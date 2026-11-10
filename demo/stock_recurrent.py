"""Recurrent corrections of the ridge's residual: linear recursions, PID filters, GRU and LSTM, walk-forward (TWStock).

Stage 1 is the base ridge of demo/stock_regime_peer.py (poly p0 + range + the US-session features, 52 features), which
gives mu_t for every (stock, day t); on every year before the forecast year Y, mu comes from the ridge fitted on the
years before Y except that year, so the residuals e_t = r_t - mu_t the corrections learn from are out of sample, like
those on Y. e_t is set to 0 where the stock did not trade. Stage 2 forecasts e_t from what is known before day t, and
the forecast is mu_t + e_hat_t. Corrections:

- AR(p): ridge on the stock's own e_{t-1} .. e_{t-p};
- EMA: ridge on exponential moving averages E_rho(t) = rho E_rho(t-1) + (1 - rho) e_{t-1}, rho in RHOS (the simple
  recursion e_hat_t = g E_rho(t) for every rho at once);
- PID: e_hat_t = Kp e_{t-1} + Ki I_rho(t) + Kd (e_{t-1} - e_{t-2}), I_rho the leaky integral sum_k rho^k e_{t-1-k};
  the gains by least squares (ridge), rho chosen with alpha on the MSE of Y - 1; on the stock's own residual, and on
  the own residual plus the same three terms of the day's mean residual over the stocks;
- GRU, LSTM: one layer, 32 hidden units, shared by the stocks, read the last L = 60 days; per day the input is
  e_{t-1}, mu_t and the traded flag ("resid"), or those and the 52 base features ("resid + X"); head Linear(32 -> 1)
  started at zero after dropout 0.5; AdamW (lr 1e-3, weight decay 1e-2), batches of 1024, trained on the years before
  Y - 1, stopped early on the MSE of Y - 1 (patience 3, up to 30 epochs); 3 seeds, averaged.
The linear corrections choose alpha on Y - 1 from the fit on the years before Y - 1 and are refitted on all years
before Y. Reported pooled 2022 - 2026 on the 100 and the 50-stock subset: the autocorrelation of e on the evaluation
years, then R2, the score of the sign, its cross-sectional part, and the paired differences to the ridge alone,
day-clustered.

    python demo/stock_recurrent.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_extra import extra_features, signals  # noqa: E402
from stock_overnight import ADR, yahoo_features  # noqa: E402
from stock_poly import ROOT, features, load  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge, clustered_mean  # noqa: E402

U = 1e4
RHOS = (0.5, 0.8, 0.9, 0.95, 0.98, 0.99)
SEEDS = (2021, 2022, 2023)
L, H = 60, 32


def base_features(dates, r, traded, gap, known, stocks):
    T, S = r.shape
    train = np.asarray(dates < "2022-01-01")
    Xb, y, bnames = features(dates, r, traded, gap, known)
    Xe, _, egroups = extra_features(signals(dates, known, S), train)
    base = np.concatenate([Xb[..., [n.endswith("p0") for n in bnames]], Xe[..., egroups == "range"]], -1)
    closes = pd.read_csv(ROOT / "market" / "2330.csv", parse_dates=["date"]).set_index("date")["etl:adj_close"]
    yh = yahoo_features(dates, closes.reindex(dates).to_numpy())
    mkt = lambda v: np.repeat(np.asarray(v, float)[:, None], S, 1)
    own = np.zeros((T, S))
    for s, sym in ADR.items():
        own[:, stocks.index(s)] = yh[{"TSM": "tsm_ret", "UMC": "umc_ret", "ASX": "asx_ret"}[sym]].to_numpy()
    us = np.stack([mkt(yh[k]) for k in ("tsm_ret", "sox_ret", "nq_ret", "adr_prem_chg")] + [own], -1)
    return np.concatenate([base, us], -1), y


def lagged(e0, k):
    """row t holds e0[t - k] (0 before the start)."""
    out = np.zeros_like(e0)
    out[k:] = e0[:-k]
    return out


def leaky(e0, rho):
    """row t: sum_k rho^k e0[t - 1 - k]."""
    out = np.zeros_like(e0)
    for t in range(1, len(e0)):
        out[t] = rho * out[t - 1] + e0[t - 1]
    return out


class Rec(nn.Module):
    def __init__(self, kind, D):
        super().__init__()
        self.rnn = (nn.GRU if kind == "GRU" else nn.LSTM)(D, H, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.head = nn.Linear(H, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.drop(self.rnn(x)[0][:, -1])).squeeze(1)


def train_rec(kind, F, tgt, tr, va, te, seed, device):
    """F (T, S, D) standardized inputs; tr/va/te boolean (T, S); returns e_hat on te (in target units), best epoch."""
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    Ft = torch.tensor(F, dtype=torch.float32, device=device)
    st = np.nanstd(tgt[tr])
    Yt = torch.tensor(np.nan_to_num(tgt / st), dtype=torch.float32, device=device)
    ar = torch.arange(L, device=device) - (L - 1)
    idx = lambda m: [torch.tensor(a, device=device) for a in np.nonzero(m)]
    win = lambda t, s: Ft[t[:, None] + ar, s[:, None]]
    (ttr, str_), (tva, sva), (tte, ste) = idx(tr), idx(va), idx(te)
    net = Rec(kind, F.shape[-1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)

    def predict(t, s):
        net.eval()
        with torch.no_grad():
            return torch.cat([net(win(t[i:i + 8192], s[i:i + 8192])) for i in range(0, len(t), 8192)])

    val = lambda: (predict(tva, sva) - Yt[tva, sva]).pow(2).mean().item()
    best, state, bad, best_ep = val(), {k: v.clone() for k, v in net.state_dict().items()}, 0, 0
    for ep in range(1, 31):
        net.train()
        order = torch.randperm(len(ttr), device=device, generator=g)
        for i in range(0, len(order), 1024):
            b = order[i:i + 1024]
            loss = (net(win(ttr[b], str_[b])) - Yt[ttr[b], str_[b]]).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        v = val()
        if v < best:
            best, state, bad, best_ep = v, {k: x.clone() for k, x in net.state_dict().items()}, 0, ep
        else:
            bad += 1
            if bad >= 3:
                break
    net.load_state_dict(state)
    return predict(tte, ste).cpu().numpy() * st, best_ep


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dates, r, traded, gap, known = load()
    T, S = r.shape
    info = json.loads((ROOT / "info.json").read_text())
    stocks, top50 = info["Top100@Y"], set(info["Top50@Y"])
    X, y = base_features(dates, r, traded, gap, known, stocks)
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y)
    trd = traded.to_numpy()
    print(f"{ok.sum()} stock-days, {X.shape[-1]} base features; device {device}", flush=True)

    names = ["ridge", "AR(5)", "AR(20)", "EMA (6 rho)", "PID own", "PID own + market",
             "GRU resid", "GRU resid + X", "LSTM resid", "LSTM resid + X"]
    res = {k: np.full((T, S), np.nan) for k in names}
    e_eval = np.full((T, S), np.nan)
    for Y in EVAL:
        rows = ok & (year < Y)
        inner, sel = ok & (year < Y - 1), ok & (year == Y - 1)
        m = Ridge(X[inner], y[inner])
        a = min(ALPHAS, key=lambda a: np.mean((m.predict(X[sel], m.weights(a)) - y[sel]) ** 2))
        mu = np.full((T, S), np.nan)
        m = Ridge(X[rows], y[rows])
        te = ok & (year == Y)
        mu[te] = m.predict(X[te], m.weights(a))
        for yy in np.unique(year[rows]):
            fit, at = rows & (year != yy), rows & (year == yy)
            mf = Ridge(X[fit], y[fit])
            mu[at] = mf.predict(X[at], mf.weights(a))
        e = np.where(ok & (year <= Y), y - mu, np.nan)
        e_eval[te] = e[te]
        e0 = np.where(np.isfinite(e) & trd, e, 0.0)
        cnt = np.maximum((np.isfinite(e) & trd).sum(1), 1)
        em = np.repeat((e0.sum(1) / cnt)[:, None], S, 1)
        res["ridge"][te] = mu[te]
        tr, va = ok & (year < Y - 1), ok & (year == Y - 1)

        def linear(Z):
            """Z (T, S, K) or a list of candidate Z; ridge of e on Z, (Z, alpha) on Y - 1; refit on all before Y."""
            cands = Z if isinstance(Z, list) else [Z]
            best = None
            for Zc in cands:
                mm = Ridge(Zc[tr], e[tr])
                for al in ALPHAS:
                    mse = np.mean((mm.predict(Zc[va], mm.weights(al)) - e[va]) ** 2)
                    if best is None or mse < best[0]:
                        best = (mse, Zc, al)
            _, Zc, al = best
            mm = Ridge(Zc[rows], e[rows])
            return mm.predict(Zc[te], mm.weights(al))

        lags = np.stack([lagged(e0, k) for k in range(1, 21)], -1)
        res["AR(5)"][te] = mu[te] + linear(lags[..., :5])
        res["AR(20)"][te] = mu[te] + linear(lags)
        res["EMA (6 rho)"][te] = mu[te] + linear(np.stack([(1 - rh) * leaky(e0, rh) for rh in RHOS], -1))
        pid = lambda x, rh: [lagged(x, 1), (1 - rh) * leaky(x, rh), lagged(x, 1) - lagged(x, 2)]
        res["PID own"][te] = mu[te] + linear([np.stack(pid(e0, rh), -1) for rh in RHOS])
        res["PID own + market"][te] = mu[te] + linear([np.stack(pid(e0, rh) + pid(em, rh), -1) for rh in RHOS])
        print(f"{Y}: alpha 1e{int(np.log10(a))}, linear corrections done", flush=True)

        # recurrent nets: standardize with the training rows
        se = np.nanstd(e[tr])
        Xs = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-12)
        mus = (mu - np.nanmean(mu[tr])) / se
        core = np.stack([lagged(e0, 1) / se, np.nan_to_num(mus), trd.astype(float)], -1)
        F = {"resid": core, "resid + X": np.concatenate([core, Xs], -1)}
        for kind in ("GRU", "LSTM"):
            for fname, Fv in F.items():
                Fv = np.nan_to_num(Fv).astype(np.float32)
                outs = [train_rec(kind, Fv, e, tr, va, te, s, device) for s in SEEDS]
                res[f"{kind} {fname}"][te] = mu[te] + np.mean([o[0] for o in outs], 0)
                print(f"{Y} {kind} {fname}: best epochs {[o[1] for o in outs]}", flush=True)

    # autocorrelation of the out-of-sample residual on the evaluation years
    ev = np.isin(year, EVAL) & ok
    ez = np.where(ev & trd, e_eval, np.nan)
    ac = lambda x, k: pd.Series(x[k:].ravel()).corr(pd.Series(x[:-k].ravel()))
    mres = np.nanmean(ez, 1)[:, None]
    print("\nautocorrelation of e on 2022-2026, own (pooled over stocks):",
          " ".join(f"lag{k} {ac(ez, k):+.4f}" for k in (1, 2, 3, 5, 10, 20)))
    print("autocorrelation of the day-mean e:", " ".join(f"lag{k} {ac(mres, k):+.4f}" for k in (1, 2, 3, 5, 10, 20)))
    print("autocorrelation of e - day mean:", " ".join(f"lag{k} {ac(ez - mres, k):+.4f}" for k in (1, 2, 3, 5, 10, 20)))

    is50 = np.array([s in top50 for s in stocks])
    fm = lambda t: f"{U * t[0]:+5.1f}e-4 ± {U * t[1]:3.1f}e-4"
    for uname, cols in (("all 100 stocks", np.ones(S, bool)), ("the 50-stock subset", is50)):
        um = ev & cols[None, :]
        y_, d_ = y[um], np.nonzero(um)[0]
        _, inv = np.unique(d_, return_inverse=True)
        mk = (np.bincount(inv, weights=y_) / np.bincount(inv))[inv]
        mb = res["ridge"][um]
        sb = np.where(mb > 0, 1.0, -1.0)
        base = np.mean((y_ - y_.mean()) ** 2)
        print(f"\n== walk-forward {EVAL[0]}-{EVAL[-1]}, {uname}; always down: score {fm(clustered_mean(-y_, d_))}")
        for name in names:
            p = res[name][um]
            sg = np.where(p > 0, 1.0, -1.0)
            dm = clustered_mean((mb - y_) ** 2 - (p - y_) ** 2, d_)
            print(f"{name:18s} R2 {1 - np.mean((p - y_) ** 2) / base:+.4f}  up {np.mean(sg > 0):4.0%}  score "
                  f"{fm(clustered_mean(sg * y_, d_))}  cross {fm(clustered_mean(sg * (y_ - mk), d_))}  | vs ridge: "
                  f"flips {np.mean(sg != sb):5.1%}  R2 {dm[0] / base:+.4f} ± {dm[1] / base:.4f}  score "
                  f"{fm(clustered_mean((sg - sb) * y_, d_))}", flush=True)


if __name__ == "__main__":
    main()
