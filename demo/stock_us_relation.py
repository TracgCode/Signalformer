"""E3a: which US stocks move which Taiwan stock at the open, a learned signed relation (TWStock).

Target of Taiwan stock s on day t: g - g_hat, its opening gap log(o_t / c_{t-1}) minus the walk-forward ridge
forecast from the 52 base price features (which already hold the TSM / SOX / NQ returns, the ADR premium change and
the own ADR of 2330 / 2303 / 3711), as in demo/news_probe.py; so only the stock-specific response to the US is left.
Input: for the 46 US stocks of dataset/peers/us (demo/fetch_peers.py), r_jt = log of the last adjusted close before
day t minus that before day t - 1, i.e. every US session that ended between the Taiwan close of t - 1 and the open
of t (summed over holidays), standardized by its std before 2022.
Model: g_hat_s = gamma_s * sum_j a_sj r_jt, gamma_s = softplus(.) >= 0, so the sign is in a_sj. Identities: Taiwan
u_s = [industry one-hot (dataset/profiles.csv), learned 8 dims (started at 0)], US v_j = learned 8 dims; scores
x_sj = (W_q u_s) . (W_k v_j) / sqrt(d), d = 8, one (W_q, W_k) pair per score. Normalizations of a:
- softmax: a = softmax_j(m) (non-negative);
- signed (main): a = softmax_j(m) * tanh(c): softmax selects the US stocks, tanh gives the direction;
- differential: a = softmax_j(m1) - lambda * softmax_j(m2), lambda = sigmoid(.) (Ye et al. 2024);
- L1: a = c / (eps + sum_j |c|);
- low rank: a = c (no normalization).
Comparisons: the training mean; a ridge per Taiwan stock on all 46 r_j (alpha shared, chosen on Y - 1); the mean of
the hand-made peers of demo/fetch_peers.py (PEERS, 50 stocks) with a coefficient per stock; controls of the signed
model: the industry one-hot permuted over the stocks, and r of each day replaced by that of another day of the same year.
Training: full batch, Adam lr 1e-2, L2 (Adam weight decay) on the embeddings and projections, its weight chosen on
Y - 1 from 1e-4 .. 1e-1 with the first seed (none on gamma and lambda), MSE on the target scaled by its training std; the number of
steps (evaluated every 25, up to 3,000) chosen on Y - 1 from the fit on the years before Y - 1, then refit on all
years before Y for that many steps; 3 seeds averaged. Walk-forward 2022-2026, pooled out-of-sample R2 against the
training mean with a bootstrap over days. Diagnostics of the signed model (fit before 2026, first seed): the largest
|a_sj| for some stocks, and the share of sum_j |a_sj| on the hand-made peers against their share of the 46.

    python demo/stock_us_relation.py            (E3a)
    python demo/stock_us_relation.py basket     (E3b: the same on top of a US-basket beta per Taiwan stock, the
        equal-weighted mean of the 46 standardized returns; per-stock ridge baselines on the basket, on the
        TSM / SOX / NQ returns and on both; the relation input of another day as the control)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_peers import PEERS  # noqa: E402
from news_probe import stats  # noqa: E402
from stock_news_gap import walk  # noqa: E402
from stock_poly import ROOT, load  # noqa: E402
from stock_recurrent import base_features  # noqa: E402
from stock_walkforward import EVAL, FIRST, Ridge  # noqa: E402

DEV = "cuda"
SEEDS = (0, 1, 2)
ALPHAS = 10.0 ** np.arange(-2, 6)
WDS = (1e-4, 1e-3, 1e-2, 1e-1)


class Relation(nn.Module):
    def __init__(self, ident, n_us, norm, d=8, d_learned=8, basket=False):
        super().__init__()
        self.basket = basket
        S = ident.shape[0]
        self.norm, self.d = norm, d
        self.register_buffer("ident", torch.as_tensor(ident, dtype=torch.float32))
        self.u = nn.Parameter(torch.zeros(S, d_learned))
        self.v = nn.Parameter(torch.randn(n_us, d_learned) / d_learned ** 0.5)
        n_scores = {"softmax": 1, "signed": 2, "differential": 2, "l1": 1, "lowrank": 1}[norm]
        din = ident.shape[1] + d_learned
        self.wq = nn.ModuleList([nn.Linear(din, d, bias=False) for _ in range(n_scores)])
        self.wk = nn.ModuleList([nn.Linear(d_learned, d, bias=False) for _ in range(n_scores)])
        self.lam = nn.Parameter(torch.tensor(-1.0))
        self.gamma = nn.Parameter(torch.full((S,), -2.0))
        self.beta = nn.Parameter(torch.zeros(S))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def relation(self):
        us = torch.cat([self.ident, self.u], -1)
        x = [(q(us) @ k(self.v).T) / self.d ** 0.5 for q, k in zip(self.wq, self.wk)]
        if self.norm == "softmax":
            return torch.softmax(x[0], -1)
        if self.norm == "signed":
            return torch.softmax(x[0], -1) * torch.tanh(x[1])
        if self.norm == "differential":
            return torch.softmax(x[0], -1) - torch.sigmoid(self.lam) * torch.softmax(x[1], -1)
        if self.norm == "l1":
            return x[0] / (1e-3 + x[0].abs().sum(-1, keepdim=True))
        return x[0]

    def forward(self, R, B=None):                              # R (T, J), B (T,) -> (T, S)
        out = F.softplus(self.gamma) * (R @ self.relation().T)
        return out + self.beta * B[:, None] if self.basket else out


def fit(make, R, Y, M, tr, va=None, steps=None, seed=0, wd=1e-3, B=None):
    torch.manual_seed(seed)
    net = make().to(DEV)
    # L2 on the embeddings and projections only: decaying gamma or lambda would pull softplus(gamma) to 0.69, not 0
    free = ("gamma", "lam", "beta")
    decay = [p for n, p in net.named_parameters() if n not in free]
    opt = torch.optim.Adam([{"params": decay, "weight_decay": wd},
                            {"params": [p for n, p in net.named_parameters() if n in free], "weight_decay": 0.0}],
                           lr=1e-2)
    Bt = (lambda idx: B[idx]) if B is not None else (lambda idx: None)
    sc = Y[tr][M[tr]].std()
    best, best_k = None, 0
    if va is not None:
        z = (Y[va][M[va]] ** 2).mean().item()
        best = z
    n = steps if steps is not None else 3000
    for k in range(1, n + 1):
        loss = (((net(R[tr], Bt(tr)) - Y[tr]) / sc)[M[tr]] ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if va is not None and k % 25 == 0:
            with torch.no_grad():
                v = ((net(R[va], Bt(va)) - Y[va])[M[va]] ** 2).mean().item()
            if v < best:
                best, best_k = v, k
            elif k - best_k >= 500:
                break
    return net, best_k, best


def prepare():
    dates, r, traded, gap, known = load()
    T, S = r.shape
    info = json.loads((ROOT / "info.json").read_text())
    stocks, top50 = info["Top100@Y"], set(info["Top50@Y"])
    X, y = base_features(dates, r, traded, gap, known, stocks)
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    g = gap.to_numpy()
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y) & np.isfinite(g)
    ok &= traded.to_numpy()
    ghat, _ = walk(X, g, ok, year)
    for yy in np.unique(year[ok & (year < EVAL[0])]):
        fit_, at = ok & (year != yy) & (year < EVAL[0]), ok & (year == yy)
        m = Ridge(X[fit_], g[fit_])
        ghat[at] = m.predict(X[at], m.weights(1e4))
    tgt = np.where(ok, g - ghat, 0.0)

    # US returns since the Taiwan close of t - 1
    us = sorted(p.stem for p in (ROOT / "peers" / "us").glob("*.csv"))
    pre = np.asarray(dates < "2022-01-01")
    cols = []
    for sym in us:
        f = pd.read_csv(ROOT / "peers" / "us" / f"{sym}.csv", parse_dates=["date"]).set_index("date")
        lc = np.log(f["adjclose"]).dropna()
        idx = np.searchsorted(lc.index.values, dates.values, side="left") - 1
        lev = np.where(idx >= 0, lc.values[np.clip(idx, 0, None)], np.nan)
        rr = np.r_[np.nan, np.diff(lev)]
        cols.append(rr)
    Rn = np.stack(cols, 1)
    Rn = np.nan_to_num(Rn / np.nanstd(Rn[pre], 0))
    J = len(us)
    print(f"{J} US stocks; target std {tgt[ok].std():.5f}; days {T}", flush=True)

    # identities and the hand-made peers
    prof = pd.read_csv(ROOT / "profiles.csv", dtype={"code": str}).set_index("code").reindex([str(c) for c in stocks])
    ind = prof.industry.fillna("?").to_numpy()
    cats = sorted(set(ind))
    IND = np.array([[float(a == c) for c in cats] for a in ind], np.float32)
    IND_p = IND[np.random.default_rng(1).permutation(S)]
    PM = np.zeros((S, J))
    for s, (u_l, _, _) in PEERS.items():
        if str(s) in [str(c) for c in stocks]:
            i = [str(c) for c in stocks].index(str(s))
            for t_ in u_l:
                if t_ in us:
                    PM[i, us.index(t_)] = 1.0
    has_peer = PM.sum(1) > 0
    print(f"hand-made peers for {has_peer.sum()} stocks, {PM.sum():.0f} links", flush=True)

    return dict(dates=dates, T=T, S=S, stocks=stocks, top50=top50, X=X, year=year, ok=ok, tgt=tgt, Rn=Rn, J=J,
                us=us, ind=ind, IND=IND, IND_p=IND_p, PM=PM, has_peer=has_peer)


def main() -> None:
    D_ = prepare()
    dates, T, S, stocks, top50, X, year, ok, tgt, Rn, J, us, ind, IND, IND_p, PM, has_peer = (D_[k] for k in (
        'dates', 'T', 'S', 'stocks', 'top50', 'X', 'year', 'ok', 'tgt', 'Rn', 'J', 'us', 'ind', 'IND', 'IND_p', 'PM', 'has_peer'))
    rng = np.random.default_rng(0)
    perm = np.arange(T)
    for yy in np.unique(dates.year):
        ii = np.nonzero(dates.year == yy)[0]
        perm[ii] = rng.permutation(ii)
    Rt = torch.tensor(Rn, dtype=torch.float32, device=DEV)
    Rs = Rt[torch.tensor(perm, device=DEV)]
    Yt = torch.tensor(tgt, dtype=torch.float32, device=DEV)
    Mt = torch.tensor(ok, device=DEV)
    yrs = dates.year.to_numpy()
    rows = lambda m: torch.tensor(np.nonzero(m)[0], device=DEV)

    V = {"softmax": (IND, "softmax", Rt), "signed": (IND, "signed", Rt), "differential": (IND, "differential", Rt),
         "L1": (IND, "l1", Rt), "low rank": (IND, "lowrank", Rt),
         "signed, industry permuted": (IND_p, "signed", Rt), "signed, US returns shuffled": (IND, "signed", Rs)}
    pred = {k: np.full((T, S), np.nan) for k in ["per-stock ridge", "hand-made peers"] + list(V)}
    diag_net = None
    for Y in EVAL:
        te, tr_s, va_s, tr_r = yrs == Y, yrs < Y - 1, yrs == Y - 1, yrs < Y
        # per-stock ridge on the 46 returns, alpha shared, chosen on Y - 1
        def ridge_all(trm, a_list):
            out = {}
            Xtr = Rn[trm]
            for a in a_list:
                B = np.zeros((S, J))
                for s in range(S):
                    m = ok[trm, s]
                    if m.sum() < 50:
                        continue
                    Xs, ys = Xtr[m], tgt[trm][m, s]
                    B[s] = np.linalg.solve(Xs.T @ Xs + a * len(Xs) * np.eye(J), Xs.T @ (ys - ys.mean()))
                out[a] = B
            return out
        Bs = ridge_all(tr_s, ALPHAS)
        a_best = min(ALPHAS, key=lambda a: np.mean(((Rn[va_s] @ Bs[a].T) - tgt[va_s])[ok[va_s]] ** 2))
        B = ridge_all(tr_r, [a_best])[a_best]
        pred["per-stock ridge"][te] = Rn[te] @ B.T
        # hand-made peers: mean of the peers' returns, a coefficient per stock (ridge, same alpha choice)
        Pf = Rn @ (PM / np.maximum(PM.sum(1, keepdims=True), 1)).T     # (T, S)
        def peer_coef(trm, a):
            c = np.zeros(S)
            for s in np.nonzero(has_peer)[0]:
                m = ok[trm, s]
                x, yy_ = Pf[trm][m, s], tgt[trm][m, s]
                c[s] = (x @ (yy_ - yy_.mean())) / (x @ x + a * len(x))
            return c
        a_p = min(ALPHAS, key=lambda a: np.mean(((Pf[va_s] * peer_coef(tr_s, a)) - tgt[va_s])[ok[va_s]] ** 2))
        pred["hand-made peers"][te] = Pf[te] * peer_coef(tr_r, a_p)
        line = [f"ridge alpha {a_best:g}, peers alpha {a_p:g}"]
        for name, (ident, norm, R) in V.items():
            make = lambda: Relation(ident, J, norm)
            ps = []
            # the L2 weight chosen on Y - 1 with the first seed
            wd = min(WDS, key=lambda w: fit(make, R, Yt, Mt, rows(tr_s), va=rows(va_s), seed=SEEDS[0], wd=w)[2])
            for seed in SEEDS:
                _, k, _ = fit(make, R, Yt, Mt, rows(tr_s), va=rows(va_s), seed=seed, wd=wd)
                if k == 0:
                    ps.append(np.zeros((te.sum(), S)))
                    continue
                net, _, _ = fit(make, R, Yt, Mt, rows(tr_r), steps=k, seed=seed, wd=wd)
                with torch.no_grad():
                    ps.append(net(R[rows(te)]).cpu().numpy())
                if name == "signed" and Y == EVAL[-1] and seed == SEEDS[0]:
                    diag_net = net
            pred[name][te] = np.mean(ps, 0)
            line.append(f"{name} wd {wd:g} steps {k}")
        print(f"{Y}: " + ", ".join(line), flush=True)
        if Y == EVAL[0]:
            print("parameters: " + ", ".join(f"{n} {Relation(v[0], J, v[1]).n_params()}" for n, v in V.items())
                  + f"; per-stock ridge {S * J}", flush=True)

    ev = ok & np.isin(year, EVAL)
    is50 = np.array([s in top50 for s in stocks])[None, :] & ev
    hp = has_peer[None, :] & ev
    means = np.zeros((T, S))
    for Y in EVAL:
        means[yrs == Y] = tgt[ok & (year < Y)].mean()
    print("\n== target g - g_hat (price), walk-forward 2022-2026, R2 against the training mean", flush=True)
    for name, P in pred.items():
        out = []
        for lab, m in (("100 stocks", ev), ("50-stock subset", is50), ("stocks with hand-made peers", hp)):
            dd = np.nonzero(m)[0]
            r2, r2s, c, cs = stats(np.nan_to_num(P[m]), tgt[m], means[m], dd, np.random.default_rng(0))
            out.append(f"{lab}: R2 {r2:+.4f} ± {r2s:.4f} corr {c:+.3f}")
        print(f"{name:28s} " + " | ".join(out), flush=True)

    if diag_net is not None:
        with torch.no_grad():
            A = diag_net.relation().cpu().numpy()
            gam = F.softplus(diag_net.gamma).cpu().numpy()
        print("\n== signed relation fitted before 2026 (first seed): largest |a_sj|, gamma_s", flush=True)
        for s in ["2330", "2303", "3711", "2317", "2454", "2308", "2382", "2603", "2881", "1301"]:
            if s in [str(c) for c in stocks]:
                i = [str(c) for c in stocks].index(s)
                top = np.argsort(-np.abs(A[i]))[:6]
                print(f"{s} ({ind[i]}), gamma {gam[i]:.3f}: " +
                      ", ".join(f"{us[j]} {A[i, j]:+.3f}" for j in top), flush=True)
        absA = np.abs(A)
        share = (absA * PM).sum(1)[has_peer] / absA.sum(1)[has_peer]
        base_share = PM.sum(1)[has_peer] / J
        print(f"share of sum |a| on the hand-made peers: mean {share.mean():.3f} (uniform {base_share.mean():.3f}); "
              f"negative a: {np.mean(A < 0):.1%} of pairs, {np.sum(np.abs(A[A < 0])) / absA.sum():.1%} of sum |a|",
              flush=True)


def perstock_ridge(F, tgt, ok, yrs, Y):
    """ridge of the target on F (T, S, k) per Taiwan stock, alpha shared and chosen on Y - 1; forecasts of year Y."""
    T, S, k = F.shape

    def coefs(trm, a):
        C = np.zeros((S, k + 1))
        for s in range(S):
            m = ok[trm, s]
            if m.sum() < 50:
                continue
            Xs, ys = F[trm][m, s], tgt[trm][m, s]
            mu = Xs.mean(0)
            b = np.linalg.solve((Xs - mu).T @ (Xs - mu) + a * len(Xs) * np.eye(k), (Xs - mu).T @ (ys - ys.mean()))
            C[s] = np.r_[ys.mean() - mu @ b, b]
        return C

    pred = lambda C, rows: C[:, 0] + np.einsum("tsk,sk->ts", F[rows], C[:, 1:])
    tr_s, va_s, tr_r, te = yrs < Y - 1, yrs == Y - 1, yrs < Y, yrs == Y
    a = min(ALPHAS, key=lambda a: np.mean((pred(coefs(tr_s, a), va_s) - tgt[va_s])[ok[va_s]] ** 2))
    return pred(coefs(tr_r, a), te), a


def paired(p, q, y, day, rng, base):
    """R2(p) - R2(q) and its bootstrap standard error over days."""
    d = ((q - y) ** 2 - (p - y) ** 2) / np.mean((y - base) ** 2)
    days, inv = np.unique(day, return_inverse=True)
    per = np.bincount(inv, weights=d) / len(y)
    bs = [per[rng.integers(0, len(days), len(days))].sum() for _ in range(500)]
    return d.mean(), np.std(bs)


def main_basket() -> None:
    """E3b: the relation on top of a US-basket sensitivity per Taiwan stock."""
    D_ = prepare()
    dates, T, S, stocks, top50, X, year, ok, tgt, Rn, J, us, ind, IND, PM, has_peer = (D_[k] for k in (
        "dates", "T", "S", "stocks", "top50", "X", "year", "ok", "tgt", "Rn", "J", "us", "ind", "IND", "PM",
        "has_peer"))
    yrs = dates.year.to_numpy()
    basket = Rn.mean(1)
    basket = basket / basket[np.asarray(dates < "2022-01-01")].std()
    rng = np.random.default_rng(0)
    perm = np.arange(T)
    for yy in np.unique(yrs):
        ii = np.nonzero(yrs == yy)[0]
        perm[ii] = rng.permutation(ii)
    Rt = torch.tensor(Rn, dtype=torch.float32, device=DEV)
    Rs = Rt[torch.tensor(perm, device=DEV)]
    Bt = torch.tensor(basket, dtype=torch.float32, device=DEV)
    Yt = torch.tensor(tgt, dtype=torch.float32, device=DEV)
    Mt = torch.tensor(ok, device=DEV)
    rows = lambda m: torch.tensor(np.nonzero(m)[0], device=DEV)
    FB = np.repeat(basket[:, None, None], S, 1)
    F3 = X[..., -5:-2]                                           # TSM, SOX, NQ returns of the base features
    FA = np.repeat(Rn[:, None, :], S, 1)
    RID = {"basket beta per stock": FB, "TSM / SOX / NQ beta per stock": F3,
           "basket + TSM / SOX / NQ per stock": np.concatenate([FB, F3], -1), "ridge per stock on 46": FA}
    NETS = {"basket + signed": ("signed", Rt), "basket + low rank": ("lowrank", Rt), "basket + softmax": ("softmax", Rt),
            "basket + signed, relation input of another day": ("signed", Rs)}
    pred = {k: np.full((T, S), np.nan) for k in list(RID) + list(NETS)}
    nets_2026 = {}
    for Y in EVAL:
        te = yrs == Y
        line = []
        for name, F_ in RID.items():
            pred[name][te], a = perstock_ridge(F_, tgt, ok, yrs, Y)
            line.append(f"{name} alpha {a:g}")
        tr_s, va_s, tr_r = yrs < Y - 1, yrs == Y - 1, yrs < Y
        for name, (norm, R) in NETS.items():
            make = lambda: Relation(IND, J, norm, basket=True)
            wd = min(WDS, key=lambda w: fit(make, R, Yt, Mt, rows(tr_s), va=rows(va_s), seed=SEEDS[0], wd=w,
                                            B=Bt)[2])
            ps = []
            for seed in SEEDS:
                _, k, _ = fit(make, R, Yt, Mt, rows(tr_s), va=rows(va_s), seed=seed, wd=wd, B=Bt)
                net, _, _ = fit(make, R, Yt, Mt, rows(tr_r), steps=max(k, 25), seed=seed, wd=wd, B=Bt)
                with torch.no_grad():
                    ps.append(net(R[rows(te)], Bt[rows(te)]).cpu().numpy())
                if Y == EVAL[-1] and seed == SEEDS[0]:
                    nets_2026[name] = net
            pred[name][te] = np.mean(ps, 0)
            line.append(f"{name} wd {wd:g} steps {k}")
        print(f"{Y}: " + ", ".join(line), flush=True)

    ev = ok & np.isin(year, EVAL)
    is50 = np.array([s in top50 for s in stocks])[None, :] & ev
    means = np.zeros((T, S))
    for Y in EVAL:
        means[yrs == Y] = tgt[ok & (year < Y)].mean()
    print("\n== target g - g_hat (price), walk-forward 2022-2026, R2 against the training mean; "
          "paired: minus the basket beta per stock", flush=True)
    ref = "basket beta per stock"
    for name, P in pred.items():
        out = []
        for lab, m in (("100 stocks", ev), ("50-stock subset", is50)):
            dd = np.nonzero(m)[0]
            r2, r2s, c, _ = stats(P[m], tgt[m], means[m], dd, np.random.default_rng(0))
            dm, ds = paired(P[m], pred[ref][m], tgt[m], dd, np.random.default_rng(0), means[m])
            out.append(f"{lab}: R2 {r2:+.4f} ± {r2s:.4f} corr {c:+.3f}, vs basket {dm:+.4f} ± {ds:.4f}")
        print(f"{name:48s} " + " | ".join(out), flush=True)

    for name in ("basket + signed", "basket + low rank"):
        net = nets_2026.get(name)
        if net is None:
            continue
        with torch.no_grad():
            A = (F.softplus(net.gamma)[:, None] * net.relation()).cpu().numpy()    # effective weight per US stock
            beta = net.beta.cpu().numpy()
        absA = np.abs(A)
        eff = absA.sum(1) ** 2 / (A ** 2).sum(1)
        share = (absA * PM).sum(1)[has_peer] / absA.sum(1)[has_peer]
        print(f"\n== {name}, fitted before 2026 (first seed): effective weights gamma_s a_sj; "
              f"effective number of US stocks per Taiwan stock: median {np.median(eff):.1f} of {J}; "
              f"share of sum |w| on the hand-made peers {share.mean():.3f} (uniform {(PM.sum(1)[has_peer] / J).mean():.3f}); "
              f"negative: {np.mean(A < 0):.1%} of pairs, {absA[A < 0].sum() / absA.sum():.1%} of sum |w|", flush=True)
        for s_ in ["2330", "2303", "3711", "2317", "2454", "2308", "2382", "3008", "2603", "2881", "1301", "2412"]:
            if s_ in [str(c) for c in stocks]:
                i = [str(c) for c in stocks].index(s_)
                top = np.argsort(-absA[i])[:6]
                print(f"  {s_} ({ind[i]}), basket beta {beta[i]:+.3f}: " +
                      ", ".join(f"{us[j]} {A[i, j]:+.3f}" for j in top), flush=True)
        cats = sorted(set(ind))
        print("  mean weight by industry (industries with >= 4 stocks), top 4:", flush=True)
        for c_ in cats:
            m = ind == c_
            if m.sum() >= 4:
                w = A[m].mean(0)
                top = np.argsort(-np.abs(w))[:4]
                print(f"    {c_} ({m.sum()}): " + ", ".join(f"{us[j]} {w[j]:+.3f}" for j in top), flush=True)


if __name__ == "__main__":
    main_basket() if "basket" in sys.argv else main()
