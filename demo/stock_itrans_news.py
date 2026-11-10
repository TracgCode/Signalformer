"""E2f: iTransformer-style stock tokens on the ridge residual, with a cross attention from the stock tokens (queries)
to the morning's news items (keys and values), walk-forward (TWStock).

Stage 1 (base): the ridge on the 52 base features plus the news-predicted gap block of demo/stock_news_gap.py
(g_hat, |g_hat|, the day's mean g_hat; 55 features), whose forecast mu is the reference. Stage 2 forecasts the
residual e = r - mu; the forecast is mu + e_hat.

Model, for one forecast day t at a time (all 100 stocks together):
- stock tokens (iTransformer: one token per variate, here per stock, from its whole window): the stock's last
  W = 60 days of [e_{k-1} / se, r_{k-1} / sd, g_{k-1} / sd, traded_{k-1}] flattened (240 values) -> Linear -> d,
  plus Linear of the day's 55 ridge inputs and mu / se -> d; dropout; one self-attention layer across the stocks
  (pre-norm, residual), so a stock's token can read the other stocks' windows;
- news: the items of the morning (own overnight items of our stocks and the market items, i.e. everything that
  enters the forecast of day t, at most N_MAX = 256, the latest kept): the first 64 principal components of their
  bge-m3 embeddings (fitted on the items before 2022) and the LLM score s = P(pos) - P(neg); keys W_k z, values W_v z,
  plus a learned null key and value (attend to nothing);
- cross attention: query W_q LN(h_s) of stock s; logit q . k_n / sqrt(d) + beta * [item n tags stock s]; the
  attended value c_s; e_hat_s = w . [h_s, c_s] (w started at 0, so the start is the ridge).
Mode newsq (python demo/stock_itrans_news.py newsq): the other direction, the standard cross attention with the
news as the query and the stock tokens as keys and values: q_s = W_q u_s, u_s = the mean of the 65-dim vectors of
stock s's own overnight items and log(1 + their count) (0 without an item, so the attention is then uniform);
k_j = W_k LN(h_j), v_j = W_v LN(h_j) for the 100 stocks j; c_s = sum_j softmax_j(q_s . k_j / sqrt(d)) v_j;
e_hat_s = w . [h_s, c_s].
Mode sigval: the news as keys and the price state as values: q_s = W_q LN(h_s), k_n = W_k z_n as in stockq (with the
own-tag bias and the null key), but v_n = W_v LN(h_{tag(n)}), the token of the stock the item tags (a learned market
token for the market items), so a stock reads the states of the stocks that the morning's news makes relevant to it.
Mode itemq: every item of the morning as its own query over the stock tokens (o_n = sum_j softmax_j(W_q z_n .
W_k LN(h_j)) W_v LN(h_j)), then each stock attends over the item outputs (query W_q2 LN(h_s), keys W_k2 o_n, the
own-tag bias and a null key), so several items are not averaged before they are read.
Versions: iT (no cross attention), iT + news, iT + news shuffled (each day gets the news pool of another day of the
same year; the tags move with the items, so the own-stock bias stays meaningful but the content is from another day).
Splits as demo/stock_news_attn2.py: select the number of steps (0 allowed) on Y - 1 with residuals out of sample
(leave one half-year block out within the years before Y - 1, Y - 1 from the fit on the years before it), patience 3,
up to 30 epochs; refit on all years before Y for that many steps; 3 seeds averaged. d = 16, dropout 0.3, AdamW
lr 1e-3, weight decay 1e-2, batches of 16 days. The gap block is out of sample in the same way (its own ridge, fitted
on the years before the block's reference year, leave one year out within them).

    python demo/stock_itrans_news.py [stockq|newsq|sigval|itemq]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_news import text_features  # noqa: E402
from stock_news_attn import report  # noqa: E402
from stock_news_attn2 import oos_mu  # noqa: E402
from stock_news_gap import gap_block  # noqa: E402
from stock_poly import ROOT, load  # noqa: E402
from stock_recurrent import base_features, lagged  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stock_model import StockNewsAttention  # noqa: E402

U = 1e4
SEEDS = (2021, 2022, 2023)
W, D, N_MAX, K_PC, DAYS = 60, 16, 256, 64, 16
VERSIONS = ("iT", "iT + news", "iT + news shuffled")


def ghat_for(Z, g, ok, year, Yb):
    """gap forecasts: on Yb from the fit on the years before Yb, on each earlier year from the fit without it."""
    inner, sel, rows = ok & (year < Yb - 1), ok & (year == Yb - 1), ok & (year < Yb)
    m = Ridge(Z[inner], g[inner])
    a = min(ALPHAS, key=lambda a: np.mean((m.predict(Z[sel], m.weights(a)) - g[sel]) ** 2))
    G = np.full(g.shape, np.nan)
    m = Ridge(Z[rows], g[rows])
    at = ok & (year == Yb)
    G[at] = m.predict(Z[at], m.weights(a))
    for yy in np.unique(year[rows]):
        fit, at = rows & (year != yy), rows & (year == yy)
        mf = Ridge(Z[fit], g[fit])
        G[at] = mf.predict(Z[at], mf.weights(a))
    return G


def news_pool(dates, stocks):
    """per day: (N_MAX, K_PC + 1) item vectors, the tagged stock index (-1 market, -2 padding)."""
    T = len(dates)
    z = np.load(ROOT / "news" / "scores" / "items.npz")
    E = np.load(ROOT / "news" / "scores" / "emb.npy")
    stock, ts, P, eid = z["stock"], pd.DatetimeIndex(z["ts"]), z["p"], z["emb_id"]
    cut = (dates + pd.Timedelta(hours=8, minutes=50)).to_numpy()
    d = np.searchsorted(cut, ts.to_numpy(), "left")
    keep = (d > 0) & (d < T)
    prev_close = np.r_[np.datetime64("NaT"), (dates[:-1] + pd.Timedelta(hours=13, minutes=30)).to_numpy()]
    over = np.zeros(len(d), bool)
    over[keep] = ts.to_numpy()[keep] > prev_close[d[keep]]
    sidx = {c: j for j, c in enumerate(stocks)}
    j = np.array([sidx.get(c, -1) for c in stock])
    use = keep & over & ((j >= 0) | (stock == "MKT"))
    tr = use & (ts < "2022-01-01")
    Etr = E[eid[tr]].astype(np.float32)
    c = Etr.mean(0)
    _, _, Vt = np.linalg.svd(Etr[:: max(1, len(Etr) // 50000)] - c, full_matrices=False)
    Z = (E[eid[use]].astype(np.float32) - c) @ Vt[:K_PC].T
    Z /= Z[ts[use] < "2022-01-01"].std(0) + 1e-6
    s = (P[use, 0] - P[use, 2]).astype(np.float32)
    Z = np.concatenate([Z, s[:, None]], 1).astype(np.float32)
    du, ju, tu = d[use], j[use], ts[use].to_numpy()
    NZ = np.zeros((T, N_MAX, K_PC + 1), np.float32)
    NT = np.full((T, N_MAX), -2, np.int64)
    order = np.lexsort((-tu.astype("int64"), du))  # by day, latest first
    du, ju, Z = du[order], ju[order], Z[order]
    starts = np.searchsorted(du, np.arange(T))
    ends = np.searchsorted(du, np.arange(T), "right")
    for t in range(T):
        n = min(ends[t] - starts[t], N_MAX)
        NZ[t, :n] = Z[starts[t]:starts[t] + n]
        NT[t, :n] = ju[starts[t]:starts[t] + n]
    cnt = ends - starts
    print(f"news pool: {use.sum()} items; per day mean {cnt[cnt > 0].mean():.0f}, max {cnt.max()}, "
          f"{np.mean(cnt > N_MAX):.1%} of days capped at {N_MAX}", flush=True)
    return NZ, NT


def ITNews(S, Fx, news, mode="stockq"):
    return StockNewsAttention(S, W * 4, Fx, K_PC + 1, use_news=news, mode=mode, d_model=D)


class Data:
    def __init__(self, Xa, y, mu, ok, sig, trd, tr, NZ, NT, dev):
        e = np.where(ok, y - mu, np.nan)
        e0 = np.where(np.isfinite(e) & trd, e, 0.0)
        self.se = np.nanstd(e[tr])
        Xs = (Xa - Xa[tr].mean(0)) / (Xa[tr].std(0) + 1e-12)
        mus = (mu - np.nanmean(mu[tr])) / self.se
        cur = np.nan_to_num(np.concatenate([Xs, mus[..., None]], -1)).astype(np.float32)
        S4 = np.concatenate([lagged(e0, 1)[..., None] / self.se, sig], -1).astype(np.float32)   # (T, S, 4)
        self.S4 = torch.tensor(np.nan_to_num(S4), device=dev)
        self.cur = torch.tensor(cur, device=dev)
        self.Y = torch.tensor(np.nan_to_num(e / self.se), dtype=torch.float32, device=dev)
        self.M = torch.tensor(np.isfinite(e), device=dev)
        self.NZ, self.NT, self.dev = NZ, NT, dev
        self.ar = torch.arange(W, device=dev) - (W - 1)

    def batch(self, t):
        win = self.S4[t[:, None] + self.ar]                              # (B, W, S, 4)
        win = win.permute(0, 2, 1, 3).flatten(2)                          # (B, S, W * 4)
        return win, self.cur[t], self.NZ[t], self.NT[t]

    def predict(self, net, days):
        net.eval()
        with torch.no_grad():
            out = torch.cat([net(*self.batch(days[i:i + 64])) for i in range(0, len(days), 64)])
        net.train()
        return out


def train(data, S, Fx, news, seed, days_tr, steps=None, days_va=None, mode="stockq"):
    torch.manual_seed(seed)
    g = torch.Generator(device=data.dev).manual_seed(seed)
    net = ITNews(S, Fx, news, mode).to(data.dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    if days_va is not None:
        Mv, Yv = data.M[days_va], data.Y[days_va]
        val = lambda: ((data.predict(net, days_va) - Yv)[Mv]).pow(2).mean().item()
        z = Yv[Mv].pow(2).mean().item()
        best, best_steps, bad, curve = val(), 0, 0, []
        curve.append(best / z)
    done = 0
    for ep in range(1, 31):
        order = days_tr[torch.randperm(len(days_tr), device=data.dev, generator=g)]
        for i in range(0, len(order), DAYS):
            if steps is not None and done >= steps:
                return net, done, None
            b = order[i:i + DAYS]
            m = data.M[b]
            loss = ((net(*data.batch(b)) - data.Y[b])[m]).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            done += 1
        if days_va is not None:
            v = val()
            curve.append(v / z)
            if v < best:
                best, best_steps, bad = v, done, 0
            else:
                bad += 1
                if bad >= 3:
                    break
        if steps is not None and done >= steps:
            break
    return net, (best_steps if days_va is not None else done), (curve if days_va is not None else None)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "stockq"
    dev = "cuda"
    dates, r, traded, gap, known = load()
    T, S = r.shape
    info = json.loads((ROOT / "info.json").read_text())
    stocks, top50 = info["Top100@Y"], set(info["Top50@Y"])
    X, y = base_features(dates, r, traded, gap, known, stocks)
    LLM, _, EMB, has = text_features(dates, stocks)
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    half = year * 2 + (np.repeat(dates.month.to_numpy()[:, None], S, axis=1) > 6)
    g = gap.to_numpy()
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y) & np.isfinite(g)
    trd = traded.to_numpy()
    Xn = np.concatenate([X, LLM, EMB], -1)
    pre = ok & (year < 2022)
    r0 = np.where(trd, np.nan_to_num(r.to_numpy()), 0.0)
    g0 = np.where(trd, np.nan_to_num(g), 0.0)
    sig = np.stack([lagged(r0, 1) / r0[pre].std(), lagged(g0, 1) / g0[pre].std(), lagged(trd.astype(float), 1)], -1)
    NZ, NT = news_pool(dates, stocks)
    rng = np.random.default_rng(0)
    perm = np.arange(T)
    yrs = dates.year.to_numpy()
    for yy in np.unique(yrs):
        idx = np.nonzero(yrs == yy)[0]
        perm[idx] = rng.permutation(idx)
    NZt, NTt = torch.tensor(NZ, device=dev), torch.tensor(NT, device=dev)
    NZs, NTs = NZt[torch.tensor(perm, device=dev)], NTt[torch.tensor(perm, device=dev)]
    del NZ, NT
    tday = np.arange(T)
    res = {k: np.full((T, S), np.nan) for k in ("ridge",) + VERSIONS}
    for Y in EVAL:
        Xa_s = np.concatenate([X, gap_block(ghat_for(Xn, g, ok, year, Y - 1), ok)], -1)
        Xa_r = np.concatenate([X, gap_block(ghat_for(Xn, g, ok, year, Y), ok)], -1)
        inner, sel = ok & (year < Y - 1), ok & (year == Y - 1)
        m = Ridge(Xa_s[inner], y[inner])
        a = min(ALPHAS, key=lambda a: np.mean((m.predict(Xa_s[sel], m.weights(a)) - y[sel]) ** 2))
        mu_s = oos_mu(Xa_s, y, ok, year, half, a, Y - 1)
        mu_r = oos_mu(Xa_r, y, ok, year, half, a, Y)
        te = ok & (year == Y)
        res["ridge"][te] = mu_r[te]
        tr_s, tr_r = ok & (year < Y - 1), ok & (year < Y)
        dsel = lambda msk: torch.tensor(tday[(msk.any(1)) & (tday >= W)], device=dev)
        d_tr_s, d_va_s = dsel(tr_s), dsel(ok & (year == Y - 1))
        d_tr_r, d_te = dsel(tr_r), dsel(te)
        for v in VERSIONS:
            news = v != "iT"
            nz, nt = (NZs, NTs) if v.endswith("shuffled") else (NZt, NTt)
            data_s = Data(Xa_s, y, mu_s, ok & (year <= Y - 1), sig, trd, tr_s, nz, nt, dev)
            data_r = Data(Xa_r, y, mu_r, ok & (year <= Y), sig, trd, tr_r, nz, nt, dev)
            Fx = data_r.cur.shape[-1]
            preds, log = [], []
            for seed in SEEDS:
                _, steps, curve = train(data_s, S, Fx, news, seed, d_tr_s, days_va=d_va_s, mode=mode)
                if steps == 0:
                    preds.append(np.zeros((len(d_te), S)))
                else:
                    net, _, _ = train(data_r, S, Fx, news, seed, d_tr_r, steps=steps, mode=mode)
                    preds.append(data_r.predict(net, d_te).cpu().numpy() * data_r.se)
                log.append(f"{steps} steps (val {min(curve):.5f})")
            p = np.mean(preds, 0)
            out = np.full((T, S), np.nan)
            out[d_te.cpu().numpy()] = p
            res[v][te] = mu_r[te] + out[te]
            print(f"{Y} {v}: {', '.join(log)}", flush=True)
            del data_s, data_r
            torch.cuda.empty_cache()

    ev = ok & np.isin(year, EVAL)
    is50 = np.array([s in top50 for s in stocks])[None, :]
    subsets = (("all 100 stocks", ev), ("the 50-stock subset", ev & is50),
               ("100 stocks, days with an own overnight item", ev & has),
               ("100 stocks, 2025-2026", ev & (year >= 2025)))
    fm = lambda t: f"{U * t[0]:+5.1f}e-4 ± {U * t[1]:3.1f}e-4"
    print(f"\n######## mode {mode}: each version vs the ridge (base + g_hat)")
    report(res, {v: "ridge" for v in VERSIONS}, y, subsets, fm)
    print("\n######## controls: news vs shuffled news, news vs no news")
    for a_, b_ in (("iT + news", "iT + news shuffled"), ("iT + news", "iT")):
        report({b_: res[b_], a_: res[a_]}, {a_: b_}, y, subsets[:3], fm)


if __name__ == "__main__":
    main()
