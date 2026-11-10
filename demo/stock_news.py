"""E2: text features from MOPS announcements and cnyes news (LLM scores and embeddings), walk-forward (TWStock).

Data: dataset/news/scores/ from demo/news_score.py (one item per text and stock; the zero-shot LLM probabilities of
positive / neutral / negative and a bge-m3 embedding). Each item enters the forecast of the first trading day d with
d 08:50 >= its time (before the 09:00 open): as "overnight" if it came after 13:30 of the trading day before d,
otherwise as "session" (it came during that day's session, which its close already reflects).
Features per stock and day:
- LLM, per source (mops, cnyes): log(1 + count) overnight, the sum of the scores s = P(positive) - P(negative)
  overnight, and the sum of s in the session; the market: the mean s and log(1 + count) of the cnyes items that tag
  none of our stocks, overnight;
- embedding: the 1024-dim vectors projected on the first K_OWN principal components of the training items (before
  2022) of our stocks, summed over the stock's overnight items; the market: the first K_MKT components of the market
  items, averaged overnight.
Base: poly p0 + range + the US-session features (52). Walk-forward 2022 - 2026 as before. Reported: the daily rank IC
of the overnight LLM score with r (days and stocks with an item), and R2, the score, its cross-sectional part and the
paired differences to the base on the 100 and the 50 stocks, on the stock-days with an overnight item, and on
2025 - 2026 (after the LLM's training data).

    python demo/stock_news.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_poly import ROOT, load  # noqa: E402
from stock_recurrent import base_features  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge, clustered_mean  # noqa: E402

U = 1e4
K_OWN, K_MKT = 16, 8


def text_features(dates, stocks, raw=False):
    """raw: also return the mean raw embedding (T, S, 1024) of the stock's own overnight items (0 without one)."""
    T, S = len(dates), len(stocks)
    z = np.load(ROOT / "news" / "scores" / "items.npz")
    E = np.load(ROOT / "news" / "scores" / "emb.npy")[z["emb_id"]].astype(np.float32)
    src, stock, ts, P = z["src"], z["stock"], pd.DatetimeIndex(z["ts"]), z["p"]
    s = P[:, 0] - P[:, 2]
    cut = (dates + pd.Timedelta(hours=8, minutes=50)).to_numpy()
    d = np.searchsorted(cut, ts.to_numpy(), "left")  # first d with d 08:50 >= ts
    keep = (d > 0) & (d < T)
    prev_close = np.r_[np.datetime64("NaT"), (dates[:-1] + pd.Timedelta(hours=13, minutes=30)).to_numpy()]
    over = np.zeros(len(d), bool)
    over[keep] = ts.to_numpy()[keep] > prev_close[d[keep]]
    sidx = {c: j for j, c in enumerate(stocks)}
    j = np.array([sidx.get(c, -1) for c in stock])
    own, mkt = keep & (j >= 0), keep & (stock == "MKT")
    print(f"{keep.sum()} items in range: {np.sum(own & over)} own overnight, {np.sum(own & ~over)} own session, "
          f"{np.sum(mkt & over)} market overnight", flush=True)

    def grid(m, w):
        out = np.zeros((T, S))
        np.add.at(out, (d[m], j[m]), w[m])
        return out

    feats, names = [], []
    for so in ("mops", "cnyes"):
        m = own & (src == so)
        feats += [np.log1p(grid(m & over, np.ones(len(d)))), grid(m & over, s), grid(m & ~over, s)]
        names += [f"{so} count over", f"{so} score over", f"{so} score session"]
    mo = mkt & over
    mc = np.bincount(d[mo], minlength=T).astype(float)
    ms = np.bincount(d[mo], weights=s[mo], minlength=T) / np.maximum(mc, 1)
    feats += [np.repeat(ms[:, None], S, 1), np.repeat(np.log1p(mc)[:, None], S, 1)]
    names += ["market score over", "market count over"]
    LLM = np.stack(feats, -1)

    tr_items = ts < "2022-01-01"
    def pca(m, k):
        X = E[m & tr_items]
        c = X.mean(0)
        _, _, Vt = np.linalg.svd(X - c, full_matrices=False)
        return (E - c) @ Vt[:k].T
    Zo, Zm = pca(own, K_OWN), pca(mkt, K_MKT)
    EO = np.zeros((T, S, K_OWN))
    np.add.at(EO, (d[own & over], j[own & over]), Zo[own & over])
    EM = np.zeros((T, K_MKT))
    np.add.at(EM, d[mo], Zm[mo])
    EM /= np.maximum(mc, 1)[:, None]
    EMB = np.concatenate([EO, np.repeat(EM[:, None], S, 1)], -1)
    has = grid(own & over, np.ones(len(d))) > 0
    if not raw:
        return LLM, names, EMB, has
    m = own & over
    R = np.zeros((T, S, E.shape[1]), np.float32)
    np.add.at(R, (d[m], j[m]), E[m])
    R /= np.maximum(grid(m, np.ones(len(d))), 1)[..., None].astype(np.float32)
    return LLM, names, EMB, has, R


def main() -> None:
    dates, r, traded, gap, known = load()
    T, S = r.shape
    info = json.loads((ROOT / "info.json").read_text())
    stocks, top50 = info["Top100@Y"], set(info["Top50@Y"])
    X, y = base_features(dates, r, traded, gap, known, stocks)
    LLM, lnames, EMB, has = text_features(dates, stocks)
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y)
    print(f"share of stock-days with an own overnight item: {np.mean(has[ok]):.1%}", flush=True)

    # model-free check: daily rank IC of the summed overnight LLM score with r, over the stocks with an item
    sc = LLM[..., lnames.index("mops score over")] + LLM[..., lnames.index("cnyes score over")]
    rows = []
    for t in range(T):
        m = ok[t] & has[t]
        if m.sum() >= 5:
            a, b = pd.Series(sc[t, m]).rank(), pd.Series(y[t, m]).rank()
            rows.append((dates[t].year, a.corr(b), np.corrcoef(sc[t, m], y[t, m])[0, 1]))
    ic = pd.DataFrame(rows, columns=["year", "rank IC", "IC"])
    print("daily IC of the overnight LLM score with r (days with >= 5 stocks with an item):")
    print(ic.groupby("year").agg(["mean", "count"]).round(4).to_string())
    for name in ("mops score over", "cnyes score over", "market score over"):
        v = LLM[..., lnames.index(name)]
        m = ok & (v != 0)
        print(f"pooled corr({name}, r) on {m.sum()} stock-days: {np.corrcoef(v[m], y[m])[0, 1]:+.4f}; "
              f"mean r where > 0: {U * y[m & (v > 0)].mean():+.1f}e-4, where < 0: {U * y[m & (v < 0)].mean():+.1f}e-4")

    li = lambda *ns: [lnames.index(n) for n in ns]
    own_m = li("mops count over", "mops score over", "mops score session")
    own_c = li("cnyes count over", "cnyes score over", "cnyes score session")
    mk = li("market score over", "market count over")
    SETS = {"base": X,
            "+ LLM mops": np.concatenate([X, LLM[..., own_m]], -1),
            "+ LLM cnyes": np.concatenate([X, LLM[..., own_c]], -1),
            "+ LLM own (both)": np.concatenate([X, LLM[..., own_m + own_c]], -1),
            "+ LLM own + market": np.concatenate([X, LLM], -1),
            "+ emb own": np.concatenate([X, EMB[..., :K_OWN]], -1),
            "+ emb own + market": np.concatenate([X, EMB], -1),
            "+ LLM + emb (all)": np.concatenate([X, LLM, EMB], -1)}
    res = {}
    for name, Z in SETS.items():
        mu = np.full((T, S), np.nan)
        for Y in EVAL:
            inner, sel, rows_, te = ok & (year < Y - 1), ok & (year == Y - 1), ok & (year < Y), ok & (year == Y)
            m = Ridge(Z[inner], y[inner])
            a = min(ALPHAS, key=lambda a: np.mean((m.predict(Z[sel], m.weights(a)) - y[sel]) ** 2))
            m = Ridge(Z[rows_], y[rows_])
            mu[te] = m.predict(Z[te], m.weights(a))
        res[name] = mu
    ev = ok & np.isin(year, EVAL)
    is50 = np.array([s in top50 for s in stocks])[None, :]
    fm = lambda t: f"{U * t[0]:+5.1f}e-4 ± {U * t[1]:3.1f}e-4"
    for uname, um in (("all 100 stocks", ev), ("the 50-stock subset", ev & is50),
                      ("100 stocks, days with an own overnight item", ev & has),
                      ("100 stocks, 2025-2026", ev & (year >= 2025)), ("50 stocks, 2025-2026", ev & is50 & (year >= 2025))):
        y_, d_ = y[um], np.nonzero(um)[0]
        _, inv = np.unique(d_, return_inverse=True)
        mkr = (np.bincount(inv, weights=y_) / np.bincount(inv))[inv]
        mb = res["base"][um]
        sb = np.where(mb > 0, 1.0, -1.0)
        base = np.mean((y_ - y_.mean()) ** 2)
        print(f"\n== walk-forward, {uname} ({um.sum()} stock-days); always down: score {fm(clustered_mean(-y_, d_))}")
        for name, mu in res.items():
            p = mu[um]
            sg = np.where(p > 0, 1.0, -1.0)
            dm = clustered_mean((mb - y_) ** 2 - (p - y_) ** 2, d_)
            print(f"{name:20s} R2 {1 - np.mean((p - y_) ** 2) / base:+.4f}  score {fm(clustered_mean(sg * y_, d_))}  "
                  f"cross {fm(clustered_mean(sg * (y_ - mkr), d_))}  | vs base: flips {np.mean(sg != sb):5.1%}  R2 "
                  f"{dm[0] / base:+.4f} ± {dm[1] / base:.4f}  score {fm(clustered_mean((sg - sb) * y_, d_))}", flush=True)


if __name__ == "__main__":
    main()
