"""E2i: which stock embedding routes the news of other stocks to a stock, a linear spillover probe (TWStock).

Sample: the stock-days (100 stocks, 2020-04 .. 2026-08) without an own overnight item (MOPS announcement or cnyes
news, after 13:30 of the previous trading day and before 08:50) on days when at least one other stock has one.
Targets of stock i on day t, as in demo/news_probe.py: g - g_hat (the opening gap minus its walk-forward forecast from
the 52 price features) and |g| - |g|_hat (the size of the gap minus its forecast from the stock's mean |g| over 5, 20
and 60 days and |g_hat|).
News of stock j on day t: s_jt = the sum over its overnight items of P(利多) - P(利空) of the direction field of
demo/news_struct.py, and n_jt = 1 if it has any. For a similarity w_ij >= 0 (w_ii = 0) the features of stock i are
  sum_j w_ij s_jt  and  sum_j w_ij n_jt   (with w = 1: the other stocks' news, equally weighted).
Similarities: equal; same industry (TWSE / TPEx industry code, dataset/profiles.csv); comovement, the cosine of
stock_model.comovement_embedding (dim 16) of the daily log returns log(c_t / c_{t-1}), for fold Y fitted on the years
before Y - 1 only; profile, the cosine of the centred bge-m3 embeddings of the company descriptions of
demo/fetch_profiles.py; cosines below 0 are set to 0. Each also with the stocks permuted (the same similarity matrix
assigned to the wrong stocks) as a control, which keeps its distribution but not the match of stocks.
Model: ridge, walk-forward: for year Y fitted on the stock-days before Y, alpha chosen on Y - 1 from the fit before
Y - 1. Pooled 2022 - 2026: out-of-sample R2 against the training mean, correlation, bootstrap standard errors over days.

    HF_HOME=<models> PYLIB=<transformers dir> python demo/stock_id_probe.py [pre]
      (pre: g_hat also from the 4 pre-open features of stock_overnight.preopen_features, E3c)
"""

import json
import os
import sys
from pathlib import Path

if os.environ.get("PYLIB"):
    sys.path.insert(0, os.environ["PYLIB"])

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from news_probe import ALPHAS, DEV, stats  # noqa: E402
from news_score import items  # noqa: E402
from news_struct import load_struct  # noqa: E402
from stock_model import comovement_embedding  # noqa: E402
from stock_news_gap import walk  # noqa: E402
from stock_poly import ROOT, load  # noqa: E402
from stock_recurrent import base_features  # noqa: E402
from stock_walkforward import EVAL, FIRST, Ridge  # noqa: E402


def cos_pos(E):
    E = E - E.mean(0)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    W = np.clip(E @ E.T, 0, None)
    np.fill_diagonal(W, 0)
    return W


def ridge_folds(Zf, y, yr):
    """walk-forward ridge; Zf(Y) gives the design for fold Y (rows of all years)."""
    yt = torch.tensor(y, dtype=torch.float64, device=DEV)
    out = np.full(len(y), np.nan)
    for Y in EVAL:
        Zt = torch.tensor(Zf(Y), dtype=torch.float64, device=DEV)

        def fit(m):
            mu, sd = Zt[m].mean(0), Zt[m].std(0) + 1e-9
            X = (Zt[m] - mu) / sd
            c = yt[m].mean()
            G, b = X.T @ X, X.T @ (yt[m] - c)
            I = torch.eye(G.shape[0], dtype=G.dtype, device=DEV)
            return lambda a, at: ((Zt[at] - mu) / sd) @ torch.linalg.solve(G + a * len(X) * I, b) + c

        tr_s, va, tr, te = [torch.tensor(v, device=DEV) for v in (yr < Y - 1, yr == Y - 1, yr < Y, yr == Y)]
        f = fit(tr_s)
        a = min(ALPHAS, key=lambda a: (f(a, va) - yt[va]).pow(2).mean().item())
        out[te.cpu().numpy()] = fit(tr)(a, te).cpu().numpy()
    return out


def main() -> None:
    dates, r, traded, gap, known = load()
    T, S = r.shape
    stocks = json.loads((ROOT / "info.json").read_text())["Top100@Y"]
    X, y = base_features(dates, r, traded, gap, known, stocks)
    if "pre" in sys.argv:  # E3c: g_hat also from the pre-open features
        from stock_overnight import preopen_features
        X = np.concatenate([X, preopen_features(dates, gap, traded)], -1)
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    g = gap.to_numpy()
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y) & np.isfinite(g)
    ghat, _ = walk(X, g, ok, year)
    for yy in np.unique(year[ok & (year < EVAL[0])]):  # forecasts for the training years: leave one year out
        fit_, at = ok & (year != yy) & (year < EVAL[0]), ok & (year == yy)
        m = Ridge(X[fit_], g[fit_])
        ghat[at] = m.predict(X[at], m.weights(1e4))
    ag = np.abs(np.nan_to_num(np.where(traded.to_numpy(), g, np.nan)))
    roll = lambda w: pd.DataFrame(ag).rolling(w, min_periods=1).mean().shift(1).to_numpy()
    VOL = np.stack([roll(5), roll(20), roll(60), np.abs(np.nan_to_num(ghat))], -1)

    # news per stock-day: direction of the struct fields, summed over the overnight items
    f = items()
    z = np.load(ROOT / "news" / "scores" / "items.npz")
    assert len(f) == len(z["ts"]) and (f.stock.to_numpy().astype(str) == z["stock"]).all()
    st = load_struct()
    assert len(st["index"]) == len(f)
    s_item = st["p_direction"][:, 0] - st["p_direction"][:, 1]
    ts = pd.DatetimeIndex(z["ts"])
    cut = (dates + pd.Timedelta(hours=8, minutes=50)).to_numpy()
    d = np.searchsorted(cut, ts.to_numpy(), "left")
    keep = (d > 0) & (d < T)
    prev_close = np.r_[np.datetime64("NaT", "s"), (dates[:-1] + pd.Timedelta(hours=13, minutes=30)).to_numpy()]
    over = np.zeros(len(d), bool)
    over[keep] = ts.to_numpy()[keep] > prev_close[d[keep]]
    sidx = {c: k for k, c in enumerate(stocks)}
    j = np.array([sidx.get(c, -1) for c in z["stock"]])
    own = keep & over & (j >= 0)
    Sn, Nn = np.zeros((T, S)), np.zeros((T, S))
    np.add.at(Sn, (d[own], j[own]), s_item[own])
    np.add.at(Nn, (d[own], j[own]), 1.0)
    Nn = (Nn > 0).astype(float)

    # sample: no own item, some other stock has one
    other = Nn.sum(1, keepdims=True) - Nn > 0
    samp = ok & (Nn == 0) & other
    dd, ii = np.nonzero(samp)
    yr = dates.year.to_numpy()[dd]
    ev = yr >= EVAL[0]
    print(f"{len(dd)} stock-days without own news, {ev.sum()} in 2022-2026; "
          f"other stocks with news per day: mean {Nn[np.unique(dd)].sum(1).mean():.1f}", flush=True)

    # similarities
    prof = pd.read_csv(ROOT / "profiles.csv", dtype={"code": str}).set_index("code").reindex([str(c) for c in stocks])
    assert prof.desc.notna().all()
    ind = prof.industry.to_numpy()
    W_ind = (ind[:, None] == ind[None, :]).astype(float)
    np.fill_diagonal(W_ind, 0)
    from news_score import embeddings
    W_prof = cos_pos(embeddings(list(prof.desc), "cuda").astype(np.float64))
    W_eq = 1.0 - np.eye(S)
    lc = (np.log1p(r.where(traded).to_numpy()) + np.where(traded.to_numpy(), g, np.nan))  # log(c_t / c_{t-1})
    W_co = {Y: cos_pos(comovement_embedding(lc[(dates.year < Y - 1) & (np.arange(T) >= 1)], 16).astype(np.float64))
            for Y in EVAL}
    perm = np.random.default_rng(0).permutation(S)
    P = lambda W: W[perm][:, perm]
    for name, W in (("industry", W_ind), ("profile", W_prof), ("comovement 2022", W_co[2022])):
        off = W[~np.eye(S, dtype=bool)]
        print(f"{name:16s} mean w {off.mean():.3f}, share > 0 {np.mean(off > 0):.2f}; "
              f"corr with comovement 2026 {np.corrcoef(off, W_co[2026][~np.eye(S, dtype=bool)])[0, 1]:+.3f}", flush=True)

    def feats(W):
        A = Sn[dd] @ W.T   # A[k, i] = sum_j W[i, j] S[d_k, j]
        B = Nn[dd] @ W.T
        return np.stack([A[np.arange(len(dd)), ii], B[np.arange(len(dd)), ii]], 1)

    fixed = {n: feats(W) for n, W in (("equal", W_eq), ("industry", W_ind), ("profile", W_prof),
                                       ("industry, permuted", P(W_ind)), ("profile, permuted", P(W_prof)))}
    co = {Y: feats(W_co[Y]) for Y in EVAL}
    co_p = {Y: feats(P(W_co[Y])) for Y in EVAL}
    REPS = {
        "equal (all other stocks)": lambda Y: fixed["equal"],
        "industry": lambda Y: fixed["industry"],
        "industry, permuted": lambda Y: fixed["industry, permuted"],
        "comovement": lambda Y: co[Y],
        "comovement, permuted": lambda Y: co_p[Y],
        "profile bge-m3": lambda Y: fixed["profile"],
        "profile, permuted": lambda Y: fixed["profile, permuted"],
        "comovement + profile": lambda Y: np.concatenate([co[Y], fixed["profile"]], 1),
        "equal + comovement + profile": lambda Y: np.concatenate([fixed["equal"], co[Y], fixed["profile"]], 1),
        "equal + industry + comovement + profile": lambda Y: np.concatenate(
            [fixed["equal"], fixed["industry"], co[Y], fixed["profile"]], 1),
    }

    rng = np.random.default_rng(0)
    day = dd[ev]
    yabs = ag[dd, ii]
    vb = VOL[dd, ii]
    base_v = ridge_folds(lambda Y: vb, yabs, yr)
    pre = ~ev
    R = Ridge(vb[pre], yabs[pre])
    base_v[pre] = R.predict(vb[pre], R.weights(1.0))
    targets = {"g - g_hat (price)": (g - ghat)[dd, ii], "|g| - |g|_hat (vol)": yabs - base_v}
    for tname, tv in targets.items():
        print(f"\n== target {tname}", flush=True)
        bm = np.array([tv[yr < Y].mean() for Y in EVAL])[yr[ev] - EVAL[0]]
        for name, Zf in REPS.items():
            p = ridge_folds(Zf, tv, yr)[ev]
            r2, r2s, c, cs = stats(p, tv[ev], bm, day, rng)
            print(f"{name:40s} R2 {r2:+.5f} ± {r2s:.5f}  corr {c:+.3f} ± {cs:.3f}", flush=True)


if __name__ == "__main__":
    main()
