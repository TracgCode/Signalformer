"""E2e: the same-day opening gap as a feature of the open-to-close return, and a news-predicted gap in its place,
walk-forward (TWStock).

g_t = log(o_t / c_{t-1}) is not known at 08:50; the rows with it are an upper bound (what a perfect gap forecast,
e.g. from the 08:30 - 09:00 pre-open match, could give). g_hat replaces it with a ridge forecast of g_t from what is
known before 09:00: either the 52 base features only (control: linear in X, so it adds nothing to a linear model), or
the base plus the news features of demo/stock_news.py (LLM + embedding). Stage 1 (the gap ridge) is walk-forward as
well: for test year Y it is fitted on the years < Y (alpha chosen on Y - 1 from the fit on < Y - 1); on the training
years g_hat comes from the fit without that year, so stage 2 never sees an in-sample g_hat. Added per stock-day: g,
|g| and the day's cross-sectional mean of g (the market gap); the same for g_hat.

    python demo/stock_news_gap.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_news import text_features  # noqa: E402
from stock_poly import ROOT, load  # noqa: E402
from stock_recurrent import base_features  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge, clustered_mean  # noqa: E402,F401
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stock_model import gap_block, walk_forward  # noqa: E402,F401

U = 1e4


def walk(Z, y, ok, year, loo=False):
    """walk-forward ridge forecasts on the EVAL years (stock_model.walk_forward)."""
    return walk_forward(Z, y, ok, year, EVAL, ALPHAS, loo)


def main() -> None:
    dates, r, traded, gap, known = load()
    T, S = r.shape
    info = json.loads((ROOT / "info.json").read_text())
    stocks, top50 = info["Top100@Y"], set(info["Top50@Y"])
    X, y = base_features(dates, r, traded, gap, known, stocks)
    LLM, _, EMB, has = text_features(dates, stocks)
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    g = gap.to_numpy()
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y) & np.isfinite(g)
    Xn = np.concatenate([X, LLM, EMB], -1)

    # stage 1: gap forecasts
    ghat, R2g = {}, {}
    for name, Z in (("price", X), ("price + news", Xn)):
        mu, oos = walk(Z, g, ok, year, loo=True)
        ghat[name] = (mu, oos)
        ev = ok & np.isin(year, EVAL)
        R2g[name] = 1 - np.mean((mu[ev] - g[ev]) ** 2) / np.mean((g[ev] - g[ev].mean()) ** 2)
        print(f"gap forecast from {name}: walk-forward R2 {R2g[name]:+.4f} (all), "
              f"{1 - np.mean((mu[ev & has] - g[ev & has]) ** 2) / np.var(g[ev & has]):+.4f} (item days)", flush=True)
    ev = ok & np.isin(year, EVAL)
    print(f"corr(g_t, r_t) pooled 2022-26: {np.corrcoef(g[ev], y[ev])[0, 1]:+.4f}", flush=True)

    # stage 2
    SETS = {"base": None, "+ realized g (upper bound)": "real",
            "+ g_hat price (control)": "price", "+ g_hat price + news": "price + news",
            "+ news features directly": "direct"}
    res = {}
    for name, kind in SETS.items():
        if kind is None:
            res[name], _ = walk(X, y, ok, year)
            continue
        if kind == "real":
            res[name], _ = walk(np.concatenate([X, gap_block(g, ok)], -1), y, ok, year)
            continue
        if kind == "direct":
            res[name], _ = walk(Xn, y, ok, year)
            continue
        mu = np.full((T, S), np.nan)
        for Y in EVAL:  # stage-2 inputs for test year Y: g_hat out of sample on every year it is fitted / applied to
            G = ghat[kind][1][Y]
            Z = np.concatenate([X, gap_block(G, ok & np.isfinite(G))], -1)
            okY = ok & np.isfinite(G)
            inner, sel, rows, te = okY & (year < Y - 1), okY & (year == Y - 1), okY & (year < Y), okY & (year == Y)
            m = Ridge(Z[inner], y[inner])
            a = min(ALPHAS, key=lambda a: np.mean((m.predict(Z[sel], m.weights(a)) - y[sel]) ** 2))
            m = Ridge(Z[rows], y[rows])
            mu[te] = m.predict(Z[te], m.weights(a))
        res[name] = mu

    is50 = np.array([s in top50 for s in stocks])[None, :]
    fm = lambda t: f"{U * t[0]:+5.1f}e-4 ± {U * t[1]:3.1f}e-4"
    for uname, um in (("all 100 stocks", ev), ("the 50-stock subset", ev & is50),
                      ("100 stocks, days with an own overnight item", ev & has),
                      ("100 stocks, 2025-2026", ev & (year >= 2025))):
        um = um & np.all([np.isfinite(v) for v in res.values()], 0)
        y_, d_ = y[um], np.nonzero(um)[0]
        mb = res["base"][um]
        sb = np.where(mb > 0, 1.0, -1.0)
        base = np.mean((y_ - y_.mean()) ** 2)
        print(f"\n== walk-forward, {uname} ({um.sum()} stock-days); always down: score {fm(clustered_mean(-y_, d_))}")
        for name, mu in res.items():
            p = mu[um]
            sg = np.where(p > 0, 1.0, -1.0)
            dm = clustered_mean((mb - y_) ** 2 - (p - y_) ** 2, d_)
            print(f"{name:28s} R2 {1 - np.mean((p - y_) ** 2) / base:+.4f}  score {fm(clustered_mean(sg * y_, d_))}  "
                  f"| vs base: flips {np.mean(sg != sb):5.1%}  R2 {dm[0] / base:+.4f} ± {dm[1] / base:.4f}  "
                  f"score {fm(clustered_mean((sg - sb) * y_, d_))}", flush=True)


if __name__ == "__main__":
    main()
