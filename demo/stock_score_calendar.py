"""E4b: calendar effects, stock intercepts and robust targets in the return ridge, judged on the sign score (TWStock).

Reference: the ridge of r (open-to-close log return) on the 52 base features + the gap block of g_hat_pre (E4a ref).
The always-down baseline lives on the negative mean of r, i.e. on the market part of the Score, which carries most of
its value; day-level effects of the calendar act on that part, stock intercepts on the cross part. Versions:
- + stock one-hot (100): one intercept per stock;
- + day of week (5 one-hot);
- + calendar: day of week, month (12), turn of the month (last trading day, first two trading days), TAIEX futures
  settlement day (third Wednesday) and the day after it, the trading days before and after a break of 4 or more
  calendar days (long holidays);
- + calendar + stock one-hot;
- the same calendar + stock columns as a second ridge on the residual of the reference (its own alpha, chosen on Y - 1;
  the shared alpha of one ridge may shrink a few strong flags as much as the 52 noisy features);
- target winsorized at its 1% / 99% training quantiles (inputs of the reference);
- target winsorized at 5% / 95%.
Alpha chosen on Y - 1 (MSE of the raw r), stage-2 rows by half-year blocks out (demo/stock_identity_news.py).
Usage: python demo/stock_score_calendar.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_identity_news import struct_features  # noqa: E402
from stock_itrans_news import U as UNIT, ghat_for  # noqa: E402
from stock_news_attn import report  # noqa: E402
from stock_news_attn2 import oos_mu  # noqa: E402
from stock_news_gap import gap_block  # noqa: E402
from stock_overnight import preopen_features  # noqa: E402
from stock_poly import ROOT, load  # noqa: E402
from stock_recurrent import base_features  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stock_model  # noqa: E402


class Ridge(stock_model.Ridge):
    """stock_model.Ridge, but columns that are constant on the fit rows (e.g. the one-hot of a stock not yet listed)
    get weight 0 instead of an exploding standardization."""

    def __init__(self, X, y):
        sd = X.std(0)
        self.mu, self.sd = X.mean(0), np.where(sd < 1e-8, np.inf, sd)
        self.c0 = y.mean()
        z = (X - self.mu) / self.sd
        self.gram, self.rhs = z.T @ z, z.T @ (y - self.c0)


stock_model.Ridge = Ridge   # used by stock_model.block_out_of_sample (oos_mu)


def calendar(dates):
    """(T, k) day-level calendar flags and their names."""
    d = pd.DatetimeIndex(dates)
    T = len(d)
    cols, names = [], []
    for k in range(5):
        cols.append(d.dayofweek == k); names.append(f"dow{k}")
    for m in range(1, 13):
        cols.append(d.month == m); names.append(f"m{m}")
    ym = d.year * 12 + d.month
    last = np.r_[ym[1:] != ym[:-1], True]
    first = np.r_[True, ym[1:] != ym[:-1]]
    second = np.r_[False, first[:-1]]
    cols += [last, first, second]; names += ["month last", "month first", "month second"]
    third_wed = (d.dayofweek == 2) & (d.day >= 15) & (d.day <= 21)
    # if the third Wednesday is a holiday, settlement moves to the next trading day: first trading day after it
    settle = np.zeros(T, bool)
    for y_, m_ in sorted(set(zip(d.year, d.month))):
        tw = pd.Timestamp(year=y_, month=m_, day=15)
        tw = tw + pd.Timedelta(days=(2 - tw.dayofweek) % 7)
        i = np.searchsorted(d.values, tw.to_datetime64())
        if i < T and d[i].year == y_ and d[i].month == m_:
            settle[i] = True
    cols += [settle, np.r_[False, settle[:-1]]]; names += ["settlement", "after settlement"]
    gapdays = np.r_[1, np.diff(d.values).astype("timedelta64[D]").astype(int)]
    post = gapdays >= 4
    pre = np.r_[post[1:], False]
    cols += [pre, post]; names += ["before break", "after break"]
    return np.stack(cols, 1).astype(float), names


def main() -> None:
    dates, r, traded, gap, known = load()
    T, S = r.shape
    info = json.loads((ROOT / "info.json").read_text())
    stocks, top50 = info["Top100@Y"], set(info["Top50@Y"])
    X, y = base_features(dates, r, traded, gap, known, stocks)
    STR = struct_features(dates, stocks)
    has = (STR[..., 0] + STR[..., 1]) > 0
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    half = year * 2 + (np.repeat(dates.month.to_numpy()[:, None], S, axis=1) > 6)
    g = gap.to_numpy()
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y) & np.isfinite(g)
    XG = np.concatenate([X, preopen_features(dates, gap, traded)], -1)
    C, cn = calendar(dates)
    print("calendar flags, share of days 2020-2026: " + ", ".join(f"{n} {v:.3f}" for n, v in zip(cn, C.mean(0))))
    CAL = np.repeat(C[:, None, :], S, 1)
    DOW = CAL[..., :5]
    SID = np.repeat(np.eye(S)[None], T, 0)
    # mean r per calendar flag before 2022 (descriptive)
    pre22 = ok & (year < 2022)
    print("mean r (e-4) on flagged stock-days before 2022 / all: " + ", ".join(
        f"{n} {1e4 * y[pre22 & (CAL[..., k] > 0)].mean():+.1f}" for k, n in enumerate(cn)) +
        f" / all {1e4 * y[pre22].mean():+.1f}", flush=True)

    ev = ok & np.isin(year, EVAL)
    is50 = np.array([s in top50 for s in stocks])[None, :]
    subsets = (("all 100 stocks", ev), ("the 50-stock subset", ev & is50),
               ("100 stocks, days with an own overnight item", ev & has),
               ("100 stocks, 2025-2026", ev & (year >= 2025)))
    fm = lambda t: f"{UNIT * t[0]:+5.1f}e-4 ± {UNIT * t[1]:3.1f}e-4"
    V = {"ref: 52 + gap block of g_hat_pre": (None, None), "+ stock one-hot": (SID, None), "+ day of week": (DOW, None),
         "+ calendar": (CAL, None), "+ calendar + stock one-hot": (np.concatenate([CAL, SID], -1), None),
         "residual ridge on calendar + stock": ("stage", np.concatenate([CAL, SID], -1)),
         "residual ridge on calendar": ("stage", CAL),
         "target winsorized 1%": (None, 0.01), "target winsorized 5%": (None, 0.05)}
    res = {k: np.full((T, S), np.nan) for k in V}
    for Y in EVAL:
        Bs, Br = gap_block(ghat_for(XG, g, ok, year, Y - 1), ok), gap_block(ghat_for(XG, g, ok, year, Y), ok)
        inner, sel, te = ok & (year < Y - 1), ok & (year == Y - 1), ok & (year == Y)
        mu_ref_s = mu_ref_r = None
        for k, (extra, opt) in V.items():
            tgt = y
            if isinstance(opt, float):
                lo, hi = np.quantile(y[ok & (year < Y)], [opt, 1 - opt])
                tgt = np.clip(y, lo, hi)
            if extra is None or isinstance(extra, str):
                Xs, Xr = np.concatenate([X, Bs], -1), np.concatenate([X, Br], -1)
            else:
                Xs, Xr = np.concatenate([X, Bs, extra], -1), np.concatenate([X, Br, extra], -1)
            if isinstance(extra, str):     # second ridge on the residual of the reference
                Z = opt
                e_s = y - mu_ref_s
                m = Ridge(Z[inner], e_s[inner])
                a = min(np.r_[ALPHAS, 1e8, 1e9, 1e10], key=lambda a: np.mean((mu_ref_s[sel] + m.predict(Z[sel], m.weights(a)) - y[sel]) ** 2))
                e_r = y - mu_ref_r
                res[k][te] = (mu_ref_r + oos_mu(Z, np.where(ok, e_r, 0.0), ok, year, half, a, Y))[te]
                print(f"{Y} {k}: alpha {a:.0e}", flush=True)
                continue
            m = Ridge(Xs[inner], tgt[inner])
            a = min(ALPHAS, key=lambda a: np.mean((m.predict(Xs[sel], m.weights(a)) - y[sel]) ** 2))
            mu_r = oos_mu(Xr, tgt, ok, year, half, a, Y)
            res[k][te] = mu_r[te]
            if mu_ref_s is None:
                mu_ref_r = mu_r
                mu_ref_s = oos_mu(Xs, y, ok, year, half, a, Y - 1)
        print(f"{Y} done", flush=True)
    print("\n######## the return ridge with calendar / stock columns and robust targets vs the reference")
    names = list(V)
    report(res, {k: names[0] for k in names[1:]}, y, subsets, fm)


if __name__ == "__main__":
    main()
