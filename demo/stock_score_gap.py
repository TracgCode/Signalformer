"""E4a: the parts of the opening-gap forecast as inputs of the return ridge, judged on the sign score (TWStock).

E3c found a gap forecast with twice the R2 of the E2 one (g_hat_pre: the 52 base features + the 4 pre-open features of
stock_overnight.preopen_features), a per-stock ridge on the pre-open betas and the 46 US returns for its residual
(+0.0185) and a Taiwan news signal on top (+0.0041). E2j step 0 showed that the better g_hat_pre barely moves the
Score (14.5e-4 -> 14.7e-4). Here the gap forecast is split into its parts, each with its own coefficient in the ridge
of r (the open-to-close log return), since the intraday reversal can differ by the source of the gap:
- P: g_hat from the 52 price features (E2e "price", the control);
- O: g_hat_pre - P (the pre-open part);
- U: the per-stock ridge of g - g_hat_pre on [US basket, Nikkei / KOSPI / futures gaps] + the 46 US returns (E3c r7);
- N: a pooled ridge of g - g_hat_pre - U on the own struct news features of E2j (43 columns).
All parts are out-of-sample on every row: on year Y from the fits on the years before Y, on each earlier year from the
fit without that year (demo/stock_itrans_news.ghat_for). Versions of the ridge of r (52 base features + the blocks,
alpha chosen on Y - 1, stage-2 rows by half-year blocks out, as demo/stock_identity_news.py):
- ref: gap block (value, |value|, day mean) of g_hat_pre (E2j step 0 pre, the reference);
- total: gap block of g_hat_pre + U + N;
- parts: value and day mean of each of P, O, U, N, and |total|;
- parts, own minus day: the same with each value replaced by value - day mean (same span; a check of the solver);
- total + total * |total|: a convex term (the reversal may grow with the size of the gap);
- parts + raw pre-open: also the 4 pre-open features themselves.
Usage: python demo/stock_score_gap.py
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
from stock_overnight import preopen_features, taifex_features, yahoo_features  # noqa: E402
from stock_poly import ROOT, load  # noqa: E402
from stock_recurrent import base_features  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge  # noqa: E402


def us_returns(dates):
    """(T, 46) US close-to-close log returns since the Taiwan close of t - 1, scaled by their sd before 2022."""
    us = sorted(p.stem for p in (ROOT / "peers" / "us").glob("*.csv"))
    pre = np.asarray(dates < "2022-01-01")
    cols = []
    for sym in us:
        f = pd.read_csv(ROOT / "peers" / "us" / f"{sym}.csv", parse_dates=["date"]).set_index("date")
        lc = np.log(f["adjclose"]).dropna()
        idx = np.searchsorted(lc.index.values, dates.values, side="left") - 1
        lev = np.where(idx >= 0, lc.values[np.clip(idx, 0, None)], np.nan)
        cols.append(np.r_[np.nan, np.diff(lev)])
    Rn = np.stack(cols, 1)
    return np.nan_to_num(Rn / np.nanstd(Rn[pre], 0))


def perstock_for(F, tgt, ok, year, Yb):
    """per-stock ridge of tgt on F (T, k) shared by all stocks, alpha shared and chosen on Yb - 1; forecasts on Yb from
    the fit on the years before Yb, on each earlier year from the fit without that year."""
    T, S = tgt.shape
    k = F.shape[1]

    def coefs(trm):
        out = {}
        for s in range(S):
            m = trm[:, s]
            if m.sum() < 50:
                continue
            Xs, ys = F[m], tgt[m, s]
            mu, ym = Xs.mean(0), ys.mean()
            Xc = Xs - mu
            out[s] = (Xc.T @ Xc, Xc.T @ (ys - ym), mu, ym, len(Xs))
        return out

    def predict(C, a, rows):
        P = np.full((T, S), np.nan)
        for s, (A, b, mu, ym, n) in C.items():
            w = np.linalg.solve(A + a * n * np.eye(k), b)
            r_ = rows[:, s]
            P[r_, s] = ym + (F[r_] - mu) @ w
        return P

    C = coefs(ok & (year < Yb - 1))
    sel = ok & (year == Yb - 1)
    a = min(ALPHAS, key=lambda a: np.nanmean((predict(C, a, sel) - tgt)[sel] ** 2))
    G = predict(coefs(ok & (year < Yb)), a, ok & (year == Yb))
    for yy in np.unique(year[ok & (year < Yb)]):
        rows = ok & (year < Yb)
        G2 = predict(coefs(rows & (year != yy)), a, rows & (year == yy))
        at = rows & (year == yy)
        G[at] = G2[at]
    return np.nan_to_num(G)


def day_mean(v, ok):
    return np.nan_to_num(np.nanmean(np.where(ok, v, np.nan), 1, keepdims=True)).repeat(v.shape[1], 1)


def main() -> None:
    dates, r, traded, gap, known = load()
    T, S = r.shape
    info = json.loads((ROOT / "info.json").read_text())
    stocks, top50 = info["Top100@Y"], set(info["Top50@Y"])
    X, y = base_features(dates, r, traded, gap, known, stocks)
    STR = struct_features(dates, stocks)
    own = STR[..., :43]
    has = (STR[..., 0] + STR[..., 1]) > 0
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    half = year * 2 + (np.repeat(dates.month.to_numpy()[:, None], S, axis=1) > 6)
    g = gap.to_numpy()
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y) & np.isfinite(g)
    trd = traded.to_numpy()
    PO = preopen_features(dates, gap, traded)
    XG = np.concatenate([X, PO], -1)
    Rn = us_returns(dates)
    closes = pd.read_csv(ROOT / "market" / "2330.csv", parse_dates=["date"]).set_index("date")["etl:adj_close"]
    yh = yahoo_features(dates, closes.reindex(dates).to_numpy())
    tx = taifex_features(dates)
    basket = Rn.mean(1)
    basket = basket / basket[np.asarray(dates < "2022-01-01")].std()
    PB = np.nan_to_num(np.c_[basket, yh["n225_gap"].to_numpy(), yh["ks11_gap"].to_numpy(), tx["fut_gap"].to_numpy()])
    FU = np.c_[PB, Rn]

    ev = ok & np.isin(year, EVAL)
    is50 = np.array([s in top50 for s in stocks])[None, :]
    subsets = (("all 100 stocks", ev), ("the 50-stock subset", ev & is50),
               ("100 stocks, days with an own overnight item", ev & has),
               ("100 stocks, 2025-2026", ev & (year >= 2025)))
    fm = lambda t: f"{UNIT * t[0]:+5.1f}e-4 ± {UNIT * t[1]:3.1f}e-4"
    names = ["ref: gap block of g_hat_pre", "total: gap block of g_hat_pre + U + N", "parts P, O, U, N",
             "parts, own minus day", "total + total*|total|", "parts + raw pre-open"]
    res = {k: np.full((T, S), np.nan) for k in names}
    comp = {k: np.full((T, S), np.nan) for k in ("P", "Pre", "U", "N")}
    okt = ok & trd
    for Y in EVAL:
        def parts(Yb):
            P_ = ghat_for(X, g, ok, year, Yb)
            Pre = ghat_for(XG, g, ok, year, Yb)
            t1 = np.where(okt, g - Pre, 0.0)
            U_ = perstock_for(FU, t1, okt & np.isfinite(Pre), year, Yb)
            t2 = np.where(okt, g - Pre - U_, 0.0)
            N_ = np.nan_to_num(ghat_for(own, t2, okt & np.isfinite(Pre), year, Yb))
            return np.nan_to_num(P_), np.nan_to_num(Pre), U_, N_

        def blocks(P_, Pre, U_, N_):
            tot = Pre + U_ + N_
            pp = [P_, Pre - P_, U_, N_]
            B = {names[0]: gap_block(Pre, ok), names[1]: gap_block(tot, ok)}
            dm = [day_mean(v, ok) for v in pp]
            B[names[2]] = np.stack(pp + dm + [np.abs(tot)], -1)
            B[names[3]] = np.stack([v - d for v, d in zip(pp, dm)] + dm + [np.abs(tot)], -1)
            B[names[4]] = np.concatenate([gap_block(tot, ok), (tot * np.abs(tot))[..., None] * 100], -1)
            B[names[5]] = np.concatenate([B[names[2]], PO], -1)
            return B

        cY = parts(Y)
        Bs, Br = blocks(*parts(Y - 1)), blocks(*cY)
        te = ok & (year == Y)
        for k, v in zip(comp, cY):
            comp[k][te] = v[te]
        inner, sel = ok & (year < Y - 1), ok & (year == Y - 1)
        for k in names:
            Xa_s, Xa_r = np.concatenate([X, Bs[k]], -1), np.concatenate([X, Br[k]], -1)
            m = Ridge(Xa_s[inner], y[inner])
            a = min(ALPHAS, key=lambda a: np.mean((m.predict(Xa_s[sel], m.weights(a)) - y[sel]) ** 2))
            res[k][te] = oos_mu(Xa_r, y, ok, year, half, a, Y)[te]
        print(f"{Y} done", flush=True)

    e = ev & trd
    gm = np.array([g[ok & (year < Y)].mean() for Y in EVAL])[year[e] - EVAL[0]]
    base = np.mean((g[e] - gm) ** 2)
    P_, Pre, U_, N_ = (comp[k][e] for k in comp)
    for nm, f in (("P (price)", P_), ("g_hat_pre", Pre), ("g_hat_pre + U", Pre + U_), ("g_hat_pre + U + N", Pre + U_ + N_)):
        print(f"R2 of the gap, {nm:18s}: {1 - np.mean((g[e] - f) ** 2) / base:+.4f}", flush=True)
    for nm, f in (("g (realized)", g[e]), ("P", P_), ("O = g_hat_pre - P", Pre - P_), ("U", U_), ("N", N_), ("g - g_hat_pre - U - N", g[e] - Pre - U_ - N_)):
        print(f"corr with r: {nm:22s} {np.corrcoef(f, y[e])[0, 1]:+.4f}", flush=True)
    print("\n######## the return ridge with each gap input vs the reference")
    report(res, {k: names[0] for k in names[1:]}, y, subsets, fm)


if __name__ == "__main__":
    main()
