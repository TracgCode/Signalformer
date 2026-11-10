"""Overnight and pre-open features added to the ridge, walk-forward (TWStock).

Data: demo/fetch_overnight.py (dataset/overnight/, not committed). For Taiwan trading day t (the forecast is made before
the 09:00 open of day t):

TAIFEX TAIEX futures (TX), the contract month that traded most in the day session of t - 1 (known at t - 1):
- fut_gap = log(day-session open of t / day-session close of t - 1); the futures open at 08:45, 15 minutes before
  the stock market opens (the most recent price available);
- night_ret = log(after-hours close / day close of t - 1), the after-hours session that ends at 05:00 of t;
- open_vs_night = log(day open of t / after-hours close);
- night_vol = log after-hours volume minus its mean over the previous 20 sessions;
- beta interactions: each of fut_gap and night_ret times the stock's gap beta, the least-squares slope of the stock's
  overnight gap on the futures' overnight gap over the previous 60 days (so the market-wide signal can rank stocks).
Yahoo (the last bar of each US market dated before t; the Asian opens of day t itself, at 08:00 Taiwan time):
- tsm_ret, sox_ret, nq_ret: log close-to-close of TSM ADR, ^SOX, Nasdaq-100 futures;
- adr_prem_chg: daily change of the TSM ADR premium, TSM x USD/TWD / (5 x 2330 close) - 1;
- n225_gap, ks11_gap: log(open of t / close of the previous session) of Nikkei 225 and KOSPI (0 on their holidays);
- own_adr_ret: the stock's own ADR return for 2330 (TSM), 2303 (UMC), 3711 (ASX), 0 for the other stocks.

Models: the walk-forward ridge of demo/stock_extra.py (alpha on the previous year, refitted on all earlier years) on
poly p0 + range (47 features) plus groups of the new features; sign(mu). Reported pooled 2022 - 2026: R2, the score,
its market and cross-sectional parts, the paired difference to the base (day-clustered), on all 100 stocks and on the
50-stock subset.

    python demo/stock_overnight.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_extra import extra_features, signals  # noqa: E402
from stock_poly import ROOT, features, load  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge, clustered_mean  # noqa: E402

OV = ROOT / "overnight"
U = 1e4
ADR = {"2330": "TSM", "2303": "UMC", "3711": "ASX"}


def taifex_features(dates):
    f = pd.read_csv(OV / "taifex_tx.csv", dtype=str)
    f.columns = [c.strip() for c in f.columns]
    f = f[f["契約"].str.strip() == "TX"].copy()
    f["month"] = f["到期月份(週別)"].str.strip()
    f = f[f["month"].str.fullmatch(r"\d{6}")]
    f["date"] = pd.to_datetime(f["交易日期"].str.strip())
    f["session"] = f["交易時段"].str.strip()
    for c, n in (("開盤價", "open"), ("收盤價", "close"), ("成交量", "volume")):
        f[n] = pd.to_numeric(f[c].str.strip(), errors="coerce")
    day = f[f["session"] == "一般"].set_index(["date", "month"])
    night = f[f["session"] == "盤後"].set_index(["date", "month"])
    tdays = sorted(day.index.get_level_values(0).unique())
    prev = {d: p for p, d in zip(tdays[:-1], tdays[1:])}
    # which day does an after-hours row dated D belong to? compare its open with the day closes of D and of prev(D)
    chk = []
    for (d, mth), row in night.iterrows():
        if d in prev and (prev[d], mth) in day.index and (d, mth) in day.index:
            chk.append((abs(row["open"] - day.loc[(prev[d], mth), "close"]), abs(row["open"] - day.loc[(d, mth), "close"])))
    chk = np.array(chk)
    starts_after_prev = np.nanmedian(chk[:, 0]) < np.nanmedian(chk[:, 1])
    print(f"after-hours rows: median |open - day close of the previous day| {np.nanmedian(chk[:, 0]):.0f}, "
          f"|open - day close of the same date| {np.nanmedian(chk[:, 1]):.0f} -> a row dated D is the session "
          f"{'from the evening of D - 1 to 05:00 of D' if starts_after_prev else 'from the evening of D to 05:00 of D + 1'}")
    near = day["volume"].groupby(level=0).idxmax().map(lambda ix: ix[1])  # most traded month per day
    out = pd.DataFrame(index=pd.DatetimeIndex(dates), columns=["fut_gap", "night_ret", "open_vs_night", "night_lvol"],
                       dtype=float)
    for d in out.index:
        if d not in prev or prev[d] not in near.index:
            continue
        p = prev[d]
        mth = near[p]
        nd = d if starts_after_prev else p  # the after-hours row that ends at 05:00 of d
        try:
            c0 = day.loc[(p, mth), "close"]
            o1 = day.loc[(d, mth), "open"]
        except KeyError:
            continue
        out.loc[d, "fut_gap"] = np.log(o1 / c0)
        if (nd, mth) in night.index:
            n = night.loc[(nd, mth)]
            out.loc[d, "night_ret"] = np.log(n["close"] / c0)
            out.loc[d, "open_vs_night"] = np.log(o1 / n["close"])
            out.loc[d, "night_lvol"] = np.log1p(n["volume"])
    out["night_vol"] = out["night_lvol"] - out["night_lvol"].shift(1).rolling(20, min_periods=5).mean()
    print(f"TAIFEX features: fut_gap on {out['fut_gap'].notna().sum()} days, night on {out['night_ret'].notna().sum()} "
          f"days (first {out['night_ret'].first_valid_index().date()})")
    return out.drop(columns="night_lvol")


def yahoo(sym):
    f = pd.read_csv(OV / f"yahoo_{sym.replace('^', '').replace('=', '_')}.csv", parse_dates=["date"]).set_index("date")
    return f.sort_index()


def last_before(series, dates):
    """For each Taiwan day t, the value of the last bar dated strictly before t (the US session that ended before
    the Taiwan open)."""
    s = series.dropna()
    idx = np.searchsorted(s.index.values, pd.DatetimeIndex(dates).values, side="left") - 1
    v = np.where(idx >= 0, s.values[np.clip(idx, 0, None)], np.nan)
    return pd.Series(v, index=dates)


def yahoo_features(dates, c2330):
    out = pd.DataFrame(index=pd.DatetimeIndex(dates))
    for name, sym in (("tsm_ret", "TSM"), ("sox_ret", "^SOX"), ("nq_ret", "NQ=F"), ("umc_ret", "UMC"),
                      ("asx_ret", "ASX")):
        c = yahoo(sym)["close"]
        out[name] = last_before(np.log(c).diff(), dates)
    tsm, twd = yahoo("TSM")["close"], yahoo("TWD=X")["close"]
    tsm_b, twd_b = last_before(tsm, dates), last_before(twd, dates)
    prem = tsm_b * twd_b / (5 * pd.Series(c2330, index=dates).shift(1)) - 1  # 2330 close of t - 1
    out["adr_prem"] = prem
    out["adr_prem_chg"] = prem.diff()
    for name, sym in (("n225_gap", "^N225"), ("ks11_gap", "^KS11")):
        f = yahoo(sym)
        gap = np.log(f["open"] / f["close"].shift(1))
        out[name] = gap.reindex(out.index).fillna(0.0)  # day t open at 08:00 Taiwan time; 0 on their holidays
    return out


def preopen_features(dates, gap, traded):
    """the pre-open features of the gap base forecast g_hat (T, S, 4), all known before 08:50 of day t: the open gaps of
    the Nikkei 225 and KOSPI (08:00 Taipei) and of the TAIEX futures (08:45), and the futures gap times each stock's gap
    beta on it over the previous 60 days. Appended to stock_recurrent.base_features they raise the walk-forward R2 of
    g_hat 2022-2026 from about .24 to .48."""
    closes = pd.read_csv(ROOT / "market" / "2330.csv", parse_dates=["date"]).set_index("date")["etl:adj_close"]
    yh = yahoo_features(dates, closes.reindex(dates).to_numpy())
    fg = np.nan_to_num(taifex_features(dates)["fut_gap"].to_numpy())
    G = np.where(traded.to_numpy(), gap.to_numpy(), np.nan)
    T, S = G.shape
    beta = np.zeros((T, S))
    for t in range(61, T):
        f_, g_ = fg[t - 60:t], G[t - 60:t]
        fc = f_ - f_.mean()
        beta[t] = np.nansum(fc[:, None] * np.where(np.isfinite(g_), g_ - np.nanmean(g_, 0), 0), 0) / max(
            (fc ** 2).sum(), 1e-12)
    mkt = lambda v: np.repeat(np.nan_to_num(np.asarray(v, float))[:, None], S, 1)
    return np.stack([mkt(yh["n225_gap"]), mkt(yh["ks11_gap"]), mkt(fg), mkt(fg) * beta], -1)


def main() -> None:
    dates, r, traded, gap, known = load()
    T, S = r.shape
    stocks = json.loads((ROOT / "info.json").read_text())["Top100@Y"]
    top50 = set(json.loads((ROOT / "info.json").read_text())["Top50@Y"])
    Xb, y, bnames = features(dates, r, traded, gap, known)
    Xe, _, egroups = extra_features(signals(dates, known, S), np.asarray(dates < "2022-01-01"))
    base = np.concatenate([Xb[..., [n.endswith("p0") for n in bnames]], Xe[..., egroups == "range"]], -1)

    tx = taifex_features(dates)
    closes = pd.read_csv(ROOT / "market" / "2330.csv", parse_dates=["date"]).set_index("date")["etl:adj_close"]
    yh = yahoo_features(dates, closes.reindex(dates).to_numpy())
    # gap beta: slope of the stock's gap on the futures gap over the previous 60 days (rows t - 60 .. t - 1)
    g = gap.where(traded).to_numpy()
    fg = tx["fut_gap"].to_numpy()
    beta = np.full((T, S), np.nan)
    for t in range(61, T):
        a, b = fg[t - 60:t, None], g[t - 60:t]
        ok = np.isfinite(a) & np.isfinite(b)
        am = np.where(ok, a, np.nan)
        ma, mb = np.nanmean(am, 0), np.nanmean(np.where(ok, b, np.nan), 0)
        cov = np.nanmean((am - ma) * (np.where(ok, b, np.nan) - mb), 0)
        var = np.nanmean((am - ma) ** 2, 0)
        beta[t] = cov / np.where(var > 0, var, np.nan)
    beta = np.clip(beta, -1, 3)
    mkt = lambda v: np.repeat(np.asarray(v, float)[:, None], S, 1)
    own = np.zeros((T, S))
    for s, sym in ADR.items():
        own[:, stocks.index(s)] = yh[{"TSM": "tsm_ret", "UMC": "umc_ret", "ASX": "asx_ret"}[sym]].to_numpy()
    new = {"fut_gap": mkt(tx["fut_gap"]), "night_ret": mkt(tx["night_ret"]), "open_vs_night": mkt(tx["open_vs_night"]),
           "night_vol": mkt(tx["night_vol"]), "fut_gap x beta": mkt(tx["fut_gap"]) * beta,
           "night_ret x beta": mkt(tx["night_ret"]) * beta, "beta": beta,
           "tsm_ret": mkt(yh["tsm_ret"]), "sox_ret": mkt(yh["sox_ret"]), "nq_ret": mkt(yh["nq_ret"]),
           "adr_prem_chg": mkt(yh["adr_prem_chg"]), "n225_gap": mkt(yh["n225_gap"]), "ks11_gap": mkt(yh["ks11_gap"]),
           "own_adr_ret": own}
    names = list(new)
    XN = np.stack([new[k] for k in names], -1)
    X = np.concatenate([base, XN], -1).reshape(T * S, -1)
    K0 = base.shape[-1]
    yf = y.reshape(-1)
    dayf, yearf = np.repeat(np.arange(T), S), np.repeat(dates.year.to_numpy(), S)
    is50 = np.tile([s in top50 for s in stocks], T)
    ok = np.repeat(np.arange(T) >= FIRST, S) & np.isfinite(X).all(1) & np.isfinite(yf)
    X, yf, dayf, yearf, is50 = X[ok], yf[ok], dayf[ok], yearf[ok], is50[ok]
    print(f"{ok.sum()} stock-days with every feature ({np.unique(dayf).size} days, from {dates[dayf.min()].date()})")
    print("correlation of the market features with the day's mean open-to-close return (2020-2026):")
    _, inv = np.unique(dayf, return_inverse=True)
    mret = np.bincount(inv, weights=yf) / np.bincount(inv)
    for k in ("fut_gap", "night_ret", "open_vs_night", "night_vol", "tsm_ret", "sox_ret", "nq_ret", "adr_prem_chg",
              "n225_gap", "ks11_gap"):
        v = np.bincount(inv, weights=X[:, K0 + names.index(k)]) / np.bincount(inv)
        print(f"  {k:14s} corr {np.corrcoef(v, mret)[0, 1]:+.3f}")

    col = lambda ks: np.array([True] * K0 + [n in ks for n in names])
    TX = ["fut_gap", "night_ret", "open_vs_night", "night_vol"]
    BETA = ["fut_gap x beta", "night_ret x beta", "beta"]
    YH = ["tsm_ret", "sox_ret", "nq_ret", "adr_prem_chg", "n225_gap", "ks11_gap", "own_adr_ret"]
    SETS = {"base (poly p0 + range)": col([]), "+ fut_gap": col(["fut_gap"]), "+ TX night (no 08:45 open)":
            col(["night_ret", "night_vol"]), "+ TX all": col(TX), "+ TX all + beta": col(TX + BETA),
            "+ Yahoo": col(YH), "+ Yahoo (no Asian opens)": col([k for k in YH if not k.endswith("_gap")]),
            "+ TX night + Yahoo (no 08:45 open)": col(["night_ret", "night_vol"] + YH),
            "+ all": col(TX + BETA + YH)}

    te = np.isin(yearf, EVAL)
    yt, dt, t50 = yf[te], dayf[te], is50[te]
    fm = lambda t: f"{U * t[0]:+5.1f}e-4 ± {U * t[1]:3.1f}e-4"
    out = {}
    for label, cols in SETS.items():
        Xs = X[:, cols]
        mu, c0s = np.empty(te.sum()), np.empty(te.sum())
        for Y in EVAL:
            inner, sel = yearf < Y - 1, yearf == Y - 1
            m = Ridge(Xs[inner], yf[inner])
            a = min(ALPHAS, key=lambda a: np.mean((m.predict(Xs[sel], m.weights(a)) - yf[sel]) ** 2))
            m = Ridge(Xs[yearf < Y], yf[yearf < Y])
            idx = yearf[te] == Y
            mu[idx], c0s[idx] = m.predict(Xs[yearf == Y], m.weights(a)), m.c0
        out[label] = (mu, c0s)

    for uname, um in (("all 100 stocks", np.ones(te.sum(), bool)), ("the 50-stock subset", t50)):
        y_, d_ = yt[um], dt[um]
        _, inv = np.unique(d_, return_inverse=True)
        cnt = np.bincount(inv)
        mk = (np.bincount(inv, weights=y_) / cnt)[inv]
        print(f"\n== walk-forward {EVAL[0]}-{EVAL[-1]}, {uname}; always down: score {fm(clustered_mean(-y_, d_))}")
        sb = np.where(out["base (poly p0 + range)"][0][um] > 0, 1.0, -1.0)
        mb = out["base (poly p0 + range)"][0][um]
        for label, (mu, c0s) in out.items():
            mu_, c0_ = mu[um], c0s[um]
            sg = np.where(mu_ > 0, 1.0, -1.0)
            base_mse = np.mean((c0_ - y_) ** 2)
            dm = clustered_mean((mb - y_) ** 2 - (mu_ - y_) ** 2, d_)
            print(f"{label:36s} R2 {1 - np.mean((mu_ - y_) ** 2) / base_mse:+.4f}  up {np.mean(sg > 0):4.0%}  score "
                  f"{fm(clustered_mean(sg * y_, d_))}  market {fm(clustered_mean(sg * mk, d_))}  cross "
                  f"{fm(clustered_mean(sg * (y_ - mk), d_))}  | vs base: R2 {dm[0] / base_mse:+.4f} ± {dm[1] / base_mse:.4f}"
                  f"  score {fm(clustered_mean((sg - sb) * y_, d_))}", flush=True)


if __name__ == "__main__":
    main()
