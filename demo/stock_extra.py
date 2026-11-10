"""The data columns of TWStock not yet used by the return ridge, added group by group, walk-forward (TWStock).

Every new column is made stationary as a daily signal (row t known at the close of day t), and each signal gives the
features: its last value and its moving averages over W = 5, 10, 20, 40, 60 days (the p = 0 readouts), all shifted so
that the forecast day t uses data up to the close of day t - 1. For signals that are levels (log volume and similar)
the features are the last value minus the moving averages instead, so they stay stationary. Groups:

- flows: the net buys of foreign investors, investment trusts, dealers (own account) and dealers (hedging), each
  divided by the shares traded that day; and the daily change of the foreign shareholding ratio (percentage points);
- volume: log shares traded, log number of trades, log shares per trade (deviations from their moving averages);
- range: log(high / low) and the close location (2 close - high - low) / (high - low) (moving averages);
- technical: RSI / 100 - 0.5, MACD / close, ATR / close, the Bollinger position (close - mid) / (upper - lower),
  SMA / close - 1 (last value only; they are functions of past prices of the stock);
- global: the daily log change of KOSPI, log VIX and its daily change, the TAIEX futures open interest of the data
  (TX_OI, in units of 10^4) and its daily change; the same for every stock, so they can only move the market part.

The base is the multi-window polynomial ridge of demo/stock_poly.py, either the 35 p = 0 features or all 230. Each
model is evaluated walk-forward on 2022 - 2026 as demo/stock_walkforward.py (alpha chosen on the previous year from
fits on the years before it, then refitted on all earlier years). Reported pooled: R2 against the training mean,
the sign score of sign(mu), its cross-sectional part, and the paired differences to the base, day-clustered.

    python demo/stock_extra.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_poly import BP, ROOT, features, load, poly_filters, windowed  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge, clustered_mean  # noqa: E402

WINDOWS = (5, 10, 20, 40, 60)
FLOWS = {"foreign": "外陸資買賣超股數(不含外資自營商)", "trust": "投信買賣超股數", "dealer": "自營商買賣超股數(自行買賣)",
         "hedge": "自營商買賣超股數(避險)"}


def load_columns(dates):
    stocks = json.loads((ROOT / "info.json").read_text())["Top100@Y"]
    out = {}
    for s in stocks:
        f = pd.read_csv(ROOT / "market" / f"{s}.csv", parse_dates=["date"]).set_index("date")
        for c in f.columns:
            out.setdefault(c, {})[s] = f[c]
    return {c: pd.DataFrame(v).reindex(dates)[stocks].to_numpy(dtype=np.float64) for c, v in out.items()}


def signals(dates, known, S):
    """name -> (group, kind, (T, S) array); kind 'mean' (moving averages) or 'dev' (minus moving averages) or 'last'."""
    col = load_columns(dates)
    with np.errstate(invalid="ignore", divide="ignore"):
        vol = col["price:成交股數"]
        sig = {}
        for k, c in FLOWS.items():
            sig[f"flow_{k}"] = ("flows", "mean", np.clip(col[f"institutional_investors_trading_summary:{c}"] / vol, -1, 1))
        hold = col["foreign_investors_shareholding:全體外資及陸資持股比率"]
        sig["hold_change"] = ("flows", "mean", np.diff(hold, axis=0, prepend=np.nan))
        trades = col["price:成交筆數"]
        sig["log_volume"] = ("volume", "dev", np.log(vol))
        sig["log_trades"] = ("volume", "dev", np.log(trades))
        sig["log_trade_size"] = ("volume", "dev", np.log(vol / trades))
        h, l, c = col["etl:adj_high"], col["etl:adj_low"], col["etl:adj_close"]
        sig["log_range"] = ("range", "mean", np.log(h / l))
        sig["close_location"] = ("range", "mean", np.where(h > l, (2 * c - h - l) / (h - l), 0.0))
        sig["rsi"] = ("technical", "last", col["RSI"] / 100 - 0.5)
        sig["macd"] = ("technical", "last", col["MACD"] / c)
        sig["atr"] = ("technical", "last", col["ATR"] / c)
        sig["bollinger"] = ("technical", "last", (c - col["BBANDS_mid"]) / (col["BBANDS_upper"] - col["BBANDS_lower"]))
        sig["sma"] = ("technical", "last", col["SMA"] / c - 1)
        # global: known[t] is as of the day before the forecast day; move it one row up (as stock_poly.features)
        up = lambda v: np.repeat(np.append(v[1:], v[-1])[:, None], S, axis=1)
        ks = np.log(known["^KS11"].ffill().to_numpy())
        vix = np.log(known["^VIX"].ffill().to_numpy())
        oi = known["TX_OI"].ffill().to_numpy() / 1e4
        sig["kospi"] = ("global", "mean", up(np.diff(ks, prepend=ks[0])))
        sig["log_vix"] = ("global", "last", up(vix))
        sig["vix_change"] = ("global", "mean", up(np.diff(vix, prepend=vix[0])))
        sig["tx_oi"] = ("global", "last", up(oi))
        sig["tx_oi_change"] = ("global", "mean", up(np.diff(oi, prepend=oi[0])))
    return sig


def extra_features(sig, train_rows):
    feats, names, groups = [], [], []
    for name, (group, kind, x) in sig.items():
        x = np.where(np.isfinite(x), x, np.nan)
        fill = np.nanmean(x[train_rows])
        x = np.where(np.isfinite(x), x, fill)  # missing days get the training mean
        cols = [("last", x)]
        if kind != "last":
            for w in WINDOWS:
                ma = windowed(x, poly_filters(w, 0)[0])
                cols.append((f"W{w}", ma if kind == "mean" else x - ma))
        for tag, v in cols:
            feats.append(v); names.append(f"{name}:{tag}"); groups.append(group)
    X = np.stack(feats, axis=-1)
    X = np.concatenate([np.full_like(X[:1], np.nan), X[:-1]], axis=0)  # row t uses data up to t - 1
    return X, names, np.array(groups)


def main() -> None:
    dates, r, traded, gap, known = load()
    T, S = r.shape
    Xb, y, bnames = features(dates, r, traded, gap, known)
    train_rows = np.asarray(dates < "2022-01-01")
    Xe, enames, egroups = extra_features(signals(dates, known, S), train_rows)
    K1, K2 = Xb.shape[-1], Xe.shape[-1]
    X = np.concatenate([Xb, Xe], axis=-1).reshape(-1, K1 + K2)
    yf = y.reshape(-1)
    dayf, yearf = np.repeat(np.arange(T), S), np.repeat(dates.year.to_numpy(), S)
    ok = np.repeat(np.arange(T) >= FIRST, S) & np.isfinite(X).all(1) & np.isfinite(yf)
    X, yf, dayf, yearf = X[ok], yf[ok], dayf[ok], yearf[ok]
    print(f"{K1} base features, {K2} new features over {len(set(egroups))} groups; {ok.sum()} stock-days", flush=True)
    deg0 = np.array([n.endswith("p0") for n in bnames] + [False] * K2)
    base230 = np.array([True] * K1 + [False] * K2)
    new = {g: np.array([False] * K1 + list(egroups == g)) for g in dict.fromkeys(egroups)}
    allnew = np.array([False] * K1 + [True] * K2)

    te = np.isin(yearf, EVAL)
    yt, dt = yf[te], dayf[te]
    days, inv = np.unique(dt, return_inverse=True)
    mkt = (np.bincount(inv, weights=yt) / np.bincount(inv))[inv]
    out = {}

    def run(label, cols):
        Xs = X[:, cols]
        mu, c0s, alphas = np.empty(te.sum()), np.empty(te.sum()), []
        for Y in EVAL:
            inner, sel = yearf < Y - 1, yearf == Y - 1
            m = Ridge(Xs[inner], yf[inner])
            a = min(ALPHAS, key=lambda a: np.mean((m.predict(Xs[sel], m.weights(a)) - yf[sel]) ** 2))
            m = Ridge(Xs[yearf < Y], yf[yearf < Y])
            idx = yearf[te] == Y
            mu[idx], c0s[idx] = m.predict(Xs[yearf == Y], m.weights(a)), m.c0
            alphas.append(int(np.log10(a)))
        out[label] = (mu, c0s)
        sgn = np.where(mu > 0, 1.0, -1.0)
        r2 = 1 - np.mean((mu - yt) ** 2) / np.mean((c0s - yt) ** 2)
        fm = lambda t: f"{BP * t[0]:+5.1f} ± {BP * t[1]:3.1f}"
        print(f"{label:34s} K {cols.sum():3d}  alpha 1e{alphas}  R2 {r2:+.4f}  score {fm(clustered_mean(sgn * yt, dt))}"
              f"  cross {fm(clustered_mean(sgn * (yt - mkt), dt))}", flush=True)

    def paired(a, b):
        (ma, ca), (mb, _) = out[a], out[b]
        sa, sb = np.where(ma > 0, 1.0, -1.0), np.where(mb > 0, 1.0, -1.0)
        base = np.mean((ca - yt) ** 2)
        dm = clustered_mean((mb - yt) ** 2 - (ma - yt) ** 2, dt)
        ds = clustered_mean((sa - sb) * yt, dt)
        dc = clustered_mean((sa - sb) * (yt - mkt), dt)
        print(f"  {a:34s} - {b:22s}: R2 {dm[0] / base:+.4f} ± {dm[1] / base:.4f}   score {BP * ds[0]:+5.1f} ± "
              f"{BP * ds[1]:3.1f}   cross {BP * dc[0]:+5.1f} ± {BP * dc[1]:3.1f}", flush=True)

    print(f"\nalways down: score {BP * clustered_mean(-yt, dt)[0]:+.1f}; pooled {EVAL[0]}-{EVAL[-1]}, walk-forward")
    run("base p=0 (35)", deg0)
    run("base all (230)", base230)
    for g, cols in new.items():
        run(f"{g} alone", cols)
    run("all new alone", allnew)
    for base_label, bcols in (("base p=0 (35)", deg0), ("base all (230)", base230)):
        short = base_label.split(" (")[0]
        for g, cols in new.items():
            run(f"{short} + {g}", bcols | cols)
        run(f"{short} + all new", bcols | allnew)
    print("\npaired differences to the base (pooled, day-clustered se)")
    for base_label in ("base p=0 (35)", "base all (230)"):
        short = base_label.split(" (")[0]
        for g in list(new) + ["all new"]:
            paired(f"{short} + {g}", base_label)


if __name__ == "__main__":
    main()
