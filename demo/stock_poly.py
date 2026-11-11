"""Minimum-variance polynomial forecasts on TWStock (fmt runs), fitted over several windows.

On each window of W past days a polynomial of degree p is fitted by least squares (the minimum-variance unbiased
linear estimator of the polynomial under white noise, Gauss-Markov) and extrapolated one day ahead. Every fit is a
fixed linear filter over the window, so each (signal, W, p) gives one forecast feature for every stock-day:

- level signals (the open-to-close return r itself): the extrapolated value is the forecast of the next r;
- integrated signals (cumulative intraday log return, log close, cumulative overnight gap, log Nasdaq / S&P 500 /
  TAIEX): "slope" = fitted next value minus fitted last value (the local trend), "reversion" = fitted next value
  minus the last observed value (the deviation from the fit; with p = 0 this is the moving average minus the last).

The models are: each single feature used as a sign rule (chosen on validation), and ridge regressions of the next r
on groups of the features (pooled over stocks, standardized, ridge strength chosen on the validation MSE), with and
without an intercept. Scores have a standard error clustered by day, since the stocks move together.

    python demo/stock_poly.py          # the multi-window models
    python demo/stock_poly.py --grid   # one model per (degree, window), written to demo/runs/stock_poly_grid.csv
    python demo/stock_poly.py --base   # the ridge on all features for every stock-day, written to
                                       # demo/runs/twstock_ridge_base.npz (train.py --residual-base)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "dataset"
SPLITS = (("train", "2020-01-01", "2024-01-01"), ("val", "2024-01-01", "2025-01-01"),
          ("test", "2025-01-01", "2026-09-01"))  # by the forecast day, as build_stock_splits
WINDOWS = (5, 10, 20, 40, 60)
DEGREES = (0, 1, 2, 3)
BP = 1e4


def poly_filters(window: int, degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Filters h (W,) with h @ y = the least-squares polynomial fit of y (oldest first) at the next step / last step."""
    x = (np.arange(window, dtype=np.float64) - (window - 1)) / window  # the last step at 0, centred for stability
    vander = np.vander(x, degree + 1, increasing=True)  # (W, p + 1)
    pinv = np.linalg.pinv(vander)  # (p + 1, W)
    at = lambda t: (t ** np.arange(degree + 1)) @ pinv
    return at(1 / window), at(0.0)


def windowed(series: np.ndarray, h: np.ndarray) -> np.ndarray:
    """series (T, S) -> (T, S): row t is h over rows t - W + 1 .. t (NaN before the first full window)."""
    w = len(h)
    out = np.full(series.shape, np.nan)
    view = np.lib.stride_tricks.sliding_window_view(series, w, axis=0)  # (T - W + 1, S, W)
    out[w - 1:] = view @ h
    return out


US_FRESH = ("^IXIC", "^GSPC")


def load(fresh_us=False):
    """fresh_us: take the US columns US_FRESH from the row dated t instead of the row before t. global_data.csv holds in
    its row dated D the US close of the date of the previous row (a US day before D), so the row dated t is known
    before the Taiwan open of t. ^TWII, ^KS11 and TX_OI stay as of the day before t (their row t is same-day). ^VIX is
    left out: its close (16:15 New York) is 05:15 Taipei in winter. The default (False) is unchanged."""
    stocks = json.loads((ROOT / "info.json").read_text())["Top100@Y"]
    opens, closes = {}, {}
    for s in stocks:
        f = pd.read_csv(ROOT / "market" / f"{s}.csv", parse_dates=["date"]).set_index("date")
        opens[s], closes[s] = f["etl:adj_open"], f["etl:adj_close"]
    o, c = pd.DataFrame(opens).sort_index(), pd.DataFrame(closes).sort_index()  # Taiwan trading days x stocks
    traded = o.notna() & c.notna()
    r = (c - o) / o
    gap = np.log(o / c.ffill().shift(1))
    g = pd.read_csv(ROOT / "global" / "global_data.csv", parse_dates=["date"]).set_index("date").sort_index()
    # the US close of the day before the forecast day is known at the Taiwan open (as of date - 1 day)
    known = g.reindex(o.index - pd.Timedelta(days=1), method="ffill")
    known.index = o.index
    if fresh_us:
        now = g.reindex(o.index, method="ffill")
        for col in US_FRESH:
            known[col] = now[col].to_numpy()
    return o.index, r, traded, gap, known


def features(dates, r, traded, gap, known, combos=None):
    """Forecast features for every (forecast day t, stock), from data up to the close of day t - 1.

    combos: the (window, degree) pairs; by default WINDOWS x DEGREES with degree < window - 1."""
    S = r.shape[1]
    level = {"r": r.where(traded, 0.0).to_numpy()}
    cum_intraday = np.cumsum(np.log1p(r.where(traded, 0.0).to_numpy()), axis=0)  # sum of log(close / open)
    cum_gap = np.cumsum(gap.where(traded, 0.0).fillna(0.0).to_numpy(), axis=0)  # sum of log(open / previous close)
    integrated = {"cum_intraday": cum_intraday, "log_close": cum_intraday + cum_gap, "cum_gap": cum_gap}
    for name, col in (("nasdaq", "^IXIC"), ("sp500", "^GSPC"), ("taiex", "^TWII")):
        # known[t] is already as of the day before the forecast day t; move it one row up so that the shift below
        # (row t uses rows up to t - 1) gives row t the series up to known[t]
        v = np.log(known[col].ffill().to_numpy())
        integrated[name] = np.repeat(np.append(v[1:], v[-1])[:, None], S, axis=1)
    feats, names = [], []
    if combos is None:
        combos = [(w, p) for w in WINDOWS for p in DEGREES if p < w - 1]
    for w, p in combos:
        if True:
            nxt, last = poly_filters(w, p)
            for name, x in level.items():
                feats.append(windowed(x, nxt)); names.append(f"{name}:W{w}p{p}")
            for name, x in integrated.items():
                fit_next = windowed(x, nxt)
                if p > 0:
                    feats.append(fit_next - windowed(x, last)); names.append(f"{name}:slope:W{w}p{p}")
                feats.append(fit_next - x); names.append(f"{name}:revert:W{w}p{p}")
    X = np.stack(feats, axis=-1)  # (T, S, K), row t uses data up to t
    X = np.concatenate([np.full_like(X[:1], np.nan), X[:-1]], axis=0)  # shift: row t uses data up to t - 1
    y = r.where(traded).to_numpy()
    return X, y, names


def score(pred, y, day):
    """Mean of sign(pred) * r and its standard error clustered by day."""
    gain = np.sign(pred) * y
    days = np.unique(day)
    per_day = np.bincount(np.searchsorted(days, day), weights=gain)
    count = np.bincount(np.searchsorted(days, day))
    mean = gain.mean()
    se = np.sqrt(np.sum((per_day - count * mean) ** 2)) / len(gain)
    return mean, se


def main() -> None:
    dates, r, traded, gap, known = load()
    X, y, names = features(dates, r, traded, gap, known)
    T, S, K = X.shape
    first = max(WINDOWS) + 1
    data = {}
    for name, a, b in SPLITS:
        rows = np.where((dates >= a) & (dates < b) & (np.arange(T) >= first))[0]
        x, t = X[rows].reshape(-1, K), y[rows].reshape(-1)
        day = np.repeat(rows, S)
        ok = np.isfinite(t) & np.isfinite(x).all(axis=1)
        data[name] = (x[ok], t[ok], day[ok])
    print(f"{K} polynomial features; stock-days " + ", ".join(f"{k} {len(v[1])}" for k, v in data.items()))

    def report(label, preds):
        cells = []
        for k in ("train", "val", "test"):
            _, t, day = data[k]
            m, se = score(preds[k], t, day)
            cells.append(f"{BP * m:+6.1f} ± {BP * se:3.1f}")
        mse = {k: np.mean((preds[k] - data[k][1]) ** 2) for k in ("val", "test")}
        const = {k: np.mean((data["train"][1].mean() - data[k][1]) ** 2) for k in ("val", "test")}
        r2 = " ".join(f"{1 - mse[k] / const[k]:+.4f}" for k in ("val", "test")) if label != "always down" else "    -       -   "
        up = np.mean(preds["test"] > 0) * 100
        # paired with always down on the same stock-days: mean of (sign(pred) + 1) * r, clustered by day
        _, t, day = data["test"]
        d, dse = score(np.ones(len(t)), (np.sign(preds["test"]) + 1) * t, day)
        cells.append(f"test - down {BP * d:+5.1f} ± {BP * dse:3.1f}")
        print(f"{label:44s} " + "  ".join(cells) + f"   R2 val/test {r2}   test up {up:4.1f}%")

    print(f"{'model':44s} {'train score (bp)':>14s}  {'val':>12s}  {'test':>12s}")
    report("always down", {k: -np.ones(len(v[1])) for k, v in data.items()})

    # single features as sign rules, oriented by their training correlation, chosen on validation
    train_corr = np.array([np.corrcoef(data["train"][0][:, j], data["train"][1])[0, 1] for j in range(K)])
    orient = np.sign(train_corr)
    val_scores = np.array([score(orient[j] * data["val"][0][:, j], data["val"][1], data["val"][2])[0] for j in range(K)])
    for rank, j in enumerate(np.argsort(-val_scores)[:3]):
        report(f"sign rule #{rank + 1} on val: {names[j]}", {k: orient[j] * v[0][:, j] for k, v in data.items()})

    def ridge(cols, intercept=True, label=""):
        xtr, ytr, _ = data["train"]
        mu, sd = xtr[:, cols].mean(0), xtr[:, cols].std(0) + 1e-12
        z = lambda a: (a[:, cols] - mu) / sd
        c0 = ytr.mean() if intercept else 0.0
        a = z(xtr)
        gram, rhs = a.T @ a, a.T @ (ytr - c0)
        best = None
        for alpha in 10.0 ** np.arange(-1, 8):
            w = np.linalg.solve(gram + alpha * np.eye(len(cols)), rhs)
            v = np.mean((z(data["val"][0]) @ w + c0 - data["val"][1]) ** 2)
            if best is None or v < best[0]:
                best = (v, alpha, w)
        _, alpha, w = best
        report(f"{label} (alpha 1e{np.log10(alpha):.0f})", {k: z(v[0]) @ w + c0 for k, v in data.items()})

    groups = {}
    for j, n in enumerate(names):
        groups.setdefault(n.split(":")[0], []).append(j)
    for g, cols in groups.items():
        ridge(cols, label=f"ridge: {g} ({len(cols)})")
    everything = list(range(K))
    ridge(everything, label=f"ridge: all ({K})")
    ridge(everything, intercept=False, label=f"ridge: all, no intercept")
    stock_only = [j for j, n in enumerate(names) if n.split(":")[0] in ("r", "cum_intraday", "log_close", "cum_gap")]
    ridge(stock_only, label=f"ridge: own-stock signals ({len(stock_only)})")
    for w in WINDOWS:
        cols = [j for j, n in enumerate(names) if n.endswith(tuple(f"W{w}p{p}" for p in DEGREES))]
        ridge(cols, label=f"ridge: window {w} only ({len(cols)})")
    for p in DEGREES:
        cols = [j for j, n in enumerate(names) if n.endswith(tuple(f"W{w}p{p}" for w in WINDOWS))]
        ridge(cols, label=f"ridge: degree {p} only ({len(cols)})")


GRID_WINDOWS = range(2, 61)
GRID_MAX_DEGREE = 8


def grid() -> None:
    """One ridge model per (degree k, window l) with k <= min(l - 1, GRID_MAX_DEGREE): only the 7 (k = 0) or 13
    features of that single polynomial fit, every signal. Writes demo/runs/stock_poly_grid.csv."""
    dates, r, traded, gap, known = load()
    T = len(dates)
    first = max(GRID_WINDOWS) + 1  # the same stock-days for every cell
    y = r.where(traded).to_numpy()
    S = y.shape[1]
    rows = {name: np.where((dates >= a) & (dates < b) & (np.arange(T) >= first))[0] for name, a, b in SPLITS}
    out = []
    for l in GRID_WINDOWS:
        for k in range(min(l - 1, GRID_MAX_DEGREE) + 1):
            X, _, _ = features(dates, r, traded, gap, known, combos=[(l, k)])
            K = X.shape[-1]
            data = {}
            for name, rr in rows.items():
                x, t = X[rr].reshape(-1, K), y[rr].reshape(-1)
                day = np.repeat(rr, S)
                ok = np.isfinite(t) & np.isfinite(x).all(axis=1)
                data[name] = (x[ok], t[ok], day[ok])
            xtr, ytr, _ = data["train"]
            mu, sd = xtr.mean(0), xtr.std(0) + 1e-12
            z = lambda a: (a - mu) / sd
            c0 = ytr.mean()
            a_ = z(xtr)
            gram, rhs = a_.T @ a_, a_.T @ (ytr - c0)
            best = None
            for alpha in 10.0 ** np.arange(-1, 8):
                w = np.linalg.solve(gram + alpha * np.eye(K), rhs)
                v = np.mean((z(data["val"][0]) @ w + c0 - data["val"][1]) ** 2)
                if best is None or v < best[0]:
                    best = (v, alpha, w)
            _, alpha, w = best
            row = dict(k=k, l=l, features=K, alpha=alpha)
            for name, (x, t, day) in data.items():
                pred = z(x) @ w + c0
                m, se = score(pred, t, day)
                const = np.mean((ytr.mean() - t) ** 2)
                row.update({f"{name}_score_bp": BP * m, f"{name}_se_bp": BP * se,
                            f"{name}_r2": 1 - np.mean((pred - t) ** 2) / const, f"{name}_up": np.mean(pred > 0)})
                d, dse = score(np.ones(len(t)), (np.sign(pred) + 1) * t, day)
                row.update({f"{name}_minus_down_bp": BP * d, f"{name}_minus_down_se_bp": BP * dse})
            out.append(row)
            print(f"k {k} l {l:2d} alpha 1e{np.log10(alpha):.0f}  val {row['val_score_bp']:+6.1f}  "
                  f"test {row['test_score_bp']:+6.1f}  test-down {row['test_minus_down_bp']:+5.1f}  "
                  f"R2 {row['val_r2']:+.4f} {row['test_r2']:+.4f}", flush=True)
    path = Path(__file__).resolve().parent / "runs" / "stock_poly_grid.csv"
    pd.DataFrame(out).to_csv(path, index=False)
    print(f"wrote {path}")


BASE_PATH = Path(__file__).resolve().parent / "runs" / "twstock_ridge_base.npz"


def base() -> None:
    """The ridge on all the multi-window features ("ridge: all" of main), fitted on train with alpha chosen on the
    validation MSE, evaluated on every (forecast day, stock). Stock-days without full features get the training
    mean. Writes the forecast day, the stock codes and the forecasts (returns, not scaled) to BASE_PATH."""
    dates, r, traded, gap, known = load()
    X, y, names = features(dates, r, traded, gap, known)
    T, S, K = X.shape
    first = max(WINDOWS) + 1
    full = np.isfinite(X).all(axis=-1) & (np.arange(T) >= first)[:, None]  # (T, S)
    data = {}
    for name, a, b in SPLITS:
        ok = full & np.isfinite(y) & ((dates >= a) & (dates < b))[:, None]
        data[name] = (X[ok], y[ok])
    xtr, ytr = data["train"]
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-12
    z = lambda a: (a - mu) / sd
    c0 = ytr.mean()
    a = z(xtr)
    gram, rhs = a.T @ a, a.T @ (ytr - c0)
    best = None
    for alpha in 10.0 ** np.arange(-1, 8):
        w = np.linalg.solve(gram + alpha * np.eye(K), rhs)
        v = np.mean((z(data["val"][0]) @ w + c0 - data["val"][1]) ** 2)
        if best is None or v < best[0]:
            best = (v, alpha, w)
    _, alpha, w = best
    pred = np.full((T, S), c0)
    pred[full] = z(X[full]) @ w + c0
    for name, (x, t) in data.items():
        p = z(x) @ w + c0
        const = np.mean((c0 - t) ** 2)
        print(f"{name:5s} score {BP * np.mean(np.sign(p) * t):+6.1f} bp  R2 {1 - np.mean((p - t) ** 2) / const:+.4f}")
    np.savez(BASE_PATH, dates=dates.to_numpy().astype("datetime64[D]"), stocks=np.array(r.columns, dtype=str),
             pred=pred)
    print(f"alpha 1e{np.log10(alpha):.0f}, {K} features; wrote {BASE_PATH}")


if __name__ == "__main__":
    import sys

    if "--grid" in sys.argv[1:]:
        grid()
    elif "--base" in sys.argv[1:]:
        base()
    else:
        main()
