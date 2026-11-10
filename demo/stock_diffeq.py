"""A reference model: local difference-equation fits in place of the polynomial fits, then the ridge (TWStock).

demo/stock_poly.py fits a polynomial of the time index over each window of W past days. Here each window is fitted
instead with a linear difference equation of order q = 1, 2 or 3, so the forecast follows the window's own dynamics
(persistence, mean reversion, oscillation) rather than a curve in time:

- level signal (the open-to-close return r): r_n = c + a_1 r_{n-1} (+ a_2 r_{n-2} + a_3 r_{n-3});
- integrated signals (cumulative intraday log return, log close, cumulative gap, log Nasdaq / S&P 500 / TAIEX), in
  the equivalent difference form: dx_n = c + b (x_{n-1} - x_last) (+ d_1 dx_{n-1} + d_2 dx_{n-2}), with dx_n = x_n - x_{n-1} and
  x_last the last value of the window (b < 0 is mean reversion, d the momentum of the increments).

The equations of a window (one per day with its lags inside the window) are solved for theta = (c, a or b, d) by
- least squares (LS): theta = (A'A + 1e-10 tr(A'A)/k I)^-1 A'y, the exact minimizer;
- gradient descent (GD): K steps theta <- theta - (A'A theta - A'y) / lambda_max(A'A) from theta = 0, with the
  regressors scaled to unit RMS within the window. Stopped early, GD shrinks theta towards 0, i.e. towards "r = 0"
  for r and towards a random walk without drift for the integrated signals (an implicit ridge); K -> inf gives LS.

Features of every (signal, W, q, method): the one-step forecast (r_hat for r, dx_hat for the integrated signals) and,
in the "+ coef" sets, the dynamic coefficients (a_1, a_2 or b, d). Windows 5 .. 60 for q = 1, 10 .. 60 for q = 2 and 3.
Every feature is clipped at its 0.1 / 99.9 percentiles of the years before 2022, and shifted so that day t uses data
up to the close of t - 1. The second stage is the ridge of demo/stock_extra.py, walk-forward on 2022 - 2026, all
models on the same stock-days. Scores are in units of 1e-4 (of the open-to-close return), day-clustered.

    python demo/stock_diffeq.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_extra import extra_features, signals  # noqa: E402
from stock_poly import features, load  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge, clustered_mean  # noqa: E402

WINDOWS = {1: (5, 10, 20, 40, 60), 2: (10, 20, 40, 60), 3: (10, 20, 40, 60)}
GD_STEPS = (10, 100)
U = 1e4  # report in units of 1e-4


def series(dates, r, traded, gap, known):
    S = r.shape[1]
    rr = r.where(traded, 0.0).to_numpy()
    ci = np.cumsum(np.log1p(rr), axis=0)
    cg = np.cumsum(gap.where(traded, 0.0).fillna(0.0).to_numpy(), axis=0)
    out = {"r": ("level", rr), "cum_intraday": ("int", ci), "log_close": ("int", ci + cg), "cum_gap": ("int", cg)}
    for name, col in (("nasdaq", "^IXIC"), ("sp500", "^GSPC"), ("taiex", "^TWII")):
        v = np.log(known[col].ffill().to_numpy())
        out[name] = ("int", np.repeat(np.append(v[1:], v[-1])[:, None], S, axis=1))
    return out


def fit_window(kind, x, W, q):
    """x (T, S) -> design of every window ending at row t: A (T, S, n, k), y (T, S, n), forecast regressors (T, S, k)."""
    T, S = x.shape
    win = np.full((T, S, W), np.nan)
    win[W - 1:] = np.lib.stride_tricks.sliding_window_view(x, W, axis=0)  # rows t - W + 1 .. t, oldest first
    if kind == "level":
        y = win[..., q:]
        lags = [win[..., q - j:W - j] for j in range(1, q + 1)]  # r_{n-1}, r_{n-2}
        nxt = [win[..., W - j] for j in range(1, q + 1)]  # r_t, r_{t-1}
    else:
        z = win - win[..., -1:]  # relative to the last value; the fit and the forecast of dx are unchanged
        dz = np.diff(z, axis=-1)  # (.., W - 1): dx_1 .. dx_{W-1}
        y = dz[..., q - 1:]  # dx_n for n = q .. W - 1
        lags = [z[..., q - 1:W - 1]]  # x_{n-1} - x_last
        nxt = [z[..., -1]]  # x_t - x_last = 0
        for j in range(1, q):
            lags.append(dz[..., q - 1 - j:W - 1 - j])  # dx_{n-j}
            nxt.append(dz[..., -j])
    ones = np.ones_like(y)
    A = np.stack([ones] + lags, axis=-1)
    f = np.stack([np.ones((T, S))] + nxt, axis=-1)
    return A, y, f


def solve(A, y, f, steps=None):
    """Forecast f @ theta and theta[1:] for every window; steps None = LS, else GD from 0 on unit-RMS regressors."""
    k = A.shape[-1]
    scale = np.sqrt(np.mean(A ** 2, axis=-2))  # (T, S, k)
    scale = np.where(scale > 0, scale, 1.0)
    As, fs = A / scale[..., None, :], f / scale
    G = np.einsum("...nk,...nl->...kl", As, As)
    g = np.einsum("...nk,...n->...k", As, y)
    if steps is None:
        tr = np.trace(G, axis1=-2, axis2=-1)[..., None, None]
        th = np.linalg.solve(G + 1e-10 * tr / k * np.eye(k) + 1e-300 * np.eye(k), g[..., None])[..., 0]
    else:
        lam = np.linalg.eigvalsh(G)[..., -1:]
        lam = np.where(lam > 0, lam, 1.0)
        th = np.zeros_like(g)
        for _ in range(steps):
            th = th - (np.einsum("...kl,...l->...k", G, th) - g) / lam
    return np.sum(fs * th, axis=-1), th[..., 1:] / scale[..., 1:]


def de_features(sig, methods):
    feats, names = [], []
    for name, (kind, x) in sig.items():
        for q, Ws in WINDOWS.items():
            for W in Ws:
                A, y, f = fit_window(kind, x, W, q)
                bad = ~np.isfinite(A).all((-1, -2)) | ~np.isfinite(y).all(-1)
                A, y, f = np.nan_to_num(A), np.nan_to_num(y), np.nan_to_num(f)
                for m in methods:
                    fc, co = solve(A, y, f, None if m == "LS" else int(m[2:]))
                    fc[bad] = np.nan
                    feats.append(fc); names.append(f"{name}:W{W}q{q}:{m}:fc")
                    for j in range(co.shape[-1]):
                        c = co[..., j].copy(); c[bad] = np.nan
                        feats.append(c); names.append(f"{name}:W{W}q{q}:{m}:coef{j + 1}")
        print(f"  {name} done", flush=True)
    X = np.stack(feats, axis=-1).astype(np.float32)
    X = np.concatenate([np.full_like(X[:1], np.nan), X[:-1]], axis=0)  # row t uses data up to t - 1
    return X, np.array(names)


def main() -> None:
    dates, r, traded, gap, known = load()
    T, S = r.shape
    train_rows = np.asarray(dates < "2022-01-01")
    methods = ["LS"] + [f"GD{k}" for k in GD_STEPS]
    Xd, dnames = de_features(series(dates, r, traded, gap, known), methods)
    lo, hi = np.nanquantile(Xd[train_rows].reshape(-1, Xd.shape[-1]), [0.001, 0.999], axis=0)
    Xd = np.clip(Xd, lo, hi)
    Xb, y, bnames = features(dates, r, traded, gap, known)
    Xe, _, egroups = extra_features(signals(dates, known, S), train_rows)
    X = np.concatenate([Xb, Xe, Xd], axis=-1).reshape(-1, Xb.shape[-1] + Xe.shape[-1] + Xd.shape[-1])
    K1, K2 = Xb.shape[-1], Xe.shape[-1]
    yf = y.reshape(-1)
    dayf, yearf = np.repeat(np.arange(T), S), np.repeat(dates.year.to_numpy(), S)
    ok = np.repeat(np.arange(T) >= FIRST, S) & np.isfinite(X).all(1) & np.isfinite(yf)
    X, yf, dayf, yearf = X[ok], yf[ok], dayf[ok], yearf[ok]
    print(f"{ok.sum()} stock-days; {len(dnames)} difference-equation features", flush=True)

    none = np.zeros(X.shape[1], bool)
    def mask(base=(), rng=False, de=None):
        m = none.copy()
        if "p0" in base:
            m[:K1] = [n.endswith("p0") for n in bnames]
        if "all" in base:
            m[:K1] = True
        if rng:
            m[K1:K1 + K2] = egroups == "range"
        if de is not None:
            meth, orders, coef = de
            sel = np.array([n.split(":")[-2] == meth and int(n.split(":")[-3][-1]) in orders and
                            (coef or n.endswith(":fc")) for n in dnames])
            m[K1 + K2:] = sel
        return m

    SETS = {"poly p0": mask(("p0",)), "poly p0 + range": mask(("p0",), True), "poly all 230": mask(("all",))}
    for meth in methods:
        SETS[f"DE {meth} q1"] = mask(de=(meth, (1,), False))
        SETS[f"DE {meth} q2"] = mask(de=(meth, (2,), False))
        SETS[f"DE {meth} q3"] = mask(de=(meth, (3,), False))
        SETS[f"DE {meth} q1+q2"] = mask(de=(meth, (1, 2), False))
        SETS[f"DE {meth} q1+q2+q3"] = mask(de=(meth, (1, 2, 3), False))
        SETS[f"DE {meth} q1+q2+q3 + coef"] = mask(de=(meth, (1, 2, 3), True))
        SETS[f"DE {meth} q1+q2+q3 + range"] = mask(rng=True, de=(meth, (1, 2, 3), False))
    SETS["poly p0 + range + DE LS q1-3"] = mask(("p0",), True, ("LS", (1, 2, 3), False))

    te = np.isin(yearf, EVAL)
    yt, dt = yf[te], dayf[te]
    _, inv = np.unique(dt, return_inverse=True)
    mkt = (np.bincount(inv, weights=yt) / np.bincount(inv))[inv]
    fm = lambda t: f"{U * t[0]:+5.1f}e-4 ± {U * t[1]:3.1f}e-4"
    out = {}
    print(f"\npooled {EVAL[0]}-{EVAL[-1]}, walk-forward; always down: score {fm(clustered_mean(-yt, dt))}")
    for label, cols in SETS.items():
        Xs = X[:, cols]
        mu, c0s, al = np.empty(te.sum()), np.empty(te.sum()), []
        for Y in EVAL:
            inner, sel = yearf < Y - 1, yearf == Y - 1
            m = Ridge(Xs[inner], yf[inner])
            a = min(ALPHAS, key=lambda a: np.mean((m.predict(Xs[sel], m.weights(a)) - yf[sel]) ** 2))
            m = Ridge(Xs[yearf < Y], yf[yearf < Y])
            idx = yearf[te] == Y
            mu[idx], c0s[idx] = m.predict(Xs[yearf == Y], m.weights(a)), m.c0
            al.append(int(np.log10(a)))
        out[label] = mu
        sg = np.where(mu > 0, 1.0, -1.0)
        nz = yt != 0
        base = np.mean((c0s - yt) ** 2)
        ref = out["poly p0 + range"] if "poly p0 + range" in out else mu
        dm = clustered_mean((ref - yt) ** 2 - (mu - yt) ** 2, dt)
        sr = np.where(ref > 0, 1.0, -1.0)
        print(f"{label:32s} K {cols.sum():3d} alpha 1e{al}  R2 {1 - np.mean((mu - yt) ** 2) / base:+.4f}  "
              f"acc {np.mean((sg[nz] > 0) == (yt[nz] > 0)):5.1%}  up {np.mean(sg > 0):4.0%}  score "
              f"{fm(clustered_mean(sg * yt, dt))}  cross {fm(clustered_mean(sg * (yt - mkt), dt))}  | vs poly p0+range: "
              f"R2 {dm[0] / base:+.4f} ± {dm[1] / base:.4f}  score {fm(clustered_mean((sg - sr) * yt, dt))}", flush=True)


if __name__ == "__main__":
    main()
