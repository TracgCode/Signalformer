"""Difference equations on block means: average m adjacent days, then difference (TWStock).

demo/stock_diffeq_pool.py showed that an order-3 equation on the last three daily values misses the information of the
polynomial ridge, which sits in 5 - 60 day averages. Here every integrated signal x (cumulative intraday log return,
log close, cumulative gap, log Nasdaq / S&P 500 / TAIEX; see demo/stock_diffeq.py) is first averaged over blocks of
m adjacent days, B_m(t) = mean(x_{t-m+1} .. x_t), and then differenced between adjacent blocks,
D_m(t, k) = B_m(t - k m) - B_m(t - (k + 1) m), k = 0, 1, 2 (the latest block first). Block sizes m = 2, 5, 10, 20.

- block lags: the ridge on D_m(t, 0..2) of every signal and m (the shared equation fitted end to end; it is linear);
- block DE, two-stage: for every (signal, m) the order-3 equation of the next block difference,
  B_m(t + m) - B_m(t) = c + a_1 D_m(t, 0) + a_2 D_m(t, 1) + a_3 D_m(t, 2), by least squares over the stock-days before
  2022, then the ridge on its forecasts.

All features are shifted so that day t uses data up to the close of t - 1; walk-forward ridge on 2022 - 2026 as
demo/stock_extra.py on the same stock-days; scores in units of the open-to-close return, day-clustered.

    python demo/stock_diffeq_block.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_diffeq import series  # noqa: E402
from stock_extra import extra_features, signals  # noqa: E402
from stock_poly import features, load, poly_filters, windowed  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge, clustered_mean  # noqa: E402

BLOCKS = (2, 5, 10, 20)
NLAG = 3
U = 1e4


def lagrows(a, k):
    out = np.full_like(a, np.nan)
    out[k:] = a[:len(a) - k] if k else a
    return out


def main() -> None:
    dates, r, traded, gap, known = load()
    T, S = r.shape
    train_end = int(np.asarray(dates < "2022-01-01").sum())
    sig = {k: x for k, (kind, x) in series(dates, r, traded, gap, known).items() if kind == "int"}
    lagf, lagn, def_, den = [], [], [], []
    for name, x in sig.items():
        for m in BLOCKS:
            B = windowed(x, poly_filters(m, 0)[0])  # block mean ending at row t
            D = np.stack([lagrows(B, k * m) - lagrows(B, (k + 1) * m) for k in range(NLAG)], axis=-1)
            nxt = np.full_like(B, np.nan)
            nxt[:-m] = B[m:] - B[:-m]  # the next block difference, B(t + m) - B(t)
            a = D[FIRST:train_end - m].reshape(-1, NLAG); b = nxt[FIRST:train_end - m].reshape(-1)
            ok = np.isfinite(a).all(1) & np.isfinite(b)
            sol = np.linalg.lstsq(np.column_stack([np.ones(ok.sum()), a[ok]]), b[ok], rcond=None)[0]
            print(f"{name:13s} m {m:2d}: c {sol[0]:+.2e}  a {np.round(sol[1:], 3)}", flush=True)
            def_.append(sol[0] + D @ sol[1:]); den.append(f"{name}:m{m}:fc")
            for k in range(NLAG):
                lagf.append(D[..., k]); lagn.append(f"{name}:m{m}:D{k}")
    sh = lambda X: np.concatenate([np.full_like(X[:1], np.nan), X[:-1]], axis=0)
    XL, XD = sh(np.stack(lagf, -1)), sh(np.stack(def_, -1))
    Xb, y, bnames = features(dates, r, traded, gap, known)
    Xe, _, egroups = extra_features(signals(dates, known, S), np.asarray(dates < "2022-01-01"))
    blocks = {"base": Xb, "extra": Xe, "lags": XL, "de": XD}
    off, o = {}, 0
    for k, v in blocks.items():
        off[k] = (o, o + v.shape[-1]); o += v.shape[-1]
    X = np.concatenate(list(blocks.values()), axis=-1).reshape(T * S, -1)
    yf = y.reshape(-1)
    dayf, yearf = np.repeat(np.arange(T), S), np.repeat(dates.year.to_numpy(), S)
    ok = np.repeat(np.arange(T) >= FIRST, S) & np.isfinite(X).all(1) & np.isfinite(yf)
    X, yf, dayf, yearf = X[ok], yf[ok], dayf[ok], yearf[ok]
    print(f"{ok.sum()} stock-days", flush=True)

    def cols(**parts):
        msk = np.zeros(X.shape[1], bool)
        for k, sel in parts.items():
            a, b = off[k]
            msk[a:b] = True if sel is True else sel
        return msk
    p0 = np.array([n.endswith("p0") for n in bnames])
    rng = egroups == "range"
    lag_m = lambda ms: np.array([int(n.split(":")[1][1:]) in ms for n in lagn])
    SETS = {"poly p0 + range": cols(base=p0, extra=rng), "poly p0": cols(base=p0)}
    for m in BLOCKS:
        SETS[f"block lags m {m}"] = cols(lags=lag_m((m,)))
    SETS["block lags all m"] = cols(lags=True)
    SETS["block DE two-stage all m"] = cols(de=True)
    SETS["block lags all m + range"] = cols(lags=True, extra=rng)
    SETS["block DE two-stage + range"] = cols(de=True, extra=rng)
    SETS["poly p0 + range + block lags"] = cols(base=p0, extra=rng, lags=True)

    te = np.isin(yearf, EVAL)
    yt, dt = yf[te], dayf[te]
    _, inv = np.unique(dt, return_inverse=True)
    mkt = (np.bincount(inv, weights=yt) / np.bincount(inv))[inv]
    fm = lambda t: f"{U * t[0]:+5.1f}e-4 ± {U * t[1]:3.1f}e-4"
    out = {}
    print(f"\npooled {EVAL[0]}-{EVAL[-1]}, walk-forward; always down: score {fm(clustered_mean(-yt, dt))}")
    for label, m_ in SETS.items():
        Xs = X[:, m_]
        mu, c0s = np.empty(te.sum()), np.empty(te.sum())
        for Y in EVAL:
            inner, sel = yearf < Y - 1, yearf == Y - 1
            m = Ridge(Xs[inner], yf[inner])
            a = min(ALPHAS, key=lambda a: np.mean((m.predict(Xs[sel], m.weights(a)) - yf[sel]) ** 2))
            m = Ridge(Xs[yearf < Y], yf[yearf < Y])
            idx = yearf[te] == Y
            mu[idx], c0s[idx] = m.predict(Xs[yearf == Y], m.weights(a)), m.c0
        out[label] = mu
        sg = np.where(mu > 0, 1.0, -1.0)
        nz = yt != 0
        base = np.mean((c0s - yt) ** 2)
        ref = out["poly p0 + range"]
        dm = clustered_mean((ref - yt) ** 2 - (mu - yt) ** 2, dt)
        sr = np.where(ref > 0, 1.0, -1.0)
        print(f"{label:30s} K {m_.sum():3d}  R2 {1 - np.mean((mu - yt) ** 2) / base:+.4f}  acc "
              f"{np.mean((sg[nz] > 0) == (yt[nz] > 0)):5.1%}  up {np.mean(sg > 0):4.0%}  score "
              f"{fm(clustered_mean(sg * yt, dt))}  cross {fm(clustered_mean(sg * (yt - mkt), dt))}  | vs poly p0+range:"
              f" R2 {dm[0] / base:+.4f} ± {dm[1] / base:.4f}  score {fm(clustered_mean((sg - sr) * yt, dt))}",
              flush=True)


if __name__ == "__main__":
    main()
