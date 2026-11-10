"""Bilinear least-squares forecasts on TWStock: r_hat = b0 + sum_k u_k^T Z v_k.

Z (F x L) holds, for one (forecast day t, stock), F stationary daily signals over the last L days (lag 0 = the day
before t): the open-to-close return r, the overnight log gap, the close-to-close log return, and the daily log change
of Nasdaq, S&P 500 and TAIEX as known at the Taiwan open. u_k (F) weighs the signals, v_k (L) is a filter over the
lags. With one of u or v fixed the model is linear in the other, so it is fitted by alternating ridge least squares
(each step the minimum-variance linear estimate under white noise). Variants: free filters v (L values), and
polynomial filters v = B c with B the degree-p Legendre basis over the lags (p + 1 values). The full ridge on vec(Z)
(F L weights) is the unconstrained reference. Pooled over stocks, standardized per signal, ridge chosen on val MSE.

    python demo/stock_bilinear.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_poly import BP, SPLITS, load, score  # noqa: E402

LAGS = (5, 10, 20, 40, 60)
ALPHAS = 10.0 ** np.arange(-1, 8)


def signals():
    dates, r, traded, gap, known = load()
    rr = r.where(traded, 0.0).to_numpy()
    g = gap.where(traded, 0.0).fillna(0.0).to_numpy()
    own = [rr, g, np.log1p(rr) + g]  # (T, S) each; row t = day t
    S = rr.shape[1]
    glob = []
    for col in ("^IXIC", "^GSPC", "^TWII"):
        v = np.log(known[col].ffill().bfill().to_numpy())
        d = np.diff(v, prepend=v[0])  # row t: change known at the open of day t
        # shift up one row so that "row t uses rows <= t - 1" below gives the change known at the open of day t
        glob.append(np.repeat(np.append(d[1:], 0.0)[:, None], S, axis=1))
    x = np.stack(own + glob, axis=-1)  # (T, S, F), row t is known at the close of day t
    y = r.where(traded).to_numpy()
    return dates, x, y


def lagged(x, L):
    """(T, S, F) -> (T, S, F, L): row t holds rows t - 1, ..., t - L (lag 0 first); NaN where not available."""
    T = x.shape[0]
    out = np.full(x.shape + (L,), np.nan, dtype=np.float32)
    for j in range(L):
        out[1 + j:, :, :, j] = x[: T - 1 - j]
    return out


def legendre(L, p):
    s = np.linspace(-1, 1, L)
    return np.polynomial.legendre.legvander(s, p)  # (L, p + 1)


def ridge(a, y, alpha):
    return np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ y)


def fit_bilinear(Z, y, R, basis, alpha, iters=15):
    """Z (N, F, L), y (N,) centred. v_k = basis @ c_k. Returns u (R, F), c (R, q)."""
    N, F, L = Z.shape
    Zb = Z @ basis  # (N, F, q)
    q = basis.shape[1]
    # init from the full ridge on vec(Zb): the leading R singular pairs of its (F, q) weights
    W = ridge(Zb.reshape(N, -1), y, alpha).reshape(F, q)
    U_, s, Vt = np.linalg.svd(W)
    u = (U_[:, :R] * np.sqrt(s[:R])).T
    c = (Vt[:R].T * np.sqrt(s[:R])).T
    for _ in range(iters):
        a = np.einsum("nfq,rq->nrf", Zb, c).reshape(N, -1)
        u = ridge(a, y, alpha).reshape(R, F)
        a = np.einsum("nfq,rf->nrq", Zb, u).reshape(N, -1)
        c = ridge(a, y, alpha).reshape(R, q)
    return u, c


def predict(Z, u, c, basis):
    return np.einsum("nfl,lq,rq,rf->n", Z, basis, c, u)


def main() -> None:
    dates, x, y = signals()
    T, S, F = x.shape
    first = max(LAGS) + 1
    split_rows = {n: np.where((dates >= a) & (dates < b) & (np.arange(T) >= first))[0] for n, a, b in SPLITS}
    Zall = lagged(x, max(LAGS))  # (T, S, F, Lmax)
    data = {}
    for n, rows in split_rows.items():
        z, t = Zall[rows].reshape(-1, F, max(LAGS)), y[rows].reshape(-1)
        day = np.repeat(rows, S)
        ok = np.isfinite(t) & np.isfinite(z).all(axis=(1, 2))
        data[n] = (z[ok], t[ok], day[ok])
    sd = data["train"][0][:, :, 0].std(0)  # per-signal scale
    data = {n: ((z / sd[None, :, None]).astype(np.float64), t, d) for n, (z, t, d) in data.items()}
    c0 = data["train"][1].mean()
    print(f"F {F} signals, stock-days " + ", ".join(f"{k} {len(v[1])}" for k, v in data.items()))

    def report(label, preds, nparam):
        cells = []
        for k in ("train", "val", "test"):
            _, t, day = data[k]
            m, se = score(preds[k], t, day)
            cells.append(f"{BP * m:+6.1f} ± {BP * se:3.1f}")
        r2 = []
        for k in ("val", "test"):
            t = data[k][1]
            r2.append(1 - np.mean((preds[k] - t) ** 2) / np.mean((c0 - t) ** 2))
        _, t, day = data["test"]
        d, dse = score(np.ones(len(t)), (np.sign(preds["test"]) + 1) * t, day)
        up = 100 * np.mean(preds["test"] > 0)
        print(f"{label:34s} {nparam:5d}  " + "  ".join(cells) + f"   R2 {r2[0]:+.4f} {r2[1]:+.4f}"
              f"   test-down {BP * d:+5.1f} ± {BP * dse:3.1f}   up {up:4.1f}%", flush=True)

    print(f"{'model':34s} {'param':>5s}  {'train':>12s}  {'val':>12s}  {'test':>12s}")
    report("always down", {k: -np.ones(len(v[1])) for k, v in data.items()}, 0)

    for L in LAGS:
        Z = {k: v[0][:, :, :L] for k, v in data.items()}
        yt = data["train"][1] - c0

        def choose(fit, pred):
            best = None
            for alpha in ALPHAS:
                params = fit(alpha)
                v = np.mean((pred(Z["val"], params) + c0 - data["val"][1]) ** 2)
                if best is None or v < best[0]:
                    best = (v, alpha, params)
            return best[1], best[2]

        # full ridge on vec(Z)
        flat = lambda z: z.reshape(len(z), -1)
        a, w = choose(lambda al: ridge(flat(Z["train"]), yt, al), lambda z, w: flat(z) @ w)
        report(f"L{L} full ridge (a1e{np.log10(a):.0f})", {k: flat(Z[k]) @ w + c0 for k in Z}, F * L)
        variants = [("free", np.eye(L))] + [(f"poly p{p}", legendre(L, p)) for p in (1, 2, 3) if p < L - 1]
        for name, B in variants:
            for R in (1, 2):
                a, (u, c) = choose(lambda al: fit_bilinear(Z["train"], yt, R, B, al),
                                   lambda z, uc: predict(z, uc[0], uc[1], B))
                report(f"L{L} bilinear {name} R{R} (a1e{np.log10(a):.0f})",
                       {k: predict(Z[k], u, c, B) + c0 for k in Z}, R * (F + B.shape[1]))


if __name__ == "__main__":
    main()
