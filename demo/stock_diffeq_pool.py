"""Difference equations with shared coefficients, and partially pooled per-window fits, then the ridge (TWStock).

demo/stock_diffeq.py fits a difference equation separately in every window; its forecast carries the noise of the
estimated coefficients. Here the coefficients are shared (or pulled towards shared values) to cut that variance.

Shared coefficients. For every signal f the lags L_f are r_{t-1}, r_{t-2}, r_{t-3} (the return) or dx_{t-1}, dx_{t-2},
dx_{t-3} (the increments of the integrated signals, see demo/stock_diffeq.py), and the shared equation of order 3
forecasts the next value (r_t or dx_t) as c_f + theta_f . L_f, the same theta_f for every stock and day.
- two-stage: theta_f by least squares on the signal's own next value (the years before 2022), then the ridge on the
  7 forecasts (as before);
- end-to-end GD: mu = b + sum_f v_f (c_f + theta_f . L_f), theta_f started at the two-stage values and v at 0,
  all trained by Adam (full batch, lr 1e-3, weight decay 1e-4 on v) on the MSE of r over the years before Y - 1,
  stopped early on the MSE of Y - 1 (checked every 50 steps, patience 5, at most 5000 steps), then forecasting Y;
  3 seeds of the step order are not needed (full batch), so one run per year;
- ridge on the 21 lags: the same function class as the end-to-end model (it is linear in L), solved exactly.

Partial pooling. In every window the equation of demo/stock_diffeq.py is solved with a prior at the pooled solution:
theta = (G + kappa n P)^-1 (g + kappa n P theta_0), G = A'A and g = A'y of the window, n its number of equations,
theta_0 the least-squares solution pooled over all windows of the years before 2022 (per signal, W, q) and P the mean
of G / n over those windows. kappa = 0 is the per-window LS, kappa -> inf the shared coefficients. kappa = 0.3, 3, 30.

All models: walk-forward ridge on 2022 - 2026 as demo/stock_extra.py on the same stock-days; scores in units of the
open-to-close return, day-clustered.

    python demo/stock_diffeq_pool.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_diffeq import WINDOWS, fit_window, series  # noqa: E402
from stock_extra import extra_features, signals  # noqa: E402
from stock_poly import features, load  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge, clustered_mean  # noqa: E402

KAPPAS = (0.3, 3.0, 30.0)
NLAG = 3
U = 1e4


def shift(X):
    return np.concatenate([np.full_like(X[:1], np.nan), X[:-1]], axis=0)  # row t uses data up to t - 1


def lag_features(sig):
    """(T, S, F, NLAG) lags known at the close of t (shifted later) and the next value (T, S, F) to forecast."""
    L, nxt = [], []
    for kind, x in sig.values():
        z = x if kind == "level" else np.diff(x, axis=0, prepend=np.nan)
        lg = np.full(z.shape + (NLAG,), np.nan)
        for j in range(NLAG):
            lg[j:, :, j] = z[:len(z) - j]  # row t: z_t, z_{t-1}, z_{t-2}
        L.append(lg)
        nxt.append(np.append(z[1:], np.full((1, z.shape[1]), np.nan), axis=0))  # row t: z_{t+1}
    return np.stack(L, axis=2), np.stack(nxt, axis=2)


def pooled_features(sig, train_end):
    feats, names = [], []
    for name, (kind, x) in sig.items():
        for q, Ws in WINDOWS.items():
            for W in Ws:
                A, y, f = fit_window(kind, x, W, q)
                bad = ~np.isfinite(A).all((-1, -2)) | ~np.isfinite(y).all(-1)
                A, y, f = np.nan_to_num(A), np.nan_to_num(y), np.nan_to_num(f)
                G = np.einsum("...nk,...nl->...kl", A, A)
                g = np.einsum("...nk,...n->...k", A, y)
                n = A.shape[-2]
                tr = ~bad[:train_end]
                theta0 = np.linalg.lstsq(G[:train_end][tr].sum(0), g[:train_end][tr].sum(0), rcond=None)[0]
                P = G[:train_end][tr].mean(0) / n
                for kp in KAPPAS:
                    th = np.linalg.solve(G + kp * n * P, (g + kp * n * P @ theta0)[..., None])[..., 0]
                    fc = np.sum(f * th, axis=-1)
                    fc[bad] = np.nan
                    feats.append(fc); names.append(f"{name}:W{W}q{q}:k{kp}")
        print(f"  {name} done", flush=True)
    return shift(np.stack(feats, axis=-1).astype(np.float32)), np.array(names)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dates, r, traded, gap, known = load()
    T, S = r.shape
    train_rows = np.asarray(dates < "2022-01-01")
    train_end = int(train_rows.sum())
    sig = series(dates, r, traded, gap, known)
    F = len(sig)
    Lraw, nxt = lag_features(sig)
    # two-stage shared coefficients: LS of each signal's next value on its own lags, years before 2022
    theta, c = np.zeros((F, NLAG)), np.zeros(F)
    for i in range(F):
        a, b = Lraw[FIRST:train_end - 1, :, i].reshape(-1, NLAG), nxt[FIRST:train_end - 1, :, i].reshape(-1)
        ok = np.isfinite(a).all(1) & np.isfinite(b)
        A = np.column_stack([np.ones(ok.sum()), a[ok]])
        sol = np.linalg.lstsq(A, b[ok], rcond=None)[0]
        c[i], theta[i] = sol[0], sol[1:]
    print("shared theta (lag 1..3) per signal: " + "; ".join(f"{k} {np.round(theta[i], 3)}" for i, k in enumerate(sig)))
    L = shift(Lraw)  # (T, S, F, NLAG), row t: lags up to t - 1
    shared_fc = c + np.einsum("tsfj,fj->tsf", L, theta)
    Xp, pnames = pooled_features(sig, train_end)
    lo, hi = np.nanquantile(Xp[train_rows].reshape(-1, Xp.shape[-1]), [0.001, 0.999], axis=0)
    Xp = np.clip(Xp, lo, hi)
    Xb, y, bnames = features(dates, r, traded, gap, known)
    Xe, _, egroups = extra_features(signals(dates, known, S), train_rows)
    blocks = {"base": Xb, "extra": Xe, "lags": L.reshape(T, S, -1), "shared": shared_fc, "pool": Xp}
    off, o = {}, 0
    for k, v in blocks.items():
        off[k] = (o, o + v.shape[-1]); o += v.shape[-1]
    X = np.concatenate(list(blocks.values()), axis=-1).reshape(T * S, -1)
    yf = y.reshape(-1)
    dayf, yearf = np.repeat(np.arange(T), S), np.repeat(dates.year.to_numpy(), S)
    ok = np.repeat(np.arange(T) >= FIRST, S) & np.isfinite(X).all(1) & np.isfinite(yf)
    X, yf, dayf, yearf = X[ok], yf[ok], dayf[ok], yearf[ok]
    print(f"{ok.sum()} stock-days; {len(pnames)} pooled features", flush=True)

    def cols(**parts):
        m = np.zeros(X.shape[1], bool)
        for k, sel in parts.items():
            a, b = off[k]
            m[a:b] = True if sel is True else sel
        return m
    p0 = np.array([n.endswith("p0") for n in bnames])
    rng = egroups == "range"
    kp = lambda k: np.array([n.endswith(f":k{k}") for n in pnames])
    SETS = {"poly p0 + range": cols(base=p0, extra=rng), "poly p0": cols(base=p0),
            "shared DE q3, two-stage (7)": cols(shared=True), "ridge on the 21 lags": cols(lags=True),
            "shared DE two-stage + range": cols(shared=True, extra=rng)}
    for k in KAPPAS:
        SETS[f"pooled DE kappa {k} (91)"] = cols(pool=kp(k))
        SETS[f"pooled DE kappa {k} + range"] = cols(pool=kp(k), extra=rng)
    SETS["poly p0 + range + pooled kappa 3"] = cols(base=p0, extra=rng, pool=kp(3.0))

    te = np.isin(yearf, EVAL)
    yt, dt = yf[te], dayf[te]
    _, inv = np.unique(dt, return_inverse=True)
    mkt = (np.bincount(inv, weights=yt) / np.bincount(inv))[inv]
    fm = lambda t: f"{U * t[0]:+5.1f}e-4 ± {U * t[1]:3.1f}e-4"
    out = {}

    def show(label, mu, c0s, k, extra=""):
        out[label] = mu
        sg = np.where(mu > 0, 1.0, -1.0)
        nz = yt != 0
        base = np.mean((c0s - yt) ** 2)
        ref = out["poly p0 + range"]
        dm = clustered_mean((ref - yt) ** 2 - (mu - yt) ** 2, dt)
        sr = np.where(ref > 0, 1.0, -1.0)
        print(f"{label:34s} K {k:3d}  R2 {1 - np.mean((mu - yt) ** 2) / base:+.4f}  acc "
              f"{np.mean((sg[nz] > 0) == (yt[nz] > 0)):5.1%}  up {np.mean(sg > 0):4.0%}  score "
              f"{fm(clustered_mean(sg * yt, dt))}  cross {fm(clustered_mean(sg * (yt - mkt), dt))}  | vs poly p0+range:"
              f" R2 {dm[0] / base:+.4f} ± {dm[1] / base:.4f}  score {fm(clustered_mean((sg - sr) * yt, dt))}{extra}",
              flush=True)

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
        show(label, mu, c0s, int(m_.sum()))

    # end-to-end GD on the shared equation
    a_, b_ = off["lags"]
    Lf = X[:, a_:b_].reshape(-1, F, NLAG)
    t = lambda v: torch.tensor(v, dtype=torch.float32, device=device)
    mu, c0s, steps = np.empty(te.sum()), np.empty(te.sum()), []
    for Y in EVAL:
        tr, va, ts = yearf < Y - 1, yearf == Y - 1, yearf == Y
        Ltr, Lva, Lte = t(Lf[tr]), t(Lf[va]), t(Lf[ts])
        ytr, yva = t(yf[tr]), t(yf[va])
        sz = yf[tr].std()
        th = torch.nn.Parameter(t(theta)); cc = torch.nn.Parameter(t(c))
        v = torch.nn.Parameter(torch.zeros(F, device=device)); b0 = torch.nn.Parameter(t(yf[tr].mean()))
        fsd = t(np.nanstd(np.einsum("nfj,fj->nf", Lf[tr], theta) + c, axis=0) + 1e-12)
        pred = lambda Lx: b0 + (((cc + (Lx * th).sum(-1)) / fsd) * v).sum(-1) * sz
        opt = torch.optim.Adam([{"params": [th, cc, b0]}, {"params": [v], "weight_decay": 1e-4}], lr=1e-3)
        best, state, bad = np.inf, None, 0
        for step in range(5001):
            if step % 50 == 0:
                with torch.no_grad():
                    vl = (pred(Lva) - yva).pow(2).mean().item()
                if vl < best:
                    best, state, bad, bstep = vl, [p.detach().clone() for p in (th, cc, v, b0)], 0, step
                else:
                    bad += 1
                    if bad >= 5:
                        break
            loss = (pred(Ltr) - ytr).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for p, s_ in zip((th, cc, v, b0), state):
                p.copy_(s_)
            idx = yearf[te] == Y
            mu[idx], c0s[idx] = pred(Lte).cpu().numpy(), yf[yearf < Y].mean()
        steps.append(bstep)
    show("shared DE q3, end-to-end GD", mu, c0s, F * (NLAG + 2) + 1, f"  (best steps {steps})")


if __name__ == "__main__":
    main()
