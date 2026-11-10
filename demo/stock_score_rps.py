"""E4n: quintile probabilities (RPS) walk-forward, a set model that outputs them, and the volatility forecast as an
input of the sign-score set model (TWStock).

Quintiles: on every day the usable stocks (the rows of the set model) are ranked by r and split into five equal
groups (q = floor(5 rank / n), 0 = lowest). A forecast gives every stock-day the five probabilities and is scored by
RPS = 1/5 sum_j (F_j - C_j)^2 (F, C cumulative forecast and outcome; uniform about 0.16, lower is better). The 50-stock
subset is scored on the quintiles of all usable stocks. All gains and differences are paired and day-clustered.

Volatility forecast sigma_hat (rebuilt walk-forward, no import of the earlier volatility script): target the
Garman-Klass variance RV = 0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2 (floored at the 1% quantile of the positive 2020
values); log RV regressed by least squares on, all known before 09:00 of day t:
- HAR terms: log of the 1-, 5- and 22-day means of RV up to t - 1;
- pre-open gaps (in place of the stock's own opening gap, which is only known at 09:00): log squares of the Nikkei and
  KOSPI open gaps, of the 08:45 TAIEX futures gap and of the futures gap times the stock's 60-day gap beta;
- market: log VIX (last US close), log 5-day mean of the squared TAIEX changes, the day mean of log RV of t - 1;
- the log volume of t - 1 relative to its 22-day mean.
For fold F the fit uses the years before F, mapped back with a smearing factor; fold Y forecasts the test year Y and
fold Y - 1 the selection year. Sanity check per year: QLIKE against the training-mean constant and EWMA (lambda chosen
on the training QLIKE), and the HAR terms alone.

Part 'a' (python demo/stock_score_rps.py a, CPU): RPS of forecasts r_i = kappa mu_i + s_i eps_i with
s_i = sbar (sigma_i / sbar)^gamma (sbar the day's mean of sigma_hat), eps from the day-demeaned standardized residuals
(r - day mean) / sigma_hat of the training years only, independent across stocks. The probabilities are computed by
Monte Carlo of the day's cross-section (500 draws on Y - 1, 1000 on Y): the four quintile thresholds of each draw are
taken from the simulated cross-section, and each stock's probabilities are its exact CDF at those thresholds averaged
over the draws (a Rao-Blackwellized estimate, no 0/1 noise). Forecasts: uniform; sigma_hat only (kappa 0); MSE ridge mu
(the E4 55-input ridge, recomputed); the set-model mu of demo/runs/e4j_final2_pred.npz (d 32 soft Score, 40 seeds),
whose outputs are not in return units and whose scale changes between folds: each day's set forecasts are
z-scored over the day's usable stocks (only the ranks within a day matter for the RPS), then mapped by the slope of the
day-demeaned r on z over the out-of-sample rows of Y - 1 (for 2022, which has no out-of-sample forecasts of 2021, the
2021 forecasts of the part c base set models trained on 2020 with the epochs chosen on 2021); each mean with and
without sigma_hat.
kappa and gamma are chosen per fold by the RPS of Y - 1 (forecasts of Y - 1 from fits on the years before Y - 1).
Forecasts to demo/runs/e4n_a_pred.npz.
Part 'b' (python demo/stock_score_rps.py b): the d 32 set model (TokNet of demo/stock_score_itoken.py, the day's
inputs, E4h block) with 5 logits per stock, softmax, trained on the mean RPS against the realized quintile of the day;
B1 the 55 inputs, B2 55 + log sigma_hat and log sigma_hat minus the day's mean, B3 = B2 with an auxiliary soft-Score
head (TokNet.out2; loss RPS + 0.1 soft Score). Epochs on Y - 1 (patience 3, at most 40, at least 1, the total loss),
refit on all years before Y, 10 seeds 0..9, probabilities averaged. Sign forecasts derived from the probabilities,
E[q] - 3 (quintiles 1..5) and P(q >= 4) - P(q <= 2), rank the stocks within the day only (no market direction); also
with the market tilt of the set model added: f = m_t + (s_i - mean_t s) sd_t(set) / sd_t(s), m_t and sd_t the day's
mean and sd of the 40-seed set-model forecasts. Forecasts to demo/runs/e4n_b_pred.npz.
Part 'c' (python demo/stock_score_rps.py c): the d 32 soft-Score set model (E4j final2 recipe) on the 55 inputs and on
55 + the two sigma_hat inputs, 10 seeds 0..9 each, the same seeds (paired); forecasts to demo/runs/e4n_c_pred.npz,
with the per-seed mean forecasts of 2021 of the base models trained on 2020 (for the calibration of part a).
Usage: python demo/stock_score_rps.py [a | b | c]
"""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_itrans_news import U as UNIT, ghat_for  # noqa: E402
from stock_news_gap import gap_block  # noqa: E402
from stock_overnight import preopen_features  # noqa: E402
from stock_poly import ROOT, load  # noqa: E402
from stock_recurrent import base_features  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge, clustered_mean  # noqa: E402

RUNS = Path(__file__).resolve().parent / "runs"
FOLDS = (EVAL[0] - 1,) + tuple(EVAL)
SEEDS = tuple(range(10))
KAPPAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
GAMMAS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5)
D_SEL, D_TEST = 500, 1000
NQ = 2001
LAM_B3 = 0.1
FINAL2 = "d_32_soft_Score_40_seeds"
fm = lambda t: f"{UNIT * t[0]:+5.1f}e-4 ± {UNIT * t[1]:3.1f}e-4"
fg4 = lambda t: f"{1e4 * t[0]:+5.1f}e-4 ± {1e4 * t[1]:3.1f}e-4"


def setup():
    t0 = time.time()
    dates, r, traded, gap, known = load()
    T, S = r.shape
    info = json.loads((ROOT / "info.json").read_text())
    stocks, top50 = info["Top100@Y"], set(info["Top50@Y"])
    X, y = base_features(dates, r, traded, gap, known, stocks)
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    g = gap.to_numpy()
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y) & np.isfinite(g)
    PO = preopen_features(dates, gap, traded)
    XG = np.concatenate([X, PO], -1)
    gh = {F: ghat_for(XG, g, ok, year, F) for F in FOLDS}
    Z = {F: np.concatenate([X, gap_block(gh[F], ok)], -1) for F in FOLDS}
    ev = ok & np.isin(year, EVAL)
    is50 = np.array([s in top50 for s in stocks])[None, :]
    dayi = np.repeat(np.arange(T)[:, None], S, 1)
    print(f"setup {time.time() - t0:.0f} s: T {T}, S {S}, {Z[EVAL[0]].shape[-1]} inputs", flush=True)
    return SimpleNamespace(dates=dates, r=r, traded=traded, gap=gap, known=known, T=T, S=S, stocks=stocks, X=X, y=y,
                           year=year, yrs=dates.year.to_numpy(), g=g, ok=ok, PO=PO, gh=gh, Z=Z, ev=ev, is50=is50,
                           dayi=dayi, trd=traded.to_numpy())


# ---------------------------------------------------------------- volatility forecast

def shift(a):
    return np.concatenate([np.full_like(a[:1], np.nan), a[:-1]], axis=0)


def vol_panel(d):
    cols = {"o": "etl:adj_open", "h": "etl:adj_high", "l": "etl:adj_low", "c": "etl:adj_close", "v": "price:成交股數"}
    p = {k: {} for k in cols}
    for s in d.stocks:
        f = pd.read_csv(ROOT / "market" / f"{s}.csv", parse_dates=["date"]).set_index("date")
        for k, c in cols.items():
            p[k][s] = f[c]
    p = {k: pd.DataFrame(v).sort_index().reindex(d.dates)[d.stocks].to_numpy() for k, v in p.items()}
    o, h, l, c = p["o"], p["h"], p["l"], p["c"]
    use = d.trd & np.isfinite(h) & np.isfinite(l) & (h >= l) & (l > 0) & (o > 0) & np.isfinite(c)
    with np.errstate(invalid="ignore", divide="ignore"):
        rv = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    rv = np.where(use, rv, np.nan)
    ref = rv[(d.yrs == 2020)]
    floor = np.nanquantile(np.where(ref > 0, ref, np.nan), 0.01)
    rv = np.where(use, np.maximum(rv, floor), np.nan)
    lrv = np.log(rv)
    rv_f = np.exp(pd.DataFrame(lrv).ffill().to_numpy())
    feats = {}
    for w in (1, 5, 22):
        feats[f"har:logRV{w}"] = shift(np.log(pd.DataFrame(rv_f).rolling(w, min_periods=1).mean().to_numpy()))
    fl = 1e-6
    for k, n in enumerate(("n225", "ks11", "futures", "futures x beta")):
        feats[f"pre:log {n} gap^2"] = np.log(d.PO[..., k] ** 2 + fl)
    S = d.S
    vix = np.log(d.known["^VIX"].ffill().bfill().to_numpy())
    tw = np.log(d.known["^TWII"].ffill().bfill().to_numpy())
    dtw2 = np.diff(tw, prepend=tw[0]) ** 2
    feats["mkt:logVIX"] = np.repeat(vix[:, None], S, 1)
    feats["mkt:logTAIEX5"] = np.repeat(np.log(pd.Series(dtw2).rolling(5, min_periods=1).mean().to_numpy() + 1e-8)[:, None],
                                       S, 1)
    feats["mkt:mean logRV1"] = shift(np.repeat(np.nanmean(lrv, 1, keepdims=True), S, 1))
    vol = np.log(np.where(d.trd, p["v"], np.nan) + 1.0)
    vol_f = pd.DataFrame(vol).ffill().to_numpy()
    feats["vol:rel22"] = shift(vol_f - pd.DataFrame(vol_f).rolling(22, min_periods=1).mean().to_numpy())
    names = list(feats)
    return rv, rv_f, np.stack([feats[n] for n in names], -1), names


def har_fit(Fx, rv, fitm):
    fin = np.isfinite(Fx).all(-1)
    m = fitm & fin & np.isfinite(rv)
    A = Fx[m]
    mu, sd = A.mean(0), A.std(0) + 1e-12
    A1 = np.column_stack([np.ones(len(A)), (A - mu) / sd])
    beta = np.linalg.lstsq(A1, np.log(rv[m]), rcond=None)[0]
    smear = np.mean(rv[m] / np.exp(A1 @ beta))
    out = np.full(rv.shape, np.nan)
    out[fin] = smear * np.exp(np.column_stack([np.ones(fin.sum()), (Fx[fin] - mu) / sd]) @ beta)
    return out, beta


def ewma(rv_f, lam):
    s = np.empty_like(rv_f)
    s[0] = np.nanmean(rv_f[:60], 0)
    for t in range(1, len(rv_f)):
        s[t] = lam * s[t - 1] + (1 - lam) * np.where(np.isfinite(rv_f[t - 1]), rv_f[t - 1], s[t - 1])
    return s


def qlike(rv, s):
    q = rv / s
    return q - np.log(q) - 1


def sigma_folds(d, verbose=False):
    """{fold F: sigma_hat (T, S) from the HAR fit on the years before F}, NaN filled with the day's median."""
    rv, rv_f, Fx, names = vol_panel(d)
    out, betas = {}, {}
    for F in FOLDS:
        s2, beta = har_fit(Fx, rv, d.ok & (d.year < F))
        med = np.nanmedian(np.where(d.ok, s2, np.nan), 1, keepdims=True)
        s2 = np.where(np.isfinite(s2), s2, np.nan_to_num(np.repeat(med, d.S, 1), nan=np.nanmedian(s2)))
        out[F], betas[F] = np.sqrt(s2), beta
    if verbose:
        print("\n######## volatility forecast, walk-forward QLIKE per year (rows: usable stock-days with RV; lower is "
              "better)")
        har_cols = [i for i, n in enumerate(names) if n.startswith("har:")]
        print("| year | stock-days | constant (train mean) | EWMA (lambda) | HAR 1/5/22 | HAR + pre-open + market + volume "
              "| var z (full) |")
        print("|---|---|---|---|---|---|---|")
        tot = {k: [] for k in ("c", "e", "h", "f")}
        for Y in EVAL:
            tr = d.ok & (d.year < Y) & np.isfinite(rv)
            te = d.ok & (d.year == Y) & np.isfinite(rv)
            c = np.mean(rv[tr])
            best = None
            for lam in (0.8, 0.85, 0.9, 0.94, 0.96, 0.97, 0.98, 0.99):
                s = ewma(rv_f, lam)
                s = np.where(np.isfinite(s), s, c)
                q = np.mean(qlike(rv[tr], s[tr]))
                if best is None or q < best[0]:
                    best = (q, lam, s)
            sh, _ = har_fit(Fx[..., har_cols], rv, d.ok & (d.year < Y))
            sh = np.where(np.isfinite(sh), sh, c)
            qs = {"c": qlike(rv[te], c), "e": qlike(rv[te], best[2][te]), "h": qlike(rv[te], sh[te]),
                  "f": qlike(rv[te], out[Y][te] ** 2)}
            for k, v in qs.items():
                tot[k].append(v)
            z = d.y[te] / out[Y][te]
            print(f"| {Y} | {te.sum()} | {qs['c'].mean():.4f} | {qs['e'].mean():.4f} ({best[1]:g}) | "
                  f"{qs['h'].mean():.4f} | {qs['f'].mean():.4f} | {np.var(z):.2f} |", flush=True)
        a = {k: np.concatenate(v) for k, v in tot.items()}
        print(f"| 2022-2026 | {len(a['c'])} | {a['c'].mean():.4f} | {a['e'].mean():.4f} | {a['h'].mean():.4f} | "
              f"{a['f'].mean():.4f} | |")
        print("standardized coefficients of the full HAR, fold 2026: " +
              ", ".join(f"{n} {b:+.3f}" for n, b in zip(names, betas[EVAL[-1]][1:])), flush=True)
    return out


def sig_inputs(sig, ok):
    """(T, S, 2): log sigma_hat and log sigma_hat minus its mean over the day's usable stocks."""
    ls = np.log(sig)
    m = np.nanmean(np.where(ok, ls, np.nan), 1, keepdims=True)
    return np.stack([ls, ls - np.nan_to_num(m)], -1)


# ---------------------------------------------------------------- quintiles and RPS

def quintiles(y, ok):
    q = np.full(y.shape, -1, dtype=int)
    for t in range(len(y)):
        i = np.nonzero(ok[t])[0]
        if len(i):
            q[t, i] = (np.argsort(np.argsort(y[t, i], kind="stable"), kind="stable") * 5) // len(i)
    return q


def rps_rows(pge, q):
    """pge (n, 4): P(q >= j), j = 1..4; q (n,) realized 0..4 -> RPS per row."""
    F = 1.0 - pge
    C = (q[:, None] < np.arange(1, 5)[None]).astype(float)
    return ((F - C) ** 2).sum(1) / 5.0


def eps_quantiles(d, sig, fit):
    """quantile grid of the day-demeaned standardized residuals (r - day mean) / sigma_hat on the rows `fit`."""
    rm = np.nanmean(np.where(d.ok, d.y, np.nan), 1, keepdims=True)
    e = ((d.y - rm) / sig)[fit & d.ok]
    e = (e - e.mean()) / e.std()
    return np.quantile(e, np.linspace(0, 1, NQ))


class MC:
    """Monte Carlo of the day's cross-section with common random numbers for every (kappa, gamma)."""

    def __init__(self, d, days, qgrid, D, seed):
        self.days, self.qg, self.lin = days, qgrid, np.linspace(0, 1, NQ)
        self.idx = [np.nonzero(d.ok[t])[0] for t in days]
        self.e = [np.interp(np.random.default_rng(seed * 100003 + t).random((D, len(i))), self.lin, qgrid)
                  for t, i in zip(days, self.idx)]

    def pge(self, m, sig, gamma):
        out = np.zeros(m.shape + (4,))
        for t, i, e in zip(self.days, self.idx, self.e):
            n = len(i)
            sg = sig[t, i]
            sb = sg.mean()
            s = sb * (sg / sb) ** gamma
            mu = m[t, i]
            srt = np.sort(mu[None] + s[None] * e, 1)
            k = np.ceil(np.arange(1, 5) * n / 5).astype(int)
            b = 0.5 * (srt[:, k - 1] + srt[:, np.minimum(k, n - 1)])            # (D, 4)
            z = (b[:, :, None] - mu[None, None]) / s[None, None]               # (D, 4, n)
            out[t, i] = (1.0 - np.interp(z, self.qg, self.lin).mean(0)).T
        return out


def rps_of(pge, q, mask):
    out = np.full(q.shape, np.nan)
    out[mask] = rps_rows(pge[mask], q[mask])
    return out


def xs_z(f, ok):
    """f z-scored over each day's usable stocks (NaN elsewhere)."""
    fm_ = np.where(ok & np.isfinite(f), f, np.nan)
    return (fm_ - np.nanmean(fm_, 1, keepdims=True)) / np.nanstd(fm_, 1, keepdims=True)


def xs_slope(mu, y, m):
    """slope of the day-demeaned y on the day-demeaned mu over the rows m."""
    mm = lambda a: a - np.nan_to_num(np.nanmean(np.where(m, a, np.nan), 1, keepdims=True))
    a, b = mm(mu)[m], mm(y)[m]
    return np.sum(a * b) / np.sum(a * a)


def rps_table(d, R, refs, title, extra=None):
    """R: name -> (T, S) RPS; refs: list of reference names for the paired difference columns."""
    ev, is50, yr, di = d.ev, d.is50, d.year, d.dayi
    u = R["uniform"]
    cm = lambda x, m: clustered_mean(x[m], di[m])
    print(f"\n######## {title}")
    hdr = ("| forecast | RPS (100) | gain vs uniform | " + " | ".join(f"− {r_}" for r_ in refs) +
           " | RPS (50) | gain (50) | " + " | ".join(f"gain {Y}" for Y in EVAL) + " |")
    print(hdr)
    print("|" + "---|" * (5 + len(refs) + len(EVAL)))
    for k, v in R.items():
        m = ev & np.isfinite(v)
        cells = [f"{np.mean(v[m]):.5f}", fg4(cm(u - v, m))]
        for r_ in refs:
            cells.append("—" if r_ == k else fg4(cm(R[r_] - v, m & np.isfinite(R[r_]))))
        m5 = m & is50
        cells += [f"{np.mean(v[m5]):.5f}", fg4(cm(u - v, m5))]
        cells += [fg4(cm(u - v, m & (yr == Y))) if (m & (yr == Y)).any() else "—" for Y in EVAL]
        print(f"| {k}{'' if extra is None else extra.get(k, '')} | " + " | ".join(cells) + " |", flush=True)


def sign_table(d, res, ridge, setref, title):
    """res: name -> (T, S) forecast; sign score = mean sign(f) r, day-clustered."""
    ev, is50, yr, di, y = d.ev, d.is50, d.year, d.dayi, np.nan_to_num(d.y)
    sg = {k: np.where(v > 0, 1.0, -1.0) for k, v in res.items()}
    print(f"\n######## {title}")
    print(f"| model | sign score (100) | Δ vs {ridge} | Δ vs {setref} | sign score (50) | Δ vs {setref} (50) | up share | "
          + " | ".join(str(Y) for Y in EVAL) + " |")
    print("|" + "---|" * (7 + len(EVAL)))
    for k in res:
        c = lambda m_, ref=None: fm(clustered_mean(((sg[k] - (0 if ref is None else sg[ref])) * y)[m_], di[m_]))
        line = (f"| {k} | {c(ev)} | {'—' if k == ridge else c(ev, ridge)} | {'—' if k == setref else c(ev, setref)} | "
                f"{c(ev & is50)} | {'—' if k == setref else c(ev & is50, setref)} | {np.mean(res[k][ev] > 0):.3f} | "
                + " | ".join(c(ev & (yr == Y)) for Y in EVAL))
        print(line + " |", flush=True)


# ---------------------------------------------------------------- part a

def part_a(d):
    t0 = time.time()
    sig = sigma_folds(d, verbose=True)
    print(f"sigma_hat folds {time.time() - t0:.0f} s", flush=True)
    q = quintiles(d.y, d.ok)
    T, S = d.T, d.S
    f2 = np.load(RUNS / "e4j_final2_pred.npz")
    setf = xs_z(f2[FINAL2], d.ok)
    cpath = RUNS / "e4n_c_pred.npz"
    cal21 = xs_z(np.load(cpath)["sel2021_base_per_seed"], d.ok) if cpath.exists() else None
    print(f"2021 calibration forecasts of the set model from part c: {'found' if cal21 is not None else 'missing'}",
          flush=True)
    rows = ["uniform", "σ̂ only", "ridge μ only", "ridge μ + σ̂", "set μ only", "set μ + σ̂"]
    R = {k: np.full((T, S), np.nan) for k in rows}
    P = {k: np.full((T, S, 5), np.nan, dtype=np.float32) for k in rows}
    ridge = np.full((T, S), np.nan)
    sig_wf = np.full((T, S), np.nan)
    chosen = {k: {} for k in rows}
    for Y in EVAL:
        t1 = time.time()
        inner, sel, rows_, te = (d.ok & (d.year < Y - 1), d.ok & (d.year == Y - 1), d.ok & (d.year < Y),
                                 d.ok & (d.year == Y))
        Zs, Zr = d.Z[Y - 1], d.Z[Y]
        m = Ridge(Zs[inner], d.y[inner])
        a = min(ALPHAS, key=lambda a_: np.mean((m.predict(Zs[sel], m.weights(a_)) - d.y[sel]) ** 2))
        mu_sel = np.full((T, S), np.nan)
        mu_sel[sel] = m.predict(Zs[sel], m.weights(a))
        mr = Ridge(Zr[rows_], d.y[rows_])
        mu_te = np.full((T, S), np.nan)
        mu_te[te] = mr.predict(Zr[te], mr.weights(a))
        ridge[te] = mu_te[te]
        sig_wf[te] = sig[Y][te]
        # set-model mu mapped to return units on Y - 1
        if Y - 1 >= EVAL[0]:
            cal = np.where(sel, setf, np.nan)
        elif cal21 is not None:
            cal = np.where(sel, cal21, np.nan)
        else:
            cal = None
        if cal is not None:
            b = xs_slope(cal, d.y, sel & np.isfinite(cal))
            set_sel, set_te = np.nan_to_num(b * cal), np.nan_to_num(b * setf)
        q_sel = eps_quantiles(d, sig[Y - 1], inner)
        q_te = eps_quantiles(d, sig[Y], rows_)
        days_sel = np.nonzero(sel.any(1))[0]
        days_te = np.nonzero(te.any(1))[0]
        mc_sel = MC(d, days_sel, q_sel, D_SEL, Y - 1)
        mc_te = MC(d, days_te, q_te, D_TEST, Y)
        zero = np.zeros((T, S))
        specs = {"σ̂ only": (zero, zero, (0.0,), GAMMAS),
                 "ridge μ only": (np.nan_to_num(mu_sel), np.nan_to_num(mu_te), KAPPAS[1:], (0.0,)),
                 "ridge μ + σ̂": (np.nan_to_num(mu_sel), np.nan_to_num(mu_te), KAPPAS, GAMMAS)}
        if cal is not None:
            specs["set μ only"] = (set_sel, set_te, KAPPAS[1:], (0.0,))
            specs["set μ + σ̂"] = (set_sel, set_te, KAPPAS, GAMMAS)
        uni = np.tile(np.array([0.8, 0.6, 0.4, 0.2]), (T, S, 1))
        R["uniform"][te] = rps_of(uni, q, te)[te]
        P["uniform"][te] = 0.2
        msg = [f"ridge alpha {a:g}"]
        if cal is not None:
            msg.append(f"set slope {b:.3g} (calibration rows: {'2021 part c' if Y - 1 < EVAL[0] else Y - 1})")
        for k, (ms, mt, kap, gam) in specs.items():
            best = None
            for kp in kap:
                for ga in gam:
                    v = np.mean(rps_of(mc_sel.pge(kp * ms, sig[Y - 1], ga), q, sel)[sel])
                    if best is None or v < best[0]:
                        best = (v, kp, ga)
            _, kp, ga = best
            pg = mc_te.pge(kp * mt, sig[Y], ga)
            R[k][te] = rps_of(pg, q, te)[te]
            cum = np.concatenate([np.ones((T, S, 1)), pg, np.zeros((T, S, 1))], -1)
            P[k][te] = (cum[..., :-1] - cum[..., 1:])[te]
            chosen[k][Y] = (kp, ga)
            msg.append(f"{k}: kappa {kp:g}, gamma {ga:g} (Y - 1 RPS {best[0]:.5f})")
        print(f"{Y} done in {time.time() - t1:.0f} s\n  " + "\n  ".join(msg), flush=True)
    ref = f2["ref_MSE_ridge"]
    print(f"\nrecomputed ridge vs saved E4 ridge: max abs diff {np.nanmax(np.abs(ridge[d.ev] - ref[d.ev])):.3g}")
    extra = {k: " (κ, γ: " + ", ".join(f"{chosen[k][Y][0]:g}/{chosen[k][Y][1]:g}" for Y in EVAL if Y in chosen[k]) + ")"
             for k in rows if chosen[k]}
    rps_table(d, R, ["σ̂ only", "ridge μ + σ̂"], "RPS 2022-2026 (gains and differences ×1e-4 written as e-4, paired, "
              "day-clustered; − X = RPS of X minus RPS of the row, > 0 = the row is better)", extra)
    if cal21 is None:
        print("(set rows: 2023-2026 only, 2022 has no calibration forecasts without part c)")
    d23 = SimpleNamespace(**{**vars(d), "ev": d.ev & (d.year >= 2023)})
    rps_table(d23, R, ["σ̂ only", "ridge μ + σ̂", "set μ + σ̂"], "RPS 2023-2026 only (the set-model calibration from "
              "its own out-of-sample forecasts of Y - 1 in every year)")
    print(f"\npart a done in {time.time() - t0:.0f} s", flush=True)
    np.savez(RUNS / "e4n_a_pred.npz", y=d.y, ok=d.ok, ev=d.ev, year=d.year, is50=d.is50, q=q, sigma_hat=sig_wf,
             ridge=ridge, **{f"rps_{k}": v for k, v in R.items()}, **{f"prob_{k}": v for k, v in P.items()},
             chosen=np.array(json.dumps({k: {str(Y): v for Y, v in c.items()} for k, c in chosen.items()})))


# ---------------------------------------------------------------- parts b and c (GPU)

def rps_loss(logits, q):
    import torch
    p = torch.softmax(logits, -1)
    F = torch.cumsum(p, -1)[:, :4]
    C = (torch.arange(4, device=q.device)[None] >= q[:, None]).float()
    return ((F - C) ** 2).sum(-1).mean() / 5.0


def train_q(inp, Q, R, M, days, seed, lam, epochs=None, vdays=None):
    """it.train with 5 softmax logits per stock and the RPS loss (+ lam x soft Score on TokNet.out2)."""
    import torch
    import torch.nn as nn
    import stock_score_itoken as it
    DEV = it.DEV
    torch.manual_seed(seed)
    g = torch.Generator(device=DEV).manual_seed(seed)
    net = it.TokNet(inp.win_dim, inp.cur_dim, "e4h", 32, False, dropout=0.2)
    net.out = nn.Linear(32, 5)
    net = net.to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_ep, bad = np.inf, 0, 0

    def run(b):
        m = M[b]
        lo = net(*inp.batch(b), m)
        return lo[m], net.out2(net.h_last).squeeze(-1)[m], Q[b][m], R[b][m]

    def loss(lo, sg, q, r):
        l = rps_loss(lo, q)
        return l + lam * it.loss_fn("soft", sg, r) if lam > 0 else l

    for ep in range(1, (epochs or 40) + 1):
        net.train()
        perm = days[torch.randperm(len(days), device=DEV, generator=g)]
        for i in range(0, len(perm), 16):
            l = loss(*run(perm[i:i + 16]))
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
        if epochs is None:
            net.eval()
            with torch.no_grad():
                parts = list(zip(*(run(vdays[i:i + 64]) for i in range(0, len(vdays), 64))))
                v = loss(*(torch.cat(p_) for p_ in parts)).item()
            if v < best:
                best, best_ep, bad = v, ep, 0
            else:
                bad += 1
                if bad >= 3:
                    break
    net.eval()
    return net, best_ep


def part_b(d):
    import torch
    import stock_score_itoken as it
    t0 = time.time()
    DEV = it.DEV
    sig = sigma_folds(d)
    q = quintiles(d.y, d.ok)
    T, S = d.T, d.S
    Qt = torch.tensor(np.maximum(q, 0), dtype=torch.long, device=DEV)
    Rt = torch.tensor(np.where(d.ok, d.y, 0.0), dtype=torch.float32, device=DEV)
    Mt = torch.tensor(d.ok, device=DEV)
    VAR = {"B1 RPS, 55 inputs": (False, 0.0), "B2 RPS, 55 + σ̂": (True, 0.0),
           "B3 RPS + aux soft Score, 55 + σ̂": (True, LAM_B3)}
    PB = {k: np.full((T, S, 5), np.nan) for k in VAR}
    SG = np.full((T, S), np.nan)
    zw = np.zeros((T, S, 3))
    for Y in EVAL:
        t1 = time.time()
        inner, rows = d.ok & (d.year < Y - 1), d.ok & (d.year < Y)
        dd = lambda cond: torch.tensor(np.nonzero(cond & d.ok.any(1))[0], device=DEV)
        d_in, d_sel, d_rows, d_te = dd(d.yrs < Y - 1), dd(d.yrs == Y - 1), dd(d.yrs < Y), dd(d.yrs == Y)
        te = d.ok & (d.year == Y)
        msg = []
        for k, (use_sig, lam) in VAR.items():
            Zs, Zr = d.Z[Y - 1], d.Z[Y]
            if use_sig:
                Zs = np.concatenate([Zs, sig_inputs(sig[Y - 1], d.ok)], -1)
                Zr = np.concatenate([Zr, sig_inputs(sig[Y], d.ok)], -1)
            inp_s = it.Inputs("day", Zs, np.zeros((T, S)), d.y, d.ok, d.trd, zw, inner)
            inp_r = it.Inputs("day", Zr, np.zeros((T, S)), d.y, d.ok, d.trd, zw, rows)
            pr, sg, eps = np.zeros((T, S, 5)), np.zeros((T, S)), []
            for seed in SEEDS:
                _, ep = train_q(inp_s, Qt, Rt, Mt, d_in, seed, lam, vdays=d_sel)
                ep = max(ep, 1)
                net, _ = train_q(inp_r, Qt, Rt, Mt, d_rows, seed, lam, epochs=ep)
                with torch.no_grad():
                    for i in range(0, len(d_te), 64):
                        b = d_te[i:i + 64]
                        lo = net(*inp_r.batch(b), Mt[b])
                        bn = b.cpu().numpy()
                        pr[bn] += torch.softmax(lo, -1).cpu().numpy()
                        sg[bn] += net.out2(net.h_last).squeeze(-1).cpu().numpy()
                eps.append(ep)
            PB[k][te] = pr[te] / len(SEEDS)
            if lam > 0:
                SG[te] = sg[te]
            msg.append(f"{k}: {Zr.shape[-1]} inputs, epochs {eps}")
            del inp_s, inp_r
            torch.cuda.empty_cache()
        print(f"{Y} done in {time.time() - t1:.0f} s\n  " + "\n  ".join(msg), flush=True)
    # RPS, paired with the part a rows when available
    R = {}
    ap = RUNS / "e4n_a_pred.npz"
    if ap.exists():
        A = np.load(ap)
        for k in ("uniform", "σ̂ only", "ridge μ + σ̂", "set μ + σ̂"):
            R[k] = A[f"rps_{k}"]
    else:
        print("part a forecasts missing: only the uniform reference", flush=True)
        R["uniform"] = np.where(d.ev, rps_of(np.tile(np.array([0.8, 0.6, 0.4, 0.2]), (T, S, 1)), q, d.ev), np.nan)
    for k in VAR:
        pge = np.flip(np.cumsum(np.flip(PB[k], -1), -1), -1)[..., 1:]   # P(q >= j), j = 1..4
        R[k] = np.where(d.ev, rps_of(np.nan_to_num(pge), q, d.ev), np.nan)
    refs = [k for k in ("σ̂ only", "ridge μ + σ̂", "set μ + σ̂") if k in R] + ["B1 RPS, 55 inputs"]
    rps_table(d, R, refs, "RPS 2022-2026, set models trained on the RPS (paired, day-clustered, e-4)")
    # sign forecasts derived from the probabilities
    f2 = np.load(RUNS / "e4j_final2_pred.npz")
    setf, ridge = f2[FINAL2], f2["ref_MSE_ridge"]
    ev = d.ev
    msk = np.where(ev, 1.0, np.nan)
    sm = np.nanmean(setf * msk, 1, keepdims=True)
    ssd = np.nanstd(setf * msk, 1, keepdims=True)

    def tilt(s):
        mm = np.nanmean(s * msk, 1, keepdims=True)
        sd = np.nanstd(s * msk, 1, keepdims=True)
        return sm + (s - mm) * ssd / np.maximum(sd, 1e-12)

    res = {"always down": np.full((T, S), -1.0), "MSE ridge": ridge, "set model, d 32 soft Score, 40 seeds": setf}
    for k in VAR:
        eq = (PB[k] * np.arange(5)).sum(-1) - 2.0
        pd_ = PB[k][..., 3:].sum(-1) - PB[k][..., :2].sum(-1)
        tag = k.split(" ")[0]
        res[f"{tag}: E[q] − 3 (cross-sectional only)"] = eq
        res[f"{tag}: P(q≥4) − P(q≤2) (cross-sectional only)"] = pd_
        res[f"{tag}: E[q] − 3 + set-model market tilt"] = tilt(eq)
        res[f"{tag}: P(q≥4) − P(q≤2) + set-model market tilt"] = tilt(pd_)
    res["B3: aux soft-Score head"] = SG
    sign_table(d, res, "MSE ridge", "set model, d 32 soft Score, 40 seeds",
               "sign score of the forecasts derived from the quintile probabilities (mean sign(f) r, day-clustered)")
    print(f"\npart b done in {time.time() - t0:.0f} s", flush=True)
    np.savez(RUNS / "e4n_b_pred.npz", y=d.y, ok=d.ok, ev=d.ev, year=d.year, is50=d.is50, q=q,
             **{f"prob_{k.split(' ')[0]}": v for k, v in PB.items()}, B3_sign_head=SG,
             **{f"rps_{k.split(' ')[0]}": R[k] for k in VAR})


def part_c(d):
    import torch
    import stock_score_itoken as it
    t0 = time.time()
    DEV = it.DEV
    sig = sigma_folds(d)
    T, S = d.T, d.S
    cfg = ("day", "e4h", 32, "r", "soft", False, "none", 0.0, 0.2, 1e-2, 1e-3, 16, "all", len(SEEDS))
    Yt = torch.tensor(np.where(d.ok, d.y, 0.0), dtype=torch.float32, device=DEV)
    Mt = torch.tensor(d.ok, device=DEV)
    zw = np.zeros((T, S, 3))
    VAR = {"set base, 10 seeds": False, "set + σ̂, 10 seeds": True}
    res = {k: np.full((T, S), np.nan) for k in VAR}
    cal21 = np.full((T, S), np.nan)
    o_s = torch.zeros((T, S), device=DEV)
    for Y in EVAL:
        t1 = time.time()
        inner, rows, te = d.ok & (d.year < Y - 1), d.ok & (d.year < Y), d.ok & (d.year == Y)
        dd = lambda cond: torch.tensor(np.nonzero(cond & d.ok.any(1))[0], device=DEV)
        d_in, d_sel, d_rows, d_te = dd(d.yrs < Y - 1), dd(d.yrs == Y - 1), dd(d.yrs < Y), dd(d.yrs == Y)
        msg = []
        for k, use_sig in VAR.items():
            Zs, Zr = d.Z[Y - 1], d.Z[Y]
            if use_sig:
                Zs = np.concatenate([Zs, sig_inputs(sig[Y - 1], d.ok)], -1)
                Zr = np.concatenate([Zr, sig_inputs(sig[Y], d.ok)], -1)
            inp_s = it.Inputs("day", Zs, np.zeros((T, S)), d.y, d.ok, d.trd, zw, inner)
            inp_r = it.Inputs("day", Zr, np.zeros((T, S)), d.y, d.ok, d.trd, zw, rows)
            o, oc, eps = np.zeros((T, S)), np.zeros((T, S)), []
            for seed in SEEDS:
                _, ep = it.train(cfg, inp_s, Yt, o_s, Mt, d_in, seed, vdays=d_sel)
                ep = max(ep, 1)
                net, _ = it.train(cfg, inp_r, Yt, o_s, Mt, d_rows, seed, epochs=ep)
                with torch.no_grad():
                    for i in range(0, len(d_te), 64):
                        b = d_te[i:i + 64]
                        o[b.cpu().numpy()] += net(*inp_r.batch(b), o_s[b], Mt[b]).cpu().numpy()
                if Y == EVAL[0] and not use_sig:  # the base model fitted on 2020 with these epochs, forecasts of 2021
                    netc, _ = it.train(cfg, inp_s, Yt, o_s, Mt, d_in, seed, epochs=ep)
                    with torch.no_grad():
                        for i in range(0, len(d_sel), 64):
                            b = d_sel[i:i + 64]
                            oc[b.cpu().numpy()] += netc(*inp_s.batch(b), o_s[b], Mt[b]).cpu().numpy()
                eps.append(ep)
            res[k][te] = o[te]
            if Y == EVAL[0] and not use_sig:
                sel = d.ok & (d.year == Y - 1)
                cal21[sel] = oc[sel] / len(SEEDS)
            msg.append(f"{k}: {Zr.shape[-1]} inputs, epochs {eps}")
            del inp_s, inp_r
            torch.cuda.empty_cache()
        print(f"{Y} done in {time.time() - t1:.0f} s\n  " + "\n  ".join(msg), flush=True)
    f2 = np.load(RUNS / "e4j_final2_pred.npz")
    out = {"always down": np.full((T, S), -1.0), "MSE ridge": f2["ref_MSE_ridge"], **res}
    sign_table(d, out, "MSE ridge", "set base, 10 seeds", "sign score: σ̂ as an input of the d 32 soft-Score set model "
               "(10 seeds 0..9, paired)")
    sg = {k: np.where(v > 0, 1.0, -1.0) for k, v in res.items()}
    print(f"flips set + σ̂ vs set base: {np.mean(sg['set + σ̂, 10 seeds'][d.ev] != sg['set base, 10 seeds'][d.ev]):.3f}")
    print(f"\npart c done in {time.time() - t0:.0f} s", flush=True)
    np.savez(RUNS / "e4n_c_pred.npz", y=d.y, ok=d.ok, ev=d.ev, year=d.year, is50=d.is50, set_base=res["set base, 10 seeds"],
             set_sigma=res["set + σ̂, 10 seeds"], sel2021_base_per_seed=cal21)


def main() -> None:
    part = next((p_ for p_ in ("a", "b", "c") if p_ in sys.argv[1:]), None)
    if part is None:
        sys.exit(__doc__)
    d = setup()
    {"a": part_a, "b": part_b, "c": part_c}[part](d)


if __name__ == "__main__":
    main()
