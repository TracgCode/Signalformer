"""E4c: losses aligned with the sign score instead of the MSE of r, linear and MLP (TWStock).

The Score is mean(sign(f) * r): a stock-day counts with the weight |r| and pays when sign(f) = sign(r). Maximizing it is
a cost-sensitive classification with labels sign(r) and weights |r|. The MSE ridge targets E[r], whose sign is the
Bayes decision, but it spends its capacity on the size of r. Here the same inputs (52 base features + gap block of
g_hat_pre, standardized on the training rows) with:
- ref: the MSE ridge of E4a / E4b (alpha on Y - 1, fitted on all years before Y);
- weighted logistic: |r| * log(1 + exp(-sign(r) f)), L2 on the weights;
- weighted hinge: |r| * max(0, 1 - sign(r) f), L2;
- soft Score: -mean(tanh(f) * r), L2 (the Score with sign replaced by tanh);
- MSE MLP and weighted-logistic MLP: one hidden layer of 64, GELU, dropout 0.2, AdamW, batches of 4096 rows, epochs
  chosen on Y - 1 (patience 3, up to 40), refit on all years before Y, 3 seeds averaged;
- mean of the ridge and the weighted logistic, each scaled by its sd on the training rows (an ensemble).
The L2 weight of the linear models is chosen on Y - 1 by the validation loss (fit on the years before Y - 1), then the
model is refitted on all years before Y. Linear models: full-batch L-BFGS.
Round r2 (python demo/stock_score_loss.py r2): the weighted logistic with L2 up to 10; the soft Score and the weighted
logistic as MLPs with 5 seeds; the soft Score on the inputs plus their day-demeaned copies (value - the day's mean over
the stocks); the ensemble of soft Score (linear), soft-Score MLP and weighted-logistic MLP (mean of the outputs, each
divided by its sd on the test rows of the fold).
Round r3 (python demo/stock_score_loss.py r3): the ensemble of r2 scaled up, each member set as its own ensemble
(soft Score linear + soft-Score MLP + weighted-logistic MLP):
- base: the inputs of the reference, MLPs with 10 seeds;
- rich: the inputs plus the 4 raw pre-open features and the 43 own struct news columns of E2j, MLPs with 5 seeds;
- deep: the inputs of the reference, MLPs with two hidden layers of 128, 5 seeds;
- all: the mean of the three ensembles.
Round r4 (python demo/stock_score_loss.py r4): wider price inputs for the Score-aligned ensemble (r3 base, 5 seeds) and
for the MSE ridge (control): the inputs of the reference plus
- poly230: the other 195 polynomial-fit features of stock_poly.features (all 230 instead of the 35 of degree 0);
- poly230 + extra: also the volume, flows, technical and global groups of stock_extra (73).
Round r5 (python demo/stock_score_loss.py r5): recency weights for the r3 base ensemble (5 seeds): every training row
weighted by 2^(-age / half-life), age in trading days before the last fit row, half-life 250 / 500 / 1000 days (about
1 / 2 / 4 years) against none.
Round r6 (python demo/stock_score_loss.py r6): the r3 base ensemble alone (10 seeds), with the forecasts of the ridge
and of every member saved to demo/runs/e4_r6_pred.npz for the per-year breakdown and the decision rules.
Usage: python demo/stock_score_loss.py [r2 | r3 | r4 | r5 | r6]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_identity_news import struct_features  # noqa: E402
from stock_itrans_news import U as UNIT, ghat_for  # noqa: E402
from stock_news_attn import report  # noqa: E402
from stock_news_gap import gap_block  # noqa: E402
from stock_overnight import preopen_features  # noqa: E402
from stock_poly import ROOT, load  # noqa: E402
from stock_recurrent import base_features  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge  # noqa: E402

DEV = "cuda"
L2S = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
SEEDS = (0, 1, 2)
SEEDS5 = (0, 1, 2, 3, 4)
L2S_WIDE = (0.1, 0.3, 1.0, 3.0, 10.0)
R2 = False


def demeaned(Z, ok):
    """Z with its day-demeaned copy appended (value - the day's mean over the usable stocks)."""
    m = np.nanmean(np.where(ok[..., None], Z, np.nan), 1, keepdims=True)
    return np.concatenate([Z, Z - np.nan_to_num(m)], -1)


def loss_fn(kind, f, r, rw=None):
    """rw: optional row weights (recency, round r5), mean 1."""
    s = torch.sign(r)
    w = r.abs() / r.abs().mean()
    rw = 1.0 if rw is None else rw
    if kind == "logistic":
        return (rw * w * nn.functional.softplus(-s * f)).mean()
    if kind == "hinge":
        return (rw * w * torch.relu(1 - s * f)).mean()
    if kind == "soft":
        return -(rw * torch.tanh(f) * r).mean() / r.abs().mean()
    if kind == "mse":
        return ((f - r / r.std()) ** 2).mean()
    raise ValueError(kind)


def fit_linear(kind, Z, r, l2, rw=None):
    Zt, rt = torch.tensor(Z, device=DEV), torch.tensor(r, device=DEV)
    rw = None if rw is None else torch.tensor(rw, dtype=torch.float32, device=DEV)
    w = torch.zeros(Z.shape[1], device=DEV, requires_grad=True)
    b = torch.zeros(1, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=500, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        l = loss_fn(kind, Zt @ w + b, rt, rw) + l2 * (w ** 2).sum()
        l.backward()
        return l
    opt.step(closure)
    return w.detach(), b.detach()


def lin_pred(p, Z):
    return (torch.tensor(Z, device=DEV) @ p[0] + p[1]).cpu().numpy()


DEEP = False


class MLP(nn.Module):
    def __init__(self, k):
        super().__init__()
        if DEEP:
            self.net = nn.Sequential(nn.Linear(k, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 128), nn.GELU(),
                                     nn.Dropout(0.2), nn.Linear(128, 1))
        else:
            self.net = nn.Sequential(nn.Linear(k, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


def fit_mlp(kind, Z, r, seed, epochs=None, Zv=None, rv=None, rw=None):
    torch.manual_seed(seed)
    rw = None if rw is None else torch.tensor(rw, dtype=torch.float32, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(seed)
    Zt, rt = torch.tensor(Z, device=DEV), torch.tensor(r, device=DEV)
    net = MLP(Z.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_ep, bad = np.inf, 0, 0
    for ep in range(1, (epochs or 40) + 1):
        net.train()
        perm = torch.randperm(len(Zt), device=DEV, generator=g)
        for i in range(0, len(perm), 4096):
            b = perm[i:i + 4096]
            l = loss_fn(kind, net(Zt[b]), rt[b], None if rw is None else rw[b])
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
        if epochs is None:
            net.eval()
            with torch.no_grad():
                v = loss_fn(kind, net(torch.tensor(Zv, device=DEV)), torch.tensor(rv, device=DEV)).item()
            if v < best:
                best, best_ep, bad = v, ep, 0
            else:
                bad += 1
                if bad >= 3:
                    break
    net.eval()
    return net, best_ep


def member_set(Zs, Zr, inner, sel, rows, te, y, seeds, hl=None, tday=None):
    """soft Score linear, soft-Score MLP and weighted-logistic MLP on (Zs: inputs of the selection fit, Zr: of the
    refit); returns the three test forecasts and a log message."""
    mu_s, sd_s = Zs[inner].mean(0), Zs[inner].std(0) + 1e-12
    mu_r, sd_r = Zr[rows].mean(0), Zr[rows].std(0) + 1e-12
    As = lambda m_: ((Zs[m_] - mu_s) / sd_s).astype(np.float32)
    Ar = lambda m_: ((Zr[m_] - mu_r) / sd_r).astype(np.float32)
    yi, ys_, yr_ = y[inner].astype(np.float32), y[sel].astype(np.float32), y[rows].astype(np.float32)
    if hl is None:
        wi = wr = None
    else:   # recency weights exp(-age / half-life * ln 2), age in trading days from the end of the fit rows, mean 1
        rec = lambda m_: (lambda a_: a_ / a_.mean())(np.exp(-(tday[m_].max() - tday[m_]) / hl * np.log(2)))
        wi, wr = rec(inner), rec(rows)
    vl = {}
    for l2 in L2S:
        p = fit_linear("soft", As(inner), yi, l2, wi)
        vl[l2] = loss_fn("soft", torch.tensor(lin_pred(p, As(sel)), device=DEV), torch.tensor(ys_, device=DEV)).item()
    l2 = min(vl, key=vl.get)
    out = [lin_pred(fit_linear("soft", Ar(rows), yr_, l2, wr), Ar(te))]
    msg = [f"soft l2 {l2:g}"]
    for kind in ("soft", "logistic"):
        o, eps = 0, []
        for seed in seeds:
            _, ep = fit_mlp(kind, As(inner), yi, seed, Zv=As(sel), rv=ys_, rw=wi)
            ep = max(ep, 1)
            net, _ = fit_mlp(kind, Ar(rows), yr_, seed, epochs=ep, rw=wr)
            with torch.no_grad():
                o = o + net(torch.tensor(Ar(te), device=DEV)).cpu().numpy()
            eps.append(ep)
        out.append(o)
        msg.append(f"{kind} MLP epochs {eps}")
    return out, ", ".join(msg)


def main_r3() -> None:
    global DEEP
    dates, r, traded, gap, known = load()
    T, S = r.shape
    info = json.loads((ROOT / "info.json").read_text())
    stocks, top50 = info["Top100@Y"], set(info["Top50@Y"])
    X, y = base_features(dates, r, traded, gap, known, stocks)
    STR = struct_features(dates, stocks)
    own = STR[..., :43]
    has = (STR[..., 0] + STR[..., 1]) > 0
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    g = gap.to_numpy()
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y) & np.isfinite(g)
    PO = preopen_features(dates, gap, traded)
    XG = np.concatenate([X, PO], -1)
    ev = ok & np.isin(year, EVAL)
    is50 = np.array([s in top50 for s in stocks])[None, :]
    subsets = (("all 100 stocks", ev), ("the 50-stock subset", ev & is50),
               ("100 stocks, days with an own overnight item", ev & has),
               ("100 stocks, 2025-2026", ev & (year >= 2025)))
    fm = lambda t: f"{UNIT * t[0]:+5.1f}e-4 ± {UNIT * t[1]:3.1f}e-4"
    R4 = "r4" in sys.argv[1:]
    R5 = "r5" in sys.argv[1:]
    HL = {"base": None, "half-life 250": 250, "half-life 500": 500, "half-life 1000": 1000}
    tday = np.repeat(np.arange(T)[:, None], S, 1)
    R6 = "r6" in sys.argv[1:]
    sets = tuple(HL) if R5 else ("base",) if R6 else ("base", "rich", "deep")
    if R4:
        from stock_extra import extra_features, signals
        from stock_poly import features
        Xb, _, bn = features(dates, r, traded, gap, known)
        Xe, _, eg = extra_features(signals(dates, known, S), np.asarray(dates < "2022-01-01"))
        P195 = np.nan_to_num(Xb[..., [not n.endswith("p0") for n in bn]])
        EXT = np.nan_to_num(Xe[..., np.isin(eg, ["volume", "flows", "technical", "global"])])
        WIDE = {"base": None, "poly230": P195, "poly230 + extra": np.concatenate([P195, EXT], -1)}
        print(f"poly230 adds {P195.shape[-1]}, extra adds {EXT.shape[-1]}", flush=True)
        sets = tuple(WIDE)
    mem = ("soft Score", "soft-Score MLP", "w-logistic MLP")
    names = ["ref: MSE ridge"] + [f"{s_}: {m_}" for s_ in sets for m_ in mem] + [f"{s_}: ensemble" for s_ in sets] + [
        "all: ensemble of the three ensembles"]
    if R4:
        names += [f"{s_}: MSE ridge" for s_ in sets[1:]]
    res = {k: np.full((T, S), np.nan) for k in names}
    for Y in EVAL:
        Zs = np.concatenate([X, gap_block(ghat_for(XG, g, ok, year, Y - 1), ok)], -1)
        Zr = np.concatenate([X, gap_block(ghat_for(XG, g, ok, year, Y), ok)], -1)
        inner, sel, rows, te = ok & (year < Y - 1), ok & (year == Y - 1), ok & (year < Y), ok & (year == Y)
        m = Ridge(Zs[inner], y[inner])
        a = min(ALPHAS, key=lambda a: np.mean((m.predict(Zs[sel], m.weights(a)) - y[sel]) ** 2))
        m = Ridge(Zr[rows], y[rows])
        res[names[0]][te] = m.predict(Zr[te], m.weights(a))
        ens_all = 0
        for s_ in sets:
            DEEP = s_ == "deep"
            if s_ == "rich":
                A, B = np.concatenate([Zs, PO, own], -1), np.concatenate([Zr, PO, own], -1)
            elif R4 and WIDE[s_] is not None:
                A, B = np.concatenate([Zs, WIDE[s_]], -1), np.concatenate([Zr, WIDE[s_]], -1)
                mm = Ridge(A[inner], y[inner])
                aa = min(ALPHAS, key=lambda a: np.mean((mm.predict(A[sel], mm.weights(a)) - y[sel]) ** 2))
                mm = Ridge(B[rows], y[rows])
                res[f"{s_}: MSE ridge"][te] = mm.predict(B[te], mm.weights(aa))
            else:
                A, B = Zs, Zr
            nseed = SEEDS5 if (R4 or R5) else (tuple(range(10)) if s_ == "base" else SEEDS5)
            out, msg = member_set(A, B, inner, sel, rows, te, y, nseed, hl=HL.get(s_) if R5 else None, tday=tday)
            ens = 0
            for m_, o in zip(mem, out):
                res[f"{s_}: {m_}"][te] = o
                ens = ens + o / o.std()
            res[f"{s_}: ensemble"][te] = ens
            ens_all = ens_all + ens / ens.std()
            print(f"{Y} {s_}: {msg}", flush=True)
        DEEP = False
        res["all: ensemble of the three ensembles"][te] = ens_all
        print(f"{Y} done: ridge alpha {a:g}", flush=True)
    print(f"\n######## {'r5: recency weights' if R5 else 'r4: wider inputs' if R4 else 'r3: scaled-up'} ensembles of Score-aligned models vs the MSE ridge (R2 meaningless except ref)")
    report(res, {k: names[0] for k in names[1:]}, y, subsets, fm)
    for k in names:
        print(f"share of up calls, {k}: {np.mean(res[k][ev] > 0):.3f}")
    if R6:
        np.savez(Path(__file__).resolve().parent / "runs" / "e4_r6_pred.npz", y=y, ok=ok, ev=ev, year=year, is50=is50,
                 **{k.replace(" ", "_").replace(":", ""): v for k, v in res.items()})


def main() -> None:
    if any(a in sys.argv[1:] for a in ("r3", "r4", "r5", "r6")):
        return main_r3()
    dates, r, traded, gap, known = load()
    T, S = r.shape
    info = json.loads((ROOT / "info.json").read_text())
    stocks, top50 = info["Top100@Y"], set(info["Top50@Y"])
    X, y = base_features(dates, r, traded, gap, known, stocks)
    STR = struct_features(dates, stocks)
    has = (STR[..., 0] + STR[..., 1]) > 0
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    g = gap.to_numpy()
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y) & np.isfinite(g)
    XG = np.concatenate([X, preopen_features(dates, gap, traded)], -1)

    ev = ok & np.isin(year, EVAL)
    is50 = np.array([s in top50 for s in stocks])[None, :]
    subsets = (("all 100 stocks", ev), ("the 50-stock subset", ev & is50),
               ("100 stocks, days with an own overnight item", ev & has),
               ("100 stocks, 2025-2026", ev & (year >= 2025)))
    fm = lambda t: f"{UNIT * t[0]:+5.1f}e-4 ± {UNIT * t[1]:3.1f}e-4"
    global R2
    R2 = "r2" in sys.argv[1:]
    names = ["ref: MSE ridge", "weighted logistic", "weighted hinge", "soft Score", "MSE MLP", "weighted-logistic MLP",
             "mean of ridge and weighted logistic"]
    LINEAR = [("logistic", names[1]), ("hinge", names[2]), ("soft", names[3])]
    MLPS = [("mse", names[4]), ("logistic", names[5])]
    DM = None
    if R2:
        names = ["ref: MSE ridge", "soft Score", "weighted logistic, L2 up to 10", "soft-Score MLP (5 seeds)",
                 "weighted-logistic MLP (5 seeds)", "soft Score, + day-demeaned inputs",
                 "ensemble: soft Score + soft MLP + w-logistic MLP"]
        LINEAR = [("soft", names[1]), ("logistic", names[2])]
        MLPS = [("soft", names[3]), ("logistic", names[4])]
        DM = names[5]
        ENS = (names[1], names[3], names[4])
    res = {k: np.full((T, S), np.nan) for k in names}
    for Y in EVAL:
        Zs = np.concatenate([X, gap_block(ghat_for(XG, g, ok, year, Y - 1), ok)], -1)
        Zr = np.concatenate([X, gap_block(ghat_for(XG, g, ok, year, Y), ok)], -1)
        inner, sel, rows, te = ok & (year < Y - 1), ok & (year == Y - 1), ok & (year < Y), ok & (year == Y)
        # ridge reference (alpha on Y - 1, fit on all before Y; forecasts of Y)
        m = Ridge(Zs[inner], y[inner])
        a = min(ALPHAS, key=lambda a: np.mean((m.predict(Zs[sel], m.weights(a)) - y[sel]) ** 2))
        m = Ridge(Zr[rows], y[rows])
        res[names[0]][te] = m.predict(Zr[te], m.weights(a))
        sd_ridge = m.predict(Zr[rows], m.weights(a)).std()
        # standardized inputs for the torch models
        mu_s, sd_s = Zs[inner].mean(0), Zs[inner].std(0) + 1e-12
        mu_r, sd_r = Zr[rows].mean(0), Zr[rows].std(0) + 1e-12
        As = lambda m_: ((Zs[m_] - mu_s) / sd_s).astype(np.float32)
        Ar = lambda m_: ((Zr[m_] - mu_r) / sd_r).astype(np.float32)
        yi, ys_, yr_ = y[inner].astype(np.float32), y[sel].astype(np.float32), y[rows].astype(np.float32)
        msg = []
        for kind, nm in LINEAR:
            vl = {}
            for l2 in (L2S_WIDE if kind == "logistic" and R2 else L2S):
                p = fit_linear(kind, As(inner), yi, l2)
                with torch.no_grad():
                    f = torch.tensor(lin_pred(p, As(sel)), device=DEV)
                    vl[l2] = loss_fn(kind, f, torch.tensor(ys_, device=DEV)).item()
            l2 = min(vl, key=vl.get)
            p = fit_linear(kind, Ar(rows), yr_, l2)
            res[nm][te] = lin_pred(p, Ar(te))
            if kind == "logistic" and not R2:
                sd_log = lin_pred(p, Ar(rows)).std()
            msg.append(f"{kind} l2 {l2:g}")
        for kind, nm in MLPS:
            out, eps = 0, []
            for seed in (SEEDS5 if R2 else SEEDS):
                _, ep = fit_mlp(kind, As(inner), yi, seed, Zv=As(sel), rv=ys_)
                ep = max(ep, 1)
                net, _ = fit_mlp(kind, Ar(rows), yr_, seed, epochs=ep)
                with torch.no_grad():
                    out = out + net(torch.tensor(Ar(te), device=DEV)).cpu().numpy()
                eps.append(ep)
            res[nm][te] = out
            msg.append(f"{kind} MLP epochs {eps}")
        if R2:
            if DM:
                Zs_, Zr_ = Zs, Zr
                Zs, Zr = demeaned(Zs, ok), demeaned(Zr, ok)
                mu_s, sd_s = Zs[inner].mean(0), Zs[inner].std(0) + 1e-12
                mu_r, sd_r = Zr[rows].mean(0), Zr[rows].std(0) + 1e-12
                vl = {}
                for l2 in L2S:
                    p = fit_linear("soft", As(inner), yi, l2)
                    with torch.no_grad():
                        vl[l2] = loss_fn("soft", torch.tensor(lin_pred(p, As(sel)), device=DEV),
                                         torch.tensor(ys_, device=DEV)).item()
                l2 = min(vl, key=vl.get)
                res[DM][te] = lin_pred(fit_linear("soft", Ar(rows), yr_, l2), Ar(te))
                msg.append(f"soft demeaned l2 {l2:g}")
                Zs, Zr = Zs_, Zr_
            ens = sum(res[k][te] / res[k][te].std() for k in ENS)
            res[names[-1]][te] = ens
        else:
            res[names[6]][te] = res[names[0]][te] / sd_ridge + res[names[1]][te] / sd_log
        print(f"{Y} done: ridge alpha {a:g}, " + ", ".join(msg), flush=True)
    print("\n######## Score-aligned losses vs the MSE ridge (R2 is meaningful only for the MSE models)")
    report(res, {k: names[0] for k in names[1:]}, y, subsets, fm)
    for k in names:
        print(f"share of up calls, {k}: {np.mean(res[k][ev] > 0):.3f}")


if __name__ == "__main__":
    main()
