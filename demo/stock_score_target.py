"""E4i: is the gain of E4c-E4h due to the Score-aligned loss, to forecasting r directly instead of the ridge residual,
or to both? A 2 x 2 design (TWStock).

Earlier networks (E2f, E2j) forecast the residual e = r - mu of the MSE ridge with the MSE and a zero-started head, and
did not beat the ridge; the E4 models forecast r directly with Score-aligned losses and do. Here every cell uses the
same network, inputs (52 base features + gap block of g_hat_pre, standardized on the training rows), protocol (epochs
chosen on Y - 1, patience 3, up to 40, refit on all years before Y, 5 seeds averaged) and evaluation (sign of the
final forecast):
- target r, MSE: the network forecasts r / sd(r);
- residual, MSE: forecast = mu / sd(r) + network (fixed offset), i.e. the network fits e / sd(r) (E2j-style);
- target r, Score loss: the network output f;
- residual, Score loss: f = a * mu / sd(mu) + network, a learnable scalar started at 1.
mu is the out-of-sample ridge forecast (stage-2 rows by half-year blocks out, demo/stock_identity_news.py). The two
residual cells are run with the usual initialization and with a zero-started output layer (as in E2j). Score losses:
soft Score and weighted logistic (demo/stock_score_loss.loss_fn). Networks: the MLP of demo/stock_score_loss.py (64,
GELU, dropout 0.2) and the 1-layer set model of demo/stock_score_set.py.
Part 'attn' (python demo/stock_score_target.py attn): the effect of the attention over the day's stocks. The set model
with its encoder layer replaced by (a) pool: each stock sees its own representation and the masked mean of the day's
representations (Linear(128, 64) + GELU + dropout 0.2 on [h, mean_day h], then Linear(64, 1)); (b) none: its own
representation only (Linear(64, 64) + GELU + dropout 0.2), trained like the set model on batches of days. Target r,
soft Score and weighted logistic, against the set model with attention of the 2 x 2 part.
Part 'pool2' (python demo/stock_score_target.py pool2): since the day mean does about as well as the attention, richer
pooled contexts: [h, day mean, industry mean] (the masked mean over the stock's TWSE / TPEx industry, profiles.csv,
the stock itself included) and [h, industry mean]; target r, both Score losses.
Usage: python demo/stock_score_target.py [2x2 | attn | pool2]   (default 2x2; forecasts to demo/runs/e4i_<part>_pred.npz)
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
from stock_news_attn2 import oos_mu  # noqa: E402
from stock_news_gap import gap_block  # noqa: E402
from stock_overnight import preopen_features  # noqa: E402
from stock_poly import ROOT, load  # noqa: E402
from stock_recurrent import base_features  # noqa: E402
from stock_score_loss import MLP, loss_fn  # noqa: E402
from stock_score_set import SetNet  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge  # noqa: E402

DEV = "cuda"
SEEDS = (0, 1, 2, 3, 4)


class NoAttnSet(nn.Module):
    """the set model without attention: 'pool' adds the day's mean representation, 'none' sees only its own."""

    IND = None   # (S, n_industries) one-hot, set in main for the pool2 modes

    def __init__(self, k, mode, d=64):
        super().__init__()
        self.mode = mode
        self.inp = nn.Sequential(nn.Linear(k, d), nn.GELU())
        mult = {"pool": 2, "none": 1, "pool_dayind": 3, "pool_ind": 2}[mode]
        self.mid = nn.Sequential(nn.Linear(mult * d, d), nn.GELU(), nn.Dropout(0.2))
        self.out = nn.Linear(d, 1)

    def forward(self, z, m):
        h = self.inp(z)
        w = m.float()[..., None]
        parts = [h]
        if self.mode in ("pool", "pool_dayind"):
            parts.append(((h * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1)).expand_as(h))
        if self.mode in ("pool_ind", "pool_dayind"):
            I = NoAttnSet.IND                                         # (S, K)
            num = torch.einsum("bsd,sk->bkd", h * w, I)
            den = torch.einsum("bs,sk->bk", w[..., 0], I)[..., None].clamp(min=1)
            parts.append(torch.einsum("bkd,sk->bsd", num / den, I))
        h = torch.cat(parts, -1)
        return self.out(self.mid(h)).squeeze(-1)


class Head(nn.Module):
    """network output + a * offset; kind 'mlp' takes rows (n, k), 'set' takes days (B, S, k) with a mask."""

    def __init__(self, arch, k, residual, learn_a, zero):
        super().__init__()
        self.arch = arch
        self.net = MLP(k) if arch == "mlp" else SetNet(k) if arch == "set" else NoAttnSet(k, arch)
        last = self.net.net[-1] if arch == "mlp" else self.net.out
        if zero:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        self.residual = residual
        self.a = nn.Parameter(torch.ones(())) if learn_a else None

    def forward(self, z, off, m=None):  # noqa: D102
        f = self.net(z) if self.arch == "mlp" else self.net(z, m)
        if self.residual:  # offset of the residual cells
            f = f + (self.a if self.a is not None else 1.0) * off
        return f


def train(arch, cell, Z, Y, OFF, M, idx, seed, epochs=None, vidx=None):
    """cell: (target, loss kind, zero head); Z (T, S, k), Y (T, S) raw r, OFF (T, S) offset; idx: rows (mlp: flat
    (t, s) pairs) or days (set)."""
    target, kind, zero = cell
    residual = target == "residual"
    torch.manual_seed(seed)
    g = torch.Generator(device=DEV).manual_seed(seed)
    net = Head(arch, Z.shape[-1], residual, residual and kind != "mse", zero).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_ep, bad = np.inf, 0, 0

    def batches(ix, bs, shuffle):
        order = ix[torch.randperm(len(ix), device=DEV, generator=g)] if shuffle else ix
        for i in range(0, len(order), bs):
            yield order[i:i + bs]

    def run(b):
        if arch == "mlp":
            t, s = b[:, 0], b[:, 1]
            return net(Z[t, s], OFF[t, s]), Y[t, s]
        m = M[b]
        return net(Z[b], OFF[b], m)[m], Y[b][m]

    lf = (lambda f, y: loss_fn("mse", f, y)) if kind == "mse" else (lambda f, y: loss_fn(kind, f, y))
    bs = 4096 if arch == "mlp" else 16
    for ep in range(1, (epochs or 40) + 1):
        net.train()
        for b in batches(idx, bs, True):
            f, y = run(b)
            l = lf(f, y)
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
        if epochs is None:
            net.eval()
            with torch.no_grad():
                fs, ys = zip(*(run(b) for b in batches(vidx, 8192 if arch == "mlp" else 64, False)))
                v = lf(torch.cat(fs), torch.cat(ys)).item()
            if v < best:
                best, best_ep, bad = v, ep, 0
            else:
                bad += 1
                if bad >= 3:
                    break
    net.eval()
    return net, best_ep


def main() -> None:
    dates, r, traded, gap, known = load()
    T, S = r.shape
    info = json.loads((ROOT / "info.json").read_text())
    stocks, top50 = info["Top100@Y"], set(info["Top50@Y"])
    X, y = base_features(dates, r, traded, gap, known, stocks)
    STR = struct_features(dates, stocks)
    has = (STR[..., 0] + STR[..., 1]) > 0
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    half = year * 2 + (np.repeat(dates.month.to_numpy()[:, None], S, axis=1) > 6)
    yrs = dates.year.to_numpy()
    g = gap.to_numpy()
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y) & np.isfinite(g)
    XG = np.concatenate([X, preopen_features(dates, gap, traded)], -1)
    ev = ok & np.isin(year, EVAL)
    is50 = np.array([s in top50 for s in stocks])[None, :]
    subsets = (("all 100 stocks", ev), ("the 50-stock subset", ev & is50),
               ("100 stocks, days with an own overnight item", ev & has),
               ("100 stocks, 2025-2026", ev & (year >= 2025)))
    fm = lambda t: f"{UNIT * t[0]:+5.1f}e-4 ± {UNIT * t[1]:3.1f}e-4"
    CELLS = {"target r, MSE": ("r", "mse", False), "residual, MSE": ("residual", "mse", False),
             "residual, MSE, zero head": ("residual", "mse", True),
             "target r, soft Score": ("r", "soft", False), "residual, soft Score": ("residual", "soft", False),
             "residual, soft Score, zero head": ("residual", "soft", True),
             "target r, w-logistic": ("r", "logistic", False), "residual, w-logistic": ("residual", "logistic", False),
             "residual, w-logistic, zero head": ("residual", "logistic", True)}
    part = "attn" if "attn" in sys.argv[1:] else "pool2" if "pool2" in sys.argv[1:] else "2x2"
    ARCHS = [("mlp", "MLP", list(CELLS)), ("set", "set", list(CELLS))]
    if part == "attn":
        ARCHS = [(a_, l_, ["target r, soft Score", "target r, w-logistic"]) for a_, l_ in
                 (("pool", "set without attention, day mean"), ("none", "set without attention, own only"))]
    if part == "pool2":
        from stock_identity_news import industry_onehot
        NoAttnSet.IND = torch.tensor(industry_onehot(stocks)[0], device=DEV)
        ARCHS = [(a_, l_, ["target r, soft Score", "target r, w-logistic"]) for a_, l_ in
                 (("pool_dayind", "pooled, day + industry mean"), ("pool_ind", "pooled, industry mean"))]
    names = ["ref: MSE ridge"] + [f"{l_}: {c}" for _, l_, cs in ARCHS for c in cs]
    res = {k: np.full((T, S), np.nan) for k in names}
    Yt = torch.tensor(np.where(ok, y, 0.0), dtype=torch.float32, device=DEV)
    Mt = torch.tensor(ok, device=DEV)
    for Y in EVAL:
        Zs = np.concatenate([X, gap_block(ghat_for(XG, g, ok, year, Y - 1), ok)], -1)
        Zr = np.concatenate([X, gap_block(ghat_for(XG, g, ok, year, Y), ok)], -1)
        inner, sel, rows, te = ok & (year < Y - 1), ok & (year == Y - 1), ok & (year < Y), ok & (year == Y)
        m = Ridge(Zs[inner], y[inner])
        a = min(ALPHAS, key=lambda a: np.mean((m.predict(Zs[sel], m.weights(a)) - y[sel]) ** 2))
        mu_s = oos_mu(Zs, y, ok, year, half, a, Y - 1)     # out of sample on years < Y - 1 (blocks) and on Y - 1
        mu_r = oos_mu(Zr, y, ok, year, half, a, Y)         # out of sample on years < Y (blocks) and on Y
        res[names[0]][te] = mu_r[te]
        nz = lambda Z_, m_: torch.tensor(np.nan_to_num((Z_ - Z_[m_].mean(0)) / (Z_[m_].std(0) + 1e-12)),
                                         dtype=torch.float32, device=DEV)
        As, Ar = nz(Zs, inner), nz(Zr, rows)
        sd_r_s, sd_r_r = y[inner].std(), y[rows].std()
        off = {}
        for kind in ("mse", "soft", "logistic"):
            if kind == "mse":
                off[kind] = (torch.tensor(np.nan_to_num(mu_s / sd_r_s), dtype=torch.float32, device=DEV),
                             torch.tensor(np.nan_to_num(mu_r / sd_r_r), dtype=torch.float32, device=DEV))
            else:
                off[kind] = (torch.tensor(np.nan_to_num(mu_s / np.nanstd(mu_s[inner])), dtype=torch.float32, device=DEV),
                             torch.tensor(np.nan_to_num(mu_r / np.nanstd(mu_r[rows])), dtype=torch.float32, device=DEV))
        rr = lambda cond: torch.tensor(np.argwhere(cond), device=DEV)
        dd = lambda cond: torch.tensor(np.nonzero(cond & ok.any(1))[0], device=DEV)
        IDX = {"mlp": (rr(inner), rr(sel), rr(rows))}
        IDX["set"] = IDX["pool"] = IDX["none"] = IDX["pool_dayind"] = IDX["pool_ind"] = (
            dd(yrs < Y - 1), dd(yrs == Y - 1), dd(yrs < Y))
        te_rows, te_days = rr(te), dd(yrs == Y)
        msg = []
        for arch, lab, cs in ARCHS:
            i_in, i_sel, i_rows = IDX[arch]
            for cname in cs:
                cell = CELLS[cname]
                o, eps = np.zeros((T, S)), []
                o_s, o_r = off[cell[1]]
                for seed in SEEDS:
                    _, ep = train(arch, cell, As, Yt, o_s, Mt, i_in, seed, vidx=i_sel)
                    ep = max(ep, 1)
                    net, _ = train(arch, cell, Ar, Yt, o_r, Mt, i_rows, seed, epochs=ep)
                    with torch.no_grad():
                        if arch == "mlp":
                            for i in range(0, len(te_rows), 16384):
                                b = te_rows[i:i + 16384]
                                t_, s_ = b[:, 0], b[:, 1]
                                o[t_.cpu().numpy(), s_.cpu().numpy()] += net(Ar[t_, s_], o_r[t_, s_]).cpu().numpy()
                        else:
                            for i in range(0, len(te_days), 64):
                                b = te_days[i:i + 64]
                                o[b.cpu().numpy()] += net(Ar[b], o_r[b], Mt[b]).cpu().numpy()
                    eps.append(ep)
                res[f"{lab}: {cname}"][te] = o[te]
                msg.append(f"{lab} {cname} epochs {eps}")
        print(f"{Y} done: ridge alpha {a:g}\n  " + "\n  ".join(msg), flush=True)
    print("\n######## 2 x 2: target (r / residual of the ridge) x loss (MSE / Score-aligned), vs the MSE ridge "
          "(R2 meaningless except ref and the MSE rows, whose scale is r / sd)")
    report(res, {k: names[0] for k in names[1:]}, y, subsets, fm)
    for k in names:
        print(f"share of up calls, {k}: {np.mean(res[k][ev] > 0):.3f}")
    np.savez(Path(__file__).resolve().parent / "runs" / f"e4i_{part}_pred.npz", y=y, ok=ok, ev=ev, year=year, is50=is50,
             **{k.replace(" ", "_").replace(":", "").replace(",", ""): v for k, v in res.items()})


if __name__ == "__main__":
    main()
