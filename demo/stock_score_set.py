"""E4h: a set model over the stocks of a day, trained on the Score-aligned losses of E4c (TWStock).

The gain of the Score-aligned models (E4c-E4g) is in the cross part: the ranking of the stocks within a day. The MLP
sees one stock-day at a time; here each stock also sees the other stocks of the same day through one self-attention
layer, so the forecast can depend on where a stock stands among the day's stocks (e.g. its gap or momentum relative to
the rest). Per day, the (S, k) inputs of the reference (52 base features + gap block of g_hat_pre, standardized on the
training rows) -> Linear(k, 64) + GELU -> one pre-norm Transformer encoder layer (4 heads, feed-forward 128, dropout
0.2; stocks not usable that day are masked) -> Linear(64, 1). Losses: soft Score and weighted logistic. AdamW lr 1e-3,
weight decay 1e-2, batches of 16 days, epochs chosen on Y - 1 (patience 3, up to 40), refit on all years before Y,
5 seeds. Ensembles: the two set models; and the two set models + the three members of the E4 r6 base ensemble
(demo/runs/e4_r6_pred.npz, each output divided by its sd on the fold's test rows).
Round r2 (python demo/stock_score_set.py r2): one layer with 10 seeds, and two encoder layers with 5 seeds; ensembles
with the r6 members; forecasts saved to demo/runs/e4h_r2_pred.npz.
Usage: python demo/stock_score_set.py [r2]
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
from stock_score_loss import loss_fn  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge  # noqa: E402

DEV = "cuda"
SEEDS = (0, 1, 2, 3, 4)
DAYS = 16


LAYERS = 1


class SetNet(nn.Module):
    def __init__(self, k, d=64):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(k, d), nn.GELU())
        layer = lambda: nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.2, batch_first=True, norm_first=True,
                                                   activation="gelu")
        self.enc = layer() if LAYERS == 1 else nn.TransformerEncoder(layer(), LAYERS, enable_nested_tensor=False)
        self.out = nn.Linear(d, 1)

    def forward(self, z, m):          # z (B, S, k), m (B, S) usable
        h = self.enc(self.inp(z), src_key_padding_mask=~m)
        return self.out(h).squeeze(-1)


def fit_set(kind, Z, Y, M, days, seed, epochs=None, vdays=None):
    torch.manual_seed(seed)
    g = torch.Generator(device=DEV).manual_seed(seed)
    net = SetNet(Z.shape[-1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_ep, bad = np.inf, 0, 0

    def vloss():
        net.eval()
        with torch.no_grad():
            f = torch.cat([net(Z[vdays[i:i + 64]], M[vdays[i:i + 64]])[M[vdays[i:i + 64]]]
                           for i in range(0, len(vdays), 64)])
            yv = torch.cat([Y[vdays[i:i + 64]][M[vdays[i:i + 64]]] for i in range(0, len(vdays), 64)])
        net.train()
        return loss_fn(kind, f, yv).item()

    for ep in range(1, (epochs or 40) + 1):
        net.train()
        perm = days[torch.randperm(len(days), device=DEV, generator=g)]
        for i in range(0, len(perm), DAYS):
            b = perm[i:i + DAYS]
            m = M[b]
            l = loss_fn(kind, net(Z[b], m)[m], Y[b][m])
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
        if epochs is None:
            v = vloss()
            if v < best:
                best, best_ep, bad = v, ep, 0
            else:
                bad += 1
                if bad >= 3:
                    break
    net.eval()
    return net, best_ep


def main() -> None:
    global LAYERS
    R2 = "r2" in sys.argv[1:]
    dates, r, traded, gap, known = load()
    T, S = r.shape
    info = json.loads((ROOT / "info.json").read_text())
    stocks, top50 = info["Top100@Y"], set(info["Top50@Y"])
    X, y = base_features(dates, r, traded, gap, known, stocks)
    STR = struct_features(dates, stocks)
    has = (STR[..., 0] + STR[..., 1]) > 0
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
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
    z6 = np.load(Path(__file__).resolve().parent / "runs" / "e4_r6_pred.npz")
    base_mem = [z6[k] for k in ("base_soft_Score", "base_soft-Score_MLP", "base_w-logistic_MLP")]
    names = ["ref: MSE ridge", "E4 r6 base ensemble", "set model, soft Score", "set model, weighted logistic",
             "ensemble of the two set models", "ensemble: r6 base members + set models"]
    CONF = [("", 1, SEEDS)]
    if R2:
        CONF = [("1 layer, 10 seeds", 1, tuple(range(10))), ("2 layers, 5 seeds", 2, SEEDS)]
        names = names[:2] + [f"{c}: {m_}" for c, _, _ in CONF for m_ in ("soft Score", "weighted logistic")] + [
            "ensemble: r6 base members + 1-layer set models", "ensemble: r6 base members + 2-layer set models",
            "ensemble: r6 base members + all set models"]
    res = {k: np.full((T, S), np.nan) for k in names}
    res[names[1]] = z6["base_ensemble"]
    Yt = torch.tensor(np.where(ok, y, 0.0), dtype=torch.float32, device=DEV)
    Mt = torch.tensor(ok, device=DEV)
    for Y in EVAL:
        Zs = np.concatenate([X, gap_block(ghat_for(XG, g, ok, year, Y - 1), ok)], -1)
        Zr = np.concatenate([X, gap_block(ghat_for(XG, g, ok, year, Y), ok)], -1)
        inner, sel, rows, te = ok & (year < Y - 1), ok & (year == Y - 1), ok & (year < Y), ok & (year == Y)
        m = Ridge(Zs[inner], y[inner])
        a = min(ALPHAS, key=lambda a: np.mean((m.predict(Zs[sel], m.weights(a)) - y[sel]) ** 2))
        m = Ridge(Zr[rows], y[rows])
        res[names[0]][te] = m.predict(Zr[te], m.weights(a))
        nz = lambda Z_, m_: torch.tensor(np.nan_to_num((Z_ - Z_[m_].mean(0)) / (Z_[m_].std(0) + 1e-12)),
                                         dtype=torch.float32, device=DEV)
        As, Ar = nz(Zs, inner), nz(Zr, rows)
        dd = lambda cond: torch.tensor(np.nonzero(cond & ok.any(1))[0], device=DEV)
        d_in, d_sel, d_rows, d_te = dd(yrs < Y - 1), dd(yrs == Y - 1), dd(yrs < Y), dd(yrs == Y)
        msg, outs, bycf = [], [], {}
        for cf, nl, seeds_ in CONF:
          LAYERS = nl
          for kind, nm in ((("soft", f"{cf}: soft Score"), ("logistic", f"{cf}: weighted logistic")) if R2 else
                           (("soft", names[2]), ("logistic", names[3]))):
            o, eps = np.zeros((T, S)), []
            for seed in seeds_:
                _, ep = fit_set(kind, As, Yt, Mt, d_in, seed, vdays=d_sel)
                ep = max(ep, 1)
                net, _ = fit_set(kind, Ar, Yt, Mt, d_rows, seed, epochs=ep)
                with torch.no_grad():
                    for i in range(0, len(d_te), 64):
                        b = d_te[i:i + 64]
                        o[b.cpu().numpy()] += net(Ar[b], Mt[b]).cpu().numpy()
                eps.append(ep)
            res[nm][te] = o[te]
            outs.append(o[te])
            bycf.setdefault(nl, []).append(o[te])
            msg.append(f"{cf} {kind} set epochs {eps}")
        LAYERS = 1
        bm_ = [bm[te] for bm in base_mem]
        if R2:
            res[names[6]][te] = sum(v / v.std() for v in bycf[1] + bm_)
            res[names[7]][te] = sum(v / v.std() for v in bycf[2] + bm_)
            res[names[8]][te] = sum(v / v.std() for v in outs + bm_)
        else:
            res[names[4]][te] = sum(v / v.std() for v in outs)
            res[names[5]][te] = sum(v / v.std() for v in outs + bm_)
        print(f"{Y} done: ridge alpha {a:g}, " + ", ".join(msg), flush=True)
    print("\n######## set models over the day's stocks, Score-aligned losses, vs the MSE ridge (R2 meaningless except ref)")
    report(res, {k: names[0] for k in names[1:]}, y, subsets, fm)
    print("\n######## the same vs the E4 r6 base ensemble")
    report({k: res[k] for k in names[1:]}, {k: names[1] for k in names[2:]}, y, subsets[:2], fm)
    for k in names:
        print(f"share of up calls, {k}: {np.mean(res[k][ev] > 0):.3f}")
    if R2:
        np.savez(Path(__file__).resolve().parent / "runs" / "e4h_r2_pred.npz", y=y, ok=ok, ev=ev, year=year, is50=is50,
                 **{k.replace(" ", "_").replace(":", "").replace(",", ""): v for k, v in res.items()})


if __name__ == "__main__":
    main()
