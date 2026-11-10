"""A faster trainer for the d 32 soft-Score set model: all seeds trained at once as one batched model (TWStock).

The profile (demo/stock_score_profile.py) showed that one network uses about 1% of the GPU: a training step is kernel
launches and host-device synchronizations. Here the K seeds of a run are K copies of the same network whose weights are
stacked along a leading dimension, so one step trains all of them:
- the same layers as stock_score_itoken.TokNet with token "day" and block "e4h": Linear(F, d) + GELU, one pre-norm
  Transformer encoder layer (4 heads, feed-forward 2d, GELU, dropout on the attention weights, after the attention and
  inside and after the feed-forward, LayerNorm eps 1e-5), Linear(d, 1); written with batched einsums;
- the initial weights of seed k are those of TokNet built after torch.manual_seed(k), as in stock_score_itoken.train;
  the order of the training days in every epoch comes from the same per-seed generator, so each seed sees the same
  batches as before; only the dropout masks differ (they are drawn for all seeds at once);
- AdamW on the stacked weights is the same element-wise update as one AdamW per seed; the loss is the sum over the
  seeds of each seed's soft-Score loss -sum(tanh(f) r) / sum|r| over its usable rows, as masked sums (no boolean
  indexing, so no synchronization per step);
- early stopping per seed (the validation loss of every seed after every epoch, patience 3, at most 40 epochs); the
  refit trains for the largest chosen epoch count and keeps each seed's weights at its own count;
- the forecast is the sum over the seeds of the raw outputs, as in final2.
The attention mask may differ from the loss mask (the pre-open mask of stock_score_itoken premask).
Part 'check' (python demo/stock_score_fast.py check): final2 (d 32, soft Score, 40 seeds) with this trainer, compared
with the saved final2 forecasts (paired; the seeds see the same data order and start from the same weights, so the two
differ only through dropout), and the run time. Forecasts to demo/runs/e4j_fast_check_pred.npz.
Usage: python demo/stock_score_fast.py check [premask]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stock_score_itoken as it  # noqa: E402

DEV = it.DEV
KEYS = ("cur.weight", "cur.bias", "enc.self_attn.in_proj_weight", "enc.self_attn.in_proj_bias",
        "enc.self_attn.out_proj.weight", "enc.self_attn.out_proj.bias", "enc.linear1.weight", "enc.linear1.bias",
        "enc.linear2.weight", "enc.linear2.bias", "enc.norm1.weight", "enc.norm1.bias", "enc.norm2.weight",
        "enc.norm2.bias", "out.weight", "out.bias")


def init_bank(seeds, n_in, d, dropout=0.2):
    """stacked initial weights (K, ...) of TokNet built after torch.manual_seed(seed) for each seed."""
    sds = []
    for s in seeds:
        torch.manual_seed(s)
        sds.append(it.TokNet(0, n_in, "e4h", d, False, dropout=dropout).state_dict())
    return {k: nn.Parameter(torch.stack([sd[k] for sd in sds]).to(DEV)) for k in KEYS}


def forward(P, z, att, train, p=0.2, heads=4):
    """z (K or 1, B, S, F), att (K or 1, B, S) bool -> f (K, B, S)."""
    K = P["out.bias"].shape[0]
    drop = (lambda x: Fn.dropout(x, p, True)) if train else (lambda x: x)
    h = torch.einsum("kbsf,kdf->kbsd", z.expand(K, *z.shape[1:]) if z.shape[0] == 1 else z, P["cur.weight"])
    x = Fn.gelu(h + P["cur.bias"][:, None, None])
    d = x.shape[-1]
    ln = lambda v, w, b: Fn.layer_norm(v, (d,)) * w[:, None, None] + b[:, None, None]
    a = ln(x, P["enc.norm1.weight"], P["enc.norm1.bias"])
    qkv = torch.einsum("kbsd,ked->kbse", a, P["enc.self_attn.in_proj_weight"]) + P["enc.self_attn.in_proj_bias"][:, None, None]
    Kb, B, S, _ = qkv.shape
    q, k_, v = (t.reshape(Kb, B, S, heads, d // heads).transpose(2, 3) for t in qkv.split(d, -1))  # (K, B, H, S, dh)
    sc = q @ k_.transpose(-1, -2) / (d // heads) ** 0.5
    am = (att.expand(K, *att.shape[1:]) if att.shape[0] == 1 else att)[:, :, None, None, :]
    sc = sc.masked_fill(~am, float("-inf"))
    w = drop(torch.softmax(sc, -1))
    o = (w @ v).transpose(2, 3).reshape(Kb, B, S, d)
    o = torch.einsum("kbsd,ked->kbse", o, P["enc.self_attn.out_proj.weight"]) + P["enc.self_attn.out_proj.bias"][:, None, None]
    x = x + drop(o)
    a = ln(x, P["enc.norm2.weight"], P["enc.norm2.bias"])
    f1 = Fn.gelu(torch.einsum("kbsd,ked->kbse", a, P["enc.linear1.weight"]) + P["enc.linear1.bias"][:, None, None])
    f2 = torch.einsum("kbse,kde->kbsd", drop(f1), P["enc.linear2.weight"]) + P["enc.linear2.bias"][:, None, None]
    x = x + drop(f2)
    return torch.einsum("kbsd,kod->kbso", x, P["out.weight"])[..., 0] + P["out.bias"][:, None, None, 0]


def soft_loss(f, y, m):
    """per seed: -sum(tanh(f) y) / sum|y| over the usable rows m; f (K, B, S), y and m (K or 1, B, S)."""
    mf = m.float()
    return -(torch.tanh(f) * y * mf).sum((1, 2)) / (y.abs() * mf).sum((1, 2)).clamp_min(1e-12)


def train_bank(seeds, Z, Y, M, ATT, days, d=32, epochs=None, vdays=None, lr=1e-3, wd=1e-2, bsz=16, dropout=0.2,
               max_epochs=40, patience=3):
    """Z (T, S, F) standardized inputs on the GPU. Selection (epochs None, vdays given): returns the best epoch of
    every seed. Refit (epochs = list per seed): returns the weights of every seed after its own epoch count."""
    K = len(seeds)
    P = init_bank(seeds, Z.shape[-1], d, dropout)
    opt = torch.optim.AdamW(list(P.values()), lr=lr, weight_decay=wd)
    gens = [torch.Generator(device=DEV).manual_seed(s) for s in seeds]
    n_ep = max_epochs if epochs is None else int(max(epochs))
    best = torch.full((K,), float("inf"), device=DEV)
    best_ep = torch.zeros(K, dtype=torch.long, device=DEV)
    bad = torch.zeros(K, dtype=torch.long, device=DEV)
    done = torch.zeros(K, dtype=torch.bool, device=DEV)
    snap = {k: v.detach().clone() for k, v in P.items()}
    ep_t = None if epochs is None else torch.tensor(epochs, device=DEV)
    for ep in range(1, n_ep + 1):
        perms = torch.stack([days[torch.randperm(len(days), device=DEV, generator=g)] for g in gens])  # (K, n)
        for i in range(0, perms.shape[1], bsz):
            b = perms[:, i:i + bsz]
            f = forward(P, Z[b], ATT[b], True, dropout)
            loss = soft_loss(f, Y[b], M[b]).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epochs is not None:
            hit = ep_t == ep
            with torch.no_grad():
                for k, v in P.items():
                    snap[k][hit] = v[hit]
            continue
        with torch.no_grad():
            num = torch.zeros(K, device=DEV)
            den = torch.zeros((), device=DEV)
            for i in range(0, len(vdays), 64):
                b = vdays[i:i + 64][None]
                f = forward(P, Z[b], ATT[b], False)
                mf = M[b].float()
                num += (torch.tanh(f) * Y[b] * mf).sum((1, 2))
                den += (Y[b].abs() * mf).sum()
            v = -num / den
            imp = (v < best) & ~done
            best = torch.where(imp, v, best)
            best_ep = torch.where(imp, torch.full_like(best_ep, ep), best_ep)
            bad = torch.where(imp, torch.zeros_like(bad), bad + (~done).long())
            done = done | (bad >= patience)
        if bool(done.all()):
            break
    if epochs is None:
        return best_ep.tolist()
    return snap


def predict(snap, Z, ATT, tdays):
    with torch.no_grad():
        out = []
        for i in range(0, len(tdays), 64):
            b = tdays[i:i + 64][None]
            out.append(forward(snap, Z[b], ATT[b], False).sum(0))
        return torch.cat(out).cpu().numpy()   # (n_days, S), summed over the seeds


def fit_predict(seeds, inp_s, inp_r, Y, M, ATT, d_in, d_sel, d_rows, d_te, d=32):
    """selection on inp_s (early stopping on d_sel), refit on inp_r, forecast of d_te; returns (sum f, epochs)."""
    eps = [max(e, 1) for e in train_bank(seeds, inp_s.cur, Y, M, ATT, d_in, d=d, vdays=d_sel)]
    snap = train_bank(seeds, inp_r.cur, Y, M, ATT, d_rows, d=d, epochs=eps)
    return predict(snap, inp_r.cur, ATT, d_te), eps


def main() -> None:
    from stock_itrans_news import U as UNIT, ghat_for
    from stock_news_attn import report
    from stock_news_attn2 import oos_mu
    from stock_news_gap import gap_block
    from stock_overnight import preopen_features
    from stock_poly import ROOT, load
    from stock_recurrent import base_features
    from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge

    premask = "premask" in sys.argv[1:]
    seeds = tuple(range(40))
    t_all = time.time()
    dates, r, traded, gap, known = load()
    T, S = r.shape
    info = json.loads((ROOT / "info.json").read_text())
    stocks, top50 = info["Top100@Y"], set(info["Top50@Y"])
    X, y = base_features(dates, r, traded, gap, known, stocks)
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    half = year * 2 + (np.repeat(dates.month.to_numpy()[:, None], S, axis=1) > 6)
    yrs = dates.year.to_numpy()
    g = gap.to_numpy()
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y) & np.isfinite(g)
    pm = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1)
    trd = traded.to_numpy()
    XG = np.concatenate([X, preopen_features(dates, gap, traded)], -1)
    ev = ok & np.isin(year, EVAL)
    is50 = np.array([s in top50 for s in stocks])[None, :]
    subsets = (("all 100 stocks", ev), ("the 50-stock subset", ev & is50))
    fm = lambda t: f"{UNIT * t[0]:+5.1f}e-4 ± {UNIT * t[1]:3.1f}e-4"
    sig = np.zeros((T, S, 3))
    Yt = torch.tensor(np.where(ok, y, 0.0), dtype=torch.float32, device=DEV)
    Mt = torch.tensor(ok, device=DEV)
    At = torch.tensor(pm | ok, device=DEV) if premask else Mt
    name = "fast trainer, d 32, soft Score, 40 seeds" + (", pre-open mask" if premask else "")
    res = {name: np.full((T, S), np.nan)}
    t_train = 0.0
    ridge_ref = np.full((T, S), np.nan)
    for Y in EVAL:
        if premask:
            gh_s, gh_r = it.ghat_rows(XG, g, ok, pm, year, Y - 1), it.ghat_rows(XG, g, ok, pm, year, Y)
            Zs = np.concatenate([X, gap_block(gh_s, pm & np.isfinite(gh_s))], -1)
            Zr = np.concatenate([X, gap_block(gh_r, pm & np.isfinite(gh_r))], -1)
        else:
            gh_s, gh_r = ghat_for(XG, g, ok, year, Y - 1), ghat_for(XG, g, ok, year, Y)
            Zs = np.concatenate([X, gap_block(gh_s, ok)], -1)
            Zr = np.concatenate([X, gap_block(gh_r, ok)], -1)
        inner, sel, rows, te = ok & (year < Y - 1), ok & (year == Y - 1), ok & (year < Y), ok & (year == Y)
        m = Ridge(Zs[inner], y[inner])
        a = min(ALPHAS, key=lambda a: np.mean((m.predict(Zs[sel], m.weights(a)) - y[sel]) ** 2))
        mr = Ridge(Zr[rows], y[rows])
        ridge_ref[te] = mr.predict(Zr[te], mr.weights(a))
        mu_s = oos_mu(Zs, y, ok, year, half, a, Y - 1)
        mu_r = oos_mu(Zr, y, ok, year, half, a, Y)
        inp_s = it.Inputs("day", Zs, mu_s, y, ok, trd, sig, inner)
        inp_r = it.Inputs("day", Zr, mu_r, y, ok, trd, sig, rows)
        dd = lambda cond: torch.tensor(np.nonzero(cond & ok.any(1))[0], device=DEV)
        d_in, d_sel, d_rows, d_te = dd(yrs < Y - 1), dd(yrs == Y - 1), dd(yrs < Y), dd(yrs == Y)
        torch.cuda.synchronize()
        t0 = time.time()
        f, eps = fit_predict(seeds, inp_s, inp_r, Yt, Mt, At, d_in, d_sel, d_rows, d_te)
        torch.cuda.synchronize()
        t_train += time.time() - t0
        o = np.zeros((T, S))
        o[d_te.cpu().numpy()] = f
        res[name][te] = o[te]
        print(f"{Y} done in {time.time() - t0:.1f} s: epochs {eps}", flush=True)
    ref = "E4j final2 (one network per seed)"
    f2 = Path(__file__).resolve().parent / "runs" / "e4j_final2_pred.npz"
    if f2.exists():
        z2 = np.load(f2)
        res[ref] = z2["d_32_soft_Score_40_seeds"]
        res["ref: MSE ridge"] = z2["ref_MSE_ridge"]
    else:  # no saved final2 run: compare with the MSE ridge only
        res["ref: MSE ridge"] = ridge_ref
        ref = "ref: MSE ridge"
    print(f"\ntraining time of the 5 folds x 40 seeds: {t_train / 60:.1f} min; whole run {(time.time() - t_all) / 60:.1f} min",
          flush=True)
    print("\n######## the fast trainer vs final2 (same seeds, initial weights and data order; dropout differs)")
    report(res, {name: ref, **({ref: "ref: MSE ridge"} if ref != "ref: MSE ridge" else {})}, y, subsets, fm)
    np.savez(Path(__file__).resolve().parent / "runs" / ("e4j_fast_check_premask_pred.npz" if premask else
                                                          "e4j_fast_check_pred.npz"),
             y=y, ok=ok, ev=ev, year=year, is50=is50, fast=res[name])


if __name__ == "__main__":
    main()
