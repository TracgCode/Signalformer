"""E2c: a minimal attention over the stock's price window on the ridge residual, with the query from a learned
constant, the price state, or the text, walk-forward with a selection and a refit phase (TWStock).

Stage 1 is the base ridge (poly p0 + range + US session, 52 features; alpha chosen on Y - 1 as before), whose
forecast mu is the reference. Stage 2 forecasts the residual e = r - mu; the forecast is mu + e_hat.
Model (D = 8, about 600 parameters): every day k of the last L = 60 days, f_k = [e_{k-1}, mu_k, traded, X_k]
(standardized), is projected h_k = P f_k; keys are h_k rotated by RoPE (relative position k - 60, base 100); values
are h_k; logits q . key_k / sqrt(D), softmax over the 60 days; e_hat = w . sum_k a_k h_k (no bias, w started at 0).
Queries:
  Q0  a learned constant vector;
  Q1  W_q h_60 (the price state of the forecast day, known before its open);
  Q2  W_q u, u = 18 text inputs (log(1 + count), the mean LLM score, the mean of the first 16 PCs of the bge-m3
      embeddings), standardized on the stock-days with an own overnight item before 2022; 0 without an item;
  Q3  W_q v, v = the mean raw 1024-dim bge-m3 embedding of the stock's own overnight items, centred on the items
      before 2022 and scaled to unit mean variance; 0 without an item;
  Q2s, Q3s  Q2, Q3 with the text permuted across the stock-days with an item within each year.
Without an item the text queries are 0, so the attention is uniform (the 60-day mean of h).
Splits for forecast year Y:
  selection: the residuals of the years before Y - 1 come from ridges fitted without their half-year block (leave
  one half-year block out within the years before Y - 1), those of Y - 1 from the ridge fitted on the years before
  Y - 1; the net is trained on the years before Y - 1 and evaluated after every epoch on the stock-days of Y - 1 with
  an own overnight item; the best number of gradient steps (0 allowed) is kept (patience 3, up to 30 epochs);
  refit: residuals of the years before Y from leave one half-year block out within them, those of Y from the ridge
  fitted on all years before Y (the reference forecast); a new net with the same seed is trained on all years before
  Y for the selected number of steps and forecasts Y.
AdamW lr 1e-3, weight decay 1e-2, batches of 1024; 3 seeds averaged.

    python demo/stock_news_attn2.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_news import text_features  # noqa: E402
from stock_news_attn import report  # noqa: E402
from stock_poly import ROOT, load  # noqa: E402
from stock_recurrent import base_features, lagged  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stock_model import block_out_of_sample  # noqa: E402

U = 1e4
SEEDS = (2021, 2022, 2023)
L, D, BATCH = 60, 8, 1024
VERSIONS = ("Q0 const", "Q1 price", "Q2 text", "Q2s shuffled", "Q3 emb", "Q3s shuffled")


class Mini(nn.Module):
    def __init__(self, F, kind, Dq):
        super().__init__()
        self.kind = kind
        self.P = nn.Linear(F, D)
        if kind == "const":
            self.q = nn.Parameter(torch.zeros(D))
        else:
            self.wq = nn.Linear(D if kind == "price" else Dq, D, bias=False)
        self.head = nn.Linear(D, 1, bias=False)
        nn.init.zeros_(self.head.weight)
        theta = 100.0 ** (-torch.arange(0, D, 2) / D)                  # (D/2,)
        ang = (torch.arange(L) - (L - 1)).float()[:, None] * theta     # (L, D/2), position relative to the last day
        self.register_buffer("cos", ang.cos())
        self.register_buffer("sin", ang.sin())

    def forward(self, w, qin):
        h = self.P(w)                                                  # (B, L, D)
        x1, x2 = h[..., 0::2], h[..., 1::2]
        k = torch.stack([x1 * self.cos - x2 * self.sin, x1 * self.sin + x2 * self.cos], -1).flatten(-2)
        if self.kind == "const":
            q = self.q.expand(len(h), D)
        elif self.kind == "price":
            q = self.wq(h[:, -1])
        else:
            q = self.wq(qin)
        a = torch.softmax((k @ q[:, :, None]).squeeze(-1) / D ** 0.5, -1)
        return self.head((a[:, :, None] * h).sum(1)).squeeze(1)


def oos_mu(X, y, ok, year, half, a, Yb):
    """mu on the years before Yb by leave one half-year block out within them, and on Yb from the fit on all before."""
    return block_out_of_sample(X, y, ok, year, half, a, Yb)


def inputs(X, y, mu, ok, trd, tr):
    e = np.where(ok, y - mu, np.nan)
    e0 = np.where(np.isfinite(e) & trd, e, 0.0)
    se = np.nanstd(e[tr])
    Xs = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-12)
    mus = (mu - np.nanmean(mu[tr])) / se
    F = np.concatenate([np.stack([lagged(e0, 1) / se, np.nan_to_num(mus), trd.astype(float)], -1), Xs], -1)
    return np.nan_to_num(F).astype(np.float32), e, se


class Runner:
    def __init__(self, F, Q, e, se, device):
        self.dev = device
        self.F = torch.tensor(F, device=device)
        self.Q = Q
        self.Y = torch.tensor(np.nan_to_num(e / se), dtype=torch.float32, device=device)
        self.se = se
        self.ar = torch.arange(L, device=device) - (L - 1)

    def idx(self, m):
        return [torch.tensor(a, device=self.dev) for a in np.nonzero(m)]

    def fwd(self, net, t, s):
        return net(self.F[t[:, None] + self.ar, s[:, None]], None if self.Q is None else self.Q[t, s])

    def predict(self, net, t, s):
        net.eval()
        with torch.no_grad():
            p = torch.cat([self.fwd(net, t[i:i + 8192], s[i:i + 8192]) for i in range(0, len(t), 8192)])
        net.train()
        return p

    def train(self, kind, Dq, seed, tr, steps=None, va=None):
        """steps given: train that many steps; else early stop on va; returns (net, best steps, val curve)."""
        torch.manual_seed(seed)
        g = torch.Generator(device=self.dev).manual_seed(seed)
        net = Mini(self.F.shape[-1], kind, Dq).to(self.dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        ttr, s_tr = self.idx(tr)
        per = (len(ttr) + BATCH - 1) // BATCH
        if va is not None:
            tva, s_va = self.idx(va)
            val = lambda: (self.predict(net, tva, s_va) - self.Y[tva, s_va]).pow(2).mean().item()
            z = self.Y[tva, s_va].pow(2).mean().item()
            best, best_steps, bad, curve = val(), 0, 0, []
            curve.append(best / z)
        done = 0
        for ep in range(1, 31):
            order = torch.randperm(len(ttr), device=self.dev, generator=g)
            for i in range(0, len(order), BATCH):
                if steps is not None and done >= steps:
                    return net, done, None
                b = order[i:i + BATCH]
                loss = (self.fwd(net, ttr[b], s_tr[b]) - self.Y[ttr[b], s_tr[b]]).pow(2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                done += 1
            if va is not None:
                v = val()
                curve.append(v / z)
                if v < best:
                    best, best_steps, bad = v, done, 0
                else:
                    bad += 1
                    if bad >= 3:
                        break
            if steps is not None and done >= steps:
                break
        return net, (best_steps if va is not None else done), (curve if va is not None else None)


def main() -> None:
    device = "cuda"
    dates, r, traded, gap, known = load()
    T, S = r.shape
    info = json.loads((ROOT / "info.json").read_text())
    stocks, top50 = info["Top100@Y"], set(info["Top50@Y"])
    X, y = base_features(dates, r, traded, gap, known, stocks)
    LLM, ln, EMB, _, RAW = text_features(dates, stocks, raw=True)
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    half = year * 2 + (np.repeat(dates.month.to_numpy()[:, None], S, axis=1) > 6)
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y)
    trd = traded.to_numpy()

    cnt = np.round(np.expm1(LLM[..., ln.index("mops count over")]) + np.expm1(LLM[..., ln.index("cnyes count over")]))
    has = cnt > 0
    c1 = np.maximum(cnt, 1)
    s_mean = (LLM[..., ln.index("mops score over")] + LLM[..., ln.index("cnyes score over")]) / c1
    raw = np.concatenate([np.log1p(cnt)[..., None], s_mean[..., None], EMB[..., :16] / c1[..., None]], -1)
    pre = has & (year < 2022) & ok
    Uz = np.where(has[..., None], (raw - raw[pre].mean(0)) / (raw[pre].std(0) + 1e-12), 0.0).astype(np.float32)
    c = RAW[pre].mean(0)
    sc = np.sqrt(((RAW[pre] - c) ** 2).mean())
    Vz = np.where(has[..., None], (RAW - c) / sc, 0.0).astype(np.float32)
    del RAW
    rng = np.random.default_rng(0)
    perm_src = np.zeros((T, S, 2), np.int64)  # for the shuffled versions: the (t, s) whose text a stock-day gets
    tt, ss = np.meshgrid(np.arange(T), np.arange(S), indexing="ij")
    perm_src[..., 0], perm_src[..., 1] = tt, ss
    for yy in np.unique(year[ok]):
        m_ = has & (year == yy)
        src = np.stack([tt[m_], ss[m_]], -1)
        perm_src[m_] = src[rng.permutation(len(src))]
    sh = lambda A: A[perm_src[..., 0], perm_src[..., 1]]
    Qs = {"Q0 const": ("const", None), "Q1 price": ("price", None), "Q2 text": ("text", Uz),
          "Q2s shuffled": ("text", sh(Uz)), "Q3 emb": ("text", Vz), "Q3s shuffled": ("text", sh(Vz))}
    print(f"{ok.sum()} stock-days; own overnight item on {np.mean(has[ok]):.1%}; text 18 dims, embedding "
          f"{Vz.shape[-1]} dims", flush=True)
    Qt = {k: (kind, None if A is None else torch.tensor(A, device=device)) for k, (kind, A) in Qs.items()}
    del Vz

    res = {k: np.full((T, S), np.nan) for k in ("ridge",) + VERSIONS}
    for Y in EVAL:
        inner, sel = ok & (year < Y - 1), ok & (year == Y - 1)
        m = Ridge(X[inner], y[inner])
        a = min(ALPHAS, key=lambda a: np.mean((m.predict(X[sel], m.weights(a)) - y[sel]) ** 2))
        mu_s = oos_mu(X, y, ok, year, half, a, Y - 1)
        mu_r = oos_mu(X, y, ok, year, half, a, Y)
        te = ok & (year == Y)
        res["ridge"][te] = mu_r[te]
        tr_s, va_s, tr_r = ok & (year < Y - 1), ok & (year == Y - 1) & has, ok & (year < Y)
        F_s, e_s, se_s = inputs(X, y, mu_s, ok, trd, tr_s)
        F_r, e_r, se_r = inputs(X, y, mu_r, ok, trd, tr_r)
        for v in VERSIONS:
            kind, Q = Qt[v]
            Dq = 0 if Q is None else Q.shape[-1]
            run_s, run_r = Runner(F_s, Q, e_s, se_s, device), Runner(F_r, Q, e_r, se_r, device)
            tte, s_te = run_r.idx(te)
            preds, info_ = [], []
            for seed in SEEDS:
                _, steps, curve = run_s.train(kind, Dq, seed, tr_s, va=va_s)
                if steps == 0:
                    preds.append(np.zeros(len(tte)))
                else:
                    net, _, _ = run_r.train(kind, Dq, seed, tr_r, steps=steps)
                    preds.append(run_r.predict(net, tte, s_te).cpu().numpy() * se_r)
                info_.append(f"{steps} steps (val {min(curve):.5f})")
            res[v][te] = mu_r[te] + np.mean(preds, 0)
            print(f"{Y} {v}: {', '.join(info_)}", flush=True)
            del run_s, run_r
            torch.cuda.empty_cache()

    ev = ok & np.isin(year, EVAL)
    is50 = np.array([s in top50 for s in stocks])[None, :]
    subsets = (("all 100 stocks", ev), ("the 50-stock subset", ev & is50),
               ("100 stocks, days with an own overnight item", ev & has),
               ("100 stocks, 2025-2026", ev & (year >= 2025)), ("50 stocks, 2025-2026", ev & is50 & (year >= 2025)))
    fm = lambda t: f"{U * t[0]:+5.1f}e-4 ± {U * t[1]:3.1f}e-4"
    print("\n######## each version vs the ridge")
    report(res, {v: "ridge" for v in VERSIONS}, y, subsets, fm)
    print("\n######## controls: Q1 vs Q0, Q2 vs Q2s, Q3 vs Q3s, Q2 vs Q1")
    for a_, b_ in (("Q1 price", "Q0 const"), ("Q2 text", "Q2s shuffled"), ("Q3 emb", "Q3s shuffled"),
                   ("Q2 text", "Q1 price")):
        report({b_: res[b_], a_: res[a_]}, {a_: b_}, y, subsets[:3], fm)


if __name__ == "__main__":
    main()
