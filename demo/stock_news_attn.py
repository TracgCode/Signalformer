"""E2b: text x price interactions. (1) Ridge with interaction terms; (2) text-queried cross-attention over the
stock's price window, with the query FiLM-modulated by the text, walk-forward (TWStock).

Text inputs per stock and day (from demo/stock_news.py text_features, overnight items of the stock only):
has (an own overnight item exists), n = log(1 + count), the mean LLM score s = P(pos) - P(neg), and the mean of the
first 16 principal components of the bge-m3 embeddings (PCA fitted on the items before 2022); standardized on the
stock-days with an item before 2022; all 0 without an item.
(1) Ridge on the base (poly p0 + range + US session, 52) plus: has and n; has x X (52); s x X (52); the first 4 PCs
    x X (208). Same walk-forward as before.
(2) Stage 1 is the base ridge with out-of-sample residuals on the training years (as in demo/stock_recurrent.py);
    stage 2 forecasts the residual e_t. Keys and values: the last L = 60 days of [e_{t-1}, mu_t, traded, X]
    (standardized), each day linearly projected to d = 16 plus a learned position embedding; one head.
    Query: q0 = W h_t (the projected current day, so the price state picks the base query), modulated by the text u:
    q = (1 + has G(u)) * q0 + has B(u), G and B one-hidden-layer MLPs (32 units) with zero-initialized outputs, so
    without an item (and at the start of training) the model is plain attention pooling. Output: dropout 0.5, then a
    zero-initialized Linear(16 -> 1). AdamW lr 1e-3, wd 1e-2, batches of 1024, early stop on Y - 1 (patience 3, up to
    30 epochs), 3 seeds averaged. Versions:
      A  u = 0 and has = 0 (no text: attention pooling with the price-state query);
      B  u shuffled across the stock-days with an item within each year (has kept, content destroyed);
      C  the real u.
    C - B is the value of the content, B - A that of the item flag.
Reported pooled 2022 - 2026 on the 100 and 50 stocks, the stock-days with an own item, and 2025 - 2026: R2, the
score of the sign, its cross-sectional part, and paired differences to a reference, day-clustered.

    python demo/stock_news_attn.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_news import text_features  # noqa: E402
from stock_poly import ROOT, load  # noqa: E402
from stock_recurrent import base_features, lagged  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge, clustered_mean  # noqa: E402

U = 1e4
SEEDS = (2021, 2022, 2023)
L, D, HID = 60, 16, 32


class Attn(nn.Module):
    def __init__(self, F, Du):
        super().__init__()
        self.proj = nn.Linear(F, D)
        self.pos = nn.Parameter(torch.zeros(L, D))
        self.wq, self.wk, self.wv = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)
        self.G = nn.Sequential(nn.Linear(Du, HID), nn.GELU(), nn.Linear(HID, D))
        self.B = nn.Sequential(nn.Linear(Du, HID), nn.GELU(), nn.Linear(HID, D))
        for m in (self.G[-1], self.B[-1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.drop = nn.Dropout(0.5)
        self.head = nn.Linear(D, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, w, u, has):
        h = self.proj(w) + self.pos                       # (B, L, D)
        q0 = self.wq(h[:, -1])
        hs = has[:, None]
        q = (1 + hs * self.G(u)) * q0 + hs * self.B(u)
        a = torch.softmax((self.wk(h) @ q[:, :, None]).squeeze(-1) / D ** 0.5, -1)
        o = (a[:, :, None] * self.wv(h)).sum(1)
        return self.head(self.drop(o)).squeeze(1)


def train_attn(F, Uin, HAS, tgt, tr, va, te, seed, device):
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    Ft = torch.tensor(F, dtype=torch.float32, device=device)
    Ut = torch.tensor(Uin, dtype=torch.float32, device=device)
    Ht = torch.tensor(HAS, dtype=torch.float32, device=device)
    st = np.nanstd(tgt[tr])
    Yt = torch.tensor(np.nan_to_num(tgt / st), dtype=torch.float32, device=device)
    ar = torch.arange(L, device=device) - (L - 1)
    idx = lambda m: [torch.tensor(a, device=device) for a in np.nonzero(m)]
    (ttr, s_tr), (tva, s_va), (tte, s_te) = idx(tr), idx(va), idx(te)
    net = Attn(F.shape[-1], Uin.shape[-1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    fwd = lambda t, s: net(Ft[t[:, None] + ar, s[:, None]], Ut[t, s], Ht[t, s])

    def predict(t, s):
        net.eval()
        with torch.no_grad():
            return torch.cat([fwd(t[i:i + 8192], s[i:i + 8192]) for i in range(0, len(t), 8192)])

    val = lambda: (predict(tva, s_va) - Yt[tva, s_va]).pow(2).mean().item()
    best, state, bad, best_ep = val(), {k: v.clone() for k, v in net.state_dict().items()}, 0, 0
    for ep in range(1, 31):
        net.train()
        order = torch.randperm(len(ttr), device=device, generator=g)
        for i in range(0, len(order), 1024):
            b = order[i:i + 1024]
            loss = (fwd(ttr[b], s_tr[b]) - Yt[ttr[b], s_tr[b]]).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        v = val()
        if v < best:
            best, state, bad, best_ep = v, {k: x.clone() for k, x in net.state_dict().items()}, 0, ep
        else:
            bad += 1
            if bad >= 3:
                break
    net.load_state_dict(state)
    return predict(tte, s_te).cpu().numpy() * st, best_ep


def report(res, refs, y, subsets, fm):
    for uname, um in subsets:
        y_, d_ = y[um], np.nonzero(um)[0]
        _, inv = np.unique(d_, return_inverse=True)
        mkr = (np.bincount(inv, weights=y_) / np.bincount(inv))[inv]
        base = np.mean((y_ - y_.mean()) ** 2)
        print(f"\n== {uname} ({um.sum()} stock-days); always down: score {fm(clustered_mean(-y_, d_))}")
        for name, mu in res.items():
            p = mu[um]
            sg = np.where(p > 0, 1.0, -1.0)
            ref = refs.get(name)
            line = (f"{name:26s} R2 {1 - np.mean((p - y_) ** 2) / base:+.4f}  score {fm(clustered_mean(sg * y_, d_))}  "
                    f"cross {fm(clustered_mean(sg * (y_ - mkr), d_))}")
            if ref is not None:
                mb = res[ref][um]
                sb = np.where(mb > 0, 1.0, -1.0)
                dm = clustered_mean((mb - y_) ** 2 - (p - y_) ** 2, d_)
                line += (f"  | vs {ref}: flips {np.mean(sg != sb):5.1%}  R2 {dm[0] / base:+.4f} ± {dm[1] / base:.4f}  "
                         f"score {fm(clustered_mean((sg - sb) * y_, d_))}")
            print(line, flush=True)


def main() -> None:
    device = "cuda"
    dates, r, traded, gap, known = load()
    T, S = r.shape
    info = json.loads((ROOT / "info.json").read_text())
    stocks, top50 = info["Top100@Y"], set(info["Top50@Y"])
    X, y = base_features(dates, r, traded, gap, known, stocks)
    LLM, ln, EMB, has = text_features(dates, stocks)
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y)
    trd = traded.to_numpy()

    cnt = np.expm1(LLM[..., ln.index("mops count over")]) + np.expm1(LLM[..., ln.index("cnyes count over")])
    cnt = np.round(cnt)
    has = cnt > 0
    c1 = np.maximum(cnt, 1)
    s_mean = (LLM[..., ln.index("mops score over")] + LLM[..., ln.index("cnyes score over")]) / c1
    pcs = EMB[..., :16] / c1[..., None]
    raw = np.concatenate([np.log1p(cnt)[..., None], s_mean[..., None], pcs], -1)
    pre = has & (year < 2022) & ok
    mu_u, sd_u = raw[pre].mean(0), raw[pre].std(0) + 1e-12
    Uz = np.where(has[..., None], (raw - mu_u) / sd_u, 0.0)
    print(f"{ok.sum()} stock-days; own overnight item on {np.mean(has[ok]):.1%}; text input {Uz.shape[-1]} dims",
          flush=True)

    ev = ok & np.isin(year, EVAL)
    is50 = np.array([s in top50 for s in stocks])[None, :]
    subsets = (("all 100 stocks", ev), ("the 50-stock subset", ev & is50),
               ("100 stocks, days with an own overnight item", ev & has),
               ("100 stocks, 2025-2026", ev & (year >= 2025)), ("50 stocks, 2025-2026", ev & is50 & (year >= 2025)))
    fm = lambda t: f"{U * t[0]:+5.1f}e-4 ± {U * t[1]:3.1f}e-4"

    # (1) ridge with interactions
    hz = has[..., None].astype(float)
    sz = Uz[..., 1:2]
    SETS = {"base": X,
            "+ has, n": np.concatenate([X, hz, Uz[..., :1]], -1),
            "+ has, n, has x X": np.concatenate([X, hz, Uz[..., :1], hz * X], -1),
            "+ has, n, s, s x X": np.concatenate([X, hz, Uz[..., :2], sz * X], -1),
            "+ has, n, PC1-4 x X": np.concatenate([X, hz, Uz[..., :1], Uz[..., 2:6]]
                                                  + [Uz[..., k:k + 1] * X for k in range(2, 6)], -1)}
    res = {}
    for name, Z in SETS.items():
        mu = np.full((T, S), np.nan)
        for Y in EVAL:
            inner, sel, rows_, te = ok & (year < Y - 1), ok & (year == Y - 1), ok & (year < Y), ok & (year == Y)
            m = Ridge(Z[inner], y[inner])
            a = min(ALPHAS, key=lambda a: np.mean((m.predict(Z[sel], m.weights(a)) - y[sel]) ** 2))
            m = Ridge(Z[rows_], y[rows_])
            mu[te] = m.predict(Z[te], m.weights(a))
        res[name] = mu
        print(f"ridge {name}: {Z.shape[-1]} features", flush=True)
    print("\n######## (1) ridge with text x price interactions")
    report(res, {k: "base" for k in SETS if k != "base"}, y, subsets, fm)

    # (2) cross-attention on the ridge residual
    names = ["ridge", "A no text", "B shuffled text", "C text"]
    res2 = {k: np.full((T, S), np.nan) for k in names}
    rng = np.random.default_rng(0)
    for Y in EVAL:
        rows = ok & (year < Y)
        inner, sel = ok & (year < Y - 1), ok & (year == Y - 1)
        m = Ridge(X[inner], y[inner])
        a = min(ALPHAS, key=lambda a: np.mean((m.predict(X[sel], m.weights(a)) - y[sel]) ** 2))
        mu = np.full((T, S), np.nan)
        m = Ridge(X[rows], y[rows])
        te = ok & (year == Y)
        mu[te] = m.predict(X[te], m.weights(a))
        for yy in np.unique(year[rows]):
            fit, at = rows & (year != yy), rows & (year == yy)
            mf = Ridge(X[fit], y[fit])
            mu[at] = mf.predict(X[at], mf.weights(a))
        e = np.where(ok & (year <= Y), y - mu, np.nan)
        e0 = np.where(np.isfinite(e) & trd, e, 0.0)
        res2["ridge"][te] = mu[te]
        tr, va = ok & (year < Y - 1), ok & (year == Y - 1)
        se = np.nanstd(e[tr])
        Xs = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-12)
        mus = (mu - np.nanmean(mu[tr])) / se
        F = np.nan_to_num(np.concatenate([np.stack([lagged(e0, 1) / se, np.nan_to_num(mus), trd.astype(float)], -1),
                                          Xs], -1)).astype(np.float32)
        Ush = Uz.copy()
        for yy in np.unique(year[ok]):
            m_ = has & (year == yy)
            v = Uz[m_]
            Ush[m_] = v[rng.permutation(len(v))]
        variants = {"A no text": (np.zeros_like(Uz), np.zeros((T, S))), "B shuffled text": (Ush, has.astype(float)),
                    "C text": (Uz, has.astype(float))}
        for vname, (Uv, Hv) in variants.items():
            outs = [train_attn(F, Uv.astype(np.float32), Hv, e, tr, va, te, s, device) for s in SEEDS]
            res2[vname][te] = mu[te] + np.mean([o[0] for o in outs], 0)
            print(f"{Y} {vname}: best epochs {[o[1] for o in outs]}", flush=True)
    print("\n######## (2) text-queried cross-attention on the ridge residual")
    report(res2, {"A no text": "ridge", "B shuffled text": "A no text", "C text": "B shuffled text"}, y, subsets, fm)
    print("\n-- C vs A and C vs ridge")
    report({k: res2[k] for k in ("ridge", "A no text", "C text")}, {"C text": "ridge"}, y, subsets[:3], fm)


if __name__ == "__main__":
    main()
