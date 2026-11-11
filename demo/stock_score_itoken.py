"""E4j: from the E2f/E2j iTransformer to the E4h set model, one difference at a time (TWStock).

The iT of E2f/E2j (stock tokens from the 60-day window, one self-attention layer across the stocks, residual e of the
ridge, MSE, zero-started head) did not beat the ridge; the E4h set model (tokens from the day's 55 features, a
Transformer encoder layer, target r, soft Score) does. The models below walk from one to the other, changing one
thing per step, so that the gain can be attributed:
- L0 iT as in E2j: token = Linear(window of [e, r, g, traded], 60 days) + Linear(day's 55 features and mu / se),
  dropout 0.3, pre-norm self-attention (2 heads, d 16, no mask, no feed-forward), head Linear(d, 1) started at 0;
  forecast mu / sd(r) + head, MSE on r / sd(r) (i.e. on the residual);
- L1: L0 with the soft Score: forecast a * mu / sd(mu) + head (a learnable, started at 1);
- L2: L1 forecasting r directly (no mu offset), head with the usual initialization;
- L3: L2 without e and mu in the token (window of [r, g, traded] + the 55 day features);
- L4: L3 with the E4h block (GELU on the token, masked Transformer encoder layer with feed-forward 2d, dropout 0.2,
  4 heads), d 16;
- L5: L4 with d 64;
- L6: L5 with the day's features only (= the E4h set model);
- L7: L5 with the window only.
e is the out-of-sample residual of the ridge (stage-2 rows by half-year blocks out), lagged by one day. Epochs chosen
on Y - 1 (patience 3, up to 40, at least 1), refit on all years before Y, 5 seeds averaged, AdamW lr 1e-3, weight
decay 1e-2, batches of 16 days.
Part 'width' (python demo/stock_score_itoken.py width): the width d of the E4h set model (day features, E4h block,
target r) at 16 / 32 / 64 / 128, under the MSE and under the soft Score, since the ladder found the step d 16 -> 64
decisive but changed the loss before it.
Part 'ident' (python demo/stock_score_itoken.py ident): stock identity in the set model (day features, E4h block,
d 32, target r, soft Score), the idea of StockIdentityAttention without the news: the token adds (a) a learned
embedding per stock, (b) Linear(industry one-hot), (c) the industry one-hot given to the wrong stocks (control),
(d) both (a) and (b); against no identity.
Part 'aux' (python demo/stock_score_itoken.py aux): the set model (day features, E4h block, d 32, target r, soft
Score) with an auxiliary head that forecasts the gap residual g - g_hat_pre (MSE on its standardized value, weight
lambda 0.3 / 1 / 3 added to the soft Score; the epochs are still chosen on the soft Score alone), since the news and
the US returns carry a signal on the gap but none on r: a shared representation might still help r. Also the day
features plus the 4 raw pre-open features (no news), without the auxiliary head.
Part 'hp' (python demo/stock_score_itoken.py hp): the training settings of the d 32 soft-Score set model, dropout
0.1 / 0.2 / 0.4 x weight decay 1e-2 / 1e-1 x lr 1e-3 / 3e-4 (12 settings), and the setting chosen per fold by the mean
best validation loss on Y - 1 over the seeds (out of sample: no test year is used for the choice).
Part 'hp2' (python demo/stock_score_itoken.py hp2): since the lower lr was worse, lr 1e-3 / 2e-3 / 3e-3 x batches of
8 / 16 / 32 days at the default dropout 0.2 and weight decay 1e-2, with the same choice on Y - 1.
Part 'rank' (python demo/stock_score_itoken.py rank): the inputs as ranks within the day. Each of the 55 day features
is replaced by its rank among the usable stocks of the day, scaled to [-1, 1] (robust to outliers and already
relative to the day), or appended to the standardized features; d 32 soft-Score set model.
Part 'top50' (python demo/stock_score_itoken.py top50): the d 32 soft-Score set model trained and run on the 50-stock
subset only (its loss, its attention context and its epochs choice restricted to those stocks), against the model
trained on all 100 stocks, evaluated on the 50-stock subset.
Part 'final' (python demo/stock_score_itoken.py final): the best single model (d 32, soft Score) and the d 64 MSE set
model with 20 seeds each, their ensemble (each divided by its sd per fold), and the Score per year; forecasts saved to
demo/runs/e4j_final_pred.npz.
Part 'final2' (python demo/stock_score_itoken.py final2): since 20 seeds beat 5, the d 32 soft-Score model with 40
seeds, and d 16 / d 128 soft Score with 20 seeds each; forecasts to demo/runs/e4j_final2_pred.npz.
Part 'premask' (python demo/stock_score_itoken.py premask): the d 32 soft-Score model with 40 seeds whose attention
context and gap-block day mean use only stocks known before 09:00 (features of t - 1 finite, past the first FIRST days),
instead of the rows usable on day t (which also require that the stock trades on t and that r and the gap of t exist).
The loss and the evaluation keep the usable rows; ĝ_pre is predicted on the pre-open rows. Paired with the final2
model (same seeds 0..39); forecasts to demo/runs/e4j_premask_pred.npz.
Usage: python demo/stock_score_itoken.py [width | ident | aux | hp | hp2 | rank | top50 | final | final2 | premask]
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
from stock_recurrent import base_features, lagged  # noqa: E402
from stock_score_loss import loss_fn  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge  # noqa: E402

DEV = "cuda"
SEEDS = (0, 1, 2, 3, 4)
W = 60
# name: (token, block, d, target, loss, zero head)
LADDER = {
    "L0 iT as in E2j (residual, MSE, zero head)": ("iT", "it", 16, "residual", "mse", True),
    "L1 + soft Score (residual, zero head)": ("iT", "it", 16, "residual", "soft", True),
    "L2 + target r": ("iT", "it", 16, "r", "soft", False),
    "L3 + token without e, mu": ("window+day", "it", 16, "r", "soft", False),
    "L4 + E4h block, d 16": ("window+day", "e4h", 16, "r", "soft", False),
    "L5 + d 64": ("window+day", "e4h", 64, "r", "soft", False),
    "L6 day features only (= E4h set)": ("day", "e4h", 64, "r", "soft", False),
    "L7 window only": ("window", "e4h", 64, "r", "soft", False),
}


WIDTH = {f"d {d_}, {lab}": ("day", "e4h", d_, "r", k_, False) for k_, lab in (("mse", "MSE"), ("soft", "soft Score"))
         for d_ in (16, 32, 64, 128)}


IDENT = {f"d 32, soft Score, identity: {k_}": ("day", "e4h", 32, "r", "soft", False, k_)
         for k_ in ("none", "learned", "industry", "industry permuted", "learned + industry")}
AUXV = {f"d 32, soft Score, aux gap head lambda {l_}": ("day", "e4h", 32, "r", "soft", False, "none", l_)
        for l_ in (0.0, 0.3, 1.0, 3.0)}
AUXV["d 32, soft Score, day features + 4 raw pre-open"] = ("day+pre", "e4h", 32, "r", "soft", False, "none", 0.0)
HP = {f"d 32, soft Score, dropout {dr_}, wd {wd_:g}, lr {lr_:g}": ("day", "e4h", 32, "r", "soft", False, "none", 0.0,
                                                                   dr_, wd_, lr_)
      for dr_ in (0.1, 0.2, 0.4) for wd_ in (1e-2, 1e-1) for lr_ in (1e-3, 3e-4)}
HP2 = {f"d 32, soft Score, lr {lr_:g}, batch {b_} days": ("day", "e4h", 32, "r", "soft", False, "none", 0.0, 0.2, 1e-2,
                                                          lr_, b_)
       for lr_ in (1e-3, 2e-3, 3e-3) for b_ in (8, 16, 32)}
RANK = {f"d 32, soft Score, inputs {k_}": (t_, "e4h", 32, "r", "soft", False)
        for k_, t_ in (("standardized", "day"), ("ranks within the day", "dayrank"),
                       ("standardized + ranks", "day+rank"))}
TOP50 = {"d 32, soft Score, trained on 100 stocks": ("day", "e4h", 32, "r", "soft", False),
         "d 32, soft Score, trained on the 50-stock subset": ("day", "e4h", 32, "r", "soft", False, "none", 0.0, 0.2,
                                                              1e-2, 1e-3, 16, "top50")}
FINAL = {"d 32, soft Score, 20 seeds": ("day", "e4h", 32, "r", "soft", False, "none", 0.0, 0.2, 1e-2, 1e-3, 16, "all", 20),
         "d 64, MSE, 20 seeds": ("day", "e4h", 64, "r", "mse", False, "none", 0.0, 0.2, 1e-2, 1e-3, 16, "all", 20)}
FINAL2 = {"d 32, soft Score, 40 seeds": ("day", "e4h", 32, "r", "soft", False, "none", 0.0, 0.2, 1e-2, 1e-3, 16, "all", 40),
          "d 16, soft Score, 20 seeds": ("day", "e4h", 16, "r", "soft", False, "none", 0.0, 0.2, 1e-2, 1e-3, 16, "all", 20),
          "d 128, soft Score, 20 seeds": ("day", "e4h", 128, "r", "soft", False, "none", 0.0, 0.2, 1e-2, 1e-3, 16, "all",
                                          20)}
GT = None    # (sel, refit) tensors of the standardized gap residual for part aux
IND = None   # (S, K) industry one-hot tensor, set in main for part ident
IND_P = None
ATT = None   # (T, S) attention mask when it differs from the loss mask (part premask)
PREMASK = {"d 32, soft Score, 40 seeds, pre-open mask": FINAL2["d 32, soft Score, 40 seeds"]}


def ghat_rows(Z, g, ok, rows_out, year, Yb):
    """ghat_for, but the forecasts are made on rows_out (a superset of ok) instead of ok only."""
    from stock_itrans_news import ghat_for as _gf
    G = _gf(Z, g, ok, year, Yb)
    extra = rows_out & ~ok & np.isfinite(Z).all(-1)
    inner, sel, rows = ok & (year < Yb - 1), ok & (year == Yb - 1), ok & (year < Yb)
    m = Ridge(Z[inner], g[inner])
    a = min(ALPHAS, key=lambda a: np.mean((m.predict(Z[sel], m.weights(a)) - g[sel]) ** 2))
    at = extra & (year == Yb)
    if at.any():
        m = Ridge(Z[rows], g[rows])
        G[at] = m.predict(Z[at], m.weights(a))
    for yy in np.unique(year[rows]):
        at = extra & (year == yy)
        if at.any():
            mf = Ridge(Z[rows & (year != yy)], g[rows & (year != yy)])
            G[at] = mf.predict(Z[at], mf.weights(a))
    return G


class TokNet(nn.Module):
    def __init__(self, win_dim, cur_dim, block, d, zero, ident="none", S=100, dropout=0.2):
        super().__init__()
        self.block = block
        self.ident = ident
        if "learned" in ident:
            self.emb = nn.Parameter(torch.zeros(S, d))
        if "industry" in ident:
            self.indp = nn.Linear(IND.shape[1], d, bias=False)
        self.win = nn.Linear(win_dim, d) if win_dim else None
        self.cur = nn.Linear(cur_dim, d) if cur_dim else None
        if block == "it":
            self.drop = nn.Dropout(0.3)
            self.ln = nn.LayerNorm(d)
            self.att = nn.MultiheadAttention(d, 2, dropout=0.1, batch_first=True)
        else:
            self.enc = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=dropout, batch_first=True, norm_first=True,
                                                  activation="gelu")
        self.out = nn.Linear(d, 1)
        self.out2 = nn.Linear(d, 1)
        self.h_last = None
        if zero:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, win, cur, m):
        h = 0
        if self.win is not None:
            h = h + self.win(win)
        if self.cur is not None:
            h = h + self.cur(cur)
        if "learned" in self.ident:
            h = h + self.emb
        if "industry" in self.ident:
            h = h + self.indp(IND_P if "permuted" in self.ident else IND)
        if self.block == "it":
            h = self.drop(h)
            x = self.ln(h)
            h = h + self.att(x, x, x, need_weights=False)[0]
        else:
            h = self.enc(nn.functional.gelu(h), src_key_padding_mask=~m)
        self.h_last = h
        return self.out(h).squeeze(-1)


class Wrap(nn.Module):
    def __init__(self, net, residual, learn_a):
        super().__init__()
        self.net, self.residual = net, residual
        self.a = nn.Parameter(torch.ones(())) if learn_a else None

    def forward(self, win, cur, off, m):
        f = self.net(win, cur, m)
        return f + (self.a if self.a is not None else 1.0) * off if self.residual else f


class Inputs:
    """window and day inputs of one fit (selection or refit) for one token kind."""

    def __init__(self, token, Zd, mu, y, ok, trd, sig, fitm):
        e = np.where(ok, y - mu, np.nan)
        se = np.nanstd(e[fitm])
        e0 = np.where(np.isfinite(e) & trd, e, 0.0)
        cur = np.nan_to_num((Zd - Zd[fitm].mean(0)) / (Zd[fitm].std(0) + 1e-12))
        if token in ("dayrank", "day+rank"):
            Zm = np.where(ok[..., None], Zd, np.nan)
            rk = np.argsort(np.argsort(np.where(np.isnan(Zm), np.inf, Zm), 1), 1).astype(float)
            n = ok.sum(1, keepdims=True)[..., None].astype(float)
            rk = np.where(ok[..., None], 2 * rk / np.maximum(n - 1, 1) - 1, 0.0)
            cur = rk if token == "dayrank" else np.concatenate([cur, rk], -1)
        if token == "iT":
            mus = np.nan_to_num((mu - np.nanmean(mu[fitm])) / se)
            cur = np.concatenate([cur, mus[..., None]], -1)
            win = np.concatenate([lagged(e0, 1)[..., None] / se, sig], -1)
        else:
            win = sig
        self.win = torch.tensor(win, dtype=torch.float32, device=DEV) if not token.startswith("day") else None
        self.cur = torch.tensor(cur, dtype=torch.float32, device=DEV) if token != "window" else None
        self.win_dim = 0 if self.win is None else W * self.win.shape[-1]
        self.cur_dim = 0 if self.cur is None else self.cur.shape[-1]
        self.ar = torch.arange(W, device=DEV) - (W - 1)

    def batch(self, t):
        win = None
        if self.win is not None:
            win = self.win[t[:, None] + self.ar].permute(0, 2, 1, 3).flatten(2)
        return win, (None if self.cur is None else self.cur[t])


def train(cfg, inp, Y, OFF, M, days, seed, epochs=None, vdays=None, G=None):
    token, block, d, target, kind, zero = cfg[:6]
    ident = cfg[6] if len(cfg) > 6 else "none"
    lam = cfg[7] if len(cfg) > 7 else 0.0
    dropout, wd, lr = cfg[8:11] if len(cfg) > 8 else (0.2, 1e-2, 1e-3)
    bsz = cfg[11] if len(cfg) > 11 else 16
    torch.manual_seed(seed)
    g = torch.Generator(device=DEV).manual_seed(seed)
    net = Wrap(TokNet(inp.win_dim, inp.cur_dim, block, d, zero, ident, dropout=dropout), target == "residual",
               target == "residual" and kind != "mse").to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    best, best_ep, bad = np.inf, 0, 0

    def run(b):
        m = M[b]
        return net(*inp.batch(b), OFF[b], m if ATT is None else ATT[b])[m], Y[b][m]

    for ep in range(1, (epochs or 40) + 1):
        net.train()
        perm = days[torch.randperm(len(days), device=DEV, generator=g)]
        for i in range(0, len(perm), bsz):
            b = perm[i:i + bsz]
            f, y = run(b)
            l = loss_fn(kind, f, y)
            if lam > 0:
                m_ = M[b]
                g2 = net.net.out2(net.net.h_last).squeeze(-1)[m_]
                l = l + lam * ((g2 - G[b][m_]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
        if epochs is None:
            net.eval()
            with torch.no_grad():
                fs, ys = zip(*(run(vdays[i:i + 64]) for i in range(0, len(vdays), 64)))
                v = loss_fn(kind, torch.cat(fs), torch.cat(ys)).item()
            if v < best:
                best, best_ep, bad = v, ep, 0
            else:
                bad += 1
                if bad >= 3:
                    break
    net.eval()
    train.last_best = best
    return net, best_ep


def main() -> None:
    global IND, IND_P, LADDER, ATT
    part = next((p_ for p_ in ("width", "ident", "aux", "hp2", "hp", "rank", "top50", "final2", "final", "premask")
                 if p_ in sys.argv[1:]),
                "ladder")
    dates, r, traded, gap, known = load()
    T, S = r.shape
    info = json.loads((ROOT / "info.json").read_text())
    stocks, top50 = info["Top100@Y"], set(info["Top50@Y"])
    X, y = base_features(dates, r, traded, gap, known, stocks)
    try:  # the news struct items only define one reporting subset; the models do not use them
        STR = struct_features(dates, stocks)
        has = (STR[..., 0] + STR[..., 1]) > 0
    except FileNotFoundError:
        print("no news struct data in dataset/news: the subset of days with an own overnight item is skipped", flush=True)
        has = None
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    half = year * 2 + (np.repeat(dates.month.to_numpy()[:, None], S, axis=1) > 6)
    yrs = dates.year.to_numpy()
    g = gap.to_numpy()
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y) & np.isfinite(g)
    trd = traded.to_numpy()
    if part == "ident":
        from stock_identity_news import industry_onehot
        I_ = industry_onehot(stocks)[0]
        IND = torch.tensor(I_, device=DEV)
        IND_P = torch.tensor(I_[np.random.default_rng(1).permutation(S)], device=DEV)
    pre = ok & (year < 2022)
    r0 = np.where(trd, np.nan_to_num(r.to_numpy()), 0.0)
    g0 = np.where(trd, np.nan_to_num(g), 0.0)
    sig = np.stack([lagged(r0, 1) / r0[pre].std(), lagged(g0, 1) / g0[pre].std(), lagged(trd.astype(float), 1)], -1)
    PO = preopen_features(dates, gap, traded)
    XG = np.concatenate([X, PO], -1)
    ev = ok & np.isin(year, EVAL)
    is50 = np.array([s in top50 for s in stocks])[None, :]
    subsets = (("all 100 stocks", ev), ("the 50-stock subset", ev & is50)) + (
        (("100 stocks, days with an own overnight item", ev & has),) if has is not None else ()) + (
        ("100 stocks, 2025-2026", ev & (year >= 2025)),)
    fm = lambda t: f"{UNIT * t[0]:+5.1f}e-4 ± {UNIT * t[1]:3.1f}e-4"
    if part == "width":
        LADDER = WIDTH
    if part == "ident":
        LADDER = IDENT
    if part == "aux":
        LADDER = AUXV
    if part == "hp":
        LADDER = HP
    if part == "hp2":
        LADDER = HP2
    if part == "rank":
        LADDER = RANK
    if part == "top50":
        LADDER = TOP50
    if part == "final":
        LADDER = FINAL
    if part == "final2":
        LADDER = FINAL2
    if part == "premask":
        LADDER = PREMASK
    pm = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1)
    print(f"pre-open rows {pm.sum()}, usable rows {ok.sum()}, pre-open but not usable {(pm & ~ok).sum()}, "
          f"usable but not pre-open {(ok & ~pm).sum()}", flush=True)
    if part == "premask":
        ATT = torch.tensor(pm | ok, device=DEV)
    names = ["ref: MSE ridge"] + list(LADDER) + (["chosen on Y - 1 by the validation loss"] if part in ("hp", "hp2") else [])
    if part == "final":
        names.append("ensemble of the two")
    res = {k: np.full((T, S), np.nan) for k in names}
    Yt = torch.tensor(np.where(ok, y, 0.0), dtype=torch.float32, device=DEV)
    Mt = torch.tensor(ok, device=DEV)
    for Y in EVAL:
        if part == "premask":
            gh_s, gh_r = ghat_rows(XG, g, ok, pm, year, Y - 1), ghat_rows(XG, g, ok, pm, year, Y)
            Zs = np.concatenate([X, gap_block(gh_s, pm & np.isfinite(gh_s))], -1)
            Zr = np.concatenate([X, gap_block(gh_r, pm & np.isfinite(gh_r))], -1)
        else:
            gh_s, gh_r = ghat_for(XG, g, ok, year, Y - 1), ghat_for(XG, g, ok, year, Y)
            Zs = np.concatenate([X, gap_block(gh_s, ok)], -1)
            Zr = np.concatenate([X, gap_block(gh_r, ok)], -1)
        inner, sel, rows, te = ok & (year < Y - 1), ok & (year == Y - 1), ok & (year < Y), ok & (year == Y)
        gres = lambda gh, fm_: np.where(ok & np.isfinite(gh), (g - gh) / np.nanstd((g - gh)[fm_]), 0.0)
        G_s = torch.tensor(np.nan_to_num(gres(gh_s, inner)), dtype=torch.float32, device=DEV)
        G_r = torch.tensor(np.nan_to_num(gres(gh_r, rows)), dtype=torch.float32, device=DEV)
        m = Ridge(Zs[inner], y[inner])
        a = min(ALPHAS, key=lambda a: np.mean((m.predict(Zs[sel], m.weights(a)) - y[sel]) ** 2))
        mu_s = oos_mu(Zs, y, ok, year, half, a, Y - 1)
        mu_r = oos_mu(Zr, y, ok, year, half, a, Y)
        res[names[0]][te] = mu_r[te]
        dd = lambda cond: torch.tensor(np.nonzero(cond & ok.any(1))[0], device=DEV)
        d_in, d_sel, d_rows, d_te = dd(yrs < Y - 1), dd(yrs == Y - 1), dd(yrs < Y), dd(yrs == Y)
        offs = {"mse": (mu_s / y[inner].std(), mu_r / y[rows].std()),
                "soft": (mu_s / np.nanstd(mu_s[inner]), mu_r / np.nanstd(mu_r[rows]))}
        offs = {k: tuple(torch.tensor(np.nan_to_num(v), dtype=torch.float32, device=DEV) for v in vv)
                for k, vv in offs.items()}
        cache, msg, vbest = {}, [], {}
        for name, cfg in LADDER.items():
            token = cfg[0]
            if token not in cache:
                ext = (lambda Z_: np.concatenate([Z_, PO], -1)) if token == "day+pre" else (lambda Z_: Z_)
                cache[token] = (Inputs(token, ext(Zs), mu_s, y, ok, trd, sig, inner),
                                Inputs(token, ext(Zr), mu_r, y, ok, trd, sig, rows))
            inp_s, inp_r = cache[token]
            o_s, o_r = offs[cfg[4]]
            Mu = Mt & torch.tensor(np.repeat(is50, T, 0), device=DEV) if (len(cfg) > 12 and cfg[12] == "top50") else Mt
            o, eps = np.zeros((T, S)), []
            for seed in (range(cfg[13]) if len(cfg) > 13 else SEEDS):
                _, ep = train(cfg, inp_s, Yt, o_s, Mu, d_in, seed, vdays=d_sel, G=G_s)
                vbest.setdefault(name, []).append(train.last_best)
                ep = max(ep, 1)
                net, _ = train(cfg, inp_r, Yt, o_r, Mu, d_rows, seed, epochs=ep, G=G_r)
                with torch.no_grad():
                    for i in range(0, len(d_te), 64):
                        b = d_te[i:i + 64]
                        o[b.cpu().numpy()] += net(*inp_r.batch(b), o_r[b], Mu[b] if ATT is None else ATT[b]).cpu().numpy()
                eps.append(ep)
            res[name][te] = o[te]
            msg.append(f"{name}: epochs {eps}, val {np.mean(vbest[name]):.5f}")
        if part == "final":
            res[names[-1]][te] = sum(res[k][te] / res[k][te].std() for k in FINAL)
        if part in ("hp", "hp2"):
            pick = min(vbest, key=lambda k_: np.mean(vbest[k_]))
            res[names[-1]][te] = res[pick][te]
            msg.append(f"chosen: {pick}")
        print(f"{Y} done: ridge alpha {a:g}\n  " + "\n  ".join(msg), flush=True)
    print("\n######## the ladder from the E2j iT to the E4h set model, vs the MSE ridge (R2 meaningless except ref, L0)")
    report(res, {k: names[0] for k in names[1:]}, y, subsets, fm)
    if part == "ladder":
        print("\n######## each step vs the step before")
        report({k: res[k] for k in names[1:]}, {names[i + 1]: names[i] for i in range(1, len(names) - 1)}, y,
               subsets[:2], fm)
    elif part in ("ident", "aux", "hp", "hp2", "rank", "top50", "final", "final2"):
        print("\n######## each version vs the first (no identity / lambda 0)")
        report({k: res[k] for k in names[1:]}, {k: names[1] for k in names[2:]}, y, subsets, fm)
    else:
        print("\n######## soft Score vs MSE at the same width, and each width vs d 16 of the same loss")
        refs = {f"d {d_}, soft Score": f"d {d_}, MSE" for d_ in (16, 32, 64, 128)}
        report({k: res[k] for k in names[1:]}, refs, y, subsets[:2], fm)
        refs = {f"d {d_}, {l_}": f"d 16, {l_}" for d_ in (32, 64, 128) for l_ in ("MSE", "soft Score")}
        report({k: res[k] for k in names[1:]}, refs, y, subsets[:2], fm)
    for k in names:
        print(f"share of up calls, {k}: {np.mean(res[k][ev] > 0):.3f}")
    if part == "premask" and (Path(__file__).resolve().parent / "runs" / "e4j_final2_pred.npz").exists():
        z2 = np.load(Path(__file__).resolve().parent / "runs" / "e4j_final2_pred.npz")
        k2 = "E4j final2: d 32, soft Score, 40 seeds, usable-row mask"
        res[k2] = z2["d_32_soft_Score_40_seeds"]
        names.append(k2)
        print("\n######## pre-open mask vs the usable-row mask of final2 (same seeds, paired)")
        report({k: res[k] for k in names}, {names[1]: k2, k2: names[0]}, y, subsets[:2], fm)
    if part in ("final", "final2", "premask"):
        print("\n######## by year (100 stocks): Score of the ridge and each model, and model - ridge")
        dayi = np.repeat(np.arange(T)[:, None], S, 1)
        from stock_walkforward import clustered_mean
        for Yy in EVAL:
            mm = ev & (year == Yy)
            line = f"{Yy}: always down {fm(clustered_mean(-y[mm], dayi[mm]))}"
            sr = np.where(res[names[0]][mm] > 0, 1.0, -1.0)
            for k in names:
                sk = np.where(res[k][mm] > 0, 1.0, -1.0)
                line += f" | {k} {fm(clustered_mean(sk * y[mm], dayi[mm]))}"
                if k != names[0]:
                    line += f" ({fm(clustered_mean((sk - sr) * y[mm], dayi[mm]))})"
            print(line, flush=True)
    np.savez(Path(__file__).resolve().parent / "runs" / ("e4j_pred.npz" if part == "ladder" else f"e4j_{part}_pred.npz"), y=y, ok=ok, ev=ev, year=year, is50=is50,
             **{(k.split(" ")[0] if part == "ladder" else k.replace(" ", "_").replace(",", "").replace(":", "")): v
                for k, v in res.items()})


if __name__ == "__main__":
    main()
