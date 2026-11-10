"""E3c: variants of the E3b relation (demo/stock_us_relation.py) and of its inputs, and the per-stock ridges (TWStock).

Same data, target (g - g_hat_base), walk-forward 2022-2026 and selection rules as demo/stock_us_relation.py main_basket:
L2 weight chosen on Y - 1 with the first seed, steps chosen on Y - 1 (every 25, patience 500, <= 3000), refit on all
years before Y, 3 seeds averaged. Variants (round 1, the form of the weights a_sj):
- signed (E3b reference): softmax(m) * tanh(c)
- sparsemax signed: sparsemax(m) * tanh(c) (exact zeros in the selection)
- tanh, L1 normalized: tanh(c) / (1e-3 + sum_j |tanh(c)|)
- tanh only: tanh(c) (bounded, no normalization)
- signed + L1 on the effective weights gamma_s a_sj (coefficient 1e-3 / 1e-2 times the MSE scale)
- signed d = 4 / d = 16 (dimension of the scores and of the learned identities)
- signed lr 3e-3
- mean of basket + signed and the per-stock ridge on the 46 returns (forecast average)
Later rounds (see demo/EXPERIMENTS.md, E3c): r2 / r2b inputs (mkt series, Japanese / Korean stock gaps, the pre-open
betas), r3 regimes, r4 US after-hours returns, r5 US news, r7 / r9 / r11 / r12 on the target g - g_hat_pre (g_hat with
stock_overnight.preopen_features), r8 the penalty of the pre-open columns in the per-stock ridge.
Usage: python demo/stock_us_variants.py r1 .. r12   (SAVE=1: the per-stock ridge forecasts to demo/runs/e3c_<round>_pred.npz)
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from news_probe import stats  # noqa: E402
from stock_us_relation import (ALPHAS, DEV, SEEDS, WDS, paired, perstock_ridge, prepare)  # noqa: E402
from stock_walkforward import EVAL  # noqa: E402


def sparsemax(z):
    zs, _ = torch.sort(z, dim=-1, descending=True)
    k = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    cs = zs.cumsum(-1)
    supp = 1 + k * zs > cs
    kz = supp.sum(-1, keepdim=True)
    tau = (cs.gather(-1, kz - 1) - 1) / kz.to(z.dtype)
    return torch.clamp(z - tau, min=0)


class Rel(nn.Module):
    def __init__(self, ident, n_us, norm, d=8, extra=None, ctx_dim=0, kb=1, blocks=1):
        super().__init__()
        S = ident.shape[0]
        # blocks > 1: the input is blocks x (n_us / blocks) columns (e.g. returns, after-hours returns, news) that share
        # one relation W; block b > 0 enters with a per-stock signed scale eta_b,s (started at 0)
        self.blocks = blocks
        n_us //= blocks
        self.eta = nn.Parameter(torch.zeros(blocks - 1, S))
        self.norm, self.d = norm, d
        self.register_buffer("ident", torch.as_tensor(ident, dtype=torch.float32))
        self.u = nn.Parameter(torch.zeros(S, d))
        self.v = nn.Parameter(torch.randn(n_us, d) / d ** 0.5)
        n_scores = 2 if norm in ("signed", "sparsemax signed") else 1
        din = ident.shape[1] + d
        self.wq = nn.ModuleList([nn.Linear(din, d, bias=False) for _ in range(n_scores)])
        self.wk = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_scores)])
        self.gamma = nn.Parameter(torch.full((S,), -2.0))
        self.beta = nn.Parameter(torch.zeros(S, kb))

    def relation(self):
        us = torch.cat([self.ident, self.u], -1)
        x = [(q(us) @ k(self.v).T) / self.d ** 0.5 for q, k in zip(self.wq, self.wk)]
        if self.norm == "signed":
            return torch.softmax(x[0], -1) * torch.tanh(x[1])
        if self.norm == "sparsemax signed":
            return sparsemax(x[0]) * torch.tanh(x[1])
        if self.norm == "tanh l1":
            t = torch.tanh(x[0])
            return t / (1e-3 + t.abs().sum(-1, keepdim=True))
        if self.norm == "tanh":
            return torch.tanh(x[0])
        raise ValueError(self.norm)

    def weights(self):
        return F.softplus(self.gamma)[:, None] * self.relation()

    def forward(self, R, B):
        B = B[:, None] if B.dim() == 1 else B
        W = self.weights()
        J = W.shape[1]
        out = R[:, :J] @ W.T + B @ self.beta.T
        for b in range(1, self.blocks):
            out = out + self.eta[b - 1] * (R[:, b * J:(b + 1) * J] @ W.T)
        return out


class RegRel(nn.Module):
    """one relation and basket beta per regime; B (T, 2) = [basket, regime id]."""

    def __init__(self, n_reg, **kw):
        super().__init__()
        self.subs = nn.ModuleList([Rel(**kw) for _ in range(n_reg)])

    def weights(self):
        return torch.cat([m.weights() for m in self.subs], -1)

    def forward(self, R, B):
        reg = B[:, -1].long()
        out = 0
        for k, m in enumerate(self.subs):
            out = out + (reg == k).float()[:, None] * m(R, B[:, :-1])
        return out


def fit(make, R, B, Y, M, tr, va=None, steps=None, seed=0, wd=1e-3, lr=1e-2, l1=0.0):
    torch.manual_seed(seed)
    net = make().to(DEV)
    free = lambda n: n.split(".")[-1] in ("gamma", "beta", "eta")
    opt = torch.optim.Adam([{"params": [p for n, p in net.named_parameters() if not free(n)], "weight_decay": wd},
                            {"params": [p for n, p in net.named_parameters() if free(n)], "weight_decay": 0.0}],
                           lr=lr)
    sc = Y[tr][M[tr]].std()
    best, best_k = None, 0
    if va is not None:
        best = (Y[va][M[va]] ** 2).mean().item()
    n = steps if steps is not None else 3000
    for k in range(1, n + 1):
        loss = (((net(R[tr], B[tr]) - Y[tr]) / sc)[M[tr]] ** 2).mean()
        if l1:
            loss = loss + l1 * net.weights().abs().sum(-1).mean() / sc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if va is not None and k % 25 == 0:
            with torch.no_grad():
                v = ((net(R[va], B[va]) - Y[va])[M[va]] ** 2).mean().item()
            if v < best:
                best, best_k = v, k
            elif k - best_k >= 500:
                break
    return net, best_k, best


def extra_inputs(dates, Rn):
    import pandas as pd
    from stock_poly import ROOT
    pre = np.asarray(dates < "2022-01-01")
    std = lambda M: np.nan_to_num(M / np.nanstd(M[pre], 0))
    cols = []
    for sym in ("BKX", "CL_F", "FXI"):
        f = pd.read_csv(ROOT / "peers" / "mkt" / f"{sym}.csv", parse_dates=["date"]).set_index("date")
        lc = np.log(f["adjclose"]).dropna()
        lc = lc[np.isfinite(lc)]
        idx = np.searchsorted(lc.index.values, dates.values, side="left") - 1
        lev = np.where(idx >= 0, lc.values[np.clip(idx, 0, None)], np.nan)
        cols.append(np.r_[np.nan, np.diff(lev)])
    mkt = std(np.stack(cols, 1))
    gaps = []
    for mk in ("jp", "kr"):
        for p in sorted((ROOT / "peers" / mk).glob("*.csv")):
            f = pd.read_csv(p, parse_dates=["date"]).set_index("date")
            f = f[(f.open > 0) & (f.close > 0)]
            fac = f.adjclose / f.close
            g = np.log(f.open * fac) - np.log((f.close * fac).shift(1))
            g = g[np.abs(g) < 0.3]
            gaps.append(g.reindex(dates).to_numpy())
    asia = std(np.stack(gaps, 1))
    print(f"inputs: mkt {mkt.shape[1]}, Asia gaps {asia.shape[1]} (share of Taiwan days with a gap: "
          f"{np.mean(asia != 0):.2f})", flush=True)
    return {"us": Rn, "us+mkt": np.c_[Rn, mkt], "us+asia": np.c_[Rn, asia], "all": np.c_[Rn, mkt, asia]}


def ah_inputs(dates, syms):
    """after-hours log return of each symbol up to 08:50 Taipei of day t: the close of the last hourly bar that starts
    at or after 16:00 New York on the last US session before t and ends by 00:50 UTC of t, over the open of the
    16:00 bar (the closing-auction print); 0 without such bars."""
    import pandas as pd
    from stock_poly import ROOT
    D = ROOT / "peers" / "alpaca_bars"
    cut = pd.DatetimeIndex(dates).tz_localize("Asia/Taipei") + pd.Timedelta(hours=8, minutes=50)
    cut = cut.tz_convert("UTC")
    out = np.zeros((len(dates), len(syms)))
    for j, sym in enumerate(syms):
        f = pd.read_csv(D / f"{sym}.csv")
        t = pd.to_datetime(f.t, utc=True)
        ny = t.dt.tz_convert("America/New_York")
        day = ny.dt.tz_localize(None).dt.normalize()
        f = f.assign(t=t, day=day.to_numpy(), hr=ny.dt.hour.to_numpy())
        ref = f[f.hr == 16].set_index("day").o                       # closing print of each session
        post = f[(f.hr >= 16)]
        end = (post.t + pd.Timedelta(hours=1)).dt.tz_localize(None).to_numpy()
        for i, c in enumerate(cut):
            # the last session day whose 16:00 bar ended before the cut-off
            k = np.searchsorted(end, c.tz_localize(None).to_datetime64(), side="right") - 1
            if k < 0:
                continue
            d0 = post.day.iloc[k]
            if d0 not in ref.index or (c - pd.Timestamp(d0).tz_localize("America/New_York")).days > 4:
                continue
            out[i, j] = np.log(post.c.iloc[k] / ref.loc[d0])
    return out


def news_inputs(dates, syms):
    """Benzinga news direction of each US peer for day t (demo/news_struct_us.py): the sum over its articles
    published after the last US regular close before 08:50 Taipei of t and before that cut-off of
    P(about the company) x (P(positive) - P(negative)); and the count of those articles."""
    import pandas as pd
    from stock_poly import ROOT
    z = np.load(ROOT / "news" / "alpaca" / "struct.npz", allow_pickle=True)
    sc = z["p_relevance"][:, 0] * (z["p_direction"][:, 0] - z["p_direction"][:, 1])
    a = pd.to_datetime(pd.Series(z["created"]), utc=True)
    cut = (pd.DatetimeIndex(dates).tz_localize("Asia/Taipei") + pd.Timedelta(hours=8, minutes=50)).tz_convert("UTC")
    # US regular closes: 16:00 New York of every session with a 16:00 bar of SPY
    f = pd.read_csv(ROOT / "peers" / "alpaca_bars" / "SPY.csv")
    ny = pd.to_datetime(f.t, utc=True).dt.tz_convert("America/New_York")
    days = ny[ny.dt.hour == 15].dt.normalize().drop_duplicates()
    closes = np.sort((days + pd.Timedelta(hours=16)).dt.tz_convert("UTC").dt.tz_localize(None).to_numpy())
    cu = cut.tz_localize(None).to_numpy()
    av = a.dt.tz_localize(None).to_numpy()
    t = np.searchsorted(cu, av, side="left")                 # first Taiwan day whose cut-off is at or after the article
    ok = t < len(cu)
    last_close = np.full(len(av), np.datetime64("NaT"), "datetime64[ns]")
    k = np.searchsorted(closes, cu[np.minimum(t, len(cu) - 1)], side="right") - 1
    last_close[ok] = closes[np.maximum(k[ok], 0)]
    ok &= av >= last_close
    jx = {s_: i for i, s_ in enumerate(syms)}
    j = np.array([jx.get(s_, -1) for s_ in z["sym"]])
    ok &= j >= 0
    N, C = np.zeros((len(dates), len(syms))), np.zeros((len(dates), len(syms)))
    np.add.at(N, (t[ok], j[ok]), sc[ok])
    np.add.at(C, (t[ok], j[ok]), 1.0)
    print(f"news: {ok.sum()} of {len(av)} scored pairs fall between the last US close and 08:50 Taipei; "
          f"days with any {np.mean(C.sum(1) > 0):.2f}; mean score {sc[ok].mean():+.3f}", flush=True)
    return N, C


VARIANTS = {
    "r1": {
        "basket + signed": dict(norm="signed"),
        "basket + sparsemax signed": dict(norm="sparsemax signed"),
        "basket + tanh, L1 normalized": dict(norm="tanh l1"),
        "basket + tanh only": dict(norm="tanh"),
        "basket + signed, L1 penalty 1e-3": dict(norm="signed", l1=1e-3),
        "basket + signed, L1 penalty 1e-2": dict(norm="signed", l1=1e-2),
        "basket + signed, d 4": dict(norm="signed", d=4),
        "basket + signed, d 16": dict(norm="signed", d=16),
        "basket + signed, lr 3e-3": dict(norm="signed", lr=3e-3),
    },
    # round 2: inputs. mkt = BKX, CL=F, FXI (US session, as the US stocks); asia = the opening gaps of day t of the
    # 12 Japanese and 9 Korean peers (dataset/peers/jp, kr), log(open_t / close_{t-1}) on adjusted prices, their open
    # (09:00 local = 08:00 Taipei) is before the Taiwan open; 0 when that market is closed on day t.
    "r2": {
        "basket + signed": dict(norm="signed"),
        "basket + signed, US + mkt": dict(norm="signed", inputs="us+mkt"),
        "basket + signed, US + Asia gaps": dict(norm="signed", inputs="us+asia"),
        "basket + signed, US + mkt + Asia gaps": dict(norm="signed", inputs="all"),
    },
}
# round 3: a relation (and basket beta) per regime of day t: sign of the basket return; size, |basket| above its
# median before 2022; volatility, the std of the basket over the 20 days before t above its median before 2022.
VARIANTS["r3"] = {
    "basket + signed": dict(norm="signed"),
    "basket + signed, regime sign": dict(norm="signed", regime="sign"),
    "basket + signed, regime size": dict(norm="signed", regime="size"),
    "basket + signed, regime volatility": dict(norm="signed", regime="vol"),
}
VARIANTS["r3"] = {
    "pre-open betas + signed": dict(norm="signed", bas="pre"),
    "pre-open betas + signed, regime sign": dict(norm="signed", regime="sign", bas="pre"),
    "pre-open betas + signed, regime size": dict(norm="signed", regime="size", bas="pre"),
    "pre-open betas + signed, regime volatility": dict(norm="signed", regime="vol", bas="pre"),
    "pre-open betas + signed, lr 3e-3, L1 1e-2": dict(norm="signed", bas="pre", lr=3e-3, l1=1e-2),
}
# round 2b: the pre-open market information missing from the 52 base features: open gaps of the Nikkei 225 and
# KOSPI (08:00 Taipei) and of the TAIEX futures (08:45), from demo/stock_overnight.py; with a beta per Taiwan stock
# ("pre-open betas" = basket + n225 + ks11 + futures gap). The relation on top of those, on the 46 US stocks and on
# US + the 21 Japanese / Korean stock gaps; controls: the Asian stock gaps of another day of the same year, and those
# of day t - 1.
VARIANTS["r2b"] = {
    "pre-open betas + signed on 46 US": dict(norm="signed", bas="pre"),
    "pre-open betas + signed on US + Asia gaps": dict(norm="signed", inputs="us+asia", bas="pre"),
    "pre-open betas + signed on US + Asia gaps, Asia of another day": dict(norm="signed", inputs="us+asia shuf",
                                                                           bas="pre"),
    "pre-open betas + signed on US + Asia gaps, Asia of t - 1": dict(norm="signed", inputs="us+asia lag", bas="pre"),
    "basket + signed on US + Asia gaps": dict(norm="signed", inputs="us+asia"),
}
# round 4: US after-hours returns to 08:50 Taipei (demo/fetch_peers.py alpaca_bars). pre-open + AH = the pre-open betas
# plus the after-hours returns of SPY, QQQ, SOXX per stock; relation inputs: the 46 close-to-close returns (R), R and
# the 46 after-hours returns as separate inputs, or their sum (the return from the close of t - 1 to 08:50 of t);
# control: the after-hours returns of another day of the same year.
VARIANTS["r4"] = {
    "pre-open + signed on R": dict(norm="signed", bas="pre"),
    "pre-open + AH ETFs + signed on R": dict(norm="signed", bas="pre+ah"),
    "pre-open + AH ETFs + signed on R, AH": dict(norm="signed", bas="pre+ah", inputs="us+ah"),
    "pre-open + AH ETFs + signed on R + AH (sum)": dict(norm="signed", bas="pre+ah", inputs="us sum ah"),
    "pre-open + AH ETFs + signed on R, AH of another day": dict(norm="signed", bas="pre+ah", inputs="us+ah shuf"),
}
# round 5: the news direction N (46) of the US peers (news_inputs). Base: pre-open + AH ETFs, the relation on the
# returns and after-hours returns (shared weights, eta per stock). News: shared with the same W (eta per stock),
# or a ridge per stock; control: the news of another day of the same year.
VARIANTS["r5"] = {
    "pre-open + signed on R": dict(norm="signed", bas="pre"),
    "pre-open + signed on R, news shared": dict(norm="signed", bas="pre", inputs="r|n", blocks=2),
    "pre-open + signed on R, news shared, news of another day": dict(norm="signed", bas="pre", inputs="r|n shuf",
                                                                     blocks=2),
    "pre-open + signed on R, news separate": dict(norm="signed", bas="pre", inputs="r+n"),
    "pre-open + signed on R, AH clipped shared": dict(norm="signed", bas="pre", inputs="r|ahc", blocks=2),
    "pre-open + signed on R, AH clipped, news shared": dict(norm="signed", bas="pre", inputs="r|ahc|n", blocks=3),
}
# round 7: the relation on the target g - g_hat_pre (base ridge with the pre-open features, stock_overnight.preopen_features)
VARIANTS["r7"] = {
    "basket + signed": dict(norm="signed"),
    "pre-open + signed on R": dict(norm="signed", bas="pre"),
    "signed on R (no betas)": dict(norm="signed", bas="none"),
}
# round 8: is the edge of the relation over the per-stock ridge (r2b, r3) only the shrinkage of the pre-open betas,
# which the ridge penalizes with the same alpha as the 46 US coefficients while the network leaves them free?
# The ridge with the pre-open columns scaled by 10 / 100 (penalty / 100, / 10^4), on the original target.
VARIANTS["r8"] = {
    "pre-open + signed on R": dict(norm="signed", bas="pre"),
}
# round 9 (target of r7): sharing the US coefficients across stocks instead of a learned relation: one ridge per
# industry (stocks of an industry pooled), and the per-stock ridge shrunk toward it (forecast average and the
# ridge on the residual of the industry forecast); pre-open columns x10 (penalty / 100) everywhere.
VARIANTS["r9"] = {}
# round 11 (target of r7): ridge-only checks of the inputs on the new target: the US returns of the day before as well
# (a delayed response), the Japanese / Korean stock gaps, the mkt series (BKX, CL=F, FXI).
VARIANTS["r11"] = {}
# round 12: all of them together, and the stocks' own TWSE peers? no: pre-open + R + mkt + Asia gaps, with the
# signed relation on the same inputs for comparison
VARIANTS["r12"] = {"pre-open + signed on R, mkt, Asia gaps": dict(norm="signed", bas="pre", inputs="all")}
RIDGES = {"r2": ("us+mkt", "us+asia", "all")}


def main(rnd):
    D_ = prepare()
    dates, T, S, stocks, top50, year, ok, tgt, Rn, J, us, IND, PM, has_peer = (D_[k] for k in (
        "dates", "T", "S", "stocks", "top50", "year", "ok", "tgt", "Rn", "J", "us", "IND", "PM", "has_peer"))
    yrs = dates.year.to_numpy()
    if rnd in ("r7", "r9", "r11", "r12"):   # target g - g_hat_pre: the pooled ridge on the 52 base features + 4 pre-open features
        from stock_overnight import preopen_features
        from stock_news_gap import walk
        from stock_poly import load
        from stock_walkforward import Ridge
        _, r_, traded_, gap_, _ = load()
        X2 = np.concatenate([D_["X"], preopen_features(dates, gap_, traded_)], -1)
        g_ = gap_.to_numpy()
        gh, _ = walk(X2, g_, ok, year)
        for yy in np.unique(year[ok & (year < EVAL[0])]):
            fit_, at = ok & (year != yy) & (year < EVAL[0]), ok & (year == yy)
            m_ = Ridge(X2[fit_], g_[fit_])
            gh[at] = m_.predict(X2[at], m_.weights(1e4))
        tgt = np.where(ok, g_ - gh, 0.0)
        print(f"r7 target g - g_hat_pre: std {tgt[ok].std():.5f}", flush=True)
    basket = Rn.mean(1)
    basket = basket / basket[np.asarray(dates < "2022-01-01")].std()
    B1 = torch.tensor(basket, dtype=torch.float32, device=DEV)
    Yt = torch.tensor(tgt, dtype=torch.float32, device=DEV)
    Mt = torch.tensor(ok, device=DEV)
    rows = lambda m: torch.tensor(np.nonzero(m)[0], device=DEV)
    FB = np.repeat(basket[:, None, None], S, 1)
    FA = np.repeat(Rn[:, None, :], S, 1)
    RID = {"basket beta per stock": FB, "ridge per stock on 46": FA}
    INP = extra_inputs(dates, Rn)
    for k in RIDGES.get(rnd, ()):
        RID[f"ridge per stock on {k} ({INP[k].shape[1]})"] = np.repeat(np.c_[INP[k], basket][:, None, :], S, 1)
    import pandas as pd
    pre = np.asarray(dates < "2022-01-01")
    vol = pd.Series(basket).rolling(20, min_periods=5).std().shift(1).to_numpy()
    REG = {"sign": (basket > 0).astype(float), "size": (np.abs(basket) > np.median(np.abs(basket[pre]))).astype(float),
           "vol": (np.nan_to_num(vol) > np.nanmedian(vol[pre])).astype(float)}
    RID_R3 = rnd == "r3"
    BAS = {}
    from stock_poly import ROOT as ROOT_
    REGT = {k: torch.tensor(np.c_[basket, z], dtype=torch.float32, device=DEV) for k, z in REG.items()}
    if True:
        from stock_overnight import taifex_features, yahoo_features
        closes = pd.read_csv(ROOT_ / "market" / "2330.csv", parse_dates=["date"]).set_index("date")["etl:adj_close"]
        yh = yahoo_features(dates, closes.reindex(dates).to_numpy())
        tx = taifex_features(dates)
        PO = np.stack([yh["n225_gap"].to_numpy(), yh["ks11_gap"].to_numpy(), tx["fut_gap"].to_numpy()], 1)
        PO = np.nan_to_num(PO / np.nanstd(PO[pre], 0))
        print(f"pre-open: corr of the daily mean target with n225 / ks11 / futures gap: " + ", ".join(
            f"{np.corrcoef(PO[:, i], np.nanmean(np.where(ok, tgt, np.nan), 1))[0, 1]:+.3f}" for i in range(3)), flush=True)
        PB = np.c_[basket, PO]
        RID["pre-open betas per stock (4)"] = np.repeat(PB[:, None, :], S, 1)
    if rnd == "r2b":
        RID["index-open betas (basket + n225 + ks11) per stock"] = np.repeat(PB[:, None, :3], S, 1)
        RID["pre-open betas + ridge on 46 US per stock"] = np.repeat(np.c_[PB, Rn][:, None, :], S, 1)
        RID["pre-open betas + ridge on US + Asia gaps per stock"] = np.repeat(np.c_[PB, INP["us+asia"]][:, None, :], S, 1)
    if True:
        A_ = INP["us+asia"][:, Rn.shape[1]:]
        rng = np.random.default_rng(0)
        perm = np.arange(T)
        for yy in np.unique(yrs):
            ii = np.nonzero(yrs == yy)[0]
            perm[ii] = rng.permutation(ii)
        INP["us+asia shuf"] = np.c_[Rn, A_[perm]]
        INP["us+asia lag"] = np.c_[Rn, np.r_[np.zeros((1, A_.shape[1])), A_[:-1]]]
        BAS["pre"] = torch.tensor(PB, dtype=torch.float32, device=DEV)
        BAS["none"] = torch.zeros((T, 1), dtype=torch.float32, device=DEV)
    if rnd in ("r4", "r5"):
        AH = ah_inputs(dates, list(D_["us"]) + ["SPY", "QQQ", "SOXX"])
        # scale: the std of the non-zero values before 2022, or of all non-zero values for stocks listed later
        sd = np.array([np.std(c[pre & (c != 0)]) if (pre & (c != 0)).sum() > 50 else np.std(c[c != 0]) for c in AH.T])
        AH = AH / sd
        print(f"after-hours: share of days with a value {np.mean(AH[:, :J] != 0):.2f}; corr of the SPY after-hours "
              f"return with the mean target {np.corrcoef(AH[:, J], np.nan_to_num(np.nanmean(np.where(ok, tgt, np.nan), 1)))[0, 1]:+.3f}",
              flush=True)
        AHu, AHm = AH[:, :J], AH[:, J:]
        PBA = np.c_[PB, AHm]
        BAS["pre+ah"] = torch.tensor(PBA, dtype=torch.float32, device=DEV)
        INP["us+ah"] = np.c_[Rn, AHu]
        INP["us sum ah"] = Rn + AHu
        INP["us+ah shuf"] = np.c_[Rn, AHu[perm]]
        RID["pre-open + AH ETFs per stock (7)"] = np.repeat(PBA[:, None, :], S, 1)
        RID["pre-open + ridge on R per stock"] = np.repeat(np.c_[PB, Rn][:, None, :], S, 1)
        RID["pre-open + AH ETFs + ridge on R, AH per stock"] = np.repeat(np.c_[PBA, Rn, AHu][:, None, :], S, 1)
        INP["r|ah"] = np.c_[Rn, AHu]
    if rnd == "r5":
        N, C = news_inputs(dates, list(D_["us"]))
        N = N / np.std(N[pre & (np.abs(N).sum(1) > 0)])
        AHc = np.clip(AHu, -5, 5)
        INP["r|n"] = np.c_[Rn, N]
        INP["r|n shuf"] = np.c_[Rn, N[perm]]
        INP["r+n"] = np.c_[Rn, N]
        INP["r|ahc"] = np.c_[Rn, AHc]
        INP["r|ahc|n"] = np.c_[Rn, AHc, N]
        RID["pre-open + ridge on R, news per stock"] = np.repeat(np.c_[PB, Rn, N][:, None, :], S, 1)
        RID["pre-open + ridge on R, AH clipped per stock"] = np.repeat(np.c_[PB, Rn, AHc][:, None, :], S, 1)
        print(f"news: share of (day, US peer) with news {np.mean(N != 0):.3f}; |N| > 3: {np.mean(np.abs(N) > 3):.4f}",
              flush=True)
        for k_ in [k_ for k_ in RID if k_.startswith(("basket", "ridge per stock on 46", "pre-open betas"))]:
            pass
    if rnd == "r8":
        RID["pre-open + ridge on R per stock"] = np.repeat(np.c_[PB, Rn][:, None, :], S, 1)
        RID["pre-open x10 + ridge on R per stock"] = np.repeat(np.c_[10 * PB, Rn][:, None, :], S, 1)
        RID["pre-open x100 + ridge on R per stock"] = np.repeat(np.c_[100 * PB, Rn][:, None, :], S, 1)
    if rnd == "r9":
        RID["pre-open x10 + ridge on R per stock"] = np.repeat(np.c_[10 * PB, Rn][:, None, :], S, 1)
        RID["pre-open x10 per stock"] = np.repeat(10 * PB[:, None, :], S, 1)
    if rnd == "r12":
        RID["pre-open + ridge on R per stock"] = np.repeat(np.c_[PB, Rn][:, None, :], S, 1)
        RID["pre-open + ridge on R, mkt, Asia gaps per stock"] = np.repeat(np.c_[PB, INP["all"]][:, None, :], S, 1)
        RID["pre-open + ridge on R, mkt, Asia gaps of another day"] = np.repeat(
            np.c_[PB, INP["us+mkt"], INP["us+asia shuf"][:, J:]][:, None, :], S, 1)
    if rnd == "r11":
        R1 = np.r_[np.zeros((1, J)), Rn[:-1]]
        RID["pre-open + ridge on R per stock"] = np.repeat(np.c_[PB, Rn][:, None, :], S, 1)
        RID["pre-open + ridge on R, R of t - 1 per stock"] = np.repeat(np.c_[PB, Rn, R1][:, None, :], S, 1)
        RID["pre-open + ridge on R, Asia stock gaps per stock"] = np.repeat(np.c_[PB, INP["us+asia"]][:, None, :], S, 1)
        RID["pre-open + ridge on R, mkt per stock"] = np.repeat(np.c_[PB, INP["us+mkt"]][:, None, :], S, 1)
        RID["pre-open + ridge on R, Asia gaps of another day"] = np.repeat(np.c_[PB, INP["us+asia shuf"]][:, None, :], S, 1)
    if rnd == "r7":
        RID["pre-open + ridge on R per stock"] = np.repeat(np.c_[PB, Rn][:, None, :], S, 1)
    if RID_R3:
        RID["pre-open betas + ridge on 46 US per stock"] = np.repeat(np.c_[PB, Rn][:, None, :], S, 1)
        for k, z in REG.items():
            z_ = z[:, None]
            RID[f"pre-open betas per stock per regime {k}"] = np.repeat(np.c_[PB * z_, PB * (1 - z_)][:, None, :], S, 1)
    INPT = {k: torch.tensor(v, dtype=torch.float32, device=DEV) for k, v in INP.items()}
    V = VARIANTS[rnd]
    pred = {k: np.full((T, S), np.nan) for k in list(RID) + list(V)}
    nets = {}
    for Y in EVAL:
        te = yrs == Y
        line = []
        for name, F_ in RID.items():
            pred[name][te], a = perstock_ridge(F_, tgt, ok, yrs, Y)
            line.append(f"{name} alpha {a:g}")
        tr_s, va_s, tr_r = yrs < Y - 1, yrs == Y - 1, yrs < Y
        for name, o in V.items():
            o = dict(o)
            lr, l1 = o.pop("lr", 1e-2), o.pop("l1", 0.0)
            Rt = INPT[o.pop("inputs", "us")]
            reg = o.pop("regime", None)
            bas = o.pop("bas", None)
            Bt = BAS[bas] if bas else B1
            Bt = Bt[:, None] if Bt.dim() == 1 else Bt
            if reg:
                Bt = torch.cat([Bt, REGT[reg][:, 1:]], 1)
            o["kb"] = Bt.shape[1] - (1 if reg else 0)
            make = (lambda: RegRel(2, ident=IND, n_us=Rt.shape[1], **o)) if reg else (lambda: Rel(IND, Rt.shape[1], **o))
            kw = dict(lr=lr, l1=l1)
            wd = min(WDS, key=lambda w: fit(make, Rt, Bt, Yt, Mt, rows(tr_s), va=rows(va_s), seed=SEEDS[0], wd=w,
                                            **kw)[2])
            ps, ks = [], []
            for seed in SEEDS:
                _, k, _ = fit(make, Rt, Bt, Yt, Mt, rows(tr_s), va=rows(va_s), seed=seed, wd=wd, **kw)
                ks.append(k)
                if k == 0 and rnd in ("r7", "r12"):   # no step improved on Y - 1: forecast 0 (as in E3a)
                    ps.append(np.zeros((te.sum(), S)))
                    continue
                net, _, _ = fit(make, Rt, Bt, Yt, Mt, rows(tr_r), steps=max(k, 25), seed=seed, wd=wd, **kw)
                with torch.no_grad():
                    ps.append(net(Rt[rows(te)], Bt[rows(te)]).cpu().numpy())
                if Y == EVAL[-1] and seed == SEEDS[0] and k > 0:
                    nets[name] = net
            pred[name][te] = np.mean(ps, 0)
            line.append(f"{name} wd {wd:g} steps {ks}")
        print(f"{Y}: " + "; ".join(line), flush=True)
    if rnd == "r9":
        ind = D_["ind"]
        groups = {"industry": ind, "all stocks": np.zeros(S, int)}
        XG = np.c_[10 * PB, Rn]
        k_ = XG.shape[1]
        for gname, lab in groups.items():
            P_ = np.full((T, S), np.nan)
            for Y in EVAL:
                tr_s, va_s, tr_r, te = yrs < Y - 1, yrs == Y - 1, yrs < Y, yrs == Y
                def fitg(trm, a):
                    out = {}
                    for c in np.unique(lab):
                        ss = np.nonzero(lab == c)[0]
                        Xs = np.concatenate([XG[trm][ok[trm, s_]] for s_ in ss])
                        ys = np.concatenate([tgt[trm][ok[trm, s_], s_] for s_ in ss])
                        mu = Xs.mean(0)
                        b = np.linalg.solve((Xs - mu).T @ (Xs - mu) + a * len(Xs) * np.eye(k_), (Xs - mu).T @ (ys - ys.mean()))
                        out[c] = (ys.mean() - mu @ b, b)
                    return out
                pr = lambda C, rows_: np.stack([C[lab[s_]][0] + XG[rows_] @ C[lab[s_]][1] for s_ in range(S)], 1)
                a = min(ALPHAS, key=lambda a: np.mean((pr(fitg(tr_s, a), va_s) - tgt[va_s])[ok[va_s]] ** 2))
                P_[te] = pr(fitg(tr_r, a), te)
            pred[f"{gname}-shared ridge on pre-open x10 + R"] = P_
            pred[f"mean of per-stock and {gname}-shared"] = 0.5 * (pred["pre-open x10 + ridge on R per stock"] + P_)
        print(f"industries: {len(np.unique(ind))}", flush=True)
    if "pre-open betas + signed" in pred:
        pred["mean of pre-open betas + signed and pre-open betas + ridge on 46"] = 0.5 * (
            pred["pre-open betas + signed"] + pred["pre-open betas + ridge on 46 US per stock"])
    if "basket + signed" in pred:
        pred["mean of basket + signed and ridge per stock on 46"] = 0.5 * (pred["basket + signed"]
                                                                           + pred["ridge per stock on 46"])

    if os.environ.get("SAVE"):
        np.savez(Path(__file__).resolve().parent / "runs" / f"e3c_{rnd}_pred.npz", tgt=tgt, ok=ok,
                 **{k.replace(" ", "_").replace("/", "_"): v for k, v in pred.items() if "ridge on R per stock" in k})
    ev = ok & np.isin(year, EVAL)
    is50 = np.array([s in top50 for s in stocks])[None, :] & ev
    means = np.zeros((T, S))
    for Y in EVAL:
        means[yrs == Y] = tgt[ok & (year < Y)].mean()
    print("\n== target g - g_hat (price), walk-forward 2022-2026, R2 against the training mean; "
          "paired differences against the basket beta and against basket + signed", flush=True)
    for name, P in pred.items():
        out = []
        for lab, m in (("100 stocks", ev), ("50-stock subset", is50)):
            dd = np.nonzero(m)[0]
            r2, r2s, c, _ = stats(P[m], tgt[m], means[m], dd, np.random.default_rng(0))
            dm, ds = paired(P[m], pred["basket beta per stock"][m], tgt[m], dd, np.random.default_rng(0), means[m])
            s_ = ""
            ref2 = next((k for k in ("pre-open + signed on R", "pre-open x10 + ridge on R per stock", "pre-open + ridge on R per stock") if k in pred and rnd in ("r5", "r7", "r8", "r9", "r11", "r12")), None) or next((k for k in ("pre-open + AH ETFs per stock (7)", "pre-open betas per stock (4)", "basket + signed")
                         if k in pred), None)
            if ref2:
                dm2, ds2 = paired(P[m], pred[ref2][m], tgt[m], dd, np.random.default_rng(0), means[m])
                s_ = f", vs {ref2[:12]} {dm2:+.4f} ± {ds2:.4f}"
            out.append(f"{lab}: R2 {r2:+.4f} ± {r2s:.4f} corr {c:+.3f}, vs basket {dm:+.4f} ± {ds:.4f}{s_}")
        print(f"{name:52s} " + " | ".join(out), flush=True)

    print("\n== fitted before 2026 (first seed): effective weights w_sj = gamma_s a_sj", flush=True)
    for name, net in nets.items():
        with torch.no_grad():
            A = net.weights().cpu().numpy()
        if isinstance(net, RegRel):
            A0, A1 = A[:, :J], A[:, J:]
            print(f"{name:52s} regime 1 vs 0: corr of the weights {np.corrcoef(A0.ravel(), A1.ravel())[0, 1]:+.3f}, "
                  f"sum |w| {np.abs(A1).sum() / np.abs(A0).sum():.2f}x; basket beta mean "
                  f"{net.subs[1].beta[:, 0].mean().item():+.3f} vs {net.subs[0].beta[:, 0].mean().item():+.3f}", flush=True)
            A = A0
        if A.shape[1] > J:
            print(f"{name:52s} share of sum |w| on the inputs beyond the 46 US stocks: "
                  f"{np.abs(A[:, J:]).sum() / np.abs(A).sum():.1%} ({A.shape[1] - J} of {A.shape[1]} inputs)", flush=True)
            A = A[:, :J]
        absA = np.abs(A)
        eff = absA.sum(1) ** 2 / np.maximum((A ** 2).sum(1), 1e-30)
        share = (absA * PM).sum(1)[has_peer] / np.maximum(absA.sum(1)[has_peer], 1e-30)
        print(f"{name:52s} effective number of US stocks median {np.median(eff):5.1f} of {J}; exact zeros "
              f"{np.mean(A == 0):.1%}; share on hand-made peers {share.mean():.3f} "
              f"(uniform {(PM.sum(1)[has_peer] / J).mean():.3f}); negative {absA[A < 0].sum() / absA.sum():.1%} of sum |w|",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
