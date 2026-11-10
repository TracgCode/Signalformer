"""E2g: how much of the opening gap the representations of a news item carry, item-level probes (TWStock).

Sample: the stock-days (100 stocks, 2020-04 .. 2026-08) with exactly one own overnight item (MOPS announcement or
cnyes news, after 13:30 of the previous trading day and before 08:50), so every target belongs to one text. Targets of
day t:
- g - g_hat: the stock's opening gap log(o_t / c_{t-1}) minus its walk-forward ridge forecast from the 52 price
  features (demo/stock_news_gap.py), i.e. the part of the gap that the price does not explain (news should be in it);
- |g| - |g|_hat: the size of the gap minus its ridge forecast from the stock's mean |g| over 5, 20 and 60 days and
  |g_hat| (does the text tell how much the price will move, whatever the direction);
- e = r - mu: the residual of the open-to-close return after the base ridge (the forecast target).
Representations of the item (all fitted on the items before 2022 where they are learned):
  LLM s (P(pos) - P(neg)), the 3 LLM probabilities; bge-m3 PCA 16 / PCA 64 / all 1024 dims; TF-IDF of the character
  1- and 2-grams of the title (the 4000 most frequent in training); the MOPS clause (款次) dummies and the source;
  all together.
Model: ridge (on the GPU), walk-forward: for year Y fitted on the items before Y, alpha chosen on Y - 1 from the fit
before Y - 1. Reported pooled 2022 - 2026: out-of-sample R2 (against the training mean), the correlation of the
forecast with the target, and their bootstrap standard errors over days, per source too.

    python demo/news_probe.py [struct] [pre [plusus]]
      struct: the representations of demo/news_struct.py (E2h);
      pre: g_hat also from the 4 pre-open features of stock_overnight.preopen_features (E3c r6);
      plusus: the target also after removing the per-stock ridge on the pre-open + 46 US returns (E3c r10; needs
        demo/runs/e3c_r7_pred.npz from SAVE=1 python demo/stock_us_variants.py r7)
"""

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from news_score import items  # noqa: E402
from stock_news_gap import walk  # noqa: E402
from stock_poly import ROOT, load  # noqa: E402
from stock_recurrent import base_features  # noqa: E402
from stock_walkforward import EVAL, FIRST  # noqa: E402

DEV = "cuda"
ALPHAS = 10.0 ** np.arange(-2, 7)


def ridge_wf(Z, y, yr):
    """walk-forward ridge forecasts for the items of EVAL years (GPU, closed form on standardized Z)."""
    Zt = torch.tensor(Z, dtype=torch.float64, device=DEV)
    yt = torch.tensor(y, dtype=torch.float64, device=DEV)
    out = np.full(len(y), np.nan)

    def fit(m):
        mu, sd = Zt[m].mean(0), Zt[m].std(0) + 1e-9
        X = (Zt[m] - mu) / sd
        c = yt[m].mean()
        G, b = X.T @ X, X.T @ (yt[m] - c)
        I = torch.eye(G.shape[0], dtype=G.dtype, device=DEV)
        return lambda a, at: ((Zt[at] - mu) / sd) @ torch.linalg.solve(G + a * len(X) * I, b) + c

    for Y in EVAL:
        tr_s, va, tr, te = [torch.tensor(v, device=DEV) for v in (yr < Y - 1, yr == Y - 1, yr < Y, yr == Y)]
        f = fit(tr_s)
        a = min(ALPHAS, key=lambda a: (f(a, va) - yt[va]).pow(2).mean().item())
        out[te.cpu().numpy()] = fit(tr)(a, te).cpu().numpy()
    return out


def stats(p, y, base_mean, day, rng):
    def one(idx):
        yy, pp = y[idx], p[idx]
        return 1 - np.mean((yy - pp) ** 2) / np.mean((yy - base_mean[idx]) ** 2), np.corrcoef(pp, yy)[0, 1]
    r2, c = one(np.arange(len(y)))
    days, inv = np.unique(day, return_inverse=True)
    groups = [np.nonzero(inv == k)[0] for k in range(len(days))]
    bs = []
    for _ in range(200):
        pick = rng.integers(0, len(days), len(days))
        bs.append(one(np.concatenate([groups[k] for k in pick])))
    bs = np.array(bs)
    return r2, bs[:, 0].std(), c, bs[:, 1].std()


def main() -> None:
    dates, r, traded, gap, known = load()
    T, S = r.shape
    stocks = json.loads((ROOT / "info.json").read_text())["Top100@Y"]
    X, y = base_features(dates, r, traded, gap, known, stocks)
    if "pre" in sys.argv:  # g_hat with the pre-open features (stock_overnight.preopen_features)
        from stock_overnight import preopen_features
        X = np.concatenate([X, preopen_features(dates, gap, traded)], -1)
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    g = gap.to_numpy()
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y) & np.isfinite(g)
    ghat, _ = walk(X, g, ok, year)                       # walk-forward on EVAL years only
    mu, _ = walk(X, y, ok, year)
    # in-sample-free forecasts for the training years too: leave one year out with a fixed alpha
    from stock_walkforward import Ridge
    for tgt, arr in ((g, ghat), (y, mu)):
        for yy in np.unique(year[ok & (year < EVAL[0])]):
            fit_, at = ok & (year != yy) & (year < EVAL[0]), ok & (year == yy)
            m = Ridge(X[fit_], tgt[fit_])
            arr[at] = m.predict(X[at], m.weights(1e4))
    ag = np.abs(np.nan_to_num(np.where(traded.to_numpy(), g, np.nan)))
    roll = lambda w: pd.DataFrame(ag).rolling(w, min_periods=1).mean().shift(1).to_numpy()
    VOL = np.stack([roll(5), roll(20), roll(60), np.abs(np.nan_to_num(ghat))], -1)

    # items, aligned with dataset/news/scores/items.npz (same order as news_score.items())
    f = items()
    z = np.load(ROOT / "news" / "scores" / "items.npz")
    assert len(f) == len(z["ts"]) and (f.stock.to_numpy().astype(str) == z["stock"]).all()
    E = np.load(ROOT / "news" / "scores" / "emb.npy")[z["emb_id"]].astype(np.float32)
    ts = pd.DatetimeIndex(z["ts"])
    cut = (dates + pd.Timedelta(hours=8, minutes=50)).to_numpy()
    d = np.searchsorted(cut, ts.to_numpy(), "left")
    keep = (d > 0) & (d < T)
    prev_close = np.r_[np.datetime64("NaT", "s"), (dates[:-1] + pd.Timedelta(hours=13, minutes=30)).to_numpy()]
    over = np.zeros(len(d), bool)
    over[keep] = ts.to_numpy()[keep] > prev_close[d[keep]]
    sidx = {c: j for j, c in enumerate(stocks)}
    j = np.array([sidx.get(c, -1) for c in z["stock"]])
    own = keep & over & (j >= 0)
    n_own = Counter(zip(d[own], j[own]))
    single = own & np.array([n_own.get((a, b), 0) == 1 for a, b in zip(d, j)])
    single &= ok[np.clip(d, 0, T - 1), np.clip(j, 0, S - 1)]
    idx = np.nonzero(single)[0]
    dd, jj = d[idx], j[idx]
    yr = dates.year.to_numpy()[dd]
    print(f"{len(idx)} single-item stock-days ({np.mean(z['src'][idx] == 'mops'):.0%} MOPS); "
          f"{np.sum(yr >= EVAL[0])} in 2022-2026", flush=True)

    # clause of the MOPS items
    clause = {}
    for fn in sorted((ROOT / "news" / "mops").glob("*.jsonl.gz")):
        for rr in map(json.loads, gzip.open(fn, "rt", encoding="utf-8")):
            if rr.get("date") and rr.get("time") and rr["date"] != "None":
                clause[(rr["co"], pd.Timestamp(rr["date"] + rr["time"].zfill(6)))] = rr.get("clause") or "?"
    src = z["src"][idx]
    cl = np.array([clause.get((z["stock"][i], ts[i]), "news") if src[k] == "mops" else "news"
                   for k, i in enumerate(idx)])
    pre = yr < 2022
    top = [c for c, n in Counter(cl[pre]).most_common(40) if n >= 30]
    CL = np.stack([cl == c for c in top] + [src == "mops"], 1).astype(np.float32)

    # representations
    P = z["p"][idx]
    s = (P[:, 0] - P[:, 2])[:, None]
    Ei = E[idx]
    tr_items = Ei[pre]
    c0 = tr_items.mean(0)
    _, _, Vt = np.linalg.svd(tr_items - c0, full_matrices=False)
    PC = (Ei - c0) @ Vt[:64].T
    titles = [f.text.iat[i].split("\n", 1)[0] for i in idx]
    grams = lambda t: [t[k:k + n] for n in (1, 2) for k in range(len(t) - n + 1)]
    df = Counter(gm for t, p_ in zip(titles, pre) if p_ for gm in set(grams(t)))
    vocab = {gm: k for k, (gm, _) in enumerate(df.most_common(4000))}
    npre = pre.sum()
    idf = np.log((1 + npre) / (1 + np.array([df[gm] for gm in vocab]))) + 1
    TF = np.zeros((len(idx), len(vocab)), np.float32)
    for k, t in enumerate(titles):
        for gm, n in Counter(grams(t)).items():
            if gm in vocab:
                TF[k, vocab[gm]] = n
    TF *= idf
    TF /= np.linalg.norm(TF, axis=1, keepdims=True) + 1e-9
    REPS = {"LLM s": s, "LLM 3 probs": P, "bge PCA16": PC[:, :16], "bge PCA64": PC, "bge 1024": Ei,
            "title TF-IDF 4000": TF, "clause + source": CL,
            "all": np.concatenate([P, Ei, TF, CL], 1)}
    if "struct" in sys.argv:  # E2h: the typed fields and summaries of demo/news_struct.py
        from news_struct import FIELDS, OUT, load_struct
        st = load_struct()
        F24 = np.concatenate([st[f"p_{k}"][idx] for k, _, _ in FIELDS], 1).astype(np.float64)
        Es = np.load(OUT / "summary_emb.npy")[idx].astype(np.float32)
        cs = Es[pre].mean(0)
        _, _, Vs = np.linalg.svd(Es[pre] - cs, full_matrices=False)
        PS = (Es - cs) @ Vs[:64].T
        REPS = {"Qwen2.5 3 probs (E2)": P, "text bge PCA16 (E2)": PC[:, :16],
                "Qwen3.5 direction 3": st["p_direction"][idx], "Qwen3.5 size 3": st["p_size"][idx],
                "Qwen3.5 all fields 24": F24, "summary bge PCA16": PS[:, :16], "summary bge PCA64": PS,
                "fields 24 + summary PCA16": np.concatenate([F24, PS[:, :16]], 1),
                "Qwen2.5 3 + fields 24": np.concatenate([P, F24], 1),
                "fields 24 + summary 16 + text 16": np.concatenate([F24, PS[:, :16], PC[:, :16]], 1)}

    targets = {"g - g_hat (price)": (g - ghat)[dd, jj], "e = r - mu": (y - mu)[dd, jj]}
    if "plusus" in sys.argv:  # with pre: also after removing the per-stock ridge on the pre-open + 46 US returns
        zp = np.load(Path(__file__).resolve().parent / "runs" / "e3c_r7_pred.npz")   # stock_us_variants.py r7, SAVE=1
        usp = np.nan_to_num(zp["pre-open_+_ridge_on_R_per_stock"])     # 0 before 2022 (forecasts on 2022-2026 only)
        t0_ = (g - ghat)[dd, jj]
        targets = {"g - g_hat (price)": t0_, "g - g_hat - US ridge": t0_ - usp[dd, jj]}
    vol_base = VOL[dd, jj]
    yabs = ag[dd, jj]
    rng = np.random.default_rng(0)
    ev = yr >= EVAL[0]
    day = dd[ev]
    for tname, tv in list(targets.items()) + ([] if "plusus" in sys.argv else [("|g| - |g|_hat (vol)", None)]):
        print(f"\n== target {tname}", flush=True)
        if tv is None:  # |g|: the representations on top of the volatility baseline
            base_p = ridge_wf(vol_base, yabs, yr)
            tv = yabs - np.where(np.isnan(base_p), 0, base_p)
            tv[~ev] = (yabs - ridge_wf_insample(vol_base, yabs, pre))[~ev]
        means = {Y: tv[yr < Y].mean() for Y in EVAL}
        bm = np.array([means[Y] for Y in yr[ev]])
        for name, Z in REPS.items():
            p = ridge_wf(Z, tv, yr)[ev]
            r2, r2s, c, cs = stats(p, tv[ev], bm, day, rng)
            per = []
            for so in ("mops", "cnyes"):
                m = src[ev] == so
                r2b = 1 - np.mean((tv[ev][m] - p[m]) ** 2) / np.mean((tv[ev][m] - bm[m]) ** 2)
                per.append(f"{so} R2 {r2b:+.4f} corr {np.corrcoef(p[m], tv[ev][m])[0, 1]:+.3f}")
            print(f"{name:20s} R2 {r2:+.4f} ± {r2s:.4f}  corr {c:+.3f} ± {cs:.3f}  | " + " | ".join(per), flush=True)


def ridge_wf_insample(Z, y, m):
    """in-sample fit on the training rows (only used to centre the |g| target before 2022)."""
    from stock_walkforward import Ridge
    R = Ridge(Z[m], y[m])
    return R.predict(Z, R.weights(1.0))


if __name__ == "__main__":
    main()
