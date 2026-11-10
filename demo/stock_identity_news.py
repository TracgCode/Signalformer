"""E2j: the typed news fields in the gap block, and StockIdentityAttention on the ridge residual, walk-forward (TWStock).

Step 0 (python demo/stock_identity_news.py gap [pre]; pre: the gap ridge also gets the 4 pre-open features of
stock_overnight.preopen_features, E3c): the gap block of demo/stock_news_gap.py (g_hat, |g_hat|, the day's
mean g_hat) with three news inputs of the gap ridge, on top of the 52 base price features:
- E2: the LLM and embedding features of demo/stock_news.py (Qwen2.5 probabilities, bge-m3 of the texts);
- struct: per stock-day the mean over its overnight items of the 24 field probabilities of demo/news_struct.py and
  the first 16 principal components of the bge-m3 embedding of their summaries (fitted on the items before 2022),
  log(1 + count) per source (MOPS, cnyes), the mean direction P(利多) - P(利空) of the stock's items of the previous
  session (43); with market: also the same 40 means and the log count of the overnight market items (84);
- E2 + struct (own part).
The market part lowers the R2 of the gap (+0.2303 against +0.2395 without it, also on the stock-days without news):
its 41 columns are the same for all stocks of a day, and the ridge overfits them.
For each, the ridge on the 55 features (52 + gap block) is fitted walk-forward as in demo/stock_itrans_news.py and
compared with the E2 version, which is the reference of E2f.

Step 1 (python demo/stock_identity_news.py net [gap input, default struct]): the residual e = r - mu of that ridge
forecast by the networks of stock_model.py, with the protocol of demo/stock_itrans_news.py (steps chosen on Y - 1 with
out-of-sample residuals, patience 3, up to 30 epochs, refit on all years before Y, 3 seeds averaged, d = 16, dropout
0.3, AdamW lr 1e-3, weight decay 1e-2, batches of 16 days). Versions:
- iT: stock tokens, no news;
- sigval, E2 news: StockNewsAttention mode sigval with the E2 item vectors (bge-m3 PCA64 of the text and s);
- sigval, struct news: the same with the struct item vectors (24 fields, summary PCA16, source flags; 42 dims);
- identity, industry: StockIdentityAttention with the struct items and the identity = one-hot of the industry
  (dataset/profiles.csv) plus 8 learned dims;
- identity, comovement: identity = stock_model.comovement_embedding (16) of the daily log returns, from the years
  before Y - 1 for the choice of steps and before Y for the refit;
- identity, learned only: identity = 0, only the learned embedding;
- identity, industry permuted: the industry rows given to the wrong stocks (control);
- identity, industry, news shuffled: each day gets the item pool of another day of the same year (control).
Diagnostics of the identity versions (refit of the first seed, test days): mean entropy of an item's attention over
the stocks, the attention on the stock it tags and on the other stocks of that stock's industry (uniform: 1 / S and
(n_industry - 1) / S), beta.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from news_struct import load_struct  # noqa: E402
from stock_itrans_news import DAYS, D, N_MAX, SEEDS, U, W, Data, ghat_for, news_pool  # noqa: E402
from stock_news import text_features  # noqa: E402
from stock_news_attn import report  # noqa: E402
from stock_news_attn2 import oos_mu  # noqa: E402
from stock_news_gap import gap_block  # noqa: E402
from stock_poly import ROOT, load  # noqa: E402
from stock_recurrent import base_features, lagged  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stock_model import PAD, StockIdentityAttention, StockNewsAttention, comovement_embedding  # noqa: E402

NEWS_DIM = 42


def item_index(dates, stocks):
    """per item: day d, stock index j (-1 market, -2 not ours), overnight flag, in-range flag."""
    T = len(dates)
    z = np.load(ROOT / "news" / "scores" / "items.npz")
    ts = pd.DatetimeIndex(z["ts"])
    cut = (dates + pd.Timedelta(hours=8, minutes=50)).to_numpy()
    d = np.searchsorted(cut, ts.to_numpy(), "left")
    keep = (d > 0) & (d < T)
    prev_close = np.r_[np.datetime64("NaT"), (dates[:-1] + pd.Timedelta(hours=13, minutes=30)).to_numpy()]
    over = np.zeros(len(d), bool)
    over[keep] = ts.to_numpy()[keep] > prev_close[d[keep]]
    sidx = {c: k for k, c in enumerate(stocks)}
    j = np.array([sidx.get(c, -1 if c == "MKT" else -2) for c in z["stock"]])
    return z, ts, d, j, over, keep


def struct_vectors(ts):
    """(n_items, 40): the 24 field probabilities and the summary PCA16 (standardized on the items before 2022)."""
    st = load_struct()
    F = np.concatenate([st[f"p_{k}"] for k in ("relation", "event", "direction", "size", "surprise")], 1)
    Es = np.load(ROOT / "news" / "struct" / "summary_emb.npy")
    pre = np.asarray(ts < "2022-01-01")
    Xp = Es[pre][:: max(1, pre.sum() // 50000)].astype(np.float32)
    c = Xp.mean(0)
    _, _, Vt = np.linalg.svd(Xp - c, full_matrices=False)
    PS = np.concatenate([(Es[i:i + 50000].astype(np.float32) - c) @ Vt[:16].T for i in range(0, len(Es), 50000)])
    PS /= PS[pre].std(0) + 1e-6
    dirn = st["p_direction"][:, 0] - st["p_direction"][:, 1]
    return np.concatenate([F, PS], 1).astype(np.float32), dirn


def struct_features(dates, stocks):
    """per stock-day gap-ridge inputs from the struct items (see the docstring), (T, S, 84)."""
    T, S = len(dates), len(stocks)
    z, ts, d, j, over, keep = item_index(dates, stocks)
    V, dirn = struct_vectors(ts)
    src = z["src"]
    own, mkt = keep & (j >= 0), keep & (j == -1)
    o = own & over
    cnt = np.zeros((T, S))
    np.add.at(cnt, (d[o], j[o]), 1.0)
    M = np.zeros((T, S, V.shape[1]))
    np.add.at(M, (d[o], j[o]), V[o])
    M /= np.maximum(cnt, 1)[..., None]
    cs = []
    for so in ("mops", "cnyes"):
        c = np.zeros((T, S))
        m = o & (src == so)
        np.add.at(c, (d[m], j[m]), 1.0)
        cs.append(np.log1p(c))
    ses = own & ~over
    sd, sc = np.zeros((T, S)), np.zeros((T, S))
    np.add.at(sd, (d[ses], j[ses]), dirn[ses])
    np.add.at(sc, (d[ses], j[ses]), 1.0)
    sd /= np.maximum(sc, 1)
    mo = mkt & over
    mc = np.bincount(d[mo], minlength=T).astype(float)
    MM = np.zeros((T, V.shape[1]))
    np.add.at(MM, d[mo], V[mo])
    MM /= np.maximum(mc, 1)[:, None]
    out = np.concatenate([np.stack(cs + [sd], -1), M, np.repeat(MM[:, None], S, 1),
                          np.repeat(np.log1p(mc)[:, None, None], S, 1)], -1)
    print(f"struct gap features: {out.shape[-1]}; stock-days with an own overnight item {np.mean(cnt > 0):.1%}",
          flush=True)
    return out


def struct_pool(dates, stocks):
    """per day: (N_MAX, 42) struct item vectors with the source flags, and the tags (stock, -1 market, -2 padding)."""
    T = len(dates)
    z, ts, d, j, over, keep = item_index(dates, stocks)
    V, _ = struct_vectors(ts)
    use = keep & over & (j >= -1)
    src = z["src"][use]
    Z = np.concatenate([V[use], (src == "mops")[:, None], (src == "cnyes")[:, None]], 1).astype(np.float32)
    du, ju, tu = d[use], j[use], ts[use].to_numpy()
    order = np.lexsort((-tu.astype("int64"), du))
    du, ju, Z = du[order], ju[order], Z[order]
    starts, ends = np.searchsorted(du, np.arange(T)), np.searchsorted(du, np.arange(T), "right")
    NZ = np.zeros((T, N_MAX, NEWS_DIM), np.float32)
    NT = np.full((T, N_MAX), PAD, np.int64)
    for t in range(T):
        n = min(ends[t] - starts[t], N_MAX)
        NZ[t, :n] = Z[starts[t]:starts[t] + n]
        NT[t, :n] = ju[starts[t]:starts[t] + n]
    return NZ, NT


def industry_onehot(stocks):
    prof = pd.read_csv(ROOT / "profiles.csv", dtype={"code": str}).set_index("code").reindex([str(c) for c in stocks])
    ind = prof.industry.fillna("?").to_numpy()
    cats = sorted(set(ind))
    return np.array([[float(a == c) for c in cats] for a in ind], np.float32), ind


def train(make, data, seed, days_tr, steps=None, days_va=None):
    torch.manual_seed(seed)
    g = torch.Generator(device=data.dev).manual_seed(seed)
    net = make().to(data.dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    if days_va is not None:
        Mv, Yv = data.M[days_va], data.Y[days_va]
        val = lambda: ((data.predict(net, days_va) - Yv)[Mv]).pow(2).mean().item()
        z = Yv[Mv].pow(2).mean().item()
        best, best_steps, bad = val(), 0, 0
        curve = [best / z]
    done = 0
    for ep in range(1, 31):
        order = days_tr[torch.randperm(len(days_tr), device=data.dev, generator=g)]
        for i in range(0, len(order), DAYS):
            if steps is not None and done >= steps:
                return net, done, None
            b = order[i:i + DAYS]
            m = data.M[b]
            loss = ((net(*data.batch(b)) - data.Y[b])[m]).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            done += 1
        if days_va is not None:
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
    return net, (best_steps if days_va is not None else done), (curve if days_va is not None else None)


def diagnostics(net, data, days, ind):
    S = len(ind)
    same = torch.tensor(ind[:, None] == ind[None, :], device=data.dev).float()
    same.fill_diagonal_(0)
    ent, own, ins, n, ntag = 0.0, 0.0, 0.0, 0, 0
    net.eval()
    with torch.no_grad():
        for i in range(0, len(days), 64):
            _, _, nz, nt = data.batch(days[i:i + 64])
            a = net.item_attention(nz, nt)                          # (B, N, S)
            tagged = nt >= 0
            v = nt != PAD
            ent += (-(a * (a + 1e-12).log()).sum(-1))[v].sum().item()
            n += v.sum().item()
            t = nt.clamp(min=0)
            own += a.gather(-1, t[..., None]).squeeze(-1)[tagged].sum().item()
            ins += (a * same[t]).sum(-1)[tagged].sum().item()
            ntag += tagged.sum().item()
    net.train()
    exp_in = float((same.sum(1)).mean().item()) / S
    return (f"entropy {ent / n:.2f} (uniform {np.log(S):.2f}), own {own / max(ntag, 1):.3f} (uniform {1 / S:.3f}), "
            f"same industry {ins / max(ntag, 1):.3f} (uniform {exp_in:.3f}), beta {net.beta.item():+.2f}")


def main() -> None:
    use_pre = "pre" in sys.argv[1:]          # E3c: the gap ridge also gets stock_overnight.preopen_features
    argv = [a for a in sys.argv if a != "pre"]
    step = argv[1] if len(argv) > 1 else "gap"
    gin = argv[2] if len(argv) > 2 else "struct"
    dev = "cuda"
    dates, r, traded, gap, known = load()
    T, S = r.shape
    info = json.loads((ROOT / "info.json").read_text())
    stocks, top50 = info["Top100@Y"], set(info["Top50@Y"])
    X, y = base_features(dates, r, traded, gap, known, stocks)
    LLM, _, EMB, has = text_features(dates, stocks)
    STR = struct_features(dates, stocks)
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    half = year * 2 + (np.repeat(dates.month.to_numpy()[:, None], S, axis=1) > 6)
    g = gap.to_numpy()
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y) & np.isfinite(g)
    trd = traded.to_numpy()
    own = STR[..., :43]   # without the market part, which overfits (same value for all stocks of a day)
    XG = X
    if use_pre:
        from stock_overnight import preopen_features
        XG = np.concatenate([X, preopen_features(dates, gap, traded)], -1)
    GIN = {"E2": np.concatenate([XG, LLM, EMB], -1), "struct": np.concatenate([XG, own], -1),
           "E2 + struct": np.concatenate([XG, LLM, EMB, own], -1),
           "struct with market": np.concatenate([XG, STR], -1)}
    if use_pre:
        GIN = {"price only (pre-open)": XG, **GIN}
    ev = ok & np.isin(year, EVAL)
    is50 = np.array([s in top50 for s in stocks])[None, :]
    subsets = (("all 100 stocks", ev), ("the 50-stock subset", ev & is50),
               ("100 stocks, days with an own overnight item", ev & has),
               ("100 stocks, 2025-2026", ev & (year >= 2025)))
    fm = lambda t: f"{U * t[0]:+5.1f}e-4 ± {U * t[1]:3.1f}e-4"

    def stage1(Xn, Y):
        Xa_s = np.concatenate([X, gap_block(ghat_for(Xn, g, ok, year, Y - 1), ok)], -1)
        G = ghat_for(Xn, g, ok, year, Y)
        Xa_r = np.concatenate([X, gap_block(G, ok)], -1)
        inner, sel = ok & (year < Y - 1), ok & (year == Y - 1)
        m = Ridge(Xa_s[inner], y[inner])
        a = min(ALPHAS, key=lambda a: np.mean((m.predict(Xa_s[sel], m.weights(a)) - y[sel]) ** 2))
        return Xa_s, Xa_r, oos_mu(Xa_s, y, ok, year, half, a, Y - 1), oos_mu(Xa_r, y, ok, year, half, a, Y), G

    if step == "gap":
        res = {f"ridge, gap from {k}": np.full((T, S), np.nan) for k in GIN}
        gh = {k: np.full((T, S), np.nan) for k in GIN}
        for Y in EVAL:
            te = ok & (year == Y)
            for k, Xn in GIN.items():
                _, _, _, mu_r, G = stage1(Xn, Y)
                res[f"ridge, gap from {k}"][te] = mu_r[te]
                gh[k][te] = G[te]
            print(f"{Y} done", flush=True)
        for k in GIN:
            gm = np.array([g[ok & (year < Y)].mean() for Y in EVAL])[year[ev] - EVAL[0]]
            r2 = 1 - np.mean((g[ev] - gh[k][ev]) ** 2) / np.mean((g[ev] - gm) ** 2)
            hh = ev & has
            gm2 = np.array([g[ok & (year < Y)].mean() for Y in EVAL])[year[hh] - EVAL[0]]
            r2h = 1 - np.mean((g[hh] - gh[k][hh]) ** 2) / np.mean((g[hh] - gm2) ** 2)
            print(f"g_hat from {k:12s}: R2 of the gap {r2:+.4f}, on days with an own item {r2h:+.4f}", flush=True)
        print("\n######## the ridge with each gap block vs the E2 gap block")
        ref = "ridge, gap from E2"
        report(res, {k: ref for k in res if k != ref}, y, subsets, fm)
        return

    # ---------------- step 1: the networks ----------------
    sig_r0 = np.where(trd, np.nan_to_num(r.to_numpy()), 0.0)
    g0 = np.where(trd, np.nan_to_num(g), 0.0)
    pre = ok & (year < 2022)
    sig = np.stack([lagged(sig_r0, 1) / sig_r0[pre].std(), lagged(g0, 1) / g0[pre].std(),
                    lagged(trd.astype(float), 1)], -1)
    NZ_e2, NT_e2 = news_pool(dates, stocks)
    NZ_st, NT_st = struct_pool(dates, stocks)
    rng = np.random.default_rng(0)
    perm = np.arange(T)
    yrs = dates.year.to_numpy()
    for yy in np.unique(yrs):
        idx = np.nonzero(yrs == yy)[0]
        perm[idx] = rng.permutation(idx)
    tP = torch.tensor(perm, device=dev)
    pools = {"e2": (torch.tensor(NZ_e2, device=dev), torch.tensor(NT_e2, device=dev)),
             "st": (torch.tensor(NZ_st, device=dev), torch.tensor(NT_st, device=dev))}
    pools["st shuffled"] = (pools["st"][0][tP], pools["st"][1][tP])
    del NZ_e2, NT_e2, NZ_st, NT_st
    IND, ind = industry_onehot(stocks)
    sperm = np.random.default_rng(1).permutation(S)
    lc = np.log1p(r.where(traded).to_numpy()) + np.where(trd, g, np.nan)
    comov = lambda Yc: torch.tensor(comovement_embedding(lc[(dates.year < Yc) & (np.arange(T) >= 1)], 16))
    zero_id = torch.zeros(S, 1)
    E2DIM = pools["e2"][0].shape[-1]
    # name: (pool, factory(Fx, Yc) where Yc = the first year not used by the identity)
    V = {
        "iT": ("e2", lambda Fx, Yc: StockNewsAttention(S, W * 4, Fx, E2DIM, use_news=False, d_model=D)),
        "sigval, E2 news": ("e2", lambda Fx, Yc: StockNewsAttention(S, W * 4, Fx, E2DIM, mode="sigval", d_model=D)),
        "sigval, struct news": ("st", lambda Fx, Yc: StockNewsAttention(S, W * 4, Fx, NEWS_DIM, mode="sigval",
                                                                       d_model=D)),
        "identity, industry": ("st", lambda Fx, Yc: StockIdentityAttention(S, W * 4, Fx, torch.tensor(IND), NEWS_DIM,
                                                                          d_model=D)),
        "identity, comovement": ("st", lambda Fx, Yc: StockIdentityAttention(S, W * 4, Fx, comov(Yc), NEWS_DIM,
                                                                            d_model=D)),
        "identity, learned only": ("st", lambda Fx, Yc: StockIdentityAttention(S, W * 4, Fx, zero_id, NEWS_DIM,
                                                                              d_model=D)),
        "identity, industry permuted": ("st", lambda Fx, Yc: StockIdentityAttention(
            S, W * 4, Fx, torch.tensor(IND[sperm]), NEWS_DIM, d_model=D)),
        "identity, industry, news shuffled": ("st shuffled", lambda Fx, Yc: StockIdentityAttention(
            S, W * 4, Fx, torch.tensor(IND), NEWS_DIM, d_model=D)),
    }
    only = [a for a in argv[3:]]
    if only:
        V = {k: v for k, v in V.items() if k in only}
    print(f"gap input: {gin}; versions: {list(V)}", flush=True)
    tday = np.arange(T)
    res = {k: np.full((T, S), np.nan) for k in ("ridge",) + tuple(V)}
    import time
    t0 = time.time()
    for Y in EVAL:
        Xa_s, Xa_r, mu_s, mu_r, _ = stage1(GIN[gin], Y)
        te = ok & (year == Y)
        res["ridge"][te] = mu_r[te]
        tr_s, tr_r = ok & (year < Y - 1), ok & (year < Y)
        dsel = lambda msk: torch.tensor(tday[(msk.any(1)) & (tday >= W)], device=dev)
        d_tr_s, d_va_s, d_tr_r, d_te = dsel(tr_s), dsel(ok & (year == Y - 1)), dsel(tr_r), dsel(te)
        for v, (pool, make) in V.items():
            nz, nt = pools[pool]
            data_s = Data(Xa_s, y, mu_s, ok & (year <= Y - 1), sig, trd, tr_s, nz, nt, dev)
            data_r = Data(Xa_r, y, mu_r, ok & (year <= Y), sig, trd, tr_r, nz, nt, dev)
            Fx = data_r.cur.shape[-1]
            preds, log, diag = [], [], ""
            for seed in SEEDS:
                _, steps, curve = train(lambda: make(Fx, Y - 1), data_s, seed, d_tr_s, days_va=d_va_s)
                if steps == 0:
                    preds.append(np.zeros((len(d_te), S)))
                else:
                    net, _, _ = train(lambda: make(Fx, Y), data_r, seed, d_tr_r, steps=steps)
                    preds.append(data_r.predict(net, d_te).cpu().numpy() * data_r.se)
                    if seed == SEEDS[0] and isinstance(net, StockIdentityAttention):
                        diag = "; " + diagnostics(net, data_r, d_te, ind)
                log.append(f"{steps} steps (val {min(curve):.5f})")
            out = np.full((T, S), np.nan)
            out[d_te.cpu().numpy()] = np.mean(preds, 0)
            res[v][te] = mu_r[te] + out[te]
            print(f"{Y} {v}: {', '.join(log)}{diag}  [{time.time() - t0:.0f} s]", flush=True)
            del data_s, data_r
            torch.cuda.empty_cache()

    print(f"\n######## gap input {gin}: each version vs the ridge")
    report(res, {v: "ridge" for v in V}, y, subsets, fm)
    print("\n######## pairs")
    pairs = (("identity, industry", "iT"), ("identity, industry", "sigval, struct news"),
             ("sigval, struct news", "sigval, E2 news"), ("identity, industry", "identity, industry permuted"),
             ("identity, industry", "identity, industry, news shuffled"), ("identity, industry", "identity, learned only"),
             ("identity, comovement", "identity, learned only"))
    for a_, b_ in pairs:
        if a_ in res and b_ in res:
            report({b_: res[b_], a_: res[a_]}, {a_: b_}, y, subsets[:3], fm)


if __name__ == "__main__":
    main()
