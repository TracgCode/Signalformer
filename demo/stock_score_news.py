"""E4k: the morning's news as tokens of the E4h set model (TWStock).

E2f/E2j could not train a news pathway: their networks were narrow (d 16), started from a zero head under the MSE on
the ridge residual, and often trained for 0 steps. E4i/E4j found the recipe that does train (target r, usual head,
d 64, at least one epoch). Here the set model of E4h gets the day's news items as extra tokens in the same encoder
layer, so a stock can attend to news items as well as to the other stocks:
- stock token: GELU(Linear(the stock's 55 day features)) as in E4h;
- news token: Linear(the 42-dim struct item vector of demo/stock_identity_news.struct_pool: 24 Qwen3.5 field
  probabilities, summary bge-m3 PCA16, source flags) + a learned news-type embedding + (for an item that tags one of
  our stocks) Linear(that stock's day features), so the item carries which stock it is about; at most 256 overnight
  items per day (own and market), padding masked;
- one pre-norm Transformer encoder layer over the stock and news tokens (4 heads, feed-forward 128, dropout 0.2),
  output Linear(64, 1) on the stock tokens only.
Versions: no news (E4h set), news, news of another day of the same year (control: the tags move with the items), news
without the tag term. Target r, soft Score; epochs on Y - 1 (patience 3, up to 40, at least 1), refit, 5 seeds.
Usage: python demo/stock_score_news.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_identity_news import struct_features, struct_pool  # noqa: E402
from stock_itrans_news import U as UNIT, ghat_for  # noqa: E402
from stock_news_attn import report  # noqa: E402
from stock_news_gap import gap_block  # noqa: E402
from stock_overnight import preopen_features  # noqa: E402
from stock_poly import ROOT, load  # noqa: E402
from stock_recurrent import base_features  # noqa: E402
from stock_score_loss import loss_fn  # noqa: E402
from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stock_model import PAD  # noqa: E402

DEV = "cuda"
SEEDS = (0, 1, 2, 3, 4)


class NewsSet(nn.Module):
    def __init__(self, k, news_dim, use_news, use_tag, d=64):
        super().__init__()
        self.use_news, self.use_tag = use_news, use_tag
        self.inp = nn.Linear(k, d)
        self.news = nn.Linear(news_dim, d)
        self.ntype = nn.Parameter(torch.zeros(d))
        self.tag = nn.Linear(k, d, bias=False)
        self.enc = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.2, batch_first=True, norm_first=True,
                                              activation="gelu")
        self.out = nn.Linear(d, 1)

    def forward(self, z, m, nz, nt):        # z (B, S, k), m (B, S), nz (B, N, nd), nt (B, N) tags
        h = nn.functional.gelu(self.inp(z))
        S = z.shape[1]
        if not self.use_news:
            return self.out(self.enc(h, src_key_padding_mask=~m)[:, :S]).squeeze(-1)
        n = self.news(nz) + self.ntype
        if self.use_tag:
            own = nt >= 0
            zt = torch.gather(z, 1, nt.clamp(min=0)[..., None].expand(-1, -1, z.shape[-1]))
            n = n + own[..., None].float() * self.tag(zt)
        x = torch.cat([h, n], 1)
        mask = torch.cat([~m, nt == PAD], 1)
        return self.out(self.enc(x, src_key_padding_mask=mask)[:, :S]).squeeze(-1)


def fit(cfg, Z, Y, M, NZ, NT, days, seed, epochs=None, vdays=None):
    use_news, use_tag = cfg
    torch.manual_seed(seed)
    g = torch.Generator(device=DEV).manual_seed(seed)
    net = NewsSet(Z.shape[-1], NZ.shape[-1], use_news, use_tag).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_ep, bad = np.inf, 0, 0
    run = lambda b: (net(Z[b], M[b], NZ[b], NT[b])[M[b]], Y[b][M[b]])
    for ep in range(1, (epochs or 40) + 1):
        net.train()
        perm = days[torch.randperm(len(days), device=DEV, generator=g)]
        for i in range(0, len(perm), 16):
            f, y = run(perm[i:i + 16])
            l = loss_fn("soft", f, y)
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
        if epochs is None:
            net.eval()
            with torch.no_grad():
                fs, ys = zip(*(run(vdays[i:i + 32]) for i in range(0, len(vdays), 32)))
                v = loss_fn("soft", torch.cat(fs), torch.cat(ys)).item()
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
    NZ, NT = struct_pool(dates, stocks)
    print(f"news tokens per day: mean {(NT != PAD).sum(1).mean():.1f}, max {(NT != PAD).sum(1).max()}", flush=True)
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    yrs = dates.year.to_numpy()
    rng = np.random.default_rng(0)
    perm = np.arange(T)
    for yy in np.unique(yrs):
        ii = np.nonzero(yrs == yy)[0]
        perm[ii] = rng.permutation(ii)
    g = gap.to_numpy()
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y) & np.isfinite(g)
    XG = np.concatenate([X, preopen_features(dates, gap, traded)], -1)
    ev = ok & np.isin(year, EVAL)
    is50 = np.array([s in top50 for s in stocks])[None, :]
    subsets = (("all 100 stocks", ev), ("the 50-stock subset", ev & is50),
               ("100 stocks, days with an own overnight item", ev & has),
               ("100 stocks, 2025-2026", ev & (year >= 2025)))
    fm = lambda t: f"{UNIT * t[0]:+5.1f}e-4 ± {UNIT * t[1]:3.1f}e-4"
    NZt = torch.tensor(NZ, device=DEV)
    NTt = torch.tensor(NT, device=DEV)
    pt = torch.tensor(perm, device=DEV)
    VERS = {"set, no news (E4h)": ((False, False), False), "set + news tokens": ((True, True), False),
            "set + news of another day (control)": ((True, True), True),
            "set + news tokens without the tag term": ((True, False), False)}
    names = ["ref: MSE ridge"] + list(VERS)
    res = {k: np.full((T, S), np.nan) for k in names}
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
        msg = []
        for name, (cfg, shuf) in VERS.items():
            NZu, NTu = (NZt[pt], NTt[pt]) if shuf else (NZt, NTt)
            o, eps = np.zeros((T, S)), []
            for seed in SEEDS:
                _, ep = fit(cfg, As, Yt, Mt, NZu, NTu, d_in, seed, vdays=d_sel)
                ep = max(ep, 1)
                net, _ = fit(cfg, Ar, Yt, Mt, NZu, NTu, d_rows, seed, epochs=ep)
                with torch.no_grad():
                    for i in range(0, len(d_te), 32):
                        b = d_te[i:i + 32]
                        o[b.cpu().numpy()] += net(Ar[b], Mt[b], NZu[b], NTu[b]).cpu().numpy()
                eps.append(ep)
            res[name][te] = o[te]
            msg.append(f"{name}: epochs {eps}")
        print(f"{Y} done: ridge alpha {a:g}\n  " + "\n  ".join(msg), flush=True)
    print("\n######## news tokens in the set model, vs the MSE ridge (R2 meaningless except ref)")
    report(res, {k: names[0] for k in names[1:]}, y, subsets, fm)
    print("\n######## vs the set model without news, and news vs the news of another day")
    refs = {names[2]: names[1], names[3]: names[1], names[4]: names[1]}
    report({k: res[k] for k in names[1:]}, refs, y, subsets, fm)
    report({k: res[k] for k in names[2:4]}, {names[2]: names[3]}, y, subsets, fm)
    for k in names:
        print(f"share of up calls, {k}: {np.mean(res[k][ev] > 0):.3f}")
    np.savez(Path(__file__).resolve().parent / "runs" / "e4k_pred.npz", y=y, ok=ok, ev=ev, year=year, is50=is50,
             **{k.replace(" ", "_").replace(",", "").replace(":", "").replace("(", "").replace(")", ""): v
                for k, v in res.items()})


if __name__ == "__main__":
    main()
