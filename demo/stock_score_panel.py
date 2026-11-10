"""E4p: the columns of the user's panel file (dataset/panel/panel.parquet) as extra inputs of the best sign-score model.

The panel has one row per stock and trading day (100 stocks, 2020-01-02 .. 2026-10-20); every column of row d is known
after the close of day d, so it is shifted by one trading day within each stock (groupby stock_id, shift(1)) and then
serves the forecast of the next day, as the panel's notes require (the US columns are already aligned so that row d
holds the US session of the night after d, so the same single shift applies). The forecast days are those of
stock_poly.load (to 2026-08-31).

Duplicates of the data already in dataset/ are dropped: the price, volume, institutional and technical-indicator columns
(identical to dataset/market/*.csv) and the g_ market columns (identical to dataset/global/global_data.csv); `y` and the
identifiers are not inputs. The remaining columns, in groups:
- shares: log shares issued, turnover, the four institutional net buys / shares issued;
- margin: the FinMind margin and short-sale counts / shares issued, usage, usage change, estimated cost relative to
  the adjusted close, maintenance, eligibility and position flags, offsetting trades / shares issued;
- sbl: the securities-lending short-sale counts / shares issued, balance ratio, sell ratio, balance-ratio change
  (the quota, which the notes describe as the next day's remaining quota, also / shares issued);
- susp: the short-sale suspension flag and the trading days to the forced cover;
- hold: the 15 brackets of the weekly shareholding distribution (percent, log people, their changes), the update-day
  flag, days since the update, the share-event flag;
- mkt: log market margin purchase, margin money and short sale, and their percent changes (the same for every stock);
- ohlc (a separate variant, the duplicated market columns made stationary): log volume / turnover amount / trades
  relative to their 20-day means, RSI, MACD / close, ATR / close, the BBANDS and SMA relative to the close.
Each column is forward-filled within the stock (never backward), clipped at its 0.5 / 99.5% quantiles of the rows
before 2021 (inside every fold's training data) and its missing values set to the pre-2021 median; the model's
standardization is the usual one on the training rows of each fold.

Model: the d 32 soft-Score set model of stock_score_itoken.py final2 (epochs on Y - 1, refit, seeds averaged), the
inputs 55 + the group's columns. Variants with seeds 0..9 (paired with the base at the same seeds): base, + each group,
+ all groups (without ohlc), + all + ohlc; the MSE ridge with and without all groups. Walk-forward 2022-2026.
Part 'r2' (python demo/stock_score_panel.py r2 <variant>): one variant with seeds 0..39, paired with final2.
Forecasts to demo/runs/e4p_pred.npz (r2: e4p_r2_pred.npz).
Part 'fast40' (python demo/stock_score_panel.py fast40): the groups that helped at 10 seeds and their unions, with
40 seeds 0..39 and the batched trainer of demo/stock_score_fast.py (same seeds, initial weights and data order as the
one-network trainer; only dropout differs), each paired with the base trained the same way. Forecasts to
demo/runs/e4p_fast40_pred.npz.
Part 'confirm' (python demo/stock_score_panel.py fast40 confirm): base and susp+sbl+shares with 40 new seeds 40..79,
to check the best fast40 union on seeds not used to pick it; forecasts to demo/runs/e4p_confirm_pred.npz.
Usage: python demo/stock_score_panel.py [r2 <variant> | fast40 [confirm]]
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_poly import ROOT  # noqa: E402

PYLIB = os.environ.get("PYLIB", "")  # optional extra package directory (pyarrow), from the environment


def panel_groups(dates, stocks):
    """{group: ((T, S, K) array aligned to (dates, stocks) after the one-day shift, [names])}"""
    if PYLIB and PYLIB not in sys.path:
        sys.path.append(PYLIB)
    p = pd.read_parquet(ROOT / "panel" / "panel.parquet")
    p["stock_id"] = p["stock_id"].astype(str)
    p = p.sort_values(["stock_id", "date"]).reset_index(drop=True)
    sh = p["shares_issued"].where(p["shares_issued"] > 0)
    close = p["etl:adj_close"]
    F = {}
    F["shares"] = {"log_shares": np.log(sh), "turnover": p["turnover"],
                   **{c: p[c] for c in ("inst_foreign_ratio", "inst_trust_ratio", "inst_dealer_self_ratio",
                                        "inst_dealer_hedge_ratio")}}
    mcount = [c for c in p.columns if c.startswith(("MarginPurchase", "ShortSale", "MarginShortSales"))
              and c not in ("MarginPurchaseLimit", "ShortSaleLimit")] + ["OffsetLoanAndShort"]
    F["margin"] = {**{c + "/sh": p[c] / sh for c in mcount},
                   **{c + "/sh": p[c] / sh for c in ("MarginPurchaseLimit", "ShortSaleLimit")},
                   **{c: p[c] for c in ("margin_usage", "short_usage", "margin_usage_change", "short_usage_change",
                                        "margin_maintenance", "short_maintenance", "margin_eligible",
                                        "has_margin_position", "has_short_position")},
                   "margin_cost/close": p["margin_cost"] / close - 1, "short_cost/close": p["short_cost"] / close - 1}
    scount = [c for c in p.columns if c.startswith("SBLShortSales")]
    F["sbl"] = {**{c + "/sh": p[c] / sh for c in scount},
                **{c: p[c] for c in ("sbl_balance_ratio", "sbl_sell_ratio", "sbl_balance_ratio_change")}}
    F["susp"] = {c: p[c] for c in ("short_susp", "days_to_short_cover")}
    hold = {}
    for c in p.columns:
        if c.startswith("hold_percent"):
            hold[c] = p[c]
        elif c.startswith("hold_people_change"):
            hold[c] = p[c]
        elif c.startswith("hold_people"):
            hold["log_" + c] = np.log1p(p[c])
    hold.update({c: p[c] for c in ("hold_share_event_week", "hold_is_update_day", "hold_days_since_update")})
    F["hold"] = hold
    F["mkt"] = {**{"log_" + c: np.log(p[c].where(p[c] > 0)) for c in ("mkt_MarginPurchase", "mkt_MarginPurchaseMoney",
                                                                        "mkt_ShortSale")},
                **{c: p[c] for c in ("mkt_MarginPurchase_chg_pct", "mkt_MarginPurchaseMoney_chg_pct",
                                     "mkt_ShortSale_chg_pct")}}
    g = p.groupby("stock_id")
    rel20 = lambda c: np.log1p(p[c].clip(lower=0)) - g[c].transform(lambda s: np.log1p(s.clip(lower=0)).rolling(20, 1).mean())
    F["ohlc"] = {"lvol_rel20": rel20("price:成交股數"), "lamt_rel20": rel20("price:成交金額"), "ltrades_rel20": rel20("price:成交筆數"),
                 "RSI": p["RSI"], "MACD/close": p["MACD"] / close, "ATR/close": p["ATR"] / close,
                 **{c + "/close": p[c] / close - 1 for c in ("SMA", "BBANDS_upper", "BBANDS_mid", "BBANDS_lower")}}
    out = {}
    idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock_id"])
    for grp, cols in F.items():
        df = pd.DataFrame(cols)
        df[["stock_id", "date"]] = p[["stock_id", "date"]]
        df = df.replace([np.inf, -np.inf], np.nan)
        val = [c for c in df.columns if c not in ("stock_id", "date")]
        df[val] = df.groupby("stock_id")[val].ffill()
        df[val] = df.groupby("stock_id")[val].shift(1)   # row d -> the forecast of the next trading day
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
        a = df.set_index(["date", "stock_id"])[val].reindex(idx).to_numpy(np.float64)
        out[grp] = (a.reshape(len(dates), len(stocks), len(val)), val)
    return out, p


def prep(F, pre):
    """clip at the pre-2021 quantiles, fill the missing values with the pre-2021 median (constant columns -> 0)."""
    F = F.copy()
    for k in range(F.shape[-1]):
        v = F[..., k]
        ref = v[pre & np.isfinite(v)]
        if len(ref) < 100:
            F[..., k] = 0.0
            continue
        if not np.all(np.isin(ref, (0.0, 1.0))):
            lo, hi = np.quantile(ref, [0.005, 0.995])
            v = np.clip(v, lo, hi)
        F[..., k] = np.where(np.isfinite(v), v, np.median(ref))
    return F


def main() -> None:
    import torch
    import stock_score_itoken as it
    from stock_itrans_news import U as UNIT, ghat_for
    from stock_news_attn import report
    from stock_news_attn2 import oos_mu
    from stock_news_gap import gap_block
    from stock_overnight import preopen_features
    from stock_poly import load
    from stock_recurrent import base_features
    from stock_walkforward import ALPHAS, EVAL, FIRST, Ridge, clustered_mean

    R2 = len(sys.argv) > 2 and sys.argv[1] == "r2"
    FAST = "fast40" in sys.argv[1:]
    CONF = "confirm" in sys.argv[1:]
    seeds = tuple(range(40, 80)) if CONF else tuple(range(40)) if (R2 or FAST) else tuple(range(10))
    dates, r, traded, gap, known = load()
    T, S = r.shape
    info = json.loads((ROOT / "info.json").read_text())
    stocks, top50 = info["Top100@Y"], set(info["Top50@Y"])
    PG, p = panel_groups(dates, stocks)
    X, y = base_features(dates, r, traded, gap, known, stocks)
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    half = year * 2 + (np.repeat(dates.month.to_numpy()[:, None], S, axis=1) > 6)
    yrs = dates.year.to_numpy()
    g = gap.to_numpy()
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y) & np.isfinite(g)
    trd = traded.to_numpy()
    pre = year < 2021

    # checks: the panel's own y and r against ours on the same day (before the shift)
    pv = p.assign(date=pd.to_datetime(p["date"]).astype("datetime64[ns]")).set_index(["date", "stock_id"])
    idx = pd.MultiIndex.from_product([dates, stocks])
    r_p = (pv["etl:adj_close"] / pv["etl:adj_open"] - 1).reindex(idx).to_numpy().reshape(T, S)
    m_ = ok & np.isfinite(r_p)
    print(f"panel rows on our days {np.isfinite(r_p).sum()} of {T * S}; r: corr {np.corrcoef(r_p[m_], y[m_])[0, 1]:.6f}, "
          f"max |diff| {np.max(np.abs(r_p[m_] - y[m_])):.2e}", flush=True)
    EX = {}
    for grp, (a, names) in PG.items():
        EX[grp] = prep(a, pre)
        cov = np.isfinite(a[ok & (year >= 2022)]).mean(0)
        print(f"group {grp}: {len(names)} columns; coverage 2022-26 min {cov.min():.2f} mean {cov.mean():.2f}", flush=True)
    EX["all"] = np.concatenate([EX[k] for k in ("shares", "margin", "sbl", "susp", "hold", "mkt")], -1)
    EX["all + ohlc"] = np.concatenate([EX["all"], EX["ohlc"]], -1)
    VAR = ["base", "shares", "margin", "sbl", "susp", "hold", "mkt", "ohlc", "all", "all + ohlc"]
    if R2:
        VAR = ["base", sys.argv[2]] if sys.argv[2] != "base" else ["base"]
    if FAST:
        VAR = ["base", "susp", "sbl", "shares", "margin", "susp+sbl", "susp+sbl+shares", "susp+sbl+shares+margin"]
        if CONF:
            VAR = ["base", "susp+sbl+shares"]
        for v in VAR:
            if "+" in v:
                EX[v] = np.concatenate([EX[k] for k in v.split("+")], -1)
    print(f"variants {VAR}, seeds {seeds[0]}..{seeds[-1]}", flush=True)

    PO = preopen_features(dates, gap, traded)
    XG = np.concatenate([X, PO], -1)
    ev = ok & np.isin(year, EVAL)
    is50 = np.array([s in top50 for s in stocks])[None, :]
    subsets = (("all 100 stocks", ev), ("the 50-stock subset", ev & is50), ("100 stocks, 2025-2026", ev & (year >= 2025)))
    fm = lambda t: f"{UNIT * t[0]:+5.1f}e-4 ± {UNIT * t[1]:3.1f}e-4"
    cfg = it.FINAL2["d 32, soft Score, 40 seeds"][:13] + (len(seeds),)
    names = ["always down", "ridge base", "ridge + all"] + [f"set {v}" for v in VAR]
    res = {k: np.full((T, S), np.nan) for k in names}
    res["always down"][:] = -1.0
    Yt = torch.tensor(np.where(ok, y, 0.0), dtype=torch.float32, device=it.DEV)
    Mt = torch.tensor(ok, device=it.DEV)
    sig = np.zeros((T, S, 3))
    for Y in EVAL:
        gh_s, gh_r = ghat_for(XG, g, ok, year, Y - 1), ghat_for(XG, g, ok, year, Y)
        Z0s = np.concatenate([X, gap_block(gh_s, ok)], -1)
        Z0r = np.concatenate([X, gap_block(gh_r, ok)], -1)
        inner, sel, rows, te = ok & (year < Y - 1), ok & (year == Y - 1), ok & (year < Y), ok & (year == Y)
        dd = lambda cond: torch.tensor(np.nonzero(cond & ok.any(1))[0], device=it.DEV)
        d_in, d_sel, d_rows, d_te = dd(yrs < Y - 1), dd(yrs == Y - 1), dd(yrs < Y), dd(yrs == Y)
        msg = []
        for rk, ek in (("ridge base", None), ("ridge + all", "all")):
            Zs = Z0s if ek is None else np.concatenate([Z0s, EX[ek]], -1)
            Zr = Z0r if ek is None else np.concatenate([Z0r, EX[ek]], -1)
            m = Ridge(Zs[inner], y[inner])
            a = min(ALPHAS, key=lambda a: np.mean((m.predict(Zs[sel], m.weights(a)) - y[sel]) ** 2))
            m = Ridge(Zr[rows], y[rows])
            res[rk][te] = m.predict(Zr[te], m.weights(a))
            msg.append(f"{rk}: alpha {a:g}")
        m = Ridge(Z0s[inner], y[inner])
        a = min(ALPHAS, key=lambda a: np.mean((m.predict(Z0s[sel], m.weights(a)) - y[sel]) ** 2))
        mu_s = oos_mu(Z0s, y, ok, year, half, a, Y - 1)
        mu_r = oos_mu(Z0r, y, ok, year, half, a, Y)
        o_s = torch.zeros((T, S), device=it.DEV)
        for v in VAR:
            Zs = Z0s if v == "base" else np.concatenate([Z0s, EX[v]], -1)
            Zr = Z0r if v == "base" else np.concatenate([Z0r, EX[v]], -1)
            inp_s = it.Inputs("day", Zs, mu_s, y, ok, trd, sig, inner)
            inp_r = it.Inputs("day", Zr, mu_r, y, ok, trd, sig, rows)
            o, eps = np.zeros((T, S)), []
            if FAST:
                import stock_score_fast as fast
                f_, eps = fast.fit_predict(seeds, inp_s, inp_r, Yt, Mt, Mt, d_in, d_sel, d_rows, d_te)
                o[d_te.cpu().numpy()] = f_
            for seed in (() if FAST else seeds):
                _, ep = it.train(cfg, inp_s, Yt, o_s, Mt, d_in, seed, vdays=d_sel)
                ep = max(ep, 1)
                net, _ = it.train(cfg, inp_r, Yt, o_s, Mt, d_rows, seed, epochs=ep)
                with torch.no_grad():
                    for i in range(0, len(d_te), 64):
                        b = d_te[i:i + 64]
                        o[b.cpu().numpy()] += net(*inp_r.batch(b), o_s[b], Mt[b]).cpu().numpy()
                eps.append(ep)
            res[f"set {v}"][te] = o[te]
            msg.append(f"set {v}: {Zr.shape[-1]} inputs, epochs {eps}")
            del inp_s, inp_r
            torch.cuda.empty_cache()
        print(f"{Y} done\n  " + "\n  ".join(msg), flush=True)

    if R2:
        z2 = np.load(Path(__file__).resolve().parent / "runs" / "e4j_final2_pred.npz")
        res["E4j final2 (40 seeds)"] = z2["d_32_soft_Score_40_seeds"]
        names.append("E4j final2 (40 seeds)")
    print("\n######## vs the MSE ridge")
    report({k: res[k] for k in names if k != "always down"}, {k: "ridge base" for k in names if k not in ("always down", "ridge base")},
           y, subsets, fm)
    print("\n######## vs the set model without the panel columns (paired, same seeds)")
    report({k: res[k] for k in names if k.startswith("set")}, {k: "set base" for k in names if k.startswith("set ") and k != "set base"},
           y, subsets, fm)
    sg = {k: np.where(res[k] > 0, 1.0, -1.0) for k in names}
    dayi = np.repeat(np.arange(T)[:, None], S, 1)
    print("\n######## table (sign score = mean sign(f) r; ± day-clustered SE; differences paired)")
    print("| model | sign score (100) | − ridge | − set base | sign score (50) | − set base (50) | up share | "
          + " | ".join(str(Y) for Y in EVAL) + " |")
    print("|" + "---|" * (7 + len(EVAL)))
    for k in names:
        c = lambda m_, ref=None: fm(clustered_mean(((sg[k] - (0 if ref is None else sg[ref])) * y)[m_], dayi[m_]))
        isset = k.startswith("set ") and k != "set base"
        print(f"| {k} | {c(ev)} | {c(ev, 'ridge base') if k != 'ridge base' else '0'} | {c(ev, 'set base') if isset else '—'} | "
              f"{c(ev & is50)} | {c(ev & is50, 'set base') if isset else '—'} | {np.mean(res[k][ev] > 0):.3f} | "
              + " | ".join(c(ev & (year == Y)) for Y in EVAL) + " |", flush=True)
    np.savez(Path(__file__).resolve().parent / "runs" / ("e4p_r2_pred.npz" if R2 else "e4p_confirm_pred.npz" if CONF else "e4p_fast40_pred.npz" if FAST else "e4p_pred.npz"), y=y, ok=ok, ev=ev,
             year=year, is50=is50, **{k.replace(" ", "_").replace("+", "p"): v for k, v in res.items()})


if __name__ == "__main__":
    main()
