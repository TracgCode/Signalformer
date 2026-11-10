"""E4m: TAIFEX market and single-stock-futures features for the d 32 soft-Score set model (TWStock).

Data: dataset/taifex/ from demo/fetch_taifex.py (TAIFEX public downloads, private). Every feature of day t is known
before 09:00 of day t: the day-session rows are those of the previous trading day p (published after 13:45 / 15:00 of
p), the after-hours rows are those of the session that ends at 05:00 of t.
Market-level features (the same for every stock):
- tx_post: the TAIEX futures (TX, near month) 13:30-13:45 move of p: b(p) - median of b over the 20 previous days of
  the same contract month, b = log(TX close 13:45 / TAIEX close 13:30) (the median removes the slow basis);
- tx_lvol, tx_oichg: log TX volume (all months, day session) vs its 20-day mean; log change of the TX open interest;
- pc_vol, pc_oi and their 1-day changes: log TXO put/call volume and open-interest ratios;
- iv, d_iv: log ATM implied volatility of the TXO near month (straddle of the strike nearest to the TX settlement:
  sigma = (C + P) / (0.798 F sqrt(tau))); a stand-in for the TAIFEX VIX, whose history is not online (3 months only);
- lt5, lt10, lt10_spec, d_lt10: large traders' net TX position (top-5 / top-10 long minus short, all months, all /
  specific legal persons) over the market open interest; and lt_opt: the same for TXO calls minus puts if usable.
Per-stock single-stock-futures (SSF) features, for the 92 of our 100 stocks with an SSF (0 and has_ssf = 0 otherwise),
the standard (2,000-share) contract, else the mini (100-share) one; near month = the first month whose expiry (third
Wednesday) is at least 3 days after p (so the roll happens before expiry):
- ssf_post: b(p) - median of b over the 20 previous days of the same contract month, b = log(SSF settlement 13:45 /
  stock adjusted close 13:30) (the 13:30-13:45 move; the adjustment factor is constant between ex-dates and the
  median removes it with the basis);
- ssf_db: b(p) - b(p - 1); ssf_lvol: log SSF volume vs its 20-day mean; ssf_oichg: log change of the open interest;
- ssf_night: log(after-hours close / day settlement of p) of the session ending 05:00 of t (only 2330 and 2303 have
  SSF after-hours trading, from 2025; 0 otherwise); has_ssf.
SSF 08:45 open (separate variant): log(day open of t / settlement of p) of the near month, only for the contracts whose
first trade of the day session was before 09:00 on at least 95% of the days in TAIFEX's tick data of the last 30
trading days (the daily open of an illiquid contract can be a trade after 09:00, i.e. after the stock opened).
Models: the MSE ridge and the d 32 soft-Score set model (E4j final2 recipe: epochs on Y - 1, refit on all years before
Y, 10 seeds 0..9 averaged, the same seeds for every variant -> paired comparisons) on the 55 inputs (52 base + gap block
of g_hat_pre), with the market features, the SSF features, both, (both + SSF open,) and with the features also in the
g_hat_pre gap ridge (XG = X + preopen_features + new). Walk-forward EVAL 2022-2026. Forecasts to
demo/runs/e4m_taifex_pred.npz.
Round r2 (python demo/stock_score_taifex.py r2): only the per-stock SSF features, since the market features hurt; 40
seeds; base, + SSF in the model, SSF in the gap ridge, both, and both with the SSF open in the gap ridge; forecasts to
demo/runs/e4m_r2_pred.npz.
Usage: python demo/stock_score_taifex.py [timing | ridge | r2]   (timing: only the 08:45 tick check; ridge: the ridges and
the gap R2 only, on the CPU, forecasts to demo/runs/e4m_taifex_ridge_pred.npz)
"""

import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_poly import ROOT  # noqa: E402

TF = ROOT / "taifex"
CACHE = TF / "derived"
SEEDS = tuple(range(10))
OPEN_SHARE = 0.95


def num(s):
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", ""), errors="coerce")


def third_wednesday(ym):
    d = pd.Timestamp(year=int(ym[:4]), month=int(ym[4:6]), day=1)
    return d + pd.Timedelta(days=(2 - d.weekday()) % 7 + 14)


def ssf_codes(stocks):
    """stock id -> (standard code, mini code) of its SSF from TAIFEX's 股票期貨標的 list (codes + 'F')."""
    s = (TF / "stocklists.html").read_text(encoding="utf-8", errors="replace")
    body = s[s.find('id="myTable"'):]
    body = body[:body.find("</table>")]
    out = {}
    for tr in re.findall(r"<tr>(.*?)</tr>", body, flags=re.S):
        cl = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", t)).strip()
              for t in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)]
        if len(cl) < 14 or cl[2] not in stocks or "是股票期貨標的" not in cl[4]:
            continue
        std, mini = out.get(cl[2], (None, None))
        if cl[11].replace(",", "") == "100":
            mini = cl[0] + "F"
        else:
            std = cl[0] + "F"
        out[cl[2]] = (std, mini)
    return out


def read_fut(codes):
    """day and after-hours rows of the given futures codes, 2019-12 .. 2026-08 (cached)."""
    path = CACHE / "fut_rows.pkl"
    if path.exists():
        return pd.read_pickle(path)
    cols = ["交易日期", "契約", "到期月份(週別)", "開盤價", "收盤價", "成交量", "結算價", "未沖銷契約數", "交易時段"]
    frames = []
    srcs = [zipfile.ZipFile(p).open(zipfile.ZipFile(p).namelist()[0]) for p in sorted((TF / "fut").glob("*_fut.zip"))]
    srcs += sorted((TF / "fut").glob("2026_*.csv"))
    for src in srcs:
        raw = src.read() if hasattr(src, "read") else Path(src).read_bytes()
        f = pd.read_csv(io.BytesIO(raw), encoding="cp950", dtype=str, index_col=False)
        f.columns = [c.strip() for c in f.columns]
        f = f[cols].dropna(subset=["契約"])
        f["契約"] = f["契約"].str.strip()
        f = f[f["契約"].isin(codes)]
        frames.append(f)
        print(f"  read {getattr(src, 'name', src)}: {len(f)} rows", flush=True)
    f = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({"date": pd.to_datetime(f["交易日期"].str.strip()), "code": f["契約"],
                        "month": f["到期月份(週別)"].str.strip(), "session": f["交易時段"].str.strip(),
                        "open": num(f["開盤價"]), "close": num(f["收盤價"]), "vol": num(f["成交量"]),
                        "settle": num(f["結算價"]), "oi": num(f["未沖銷契約數"])})
    out = out[out["month"].str.fullmatch(r"\d{6}")].drop_duplicates(["date", "code", "month", "session"])
    out = out[(out["date"] >= "2019-11-01") & (out["date"] <= "2026-08-31")]
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_pickle(path)
    return out


def night_dating(day, night):
    """True if an after-hours row dated D is the session from the evening before D to 05:00 of D (as stock_overnight)."""
    d = day.set_index(["date", "month"])["close"]
    tdays = sorted(day["date"].unique())
    prev = dict(zip(tdays[1:], tdays[:-1]))
    a, b = [], []
    for r in night.itertuples():
        if r.date in prev and (prev[r.date], r.month) in d.index and (r.date, r.month) in d.index and r.open > 0:
            a.append(abs(r.open - d[(prev[r.date], r.month)]))
            b.append(abs(r.open - d[(r.date, r.month)]))
    return np.nanmedian(a) < np.nanmedian(b)


def near_month(dates_p, months):
    """per date: the first month whose expiry is >= 3 days after the date."""
    ms = sorted(months)
    exp = np.array([third_wednesday(m) for m in ms], dtype="datetime64[ns]")
    out = []
    for p in dates_p:
        k = np.searchsorted(exp, np.datetime64(p + pd.Timedelta(days=3)), "left")
        out.append(ms[k] if k < len(ms) else None)
    return out


def post_close(day, ref, price="settle", w=20):
    """per date p (index of ref): b(p) - median(b over the w previous dates of the same month), b = log(price / ref),
    on the near month of p; also b(p) - b(p - 1), the month used."""
    px = day.pivot_table(index="date", columns="month", values=price, aggfunc="first")
    if price == "settle":  # fall back on the close where the settlement is missing
        px = px.fillna(day.pivot_table(index="date", columns="month", values="close", aggfunc="first"))
    px = px.reindex(ref.index)
    B = np.log(px.where(px > 0).to_numpy() / ref.to_numpy()[:, None])
    nm = near_month(ref.index, px.columns)
    col = {m: j for j, m in enumerate(px.columns)}
    post, db = np.full(len(ref), np.nan), np.full(len(ref), np.nan)
    for i, m in enumerate(nm):
        if m is None:
            continue
        j = col[m]
        hist = B[max(0, i - w):i, j]
        hist = hist[np.isfinite(hist)]
        if np.isfinite(B[i, j]) and len(hist) >= 5:
            post[i] = B[i, j] - np.median(hist)
        if i > 0 and np.isfinite(B[i, j]) and np.isfinite(B[i - 1, j]):
            db[i] = B[i, j] - B[i - 1, j]
    return post, db, nm


def rel_lvol(v, w=20):
    lv = np.log1p(np.asarray(v, float))
    m = pd.Series(lv).shift(1).rolling(w, min_periods=5).mean().to_numpy()
    return lv - m


def by_prev(dates, series_p):
    """value for day t = series at the previous trading day dates[t - 1]."""
    s = pd.Series(series_p).reindex(dates)
    return np.r_[np.nan, s.to_numpy()[:-1]]


def market_features(dates, fut):
    """(T, K) market-level features of day t from the data of p = dates[t - 1], plus the names."""
    dates = pd.DatetimeIndex(dates)
    g = pd.read_csv(ROOT / "global" / "global_data.csv", parse_dates=["date"]).set_index("date").sort_index()
    taiex = g["^TWII"].reindex(dates)
    tx = fut[(fut["code"] == "TX") & (fut["session"] == "一般")]
    post, _, _ = post_close(tx, taiex, price="close")
    F = {"tx_post": np.r_[np.nan, post[:-1]]}
    agg = tx.groupby("date").agg(vol=("vol", "sum"), oi=("oi", "sum")).reindex(dates)
    F["tx_lvol"] = np.r_[np.nan, rel_lvol(agg["vol"].fillna(0))[:-1]]
    loi = np.log(agg["oi"].to_numpy())
    F["tx_oichg"] = np.r_[np.nan, np.r_[np.nan, np.diff(loi)][:-1]]
    # put/call ratios
    pc = pd.concat([pd.read_csv(p, encoding="cp950", index_col=False) for p in sorted((TF / "pcratio").glob("*.csv"))])
    pc.columns = [c.strip() for c in pc.columns]
    pc["date"] = pd.to_datetime(pc["日期"].astype(str).str.strip())
    pc = pc.drop_duplicates("date").set_index("date").sort_index()
    for c, n in (("買賣權成交量比率%", "pc_vol"), ("買賣權未平倉量比率%", "pc_oi")):
        v = np.log(num(pc[c]).where(lambda x: x > 0))
        F[n] = by_prev(dates, v)
        F["d_" + n] = by_prev(dates, v.diff())
    # ATM implied volatility of the TXO near month
    iv = txo_iv(tx)
    F["iv"] = by_prev(dates, np.log(iv))
    F["d_iv"] = by_prev(dates, np.log(iv).diff())
    # large traders
    lt = large_traders()
    for c in lt.columns:
        F[c] = by_prev(dates, lt[c])
    names = list(F)
    return np.stack([F[k] for k in names], -1), names


def txo_iv(tx):
    path = CACHE / "txo_iv.pkl"
    if path.exists():
        return pd.read_pickle(path)
    rows = []
    srcs = [(p, n) for p in sorted((TF / "opt").glob("*_opt.zip")) for n in zipfile.ZipFile(p).namelist()]
    srcs += [(p, None) for p in sorted((TF / "opt").glob("2026_*_TXO.csv"))]
    cols = ["交易日期", "契約", "到期月份(週別)", "履約價", "買賣權", "結算價", "交易時段"]
    for p, n in srcs:
        raw = zipfile.ZipFile(p).open(n).read() if n else p.read_bytes()
        for f in pd.read_csv(io.BytesIO(raw), encoding="cp950", dtype=str, index_col=False, chunksize=500000):
            f.columns = [c.strip() for c in f.columns]
            f = f[cols].dropna(subset=["契約"])
            f = f[(f["契約"].str.strip() == "TXO") & (f["交易時段"].str.strip() == "一般")]
            f = f[f["到期月份(週別)"].str.strip().str.fullmatch(r"\d{6}")]
            rows.append(pd.DataFrame({"date": pd.to_datetime(f["交易日期"].str.strip()),
                                      "month": f["到期月份(週別)"].str.strip(), "K": num(f["履約價"]),
                                      "cp": f["買賣權"].str.strip(), "settle": num(f["結算價"])}))
        print(f"  TXO read {p.name} {n or ''}", flush=True)
    o = pd.concat(rows, ignore_index=True).dropna()
    o = o[o["settle"] > 0]
    piv = o.pivot_table(index=["date", "month", "K"], columns="cp", values="settle", aggfunc="first").dropna()
    txs = tx.set_index(["date", "month"])["settle"].dropna()
    out = {}
    for d, sub in piv.groupby(level=0):
        ms = sorted(sub.index.get_level_values(1).unique())
        ms = [m for m in ms if third_wednesday(m) >= d + pd.Timedelta(days=5)]
        if not ms or (d, ms[0]) not in txs.index:
            continue
        m = ms[0]
        F = txs[(d, m)]
        s = sub.xs(m, level=1).droplevel(0)
        K = s.index.to_numpy()
        k = np.argmin(np.abs(K - F))
        tau = ((third_wednesday(m) - d).days + 0.5) / 365
        out[d] = (s.iloc[k]["買權"] + s.iloc[k]["賣權"]) / (0.7979 * F * np.sqrt(tau))
    iv = pd.Series(out).sort_index()
    iv.to_pickle(path)
    return iv


def read_lt_fut():
    path = CACHE / "lt_fut.pkl"
    if path.exists():
        return pd.read_pickle(path)
    f = pd.concat([pd.read_csv(p, encoding="cp950", dtype=str, index_col=False)
                   for p in sorted((TF / "largetrader_fut").glob("*.csv"))])
    f.columns = [c.strip() for c in f.columns]
    f = f[f["到期月份(週別)"].str.strip() == "999999"].copy()
    f["商品(契約)"] = f["商品(契約)"].str.strip()
    CACHE.mkdir(parents=True, exist_ok=True)
    f.to_pickle(path)
    return f


def ssf_large_traders():
    """date x 2-letter SSF code: top-10 traders' net position (long - short) over the open interest, all months."""
    f = read_lt_fut()
    f = f[f["交易人類別"].str.strip() == "0"].copy()
    f["date"] = pd.to_datetime(f["日期"].str.strip())
    f["n10"] = (num(f["前十大交易人買方"]) - num(f["前十大交易人賣方"])) / num(f["全市場未沖銷部位數"]).where(
        lambda x: x > 0)
    return f.pivot_table(index="date", columns="商品(契約)", values="n10", aggfunc="first")


def large_traders():
    """per date: net positions of the large TX traders over the market open interest (all months)."""
    f = read_lt_fut()
    f = f[f["商品(契約)"] == "TX"].copy()
    f["date"] = pd.to_datetime(f["日期"].str.strip())
    oi = num(f["全市場未沖銷部位數"])
    f["n5"] = (num(f["前五大交易人買方"]) - num(f["前五大交易人賣方"])) / oi
    f["n10"] = (num(f["前十大交易人買方"]) - num(f["前十大交易人賣方"])) / oi
    f["typ"] = f["交易人類別"].str.strip()
    a = f[f["typ"] == "0"].drop_duplicates("date").set_index("date").sort_index()
    s = f[f["typ"] == "1"].drop_duplicates("date").set_index("date").sort_index()
    out = pd.DataFrame({"lt5": a["n5"], "lt10": a["n10"], "lt10_spec": s["n10"].reindex(a.index)})
    out["d_lt10"] = out["lt10"].diff()
    o = pd.concat([pd.read_csv(p, encoding="cp950", dtype=str, index_col=False)
                   for p in sorted((TF / "largetrader_opt").glob("*.csv"))])
    o.columns = [c.strip() for c in o.columns]
    if "買賣權" in o.columns:
        o = o[(o["商品(契約)"].str.strip() == "TXO") & (o["到期月份(週別)"].str.strip() == "999999")
              & (o["交易人類別"].str.strip() == "0")]
        o["date"] = pd.to_datetime(o["日期"].str.strip())
        o["net"] = (num(o["前十大交易人買方"]) - num(o["前十大交易人賣方"])) / num(o["全市場未沖銷部位數"])
        cp = o.pivot_table(index="date", columns=o["買賣權"].str.strip(), values="net", aggfunc="first")
        if {"買權", "賣權"} <= set(cp.columns):
            out["lt_opt"] = (cp["買權"] - cp["賣權"]).reindex(out.index)
    return out


def ssf_features(dates, stocks, fut, adj_close, codes, minvol=None):
    """(T, S, K) per-stock SSF features of day t, the names, and (T, S) the SSF 08:45 open feature (unfiltered)."""
    dates = pd.DatetimeIndex(dates)
    T, S = len(dates), len(stocks)
    names = ["ssf_post", "ssf_db", "ssf_lvol", "ssf_oichg", "ssf_night", "has_ssf", "ssf_lt10", "ssf_dlt10"]
    LT = ssf_large_traders()
    X = np.full((T, S, len(names)), np.nan)
    OPEN = np.full((T, S), np.nan)
    allday = fut[fut["session"] == "一般"]
    allnight = fut[fut["session"] == "盤後"]
    nd = None
    for j, s in enumerate(stocks):
        if s not in codes:
            continue
        std, mini = codes[s]
        day = allday[allday["code"] == (std or mini)]
        if len(day) < 100 and mini:
            day = allday[allday["code"] == mini]
        if len(day) == 0:
            continue
        post, db, nm = post_close(day, adj_close[s].reindex(dates), price="settle")
        X[:, j, 0] = np.r_[np.nan, post[:-1]]
        X[:, j, 1] = np.r_[np.nan, db[:-1]]
        # volume (standard-contract equivalents) and open interest over all months, both sizes
        both = allday[allday["code"].isin([c for c in (std, mini) if c])].copy()
        wgt = np.where(both["code"] == mini, 1 / 20, 1.0) if std else 1.0
        both["v"], both["o"] = both["vol"] * wgt, both["oi"] * wgt
        agg = both.groupby("date")[["v", "o"]].sum().reindex(dates)
        X[:, j, 2] = np.r_[np.nan, rel_lvol(agg["v"].fillna(0))[:-1]]
        loi = np.log(agg["o"].where(agg["o"] > 0).to_numpy())
        X[:, j, 3] = np.r_[np.nan, np.r_[np.nan, np.diff(loi)][:-1]]
        X[:, j, 5] = np.r_[0, np.isfinite(post[:-1]).astype(float)]
        c2 = (std or mini)[:-1]
        if c2 in LT.columns:
            lt = LT[c2].reindex(dates).to_numpy()
            X[:, j, 6] = np.r_[np.nan, lt[:-1]]
            X[:, j, 7] = np.r_[np.nan, np.r_[np.nan, np.diff(lt)][:-1]]
        # after-hours session ending at 05:00 of t, near month of p, vs the day settlement of p
        night = allnight[allnight["code"] == day["code"].iloc[0]]
        sett = day.pivot_table(index="date", columns="month", values="settle", aggfunc="first").reindex(dates)
        if len(night):
            if nd is None:
                nd = night_dating(allday[allday["code"] == "TX"], allnight[allnight["code"] == "TX"])
                print(f"after-hours rows dated D are the session {'ending 05:00 of D' if nd else 'starting the evening of D'}",
                      flush=True)
            nc = night[night["vol"] > 0].pivot_table(index="date", columns="month", values="close", aggfunc="first")
            for t in range(1, T):
                m = nm[t - 1]
                key = dates[t] if nd else dates[t - 1]
                if m is not None and key in nc.index and m in nc.columns and m in sett.columns:
                    v = np.log(nc.at[key, m] / sett.iloc[t - 1][m])
                    if np.isfinite(v):
                        X[t, j, 4] = v
        # 08:45 open of t of the near month of p vs the settlement of p
        op = day.pivot_table(index="date", columns="month", values="open", aggfunc="first").reindex(dates)
        vol = day.pivot_table(index="date", columns="month", values="vol", aggfunc="first").reindex(dates)
        for t in range(1, T):
            m = nm[t - 1]
            # guard: the near month must have traded at least half its tick-window median volume on p (known)
            if (m is not None and m in op.columns and vol.iloc[t][m] > 0
                    and vol.iloc[t - 1][m] >= (minvol or {}).get(day["code"].iloc[0], np.inf)):
                OPEN[t, j] = np.log(op.iloc[t][m] / sett.iloc[t - 1][m])
    X[..., 4] = np.nan_to_num(X[..., 4])
    return X, names, OPEN


def open_timing(codes):
    """per contract: share of the days it traded on which the first day-session trade (08:45-13:45) of its most traded
    month was before 09:00, from the TAIFEX tick files of the last 30 trading days; and the number of days traded."""
    want = {c for v in codes.values() for c in v if c}
    first, alldays, qty = {}, set(), {}
    for p in sorted((TF / "ticks").glob("Daily_*.zip")):
        z = zipfile.ZipFile(p)
        for n in z.namelist():
            f = pd.read_csv(io.BytesIO(z.open(n).read()), encoding="cp950", dtype=str, index_col=False)
            code, tm = f.iloc[:, 1].str.strip(), f.iloc[:, 3].str.strip().str.zfill(6)
            dayses = (tm >= "084500") & (tm <= "134500")
            alldays |= set(f.iloc[:, 0].str.strip()[dayses & (code == "TX")])
            keep = code.isin(want) & dayses
            g = pd.DataFrame({"code": code[keep], "day": f.iloc[:, 0].str.strip()[keep],
                              "month": f.iloc[:, 2].str.strip()[keep], "tm": tm[keep],
                              "q": pd.to_numeric(f.iloc[:, 5][keep], errors="coerce")})
            g = g.groupby(["code", "day", "month"]).agg(tm=("tm", "min"), q=("q", "sum")).reset_index()
            g = g.sort_values("q").drop_duplicates(["code", "day"], keep="last")
            for r in g.itertuples():
                first.setdefault(r.code, {})[r.day] = r.tm
                qty.setdefault(r.code, {})[r.day] = r.q / 2  # B+S counts both sides
    out = {}
    for c in sorted(want):
        v = first.get(c, {})
        out[c] = (sum(x < "090000" for x in v.values()) / max(len(v), 1), len(v),
                  float(np.median(list(qty[c].values()))) if c in qty else 0.0)
    return out, len(alldays)


def clean(F, pre):
    """NaN -> 0 after clipping each column at its 0.5% / 99.5% quantiles on the rows before 2022 (no eval data)."""
    F = F.copy()
    for k in range(F.shape[-1]):
        v = F[..., k]
        ref = v[pre & np.isfinite(v)]
        if len(ref) > 100 and not np.all(np.isin(ref, (0.0, 1.0))):
            lo, hi = np.quantile(ref, [0.005, 0.995])
            F[..., k] = np.clip(v, lo, hi)
    return np.nan_to_num(F)


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

    global SEEDS
    RIDGE = "ridge" in sys.argv[1:]
    R2 = "r2" in sys.argv[1:]
    if R2:
        SEEDS = tuple(range(40))
    if RIDGE:
        it.DEV = "cpu"
    dates, r, traded, gap, known = load()
    T, S = r.shape
    info = json.loads((ROOT / "info.json").read_text())
    stocks, top50 = info["Top100@Y"], set(info["Top50@Y"])
    codes = ssf_codes(stocks)
    print(f"{len(codes)} of {S} stocks have an SSF; e.g. 2330 -> {codes.get('2330')}", flush=True)
    tim, ndays = open_timing(codes)
    print(f"\n######## SSF 08:45 timing check, {ndays} tick days: share of days with the first day-session trade "
          f"before 09:00 (code: share, days traded)")
    allowed = {c for c, (sh, n, _) in tim.items() if sh >= OPEN_SHARE and n >= 0.8 * ndays}
    print("  " + ", ".join(f"{c} {sh:.2f}/{n}" for c, (sh, n, _) in sorted(tim.items(), key=lambda kv: -kv[1][0])))
    print("  median daily near-month volume in the tick window (allowed): " +
          ", ".join(f"{c} {tim[c][2]:.0f}" for c in sorted(allowed)))
    print(f"  contracts with >= {OPEN_SHARE:.0%} and traded on >= 80% of the days: {len(allowed)}: {sorted(allowed)}", flush=True)
    if "timing" in sys.argv[1:]:
        return
    want = {"TX"} | {c for v in codes.values() for c in v if c}
    fut = read_fut(want)
    print(f"futures rows: {len(fut)}, {fut['date'].min():%Y-%m-%d} .. {fut['date'].max():%Y-%m-%d}", flush=True)
    adj = pd.DataFrame({s: pd.read_csv(ROOT / "market" / f"{s}.csv", parse_dates=["date"]).set_index("date")[
        "etl:adj_close"] for s in stocks}).sort_index()
    XMr, mnames = market_features(dates, fut)
    minvol = {c: 0.5 * tim[c][2] for c in allowed}
    XSr, snames, OPENr = ssf_features(dates, stocks, fut, adj, codes, minvol)
    # the open variant: only stocks whose used contract passed the timing check
    use_open = np.array([s in codes and (codes[s][0] or codes[s][1]) in allowed for s in stocks])
    OPENr = np.where(use_open[None, :], OPENr, np.nan)
    year = np.repeat(dates.year.to_numpy()[:, None], S, axis=1)
    pre = year < 2022
    XM = clean(np.repeat(XMr[:, None, :], S, 1), pre)
    XS = clean(XSr, pre)
    XO = clean(OPENr[..., None], pre)
    print(f"\nmarket features {mnames}\nSSF features {snames}; SSF open on {use_open.sum()} stocks", flush=True)
    yrs_ = dates.year.to_numpy()
    for k, n in enumerate(mnames):
        v = XMr[:, k]
        print(f"  {n:10s} coverage 2020-21 {np.isfinite(v[yrs_ < 2022]).mean():.2f}  2022-26 "
              f"{np.isfinite(v[yrs_ >= 2022]).mean():.2f}  sd {np.nanstd(v):.4g}")
    for k, n in enumerate(snames):
        v = XSr[..., k]
        print(f"  {n:10s} non-zero share {np.mean(np.nan_to_num(v) != 0):.2f}  sd {np.nanstd(v):.4g}")
    print(f"  ssf_open   non-zero share {np.mean(XO != 0):.3f}", flush=True)

    X, y = base_features(dates, r, traded, gap, known, stocks)
    half = year * 2 + (np.repeat(dates.month.to_numpy()[:, None], S, axis=1) > 6)
    yrs = dates.year.to_numpy()
    g = gap.to_numpy()
    ok = (np.arange(T) >= FIRST)[:, None] & np.isfinite(X).all(-1) & np.isfinite(y) & np.isfinite(g)
    trd = traded.to_numpy()
    PO = preopen_features(dates, gap, traded)
    XG = np.concatenate([X, PO], -1)
    ev = ok & np.isin(year, EVAL)
    is50 = np.array([s in top50 for s in stocks])[None, :]
    has = np.repeat(np.array([s in codes for s in stocks])[None, :], T, 0)
    subsets = (("all 100 stocks", ev), ("the 50-stock subset", ev & is50), ("stocks with an SSF", ev & has),
               ("100 stocks, 2025-2026", ev & (year >= 2025)))
    fm = lambda t: f"{UNIT * t[0]:+5.1f}e-4 ± {UNIT * t[1]:3.1f}e-4"

    # extra inputs of the models (appended to the 55) and of the gap ridge
    EXTRA = {"base": None, "+market": XM, "+SSF": XS, "+both": np.concatenate([XM, XS], -1)}
    GEXTRA = {"base": None, "+market": XM, "+SSF": XS, "+both": EXTRA["+both"]}
    if allowed:
        EXTRA["+both +SSF open"] = np.concatenate([XM, XS, XO], -1)
        GEXTRA["+SSF open"] = XO
        GEXTRA["+both +SSF open"] = EXTRA["+both +SSF open"]
    # variant: (model inputs, gap-ridge inputs)
    VAR = {"base": ("base", "base"), "+market": ("+market", "base"), "+SSF": ("+SSF", "base"),
           "+both": ("+both", "base"), "g_hat with both": ("base", "+both")}
    if allowed:
        VAR["+both +SSF open"] = ("+both +SSF open", "base")
        VAR["g_hat with both +SSF open"] = ("base", "+both +SSF open")
    if R2:  # round r2: only the per-stock SSF features, in the model and/or the gap ridge, 40 seeds
        GEXTRA = {"base": None, "+SSF": XS}
        VAR = {"base": ("base", "base"), "+SSF": ("+SSF", "base"), "g_hat with SSF": ("base", "+SSF"),
               "+SSF, g_hat with SSF": ("+SSF", "+SSF")}
        if allowed:
            GEXTRA["+SSF +SSF open"] = np.concatenate([XS, XO], -1)
            VAR["+SSF, g_hat with SSF +SSF open"] = ("+SSF", "+SSF +SSF open")
    cfg = ("day", "e4h", 32, "r", "soft", False, "none", 0.0, 0.2, 1e-2, 1e-3, 16, "all", len(SEEDS))
    names = ["always down"] + [f"ridge {v}" for v in VAR] + ([] if RIDGE else [f"set {v}" for v in VAR])
    res = {k: np.full((T, S), np.nan) for k in names}
    res["always down"][:] = -1.0
    G = {k: np.full((T, S), np.nan) for k in GEXTRA}
    Yt = torch.tensor(np.where(ok, y, 0.0), dtype=torch.float32, device=it.DEV)
    Mt = torch.tensor(ok, device=it.DEV)
    sig = np.zeros((T, S, 3))
    for Y in EVAL:
        gh = {}
        for gk, ge in GEXTRA.items():
            XGk = XG if ge is None else np.concatenate([XG, ge], -1)
            gh[gk] = (ghat_for(XGk, g, ok, year, Y - 1), ghat_for(XGk, g, ok, year, Y))
            G[gk][ok & (year == Y)] = gh[gk][1][ok & (year == Y)]
        inner, sel, rows, te = ok & (year < Y - 1), ok & (year == Y - 1), ok & (year < Y), ok & (year == Y)
        dd = lambda cond: torch.tensor(np.nonzero(cond & ok.any(1))[0], device=it.DEV)
        d_in, d_sel, d_rows, d_te = dd(yrs < Y - 1), dd(yrs == Y - 1), dd(yrs < Y), dd(yrs == Y)
        msg = []
        for v, (ek, gk) in VAR.items():
            ext = (lambda Z_: Z_) if EXTRA[ek] is None else (lambda Z_: np.concatenate([Z_, EXTRA[ek]], -1))
            Zs = ext(np.concatenate([X, gap_block(gh[gk][0], ok)], -1))
            Zr = ext(np.concatenate([X, gap_block(gh[gk][1], ok)], -1))
            m = Ridge(Zs[inner], y[inner])
            a = min(ALPHAS, key=lambda a: np.mean((m.predict(Zs[sel], m.weights(a)) - y[sel]) ** 2))
            mu_s = oos_mu(Zs, y, ok, year, half, a, Y - 1)
            mu_r = oos_mu(Zr, y, ok, year, half, a, Y)
            res[f"ridge {v}"][te] = mu_r[te]
            if RIDGE:
                msg.append(f"{v}: {Zr.shape[-1]} inputs, ridge alpha {a:g}")
                continue
            inp_s = it.Inputs("day", Zs, mu_s, y, ok, trd, sig, inner)
            inp_r = it.Inputs("day", Zr, mu_r, y, ok, trd, sig, rows)
            o_s = torch.zeros((T, S), device=it.DEV)
            o, eps = np.zeros((T, S)), []
            for seed in SEEDS:
                _, ep = it.train(cfg, inp_s, Yt, o_s, Mt, d_in, seed, vdays=d_sel)
                ep = max(ep, 1)
                net, _ = it.train(cfg, inp_r, Yt, o_s, Mt, d_rows, seed, epochs=ep)
                with torch.no_grad():
                    for i in range(0, len(d_te), 64):
                        b = d_te[i:i + 64]
                        o[b.cpu().numpy()] += net(*inp_r.batch(b), o_s[b], Mt[b]).cpu().numpy()
                eps.append(ep)
            res[f"set {v}"][te] = o[te]
            msg.append(f"{v}: {Zr.shape[-1]} inputs, ridge alpha {a:g}, set epochs {eps}")
            del inp_s, inp_r
            torch.cuda.empty_cache()
        print(f"{Y} done\n  " + "\n  ".join(msg), flush=True)

    print("\n######## gap forecast g_hat_pre, walk-forward R2 on 2022-2026 (ok rows), per gap-ridge input set")
    gb = np.mean((g[ev] - g[ev].mean()) ** 2)
    dayi = np.repeat(np.arange(T)[:, None], S, 1)
    for gk in GEXTRA:
        e_ = (g[ev] - G[gk][ev]) ** 2
        dm = clustered_mean(((g[ev] - G["base"][ev]) ** 2 - e_), dayi[ev])
        print(f"  g_hat {gk:18s} R2 {1 - e_.mean() / gb:.4f}  vs base {dm[0] / gb:+.4f} ± {dm[1] / gb:.4f}", flush=True)

    print("\n######## vs the MSE ridge without the features")
    report(res, {k: "ridge base" for k in names if k != "ridge base"}, y, subsets, fm)
    print("\n######## vs the set model without the features (paired, same seeds)")
    report({k: res[k] for k in names if k.startswith("set")}, {k: "set base" for k in names if k.startswith("set ")
                                                                and k != "set base"}, y, subsets, fm)
    print("\n######## ridge variants vs ridge base")
    report({k: res[k] for k in names if k.startswith("ridge")}, {k: "ridge base" for k in names if k.startswith("ridge ")
                                                                  and k != "ridge base"}, y, subsets[:2], fm)
    for k in names:
        print(f"share of up calls, {k}: {np.mean(res[k][ev] > 0):.3f}")

    print("\n######## table (Score = mean sign(f) r; ± day-clustered SE)")
    sg = {k: np.where(res[k] > 0, 1.0, -1.0) for k in names}
    hdr = "| model | Score (100) | Δ vs ridge | Δ vs set base | Score (50) | " + " | ".join(str(Y) for Y in EVAL) + " |"
    print(hdr)
    print("|" + "---|" * (5 + len(EVAL)))
    for k in names:
        c = lambda m_, ref=None: fm(clustered_mean(((sg[k] - (0 if ref is None else sg[ref])) * y)[m_], dayi[m_]))
        dr = c(ev, "ridge base") if k not in ("ridge base",) else "—"
        dsb = c(ev, "set base") if not RIDGE and k.startswith("set ") and k != "set base" else "—"
        line = f"| {k} | {c(ev)} | {dr} | {dsb} | {c(ev & is50)} | " + " | ".join(c(ev & (year == Y)) for Y in EVAL)
        print(line + " |", flush=True)
    np.savez(Path(__file__).resolve().parent / "runs" / ("e4m_taifex_ridge_pred.npz" if RIDGE else "e4m_r2_pred.npz" if R2 else "e4m_taifex_pred.npz"), y=y, ok=ok, ev=ev, year=year,
             is50=is50, mnames=np.array(mnames), snames=np.array(snames),
             **{k.replace(" ", "_").replace("+", "p"): v for k, v in res.items()},
             **{"ghat_" + k.replace(" ", "_").replace("+", "p"): v for k, v in G.items()})


if __name__ == "__main__":
    main()
