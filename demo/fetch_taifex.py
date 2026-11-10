"""Download TAIFEX (臺灣期貨交易所, www.taifex.com.tw) public market data into dataset/taifex/ (gitignored, private).

Sources (all free public download forms, no login, no key; robots.txt does not exist (HTTP 404) on 2026-11-10):
- fut/: 期貨每日交易行情, every futures contract, contract month and session (一般 = day 08:45-13:45, 盤後 = night
  15:00-05:00, dated by the day it is attributed to, i.e. the morning it ends). Years 2019-2025 from the annual zip
  (POST /cht/3/futDataDown down_type=2 his_year=Y); 2026-01..2026-08 by month (down_type=1 commodity_id=all, the form
  allows at most one month per request).
- opt/: 選擇權每日交易行情 (all option contracts; we use TXO for an ATM implied-volatility proxy, because TAIFEX keeps
  only the last 3 months of its VIX files). Annual zip /cht/3/optDataDown, 2026 by month /cht/3/dlOptDataDown TXO.
- pcratio/: 臺指選擇權 Put/Call ratios, /cht/3/pcRatioDown, one month per request.
- largetrader_fut/, largetrader_opt/: 大額交易人未沖銷部位 (top-5 / top-10 traders' long and short open interest, all
  contracts), /cht/3/dlLargeTraderFutDown and dlLargeTraderOptDown, at most three months per request.
- vix/: 臺指選擇權波動率指數 daily close; only the last 3 months are online (/file/taifex/Dailydownload/vix/log2data/
  YYYYMMnew.txt), so only 2026-08 falls in the project period. Kept for reference.
- stocklists.html, 2_stockinfo.ods: 股票期貨標的 list (contract code -> stock id), /cht/2/stockLists.
- ticks/: 前30個交易日期貨每筆成交資料 (CSV zips) for the 08:45-open timing check.

Terms: TAIFEX's 使用條款 reserve all rights; reproduction or redistribution needs written consent (except the
data.gov.tw open-data sets). Private research only; never commit or publish dataset/taifex.
Rate: one sequential request at a time, a pause of 2.5 s between requests; on an error wait 60 s, then 600 s, then
give up on that file. User-Agent "research-script/1.0 (non-commercial academic research)". Resume-safe: files that
already exist are skipped.

    python demo/fetch_taifex.py
"""

import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "dataset" / "taifex"
BASE = "https://www.taifex.com.tw"
UA = {"User-Agent": "research-script/1.0 (non-commercial academic research)"}
PAUSE = 2.5
_last = [0.0]


def request(url, data=None):
    """One request, at least PAUSE s after the previous one; back off 60 s, then 600 s on errors."""
    for wait in (60, 600, None):
        dt = time.time() - _last[0]
        if dt < PAUSE:
            time.sleep(PAUSE - dt)
        _last[0] = time.time()
        try:
            body = urllib.parse.urlencode(data).encode() if data is not None else None
            with urllib.request.urlopen(urllib.request.Request(url, body, UA), timeout=180) as r:
                return r.read(), r.geturl()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"  error {url} {data}: {e}", flush=True)
            if wait is None:
                return None, None
            time.sleep(wait)


def save(path, url, data=None, kind="csv"):
    """Download to path unless it exists. kind: csv (cp950 text starting with a header), zip, txt, html."""
    if path.exists():
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        raw, final = request(url, data)
        if raw is None:
            return False
        ok = {"zip": raw[:2] == b"PK",
              "csv": raw[:40].decode("cp950", "replace").lstrip("﻿").startswith(("交易日期", "日期")),
              "txt": final is not None and "404" not in final and len(raw) > 0,
              "html": len(raw) > 1000}[kind]
        if kind == "csv" and not ok and len(raw) < 400 and raw.count(b"\n") <= 1:
            print(f"  {path.name}: empty answer {raw[:120]!r}", flush=True)
            return False
        if ok:
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(raw)
            tmp.rename(path)
            print(f"{path.relative_to(OUT)}: {len(raw):,} bytes", flush=True)
            return True
        print(f"  {path.name}: unexpected answer ({len(raw)} bytes) {raw[:160]!r}", flush=True)
        time.sleep(60 if attempt == 0 else 0)
    return False


def months(a="2019-12-01", b="2026-08-31"):
    for m in pd.date_range(a, b, freq="MS"):
        yield m, min(m + pd.offsets.MonthEnd(0), pd.Timestamp(b))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / "stocklists.html", f"{BASE}/cht/2/stockLists", kind="html")
    save(OUT / "2_stockinfo.ods", f"{BASE}/file/taifex/CHINESE/2/2_stockinfo.ods", kind="zip")
    for y in range(2019, 2026):
        save(OUT / "fut" / f"{y}_fut.zip", f"{BASE}/cht/3/futDataDown", dict(down_type=2, his_year=y), "zip")
    for m, e in months("2026-01-01"):
        save(OUT / "fut" / f"{m:%Y_%m}.csv", f"{BASE}/cht/3/futDataDown",
             dict(down_type=1, commodity_id="all", commodity_id2="", queryStartDate=f"{m:%Y/%m/%d}",
                  queryEndDate=f"{e:%Y/%m/%d}"))
    for m, e in months():
        save(OUT / "pcratio" / f"{m:%Y_%m}.csv", f"{BASE}/cht/3/pcRatioDown",
             dict(down_type="", queryStartDate=f"{m:%Y/%m/%d}", queryEndDate=f"{e:%Y/%m/%d}"))
    for y in range(2019, 2026):
        save(OUT / "opt" / f"{y}_opt.zip", f"{BASE}/cht/3/optDataDown", dict(down_type=2, his_year=y), "zip")
    for m, e in months("2026-01-01"):
        save(OUT / "opt" / f"{m:%Y_%m}_TXO.csv", f"{BASE}/cht/3/dlOptDataDown",
             dict(down_type=1, commodity_id="TXO", commodity_id2="", queryStartDate=f"{m:%Y/%m/%d}",
                  queryEndDate=f"{e:%Y/%m/%d}"))
    for q in pd.date_range("2019-12-01", "2026-08-31", freq="3MS"):
        e = min(q + pd.offsets.MonthEnd(3), pd.Timestamp("2026-08-31"))
        for kind, ep in (("fut", "dlLargeTraderFutDown"), ("opt", "dlLargeTraderOptDown")):
            save(OUT / f"largetrader_{kind}" / f"{q:%Y_%m}.csv", f"{BASE}/cht/3/{ep}",
                 dict(queryStartDate=f"{q:%Y/%m/%d}", queryEndDate=f"{e:%Y/%m/%d}"))
    save(OUT / "vix" / "202608new.txt", f"{BASE}/file/taifex/Dailydownload/vix/log2data/202608new.txt", kind="txt")
    # tick data of the last 30 trading days (only for the 08:45 timing check)
    raw, _ = request(f"{BASE}/cht/3/dlFutPrevious30DaysSalesData")
    if raw:
        for u in sorted(set(re.findall(r"https://www\.taifex\.com\.tw/file/taifex/Dailydownload/DailydownloadCSV/"
                                       r"Daily_\d{4}_\d\d_\d\d\.zip", raw.decode("utf-8", "replace")))):
            save(OUT / "ticks" / u.rsplit("/", 1)[1], u, kind="zip")
    print("done", flush=True)


if __name__ == "__main__":
    main()
