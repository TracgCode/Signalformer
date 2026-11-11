"""Download the overnight data used by demo/stock_overnight.py into dataset/overnight/ (gitignored, not committed).

- TAIFEX TAIEX futures (TX) daily data, day (一般) and after-hours (盤後) sessions, every contract month, from the
  public download form (one month per request, a pause between requests);
- Yahoo daily bars (unofficial chart API) of the Taiwan ADRs and the overnight markets, dated by the exchange's local
  trading day.

    python demo/fetch_overnight.py
"""

import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "dataset" / "overnight"
START, END = "2019-12-01", "2026-09-01"
YAHOO = ["TSM", "UMC", "ASX", "^SOX", "^N225", "^KS11", "TWD=X", "2330.TW", "2303.TW", "3711.TW", "NQ=F", "ES=F"]
UA = {"User-Agent": "Mozilla/5.0"}


def taifex():
    frames = []
    for m in pd.date_range(START, END, freq="MS")[:-1]:
        a, b = m.strftime("%Y/%m/%d"), (m + pd.offsets.MonthEnd(0)).strftime("%Y/%m/%d")
        data = urllib.parse.urlencode(dict(down_type=1, commodity_id="TX", queryStartDate=a, queryEndDate=b)).encode()
        for attempt in range(4):
            raw = urllib.request.urlopen(urllib.request.Request("https://www.taifex.com.tw/cht/3/futDataDown", data,
                                                                UA), timeout=60).read().decode("cp950")
            if raw.startswith("交易日期"):
                break
            time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"TAIFEX {a}: {raw[:200]}")
        f = pd.read_csv(io.StringIO(raw), dtype=str, index_col=False)  # rows end with a trailing comma
        frames.append(f)
        print(f"TAIFEX {a[:7]}: {len(f)} rows", flush=True)
        time.sleep(3)
    f = pd.concat(frames, ignore_index=True)
    f.columns = [c.strip() for c in f.columns]
    f.to_csv(OUT / "taifex_tx.csv", index=False)


def bar_dates(res, sym):
    """Date of each daily bar in the exchange's own time zone (DST-aware). A fixed res["meta"]["gmtoffset"] is the
    offset of the fetch day only: series stamped 00:00 local (futures, FX) then get the previous day in the other
    DST regime."""
    tz = ZoneInfo(res["meta"]["exchangeTimezoneName"])
    d = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz)
    wkend = int((d.dayofweek >= 5).sum())
    if wkend:
        print(f"WARNING {sym}: {wkend} weekend-dated bars ({tz})", flush=True)
    return d.strftime("%Y-%m-%d")


def yahoo(sym):
    p1, p2 = int(pd.Timestamp(START).timestamp()), int(pd.Timestamp(END).timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym)}?period1={p1}&period2={p2}"
           "&interval=1d&events=div%2Csplit")
    res = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read())["chart"]
    res = res["result"][0]
    q = res["indicators"]["quote"][0]
    f = pd.DataFrame({"date": bar_dates(res, sym),
                      "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
                      "volume": q["volume"]})
    if "adjclose" in res["indicators"]:
        f["adjclose"] = res["indicators"]["adjclose"][0]["adjclose"]
    f = f.dropna(subset=["close"]).drop_duplicates("date", keep="last")
    f.to_csv(OUT / f"yahoo_{sym.replace('^', '').replace('=', '_')}.csv", index=False)
    print(f"Yahoo {sym}: {len(f)} rows, {f['date'].iloc[0]} .. {f['date'].iloc[-1]}", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for s in YAHOO:
        yahoo(s)
        time.sleep(1)
    taifex()
