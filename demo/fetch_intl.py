"""Download daily bars of Korean and Japanese large caps for demo/stock_pool.py into dataset/intl/ (not committed).

Yahoo chart API (unofficial), 30 KOSPI and 30 Nikkei large caps, 2019-12 .. 2026-08, with the adjusted close so that
open and close can be put on the same split/dividend basis.

    python demo/fetch_intl.py
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "dataset" / "intl"
START, END = "2019-12-01", "2026-09-01"
KR = ["005930", "000660", "373220", "207940", "005380", "000270", "068270", "005490", "035420", "051910", "006400",
      "105560", "055550", "012330", "035720", "028260", "066570", "003550", "096770", "017670", "034730", "015760",
      "086790", "032830", "011200", "009150", "018260", "010130", "033780", "003670"]
JP = ["7203", "6758", "8306", "6861", "9984", "6098", "8035", "9432", "4063", "9983", "6501", "8316", "7974", "4568",
      "8058", "6367", "4519", "6902", "7267", "8001", "6981", "6857", "4502", "6594", "8411", "6954", "7741", "9433",
      "6273", "6723"]


def yahoo(sym, path):
    p1, p2 = int(pd.Timestamp(START).timestamp()), int(pd.Timestamp(END).timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym)}?period1={p1}&period2={p2}"
           "&interval=1d&events=div%2Csplit")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    res = json.loads(urllib.request.urlopen(req, timeout=60).read())["chart"]["result"][0]
    q, off = res["indicators"]["quote"][0], res["meta"]["gmtoffset"]
    f = pd.DataFrame({"date": pd.to_datetime([t + off for t in res["timestamp"]], unit="s").strftime("%Y-%m-%d"),
                      "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
                      "volume": q["volume"], "adjclose": res["indicators"]["adjclose"][0]["adjclose"]})
    f = f.dropna(subset=["close", "open"]).drop_duplicates("date", keep="last")
    f.to_csv(path, index=False)
    return len(f)


if __name__ == "__main__":
    for mkt, codes, suf in (("kr", KR, ".KS"), ("jp", JP, ".T")):
        (OUT / mkt).mkdir(parents=True, exist_ok=True)
        for c in codes:
            print(f"{mkt} {c}: {yahoo(c + suf, OUT / mkt / f'{c}.csv')} rows", flush=True)
            time.sleep(1)
