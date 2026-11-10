"""Build a short business description of each of the 100 stocks for the stock-identity embeddings
(demo/stock_itrans_id.py) into dataset/profiles.csv (not committed).

- TWSE / TPEx OpenAPI company basic data (keyless): name, short name, industry code;
- the lead section of the company's Chinese Wikipedia article (REST summary API, keyless, a descriptive User-Agent,
  one request per second; Taiwan variant), found from the full company name, then the short name; empty when absent.
The description is "<short name>（<code>），<industry>。<Wikipedia lead, at most 400 characters>".

    python demo/fetch_profiles.py
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "dataset"
UA = {"User-Agent": "TWStock-research/0.1 (private academic study)", "Accept-Language": "zh-tw"}
IND = {"01": "水泥", "02": "食品", "03": "塑膠", "04": "紡織纖維", "05": "電機機械", "06": "電器電纜", "08": "玻璃陶瓷",
       "09": "造紙", "10": "鋼鐵", "11": "橡膠", "12": "汽車", "14": "建材營造", "15": "航運", "16": "觀光餐旅",
       "17": "金融保險", "18": "貿易百貨", "20": "其他", "21": "化學", "22": "生技醫療", "23": "油電燃氣",
       "24": "半導體", "25": "電腦及週邊設備", "26": "光電", "27": "通信網路", "28": "電子零組件", "29": "電子通路",
       "30": "資訊服務", "31": "其他電子", "35": "綠能環保", "36": "數位雲端", "37": "運動休閒", "38": "居家生活"}


def get(url, tries=5):
    for k in range(tries):
        try:
            res = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
            time.sleep(1.0)
            return json.loads(res)
        except urllib.error.HTTPError as e:
            time.sleep(1.0)
            if e.code == 404:
                return None
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  retry {k + 1}: {e}", flush=True)
            time.sleep(10 * (k + 1))
    return None


def wiki(title):
    d = get("https://zh.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title))
    if not d or d.get("type") == "disambiguation":
        return None
    return d.get("extract") or None


def main():
    stocks = json.loads((ROOT / "info.json").read_text())["Top100@Y"]
    base = {}
    for d in get("https://openapi.twse.com.tw/v1/opendata/t187ap03_L"):
        base[d["公司代號"]] = (d["公司名稱"], d["公司簡稱"], d["產業別"])
    for d in get("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"):
        base.setdefault(d["SecuritiesCompanyCode"], (d["CompanyName"], d["CompanyAbbreviation"],
                                                     d["SecuritiesIndustryCode"]))
    rows = []
    for s in stocks:
        full, short, ind = base.get(s, ("", s, ""))
        names = [full.replace("股份有限公司", "").strip(), full, short]
        lead = None
        for n in dict.fromkeys(x for x in names if x):
            lead = wiki(n)
            if lead:
                break
        desc = f"{short}（{s}），{IND.get(ind, '')}業。" + (lead or "")[:400]
        rows.append(dict(code=s, name=full, short=short, industry=IND.get(ind, ind), wiki=bool(lead), desc=desc))
        print(f"{s} {short} {IND.get(ind, ind)} wiki={bool(lead)}", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "profiles.csv", index=False)


if __name__ == "__main__":
    main()
