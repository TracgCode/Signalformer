"""Download dated text (news and material announcements) for the E2 text features into dataset/news/ (not committed).

Two sources, one file per calendar day, resumable (days already on disk are skipped):
- mops: the whole market's material information (重大訊息) of a day from MOPS t05st02 (公開資訊觀測站), with the company
  code, the announcement date and time to the second, the subject, the clause and the full text.
  -> dataset/news/mops/YYYYMMDD.jsonl.gz
- cnyes: the Taiwan-stock news list of 鉅亨網 (category tw_stock, 30 items per page), with the publication time
  (unix seconds), the tagged stock codes, the title, the summary and the text with the HTML tags removed.
  -> dataset/news/cnyes/YYYYMMDD.jsonl.gz
- cnyes_wd: the same for the international-stock list of 鉅亨網 (category wd_stock: US, Europe and Asia, tagged with
  the foreign tickers). -> dataset/news/cnyes_wd/YYYYMMDD.jsonl.gz
Private research use only: the MOPS robots.txt disallows automated access and the cnyes terms forbid reproduction, so
the data stay in dataset/ and are never published. One request every PAUSE seconds per source.

    python demo/fetch_news.py mops|cnyes|cnyes_wd [start YYYY-MM-DD] [end YYYY-MM-DD]
"""

import gzip
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "dataset" / "news"
START, END = "2020-01-01", "2026-08-31"
PAUSE = 3.0
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def fetch(url, data=None, headers=None):
    h = dict(UA, **(headers or {}))
    body = urllib.parse.urlencode(data).encode() if data else None
    for attempt in range(8):
        try:
            res = urllib.request.urlopen(urllib.request.Request(url, data=body, headers=h), timeout=60).read()
            time.sleep(PAUSE)
            return res
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            wait = min(60 * 2 ** attempt, 1800)
            print(f"  {e}; retry in {wait} s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed: {url}")


def clean(s):
    s = html.unescape(html.unescape(s or ""))
    return re.sub(r"\s+\n", "\n", re.sub(r"<[^>]+>", "", s)).replace("\xa0", " ").strip()


def mops_day(day):
    data = dict(encodeURIComponent=1, step=1, step00=0, firstin=1, off=1, TYPEK="all", year=day.year - 1911,
                month=f"{day.month:02d}", day=f"{day.day:02d}")
    hdr = {"Referer": "https://mopsov.twse.com.tw/mops/web/t05st02", "X-Requested-With": "XMLHttpRequest"}
    for attempt in range(6):
        t = fetch("https://mopsov.twse.com.tw/mops/web/ajax_t05st02", data, hdr).decode("utf-8", "replace")
        if "公司當日重大訊息" in t or "查無" in t or "無符合" in t:
            break
        print(f"  {day.date()}: unexpected page ({len(t)} B, e.g. too frequent); wait", flush=True)
        time.sleep(min(120 * 2 ** attempt, 1800))
    else:
        raise RuntimeError(f"mops {day.date()}: no valid page")
    rows = []
    for tr in re.findall(r"<tr class='(?:even|odd)'>(.*?)</tr>", t, re.S):
        hv = {k: v for k, v in re.findall(r"name='(h\d+)' value='(.*?)'>", tr, re.S)}
        hv = {"h0" + k[-1]: v for k, v in hv.items()}
        tk = re.search(r'TYPEK.value="(\w+)"', tr)
        rows.append(dict(co=hv.get("h01"), name=hv.get("h00"), date=hv.get("h02"), time=hv.get("h03"),
                         subject=clean(hv.get("h04")), seq=hv.get("h05"), clause=hv.get("h06"),
                         fact_date=hv.get("h07"), text=clean(hv.get("h08")), typek=tk[1] if tk else None))
    return rows


def cnyes_day(day, cat="tw_stock"):
    t0 = int(pd.Timestamp(day, tz="Asia/Taipei").timestamp())
    rows, page, last = [], 1, 1
    while page <= last:
        q = urllib.parse.urlencode(dict(limit=30, page=page, startAt=t0, endAt=t0 + 86399))
        it = json.loads(fetch(f"https://api.cnyes.com/media/api/v1/newslist/category/{cat}?{q}"))["items"]
        last = it.get("last_page") or 1
        for x in it["data"]:
            rows.append(dict(id=x.get("newsId"), publish=x.get("publishAt"), stock=x.get("stock"),
                             market=[m.get("code") for m in (x.get("market") or [])], keyword=x.get("keyword"),
                             title=clean(x.get("title")), summary=clean(x.get("summary")),
                             text=clean(x.get("content"))))
        page += 1
    return rows


def main():
    src = sys.argv[1]
    start = sys.argv[2] if len(sys.argv) > 2 else START
    end = sys.argv[3] if len(sys.argv) > 3 else END
    get = {"mops": mops_day, "cnyes": cnyes_day, "cnyes_wd": lambda d: cnyes_day(d, "wd_stock")}[src]
    out = OUT / src
    out.mkdir(parents=True, exist_ok=True)
    for day in pd.date_range(start, end):
        path = out / f"{day:%Y%m%d}.jsonl.gz"
        if path.exists():
            continue
        rows = get(day)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()))
        tmp.rename(path)
        print(f"{src} {day.date()}: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
