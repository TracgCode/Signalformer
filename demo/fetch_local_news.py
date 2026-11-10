"""Download Japanese and Korean local-language disclosures and Korean English news into dataset/news/ (not committed).

- tdnet: Japanese timely disclosures (適時開示, TDnet) of the Japanese peers and the 30 Nikkei large caps, through the
  unofficial yanoshin WEB-API (keyless; one request per company; pubdate in JST to the minute)
  -> dataset/news/tdnet/CODE.csv
- kind: Korean exchange disclosures of the Korean peers and the 30 KOSPI large caps from KIND (kind.krx.co.kr, the
  search form; time in KST to the minute; title, submitter and the receipt number)
  -> dataset/news/kind/CODE.csv
- dart: the Korean disclosure list of OpenDART (FSS) for the same Korean companies (receipt date only, no time; used
  to check KIND for completeness), with the key read from keys/opendart outside the repo
  -> dataset/news/dart/CODE.csv
- kherald: every article of The Korea Herald (English) listed in its daily sitemaps (robots.txt: Allow /), with the
  title, the published time (to the second, with the zone), the description and the body text
  -> dataset/news/kherald/YYYYMMDD.jsonl.gz (resumable per day)
Private research use only; the data stay in dataset/ and are never published. One request every PAUSES[source]
seconds (3 s for the sites without a published limit); failed requests wait 10, 20, ... s and the run stops after 8
failures in a row.

    python demo/fetch_local_news.py tdnet|kind|dart|kherald [start YYYY-MM-DD] [end YYYY-MM-DD]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_intl import JP as JP_INTL, KR as KR_INTL  # noqa: E402
from fetch_peers import JP as JP_PEER, KR as KR_PEER  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "dataset" / "news"
START, END = "2019-12-01", "2026-08-31"
PAUSE = 1.5
# seconds between requests per source, kept well below each site's limits (KIND and the Korea Herald publish none, so
# one request every 3 s; OpenDART allows about 20,000 a day and this needs about 1,500; yanoshin about 50 in all)
PAUSES = {"tdnet": 3.0, "kind": 3.0, "dart": 1.5, "kherald": 3.0}
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
JP = sorted(set(JP_INTL) | set(JP_PEER))
KR = sorted(set(KR_INTL) | set(KR_PEER))


def fetch(url, data=None, headers=None, tries=8):
    body = urllib.parse.urlencode(data).encode() if data else None
    for k in range(tries):
        try:
            res = urllib.request.urlopen(urllib.request.Request(url, data=body, headers=dict(UA, **(headers or {}))),
                                         timeout=90).read()
            time.sleep(PAUSE)
            return res.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                time.sleep(PAUSE)
                return None
            print(f"  retry {k + 1}: {e}", flush=True)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"  retry {k + 1}: {e}", flush=True)
        time.sleep(10 * (k + 1))
    raise RuntimeError(url)


def clean(s):
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", s or "", flags=re.S)
    s = re.sub(r"<br\s*/?>|</p>", "\n", s)
    return re.sub(r"[ \t\r\f\v]+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def run_tdnet():
    (OUT / "tdnet").mkdir(parents=True, exist_ok=True)
    for c in JP:
        path = OUT / "tdnet" / f"{c}.csv"
        if path.exists():
            continue
        d = json.loads(fetch(f"https://webapi.yanoshin.jp/webapi/tdnet/list/{c}.json?limit=5000"))
        rows = [x["Tdnet"] for x in d["items"]]
        df = pd.DataFrame([dict(id=x["id"], pubdate=x["pubdate"], code=x["company_code"], name=x["company_name"],
                                title=x["title"], url=x["document_url"]) for x in rows])
        if len(df):
            df = df[(df["pubdate"] >= START) & (df["pubdate"] <= END + " 23:59")]
        df.to_csv(path, index=False)
        print(f"tdnet {c}: {len(df)} of {d['total_count']}", flush=True)


ROW = re.compile(r'<td class="first txc">\s*\d+\s*</td>\s*<td class="txc">([\d\- :]+)</td>.*?'
                 r"openDisclsViewer\('(\d+)'[^>]*title='([^']*)'.*?</td>\s*<td>([^<]*)</td>", re.S)


def run_kind():
    (OUT / "kind").mkdir(parents=True, exist_ok=True)
    for c in KR:
        path = OUT / "kind" / f"{c}.csv"
        if path.exists():
            continue
        rows = []
        for y in range(int(START[:4]), int(END[:4]) + 1):
            a, b = max(f"{y}-01-01", START), min(f"{y}-12-31", END)
            page = 1
            while True:
                s = fetch("https://kind.krx.co.kr/disclosure/details.do", data=dict(
                    method="searchDetailsSub", forward="details_sub", currentPageSize=100, pageIndex=page,
                    orderMode=0, orderStat="D", repIsuSrtCd=f"A{c}", fromDate=a, toDate=b, lastReport="T"),
                    headers={"Referer": "https://kind.krx.co.kr/disclosure/details.do?method=searchDetailsMain"})
                got = ROW.findall(s)
                seen = {r["acptno"] for r in rows}
                new = [g_ for g_ in got if g_[1] not in seen]
                rows += [dict(time=t.strip(), acptno=n, title=html.unescape(ti), submitter=sb.strip(), code=c)
                         for t, n, ti, sb in new]
                if len(got) < 100 or not new or page >= 200:  # the site repeats the last page past the end
                    break
                page += 1
        df = pd.DataFrame(rows).drop_duplicates("acptno") if rows else pd.DataFrame()
        df.to_csv(path, index=False)
        print(f"kind {c}: {len(df)}", flush=True)


def run_dart():
    import io
    import zipfile
    key = (Path(__file__).resolve().parents[2] / "keys" / "opendart").read_text().strip()
    (OUT / "dart").mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={key}", timeout=120).read()
    xml = zipfile.ZipFile(io.BytesIO(raw)).read("CORPCODE.xml").decode("utf-8")
    corp = {}
    for blk in re.findall(r"<list>(.*?)</list>", xml, re.S):  # one entry per block (most have no stock code)
        cc, sc = re.search(r"<corp_code>(\d+)</corp_code>", blk), re.search(r"<stock_code>\s*(\d{6})\s*<", blk)
        if cc and sc:
            corp[sc.group(1)] = cc.group(1)
    for c in KR:
        path = OUT / "dart" / f"{c}.csv"
        if path.exists():
            continue
        rows, page, last = [], 1, 1
        while page <= last:
            q = urllib.parse.urlencode(dict(crtfc_key=key, corp_code=corp[c], bgn_de=START.replace("-", ""),
                                            end_de=END.replace("-", ""), page_no=page, page_count=100))
            d = json.loads(fetch(f"https://opendart.fss.or.kr/api/list.json?{q}"))
            if d["status"] != "000":
                print(f"dart {c}: status {d['status']} {d.get('message')}", flush=True)
                break
            rows += d["list"]
            last = d["total_page"]
            page += 1
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"dart {c}: {len(rows)}", flush=True)


def kh_article(url):
    s = fetch(url)
    if s is None:
        return None
    meta = lambda p: (re.search(rf'<meta (?:property|name)="{p}"\s+content="([^"]*)"', s) or [None, None])[1]
    m = re.search(r'<article class="news_content[^"]*" id="articleText"[^>]*>(.*?)</article>', s, re.S)
    return dict(url=url, published=meta("article:published_time"), title=html.unescape(meta("og:title") or ""),
                description=html.unescape(meta("og:description") or ""), text=clean(m.group(1)) if m else "")


def run_kherald(start, end):
    out = OUT / "kherald"
    out.mkdir(parents=True, exist_ok=True)
    for day in pd.date_range(start, end):
        path = out / f"{day:%Y%m%d}.jsonl.gz"
        if path.exists():
            continue
        sm = fetch(f"https://www.koreaherald.com/sitemap/articles/{day:%Y}/{day:%Y%m%d}") or ""
        urls = list(dict.fromkeys(re.findall(r"<loc>\s*(https://www\.koreaherald\.com/article/\d+)\s*</loc>", sm)))
        rows = [r for r in (kh_article(u) for u in urls) if r]
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()))
        tmp.rename(path)
        print(f"kherald {day:%Y-%m-%d}: {len(rows)}", flush=True)


if __name__ == "__main__":
    src = sys.argv[1]
    PAUSE = PAUSES[src]
    if src == "kherald":
        run_kherald(sys.argv[2] if len(sys.argv) > 2 else START, sys.argv[3] if len(sys.argv) > 3 else END)
    else:
        {"tdnet": run_tdnet, "kind": run_kind, "dart": run_dart}[src]()
