"""Download political and policy news of the US, Taiwan, China, Japan and Korea into dataset/politics/ (not committed).

- fedreg: Federal Register public-inspection documents (executive orders, BIS export-control rules, USTR notices, ...)
  by the day they went on public inspection, with filed_at (to the minute, with the zone; usually 08:45 or 16:15 ET,
  a day before publication). Official API, keyless -> dataset/politics/fedreg/YYYYMM.jsonl.gz
- truth: Donald Trump's Truth Social posts from the trumpstruth.org archive RSS (robots.txt: Disallow empty), from
  2022-02-14, pubDate to the second -> dataset/politics/truth/YYYYMM.jsonl.gz
- mfa: the transcripts of the regular press conferences of the PRC Ministry of Foreign Affairs (mfa.gov.cn; about
  15:00 Beijing time, so usable from the next Taiwan session) -> dataset/politics/mfa/YYYYMM.jsonl.gz
- mac: the press releases of Taiwan's Mainland Affairs Council (data.gov.tw dataset 55229, one CSV; date only)
  -> dataset/politics/mac.csv
- koreakr: the Korean government press releases (보도자료) listed in the korea.kr sitemap (robots.txt: Allow /)
  -> dataset/politics/koreakr.jsonl.gz
- gdelt: GDELT 2.0 event files (every 15 minutes, DATEADDED to the 15 minutes, UTC), only the events whose two
  actors are both among USA, CHN, TWN, JPN, KOR, PRK, with the CAMEO code, quad class, Goldstein scale, tone, counts and
  the source URL -> dataset/politics/gdelt/YYYYMMDD.csv.gz (resumable per day)
Private research use only; the data stay in dataset/ and are never published. One connection per source; one request
every PAUSE seconds (3 s for web pages, 1 s for the static GDELT files); failed requests wait 10, 20, ... s and the
run stops after 8 failures in a row. Not fetched: the MND PLA-activity reports (robots.txt: Disallow / for all but
Googlebot).

    python demo/fetch_politics.py fedreg|truth|mfa|mac|koreakr|gdelt
"""

import csv
import gzip
import html
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "dataset" / "politics"
START, END = "2019-12-01", "2026-08-31"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
PAUSE = 3.0
COUNTRIES = {"USA", "CHN", "TWN", "JPN", "KOR", "PRK"}


def fetch(url, tries=8, raw=False):
    for k in range(tries):
        try:
            res = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()
            time.sleep(PAUSE)
            return res if raw else res.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                time.sleep(PAUSE)
                return None
            print(f"  retry {k + 1}: {e} {url}", flush=True)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"  retry {k + 1}: {e} {url}", flush=True)
        time.sleep(10 * (k + 1))
    raise RuntimeError(url)


def clean(s):
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", s or "", flags=re.S)
    s = re.sub(r"<br\s*/?>|</p>", "\n", s)
    return re.sub(r"[ \t\r\f\v]+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def save_jsonl(path, rows):
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()))
    tmp.rename(path)


def months():
    return pd.date_range(START, END, freq="MS")


def run_fedreg():
    out = OUT / "fedreg"
    out.mkdir(parents=True, exist_ok=True)
    for m in months():
        path = out / f"{m:%Y%m}.jsonl.gz"
        if path.exists():
            continue
        rows = []
        for day in pd.date_range(m, m + pd.offsets.MonthEnd(0)):
            if day.weekday() >= 5:
                continue
            q = urllib.parse.urlencode({"conditions[available_on]": f"{day:%Y-%m-%d}", "per_page": 1000})
            s = fetch(f"https://www.federalregister.gov/api/v1/public-inspection-documents.json?{q}")
            for d in json.loads(s).get("results", []) if s else []:
                rows.append({k: d.get(k) for k in ("document_number", "filed_at", "filing_type", "type",
                                                   "agency_names", "title", "subject_1", "subject_2", "toc_subject",
                                                   "toc_doc", "publication_date", "html_url", "pdf_url")})
        save_jsonl(path, rows)
        print(f"fedreg {m:%Y-%m}: {len(rows)}", flush=True)


def run_truth():
    out = OUT / "truth"
    out.mkdir(parents=True, exist_ok=True)
    for m in months():
        if m < pd.Timestamp("2022-02-01"):
            continue
        path = out / f"{m:%Y%m}.jsonl.gz"
        if path.exists():
            continue
        rows = []
        for a in pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="2D"):  # 2-day chunks stay below 500 posts
            b = min(a + pd.Timedelta(days=1), m + pd.offsets.MonthEnd(0))
            q = urllib.parse.urlencode(dict(start_date=f"{a:%Y-%m-%d}", end_date=f"{b:%Y-%m-%d}", limit=500))
            s = fetch(f"https://www.trumpstruth.org/feed?{q}")
            items = ET.fromstring(s.encode()).findall("./channel/item") if s else []
            if len(items) >= 500:
                print(f"  truth {a:%Y-%m-%d}: 500 items, may be truncated", flush=True)
            for it in items:
                rows.append(dict(title=clean(it.findtext("title")), text=clean(it.findtext("description")),
                                 pubdate=it.findtext("pubDate"), link=it.findtext("link"), guid=it.findtext("guid")))
        rows = list({r["guid"] or r["link"]: r for r in rows}.values())
        save_jsonl(path, rows)
        print(f"truth {m:%Y-%m}: {len(rows)}", flush=True)


def run_mfa():
    out = OUT / "mfa"
    out.mkdir(parents=True, exist_ok=True)
    base = "https://www.mfa.gov.cn/web/wjdt_674879/fyrbt_674889/"
    links, page = {}, 0
    while True:
        s = fetch(base + ("index.shtml" if page == 0 else f"index_{page}.shtml"))
        found = re.findall(r'<a href="\./(\d{6}/t\d+_\d+\.shtml)"[^>]*>([^<]*记者会[^<]*)', s or "")
        if not found:
            break
        for href, title in found:
            m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title)
            if m:
                links[href] = (f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}", title.strip())
        oldest = min(d for d, _ in links.values())
        print(f"mfa list page {page}: {len(links)} links, oldest {oldest}", flush=True)
        if oldest < START:
            break
        page += 1
    by_month = {}
    for href, (d, title) in links.items():
        if START <= d <= END:
            by_month.setdefault(d[:7].replace("-", ""), []).append((d, href, title))
    for mon in sorted(by_month):
        path = out / f"{mon}.jsonl.gz"
        if path.exists():
            continue
        rows = []
        for d, href, title in sorted(by_month[mon]):
            s = fetch(base + href) or ""
            body = re.search(r'<div class="trs_editor_view[^"]*"[^>]*>(.*?)<div class="news-foot"', s, re.S)
            rows.append(dict(date=d, title=title, url=base + href, text=clean(body.group(1)) if body else ""))
        save_jsonl(path, rows)
        print(f"mfa {mon}: {len(rows)}", flush=True)


def run_mac():
    OUT.mkdir(parents=True, exist_ok=True)
    url = "https://www.mac.gov.tw/big5/data/" + urllib.parse.quote("55229_大陸委員會新聞稿.csv")
    (OUT / "mac.csv").write_bytes(fetch(url, raw=True))
    print(f"mac: {len(pd.read_csv(OUT / 'mac.csv'))} rows", flush=True)


def run_koreakr():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "koreakr.jsonl.gz"
    sm = fetch("https://www.korea.kr/sitemap_brief.xml")
    urls = list(dict.fromkeys(re.findall(r"<loc>(https://www\.korea\.kr/briefing/pressReleaseView\.do\?newsId=\d+)</loc>",
                                         sm)))
    print(f"koreakr: {len(urls)} press releases in the sitemap", flush=True)
    rows = []
    for i, u in enumerate(urls):
        s = fetch(u) or ""
        title = re.search(r'<meta property="og:title" content="([^"]*)"', s)
        date = re.search(r"(\d{4}\.\d{2}\.\d{2})", s)
        body = re.search(r'<div class="view_cont"[^>]*>(.*?)</div>\s*<div', s, re.S)
        rows.append(dict(url=u, title=html.unescape(title.group(1)) if title else "",
                         date=date.group(1) if date else None, text=clean(body.group(1))[:20000] if body else ""))
        if (i + 1) % 500 == 0:
            print(f"koreakr {i + 1}/{len(urls)}", flush=True)
            save_jsonl(path, rows)
    save_jsonl(path, rows)
    print(f"koreakr: {len(rows)}", flush=True)


GDELT_COLS = {0: "id", 1: "sqldate", 7: "a1_country", 17: "a2_country", 26: "event_code", 28: "root_code",
              29: "quad", 30: "goldstein", 31: "mentions", 32: "sources", 33: "articles", 34: "tone",
              59: "dateadded", 60: "url"}


def run_gdelt():
    global PAUSE
    PAUSE = 1.0
    out = OUT / "gdelt"
    out.mkdir(parents=True, exist_ok=True)
    master = fetch("http://data.gdeltproject.org/gdeltv2/masterfilelist.txt")
    files = {}
    for line in master.splitlines():
        p = line.split()
        if len(p) == 3 and p[2].endswith(".export.CSV.zip"):
            stamp = p[2].rsplit("/", 1)[1][:14]
            if START.replace("-", "") <= stamp[:8] <= END.replace("-", ""):
                files.setdefault(stamp[:8], []).append(p[2])
    for day in sorted(files):
        path = out / f"{day}.csv.gz"
        if path.exists():
            continue
        rows = []
        for url in sorted(files[day]):
            raw = fetch(url, raw=True)
            if raw is None:
                continue
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                text = z.read(z.namelist()[0]).decode("utf-8", "replace")
            for rec in csv.reader(io.StringIO(text), delimiter="\t"):
                if len(rec) > 60 and rec[7] in COUNTRIES and rec[17] in COUNTRIES:
                    rows.append([rec[i] for i in GDELT_COLS])
        pd.DataFrame(rows, columns=list(GDELT_COLS.values())).to_csv(path, index=False, compression="gzip")
        print(f"gdelt {day}: {len(rows)} events from {len(files[day])} files", flush=True)


if __name__ == "__main__":
    {"fedreg": run_fedreg, "truth": run_truth, "mfa": run_mfa, "mac": run_mac, "koreakr": run_koreakr,
     "gdelt": run_gdelt}[sys.argv[1]]()
