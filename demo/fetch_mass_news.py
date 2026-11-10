"""Download mass-audience Taiwan news into dataset/news/<source>/ (TWStock, private research, not committed).

- setn: 三立新聞網 (SETN) articles by their sequential id (the sitemaps hold only the last days), from about 2020-01 to
  the newest id of the news sitemap. robots.txt allows all crawlers, AI crawlers included (checked 2026-11-07).
  Per article: id, the publication time as shown on the page (Taipei time; the "+00:00" of its JSON-LD is wrong),
  section (articleSection), title, text (div#newsContent). Missing ids redirect to an error page and are skipped.
  One request a second, a descriptive User-Agent; one file per 1,000 ids (dataset/news/setn/NNNN.jsonl.gz, written
  to a temporary name and renamed), so the run resumes. When the network is down (no route, DNS failure) it waits
  and retries for up to an hour instead of stopping.

- ettoday: ETtoday 新聞雲, every day 2019-12-01 .. 2026-08-31 from the day list (news-list-YYYY-MM-DD-0.htm, newest 100)
  and the "show_roll.php" POST behind its infinite scroll (10 more per request, walked until the previous day). The
  full list is kept (published as listed, Taipei time; category tag; title; url); bodies (div.story, with the page's
  article:published_time, +08:00, and articleSection) only for the tags in ET_BODY. One file per day
  (dataset/news/ettoday/YYYY/YYYYMMDD.jsonl.gz).
- ltn: 自由時報 (news.ltn.com.tw and ec.ltn.com.tw 自由財經) by article id. All LTN sites share one id sequence and
  news.ltn.com.tw/news/politics/breakingnews/<id> redirects to the right site and section, so one request per id
  gives url, site and section of every id; bodies (JSON-LD datePublished, +08:00) are fetched for ec and the news
  sections in LTN_NEWS. Its list pages stop after about 500 items and its search after a few pages, so ids are the
  complete route. One file per 1,000 ids (dataset/news/ltn/NNNN.jsonl.gz).
- chinatimes: 中時新聞網 via its all-article sitemaps (sitemap_article_N.xml, N from 1 = 2009 to about 1,023 now, about
  a week each). Per article: id (14 digits, date first), url, category code; the body page (time shown on the page,
  Taipei time; articleSection; source 中時/工商/旺報) is fetched once for each new code (to learn its section name, kept
  in codes.json) and then for codes whose section matches CT_KEEP. One file per sitemap.
- chosun: 朝鮮日報 via the per-section Arc sitemaps (100 a page, offsets up to 9,900: the 10,000 most recently modified
  URLs of each section, so mostly the last months, e.g. economy from about 2026-04, politics from 2025-10, plus a few
  older ones; /search is disallowed by robots.txt, the old news.chosun.com lists (/svc) too), so older years are missing.
  Bodies (Fusion.globalContent JSON; first_publish_date in UTC, stored as published = Taipei time UTC+8 and
  published_utc) for the sections in CH_BODY. One file per month.
All sources: one request a second, the same User-Agent, the `User-agent: *` rules of each host's robots.txt obeyed;
403/429 back off and then stop the run. Dcard (Cloudflare challenge, 403) and CMoney 爆料同學會 (post lists only through
its token API; ids ~50k a day) are not collected.

    python demo/fetch_mass_news.py setn [first_id last_id]
    python demo/fetch_mass_news.py ettoday|chosun [first_day last_day]
    python demo/fetch_mass_news.py ltn [first_id last_id]
    python demo/fetch_mass_news.py chinatimes [first_day last_day [first_sitemap last_sitemap]]
"""

import datetime as dt
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

OUT = Path(__file__).resolve().parents[1] / "dataset" / "news"
UA = {"User-Agent": "TWStock-research/0.1 (private academic study)"}
PAUSE = 1.0


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


NOREDIR = urllib.request.build_opener(_NoRedirect)
ROBOTS = {}


def fetch(url, tries=8, data=None, follow=True):
    """the page text and the final URL; None for 404. Network outages are waited out (up to an hour); 403/429 back off
    (1, 2, 4, 8 min) and then stop the run. With follow=False a redirect gives ("", its target)."""
    k, waited, refused = 0, 0, 0
    while True:
        try:
            req = urllib.request.Request(url, data=data.encode() if data else None, headers=UA)
            r = (urllib.request.urlopen if follow else NOREDIR.open)(req, timeout=60)
            body = r.read().decode("utf-8", "replace")
            time.sleep(PAUSE)
            return body, r.geturl()
        except urllib.error.HTTPError as e:
            time.sleep(PAUSE)
            if e.code in (404, 410):
                return None, url
            if not follow and e.code in (301, 302, 303, 307, 308):
                return "", urllib.parse.urljoin(url, e.headers.get("Location", ""))
            if e.code in (403, 429):
                refused += 1
                if refused > 4:
                    raise RuntimeError(f"refused ({e.code}) {url}")
                print(f"  {e.code} {url}, backing off {60 * 2 ** (refused - 1)} s", flush=True)
                time.sleep(60 * 2 ** (refused - 1))
                continue
            k += 1
            if e.code >= 500:  # deleted articles answer 500: one retry after 3 s, then skip
                if k >= 2:
                    print(f"  skip {url}: {e}", flush=True)
                    return None, url
                time.sleep(3)
                continue
            print(f"  retry {k}: {e} {url}", flush=True)
            if k >= tries:
                raise RuntimeError(url)
            time.sleep(10 * k)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            reason = str(getattr(e, "reason", e))
            if ("unreachable" in reason or "name resolution" in reason) and waited < 3600:
                print(f"  network down ({reason}), waiting 120 s", flush=True)
                time.sleep(120)
                waited += 120
                continue
            k += 1
            print(f"  retry {k}: {e} {url}", flush=True)
            if k >= tries:
                raise RuntimeError(url)
            time.sleep(10 * k)


def clean(s):
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", s or "", flags=re.S)
    s = re.sub(r"<br\s*/?>|</p>", "\n", s)
    s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    return "\n".join(x for x in (re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in s.split("\n")) if x)


def setn_article(i):
    s, final = fetch(f"https://www.setn.com/news/{i}")
    if s is None or "page-error" in final:
        return None
    pub = re.search(r'"datePublished":\s*"([^"]+)"', s)
    sec = re.search(r'"articleSection":\s*"([^"]*)"', s)
    title = re.search(r'<meta property="og:title" content="([^"]*)"', s)
    a = s.find('id="newsContent"')
    body = s[a:s.find("</article>", a)] if a >= 0 else ""
    return dict(id=i, published=pub.group(1).split(" +")[0] if pub else None, section=sec.group(1) if sec else None,
                title=html.unescape(title.group(1)).split(" | ")[0] if title else "", text=clean(body[17:]))


def newest_setn():
    s, _ = fetch("https://www.setn.com/sitemapGoogleNews.xml")
    return max(int(x) for x in re.findall(r"setn\.com/[Nn]ews/(\d+)", s or ""))


def run_setn(first, last):
    out = OUT / "setn"
    out.mkdir(parents=True, exist_ok=True)
    t0, n_done = time.time(), 0
    for b in range(first // 1000, last // 1000 + 1):
        path = out / f"{b:04d}.jsonl.gz"
        if path.exists():
            continue
        rows = [r for r in (setn_article(i) for i in range(max(b * 1000, first), min(b * 1000 + 1000, last + 1))) if r]
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(gzip.compress("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()))
        tmp.rename(path)
        n_done += 1
        dates = sorted(r["published"] for r in rows if r["published"])
        print(f"setn block {b}: {len(rows)} articles, {dates[0] if dates else '-'} .. {dates[-1] if dates else '-'}, "
              f"{(time.time() - t0) / n_done:.0f} s per block", flush=True)



def robots_ok(url):
    """our User-agent is not named in any robots.txt, so the `User-agent: *` group decides (longest match wins)."""
    host = urllib.parse.urlsplit(url).netloc
    if host not in ROBOTS:
        s, _ = fetch(f"https://{host}/robots.txt")
        rules, star, prev_ua = [], False, False
        for ln in (s or "").splitlines():
            k, _, v = ln.split("#")[0].partition(":")
            k, v = k.strip().lower(), v.strip()
            if k == "user-agent":
                star = (star and prev_ua) or v == "*"
                prev_ua = True
                continue
            prev_ua = False
            if star and k in ("allow", "disallow") and v:
                rules.append((len(v), k == "allow", re.compile(re.escape(v).replace(r"\*", ".*").replace(r"\$", "$"))))
        ROBOTS[host] = sorted(rules, key=lambda r: -r[0])
    path = urllib.parse.urlsplit(url).path or "/"
    return next((allow for _, allow, rx in ROBOTS[host] if rx.match(path)), True)


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(gzip.compress("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()))
    tmp.rename(path)


def meta(s, rx):
    m = re.search(rx, s)
    return html.unescape(m.group(1)).strip() if m else None


def days(d0, d1):
    d = dt.date.fromisoformat(d0)
    while d <= dt.date.fromisoformat(d1):
        yield d
        d += dt.timedelta(days=1)


# ---- ETtoday ----
ET_BODY = {"財經", "政治", "國際", "大陸", "生活", "社會", "房產雲", "AI科技", "3C家電", "保險", "ESG", "軍武", "法律", "消費"}


def et_items(s):
    return [dict(published=a.replace("/", "-"), category=b, url=c, title=clean(d)) for a, b, c, d in re.findall(
        r'<span class="date">([^<]+)</span><em class="tag[^"]*">([^<]*)</em><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', s)]


def et_article(url):
    s, final = fetch(url)
    if s is None:
        return {}
    a = s.find('itemprop="articleBody"')
    e = s.find("<!--本文 結束-->", a)
    body = s[s.find(">", a) + 1:e if e > 0 else a + 200000] if a >= 0 else ""
    body = re.sub(r'<div class="ad_readmore">.*?</div>', " ", body, flags=re.S)
    return dict(url=final, published_page=meta(s, r'"datePublished":\s*"([^"]+)"'),
                section=meta(s, r'"articleSection":\s*"([^"]*)"'), text=clean(body))


def run_ettoday(d0, d1):
    for day in days(d0, d1):
        path = OUT / "ettoday" / f"{day:%Y}" / f"{day:%Y%m%d}.jsonl.gz"
        if path.exists():
            continue
        t0, ymd, n_req = time.time(), f"{day:%Y/%m/%d}", 1
        s, _ = fetch(f"https://www.ettoday.net/news/news-list-{day:%Y-%m-%d}-0.htm")
        rows = et_items(s or "")
        for off in range(1, 400):  # the day page holds the newest 100, then 10 per "show_roll" request
            s, _ = fetch("https://www.ettoday.net/show_roll.php",
                         data=f"offset={off}&tPage=3&tFile={day:%Y%m%d}.xml&tOt=0&tSi=100&tAr=0")
            n_req += 1
            more = et_items(s or "")
            rows += more
            if not more or min(r["published"] for r in more) < ymd.replace("/", "-"):
                break
        seen, keep = set(), []
        for r in rows:
            if r["published"].startswith(ymd.replace("/", "-")) and r["url"] not in seen:
                seen.add(r["url"])
                keep.append(r)
        for r in keep:
            r["id"] = int(re.search(r"(\d+)(?:\.htm)?$", r["url"]).group(1))
            if r["category"] in ET_BODY and robots_ok(r["url"]):
                r.update(et_article(r["url"]))
                n_req += 1
        save(path, keep)
        print(f"ettoday {day}: {len(keep)} listed, {sum('text' in r for r in keep)} bodies, {n_req} requests, "
              f"{time.time() - t0:.0f} s", flush=True)


# ---- 自由時報 (LTN) ----
LTN_NEWS = {"politics", "world", "society", "life", "focus"}  # sections of news.ltn.com.tw whose bodies we fetch
LTN_DROP = ("一手掌握經濟脈動", "點我訂閱自由財經", "不用抽 不用搶")


def ltn_body(s):
    a = s.find('class="text')
    e = min(x for x in (s.find('class="appE1121"', a), s.find("<!-- 投票箱 -->", a), a + 300000) if x > 0)
    paras = (clean(p) for p in re.findall(r"<p>(.*?)</p>", s[a:e], flags=re.S))
    return "\n".join(p for p in paras if p and not p.startswith(LTN_DROP))


def ltn_article(i):
    s, url = fetch(f"https://news.ltn.com.tw/news/politics/breakingnews/{i}", follow=False)
    if s is None:
        return None
    host = urllib.parse.urlsplit(url).netloc.split(".")[0]
    m = re.search(r"/news/([a-z]+)/breakingnews/", url)
    row = dict(id=i, url=url, site=host, category=m.group(1) if m else host)
    if s == "":  # redirect: fetch the body only for the wanted sections
        if not (host == "ec" or (host == "news" and row["category"] in LTN_NEWS)):
            return row
        s, url = fetch(url)
        if not s:
            return row
    row.update(published=meta(s, r'"datePublished":\s*"([^"]+)"'), section=meta(s, r'"articleSection":\s*"([^"]*)"'),
               title=meta(s, r'<meta property="og:title" content="([^"]*)"') or meta(s, r"<title>([^<]*)"),
               text=ltn_body(s))
    row["title"] = re.sub(r" - (自由財經|[^-]+ - 自由時報電子報)$", "", row["title"] or "")
    return row


def run_ltn(first, last):
    out = OUT / "ltn"
    t0, n_done = time.time(), 0
    for b in range(first // 1000, last // 1000 + 1):
        path = out / f"{b:04d}.jsonl.gz"
        if path.exists():
            continue
        rows = [r for r in (ltn_article(i) for i in range(max(b * 1000, first), min(b * 1000 + 1000, last + 1))) if r]
        save(path, rows)
        n_done += 1
        dates = sorted(r["published"] for r in rows if r.get("published"))
        print(f"ltn block {b}: {len(rows)} ids, {sum('text' in r for r in rows)} bodies, "
              f"{dates[0] if dates else '-'} .. {dates[-1] if dates else '-'}, {(time.time() - t0) / n_done:.0f} s per block",
              flush=True)


# ---- 中時 (chinatimes) ----
CT_KEEP = ("政治", "財經", "國際", "兩岸", "社會", "生活", "科技", "證券", "要聞", "全球", "產業", "金融", "政經", "股", "經濟",
           "理財", "房", "稅", "大陸", "中國", "焦點")
CT_SITEMAP = "https://www.chinatimes.com/sitemaps/article_sitemaps/sitemap_article_{}.xml"


def ct_sitemap(n):
    s, _ = fetch(CT_SITEMAP.format(n))
    return [dict(id=int(i), date=i[:8], kind=k, code=c, url=f"https://www.chinatimes.com/{k}/{i}-{c}")
            for k, i, c in re.findall(r"<loc>https://www\.chinatimes\.com/([a-z]+)/(\d{14})-(\d+)", s or "")]


def ct_article(url):
    s, _ = fetch(url)
    if s is None:
        return {}
    a = s.find('class="article-body"')
    e = min(x for x in (s.find('id="donate-form-container"', a), s.find("<!--推廣連結開始-->", a),
                        s.find("<!--主題標籤開始-->", a), a + 300000) if x > 0)
    sec = meta(s, r'"articleSection":\s*"([^"]*)"')
    title = meta(s, r'<meta property="og:title"[^>]*content="([^"]*)"') or ""
    return dict(published=meta(s, r'<time datetime="([^"]+)"'), section=sec, source=meta(s, r'<div class="source">([^<]*)'),
                title=title[:-len(f" - {sec}")] if sec and title.endswith(f" - {sec}") else title,
                text=clean(s[s.find(">", a) + 1:e]) if a >= 0 else "")


def run_chinatimes(d0, d1, first=None, last=None):
    out = OUT / "chinatimes"
    out.mkdir(parents=True, exist_ok=True)
    cpath = out / "codes.json"
    codes = json.loads(cpath.read_text()) if cpath.exists() else {}  # category code -> section name, learnt from a page
    lo, hi = d0.replace("-", ""), d1.replace("-", "")
    if first is None:  # sitemaps run oldest (1) to newest, about a week each: bisect for the first one reaching d0
        a, z = 1, 1100
        while a < z:
            m = (a + z) // 2
            rows = ct_sitemap(m)
            if rows and max(r["date"] for r in rows) < lo:
                a = m + 1
            else:
                z = m
        first = a
        print(f"chinatimes: first sitemap {first}", flush=True)
    t0, n_done = time.time(), 0
    for n in range(first, (last or 100000) + 1):
        path = out / f"{n:04d}.jsonl.gz"
        if path.exists():
            continue
        rows = ct_sitemap(n)
        if not rows:
            print(f"chinatimes sitemap {n}: empty, stop", flush=True)
            break
        if min(r["date"] for r in rows) > hi:
            print(f"chinatimes sitemap {n}: past {d1}, stop", flush=True)
            break
        byid = {}
        for r in rows:
            if lo <= r["date"] <= hi:
                byid.setdefault(r["id"], r).setdefault("codes", []).append(r["code"])
        for r in byid.values():
            sec = codes.get(r["code"])
            if sec is None or (any(k in sec for k in CT_KEEP) and "特刊" not in sec):
                r.update(ct_article(r["url"]))
                if sec is None:
                    codes[r["code"]] = r.get("section") or ""
                    tmp = cpath.with_name("codes.json.tmp")
                    tmp.write_text(json.dumps(codes, ensure_ascii=False, indent=0))
                    tmp.rename(cpath)
            r.setdefault("section", codes.get(r["code"]))
        save(path, list(byid.values()))
        n_done += 1
        dates = sorted(r["date"] for r in byid.values())
        print(f"chinatimes sitemap {n}: {len(byid)} articles, {sum('text' in r for r in byid.values())} bodies, "
              f"{dates[0] if dates else '-'} .. {dates[-1] if dates else '-'}, {(time.time() - t0) / n_done:.0f} s per sitemap",
              flush=True)


# ---- 朝鮮日報 (Chosun) ----
CH_CATS = ["economy", "politics", "international", "national", "opinion", "medical", "sports", "culture-life",
           "entertainments"]
CH_BODY = {"economy", "politics", "international", "national", "opinion"}


def ch_article(url):
    s, _ = fetch(url)
    m = re.search(r"Fusion\.globalContent=(\{.*?\});Fusion\.", s or "", flags=re.S)
    if not m:
        return {}
    d = json.loads(m.group(1))
    utc = d.get("first_publish_date") or d.get("display_date")
    t = dt.datetime.fromisoformat(utc.replace("Z", "+00:00")) if utc else None
    sec = (d.get("taxonomy") or {}).get("primary_section") or {}
    return dict(published=(t + dt.timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S") if t else None, published_utc=utc,
                section=sec.get("name"), section_path=sec.get("_id"),
                title=((d.get("headlines") or {}).get("basic") or ""),
                text="\n".join(clean(e.get("content")) for e in d.get("content_elements", []) if e.get("type") == "text"))


def run_chosun(d0, d1):
    out = OUT / "chosun"
    lpath = out / "list.jsonl.gz"
    if not lpath.exists():  # the per-section sitemaps page by 100 up to an offset of 9,900 (newest lastmod first)
        rows, seen = [], set()
        for cat in CH_CATS:
            for off in range(0, 10000, 100):
                s, _ = fetch(f"https://www.chosun.com/arc/outboundfeeds/sitemap/category/{cat}/?outputType=xml&from={off}")
                locs = re.findall(r"<loc>(https://www\.chosun\.com/([a-z_-]+)/[^<]*?(\d{4})/(\d\d)/(\d\d)/([A-Z0-9]+)/)</loc>",
                                  s or "")
                for u, c, y, mo, d, i in locs:
                    if i not in seen:
                        seen.add(i)
                        rows.append(dict(id=i, url=u, category=c, date=f"{y}-{mo}-{d}"))
                if not locs:
                    break
            print(f"chosun list {cat}: {len(rows)} urls so far, oldest {min(r['date'] for r in rows)}", flush=True)
        save(lpath, rows)
    rows = [json.loads(x) for x in gzip.decompress(lpath.read_bytes()).decode().splitlines()]
    rows = sorted((r for r in rows if d0 <= r["date"] <= d1), key=lambda r: r["date"])
    for month in sorted({r["date"][:7] for r in rows}):
        path = out / f"{month.replace('-', '')}.jsonl.gz"
        if path.exists():
            continue
        t0, mrows = time.time(), [r for r in rows if r["date"][:7] == month]
        for r in mrows:
            if r["category"] in CH_BODY and robots_ok(r["url"]):
                r.update(ch_article(r["url"]))
        save(path, mrows)
        print(f"chosun {month}: {len(mrows)} listed, {sum('text' in r for r in mrows)} bodies, {time.time() - t0:.0f} s",
              flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "setn":
        lo = int(sys.argv[2]) if len(sys.argv) > 2 else 660000
        hi = int(sys.argv[3]) if len(sys.argv) > 3 else newest_setn()
        print(f"setn ids {lo} .. {hi}", flush=True)
        run_setn(lo, hi)
    elif sys.argv[1] == "ettoday":
        run_ettoday(*(sys.argv[2:4] if len(sys.argv) > 3 else ("2019-12-01", "2026-08-31")))
    elif sys.argv[1] == "ltn":
        run_ltn(*(map(int, sys.argv[2:4]) if len(sys.argv) > 3 else (2995000, 5559000)))
    elif sys.argv[1] == "chinatimes":
        a = sys.argv[2:4] if len(sys.argv) > 3 else ("2019-12-01", "2026-08-31")
        run_chinatimes(*a, *(int(x) for x in sys.argv[4:6]))
    elif sys.argv[1] == "chosun":
        run_chosun(*(sys.argv[2:4] if len(sys.argv) > 3 else ("2019-12-01", "2026-08-31")))
