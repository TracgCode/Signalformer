"""Download the US / Japan / Korea data on the foreign peers of the 50-stock subset into dataset/peers/ (not committed).

- yahoo: daily bars (unofficial chart API, as in demo/fetch_intl.py) of the peers below and four market series,
  2019-12 .. 2026-08 -> dataset/peers/{us,jp,kr,mkt}/SYMBOL.csv (Korean codes are tried on KOSPI, then KOSDAQ);
- eu: daily bars (same API) of the European peers below and three European indices -> dataset/peers/eu/SYMBOL.csv;
  the European session of t - 1 closes at 00:30 Taipei (23:30 in the summer), before the US close;
- edgar: the SEC EDGAR filing index of every US peer (form, 8-K items, acceptance time to the second; keyless, a
  User-Agent with a contact address read from keys/sec_ua.txt outside the repo, well under 10 requests a second) -> dataset/peers/edgar/TICKER.csv. Item 2.02 of an 8-K is
  the earnings release; foreign filers (ASML, ARM, ZIM, CLS) file 6-K without items;
- gdelt: the GDELT DOC 2.0 daily timelines (UTC days) of the article volume and the average tone of each peer's name,
  2020 .. 2026-08 (one request every 20 s, pauses of 2, 4, 6 ... min when throttled) -> dataset/peers/gdelt/TICKER_{vol,tone}.csv.- alpaca: the Benzinga news of the US peers from the Alpaca market-data API (free key read from keys/alpaca outside
  the repo: key id, then secret; created_at in UTC to the second; headline, summary, full text, tagged symbols), one
  file per month -> dataset/news/alpaca/YYYYMM.jsonl.gz (resumable per month);
- alpaca_bars: hourly bars with the extended hours (SIP feed, raw prices; same key) of the US peers and SPY / QQQ /
  SOXX, 2019-11-25 .. 2026-08-31, 1 request a second -> dataset/peers/alpaca_bars/SYM.csv (t = bar start UTC);
- alpaca_bars_x: the same hourly bars of the country ETFs and Asian ADRs of EXTRA_BARS;
- crypto_bars: hourly BTC/USD and ETH/USD (around the clock, Alpaca crypto data, no key);
- edgar_docs: the text of the press release (exhibit 99.x, else the largest other document) of every 8-K with item
  2.02 and every 6-K since 2019-11 (SEC EDGAR, same User-Agent, 1 request a second) -> dataset/peers/edgar_docs/
  TICKER.jsonl.gz (resumable per ticker).

    python demo/fetch_peers.py yahoo|eu|edgar|gdelt|alpaca|alpaca_bars|alpaca_bars_x|crypto_bars|edgar_docs
"""

import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_intl import yahoo  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "dataset" / "peers"
KEYS = Path(__file__).resolve().parents[2] / "keys"  # outside the repo
SEC_UA = {"User-Agent": (KEYS / "sec_ua.txt").read_text().strip() if (KEYS / "sec_ua.txt").exists() else "TWStock research"}

# Taiwan stock -> foreign peers (US tickers, Tokyo codes, Seoul codes); the Japanese and Korean codes already in
# dataset/intl/ are listed too and copied from there by the feature code, not downloaded again
PEERS = {
    "2330": (["TSM", "NVDA", "AAPL", "AMD", "AVGO", "QCOM", "ASML", "AMAT", "LRCX", "KLAC"],
             ["8035", "6857", "4063", "3436"], ["005930", "000660"]),
    "2303": (["UMC", "GFS", "INTC"], ["6723"], ["005930"]),
    "3711": (["ASX", "AMKR"], ["6857"], []),
    "2454": (["QCOM", "ARM", "AVGO"], [], ["005930"]),
    "3034": (["AAPL"], [], ["108320", "034220"]),
    "2379": (["AVGO", "MRVL", "SYNA"], [], []),
    "3661": (["AVGO", "MRVL", "AMZN"], [], []),
    "8069": (["AMZN"], [], []),
    "2317": (["AAPL", "NVDA", "DELL", "HPE"], ["6753"], []),
    "2382": (["NVDA", "SMCI", "DELL", "HPE", "MSFT", "META"], [], []),
    "3231": (["NVDA", "SMCI", "DELL", "HPE", "MSFT", "META"], [], []),
    "6669": (["NVDA", "SMCI", "DELL", "HPE", "MSFT", "META"], [], []),
    "2308": (["NVDA", "VRT", "ETN", "TSLA"], ["6594"], []),
    "3017": (["NVDA", "SMCI", "VRT", "AMD"], [], []),
    "3653": (["NVDA", "SMCI", "VRT", "AMD"], [], []),
    "2059": (["NVDA", "SMCI", "VRT", "AMD"], [], []),
    "2383": (["NVDA", "AVGO", "ANET"], [], ["000150"]),
    "2345": (["ANET", "CSCO", "CLS"], [], []),
    "2357": (["DELL", "HPQ"], [], []),
    "2301": (["DELL", "HPQ"], [], []),
    "2327": (["VSH"], ["6981", "6762", "6976"], ["009150"]),
    "3008": (["AAPL"], ["6758"], ["011070", "009150"]),
    "2395": (["ROK"], ["6861", "6954"], []),
    **{s: (["^BKX", "MET", "PRU", "^TNX"], ["8306", "8316", "8411", "8750"], ["105560", "055550", "086790", "032830"])
       for s in ("2881", "2882", "2891", "2886", "2884", "2885", "2892", "2880", "5880", "2890", "2883", "2887")},
    "5871": (["^BKX", "^TNX", "FXI"], [], []),
    **{s: (["DOW", "LYB", "CL=F"], ["4063", "4183"], ["051910", "011170"]) for s in ("1301", "1303", "6505")},
    "2002": (["NUE", "STLD"], ["5401"], ["005490", "004020"]),
    **{s: (["ZIM", "MATX"], ["9101", "9104", "9107"], ["011200"]) for s in ("2603", "2609", "2615")},
    **{s: (["T", "VZ", "TMUS"], ["9432", "9433"], ["017670", "030200"]) for s in ("2412", "3045", "4904")},
    "2207": ([], ["7203"], []),
    "2912": ([], ["3382"], []),
    "3293": (["RBLX", "TTWO", "EA"], ["7974", "9697"], ["259960", "036570"]),
}
# European peers of the Taiwan stocks (Yahoo symbols) and European indices (STOXX 600, DAX, Euro Stoxx 50)
EU_PEERS = {
    **{s: ["ASML.AS", "ASM.AS", "BESI.AS"] for s in ("2330", "2303", "3711")},
    **{s: ["IFX.DE", "STMPA.PA"] for s in ("2454", "3034", "2379", "2327")},
    **{s: ["SAP.DE", "NOKIA.HE", "ERIC-B.ST"] for s in ("2345", "2383")},
    **{s: ["SU.PA", "SIE.DE"] for s in ("2308", "2395")},
    **{s: ["HSBA.L", "BNP.PA", "ALV.DE"] for s in ("2881", "2882", "2891", "2886", "2884", "2885", "2892", "2880",
                                                  "5880", "2890", "2883", "2887", "5871")},
    **{s: ["BAS.DE"] for s in ("1301", "1303", "6505")},
    "2002": ["MT.AS"],
    **{s: ["MAERSK-B.CO", "HLAG.DE"] for s in ("2603", "2609", "2615")},
    **{s: ["DTE.DE", "ORA.PA"] for s in ("2412", "3045", "4904")},
}
EU_INDEX = ["^STOXX", "^GDAXI", "^STOXX50E"]
MARKET = {"^BKX", "^TNX", "CL=F", "FXI"}
US = sorted({t for u, _, _ in PEERS.values() for t in u} - MARKET)
JP = sorted({t for _, j, _ in PEERS.values() for t in j})
KR = sorted({t for _, _, k in PEERS.values() for t in k})
NAMES = {"TSM": "TSMC", "NVDA": "Nvidia", "AAPL": "Apple", "AMD": "AMD", "AVGO": "Broadcom", "QCOM": "Qualcomm",
         "ASML": "ASML", "AMAT": "Applied Materials", "LRCX": "Lam Research", "KLAC": "KLA Corporation",
         "UMC": "United Microelectronics", "GFS": "GlobalFoundries", "INTC": "Intel", "ASX": "ASE Technology",
         "AMKR": "Amkor", "ARM": "Arm Holdings", "MRVL": "Marvell", "SYNA": "Synaptics", "AMZN": "Amazon",
         "DELL": "Dell", "HPE": "Hewlett Packard Enterprise", "SMCI": "Supermicro", "MSFT": "Microsoft",
         "META": "Meta", "VRT": "Vertiv", "ETN": "Eaton", "TSLA": "Tesla", "ANET": "Arista Networks",
         "CSCO": "Cisco", "CLS": "Celestica", "HPQ": "HP Inc", "VSH": "Vishay", "ROK": "Rockwell Automation",
         "MET": "MetLife", "PRU": "Prudential Financial", "DOW": "Dow Inc", "LYB": "LyondellBasell",
         "NUE": "Nucor", "STLD": "Steel Dynamics", "ZIM": "ZIM Integrated", "MATX": "Matson", "T": "AT&T",
         "VZ": "Verizon", "TMUS": "T-Mobile", "RBLX": "Roblox", "TTWO": "Take-Two", "EA": "Electronic Arts"}


def get(url, headers, tries=8, pause=10):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120).read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"  retry {k + 1} after {e}", flush=True)
            time.sleep(pause * (k + 1))
    raise RuntimeError(url)


def run_yahoo():
    intl = OUT.parent / "intl"
    for mkt, codes, sufs in (("us", US, ("",)), ("mkt", sorted(MARKET), ("",)), ("jp", JP, (".T",)),
                             ("kr", KR, (".KS", ".KQ"))):
        (OUT / mkt).mkdir(parents=True, exist_ok=True)
        for c in codes:
            path = OUT / mkt / f"{c.replace('^', '').replace('=', '_')}.csv"
            if path.exists() or (mkt in ("jp", "kr") and (intl / mkt / f"{c}.csv").exists()):
                continue
            for suf in sufs:
                try:
                    print(f"{mkt} {c}{suf}: {yahoo(c + suf, path)} rows", flush=True)
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"{mkt} {c}{suf}: failed ({e})", flush=True)
            time.sleep(1)


def run_eu():
    (OUT / "eu").mkdir(parents=True, exist_ok=True)
    for c in sorted({t for v in EU_PEERS.values() for t in v}) + EU_INDEX:
        path = OUT / "eu" / f"{c.replace('^', '')}.csv"
        if path.exists():
            continue
        try:
            print(f"eu {c}: {yahoo(c, path)} rows", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"eu {c}: failed ({e})", flush=True)
        time.sleep(1)


def run_edgar():
    (OUT / "edgar").mkdir(parents=True, exist_ok=True)
    cik = {v["ticker"]: v["cik_str"] for v in json.loads(get("https://www.sec.gov/files/company_tickers.json",
                                                              SEC_UA)).values()}
    for t in US:
        path = OUT / "edgar" / f"{t}.csv"
        if path.exists():
            continue
        if t not in cik:
            print(f"{t}: no CIK", flush=True)
            continue
        base = "https://data.sec.gov/submissions/"
        sub = json.loads(get(f"{base}CIK{cik[t]:010d}.json", SEC_UA))
        parts = [pd.DataFrame(sub["filings"]["recent"])]
        for f in sub["filings"].get("files", []):
            parts.append(pd.DataFrame(json.loads(get(base + f["name"], SEC_UA))))
            time.sleep(0.2)
        df = pd.concat(parts)
        keep = ["accessionNumber", "filingDate", "acceptanceDateTime", "form", "items", "primaryDocument",
                "primaryDocDescription"]
        df = df[[c for c in keep if c in df]]
        df = df[(df["filingDate"] >= "2019-12-01") & df["form"].isin(["8-K", "8-K/A", "6-K", "10-Q", "10-K", "20-F"])]
        df.to_csv(path, index=False)
        print(f"{t}: {len(df)} filings, {(df['form'] == '8-K').sum()} 8-K", flush=True)
        time.sleep(0.3)


def run_gdelt():
    (OUT / "gdelt").mkdir(parents=True, exist_ok=True)
    for t in US:
        q = NAMES[t]
        q = f'"{q}"' if " " in q else q
        for mode, tag in (("timelinevolraw", "vol"), ("timelinetone", "tone")):
            path = OUT / "gdelt" / f"{t}_{tag}.csv"
            if path.exists():
                continue
            frames = []
            for y0 in range(2020, 2027):  # one request per year so that the resolution stays daily
                a, b = f"{y0}0101000000", (f"{y0 + 1}0101000000" if y0 < 2026 else "20260901000000")
                url = ("https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(
                    dict(query=q, mode=mode, format="csv", startdatetime=a, enddatetime=b)))
                for k in range(12):
                    time.sleep(20)  # GDELT asks for at most one request every 5 s; it throttled us at 6 s
                    txt = get(url, {"User-Agent": "Mozilla/5.0"}).decode("utf-8-sig")
                    if not txt.startswith("Please limit"):
                        break
                    time.sleep(120 * (k + 1))
                else:
                    raise RuntimeError(f"GDELT kept throttling on {t} {y0}")
                if txt.strip():
                    frames.append(pd.read_csv(io.StringIO(txt)))
            df = pd.concat(frames) if frames else pd.DataFrame()
            df.to_csv(path, index=False)
            print(f"{t} {tag}: {len(df)} rows", flush=True)


def run_alpaca():
    import gzip
    kid, sec = (KEYS / "alpaca").read_text().split()[:2]
    h = {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}
    out = OUT.parent / "news" / "alpaca"
    out.mkdir(parents=True, exist_ok=True)
    syms = ",".join(t for t in US if t != "EA")
    for m in pd.date_range("2019-12-01", "2026-08-01", freq="MS"):
        path = out / f"{m:%Y%m}.jsonl.gz"
        if path.exists():
            continue
        rows, token = [], None
        while True:
            q = dict(symbols=syms, start=f"{m:%Y-%m-%d}T00:00:00Z",
                     end=f"{m + pd.offsets.MonthBegin(1):%Y-%m-%d}T00:00:00Z", limit=50, sort="asc",
                     include_content="true")
            if token:
                q["page_token"] = token
            d = json.loads(get("https://data.alpaca.markets/v1beta1/news?" + urllib.parse.urlencode(q), h))
            rows += d["news"]
            token = d.get("next_page_token")
            time.sleep(0.6)  # about 90 requests a minute; the free plan allows 200
            if not token:
                break
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()))
        tmp.rename(path)
        print(f"alpaca {m:%Y-%m}: {len(rows)}", flush=True)


_last = [0.0]


def paced(url, headers, stop=(401, 403, 422)):
    """GET at most once a second; 404 -> None; codes in `stop` end the run; others wait 30 s, 1, 2 ... min."""
    for k in range(6):
        w = 1.0 - (time.time() - _last[0])
        if w > 0:
            time.sleep(w)
        _last[0] = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            print(f"  HTTP {e.code} on {url}", flush=True)
            if e.code in stop:
                raise SystemExit("not allowed, stopping")
            time.sleep(30 * 2 ** k)
        except Exception as e:  # network
            print(f"  {e} on {url}", flush=True)
            time.sleep(60)
    raise SystemExit(f"giving up on {url}")


# country ETFs and Asian ADRs traded in the US session and its extended hours (alpaca_bars_x)
EXTRA_BARS = ["EWT", "EWJ", "EWY", "FXI", "KWEB", "SONY", "TM", "MUFG", "SMFG", "KB", "SHG", "PKX", "SMH", "TLT", "UUP"]


def run_alpaca_bars(extra=False):
    import os
    kid, sec = (KEYS / "alpaca").read_text().split()[:2]
    h = {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}
    out = OUT / "alpaca_bars"
    out.mkdir(exist_ok=True)
    syms = EXTRA_BARS if extra else sorted(p.stem for p in (OUT / "us").glob("*.csv")) + ["SPY", "QQQ", "SOXX"]
    for s_ in syms:
        dst = out / f"{s_}.csv"
        if dst.exists():
            continue
        rows, tok = [], None
        while True:
            q = dict(timeframe="1Hour", start="2019-11-25T00:00:00Z", end="2026-09-01T00:00:00Z", limit=10000,
                     adjustment="raw", feed="sip")
            if tok:
                q["page_token"] = tok
            d = json.loads(paced(f"https://data.alpaca.markets/v2/stocks/{s_}/bars?" + urllib.parse.urlencode(q), h,
                                 stop=(401, 403, 422)))
            rows += d.get("bars") or []
            tok = d.get("next_page_token")
            if not tok:
                break
        f = pd.DataFrame(rows)
        f.to_csv(dst.with_suffix(".tmp"), index=False)
        os.replace(dst.with_suffix(".tmp"), dst)
        print(f"{s_}: {len(f)} bars", flush=True)


def run_crypto_bars():
    """hourly BTC/USD and ETH/USD bars (Alpaca crypto market data, no key needed), 2019-11-25 .. 2026-08-31, around the
    clock -> dataset/peers/alpaca_bars/BTC_USD.csv, ETH_USD.csv (t = bar start UTC)."""
    import os
    out = OUT / "alpaca_bars"
    out.mkdir(exist_ok=True)
    for s_ in ("BTC/USD", "ETH/USD"):
        dst = out / f"{s_.replace('/', '_')}.csv"
        if dst.exists():
            continue
        rows, tok = [], None
        while True:
            q = dict(symbols=s_, timeframe="1Hour", start="2019-11-25T00:00:00Z", end="2026-09-01T00:00:00Z", limit=10000)
            if tok:
                q["page_token"] = tok
            d = json.loads(paced("https://data.alpaca.markets/v1beta3/crypto/us/bars?" + urllib.parse.urlencode(q), {},
                                 stop=(401, 403, 422)))
            rows += (d.get("bars") or {}).get(s_) or []
            tok = d.get("next_page_token")
            if not tok:
                break
        f = pd.DataFrame(rows)
        f.to_csv(dst.with_suffix(".tmp"), index=False)
        os.replace(dst.with_suffix(".tmp"), dst)
        print(f"{s_}: {len(f)} bars", flush=True)


def run_edgar_docs():
    import gzip
    import html
    import os
    import re
    out = OUT / "edgar_docs"
    out.mkdir(exist_ok=True)
    text_of = lambda raw: re.sub(r"\s+", " ", html.unescape(re.sub(r"(?s)<[^>]+>", " ", re.sub(
        r"(?is)<(script|style).*?</\1>", " ", raw.decode("utf-8", "ignore"))))).strip()
    cik = {v["ticker"]: v["cik_str"] for v in json.loads(paced("https://www.sec.gov/files/company_tickers.json",
                                                              SEC_UA, stop=(403,))).values()}
    for p in sorted((OUT / "edgar").glob("*.csv")):
        t = p.stem
        dst = out / f"{t}.jsonl.gz"
        if dst.exists() or t not in cik:
            continue
        f = pd.read_csv(p, dtype=str)
        f = f[(f.acceptanceDateTime >= "2019-11-01") & (
            (f.form == "6-K") | ((f.form == "8-K") & f["items"].fillna("").str.contains("2.02", regex=False)))]
        rows = []
        for _, r in f.iterrows():
            base = f"https://www.sec.gov/Archives/edgar/data/{cik[t]}/{r.accessionNumber.replace('-', '')}/"
            idx = paced(base + "index.json", SEC_UA, stop=())
            if idx is None:
                continue
            htm = [(i["name"], int(i.get("size") or 0)) for i in json.loads(idx)["directory"]["item"]
                   if i["name"].lower().endswith((".htm", ".html", ".txt")) and i["name"] != r.primaryDocument
                   and not re.match(r"(?i)(R\d+\.htm|FilingSummary|.*-index)", i["name"])
                   and not i["name"].startswith(r.accessionNumber)]
            ex = [n for n, _ in htm if re.search(r"(?i)ex[-_]?99|exhibit[-_]?99|dex99", n)]
            doc = sorted(ex)[0] if ex else (max(htm, key=lambda x: x[1])[0] if htm else r.primaryDocument)
            raw = paced(base + doc, SEC_UA, stop=())
            if raw is not None:
                rows.append(dict(accession=r.accessionNumber, form=r.form, items=r["items"],
                                 accepted=r.acceptanceDateTime, doc=doc, text=text_of(raw)[:20000]))
        with gzip.open(dst.with_suffix(".tmp"), "wt") as g:
            g.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
        os.replace(dst.with_suffix(".tmp"), dst)
        print(f"{t}: {len(rows)} of {len(f)} filings", flush=True)


if __name__ == "__main__":
    {"yahoo": run_yahoo, "eu": run_eu, "edgar": run_edgar, "gdelt": run_gdelt, "alpaca": run_alpaca, "alpaca_bars": run_alpaca_bars,
     "alpaca_bars_x": lambda: run_alpaca_bars(extra=True), "crypto_bars": run_crypto_bars,
     "edgar_docs": run_edgar_docs}[sys.argv[1]]()
