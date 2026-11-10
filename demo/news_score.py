"""Score the E2 texts (MOPS announcements and cnyes news) with an LLM and embed them (TWStock, not committed data).

Items (dataset/news/, from demo/fetch_news.py):
- MOPS: every announcement of our 100 stocks (deduplicated on company, date, time and sequence number); the item's
  stock is the announcing company, its time the announcement's time (Taipei), its text the subject and the first
  TEXT_CHARS characters of the body;
- cnyes: every article; one item per tagged stock among our 100 (up to 3), or one market item (stock "MKT") if it
  tags none of them; time the publication time converted to Taipei, text the title and the summary.
1. LLM score (zero-shot): Qwen2.5-3B-Instruct reads the item and the question whether it is positive, neutral or
   negative news for the company's share price on the next trading day (for market items: for the Taiwan market);
   from the logits of the first token of the answers 正面 / 中性 / 負面 at the answer position,
   score = P(正面) - P(負面) (renormalized over the three).
2. Embedding: BAAI/bge-m3 (the CLS vector, L2-normalized, 1024 dims, float16) of every distinct text.
Output dataset/news/scores/items.npz: src, stock, ts (Taipei, datetime64[s]), p (N, 3), emb_id; emb.npy.

    HF_HOME=<models> PYLIB=<transformers target dir> python demo/news_score.py [--test]
"""

import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if os.environ.get("PYLIB"):  # transformers and its dependencies, installed outside the environment
    sys.path.insert(0, os.environ["PYLIB"])
ROOT = Path(__file__).resolve().parents[1] / "dataset"
OUT = ROOT / "news" / "scores"
TEXT_CHARS, MAX_TOK, MAX_TAGS = 600, 512, 3
LLM, EMB = "Qwen/Qwen2.5-3B-Instruct", "BAAI/bge-m3"
LABELS = ("正面", "中性", "負面")


def items():
    stocks = json.loads((ROOT / "info.json").read_text())["Top100@Y"]
    ours = set(stocks)
    names, rows, seen = {}, [], set()
    for f in sorted((ROOT / "news" / "mops").glob("*.jsonl.gz")):
        for r in map(json.loads, gzip.open(f, "rt", encoding="utf-8")):
            if not r.get("date") or r["date"] == "None" or not r.get("time") or r["co"] not in ours:
                continue
            k = (r["co"], r["date"], r["time"], r["seq"])
            if k in seen:
                continue
            seen.add(k)
            names[r["co"]] = r["name"]
            rows.append(("mops", r["co"], pd.Timestamp(r["date"] + r["time"].zfill(6)),
                         (r["subject"] + "\n" + r["text"][:TEXT_CHARS]).strip(), "公告"))
    seen = set()
    for f in sorted((ROOT / "news" / "cnyes").glob("*.jsonl.gz")):
        for r in map(json.loads, gzip.open(f, "rt", encoding="utf-8")):
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            ts = pd.Timestamp(int(r["publish"]), unit="s", tz="UTC").tz_convert("Asia/Taipei").tz_localize(None)
            text = (r["title"] + "\n" + (r["summary"] or r["text"][:TEXT_CHARS])).strip()
            tags = [s for s in dict.fromkeys(r.get("stock") or []) if s in ours][:MAX_TAGS]
            for s in tags or ["MKT"]:
                rows.append(("cnyes", s, ts, text, "新聞"))
    f = pd.DataFrame(rows, columns=["src", "stock", "ts", "text", "kind"])
    f["name"] = f.stock.map(names).fillna(f.stock)
    return f


def prompt(tok, row):
    if row.stock == "MKT":
        q = f"以下是一則台股{row.kind}（{row.ts:%Y-%m-%d %H:%M}）：\n{row.text}\n\n這則訊息對台股大盤下一個交易日的走勢是正面、中性還是負面？只回答：正面、中性或負面。"
    else:
        q = (f"以下是一則關於{row.name}（{row.stock}）的{row.kind}（{row.ts:%Y-%m-%d %H:%M}）：\n{row.text}\n\n"
             f"這則訊息對{row.name}下一個交易日的股價是正面、中性還是負面？只回答：正面、中性或負面。")
    msgs = [{"role": "system", "content": "你是台灣股市的分析師。"}, {"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def llm_scores(f, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLM, padding_side="left")
    ids = [tok.encode(w, add_special_tokens=False)[0] for w in LABELS]
    assert len(set(ids)) == 3, ids
    model = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.bfloat16).to(device).eval()
    texts = [prompt(tok, r) for r in f.itertuples()]
    enc = [tok.encode(t, add_special_tokens=False)[-MAX_TOK:] for t in texts]
    order = np.argsort([len(e) for e in enc])
    P = np.zeros((len(texts), 3), np.float32)
    t0, B = time.time(), 32
    for i in range(0, len(order), B):
        b = order[i:i + B]
        L = max(len(enc[j]) for j in b)
        x = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
        m = torch.zeros((len(b), L), dtype=torch.long)
        for k, j in enumerate(b):
            x[k, L - len(enc[j]):] = torch.tensor(enc[j])
            m[k, L - len(enc[j]):] = 1
        with torch.no_grad():
            lg = model(input_ids=x.to(device), attention_mask=m.to(device), logits_to_keep=1).logits[:, -1, ids]
        P[b] = torch.softmax(lg.float(), -1).cpu().numpy()
        if (i // B) % 200 == 0:
            print(f"LLM {i + len(b)}/{len(order)}, {time.time() - t0:.0f} s", flush=True)
    del model
    torch.cuda.empty_cache()
    return P


def embeddings(texts, device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMB)
    model = AutoModel.from_pretrained(EMB, dtype=torch.float16).to(device).eval()
    order = np.argsort([len(t) for t in texts])
    E = np.zeros((len(texts), model.config.hidden_size), np.float16)
    t0 = time.time()
    for i in range(0, len(order), 128):
        b = order[i:i + 128]
        x = tok([texts[j] for j in b], padding=True, truncation=True, max_length=MAX_TOK, return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**x).last_hidden_state[:, 0]
        E[b] = torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy().astype(np.float16)
        if (i // 128) % 200 == 0:
            print(f"embed {i + len(b)}/{len(order)}, {time.time() - t0:.0f} s", flush=True)
    return E


def main():
    device = "cuda"
    OUT.mkdir(parents=True, exist_ok=True)
    f = items()
    print(f"{len(f)} items: " + ", ".join(f"{k} {v}" for k, v in f.groupby(["src", f.stock == 'MKT']).size().items()),
          flush=True)
    if "--test" in sys.argv:
        f = f.sample(512, random_state=0).reset_index(drop=True)
    uniq, emb_id = np.unique(f.text.to_numpy(), return_inverse=True)
    E = embeddings(list(uniq), device)
    P = llm_scores(f, device)
    tag = "_test" if "--test" in sys.argv else ""
    np.save(OUT / f"emb{tag}.npy", E)
    np.savez(OUT / f"items{tag}.npz", src=f.src.to_numpy().astype(str), stock=f.stock.to_numpy().astype(str),
             ts=f.ts.to_numpy().astype("datetime64[s]"), p=P, emb_id=emb_id)
    s = P[:, 0] - P[:, 2]
    print(f"score mean {s.mean():+.3f}, sd {s.std():.3f}; argmax shares " +
          ", ".join(f"{w} {np.mean(P.argmax(1) == k):.1%}" for k, w in enumerate(LABELS)), flush=True)
    if "--test" in sys.argv:
        for r, sc in list(zip(f.itertuples(), s))[:8]:
            print(f"{sc:+.2f} {r.src} {r.stock} {r.text[:70]!r}")


if __name__ == "__main__":
    main()
