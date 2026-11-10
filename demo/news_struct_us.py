"""E3c r5: direction and size of the Benzinga news (dataset/news/alpaca, demo/fetch_peers.py alpaca) about the 46 US peers that
appeared after the US regular close (16:00 New York) and before 04:00 New York, or on a weekend, i.e. not yet in the
close-to-close returns, by Qwen3.5-4B with the TypedForm (headline + summary, greedy, thinking off).
One row per (article, US peer named in its symbols). Output dataset/news/alpaca/struct.npz: id, symbol, created (UTC),
p_relevance (about the company itself / mentions it / unrelated), p_direction (positive / negative / neutral),
p_size (small / medium / large).

    HF_HOME=<models> PYLIB=<transformers dir> python demo/news_struct_us.py [--test N] [--batch B]
"""

import argparse
import glob
import gzip
import html
import json
import os
import re
import sys
import time
from pathlib import Path

if os.environ.get("PYLIB"):
    sys.path.insert(0, os.environ["PYLIB"])

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_poly import ROOT as DS  # noqa: E402
from stock_model import LETTERS, TypedForm  # noqa: E402

LLM = "Qwen/Qwen3.5-4B"
FIELDS = [("relevance", "Relevance", ["about the company itself", "mentions the company", "unrelated"]),
          ("direction", "Direction for the stock", ["positive", "negative", "neutral"]),
          ("size", "Size of the likely price move", ["small", "medium", "large"])]


def pairs():
    us = {Path(p).stem for p in glob.glob(str(DS / "peers" / "us" / "*.csv"))}
    rows = []
    for p in sorted((DS / "news" / "alpaca").glob("*.jsonl.gz")):
        with gzip.open(p, "rt") as g:
            for l in g:
                r = json.loads(l)
                txt = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", r.get("summary") or ""))).strip()
                for s in r["symbols"]:
                    if s in us:
                        rows.append((r["id"], s, r["created_at"], r["headline"], txt, len(r["symbols"])))
    f = pd.DataFrame(rows, columns=["id", "sym", "created", "headline", "summary", "n_sym"])
    t = pd.to_datetime(f.created).dt.tz_convert("America/New_York")
    over = (t.dt.hour >= 16) | (t.dt.hour < 4) | (t.dt.day_of_week >= 5)
    return f[over].reset_index(drop=True)


def prompt(tok, r):
    form = "\n".join(f"{lab}: (" + " / ".join(f"{LETTERS[k]}={o}" for k, o in enumerate(opts)) + ")"
                     for _, lab, opts in FIELDS)
    q = (f"News published {r.created} UTC.\nHeadline: {r.headline}\n" + (f"Summary: {r.summary[:600]}\n" if r.summary else "")
         + f"\nJudge only from this text, without outside knowledge, what it means for the stock price of {r.sym}. "
         f"Answer line by line with the option letter only:\n{form}")
    msgs = [{"role": "system", "content": "You are a careful equity analyst who answers only from the given text."},
            {"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--batch", type=int, default=48)
    a = ap.parse_args()
    import transformers as T
    from transformers import LogitsProcessorList
    f = pairs()
    print(f"{len(f)} (article, US peer) pairs in the overnight window", flush=True)
    if a.test:
        f = f.sample(a.test, random_state=0).reset_index(drop=True)
    tok = T.AutoTokenizer.from_pretrained(LLM, padding_side="left")
    model = T.AutoModelForImageTextToText.from_pretrained(LLM, dtype=torch.bfloat16).to("cuda").eval()
    texts = [prompt(tok, r) for r in f.itertuples()]
    order = np.argsort([len(x) for x in texts])
    P = {n: np.zeros((len(f), len(o)), np.float32) for n, _, o in FIELDS}
    t0 = time.time()
    for i in range(0, len(order), a.batch):
        b = order[i:i + a.batch]
        x = tok([texts[k] for k in b], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        form = TypedForm(lambda s: tok.encode(s, add_special_tokens=False), FIELDS, tail="\n", sep=": ")
        with torch.no_grad():
            model.generate(**x, max_new_tokens=len(form.plan), do_sample=False,
                           logits_processor=LogitsProcessorList([form]), pad_token_id=tok.pad_token_id)
        for n in P:
            P[n][b] = form.probs[n]
        if (i // a.batch) % 100 == 0:
            print(f"{i + len(b)} / {len(f)}, {time.time() - t0:.0f} s", flush=True)
    if a.test:
        for k, r in enumerate(f.itertuples()):
            print(f"== {r.sym} {r.created} {r.headline}\n-- " + "  ".join(
                f"{n} " + "/".join(f"{v:.2f}" for v in P[n][k]) for n in P))
        return
    dst = DS / "news" / "alpaca" / "struct.npz"
    np.savez(dst.with_suffix(".tmp.npz"), id=f.id.to_numpy(), sym=f.sym.to_numpy().astype(str),
             created=f.created.to_numpy().astype(str), n_sym=f.n_sym.to_numpy(), **{f"p_{n}": v for n, v in P.items()})
    os.replace(dst.with_suffix(".tmp.npz"), dst)
    print(f"{len(f)} pairs in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
