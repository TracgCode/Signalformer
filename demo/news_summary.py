"""Structured summaries of the E2 texts (MOPS announcements and cnyes news) by a local LLM (TWStock, not committed).

For every item (one text and the stock it is about, as in demo/news_score.py) a local instruction model (--model,
default Qwen2.5-3B-Instruct; thinking turned off where the chat template has it) writes, from the
text only, a fixed form: event type, direction for the stock, size, surprise, key numbers and a one-sentence summary
(at most 60 characters). Greedy decoding. With --test N it summarizes a stratified sample of N items (half MOPS, half
cnyes, spread over the years) and writes the texts and the forms side by side to the file given by --out.

    HF_HOME=<models> PYLIB=<transformers target dir> python demo/news_summary.py [--model M] --test 50 --out <file>
"""

import argparse
import os
import sys
import time
from pathlib import Path

if os.environ.get("PYLIB"):
    sys.path.insert(0, os.environ["PYLIB"])
if os.environ.get("PYLIB_EXTRA"):  # e.g. the `kernels` package for FP8 checkpoints, after PYLIB
    sys.path.append(os.environ["PYLIB_EXTRA"])

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from news_score import items  # noqa: E402

LLM = "Qwen/Qwen2.5-3B-Instruct"
FORM = """事件類型：（財報／月營收／法說會／股利／增減資／併購或投資／重大訂單或合作／產能或擴廠／人事／訴訟或裁罰／股東會或董事會例行事項／其他）
方向：（利多／利空／中性）
幅度：（小／中／大）
是否超出預期：（是／否／未提及）
關鍵數字：（原文中最重要的數字，沒有就寫「無」）
摘要：（一句話，60字以內）"""


EVENTS = ["財報", "月營收", "法說會", "股利", "增減資", "併購或投資", "重大訂單或合作", "產能或擴廠", "人事", "訴訟或裁罰",
          "股東會或董事會例行事項", "其他"]
NX_TEMPLATE = {"事件類型": EVENTS, "方向": ["利多", "利空", "中性"], "幅度": ["小", "中", "大"],
               "是否超出預期": ["是", "否", "未提及"], "關鍵數字": ["verbatim-string"], "摘要": "string"}


def prompt(tok, row, model=LLM):
    who = "台股大盤" if row.stock == "MKT" else f"{row.name}（{row.stock}）"
    if "nuextract" in model.lower():  # NuExtract: the text, a JSON template and instructions (structured mode)
        import json
        msgs = [{"role": "user", "content": [{"type": "text", "text":
                 f"以下是一則{row.kind}，發布時間 {row.ts:%Y-%m-%d %H:%M}：\n{row.text}"}]}]
        ins = (f"只根據上文、不要使用任何外部知識，評估它對{who}的意義：方向與幅度是對{who}股價的影響；"
               "摘要為一句繁體中文，60字以內。")
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                                       template=json.dumps(NX_TEMPLATE, ensure_ascii=False, indent=4),
                                       instructions=ins)
    q = (f"以下是一則{row.kind}，發布時間 {row.ts:%Y-%m-%d %H:%M}：\n{row.text}\n\n"
         f"請只根據上文、不要使用任何外部知識，評估它對{who}的意義，依下列格式逐行填寫，不要加其他內容：\n{FORM}")
    msgs = [{"role": "system", "content": "你是嚴謹的台灣股市分析師，只根據提供的文字回答。"},
            {"role": "user", "content": q}]
    if "gemma" in model.lower():  # no system role in the Gemma templates
        msgs = [{"role": "user", "content": msgs[0]["content"] + "\n\n" + q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def load_model(name, device):
    import transformers as T
    tok = T.AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for cls in (T.AutoModelForCausalLM, T.AutoModelForImageTextToText):
        try:
            return tok, cls.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
        except (ValueError, KeyError) as e:
            print(f"{cls.__name__}: {e}", flush=True)
    raise RuntimeError(name)


def summarize(rows, device, batch=32, max_new=160, name=LLM):
    tok, model = load_model(name, device)
    print(f"{name}: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB after loading", flush=True)
    texts = [prompt(tok, r, name) for r in rows.itertuples()]
    out = [None] * len(texts)
    order = np.argsort([len(t) for t in texts])
    t0 = time.time()
    for i in range(0, len(order), batch):
        b = order[i:i + batch]
        x = tok([texts[k] for k in b], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            y = model.generate(**x, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for k, seq in zip(b, y[:, x["input_ids"].shape[1]:]):
            out[k] = tok.decode(seq, skip_special_tokens=True).strip()
        print(f"{i + len(b)}/{len(order)}, {time.time() - t0:.0f} s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--out", default="news_summary_test.txt")
    ap.add_argument("--model", default=LLM)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    f = items()
    f = f[f.stock != "MKT"].copy()
    f["year"] = f.ts.dt.year
    if a.test:
        k = a.test // 2
        parts = []
        for so in ("mops", "cnyes"):
            g = f[f.src == so]
            parts.append(g.groupby("year", group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, k // g.year.nunique() + 1)), random_state=0)).head(k))
        f = __import__("pandas").concat(parts).reset_index(drop=True)
    t0 = time.time()
    s = summarize(f, "cuda", batch=a.batch, name=a.model)
    print(f"{len(f)} items in {time.time() - t0:.0f} s", flush=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r, out in zip(f.itertuples(), s):
            fh.write(f"===== {r.src} {r.stock} {r.name} {r.ts:%Y-%m-%d %H:%M}\n--- 原文\n{r.text}\n--- 摘要\n{out}\n\n")


if __name__ == "__main__":
    main()
