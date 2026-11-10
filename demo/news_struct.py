"""Typed fields with choice probabilities and a one-sentence summary for every E2 item, by Qwen3.5-4B (TWStock).

For every item of demo/news_score.py (one text and the stock it is about; MKT = the market) the model reads the text
(only the text, no outside knowledge) and fills a fixed form. The categorical fields are lettered choices; at each of
them the generation is constrained to the allowed letters, the probabilities of the letters (renormalized over the
allowed ones) are kept, and the most likely letter is written before the next field, so the later fields are
conditioned on the earlier answers. All other tokens of the form are forced. After the fields the model writes a
free one-sentence summary (at most 80 new tokens). Fields:
  relation   A direct / B indirect / C unrelated (to the stock named in the question)
  event      A..L: 財報, 月營收, 法說會, 股利, 增減資, 併購或投資, 重大訂單或合作, 產能或擴廠, 人事, 訴訟或裁罰,
             股東會或董事會例行事項, 其他
  direction  A 利多 / B 利空 / C 中性 (for the stock's price)
  size       A 小 / B 中 / C 大
  surprise   A 是 / B 否 / C 未提及
Output, resumable in parts of 4,096 items: dataset/news/struct/part_NNNN.npz with the item index, the probabilities of
every field and the summaries; with --embed, the bge-m3 embeddings of the summaries (as in demo/news_score.py, CLS,
normalized, float16) into dataset/news/struct/summary_emb.npy. Greedy, thinking off, the text cut to 1,024 tokens as in demo/news_summary.py.

    HF_HOME=<models> PYLIB=<transformers dir> python demo/news_struct.py [--test N --out FILE] [--batch B] [--embed]
"""

import argparse
import os
import sys
import time
from pathlib import Path

if os.environ.get("PYLIB"):
    sys.path.insert(0, os.environ["PYLIB"])

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from news_score import ROOT, items  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stock_model import LETTERS, TypedForm  # noqa: E402

LLM = "Qwen/Qwen3.5-4B"
OUT = ROOT / "news" / "struct"
PART = 4096
EVENTS = ["財報", "月營收", "法說會", "股利", "增減資", "併購或投資", "重大訂單或合作", "產能或擴廠", "人事", "訴訟或裁罰",
          "股東會或董事會例行事項", "其他"]
FIELDS = [("relation", "與標的的關係", ["直接相關", "間接相關", "無關"]),
          ("event", "事件類型", EVENTS),
          ("direction", "方向", ["利多", "利空", "中性"]),
          ("size", "幅度", ["小", "中", "大"]),
          ("surprise", "是否超出預期", ["是", "否", "未提及"])]


def prompt(tok, row):
    who = "台股大盤" if row.stock == "MKT" else f"{row.name}（{row.stock}）"
    form = "\n".join(f"{lab}：（" + "／".join(f"{LETTERS[k]}={o}" for k, o in enumerate(opts)) + "）"
                     for _, lab, opts in FIELDS)
    q = (f"以下是一則{row.kind}，發布時間 {row.ts:%Y-%m-%d %H:%M}：\n{row.text}\n\n"
         f"請只根據上文、不要使用任何外部知識，評估它對{who}的意義。例行公告與和{who}無直接關係的內容應判為中性、小。"
         f"依下列格式逐行作答，選項只寫代號字母，最後一行寫一句60字以內的繁體中文摘要：\n{form}\n摘要：（一句話）")
    msgs = [{"role": "system", "content": "你是嚴謹的台灣股市分析師，只根據提供的文字回答。"},
            {"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def Form(tok):
    """the form of FIELDS as a stock_model.TypedForm logits processor."""
    return TypedForm(lambda t: tok.encode(t, add_special_tokens=False), FIELDS)


def run(rows, tok, model, batch, device):
    from transformers import LogitsProcessorList
    texts = [prompt(tok, r) for r in rows.itertuples()]
    order = np.argsort([len(t) for t in texts])
    P = {name: np.zeros((len(texts), len(opts)), np.float32) for name, _, opts in FIELDS}
    summ = [""] * len(texts)
    for i in range(0, len(order), batch):
        b = order[i:i + batch]
        x = tok([texts[k] for k in b], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        form = Form(tok)
        with torch.no_grad():
            y = model.generate(**x, max_new_tokens=len(form.plan) + 80, do_sample=False,
                               logits_processor=LogitsProcessorList([form]), pad_token_id=tok.pad_token_id)
        for name in P:
            P[name][b] = form.probs[name]
        for k, seq in zip(b, y[:, x["input_ids"].shape[1] + len(form.plan):]):
            summ[k] = tok.decode(seq, skip_special_tokens=True).strip().split("\n")[0]
    return P, summ


def load_struct():
    """all parts joined, in the order of news_score.items(): index, summary, p_<field>."""
    parts = [np.load(fn, allow_pickle=True) for fn in sorted(OUT.glob("part_[0-9][0-9][0-9][0-9].npz"))]
    out = {k: np.concatenate([z[k] for z in parts]) for k in parts[0].files}
    assert (out["index"] == np.arange(len(out["index"]))).all()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--out", default="news_struct_test.txt")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--embed", action="store_true", help="embed the finished summaries with bge-m3 (summary_emb.npy)")
    a = ap.parse_args()
    if a.embed:
        from news_score import embeddings
        S = load_struct()
        np.save(OUT / "summary_emb.npy", embeddings(list(S["summary"]), "cuda"))
        return
    import transformers as T
    device = "cuda"
    tok = T.AutoTokenizer.from_pretrained(LLM, padding_side="left")
    model = T.AutoModelForImageTextToText.from_pretrained(LLM, dtype=torch.bfloat16).to(device).eval()
    f = items()
    if a.test:
        f = f[f.stock != "MKT"].sample(a.test, random_state=1).reset_index(drop=True)
        t0 = time.time()
        P, summ = run(f, tok, model, a.batch, device)
        print(f"{len(f)} items in {time.time() - t0:.0f} s", flush=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            for k, r in enumerate(f.itertuples()):
                fields = "  ".join(f"{name} " + "/".join(f"{v:.2f}" for v in P[name][k]) for name, _, _ in FIELDS
                                   if name != "event") + f"  event {EVENTS[P['event'][k].argmax()]} {P['event'][k].max():.2f}"
                fh.write(f"===== {r.src} {r.stock} {r.name} {r.ts:%Y-%m-%d %H:%M}\n{r.text[:300]}\n--- {fields}\n"
                         f"--- 摘要：{summ[k]}\n\n")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    n_parts = (len(f) + PART - 1) // PART
    t0 = time.time()
    for part in range(n_parts):
        path = OUT / f"part_{part:04d}.npz"
        if path.exists():
            continue
        rows = f.iloc[part * PART:(part + 1) * PART]
        P, summ = run(rows, tok, model, a.batch, device)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, index=np.arange(part * PART, part * PART + len(rows)), summary=np.array(summ, dtype=object),
                 **{f"p_{k}": v for k, v in P.items()})
        tmp.rename(path)
        print(f"part {part + 1}/{n_parts} done, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
