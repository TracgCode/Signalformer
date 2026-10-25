"""Run the benchmark matrix (datasets x horizons x variants) and write a comparison table.

    python demo/run_benchmarks.py                                   # default variants and seeds, horizon 96
    python demo/run_benchmarks.py --seeds 2021                      # a single seed
    python demo/run_benchmarks.py --datasets ETTm1 Weather --variants base no_revin
    python demo/run_benchmarks.py --pred-lens 96 192 336 720
    python demo/run_benchmarks.py --dry-run                         # print the commands only

Each run is a separate `train.py` process; up to --jobs (default 2) run at the same time on the same GPU, and
each writes its output to demo/runs/benchmark/logs/. Results go to
demo/runs/benchmark/<dataset>_<pred_len>_<variant>_s<seed>.json; finished runs are skipped on the next invocation (use
--rerun to repeat them), so an interrupted sweep resumes where it stopped.

Variants that resolve to the same configuration for a dataset (e.g. `auto_dmodel` on ETT, where 'auto' gives the
same width as the preset) are trained once and the result is copied. Runs use torch.compile unless
--no-compile is given. The summary, including a naive repeat-last-value baseline, is written to
demo/runs/benchmark/summary.md.
"""

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch

from datasets import build_splits, num_signals
from train import effective_config
from train import parse_args as parse_train_args

DEMO_DIR = Path(__file__).resolve().parent
OUT_DIR = DEMO_DIR / "runs" / "benchmark"

DATASETS = ("ETTm1", "ETTm2", "Weather", "Traffic")

# Extra arguments applied to every variant of a dataset
DATASET_ARGS = {
    "Traffic": ["--kernel-size", "16", "--conv-stride", "8", "--n-rank", "8"],
}

# The train.py defaults are channel independence with the raw-input skip ("ci_raw"). Earlier rounds used other
# defaults, so every variant spells out its mode and keeps the meaning it had when its results were produced:
# SHARED = one shared token sequence for all signals, no raw skip (the default of the first rounds).
SHARED = ["--shared", "--no-raw-skip"]
# Patching with one token per 8 steps (kernel 16, rank 8), the best tokenizer of round 6
PATCH8 = ["--kernel-size", "16", "--conv-stride", "8", "--n-rank", "8"]
VARIANTS = {
    # Round 1-2: shared token sequence
    "base": SHARED,
    "auto_dmodel": [*SHARED, "--d-model", "auto"],
    "dropout_high": [*SHARED, "--dropout", "0.3"],
    "lr_low": [*SHARED, "--lr", "1e-4", "--epochs", "20"],
    "combined": [*SHARED, "--d-model", "auto", "--dropout", "0.3", "--lr", "1e-4", "--epochs", "20"],
    "muon": [*SHARED, "--optimizer", "muon"],
    "muon_lr_low": [*SHARED, "--optimizer", "muon", "--lr", "1e-4", "--epochs", "20"],
    "no_revin": [*SHARED, "--no-revin"],
    "with_time": [*SHARED, "--time-features", "hour,dow"],
    "no_gate_penalty": [*SHARED, "--gate-lambda", "0"],
    "large": [*SHARED, "--d-model", "64", "--encoder-layers", "2", "--decoder-layers", "2", "--dropout", "0.1"],
    "full_val": [*SHARED, "--val-windows", "0"],
    # Round 3-4: shared vs channel independence, front end, width, latent projection, raw skip, kernels
    "ci": ["--no-raw-skip"],
    "conv_bank": [*SHARED, "--front-end", "conv_bank"],
    "ci_conv_bank": ["--no-raw-skip", "--front-end", "conv_bank"],
    "half_dmodel": [*SHARED, "--d-model", "half"],
    "ci_proj": ["--no-raw-skip", "--ci-latents", "half"],
    "ci_proj_norm": ["--no-raw-skip", "--ci-latents", "half", "--latent-norm"],
    "raw": ["--shared"],
    "ci_raw": [],  # the current train.py defaults
    "ci_conv_bank_raw": ["--front-end", "conv_bank"],
    "multi_kernel": [*SHARED, "--kernel-size", "3,7,15"],
    "ci_conv_bank_mk": ["--no-raw-skip", "--front-end", "conv_bank", "--kernel-size", "3,7,15"],
    # Round 5 (on the ci_raw defaults). No moving-average decomposition: the raw signal goes straight into the
    # tokenizer conv, with more filters (n_rank) and longer kernels to make up for the missing bands
    "no_ma_r4": ["--front-end", "none", "--n-rank", "4"],
    "no_ma_long": ["--front-end", "none", "--n-rank", "4", "--kernel-size", "48"],
    "no_ma_mk": ["--front-end", "none", "--n-rank", "4", "--kernel-size", "4,16,48"],
    # Patching: one token per stride steps
    "patch8": PATCH8,
    "patch4": ["--kernel-size", "8", "--conv-stride", "4", "--n-rank", "4"],
    "patch1": ["--kernel-size", "3", "--conv-stride", "1", "--n-rank", "1"],
    # Frequency-domain memory, and a longer input window where it can resolve slower cycles
    "freq": ["--freq-tokens"],
    "seq336": ["--seq-len", "336"],
    "freq_seq336": ["--seq-len", "336", "--freq-tokens"],
    # Tree-merging architecture (tree_model.py) with each pairing method
    "tree_index": ["--architecture", "tree", "--pairing", "index"],
    "tree_corr": ["--architecture", "tree", "--pairing", "corr"],
    "tree_random": ["--architecture", "tree", "--pairing", "random"],
    "tree_soft": ["--architecture", "tree", "--pairing", "soft"],
    "tree_sampled": ["--architecture", "tree", "--pairing", "sampled"],
    # Diagnostic for the ci_proj collapse on Traffic (all gates closed in epoch 1): same run without gate penalty
    "ci_proj_nogate": ["--no-raw-skip", "--ci-latents", "half", "--gate-lambda", "0"],
    # Round 7. Raw input as a decoder cross-attention memory instead of (or besides) the raw skip
    "raw_xattn": ["--no-raw-skip", "--raw-tokens"],
    "patch8_raw_xattn": [*PATCH8, "--no-raw-skip", "--raw-tokens"],
    "patch8_raw_both": [*PATCH8, "--raw-tokens"],
    # Longer patches, patching without the MA decomposition (one band, so rank 16 for the same width), a longer
    # input window, and more dropout against the early overfitting seen with patch8
    "patch16": ["--kernel-size", "32", "--conv-stride", "16", "--n-rank", "16"],
    "patch8_no_ma": [*PATCH8[:-1], "16", "--front-end", "none"],
    "patch8_seq336": [*PATCH8, "--seq-len", "336"],
    "patch8_freq_seq336": [*PATCH8, "--seq-len", "336", "--freq-tokens"],
    "patch8_drop": [*PATCH8, "--dropout", "0.3"],
    # Cross Transformer: channel-independent tokenizer and encoder, then a decoder in which every signal
    # cross-attends to every signal's memory in signal order; the raw input through the skip or the decoder
    "cross": ["--architecture", "cross"],
    "cross_raw_xattn": ["--architecture", "cross", "--no-raw-skip", "--raw-tokens"],
    "cross_patch8": ["--architecture", "cross", *PATCH8],
    "cross_patch8_raw_xattn": ["--architecture", "cross", *PATCH8, "--no-raw-skip", "--raw-tokens"],
    # Learnable gate per (receiving, sending) signal pair, starting as plain channel independence, with L1 sparsity
    "cross_patch8_gate": ["--architecture", "cross", *PATCH8, "--edge-gates"],
}
# seq336 (round 6) is the reference for patch8_seq336; on Traffic, which always patches by 8, the two are the same
# configuration and share results. Frequency tokens stay available (--freq-tokens, the freq* variants) but are left
# out of the default sweep: they only helped at input 336 on two datasets, within noise.
DEFAULT_VARIANTS = ["ci_raw", "patch8", "seq336", "raw_xattn", "patch8_raw_xattn", "patch8_raw_both", "patch16",
                    "patch8_no_ma", "patch8_seq336", "patch8_drop", "cross", "cross_raw_xattn", "cross_patch8",
                    "cross_patch8_raw_xattn", "cross_patch8_gate"]
DEFAULT_SEEDS = [2021, 2022, 2023]
# Variants that only run on some datasets (others show "–" in the summary)
VARIANT_DATASETS = {
    # Learnable pairings are validated on the smaller datasets first
    "tree_soft": ("ETTm1", "ETTm2", "Weather"),
    "tree_sampled": ("ETTm1", "ETTm2", "Weather"),
    "ci_proj_nogate": ("Traffic",),
    # C cross-attentions per decoder layer over C sequences: too costly for Traffic's 862 signals
    "cross": ("ETTm1", "ETTm2", "Weather"),
    "cross_raw_xattn": ("ETTm1", "ETTm2", "Weather"),
    "cross_patch8": ("ETTm1", "ETTm2", "Weather"),
    "cross_patch8_raw_xattn": ("ETTm1", "ETTm2", "Weather"),
    "cross_patch8_gate": ("ETTm1", "ETTm2", "Weather"),
}
VARIANT_OVERRIDES = {
    # Traffic's "large" doubles its own preset width instead
    ("Traffic", "large"): [*SHARED, "--d-model", "128", "--encoder-layers", "2", "--decoder-layers", "2",
                           "--dropout", "0.1"],
    # Kernels must cover Traffic's patch stride of 8
    ("Traffic", "multi_kernel"): [*SHARED, "--kernel-size", "8,16,32"],
    ("Traffic", "ci_conv_bank_mk"): ["--no-raw-skip", "--front-end", "conv_bank", "--kernel-size", "8,16,32"],
    ("Traffic", "no_ma_r4"): ["--front-end", "none", "--n-rank", "16"],
    ("Traffic", "no_ma_long"): ["--front-end", "none", "--n-rank", "32", "--kernel-size", "48"],
    ("Traffic", "no_ma_mk"): ["--front-end", "none", "--n-rank", "8", "--kernel-size", "8,16,48"],
    # 431 latents per window: 8 windows per batch = 3448 sequences, as for channel independence with batch 4
    ("Traffic", "ci_proj"): ["--no-raw-skip", "--ci-latents", "half", "--batch-size", "8"],
    ("Traffic", "ci_proj_norm"): ["--no-raw-skip", "--ci-latents", "half", "--latent-norm", "--batch-size", "8"],
    ("Traffic", "ci_proj_nogate"): ["--no-raw-skip", "--ci-latents", "half", "--gate-lambda", "0",
                                    "--batch-size", "8"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--pred-lens", nargs="+", type=int, default=[96])
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                   help="each configuration runs once per seed; the summary reports mean ± std")
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--jobs", type=int, default=2, help="runs executed at the same time")
    p.add_argument("--no-compile", action="store_true", help="do not pass --compile to train.py")
    p.add_argument("--rerun", action="store_true", help="repeat runs that already have a result file")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="arguments after -- are passed to every train.py run")
    args = p.parse_args()
    args.extra = [a for a in args.extra if a != "--"]
    return args


def result_path(dataset: str, pred_len: int, variant: str, seed: int) -> Path:
    return OUT_DIR / f"{dataset}_{pred_len}_{variant}_s{seed}.json"


@dataclass
class Job:
    dataset: str
    pred_len: int
    variant: str
    seed: int
    train_argv: list[str]  # train.py arguments that define the configuration
    duplicate_of: "Job | None" = None

    @property
    def name(self) -> str:
        return f"{self.dataset}_{self.pred_len}_{self.variant}_s{self.seed}"

    @property
    def output(self) -> Path:
        return result_path(self.dataset, self.pred_len, self.variant, self.seed)

    def command(self, compile_model: bool) -> list[str]:
        cmd = [sys.executable, "-u", str(DEMO_DIR / "train.py"), *self.train_argv, "--tag", self.variant]
        if compile_model:
            cmd.append("--compile")
        return [*cmd, "--output", str(self.output)]


def build_jobs(args: argparse.Namespace) -> list[Job]:
    """All jobs, with each repeated configuration linked to the first job that trains it."""
    jobs, first_by_config = [], {}
    for pred_len in args.pred_lens:
        for dataset in args.datasets:
            for variant in args.variants:
                if dataset not in VARIANT_DATASETS.get(variant, (dataset,)):
                    continue
                for seed in args.seeds:
                    train_argv = [
                        "--dataset", dataset, "--seq-len", str(args.seq_len), "--pred-len", str(pred_len),
                        *DATASET_ARGS.get(dataset, []),
                        *(VARIANT_OVERRIDES.get((dataset, variant)) or VARIANTS[variant]),
                        *args.extra,
                        "--seed", str(seed),
                    ]
                    job = Job(dataset, pred_len, variant, seed, train_argv)
                    config = effective_config(parse_train_args(train_argv), num_signals(dataset))
                    key = json.dumps(config, sort_keys=True)
                    job.duplicate_of = first_by_config.get(key)
                    first_by_config.setdefault(key, job)
                    jobs.append(job)
    return jobs


def copy_result(source: Job, duplicate: Job) -> None:
    result = json.loads(source.output.read_text())
    result["args"]["tag"] = duplicate.variant
    result["duplicate_of"] = source.variant
    duplicate.output.write_text(json.dumps(result, indent=2))


def naive_baseline(dataset: str, seq_len: int, pred_len: int) -> tuple[float, float]:
    """MSE / MAE of repeating the last input value, on the same z-scored test windows as train.py."""
    test = build_splits(dataset, seq_len, pred_len, time_fields=())["test"]
    windows = test.values.unfold(1, seq_len + pred_len, 1)  # (C, num_windows, seq_len + pred_len)
    err = windows[..., seq_len:] - windows[..., seq_len - 1 : seq_len]
    return err.pow(2).mean().item(), err.abs().mean().item()


def format_cell(runs: list[tuple[float, float]], num_seeds: int, best: bool) -> str:
    """'mean MSE ± std / mean MAE'; std is omitted for a single run, and a partial seed count is noted."""
    mse = torch.tensor([r[0] for r in runs], dtype=torch.float64)
    mae = sum(r[1] for r in runs) / len(runs)
    text = f"{mse.mean():.3f}"
    if len(runs) > 1:
        text += f" ± {mse.std():.3f}"
    text += f" / {mae:.3f}"
    if 1 < num_seeds != len(runs):
        text += f" (n={len(runs)})"
    return f"**{text}**" if best else text


def write_summary(args: argparse.Namespace, failures: list[str]) -> str:
    lines = [
        "# Benchmark summary",
        "",
        f"Input length {args.seq_len}, seeds {', '.join(map(str, args.seeds))}. Test MSE (mean ± std over seeds) "
        "/ MAE (mean) on z-scored data; bold marks the lowest mean MSE in each row.",
        "",
    ]
    columns = ["naive", *args.variants]
    for pred_len in args.pred_lens:
        lines += [f"## Horizon {pred_len}", "", "| Dataset | " + " | ".join(columns) + " |",
                  "|---|" + "---|" * len(columns)]
        for dataset in args.datasets:
            scores: dict[str, list[tuple[float, float]]] = {}
            try:
                scores["naive"] = [naive_baseline(dataset, args.seq_len, pred_len)]  # deterministic
            except FileNotFoundError:
                pass
            for variant in args.variants:
                runs = []
                for seed in args.seeds:
                    path = result_path(dataset, pred_len, variant, seed)
                    if path.exists():
                        r = json.loads(path.read_text())
                        runs.append((r["test_mse"], r["test_mae"]))
                if runs:
                    scores[variant] = runs
            means = {c: sum(r[0] for r in runs) / len(runs) for c, runs in scores.items()}
            best = min(means.values(), default=None)
            cells = [
                format_cell(scores[c], 1 if c == "naive" else len(args.seeds), means[c] == best)
                if c in scores else "–"
                for c in columns
            ]
            lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Variants", ""]
    duplicates = []
    for pred_len in args.pred_lens:
        for dataset in args.datasets:
            for variant in args.variants:
                path = result_path(dataset, pred_len, variant, args.seeds[0])
                if path.exists():
                    source = json.loads(path.read_text()).get("duplicate_of")
                    if source:
                        duplicates.append(f"{dataset} / {pred_len}: `{variant}` is the same configuration as `{source}`")
    lines += [f"- `{name}`: {' '.join(VARIANTS[name]) or 'train.py defaults'}" for name in args.variants]
    if duplicates:
        lines += ["", "## Shared results", "", *[f"- {d}" for d in duplicates]]
    if failures:
        lines += ["", "## Failed runs", "", *[f"- {f}" for f in failures]]
    summary = "\n".join(lines) + "\n"
    (OUT_DIR / "summary.md").write_text(summary)
    return summary


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = OUT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    if not args.dry_run and not torch.cuda.is_available():
        print("warning: no GPU detected; full training runs on CPU will be very slow")

    jobs = build_jobs(args)
    to_train = [j for j in jobs if j.duplicate_of is None and (args.rerun or not j.output.exists())]
    duplicates = [j for j in jobs if j.duplicate_of is not None and (args.rerun or not j.output.exists())]
    print(f"{len(jobs)} jobs: {len(to_train)} to train, {len(duplicates)} sharing another job's configuration, "
          f"{len(jobs) - len(to_train) - len(duplicates)} already done; up to {args.jobs} at a time")
    for job in duplicates:
        print(f"  {job.name} = {job.duplicate_of.name if job.duplicate_of else ''}")
    if args.dry_run:
        for job in to_train:
            print(" ".join(job.command(not args.no_compile)[1:]))
        return

    pending, running, failures = deque(to_train), {}, []
    done = 0
    while pending or running:
        while pending and len(running) < args.jobs:
            job = pending.popleft()
            log = open(log_dir / f"{job.name}.log", "w")
            process = subprocess.Popen(job.command(not args.no_compile), stdout=log, stderr=subprocess.STDOUT)
            running[process] = (job, log, time.time())
            print(f"start  {job.name}", flush=True)
        time.sleep(2)
        for process in [p for p in running if p.poll() is not None]:
            job, log, start = running.pop(process)
            log.close()
            done += 1
            status = "done  " if process.returncode == 0 else "FAILED"
            print(f"{status} {job.name}  {time.time() - start:.0f}s  [{done}/{len(to_train)}]", flush=True)
            if process.returncode != 0:
                failures.append(f"{job.dataset} / {job.pred_len} / {job.variant} (see logs/{job.name}.log)")

    for job in duplicates:
        if job.duplicate_of is not None and job.duplicate_of.output.exists():
            copy_result(job.duplicate_of, job)

    print(write_summary(args, failures))


if __name__ == "__main__":
    main()
