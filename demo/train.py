"""Train and evaluate Signalformer on a long-term forecasting benchmark.

    python demo/train.py --dataset ETTm1 --pred-len 96
    python demo/train.py --dataset Traffic --pred-len 96 --conv-stride 8 --kernel-size 16 --n-rank 8

Reports MSE / MAE on z-scored test data (the convention used by Informer, Autoformer, PatchTST, ...), and writes
the result to demo/runs/<dataset>_<pred_len>_<timestamp>.json.

FI2010 is a classification task instead (mid-price up / stationary / down after --pred-len events, one of 10, 20,
30, 50, 100): the model classifies each input window, trains with cross-entropy and reports accuracy and macro F1.

    python demo/train.py --dataset FI2010 --pred-len 10

Speed: every split is kept on the training device and batches are sliced there directly, without a DataLoader.
Early stopping checks a fixed, evenly spaced subset of validation windows (--val-windows); the best model is
then scored on the full validation and test splits, keeping every window (no drop_last).
"""

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Signalformer  # noqa: E402
from baselines import CycleNet, PAMNet  # noqa: E402
from polyconv import PolyConv, PolyMLP  # noqa: E402
from tree_model import PAIRINGS, TreeSignalformer, correlation_embedding, correlation_tree, random_tree  # noqa: E402

from datasets import (  # noqa: E402
    LOB_ALL_FEATURES,
    LOB_CLASSES,
    LOB_DATASETS,
    LOB_FEATURES,
    LOB_HORIZONS,
    STOCK_DATASETS,
    TIME_FIELDS,
    ForecastWindows,
    LobWindows,
    StockWindows,
    build_lob_splits,
    build_splits,
    build_stock_splits,
)

RUNS_DIR = Path(__file__).resolve().parent / "runs"
BASELINES = ("cyclenet_linear", "cyclenet_mlp", "pamnet")  # reimplemented published models (baselines.py)

# Per-dataset defaults; every value can be overridden on the command line. Models are kept small: the first
# benchmark runs overfit within two or three epochs at d_model=64 with two encoder and two decoder layers.
PRESETS = {
    "ETTm1": dict(d_model=32, encoder_layers=1, decoder_layers=1, dropout=0.2, batch_size=64, lr=1e-3),
    "ETTm2": dict(d_model=32, encoder_layers=1, decoder_layers=1, dropout=0.2, batch_size=64, lr=1e-3),
    "Weather": dict(d_model=32, encoder_layers=1, decoder_layers=1, dropout=0.2, batch_size=64, lr=1e-3),
    "Traffic": dict(d_model=64, encoder_layers=2, decoder_layers=1, dropout=0.1, batch_size=16, lr=5e-4),
    # 40 order book features; with channel independence a batch of 64 windows is 2560 sequences
    "FI2010": dict(d_model=32, encoder_layers=1, decoder_layers=1, dropout=0.2, batch_size=64, lr=1e-3),
    # 100 stocks per window and about 1,000 training days: a batch of 16 days is 1,600 sequences
    "TWStock": dict(d_model=32, encoder_layers=1, decoder_layers=1, dropout=0.2, batch_size=16, lr=1e-3),
}
# Steps per day, the period of --cycle-len daily; steps per week, the period of --cycle-len weekly (TWStock has one
# step per weekday and no cycle within a day)
DAILY_STEPS = {"ETTm1": 96, "ETTm2": 96, "Weather": 144, "Traffic": 24}
WEEKLY_STEPS = {"TWStock": 5}
# Channel independence (the default) runs every signal as its own sequence: Traffic needs smaller batches
CI_PRESETS = {"Traffic": dict(batch_size=4)}

# Arguments that do not change what is trained (used to detect duplicate configurations)
NON_CONFIG_ARGS = {"tag", "output", "device", "compile", "log_every", "checkpoint", "ckpt_every"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(PRESETS))
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int,
                   help="forecast horizon, usually 96 (the default), 192, 336 or 720; for FI2010 the label horizon "
                   f"in events, one of {', '.join(map(str, LOB_HORIZONS))} (default 10)")
    p.add_argument("--lob-features", type=int, default=LOB_FEATURES, choices=(LOB_FEATURES, LOB_ALL_FEATURES),
                   help="FI2010 inputs: the 40 order book features, or 144 with the dataset's handcrafted features")
    # Model
    p.add_argument("--d-model", help="a multiple of num_heads * 8; 'auto' for 2 x signals clamped to [32, 256]; "
                   "'half' for signals / 2, at least 32")
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--kernel-size", default="3",
                   help="tokenizer conv kernel size, or comma-separated sizes for parallel multi-kernel convs")
    p.add_argument("--conv-stride", type=int, default=1)
    p.add_argument("--n-rank", type=int, default=1)
    p.add_argument("--no-cross-channel", action="store_true", help="group the tokenizer channel conv by signal")
    p.add_argument("--ma-windows", default="4,16,48",
                   help="comma-separated, strictly increasing; also the kernel sizes of --front-end conv_bank")
    p.add_argument("--front-end", choices=("ma", "conv_bank", "none"), default="ma",
                   help="moving-average decomposition, a learnable bank of causal depthwise convolutions, or none "
                   "(raw signal straight into the tokenizer conv, no L0 gate)")
    p.add_argument("--bank-filters", type=int, default=8, help="conv_bank filters per kernel size")
    p.add_argument("--raw-skip", action=argparse.BooleanOptionalAction, default=True,
                   help="also project the normalized raw input of each token's patch straight into the token")
    p.add_argument("--freq-tokens", action="store_true",
                   help="add rFFT bins of the input window as a memory the decoder cross-attends to")
    p.add_argument("--raw-tokens", action="store_true",
                   help="add every normalized input step as a memory the decoder cross-attends to (with RoPE)")
    p.add_argument("--encoder-layers", type=int)
    p.add_argument("--decoder-layers", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--no-revin", action="store_true", help="disable reversible instance normalization")
    p.add_argument("--shared", action="store_true",
                   help="feed all signals into one shared token sequence instead of the default channel "
                   "independence (every signal through the model separately with shared weights)")
    p.add_argument("--channel-independent", action="store_true",
                   help="the default; accepted so older commands keep working")
    p.add_argument("--ci-latents", default="0",
                   help="with --channel-independent: project the signals to this many latent signals first "
                   "(an integer, or 'half' for ceil(signals / 2)); 0 = off")
    p.add_argument("--latent-norm", action="store_true",
                   help="with --ci-latents: normalize each latent over its window before the model")
    # tree: tree-merging architecture (tree_model.py); cross: Cross Transformer, channel-independent encoders and
    # a decoder in which every signal cross-attends to every signal's memory in signal order
    # cyclenet_linear / cyclenet_mlp / pamnet: reimplemented published models (baselines.py), with --cycle-len;
    # --d-model is the MLP hidden size of cyclenet_mlp and the embedding width of pamnet
    p.add_argument("--architecture", choices=("signalformer", "tree", "cross", "polyconv", "polymlp", *BASELINES),
                   default="signalformer")
    p.add_argument("--cross-parallel", action="store_true",
                   help="cross architecture: run the cross-attentions to all signals in parallel and merge them with "
                   "one projection, like the heads of multi-head attention, instead of one after another")
    p.add_argument("--cross-fused", nargs="?", const="ffn", choices=("ffn", "mlp"),
                   help="with --architecture cross: concatenate the encoded tokens of all signals at each position, "
                   "project them to d_model ('ffn': C*d -> 4d -> d; 'mlp': one layer C*d -> d, then GELU), and let "
                   "every signal's decoder cross-attend to that one memory")
    p.add_argument("--fused-own", action="store_true",
                   help="with --cross-fused: add each signal's own encoded tokens to the fused memory")
    p.add_argument("--edge-gates", nargs="?", const="l1", choices=("l1", "l0"),
                   help="cross architecture: learnable gate per (receiving, sending) signal pair, starting as the "
                   "identity (plain channel independence); l1 (the default) penalizes the gate values, l0 adds a "
                   "Hard Concrete mask per cross-signal edge and penalizes the fraction of open edges")
    p.add_argument("--edge-lambda", type=float,
                   help="weight of the edge penalty (default 1e-3 for l1, 1e-2 for l0)")
    p.add_argument("--edge-mask-lr", type=float,
                   help="learning rate of the l0 edge masks (default: --gate-lr)")
    p.add_argument("--cycle-len", default="0",
                   help="residual cycle forecasting with a learnable cycle of this many steps per signal; 'daily' "
                   "uses one day of the dataset (ETTm 96, Weather 144, Traffic 24); 0 disables it")
    p.add_argument("--phase-edges", action="store_true",
                   help="with --cycle-len, --cross-parallel and --edge-gates: add a learnable offset per phase "
                   "(step modulo the cycle length) to the C x C edge gates, as TQNet's periodic queries")
    p.add_argument("--cycle-harmonics", type=int, default=0,
                   help="with --cycle-len: describe the cycle by this many harmonics of its period (frequency "
                   "domain, evaluated at each step) instead of a table of values; every harmonic spans the same "
                   "cycles as the table with -1. 0 keeps the table")
    p.add_argument("--cycle-weekly", type=int, default=0,
                   help="with --cycle-harmonics: add this many harmonics of a week (7 times the cycle length)")
    p.add_argument("--phase-harmonics", type=int, default=0,
                   help="with --phase-edges: describe the phase offsets of every edge by this many harmonics "
                   "instead of a table of one offset per phase")
    p.add_argument("--rot-conv", type=int, default=0,
                   help="with --cycle-len: rotate the outputs of this many pairs of tokenizer filters by harmonics "
                   "1..P of the time of day of each token (a kernel that turns with the phase); at most n_rank / 2")
    p.add_argument("--rot-offsets", action="store_true",
                   help="with --rot-conv: add a learnable angle per signal and filter pair")
    p.add_argument("--carrier", choices=("none", "table", "fourier"), default="none",
                   help="with --cycle-len: modulate the encoder memory with PAMNet's phase and amplitude carriers, "
                   "a table of one carrier per phase or --carrier-harmonics harmonics")
    p.add_argument("--carrier-harmonics", type=int, default=3, help="with --carrier fourier: harmonics of the day")
    p.add_argument("--share-layers", action="store_true",
                   help="adjacent pairs of encoder blocks and of decoder blocks share all their weights")
    p.add_argument("--cycle-norm", choices=("twice", "once"), default="twice",
                   help="with --cycle-len: 'twice' lets RevIN normalize the residual again (round 15), 'once' "
                   "normalizes each window only before the cycle is subtracted, as in CycleNet")
    p.add_argument("--tree-widths", default="",
                   help="comma-separated width per merge level (multiples of num_heads * 8); the last one repeats; "
                   "empty = d_model at every level")
    p.add_argument("--pairing", choices=PAIRINGS, default="corr", help="how the tree pairs sequences")
    p.add_argument("--pair-lambda", type=float, default=1e-2,
                   help="soft pairing: weight of the concentration / usage penalty, ramped up from 0")
    p.add_argument("--pair-warmup", type=float, default=0.3,
                   help="soft pairing: fraction of training steps over which --pair-lambda ramps up")
    p.add_argument("--pair-lr", type=float, default=1e-2, help="learning rate of the pairing parameters")
    p.add_argument("--pair-temp", default="1.0,0.1",
                   help="sampled pairing: start,end sampling temperature, annealed linearly")
    p.add_argument("--time-features", default="none",
                   help=f"comma-separated subset of {','.join(TIME_FIELDS)}, or 'none'")
    # Optimization
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--gate-lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--gate-lambda", type=float, default=1e-2,
                   help="weight of the L0 gate penalty (expected fraction of open gates)")
    p.add_argument("--poly-kernel", type=int, default=8, help="polyconv: length of the learnable filters")
    p.add_argument("--poly-filters", type=int, default=2, help="polyconv: learnable filters per input feature")
    p.add_argument("--poly-windows", default="5,10,20,40", help="polyconv: polynomial fit windows (steps)")
    p.add_argument("--poly-degrees", default="0,1,2", help="polyconv: polynomial degrees")
    p.add_argument("--poly-l2", type=float, default=1.0,
                   help="polyconv: ridge penalty on the head weights, added to the mean training loss")
    p.add_argument("--poly-readouts", default="next,slope,revert",
                   help="polyconv: readout kinds of every polynomial fit (next, slope, revert)")
    p.add_argument("--poly-sum-penalty", type=float, default=0.0,
                   help="polyconv: penalty on (sum of each filter - 1)^2; > 0 also turns off the filter weight norm")
    p.add_argument("--poly-deconv", choices=("", "target", "all"), default="",
                   help="polyconv: forecast in the filtered domain and deconvolve (monic filters); the head reads "
                        "the deconvolved forecasts of r only (target) or of every feature (all)")
    p.add_argument("--poly-deconv-filter", choices=("monic", "norm", "monic_norm"), default="monic",
                   help="polyconv --poly-deconv: last tap 1 (monic), the whole filter normalized (norm, divide by "
                        "the last tap), or last tap 1 and the other taps normalized (monic_norm)")
    p.add_argument("--poly-fixed", action="store_true",
                   help="polyconv: keep the filters at their initial identity (no learnable convolution)")
    p.add_argument("--mlp-window", type=int, default=5,
                   help="polymlp: the last days of past residuals and features the MLP reads")
    p.add_argument("--mlp-shrink", type=int, default=2, help="polymlp: hidden width = inputs / this")
    p.add_argument("--mlp-diff", action="store_true",
                   help="polymlp: day-to-day changes of the other features instead of their values")
    p.add_argument("--residual-base", type=Path,
                   help="TWStock: an .npz of fixed forecasts of r (demo/stock_poly.py --base) the model's output is "
                        "added to, so the model learns their residual")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw",
                   help="muon: Muon for the 2-D weights inside attention / FFN blocks, AdamW for the rest")
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--loss", choices=("mse", "hybrid_mae"), default="mse",
                   help="hybrid_mae: (1 - alpha) * MAE + alpha * MAE of the forecasts' Fourier coefficients (as "
                   "PAMNet, after FreDF), alpha = --freq-alpha")
    p.add_argument("--freq-alpha", type=float, default=0.0)
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16", help="bf16 autocast where supported")
    p.add_argument("--grad-clip", type=float, default=1.0, help="gradient norm clipping (0 = none)")
    p.add_argument("--scheduler", choices=("cosine", "onecycle", "type3"), default="cosine",
                   help="cosine decays from the start; onecycle warms up for the first 10%% of steps")
    p.add_argument("--warmup", type=float,
                   help="cosine scheduler: linear warmup over this fraction of all steps (default 0.01 for FI2010, "
                   "0 otherwise). Without it the first full-size Adam step can push every hidden unit of the wide "
                   "classifier head below zero, and the run never leaves the uniform prediction (loss ln 3)")
    p.add_argument("--log-every", type=int, default=200,
                   help="print the training loss, gradient norm and learning rate every this many steps (and every "
                   "step of the first 10); 0 = once per epoch only")
    p.add_argument("--patience", type=int, default=3,
                   help="early stopping on validation MSE (cross-entropy for FI2010)")
    p.add_argument("--val-windows", type=int, default=2048,
                   help="validation windows checked each epoch for early stopping (0 = all)")
    p.add_argument("--max-steps", type=int, help="stop each epoch after this many steps (smoke tests)")
    p.add_argument("--seed", type=int, default=2021)
    # Execution
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (CUDA only)")
    p.add_argument("--tag", default="", help="label stored in the result file and its name")
    p.add_argument("--output", type=Path, help="result JSON path (default: demo/runs/<dataset>_<pred_len>_...)")
    p.add_argument("--checkpoint", type=Path,
                   help="training state saved every --ckpt-every steps and at the end of every epoch; a run started "
                   "again with the same file resumes from it, and it is deleted once the result is saved (default: "
                   "ckpt/<output name>.pt next to --output, none without --output)")
    p.add_argument("--ckpt-every", type=int, default=500, help="steps between checkpoints (0 = end of epoch only)")
    args = p.parse_args(argv)
    args.channel_independent = not args.shared
    if args.ci_latents != "0" and args.shared:
        p.error("--ci-latents requires channel independence (drop --shared)")
    if args.architecture == "tree" and (args.shared or args.ci_latents != "0" or args.freq_tokens or args.raw_tokens):
        p.error("the tree architecture is channel-independent and supports none of --ci-latents, --freq-tokens "
                "and --raw-tokens")
    if args.architecture == "cross" and (args.shared or args.ci_latents != "0" or args.freq_tokens):
        p.error("the cross architecture is channel-independent and supports neither --ci-latents nor --freq-tokens")
    if args.cross_fused and (args.architecture != "cross" or args.cross_parallel or args.edge_gates or args.raw_tokens):
        p.error("--cross-fused requires --architecture cross, without --cross-parallel, --edge-gates or --raw-tokens")
    if args.fused_own and not args.cross_fused:
        p.error("--fused-own requires --cross-fused")
    if args.cross_parallel and args.architecture != "cross":
        p.error("--cross-parallel requires --architecture cross")
    if args.edge_gates and args.architecture != "cross":
        p.error("--edge-gates requires --architecture cross")
    if args.dataset in LOB_DATASETS:
        if args.pred_len is None:
            args.pred_len = LOB_HORIZONS[0]
        if args.pred_len not in LOB_HORIZONS:
            p.error(f"FI2010 horizons are {', '.join(map(str, LOB_HORIZONS))}")
        if args.architecture == "tree" or args.time_features != "none":
            p.error("FI2010 supports neither the tree architecture nor time features")
    elif args.pred_len is None:
        args.pred_len = 1 if args.dataset in STOCK_DATASETS else 96
    if args.dataset in STOCK_DATASETS and (args.architecture in ("tree", "cyclenet_linear", "cyclenet_mlp")
                                           or args.shared or args.ci_latents != "0" or args.freq_tokens
                                           or args.raw_tokens or args.time_features != "none"):
        p.error("TWStock supports the signalformer, cross and pamnet architectures with channel independence, "
                "without --ci-latents, --freq-tokens, --raw-tokens or time features")
    if args.architecture in ("polyconv", "polymlp") and args.dataset not in STOCK_DATASETS:
        p.error(f"--architecture {args.architecture} is for TWStock")
    if args.architecture == "polymlp" and args.residual_base is None:
        p.error("--architecture polymlp forecasts the residual of --residual-base")
    if args.residual_base is not None and args.dataset not in STOCK_DATASETS:
        p.error("--residual-base is for TWStock")
    if args.dataset in STOCK_DATASETS and args.loss != "mse":
        p.error("TWStock is trained with the MSE of the returns (--loss mse)")
    if args.edge_lambda is None:
        args.edge_lambda = 1e-2 if args.edge_gates == "l0" else 1e-3
    if args.edge_mask_lr is None:
        args.edge_mask_lr = args.gate_lr
    if args.warmup is None:
        args.warmup = 0.01 if args.dataset in LOB_DATASETS else 0.0
    if args.warmup and args.scheduler != "cosine":
        p.error("--warmup applies to the cosine scheduler; onecycle has its own warmup")
    if args.time_features == "none":
        args.time_fields = ()
    else:
        args.time_fields = tuple(args.time_features.split(","))
        unknown = set(args.time_fields) - set(TIME_FIELDS)
        if unknown:
            p.error(f"unknown time features: {', '.join(sorted(unknown))}")
    if args.cycle_len == "daily":
        if args.dataset not in DAILY_STEPS:
            p.error(f"--cycle-len daily is not defined for {args.dataset}")
        args.cycle_len = DAILY_STEPS[args.dataset]
    elif args.cycle_len == "weekly":
        if args.dataset not in WEEKLY_STEPS:
            p.error(f"--cycle-len weekly is not defined for {args.dataset}")
        args.cycle_len = WEEKLY_STEPS[args.dataset]
    else:
        args.cycle_len = int(args.cycle_len)
    if args.architecture in BASELINES and (not args.cycle_len or args.dataset in LOB_DATASETS):
        p.error(f"--architecture {args.architecture} requires --cycle-len and a forecasting dataset")
    if args.cycle_len and (args.dataset in LOB_DATASETS or args.architecture == "tree"):
        p.error("--cycle-len applies to forecasting with the plain or cross architecture")
    if (args.cycle_harmonics or args.cycle_weekly) and not args.cycle_len:
        p.error("--cycle-harmonics and --cycle-weekly require --cycle-len")
    if args.cycle_harmonics < 0:  # -1: every harmonic the period has (the same cycles as the table)
        args.cycle_harmonics = args.cycle_len // 2
    if args.cycle_weekly and not args.cycle_harmonics:
        p.error("--cycle-weekly requires --cycle-harmonics")
    if args.phase_harmonics and not args.phase_edges:
        p.error("--phase-harmonics requires --phase-edges")
    if args.phase_edges and not (args.cycle_len and args.cross_parallel and args.edge_gates):
        p.error("--phase-edges requires --cycle-len, --cross-parallel and --edge-gates")
    if (args.rot_conv or args.carrier != "none") and (not args.cycle_len or args.architecture not in ("signalformer", "cross")):
        p.error("--rot-conv and --carrier require --cycle-len (for the phase) and the signalformer or cross architecture")
    if args.rot_offsets and not args.rot_conv:
        p.error("--rot-offsets requires --rot-conv")
    if args.carrier != "none" and (args.shared or args.ci_latents != "0"):
        p.error("--carrier requires channel independence without --ci-latents")
    if args.share_layers and args.architecture == "tree":
        p.error("--share-layers applies to the plain or cross architecture")
    presets = dict(PRESETS[args.dataset])
    if args.channel_independent:
        presets.update(CI_PRESETS.get(args.dataset, {}))
    for key, value in presets.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


HEAD_DIM_MULTIPLE = 8  # FlashAttention kernels expect head widths that are multiples of 8


def resolve_d_model(value, num_signals: int, num_heads: int) -> int:
    """'auto': twice the number of signals, clamped to [32, 256]; 'half': half the number of signals, at least
    32. Both are rounded up to a multiple of num_heads * 8. Every attention head then has a width that is a multiple of 8, which keeps the attention
    on the FlashAttention kernel (and even, as RoPE requires). Explicit values only need an even head width (tiny
    models such as d_model 4 with one head then run on the slower math kernel)."""
    step = HEAD_DIM_MULTIPLE * num_heads
    if str(value) not in ("auto", "half"):
        d_model = int(value)
        if d_model % num_heads != 0 or (d_model // num_heads) % 2 != 0:
            raise ValueError(f"d_model={d_model} must split into {num_heads} heads of an even width")
        return d_model
    width = min(max(2 * num_signals, 32), 256) if value == "auto" else max(-(-num_signals // 2), 32)
    return -(-width // step) * step


def resolve_ci_latents(value, num_signals: int) -> int:
    return -(-num_signals // 2) if str(value) == "half" else int(value)


def parse_kernel_size(value: str) -> int | tuple[int, ...]:
    sizes = tuple(int(k) for k in str(value).split(","))
    return sizes[0] if len(sizes) == 1 else sizes


def effective_config(args: argparse.Namespace, num_signals: int) -> dict:
    """Everything that determines the trained model and its scores, with 'auto' values resolved."""
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in NON_CONFIG_ARGS}
    config["d_model"] = resolve_d_model(args.d_model, num_signals, args.num_heads)
    config["ci_latents"] = resolve_ci_latents(args.ci_latents, num_signals)
    config["time_fields"] = list(args.time_fields)
    return config


class DeviceWindows:
    """A split held on the training device; batches of windows are sliced there by index.

    Forecasting windows yield (x, y, t_in, t_out, phase) with y (B, C, pred_len) and phase (B,) the absolute time
    step of each window's first input step; classification windows (LobWindows) yield (x, y, None, None, None)
    with y (B,) the class of the window's last event. Stock windows (StockWindows) yield (x, y, None, None, phase)
    with y (B, S, pred_len) the scaled returns of the S stocks, NaN where a stock did not trade. Every batch also
    yields `base`, the fixed forecast (B, S, pred_len) the model's output is added to (--residual-base), or None."""

    def __init__(self, windows: ForecastWindows | LobWindows | StockWindows, device: str, use_time: bool) -> None:
        self.values = windows.values.to(device)  # (C, T)
        self.times = windows.times.to(device) if use_time else None  # (T, time_dim)
        self.seq_len = windows.seq_len
        self.labels = windows.labels.to(device) if isinstance(windows, LobWindows) else None  # (T,)
        self.starts = windows.starts.to(device) if isinstance(windows, (LobWindows, StockWindows)) else None
        self.targets = windows.targets.to(device) if isinstance(windows, StockWindows) else None  # (S, T)
        self.return_scale = getattr(windows, "return_scale", 1.0)  # scaled stock returns * this = returns
        base = getattr(windows, "base", None)
        self.base = None if base is None else base.to(device)  # (S, T) fixed forecast added to the model's
        span = windows.seq_len if self.labels is not None else windows.seq_len + windows.pred_len
        self.offsets = torch.arange(span, device=device)
        self.num_windows = len(windows)
        self.first = getattr(windows, "first", 0)
        self.device = device

    def batch(self, index: torch.Tensor):
        starts = self.starts[index] if self.starts is not None else index
        positions = starts[:, None] + self.offsets  # (B, span)
        segment = self.values[:, positions].permute(1, 0, 2)  # (B, C, span)
        if self.labels is not None:
            return segment, self.labels[starts + self.seq_len - 1], None, None, None, None
        x, y = segment[..., : self.seq_len], segment[..., self.seq_len :]
        base = None
        if self.targets is not None:
            y = self.targets[:, positions[:, self.seq_len :]].permute(1, 0, 2)  # (B, S, pred_len)
            if self.base is not None:
                base = self.base[:, positions[:, self.seq_len :]].permute(1, 0, 2)
        phase = starts + self.first
        if self.times is None:
            return x, y, None, None, phase, base
        times = self.times[positions]  # (B, seq_len + pred_len, time_dim)
        return x, y, times[:, : self.seq_len], times[:, self.seq_len :], phase, base

    def batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        subset: int = 0,
        order: torch.Tensor | None = None,
        start: int = 0,
    ) -> Iterator[tuple]:
        # order: a given window order (e.g. the shuffle of a resumed epoch); start: index of the first batch
        if order is None:
            order = self.window_order(shuffle, subset)
        stop = len(order) - len(order) % batch_size if drop_last else len(order)
        for i in range(start * batch_size, stop, batch_size):
            yield self.batch(order[i : i + batch_size])

    def window_order(self, shuffle: bool = False, subset: int = 0) -> torch.Tensor:
        if subset and subset < self.num_windows:
            # Fixed, evenly spaced windows so every epoch is scored on the same subset
            return torch.linspace(0, self.num_windows - 1, subset, device=self.device).long()
        if shuffle:
            return torch.randperm(self.num_windows, device=self.device)
        return torch.arange(self.num_windows, device=self.device)

    def num_batches(self, batch_size: int, drop_last: bool = False) -> int:
        return self.num_windows // batch_size if drop_last else -(-self.num_windows // batch_size)


def build_optimizers(model: "Signalformer | TreeSignalformer", args: argparse.Namespace) -> list[torch.optim.Optimizer]:
    """AdamW for everything, or Muon for hidden matrices plus AdamW for the rest.

    Muon orthogonalizes the update of each 2-D weight, which only suits hidden matrices: the Linear weights
    inside the encoder and decoder blocks (attention projections and FFNs). Biases, norms, RevIN, the L0 gate,
    the tokenizer convolutions, the decoder queries and the output head stay on AdamW, as recommended for
    Muon's input, output and scalar parameters. adjust_lr_fn="match_rms_adamw" scales Muon's step to the
    RMS of an AdamW step, so both optimizers share the same --lr.
    """
    gate = [model.tokenizer.gate.log_alpha] if model.tokenizer.gate is not None else []
    pairing = [p for name, p in model.named_parameters()
               if name.startswith(("query_logits", "memory_logits", "pair_embedding"))]
    # Edge gates start as the identity; weight decay would pull the self edges toward 0. The edge masks need the
    # faster gate learning rate: at the model learning rate (round 8) they never closed a single edge
    edges = [p for name, p in model.named_parameters() if name.endswith("edge_gate")]
    masks = [p for name, p in model.named_parameters() if name.endswith("edge_mask.log_alpha")]
    # The learnable cycle is a signal level, not a weight: no weight decay
    cycle = getattr(model, "cycle", None)
    cycle = [] if cycle is None else [cycle] if isinstance(cycle, torch.nn.Parameter) else list(cycle.parameters())
    muon_params = []
    if args.optimizer == "muon":
        if isinstance(model, TreeSignalformer):
            blocks = list(model.merges)
        else:
            blocks = [*model.encoder.blocks, *model.decoder.blocks]
        # dict.fromkeys drops the repeats of blocks shared by --share-layers
        muon_params = list(dict.fromkeys(m.weight for b in blocks for m in b.modules() if isinstance(m, torch.nn.Linear)))
    special = {id(p) for p in [*muon_params, *gate, *pairing, *edges, *masks, *cycle]}
    adamw_params = [p for p in model.parameters() if id(p) not in special]
    groups = [{"params": adamw_params, "lr": args.lr, "weight_decay": args.weight_decay}]
    if gate:
        groups.append({"params": gate, "lr": args.gate_lr, "weight_decay": 0.0})
    if pairing:
        groups.append({"params": pairing, "lr": args.pair_lr, "weight_decay": 0.0})
    if edges:
        groups.append({"params": edges, "lr": args.lr, "weight_decay": 0.0})
    if masks:
        groups.append({"params": masks, "lr": args.edge_mask_lr, "weight_decay": 0.0})
    if cycle:
        groups.append({"params": cycle, "lr": args.lr, "weight_decay": 0.0})
    optimizers: list[torch.optim.Optimizer] = [torch.optim.AdamW(groups)]
    if muon_params:
        optimizers.append(torch.optim.Muon(
            muon_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.muon_momentum,
            adjust_lr_fn="match_rms_adamw",
        ))
    return optimizers


@torch.no_grad()
def evaluate(forward, model, data: DeviceWindows, batch_size: int, autocast, subset: int = 0,
             max_steps: int | None = None) -> dict[str, Any]:
    """Forecasting: {"loss": MSE, "mse", "mae"}. Classification: {"loss": cross-entropy, "ce", "acc", "f1" (macro),
    "confusion" (true class x predicted class)}. Stocks, over the days a stock traded: the forecasting metrics of the
    scaled returns, plus "score" (the mean of (2 y_hat - 1) r with y_hat = [prediction > 0] and r the unscaled
    return), "acc", "f1" (macro) and "confusion" of up (r > 0) against down or flat (rows the true class)."""
    model.eval()
    classify = data.labels is not None
    loss = torch.zeros((), device=data.device)
    ae = torch.zeros((), device=data.device)
    updown = data.targets is not None
    classes = 2 if updown else len(LOB_CLASSES)
    confusion = torch.zeros(classes, classes, device=data.device)
    gain = torch.zeros((), device=data.device)
    count = 0
    for step, (x, y, t_in, t_out, phase, base) in enumerate(data.batches(batch_size, subset=subset)):
        if max_steps is not None and step >= max_steps:
            break
        with autocast():
            pred = forward(x, t_in, t_out, phase=phase) if takes_phase(model) else forward(x, t_in, t_out)
        if base is not None:
            pred = pred.float() + base  # the model forecasts the residual of the fixed forecast
        if classify:
            loss += torch.nn.functional.cross_entropy(pred.float(), y, reduction="sum")
            confusion.index_put_((y, pred.argmax(dim=1)), torch.ones_like(y, dtype=confusion.dtype), accumulate=True)
            count += y.numel()
        elif updown:
            valid = y.isfinite()  # stock days without trading have no target
            r, guess = y[valid], pred.float()[valid]
            up = guess > 0
            gain += torch.where(up, r, -r).sum()
            loss += (guess - r).pow(2).sum()
            ae += (guess - r).abs().sum()
            confusion.index_put_(((r > 0).long(), up.long()), torch.ones_like(r), accumulate=True)
            count += valid.sum()
        else:
            err = pred.float() - y  # metrics in fp32
            loss += err.pow(2).sum()
            ae += err.abs().sum()
            count += err.numel()
    count = int(count)
    if not classify and not updown:
        mse = loss.item() / count  # one host sync per evaluation
        return dict(loss=mse, mse=mse, mae=ae.item() / count)
    confusion = confusion.cpu()
    tp = confusion.diag()
    precision = tp / confusion.sum(dim=0).clamp(min=1)
    recall = tp / confusion.sum(dim=1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    if updown:
        mse = loss.item() / count
        return dict(loss=mse, mse=mse, mae=ae.item() / count, score=gain.item() / count * data.return_scale,
                    acc=(tp.sum() / count).item(), f1=f1.mean().item(), confusion=confusion.tolist())
    ce = loss.item() / count
    return dict(loss=ce, ce=ce, acc=(tp.sum() / count).item(), f1=f1.mean().item(), confusion=confusion.tolist())


def describe(metrics: dict[str, Any]) -> str:
    if "score" in metrics:
        return (f"mse {metrics['mse']:.4f}  score {metrics['score'] * 1e4:.2f} bp  acc {metrics['acc']:.4f} "
                f"f1 {metrics['f1']:.4f}")
    if "f1" in metrics:
        return f"ce {metrics['ce']:.4f} acc {metrics['acc']:.4f} f1 {metrics['f1']:.4f}"
    return f"mse {metrics['mse']:.4f} mae {metrics['mae']:.4f}"


def takes_phase(model: torch.nn.Module) -> bool:
    """Whether the model needs each window's absolute time step (a learnable cycle, or a baseline's carriers)."""
    return getattr(model, "cycle", None) is not None or getattr(model, "uses_phase", False)


def forecast_loss(pred: torch.Tensor, y: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    """Per-window training loss of (B, C, H) forecasts: MSE, or PAMNet's hybrid MAE in time and frequency."""
    if args.loss == "mse":
        # stock targets are NaN on days a stock did not trade: the mean over the valid ones
        valid = y.isfinite()
        err = torch.where(valid, pred - y, 0.0)
        return err.pow(2).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp(min=1)
    time_mae = (pred - y).abs().mean(dim=(1, 2))
    freq_mae = (torch.fft.rfft(pred, dim=-1) - torch.fft.rfft(y, dim=-1)).abs().mean(dim=(1, 2))
    return (1 - args.freq_alpha) * time_mae + args.freq_alpha * freq_mae


def build_model(args: argparse.Namespace, num_signals: int, train_values: torch.Tensor, num_classes: int = 0):
    """Signalformer, or TreeSignalformer with its pairing tree computed from the training split only."""
    if args.architecture in ("cyclenet_linear", "cyclenet_mlp"):
        hidden = args.d_model if args.architecture == "cyclenet_mlp" else 0
        return CycleNet(num_signals, args.seq_len, args.pred_len, args.cycle_len, hidden=hidden)
    if args.architecture == "polymlp":
        return PolyMLP(num_signals, args.signal_features, args.pred_len, args.mlp_window, args.dropout, args.mlp_diff,
                       args.mlp_shrink)
    if args.architecture == "polyconv":
        ints = lambda v: tuple(int(a) for a in v.split(","))
        return PolyConv(num_signals, args.signal_features, args.seq_len, args.pred_len, args.poly_kernel,
                        args.poly_filters, ints(args.poly_windows), ints(args.poly_degrees), args.poly_fixed,
                        args.poly_l2, tuple(args.poly_readouts.split(",")), args.poly_sum_penalty,
                        args.poly_deconv, args.poly_deconv_filter)
    if args.architecture == "pamnet":
        return PAMNet(num_signals, args.seq_len, args.pred_len, args.cycle_len, args.d_model, args.dropout,
                      signal_features=getattr(args, "signal_features", 1))
    common: dict[str, Any] = dict(
        in_channels=num_signals,
        in_len=args.seq_len,
        d_model=args.d_model,
        kernel_size=parse_kernel_size(args.kernel_size),
        output_length=args.pred_len,
        num_heads=args.num_heads,
        ma_windows=tuple(int(w) for w in args.ma_windows.split(",")),
        n_rank=args.n_rank,
        dropout=args.dropout,
        conv_stride=args.conv_stride,
        revin=not args.no_revin,
        front_end=args.front_end,
        bank_filters=args.bank_filters,
        raw_skip=args.raw_skip,
    )
    if args.architecture == "tree":
        tree = pair_embedding = None
        if args.pairing in ("corr", "soft"):
            tree = correlation_tree(train_values)
        elif args.pairing == "random":
            tree = random_tree(num_signals, seed=args.seed)
        elif args.pairing == "sampled":
            pair_embedding = correlation_embedding(train_values)
        widths = tuple(resolve_d_model(w, num_signals, args.num_heads) for w in args.tree_widths.split(",") if w)
        return TreeSignalformer(
            **common, tree_widths=widths, pairing=args.pairing, tree=tree, pair_embedding=pair_embedding
        )
    return Signalformer(
        **common,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        time_dim=2 * len(args.time_fields),
        out_channels=num_signals,
        cross_channel=not args.no_cross_channel,
        channel_independent=args.channel_independent,
        ci_latents=args.ci_latents,
        latent_norm=args.latent_norm,
        freq_tokens=args.freq_tokens,
        raw_tokens=args.raw_tokens,
        cross_decoder=args.architecture == "cross",
        cross_parallel=args.cross_parallel,
        cross_fused=args.cross_fused or False,
        fused_own=args.fused_own,
        edge_gates=args.edge_gates,
        num_classes=num_classes,
        cycle_len=args.cycle_len,
        cycle_renorm=args.cycle_norm == "twice",
        phase_edges=args.phase_edges,
        share_layers=args.share_layers,
        cycle_harmonics=args.cycle_harmonics,
        cycle_weekly=args.cycle_weekly,
        phase_harmonics=args.phase_harmonics,
        rot_conv=args.rot_conv,
        rot_offsets=args.rot_offsets,
        carrier="" if args.carrier == "none" else args.carrier,
        carrier_harmonics=args.carrier_harmonics if args.carrier == "fourier" else 0,
        signal_features=getattr(args, "signal_features", 1),
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        print("warning: training on CPU; this is only practical for smoke tests (--max-steps)")

    classify = args.dataset in LOB_DATASETS
    if classify:
        splits = build_lob_splits(args.seq_len, args.pred_len, features=args.lob_features)
    elif args.dataset in STOCK_DATASETS:
        splits = build_stock_splits(args.seq_len, args.pred_len, args.residual_base,
                                    base_feature=args.architecture == "polymlp")
    else:
        splits = build_splits(args.dataset, args.seq_len, args.pred_len, args.time_fields)
    num_signals = splits["train"].values.size(0)
    if args.dataset in STOCK_DATASETS:
        # one signal per stock, each with its features
        args.signal_features = splits["train"].num_features
        num_signals //= args.signal_features
    args.d_model = resolve_d_model(args.d_model, num_signals, args.num_heads)
    args.ci_latents = resolve_ci_latents(args.ci_latents, num_signals)
    use_time = len(args.time_fields) > 0
    train, val, test = (DeviceWindows(splits[s], args.device, use_time) for s in ("train", "val", "test"))

    model = build_model(args, num_signals, splits["train"].values, len(LOB_CLASSES) if classify else 0)
    model = model.to(args.device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{args.dataset}: {num_signals} signals, train/val/test windows = "
          f"{train.num_windows}/{val.num_windows}/{test.num_windows}, params = {num_params:,}")
    tree_model = model if isinstance(model, TreeSignalformer) else None
    gate = model.tokenizer.gate

    # The compiled module shares parameters with `model`; state_dict, penalty and the gate use `model`
    forward = model
    if args.compile:
        if args.device == "cuda":
            forward = torch.compile(model)
        else:
            print("warning: --compile is ignored on CPU")

    optimizers = build_optimizers(model, args)
    steps_per_epoch = train.num_batches(args.batch_size, drop_last=True)
    if args.max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
    total_steps = max(1, args.epochs * steps_per_epoch)
    temp_start, temp_end = (float(t) for t in args.pair_temp.split(","))
    schedulers = []
    for optimizer in optimizers:
        if args.scheduler == "onecycle":
            max_lr = [group["lr"] for group in optimizer.param_groups]
            schedulers.append(torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.1
            ))
        elif args.scheduler == "type3":
            # the CycleNet / TQNet code's 'type3': the full rate for 3 epochs, then x 0.8 every epoch
            schedulers.append(torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: 0.8 ** max(0, step // steps_per_epoch - 2)
            ))
        else:
            warmup_steps = round(args.warmup * total_steps)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
            if warmup_steps > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_steps)
                cosine = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], [warmup_steps])
            schedulers.append(cosine)
    if len(optimizers) > 1:
        print(f"muon: {sum(p.numel() for p in optimizers[1].param_groups[0]['params']):,} parameters")
    # bf16 autocast on GPUs that support it; no GradScaler needed
    use_bf16 = args.device == "cuda" and torch.cuda.is_bf16_supported() and args.precision == "bf16"

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16)

    best_val, best_state, bad_epochs, history, step_log = math.inf, None, 0, [], []
    tag = f"_{args.tag}" if args.tag else ""
    out = args.output or RUNS_DIR / f"{args.dataset}_{args.pred_len}{tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    checkpoint = args.checkpoint or (args.output.parent / "ckpt" / f"{args.output.stem}.pt" if args.output else None)
    config = effective_config(args, num_signals)

    def save_checkpoint(epoch: int, step: int, order: torch.Tensor | None, total: float) -> None:
        # epoch / step: where training continues; order: that epoch's window order (None: a new epoch)
        state = dict(
            config=config, epoch=epoch, step=step, order=None if order is None else order.cpu(), total=total,
            model=model.state_dict(), optimizers=[o.state_dict() for o in optimizers],
            schedulers=[s.state_dict() for s in schedulers], best_val=best_val, best_state=best_state,
            bad_epochs=bad_epochs, history=history, step_log=step_log, rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device == "cuda" else None,
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        partial = checkpoint.with_suffix(".partial")
        torch.save(state, partial)
        os.replace(partial, checkpoint)  # atomic: an interrupted save never leaves a broken checkpoint

    first_epoch, first_step, resumed_order, resumed_total = 1, 0, None, 0.0
    if checkpoint is not None and checkpoint.exists():
        state = torch.load(checkpoint, map_location=args.device, weights_only=False)
        if state["config"] != config:
            print(f"warning: {checkpoint} was saved with other settings; starting from scratch")
        else:
            model.load_state_dict(state["model"])
            for optimizer, saved in zip(optimizers, state["optimizers"]):
                optimizer.load_state_dict(saved)
            for scheduler, saved in zip(schedulers, state["schedulers"]):
                scheduler.load_state_dict(saved)
            best_val, best_state, bad_epochs = state["best_val"], state["best_state"], state["bad_epochs"]
            history, step_log = state["history"], state["step_log"]
            torch.set_rng_state(state["rng"].cpu())
            if state["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
            first_epoch, first_step, resumed_total = state["epoch"], state["step"], state["total"]
            if state["order"] is not None:
                resumed_order = state["order"].to(args.device)
            print(f"resumed from {checkpoint} at epoch {first_epoch}, step {first_step}")

    for epoch in range(first_epoch, args.epochs + 1):
        if bad_epochs >= args.patience:
            break  # the checkpoint was saved right before early stopping
        model.train()
        start = time.time()
        if resumed_order is not None:
            order, first, total = resumed_order, first_step, torch.tensor(resumed_total, device=args.device)
            resumed_order = None
        else:
            order, first, total = train.window_order(shuffle=True), 0, torch.zeros((), device=args.device)
        interval, interval_steps = torch.zeros((), device=args.device), 0
        batches = train.batches(args.batch_size, drop_last=True, order=order, start=first)
        for step, (x, y, t_in, t_out, phase, base) in enumerate(batches, start=first):
            if step >= steps_per_epoch:
                break
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            global_step = (epoch - 1) * steps_per_epoch + step
            if tree_model is not None and tree_model.pairing == "sampled":
                progress = global_step / max(1, total_steps - 1)
                tree_model.pair_temperature = temp_start + (temp_end - temp_start) * progress
            with autocast():
                pred = forward(x, t_in, t_out, phase=phase) if takes_phase(model) else forward(x, t_in, t_out)
                if classify:
                    sample_loss = torch.nn.functional.cross_entropy(pred.float(), y, reduction="none")  # (B,)
                else:
                    if base is not None:
                        pred = pred.float() + base  # the model forecasts the residual of the fixed forecast
                    sample_loss = forecast_loss(pred.float(), y, args)  # (B,)
                task_loss = sample_loss.mean()
                loss = task_loss + args.gate_lambda * model.penalty()
                if isinstance(model, PolyConv):
                    loss = loss + model.ridge_penalty()
                if args.edge_gates:
                    loss = loss + args.edge_lambda * model.edge_penalty()
            if tree_model is not None and tree_model.pairing == "soft":
                ramp = min(1.0, global_step / max(1.0, args.pair_warmup * total_steps))
                loss = loss + args.pair_lambda * ramp * tree_model.pairing_penalty()
            if tree_model is not None and tree_model.last_log_prob is not None:
                # REINFORCE for the sampled matchings: every window drew its own matching, and the mean loss of
                # the other windows in the batch is its baseline (leave-one-out)
                b = sample_loss.detach()
                baseline = (b.sum() - b) / max(1, b.numel() - 1)
                loss = loss + ((b - baseline) * tree_model.last_log_prob).mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip or math.inf)
            lr = optimizers[0].param_groups[0]["lr"]
            for optimizer, scheduler in zip(optimizers, schedulers):
                optimizer.step()
                scheduler.step()
            total += task_loss.detach()  # summed on the device to avoid a host sync every step
            interval += task_loss.detach()
            interval_steps += 1
            if args.log_every and (global_step < 10 or (global_step + 1) % args.log_every == 0):
                entry = dict(step=global_step + 1, loss=interval.item() / interval_steps,
                             grad_norm=grad_norm.item(), lr=lr)
                line = (f"step {global_step + 1:6d}/{total_steps}  train {entry['loss']:.4f}  "
                        f"grad {entry['grad_norm']:.3g}  lr {lr:.2e}")
                if classify:
                    # spread of the logits over the batch: near 0 the predictions no longer depend on the input
                    # (e.g. every hidden unit of the classifier head is below zero) and the run is stuck
                    entry["logit_spread"] = pred.detach().float().std(dim=0).mean().item()
                    line += f"  logit spread {entry['logit_spread']:.3g}"
                    if entry["logit_spread"] < 1e-3:
                        line += "  STUCK: predictions do not depend on the input"
                step_log.append(entry)
                print(line, flush=True)
                interval, interval_steps = torch.zeros((), device=args.device), 0
            if checkpoint is not None and args.ckpt_every and (step + 1) % args.ckpt_every == 0 and step + 1 < steps_per_epoch:
                save_checkpoint(epoch, step + 1, order, total.item())
        train_loss = total.item() / steps_per_epoch
        val_metrics = evaluate(forward, model, val, args.batch_size, autocast, args.val_windows, args.max_steps)
        val_metrics.pop("confusion", None)
        history.append(dict(epoch=epoch, train_loss=train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}))
        gates = f"open gates {int((gate.gate_values() > 0).sum())}/{gate.log_alpha.numel()}  " if gate else ""
        if args.edge_gates == "l0":
            model.eval()
            with torch.no_grad():
                weights = torch.stack(model.edge_weights())
            off = ~torch.eye(weights.size(-1), dtype=torch.bool, device=weights.device).expand_as(weights)
            gates += f"open edges {int((weights[off] != 0).sum())}/{int(off.sum())}  "
            model.train()
        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {describe(val_metrics)}  "
              f"{gates}{time.time() - start:.0f}s")
        if val_metrics["loss"] < best_val:
            best_val, bad_epochs = val_metrics["loss"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if checkpoint is not None:
            save_checkpoint(epoch + 1, 0, None, 0.0)
        if bad_epochs >= args.patience:
            print("early stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    # Final scores on the complete validation and test splits
    val_metrics = evaluate(forward, model, val, args.batch_size, autocast, 0, args.max_steps)
    test_metrics = evaluate(forward, model, test, args.batch_size, autocast, 0, args.max_steps)
    print(f"val   {describe(val_metrics)}  (full split)")
    print(f"test  {describe(test_metrics)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    out.write_text(json.dumps(dict(
        args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        num_params=num_params,
        **{f"val_{k}": v for k, v in val_metrics.items() if k != "loss"},
        **{f"test_{k}": v for k, v in test_metrics.items() if k != "loss"},
        history=history, steps=step_log, gate_values=gate.gate_values().tolist() if gate else [],
        tree=tree_model.tree if tree_model is not None else None,
        edge_gates=[w.tolist() for w in model.edge_weights()] if args.edge_gates else None,
        polyconv=model.report() if isinstance(model, PolyConv) else None,
    ), indent=2))
    print(f"saved {out}")
    if checkpoint is not None:
        checkpoint.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
