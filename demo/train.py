"""Train and evaluate Signalformer on a long-term forecasting benchmark.

    python demo/train.py --dataset ETTm1 --pred-len 96
    python demo/train.py --dataset Traffic --pred-len 96 --conv-stride 8 --kernel-size 16 --n-rank 8

Reports MSE / MAE on z-scored test data (the convention used by Informer, Autoformer, PatchTST, ...), and writes
the result to demo/runs/<dataset>_<pred_len>_<timestamp>.json.

Speed: every split is kept on the training device and batches are sliced there directly, without a DataLoader.
Early stopping checks a fixed, evenly spaced subset of validation windows (--val-windows); the best model is
then scored on the full validation and test splits, keeping every window (no drop_last).
"""

import argparse
import json
import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Signalformer  # noqa: E402

from datasets import TIME_FIELDS, ForecastWindows, build_splits  # noqa: E402

RUNS_DIR = Path(__file__).resolve().parent / "runs"

# Per-dataset defaults; every value can be overridden on the command line. Models are kept small: the first
# benchmark runs overfit within two or three epochs at d_model=64 with two encoder and two decoder layers.
PRESETS = {
    "ETTm1": dict(d_model=32, encoder_layers=1, decoder_layers=1, dropout=0.2, batch_size=64, lr=1e-3),
    "ETTm2": dict(d_model=32, encoder_layers=1, decoder_layers=1, dropout=0.2, batch_size=64, lr=1e-3),
    "Weather": dict(d_model=32, encoder_layers=1, decoder_layers=1, dropout=0.2, batch_size=64, lr=1e-3),
    "Traffic": dict(d_model=64, encoder_layers=2, decoder_layers=1, dropout=0.1, batch_size=16, lr=5e-4),
}

# Arguments that do not change what is trained (used to detect duplicate configurations)
NON_CONFIG_ARGS = {"tag", "output", "device", "compile"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(PRESETS))
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=96, help="usually 96, 192, 336 or 720")
    # Model
    p.add_argument("--d-model", help="a multiple of num_heads * 8, or 'auto' for 2 x signals clamped to [32, 256]")
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--conv-stride", type=int, default=1)
    p.add_argument("--n-rank", type=int, default=1)
    p.add_argument("--no-cross-channel", action="store_true", help="group the tokenizer channel conv by signal")
    p.add_argument("--ma-windows", default="4,16,48", help="comma-separated, strictly increasing")
    p.add_argument("--encoder-layers", type=int)
    p.add_argument("--decoder-layers", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--no-revin", action="store_true", help="disable reversible instance normalization")
    p.add_argument("--channel-independent", action="store_true",
                   help="run every signal through the model separately with shared weights (PatchTST-style)")
    p.add_argument("--time-features", default="none",
                   help=f"comma-separated subset of {','.join(TIME_FIELDS)}, or 'none'")
    # Optimization
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--gate-lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--gate-lambda", type=float, default=1e-2,
                   help="weight of the L0 gate penalty (expected fraction of open gates)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw",
                   help="muon: Muon for the 2-D weights inside attention / FFN blocks, AdamW for the rest")
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--scheduler", choices=("cosine", "onecycle"), default="cosine",
                   help="cosine decays from the start; onecycle warms up for the first 10%% of steps")
    p.add_argument("--patience", type=int, default=3, help="early stopping on validation MSE")
    p.add_argument("--val-windows", type=int, default=2048,
                   help="validation windows checked each epoch for early stopping (0 = all)")
    p.add_argument("--max-steps", type=int, help="stop each epoch after this many steps (smoke tests)")
    p.add_argument("--seed", type=int, default=2021)
    # Execution
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (CUDA only)")
    p.add_argument("--tag", default="", help="label stored in the result file and its name")
    p.add_argument("--output", type=Path, help="result JSON path (default: demo/runs/<dataset>_<pred_len>_...)")
    args = p.parse_args(argv)
    if args.time_features == "none":
        args.time_fields = ()
    else:
        args.time_fields = tuple(args.time_features.split(","))
        unknown = set(args.time_fields) - set(TIME_FIELDS)
        if unknown:
            p.error(f"unknown time features: {', '.join(sorted(unknown))}")
    for key, value in PRESETS[args.dataset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


HEAD_DIM_MULTIPLE = 8  # FlashAttention kernels expect head widths that are multiples of 8


def resolve_d_model(value, num_signals: int, num_heads: int) -> int:
    """'auto': twice the number of signals, clamped to [32, 256] and rounded up to a multiple of
    num_heads * 8. Every attention head then has a width that is a multiple of 8, which keeps the attention
    on the FlashAttention kernel (and even, as RoPE requires). Explicit values must satisfy the same rule."""
    step = HEAD_DIM_MULTIPLE * num_heads
    if str(value) != "auto":
        d_model = int(value)
        if d_model % step != 0:
            raise ValueError(f"d_model={d_model} must be a multiple of num_heads * {HEAD_DIM_MULTIPLE} = {step}")
        return d_model
    width = min(max(2 * num_signals, 32), 256)
    return -(-width // step) * step


def effective_config(args: argparse.Namespace, num_signals: int) -> dict:
    """Everything that determines the trained model and its scores, with 'auto' values resolved."""
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in NON_CONFIG_ARGS}
    config["d_model"] = resolve_d_model(args.d_model, num_signals, args.num_heads)
    config["time_fields"] = list(args.time_fields)
    return config


class DeviceWindows:
    """A split held on the training device; batches of windows are sliced there by index."""

    def __init__(self, windows: ForecastWindows, device: str, use_time: bool) -> None:
        self.values = windows.values.to(device)  # (C, T)
        self.times = windows.times.to(device) if use_time else None  # (T, time_dim)
        self.seq_len = windows.seq_len
        self.offsets = torch.arange(windows.seq_len + windows.pred_len, device=device)
        self.num_windows = len(windows)
        self.device = device

    def batch(self, starts: torch.Tensor):
        positions = starts[:, None] + self.offsets  # (B, seq_len + pred_len)
        segment = self.values[:, positions].permute(1, 0, 2)  # (B, C, seq_len + pred_len)
        x, y = segment[..., : self.seq_len], segment[..., self.seq_len :]
        if self.times is None:
            return x, y, None, None
        times = self.times[positions]  # (B, seq_len + pred_len, time_dim)
        return x, y, times[:, : self.seq_len], times[:, self.seq_len :]

    def batches(
        self, batch_size: int, shuffle: bool = False, drop_last: bool = False, subset: int = 0
    ) -> Iterator[tuple]:
        if subset and subset < self.num_windows:
            # Fixed, evenly spaced windows so every epoch is scored on the same subset
            order = torch.linspace(0, self.num_windows - 1, subset, device=self.device).long()
        elif shuffle:
            order = torch.randperm(self.num_windows, device=self.device)
        else:
            order = torch.arange(self.num_windows, device=self.device)
        stop = len(order) - len(order) % batch_size if drop_last else len(order)
        for i in range(0, stop, batch_size):
            yield self.batch(order[i : i + batch_size])

    def num_batches(self, batch_size: int, drop_last: bool = False) -> int:
        return self.num_windows // batch_size if drop_last else -(-self.num_windows // batch_size)


def build_optimizers(model: Signalformer, args: argparse.Namespace) -> list[torch.optim.Optimizer]:
    """AdamW for everything, or Muon for hidden matrices plus AdamW for the rest.

    Muon orthogonalizes the update of each 2-D weight, which only suits hidden matrices: the Linear weights
    inside the encoder and decoder blocks (attention projections and FFNs). Biases, norms, RevIN, the L0 gate,
    the tokenizer convolutions, the decoder queries and the output head stay on AdamW, as recommended for
    Muon's input, output and scalar parameters. adjust_lr_fn="match_rms_adamw" scales Muon's step to the
    RMS of an AdamW step, so both optimizers share the same --lr.
    """
    gate = model.tokenizer.gate.log_alpha
    muon_params = []
    if args.optimizer == "muon":
        blocks = [*model.encoder.blocks, *model.decoder.blocks]
        muon_params = [m.weight for b in blocks for m in b.modules() if isinstance(m, torch.nn.Linear)]
    muon_ids = {id(p) for p in muon_params}
    adamw_params = [p for p in model.parameters() if p is not gate and id(p) not in muon_ids]
    optimizers: list[torch.optim.Optimizer] = [torch.optim.AdamW([
        {"params": adamw_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": [gate], "lr": args.gate_lr, "weight_decay": 0.0},
    ])]
    if muon_params:
        optimizers.append(torch.optim.Muon(
            muon_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.muon_momentum,
            adjust_lr_fn="match_rms_adamw",
        ))
    return optimizers


@torch.no_grad()
def evaluate(forward, model, data: DeviceWindows, batch_size: int, autocast, subset: int = 0,
             max_steps: int | None = None) -> tuple[float, float]:
    model.eval()
    se = torch.zeros((), device=data.device)
    ae = torch.zeros((), device=data.device)
    count = 0
    for step, (x, y, t_in, t_out) in enumerate(data.batches(batch_size, subset=subset)):
        if max_steps is not None and step >= max_steps:
            break
        with autocast():
            pred = forward(x, t_in, t_out)
        err = pred.float() - y  # metrics in fp32
        se += err.pow(2).sum()
        ae += err.abs().sum()
        count += err.numel()
    return se.item() / count, ae.item() / count  # one host sync per evaluation


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        print("warning: training on CPU; this is only practical for smoke tests (--max-steps)")

    splits = build_splits(args.dataset, args.seq_len, args.pred_len, args.time_fields)
    num_signals = splits["train"].values.size(0)
    args.d_model = resolve_d_model(args.d_model, num_signals, args.num_heads)
    use_time = len(args.time_fields) > 0
    train, val, test = (DeviceWindows(splits[s], args.device, use_time) for s in ("train", "val", "test"))

    model = Signalformer(
        in_channels=num_signals,
        in_len=args.seq_len,
        d_model=args.d_model,
        kernel_size=args.kernel_size,
        output_length=args.pred_len,
        num_heads=args.num_heads,
        ma_windows=tuple(int(w) for w in args.ma_windows.split(",")),
        n_rank=args.n_rank,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        time_dim=2 * len(args.time_fields),
        out_channels=num_signals,
        conv_stride=args.conv_stride,
        cross_channel=not args.no_cross_channel,
        revin=not args.no_revin,
        channel_independent=args.channel_independent,
    ).to(args.device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{args.dataset}: {num_signals} signals, train/val/test windows = "
          f"{train.num_windows}/{val.num_windows}/{test.num_windows}, params = {num_params:,}")

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
    schedulers = []
    for optimizer in optimizers:
        if args.scheduler == "onecycle":
            max_lr = [group["lr"] for group in optimizer.param_groups]
            schedulers.append(torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.1
            ))
        else:
            schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps))
    if len(optimizers) > 1:
        print(f"muon: {sum(p.numel() for p in optimizers[1].param_groups[0]['params']):,} parameters")
    # bf16 autocast on GPUs that support it; no GradScaler needed
    use_bf16 = args.device == "cuda" and torch.cuda.is_bf16_supported()

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16)

    best_val, best_state, bad_epochs, history = math.inf, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        total = torch.zeros((), device=args.device)
        for step, (x, y, t_in, t_out) in enumerate(train.batches(args.batch_size, shuffle=True, drop_last=True)):
            if step >= steps_per_epoch:
                break
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            with autocast():
                pred = forward(x, t_in, t_out)
                task_loss = torch.nn.functional.mse_loss(pred.float(), y)
                loss = task_loss + args.gate_lambda * model.penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for optimizer, scheduler in zip(optimizers, schedulers):
                optimizer.step()
                scheduler.step()
            total += task_loss.detach()  # summed on the device to avoid a host sync every step
        train_mse = total.item() / steps_per_epoch
        val_mse, val_mae = evaluate(forward, model, val, args.batch_size, autocast, args.val_windows, args.max_steps)
        open_gates = int((model.tokenizer.gate.gate_values() > 0).sum())
        history.append(dict(epoch=epoch, train_mse=train_mse, val_mse=val_mse, val_mae=val_mae))
        print(f"epoch {epoch:3d}  train {train_mse:.4f}  val mse {val_mse:.4f} mae {val_mae:.4f}  "
              f"open gates {open_gates}/{model.tokenizer.gate.log_alpha.numel()}  {time.time() - start:.0f}s")
        if val_mse < best_val:
            best_val, bad_epochs = val_mse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    # Final scores on the complete validation and test splits
    full_val_mse, full_val_mae = evaluate(forward, model, val, args.batch_size, autocast, 0, args.max_steps)
    test_mse, test_mae = evaluate(forward, model, test, args.batch_size, autocast, 0, args.max_steps)
    print(f"val   mse {full_val_mse:.4f}  mae {full_val_mae:.4f}  (full split)")
    print(f"test  mse {test_mse:.4f}  mae {test_mae:.4f}")

    tag = f"_{args.tag}" if args.tag else ""
    out = args.output or RUNS_DIR / f"{args.dataset}_{args.pred_len}{tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    out.write_text(json.dumps(dict(
        args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        num_params=num_params, val_mse=full_val_mse, val_mae=full_val_mae, test_mse=test_mse, test_mae=test_mae,
        history=history, gate_values=model.tokenizer.gate.gate_values().tolist(),
    ), indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
