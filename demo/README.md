# Benchmark demo

Train and evaluate Signalformer on standard long-term forecasting benchmarks: ETTm1, ETTm2, Weather and Traffic.

## Data

The datasets are not part of this repository. They belong to their original authors and are distributed under their own terms, so they are downloaded locally into `demo/data/`, which git ignores.

```bash
python demo/download_data.py
```

| Dataset | Signals | Frequency | Source | Terms |
|---|---|---|---|---|
| ETTm1, ETTm2 | 7 | 15 min | [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset) | CC BY-ND 4.0 |
| Weather | 21 | 10 min | Max Planck Institute for Biogeochemistry, Jena weather station; benchmark file from [Autoformer](https://github.com/thuml/Autoformer) | See the original provider |
| Traffic | 862 | 1 hour | California Department of Transportation (PeMS); benchmark file from [Autoformer](https://github.com/thuml/Autoformer) | See the original provider |

Weather and Traffic are fetched from the Autoformer benchmark archive mirrored by [Nixtla datasetsforecast](https://nixtlaverse.nixtla.io/datasetsforecast/long_horizon.html). Check each provider's terms before redistributing the data or results derived from it.

## Protocol

`datasets.py` follows the protocol used by Informer, Autoformer and PatchTST, so results are comparable with published numbers:

- ETT: 12 / 4 / 4 months for train / validation / test. Weather and Traffic: 70% / 10% / 20%, in time order.
- Z-score normalization with training-split statistics.
- MSE and MAE on normalized test data, averaged over every test window (no `drop_last`).
- No calendar features by default. Enable them with `--time-features` (any of `hour,dow,doy`). With RevIN they did not help on these benchmarks, and day of year acts as a date ID the model memorizes on datasets spanning one or two years.
- Reversible instance normalization (RevIN) is on by default; disable with `--no-revin`.

## Benchmark sweep

`run_benchmarks.py` runs every dataset, horizon, variant and seed as a separate `train.py` process and writes a comparison table with a naive repeat-last-value baseline to `demo/runs/benchmark/summary.md`. Scores are reported as mean ± std over seeds, since single-run differences of about 0.01 MSE are within seed noise.

- Up to `--jobs` runs (default 2) share the GPU; each writes its log to `demo/runs/benchmark/logs/`. Without NVIDIA MPS the processes are time-sliced, so the gain comes from filling each other's idle GPU time.
- Runs use `torch.compile` unless `--no-compile` is given.
- Finished runs are skipped on the next call (`--rerun` repeats them), so an interrupted sweep resumes where it stopped.
- Variants that resolve to the same configuration for a dataset are trained once and the result is copied.
- Do not edit `train.py` or `datasets.py` while a sweep runs: each run starts a new process that loads the current files.

```bash
python demo/run_benchmarks.py                                  # round 7: 4 datasets x horizon 96 x 10 variants x 3 seeds
python demo/run_benchmarks.py --seeds 2021                     # a single seed
python demo/run_benchmarks.py --pred-lens 96 192 336 720
python demo/run_benchmarks.py --datasets ETTm1 Weather --variants ci_raw tree_corr
python demo/run_benchmarks.py --dry-run                        # print the commands only
python demo/run_benchmarks.py -- --epochs 5                    # arguments after -- go to every run
```

The `train.py` defaults are channel independence (every signal runs through the model separately with shared weights; Traffic uses batch size 4) with the raw-input skip: the `ci_raw` configuration. Earlier rounds used other defaults, so each variant spells out its mode and keeps its original meaning. `SHARED` below means `--shared --no-raw-skip`: all signals in one token sequence, as in the first rounds. Results and conclusions of every round are in [EXPERIMENTS.md](EXPERIMENTS.md).

Round 7, the default sweep (`ci_raw`, `patch8` and `seq336` are round 6 references and are not retrained):

| Variant | Change from the defaults |
|---|---|
| `ci_raw` | none (reference) |
| `patch8` | one token per 8 steps: kernel 16, stride 8, n_rank 8 (Traffic's defaults already patch by 8) |
| `seq336` | input length 336 |
| `raw_xattn` | no raw skip; instead every normalized input step is a memory token the decoder cross-attends to (`--raw-tokens`) |
| `patch8_raw_xattn`, `patch8_raw_both` | `patch8` with the raw cross-attention instead of, or besides, the raw skip |
| `patch16` | one token per 16 steps: kernel 32, stride 16, n_rank 16 |
| `patch8_no_ma` | `patch8` without the moving-average decomposition and L0 gate, n_rank 16 |
| `patch8_seq336` | `patch8` with input length 336 |
| `patch8_drop` | `patch8` with dropout 0.3 |
| `cross`, `cross_raw_xattn`, `cross_patch8`, `cross_patch8_raw_xattn` | Cross Transformer (`--architecture cross`): channel-independent tokenizer and encoder, and a decoder in which every signal cross-attends to every signal's memory in signal order; on the defaults, `raw_xattn`, `patch8` and `patch8_raw_xattn` settings (ETT and Weather only) |
| `cross_patch8_gate` | `cross_patch8` with a learnable gate per (receiving, sending) signal pair, starting as the identity, with an L1 penalty on the cross-signal edges |

Frequency tokens (`--freq-tokens`) remain available but are left out of the default sweep.

Round 6:

| Variant | Change from the defaults |
|---|---|
| `no_ma_r4` | no moving-average decomposition and no L0 gate; n_rank 4 (Traffic 16) |
| `no_ma_long` | as `no_ma_r4` with kernel 48 (Traffic: n_rank 32) |
| `no_ma_mk` | as `no_ma_r4` with kernels 4, 16, 48 (Traffic: 8, 16, 48, n_rank 8) |
| `patch8`, `patch4`, `patch1` | one token per 8 / 4 / 1 steps (kernel 16 / 8 / 3, n_rank 8 / 4 / 1); shared results where equal to the defaults |
| `freq` | rFFT bins of the input window as an extra memory for the decoder |
| `seq336`, `freq_seq336` | input length 336, without and with `freq` |
| `tree_index`, `tree_corr`, `tree_random` | tree-merging architecture (`tree_model.py`) with a fixed pairing: adjacent indices, greedy by correlation of first differences on the training split, random |
| `tree_soft`, `tree_sampled` | tree with learnable pairing: soft mixtures with a negative-variance penalty, or sampled matchings trained with REINFORCE (ETT and Weather only) |

Earlier variants:

| Variant | Configuration |
|---|---|
| `base` | `SHARED` |
| `muon` | `SHARED`, Muon for the matrices inside attention / FFN blocks, AdamW for the rest |
| `auto_dmodel`, `half_dmodel` | `SHARED`, `d_model` = 2 × signals clamped to [32, 256], or signals / 2 (at least 32) |
| `dropout_high`, `lr_low`, `combined`, `muon_lr_low` | `SHARED`, dropout 0.3, learning rate 1e-4 for up to 20 epochs, and combinations |
| `no_revin`, `with_time`, `no_gate_penalty`, `large`, `full_val` | `SHARED`, RevIN off, hour + day-of-week features, gate penalty 0, a larger model, early stopping on the full validation split |
| `conv_bank`, `multi_kernel` | `SHARED`, learnable causal conv bank instead of the MA split, tokenizer kernels 3, 7, 15 |
| `raw` | `--shared` with the raw-input skip |
| `ci`, `ci_conv_bank`, `ci_conv_bank_mk` | channel independence without the raw skip, plus conv bank and multi-kernel |
| `ci_conv_bank_raw` | the defaults with the conv bank |
| `ci_proj`, `ci_proj_norm`, `ci_proj_nogate` | channel independence over signals / 2 learned latents (no raw skip), plus per-latent normalization or no gate penalty (Traffic: batch size 8) |

## Training a single run

A GPU is strongly recommended. CPU runs are only practical as smoke tests with `--max-steps`.

```bash
# Default setting: input 96, horizon 96
python demo/train.py --dataset ETTm1 --pred-len 96
python demo/train.py --dataset ETTm2 --pred-len 96
python demo/train.py --dataset Weather --pred-len 96

# Traffic has 862 signals; patching keeps the token sequence short
python demo/train.py --dataset Traffic --pred-len 96 --kernel-size 16 --conv-stride 8 --n-rank 8

# All four standard horizons
for h in 96 192 336 720; do python demo/train.py --dataset ETTm1 --pred-len $h; done

# Smoke test on CPU
python demo/train.py --dataset ETTm1 --epochs 1 --max-steps 2
```

`python demo/train.py --help` lists every option. Each run keeps every split on the GPU and slices batches there, without a DataLoader. It prints per-epoch validation scores on a fixed subset of `--val-windows` windows (default 2048) and the number of open gates, keeps the best checkpoint in memory, then reports MSE / MAE on the full validation and test splits. Arguments, scores, training history and final gate values are saved to `demo/runs/` (also ignored by git). Parsed CSVs are cached as `.npz` next to the data.

`--d-model` must be a multiple of `num_heads × 8` so every head stays on the FlashAttention kernel; `--d-model auto` rounds accordingly. Training uses bf16 autocast on GPUs that support it.

## Notes

- **Moving-average windows.** The default `--ma-windows 4,16,48` is sized for `--seq-len 96`. The first `max(windows) − 1` steps of each input use incomplete windows, so raise the input length (e.g. `--seq-len 336`) before using longer windows.
- **Compare against baselines.** Published DLinear and PatchTST results on the same split are the reference points. A model that does not beat them is not yet useful.
- **Ablations.** Useful comparisons are `--conv-stride 1` vs patching, `--no-cross-channel`, different `--n-rank`, and `--gate-lambda 0` (gate without sparsity pressure).
