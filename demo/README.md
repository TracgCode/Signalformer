# Benchmark demo

Train and evaluate Signalformer on standard long-term forecasting benchmarks (ETTm1, ETTm2, Weather and Traffic) and on FI-2010, a limit order book classification benchmark.

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
| FI-2010 | 40 | every 10 order book events | Ntakaris et al., *Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods* (5 Nasdaq Nordic stocks, 10 days of June 2010); the DecPre files distributed with [DeepLOB](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books) | CC BY 4.0 |

Weather and Traffic are fetched from the Autoformer benchmark archive mirrored by [Nixtla datasetsforecast](https://nixtlaverse.nixtla.io/datasetsforecast/long_horizon.html). Check each provider's terms before redistributing the data or results derived from it.

## Protocol

`datasets.py` follows the protocol used by Informer, Autoformer and PatchTST, so results are comparable with published numbers:

- ETT: 12 / 4 / 4 months for train / validation / test. Weather and Traffic: 70% / 10% / 20%, in time order.
- Z-score normalization with training-split statistics.
- MSE and MAE on normalized test data, averaged over every test window (no `drop_last`).
- No calendar features by default. Enable them with `--time-features` (any of `hour,dow,doy`). With RevIN they did not help on these benchmarks, and day of year acts as a date ID the model memorizes on datasets spanning one or two years.
- Reversible instance normalization (RevIN) is on by default; disable with `--no-revin`.

FI-2010 follows Setup 2 of the DeepLOB paper: the first 7 days train, the last 3 days test, and the last 20% of every stock's training events validate. The inputs are the 40 order book features (ask / bid price and volume of the 10 best levels), or with `--lob-features 144` also the dataset's 104 handcrafted features, z-scored again with training statistics; no window crosses a stock boundary. The label of a window is the mid-price move (up / stationary / down) of its last event over the next `--pred-len` events (10, 20, 30, 50 or 100). The model averages its decoder features over the decoder steps, concatenates them over the signals and classifies with an FFN head; it trains with cross-entropy (with a linear learning-rate warmup over the first 1% of steps, `--warmup`), stops early on validation cross-entropy and reports macro F1 and accuracy. The test set is 71% stationary at horizon 10, so accuracy alone is misleading; DeepLOB reports accuracy 84.5% and F1 83.4% at k = 10 in Setup 2, with precision, recall and F1 weighted by class support; the benchmark summary prints macro F1, and EXPERIMENTS.md also gives weighted F1 for that comparison.

## Benchmark sweep

`run_benchmarks.py` runs every dataset, horizon, variant and seed as a separate `train.py` process and writes a comparison table with a naive repeat-last-value baseline to `demo/runs/benchmark/summary.md`. Scores are reported as mean ± std over seeds, since single-run differences of about 0.01 MSE are within seed noise.

- Up to `--jobs` runs (default 1) share the GPU; each writes its log to `demo/runs/benchmark/logs/`. Without NVIDIA MPS the processes are time-sliced, so the gain comes from filling each other's idle GPU time.
- Runs use `torch.compile` unless `--no-compile` is given.
- Finished runs are skipped on the next call (`--rerun` repeats them), so an interrupted sweep resumes where it stopped. A run in progress saves its full training state to `demo/runs/benchmark/ckpt/` every 500 steps (`--ckpt-every`) and at the end of every epoch, resumes from it when it is started again and deletes it once its result is saved; `--rerun` starts from scratch.
- Every run logs the training loss, gradient norm and learning rate every 200 steps (`--log-every`), and on FI-2010 the spread of the logits over the batch, flagged `STUCK` when the predictions no longer depend on the input.
- Variants that resolve to the same configuration for a dataset are trained once and the result is copied.
- Do not edit `train.py` or `datasets.py` while a sweep runs: each run starts a new process that loads the current files.

```bash
python demo/run_benchmarks.py --datasets ETTm1 ETTm2 Weather --pred-lens 96 192 336 720   # round 16: cycle normalized once, 3 seeds
python demo/run_benchmarks.py --seeds 2021                     # a single seed
python demo/run_benchmarks.py --pred-lens 96 192 336 720
python demo/run_benchmarks.py --datasets ETTm1 Weather --variants ci_raw tree_corr
python demo/run_benchmarks.py --dry-run                        # print the commands only
python demo/run_benchmarks.py -- --epochs 5                    # arguments after -- go to every run
```

The `train.py` defaults are channel independence (every signal runs through the model separately with shared weights; Traffic uses batch size 4) with the raw-input skip: the `ci_raw` configuration. Earlier rounds used other defaults, so each variant spells out its mode and keeps its original meaning. `SHARED` below means `--shared --no-raw-skip`: all signals in one token sequence, as in the first rounds. Results and conclusions of every round are in [EXPERIMENTS.md](EXPERIMENTS.md).

Round 16, the default sweep (ETTm1, ETTm2 and Weather at horizons 96, 192, 336 and 720):

| Variant | Change |
|---|---|
| `patch8_cycle_once` | `patch8_cycle` with `--cycle-norm once`: the window is normalized once, before the cycle is subtracted, and RevIN does not normalize the residual again (as in CycleNet) |
| `cross_par_l0_cycle_once` | `cross_par_l0_cycle` with `--cycle-norm once` |

Round 15 (ETTm1, ETTm2 and Weather at horizons 96, 192, 336 and 720):

| Variant | Change |
|---|---|
| `patch8_cycle` | `patch8` with `--cycle-len daily`: residual cycle forecasting with a learnable cycle of one day per signal (96 steps on ETTm, 144 on Weather), indexed by the absolute time step; the model forecasts the normalized window minus the cycle, and the cycle is added back |
| `cross_par_l0_cycle` | `cross_par_l0` with `--cycle-len daily` |
| `cross_par_l0_drop` | `cross_par_l0` with dropout 0.3 instead of 0.2 |

Round 14: the variants (`patch8`, `cross_gate_l0_fast`, `cross_par_l0`) on ETTm1, ETTm2 and Weather at horizons 192, 336 and 720; no new variants.

Round 13 (ETTm1, ETTm2 and Weather at horizon 96, FI-2010 with the 40 order book features at horizons 10, 50 and 100):

| Variant | Change from `cross_patch8` |
|---|---|
| `cross_par` | `--cross-parallel`: the cross-attentions to all signals run in parallel with the same queries, each source signal with its own softmax, and are merged by one concatenation and a C * D -> D projection, like the heads of multi-head attention, then the FFN |
| `cross_par_l0` | `cross_par` with the L0 edge masks of `cross_gate_l0_fast` (masks at `--gate-lr`) |

Round 12 (FI-2010 only, at horizons 10, 50 and 100; all three start from `no_ma_f144`):

| Variant | Change from `no_ma_f144` |
|---|---|
| `f144_e20` | 20 epochs instead of 10 (early stopping still applies) |
| `f144_d64` | d_model 64 instead of 32 |
| `f144_e2d2` | two encoder and two decoder layers |

Round 11 (FI-2010 only, at horizons 10, 50 and 100; all four start from `patch8_no_ma`):

| Variant | Change from `patch8_no_ma` |
|---|---|
| `no_ma_lr_low` | learning rate 3e-4 instead of 1e-3, with 20 epochs instead of 10 (TLOB and MLPLOB train at 1e-4 and 3e-4) |
| `no_ma_d64` | d_model 64 instead of 32 |
| `no_ma_e2d2` | two encoder and two decoder layers |
| `no_ma_f144` | `--lob-features 144`: the 104 handcrafted features of FI-2010 as well, as TLOB and MLPLOB read them |

Round 10 (FI-2010 at horizons 10, 50 and 100, `--lob-horizons`):

| Variant | Change from the defaults |
|---|---|
| `patch8`, `cross_gate_l0_fast` | as in round 9, now at horizons 50 and 100 |
| `patch8_no_ma` | no moving-average bands (`--front-end none`, rank 16): at horizon 10 every band gate closed |
| `patch8_seq336` | input of 336 events; the strongest published models read 128 (TLOB) and 384 (MLPLOB) |

Round 9 (FI-2010 at horizon 10, and the L0 edge masks again on ETT and Weather):

| Variant | Change from the defaults |
|---|---|
| `ci_raw`, `patch8`, `cross_patch8`, `cross_patch8_gate` | as before, now also on FI-2010 |
| `shared_patch8` | FI-2010 only: all 40 order book features in one token sequence, patching by 8, no raw skip |
| `cross_gate_l0_fast` | `cross_gate_l0` with the edge masks at `--gate-lr` (1e-2) instead of the model learning rate (ETT, Weather, FI-2010) |

Round 8:

| Variant | Change from the defaults |
|---|---|
| `patch8_seq512` | `patch8` with input length 512 |
| `patch8_d64`, `patch8_e2d2` | `patch8` with d_model 64, or with two encoder and two decoder layers (ETT and Weather only) |
| `cross_gate_seq336` | `cross_patch8_gate` with input length 336 (ETT and Weather only, as all cross variants) |
| `cross_gate_raw_xattn` | `cross_patch8_gate` with the raw cross-attention instead of the raw skip |
| `cross_gate_l1_low`, `cross_gate_l1_high` | `cross_patch8_gate` with `--edge-lambda` 1e-4 or 1e-2 instead of 1e-3 |
| `cross_gate_l0` | `cross_patch8_gate` with a Hard Concrete mask on every cross-signal edge and an L0 penalty on the fraction of open edges (`--edge-gates l0`, `--edge-lambda` 1e-2) |
| `ci_raw_e20`, `patch4_e20` | `ci_raw` and `patch4` with 20 epochs instead of 10 (Traffic only) |
| `patch4_seq336` | `patch4` with input length 336 (Traffic only) |
| `ci_proj_nogate` | the Traffic collapse diagnostic: `ci_proj` without the gate penalty (Traffic only) |

Round 7:

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
