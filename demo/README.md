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
python demo/run_benchmarks.py                                  # 4 datasets x horizon 96 x 3 variants x 3 seeds
python demo/run_benchmarks.py --seeds 2021                     # a single seed
python demo/run_benchmarks.py --pred-lens 96 192 336 720
python demo/run_benchmarks.py --datasets ETTm1 Weather --variants base no_revin
python demo/run_benchmarks.py --dry-run                        # print the commands only
python demo/run_benchmarks.py -- --epochs 5                    # arguments after -- go to every run
```

| Variant | Change from the defaults | Default sweep |
|---|---|---|
| `base` | none: small model, RevIN, no calendar features, gate penalty 1e-2, AdamW | yes |
| `muon` | Muon for the matrices inside attention / FFN blocks, AdamW for the rest | yes |
| `ci` | channel independence: every signal runs through the model separately (Traffic: batch size 4) | yes |
| `auto_dmodel` | `d_model` = 2 × signals, clamped to [32, 256], heads a multiple of 8 wide | |
| `dropout_high` | dropout 0.3 | |
| `lr_low` | learning rate 1e-4, up to 20 epochs | |
| `combined` | `auto_dmodel` + `dropout_high` + `lr_low` | |
| `muon_lr_low` | `muon` + `lr_low` | |
| `no_revin` | RevIN off | |
| `with_time` | hour and day-of-week features | |
| `no_gate_penalty` | gate penalty 0 | |
| `large` | `d_model` 64 (Traffic 128), 2 encoder and 2 decoder layers, dropout 0.1 | |

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
