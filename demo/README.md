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
python demo/run_benchmarks.py --datasets ETTm1 ETTm2 Weather --pred-lens 96 192 336 720   # round 23: rotating tokenizer kernel and PAMNet's carriers, 3 seeds
python demo/run_benchmarks.py --seeds 2021                     # a single seed
python demo/run_benchmarks.py --pred-lens 96 192 336 720
python demo/run_benchmarks.py --datasets ETTm1 Weather --variants ci_raw tree_corr
python demo/run_benchmarks.py --dry-run                        # print the commands only
python demo/run_benchmarks.py -- --epochs 5                    # arguments after -- go to every run
```

The `train.py` defaults are channel independence (every signal runs through the model separately with shared weights; Traffic uses batch size 4) with the raw-input skip: the `ci_raw` configuration. Earlier rounds used other defaults, so each variant spells out its mode and keeps its original meaning. `SHARED` below means `--shared --no-raw-skip`: all signals in one token sequence, as in the first rounds. Results and conclusions of every round are in [EXPERIMENTS.md](EXPERIMENTS.md).

Round 23, the default sweep (each dataset's best model and recipe at horizons 96, 192, 336 and 720): `<best>_rotconv`, `_rotconv_half` and `_rotconv_sig` turn the tokenizer kernel with the time of day (`--rot-conv`, `--rot-offsets`); `<best>_pamod` and `_pamod_table` add PAMNet's phase and amplitude carriers to the encoder memory (`--carrier`); `patch8_cycle_once_fourier_k3_d64` is the Weather capacity control. Round 22: `<best>_precipe_lr0.001` and `_precipe_lr0.002`. Round 21 (run with `--variants`):

| Variant | Change |
|---|---|
| `<best>_hmae` | PAMNet's hybrid MAE loss in time and frequency, with PAMNet's alpha for the dataset (0.05 / 0.35 / 0.2) |
| `<best>_ptrain` | PAMNet's schedule: Adam without weight decay, batch 256, up to 30 epochs with patience 5 on the full validation split, 'type3' decay, lr 0.005 |
| `<best>_precipe` | both |

`<best>` is `cross_par_l0_cycle_pedge_fourier_k3` on ETTm1, `patch8_cycle_once_fourier_full` on ETTm2 and `patch8_cycle_once_fourier_k3` on Weather.

Reimplemented published models (`baselines.py`, trained in our pipeline with their own published settings and sized like our models; see EXPERIMENTS.md):

| Variant | Model |
|---|---|
| `cyclenet_linear` | CycleNet/Linear, as the official code |
| `cyclenet_mlp128` | CycleNet/MLP with 128 hidden units (the paper: 512) |
| `pamnet_d32_a0.05` / `_a0.35` / `_a0.2` | PAMNet with d = 32 (the paper: 512) and dropout 0.1, lr 0.005, the frequency-loss weight chosen on validation for ETTm1 / ETTm2 / Weather |

```bash
python demo/run_benchmarks.py --datasets ETTm1 ETTm2 Weather --pred-lens 96 192 336 720 --variants cyclenet_linear cyclenet_mlp128 pamnet_d32_a0.05 pamnet_d32_a0.35 pamnet_d32_a0.2
```

Round 20, the default sweep (ETTm1, ETTm2 and Weather at horizons 96, 192, 336 and 720):

| Variant | Change |
|---|---|
| `cross_par_l0_cycle_pedge_fourier_k3` | the best ETTm1 model of round 19, now also on ETTm2 and Weather |
| `cross_par_l0_cycle_pedge2_fourier_k3` | the same with 2 harmonics for the phase-dependent edges instead of 4; ETTm1 only |
| `cross_par_l0_cycle_pedge8_fourier_k3` | the same with 8 harmonics; ETTm1 only |

Round 19 (ETTm1, ETTm2 and Weather at horizons 96, 192, 336 and 720):

| Variant | Change |
|---|---|
| `cross_par_l0_cycle_fourier_full` | `cross_par_l0_cycle` with `--cycle-harmonics -1`: the cycle in the frequency domain with every harmonic |
| `cross_par_l0_cycle_fourier_k3` | the same with 3 harmonics |
| `cross_par_l0_cycle_pedge_fourier_k3` | `cross_par_l0_cycle_pedge_fourier` with the 3-harmonic cycle; ETTm1 only |
| `cross_fused_own_mlp_cycle` | `cross_fused_mlp_cycle` with `--fused-own`: each signal's own encoded tokens are added to the fused memory |

Round 18 (ETTm1, ETTm2 and Weather at horizons 96, 192, 336 and 720):

| Variant | Change |
|---|---|
| `patch8_cycle_once_fourier_full` | `patch8_cycle_once` with `--cycle-harmonics -1`: the cycle is described by a constant and the cosine and sine coefficients of every harmonic of its period (frequency domain) and evaluated at each step; it spans the same cycles as the table |
| `patch8_cycle_once_fourier_k8` | the same with the first 8 harmonics only (a smooth cycle, 17 numbers per signal instead of 96 or 144) |
| `patch8_cycle_once_fourier_k3` | the same with the first 3 harmonics |
| `patch8_cycle_once_fourier_k8_weekly` | `_k8` with `--cycle-weekly 8`: 8 harmonics of a week (7 cycle lengths) added to the daily ones; ETTm1 and ETTm2 only |
| `cross_par_l0_cycle_pedge_fourier` | `cross_par_l0_cycle_pedge` with `--phase-harmonics 4`: the phase offsets of every edge are 4 harmonics of the cycle instead of one offset per phase (392 parameters on ETTm, 3,528 on Weather, instead of 4,704 and 63,504) |
| `cross_fused_mlp_cycle` | the Cross Transformer with `--cross-fused mlp` and the cycle: at each position the encoded tokens of all signals are concatenated and projected by one layer (C x d -> d, then GELU) to one memory that every signal's decoder cross-attends to, instead of the complete graph of edges |

Round 17 (ETTm1, ETTm2 and Weather at horizons 96, 192, 336 and 720):

| Variant | Change |
|---|---|
| `cross_par_l0_cycle_pedge` | `cross_par_l0_cycle` with `--phase-edges`: a learnable W x C x C table (W = the cycle length) adds an offset to the edge gates at the phase of every forecast step, so how much each signal reads every other one changes with the time of day (as TQNet's periodic queries); the L0 masks stay shared by all phases and the offsets start at zero |
| `cross_par_l0_cycle_e2d2` | `cross_par_l0_cycle` with two encoder and two decoder layers |
| `cross_par_l0_cycle_e2d2_shared` | `cross_par_l0_cycle_e2d2` with `--share-layers`: the two encoder layers share all weights, and so do the two decoder layers, so it has the parameters of `cross_par_l0_cycle` |

Round 16 (ETTm1, ETTm2 and Weather at horizons 96, 192, 336 and 720):

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

## Taiwan stocks: return forecasts with news (`stock_model.py`)

These scripts build the data, the features and the walk-forward experiments of the stock models in `stock_model.py` (see the main README). The data are not part of this repository: the market data, the downloaded announcements and news, and every score or embedding derived from them stay in the git-ignored `dataset/` at the repository root. The fetchers download only from public, keyless endpoints, or with the user's own free API keys read from a `keys/` directory outside the repository. They pause between requests and follow each site's robots.txt and terms. Check those terms before using or redistributing anything they download.

| Step | Scripts |
|---|---|
| Data | `fetch_overnight.py` (US session: TSM / UMC / ASX ADRs, SOX, Nasdaq futures), `fetch_news.py` (MOPS material announcements, cnyes news), `fetch_intl.py`, `fetch_peers.py` (US / Japanese / Korean peers: prices, EDGAR filings, Alpaca news), `fetch_local_news.py` (TDnet, KIND, OpenDART, Korea Herald), `fetch_politics.py` (Federal Register, Truth Social archive, PRC MFA, MAC, korea.kr, GDELT events), `fetch_profiles.py` (company profiles), `fetch_mass_news.py` (SETN, LTN, ETtoday, China Times by article id or sitemap), `fetch_taifex.py` (futures-exchange daily reports, put/call ratios, large traders) |
| Features | `stock_poly.py` (prices, multi-window polynomial fits), `stock_extra.py` (daily range), `stock_overnight.py`, `stock_recurrent.py` (`base_features`: the 52 base features) |
| Stage 1 | `stock_walkforward.py` (walk-forward ridge), `stock_news_gap.py` (news-predicted opening gap) |
| Comparison models | `stock_poly.py` (multi-window polynomial ridge), `stock_bilinear.py` (bilinear least squares over signals and lags), `stock_diffeq.py`, `stock_diffeq_pool.py`, `stock_diffeq_block.py`, `stock_diffeq_stride.py`, `stock_diffeq_order.py` (difference equations fitted per window, pooled, on block means) |
| News | `news_score.py` (LLM direction probabilities, bge-m3 embeddings), `news_struct.py` (typed fields with `TypedForm`), `news_summary.py` (comparison of local LLMs), `news_probe.py` (what one item's representations predict), `stock_news.py` (daily text features) |
| Stage 2 | `stock_news_attn.py`, `stock_news_attn2.py` (text-queried attention; shared splits and reports), `stock_itrans_news.py` (`StockNewsAttention`, all four modes) |
| Spillover and US relation | `stock_id_probe.py` (which stock embedding routes other stocks' news), `stock_identity_news.py` (`StockIdentityAttention`), `news_struct_us.py` (typed fields of US news), `stock_us_relation.py`, `stock_us_variants.py` (signed relation to US stocks, pre-open information) |
| Sign score | `stock_score_gap.py`, `stock_score_calendar.py`, `stock_score_loss.py` (score-aligned losses), `stock_score_set.py`, `stock_score_target.py`, `stock_score_itoken.py` (set model over the day's stocks; the current best model), `stock_score_news.py` (news tokens), `stock_score_taifex.py` (futures-exchange features) |

Every experiment is walk-forward: year Y (2022–2026) is forecast by models fitted on the years before Y, and every choice (ridge penalty, number of training steps) is made on Y − 1 from the fit on the years before it. Results and their discussion are in `EXPERIMENTS.md`.

```bash
python demo/stock_walkforward.py                    # the ridge
python demo/stock_news_gap.py                       # + the opening-gap block
python demo/stock_itrans_news.py sigval             # stage 2, one of stockq | newsq | sigval | itemq
HF_HOME=<models> python demo/news_struct.py --batch 12   # typed fields for every news item (Qwen3.5-4B)
```
