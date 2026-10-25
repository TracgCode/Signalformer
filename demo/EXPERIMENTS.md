# Experiment log

Signalformer on the long-term forecasting benchmarks, horizon 96, input length 96 unless noted. Scores are test MSE / MAE on z-scored data (lower is better). Multi-seed rows are mean ± std over seeds 2021, 2022 and 2023; single-seed rows are marked. Per-run JSON files and logs are in `demo/runs/benchmark/` (git-ignored); `summary.md` there holds the latest table.

Variant names refer to `VARIANTS` in `run_benchmarks.py`, which still reproduces every configuration below. The train.py defaults changed over time (see [Defaults](#current-defaults)), so each variant spells out its mode.

## Reference points

Published results for input 96, horizon 96, as collected in the iTransformer paper (ICLR 2024). They come from single runs, and several codebases drop the last incomplete test batch, so they are not strictly comparable with the full-test, three-seed means here.

| Model | ETTm1 | ETTm2 | Weather | Traffic |
|---|---|---|---|---|
| Naive (repeat last value, our split) | 1.214 | 0.266 | 0.259 | 2.714 |
| DLinear | 0.345 | 0.193 | 0.196 | 0.650 |
| TimesNet | 0.338 | 0.187 | 0.172 | 0.593 |
| PatchTST | 0.329 | 0.175 | 0.177 | 0.462 |
| iTransformer | 0.334 | 0.180 | 0.174 | 0.395 |

## Round 1: calendar features and basic ablations (single seed)

Shared mode (all signals in one token sequence), MA decomposition, d_model 32 (Traffic 64), RevIN.

| Variant | ETTm1 | ETTm2 | Weather | Traffic |
|---|---|---|---|---|
| `full` (hour + day-of-week features) | 0.388 | 0.206 | 0.186 | 0.642 |
| `no_time` (no calendar features) | **0.352** | 0.207 | 0.187 | **0.618** |
| `no_revin` | 0.656 | 0.838 | 0.195 | 0.678 |
| `no_gate_penalty` | 0.394 | 0.205 | 0.185 | 0.625 |
| `large` (d_model 64 / 128, 2+2 layers) | 0.417 | **0.196** | 0.186 | 0.649 |

- Calendar features did not help with RevIN; day-of-year had earlier acted as a date ID the model memorized. Calendar features are off by default since.
- RevIN is essential: without it ETT errors double to quadruple.
- A larger model overfits ETTm1 and does not help elsewhere.

## Round 2: width, regularization, learning rate, optimizer (single seed)

| Variant | ETTm1 | ETTm2 | Weather | Traffic |
|---|---|---|---|---|
| `base` | **0.352** | 0.198 | 0.186 | 0.624 |
| `auto_dmodel` (2 × signals, clamped to [32, 256]) | 0.352 | **0.197** | 0.182 | **0.614** |
| `dropout_high` (0.3) | 0.357 | 0.207 | 0.189 | 0.652 |
| `lr_low` (1e-4, 20 epochs) | 0.373 | 0.211 | 0.186 | 0.634 |
| `combined` | 0.388 | 0.216 | 0.188 | 0.666 |
| `muon` | 0.358 | 0.202 | **0.179** | 0.622 |
| `muon_lr_low` | 0.373 | 0.203 | 0.186 | 0.622 |

- No sign of overfitting: more dropout and a lower learning rate both hurt.
- Quadrupling Traffic's width (64 → 256) gained only 0.010: the bottleneck is not width.
- Muon helped Weather.

## Round 3: channel independence (3 seeds)

| Variant | ETTm1 | ETTm2 | Weather | Traffic |
|---|---|---|---|---|
| `base` | 0.374 ± 0.005 | 0.209 ± 0.006 | 0.184 ± 0.002 | 0.623 ± 0.002 |
| `muon` | 0.367 ± 0.005 | 0.208 ± 0.009 | 0.176 ± 0.003 | 0.617 ± 0.005 |
| `ci` (channel independence) | **0.343 ± 0.013** | **0.187 ± 0.002** | **0.176 ± 0.002** | **0.449 ± 0.001** |

- Channel independence (every signal through the model separately, shared weights) is the largest gain so far, above all on Traffic (0.623 → 0.449, better than PatchTST's 0.462).
- `base` is 0.022 worse on ETTm1 than the single-seed round 2 run. Early stopping on the full validation split (`full_val`, round 5) gave the same 0.374 ± 0.005, so the validation subset is not the cause; the round 2 run was a favorable seed.

## Round 4: front end, width, latent projection (3 seeds)

| Variant | ETTm1 | ETTm2 | Weather | Traffic |
|---|---|---|---|---|
| `conv_bank` (learnable causal conv bank instead of MA) | 0.367 ± 0.009 | 0.208 ± 0.007 | 0.187 ± 0.003 | 0.623 ± 0.004 |
| `ci_conv_bank` | 0.346 ± 0.011 | 0.190 ± 0.002 | 0.183 ± 0.002 | **0.443 ± 0.001** |
| `half_dmodel` (d_model = signals / 2, only Traffic changes: 448) | = base | = base | = base | 0.603 ± 0.004 |
| `ci_proj` (signals → signals / 2 latents, each channel-independent) | 0.348 ± 0.007 | 0.194 ± 0.011 | **0.174 ± 0.003** | 1.408 ± 0.000 (collapsed) |

- The learnable conv bank is on par with the fixed MA decomposition; only Traffic in ci mode gains (0.449 → 0.443).
- `ci_proj` gives the best Weather score but collapses on Traffic: all four L0 gates close in the first epoch, the gates' gradient is zero once clamped at 0, and the model outputs the window mean (train loss stuck at 1.011).

## Round 5: raw-input skip, multi-kernel tokenizer, latent normalization (3 seeds)

| Variant | ETTm1 | ETTm2 | Weather | Traffic |
|---|---|---|---|---|
| `raw` (shared + raw-input skip) | 0.368 ± 0.013 | 0.225 ± 0.037 | 0.184 ± 0.003 | 0.619 ± 0.004 |
| `ci_raw` | **0.334 ± 0.001** | 0.192 ± 0.002 | 0.178 ± 0.001 | 0.445 ± 0.002 |
| `ci_conv_bank_raw` | 0.338 ± 0.012 | 0.191 ± 0.003 | 0.185 ± 0.003 | **0.441 ± 0.003** |
| `multi_kernel` (tokenizer kernels 3, 7, 15) | 0.390 ± 0.019 | 0.199 ± 0.006 | 0.184 ± 0.000 | 0.621 ± 0.003 |
| `ci_conv_bank_mk` | 0.356 ± 0.005 | 0.190 ± 0.004 | 0.183 ± 0.001 | 0.442 ± 0.001 |
| `ci_proj_norm` (`ci_proj` + per-latent window normalization) | 0.348 ± 0.005 | 0.193 ± 0.011 | 0.178 ± 0.003 | 1.408 ± 0.000 (collapsed) |
| `full_val` (early stopping on the full validation split, ETTm1 only) | 0.374 ± 0.005 | | | |

- The raw-input skip (the normalized input of each token's patch projected straight into the token, bypassing decomposition, gate and conv) helps in ci mode: ETTm1 0.343 → 0.334 with a much smaller spread. It does not help in shared mode.
- Parallel kernel sizes in the tokenizer do not help: the decomposition and attention already cover longer scales.
- Normalizing the latents did not stop the Traffic collapse, so latent scale is not its cause.

## Best results so far

| | ETTm1 | ETTm2 | Weather | Traffic |
|---|---|---|---|---|
| Signalformer, input 96 | 0.323 (`patch8`) | 0.182 (`patch8`) | 0.174 (`ci_proj`) | 0.436 (`patch4`) |
| Signalformer, input 336 | 0.311 (`freq_seq336`) | 0.185 (`seq336`) | 0.154 (`freq_seq336`) | 0.373 (`seq336`) |
| Best reference above (input 96) | 0.329 (PatchTST) | 0.175 (PatchTST) | 0.172 (TimesNet) | 0.395 (iTransformer) |

At input 96 Signalformer now beats every reference on ETTm1, beats DLinear everywhere and PatchTST on Traffic, matches iTransformer and TimesNet on Weather, and trails PatchTST on ETTm2 by 0.007. The input-336 rows should be compared with results at that length (the PatchTST paper reports roughly 0.29 / 0.165 / 0.15 for ETTm1 / ETTm2 / Weather), not with the input-96 references.

## Current defaults

After round 5, `train.py` defaults to channel independence with the raw-input skip (the `ci_raw` configuration). `--shared` restores one shared token sequence and `--no-raw-skip` removes the skip. Traffic uses batch size 4 in channel-independent mode.

## Round 6: no MA, patching, frequency tokens, input 336, tree architecture (3 seeds)

All on the `ci_raw` defaults; `ci_raw` is the reference. ETT and Weather patch by 1 by default, so `patch1` equals `ci_raw` there; Traffic patches by 8 by default, so `patch8` equals `ci_raw` there, and its no-MA variants use more filters (rank 16 / 32 / 8, kernels 16 / 48 / 8,16,48) to cover the stride. The learnable tree pairings were not run on Traffic. The tree variants have about 59k parameters on ETT and 85k on Weather, against 37k for `ci_raw`.

| Variant | Change | ETTm1 | ETTm2 | Weather | Traffic |
|---|---|---|---|---|---|
| `ci_raw` | reference | 0.334 ± 0.001 | 0.192 ± 0.002 | 0.178 ± 0.001 | 0.445 ± 0.002 |
| `no_ma_r4` | no MA, rank 4, kernel 3 | 0.339 ± 0.013 | 0.194 ± 0.004 | 0.178 ± 0.002 | 0.447 ± 0.001 |
| `no_ma_long` | no MA, rank 4, kernel 48 | 0.338 ± 0.002 | 0.192 ± 0.002 | 0.179 ± 0.001 | 0.441 ± 0.001 |
| `no_ma_mk` | no MA, rank 4, kernels 4/16/48 | 0.337 ± 0.006 | 0.189 ± 0.003 | 0.183 ± 0.005 | 0.441 ± 0.001 |
| `patch8` | kernel 16, stride 8, rank 8 | **0.323 ± 0.007** | **0.182 ± 0.001** | **0.175 ± 0.001** | = `ci_raw` |
| `patch4` | kernel 8, stride 4, rank 4 | 0.331 ± 0.008 | 0.187 ± 0.001 | 0.179 ± 0.003 | **0.436 ± 0.000** |
| `patch1` | kernel 3, stride 1, rank 1 | = `ci_raw` | = `ci_raw` | = `ci_raw` | 0.441 ± 0.006 |
| `freq` | frequency tokens | 0.330 ± 0.012 | 0.190 ± 0.005 | 0.181 ± 0.003 | 0.450 ± 0.002 |
| `seq336` | input 336 | 0.317 ± 0.013 | 0.185 ± 0.003 | 0.158 ± 0.005 | 0.373 ± 0.003 |
| `freq_seq336` | input 336 + frequency tokens | 0.311 ± 0.019 | 0.190 ± 0.003 | 0.154 ± 0.002 | 0.373 ± 0.001 |
| `tree_index` | tree, adjacent pairs | 0.414 ± 0.006 | 0.197 ± 0.002 | 0.187 ± 0.001 | 0.587 ± 0.002 |
| `tree_corr` | tree, correlation pairs | 0.418 ± 0.006 | 0.200 ± 0.007 | 0.182 ± 0.005 | 0.586 ± 0.002 |
| `tree_random` | tree, random pairs | 0.404 ± 0.003 | 0.211 ± 0.008 | 0.187 ± 0.002 | 0.590 ± 0.001 |
| `tree_soft` | tree, learnable soft mixtures | 0.377 ± 0.008 | 0.206 ± 0.001 | 0.191 ± 0.002 | – |
| `tree_sampled` | tree, sampled matchings (REINFORCE) | 0.378 ± 0.006 | 0.198 ± 0.003 | 0.184 ± 0.006 | – |

Bold marks the best input-96 score per dataset. Every Traffic run used all 10 epochs without early stopping, so Traffic is probably undertrained.

- Patching works. `patch8` beats the reference on all three datasets (ETTm1 0.334 → 0.323, ETTm2 0.192 → 0.182) with about the same parameter count and an eighth of the tokens. `patch4` falls in between, so a longer stride is better here. It is the first input-96 result below PatchTST on ETTm1.
- The MA decomposition can go: without it, with rank 4 and any of the kernel choices, scores are within noise of the reference (one exception: kernels 4/16/48 on Weather, 0.183). It does not help either, so this simplifies the model rather than improving it.
- Frequency tokens do not help at input 96, where the FFT resolves little beyond what attention already sees. At input 336 they help on ETTm1 (0.317 → 0.311) and Weather (0.158 → 0.154) but hurt ETTm2 (0.185 → 0.190), and the ETTm1 spread (± 0.019) is larger than the gain.
- A longer input helps most: Weather 0.178 → 0.158, ETTm1 0.334 → 0.317.
- On Traffic the best patch is shorter: stride 4 (0.436) beats both stride 8 (the default, 0.445) and stride 1 (0.441). Traffic's hourly series have sharp daily peaks that finer tokens may keep, while the 10-minute and 15-minute small datasets prefer coarser tokens. Since Traffic was still improving at epoch 10, this comparison should be repeated with longer training.
- Input 336 is the largest gain of the round on Traffic: 0.445 → 0.373, well below every input-96 reference (iTransformer 0.395). Frequency tokens add nothing there.
- The tree architecture loses everywhere, badly on ETTm1 (0.40–0.42 with fixed pairings), while using more parameters. Correlation pairing is no better than random on ETTm1 and only slightly better on Weather, so which channels meet matters less than the fact that they are mixed at all. This matches round 3, where mixing channels hurt ETTm1 most. Learnable pairings recover part of the gap on ETTm1 (0.377), which suggests they learn to soften the mixing, but they do not help on ETTm2 or Weather. On Traffic the tree is far behind (0.59 against 0.445) whatever the pairing, with three times the parameters.

`ci_proj_nogate` (the Traffic collapse diagnostic) has not been run yet.

## Round 7 (prepared, not run yet)

All on the `ci_raw` defaults. `ci_raw`, `patch8` and `seq336` are references from round 6 and are not retrained. On Traffic, which already patches by 8, the `patch8_*` variants equal their counterparts without the prefix and reuse their results. Frequency tokens are kept as an option (`--freq-tokens`, and the `freq*` variants including `patch8_freq_seq336`) but are off in this and later sweeps: across round 6 they only helped at input 336 on ETTm1 and Weather, by less than the seed spread on ETTm1.

| Variant | Question |
|---|---|
| `raw_xattn` | The raw input as a decoder cross-attention memory (`--raw-tokens`: every normalized input step is one token, with RoPE at its input position) instead of the raw skip. |
| `patch8_raw_xattn`, `patch8_raw_both` | The same with patching by 8, where the raw memory keeps full time resolution that the 12 patch tokens lose; with and without the raw skip. |
| `patch16` | Longer patches: kernel 32, stride 16, rank 16. |
| `patch8_no_ma` | Patching without the MA decomposition (one band, rank 16). |
| `patch8_seq336` | Patching by 8 with input 336 (42 tokens). |
| `patch8_drop` | Dropout 0.3 instead of 0.2: `patch8` validation loss starts rising after 2 to 3 epochs. |
| `cross`, `cross_raw_xattn`, `cross_patch8`, `cross_patch8_raw_xattn` (ETT and Weather) | The Cross Transformer (`--architecture cross`): every signal is tokenized and encoded on its own with shared weights, as in `ci_raw`, but in the decoder every signal cross-attends to the encoder memory of every signal, itself included, one after another in dataset order (a complete graph). With `--raw-tokens` the raw-input cross-attention comes first. One cross-attention module per layer is shared by all signals, a learnable signal embedding marks the source memory and the signal being forecast, and each per-signal residual is scaled by 1/sqrt(C). Compared with `ci_raw`, `raw_xattn`, `patch8` and `patch8_raw_xattn`, which have the same tokenizer and encoder but no exchange between signals. Unlike the tree, each signal keeps its own decoder, so this tests cross-signal information without the bottleneck of merging. It costs C cross-attentions per layer, so Traffic is skipped. |
| `cross_patch8_gate` (ETT and Weather) | `cross_patch8` with edge gates (`--edge-gates`): the 1/sqrt(C) scale is replaced by a learnable C x C gate per decoder layer on the update each signal receives from each signal. Every per-signal softmax sums to 1, so without gates a signal cannot learn to ignore another one; the gates start as the identity (plain channel independence) and have an L1 penalty on the cross-signal edges (`--edge-lambda`, default 1e-3, no weight decay). The learned gates are saved in the result JSON. |

```bash
python demo/run_benchmarks.py                                       # round 7
python demo/run_benchmarks.py --datasets ETTm1 ETTm2 Weather        # the fast datasets first
python demo/run_benchmarks.py --datasets Traffic --variants ci_proj_nogate --seeds 2021
```
