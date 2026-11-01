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
| Signalformer, input 96 | 0.320 (`cross_gate_l0`) | 0.182 (`patch8`) | 0.158 (`cross_gate_raw_xattn`) | 0.436 (`patch4`) |
| Signalformer, input 336 or 512 | 0.311 (`freq_seq336`) | 0.183 (`patch8_seq512`) | 0.153 (`patch8_seq512`, `cross_gate_seq336`) | 0.373 (`seq336`) |
| Best reference above (input 96) | 0.329 (PatchTST) | 0.175 (PatchTST) | 0.172 (TimesNet) | 0.395 (iTransformer) |

At input 96 Signalformer now beats every reference on ETTm1 and Weather (the Cross Transformer with edge gates, 0.158 against TimesNet's 0.172), beats DLinear everywhere and PatchTST on Traffic, and trails PatchTST on ETTm2 by 0.007. `patch8_seq336` reaches 0.313 / 0.185 / 0.155 at input 336 without frequency tokens. The input-336 rows should be compared with results at that length (the PatchTST paper reports roughly 0.29 / 0.165 / 0.15 for ETTm1 / ETTm2 / Weather), not with the input-96 references.

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

`ci_proj_nogate` (the Traffic collapse diagnostic) has not been run yet; it is in round 8.

## Round 7: raw-input cross-attention, patching variants, Cross Transformer (3 seeds)

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

Results (the Cross Transformer variants were not run on Traffic; `patch4`, the best Traffic score at input 96, is 0.436 from round 6):

| Variant | ETTm1 | ETTm2 | Weather | Traffic |
|---|---|---|---|---|
| `ci_raw` (reference) | 0.334 ± 0.001 | 0.192 ± 0.002 | 0.178 ± 0.001 | 0.445 ± 0.002 |
| `patch8` (reference) | 0.323 ± 0.007 | **0.182 ± 0.001** | 0.175 ± 0.001 | = `ci_raw` |
| `seq336` (reference) | 0.317 ± 0.013 | 0.185 ± 0.003 | 0.158 ± 0.005 | 0.373 ± 0.003 |
| `raw_xattn` | 0.340 ± 0.007 | 0.187 ± 0.004 | 0.177 ± 0.005 | 0.452 ± 0.021 |
| `patch8_raw_xattn` | 0.334 ± 0.010 | **0.182 ± 0.001** | 0.175 ± 0.001 | = `raw_xattn` |
| `patch8_raw_both` | 0.327 ± 0.010 | 0.183 ± 0.000 | 0.177 ± 0.007 | **0.443 ± 0.004** |
| `patch16` | 0.331 ± 0.004 | 0.183 ± 0.001 | 0.176 ± 0.001 | 0.456 ± 0.001 |
| `patch8_no_ma` | 0.334 ± 0.018 | 0.183 ± 0.001 | 0.176 ± 0.002 | 0.447 ± 0.001 |
| `patch8_seq336` | 0.313 ± 0.009 | 0.185 ± 0.006 | 0.155 ± 0.001 | = `seq336` |
| `patch8_drop` | 0.335 ± 0.003 | 0.183 ± 0.001 | 0.178 ± 0.001 | 0.486 ± 0.014 |
| `cross` | 0.377 ± 0.005 | 0.202 ± 0.003 | 0.184 ± 0.001 | – |
| `cross_raw_xattn` | 0.343 ± 0.011 | 0.190 ± 0.005 | 0.163 ± 0.002 | – |
| `cross_patch8` | 0.362 ± 0.004 | 0.192 ± 0.001 | 0.185 ± 0.002 | – |
| `cross_patch8_raw_xattn` | 0.336 ± 0.007 | 0.190 ± 0.004 | 0.162 ± 0.002 | – |
| `cross_patch8_gate` | **0.322 ± 0.002** | 0.185 ± 0.003 | **0.160 ± 0.001** | – |

Bold marks the best input-96 score per dataset. Parameter counts on ETT: `patch8` 38,951, `cross_patch8` 39,175, `cross_patch8_raw_xattn` 43,239, `cross_patch8_gate` 39,224 (Weather: 39,623 and 40,064 for the last two cross variants).

- The Cross Transformer with edge gates is the best input-96 model on Weather by a wide margin: 0.175 → 0.160, close to `seq336` (0.158) from 96 input steps, with 270 more parameters than `patch8`. On ETTm1 it ties `patch8` (0.322 against 0.323) with a third of the spread; on ETTm2 it is slightly worse (0.185 against 0.182).
- Without gates, cross-signal attention hurts on ETT (`cross_patch8` ETTm1 0.362, against 0.323 for `patch8`) and on Weather unless raw tokens are added. With the 1/sqrt(C) scale every signal must take in every other signal, because each per-source softmax sums to 1, and on ETT that forced mixing costs as much as it did in round 3. The raw-input cross-attention softens it (ETTm1 0.362 → 0.336, Weather 0.185 → 0.162), probably because it gives each signal a path to its own values that the mixing cannot dilute. The gates solve the same problem directly, by letting a signal ignore the others, and do better without raw tokens.
- The learned gates make physical sense and are almost identical across seeds (the decoder has one layer). On ETTm1 the HUFL and MUFL loads take about 0.5 from each other, as do HULL and MULL, while OT keeps a self gate of 1.15 and takes almost nothing from the loads (|g| ≤ 0.1). On Weather the strongest edges are among the radiation signals (SWDR, PAR, max. PAR, 0.45 to 0.57) and from rain to raining (0.4); OT again takes little from the others (mean |g| 0.06). The L1 penalty at 1e-3 leaves the mean off-diagonal |g| at 0.10 to 0.12, so most weak edges are not pruned to zero.
- Raw-input cross-attention adds nothing in channel-independent mode: `raw_xattn` 0.340 against `ci_raw` 0.334 on ETTm1, and the patch-8 versions equal or trail `patch8`. The 12 patch tokens with the raw skip already carry what the model uses.
- `patch16` is slightly worse than `patch8` on all three datasets, so stride 8 stays.
- Removing the MA decomposition from `patch8` costs ETTm1 0.323 → 0.334 with a large spread (± 0.018) and does nothing on ETTm2 and Weather. With patching the decomposition is worth keeping.
- Patching and input 336 add up on ETTm1 (0.317 → 0.313) and Weather (0.158 → 0.155), matching `freq_seq336` without frequency tokens.
- Dropout 0.3 is slightly worse everywhere, so the early rise of the `patch8` validation loss is not fixed by more dropout.
- On Traffic, stride 16 is worse again (0.456), which completes the stride ranking there: 4 (0.436) < 1 (0.441) < 8 (0.445) < 16 (0.456). Dropping the MA (0.447) changes nothing. The raw cross-attention gives 0.441 and 0.439 on two seeds but 0.476 on the third, and with the raw skip kept (`patch8_raw_both`, 0.443) it is within noise of the reference. Dropout 0.3 hurts badly (0.486 ± 0.014, training loss 0.297 against about 0.266): every Traffic run still stops at the 10-epoch cap, so Traffic is underfitting rather than overfitting.
- The L0 gate closes one to four of the four front-end bands in many Traffic runs, but closed bands do not track the score (`patch8_raw_both` with all four closed scores 0.447), so they do not explain the `raw_xattn` outlier.

Next candidates: edge gates with input 336 and with raw tokens, a sweep of `--edge-lambda`, and a cheaper sparse version of the gated decoder for Traffic (862 signals).

## Round 8: longer input, width and depth, gated Cross Transformer variants, L0 edges (3 seeds)

`ci_raw`, `patch8`, `patch8_seq336`, `cross_patch8_gate` and, on Traffic, `patch4` are references from earlier rounds and are not retrained. Horizon 96 first; the longer horizons (192, 336, 720) follow for the best Signalformer and Cross Transformer settings.

| Variant | Question |
|---|---|
| `patch8_seq512` | Input 336 was the largest gain so far. Does 512 help further (64 tokens with patching)? |
| `patch8_d64`, `patch8_e2d2` (ETT and Weather) | Patching leaves an eighth of the tokens. Does a wider (d_model 64) or deeper (two encoder and two decoder layers) model help now? Round 2 found larger models overfit, but with one token per step. |
| `cross_gate_seq336` (ETT and Weather, as every cross variant) | Do cross-signal edges and a longer input add up? |
| `cross_gate_raw_xattn` | Without gates the raw cross-attention helped the Cross Transformer a lot. Does it still help with gates? |
| `cross_gate_l1_low`, `cross_gate_l1_high` | L1 weight 1e-4 and 1e-2 instead of 1e-3, which left the mean off-diagonal gate at 0.10 to 0.12. |
| `cross_gate_l0` | L0 instead of L1 (`--edge-gates l0`): every off-diagonal gate is multiplied by a Hard Concrete mask, and the penalty is the expected fraction of open edges (`--edge-lambda` 1e-2 by default), so edges can close exactly and the graph becomes sparse. The gates keep their sign and the self edges are never masked. The masks train at the model learning rate, not the faster `--gate-lr`, so the penalty cannot close them before the gates have learned anything. The JSON stores the effective weights, gate times mask. |
| `ci_raw_e20`, `patch4_e20` (Traffic) | Every Traffic run so far stopped at the 10-epoch cap. How much does 20 epochs help, and does stride 4 still beat stride 8? |
| `patch4_seq336` (Traffic) | The two best Traffic settings together. |
| `ci_proj_nogate` (Traffic) | The pending diagnostic of the round 5 collapse. |

Results on ETT and Weather (the Traffic variants have not been run yet):

| Variant | ETTm1 | ETTm2 | Weather |
|---|---|---|---|
| `patch8` (reference) | 0.323 ± 0.007 | **0.182 ± 0.001** | 0.175 ± 0.001 |
| `patch8_seq336` (reference) | **0.313 ± 0.009** | 0.185 ± 0.006 | 0.155 ± 0.001 |
| `cross_patch8_gate` (reference) | 0.322 ± 0.002 | 0.185 ± 0.003 | 0.160 ± 0.001 |
| `patch8_seq512` | 0.317 ± 0.003 | 0.183 ± 0.004 | **0.153 ± 0.005** |
| `patch8_d64` | 0.327 ± 0.009 | **0.182 ± 0.003** | 0.173 ± 0.001 |
| `patch8_e2d2` | 0.329 ± 0.010 | 0.183 ± 0.002 | 0.174 ± 0.001 |
| `cross_gate_seq336` | 0.317 ± 0.012 | 0.185 ± 0.007 | **0.153 ± 0.002** |
| `cross_gate_raw_xattn` | 0.333 ± 0.001 | 0.184 ± 0.002 | 0.158 ± 0.001 |
| `cross_gate_l1_low` | 0.322 ± 0.002 | 0.185 ± 0.003 | 0.160 ± 0.001 |
| `cross_gate_l1_high` | 0.321 ± 0.002 | 0.184 ± 0.003 | 0.159 ± 0.001 |
| `cross_gate_l0` | 0.320 ± 0.004 | 0.182 ± 0.001 | 0.160 ± 0.002 |

- Input 512 helps little over 336: Weather 0.155 → 0.153 (one seed 0.149, the PatchTST input-512 level), ETTm2 0.185 → 0.183, ETTm1 0.313 → 0.317. The gain from a longer input is mostly used up at 336.
- A wider or deeper model does not help on ETTm1 (0.327 and 0.329 against 0.323, with larger spreads) and is neutral on ETTm2. On Weather, with 21 signals, both gain a little (0.175 → 0.173 / 0.174). `patch8_e2d2` has 68,519 parameters against 38,951.
- The gated Cross Transformer and input 336 add up on Weather only: 0.153 ± 0.002, tied with `patch8_seq512` and better than either part alone (0.160 and 0.155). On ETTm1 it equals `patch8_seq336` within a larger spread, on ETTm2 nothing changes.
- With gates, the raw-input cross-attention hurts ETTm1 (0.322 → 0.333) but helps Weather (0.160 → 0.158), the best input-96 Weather score so far.
- The L1 weight hardly matters: 1e-4, 1e-3 and 1e-2 all give 0.321–0.322 on ETTm1, 0.184–0.185 on ETTm2 and 0.159–0.160 on Weather.
- The L0 edges did not close a single edge (42/42 open on ETT, 420/420 on Weather, in every run), and the learned weights match the L1 gates (HUFL/MUFL and HULL/MULL about 0.5, OT reading mostly itself). The masks start at log_alpha 2 and must fall to about -2.4 before an edge is exactly 0; at the model learning rate (1e-3), with a penalty gradient far smaller than the task gradient, ten epochs are not enough. So `cross_gate_l0` is in effect an L1 gate with a little mask noise, and its scores (ETTm1 0.320, ETTm2 0.182) should not be read as an L0 result. A real test needs the masks at the faster `--gate-lr` (1e-2) or a lower initial log_alpha.

```bash
python demo/run_benchmarks.py                                       # round 8
python demo/run_benchmarks.py --datasets ETTm1 ETTm2 Weather        # the fast datasets first
```

## Round 9: FI-2010 and the L0 edges again (3 seeds)

FI-2010 is the first financial dataset and the first classification task: from the 40 order book features (10 ask and 10 bid levels, price and volume) of the last 96 events, predict whether the mid-price moves up, stays or moves down over the next 10 events. Setup 2 of DeepLOB (7 days train, 3 days test); see the README for the protocol. It is a step toward M6-style tasks, which also need class probabilities instead of point forecasts. The classifier averages the decoder features over the decoder steps and concatenates them over the signals, so with channel independence the classification head is the only place where the order book levels meet; the Cross Transformer lets them exchange information before that.

The test set is 71% stationary (training 61%), so always predicting "stationary" gives 71% accuracy but a macro F1 of about 0.28; macro F1 is the main score. DeepLOB reports accuracy 84.47% and F1 83.40% at k = 10 with an input of 100 events (Setup 2, Table II). Its recall equals its accuracy, so its precision, recall and F1 are weighted by class support, not macro averages; compare it with our weighted F1. (F1 77.66% / accuracy 78.91% is its Setup 1 result.)

| Variant | Question |
|---|---|
| `ci_raw`, `patch8` | Channel independence on order book data, where the levels are strongly coupled. |
| `shared_patch8` | All 40 features in one token sequence, the counterpart that mixes the levels from the first layer. |
| `cross_patch8`, `cross_patch8_gate` | The Cross Transformer with 40 signals (1,560 cross-signal edges), without and with L1 gates. |
| `cross_gate_l0_fast` (also ETT and Weather) | L0 edge masks at `--gate-lr` (1e-2). In round 8 the masks trained at the model learning rate and never closed an edge. On a synthetic task where only one edge matters (signal 0 is signal 1 delayed), this setting closes the 11 other edges within about 1,500 steps and keeps that one at weight 0.70. On FI-2010, the open edges show which order book levels the model reads. |

The round 8 Traffic variants are deferred.

Results on FI-2010, k = 10, test set. Weighted F1 (by class support) is the metric DeepLOB reports; the published rows are copied from DeepLOB's Setup 2 table (input 100 events).

| Model | Weighted F1 | Macro F1 | Accuracy | Seconds per epoch |
|---|---|---|---|---|
| *Always "stationary"* | *0.587* | *0.276* | *0.708* | |
| *LSTM (Tsantekidis et al. 2017)* | *0.663* | | | |
| *B(TABL) (Tran et al. 2018)* | *0.692* | | *0.789* | |
| *C(TABL) (Tran et al. 2018)* | *0.776* | | *0.847* | |
| *DeepLOB (Zhang et al. 2019)* | *0.834* | | *0.845* | |
| `ci_raw` | 0.866 ± 0.004 | 0.801 ± 0.008 | 0.870 | ~25 |
| `patch8` | 0.873 ± 0.001 | 0.811 ± 0.001 | 0.876 | ~20 |
| `shared_patch8` | 0.708 ± 0.112 | 0.518 ± 0.219 | 0.763 | ~8 |
| `cross_patch8` | 0.870 ± 0.002 | 0.804 ± 0.005 | 0.877 | ~220 |
| `cross_patch8_gate` | **0.874 ± 0.000** | 0.811 ± 0.001 | **0.879** | ~215 |
| `cross_gate_l0_fast` | **0.874 ± 0.000** | **0.812 ± 0.001** | 0.878 | ~215 |

Stronger LOB models at k = 10. Published F1 scores mix two metrics: the papers up to 2022 report weighted F1 (their recall equals their accuracy), while the LOBCAST benchmark (Prata et al. 2024) and TLOB (Berti et al. 2025) retrain every model in one framework and report macro F1. The macro rows are the fair comparison for our macro F1.

| Model | Source | F1 type | k = 10 F1 | Input | Notes |
|---|---|---|---|---|---|
| *DeepLOB* | own paper, Setup 2 | weighted | *0.834* | 100 events, 40 features | |
| *Axial-LOB (2022)* | own paper, 7/3 days | weighted | *0.851* | 40 features | |
| *TransLOB (2020)* | own paper, 7/3 days | weighted | *0.887* | 100 events | LOBCAST could not reproduce it: 59.4 against a claimed 87.3 averaged over horizons, the largest gap in the benchmark |
| *DeepLOB* | TLOB table, retrained | macro | *0.711* | | 10 points below its own paper |
| *Axial-LOB* | TLOB table, retrained | macro | *0.732* | | |
| *BiN-CTABL (2021)* | TLOB table, retrained | macro | *0.811* | | the most robust model in LOBCAST (score 99.7) |
| *TLOB (2025)* | own paper | macro | *0.816* | 128 events, 144 features | 40 order book + 104 handcrafted features (about +1 F1) |
| *MLPLOB (2025)* | own paper | macro | ***0.816*** | 384 events, 144 features | best published k = 10 macro F1 found |
| `cross_gate_l0_fast` | this round | macro | 0.812 ± 0.001 | 96 events, 40 features | |
| `patch8` | this round | macro | 0.811 ± 0.001 | 96 events, 40 features | |

- Against the retrained macro F1 scores, our best runs (0.811 to 0.812) tie BiN-CTABL (0.811), TLOB (0.8155) and MLPLOB (0.8164) at k = 10, with only the 40 raw order book features and a shorter input. TLOB and MLPLOB pull ahead at the longer horizons (k = 50: 0.900 / 0.914, k = 100: 0.928 / 0.926), where BiN-CTABL also reaches 0.877 and 0.921; k = 50 and 100 are the next test.
- The weighted F1 of the older papers is not comparable with these, and it flatters the score on this data: our own runs give 0.874 weighted against 0.812 macro.
- T-KAN (2026) reports macro F1 on FI-2010 only for its own re-implemented baselines, under a different split; it is not a reference.

`cross_gate_l0_fast` on ETT and Weather (MSE, horizon 96):

| Variant | ETTm1 | ETTm2 | Weather |
|---|---|---|---|
| *PatchTST (2023)* | *0.329* | *0.175* | *0.177* |
| *iTransformer (2024)* | *0.334* | *0.180* | *0.174* |
| *TimeXer (2024)* | *0.318* | *0.171* | *0.157* |
| *CycleNet (2024)* | *0.319* | *0.163* | *0.158* |
| *TQNet (2025)* | *0.311* | *0.173* | *0.157* |
| *PAMNet (2026)* | *0.308* | *0.165* | *0.154* |
| `cross_patch8_gate` (reference) | 0.322 ± 0.002 | 0.185 ± 0.003 | 0.160 ± 0.001 |
| `cross_gate_l0` (round 8, masks at 1e-3) | 0.320 ± 0.004 | 0.182 ± 0.001 | 0.160 ± 0.002 |
| `cross_gate_l0_fast` | 0.321 ± 0.006 | 0.185 ± 0.004 | 0.159 ± 0.002 |

- On the same metric, every channel-independent or cross variant beats DeepLOB's Setup 2 result: weighted F1 0.866 to 0.874 against 0.834, accuracy 0.870 to 0.879 against 0.845. The comparison is not exact: the input is 96 events instead of 100, and our validation split (the last 20% of each stock's training days) and early stopping differ from the paper. Stronger published models are compared in the next table. The macro F1 comparisons below are among our own variants. The best scores are tied within the seed spread: `patch8` 0.811, `cross_patch8_gate` 0.811, `cross_gate_l0_fast` 0.812. Patching helps channel independence (0.801 → 0.811) and the gates help the Cross Transformer (0.804 → 0.811), but exchanging information between the order book levels before the classifier adds nothing over channel independence, at about ten times the cost per epoch.
- `shared_patch8` is unstable: 0.705, 0.276 and 0.572. The seed at 0.276 predicts only "stationary". Mixing all 40 features in one token from the first layer is the worst choice on this data.
- The fixed L0 masks now close edges. On FI-2010, 1,452 of 4,680 cross-signal edges stay open (31%; 449 to 525 per seed), and F1 equals the L1 gates. The level-1 features (best ask and bid, price and volume) are the source of 67 to 94 open edges against 32 to 55 for each deeper level; across feature types the volumes, bid volume above all, lose more edges than the prices. On ETT 31 of 126 edges stay open and on Weather 727 of 1,260; the MSE is unchanged against the L1 gates. Sparse edges cost nothing, but they do not improve the score either.
- The band gates close completely on FI-2010: all four L0 band gates of the MA decomposition end at 0 in `ci_raw` and in every Cross Transformer run, and `patch8` keeps only the finest band (x − MA1). In those runs the decomposition branch feeds zeros and the model reads the order book only through the raw-input skip. On order book features the moving-average bands carry no useful signal at this horizon; a FI-2010 setting should drop the decomposition rather than rely on the gates to remove it.
- Validation and test differ: validation F1 is about 0.80 and its cross-entropy 0.46, against 0.81 and 0.35 on test, because the last 20% of the training days and the test days have different class mixes (the test days are 71% stationary). Compare variants on the same split only.

```bash
python demo/run_benchmarks.py                                       # round 9
python demo/run_benchmarks.py --datasets FI2010 --lob-horizons 10 50 100   # more horizons later
```

## Round 10: FI-2010 at longer horizons (3 seeds)

Round 9 tied the best published macro F1 at k = 10, but TLOB, MLPLOB and BiN-CTABL reach 0.88 to 0.91 at k = 50 and 0.92 to 0.93 at k = 100. 30 runs, all on FI-2010:

| Variant | Horizons | Question |
|---|---|---|
| `patch8`, `cross_gate_l0_fast` | 50, 100 | Do the k = 10 results hold at the horizons where the published models are strongest? Does exchanging information between the levels help more there? |
| `patch8_no_ma` | 10, 50, 100 | Every band gate closed at k = 10. Is the model as good without the decomposition (one band, rank 16)? |
| `patch8_seq336` | 10, 50, 100 | A longer input of 336 events; TLOB reads 128 and MLPLOB 384. |

Results, test macro F1 (accuracy in brackets). Published rows are macro F1 from the TLOB paper (Berti et al. 2025), which retrains the older models in one framework; TLOB and MLPLOB read 144 features (40 order book + 104 handcrafted), we read the 40 order book features.

| Model | k = 10 | k = 50 | k = 100 |
|---|---|---|---|
| *Test class shares (up / stationary / down)* | *15 / 71 / 14%* | *27 / 47 / 25%* | *34 / 35 / 31%* |
| *DeepLOB (retrained)* | *0.711* | *0.754* | *0.776* |
| *Axial-LOB (retrained)* | *0.732* | *0.783* | *0.792* |
| *BiN-CTABL (retrained)* | *0.811* | *0.877* | *0.921* |
| *TLOB (2025)* | *0.816* | *0.900* | ***0.928*** |
| *MLPLOB (2025)* | ***0.816*** | ***0.914*** | *0.926* |
| `patch8` | 0.811 ± 0.001 (0.876) | **0.880 ± 0.000** (0.884) | **0.927 ± 0.001** (0.926) |
| `patch8_no_ma` | 0.811 ± 0.001 (0.878) | 0.879 ± 0.000 (0.884) | 0.925 ± 0.000 (0.925) |
| `patch8_seq336` | **0.812 ± 0.001** (0.879) | 0.631 ± 0.423 (0.678) | 0.923 ± 0.003 (0.923) |
| `cross_gate_l0_fast` | **0.812 ± 0.001** (0.878) | 0.879 ± 0.001 (0.883) | 0.926 ± 0.002 (0.926) |

- At k = 100 the model ties the best published scores: `patch8` 0.927 against TLOB 0.928 and MLPLOB 0.926, with three times fewer input features. At k = 10 it is 0.004 behind them, within what the handcrafted features add (about 1 F1 point in the TLOB paper). At k = 50 it is clearly behind: 0.880 against MLPLOB 0.914 and TLOB 0.900, only level with BiN-CTABL (0.877). k = 50 is the horizon to work on.
- Our four variants are tied at every horizon (within 0.002), so neither the Cross Transformer, nor the moving-average bands, nor a longer input helps on FI-2010. The edges between the levels and the decomposition can be dropped for this data; `patch8_no_ma` is the simplest setting and the most stable one (spread 0.0003 to 0.0007).
- The band gates depend on the horizon: at k = 10 and k = 50 nearly all of them close, at k = 100 `patch8` keeps the two finest bands in every seed and `cross_gate_l0_fast` keeps three. Slower bands only matter for the longer horizon, but `patch8_no_ma` matches `patch8` there too.
- One run failed to train: `patch8_seq336` at k = 50, seed 2021. Its training loss stayed at ln 3 = 1.0986 from the first epoch, it predicts "up" for every window (macro F1 0.142), and early stopping ended it after 5 epochs. The other two seeds give 0.876 and 0.875. Together with the `shared_patch8` seed that predicted only "stationary" in round 9, this is the second classification run that never leaves the uniform prediction. A stuck start is not caught by early stopping; the cause is not known yet (all four band gates are closed in that run, but they are in successful runs too).

```bash
python demo/run_benchmarks.py                                       # round 10
```

## Round 11: FI-2010, capacity and inputs (3 seeds)

All four variants start from `patch8_no_ma` (the simplest tied setting of round 10) and run at k = 10, 50 and 100; 36 runs. The target is k = 50, where MLPLOB (0.914) and TLOB (0.900) are ahead of our 0.880.

| Variant | Question |
|---|---|
| `no_ma_lr_low` | Learning rate 3e-4 and 20 epochs instead of 1e-3 and 10. In round 10 early stopping ended most runs, with the best validation loss at epochs 5 to 9, so more epochs alone would not help; TLOB and MLPLOB train at 1e-4 and 3e-4. |
| `no_ma_d64`, `no_ma_e2d2` | The FI-2010 preset is small (d_model 32, one encoder and one decoder layer, about 200k parameters, most of them in the classifier head). With 200k training windows, does a wider or deeper model help, unlike on ETT? |
| `no_ma_f144` | The 104 handcrafted features (spreads, mid-prices, differences, means, accumulated differences, derivatives, intensities) as 104 more signals. TLOB and MLPLOB read them; their paper reports about +1 F1. Nine of them are zero in every file and stay zero. |

Results, test macro F1 (accuracy in brackets), published rows as in round 10. The three `no_ma_f144` runs at k = 100 first ran out of GPU memory because another process held 26 GB of the card, and were rerun; `run_benchmarks.py` now runs one job at a time by default.

| Model | Params | k = 10 | k = 50 | k = 100 |
|---|---|---|---|---|
| *Test class shares (up / stationary / down)* | | *15 / 71 / 14%* | *27 / 47 / 25%* | *34 / 35 / 31%* |
| *DeepLOB (retrained)* | | *0.711* | *0.754* | *0.776* |
| *Axial-LOB (retrained)* | | *0.732* | *0.783* | *0.792* |
| *BiN-CTABL (retrained)* | | *0.811* | *0.877* | *0.921* |
| *TLOB (2025, 144 features)* | | *0.816* | *0.900* | *0.928* |
| *MLPLOB (2025, 144 features)* | | *0.816* | ***0.914*** | *0.926* |
| `patch8_no_ma` (round 10) | 200k | 0.811 ± 0.001 (0.878) | 0.879 ± 0.000 (0.884) | 0.925 ± 0.000 (0.925) |
| `no_ma_lr_low` | 200k | 0.811 ± 0.001 (0.877) | 0.877 ± 0.001 (0.882) | 0.924 ± 0.001 (0.924) |
| `no_ma_d64` | 792k | 0.811 ± 0.000 (0.877) | 0.878 ± 0.000 (0.883) | 0.926 ± 0.000 (0.926) |
| `no_ma_e2d2` | 229k | 0.811 ± 0.001 (0.880) | 0.879 ± 0.000 (0.883) | 0.925 ± 0.001 (0.925) |
| `no_ma_f144` | 626k | **0.827 ± 0.000** (0.889) | 0.912 ± 0.001 (0.916) | **0.932 ± 0.001** (0.932) |

- Capacity and learning rate change nothing: a lower learning rate, four times the parameters (`no_ma_d64`) or twice the depth (`no_ma_e2d2`) stay within 0.002 of `patch8_no_ma` at every horizon. With the 40 order book features the model is limited by its input, not by its size or its optimization.
- The 104 handcrafted features are the gap. `no_ma_f144` gains 0.016 at k = 10, 0.033 at k = 50 and 0.007 at k = 100. At k = 10 and k = 100 it is ahead of every published model (0.827 against 0.816, 0.932 against 0.928), at k = 50 it is level with MLPLOB (0.912 against 0.914) and ahead of TLOB (0.900). With the same 144 inputs as TLOB and MLPLOB the comparison is now like for like. The gain is larger than the "about +1 F1" the TLOB paper reports for these features.
- Test cross-entropy drops even more than F1 (k = 10: 0.26 against 0.35), so the probabilities improve beyond the arg max.
- At k = 10 all three `no_ma_f144` runs reached the 10-epoch limit; the validation loss had flattened (best at epochs 8 to 10), so a few more epochs might add a little.
- No run got stuck at ln 3 this round.

```bash
python demo/run_benchmarks.py                                       # round 11
```

## Round 12: FI-2010, training length and size with all 144 features (3 seeds)

Round 11 showed that the gap to TLOB and MLPLOB came from the inputs: with the 40 order book features, neither a lower learning rate, nor a wider or deeper model helped, while the 104 handcrafted features added 0.016 / 0.033 / 0.007 macro F1. With 144 signals the model has more to fit, so the round 11 capacity questions are asked again. All three variants start from `no_ma_f144` (round 11 gives its results) and run at k = 10, 50 and 100; 27 runs, one at a time.

| Variant | Question |
|---|---|
| `f144_e20` | Every k = 10 run of `no_ma_f144` reached its 10 epochs with the validation loss still close to its best. Do 20 epochs (the cosine schedule stretches with them) add anything? |
| `f144_d64`, `f144_e2d2` | Does a wider or deeper model help once the input carries more information? Most of all at k = 50, where MLPLOB is still 0.002 ahead. |

The Cross Transformer is left out: its cost grows with the square of the number of signals, and at 144 signals one epoch would take about 13 times the 215 s it took with 40.

First run (2026-10-28), not used for conclusions:
- Four of the six `f144_d64` and `f144_e2d2` runs at k = 50 got stuck at the uniform prediction (training loss ln 3 = 1.0986 from the first epoch, macro F1 0.13 to 0.14), and all six at k = 100 ran out of GPU memory because another process held 26 GB of the card.
- The completed runs: `f144_e20` 0.822 / 0.909 / 0.931 (k = 10 / 50 / 100), below `no_ma_f144`. It stops early after 10 or 11 epochs anyway, while the cosine schedule, stretched to 20 epochs, has not decayed by then. `f144_d64` 0.828 at k = 10; `f144_e2d2` 0.825.

The stuck runs, explained: the classifier head reads the concatenated features of all signals (144 x 64 = 9216 inputs with `f144_d64`). The first AdamW step moves every weight by about the full learning rate, so the pre-activations of the head's hidden layer shift by several units at once. Traced on CPU for `f144_d64`, k = 50, seed 2021:
- At step 1 the loss jumped from 1.12 to 17.3 and 42% of the hidden units were below zero for the whole batch. By step 70 all of them were.
- The logits then no longer depend on the input (spread over the batch 0.0003) and no gradient reaches the model.
- A wider head (more signals, larger d_model) makes this likelier, which is why `seq336`, `shared_patch8` and now `d64` / `e2d2` with 144 features were hit, and why forecasting runs never were.
- With a linear warmup over the first 300 steps no unit died and the logit spread grew normally.

Fix: `train.py --warmup` (linear warmup over a fraction of all steps, then the cosine decay), 0.01 by default on FI-2010 and 0 elsewhere, so forecasting runs are unchanged. `train.py` now also logs the training loss, gradient norm, learning rate and, on FI-2010, the logit spread every `--log-every` steps (and every step of the first 10), flagging `STUCK` when the spread falls below 1e-3. Round 12 is rerun in full with the warmup (`--rerun`).

Results of the rerun with the warmup, test macro F1 (accuracy in brackets). No run got stuck: the smallest logit spread logged in the 27 runs was 0.07, far above the `STUCK` threshold of 1e-3. `no_ma_f144` is from round 11, without the warmup.

| Model | Params | k = 10 | k = 50 | k = 100 |
|---|---|---|---|---|
| *Test class shares (up / stationary / down)* | | *15 / 71 / 14%* | *27 / 47 / 25%* | *34 / 35 / 31%* |
| *DeepLOB (retrained)* | | *0.711* | *0.754* | *0.776* |
| *BiN-CTABL (retrained)* | | *0.811* | *0.877* | *0.921* |
| *TLOB (2025, 144 features)* | | *0.816* | *0.900* | *0.928* |
| *MLPLOB (2025, 144 features)* | | *0.816* | ***0.914*** | *0.926* |
| `no_ma_f144` (round 11) | 626k | **0.827 ± 0.000** (0.889) | 0.912 ± 0.001 (0.916) | **0.932 ± 0.001** (0.932) |
| `f144_e20` | 626k | 0.823 ± 0.002 (0.887) | 0.911 ± 0.000 (0.915) | 0.931 ± 0.001 (0.931) |
| `f144_d64` | 2.5M | 0.825 ± 0.004 (0.889) | 0.912 ± 0.001 (0.916) | **0.932 ± 0.001** (0.932) |
| `f144_e2d2` | 655k | 0.826 ± 0.001 (0.889) | ***0.914 ± 0.001*** (0.918) | **0.932 ± 0.001** (0.932) |

- With the warmup every run trains: the 4 stuck runs of the first attempt are gone, and the step logs never came near the threshold.
- Neither training length nor model size helps with 144 features either. All four settings are within 0.004 of each other at every horizon, about the size of the seed spread. `f144_e2d2` is the only one level with MLPLOB at k = 50 (0.914), by 0.002 over `no_ma_f144`, which is within noise.
- `f144_e20` again stops early after 8 to 11 epochs; with the longer cosine schedule the learning rate is still high when it stops, and it ends slightly lower than the 10-epoch runs.
- Overall the model now matches or beats the published FI-2010 results at all three horizons with the same 144 inputs (k = 10: 0.827 vs 0.816, k = 50: 0.912 to 0.914 vs 0.914, k = 100: 0.932 vs 0.928), at 0.6M parameters. `no_ma_f144` stays the FI-2010 default: the smallest of the tied settings.

```bash
python demo/run_benchmarks.py --rerun                               # round 12, with the warmup
```

## Round 13: the parallel Cross Transformer (3 seeds)

The serial Cross Transformer reads the C signals one after another: for j = 1 .. C every signal's decoder cross-attends to signal j's memory and adds the result before it reads signal j + 1. The result depends on the order of the signals in the dataset, and the keys and values of every signal are projected C times. It tied channel independence on FI-2010 (rounds 9 and 10) at about ten times the cost.

`--cross-parallel` (`SignalParallelCrossBlock`) reads all signals at once, merged like the heads of multi-head attention:
- every receiving signal's decoder cross-attends to every signal's memory with the same queries (the block input after self-attention), each source signal with its own softmax;
- the C outputs are concatenated and merged by one C * D -> D projection, so each source signal has its own slice of the output projection, then the FFN;
- the edge gates and L0 masks scale each source's output before the concatenation, as in the serial block;
- keys and values are projected once per signal, and all C x C attentions run as one batched call. On CPU one block is 1.8 to 2.5 times faster than the serial one (C = 7 to 40); the projection adds C * D * D weights per layer (about 40k at C = 40, D = 32).

A comparison against an explicit loop over all (receiver, source) pairs agrees to 5e-7, with and without gates.

36 runs, 3 seeds:

| Variant | Datasets | Question |
|---|---|---|
| `cross_par` | ETTm1, ETTm2, Weather (96); FI-2010, 40 features (k = 10, 50, 100) | Does exchanging information between the signals help once it is read in parallel and merged by a learned projection? |
| `cross_par_l0` | same | The same with the L0 edges; against the serial `cross_gate_l0_fast` of rounds 9 and 10. |

On FI-2010 the new runs use the warmup of round 12; the serial runs of rounds 9 and 10 did not, but none of them got stuck. ETT and Weather are unaffected (no warmup there).

```bash
python demo/run_benchmarks.py                                       # round 13
```

Results, ETT and Weather at horizon 96, test MSE. Seconds per epoch are medians from the logs; they depend on how many runs shared the GPU, so they are only indicative.

| Model | Params (ETTm1) | ETTm1 | ETTm2 | Weather | s / epoch (Weather) |
|---|---|---|---|---|---|
| *Naive (repeat last value)* | | *1.214* | *0.266* | *0.259* | |
| *PatchTST* | | *0.329* | *0.175* | *0.177* | |
| *iTransformer* | | *0.334* | *0.180* | *0.174* | |
| `patch8` (round 7) | 39k | 0.323 ± 0.007 | 0.182 ± 0.001 | 0.175 ± 0.001 | 5 |
| `cross_patch8` (serial, round 7) | 39k | 0.362 ± 0.004 | 0.192 ± 0.001 | 0.185 ± 0.002 | 15 |
| `cross_gate_l0_fast` (serial, round 9) | 39k | **0.321 ± 0.006** | 0.185 ± 0.004 | 0.159 ± 0.001 | 16 |
| `cross_par` | 45k | 0.356 ± 0.007 | 0.206 ± 0.008 | 0.166 ± 0.004 | 11 |
| `cross_par_l0` | 45k | 0.325 ± 0.005 | **0.181 ± 0.003** | **0.159 ± 0.003** | 13 |

FI-2010, 40 features, test macro F1. "Edges kept" is the share of off-diagonal L0 edges still open at the end of training.

| Model | Params (k = 10) | k = 10 | k = 50 | k = 100 | s / epoch (k = 10) | Edges kept |
|---|---|---|---|---|---|---|
| *Test class shares (up / stationary / down)* | | *15 / 71 / 14%* | *27 / 47 / 25%* | *34 / 35 / 31%* | | |
| *DeepLOB (retrained)* | | *0.711* | *0.754* | *0.776* | | |
| *BiN-CTABL (retrained)* | | *0.811* | *0.877* | *0.921* | | |
| *TLOB (2025, 144 features)* | | *0.816* | *0.900* | *0.928* | | |
| *MLPLOB (2025, 144 features)* | | *0.816* | *0.914* | *0.926* | | |
| `patch8` (round 10) | 201k | 0.811 ± 0.001 | 0.880 ± 0.000 | 0.927 ± 0.001 | 19 | |
| `cross_gate_l0_fast` (serial, round 10) | 205k | **0.812 ± 0.001** | 0.879 ± 0.001 | 0.926 ± 0.002 | 221 | 31 / 56 / 59% |
| `cross_par` | 242k | 0.810 ± 0.001 | **0.880 ± 0.001** | 0.927 ± 0.001 | 62 | |
| `cross_par_l0` | 245k | **0.812 ± 0.001** | **0.880 ± 0.001** | **0.928 ± 0.000** | 65 | 3 / 9 / 12% |

- With the L0 edges the parallel block matches the serial one everywhere: the six differences are all within the seed spread. On FI-2010 it is about 3.4 times faster per epoch (65 s against 221 s at k = 10), and still faster on ETT and Weather. It costs about 15% more parameters, almost all in the C * D -> D output projection. The parallel block replaces the serial one from here on.
- Without edge gates, reading every signal hurts, in parallel as well as in series: `cross_par` is 0.033 worse than `patch8` on ETTm1 and 0.024 worse on ETTm2. Only with the L0 edges, which close most connections (67 to 74% on ETT), does it tie `patch8`.
- Weather is the one dataset where exchanging information between the signals clearly pays: 0.175 → 0.159, below every reference at input 96. Its 21 weather variables are strongly related, and the L0 edges keep 58% of the connections there, far more than on ETT.
- On FI-2010 with 40 features the parallel L0 block closes almost every edge (3% kept at k = 10, 9% at k = 50, 12% at k = 100, against 31 to 59% for the serial block) and the score does not move (0.928 at k = 100 is 0.001 above `patch8`, within noise). The serial block's open edges were not doing anything either. The order book levels need no cross-signal exchange beyond what channel independence gives; the gains on FI-2010 came from the 104 handcrafted features (rounds 11 and 12).
- The parallel FI-2010 runs use the warmup of round 12; none was flagged `STUCK`.

During the sweep the disk holding `demo/runs` filled up (from other data, not the checkpoints) and two FI-2010 k = 100 runs failed while saving a checkpoint. Resuming them then exposed a bug: `torch.load(map_location="cuda")` moved the saved RNG states to the GPU, and `torch.set_rng_state` only accepts CPU tensors. Fixed by moving them back to the CPU before restoring; the two runs resumed from their last checkpoints (epoch 6).


## Round 14: longer horizons on ETT and Weather (3 seeds)

Every forecasting result so far is at horizon 96, while published tables report horizons 96, 192, 336 and 720 and their average. This round fills in the longer horizons for channel independence with patching and for the two L0 Cross Transformers, so the comparison covers the full table.

| Variant | Question |
|---|---|
| `patch8` | Channel independence with patching: does its lead over PatchTST at 96 (ETTm1) hold at longer horizons? |
| `cross_gate_l0_fast`, `cross_par_l0` | Does the Weather gain from exchanging information between the signals (0.175 → 0.159 at 96) grow or shrink with the horizon, and do the serial and parallel blocks still tie? |

ETTm1, ETTm2 and Weather at horizons 192, 336 and 720 (the horizon-96 runs exist), input 96, 3 seeds: 81 runs.

Published results at input 96, test MSE. TimeXer from the full results table of its paper (NeurIPS 2024, Table 12); all others from the full results table of the PAMNet paper (arXiv 2605.02938, Table 7), whose DLinear, TimesNet, PatchTST and iTransformer rows match the iTransformer paper (ICLR 2024). All are single runs.

| Dataset | Model | 96 | 192 | 336 | 720 | Avg |
|---|---|---|---|---|---|---|
| ETTm1 | DLinear (2023) | 0.345 | 0.380 | 0.413 | 0.474 | 0.403 |
|  | TimesNet (2023) | 0.338 | 0.374 | 0.410 | 0.478 | 0.400 |
|  | PatchTST (2023) | 0.329 | 0.367 | 0.399 | 0.454 | 0.387 |
|  | iTransformer (2024) | 0.334 | 0.377 | 0.426 | 0.491 | 0.407 |
|  | TimeXer (2024) | 0.318 | 0.362 | 0.395 | 0.452 | 0.382 |
|  | CycleNet (2024) | 0.319 | 0.360 | 0.389 | 0.447 | 0.379 |
|  | TQNet (2025) | 0.311 | 0.356 | 0.390 | 0.452 | 0.377 |
|  | PAMNet (2026) | 0.308 | 0.353 | 0.375 | 0.425 | 0.365 |
| ETTm2 | DLinear (2023) | 0.193 | 0.284 | 0.369 | 0.554 | 0.350 |
|  | TimesNet (2023) | 0.187 | 0.249 | 0.321 | 0.408 | 0.291 |
|  | PatchTST (2023) | 0.175 | 0.241 | 0.305 | 0.402 | 0.281 |
|  | iTransformer (2024) | 0.180 | 0.250 | 0.311 | 0.412 | 0.288 |
|  | TimeXer (2024) | 0.171 | 0.237 | 0.296 | 0.392 | 0.274 |
|  | CycleNet (2024) | 0.163 | 0.229 | 0.284 | 0.389 | 0.266 |
|  | TQNet (2025) | 0.173 | 0.238 | 0.301 | 0.397 | 0.277 |
|  | PAMNet (2026) | 0.165 | 0.227 | 0.285 | 0.380 | 0.264 |
| Weather | DLinear (2023) | 0.196 | 0.237 | 0.283 | 0.345 | 0.265 |
|  | TimesNet (2023) | 0.172 | 0.219 | 0.280 | 0.365 | 0.259 |
|  | PatchTST (2023) | 0.177 | 0.225 | 0.278 | 0.354 | 0.259 |
|  | iTransformer (2024) | 0.174 | 0.221 | 0.278 | 0.358 | 0.258 |
|  | TimeXer (2024) | 0.157 | 0.204 | 0.261 | 0.340 | 0.241 |
|  | CycleNet (2024) | 0.158 | 0.207 | 0.262 | 0.344 | 0.243 |
|  | TQNet (2025) | 0.157 | 0.206 | 0.262 | 0.344 | 0.242 |
|  | PAMNet (2026) | 0.154 | 0.202 | 0.260 | 0.343 | 0.240 |

```bash
python demo/run_benchmarks.py --datasets ETTm1 ETTm2 Weather --pred-lens 96 192 336 720   # round 14
```

Results, test MSE (Avg is the mean over the four horizons). The published rows are the strongest of the table above plus PatchTST; the horizon-96 runs are from round 13.

| Dataset | Model | 96 | 192 | 336 | 720 | Avg |
|---|---|---|---|---|---|---|
| ETTm1 | *Naive (repeat last value)* | *1.214* | *1.261* | *1.287* | *1.322* | *1.271* |
| | *PatchTST (2023)* | *0.329* | *0.367* | *0.399* | *0.454* | *0.387* |
| | *TQNet (2025)* | *0.311* | *0.356* | *0.390* | *0.452* | *0.377* |
| | *PAMNet (2026)* | ***0.308*** | ***0.353*** | ***0.375*** | ***0.425*** | ***0.365*** |
| | `patch8` | 0.323 ± 0.007 | 0.371 ± 0.006 | 0.392 ± 0.001 | 0.452 ± 0.002 | 0.384 |
| | `cross_gate_l0_fast` | 0.321 ± 0.006 | 0.357 ± 0.003 | 0.385 ± 0.003 | 0.445 ± 0.003 | 0.377 |
| | `cross_par_l0` | 0.325 ± 0.005 | 0.358 ± 0.004 | 0.385 ± 0.001 | 0.441 ± 0.003 | 0.377 |
| ETTm2 | *Naive (repeat last value)* | *0.266* | *0.340* | *0.412* | *0.522* | *0.385* |
| | *PatchTST (2023)* | *0.175* | *0.241* | *0.305* | *0.402* | *0.281* |
| | *CycleNet (2024)* | ***0.163*** | *0.229* | ***0.284*** | *0.389* | *0.266* |
| | *PAMNet (2026)* | *0.165* | ***0.227*** | *0.285* | ***0.380*** | ***0.264*** |
| | `patch8` | 0.182 ± 0.001 | 0.252 ± 0.001 | 0.322 ± 0.005 | 0.417 ± 0.002 | 0.293 |
| | `cross_gate_l0_fast` | 0.185 ± 0.004 | 0.254 ± 0.002 | 0.330 ± 0.006 | 0.461 ± 0.011 | 0.307 |
| | `cross_par_l0` | 0.181 ± 0.003 | 0.261 ± 0.008 | 0.341 ± 0.024 | 0.436 ± 0.015 | 0.305 |
| Weather | *Naive (repeat last value)* | *0.259* | *0.309* | *0.376* | *0.465* | *0.352* |
| | *PatchTST (2023)* | *0.177* | *0.225* | *0.278* | *0.354* | *0.259* |
| | *TimeXer (2024)* | *0.157* | *0.204* | *0.261* | ***0.340*** | *0.241* |
| | *PAMNet (2026)* | ***0.154*** | ***0.202*** | ***0.260*** | *0.343* | ***0.240*** |
| | `patch8` | 0.175 ± 0.001 | 0.223 ± 0.001 | 0.278 ± 0.000 | 0.357 ± 0.001 | 0.258 |
| | `cross_gate_l0_fast` | 0.159 ± 0.001 | 0.207 ± 0.002 | 0.268 ± 0.001 | 0.349 ± 0.003 | 0.246 |
| | `cross_par_l0` | 0.159 ± 0.003 | 0.210 ± 0.002 | 0.269 ± 0.002 | 0.352 ± 0.003 | 0.248 |

- Against the 2023 and 2024 baselines the picture of horizon 96 holds: both L0 Cross Transformers beat PatchTST and iTransformer on average on ETTm1 (0.377 against 0.387 / 0.407) and Weather (0.246 to 0.248 against 0.259 / 0.258), and tie TQNet on ETTm1. Against the newest models they do not: PAMNet is 0.012 ahead on ETTm1 and 0.006 to 0.008 on Weather, and TimeXer, CycleNet and TQNet are 0.004 to 0.006 ahead on Weather.
- On ETTm1 exchanging information between the signals helps more with the horizon: at 96 all three variants tie, from 192 on the Cross Transformers lead `patch8` by 0.007 to 0.014.
- On Weather the gain of the cross blocks over `patch8` shrinks with the horizon (0.016 at 96, 0.010 at 336, 0.005 to 0.008 at 720). The L0 edges keep 47 to 59% of the connections at every horizon.
- On ETTm2 the Cross Transformers overfit at the long horizons: their training loss ends lower than `patch8`'s (0.49 to 0.52 against 0.56 at 720) while validation is the same or worse and test is worse (0.436 to 0.461 against 0.417), with a seed spread up to 0.024. `patch8` is the best of ours there and still 0.012 to 0.029 behind PatchTST at every horizon. ETTm2 is where channel independence without exchange is the safe choice, as it was at 96.
- Serial and parallel blocks still tie (averages within 0.002 on every dataset), so the parallel block stays the default Cross Transformer.
- Nine runs failed at first because another process held 25.7 GB of the GPU, and the six Weather horizon-720 Cross Transformer runs were finished on a second machine. Its environment first had Python 3.15, where `torch.compile` is not supported: the three `cross_gate_l0_fast` runs ran with `--no-compile`, and the three `cross_par_l0` runs ran out of memory two at a time (about 21 GB each without compilation) and were run one at a time after a downgrade to Python 3.14, with compilation (the first resumed from its epoch-5 checkpoint). Compilation changes only speed and memory, not the results.

