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

FI-2010 is the first financial dataset and the first classification task: from the 40 order book features (10 ask and 10 bid levels, price and volume) of the last 96 events, predict whether the mid-price moves up, stays or moves down over the next 10 events. Setup 2 of DeepLOB (7 days train, 3 days test); see the README for the protocol. An "event" in these FI-2010 sections is one column of the dataset files, which summarizes a block of 10 order book events, the unit in which DeepLOB, TLOB and MLPLOB also count their inputs. It is a step toward M6-style tasks, which also need class probabilities instead of point forecasts. The classifier averages the decoder features over the decoder steps and concatenates them over the signals, so with channel independence the classification head is the only place where the order book levels meet; the Cross Transformer lets them exchange information before that.

The test set is 71% stationary (training 61%), so always predicting "stationary" gives 71% accuracy but a macro F1 of about 0.28; macro F1 is the main score. DeepLOB reports accuracy 84.47% and F1 83.40% at k = 10 with an input of 100 events (Setup 2, Table II). Its recall equals its accuracy, so its precision, recall and F1 are weighted by class support, not macro averages; compare it with our weighted F1. (F1 77.66% / accuracy 78.91% is its Setup 1 result.)

| Variant | Question |
|---|---|
| `ci_raw`, `patch8` | Channel independence on order book data, where the levels are strongly coupled. |
| `shared_patch8` | All 40 features in one token sequence, the counterpart that mixes the levels from the first layer. |
| `cross_patch8`, `cross_patch8_gate` | The Cross Transformer with 40 signals (1,560 cross-signal edges), without and with L1 gates. |
| `cross_gate_l0_fast` (also ETT and Weather) | L0 edge masks at `--gate-lr` (1e-2). In round 8 the masks trained at the model learning rate and never closed an edge. On a synthetic task where only one edge matters (signal 0 is signal 1 delayed), this setting closes the 11 other edges within about 1,500 steps and keeps that one at weight 0.70. On FI-2010, the open edges show which order book levels the model reads. |

The round 8 Traffic variants are deferred.

Results on FI-2010, k = 10, test set. The published rows are copied from DeepLOB's Setup 2 table (input 100 events). Only DeepLOB's own F1 is weighted by class support (its recall equals its accuracy); the LSTM and TABL papers average precision, recall and F1 over the three classes, which is macro F1. The LSTM row is not strictly Setup 2: that paper uses its own preprocessing of the raw events (z-scored with the previous day) and its own labels.

| Model | Weighted F1 | Macro F1 | Accuracy | Seconds per epoch |
|---|---|---|---|---|
| *Always "stationary"* | *0.587* | *0.276* | *0.708* | |
| *LSTM (Tsantekidis et al. 2017)* | | *0.663* | | |
| *B(TABL) (Tran et al. 2018)* | | *0.692* | *0.789* | |
| *C(TABL) (Tran et al. 2018)* | | *0.776* | *0.847* | |
| *DeepLOB (Zhang et al. 2019)* | *0.834* | | *0.845* | |
| `ci_raw` | 0.866 ± 0.004 | 0.801 ± 0.008 | 0.870 | ~25 |
| `patch8` | 0.873 ± 0.001 | 0.811 ± 0.001 | 0.876 | ~20 |
| `shared_patch8` | 0.708 ± 0.112 | 0.518 ± 0.219 | 0.763 | ~8 |
| `cross_patch8` | 0.870 ± 0.002 | 0.804 ± 0.005 | 0.877 | ~220 |
| `cross_patch8_gate` | **0.874 ± 0.000** | 0.811 ± 0.001 | **0.879** | ~215 |
| `cross_gate_l0_fast` | **0.874 ± 0.000** | **0.812 ± 0.001** | 0.878 | ~215 |

Stronger LOB models at k = 10. Published F1 scores mix two metrics: DeepLOB, TransLOB and Axial-LOB report weighted F1 (their recall equals their accuracy), TABL averages over the classes (macro), while the LOBCAST benchmark (Prata et al. 2024) and TLOB (Berti et al. 2025) retrain every model in one framework and report macro F1. The macro rows are the fair comparison for our macro F1.

| Model | Source | F1 type | k = 10 F1 | Input | Notes |
|---|---|---|---|---|---|
| *DeepLOB* | own paper, Setup 2 | weighted | *0.834* | 100 events, 40 features | |
| *Axial-LOB (2022)* | own paper, 7/3 days | weighted | *0.851* | 40 features | |
| *TransLOB (2020)* | own paper, 7/3 days | weighted | *0.887* | 100 events | LOBCAST could not reproduce it: 59.4 against a claimed 87.3 averaged over horizons, the largest gap in the benchmark (ATNBoF is next, 40.9 against 67.1, and has the lowest robustness score, 66.1 against TransLOB's 69.9) |
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

## Round 15: a learnable daily cycle, and more dropout (3 seeds)

The models ahead of ours in round 14 share one ingredient: CycleNet, TQNet and PAMNet all model the periodic structure of the data explicitly. CycleNet's residual cycle forecasting is the simplest form: a learnable cycle of W steps per signal, indexed by the absolute time step modulo W, is subtracted from the normalized input window, the backbone forecasts the residual, and the cycle of the forecast steps is added back. It needs no new layers, only C x W parameters.

`--cycle-len W` (`Signalformer(cycle_len=W)`) adds it to our model, with W one day (`daily`: 96 steps on ETTm, 144 on Weather). The cycle is learned with the model learning rate and no weight decay; each window is normalized by its own mean and std before the cycle is subtracted, and the model's RevIN then runs on the residual. The absolute time step of each window comes from the split borders, so the phase is the same in training, validation and test.

| Variant | Question |
|---|---|
| `patch8_cycle` | Does the learnable cycle close the gap to CycleNet / PAMNet with channel independence, above all on ETTm2 where the Cross Transformer did not help? |
| `cross_par_l0_cycle` | Does it add to the exchange between the signals, or do they capture the same structure (Weather, ETTm1)? |
| `cross_par_l0_drop` | Does dropout 0.3 (instead of 0.2) stop the Cross Transformer from overfitting ETTm2 at the long horizons? |

ETTm1, ETTm2 and Weather at horizons 96, 192, 336 and 720, input 96, 3 seeds: 108 runs. The references are the table of round 14.

```bash
python demo/run_benchmarks.py --datasets ETTm1 ETTm2 Weather --pred-lens 96 192 336 720   # round 15
```

Results, test MSE (Avg is the mean over the four horizons). `patch8` and `cross_par_l0` are the runs of rounds 13 and 14.

| Dataset | Model | 96 | 192 | 336 | 720 | Avg |
|---|---|---|---|---|---|---|
| ETTm1 | *Naive (repeat last value)* | *1.214* | *1.261* | *1.287* | *1.322* | *1.271* |
|  | *PatchTST (2023)* | *0.329* | *0.367* | *0.399* | *0.454* | *0.387* |
|  | *CycleNet (2024)* | *0.319* | *0.360* | *0.389* | *0.447* | *0.379* |
|  | *TQNet (2025)* | *0.311* | *0.356* | *0.390* | *0.452* | *0.377* |
|  | *PAMNet (2026)* | ***0.308*** | ***0.353*** | ***0.375*** | ***0.425*** | ***0.365*** |
| | `patch8` | 0.323 ± 0.007 | 0.371 ± 0.006 | 0.392 ± 0.001 | 0.452 ± 0.002 | 0.384 |
| | `patch8_cycle` | 0.321 ± 0.004 | 0.361 ± 0.002 | 0.392 ± 0.002 | 0.446 ± 0.001 | 0.380 |
| | `cross_par_l0` | 0.325 ± 0.005 | 0.358 ± 0.004 | 0.385 ± 0.001 | 0.441 ± 0.003 | 0.377 |
| | `cross_par_l0_cycle` | 0.321 ± 0.001 | 0.364 ± 0.008 | 0.389 ± 0.008 | 0.447 ± 0.004 | 0.380 |
| | `cross_par_l0_drop` | 0.320 ± 0.003 | 0.357 ± 0.005 | 0.385 ± 0.003 | 0.442 ± 0.001 | 0.376 |
| ETTm2 | *Naive (repeat last value)* | *0.266* | *0.340* | *0.412* | *0.522* | *0.385* |
|  | *PatchTST (2023)* | *0.175* | *0.241* | *0.305* | *0.402* | *0.281* |
|  | *TQNet (2025)* | *0.173* | *0.238* | *0.301* | *0.397* | *0.277* |
|  | *CycleNet (2024)* | ***0.163*** | *0.229* | ***0.284*** | *0.389* | *0.266* |
|  | *PAMNet (2026)* | *0.165* | ***0.227*** | *0.285* | ***0.380*** | ***0.264*** |
| | `patch8` | 0.182 ± 0.001 | 0.252 ± 0.001 | 0.322 ± 0.005 | 0.417 ± 0.002 | 0.293 |
| | `patch8_cycle` | 0.166 ± 0.001 | 0.233 ± 0.002 | 0.302 ± 0.003 | 0.405 ± 0.003 | 0.277 |
| | `cross_par_l0` | 0.181 ± 0.003 | 0.261 ± 0.008 | 0.341 ± 0.024 | 0.436 ± 0.015 | 0.305 |
| | `cross_par_l0_cycle` | 0.169 ± 0.002 | 0.242 ± 0.008 | 0.304 ± 0.004 | 0.397 ± 0.008 | 0.278 |
| | `cross_par_l0_drop` | 0.184 ± 0.003 | 0.258 ± 0.003 | 0.331 ± 0.009 | 0.439 ± 0.010 | 0.303 |
| Weather | *Naive (repeat last value)* | *0.259* | *0.309* | *0.376* | *0.465* | *0.352* |
|  | *PatchTST (2023)* | *0.177* | *0.225* | *0.278* | *0.354* | *0.259* |
|  | *CycleNet (2024)* | *0.158* | *0.207* | *0.262* | *0.344* | *0.243* |
|  | *TimeXer (2024)* | *0.157* | *0.204* | *0.261* | ***0.340*** | *0.241* |
|  | *PAMNet (2026)* | ***0.154*** | ***0.202*** | ***0.260*** | *0.343* | ***0.240*** |
| | `patch8` | 0.175 ± 0.001 | 0.223 ± 0.001 | 0.278 ± 0.000 | 0.357 ± 0.001 | 0.258 |
| | `patch8_cycle` | 0.163 ± 0.001 | 0.213 ± 0.000 | 0.273 ± 0.001 | 0.351 ± 0.000 | 0.250 |
| | `cross_par_l0` | 0.159 ± 0.003 | 0.210 ± 0.002 | 0.269 ± 0.002 | 0.352 ± 0.003 | 0.248 |
| | `cross_par_l0_cycle` | 0.157 ± 0.001 | 0.210 ± 0.002 | 0.269 ± 0.002 | 0.349 ± 0.003 | 0.246 |
| | `cross_par_l0_drop` | 0.161 ± 0.002 | 0.212 ± 0.002 | 0.272 ± 0.002 | 0.355 ± 0.003 | 0.250 |

- The cycle helps most where the Cross Transformer did not: on ETTm2 it lowers `patch8` from 0.293 to 0.277 and `cross_par_l0` from 0.305 to 0.278, at every horizon. Both now tie TQNet (0.277), beat PatchTST (0.281) and are 0.003 behind TimeXer, but stay 0.011 to 0.014 behind CycleNet and PAMNet.
- On ETTm2 the cycle also removes the long-horizon trouble of the Cross Transformer: at 720 its test MSE falls from 0.436 to 0.397 (the best of ours, level with TQNet's 0.397) and the seed spread from 0.015 to 0.008, while its training loss stays the same (0.50). So the cross blocks were not overfitting noise in general: without the cycle they fitted the daily pattern through the other signals, and that did not carry over to the test period.
- On Weather the cycle and the exchange between the signals capture mostly the same structure: the cycle lowers `patch8` from 0.258 to 0.250, the cross blocks from 0.258 to 0.248, and both together reach 0.246, the same as the serial `cross_gate_l0_fast`. PAMNet, TimeXer, TQNet and CycleNet remain 0.003 to 0.006 ahead.
- On ETTm1 the cycle helps channel independence a little (0.384 → 0.380) and the Cross Transformer not at all (0.377 → 0.380, within the seed spread at 192 and 336). The best of ours stays 0.376 to 0.377, level with TQNet, 0.012 behind PAMNet.
- Dropout 0.3 does not fix ETTm2 (0.303 against 0.305 with 0.2) and changes nothing elsewhere (within 0.002). Dropout 0.2 stays the default.
- Our cycle normalizes each window before subtracting the cycle and then runs the model's RevIN again on the residual; CycleNet runs RevIN only once. The second normalization may cost part of the gain where the cycle is weak relative to the residual; it is the first thing to test before adding the cycle to more variants.
- Five Weather horizon-720 Cross Transformer runs ran out of memory two at a time (another user held 5.5 GB of the GPU) and were rerun one at a time.


## Round 16: the cycle normalized once (3 seeds)

In round 15 each window is normalized by its own mean and std, the cycle is subtracted, and the model's RevIN then normalizes the residual a second time. CycleNet normalizes only once: the backbone sees the normalized window minus the cycle at its own scale. The second normalization rescales every residual to std 1, so a window that the cycle explains well (small residual) has its leftover noise blown up to the same size as a window it explains badly, and the backbone no longer sees how much of the window the cycle explained. That could be why the cycle helped little on ETTm1 and did not add to the Cross Transformer there.

`--cycle-norm once` (`Signalformer(cycle_renorm=False)`) skips RevIN on the residual; `twice`, the round-15 behavior, stays the default so the earlier runs keep their meaning.

| Variant | Question |
|---|---|
| `patch8_cycle_once` | Does one normalization, as in CycleNet, give a larger gain than round 15's `patch8_cycle` (ETTm1 0.380, ETTm2 0.277, Weather 0.250)? |
| `cross_par_l0_cycle_once` | Does it let the cycle add to the Cross Transformer on ETTm1 and Weather (round 15: 0.380 and 0.246 against 0.377 and 0.248 without the cycle)? |

ETTm1, ETTm2 and Weather at horizons 96, 192, 336 and 720, input 96, 3 seeds: 72 runs. The references are the table of round 14.

```bash
python demo/run_benchmarks.py --datasets ETTm1 ETTm2 Weather --pred-lens 96 192 336 720   # round 16
```

Results, test MSE (Avg is the mean over the four horizons). The `_cycle` rows (normalized twice) are the runs of round 15.

| Dataset | Model | 96 | 192 | 336 | 720 | Avg |
|---|---|---|---|---|---|---|
| ETTm1 | *Naive (repeat last value)* | *1.214* | *1.261* | *1.287* | *1.322* | *1.271* |
|  | *PatchTST (2023)* | *0.329* | *0.367* | *0.399* | *0.454* | *0.387* |
|  | *CycleNet (2024)* | *0.319* | *0.360* | *0.389* | *0.447* | *0.379* |
|  | *TQNet (2025)* | *0.311* | *0.356* | *0.390* | *0.452* | *0.377* |
|  | *PAMNet (2026)* | ***0.308*** | ***0.353*** | ***0.375*** | ***0.425*** | ***0.365*** |
| | `patch8_cycle` | 0.321 ± 0.004 | 0.361 ± 0.002 | 0.392 ± 0.002 | 0.446 ± 0.001 | 0.380 |
| | `patch8_cycle_once` | 0.321 ± 0.005 | 0.364 ± 0.004 | 0.391 ± 0.000 | 0.443 ± 0.003 | 0.380 |
| | `cross_par_l0_cycle` | 0.321 ± 0.001 | 0.364 ± 0.008 | 0.389 ± 0.008 | 0.447 ± 0.004 | 0.380 |
| | `cross_par_l0_cycle_once` | 0.320 ± 0.002 | 0.360 ± 0.005 | 0.389 ± 0.004 | 0.447 ± 0.004 | 0.379 |
| ETTm2 | *Naive (repeat last value)* | *0.266* | *0.340* | *0.412* | *0.522* | *0.385* |
|  | *PatchTST (2023)* | *0.175* | *0.241* | *0.305* | *0.402* | *0.281* |
|  | *TQNet (2025)* | *0.173* | *0.238* | *0.301* | *0.397* | *0.277* |
|  | *CycleNet (2024)* | ***0.163*** | *0.229* | ***0.284*** | *0.389* | *0.266* |
|  | *PAMNet (2026)* | *0.165* | ***0.227*** | *0.285* | ***0.380*** | ***0.264*** |
| | `patch8_cycle` | 0.166 ± 0.001 | 0.233 ± 0.002 | 0.302 ± 0.003 | 0.405 ± 0.003 | 0.277 |
| | `patch8_cycle_once` | 0.164 ± 0.001 | 0.230 ± 0.001 | 0.300 ± 0.001 | 0.403 ± 0.005 | 0.274 |
| | `cross_par_l0_cycle` | 0.169 ± 0.002 | 0.242 ± 0.008 | 0.304 ± 0.004 | 0.397 ± 0.008 | 0.278 |
| | `cross_par_l0_cycle_once` | 0.169 ± 0.002 | 0.244 ± 0.008 | 0.313 ± 0.016 | 0.405 ± 0.009 | 0.283 |
| Weather | *Naive (repeat last value)* | *0.259* | *0.309* | *0.376* | *0.465* | *0.352* |
|  | *PatchTST (2023)* | *0.177* | *0.225* | *0.278* | *0.354* | *0.259* |
|  | *CycleNet (2024)* | *0.158* | *0.207* | *0.262* | *0.344* | *0.243* |
|  | *TimeXer (2024)* | *0.157* | *0.204* | *0.261* | ***0.340*** | *0.241* |
|  | *PAMNet (2026)* | ***0.154*** | ***0.202*** | ***0.260*** | *0.343* | ***0.240*** |
| | `patch8_cycle` | 0.163 ± 0.001 | 0.213 ± 0.000 | 0.273 ± 0.001 | 0.351 ± 0.000 | 0.250 |
| | `patch8_cycle_once` | 0.158 ± 0.000 | 0.208 ± 0.001 | 0.267 ± 0.001 | 0.347 ± 0.001 | 0.245 |
| | `cross_par_l0_cycle` | 0.157 ± 0.001 | 0.210 ± 0.002 | 0.269 ± 0.002 | 0.349 ± 0.003 | 0.246 |
| | `cross_par_l0_cycle_once` | 0.159 ± 0.002 | 0.211 ± 0.002 | 0.271 ± 0.001 | 0.350 ± 0.002 | 0.247 |

- One normalization helps channel independence: `patch8_cycle_once` improves on `patch8_cycle` on ETTm2 (0.277 → 0.274, at every horizon) and on Weather (0.250 → 0.245, at every horizon, the seed spread of the averages is below 0.001), and ties it on ETTm1 (0.380). On Weather it is now the best of ours, level with the Cross Transformer (0.246) and 0.002 behind CycleNet (0.243); on ETTm2 it beats TQNet (0.277) but stays 0.008 to 0.010 behind CycleNet and PAMNet.
- It does not help the Cross Transformer: `cross_par_l0_cycle_once` ties `cross_par_l0_cycle` on ETTm1 (0.379 against 0.380) and Weather (0.247 against 0.246) and is worse on ETTm2 (0.283 against 0.278, above all at 336 and 720 with a larger seed spread). With the residual at its own scale, the cross blocks seem to fit the other signals' leftover daily pattern again, as without the cycle in round 14.
- Once the cycle is modeled, the exchange between the signals gains little: on Weather and ETTm2 the channel-independent `patch8_cycle_once` is the best of ours or level with it, and only on ETTm1 does the Cross Transformer without the cycle stay ahead (0.376 to 0.377 against 0.380), by about two seed spreads. The Cross Transformer keeps the round-15 cycle (normalized twice) as its base in round 17.
- Two Weather horizon-720 Cross Transformer runs ran out of memory (another user's process on the GPU) and were rerun.


## Round 17: edges that depend on the time of day, and shared layers (3 seeds)

TQNet makes its attention across the signals depend on the time of day: its queries are learnable vectors indexed by the time step modulo the period. Our parallel Cross Transformer already weighs every edge, the update signal i receives from signal j, by a gate g[i, j] times an L0 mask m[i, j]. `--phase-edges` (`Signalformer(phase_edges=True)`, which needs `--cycle-len`) adds a learnable offset per phase: receiver i weighs source j at forecast step t by (g[i, j] + p[t mod W, i, j]) · m[i, j], with W the cycle length. The offsets are indexed by the phase of each forecast step, so the steps of one 720-step forecast use different weights; they start at zero (the model starts as `cross_par_l0_cycle`), get the model's weight decay (which pulls them toward a fixed graph), and the masks stay shared by all phases, so an edge closed by L0 is closed at every time of day. This adds W x C x C parameters per decoder layer: 4,704 on ETTm (C = 7, W = 96) and 63,504 on Weather (C = 21, W = 144), about as many as the rest of the Weather model.

`--share-layers` makes adjacent pairs of encoder blocks and of decoder blocks one module. With two layers each, the model has the depth (and compute) of two layers and the parameters of one: if it beats the unshared two-layer model, sharing regularizes; if it beats the one-layer model, the extra depth helps without extra parameters.

| Variant | Question |
|---|---|
| `cross_par_l0_cycle_pedge` | Do signal exchanges that change with the time of day close the gap to TQNet / CycleNet on Weather and ETTm1, where the exchange helps? |
| `cross_par_l0_cycle_e2d2` | Does a second encoder and decoder layer help (round 7: `patch8_e2d2` did not at horizon 96)? |
| `cross_par_l0_cycle_e2d2_shared` | Does sharing the weights of the two layers regularize, above all on ETTm2 where the Cross Transformer overfits? |

The base is `cross_par_l0_cycle`, the cycle normalized twice, since one normalization did not help the Cross Transformer in round 16. ETTm1, ETTm2 and Weather at horizons 96, 192, 336 and 720, input 96, 3 seeds: 108 runs. The references are the table of round 16.

```bash
python demo/run_benchmarks.py --datasets ETTm1 ETTm2 Weather --pred-lens 96 192 336 720   # round 17
```

Results, test MSE (Avg is the mean over the four horizons). The `cross_par_l0_cycle` and `patch8_cycle_once` rows are the runs of rounds 15 and 16.

| Dataset | Model | 96 | 192 | 336 | 720 | Avg |
|---|---|---|---|---|---|---|
| ETTm1 | *Naive (repeat last value)* | *1.214* | *1.261* | *1.287* | *1.322* | *1.271* |
|  | *PatchTST (2023)* | *0.329* | *0.367* | *0.399* | *0.454* | *0.387* |
|  | *CycleNet (2024)* | *0.319* | *0.360* | *0.389* | *0.447* | *0.379* |
|  | *TQNet (2025)* | *0.311* | *0.356* | *0.390* | *0.452* | *0.377* |
|  | *PAMNet (2026)* | ***0.308*** | ***0.353*** | ***0.375*** | ***0.425*** | ***0.365*** |
| | `patch8_cycle_once` | 0.321 ± 0.005 | 0.364 ± 0.004 | 0.391 ± 0.000 | 0.443 ± 0.003 | 0.380 |
| | `cross_par_l0_cycle` | 0.321 ± 0.001 | 0.364 ± 0.008 | 0.389 ± 0.008 | 0.447 ± 0.004 | 0.380 |
| | `cross_par_l0_cycle_pedge` | 0.311 ± 0.006 | 0.367 ± 0.028 | 0.380 ± 0.010 | 0.441 ± 0.005 | 0.375 |
| | `cross_par_l0_cycle_e2d2` | 0.321 ± 0.003 | 0.372 ± 0.012 | 0.402 ± 0.009 | 0.449 ± 0.002 | 0.386 |
| | `cross_par_l0_cycle_e2d2_shared` | 0.319 ± 0.004 | 0.367 ± 0.006 | 0.391 ± 0.006 | 0.449 ± 0.001 | 0.381 |
| ETTm2 | *Naive (repeat last value)* | *0.266* | *0.340* | *0.412* | *0.522* | *0.385* |
|  | *PatchTST (2023)* | *0.175* | *0.241* | *0.305* | *0.402* | *0.281* |
|  | *TQNet (2025)* | *0.173* | *0.238* | *0.301* | *0.397* | *0.277* |
|  | *CycleNet (2024)* | ***0.163*** | *0.229* | ***0.284*** | *0.389* | *0.266* |
|  | *PAMNet (2026)* | *0.165* | ***0.227*** | *0.285* | ***0.380*** | ***0.264*** |
| | `patch8_cycle_once` | 0.164 ± 0.001 | 0.230 ± 0.001 | 0.300 ± 0.001 | 0.403 ± 0.005 | 0.274 |
| | `cross_par_l0_cycle` | 0.169 ± 0.002 | 0.242 ± 0.008 | 0.304 ± 0.004 | 0.397 ± 0.008 | 0.278 |
| | `cross_par_l0_cycle_pedge` | 0.175 ± 0.002 | 0.247 ± 0.003 | 0.310 ± 0.001 | 0.403 ± 0.012 | 0.284 |
| | `cross_par_l0_cycle_e2d2` | 0.172 ± 0.002 | 0.244 ± 0.000 | 0.319 ± 0.012 | 0.417 ± 0.010 | 0.288 |
| | `cross_par_l0_cycle_e2d2_shared` | 0.170 ± 0.001 | 0.246 ± 0.005 | 0.304 ± 0.001 | 0.414 ± 0.006 | 0.283 |
| Weather | *Naive (repeat last value)* | *0.259* | *0.309* | *0.376* | *0.465* | *0.352* |
|  | *PatchTST (2023)* | *0.177* | *0.225* | *0.278* | *0.354* | *0.259* |
|  | *CycleNet (2024)* | *0.158* | *0.207* | *0.262* | *0.344* | *0.243* |
|  | *TimeXer (2024)* | *0.157* | *0.204* | *0.261* | ***0.340*** | *0.241* |
|  | *PAMNet (2026)* | ***0.154*** | ***0.202*** | ***0.260*** | *0.343* | ***0.240*** |
| | `patch8_cycle_once` | 0.158 ± 0.000 | 0.208 ± 0.001 | 0.267 ± 0.001 | 0.347 ± 0.001 | 0.245 |
| | `cross_par_l0_cycle` | 0.157 ± 0.001 | 0.210 ± 0.002 | 0.269 ± 0.002 | 0.349 ± 0.003 | 0.246 |
| | `cross_par_l0_cycle_pedge` | 0.160 ± 0.001 | 0.210 ± 0.002 | 0.269 ± 0.002 | 0.350 ± 0.003 | 0.247 |
| | `cross_par_l0_cycle_e2d2` | 0.157 ± 0.002 | 0.209 ± 0.001 | 0.266 ± 0.001 | 0.348 ± 0.002 | 0.245 |
| | `cross_par_l0_cycle_e2d2_shared` | 0.159 ± 0.003 | 0.208 ± 0.002 | 0.268 ± 0.002 | 0.347 ± 0.002 | 0.245 |

- Phase-dependent edges help only on ETTm1: `cross_par_l0_cycle_pedge` improves on `cross_par_l0_cycle` there (0.380 → 0.375, best at 96 with 0.311, level with TQNet), the best of ours on ETTm1, but the seed spread at 192 is large (± 0.028). On ETTm2 it is worse (0.278 → 0.284, at every horizon but 720) and on Weather it is level or slightly worse (0.247 against 0.246), where the 63,504 phase offsets are as many as the rest of the model. A smoother description of the offsets (a few harmonics instead of one per phase) is tested in round 18.
- A second layer does not help: `cross_par_l0_cycle_e2d2` is worse on ETTm1 (0.386) and ETTm2 (0.288) and only 0.001 better on Weather (0.245), within the seed spread.
- Sharing the two layers regularizes but does not beat one layer: `_e2d2_shared` improves on the unshared two layers on ETTm1 (0.386 → 0.381) and ETTm2 (0.288 → 0.283) with the same compute and the parameters of one layer, but stays level with or behind the one-layer `cross_par_l0_cycle` (0.380, 0.278). Depth and sharing are dropped from the next rounds; the options stay.
- No Cross Transformer variant beats the channel-independent `patch8_cycle_once` on ETTm2 (0.274) or clearly on Weather (0.245); only on ETTm1 is the exchange ahead (0.375 against 0.380).

## Round 18: the cycle in the frequency domain, and a fused cross memory (3 seeds)

The learnable cycle of rounds 15 to 17 is a table of one value per phase and signal (W x C). `--cycle-harmonics K` describes it in the frequency domain instead: c(t) = a0 + sum over k of a_k cos(2 pi k t / W) + b_k sin(2 pi k t / W), evaluated directly at the steps of each window (the input is not transformed; only the cycle's parameters are Fourier coefficients). With every harmonic (K = W / 2, `-1`) it spans the same cycles as the table, so only the parameterization and the optimization change; with few harmonics the cycle is forced to be smooth, which may regularize it. `--cycle-weekly K` adds K harmonics of a week (7 W), which a table of one day cannot hold. `--phase-harmonics K` describes the phase offsets of round 17's phase-dependent edges the same way (2K x C x C instead of W x C x C).

`--cross-fused mlp` replaces the complete graph of the parallel Cross Transformer by one shared memory: at each token position the encoded tokens of all C signals are concatenated (C x d) and projected by one layer to d, then GELU, and every signal's decoder (with its signal embedding added to the queries) cross-attends to that memory. It has no edge gates and no phase parameters. A deeper projection (`--cross-fused ffn`, C x d -> 4d -> d) is kept for the case that the one layer underfits.

| Variant | Question |
|---|---|
| `patch8_cycle_once_fourier_full` | Does the frequency-domain description alone (same cycles as the table) train to a better cycle than `patch8_cycle_once`? |
| `patch8_cycle_once_fourier_k8`, `_k3` | Does a smooth cycle (8 or 3 harmonics) generalize better, or does it lose the sharp parts of the day? |
| `patch8_cycle_once_fourier_k8_weekly` | Does a weekly cycle on top of the daily one help on ETTm? |
| `cross_par_l0_cycle_pedge_fourier` | Do smooth phase offsets keep the ETTm1 gain of `_pedge` (0.375) without its losses on ETTm2 and Weather? |
| `cross_fused_mlp_cycle` | Does one fused memory exchange the signals as well as the complete graph (`cross_par_l0_cycle`: 0.380, 0.278, 0.246) with fewer parameters on Weather? |

The decoder now passes the phase of every step only to the blocks with phase-dependent edges; the round-17 code passed it to every block, which broke the channel-independent variants with the cycle (`patch8_cycle`, `patch8_cycle_once`) in that version. ETTm1, ETTm2 and Weather at horizons 96, 192, 336 and 720 (`_weekly` on ETTm only), input 96, 3 seeds: 204 runs. The references are the table of round 17.

```bash
python demo/run_benchmarks.py --datasets ETTm1 ETTm2 Weather --pred-lens 96 192 336 720   # round 18
```

Results, test MSE (Avg is the mean over the four horizons). The `patch8_cycle_once`, `cross_par_l0_cycle` and `cross_par_l0_cycle_pedge` rows are the runs of rounds 15 to 17.

| Dataset | Model | 96 | 192 | 336 | 720 | Avg |
|---|---|---|---|---|---|---|
| ETTm1 | *Naive (repeat last value)* | *1.214* | *1.261* | *1.287* | *1.322* | *1.271* |
|  | *PatchTST (2023)* | *0.329* | *0.367* | *0.399* | *0.454* | *0.387* |
|  | *CycleNet (2024)* | *0.319* | *0.360* | *0.389* | *0.447* | *0.379* |
|  | *TQNet (2025)* | *0.311* | *0.356* | *0.390* | *0.452* | *0.377* |
|  | *PAMNet (2026)* | ***0.308*** | ***0.353*** | ***0.375*** | ***0.425*** | ***0.365*** |
| | `patch8_cycle_once` | 0.321 ± 0.005 | 0.364 ± 0.004 | 0.391 ± 0.000 | 0.443 ± 0.003 | 0.380 |
| | `patch8_cycle_once_fourier_full` | 0.321 ± 0.002 | 0.362 ± 0.004 | 0.390 ± 0.003 | 0.443 ± 0.003 | 0.379 |
| | `patch8_cycle_once_fourier_k8` | 0.321 ± 0.004 | 0.362 ± 0.004 | 0.392 ± 0.006 | 0.442 ± 0.001 | 0.380 |
| | `patch8_cycle_once_fourier_k3` | 0.323 ± 0.006 | 0.365 ± 0.002 | 0.388 ± 0.004 | 0.440 ± 0.003 | 0.379 |
| | `patch8_cycle_once_fourier_k8_weekly` | 0.325 ± 0.003 | 0.366 ± 0.001 | 0.400 ± 0.004 | 0.460 ± 0.000 | 0.388 |
| | `cross_par_l0_cycle` | 0.321 ± 0.001 | 0.364 ± 0.008 | 0.389 ± 0.008 | 0.447 ± 0.004 | 0.380 |
| | `cross_par_l0_cycle_pedge` | 0.311 ± 0.006 | 0.367 ± 0.028 | 0.380 ± 0.010 | 0.441 ± 0.005 | 0.375 |
| | `cross_par_l0_cycle_pedge_fourier` | 0.313 ± 0.009 | 0.366 ± 0.029 | 0.384 ± 0.009 | 0.445 ± 0.005 | 0.377 |
| | `cross_fused_mlp_cycle` | 0.359 ± 0.017 | 0.446 ± 0.014 | 0.479 ± 0.028 | 0.496 ± 0.019 | 0.445 |
| ETTm2 | *Naive (repeat last value)* | *0.266* | *0.340* | *0.412* | *0.522* | *0.385* |
|  | *PatchTST (2023)* | *0.175* | *0.241* | *0.305* | *0.402* | *0.281* |
|  | *TQNet (2025)* | *0.173* | *0.238* | *0.301* | *0.397* | *0.277* |
|  | *CycleNet (2024)* | ***0.163*** | *0.229* | ***0.284*** | *0.389* | *0.266* |
|  | *PAMNet (2026)* | *0.165* | ***0.227*** | *0.285* | ***0.380*** | ***0.264*** |
| | `patch8_cycle_once` | 0.164 ± 0.001 | 0.230 ± 0.001 | 0.300 ± 0.001 | 0.403 ± 0.005 | 0.274 |
| | `patch8_cycle_once_fourier_full` | 0.164 ± 0.001 | 0.229 ± 0.000 | 0.295 ± 0.003 | 0.394 ± 0.002 | 0.270 |
| | `patch8_cycle_once_fourier_k8` | 0.164 ± 0.000 | 0.229 ± 0.001 | 0.295 ± 0.001 | 0.397 ± 0.005 | 0.271 |
| | `patch8_cycle_once_fourier_k3` | 0.167 ± 0.001 | 0.232 ± 0.001 | 0.296 ± 0.002 | 0.398 ± 0.005 | 0.273 |
| | `patch8_cycle_once_fourier_k8_weekly` | 0.168 ± 0.001 | 0.239 ± 0.002 | 0.305 ± 0.004 | 0.402 ± 0.001 | 0.279 |
| | `cross_par_l0_cycle` | 0.169 ± 0.002 | 0.242 ± 0.008 | 0.304 ± 0.004 | 0.397 ± 0.008 | 0.278 |
| | `cross_par_l0_cycle_pedge` | 0.175 ± 0.002 | 0.247 ± 0.003 | 0.310 ± 0.001 | 0.403 ± 0.012 | 0.284 |
| | `cross_par_l0_cycle_pedge_fourier` | 0.177 ± 0.003 | 0.244 ± 0.003 | 0.305 ± 0.006 | 0.400 ± 0.006 | 0.281 |
| | `cross_fused_mlp_cycle` | 0.201 ± 0.006 | 0.262 ± 0.011 | 0.319 ± 0.005 | 0.425 ± 0.015 | 0.302 |
| Weather | *Naive (repeat last value)* | *0.259* | *0.309* | *0.376* | *0.465* | *0.352* |
|  | *PatchTST (2023)* | *0.177* | *0.225* | *0.278* | *0.354* | *0.259* |
|  | *CycleNet (2024)* | *0.158* | *0.207* | *0.262* | *0.344* | *0.243* |
|  | *TimeXer (2024)* | *0.157* | *0.204* | *0.261* | ***0.340*** | *0.241* |
|  | *PAMNet (2026)* | ***0.154*** | ***0.202*** | ***0.260*** | *0.343* | ***0.240*** |
| | `patch8_cycle_once` | 0.158 ± 0.000 | 0.208 ± 0.001 | 0.267 ± 0.001 | 0.347 ± 0.001 | 0.245 |
| | `patch8_cycle_once_fourier_full` | 0.158 ± 0.000 | 0.208 ± 0.001 | 0.267 ± 0.002 | 0.347 ± 0.000 | 0.245 |
| | `patch8_cycle_once_fourier_k8` | 0.157 ± 0.001 | 0.207 ± 0.000 | 0.266 ± 0.002 | 0.346 ± 0.001 | 0.244 |
| | `patch8_cycle_once_fourier_k3` | 0.157 ± 0.000 | 0.206 ± 0.000 | 0.264 ± 0.001 | 0.345 ± 0.001 | 0.243 |
| | `cross_par_l0_cycle` | 0.157 ± 0.001 | 0.210 ± 0.002 | 0.269 ± 0.002 | 0.349 ± 0.003 | 0.246 |
| | `cross_par_l0_cycle_pedge` | 0.160 ± 0.001 | 0.210 ± 0.002 | 0.269 ± 0.002 | 0.350 ± 0.003 | 0.247 |
| | `cross_par_l0_cycle_pedge_fourier` | 0.159 ± 0.001 | 0.210 ± 0.002 | 0.268 ± 0.001 | 0.348 ± 0.004 | 0.246 |
| | `cross_fused_mlp_cycle` | 0.180 ± 0.004 | 0.232 ± 0.003 | 0.285 ± 0.001 | 0.369 ± 0.003 | 0.267 |

- The frequency-domain cycle helps channel independence, and the best number of harmonics depends on the dataset. On ETTm2 every harmonic (`_fourier_full`) improves on the table (0.274 → 0.270, above all at 336 and 720: 0.300 → 0.295, 0.403 → 0.394), although it spans the same cycles: only the parameterization changes, and with it the optimization (each Adam step moves whole harmonics instead of single phases). On Weather the smoothest cycle is best (`_k3`: 0.245 → 0.243, at every horizon), level with CycleNet (0.243); on ETTm1 all versions tie the table (0.379 to 0.380). `_k8` is never the worst and within 0.001 of the best on ETTm2 and Weather. ETTm2 is still 0.004 to 0.006 behind CycleNet and PAMNet.
- A weekly cycle hurts: `_k8_weekly` is worse than `_k8` on ETTm1 (0.380 → 0.388, above all at 720: 0.442 → 0.460) and ETTm2 (0.271 → 0.279). With a 96-step input (one day) the model sees a seventh of a week, and the training split holds about 51 weeks (12 months), so the weekly harmonics fit the training weeks.
- Smooth phase offsets lose less but gain less: `cross_par_l0_cycle_pedge_fourier` (392 offsets on ETTm, 3,528 on Weather) is between the base and `_pedge` on ETTm1 (0.377 against 0.380 and 0.375) and ETTm2 (0.281 against 0.278 and 0.284), and ties the base on Weather (0.246). The phase-dependent edges still help only on ETTm1.
- One fused memory is clearly worse than the complete graph: `cross_fused_mlp_cycle` is worse on every dataset and horizon (0.445, 0.302, 0.267 against 0.380, 0.278, 0.246). It does not underfit: its training loss falls below that of `cross_par_l0_cycle` (ETTm2 96, seed 2021, last epoch: 0.161 against 0.181) while its best validation loss stays higher (0.132 against 0.119), so the deeper `--cross-fused ffn` is not run. The likely cause is that every signal reads its own past only through the fused C x d -> d projection, mixed with all the others, instead of attending to its own tokens.


## Round 19: the frequency-domain cycle in the Cross Transformer, and the fused memory plus each signal's own tokens (3 seeds)

Round 18's Fourier cycle improved channel independence on ETTm2 (every harmonic) and Weather (3 harmonics) but was only run with `patch8`. Round 19 moves it into the Cross Transformer, and combines it with the smooth phase-dependent edges on ETTm1, the one dataset where those help. `--fused-own` (`Signalformer(fused_own=True)`) adds each signal's own encoded tokens to the fused memory of `--cross-fused`, so a decoder reads its own past directly, as with channel independence, and the other signals only through the fused summary; it adds no parameters.

| Variant | Question |
|---|---|
| `cross_par_l0_cycle_fourier_full` | Does the Fourier cycle help the Cross Transformer as it helped `patch8` on ETTm2 (0.274 → 0.270)? |
| `cross_par_l0_cycle_fourier_k3` | Does the smooth cycle help it on Weather (`patch8`: 0.245 → 0.243)? |
| `cross_par_l0_cycle_pedge_fourier_k3` | Do the smooth cycle and the smooth phase-dependent edges add up on ETTm1 (0.377 with the table cycle)? |
| `cross_fused_own_mlp_cycle` | Was the fused memory worse (round 18: 0.445, 0.302, 0.267) because the decoders lost direct access to their own signal? |

The cycle is normalized twice, the Cross Transformer base since round 16. ETTm1, ETTm2 and Weather at horizons 96, 192, 336 and 720 (`_pedge_fourier_k3` on ETTm1 only), input 96, 3 seeds: 120 runs. The references are the table of round 18.

```bash
python demo/run_benchmarks.py --datasets ETTm1 ETTm2 Weather --pred-lens 96 192 336 720   # round 19
```

Results, test MSE (Avg is the mean over the four horizons). The rows of earlier variants are the runs of rounds 15 to 18.

| Dataset | Model | 96 | 192 | 336 | 720 | Avg |
|---|---|---|---|---|---|---|
| ETTm1 | *Naive (repeat last value)* | *1.214* | *1.261* | *1.287* | *1.322* | *1.271* |
|  | *PatchTST (2023)* | *0.329* | *0.367* | *0.399* | *0.454* | *0.387* |
|  | *CycleNet (2024)* | *0.319* | *0.360* | *0.389* | *0.447* | *0.379* |
|  | *TQNet (2025)* | *0.311* | *0.356* | *0.390* | *0.452* | *0.377* |
|  | *PAMNet (2026)* | ***0.308*** | ***0.353*** | ***0.375*** | ***0.425*** | ***0.365*** |
| | `patch8_cycle_once` | 0.321 ± 0.005 | 0.364 ± 0.004 | 0.391 ± 0.000 | 0.443 ± 0.003 | 0.380 |
| | `patch8_cycle_once_fourier_full` | 0.321 ± 0.002 | 0.362 ± 0.004 | 0.390 ± 0.003 | 0.443 ± 0.003 | 0.379 |
| | `patch8_cycle_once_fourier_k3` | 0.323 ± 0.006 | 0.365 ± 0.002 | 0.388 ± 0.004 | 0.440 ± 0.003 | 0.379 |
| | `cross_par_l0_cycle` | 0.321 ± 0.001 | 0.364 ± 0.008 | 0.389 ± 0.008 | 0.447 ± 0.004 | 0.380 |
| | `cross_par_l0_cycle_fourier_full` | 0.320 ± 0.002 | 0.369 ± 0.009 | 0.390 ± 0.005 | 0.448 ± 0.003 | 0.382 |
| | `cross_par_l0_cycle_fourier_k3` | 0.320 ± 0.004 | 0.362 ± 0.005 | 0.387 ± 0.001 | 0.445 ± 0.002 | 0.378 |
| | `cross_par_l0_cycle_pedge_fourier` | 0.313 ± 0.009 | 0.366 ± 0.029 | 0.384 ± 0.009 | 0.445 ± 0.005 | 0.377 |
| | `cross_par_l0_cycle_pedge_fourier_k3` | 0.310 ± 0.004 | 0.352 ± 0.004 | 0.385 ± 0.006 | 0.447 ± 0.007 | 0.374 |
| | `cross_fused_mlp_cycle` | 0.359 ± 0.017 | 0.446 ± 0.014 | 0.479 ± 0.028 | 0.496 ± 0.019 | 0.445 |
| | `cross_fused_own_mlp_cycle` | 0.340 ± 0.012 | 0.396 ± 0.002 | 0.440 ± 0.010 | 0.483 ± 0.012 | 0.415 |
| ETTm2 | *Naive (repeat last value)* | *0.266* | *0.340* | *0.412* | *0.522* | *0.385* |
|  | *PatchTST (2023)* | *0.175* | *0.241* | *0.305* | *0.402* | *0.281* |
|  | *TQNet (2025)* | *0.173* | *0.238* | *0.301* | *0.397* | *0.277* |
|  | *CycleNet (2024)* | ***0.163*** | *0.229* | ***0.284*** | *0.389* | *0.266* |
|  | *PAMNet (2026)* | *0.165* | ***0.227*** | *0.285* | ***0.380*** | ***0.264*** |
| | `patch8_cycle_once` | 0.164 ± 0.001 | 0.230 ± 0.001 | 0.300 ± 0.001 | 0.403 ± 0.005 | 0.274 |
| | `patch8_cycle_once_fourier_full` | 0.164 ± 0.001 | 0.229 ± 0.000 | 0.295 ± 0.003 | 0.394 ± 0.002 | 0.270 |
| | `patch8_cycle_once_fourier_k3` | 0.167 ± 0.001 | 0.232 ± 0.001 | 0.296 ± 0.002 | 0.398 ± 0.005 | 0.273 |
| | `cross_par_l0_cycle` | 0.169 ± 0.002 | 0.242 ± 0.008 | 0.304 ± 0.004 | 0.397 ± 0.008 | 0.278 |
| | `cross_par_l0_cycle_fourier_full` | 0.167 ± 0.001 | 0.240 ± 0.008 | 0.300 ± 0.001 | 0.392 ± 0.004 | 0.275 |
| | `cross_par_l0_cycle_fourier_k3` | 0.169 ± 0.001 | 0.239 ± 0.006 | 0.304 ± 0.002 | 0.396 ± 0.012 | 0.277 |
| | `cross_par_l0_cycle_pedge_fourier` | 0.177 ± 0.003 | 0.244 ± 0.003 | 0.305 ± 0.006 | 0.400 ± 0.006 | 0.281 |
| | `cross_fused_mlp_cycle` | 0.201 ± 0.006 | 0.262 ± 0.011 | 0.319 ± 0.005 | 0.425 ± 0.015 | 0.302 |
| | `cross_fused_own_mlp_cycle` | 0.183 ± 0.011 | 0.253 ± 0.002 | 0.309 ± 0.004 | 0.418 ± 0.006 | 0.291 |
| Weather | *Naive (repeat last value)* | *0.259* | *0.309* | *0.376* | *0.465* | *0.352* |
|  | *PatchTST (2023)* | *0.177* | *0.225* | *0.278* | *0.354* | *0.259* |
|  | *CycleNet (2024)* | *0.158* | *0.207* | *0.262* | *0.344* | *0.243* |
|  | *TimeXer (2024)* | *0.157* | *0.204* | *0.261* | ***0.340*** | *0.241* |
|  | *PAMNet (2026)* | ***0.154*** | ***0.202*** | ***0.260*** | *0.343* | ***0.240*** |
| | `patch8_cycle_once` | 0.158 ± 0.000 | 0.208 ± 0.001 | 0.267 ± 0.001 | 0.347 ± 0.001 | 0.245 |
| | `patch8_cycle_once_fourier_full` | 0.158 ± 0.000 | 0.208 ± 0.001 | 0.267 ± 0.002 | 0.347 ± 0.000 | 0.245 |
| | `patch8_cycle_once_fourier_k3` | 0.157 ± 0.000 | 0.206 ± 0.000 | 0.264 ± 0.001 | 0.345 ± 0.001 | 0.243 |
| | `cross_par_l0_cycle` | 0.157 ± 0.001 | 0.210 ± 0.002 | 0.269 ± 0.002 | 0.349 ± 0.003 | 0.246 |
| | `cross_par_l0_cycle_fourier_full` | 0.158 ± 0.001 | 0.210 ± 0.001 | 0.270 ± 0.001 | 0.351 ± 0.005 | 0.247 |
| | `cross_par_l0_cycle_fourier_k3` | 0.157 ± 0.001 | 0.209 ± 0.000 | 0.267 ± 0.004 | 0.349 ± 0.003 | 0.246 |
| | `cross_par_l0_cycle_pedge_fourier` | 0.159 ± 0.001 | 0.210 ± 0.002 | 0.268 ± 0.001 | 0.348 ± 0.004 | 0.246 |
| | `cross_fused_mlp_cycle` | 0.180 ± 0.004 | 0.232 ± 0.003 | 0.285 ± 0.001 | 0.369 ± 0.003 | 0.267 |
| | `cross_fused_own_mlp_cycle` | 0.163 ± 0.003 | 0.213 ± 0.001 | 0.272 ± 0.002 | 0.360 ± 0.007 | 0.252 |

- The Fourier cycle helps the Cross Transformer less than channel independence. On ETTm2 every harmonic improves `cross_par_l0_cycle` by 0.003 (0.278 → 0.275, mostly at 336 and 720), against 0.004 for `patch8`; on Weather 3 harmonics tie the table (0.246) instead of improving it, and on ETTm1 3 harmonics are 0.002 better (0.378) and every harmonic 0.002 worse (0.382). The channel-independent `patch8_cycle_once_fourier_full` / `_k3` stay the best of ours on ETTm2 (0.270) and Weather (0.243).
- The smooth cycle and the smooth phase-dependent edges add up on ETTm1: `cross_par_l0_cycle_pedge_fourier_k3` reaches 0.374, the best of ours on ETTm1, ahead of TQNet (0.377) and CycleNet (0.379) but 0.009 behind PAMNet (0.365). It also removes the large seed spread at 192 of both earlier phase-edge variants (± 0.004 against ± 0.028 and ± 0.029); the gain is at 96 and 192 (0.310, 0.352), while 336 and 720 equal `_pedge_fourier`.
- The decoders' own tokens explain part of the fused memory's loss, not all of it: `cross_fused_own_mlp_cycle` improves on `cross_fused_mlp_cycle` everywhere (0.445 → 0.415, 0.302 → 0.291, 0.267 → 0.252) but stays clearly behind the complete graph (0.380, 0.278, 0.246) and behind channel independence, which reads the same own tokens without the fused summary. Adding the fused summary of all signals hurts, so the fused memory is dropped.
- Thirteen Weather runs (six at 192, seven at 336) ran out of GPU memory while another user's processes held about 28 GB, and were rerun.


## Round 20: harmonics of the phase-dependent edges (3 seeds)

`cross_par_l0_cycle_pedge_fourier_k3` (a 3-harmonic cycle and phase-dependent edges described by 4 harmonics) is the best of ours on ETTm1 (0.374). Round 20 runs it on ETTm2 and Weather, to see whether one Cross Transformer configuration can serve all three datasets, and varies the number of harmonics of the edges on ETTm1.

| Variant | Question |
|---|---|
| `cross_par_l0_cycle_pedge_fourier_k3` | Does it lose on ETTm2 and Weather, as the phase-dependent edges did with the table cycle (0.281, 0.246 against `cross_par_l0_cycle`'s 0.278, 0.246)? |
| `cross_par_l0_cycle_pedge2_fourier_k3` | Are 2 harmonics per edge (twice a day) enough on ETTm1? |
| `cross_par_l0_cycle_pedge8_fourier_k3` | Do 8 harmonics per edge add to the 4 of round 19? |

ETTm1 (the three variants) and ETTm2 and Weather (the first only) at horizons 96, 192, 336 and 720, input 96, 3 seeds: 48 runs (the 12 ETTm1 runs of the first variant are those of round 19). No new code.

```bash
python demo/run_benchmarks.py --datasets ETTm1 ETTm2 Weather --pred-lens 96 192 336 720   # round 20
```

Results, test MSE (Avg is the mean over the four horizons). The rows of earlier variants are the runs of rounds 15 to 19.

| Dataset | Model | 96 | 192 | 336 | 720 | Avg |
|---|---|---|---|---|---|---|
| ETTm1 | *Naive (repeat last value)* | *1.214* | *1.261* | *1.287* | *1.322* | *1.271* |
|  | *PatchTST (2023)* | *0.329* | *0.367* | *0.399* | *0.454* | *0.387* |
|  | *CycleNet (2024)* | *0.319* | *0.360* | *0.389* | *0.447* | *0.379* |
|  | *TQNet (2025)* | *0.311* | *0.356* | *0.390* | *0.452* | *0.377* |
|  | *PAMNet (2026)* | ***0.308*** | ***0.353*** | ***0.375*** | ***0.425*** | ***0.365*** |
| | `patch8_cycle_once_fourier_full` | 0.321 ± 0.002 | 0.362 ± 0.004 | 0.390 ± 0.003 | 0.443 ± 0.003 | 0.379 |
| | `patch8_cycle_once_fourier_k3` | 0.323 ± 0.006 | 0.365 ± 0.002 | 0.388 ± 0.004 | 0.440 ± 0.003 | 0.379 |
| | `cross_par_l0_cycle` | 0.321 ± 0.001 | 0.364 ± 0.008 | 0.389 ± 0.008 | 0.447 ± 0.004 | 0.380 |
| | `cross_par_l0_cycle_fourier_k3` | 0.320 ± 0.004 | 0.362 ± 0.005 | 0.387 ± 0.001 | 0.445 ± 0.002 | 0.378 |
| | `cross_par_l0_cycle_pedge_fourier` | 0.313 ± 0.009 | 0.366 ± 0.029 | 0.384 ± 0.009 | 0.445 ± 0.005 | 0.377 |
| | `cross_par_l0_cycle_pedge_fourier_k3` | 0.310 ± 0.004 | 0.352 ± 0.004 | 0.385 ± 0.006 | 0.447 ± 0.007 | 0.374 |
| | `cross_par_l0_cycle_pedge2_fourier_k3` | 0.312 ± 0.004 | 0.361 ± 0.023 | 0.382 ± 0.006 | 0.442 ± 0.006 | 0.374 |
| | `cross_par_l0_cycle_pedge8_fourier_k3` | 0.314 ± 0.003 | 0.355 ± 0.013 | 0.385 ± 0.010 | 0.444 ± 0.001 | 0.375 |
| ETTm2 | *Naive (repeat last value)* | *0.266* | *0.340* | *0.412* | *0.522* | *0.385* |
|  | *PatchTST (2023)* | *0.175* | *0.241* | *0.305* | *0.402* | *0.281* |
|  | *TQNet (2025)* | *0.173* | *0.238* | *0.301* | *0.397* | *0.277* |
|  | *CycleNet (2024)* | ***0.163*** | *0.229* | ***0.284*** | *0.389* | *0.266* |
|  | *PAMNet (2026)* | *0.165* | ***0.227*** | *0.285* | ***0.380*** | ***0.264*** |
| | `patch8_cycle_once_fourier_full` | 0.164 ± 0.001 | 0.229 ± 0.000 | 0.295 ± 0.003 | 0.394 ± 0.002 | 0.270 |
| | `patch8_cycle_once_fourier_k3` | 0.167 ± 0.001 | 0.232 ± 0.001 | 0.296 ± 0.002 | 0.398 ± 0.005 | 0.273 |
| | `cross_par_l0_cycle` | 0.169 ± 0.002 | 0.242 ± 0.008 | 0.304 ± 0.004 | 0.397 ± 0.008 | 0.278 |
| | `cross_par_l0_cycle_fourier_k3` | 0.169 ± 0.001 | 0.239 ± 0.006 | 0.304 ± 0.002 | 0.396 ± 0.012 | 0.277 |
| | `cross_par_l0_cycle_pedge_fourier` | 0.177 ± 0.003 | 0.244 ± 0.003 | 0.305 ± 0.006 | 0.400 ± 0.006 | 0.281 |
| | `cross_par_l0_cycle_pedge_fourier_k3` | 0.176 ± 0.001 | 0.246 ± 0.003 | 0.310 ± 0.005 | 0.401 ± 0.008 | 0.283 |
| Weather | *Naive (repeat last value)* | *0.259* | *0.309* | *0.376* | *0.465* | *0.352* |
|  | *PatchTST (2023)* | *0.177* | *0.225* | *0.278* | *0.354* | *0.259* |
|  | *CycleNet (2024)* | *0.158* | *0.207* | *0.262* | *0.344* | *0.243* |
|  | *TimeXer (2024)* | *0.157* | *0.204* | *0.261* | ***0.340*** | *0.241* |
|  | *PAMNet (2026)* | ***0.154*** | ***0.202*** | ***0.260*** | *0.343* | ***0.240*** |
| | `patch8_cycle_once_fourier_full` | 0.158 ± 0.000 | 0.208 ± 0.001 | 0.267 ± 0.002 | 0.347 ± 0.000 | 0.245 |
| | `patch8_cycle_once_fourier_k3` | 0.157 ± 0.000 | 0.206 ± 0.000 | 0.264 ± 0.001 | 0.345 ± 0.001 | 0.243 |
| | `cross_par_l0_cycle` | 0.157 ± 0.001 | 0.210 ± 0.002 | 0.269 ± 0.002 | 0.349 ± 0.003 | 0.246 |
| | `cross_par_l0_cycle_fourier_k3` | 0.157 ± 0.001 | 0.209 ± 0.000 | 0.267 ± 0.004 | 0.349 ± 0.003 | 0.246 |
| | `cross_par_l0_cycle_pedge_fourier` | 0.159 ± 0.001 | 0.210 ± 0.002 | 0.268 ± 0.001 | 0.348 ± 0.004 | 0.246 |
| | `cross_par_l0_cycle_pedge_fourier_k3` | 0.159 ± 0.001 | 0.211 ± 0.003 | 0.268 ± 0.003 | 0.345 ± 0.003 | 0.246 |

- The phase-dependent edges stay an ETTm1 feature: on ETTm2 `cross_par_l0_cycle_pedge_fourier_k3` is worse than the same model without them (0.283 against `cross_par_l0_cycle_fourier_k3`'s 0.277, at every horizon), and on Weather it ties it (0.246). No single Cross Transformer configuration is the best on all three datasets.
- The number of harmonics of the edges matters little on ETTm1: 2, 4 and 8 give 0.374, 0.374 and 0.375. With 2 the large seed spread at 192 comes back (± 0.023), with 8 it is in between (± 0.013), so 4 stays the choice.
- Best of ours after round 20, per dataset: ETTm1 0.374 (`cross_par_l0_cycle_pedge_fourier_k3`; PAMNet 0.365, TQNet 0.377, CycleNet 0.379), ETTm2 0.270 (`patch8_cycle_once_fourier_full`; PAMNet 0.264, CycleNet 0.266), Weather 0.243 (`patch8_cycle_once_fourier_k3`; PAMNet 0.240, TimeXer 0.241, CycleNet 0.243).

## Baselines reimplemented in our pipeline (3 seeds)

PAMNet (2026) is not peer-reviewed and has no public code, and the published numbers of every reference come from other pipelines. `baselines.py` reimplements CycleNet (from the official code, github.com/ACAT-SCUT/CycleNet) and PAMNet (from the paper's equations, Algorithm 1 and Figures 2 and 8 to 10), and `train.py --architecture cyclenet_linear | cyclenet_mlp | pamnet` trains them on our splits and scores them like our models. They use their own published training settings: Adam (no weight decay), batch 256, up to 30 epochs with patience 5 on the full validation split, the CycleNet code's 'type3' decay (the full rate for 3 epochs, then x 0.8 per epoch), fp32 and no gradient clipping; CycleNet trains on MSE (lr 0.01 for Linear, 0.005 for MLP, as its scripts), PAMNet on its hybrid MAE in time and frequency.

The models are sized like ours (about 10k to 100k parameters): CycleNet/Linear is unchanged, CycleNet/MLP has 128 hidden units instead of 512, PAMNet has d = 32 instead of 512 (its per-signal amplitude carriers, 144 x 21 x 32, make it about 2.5 times our size on Weather). PAMNet choices the paper leaves open: the phase and amplitude modulators have separate weights (Figure 2), dropout comes after the product (the equation and Figure 2; Algorithm 1 differs), M_X = M_P + M_A has no residual from E_X, the output MLP is Linear(d -> d) -> SiLU -> Dropout -> Linear(d -> H), batch 256 and 'type3' decay as CycleNet. The paper's lr range [0.001, 0.005] and alpha range [0.05, 0.35] were searched on validation at horizon 96 with one seed: lr 0.005 on all three datasets, alpha 0.05 (ETTm1), 0.35 (ETTm2), 0.2 (Weather). The paper's dropout 0.5 underfits at d = 32 (ETTm1 96: validation MSE 0.391, test 0.332, worse than CycleNet/Linear), so the reference uses dropout 0.1 (validation 0.358, test 0.302).

| Dataset | Model | Parameters (H 96–720) | 96 | 192 | 336 | 720 | Avg |
|---|---|---|---|---|---|---|---|
| ETTm1 | *Naive (repeat last value)* | – | *1.214* | *1.261* | *1.287* | *1.322* | *1.271* |
|  | *CycleNet/Linear (paper)* | | | | | | *0.386* |
|  | *CycleNet/MLP (paper, hidden 512)* | | *0.319* | *0.360* | *0.389* | *0.447* | *0.379* |
|  | *PAMNet (paper, d = 512)* | | *0.308* | *0.353* | *0.375* | *0.425* | *0.365* |
| | `cyclenet_linear` | 10k–71k | 0.327 ± 0.002 | 0.366 ± 0.001 | 0.397 ± 0.001 | 0.458 ± 0.000 | 0.387 |
| | `cyclenet_mlp128` | 25k–106k | 0.317 ± 0.001 | 0.356 ± 0.001 | 0.391 ± 0.001 | 0.449 ± 0.001 | 0.378 |
| | `pamnet_d32_a0.05` | 38k–59k | 0.301 ± 0.000 | 0.350 ± 0.001 | 0.382 ± 0.001 | 0.444 ± 0.001 | 0.369 |
| | `cross_par_l0_cycle_pedge_fourier_k3` | 46k–66k | 0.310 ± 0.004 | 0.352 ± 0.004 | 0.385 ± 0.006 | 0.447 ± 0.007 | 0.374 |
| | `patch8_cycle_once_fourier_full` | 40k–60k | 0.321 ± 0.002 | 0.362 ± 0.004 | 0.390 ± 0.003 | 0.443 ± 0.003 | 0.379 |
| ETTm2 | *Naive (repeat last value)* | – | *0.266* | *0.340* | *0.412* | *0.522* | *0.385* |
|  | *CycleNet/Linear (paper)* | | | | | | *0.272* |
|  | *CycleNet/MLP (paper, hidden 512)* | | *0.163* | *0.229* | *0.284* | *0.389* | *0.266* |
|  | *PAMNet (paper, d = 512)* | | *0.165* | *0.227* | *0.285* | *0.380* | *0.264* |
| | `cyclenet_linear` | 10k–71k | 0.167 ± 0.001 | 0.233 ± 0.001 | 0.295 ± 0.002 | 0.396 ± 0.001 | 0.273 |
| | `cyclenet_mlp128` | 25k–106k | 0.163 ± 0.001 | 0.229 ± 0.002 | 0.285 ± 0.000 | 0.388 ± 0.004 | 0.266 |
| | `pamnet_d32_a0.35` | 38k–59k | 0.166 ± 0.000 | 0.229 ± 0.000 | 0.289 ± 0.002 | 0.384 ± 0.001 | 0.267 |
| | `patch8_cycle_once_fourier_full` | 40k–60k | 0.164 ± 0.001 | 0.229 ± 0.000 | 0.295 ± 0.003 | 0.394 ± 0.002 | 0.270 |
| | `cross_par_l0_cycle_fourier_full` | 46k–66k | 0.167 ± 0.001 | 0.240 ± 0.008 | 0.300 ± 0.001 | 0.392 ± 0.004 | 0.275 |
| Weather | *Naive (repeat last value)* | – | *0.259* | *0.309* | *0.376* | *0.465* | *0.352* |
|  | *CycleNet/Linear (paper)* | | | | | | *0.254* |
|  | *CycleNet/MLP (paper, hidden 512)* | | *0.158* | *0.207* | *0.262* | *0.344* | *0.243* |
|  | *PAMNet (paper, d = 512)* | | *0.154* | *0.202* | *0.260* | *0.343* | *0.240* |
| | `cyclenet_linear` | 12k–73k | 0.171 ± 0.000 | 0.223 ± 0.000 | 0.276 ± 0.000 | 0.350 ± 0.000 | 0.255 |
| | `cyclenet_mlp128` | 28k–108k | 0.161 ± 0.000 | 0.208 ± 0.000 | 0.265 ± 0.000 | 0.346 ± 0.001 | 0.245 |
| | `pamnet_d32_a0.2` | 115k–136k | 0.154 ± 0.000 | 0.202 ± 0.001 | 0.259 ± 0.000 | 0.339 ± 0.000 | 0.239 |
| | `patch8_cycle_once_fourier_k3` | 39k–59k | 0.157 ± 0.000 | 0.206 ± 0.000 | 0.264 ± 0.001 | 0.345 ± 0.001 | 0.243 |
| | `cross_par_l0_cycle_fourier_k3` | 61k–81k | 0.157 ± 0.001 | 0.209 ± 0.000 | 0.267 ± 0.004 | 0.349 ± 0.003 | 0.246 |

- The pipeline matches the literature: CycleNet/Linear, unchanged from the official code, gives 0.387, 0.273 and 0.255 against the published 0.386, 0.272 and 0.254. The published references of the earlier rounds are comparable with our numbers.
- PAMNet reproduces at a sixteenth of its width: 0.369, 0.267 and 0.239 against the published 0.365, 0.264 and 0.240 (d = 512). CycleNet/MLP with 128 hidden units equals the published 512-unit model (0.378, 0.266, 0.245 against 0.379, 0.266, 0.243).
- Our best models lose to PAMNet at the same size on every dataset: ETTm1 0.374 against 0.369, ETTm2 0.270 against 0.267 (and against CycleNet/MLP's 0.266), Weather 0.243 against 0.239 (where PAMNet is 2.5 times larger). PAMNet is ahead at every horizon on ETTm1 and Weather; on ETTm2 we are ahead at 96 (0.164 against 0.166), tie at 192 and lose at 336 and 720 (0.295 and 0.394 against 0.289 and 0.384).
- PAMNet differs from our models in two ways that the next rounds can separate: the architecture (the forecast is conditioned multiplicatively on learnable carriers indexed by the phase of the window, instead of subtracting a cycle) and the training recipe (MAE in time and frequency, Adam without weight decay, batch 256, up to 30 epochs with the full validation split, against our MSE, AdamW, batch 64, 10 epochs and a 2,048-window validation subset).


## Round 21: our best models with PAMNet's training recipe

The reimplemented PAMNet beats our best model on every dataset at the same size (0.369 / 0.267 / 0.239 against 0.374 / 0.270 / 0.243), but it also trains differently. Round 21 trains each dataset's best model with PAMNet's recipe, split in two, to see how much of the gap the training explains before any architecture is borrowed.

| Variant | Question |
|---|---|
| `_hmae` | Does PAMNet's loss, (1 - alpha) MAE + alpha MAE of the Fourier coefficients, help our model (alpha as chosen for PAMNet: 0.05 / 0.35 / 0.2)? |
| `_ptrain` | Does PAMNet's schedule help: Adam without weight decay, batch 256 instead of 64, up to 30 epochs with patience 5 instead of 10 with patience 3, the full validation split instead of 2,048 windows, 'type3' decay instead of cosine, lr 0.005 instead of 0.001? |
| `_precipe` | Do both add up? |

The base models are `cross_par_l0_cycle_pedge_fourier_k3` (ETTm1), `patch8_cycle_once_fourier_full` (ETTm2) and `patch8_cycle_once_fourier_k3` (Weather). bf16 autocast and gradient clipping stay on, as in all our runs. Horizons 96, 192, 336 and 720, input 96, 3 seeds: 108 runs. No new code.

```bash
python demo/run_benchmarks.py --datasets ETTm1 ETTm2 Weather --pred-lens 96 192 336 720   # round 21
```

### Results (test MSE, mean ± std over 3 seeds)

Italic rows are published results; `cyclenet_*` and `pamnet_*` are our reimplementations from the baselines section above, trained with their own recipes.

| Dataset | Model | Params | 96 | 192 | 336 | 720 | Avg |
|---|---|---|---|---|---|---|---|
| ETTm1 | *Naive (repeat last value)* | | *1.214* | *1.261* | *1.287* | *1.322* | *1.271* |
|  | *PatchTST (2023)* | | *0.329* | *0.367* | *0.399* | *0.454* | *0.387* |
|  | *CycleNet (2024)* | | *0.319* | *0.360* | *0.389* | *0.447* | *0.379* |
|  | *TQNet (2025)* | | *0.311* | *0.356* | *0.390* | *0.452* | *0.377* |
|  | *PAMNet (2026)* | | ***0.308*** | ***0.353*** | ***0.375*** | ***0.425*** | ***0.365*** |
| | `cyclenet_linear` | 10k–71k | 0.327 ± 0.002 | 0.366 ± 0.001 | 0.397 ± 0.001 | 0.458 ± 0.000 | 0.387 |
| | `cyclenet_mlp128` | 25k–106k | 0.317 ± 0.001 | 0.356 ± 0.001 | 0.391 ± 0.001 | 0.449 ± 0.001 | 0.378 |
| | `pamnet_d32_a0.05` | 38k–59k | 0.301 ± 0.000 | 0.350 ± 0.001 | 0.382 ± 0.001 | 0.444 ± 0.001 | 0.369 |
| | `cross_par_l0_cycle_pedge_fourier_k3` | 46k–66k | 0.310 ± 0.004 | 0.352 ± 0.004 | 0.385 ± 0.006 | 0.447 ± 0.007 | 0.374 |
| | `cross_par_l0_cycle_pedge_fourier_k3_hmae` | 46k–66k | 0.315 ± 0.003 | 0.350 ± 0.002 | 0.374 ± 0.001 | 0.440 ± 0.005 | 0.370 |
| | `cross_par_l0_cycle_pedge_fourier_k3_ptrain` | 46k–66k | 0.330 ± 0.015 | 0.383 ± 0.010 | 0.414 ± 0.013 | 0.449 ± 0.009 | 0.394 |
| | `cross_par_l0_cycle_pedge_fourier_k3_precipe` | 46k–66k | 0.307 ± 0.003 | 0.358 ± 0.003 | 0.377 ± 0.008 | 0.430 ± 0.004 | 0.368 |
| ETTm2 | *Naive (repeat last value)* | | *0.266* | *0.340* | *0.412* | *0.522* | *0.385* |
|  | *PatchTST (2023)* | | *0.175* | *0.241* | *0.305* | *0.402* | *0.281* |
|  | *TQNet (2025)* | | *0.173* | *0.238* | *0.301* | *0.397* | *0.277* |
|  | *CycleNet (2024)* | | ***0.163*** | *0.229* | ***0.284*** | *0.389* | *0.266* |
|  | *PAMNet (2026)* | | *0.165* | ***0.227*** | *0.285* | ***0.380*** | ***0.264*** |
| | `cyclenet_linear` | 10k–71k | 0.167 ± 0.001 | 0.233 ± 0.001 | 0.295 ± 0.002 | 0.396 ± 0.001 | 0.273 |
| | `cyclenet_mlp128` | 25k–106k | 0.163 ± 0.001 | 0.229 ± 0.002 | 0.285 ± 0.000 | 0.388 ± 0.004 | 0.266 |
| | `pamnet_d32_a0.35` | 38k–59k | 0.166 ± 0.000 | 0.229 ± 0.000 | 0.289 ± 0.002 | 0.384 ± 0.001 | 0.267 |
| | `patch8_cycle_once_fourier_full` | 40k–60k | 0.164 ± 0.001 | 0.229 ± 0.000 | 0.295 ± 0.003 | 0.394 ± 0.002 | 0.270 |
| | `patch8_cycle_once_fourier_full_hmae` | 40k–60k | 0.163 ± 0.001 | 0.229 ± 0.001 | 0.290 ± 0.001 | 0.389 ± 0.006 | 0.268 |
| | `patch8_cycle_once_fourier_full_ptrain` | 40k–60k | 0.165 ± 0.003 | 0.230 ± 0.001 | 0.296 ± 0.001 | 0.394 ± 0.001 | 0.271 |
| | `patch8_cycle_once_fourier_full_precipe` | 40k–60k | 0.165 ± 0.001 | 0.231 ± 0.000 | 0.290 ± 0.002 | 0.393 ± 0.002 | 0.270 |
| Weather | *Naive (repeat last value)* | | *0.259* | *0.309* | *0.376* | *0.465* | *0.352* |
|  | *PatchTST (2023)* | | *0.177* | *0.225* | *0.278* | *0.354* | *0.259* |
|  | *CycleNet (2024)* | | *0.158* | *0.207* | *0.262* | *0.344* | *0.243* |
|  | *TimeXer (2024)* | | *0.157* | *0.204* | *0.261* | ***0.340*** | *0.241* |
|  | *PAMNet (2026)* | | ***0.154*** | ***0.202*** | ***0.260*** | *0.343* | ***0.240*** |
| | `cyclenet_linear` | 12k–73k | 0.171 ± 0.000 | 0.223 ± 0.000 | 0.276 ± 0.000 | 0.350 ± 0.000 | 0.255 |
| | `cyclenet_mlp128` | 28k–108k | 0.161 ± 0.000 | 0.208 ± 0.000 | 0.265 ± 0.000 | 0.346 ± 0.001 | 0.245 |
| | `pamnet_d32_a0.2` | 115k–136k | 0.154 ± 0.000 | 0.202 ± 0.001 | 0.259 ± 0.000 | 0.339 ± 0.000 | 0.239 |
| | `patch8_cycle_once_fourier_k3` | 39k–59k | 0.157 ± 0.000 | 0.206 ± 0.000 | 0.264 ± 0.001 | 0.345 ± 0.001 | 0.243 |
| | `patch8_cycle_once_fourier_k3_hmae` | 39k–59k | 0.158 ± 0.000 | 0.209 ± 0.001 | 0.266 ± 0.000 | 0.348 ± 0.001 | 0.245 |
| | `patch8_cycle_once_fourier_k3_ptrain` | 39k–59k | 0.156 ± 0.000 | 0.206 ± 0.001 | 0.265 ± 0.002 | 0.346 ± 0.002 | 0.243 |
| | `patch8_cycle_once_fourier_k3_precipe` | 39k–59k | 0.158 ± 0.000 | 0.207 ± 0.000 | 0.265 ± 0.001 | 0.346 ± 0.000 | 0.244 |

### Conclusions

- **PAMNet's loss is most of the training gap on ETTm1 and ETTm2.** `_hmae` brings ETTm1 from 0.374 to 0.370 and ETTm2 from 0.270 to 0.268. The gain is at the long horizons (ETTm1 336: 0.385 to 0.374, 720: 0.447 to 0.440), and the seed spread drops to at most 0.004. On Weather it loses a little (0.243 to 0.245).
- **The schedule alone hurts the Cross Transformer.** `_ptrain` on ETTm1 is 0.394 with spreads up to 0.012: lr 0.005 is too high for the attention model with MSE. At horizon 96 the best validation epoch is the 1st, 2nd and 4th for the three seeds. The CI patch model on ETTm2 and Weather tolerates it (0.271, 0.243, both ties).
- **Both together give the best ETTm1 result so far.** `_precipe` is 0.368, below our reimplemented PAMNet (0.369) at a similar size (46k–66k against 38k–59k), and 0.430 at horizon 720 against PAMNet's 0.444. MAE is a gentler gradient than MSE, which seems to be what makes lr 0.005 usable. It is still above PAMNet's published 0.365.
- **ETTm2 and Weather stay behind PAMNet.** ETTm2's best is now 0.268 (`_hmae`) against 0.267; Weather stays at 0.243 against 0.239. Weather's gap is not training, and PAMNet's Weather model is about twice our size (115k–136k), because its amplitude carrier grows with the number of signals.
- **Next:** keep the hybrid MAE loss for ETTm1 and ETTm2. Try `_precipe` at lr 0.001 and 0.002, so the gain from the loss and the longer training keeps without the instability. On the architecture side, try PAMNet's multiplicative per-phase carriers in our model, aimed at Weather.

## Round 22: PAMNet's recipe at a lower learning rate

Round 21 gave the best ETTm1 result with PAMNet's whole recipe (`_precipe`, 0.368), but the schedule alone at lr 0.005 made the Cross Transformer unstable (`_ptrain`, 0.394, spreads up to 0.012). Round 22 runs `_precipe` at lr 0.001 (our usual lr) and 0.002, on the same three base models, to keep the hybrid MAE loss and the longer training without the high lr.

| Variant | Question |
|---|---|
| `<best>_precipe_lr0.001` | Does the recipe gain more at our usual lr? |
| `<best>_precipe_lr0.002` | Is a middle lr better than both? |

Horizons 96, 192, 336 and 720, input 96, 3 seeds: 72 runs. The only code change is the two variants.

```bash
python demo/run_benchmarks.py --datasets ETTm1 ETTm2 Weather --pred-lens 96 192 336 720   # round 22
```

### Results (test MSE, mean ± std over 3 seeds)

Italic rows are published results; `cyclenet_*` and `pamnet_*` are our reimplementations. `_precipe` (lr 0.005) is from round 21.

| Dataset | Model | Params | 96 | 192 | 336 | 720 | Avg |
|---|---|---|---|---|---|---|---|
| ETTm1 | *Naive (repeat last value)* | | *1.214* | *1.261* | *1.287* | *1.322* | *1.271* |
|  | *PatchTST (2023)* | | *0.329* | *0.367* | *0.399* | *0.454* | *0.387* |
|  | *CycleNet (2024)* | | *0.319* | *0.360* | *0.389* | *0.447* | *0.379* |
|  | *TQNet (2025)* | | *0.311* | *0.356* | *0.390* | *0.452* | *0.377* |
|  | *PAMNet (2026)* | | ***0.308*** | ***0.353*** | ***0.375*** | ***0.425*** | ***0.365*** |
| | `cyclenet_linear` | 10k–71k | 0.327 ± 0.002 | 0.366 ± 0.001 | 0.397 ± 0.001 | 0.458 ± 0.000 | 0.387 |
| | `cyclenet_mlp128` | 25k–106k | 0.317 ± 0.001 | 0.356 ± 0.001 | 0.391 ± 0.001 | 0.449 ± 0.001 | 0.378 |
| | `pamnet_d32_a0.05` | 38k–59k | 0.301 ± 0.000 | 0.350 ± 0.001 | 0.382 ± 0.001 | 0.444 ± 0.001 | 0.369 |
| | `cross_par_l0_cycle_pedge_fourier_k3` | 46k–66k | 0.310 ± 0.004 | 0.352 ± 0.004 | 0.385 ± 0.006 | 0.447 ± 0.007 | 0.374 |
| | `cross_par_l0_cycle_pedge_fourier_k3_precipe` | 46k–66k | 0.307 ± 0.003 | 0.358 ± 0.003 | 0.377 ± 0.008 | 0.430 ± 0.004 | 0.368 |
| | `cross_par_l0_cycle_pedge_fourier_k3_precipe_lr0.001` | 46k–66k | 0.310 ± 0.001 | 0.354 ± 0.001 | 0.383 ± 0.002 | 0.451 ± 0.009 | 0.374 |
| | `cross_par_l0_cycle_pedge_fourier_k3_precipe_lr0.002` | 46k–66k | 0.317 ± 0.005 | 0.350 ± 0.001 | 0.378 ± 0.001 | 0.443 ± 0.006 | 0.372 |
| ETTm2 | *Naive (repeat last value)* | | *0.266* | *0.340* | *0.412* | *0.522* | *0.385* |
|  | *PatchTST (2023)* | | *0.175* | *0.241* | *0.305* | *0.402* | *0.281* |
|  | *TQNet (2025)* | | *0.173* | *0.238* | *0.301* | *0.397* | *0.277* |
|  | *CycleNet (2024)* | | ***0.163*** | *0.229* | ***0.284*** | *0.389* | *0.266* |
|  | *PAMNet (2026)* | | *0.165* | ***0.227*** | *0.285* | ***0.380*** | ***0.264*** |
| | `cyclenet_linear` | 10k–71k | 0.167 ± 0.001 | 0.233 ± 0.001 | 0.295 ± 0.002 | 0.396 ± 0.001 | 0.273 |
| | `cyclenet_mlp128` | 25k–106k | 0.163 ± 0.001 | 0.229 ± 0.002 | 0.285 ± 0.000 | 0.388 ± 0.004 | 0.266 |
| | `pamnet_d32_a0.35` | 38k–59k | 0.166 ± 0.000 | 0.229 ± 0.000 | 0.289 ± 0.002 | 0.384 ± 0.001 | 0.267 |
| | `patch8_cycle_once_fourier_full` | 40k–60k | 0.164 ± 0.001 | 0.229 ± 0.000 | 0.295 ± 0.003 | 0.394 ± 0.002 | 0.270 |
| | `patch8_cycle_once_fourier_full_precipe` | 40k–60k | 0.165 ± 0.001 | 0.231 ± 0.000 | 0.290 ± 0.002 | 0.393 ± 0.002 | 0.270 |
| | `patch8_cycle_once_fourier_full_precipe_lr0.001` | 40k–60k | 0.163 ± 0.001 | 0.229 ± 0.000 | 0.290 ± 0.001 | 0.391 ± 0.003 | 0.268 |
| | `patch8_cycle_once_fourier_full_precipe_lr0.002` | 40k–60k | 0.162 ± 0.001 | 0.228 ± 0.001 | 0.288 ± 0.001 | 0.390 ± 0.003 | 0.267 |
| Weather | *Naive (repeat last value)* | | *0.259* | *0.309* | *0.376* | *0.465* | *0.352* |
|  | *PatchTST (2023)* | | *0.177* | *0.225* | *0.278* | *0.354* | *0.259* |
|  | *CycleNet (2024)* | | *0.158* | *0.207* | *0.262* | *0.344* | *0.243* |
|  | *TimeXer (2024)* | | *0.157* | *0.204* | *0.261* | ***0.340*** | *0.241* |
|  | *PAMNet (2026)* | | ***0.154*** | ***0.202*** | ***0.260*** | *0.343* | ***0.240*** |
| | `cyclenet_linear` | 12k–73k | 0.171 ± 0.000 | 0.223 ± 0.000 | 0.276 ± 0.000 | 0.350 ± 0.000 | 0.255 |
| | `cyclenet_mlp128` | 28k–108k | 0.161 ± 0.000 | 0.208 ± 0.000 | 0.265 ± 0.000 | 0.346 ± 0.001 | 0.245 |
| | `pamnet_d32_a0.2` | 115k–136k | 0.154 ± 0.000 | 0.202 ± 0.001 | 0.259 ± 0.000 | 0.339 ± 0.000 | 0.239 |
| | `patch8_cycle_once_fourier_k3` | 39k–59k | 0.157 ± 0.000 | 0.206 ± 0.000 | 0.264 ± 0.001 | 0.345 ± 0.001 | 0.243 |
| | `patch8_cycle_once_fourier_k3_precipe` | 39k–59k | 0.158 ± 0.000 | 0.207 ± 0.000 | 0.265 ± 0.001 | 0.346 ± 0.000 | 0.244 |
| | `patch8_cycle_once_fourier_k3_precipe_lr0.001` | 39k–59k | 0.160 ± 0.001 | 0.213 ± 0.001 | 0.273 ± 0.000 | 0.350 ± 0.000 | 0.249 |
| | `patch8_cycle_once_fourier_k3_precipe_lr0.002` | 39k–59k | 0.158 ± 0.000 | 0.209 ± 0.000 | 0.268 ± 0.001 | 0.348 ± 0.000 | 0.246 |

### Conclusions

- **A lower lr does not help; lr 0.005 stays best for ETTm1 and Weather.** On ETTm1 the recipe gives 0.368 at lr 0.005, 0.372 at 0.002 and 0.374 at 0.001, which is back to the base model. So the round-21 gain on ETTm1 needs the hybrid MAE loss and the high lr together. A likely reason, not checked: at lr 0.001 or 0.002 the 'type3' decay (×0.8 per epoch after 3 epochs) brings the lr down before the model has trained enough. On Weather the lower lr is worse too (0.246, 0.249 against 0.244).
- **ETTm2 ties the reimplemented PAMNet at lr 0.002.** It reaches 0.267, the same as `pamnet_d32_a0.35`, and is better at horizon 96 (0.162 against 0.166) and 192 (0.228 against 0.229). It is still behind at 336 and 720, and behind PAMNet's published 0.264.
- **Best per dataset after round 22:**

  | Dataset | Our model | Reimplemented PAMNet |
  |---|---|---|
  | ETTm1 | 0.368 (`cross_par_l0_cycle_pedge_fourier_k3_precipe`) | 0.369 |
  | ETTm2 | 0.267 (`patch8_cycle_once_fourier_full_precipe_lr0.002`) | 0.267 |
  | Weather | 0.243 (`patch8_cycle_once_fourier_k3`, our own recipe) | 0.239 |

  Training is no longer the gap on ETTm1 and ETTm2. Weather's gap needs a change to the model.

## Round 23 (paused at 39 of 192 runs for fmt run 1): a kernel that turns with the time of day, and PAMNet's carriers

After rounds 21 and 22 the training is no longer the gap to the reimplemented PAMNet on ETTm1 and ETTm2, but Weather stays behind (0.243 against 0.239). Our cycle is additive: it removes a learned daily curve from the values. PAMNet's carriers are multiplicative: the features of the window are multiplied by a vector looked up by the time of day, so the mapping from the past to the future itself can change with the time of day. Round 23 tests two ways to give our model that, side by side.

**Rotating tokenizer kernel** (`--rot-conv P`, new). The tokenizer's 8 time filters per signal are taken in pairs, and the outputs of pair j = 1..P are rotated by the angle 2πjt/W, with t the time of day of the token's last input step (as RoPE, but by the time of day instead of the position). The channel mixing after it is linear, so this is the same as a kernel that turns, within the span of the 8 filters, with the time of day. Each output is then cos(ωt)·(a₁∗x) − sin(ωt)·(a₂∗x), a product of the time of day and the content, which is the kind of term PAMNet's modulator makes. Unlike PAMNet, every token gets the phase of its own patch, and it adds no parameters. But the dependence on the time of day is limited to P harmonics, and every signal turns by the same angle. `--rot-offsets` adds a learnable angle per signal and pair (21 × 4 parameters on Weather), so the signals can be shifted in phase against each other.

**PAMNet's carriers** (`--carrier table|fourier`, new): m' = m + M_P(m, e_p[t]) + M_A(m, e_a[t, c]) on the encoder memory, with PAMNet's modulator M(x, s) = W₃ Dropout(SiLU(W₁x) ⊙ W₂s), t the phase of the last input step and c the signal. e_p is shared by the signals and e_a is one per signal. They are a table of W carriers (as PAMNet) or 3 harmonics of the day. W₃ starts at zero, so the model starts exactly as the base model, which PAMNet (without the residual) does not.

| Variant | Question |
|---|---|
| `<best>_rotconv` | Does a kernel that turns with the time of day (4 pairs, harmonics 1 to 4) help? |
| `<best>_rotconv_half` | Is it better to keep half of the filters (4 of 8) fixed, so a path that ignores the phase remains? |
| `<best>_rotconv_sig` | Does a phase offset per signal help, a cheap stand-in for PAMNet's per-signal carriers? |
| `<best>_pamod` | Do PAMNet's carriers help, with 3 harmonics (+8k to +11k parameters)? |
| `<best>_pamod_table` | With a table, as PAMNet (+31k on ETTm, +108k on Weather)? |
| `patch8_cycle_once_fourier_k3_d64` | Weather only: the base model at d_model 64, as large as `_pamod_table` (143k–183k against 147k–167k). If both gain, Weather needs capacity rather than carriers. ETTm has no width between 32 and 64 that matches. |

`<best>` is each dataset's best model and recipe: `cross_par_l0_cycle_pedge_fourier_k3_precipe` (ETTm1), `patch8_cycle_once_fourier_full_precipe_lr0.002` (ETTm2) and `patch8_cycle_once_fourier_k3` (Weather). Horizons 96 to 720, input 96, 3 seeds: 192 runs. The additive cycle and the phase-dependent edges stay.

```bash
python demo/run_benchmarks.py --datasets ETTm1 ETTm2 Weather --pred-lens 96 192 336 720   # round 23
```

# fmt runs: next-day open-to-close direction of 100 Taiwan stocks (TWStock)

## Data and target

The data are private and are not in the repository. Scripts marked "not included" ran the experiments of their section but are kept outside the repository; the models and the data pipeline in use, and the polynomial, bilinear and difference-equation comparison models, are included (`stock_model.py`, `demo/README.md`).

`dataset/` holds 100 Taiwan stocks (`info.json`, Top100) from 2020-01-02 to 2026-08-31, 1,603 to 1,618 trading days each, and 6 global series (VIX, Nasdaq, S&P 500, TAIEX, KOSPI, TAIFEX open interest). The file's `y` is 1 when that day's close is above its open (checked on every day of all 100 files; a flat day is 0). The target is the next day's `y`. The models forecast r = (close - open) / open of the next day for every stock, are trained with the MSE of r, and r̂ > 0 is read as up. They are evaluated with the score mean((2ŷ - 1) r) over all (stock, day) pairs, plus accuracy and macro F1.

- One signal per stock (100), each with 26 features: r itself, the 19 market columns and the 6 global columns (the same for every stock). Prices as logs, counts as log1p, net buys as signed log1p, MACD and ATR divided by the close, then every feature of every stock z-scored with its training statistics. r is only divided by one training std, so its sign is kept. The tokenizer of each stock reads its 26 features together, and each stock's decoder forecasts its own r (`signal_features` in `Signalformer` and `PAMNet`).
- One step per weekday, so a week is exactly 5 steps (the cycle of `--cycle-len weekly`). Days a stock did not trade carry its last values (r = 0) as inputs and no target. The global row of day d holds the US close of day d, known before the Taiwan open of day d + 1.
- Splits by the forecast day: train 2020 to 2023 (921 days), validation 2024 (242), test 2025-01 to 2026-08 (402, 40,186 stock-days). Input 60 days.

## fmt run 1: PAMNet and our best model of each dataset (3 seeds)

The best models are moved to the weekly cycle. A period of 5 has only 2 harmonics, so their 3 or 4 harmonics become 2, and the ETTm2 and Weather models become the same configuration (`fmt_weather_best` shares the results of `fmt_ettm2_best`). PAMNet trains with its own recipe (lr 0.005, 'type3', up to 30 epochs, d 32, dropout 0.1), our models with the TWStock defaults (lr 1e-3, batch 16 days, up to 10 epochs). All use the MSE of r.

| Model | Params | Test score (bp) | Test macro F1 | Test accuracy | Validation score (bp) |
|---|---|---|---|---|---|
| *Always down* | | ***11.0*** | *0.365* | *0.574* | *8.2* |
| *Always up* | | *-11.0* | *0.299* | *0.426* | *-8.2* |
| *Previous day's direction* | | *-0.9* | *0.500* | | |
| *Window-average trend (FI-2010 rule), b = 20 chosen on validation* | | *-1.9* | *0.499* | | *4.9* |
| `fmt_pamnet` | 74k | 7.7 ± 2.5 | 0.501 | 0.537 | 9.5 |
| `fmt_ettm1_best` | 259k | 8.1 ± 2.0 | 0.501 | 0.538 | 9.0 |
| `fmt_ettm2_best` (= `fmt_weather_best`) | 95k | 8.2 ± 1.2 | 0.498 | 0.541 | 10.0 |
| `fmt_fi2010_best` | 61k | 8.6 ± 0.3 | 0.507 | 0.540 | 9.5 |

1 bp = 0.01% per prediction.

- **No model beats always predicting down.** The open-to-close move of these stocks is down more often than up in every split (up share 44.5% in training, 43.4% in validation, 42.6% in test), and the score of always down grows from 5.7 bp in training to 8.2 and 11.0 bp. The models learn this bias (they predict up for only 22 to 43% of the stock-days) but add noise around it: on validation they are about level with always down (9.0 to 10.0 against 8.2), on test 2.4 to 3.3 bp below it.
- **The models are indistinguishable.** All four are within one seed spread of each other, and their validation MSE is 1.004 to 1.008 (the variance of the scaled returns is about 1), so they explain almost none of the variance. They stop after 4 to 7 epochs: the validation MSE is lowest after the first one or two.
- **Price trends carry no signal here.** Neither the previous day's direction nor the window-average trend of any length b (1 to 59 days) scores more than ±7 bp on validation or test, and the b that is best on validation is not on test. On FI-2010 that rule came close to the published models because its labels are averages of the prices in the window; this `y` shares nothing with the input window.
- The test MSE (1.45) is far above the validation MSE (1.00): 2025 to 2026 is more volatile than the training years.

```bash
python demo/run_benchmarks.py --datasets TWStock --pred-lens 1 --seq-len 60 --variants fmt_pamnet fmt_ettm1_best fmt_ettm2_best fmt_weather_best fmt_fi2010_best   # fmt run 1
```


## fmt runs 2 to 5: making the models smaller against overfitting (3 seeds)

Every run of fmt run 1 has its lowest validation MSE after one or two epochs. Runs 2 to 4 take out the encoder's self-attention blocks (`_noenc`, the tokens go through the final RMSNorm straight to the decoder), halve d_model to 16 (`_d16`, 4 heads of width 4) and cut the tokenizer's filters per feature (`_rank`: n_rank 8 to 4 or 2, FI-2010's 16 to 8 or 4), alone and together. fmt run 5 is a ladder below that on the FI-2010 and ETTm2 models, every step on top of the previous one (`FMT_CUTS`): cut1 no encoder, d 8 (2 heads), rank 4; cut2 rank 2 and no raw skip; cut3 d 4 (1 head); cut4 rank 1 and kernel 8; cut5 d 2. The ETTm2 models keep the weekly cycle, 13k of their parameters (2,600 channels). Best epoch is the epoch of the lowest validation MSE (up to 10 epochs, patience 3); rise is the validation MSE of the last epoch minus the lowest.

| Model | Params | Test score (bp) | Test macro F1 | Validation MSE | Test MSE | Best epoch (3 seeds) | Rise |
|---|---|---|---|---|---|---|---|
| *Always down* | | ***11.0*** | *0.365* | | | | |
| *Constant forecast (training mean)* | | | | ***1.0030*** | ***1.4375*** | | |
| `fmt_pamnet` | 74k | 7.7 ± 2.1 | 0.501 | 1.0060 | 1.4465 | 1, 1, 2 | 0.041 |
| `fmt_ettm1_best` | 259k | 8.1 ± 1.6 | 0.501 | 1.0066 | 1.4491 | 2, 2, 3 | 0.054 |
| `fmt_ettm1_best_noenc` | 247k | 6.0 ± 1.6 | 0.508 | 1.0072 | 1.4495 | 2, 2, 2 | 0.053 |
| `fmt_ettm1_best_rank2` | 229k | 8.1 ± 0.7 | 0.506 | 1.0065 | 1.4496 | 1, 2, 3 | 0.053 |
| `fmt_ettm1_best_d16` | 140k | 8.5 ± 0.6 | 0.495 | 1.0066 | 1.4511 | 2, 2, 4 | 0.038 |
| `fmt_ettm1_best_noenc_d16_rank2` | 116k | 7.4 ± 1.4 | 0.507 | 1.0063 | 1.4482 | 3, 3, 3 | 0.032 |
| `fmt_ettm2_best` | 95k | 8.2 ± 1.0 | 0.498 | 1.0049 | 1.4456 | 2, 1, 2 | 0.008 |
| `fmt_ettm2_best_noenc` | 82k | 8.6 ± 0.8 | 0.503 | 1.0046 | 1.4452 | 2, 4, 3 | 0.007 |
| `fmt_ettm2_best_rank4` | 74k | 8.8 ± 0.6 | 0.500 | 1.0040 | 1.4452 | 2, 1, 2 | 0.011 |
| `fmt_ettm2_best_d16` | 53k | 8.7 ± 0.5 | 0.504 | 1.0042 | 1.4454 | 3, 1, 4 | 0.003 |
| `fmt_ettm2_best_noenc_d16` | 50k | 9.0 ± 0.4 | 0.504 | 1.0045 | 1.4450 | 3, 2, 4 | 0.004 |
| `fmt_ettm2_best_noenc_d16_rank4` | 36k | 8.9 ± 0.0 | 0.500 | 1.0043 | 1.4453 | 3, 1, 4 | 0.003 |
| `fmt_ettm2_best_noenc_d16_rank2` | 29k | 8.6 ± 0.5 | 0.509 | 1.0041 | 1.4438 | 3, 3, 4 | 0.004 |
| `fmt_ettm2_best_cut1` | 27k | 8.5 ± 0.5 | 0.508 | 1.0043 | 1.4446 | 3, 3, 2 | 0.001 |
| `fmt_ettm2_best_cut2` | 20k | 8.2 ± 0.5 | 0.502 | 1.0048 | 1.4456 | 4, 1, 2 | 0.001 |
| `fmt_ettm2_best_cut3` | 18k | 7.9 ± 0.4 | 0.505 | 1.0050 | 1.4461 | 5, 6, 4 | 0.001 |
| `fmt_ettm2_best_cut4` | 15k | 7.7 ± 0.2 | 0.506 | 1.0048 | 1.4455 | 5, 3, 4 | 0.000 |
| `fmt_ettm2_best_cut5` | 14k | 5.9 ± 0.8 | 0.508 | 1.0083 | 1.4514 | 1, 7, 3 | 0.000 |
| `fmt_fi2010_best` | 61k | 8.6 ± 0.3 | 0.507 | 1.0047 | 1.4460 | 2, 1, 2 | 0.009 |
| `fmt_fi2010_best_noenc` | 49k | 6.2 ± 0.7 | 0.510 | 1.0051 | 1.4472 | 2, 4, 4 | 0.008 |
| `fmt_fi2010_best_rank8` | 51k | 9.4 ± 0.3 | 0.504 | 1.0045 | 1.4462 | 2, 4, 2 | 0.011 |
| `fmt_fi2010_best_rank4` | 46k | 9.4 ± 0.6 | 0.502 | 1.0050 | 1.4458 | 3, 1, 2 | 0.008 |
| `fmt_fi2010_best_d16` | 26k | 9.4 ± 0.7 | 0.502 | 1.0048 | 1.4463 | 3, 1, 4 | 0.003 |
| `fmt_fi2010_best_noenc_d16` | 23k | 9.6 ± 1.1 | 0.504 | 1.0051 | 1.4467 | 3, 2, 4 | 0.005 |
| `fmt_fi2010_best_noenc_d16_rank8` | 16k | 9.6 ± 0.8 | 0.503 | 1.0049 | 1.4461 | 3, 2, 4 | 0.005 |
| `fmt_fi2010_best_noenc_d16_rank4` | 12k | 8.3 ± 1.0 | 0.506 | 1.0054 | 1.4462 | 3, 4, 4 | 0.004 |
| `fmt_fi2010_best_cut1` | 5.8k | 7.8 ± 1.4 | 0.505 | 1.0058 | 1.4472 | 5, 8, 3 | 0.001 |
| `fmt_fi2010_best_cut2` | 2.9k | 8.9 ± 0.9 | 0.504 | 1.0052 | 1.4459 | 5, 9, 7 | 0.000 |
| `fmt_fi2010_best_cut3` | 1.6k | 6.0 ± 1.0 | 0.508 | 1.0076 | 1.4502 | 10, 9, 10 | 0.000 |
| `fmt_fi2010_best_cut4` | 827 | 5.5 ± 0.5 | 0.507 | 1.0090 | 1.4526 | 2, 10, 10 | 0.000 |
| `fmt_fi2010_best_cut5` | 477 | 4.9 ± 0.1 | 0.506 | 1.0101 | 1.4544 | 3, 7, 5 | 0.000 |

- **The overfitting goes away at a few thousand parameters.** d_model 16 already cuts the rise of the validation MSE by two thirds (0.009 to 0.003). From the ladder's cut1 and cut2 (FI-2010: 5.8k and 2.9k) it is gone: the best epoch moves to 5 to 9 and the validation MSE no longer rises. The ETTm1 model keeps overfitting at every size tried (rise 0.03 to 0.05), because 90k of its parameters are in the cross decoder, which none of these cuts touch.
- **Below that the models underfit and are not trained to convergence.** From cut3 (1.6k) the validation MSE is still falling at epoch 10, the limit, and both MSEs and the score get worse. fmt run 6 repeats the ladder with up to 40 epochs.
- **Removing the overfitting does not add signal.** The best scores, 9.4 to 9.6 bp, come from 16k to 51k parameters, are within one seed spread of each other and of the original models, and stay below always down (11.0 bp). No model of any size reaches the validation MSE of the constant forecast: the models only fit noise more slowly.

```bash
python demo/run_benchmarks.py --datasets TWStock --pred-lens 1 --seq-len 60 --variants fmt_ettm1_best_noenc fmt_ettm2_best_noenc fmt_fi2010_best_noenc   # fmt run 2
python demo/run_benchmarks.py --datasets TWStock --pred-lens 1 --seq-len 60 --variants fmt_ettm1_best_d16 fmt_ettm2_best_d16 fmt_fi2010_best_d16   # fmt run 3
python demo/run_benchmarks.py --datasets TWStock --pred-lens 1 --seq-len 60 --variants fmt_ettm1_best_noenc_d16 fmt_ettm2_best_noenc_d16 fmt_fi2010_best_noenc_d16 fmt_ettm1_best_rank4 fmt_ettm1_best_rank2 fmt_ettm2_best_rank4 fmt_ettm2_best_rank2 fmt_fi2010_best_rank8 fmt_fi2010_best_rank4 fmt_ettm1_best_noenc_d16_rank4 fmt_ettm1_best_noenc_d16_rank2 fmt_ettm2_best_noenc_d16_rank4 fmt_ettm2_best_noenc_d16_rank2 fmt_fi2010_best_noenc_d16_rank8 fmt_fi2010_best_noenc_d16_rank4   # fmt run 4
python demo/run_benchmarks.py --datasets TWStock --pred-lens 1 --seq-len 60 --variants fmt_fi2010_best_cut{1..5} fmt_ettm2_best_cut{1..5}   # fmt run 5
```

## fmt run 6: the ladder with up to 40 epochs (3 seeds)

The same ladder with `--epochs 40` (`_e40`, patience 3 as before).

| Model | Params | Test score (bp) | Test macro F1 | Validation MSE | Test MSE | Best epoch (3 seeds) | Rise |
|---|---|---|---|---|---|---|---|
| *Always down* | | ***11.0*** | *0.365* | | | | |
| *Constant forecast (training mean)* | | | | ***1.0030*** | ***1.4375*** | | |
| `fmt_fi2010_best_noenc_d16` (fmt run 4, 10 epochs) | 23k | 9.6 ± 1.1 | 0.504 | 1.0051 | 1.4467 | 3, 2, 4 | 0.005 |
| `fmt_fi2010_best_cut1_e40` | 5.8k | 8.5 ± 1.5 | 0.502 | 1.0055 | 1.4470 | 4, 7, 3 | 0.005 |
| `fmt_fi2010_best_cut2_e40` | 2.9k | 9.4 ± 0.3 | 0.500 | 1.0053 | 1.4461 | 6, 8, 8 | 0.006 |
| `fmt_fi2010_best_cut3_e40` | 1.6k | 9.3 ± 0.7 | 0.497 | 1.0055 | 1.4472 | 10, 8, 12 | 0.002 |
| `fmt_fi2010_best_cut4_e40` | 827 | 6.8 ± 1.6 | 0.505 | 1.0068 | 1.4496 | 2, 19, 19 | 0.001 |
| `fmt_fi2010_best_cut5_e40` | 477 | 5.0 ± 0.2 | 0.507 | 1.0097 | 1.4539 | 3, 3, 39 | 0.000 |
| `fmt_ettm2_best_cut1_e40` | 27k | 8.4 ± 0.5 | 0.509 | 1.0044 | 1.4447 | 3, 3, 2 | 0.002 |
| `fmt_ettm2_best_cut2_e40` | 20k | 7.9 ± 0.6 | 0.504 | 1.0049 | 1.4453 | 3, 1, 2 | 0.001 |
| `fmt_ettm2_best_cut3_e40` | 18k | 7.2 ± 1.1 | 0.508 | 1.0053 | 1.4465 | 4, 3, 3 | 0.001 |
| `fmt_ettm2_best_cut4_e40` | 15k | 7.0 ± 0.6 | 0.509 | 1.0050 | 1.4456 | 4, 3, 4 | 0.001 |
| `fmt_ettm2_best_cut5_e40` | 14k | 6.2 ± 0.9 | 0.507 | 1.0073 | 1.4497 | 5, 7, 3 | 0.001 |

- **Trained to convergence, 1.6k parameters do as well as 23k.** The FI-2010 model at cut3 (d 4, rank 2, no raw skip) now reaches its lowest validation MSE at epochs 8 to 12 and scores 9.3 ± 0.7 bp, level with the best of fmt runs 1 to 5. cut2 (2.9k) scores 9.4 ± 0.3 with the best epoch at 6 to 8. Every model still overfits once it has passed its best epoch (the rise of 0.002 to 0.006 is the patience of 3 epochs), but it gets there much later.
- **cut4 and below underfit.** At 827 and 477 parameters (rank 1, kernel 8, d 4 or 2) the scores drop to 5 to 7 bp even after 19 to 39 epochs.
- **On ETTm2 the weekly cycle's 13k parameters set a floor.** All five steps stay at 14k to 27k parameters, overfit after 1 to 5 epochs, and are 1 to 2 bp below the FI-2010 ladder. With 10 epochs they scored slightly better.
- **Model size is not what holds the score back.** From 1.6k to 259k parameters, every model that trains properly lands at 8 to 9.6 bp and a validation MSE of 1.004 to 1.007. None beats always down (11.0 bp) or the constant forecast's validation MSE (1.0030).

```bash
python demo/run_benchmarks.py --datasets TWStock --pred-lens 1 --seq-len 60 --variants fmt_fi2010_best_cut{1..5}_e40 fmt_ettm2_best_cut{1..5}_e40   # fmt run 6
```

## Why nothing beats always down: data analysis

Computed on the raw open and close prices and on the model inputs (`build_stock_splits`), with the same splits. bp = 0.01%.

| | Train | Validation | Test |
|---|---|---|---|
| Up share of stock-days | 44.6% | 43.4% | 42.6% |
| Mean r = minus the score of always down (bp) | -5.7 | -8.2 | -11.0 |
| Standard error of a score, clustered by day (bp) | 2.7 | 4.7 | 4.7 |
| Std of r (bp) | 200 | 201 | 240 |
| Mean overnight gap open / previous close - 1 (bp) | +16.7 | +16.1 | +32.0 |
| Share of the variance of r that is the market mean | 0.18 | 0.13 | 0.15 |
| Oracle scores: per stock / market timing only / relative to the market (bp) | 135 / 60 / 118 | 140 / 54 / 127 | 165 / 69 / 149 |
| Lag-1 autocorrelation of r | -0.037 | -0.032 | -0.013 |
| Mean z-score of log close (training statistics) | 0 | 1.64 | 2.49 |
| Stock-days with the log close above its training maximum | 0 | 38% | 56% |

- **The edge of always down is the known overnight/intraday split of returns.** These stocks gap up overnight on average (+16 to +32 bp) and give part of it back between open and close (-6 to -11 bp). The bias is real in every split but small: always down is only about 2 standard errors from zero.
- **The differences between the models are noise.** A test score has a day-clustered standard error of about 4.7 bp (402 days, the stocks move together), so 8 to 9.6 bp against 11.0 bp is within one standard error, and so are all the differences between our models in fmt runs 1 to 6.
- **What the inputs say about the next day is weak and mostly market-wide.** The strongest features are the previous US session's moves (daily change of Nasdaq and S&P 500, IC +0.06 in training and +0.08 on test, +0.01 to +0.02 in 2024) and of the TAIFEX open interest. All pooled ICs are below 0.09. Because they act on all stocks at once, an IC of about 0.1 would be needed to be 2 standard errors on 402 test days. Weekdays and lagged returns carry almost nothing (autocorrelations of -0.01 to -0.04, weekday means without a consistent sign).
- **A linear model does as well as the networks.** Ridge regression on the 26 features' last values and 1-day and 5-day changes (78 inputs, alpha 10^6 chosen on validation) has a validation R² of -0.0004 against the constant and +0.0026 on test, scores 5.8 bp on validation and 12.9 bp on test, and predicts up for 8 to 15% of the stock-days: always down with a few market-wide exceptions.
- **The price-level inputs move out of their training range.** 8 of the 26 features are log price levels (close, open, high, low, SMA, Bollinger bands), z-scored with training statistics. Over 2024 to 2026 these stocks rose so much that the log close is on average 1.6 (validation) and 2.5 (test) training standard deviations above its training mean, above the training maximum on 38% and 56% of the stock-days. Their IC flips sign between training (-0.04) and validation (+0.02). A model that leans on them in training extrapolates later, which fits the overfitting after 1 to 2 epochs.

## Minimum-variance polynomial forecasts over several windows (`demo/stock_poly.py`)

On each window of W past days (W = 5, 10, 20, 40, 60) a polynomial of degree p = 0 to 3 is fitted by least squares, the minimum-variance unbiased linear estimator of a polynomial in white noise, and extrapolated one day ahead. Every fit is a fixed linear filter. The signals:
- the open-to-close return r, extrapolated directly;
- the cumulative intraday log return, the log close, the cumulative overnight gap, and the log Nasdaq, S&P 500 and TAIEX, each giving two features: the fitted trend (next fitted value minus last fitted value) and the reversion (next fitted value minus the last observed value).

That makes 230 features per stock-day, all from data up to the previous close (the US indices up to the US session before the open). The models are ridge regressions of the next r on groups of these features, pooled over stocks, with the strength chosen on the validation MSE. As a check, single features are also used as sign rules chosen on validation. "Test - down" is the paired score difference with always down on the same stock-days, clustered by day.

| Model | Val score (bp) | Test score (bp) | Test - down (bp) | Val / test R² vs constant | Test up share |
|---|---|---|---|---|---|
| *Always down* | *8.2 ± 4.7* | *11.0 ± 4.7* | | | *0%* |
| *Constant forecast (training mean)* | | | | *0 / 0* | |
| Best network, `fmt_fi2010_best_noenc_d16` | | 9.6 ± 1.1 (seeds) | | -0.0021 / -0.0064 | |
| Ridge on the raw inputs (78: last value, 1- and 5-day change of the 26 features) | 5.8 | 12.9 | | -0.0004 / +0.0026 | 10% |
| Best single feature on validation (TAIEX reversion, W 5, p 3) | 12.5 ± 4.7 | -3.2 ± 4.7 | -14.2 ± 6.5 | | 49% |
| Ridge: r (20) | 8.1 ± 3.6 | 11.6 ± 3.7 | +0.6 ± 2.0 | +0.0021 / +0.0015 | 11% |
| Ridge: cumulative overnight gap (35) | 11.5 ± 3.4 | 11.9 ± 4.1 | +0.9 ± 3.7 | +0.0049 / +0.0066 | 19% |
| Ridge: TAIEX (35) | 8.6 ± 4.7 | 16.1 ± 4.7 | +5.1 ± 3.7 | +0.0011 / +0.0019 | 9% |
| Ridge: Nasdaq or S&P 500 (35 each) | 8.2 | 11.0 | 0 | 0 / +0.0005 | 0% |
| Ridge: own-stock signals (125) | 13.6 ± 2.8 | 14.5 ± 3.9 | +3.5 ± 3.8 | +0.0056 / +0.0083 | 25% |
| Ridge: all (230) | 11.8 ± 3.3 | 16.3 ± 4.1 | +5.3 ± 3.9 | +0.0052 / +0.0129 | 28% |
| Ridge: all, no intercept | 9.4 ± 3.2 | 15.6 ± 4.0 | +4.6 ± 4.4 | +0.0032 / +0.0128 | 39% |
| Ridge: W 10 only (46) | 16.0 ± 3.4 | 12.7 ± 3.9 | +1.7 ± 4.2 | +0.0105 / +0.0091 | 34% |
| Ridge: W 20 only (46) | 13.1 ± 3.4 | 16.3 ± 4.0 | +5.4 ± 3.9 | +0.0053 / +0.0094 | 26% |
| Ridge: p = 0 only (35, moving averages) | 10.5 ± 3.3 | 18.0 ± 4.2 | +7.1 ± 3.8 | +0.0042 / +0.0087 | 25% |
| Ridge: p = 1 only (65) | 12.8 ± 3.3 | 15.7 ± 4.3 | +4.7 ± 3.7 | +0.0095 / +0.0097 | 25% |
| Ridge: p = 2 only (65) | 10.6 ± 3.3 | 16.8 ± 3.9 | +5.8 ± 4.4 | +0.0049 / +0.0124 | 33% |

The other window and degree groups (W 5, 40, 60; p = 3) and the cumulative intraday and log-close groups land in between, at 12 to 16 bp on test.

- **These are the first models that beat the constant forecast on both validation and test.** Every ridge with own-stock signals has a positive R² in both periods (+0.002 to +0.013). Every network of fmt runs 1 to 6 was below the constant in both. They also score 12 to 18 bp on test, against 11.0 for always down and at most 9.6 for the networks, and predict up for 10 to 40% of the stock-days instead of never.
- **Against always down the gain is still within the noise.** The paired differences are +0.6 to +7.1 bp with standard errors of 2 to 4.5 bp, at most 1.9 standard errors (moving averages, p = 0). The R² improvement over the constant is more consistent than the score gain.
- **The fitting has to be pooled and shrunk: single filters do not hold up.** The best single feature on validation (12.5 bp) is negative on test, and all of the three best are at or below always down. The ridge strength chosen on validation is large (10^4 to 10^6) in nearly every group.
- **The information is in the stock's own recent intraday returns and overnight gaps, and in the TAIEX trend.** The US indices are shrunk to zero because their relation to the next day is weak in 2024, the validation year. Together with the data analysis this points to the networks' inputs: these features are stationary transforms of returns, while 8 of the 26 network inputs are price levels that leave their training range.

### The (degree k, window l) grid

One ridge model per feasible cell: every window l = 2 to 60 days, every degree k ≤ min(l - 1, 8), 503 cells (`python demo/stock_poly.py --grid`, results in `demo/runs/stock_poly_grid.csv`). Each model uses only the 7 (k = 0) or 13 features of its own fit, for all seven signals. Degrees above 8 are left out: their extrapolation filters amplify the noise by orders of magnitude (for example, the sum of |h| is 511 at k = 8, l = 9).

| Effective smoothing length l / (k + 1) (days) | Cells | Val R² | Test R² | Val score (bp) | Test score (bp) | Test - down (bp) |
|---|---|---|---|---|---|---|
| ≤ 1.5 | 28 | +0.0010 | +0.0007 | 8.2 | 9.7 | -1.2 |
| 1.5 to 2.5 | 45 | +0.0017 | +0.0021 | 8.8 | 10.0 | -0.9 |
| 2.5 to 3.5 | 45 | +0.0047 | +0.0040 | 10.6 | 8.7 | -2.3 |
| 3.5 to 5 | 70 | +0.0036 | +0.0059 | 9.1 | 10.9 | -0.1 |
| 5 to 8 | 119 | +0.0027 | +0.0057 | 9.1 | 12.5 | +1.5 |
| 8 to 12 | 76 | +0.0031 | +0.0048 | 10.4 | 13.6 | +2.7 |
| 12 to 20 | 60 | +0.0023 | +0.0038 | 9.8 | 14.4 | +3.5 |
| 20 to 60 | 60 | +0.0025 | +0.0028 | 9.7 | 14.3 | +3.4 |

| By degree (mean over l) | k = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| Test - down (bp) | +2.8 | **+3.7** | +2.8 | +2.0 | +1.1 | +0.6 | 0.0 | -0.9 | -1.6 |
| Test R² | +0.0031 | +0.0044 | +0.0047 | +0.0050 | +0.0052 | +0.0050 | +0.0044 | +0.0034 | +0.0030 |

- **Almost every cell beats the constant forecast.** 474 of the 503 cells have a positive R² on both validation and test. The improvement in MSE does not depend on a lucky choice of k and l.
- **What matters is the effective smoothing length l / (k + 1), not k and l separately.** The cells with the best R² lie on a diagonal: (k, l) = (1, 5), (2, 8), (3, 11 to 12), (4, 14), that is, about 3 days per degree of freedom on validation and 3.5 to 8 on test. Shorter fits pass too much noise, and so does a high degree on a short window. Longer fits average the information away.
- **Direction prefers longer smoothing than MSE.** The score against always down rises with the smoothing length, to +3.4 bp from about 12 days per degree of freedom on. It is best at degree 1 (local linear trend, +3.7 bp averaged over l) and turns negative from degree 7. 334 of the 503 cells score above always down on test, but none by 2 standard errors, the largest being +5.0 ± 3.6 bp (k = 1, l = 35).
- **A single cell picked on validation does not carry over to test.** The validation and test scores of the cells correlate at 0.19 (R²: 0.53). The cells best on validation, short fits with almost no shrinkage, are at or below always down on test. A whole region of the plane is a safer choice than its best cell, for example k = 1 to 2 with l = 15 to 40 days, or the multi-window ridge on all of it (16 to 18 bp in the table above).

## Bilinear least squares over signals and lags (`demo/stock_bilinear.py`)

r̂ = b₀ + Σₖ uₖᵀ Z vₖ with Z (F × L) the last L days of F = 6 stationary daily signals of the stock (open-to-close return, overnight log gap, close-to-close log return, the daily log change of Nasdaq, S&P 500 and TAIEX known at the open). uₖ weighs the signals and vₖ is a filter over the lags, either free (L values) or a degree-p Legendre polynomial (p + 1 values). With one factor fixed the model is linear in the other, so it is fitted by alternating ridge least squares, started from the leading singular pairs of the full ridge. The full ridge on vec(Z) (F L weights) is the reference. Ridge strength chosen on the validation MSE; scores in bp with standard errors clustered by day.

| Model | Params | Val score | Test score | R² val / test | Test - down |
|---|---|---|---|---|---|
| Always down | 0 | 8.2 | 11.0 | - | 0 |
| Multi-window polynomial ridge (above) | 230 | 11.8 | 16.3 | +0.0052 / +0.0129 | +5.3 ± 3.9 |
| Full ridge on the raw lags, L = 20 | 120 | 9.3 | 17.5 | +0.0005 / +0.0042 | +6.5 ± 3.4 |
| Full ridge on the raw lags, L = 60 | 360 | 9.1 | 13.2 | +0.0017 / +0.0042 | +2.3 ± 3.6 |
| Bilinear, free filters, L = 20, R = 2 | 52 | 6.2 | 18.1 | +0.0014 / +0.0101 | +7.1 ± 4.3 |
| Bilinear, cubic filters, L = 20, R = 2 | 20 | 11.5 | 16.9 | +0.0047 / +0.0070 | +5.9 ± 3.7 |
| **Bilinear, cubic filters, L = 10, R = 2** (best validation R²) | 20 | 13.4 | 13.9 | **+0.0075** / +0.0099 | +2.9 ± 4.2 |

- **Polynomial filters over the lags are what helps.** The full ridge and the free bilinear filters fit the training years much better than validation (validation R² +0.000 to +0.003). With the filters restricted to polynomials of degree 1 to 3 the validation score rises to 9 to 13 bp and the validation R² to +0.002 to +0.0075.
- **Twenty parameters do as well as 230.** The cubic rank-2 model on 10 lags has the highest validation R² of everything tried on TWStock, above the multi-window ridge, and a test R² a little below it.
- **Still not significant.** No model beats always down by 2 standard errors. The highest test scores (18.1, 18.6) belong to models that are poor on validation.
- Some cells choose a ridge strength that shrinks both factors to zero, which leaves the constant forecast (R² 0): the shrinkage applies once per factor, so it bites harder than in a single ridge.

## fmt run 7: the networks learn the residual of the polynomial ridge

The multi-window polynomial ridge (`python demo/stock_poly.py --base`, fitted on train, test 16.3 bp) is a fixed forecast; every network's output is added to it (`--residual-base`), so the network is trained on what the ridge leaves. All fmt run 1 models and the fmt run 6 ladder, 3 seeds. Scaled-return MSE (constant: val 1.0030, test 1.4375; the ridge alone: about 0.9978 and 1.419).

| Model | Params | Val MSE plain / residual | Test MSE plain / residual | Test score plain / residual (bp) |
|---|---|---|---|---|
| Always down | 0 | - | - | 11.0 |
| Ridge alone | 230 | 0.9978 | 1.419 | 16.3 |
| PAMNet | 73,537 | 1.0060 / 1.0090 | 1.4465 / 1.4399 | 7.7 / 13.9 ± 1.1 |
| ETTm1 best | 259,237 | 1.0066 / 1.0082 | 1.4491 / 1.4428 | 8.1 / 14.7 ± 0.3 |
| ETTm2 best | 94,661 | 1.0049 / 1.0066 | 1.4456 / 1.4418 | 8.2 / 14.8 ± 0.4 |
| FI-2010 best | 61,173 | 1.0047 / 1.0072 | 1.4460 / 1.4401 | 8.6 / 15.2 ± 0.1 |
| ETTm2 cut 1 to 5 | 26,741 to 14,439 | 1.0044–1.0073 / 1.0063–1.0092 | 1.4447–1.4497 / 1.4400–1.4427 | 6.2–8.4 / 14.3–15.1 |
| FI-2010 cut 1 to 5 | 5,837 to 477 | 1.0053–1.0097 / 1.0078–1.0122 | 1.4461–1.4539 / 1.4414–1.4457 | 5.0–9.5 / 14.7–15.4 |

- **Every residual network is worse than the ridge alone.** The networks add 0.008 to 0.014 to the validation MSE and 0.02 to 0.03 to the test MSE of the ridge, and lower its test score from 16.3 to 14 to 15.4 bp. What they learn on the residual does not carry over: there is no structure left that the networks can find in four training years.
- **The ridge carries the score.** With the residual every network scores 14 to 15 bp on test, against 5 to 9.6 bp on its own, and the seed spread drops to 0.1 to 1 bp. The ranking among the networks is gone; even the 477-parameter model is level with the 259k one.
- Since even their best epoch is worse than adding nothing, a network that starts at zero output (a zero-initialized output layer, and the epoch-0 model among the early-stopping candidates) would at least keep the ridge's result.

## fmt run 8: PolyConv, learnable filters in front of the polynomial fits (`polyconv.py`)

Every one of the 26 input features of a stock goes through M learnable filters of length K (depthwise, shared by the stocks, no bias). On every filtered series fixed least-squares polynomial readouts are taken (windows 5, 10, 20, 40, degrees 0 to 2; "next" = the fit one step ahead, "slope" = next minus the fit at the last step, "revert" = next minus the last value; every readout column has unit norm), and a linear head shared by the stocks, with a ridge penalty l2 |w|² (default 1, about the α/N the polynomial ridge chooses), gives the forecast. The filters start at the identity (a delta at the last tap) and are weight-normalized, or, with `--poly-sum-penalty λ`, not normalized but held near a sum of 1 by λ Σ (Σⱼ wⱼ - 1)². The head starts at zero. AdamW, lr 1e-3, up to 40 epochs, patience 3, 3 seeds. A first version with batch norm on the readouts and weight decay only was dropped: its validation MSE jumped between 1.03 and 1.5 from epoch to epoch.

| Model | Params | Val MSE | Test MSE | Val score (bp) | Test score (bp) |
|---|---|---|---|---|---|
| Constant / always down | 0 | 1.0030 | 1.4375 | 8.2 | 11.0 |
| Polynomial ridge (7 stationary signals) | 230 | 0.9978 | 1.419 | 11.8 | 16.3 |
| Fixed identity filters, all readouts | 1,041 | 1.0248 ± 0.0039 | 1.5194 | 4.8 | 8.0 ± 3.0 |
| **Fixed identity filters, slope and revert only** | 729 | **1.0023 ± 0.0008** | **1.4357** | 8.8 | 10.4 ± 0.2 |
| Learnable, K 8, M 2, all readouts | 2,081 | 1.0231 ± 0.0126 | 1.4806 | 5.9 | 3.7 ± 3.1 |
| Learnable, slope and revert only | 1,457 | 1.0046 ± 0.0026 | 1.4385 | 10.3 | 10.7 ± 0.4 |
| Learnable, K 4 / K 16 | 1,873 / 2,497 | 1.0477 / 1.0217 | 1.5968 / 1.4950 | 5.9 / 6.4 | 8.6 / 2.9 |
| Learnable, M 1 / M 4 | 1,041 / 4,161 | 1.0149 / 1.0321 | 1.5072 / 1.4808 | 6.6 / 3.2 | 6.5 / 2.9 |
| Learnable, l2 0.1 / 10 | 2,081 | 1.0270 / 1.0105 | 1.4921 / 1.4559 | 6.0 / 6.0 | 4.1 / 3.7 |
| Sum penalty λ 0.01 / 0.1 / 1 | 2,081 | 1.0575 / 1.0290 / 1.0719 | 1.6196 / 1.5404 / 1.6985 | 6.4 / 5.4 / 4.1 | 8.9 / 2.7 / 9.0 |
| Sum penalty 0.1, slope and revert only | 1,457 | 1.0049 ± 0.0021 | 1.4389 | 10.0 | 10.3 ± 0.5 |
| Residual of the ridge: learnable / fixed / sum 0.1 | 2,081 / 1,041 / 2,081 | 1.0162 / 1.0241 / 1.0365 | 1.4706 / 1.5754 / 1.4967 | 9.5 / 8.6 / 8.4 | 9.2 / 13.3 / 9.2 |

- **The "next" readouts are what breaks the model.** They keep the level of the filtered series, and for the price features the level drifts out of its training range. With only the level-free slope and reversion readouts the validation MSE drops from 1.02–1.05 to 1.002–1.005, and the fixed-filter model is the first network-trained model below the constant on both validation and test (1.0023 and 1.4357). It still scores below always down and well below the polynomial ridge, which uses 7 stationary signals instead of 26 features.
- **What the filters learn: they remove the level.** With all readouts and weight-normalized filters every learned filter keeps its last tap near 1 and turns the older taps negative and nearly equal: x_t minus a short moving average, a high-pass filter. The filter sums fall from 1 to 0.2–0.5 for the global series (VIX, KOSPI, Nasdaq, TAIFEX open interest), MACD and ATR, and stay at 0.84–0.89 for the return r, which is already stationary. With only level-free readouts there is nothing to remove and the filters stay near the identity (sums 0.8 to 1.0).
- **The sum penalty works against this.** Holding the sums at 1 keeps the filters at the identity (largest deviation 0.18, sums 0.99 to 1.0), which blocks the level removal the filters were learning, and the all-readout models get worse (validation MSE 1.029 to 1.072). With the level-free readouts the penalty changes nothing (1.0049 against 1.0046).
- **Learning the filters does not help.** At every setting the learnable filters are at or below the fixed identity with the same readouts; training stops after 1 to 7 epochs, and the filters move by at most 0.2–0.4 from the delta. Longer or more filters, and a weaker head penalty, are worse. As the residual of the ridge every variant again makes the ridge worse.
- The head weights are spread nearly evenly (|w| 0.01–0.02 per feature and filter); the largest are on TAIFEX open interest, VIX and the S&P 500, which are the same for every stock on a day.

## fmt run 9: forecast in the filtered domain, then deconvolve (`--poly-deconv`)

PolyConv with monic filters (last tap fixed at 1, the other K - 1 taps learnable from 0): the polynomial fits forecast the next filtered value ẑ, and x̂ₜ₊₁ = ẑ - Σ_{j<K-1} wⱼ xₜ₋ₖ₊₂₊ⱼ turns it back into a forecast of the series itself (as the I of ARIMA: w = [-1, 1] forecasts the change and adds it to the last value). The head reads the deconvolved forecasts of r only (`target`: 12 readouts × M filters) or of all 26 features (`all`). Same training as fmt run 8, 3 seeds.

| Model | Params | Val MSE | Test MSE | Val score (bp) | Test score (bp) | Test up |
|---|---|---|---|---|---|---|
| Constant / always down | 0 | 1.0030 | 1.4375 | 8.2 | 11.0 | 0% |
| Polynomial ridge | 230 | 0.9978 | 1.419 | 11.8 | 16.3 | 28% |
| r, identity filter (polynomial extrapolation of r) | 21 | 1.0017 ± 0.0001 | 1.4366 | 7.2 | 10.1 ± 0.5 | 18% |
| r, K 8, M 2 | 41 | 1.0019 ± 0.0003 | 1.4360 | 6.1 | 11.2 ± 0.2 | 15% |
| r, K 2 / K 16 / M 4 | 29 / 57 / 81 | 1.0016 / 1.0019 / 1.0017 | 1.4366 / 1.4366 / 1.4357 | 6.0 / 5.8 / 5.8 | 9.6 / 9.4 / 10.5 | 18–22% |
| r, sum penalty 0.1 | 41 | 1.0019 ± 0.0003 | 1.4360 | 6.0 | 11.2 ± 0.2 | 15% |
| **r, residual of the ridge** | 41 | **0.9978 ± 0.0001** | **1.4190** | 11.3 | 15.7 ± 0.1 | 29% |
| All features, identity / learnable | 521 / 1,041 | 1.0109 / 1.0115 | 1.4705 / 1.4671 | 6.3 / 5.2 | 7.5 / 5.9 | 37–44% |

- **Deconvolving r is stable and beats the constant, with 21 to 81 parameters.** Every r-only model is below the constant on validation and test, with a seed spread of 0.0001 to 0.0005 in MSE (the networks of fmt runs 1 to 8: 0.002 to 0.06). It scores at always down, 9.4 to 11.2 bp.
- **The learned filters stay at the identity.** All free taps stay within ±0.1 of 0 (sums 0.93 to 1.2) at every K and M, with no common shape across seeds, and the models equal the identity-filter control. r is already stationary, so there is nothing for the filter to remove; the sum penalty therefore changes nothing either.
- **The first residual model that does not hurt the ridge.** On top of the ridge the r-only model keeps the ridge's MSE exactly (0.9978, 1.419) and its score within noise (15.7 against 16.3): it learns to add nothing. Every network of fmt runs 7 and 8 made the ridge worse.
- **The information is not in r's own past.** The polynomial extrapolation of r alone gains 0.13% of the validation MSE, the ridge 0.52%: most of the ridge's gain comes from the gaps, the close-to-close returns and the TAIEX trend.
- **Deconvolving every feature brings the levels back.** The forecasts of the next price, volume or index are levels again, and the all-feature models are back above the constant (1.011).

## fmt run 10: normalized deconvolution filters and higher polynomial degrees

The r-only deconvolving PolyConv of fmt run 9 with three filter forms: `monic` (last tap 1, the others free, fmt run 9), `norm` (the whole filter weight-normalized, w / |w|, and the forecast divided by its last tap) and `monic_norm` (last tap 1, the other K - 1 taps normalized to unit norm, started random), each with polynomial degrees up to 2, 4 and 6 (windows 5, 10, 20, 40; degree ≤ W - 2). K 8, M 2, 3 seeds.

| Filter | Degrees ≤ 2 | ≤ 4 | ≤ 6 |
|---|---|---|---|
| monic | 1.0019 / 1.4360 / 11.2 | 1.0031 / 1.4374 / 7.5 | 1.0020 / 1.4374 / 7.4 |
| norm | 1.0019 / 1.4360 / 11.2 | 1.0031 / 1.4374 / 7.4 | 1.0020 / 1.4385 / 6.6 |
| monic_norm | 1.0025 / 1.4376 / 8.2 | 1.0033 / 1.4382 / 5.1 | 1.0056 / 1.4427 / 2.2 |

Val MSE / test MSE / test score (bp); constant 1.0030 / 1.4375 / always down 11.0. Residual of the ridge at degree ≤ 4: norm 0.9982 / 1.4203 / 16.0, monic_norm 0.9984 / 1.4202 / 15.9 (the ridge alone 0.9978 / 1.419 / 16.3). All 26 features at degree ≤ 4: norm 1.0242 / 1.4871 / 8.4, monic_norm 1.0381 / 1.5453 / -0.1.

- **The normalized filter learns to be monic.** Under `norm` the last tap stays at 0.99 to 1.0 and the others within ±0.08 of 0, the same filters as `monic` to two decimals, so the two give the same results at every degree.
- **With normalized free taps the filters do not learn.** Under `monic_norm` the free taps keep their random start (for each seed the filters are the same, to two decimals, with and without the ridge base): a unit-norm random AR part is forced on r, and it only adds noise, more so at higher degrees (validation MSE 1.0025 to 1.0056).
- **Higher degrees do not help.** Degree ≤ 2 is best for every filter; with degrees up to 4 the validation MSE is back at the constant. As in the (k, l) grid, higher-degree extrapolation passes more noise than it gains.
- On top of the ridge the degree-4 models now make it slightly worse (0.9982 against 0.9978), and the all-feature models are again above the constant.

## fmt run 11: the polynomial ridge plus an MLP on its residual (`polyconv.PolyMLP`)

The ridge forecast is added as a 27th input feature of every stock (`base_feature`), so the past residuals e = r - r̂_ridge are known up to the day before the forecast day. For every stock a two-layer MLP shared by the stocks reads the last w days of e and of the 25 other features (their values, or their day-to-day changes with `--mlp-diff`): D = 26 w inputs, Linear(D → D/2), GELU, dropout 0.2, Linear(D/2 → 1), the last layer started at zero. Its output is added to the ridge forecast. Up to 40 epochs, patience 3, 3 seeds.

| Model | Params | Val MSE | Test MSE | Val score (bp) | Test score (bp) | Best epoch |
|---|---|---|---|---|---|---|
| Constant / always down | 0 | 1.0030 | 1.4375 | 8.2 | 11.0 | - |
| Polynomial ridge alone | 230 | 0.9978 | 1.419 | 11.8 | 16.3 | - |
| w 5, values | 8,581 | 1.0021 ± 0.0014 | 1.4285 | 9.5 | 12.8 ± 1.0 | 4, 1, 1 |
| w 10, values | 34,061 | 1.0034 ± 0.0037 | 1.4349 | 9.0 | 14.2 ± 1.0 | 1, 1, 1 |
| w 20, values | 135,721 | 1.0066 ± 0.0078 | 1.4430 | 7.9 | 12.2 ± 4.0 | 2, 1, 3 |
| **w 5, changes** | 8,581 | **0.9974 ± 0.0005** | 1.4198 | 11.7 | 15.4 ± 0.7 | 1, 1, 5 |
| w 10, changes | 34,061 | 0.9985 ± 0.0000 | 1.4205 | 11.1 | 15.5 ± 0.3 | 1, 1, 1 |
| w 20, changes | 135,721 | 0.9989 ± 0.0003 | 1.4227 | 10.8 | 15.4 ± 0.4 | 1, 1, 1 |

- **With the feature changes, the first model below the ridge on validation.** w = 5 lowers the ridge's validation MSE by 0.0004 (0.9974 against 0.9978) and keeps its test MSE (1.4198 against 1.419); the score stays within noise of the ridge (15.4 against 16.3 bp). The gain is a tenth of what the ridge gains over the constant.
- **With the feature values it is worse than the ridge**, increasingly so with w (validation MSE 1.002 to 1.007): the price levels again. Differencing them is what makes the MLP usable, as the level-free readouts did for PolyConv.
- **It overfits from the second epoch on.** The best epoch is the first in 13 of the 18 runs; the training loss is then already 0.95, against a validation MSE near 1.0. Longer windows only add parameters (w 20: 136k) and do worse.

### fmt run 12: the hidden layer at a quarter of the inputs (`--mlp-shrink 4`)

| Model | Params | Val MSE | Test MSE | Test score (bp) | Best epoch |
|---|---|---|---|---|---|
| Polynomial ridge alone | 230 | 0.9978 | 1.419 | 16.3 | - |
| w 5, changes, D/2 → D/4 | 8,581 → 4,225 | 0.9974 → 0.9976 | 1.4198 → 1.4194 | 15.4 ± 0.7 → 15.9 ± 0.2 | 1, 1, 5 → 1, 1, 1 |
| w 10, changes | 34,061 → 17,031 | 0.9985 → 0.9983 | 1.4205 → 1.4199 | 15.5 → 15.4 | 1, 1, 1 → 1, 1, 1 |
| w 20, changes | 135,721 → 67,861 | 0.9989 → 0.9984 | 1.4227 → 1.4216 | 15.4 → 15.5 | 1, 1, 1 → 1, 1, 1 |
| w 5, values | 8,581 → 4,225 | 1.0021 → 1.0023 | 1.4285 → 1.4342 | 12.8 → 14.2 | 4, 1, 1 → 1, 1, 1 |

- **Halving the hidden layer again changes almost nothing.** The MSEs move by at most 0.0005 and the seed spread shrinks a little; the best models remain within 0.0004 of the ridge on validation and test. Every run but one is best after the first epoch, so the width is not what limits it: whatever generalizes is learned in one pass, and the rest is noise.

## A standalone MLP on the residuals of the polynomial fit (`demo/stock_resmlp.py`, not included)

Window of W days, fitted by least squares with a degree-p polynomial; the residual series of the window is e_n = z_n - ẑ_n (n < W) and, for the next day, e_W = r_t - ẑ_next. An MLP (Linear(D → D/4), GELU, dropout 0.2, Linear(→ 1), last layer started at zero) slides a sub-window of w residuals over the series, e_k .. e_{k+w-1} → e_{k+w}, trained on every window position of the training years times every shift (1 to 3.4 million samples), and is evaluated on the last shift only: r̂_t = ẑ_next + ê_W. Inputs: the residuals of r, or of all 6 stationary signals (each detrended in its own window); optionally the target position (k + w) / W. 3 seeds, averaged.

| Model | Val R² | Test R² | Val score | Test score |
|---|---|---|---|---|
| Always down | - | - | 8.2 | 11.0 |
| Polynomial ridge (multi-window) | +0.0052 | +0.0129 | 11.8 | 16.3 |
| Extrapolation of r only, W 20 / 40, p 1 | -0.210 / -0.092 | -0.196 / -0.098 | 0.2 / 3.2 | 1.8 / 3.4 |
| Extrapolation of r only, W 20 / 40, p 2 | -0.534 / -0.238 | -0.550 / -0.235 | -1.2 / -0.2 | -5.5 / 0.5 |
| + MLP on r, any w, with or without position | within 0.003 of the extrapolation | | | |
| + MLP on all 6, W 40, w 5, p 1 (best) | -0.080 | -0.091 | 6.3 | 6.3 |

- **The extrapolation of r itself is a poor forecast, and the MLP cannot repair it.** r is close to white noise, so extrapolating a fitted line or parabola one day ahead adds the variance of the fit (‖h_next‖² σ²): R² -0.09 to -0.55. The MLP only sees the in-window residuals (I - A A⁺) z, which under white noise are uncorrelated with the fitted part A A⁺ z that carries the extrapolation error, so it has nothing to correct it with. In 50 of the 60 runs no epoch beat the zero output.
- **With r alone the MLP is degenerate**: D = w gives 1 or 2 hidden units (8 to 27 parameters).
- **The in-window targets are not the next-day target.** The interpolation residuals inside the window have a smaller variance than the extrapolation residual e_W; the position input did not change anything.
- **The R² of the extrapolation is the white-noise value.** If r_t is uncorrelated with the window, R² = -‖h_next‖²: -0.216 / -0.554 / -0.104 / -0.249 for W 20 p 1 / W 20 p 2 / W 40 p 1 / W 40 p 2, against the measured -0.210 / -0.534 / -0.092 / -0.238.

### Second version: a shrunk base and a fixed hidden width

The base is c0 + β (ẑ_next - m), with β fitted by least squares on train; the hidden layer has 16 units; W ∈ {20, 40}, p ∈ {0, 1}, w 5, r or all 6 signals (`demo/runs/stock_resmlp_v2.log`).

- **β is zero, as the white-noise argument predicts**: -0.010 to +0.055, so the base is the constant forecast (val R² 0.0000 to +0.0006; the score is the one of always down, since the constant is negative).
- **The MLP learns nothing on top**: in all 24 runs the best epoch is 0 (the zero output). The residual-MLP line is closed: the in-window residuals of a polynomial fit carry no information on the next return.

### Autocorrelation of r (train, day-clustered standard errors in brackets)

| Lag | 1 | 2 | 3 | 4 | 5 | 10 |
|---|---|---|---|---|---|---|
| r | -0.037 (0.017) | +0.005 | +0.008 | -0.030 | +0.022 | -0.006 |
| \|r\| | +0.184 (0.018) | +0.155 | +0.122 | +0.132 | +0.120 | +0.088 |
| r² | +0.180 (0.030) | +0.146 | +0.107 | +0.130 | +0.110 | +0.065 |

r is close to uncorrelated (a small reversal at lag 1, -0.03 in train and val, -0.013 in test; Σρ² over 10 lags ≈ 0.004, the order of the R² of the ridge), but not independent: |r| and r² have a strong, slowly decaying autocorrelation (volatility clustering; weaker in val, lag 1 +0.08, and test, +0.12). The Ljung-Box test of r rejects at 5% for 45% / 22% / 26% of the stocks (train / val / test), but it assumes iid and is oversized under the clustering.

## Volatility forecasts (`demo/stock_vol.py`, not included)

Target: the Garman-Klass variance of the open-to-close move, RV = ½ ln(H/L)² - (2 ln 2 - 1) ln(C/O)² (floored at the 1% quantile). Forecasts for day t use the data up to the close of t - 1 plus the overnight gap of day t, VIX and the TAIEX change known at the open. Log-scale models are mapped back with a smearing factor. QLIKE = RV/s - ln(RV/s) - 1 (lower is better, Patton 2011); MSE of the variance ×10⁸; var z is the variance of r / √s (1 if calibrated); acf |z| is its lag-1 autocorrelation (0 if the clustering is explained).

| Model | Params | Val QLIKE | Test QLIKE | Test MSE | Test var z | Test acf \|z\| |
|---|---|---|---|---|---|---|
| Constant (pooled) | 1 | 0.721 | 0.873 | 67.9 | 1.59 | +0.235 |
| Constant per stock | 100 | 0.513 | 0.895 | 63.9 | 1.98 | +0.227 |
| EWMA (λ 0.85) | 1 | 0.348 | 0.381 | 43.9 | 1.22 | +0.053 |
| HAR (1, 5, 22 days) | 4 | 0.339 | 0.386 | 44.2 | 1.21 | +0.020 |
| HAR + overnight gap | 5 | 0.322 | 0.369 | 46.7 | 1.17 | +0.014 |
| HAR + gap + market + volume | 9 | 0.313 | **0.361** | 46.8 | 1.14 | -0.002 |
| Polynomial readouts of log RV + ridge | 26 | 0.338 | 0.414 | 44.3 | 1.27 | +0.031 |
| Polynomial readouts + gap + market + volume + ridge | 31 | **0.310** | 0.382 | 47.3 | 1.17 | +0.004 |

- **The volatility is very predictable**: QLIKE falls from 0.87 (constant) to 0.36 on test, and the clustering of |z| disappears (acf +0.235 → 0.00).
- **The long average dominates**: standardized coefficients 22-day +0.55, 5-day +0.31, 1-day +0.12; the overnight gap (+0.13) and the volume relative to its 22-day mean (+0.15) help; VIX, TAIEX and the market mean add nothing once the stock's own history is in.
- **The polynomial readouts win on validation but lose on test** (0.382 vs 0.361): the extra slope and deviation terms of HAR's structure overfit the validation year.
- **Slightly too narrow**: var z 1.14 on test, from the heavy tails and a more volatile test period.

### Weighted least squares and risk-scaled positions

The 230-feature ridge refitted with weights 1 / ŝ (HAR + gap + market + volume, mean 1 on train), α chosen on the unweighted validation MSE; "risk-scaled" is the score of positions sign(r̂) / √ŝ normalized to a mean size of 1.

| Model | Val R² | Test R² | Val score | Test score | Val risk-scaled | Test risk-scaled |
|---|---|---|---|---|---|---|
| Always down | - | - | 8.2 ± 4.7 | 11.0 ± 4.7 | | |
| Ridge, unweighted | +0.0052 | +0.0129 | 11.8 ± 3.3 | 16.3 ± 4.1 | 8.2 ± 2.5 | 11.9 ± 3.0 |
| Ridge, weights 1 / ŝ | +0.0009 | +0.0083 | 8.6 ± 3.0 | 16.6 ± 3.7 | 8.2 ± 2.3 | 11.9 ± 2.7 |

- **Weighting does not help the forecast.** The (unweighted) R² drops on both splits and the score is unchanged on test, lower on validation. The predictable part of r seems concentrated in the volatile stock-days, which the weights discount.
- **Risk scaling lowers the score in bp but also its standard error**: the ratio is about the same (val 3.6 → 3.3, test 4.0 → 4.0), so it does not improve the information ratio either.

## Quintile probabilities and volatility interactions (`demo/stock_rps.py`, not included)

Quintiles as in M6: on every forecast day the traded stocks are ranked by r and split into five equal groups. A forecast gives every stock-day the probabilities of the five quintiles; RPS = 1/5 Σ_j (F_j - C_j)² with F, C the cumulative forecast and outcome (uniform = 0.16, lower is better). The forecast distribution is r_i = c0 + κ (μ_i - c0) + σ_i^γ ε_i, with μ from a return ridge, σ the volatility forecast (HAR + gap + market + volume), ε drawn from the standardized training residuals, 1000 Monte Carlo draws of each day's cross-section; κ and γ chosen on the validation RPS. Gains are uniform minus RPS, ×10⁻⁴, day-clustered SE.

### Return ridge with interactions (v = σ / mean σ)

| Model | Features | Val R² | Test R² | Val score | Test score |
|---|---|---|---|---|---|
| Always down | - | - | - | 8.2 ± 4.7 | 11.0 ± 4.7 |
| Ridge on x | 230 | +0.0052 | +0.0129 | 11.8 ± 3.3 | 16.3 ± 4.1 |
| Ridge on x, v x | 460 | +0.0064 | +0.0113 | 11.7 ± 3.5 | 17.3 ± 4.2 |
| Ridge on x, v x, log v | 461 | +0.0065 | +0.0113 | 11.8 ± 3.6 | 16.9 ± 4.2 |

The interactions raise the validation R² (+0.0012) and lower the test R² (-0.0016); the scores move well within their errors. No evidence that the return signals act differently on volatile stock-days.

### RPS

| Forecast | κ, γ | Val RPS | Val gain | Test RPS | Test gain |
|---|---|---|---|---|---|
| Uniform | - | 0.16000 | 0 | 0.16000 | 0 |
| Volatility only, per-stock historical σ | 0, 1 | 0.15778 | +22.2 ± 1.9 | 0.15919 | +8.1 ± 1.5 |
| Volatility only, forecast σ | 0, 1 | 0.15636 | +36.4 ± 2.0 | 0.15705 | +29.5 ± 1.6 |
| Mean only, ridge on x, equal σ | 1, - | 0.15908 | +9.2 ± 2.3 | 0.15962 | +3.8 ± 2.2 |
| Ridge on x + forecast σ | 1, 1 | 0.15542 | +45.8 ± 3.1 | 0.15643 | +35.7 ± 2.4 |
| Ridge on x, v x, log v + forecast σ | 1, 1 | 0.15516 | +48.4 ± 3.3 | 0.15616 | +38.4 ± 2.7 |

- **The volatility forecast carries most of the RPS gain.** A volatile stock is more likely to end in quintile 1 or 5 and a calm one in the middle, whatever the direction; the time-varying forecast (test +29.5) is far better than the per-stock historical volatility (+8.1).
- **The mean forecast alone barely helps** (test +3.8 ± 2.2), but adds about +6 to the volatility forecast; with the interaction ridge the total is +38.4 ± 2.7 on test (RPS 0.1562, 2.4% below uniform). The paired error of the differences between rows was not computed.
- κ = 1 and γ = 1 (the forecasts as they are) are chosen on validation; the mean-only grid starts at κ = 1, so its κ is on the boundary.

## Size and calibration of the ridge forecasts (`demo/stock_mu.py`, not included)

The ridge on the 230 multi-window features (α 1e5, intercept c0 = -5.8 bp, the training mean of r).

| Split | sd of r | sd of μ | μ 1% / 50% / 99% | Share μ > 0 | Share \|μ\| > 10 / 30 / 50 bp | Slope of r on μ |
|---|---|---|---|---|---|---|
| Train | 196.5 | 19.0 | -55.3 / -4.6 / +37.0 | 0.39 | 0.57 / 0.12 / 0.02 | 1.21 ± 0.13 |
| Val | 200.7 | 17.2 | -49.8 / -2.8 / +33.7 | 0.43 | 0.53 / 0.09 / 0.01 | 0.87 ± 0.24 |
| Test | 240.2 | 24.5 | -82.4 / -11.4 / +43.6 | 0.28 | 0.65 / 0.23 / 0.07 | 1.10 ± 0.33 |

Realized mean of r (bp, day-clustered SE) in deciles of μ:

| Decile of μ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Val, mean μ | -37.0 | -21.7 | -14.6 | -9.4 | -4.9 | -0.8 | +3.1 | +7.2 | +12.6 | +23.7 |
| Val, mean r | -39.5 ± 13.7 | -18.9 ± 9.7 | -19.4 ± 8.0 | -11.6 ± 6.7 | -6.6 ± 6.1 | -4.0 ± 6.0 | -0.1 ± 5.7 | -0.5 ± 5.3 | +7.4 ± 5.6 | +10.7 ± 6.2 |
| Test, mean μ | -60.2 | -36.2 | -26.5 | -19.6 | -14.0 | -8.9 | -3.9 | +1.5 | +8.2 | +26.5 |
| Test, mean r | -52.2 ± 19.2 | -27.5 ± 8.8 | -18.5 ± 6.6 | -18.3 ± 6.6 | -8.8 ± 5.8 | -6.0 ± 5.0 | -3.2 ± 4.6 | -5.8 ± 4.8 | +1.2 ± 5.3 | +29.7 ± 15.4 |

- **The forecasts are not small and they are calibrated.** The spread of μ is about 20 bp (a tenth of the spread of r), 9 to 23% of the stock-days have |μ| above 30 bp, and the slope of the realized r on μ is 1 within its error on every split. The realized means rise almost monotonically over the deciles: decile 1 to decile 10 spans -52 to +30 bp on test, -40 to +11 on validation.
- **The R² is small because the noise is large, not because the forecast is.** For a calibrated forecast R² ≈ var(μ) / var(r) = (24.5 / 240)² ≈ 0.010 on test (measured 0.013). The ridge penalty (α chosen on the validation MSE) shrinks the directions with little data but leaves the forecast at its right scale (slope ≈ 1).
- **The sign score dilutes the forecast**: it weighs the middle deciles (|μ| of a few bp, realized means within their errors of 0) as much as the extremes. Most of the information is in the outer deciles.
- **The forecast leans down, more so on test** (72% of the test stock-days below 0), so part of the ridge's score is the drift that always down also earns.

### The sign score: decision threshold, market and cross-sectional parts, noise of a short test period

The sign score is mean((2ŷ - 1) r) with ŷ ∈ {0, 1}, the score used throughout: every stock-day gets a direction and all count equally.

- **Threshold "up iff μ > τ"**: the validation score is flat for τ from 0 to +10 bp (11.8 to 12.0) and falls off outside; τ = +10 is chosen (test 17.3 vs 16.3 at τ = 0, within the error). With a calibrated μ the Bayes rule is τ = 0, and the stock-days near the threshold carry almost nothing either way.
- **Split** (the day's mean return over the stocks m_t; market part = mean(sign · m_t), cross-sectional part = mean(sign · (r - m_t))):

| Split | Model | Score | Market part | Cross-sectional part |
|---|---|---|---|---|
| Val | Always down | 8.2 | 8.2 ± 4.7 | 0 |
| Val | Ridge | 11.8 | 4.0 ± 2.9 | 7.8 ± 1.7 |
| Test | Always down | 11.0 | 11.0 ± 4.7 | 0 |
| Test | Ridge | 16.3 | 14.0 ± 3.8 | 2.2 ± 1.2 |

  The cross-sectional part has a much smaller error, since the day's common move cancels in it: on validation it is 4.6 SE above zero, on test 1.8 SE. The market part follows the daily mean return (sd 73 bp on val, 94 bp on test) and carries most of the noise.
- **Noise of a short test period** (test days resampled):

| Period | Ridge | Always down | Ridge - always down |
|---|---|---|---|
| 42 days × 100 stocks | 16.1 ± 13.0 | 10.7 ± 14.6 | 5.1 ± 12.1 |
| 1 day × 50 stocks | 15.3 ± 82.7 | 11.6 ± 100.0 | 6.4 ± 80.6 |

  A single day is decided by the day's market move; 42 days still have an error of about 13 bp, more than twice the edge of the ridge over always down.

### One direction per day, shared by all stocks

With one sign per day the cross-sectional part is exactly 0 and the score is the market part. Day-level ridge: the day's mean return on the day's mean of the 230 features, 913 training days, α chosen on validation.

| Split | Model | Score | Market part | Cross-sectional part | vs always down | Up days |
|---|---|---|---|---|---|---|
| Val | Stock-level ridge | 11.8 ± 3.3 | 4.0 | 7.8 ± 1.7 | +3.5 ± 3.8 | 42%* |
| Val | Always down | 8.2 ± 4.7 | 8.2 | 0 | 0 | 0% |
| Val | Sign of the day's mean ridge forecast | 3.7 ± 4.7 | 3.7 | 0 | -4.6 ± 5.1 | 36% |
| Val | Day-level ridge | 8.2 ± 4.7 | 8.2 | 0 | 0 | 0% |
| Test | Stock-level ridge | 16.3 ± 4.1 | 14.0 | 2.2 ± 1.2 | +5.3 ± 3.9 | 21%* |
| Test | Always down | 11.0 ± 4.7 | 11.0 | 0 | 0 | 0% |
| Test | Sign of the day's mean ridge forecast | 16.3 ± 4.7 | 16.3 | 0 | +5.3 ± 4.3 | 17% |
| Test | Day-level ridge | 15.6 ± 4.7 | 15.6 | 0 | +4.6 ± 3.2 | 3% |

\* days on which more than half of the stocks are forecast up.

- **Sharing the direction loses the cross-sectional part** (7.8 on validation, 2.2 on test) and does not gain a reliable market part: the sign of the mean ridge forecast is worse than always down on validation (-4.6) and equal to the stock-level ridge on test.
- **The day-level ridge finds no market-timing signal**: with 913 days the validation choice is the strongest penalty, i.e. the constant (always down on every validation day, up on 3% of the test days).
- The stock-level ridge is at least as good as every shared version on both splits.

### The market forecasts judged on the day's mean return, and a fixed share of up calls

One sample per day (val 242 days, 56.2% down; test 402 days, 53.0% down):

| Split | Market forecast | Hit rate | R² | Mean sign × day return |
|---|---|---|---|---|
| Val | Constant (always down) | 0.562 | 0 | 8.2 ± 4.7 |
| Val | Day's mean ridge forecast | 0.545 | -0.0058 | 3.7 ± 4.7 |
| Val | Day-level ridge | 0.562 | +0.0007 | 8.2 ± 4.7 |
| Test | Constant (always down) | 0.530 | 0 | 11.0 ± 4.7 |
| Test | Day's mean ridge forecast | 0.547 | +0.0671 | 16.3 ± 4.7 |
| Test | Day-level ridge | 0.532 | +0.0127 | 15.6 ± 4.7 |

The market direction is not forecast reliably: on validation no forecast beats the constant; on test the day's mean ridge forecast does (R² +0.067), but the two years disagree and the standard error of a hit rate over 240 to 400 days is about 3 points.

Fixed share k of up calls per day (the k of the stocks with the highest ridge forecast of the day are called up): market part (1 - 2k) × the always-down score, cross-sectional part from the ranking.

| k | Val score | Val cross-section | Test score | Test cross-section |
|---|---|---|---|---|
| 0 (always down) | 8.2 ± 4.7 | 0 | 11.0 ± 4.7 | 0 |
| 0.1 | 10.7 ± 3.9 | 4.1 ± 0.8 | 11.7 ± 4.0 | 2.9 ± 0.9 |
| 0.2 | 11.8 ± 3.4 | 6.9 ± 1.3 | 11.5 ± 3.5 | 5.0 ± 1.3 |
| 0.3 | 12.9 ± 2.9 | 9.6 ± 1.7 | 11.8 ± 3.1 | 7.4 ± 1.8 |
| 0.4 (chosen on val) | 13.0 ± 2.4 | 11.4 ± 2.0 | 10.0 ± 2.6 | 7.8 ± 2.1 |
| 0.5 | 12.2 ± 2.2 | 12.2 ± 2.2 | 9.0 ± 2.3 | 9.0 ± 2.3 |
| Stock-level ridge (sign of μ) | 11.8 ± 3.3 | 7.8 ± 1.7 | 16.3 ± 4.1 | 2.2 ± 1.2 |

- **The ranking within the day is the robust skill of the ridge**: with half the stocks up each day (k 0.5) the cross-sectional part is 12.2 ± 2.2 on validation and 9.0 ± 2.3 on test (3.9 SE), much more than in the sign rule (7.8, 2.2), whose days with many up calls mix market guesses into it.
- **The sign score still prefers the sign rule on test** (16.3 vs 10.0 at the chosen k), because on test the ridge's market tilt paid (market part 14.0); on validation the fixed share wins (13.0 vs 11.8). Which one is better depends on whether the market tilt of the ridge holds in the evaluation period, which these two years do not settle.

### A learnable day-varying threshold

Up iff c + λ (mean_t μ - c0) + (μ - mean_t μ) > 0: λ weighs the ridge's market tilt (λ 1, c = c0: the sign rule; λ 0: a fixed threshold), c is an offset. Score (bp) on a grid of λ × c:

| λ | Val, c -10 / -5 / 0 | Test, c -15 / -10 / -5 / 0 |
|---|---|---|
| 0 | 13.5 / **13.7** / 11.9 | 13.4 / 13.2 / 10.4 / 7.9 |
| 0.5 | 13.6 / 12.8 / 10.7 | 16.7 / 16.8 / 15.7 / 13.5 |
| 1 | 11.8 / 11.8 / 9.4 | 17.2 / 16.9 / 16.1 / 15.6 |
| 2 | 8.5 / 7.9 / 7.9 | **17.2** / 16.6 / 16.1 / 16.2 |

- **The offset is stable, the market weight is not.** The best c is -5 to -15 bp on both splits, near the training mean of r (c0 = -5.8 bp): the threshold sits at the drift. The best λ is 0 on validation (ignore the ridge's market tilt) and 2 on test (double it).
- Chosen on validation (λ 0, c -5): test 10.4 ± 2.6, against 16.3 for the sign rule. One validation year cannot learn how much of the market tilt to trust; a walk-forward over more years would be needed.

## The decision threshold learned on a very large window, walk-forward (`demo/stock_threshold.py`, not included)

For every year Y from 2021 the ridge (230 features, α fixed at 1e5, chosen on 2024, a small leak into that year) is fitted on all the stock-days before Y and forecasts Y. The day-varying threshold (λ, c) and the fixed share k are learned for Y on the out-of-sample forecasts of all earlier years (expanding window) or of the previous year only. Score in bp, day-clustered SE.

| Year | Always down | Sign | Threshold, expanding window (λ, c) | Threshold, previous year (λ, c) | Fixed share, expanding (k) | Best in hindsight (λ, c) |
|---|---|---|---|---|---|---|
| 2022 | 6.7 ± 5.5 | 6.3 ± 4.1 | 5.9 ± 2.6 (0.25, -5) | 5.9 ± 2.6 (0.25, -5) | 4.3 ± 2.1 (0.40) | 0.75, -28 |
| 2023 | 3.3 ± 3.9 | 4.6 ± 2.8 | 5.6 ± 2.1 (0.50, 0) | 3.3 ± 3.8 (0.75, -28) | 6.5 ± 2.2 (0.40) | 1.50, 0 |
| 2024 | 8.2 ± 4.7 | 11.8 ± 3.3 | 12.6 ± 2.5 (0.25, -2) | 8.8 ± 3.7 (1.50, 0) | 13.0 ± 2.4 (0.40) | 0.25, -8 |
| 2025 | 9.0 ± 5.3 | 14.7 ± 4.5 | 13.2 ± 3.9 (0.25, -5) | 14.0 ± 4.1 (0.25, -8) | 9.5 ± 2.5 (0.40) | 1.25, -10 |
| 2026 (to Aug) | 13.9 ± 8.8 | 15.5 ± 7.8 | 13.6 ± 6.6 (0.25, -8) | 14.4 ± 8.0 (1.25, -10) | 11.8 ± 5.7 (0.35) | 2.00, +12 |
| Pooled 2022-2026 | 7.8 ± 2.4 | 10.2 ± 2.0 | 9.9 ± 1.5 | 8.9 ± 1.9 | 8.8 ± 1.3 | |
| vs always down | 0 | +2.4 ± 2.4 | +2.1 ± 2.2 | +1.1 ± 2.2 | +1.0 ± 1.9 | |

- **The large window makes the threshold stable**: λ settles at 0.25 (0.5 once) and c at -2 to -8 bp (near the drift c0 ≈ -6 bp), while the previous-year choice jumps (λ 0.25 → 0.75 → 1.5 → 0.25 → 1.25) and the best value in hindsight ranges over 0.25 to 2.
- **It does not raise the score**: pooled over five years it equals the sign rule (9.9 vs 10.2), with a smaller error (1.5 vs 2.0) since it keeps only a quarter of the ridge's market tilt. The previous-year threshold is worse (8.9).
- **Over five out-of-sample years no rule beats always down significantly**: the sign rule is +2.4 ± 2.4 bp. The ridge is weaker in 2022 and 2023, when it is trained on one to three years; the val/test edge (+3.5, +5.3) belongs to the later, larger training sets.

## Walk-forward ridge without a fixed α (`demo/stock_walkforward.py`)

For every year Y from 2022, α is chosen by fitting on the years before Y - 1 and taking the best MSE on Y - 1; the ridge is then refitted with it on the years before Y and forecasts Y. Every forecast and every choice uses only the past. Score in bp, day-clustered SE; "vs down" is the score minus always down.

| Year | Training stock-days | α | R² | Always down | Score | Market part | Cross-sectional part | vs down |
|---|---|---|---|---|---|---|---|---|
| 2022 | 42,796 | 1e5 | -0.0070 | 6.7 ± 5.5 | 6.3 ± 4.1 | 6.4 | -0.1 ± 1.4 | -0.3 ± 7.7 |
| 2023 | 67,368 | 1e6 | +0.0015 | 3.3 ± 3.9 | 4.3 ± 3.4 | 3.0 | +1.3 ± 1.1 | +1.0 ± 1.9 |
| 2024 | 91,267 | 1e6 | +0.0048 | 8.2 ± 4.7 | 13.1 ± 3.8 | 6.5 | +6.6 ± 1.4 | +4.9 ± 2.6 |
| 2025 | 115,463 | 1e5 | +0.0191 | 9.0 ± 5.3 | 14.7 ± 4.5 | 11.2 | +3.5 ± 1.5 | +5.7 ± 5.8 |
| 2026 (to Aug) | 139,750 | 1e5 | +0.0011 | 13.9 ± 8.8 | 15.5 ± 7.8 | 16.7 | -1.2 ± 1.7 | +1.6 ± 5.4 |
| Pooled 2022-2026 | | | +0.0040 | 7.8 ± 2.4 | 10.5 ± 2.0 | 8.2 | **+2.3 ± 0.6** | +2.6 ± 2.3 |

Training on the last L years only (α chosen the same way), pooled:

| Training window | Years evaluated | R² | Score | Cross-sectional part | vs down |
|---|---|---|---|---|---|
| Last 1 year | 2022-2026 | +0.0006 | 7.9 ± 2.1 | +0.2 ± 0.5 | +0.1 ± 2.3 |
| All earlier years | 2022-2026 | +0.0040 | 10.5 ± 2.0 | +2.3 ± 0.6 | +2.6 ± 2.3 |
| Last 2 years | 2023-2026 | +0.0011 | 9.1 ± 2.3 | +2.4 ± 0.7 | +1.0 ± 2.3 |
| Last 3 years | 2024-2026 | +0.0047 | 14.3 ± 2.9 | +4.6 ± 0.9 | +4.4 ± 2.9 |
| All earlier years | 2024-2026 | | 14.4 | +3.3 (6.6, 3.5, -1.2 per year) | |

- **Out of sample over five years the ridge's cross-sectional skill is significant** (+2.3 ± 0.6, 3.8 SE) but its total edge over always down is not (+2.6 ± 2.3): the market part carries the noise.
- **The skill needs data**: trained on one year the ridge learns nothing (cross-section +0.2, α up to 1e7); with two or more years it appears, and it grows with the training set (2022 → 2024). Three years and all earlier years are about equal on 2024-2026 (cross-section 8.0 / 3.9 / 0.6 vs 6.6 / 3.5 / -1.2), so dropping the oldest years (2020-2021) does not hurt and may help slightly.
- **The skill varies by year**: the cross-sectional part is 6.6 in 2024 and -1.2 in 2026 (to August).
- Choosing α on the previous year gives 1e5 or 1e6 (1e6 in 2023 and 2024); in 2024 that lifts the score from 11.8 (α 1e5) to 13.1.

## Degree sets, walk-forward

The same walk-forward with the multi-window features (W 5 to 60) restricted to degree sets, pooled 2022-2026; paired differences on the same stock-days, day-clustered SE.

| Features | Count | R² | Score | Cross-sectional part |
|---|---|---|---|---|
| Always down | 0 | - | 7.8 ± 2.4 | 0 |
| p = 0 only | 35 | +0.0046 | **12.0 ± 2.0** | **+3.6 ± 0.8** |
| p = 1 only | 65 | +0.0025 | 9.8 ± 2.1 | +2.4 ± 0.6 |
| p = 2 only | 65 | **+0.0055** | 11.0 ± 2.0 | +1.4 ± 0.6 |
| p = 3 only | 65 | +0.0035 | 10.1 ± 2.0 | +1.4 ± 0.6 |
| p ≤ 1 | 100 | +0.0022 | 10.3 ± 2.1 | +2.4 ± 0.6 |
| p ≤ 2 | 165 | +0.0052 | 10.4 ± 2.1 | +1.9 ± 0.6 |
| p ≤ 3 (the 230) | 230 | +0.0040 | 10.5 ± 2.0 | +2.3 ± 0.6 |
| p ≤ 5 | 334 | +0.0031 | 10.6 ± 2.0 | +2.6 ± 0.6 |

- **Higher degrees do not raise the score**: each added degree moves it by 0.1 to 0.2 bp. The 230 features rank the stocks worse than the 35 moving averages (cross-sectional part -1.3 ± 0.6).
- **Only the quadratic terms help the MSE**, on top of p ≤ 1 (+0.0030 ± 0.0011); p ≤ 2 and p = 0 are equal within the noise, and degrees 3 to 5 lower R².

## The unused data columns, walk-forward (`demo/stock_extra.py`)

Every column of the data not yet used by the return ridge, made stationary as a daily signal and turned into its last value and its moving averages over 5 to 60 days (deviations from them for log levels), in five groups: flows (net buys of foreign investors, investment trusts and the two dealer accounts as a share of the day's volume, and the change of the foreign shareholding), volume (log shares, log trades, log shares per trade), range (log high / low and the close location (2c - h - l) / (h - l)), technical (RSI, MACD, ATR, Bollinger position, SMA, last value only), global (KOSPI change, log VIX and its change, TX_OI and its change). Walk-forward as above, pooled 2022-2026 (`demo/runs/stock_extra.log`).

| Model | Features | R² | Score | Cross-sectional part |
|---|---|---|---|---|
| Always down | 0 | - | 7.8 | 0 |
| Base p = 0 | 35 | +0.0046 | 12.0 ± 2.0 | +3.6 ± 0.8 |
| Base all | 230 | +0.0040 | 10.5 ± 2.0 | +2.3 ± 0.6 |
| Flows alone | 30 | +0.0002 | 8.0 ± 2.3 | +1.1 ± 0.3 |
| Volume alone | 18 | +0.0012 | 8.5 ± 2.3 | +1.9 ± 0.4 |
| Range alone | 12 | +0.0020 | 10.0 ± 2.1 | **+5.1 ± 0.8** |
| Technical alone | 5 | +0.0004 | 6.3 ± 2.0 | +2.2 ± 0.5 |
| Global alone | 20 | -0.0012 | 8.2 ± 2.4 | 0 |
| All new alone | 85 | +0.0004 | 11.2 ± 2.2 | +3.2 ± 0.7 |
| Base all + all new | 315 | +0.0048 | 12.3 ± 2.1 | +3.3 ± 0.7 |

Paired differences to the base:

| Added group | To base p = 0: R² / score / cross-section | To base all: R² / score / cross-section |
|---|---|---|
| Flows | -0.0007 ± 0.0008 / +0.4 ± 0.4 / +0.7 ± 0.3 | +0.0001 ± 0.0001 / +0.7 ± 0.2 / **+0.8 ± 0.2** |
| Volume | -0.0004 ± 0.0005 / -0.3 ± 0.4 / +0.0 ± 0.3 | -0.0001 ± 0.0002 / +0.3 ± 0.2 / +0.2 ± 0.2 |
| Range | -0.0000 ± 0.0010 / +0.7 ± 0.5 / +0.9 ± 0.4 | +0.0005 ± 0.0002 / +0.7 ± 0.3 / +0.7 ± 0.3 |
| Technical | -0.0005 ± 0.0004 / +0.4 ± 0.2 / +0.6 ± 0.2 | +0.0000 ± 0.0001 / +0.2 ± 0.1 / +0.2 ± 0.1 |
| Global | -0.0011 ± 0.0022 / -0.6 ± 1.5 / -0.9 ± 0.7 | -0.0009 ± 0.0021 / +1.2 ± 1.0 / +0.2 ± 0.5 |
| All new | -0.0002 ± 0.0021 / +0.5 ± 1.5 / -0.0 ± 0.6 | +0.0007 ± 0.0018 / +1.9 ± 1.0 / +1.0 ± 0.5 |

- **None of the new data improves the MSE**: every R² change is within ±0.001 and within about two standard errors.
- **The price range ranks stocks best on its own**: 12 features of log high / low and close location give a cross-sectional part of +5.1 ± 0.8, above the 35 moving averages (+3.6). Added to the base they keep only +0.7 to +0.9, so most of it overlaps with the return features.
- **The institutional flows add a small, consistent ranking gain**: +0.7 to +0.8 on the cross-sectional part, 4 SE on the 230-feature base.
- **Volume and the technical indicators are redundant** with the price features (≤ +0.2 on the 230 base); the global series, the same for every stock, only add noise to the market part and lower R².
- α is chosen on the MSE, which the market part dominates; a group that ranks stocks well can be shrunk away for that reason (range alone beats every combination with the base on the cross-section).

## The cross-sectional target, walk-forward (`demo/stock_cross.py`, not included)

The ridge trained on r - mean_t r (the return minus the day's mean over the stocks) with α chosen on the previous year's MSE of that target, against the ridge on r. Three decision rules: sign(μ) (raw only); ranked: up iff μ is above its day mean (about half the stocks up); ranked + c0: up iff c0 + μ - mean_t μ > 0, with c0 the training mean of r (-5.8 bp), an always-down tilt with the ranking on top. R²x is the R² of the cross-sectional target by μ - mean_t μ. Pooled 2022-2026; demeaning the features per day as well changes nothing (≤ 0.3 bp) and is left out (`demo/runs/stock_cross.log`).

| Features | Target, rule | R²x | Up | Score | Market part | Cross-sectional part |
|---|---|---|---|---|---|---|
| - | Always down | - | 0% | 7.8 ± 2.4 | 7.8 | 0 |
| p = 0 (35) | r, sign | +0.0033 | 36% | 12.0 ± 2.0 | 8.4 | 3.6 ± 0.8 |
| | r, ranked | +0.0033 | 55% | 8.5 ± 1.1 | -0.7 | 9.2 ± 1.2 |
| | r, ranked + c0 | +0.0033 | 30% | 10.1 ± 1.5 | 2.9 | 7.1 ± 1.0 |
| | r - mean, ranked | +0.0025 | 55% | 9.0 ± 1.1 | -0.5 | 9.5 ± 1.2 |
| | r - mean, ranked + c0 | +0.0025 | 24% | 10.0 ± 1.7 | 4.4 | 5.6 ± 0.8 |
| p = 0 + range (47) | r, sign | +0.0036 | 35% | **12.7 ± 2.0** | 8.2 | 4.5 ± 0.8 |
| | r, ranked | +0.0036 | 53% | 8.9 ± 1.1 | -0.6 | 9.5 ± 1.2 |
| | r - mean, ranked | +0.0031 | 53% | 9.7 ± 1.2 | -0.5 | 10.1 ± 1.3 |
| p = 0 + range + flows (77) | r, sign | +0.0035 | 35% | 12.6 ± 2.0 | 8.2 | 4.4 ± 0.8 |
| | r, ranked | +0.0035 | 52% | 8.7 ± 1.1 | -0.4 | 9.2 ± 1.2 |
| | r - mean, ranked | +0.0036 | 52% | 9.9 ± 1.2 | -0.4 | 10.3 ± 1.3 |
| 230 | r, sign | +0.0025 | 33% | 10.5 ± 2.0 | 8.2 | 2.3 ± 0.6 |
| | r, ranked | +0.0025 | 55% | 7.8 ± 1.1 | -0.9 | 8.7 ± 1.2 |
| | r - mean, ranked | +0.0023 | 55% | 7.9 ± 1.0 | -0.7 | 8.6 ± 1.1 |
| 230 + all new (315) | r, sign | +0.0033 | 32% | 12.3 ± 2.1 | 9.1 | 3.3 ± 0.7 |
| | r - mean, ranked | +0.0030 | 54% | 9.4 ± 1.1 | -0.7 | 10.1 ± 1.2 |

- **The target does not change the ranking.** With the same ranked rule the cross-sectional target is within -0.1 to +1.1 bp of the ridge on r for every feature set (the largest, p = 0 + range + flows, +1.1 ± 0.5), and its R²x is equal or lower. The ridge on r already ranks the stocks as well; the MSE on r does not shrink the ranking features away.
- **The large cross-sectional part (3.6 → 9.2) comes from the rule, not the target.** The sign rule calls a third of the stocks up and uses the ranking only there; the ranked rule calls half of them up and uses it everywhere.
- **For the sign score the sign rule is still best**: the ranked rule drops the market part (8.4 → -0.7), which is mostly the always-down tilt, and loses 3 to 4 bp in total (-3.5 ± 1.8 for p = 0). Its standard error is about half (1.1 vs 2.0), since it does not bet on the day's direction.
- The earlier finding that range alone ranks better than base + range (+5.1 vs +4.5) is also a matter of the rule: range alone calls only 22% up; ranked, the two are 8.7 and 9.5.

## The ridge plus an MLP on its residual, walk-forward (`demo/stock_ridge_mlp.py`, not included)

Stage 1 is the ridge on the 35 p = 0 features and the range group (47 features). Stage 2 is an MLP (hidden 32, dropout 0.5) that forecasts the ridge residual r − μ from μ and, depending on the variant, from the ridge's past 20 residuals of the stock, the polynomial residuals (W 20, p 1, last 5 positions of 6 signals) and the 47 features. The ridge residuals that the MLP trains on are out of sample (leave one year out). With a bound B, the MLP output is B·sd(μ)·tanh(·). The decision rule is sign(μ + ê). Results are pooled over 2022–2026 with day-clustered standard errors.

| Model | R² | Score (bp) | Cross (bp) | Score − ridge (bp) |
|---|---|---|---|---|
| always down / constant | — | +7.8 ± 2.4 | — | — |
| ridge (p0 + range) | +0.0046 | +12.7 ± 2.0 | +4.5 ± 0.8 | 0 |
| μ only, B none / 0.5 / 1 | +.0047 / +.0056 / +.0050 | 12.9 / 12.6 / 12.8 | 4.7 / 4.3 / 4.6 | +0.2 / −0.1 / +0.1 (se 0.1–0.2) |
| μ + ridge resid, B none / 0.5 / 1 | +.0048 / +.0048 / +.0048 | 12.8 / 12.7 / 12.8 | 4.4 / 4.4 / 4.5 | +0.1 / 0.0 / +0.1 (se 0.2) |
| μ + poly resid, B none / 0.5 / 1 | +.0046 / +.0051 / +.0055 | 12.5 / 12.7 / 12.8 | 4.0 / 4.1 / 4.3 | −0.2 / 0.0 / +0.1 (se 0.3–0.5) |
| μ + both resid, B none / 0.5 / 1 | +.0050 / +.0051 / +.0051 | 12.3 / 12.5 / 12.6 | 4.0 / 4.1 / 4.4 | −0.4 / −0.2 / −0.1 (se 0.3–0.4) |
| μ + both resid + X, B none / 0.5 / 1 | +.0044 / +.0047 / +.0046 | 12.6 / 12.7 / 12.7 | 4.5 / 4.5 / 4.5 | −0.1 / 0.0 / 0.0 (se 0.1–0.2) |

The MLP flips the sign of 0.4–5% of the ridge forecasts, and no variant changes the Score by more than its standard error. Early stopping picks epoch 0 (the zero-initialized output, which is the ridge itself) in 107 of the 225 runs. The bound raises R² slightly (μ only with B = 0.5: +0.0010 ± 0.0005) but does not change the Score.

## Difference-equation fits in place of the polynomial fits, walk-forward (`demo/stock_diffeq.py`)

This reference model replaces the polynomial fits with linear difference equations of order q = 1, 2, 3, fitted in each window. The coefficients are learned by least squares (LS) or by K = 10 or 100 gradient-descent steps from zero (GD). The features are the one-step forecasts (+ coef adds the dynamic coefficients). The second stage is the same walk-forward ridge. Scores are in units of the open-to-close return.

| Model | K | R² | acc | up | Score | Cross | Score − poly p0 + range |
|---|---|---|---|---|---|---|---|
| always down | — | — | 52.8% | 0% | +7.8e-4 ± 2.4e-4 | — | — |
| poly p0 + range | 47 | +0.0046 | 53.9% | 35% | +12.7e-4 ± 2.0e-4 | +4.5e-4 | 0 |
| poly p0 | 35 | +0.0046 | 53.4% | 36% | +12.0e-4 ± 2.0e-4 | +3.6e-4 | — |
| DE LS q1 / q2 / q3 | 35 / 28 / 28 | −.0014 / −.0029 / +.0003 | 52.5–53.1% | 11–19% | +8.5 / +6.8 / +8.9e-4 | 0.2–0.7e-4 | −4.2 / −5.9 / −3.8e-4 (se ≈ 2e-4) |
| DE LS q1–3 | 91 | −0.0002 | 52.9% | 16% | +8.3e-4 ± 2.2e-4 | +0.5e-4 | −4.4e-4 ± 1.9e-4 |
| DE LS q1–3 + coef | 266 | −0.0011 | 52.7% | 15% | +7.0e-4 | +0.5e-4 | −5.7e-4 ± 1.8e-4 |
| DE LS q1–3 + range | 103 | −0.0000 | 53.4% | 24% | +9.8e-4 | +3.3e-4 | −2.9e-4 ± 1.9e-4 |
| DE GD10 q1–3 (+ range) | 91 (103) | +.0001 (+.0015) | 52.8% (53.2%) | 13% (22%) | +7.7e-4 (+8.7e-4) | 0.3e-4 (3.2e-4) | −5.0e-4 (−4.0e-4) |
| DE GD100 q1–3 (+ range) | 91 (103) | −.0012 (−.0017) | 52.7% (53.1%) | 20% (26%) | +8.3e-4 (+9.2e-4) | 1.1e-4 (3.2e-4) | −4.4e-4 (−3.5e-4) |
| poly p0 + range + DE LS q1–3 | 138 | +0.0021 | 53.6% | 29% | +10.6e-4 | +3.7e-4 | −2.1e-4 ± 1.5e-4 |

Every difference-equation set scores below the polynomial ridge, and adding the difference-equation features to it lowers R² (−0.0025 ± 0.0018). The difference-equation forecasts carry almost no cross-sectional signal (0.2–1.5e-4 against 3.6e-4 for poly p0). The forecast of a model fitted on 5–60 noisy points depends on estimated coefficients, so it has a much higher variance than a moving average. The gradient-descent shrinkage (GD10) raises R² slightly but not the Score.

## Shared and partially pooled difference equations, walk-forward (`demo/stock_diffeq_pool.py`)

This test checks whether the difference equations failed because their coefficients were estimated noisily in each window.
- **Shared coefficients:** one order-3 equation per signal, the same for every stock and day. Two-stage fits the equation by least squares on the signal's own next value and then runs the ridge. End-to-end trains the equation and the readout jointly by Adam on r.
- **Ridge on the 21 lags:** the exact solution of the same linear function class.
- **Partial pooling:** each window's equation is fitted with a prior at the pooled solution, θ = (G + κnP)⁻¹(g + κnPθ₀).

| Model | K | R² | acc | up | Score | Cross | Score − poly p0 + range |
|---|---|---|---|---|---|---|---|
| always down | — | — | 52.8% | 0% | +7.8e-4 ± 2.4e-4 | — | — |
| poly p0 + range | 47 | +0.0046 | 53.9% | 35% | +12.7e-4 ± 2.0e-4 | +4.5e-4 | 0 |
| shared DE q3, two-stage | 7 | +0.0001 | 52.4% | 18% | +6.7e-4 | −0.7e-4 | −6.0e-4 ± 1.8e-4 |
| shared DE q3, end-to-end GD | 36 | −0.0004 | 52.2% | 20% | +7.0e-4 | +0.8e-4 | −5.7e-4 ± 1.8e-4 |
| ridge on the 21 lags | 21 | +0.0015 | 52.2% | 24% | +7.8e-4 | +0.3e-4 | −4.9e-4 ± 1.8e-4 |
| shared DE two-stage + range | 19 | +0.0023 | 53.6% | 28% | +10.6e-4 | +4.0e-4 | −2.1e-4 ± 1.5e-4 |
| pooled κ 0.3 / 3 / 30 | 91 | +.0002 / −.0014 / +.0006 | 52.5% | 21–27% | +8.2 / +8.8 / +8.8e-4 | 0.1–1.4e-4 | −4.5 / −3.9 / −3.9e-4 |
| pooled κ 0.3 / 3 / 30 + range | 103 | +.0008 / −.0007 / +.0026 | 53.0–53.2% | 26–34% | +9.5 / +9.7 / +10.5e-4 | 1.9–2.9e-4 | −3.2 / −3.0 / −2.2e-4 |
| poly p0 + range + pooled κ 3 | 138 | +0.0009 | 53.2% | 31% | +10.3e-4 | +2.8e-4 | −2.4e-4 ± 1.2e-4 |

Removing the estimation variance does not help: even the exact ridge on the last three lags reaches only R² +0.0015 and no cross-sectional signal. The predictive information of the polynomial ridge is in long moving averages (5–60 days, e.g. the summed gaps and the S&P 500 trend), which an order-3 equation on the last three values cannot represent.

## Difference equations on block means, walk-forward (`demo/stock_diffeq_block.py`)

This test averages each integrated signal over blocks of m = 2, 5, 10, 20 adjacent days and differences adjacent blocks, D_m(t, k) = B_m(t − km) − B_m(t − (k+1)m) with k = 0..2.
- **Block lags:** the ridge on all D.
- **Block DE, two-stage:** one order-3 equation per (signal, m) for the next block difference, fitted by least squares before 2022; the ridge then reads its forecasts.

The sample is 153,649 stock-days (the m = 20 lags need 80 days of history), so the reference numbers differ slightly from the earlier sections.

| Model | K | R² | acc | up | Score | Cross | Score − poly p0 + range |
|---|---|---|---|---|---|---|---|
| always down | — | — | — | 0% | +7.8e-4 ± 2.4e-4 | — | — |
| poly p0 + range | 47 | +0.0053 | 53.9% | 32% | +13.1e-4 ± 2.0e-4 | +4.2e-4 | 0 |
| poly p0 | 35 | +0.0045 | 53.4% | 33% | +12.3e-4 | +3.3e-4 | −0.8e-4 ± 0.5e-4 |
| block lags m 2 / 5 / 10 / 20 | 18 each | +.0019 / +.0003 / −.0000 / +.0005 | 52.6–53.2% | 19–26% | +9.6 / +9.2 / +7.2 / +6.3e-4 | 2.4–3.4e-4 | −3.5 / −3.9 / −5.9 / −6.8e-4 |
| block lags all m | 72 | +0.0003 | 53.0% | 26% | +9.3e-4 | +2.8e-4 | −3.8e-4 ± 1.4e-4 |
| block DE two-stage all m | 24 | +0.0032 | 53.4% | 29% | +10.5e-4 | +3.4e-4 | −2.6e-4 ± 1.2e-4 |
| block DE two-stage + range | 36 | +0.0031 | 53.7% | 30% | +11.1e-4 | +4.1e-4 | −2.0e-4 ± 1.1e-4 |
| poly p0 + range + block lags | 119 | +0.0016 | 53.7% | 33% | +11.4e-4 | +3.2e-4 | −1.7e-4 ± 1.0e-4 |

Block averaging recovers most of the cross-sectional signal that the daily equations missed (2.4–4.1e-4 against 0.3e-4 for the 21 daily lags), and the two-stage block equation is the best difference-equation model so far. The ridge still trails the polynomial p0 features. The p0 "revert" feature MA_W − x_t compares the latest value with the window average, a level deviation, whereas block differences contain only changes between block means and not the position of the latest value.

## Block length × stride grid for the block difference equations, walk-forward (`demo/stock_diffeq_stride.py`)

This grid places the blocks s days apart, D(t, k) = B_m(t − ks) − B_m(t − (k+1)s), so adjacent blocks overlap by m − s days. Each cell is the two-stage order-3 block equation for one (m, s), i.e. 6 forecasts; the second number in each cell adds the range group. The sample is the same 153,649 stock-days. The reference is poly p0 + range at +13.1e-4 (R² +0.0053), and always down is at +7.8e-4.

Score (alone / + range), ×1e-4:

| m \ s | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| 2 | 7.4 / 10.9 | 9.5 / 11.2 | | | | |
| 3 | 10.5 / 12.1 | 9.4 / 11.2 | 7.3 / 9.1 | | | |
| 5 | 8.3 / 10.8 | 9.1 / 10.9 | 7.6 / 10.3 | 8.2 / 9.8 | | |
| 10 | 7.4 / 9.3 | 9.4 / 9.8 | 8.5 / 9.1 | 7.9 / 9.7 | 6.7 / 7.1 | |
| 20 | 10.5 / 9.9 | 10.5 / 9.8 | 9.3 / 8.6 | 8.6 / 8.4 | 6.7 / 9.5 | 9.4 / 9.9 |

Single cells have a Score se of about 2e-4 (paired se against the reference 1.2–1.9e-4), so most differences between cells are noise. The sets over all m at one overlap rule are clearer:

| Set | K | R² | acc | Score | Cross | Score − poly p0 + range |
|---|---|---|---|---|---|---|
| s = 1 (max overlap), all m | 30 | +0.0049 | 53.1% | +11.0e-4 | +2.6e-4 | −2.0e-4 ± 1.3e-4 |
| s = 1, all m + range | 42 | +0.0055 | 53.7% | +12.5e-4 ± 2.0e-4 | +4.3e-4 | −0.6e-4 ± 1.2e-4 (R² +0.0003 ± 0.0034) |
| s ≈ m/2, all m + range | 42 | +0.0047 | 53.6% | +11.8e-4 | +4.2e-4 | −1.3e-4 ± 1.2e-4 |
| s = m (no overlap), all m + range | 42 | +0.0031 | 53.6% | +10.5e-4 | +3.9e-4 | −2.6e-4 ± 1.2e-4 |
| all 20 pairs + range | 132 | +0.0049 | 53.9% | +12.4e-4 | +3.4e-4 | −0.7e-4 ± 1.1e-4 |

More overlap is better: with s = 1 and all block lengths, the block equations match the polynomial ridge in R² and come within 0.6e-4 ± 1.2e-4 in Score. With s = 1 the block difference is B_m(t) − B_m(t − 1) = (x_t − x_{t−m}) / m, an m-day momentum, and the lags of several m together span the same information as the nested moving averages of p0.

## High-overlap, high-order block difference equations, walk-forward (`demo/stock_diffeq_order.py`)

This test restricts the strides to s = 1, 2 and raises the order to q = 1, 2, 3, 5, 10, 20. The ridge reads the two-stage forecasts of all m = 2, 3, 5, 10, 20. The sample is 155,649 stock-days, the same as the polynomial sections; the longest lag needs 59 days.

| Set | K | R² | acc | Score | Cross | Score − poly p0 + range | R² − poly p0 + range |
|---|---|---|---|---|---|---|---|
| always down | — | — | 52.8% | +7.8e-4 ± 2.4e-4 | — | — | — |
| poly p0 + range | 47 | +0.0046 | 53.9% | +12.7e-4 ± 2.0e-4 | +4.5e-4 | 0 | 0 |
| s 1, q 1 / 2 / 3 / 5 / 10 / 20, all m | 30 | +.0060 / .0063 / .0056 / .0056 / .0062 / .0052 | 53.0–53.5% | +12.8 / 12.1 / 11.4 / 11.3 / 12.5 / 11.4e-4 | 2.3–2.9e-4 | +0.1 / −0.6 / −1.3 / −1.4 / −0.2 / −1.3e-4 (se 1.1–1.4) | +.0005 to +.0016 |
| s 1, q 1 / 2 / 3 / 5 / 10 / 20, + range | 42 | +.0071 / .0063 / .0063 / .0065 / .0070 / .0061 | 53.7–54.0% | +13.6 / 13.2 / 13.1 / 12.6 / 13.9 / 13.0e-4 | 3.8–4.4e-4 | +0.9 / +0.5 / +0.4 / −0.1 / +1.2 / +0.3e-4 (se 1.1–1.3) | +.0015 to +.0024 (se .0015–.0040) |
| s 2, q 1 / 2 / 3 / 5 / 10 / 20, + range | 42 | +.0048 / .0054 / .0041 / .0042 / .0026 / .0024 | 53.7–54.0% | +12.1 / 12.4 / 12.0 / 12.4 / 11.3 / 11.4e-4 | 4.0–4.8e-4 | −0.6 to −1.4e-4 | −.0023 to +.0008 |
| s 1, all q, all m + range | 192 | +0.0069 | 53.8% | +13.3e-4 | +3.6e-4 | +0.6e-4 ± 1.2e-4 | +.0023 ± .0030 |

With s = 1 and the range group, every order reaches R² 0.0061–0.0071, above the polynomial ridge (0.0046). The best Scores are q = 10 at +13.9e-4 and q = 1 at +13.6e-4. None of the differences exceeds about 1.6 standard errors. A higher order does not help steadily: q = 1, the plain m-day momentum with one coefficient, is as good as q = 10. Stride 2 is worse than stride 1 at every order, and it gets worse as the order grows. The block equations of the previous section (q = 3, s = 1, + range) gave R² +0.0055 on the 153,649-day sample, and here they give +0.0063. Dropping 20 early training days moved R² by 0.0008, so R² differences of this size are within the sampling noise.

## Overnight and pre-open features, walk-forward (`demo/stock_overnight.py`, data from `demo/fetch_overnight.py`)

The new data are downloaded into dataset/overnight/, which is not committed:
- **TAIFEX TX futures:** day and after-hours sessions, 2019-12 – 2026-08. An after-hours row dated D is the session from the evening of D − 1 to 05:00 of D; this was checked by comparing its open with the day closes.
- **Yahoo daily bars:** TSM/UMC/ASX ADRs, ^SOX, NQ=F, TWD=X, ^N225, ^KS11.

The features for day t, all known before 09:00:
- **TX futures:** the futures gap to the 08:45 open, the night return to 05:00, the 08:45 open versus the night close, night volume, and their products with each stock's 60-day gap beta.
- **US session:** the TSM/SOX/NQ returns and the change of the TSM ADR premium.
- **Asian opens (08:00):** the Nikkei and KOSPI open gaps.
- **Own ADR:** the stock's own ADR return for 2330, 2303 and 3711.

The base is poly p0 + range, and the common sample starts on 2020-04-10, so the base Score differs slightly from earlier sections. Correlations of the daily means with the day's mean open-to-close return over 2020–26:
- **Negative:** ADR premium change −0.20, TSM −0.09, night return −0.09, SOX −0.08, futures gap −0.06.
- **Positive:** NQ +0.07.

| Set (walk-forward 2022–26) | 100 stocks: R² | Score | Score − base | 50 stocks: Score | Score − base |
|---|---|---|---|---|---|
| always down | — | +8.6e-4 | — | +4.1e-4 | — |
| base | +0.0046 | +12.1e-4 ± 2.1e-4 | 0 | +8.8e-4 ± 1.9e-4 | 0 |
| + futures gap (08:45) | +0.0049 | +12.2e-4 | +0.2e-4 ± 0.2e-4 | +8.9e-4 | +0.0e-4 ± 0.3e-4 |
| + TX night | +0.0055 | +12.8e-4 | +0.7e-4 ± 0.7e-4 | +9.5e-4 | +0.7e-4 ± 0.7e-4 |
| + TX all + beta | +0.0057 | +12.8e-4 | +0.8e-4 ± 1.0e-4 | +9.1e-4 | +0.3e-4 ± 1.0e-4 |
| + Yahoo | +0.0081 | +14.2e-4 | +2.1e-4 ± 1.4e-4 | +10.0e-4 | +1.2e-4 ± 1.4e-4 |
| + Yahoo without Asian opens | +0.0080 | +14.5e-4 | +2.4e-4 ± 1.4e-4 | +10.5e-4 | +1.7e-4 ± 1.4e-4 |
| + all | +0.0085 | +13.4e-4 | +1.4e-4 ± 1.4e-4 | +8.9e-4 | +0.1e-4 ± 1.4e-4 |

The US-session features (mainly the ADR premium change and the TSM/SOX returns, with negative weights) raise R² the most (+0.0035 ± 0.0027). They improve the market part of the Score (+2.4e-4) rather than the stock selection. The futures gap at 08:45 adds almost nothing on top of the base. The futures night session gives a small, insignificant gain. None of the Score differences exceeds 1.7 standard errors.

## State-dependent weights and peer features, walk-forward (`demo/stock_regime_peer.py`, not included)

This section tests the linear versions of two attention designs.
- **Weights that depend on the market state:** the state is log VIX, the 20-day sd of the market's mean return and |SOX| overnight, standardized on the years before 2022. The ridge reads x and x·c; the other version fits two ridges split by the 20-day market sd (a hard gate).
- **Peer stocks:** for each stock, the mean of the base features, the last return and the gap of its 5 most correlated stocks, with the correlation of r − day mean taken over 2020–21.

The base is poly p0 + range + the US-session features (52), and the sample starts on 2020-04-10. "Ranked cross" is the cross-sectional part under the median rule.

| Model | 100 stocks: R² | Score | Score − base | R² − base | Ranked cross (− base) | 50 stocks: Score − base |
|---|---|---|---|---|---|---|
| base | +0.0087 | +14.3e-4 ± 2.1e-4 | 0 | 0 | +9.5e-4 (0) | 0 (Score +10.6e-4) |
| + x · state | +0.0015 | +10.7e-4 | −3.5e-4 ± 1.6e-4 | −0.0072 ± 0.0035 | +9.0e-4 (−0.5 ± 0.8) | −3.2e-4 ± 1.5e-4 |
| + peers | +0.0079 | +13.9e-4 | −0.4e-4 ± 0.7e-4 | −0.0008 ± 0.0015 | +9.8e-4 (+0.3 ± 0.6) | −0.4e-4 ± 0.7e-4 |
| + peers + x · state | +0.0018 | +10.6e-4 | −3.7e-4 ± 1.5e-4 | −0.0069 ± 0.0031 | +9.5e-4 (−0.0 ± 0.8) | −3.0e-4 ± 1.4e-4 |
| two experts by market sd | +0.0016 | +12.2e-4 | −2.1e-4 ± 0.9e-4 | −0.0072 ± 0.0016 | +8.7e-4 (−0.8 ± 0.5) | −1.7e-4 ± 0.9e-4 |
| two experts + peers | −0.0014 | +11.8e-4 | −2.5e-4 ± 1.1e-4 | −0.0102 ± 0.0030 | +9.0e-4 (−0.5 ± 0.8) | −2.0e-4 ± 1.0e-4 |

Weights that change with the market state make the forecast worse, significantly for R², because the extra coefficients are estimated from too few independent days. The peer averages add nothing. The correlation-based peers are noisy: for 2330 they are 2454, 2912, 3045, 2404, 3711, while for 2881 they are the other financial holding companies.

## E1: the Taiwan ridge shrunk toward a Korea + Japan pool, walk-forward (`demo/stock_pool.py`, not included, data from `demo/fetch_intl.py`)

This test builds the same 47 features (poly p0 + range, with each market's own index in place of TAIEX) for 30 KOSPI and 30 Nikkei large caps from Yahoo daily bars, 2019-12 – 2026-08. The samples are 46,708 and 47,040 stock-days, against 155,649 for Taiwan. Features and r are standardized per market. The Taiwan coefficients are shrunk toward a pooled ridge, β_TW = argmin |y − Xb|² + a|b − β_pool|², with (a_p, a) chosen on the previous year. IC is the mean daily rank correlation of μ with r.

| Model | 100 stocks: R² | Score | Cross | IC | IC − TW ridge | Score − TW ridge | 50 stocks: IC − TW ridge |
|---|---|---|---|---|---|---|---|
| always down | — | +7.8e-4 | — | — | — | — | — |
| Taiwan ridge | +0.0045 | +12.5e-4 ± 2.0e-4 | +4.2e-4 | +0.0877 | 0 | 0 | 0 |
| shrunk to KR + JP | +0.0049 | +12.0e-4 | +3.8e-4 | +0.0818 | −0.0059 ± 0.0018 | −0.4e-4 ± 0.4e-4 | −0.0054 ± 0.0021 |
| shrunk to KR + JP + TW | +0.0049 | +12.3e-4 | +4.0e-4 | +0.0858 | −0.0018 ± 0.0009 | −0.1e-4 ± 0.4e-4 | −0.0016 ± 0.0011 |
| the KR + JP pool applied to Taiwan | +0.0049 | +10.3e-4 | +0.8e-4 | +0.0242 | −0.0635 ± 0.0070 | −2.2e-4 ± 1.4e-4 | −0.0639 ± 0.0078 |

Pooling does not help. The model fitted on Korea and Japan alone carries almost none of Taiwan's cross-sectional signal (IC 0.024 against 0.088), so the coefficients are market-specific. Shrinking toward the pool lowers the IC significantly (−0.006 ± 0.002) while raising R² slightly; the R² gain comes from the market-level part.

## Chip (ownership-flow) features, walk-forward (`demo/stock_chips.py`, not included, data from `demo/fetch_chips.py`, not included)

The data come from FinMind (free tier, no token, 302 requests) and are kept in dataset/chips/, not committed. Each group's daily signals get the last value and the 5–60-day means, as in `demo/stock_extra.py`.
- **margin:** change of the margin and short balances / shares traded, short/margin ratio, offsetting trades / shares traded, log margin balance (deviation).
- **sbl:** securities-lending short sales and balance change / shares traded, log balance (deviation).
- **daytrade:** day-trading volume / shares traded.
- **futures:** foreign/trust/dealer net TX open interest and its change; foreign net TXO call and put open interest and their change.

Coverage is 99–100%. The base is poly p0 + range + US (52 features).

| Set | K | 100 stocks: R² | Score − base | R² − base | Ranked cross − base | 50 stocks: Score − base |
|---|---|---|---|---|---|---|
| base | 52 | +0.0087 | 0 (Score +14.3e-4 ± 2.1e-4) | 0 | 0 (+9.5e-4) | 0 (Score +10.6e-4) |
| + margin | 82 | +0.0086 | +0.0e-4 ± 0.3e-4 | −0.0002 ± 0.0004 | +0.0e-4 ± 0.3e-4 | −0.0e-4 ± 0.3e-4 |
| + sbl | 70 | +0.0083 | −0.1e-4 ± 0.3e-4 | −0.0004 ± 0.0003 | −0.5e-4 ± 0.3e-4 | −0.2e-4 ± 0.3e-4 |
| + daytrade | 58 | +0.0089 | +0.1e-4 ± 0.2e-4 | +0.0002 ± 0.0002 | +0.3e-4 ± 0.2e-4 | −0.0e-4 ± 0.3e-4 |
| + futures | 87 | +0.0079 | −0.1e-4 ± 1.3e-4 | −0.0008 ± 0.0019 | +0.1e-4 ± 0.3e-4 | +0.6e-4 ± 1.2e-4 |
| + all chips | 141 | +0.0085 | −0.0e-4 ± 1.2e-4 | −0.0002 ± 0.0019 | +0.1e-4 ± 0.4e-4 | +0.8e-4 ± 1.2e-4 |

No chip group adds predictive information for the next day's open-to-close return: every difference is within 1.7 standard errors, most within one.

## Recurrent corrections of the ridge residual: linear recursions, PID filters, GRU and LSTM, walk-forward (`demo/stock_recurrent.py`)

Base: the ridge on poly p0 + range + the US-session features (52 features). Stage 2 forecasts the ridge's out-of-sample residual e_t = r_t - mu_t (on the training years mu comes from the ridge fitted without that year) from what is known before day t, and the forecast is mu_t + e_hat_t:

- AR(p): ridge on e_{t-1} .. e_{t-p} of the stock;
- EMA: ridge on 6 exponential moving averages E_rho(t) = rho E_rho(t-1) + (1 - rho) e_{t-1}, rho = 0.5 .. 0.99;
- PID: e_hat_t = Kp e_{t-1} + Ki I_rho(t) + Kd (e_{t-1} - e_{t-2}), I_rho the leaky integral of the past residuals; gains by least squares, rho and alpha chosen on Y - 1; on the own residual, and on the own residual plus the same terms of the day's mean residual;
- GRU, LSTM: one layer, 32 units, shared by the stocks, 60 days of (e_{t-1}, mu_t, traded flag), or those plus the 52 features; head started at zero; early stopping on Y - 1; 3 seeds.

Autocorrelation of e on 2022-2026 (pooled): own lag 1 +0.017, lag 2 -0.000, lag 5 +0.005, lag 20 -0.008; day-mean residual lag 1 +0.046, lag 2 -0.025; e minus the day mean lag 1 +0.012.

Walk-forward 2022-2026, differences paired with the ridge and day-clustered (1e-4 units):

| Model | R² (100) | Score (100) | Cross (100) | Score − ridge (100) | R² − ridge (100) | Score (50) | Score − ridge (50) |
|---|---|---|---|---|---|---|---|
| *Always down* | | *+7.8 ± 2.4* | | | | *+3.4 ± 2.3* | |
| Ridge (base) | +0.0087 | +14.3 ± 2.1 | +2.7 ± 0.7 | 0 | 0 | +10.6 ± 1.9 | 0 |
| AR(5) | +0.0086 | +14.1 ± 2.1 | +2.2 ± 0.7 | −0.2 ± 0.3 | −0.0001 ± 0.0002 | +10.1 ± 2.0 | −0.5 ± 0.3 |
| AR(20) | +0.0087 | +14.0 ± 2.1 | +2.1 ± 0.7 | −0.2 ± 0.3 | −0.0000 ± 0.0003 | +9.9 ± 1.9 | −0.7 ± 0.3 |
| EMA (6 rho) | +0.0087 | +14.3 ± 2.1 | +2.4 ± 0.7 | +0.1 ± 0.3 | −0.0000 ± 0.0003 | +10.3 ± 2.0 | −0.3 ± 0.4 |
| PID own | +0.0085 | +14.3 ± 2.1 | +2.4 ± 0.7 | +0.0 ± 0.3 | −0.0002 ± 0.0003 | +10.3 ± 1.9 | −0.4 ± 0.4 |
| PID own + market | +0.0087 | +14.2 ± 2.1 | +2.3 ± 0.7 | −0.1 ± 0.3 | −0.0000 ± 0.0002 | +10.0 ± 2.0 | −0.6 ± 0.3 |
| GRU resid | +0.0091 | +14.5 ± 2.1 | +2.7 ± 0.7 | +0.2 ± 0.1 | +0.0004 ± 0.0004 | +10.9 ± 1.9 | +0.3 ± 0.2 |
| GRU resid + X | +0.0087 | +14.3 ± 2.1 | +2.7 ± 0.7 | +0.0 ± 0.0 | +0.0000 ± 0.0000 | +10.6 ± 1.9 | +0.0 ± 0.0 |
| LSTM resid | +0.0090 | +14.5 ± 2.1 | +2.8 ± 0.7 | +0.2 ± 0.1 | +0.0003 ± 0.0003 | +10.9 ± 1.9 | +0.3 ± 0.1 |
| LSTM resid + X | +0.0087 | +14.3 ± 2.1 | +2.7 ± 0.7 | +0.0 ± 0.0 | +0.0000 ± 0.0000 | +10.6 ± 1.9 | +0.0 ± 0.0 |

- **The ridge residual has almost no memory.** Its own lag-1 autocorrelation is 0.017; the largest, 0.046 for the day-mean residual, is a market-level effect worth R² ≈ 0.002 on that one series. The linear recursions (AR, EMA, PID) flip 4 to 5% of the signs and lose 0.2 to 0.7e-4 on the 50 stocks.
- **The recurrent nets learn nothing that generalizes.** In 2022 - 2025 every seed of every GRU and LSTM keeps its initial state (best epoch 0: the head starts at zero, so the forecast is the ridge's). A per-epoch check (Y = 2024, GRU resid + X) shows overfitting, not a bug: the training MSE falls 8% in 10 epochs (lr 1e-3) while the validation MSE rises 0.6 to 2.4% above that of e_hat = 0; at lr 1e-4 the same at a tenth of the speed. Only for 2026 do the "resid" nets train 1 to 2 epochs; their +0.2e-4 comes from that year alone and changes 0.2 to 0.3% of the signs.

```bash
python demo/stock_recurrent.py
```

## MOPS material-information announcements without the text, walk-forward (`demo/stock_mops.py`, not included, data from `demo/fetch_news.py mops`)

The whole market's 重大訊息 of 2020-01 .. 2026-08 from MOPS t05st02, one request per calendar day: 2,435 days, 515,178 unique announcements (deduplicated on company, date, time and sequence number, since a day's page also lists the previous evening's), 56,824 of them by our 100 stocks, mostly 14:00 - 19:00. For the forecast of day t (before 09:00): log(1 + count) of the stock's announcements overnight (after 13:30 of the previous trading day), during the previous session and in the 5 trading days before; overnight counts of the 10 most frequent clauses (款次, training years); the market-wide overnight count. 17.9% of the stock-days have an overnight announcement. Base: poly p0 + range + US (52 features). Paired differences to the base, day-clustered, 1e-4 units:

| Features | R² (100) | Score (100) | Score − base (100) | R² − base (100) | Score − base (50) | Score − base (announcement days) |
|---|---|---|---|---|---|---|
| *Always down* | | *+7.8 ± 2.4* | | | *(50: +3.4 ± 2.3)* | *(+0.7 ± 2.8)* |
| Base | +0.0087 | +14.3 ± 2.1 | 0 | 0 | 0 (+10.6) | 0 (+11.1) |
| + counts (3 windows) | +0.0088 | +14.5 ± 2.0 | +0.3 ± 0.1 | +0.0001 ± 0.0001 | +0.2 ± 0.2 | +0.0 ± 0.4 |
| + counts + 10 clauses | +0.0087 | +14.4 ± 2.0 | +0.1 ± 0.2 | −0.0000 ± 0.0001 | +0.2 ± 0.3 | −0.8 ± 0.7 |
| + all (and market count) | +0.0087 | +14.2 ± 2.0 | −0.0 ± 0.2 | +0.0000 ± 0.0002 | +0.0 ± 0.3 | −0.8 ± 0.7 |

- **The existence and the type of an announcement carry no usable signal.** The counts add +0.3e-4 on the 100 stocks, but nothing on the days that have an announcement, where they should act; the clause counts lose 0.8e-4 there.
- **Announcement days drift down less.** Always down scores +0.7e-4 on the stock-days with an overnight announcement against +7.8e-4 on all, while the base ridge keeps +11.1e-4 there. Whether an announcement is good or bad news needs its text: that is the LLM step.

```bash
python demo/stock_mops.py
```

## E2: LLM scores and embeddings of the announcements and news, walk-forward (`demo/news_score.py`, `demo/stock_news.py`)

Texts: the 56,824 MOPS announcements of our 100 stocks and the 138,374 cnyes articles (one item per tagged stock among ours, up to 3, or one market item when an article tags none of them): 230,336 items. Every item goes through two models on the 5090 (about 40 minutes):

- **Zero-shot LLM score**: Qwen2.5-3B-Instruct (bf16; a 7B model does not fit next to another user's 17 GB) is asked whether the item is positive, neutral or negative for the company's share price on the next trading day (for market items: for the Taiwan market). s = P(正面) - P(負面) from the first-token logits of the three answers. The model is optimistic: argmax 55.1% positive, 39.3% neutral, 5.6% negative; mean s +0.49, sd 0.55.
- **Embedding**: BAAI/bge-m3 (CLS, 1024 dims), projected on the first 16 (own) or 8 (market) principal components of the training items (before 2022).

An item enters the forecast of the first trading day d with d 08:50 >= its time, as "overnight" if it came after 13:30 of the trading day before. 31.8% of the stock-days have an own overnight item. Features: per source, log(1 + count) overnight, the sum of s overnight and the sum of s in the previous session; the market's mean s and count overnight; the summed own embedding components and the mean market components. Base: poly p0 + range + US (52 features).

Model-free check: the daily rank IC of the summed overnight s with r over the stocks with an item is -0.012, -0.005, +0.001, +0.006, +0.019, -0.008, +0.007 in 2020 .. 2026; the pooled correlation of the own overnight s with r is -0.022 (MOPS) and -0.002 (cnyes). The market's mean s has no monotonic relation with the next day's mean r (quintiles -6.4, -11.1, -7.0, -9.4, -3.0e-4) and its yearly correlation changes sign.

Walk-forward 2022-2026, paired differences to the base, day-clustered, 1e-4 units:

| Features | Score (100) | Score − base (100) | R² − base (100) | Score − base (50) | Score − base (own-item days) | Score − base (100, 2025-26) | Score − base (50, 2025-26) |
|---|---|---|---|---|---|---|---|
| *Always down* | *+7.8 ± 2.4* | | | *(50: +3.4)* | *(+4.9)* | *(+11.0)* | *(+5.5)* |
| Base | +14.3 ± 2.1 | 0 | 0 (R² +0.0087) | 0 (+10.6) | 0 (+14.1) | 0 (+18.2) | 0 (+13.5) |
| + LLM MOPS | +14.2 ± 2.1 | −0.0 ± 0.1 | +0.0000 ± 0.0000 | −0.1 ± 0.1 | −0.1 ± 0.2 | −0.0 ± 0.1 | −0.1 ± 0.2 |
| + LLM cnyes | +14.4 ± 2.0 | +0.1 ± 0.2 | −0.0001 ± 0.0001 | +0.0 ± 0.2 | −0.4 ± 0.4 | −0.2 ± 0.3 | −0.1 ± 0.3 |
| + LLM own (both) | +14.5 ± 2.0 | +0.2 ± 0.2 | −0.0000 ± 0.0001 | +0.1 ± 0.2 | −0.1 ± 0.4 | +0.2 ± 0.3 | +0.2 ± 0.4 |
| + LLM own + market | +14.9 ± 2.1 | +0.6 ± 0.3 | +0.0004 ± 0.0003 | +0.5 ± 0.3 | +0.3 ± 0.5 | −0.0 ± 0.4 | +0.3 ± 0.5 |
| + embedding own | +14.3 ± 2.0 | +0.0 ± 0.2 | +0.0000 ± 0.0001 | −0.0 ± 0.2 | −0.3 ± 0.5 | −0.1 ± 0.3 | −0.1 ± 0.4 |
| + embedding own + market | +13.6 ± 2.1 | −0.7 ± 0.7 | −0.0000 ± 0.0011 | −0.9 ± 0.7 | −1.9 ± 0.8 | +0.1 ± 0.7 | −0.1 ± 0.7 |
| + LLM + embedding (all) | +13.6 ± 2.1 | −0.7 ± 0.7 | +0.0005 ± 0.0011 | −0.8 ± 0.7 | −1.4 ± 0.9 | +0.0 ± 0.6 | +0.4 ± 0.6 |

- **The stock's own texts do not help.** Neither the LLM's reading of the announcements and news nor their embeddings move the score by more than 0.4e-4 on any subset, including the stock-days that have an overnight item, where they should act, and 2025-2026, after the LLM's training data (so the null is not a look-ahead artefact hiding a gain, and look-ahead did not create one either).
- **The market's mean LLM score gives +0.6e-4 ± 0.3e-4 on the 100 stocks, but it is not stable**: 0 in 2025-2026, no monotonic pattern across its quintiles, and the sign of its correlation with the next day's return changes from year to year.
- **The embeddings overfit.** With the 8 market components the forecast flips 9% of the signs and loses 0.7 to 1.9e-4.
- **Why**: most of an overnight item's information is in the opening price, and r starts at the open. What is left for the intraday move is small against an sd of r of about 2.4e-2.

```bash
HF_HOME=<models> PYLIB=<transformers target dir> python demo/news_score.py   # 230k items, ~40 min on a 5090
python demo/stock_news.py
```

## E2b: text × price interactions and text-queried attention on the ridge residual, walk-forward (`demo/stock_news_attn.py`, `demo/stock_news_attn2.py`)

Text inputs per stock-day (own overnight items only): has, log(1 + count), the mean LLM score s, and the mean of the first 16 PCs of the bge-m3 embeddings (18 dims, standardized on the item days before 2022), or the mean raw 1024-dim embedding. Scores ×1e-4, paired with the base ridge (+14.3 ± 2.1 on 100 stocks, +10.6 ± 1.9 on 50).

**Ridge with interactions** (target r directly):

| Features | 100 stocks: Score − base | R² − base | Item days: Score − base | 50 stocks: Score − base |
|---|---|---|---|---|
| + has, n | +0.1 ± 0.1 | −0.0001 | −0.2 ± 0.3 | −0.1 ± 0.2 |
| + has × X (52) | +0.1 ± 0.3 | −0.0007 ± 0.0003 | +0.9 ± 0.7 | +0.1 ± 0.3 |
| + s, s × X (52) | −0.1 ± 0.2 | −0.0003 | −0.1 ± 0.6 | −0.2 ± 0.3 |
| + PC1–4 × X (208) | −0.6 ± 0.4 | −0.0010 ± 0.0005 | −1.2 ± 1.1 | −0.3 ± 0.5 |

**Attention, first version** (FiLM-modulated query from the price state, 16 dims, about 5,000 parameters; versions A no text, B shuffled text, C text): the best epoch was 0 in almost every year and seed, so every version equals the ridge (flips 0.2%). Diagnostics on Y = 2024: the training MSE falls 5% in 10 epochs while the validation MSE rises 1.4% (A) and 1.7% (C) above ê = 0, and no checkpoint of any learning rate (1e-3, 3e-4, 1e-4), with or without the head bias, gets below ê = 0. The head bias alone costs 0.09% because the yearly mean of the out-of-sample residual moves (2022 −7.5e-4, 2023 +2.0e-4). A (no text) overfits as much as C, so the overfitting comes from the price path. Two flaws of this version: the net was not refitted on Y − 1 (in 2022 it had 9 months of training data), and its training targets used ridges fitted with Y − 1.

**Attention, minimal version** (D = 8, RoPE keys, values = keys before rotation, no head bias, about 600 parameters; selection on the item days of Y − 1 with residuals from ridges that did not see Y − 1, leave one half-year block out; refit on all years before Y for the selected number of steps):

| Query | 100 stocks: Score − ridge | R² − ridge | Item days: Score − ridge | 50 stocks: Score − ridge | 2025–26, 100: Score − ridge |
|---|---|---|---|---|---|
| Q0 learned constant | −0.0 ± 0.1 | +0.0000 | −0.3 ± 0.2 | −0.2 ± 0.2 | −0.1 ± 0.3 |
| Q1 price state W_q h_60 | −0.4 ± 0.4 | −0.0013 ± 0.0014 | −0.4 ± 0.6 | −0.5 ± 0.4 | −1.1 ± 1.1 |
| Q2 text u (18) | −0.1 ± 0.1 | +0.0000 | −0.4 ± 0.2 | −0.1 ± 0.1 | −0.2 ± 0.3 |
| Q2s shuffled | −0.1 ± 0.1 | −0.0001 | −0.4 ± 0.2 | −0.1 ± 0.1 | −0.2 ± 0.3 |
| Q3 raw embedding (1024) | −0.2 ± 0.1 | +0.0000 | −0.5 ± 0.2 | −0.3 ± 0.2 | −0.5 ± 0.3 |
| Q3s shuffled | −0.3 ± 0.1 | −0.0003 ± 0.0002 | −0.7 ± 0.3 | −0.5 ± 0.2 | −0.9 ± 0.4 |

The selection kept 0 steps in 2022 and 2023 for every version; in 2024 and 2025 some seeds kept 66–630 steps at a validation MSE 0.1–0.2% below ê = 0, and these gains did not carry over to the test years. Text vs shuffled text: Q2 − Q2s is 0.0 ± 0.1 in score; Q3 − Q3s is +0.0003 ± 0.0001 in R², but only because Q3s is worse than the ridge while Q3 equals it. No query beats the ridge on any subset. Text, its embedding, a price-state query and a learned temporal pooling all add nothing to the ridge's residual, and the price-state query (the most flexible) is the worst.

**The minimal attention under strong regularization** (`demo/stock_news_attn3.py l1|l0|drop`, not included; same splits and selection; L1 on all weights λ ∈ {1e-5, 1e-4, 1e-3}; L0 hard-concrete gates on the inputs of P and of W_q, λ per expected open gate ∈ {1e-5, 1e-4, 1e-3}; dropout on every daily input element p ∈ {0.5, 0.8, 0.9}; "chosen" picks the strength per seed on Y − 1 together with the steps). The weight decay of the earlier runs (AdamW 1e-2 at lr 1e-3, a factor 1 − 1e-5 per step) was negligible. Score − ridge, ×1e-4, for the chosen strength:

| Query | L1: 100 stocks | L1: item days | L0: 100 stocks | L0: item days | Dropout: 100 stocks | Dropout: item days | Dropout: 2025–26 |
|---|---|---|---|---|---|---|---|
| Q0 constant | −0.0 ± 0.1 | −0.3 ± 0.2 | −0.3 ± 0.2 | −0.4 ± 0.2 | +0.0 ± 0.1 | −0.3 ± 0.2 | −0.1 ± 0.3 |
| Q1 price state | +0.0 ± 0.5 | +0.1 ± 0.6 | −0.5 ± 0.4 | −0.5 ± 0.4 | −0.2 ± 0.4 | −0.4 ± 0.4 | −0.0 ± 0.9 |
| Q2 text | +0.1 ± 0.1 | −0.3 ± 0.2 | −0.1 ± 0.1 | −0.2 ± 0.2 | +0.2 ± 0.2 | +0.0 ± 0.4 | +0.1 ± 0.4 |
| Q3 embedding | +0.0 ± 0.1 | −0.2 ± 0.2 | −0.1 ± 0.1 | −0.3 ± 0.2 | −0.0 ± 0.2 | +0.1 ± 0.4 | −0.4 ± 0.4 |

Of the 48 fixed-strength rows and 12 chosen rows, none differs from the ridge in score by more than 1.5 standard errors on the 100 stocks; the largest R² gain is Q2 with the chosen dropout on the item days, +0.0005 ± 0.0002, with a score change of +0.0 ± 0.4, one of about 240 comparisons and without its shuffled control. The strongest settings (L1 1e-3, dropout 0.9) mostly pull the correction to zero (flips 0.2–0.4%), which removes the price-state query's loss (−0.4 without regularization) without adding anything. L1, L0 and large dropout therefore only reduce the damage of the correction; none finds a residual structure that carries over to the next year.

## E2e: the same-day opening gap and a news-predicted gap, walk-forward (`demo/stock_news_gap.py`)

g_t = log(o_t / c_{t−1}) is not known at 08:50, so the rows with it are an upper bound. ĝ is a ridge forecast of g_t from what is known before 09:00: either the 52 base features (a control) or the base plus the news features. Stage 1 is walk-forward as well, and on the training years ĝ comes from the fit without that year. Each row adds g, |g| and the day's mean g (or the same for ĝ). Score differences to the base, ×1e-4, day-clustered.

| Set | 100 stocks | 50 stocks | Item days | 2025–26 |
|---|---|---|---|---|
| base (Score) | +14.3 ± 2.1 | +10.6 ± 1.9 | +14.1 ± 2.4 | +18.2 ± 4.2 |
| + realized g (upper bound) | +2.3 ± 0.8 | +1.9 ± 0.8 | +1.8 ± 1.3 | +4.2 ± 1.8 |
| + ĝ from price (control) | +0.2 ± 0.3 | +0.4 ± 0.3 | +0.2 ± 0.4 | +0.8 ± 0.6 |
| + ĝ from price + news | +0.2 ± 0.3 | +0.4 ± 0.4 | +0.3 ± 0.5 | +0.7 ± 0.7 |
| + news features directly | −0.7 ± 0.7 | −0.8 ± 0.7 | −1.4 ± 0.9 | +0.0 ± 0.6 |

The walk-forward R² of the gap is 0.2348 from price and 0.2361 with the news, and corr(g_t, r_t) = −0.070 (the open partly reverts). The realized gap is worth +2.3e-4 (2.9 standard errors), but the part of it that the price and the news forecast is worth only +0.2e-4: what helps is the information that arrives before the open. Only the 08:30–09:00 pre-open match could approach it. From here on ĝ (from price + news) is part of the base (55 features).

## E2f: iTransformer stock tokens with a cross attention to the morning's news, walk-forward (`demo/stock_itrans_news.py`)

Base: the ridge on the 52 features plus the ĝ block (Score +14.5e-4 ± 2.0e-4, R² 0.0095). Stage 2 forecasts its residual. Each stock is one token, made from its whole 60-day window of [e, r, g, traded] (240 values → d = 16) plus the day's ridge inputs. One self-attention layer runs across the 100 stocks. The news pool of a day holds every overnight item of our stocks and the market items, at most 256 (mean 117). Each item is a 65-dim vector: the first 64 PCs of bge-m3 plus the LLM score. The four modes differ in who is the query, the key and the value:

- **stockq**: query = stock token; key = value = news. There is a learned bias for the stock's own items and a null key.
- **newsq**: query = the mean of the stock's own items; key = value = the stock tokens.
- **sigval**: query = stock token; key = news; value = the token of the stock the item tags (a learned market token for market items).
- **itemq**: every item is its own query over the stock tokens, and each stock then attends over the item outputs.

"Shuffled" gives every day the news pool of another day of the same year. Splits and selection are as in E2c: the number of steps (0 allowed) is chosen on Y − 1, the model is refit on all years before Y, and 3 seeds are averaged. Score differences ×1e-4.

| Mode | iT − ridge | iT + news − ridge | shuffled − ridge | news − shuffled: Score | news − shuffled: R² |
|---|---|---|---|---|---|
| stockq | −0.2 ± 0.2 | −0.2 ± 0.2 | −0.1 ± 0.3 | −0.1 ± 0.2 | −0.0003 ± 0.0005 |
| newsq | −0.2 ± 0.2 | −0.3 ± 0.2 | −0.3 ± 0.2 | 0.0 (identical) | 0.0000 |
| sigval | −0.2 ± 0.2 | −0.0 ± 0.1 | −0.2 ± 0.2 | +0.2 ± 0.1 | +0.0003 ± 0.0001 |
| itemq | −0.2 ± 0.2 | −0.2 ± 0.2 | −0.2 ± 0.2 | +0.0 ± 0.0 | −0.0000 ± 0.0000 |

- No mode beats the ridge. The selection picks 0 steps in most folds (all of 2022 and 2024), so the forecasts are mostly the ridge itself.
- In newsq and itemq, the news and the shuffled news give the same forecasts. The weights of the news branch stay near their zero start, so the model does not use it.
- sigval is the only mode where the real news beat the shuffled news: R² +0.0003 ± 0.0001, and +0.0004 ± 0.0002 on the 50 stocks. Here the news decide which stocks' states a stock reads. The gain does not reach the ridge, though (−0.0002 ± 0.0002 in R²), and it is one of about 16 comparisons in this table.

## E2g: what the representations of one news item carry, item-level probes (`demo/news_probe.py`)

The question is whether the text representations hold information on the price at all, apart from any model built on them. The sample is 26,258 stock-days with exactly one overnight item of the stock (after 13:30 of the previous trading day and before 08:50; 46% MOPS), 18,797 of them in 2022–2026. Each target belongs to one text. The targets are:

- g − ĝ: the opening gap minus its walk-forward forecast from the 52 price features, i.e. the part of the gap that the price does not explain;
- |g| − |g|_hat: the size of the gap minus its ridge forecast from the stock's mean |g| over 5, 20 and 60 days and |ĝ|;
- e = r − μ: the residual of the forecast target after the base ridge.

The probe is a ridge on one representation, walk-forward: α is chosen on Y − 1 and the model is refit before Y. PCA and the TF-IDF vocabulary are fitted on the items before 2022. The tables give the out-of-sample R² pooled over 2022–2026 against the training mean, and its standard error from a bootstrap over days.

| Representation | g − ĝ: R² | g − ĝ: corr | \|g\| − \|g\|_hat: R² | \|g\| − \|g\|_hat: corr | e: R² |
|---|---|---|---|---|---|
| training mean (baseline) | 0 | — | 0 | — | 0 |
| LLM s = P(pos) − P(neg) | +0.0013 ± 0.0005 | +0.033 | +0.0000 ± 0.0001 | −0.006 | +0.0000 ± 0.0002 |
| LLM 3 probabilities | **+0.0020 ± 0.0007** | +0.043 | −0.0002 ± 0.0003 | +0.003 | −0.0000 ± 0.0003 |
| bge-m3 PCA16 | +0.0008 ± 0.0005 | +0.026 | **+0.0050 ± 0.0011** | +0.069 | +0.0006 ± 0.0004 |
| bge-m3 PCA64 | +0.0007 ± 0.0006 | +0.025 | +0.0032 ± 0.0012 | +0.058 | −0.0002 ± 0.0006 |
| bge-m3 all 1024 | −0.0002 ± 0.0004 | +0.002 | +0.0023 ± 0.0013 | +0.051 | +0.0001 ± 0.0007 |
| title TF-IDF (char 1–2-grams, 4000) | +0.0003 ± 0.0001 | +0.009 | +0.0035 ± 0.0009 | +0.057 | +0.0001 ± 0.0002 |
| MOPS clause + source dummies | −0.0002 ± 0.0001 | −0.036 | +0.0002 ± 0.0009 | +0.025 | +0.0002 ± 0.0003 |
| all together | +0.0005 ± 0.0002 | +0.020 | +0.0047 ± 0.0012 | +0.067 | +0.0005 ± 0.0003 |

- The direction of the unexplained gap is carried by the LLM probabilities, not by the embedding. Most of it comes from cnyes news (R² +0.0033) rather than MOPS (+0.0004).
- The size of the gap is carried by the embedding and the title words, not by the LLM. The 16 leading components do best; with more dimensions the ridge overfits (PCA64 and 1024 are lower).
- No representation explains e. The information that the text has on the day is spent at the open, which agrees with E2e.
- This points to the inputs for the later models: LLM probabilities for direction plus a low-dimensional embedding for magnitude, and the gap as the target to pretrain the text encoder on.

## E2h: choice of the local LLM for typed fields (`demo/news_summary.py`, `demo/news_struct.py`)

Every model filled the same form on the same 50 items (25 MOPS, 25 cnyes, spread over the years): event type, direction, size, surprise, key numbers and a one-sentence summary. Greedy decoding was used, thinking was off, and the model saw the text only. There are no human labels, so the judgment column is my reading of the 50 outputs, not a measured accuracy.

| Model | Format errors / 50 | Judgment | Speed and memory on the shared RTX 5090 (≈15 GB free) |
|---|---|---|---|
| Qwen2.5-3B-Instruct (used in E2) | 8 | direction mostly 利多 | fast |
| **Qwen3.5-4B** | 0 | direction 21 / 23 / 6 (利多 / 利空 / 中性), size mostly 小; marks items unrelated to the stock | batch 12, ≈ 8 h for all items |
| Gemma 4 E2B | — | size almost always 中 | fast |
| Ministral 3 8B (FP8) | ≈ 25 (parentheses) | best financial reasoning | out of memory at batch 8; 148 s / 50 at batch 2, ≈ 190 h for all items |
| NuExtract3 (template mode) | — | weak on direction; overuses 大 | best at extracting the numbers |

Jev (TypeSafe AI) was not tried: it is API-only, so the private texts would leave the machine, and it needs a paid key. `demo/news_struct.py` reaches the same typed, calibrated output locally. The form tokens are forced, and at each field the generation is restricted to the allowed letters. The softmax over those letters is kept as the field's probabilities, and the most likely letter is written before the next field, so format errors cannot occur. Fields: relation (direct / indirect / unrelated), event (12 types), direction, size, surprise, then a free summary of at most 80 tokens. Qwen3.5-4B runs this over all 233k items (57 parts of 4,096). Its Gated DeltaNet layers run on the PyTorch reference path, because flash-linear-attention and causal-conv1d are not installed; this changes the speed only. The probe of E2g is to be repeated on these probabilities and on embeddings of the summaries.

The model choice rests mainly on speed and memory. With constrained decoding, Ministral's format errors no longer matter. An objective comparison without labels would be the E2g probe of g − ĝ on the 26k single-item days, scored by a second model.

### The E2g probe on the typed fields and the summaries (`demo/news_probe.py struct`)

The probe of E2g was repeated on the same 26,258 single-item stock-days (18,797 in 2022–2026) with the outputs of `demo/news_struct.py`: the probabilities of the direction field (3), of the size field (3) and of all five fields (24: relation 3, event 12, direction 3, size 3, surprise 3), and the bge-m3 embedding of the one-sentence summary, projected on its principal components fitted on the items before 2022. The first two rows are the best E2 representations of the same items. Out-of-sample R² pooled over 2022–2026 against the training mean, with its standard error from a bootstrap over days (`demo/runs/news_probe_struct.log`):

| Representation | g − ĝ: R² | corr | \|g\| − \|g\|_hat: R² | corr | e: R² |
|---|---|---|---|---|---|
| training mean (baseline) | 0 | — | 0 | — | 0 |
| Qwen2.5 3 probabilities (E2) | +0.0020 ± 0.0007 | +0.043 | −0.0002 ± 0.0003 | +0.003 | −0.0000 ± 0.0002 |
| text bge-m3 PCA16 (E2) | +0.0008 ± 0.0005 | +0.026 | +0.0050 ± 0.0010 | +0.069 | +0.0006 ± 0.0004 |
| Qwen3.5 direction, 3 | +0.0031 ± 0.0009 | +0.055 | +0.0004 ± 0.0004 | +0.018 | −0.0001 ± 0.0002 |
| Qwen3.5 size, 3 | +0.0002 ± 0.0004 | +0.011 | −0.0001 ± 0.0005 | +0.009 | −0.0001 ± 0.0001 |
| Qwen3.5 all fields, 24 | +0.0027 ± 0.0009 | +0.051 | +0.0035 ± 0.0010 | +0.058 | −0.0007 ± 0.0005 |
| summary bge-m3 PCA16 | +0.0034 ± 0.0009 | +0.059 | **+0.0063 ± 0.0011** | +0.078 | −0.0006 ± 0.0005 |
| summary bge-m3 PCA64 | +0.0016 ± 0.0006 | +0.037 | +0.0041 ± 0.0012 | +0.062 | −0.0006 ± 0.0007 |
| fields 24 + summary PCA16 | **+0.0053 ± 0.0011** | +0.075 | +0.0054 ± 0.0013 | +0.072 | −0.0009 ± 0.0007 |
| Qwen2.5 3 + fields 24 | +0.0034 ± 0.0010 | +0.057 | +0.0024 ± 0.0009 | +0.049 | −0.0005 ± 0.0005 |
| fields 24 + summary 16 + text 16 | +0.0029 ± 0.0009 | +0.052 | +0.0046 ± 0.0011 | +0.066 | −0.0005 ± 0.0006 |

- Direction of the unexplained gap: the Qwen3.5 direction probabilities (+0.0031) carry more than the Qwen2.5 probabilities (+0.0020), mainly on MOPS (R² +0.0019 against +0.0004; cnyes +0.0041 against +0.0033). The summary embedding carries as much as the fields (+0.0034), unlike the embedding of the full text (+0.0008), and the fields with the summary together reach +0.0053, more than either alone.
- Size of the gap: the size field itself predicts nothing (−0.0001); the 24 fields together do (+0.0035, mostly through the event and relation fields on cnyes). The summary embedding keeps the size information of the full text and exceeds it (+0.0063 against +0.0050), especially on MOPS (+0.0060 against +0.0040).
- As in E2g, PCA64 is below PCA16, and adding the text PCA16 to the fields and the summary lowers both R²: with one ridge penalty for all blocks the extra dimensions are overfit.
- No representation explains e (all R² between −0.0009 and +0.0006); the information is spent at the open.
- The differences between rows are not paired tests; the standard errors are those of each row. The inputs for the later models become the 24 field probabilities plus the summary PCA16.

## E2i: which stock embedding routes the news of other stocks, a linear spillover probe (`demo/stock_id_probe.py`)

`StockIdentityAttention` lets each news item choose its stocks by their identity embedding. This probe checks, before any network, whether a candidate embedding sends the news of one stock to the stocks it moves. The sample is the stock-days without an own overnight item on days when at least one other stock has one: 106,127 stock-days, 79,218 of them in 2022–2026, with on average 31.8 other stocks with news per day. The targets are those of E2g for stock i. With a similarity w_ij ≥ 0 (w_ii = 0) the two features of stock i are Σ_j w_ij s_j and Σ_j w_ij n_j. Here s_j is the sum of P(利多) − P(利空) of the direction field of `demo/news_struct.py` over the overnight items of stock j, and n_j = 1 if stock j has any. The similarities are:

- equal: w = 1 for all other stocks;
- industry: 1 for the same TWSE / TPEx industry code;
- comovement: the cosine of `comovement_embedding` (dim 16) of the daily log returns, for fold Y fitted on the years before Y − 1;
- profile: the cosine of the centred bge-m3 embeddings of the company descriptions (`demo/fetch_profiles.py`).

Negative cosines are set to 0. Each similarity is also given to the wrong stocks (rows and columns permuted), a control that keeps the distribution of the weights. The off-diagonal weights correlate with those of the 2026 comovement at +0.59 (industry), +0.54 (profile) and +0.71 (comovement 2022). The ridge is walk-forward as in E2g; the table gives the out-of-sample R² pooled over 2022–2026 against the training mean, with bootstrap standard errors over days (`demo/runs/stock_id_probe.log`).

| Similarity | g − ĝ: R² | \|g\| − \|g\|_hat: R² | corr |
|---|---|---|---|
| training mean (baseline) | 0 | 0 | — |
| equal (all other stocks) | −0.00011 ± 0.00076 | +0.00359 ± 0.00119 | +0.051 |
| industry | −0.00003 ± 0.00025 | +0.00150 ± 0.00032 | +0.022 |
| industry, permuted | −0.00005 ± 0.00008 | +0.00029 ± 0.00024 | −0.013 |
| comovement | −0.00007 ± 0.00015 | +0.00029 ± 0.00036 | −0.009 |
| comovement, permuted | +0.00003 ± 0.00022 | +0.00070 ± 0.00041 | +0.004 |
| profile bge-m3 | −0.00008 ± 0.00014 | +0.00012 ± 0.00030 | −0.018 |
| profile, permuted | −0.00001 ± 0.00028 | +0.00118 ± 0.00038 | +0.015 |
| comovement + profile | −0.00008 ± 0.00013 | +0.00037 ± 0.00041 | −0.005 |
| equal + comovement + profile | −0.00023 ± 0.00050 | +0.00417 ± 0.00130 | +0.057 |
| equal + industry + comovement + profile | −0.00012 ± 0.00052 | **+0.00517 ± 0.00128** | +0.067 |

- No similarity carries the direction of other stocks' news to a stock (all R² of g − ĝ within ±0.0002 of 0). A plain weighted sum cannot express that the same news can be good for one stock and bad for its competitor, so this rules out only linear, same-sign spillover.
- The size of the gap rises with the news of the other stocks: the equal weights, in effect the number of stocks with news, give +0.0036. This is a market-wide effect, such as the days of earnings and monthly revenue releases, not a routing by identity.
- Among the identities only the industry routes: +0.0015 against +0.0003 for the permuted control, and +0.0010 on top of equal, comovement and profile (+0.0052 against +0.0042). Comovement and profile do not beat their permuted controls (+0.0003 against +0.0007, +0.0001 against +0.0012), although their weights correlate with the industry; their spread-out weights mix in too many unrelated stocks.
- For `StockIdentityAttention` this favours an industry code (one-hot) plus the learned part as the identity, with the comovement embedding as a variant to test in the network, where the attention can pick sharper neighbours and signed effects than this linear probe.

## E2j: the typed news fields in the gap block, and `StockIdentityAttention` on the ridge residual (`demo/stock_identity_news.py`)

### Step 0: the news input of the gap ridge

The gap block (ĝ, |ĝ|, the day's mean ĝ) is made by a ridge that forecasts the opening gap g from the 52 price features and a news input. Here that input is either E2 (the features of `demo/stock_news.py`: Qwen2.5 probabilities and bge-m3 of the texts, own and market items) or struct. Struct holds, per stock-day, the means over its overnight items of the 24 field probabilities and the summary PCA16, the log counts per source, and the mean direction of the previous session's items (43 columns). The struct version with market also adds the same 40 means and the log count of the market items (84 columns). The R² of g is pooled over 2022–2026 against the training mean (`demo/runs/stock_identity_gap_diag.log`, `demo/runs/stock_identity_gap.log`):

| ĝ ridge input | Columns | R² of g | days with an own item | days without |
|---|---|---|---|---|
| price only (baseline) | 52 | +0.2369 | +0.2279 | +0.2430 |
| + E2 (reference, E2f) | 84 | +0.2382 | +0.2291 | +0.2444 |
| + struct | 95 | **+0.2395** | **+0.2327** | +0.2441 |
| + struct with market | 136 | +0.2303 | +0.2260 | +0.2332 |
| + struct counts, direction, session | 58 | +0.2390 | +0.2314 | +0.2440 |
| + E2 + struct | 127 | +0.2396 | +0.2322 | +0.2445 |

- The market part of struct hurts even on the stock-days without news (−0.0109 against price only), where the stock's own news columns are 0, so the loss comes from the market columns. These 41 columns are the same for all stocks of a day, and the ridge overfits them. Without them, struct doubles the gain of the news over price (+0.0026 against +0.0013 for E2; on days with an own item +0.0048 against +0.0012), in line with E2h.
- A better ĝ does not give a better forecast target. The final ridge (52 + the gap block) on all 100 stocks: E2 gap R² +0.0095, score +14.5e-4 ± 2.0e-4; struct gap R² +0.0090, score +14.3e-4 ± 2.0e-4, paired R² −0.0005 ± 0.0002 and score −0.2e-4 ± 0.2e-4 against E2; E2 + struct equal to E2 (paired R² −0.0001 ± 0.0000). The always-down baseline scores +7.8e-4 ± 2.4e-4. A likely reason is that the gap block helps through the intraday reversal of price-driven gaps, while the gap explained by news need not reverse. That would call for the news part and the price part of ĝ as separate columns (not tested).

### Steps 1–2: the networks

The networks use the E2 gap block, so the ridge reference is that of E2f. The protocol is that of `demo/stock_itrans_news.py`: steps chosen on Y − 1, refit before Y, 3 seeds. The versions differ in the news items and the attention. Struct items are 42-dim: 24 fields, summary PCA16 and 2 source flags. The identity is the industry one-hot, the comovement embedding refitted per fold, or zero (learned part only); 8 learned dimensions are added in each case. The two controls are the industry given to the wrong stocks, and each day's item pool replaced by another day's of the same year. Walk-forward 2022–2026, all 100 stocks, paired against the ridge (`demo/runs/stock_identity_net.log`):

| Version | R² | Score | vs ridge: R² | vs ridge: score |
|---|---|---|---|---|
| always down (trivial baseline) | — | +7.8e-4 ± 2.4e-4 | | |
| ridge, 52 + E2 gap block (reference) | +0.0095 | +14.5e-4 ± 2.0e-4 | | |
| iT, no news | +0.0092 | +14.3e-4 ± 2.1e-4 | −0.0003 ± 0.0003 | −0.2e-4 ± 0.2e-4 |
| sigval, E2 news | +0.0093 | +14.4e-4 ± 2.0e-4 | −0.0002 ± 0.0002 | −0.0e-4 ± 0.1e-4 |
| sigval, struct news | +0.0091 | +14.1e-4 ± 2.1e-4 | −0.0004 ± 0.0003 | −0.4e-4 ± 0.2e-4 |
| identity, industry | +0.0088 | +14.1e-4 ± 2.1e-4 | −0.0007 ± 0.0003 | −0.3e-4 ± 0.2e-4 |
| identity, comovement | +0.0091 | +14.2e-4 ± 2.1e-4 | −0.0003 ± 0.0004 | −0.3e-4 ± 0.2e-4 |
| identity, learned only | +0.0094 | +14.4e-4 ± 2.0e-4 | −0.0001 ± 0.0002 | −0.1e-4 ± 0.1e-4 |
| identity, industry permuted (control) | +0.0088 | +14.1e-4 ± 2.1e-4 | −0.0007 ± 0.0003 | −0.3e-4 ± 0.2e-4 |
| identity, industry, news shuffled (control) | +0.0088 | +14.1e-4 ± 2.1e-4 | −0.0007 ± 0.0003 | −0.3e-4 ± 0.2e-4 |

- No network beats the ridge, and the news attention is not learned at all. In 50 of the 75 (version, fold, seed) choices of the identity versions (78 of 120 over all versions) the number of steps chosen on Y − 1 is 0; otherwise it is one epoch (54–73 steps). The attention of the items over the stocks stays uniform: entropy 4.60 against 4.61 for uniform, 0.010 on the tagged stock (uniform 0.010), β ≈ +0.02. The share on the same industry is 0.11 against 0.094 for uniform. The industry version and both controls flip 0.0–0.1% of the signs against each other, so the differences from the ridge come from the stock tokens, not from the news.
- The forecast target therefore gives too little signal to train a news pathway with the zero-started head in about one epoch, as in E2f. Whether the identity routes news cannot be decided on e this way. The probes (E2g, E2h, E2i) found the signal of the news in the gap g − ĝ, so the natural next test is to train the item-over-stock attention on g − ĝ (where a routing would show up against the two controls), and to use what it learns for e afterwards.

## E3a: a learned signed relation between the Taiwan stocks and 46 US stocks, on the opening gap (`demo/stock_us_relation.py`)

E2j showed that a news attention does not learn on the forecast target e. This test therefore moves the learning to the opening gap, where the overnight foreign information arrives, and starts from prices only. The target of Taiwan stock s on day t is g − ĝ: the opening gap minus the walk-forward ridge forecast from the 52 base features. Those features already hold the TSM / SOX / NQ returns, the ADR premium change and the own ADR, with coefficients shared by all stocks. The input r_jt is the log return of each of the 46 US stocks of `dataset/peers/us` over the US sessions that ended between the Taiwan close of t − 1 and the open of t, standardized before 2022.

The model is ĝ_s = γ_s Σ_j a_sj r_jt, with γ_s = softplus(·) ≥ 0 so that the sign of the relation is in a_sj. The scores are x_sj = (W_q u_s)·(W_k v_j)/√8. Here u_s is the Taiwan identity (industry one-hot plus 8 learned dimensions started at 0) and v_j is the US identity (8 learned dimensions). The normalizations of a are:

- softmax: softmax_j(m);
- signed: softmax_j(m)·tanh(c), where softmax selects and tanh gives the direction;
- differential: softmax_j(m₁) − λ softmax_j(m₂);
- L1: c / (ε + Σ_j |c|);
- low rank: c with no normalization.

The comparisons are a ridge per Taiwan stock on all 46 returns (4,600 coefficients, α shared and chosen on Y − 1) and the mean of the hand-made peers of `demo/fetch_peers.py` (46 stocks, 126 links) with a coefficient per stock. Training is full batch with Adam, lr 1e-2. The L2 on the embeddings and projections is chosen on Y − 1 from 1e-4 to 1e-1 (none on γ and λ), the steps are chosen on Y − 1, and 3 seeds are averaged. The networks have 1,589–1,909 parameters. The table gives the R² pooled over 2022–2026 against the training mean, with bootstrap standard errors over days (`demo/runs/stock_us_relation.log`):

| Model | 100 stocks: R² | corr | 50-stock subset: R² | stocks with hand-made peers: R² |
|---|---|---|---|---|
| training mean (baseline) | 0 | — | 0 | 0 |
| hand-made peers, coefficient per stock | +0.0277 ± 0.0053 | +0.172 | +0.0644 | +0.0677 |
| ridge per stock on 46 returns | **+0.1004 ± 0.0119** | +0.330 | +0.0888 | +0.0901 |
| softmax | +0.0617 ± 0.0141 | +0.263 | +0.0485 | +0.0516 |
| **signed** | +0.0992 ± 0.0111 | +0.320 | +0.0875 | +0.0886 |
| differential | +0.0737 ± 0.0113 | +0.278 | +0.0651 | +0.0670 |
| L1 | +0.0676 ± 0.0117 | +0.262 | +0.0639 | +0.0653 |
| low rank | +0.0960 ± 0.0113 | +0.319 | +0.0828 | +0.0843 |
| signed, industry permuted (control) | +0.0987 ± 0.0115 | +0.319 | +0.0869 | +0.0883 |
| signed, US returns of another day (control) | −0.0001 ± 0.0008 | +0.015 | −0.0002 | −0.0001 |

- The US stocks explain about a tenth of the gap that the base ridge leaves, and the control with the returns of another day gives 0, so this is real overnight information. Among the normalizations that allow a negative relation, signed (+0.0992) and low rank (+0.0960) reach the per-stock ridge (+0.1004) with less than half its parameters. Softmax, which allows only non-negative weights, is far lower (+0.0617). Differential (+0.0737) and L1 (+0.0676) lie in between.
- What the signed model learned is mostly not a pairwise relation. For every stock, the largest weights fall on the same US stocks (DOW, AAPL, AMD, ASML, INTC), all with |a| ≤ 0.009, and this holds for the banks and the shipping lines as well. Only 0.070 of Σ|a| is on the hand-made peers, against 0.060 for uniform weights. Negative weights are 41% of the pairs but 28% of Σ|a|. The industry one-hot adds nothing (the permuted control is equal). The gain therefore comes mainly from a stock-specific sensitivity γ_s to a broad US basket: the base ridge uses the same US coefficients for all stocks. The pairwise structure stays to be isolated by a baseline with one US-basket beta per stock and by a target net of it.

## E3b: the relation on top of a US-basket sensitivity per Taiwan stock (`demo/stock_us_relation.py basket`)

The basket is the equal-weighted mean of the 46 standardized US returns. Every network now also has a free coefficient β_s per Taiwan stock on the basket (no L2), so the relation can only add what one basket beta per stock does not explain. The ridge baselines are per stock, with α shared and chosen on Y − 1. The control gives the relation part the US returns of another day of the same year, but the basket of the true day. The paired column is the R² difference from the basket beta, with bootstrap standard errors over days (`demo/runs/stock_us_relation_basket.log`). The target is g − ĝ, as in E3a.

| Model | 100 stocks: R² | vs basket beta | 50-stock subset: R² | vs basket beta |
|---|---|---|---|---|
| training mean (baseline) | 0 | | 0 | |
| TSM / SOX / NQ beta per stock | +0.0081 ± 0.0008 | −0.0805 ± 0.0149 | +0.0067 | −0.0647 ± 0.0138 |
| **basket beta per stock** | +0.0886 ± 0.0123 | 0 | +0.0715 | 0 |
| basket + TSM / SOX / NQ per stock | +0.0896 ± 0.0126 | +0.0010 ± 0.0006 | +0.0721 | +0.0007 ± 0.0005 |
| ridge per stock on 46 | +0.1009 ± 0.0121 | +0.0123 ± 0.0028 | +0.0890 | +0.0175 ± 0.0035 |
| basket + signed | +0.1007 ± 0.0116 | **+0.0121 ± 0.0040** | +0.0906 | **+0.0192 ± 0.0046** |
| basket + low rank | +0.0998 ± 0.0116 | +0.0112 ± 0.0031 | +0.0881 | +0.0166 ± 0.0035 |
| basket + softmax | +0.0989 ± 0.0105 | +0.0103 ± 0.0040 | +0.0880 | +0.0165 ± 0.0047 |
| basket + signed, relation input of another day (control) | +0.0873 ± 0.0119 | −0.0013 ± 0.0014 | +0.0700 | −0.0015 ± 0.0016 |

- Most of the gain of E3a is one sensitivity per Taiwan stock to the broad US basket: +0.0886 of the +0.1007. The three series of the base features, given a beta per stock, explain only +0.0081.
- The stock-specific relation beyond the basket is real: +0.0121 ± 0.0040 for 100 stocks and +0.0192 ± 0.0046 for the 50-stock subset, while the control with the returns of another day gives −0.0013. All three normalizations and the unconstrained ridge reach about the same level.
- The learned relation is diffuse. The effective number of US stocks per Taiwan stock, (Σ|w|)²/Σw², is a median 29.6 of 46. The hand-made peers carry 0.065 of Σ|w| (uniform 0.060), and about half of the weight is negative. A few weights make sense: ZIM is the largest for the shipping line 2603, TSM for 2330, 3711 and 2382, and banks and plastics weigh TSM and AMAT negatively (a rotation away from technology). But the weights are small (|w| ≤ 0.002 per standardized unit) and spread over many stocks. So the normalization does not make the relation sparse, and the relation behaves like a few US sector factors per Taiwan stock rather than a list of partners.


## E3c: variants of the relation, the pre-open market information, and US after-hours prices and news (`demo/stock_us_variants.py`)

Rounds r1–r12 test the form of the relation of E3b, more inputs, and a new gap base. Unless stated, the setup is that of E3b: walk-forward 2022–2026, the L2 weight and the number of steps chosen on Y − 1, 3 seeds, R² against the training mean with bootstrap standard errors over days (`demo/runs/e3c_r*.log`).

### r1: the form of the weights (target g − ĝ of E3b, on top of the basket beta)

| Model | 100 stocks: R² | vs signed | effective number of US stocks |
|---|---|---|---|
| basket beta per stock (baseline) | +0.0886 | −0.0121 ± 0.0041 | — |
| ridge per stock on 46 (baseline) | +0.1009 | +0.0002 ± 0.0031 | — |
| basket + signed (E3b) | +0.1007 | 0 | 28.1 |
| sparsemax · tanh | +0.1000 | −0.0007 ± 0.0015 | 24.4 (5% exact zeros) |
| tanh, L1-normalized | +0.0416 | −0.0591 ± 0.0094 | 41.1 |
| tanh only | +0.0992 | −0.0015 ± 0.0023 | 30.7 |
| signed + L1 penalty 1e-2 on γ_s a_sj | +0.1024 | +0.0017 ± 0.0025 | 22.6 |
| signed, d 4 / d 16 | +0.0977 / +0.1001 | −0.0030 / −0.0005 | — / 28.7 |
| signed, lr 3e-3 | +0.1017 | +0.0010 ± 0.0026 | 33.8 |

No form beats softmax · tanh. Sparser weights (sparsemax, the L1 penalty) leave R² unchanged, and the share of the hand-made peers stays at 0.063–0.075 (uniform 0.060). Normalizing tanh by its L1 norm fixes the total weight per stock and is much worse.

### r2, r2b: Asian opens and the pre-open betas

The Japanese and Korean peers (12 and 9 stocks) open at 08:00 Taipei, so their opening gaps log(o_t / c_{t−1}) of day t are known before the Taiwan open. Added to the per-stock ridge on the 46 US returns they raise R² from +0.1009 to +0.1562 (+0.0555 ± 0.0151). The 52 base features hold no pre-open information, so this tests whether the gain is market-wide. The pre-open betas are one coefficient per Taiwan stock on four series: the US basket, the open gaps of the Nikkei 225 and KOSPI, and the TAIEX futures gap at 08:45 (`demo/stock_overnight.py`).

| Model (target g − ĝ of E3b) | 100 stocks: R² | vs pre-open betas | 50-stock subset: R² |
|---|---|---|---|
| basket beta per stock | +0.0886 | −0.1304 ± 0.0222 | +0.0715 |
| basket + Nikkei + KOSPI betas | +0.1267 | −0.0923 ± 0.0219 | +0.0970 |
| **pre-open betas (4)** | +0.2190 | 0 | +0.1918 |
| pre-open betas + signed on 46 US | +0.2517 | +0.0327 ± 0.0092 | +0.2339 |
| pre-open betas + signed on US + 21 Asian stock gaps | +0.2527 | +0.0337 ± 0.0080 | +0.2378 |
| same, Asian gaps of another day (control) | +0.2494 | +0.0304 ± 0.0092 | +0.2307 |
| same, Asian gaps of t − 1 (control) | +0.2521 | +0.0331 ± 0.0100 | +0.2350 |

The futures gap at 08:45 carries most of the gap. The Asian stock gaps add nothing beyond the market series (the controls are equal).

### r3: weights that depend on the day's state

One relation and one set of betas per regime of day t: the sign of the US basket, its size (|basket| above the median before 2022), and its volatility over the 20 days before t. On top of the pre-open betas the single relation gives +0.0327; the regimes give +0.0261 (sign), +0.0320 (size) and +0.0218 (volatility). The two relations of a split correlate only 0.3–0.6, so each half has too few days. The pre-open betas split by regime are also slightly worse than one set.

### r4, r5: US after-hours returns and US news

- **After-hours returns:** hourly bars with the extended hours of the US peers and SPY / QQQ / SOXX (`demo/fetch_peers.py alpaca_bars`). For each Taiwan day, the return from the closing print (the open of the 16:00 New York bar) to the close of the last hourly bar that ended by 08:50 Taipei.
- **News:** the Benzinga news of the US peers published after the last US close and before 08:50 Taipei, 24,719 (article, peer) pairs. Each is scored by Qwen3.5 (`demo/news_struct_us.py`) as P(about the company) × (P(positive) − P(negative)) and summed per peer and day.

| Model (target g − ĝ of E3b) | 100 stocks: R² | vs pre-open + signed |
|---|---|---|
| pre-open betas (baseline) | +0.2190 | −0.0327 ± 0.0092 |
| pre-open betas + ridge per stock on 46 (baseline) | +0.2341 | −0.0177 ± 0.0059 |
| **pre-open betas + signed on 46 US (reference)** | +0.2517 | 0 |
| + SPY / QQQ / SOXX after-hours betas | +0.2479 | −0.0038 |
| + the 46 after-hours returns as a second input | +0.2411 | −0.0106 |
| + news, through the same weights (η_s per stock) | +0.2511 | −0.0006 ± 0.0013 |
| + news of another day (control) | +0.2529 | +0.0012 ± 0.0008 |
| + news, with its own weights | +0.2450 | −0.0067 ± 0.0020 |
| + after-hours clipped at ±5 sd, shared weights | +0.2499 | −0.0019 ± 0.0013 |

Neither the after-hours prices nor the news direction add anything on top of the futures gap at 08:45. The SPY after-hours return correlates +0.18 with the day's mean target, but the futures open already prices it. A test scoring of the press releases of the US peers (8-K item 2.02, 6-K; `demo/fetch_peers.py edgar_docs`) gave P(positive) ≈ 0.9 for almost every release, so they were not used.

### r8: the edge of the relation over the ridge was the penalty of the pre-open betas

The per-stock ridge penalizes the 4 pre-open betas with the same α as the 46 US coefficients (α ≈ 1), while the network leaves its betas free. Scaling the pre-open columns by 10 divides their penalty by 100:

| Model (target g − ĝ of E3b) | 100 stocks: R² | vs pre-open + signed | 50-stock subset: vs |
|---|---|---|---|
| pre-open + ridge on 46 per stock | +0.2341 | −0.0177 ± 0.0059 | −0.0176 |
| pre-open ×10 + ridge on 46 per stock | +0.2540 | +0.0023 ± 0.0033 | +0.0034 |
| pre-open ×100 + ridge on 46 per stock | +0.2537 | +0.0019 ± 0.0031 | +0.0030 |
| pre-open + signed | +0.2517 | 0 | 0 |

With a fair penalty the ridge equals the learned relation. The same applies to E3b, where the basket beta of the networks is also free: there, too, the per-stock ridge on 46 (+0.1009) equals basket + signed (+0.1007).

### A new gap base: ĝ with the pre-open features

The gap base ĝ is from now on the walk-forward ridge on the 52 base features plus 4 pre-open features (`stock_overnight.preopen_features`): the Nikkei and KOSPI open gaps, the futures gap at 08:45, and the futures gap times each stock's 60-day gap beta on it. Its R² over 2022–2026 rises from about 0.24 to 0.478. The new target is g − ĝ_pre (sd 0.0108).

**r7, r9, r11, r12 on g − ĝ_pre:**

| Model | 100 stocks: R² | vs pre-open + ridge on 46 | 50-stock subset: R² |
|---|---|---|---|
| basket beta per stock (baseline) | +0.0065 ± 0.0037 | −0.0120 ± 0.0026 | +0.0011 |
| pre-open betas per stock | +0.0075 | −0.0110 ± 0.0018 | +0.0041 |
| ridge per stock on 46 (baseline) | +0.0146 | −0.0039 ± 0.0010 | +0.0114 |
| **pre-open + ridge on 46 per stock** | **+0.0185 ± 0.0037** | 0 | +0.0166 |
| pre-open ×10 + ridge on 46 per stock | +0.0140 | — | +0.0118 |
| pre-open + signed relation | +0.0115 ± 0.0033 | −0.0070 ± 0.0029 | +0.0135 |
| signed relation, no betas | +0.0127 | — | +0.0090 |
| one ridge per industry (24) | +0.0138 | — | +0.0096 |
| mean of per-stock (×10) and per-industry | +0.0162 | — | +0.0133 |
| + US returns of t − 1 | +0.0136 | −0.0049 ± 0.0016 | +0.0109 |
| + Asian stock gaps | +0.0193 | +0.0008 ± 0.0011 | +0.0183 |
| + BKX, CL=F, FXI | +0.0195 | +0.0010 ± 0.0004 | +0.0177 |
| + both | +0.0200 | +0.0015 ± 0.0011 | +0.0191 |
| + both, Asian gaps of another day (control) | +0.0175 | −0.0011 ± 0.0009 | +0.0150 |

On the new target the shrinkage of the pre-open betas helps (×10 is worse), the learned relation stays below the per-stock ridge, and sharing coefficients by industry does not help. More inputs add at most about 0.001.

**r6, r10: the Taiwan news probe of E2h on the new target** (`demo/news_probe.py struct pre [plusus]`, 26,258 single-item stock-days):

| Representation | g − ĝ (E2h) | g − ĝ_pre | g − ĝ_pre − US ridge | \|g\| − \|g\|_hat (E2h) | with ĝ_pre |
|---|---|---|---|---|---|
| Qwen2.5 3 probabilities (E2) | +0.0020 | +0.0015 | +0.0017 | −0.0002 | +0.0003 |
| Qwen3.5 direction | +0.0031 | +0.0028 | +0.0024 | +0.0004 | +0.0004 |
| Qwen3.5 fields 24 | +0.0027 | +0.0036 | +0.0037 | +0.0035 | +0.0027 |
| summary bge PCA16 | +0.0034 | +0.0036 | +0.0032 | +0.0063 | +0.0021 |
| **fields 24 + summary PCA16** | +0.0053 | **+0.0057 ± 0.0012** | +0.0041 ± 0.0011 | +0.0054 | +0.0022 |

The Taiwan news keeps its signal on the direction of the gap with the new base, and it mostly adds to the US part (the per-stock ridge on pre-open + 46 US returns, removed in the third column; its R² on these days is +0.0191). The signal on the size of the gap mostly goes: |ĝ_pre| already holds it. The e = r − μ column stays at about 0 for all representations.

**E2j step 0 and E2i with ĝ_pre** (`demo/stock_identity_news.py gap pre`, `demo/stock_id_probe.py pre`; logs `demo/runs/stock_identity_news_gap_pre.log`, `demo/runs/stock_id_probe_pre.log`). The gap ridge now also gets the 4 pre-open features; its R² of g is .479–.482 for all news inputs. The return ridge (52 + gap block), walk-forward 2022–2026:

| Gap ridge input | 100 stocks: R² | Score | cross part | vs E2 gap block: Score | 50-stock subset: Score |
|---|---|---|---|---|---|
| always down (trivial baseline) | — | +7.8e-4 ± 2.4e-4 | 0 | | +3.4e-4 ± 2.3e-4 |
| E2 gap block, no pre-open (reference of E2j) | +0.0095 | +14.5e-4 ± 2.0e-4 | +2.6e-4 | | |
| price only (pre-open) | +0.0097 | +14.7e-4 ± 2.0e-4 | +3.2e-4 ± 0.8e-4 | −0.2e-4 ± 0.2e-4 | +11.0e-4 ± 1.9e-4 |
| E2 (pre-open) | +0.0098 | +14.8e-4 ± 2.0e-4 | +3.1e-4 ± 0.7e-4 | 0 | +11.1e-4 ± 1.9e-4 |
| struct (pre-open) | +0.0097 | +14.7e-4 ± 2.0e-4 | +3.2e-4 ± 0.7e-4 | −0.1e-4 ± 0.1e-4 | +10.9e-4 ± 1.9e-4 |
| E2 + struct (pre-open) | +0.0098 | +14.7e-4 ± 2.0e-4 | +3.1e-4 ± 0.7e-4 | −0.1e-4 ± 0.1e-4 | +11.0e-4 ± 1.9e-4 |

The doubled R² of the gap moves the Score by only +0.3e-4 (cross part +0.5e-4), and the news inputs of the gap ridge add nothing once the pre-open features are in it.

E2i on g − ĝ_pre: no similarity carries the direction of other stocks' news (all R² between −0.0002 and +0.0001, controls alike). On |g| − |g|_hat the equal weights keep +0.0037 ± 0.0009, as before; industry is no longer above its permuted control (+0.0005 against +0.0004).

## E4: raising the sign score

From here on the goal is the Score itself: the mean of sign(μ) × r, with r the open-to-close log return. The reference is the return ridge of E2j step 0 with ĝ_pre: the 52 base features plus the gap block of ĝ_pre (+14.7e-4 on all 100 stocks).

### E4a: the parts of the gap forecast as separate inputs (`demo/stock_score_gap.py`)

The gap forecast is split into four parts, all out-of-sample on every row: on Y from the fits before Y, on each earlier year from the fit without that year.
- **P:** ĝ from the 52 price features.
- **O = ĝ_pre − P:** the pre-open part.
- **U:** the per-stock ridge of g − ĝ_pre on the pre-open betas and the 46 US returns (E3c r7).
- **N:** a pooled ridge of g − ĝ_pre − U on the 43 own struct news columns (E2j).

The hypothesis is that the intraday reversal differs by the source of the gap. The R² of g is .2369 for P, .4794 for ĝ_pre, .4884 with U and .4897 with N. Their correlations with r over 2022–2026 are:

| Part | g (realized, not known at 08:50) | P | O | U | N | g − ĝ_pre − U − N |
|---|---|---|---|---|---|---|
| corr with r | −0.070 | −0.063 | −0.018 | −0.027 | +0.005 | −0.047 |

Return ridge, walk-forward 2022–2026, paired against the reference (`demo/runs/stock_score_gap.log`):

| Gap input of the return ridge | 100 stocks: R² | Score | cross part | vs ref: Score | 50-stock subset: Score | 2025–26: Score |
|---|---|---|---|---|---|---|
| always down (trivial baseline) | — | +7.8e-4 ± 2.4e-4 | 0 | | +3.4e-4 ± 2.3e-4 | +11.0e-4 ± 4.7e-4 |
| ref: gap block of ĝ_pre | +0.0097 | **+14.7e-4 ± 2.0e-4** | +3.2e-4 ± 0.8e-4 | 0 | +11.0e-4 | +19.4e-4 |
| gap block of ĝ_pre + U + N | +0.0096 | +14.5e-4 | +3.2e-4 | −0.1e-4 ± 0.2e-4 | +10.7e-4 | +19.4e-4 |
| parts P, O, U, N (value, day mean) + \|total\| | +0.0090 | +14.2e-4 | +2.7e-4 | −0.5e-4 ± 0.4e-4 | +10.5e-4 | +18.4e-4 |
| parts, value − day mean | +0.0084 | +13.6e-4 | +2.9e-4 | −1.0e-4 ± 0.6e-4 | +10.5e-4 | +16.9e-4 |
| gap block of the total + total·\|total\| | +0.0077 | +14.0e-4 | +2.9e-4 | −0.7e-4 ± 0.6e-4 | +10.2e-4 | +18.4e-4 |
| parts + the 4 raw pre-open features | +0.0084 | +13.8e-4 | +2.7e-4 | −0.9e-4 ± 0.4e-4 | +10.3e-4 | +17.5e-4 |

- **The predictable part of the gap does not revert much.** Most of the reversal (corr −0.070 of the realized g) lies in the part that no pre-open input forecasts (−0.047). The pre-open, US and news parts are only weakly related to r (−0.018, −0.027, +0.005).
- **Splitting the gap block costs Score.** The separate day-mean columns are the same for all stocks of a day, and the ridge overfits them, as with the market news columns in E2j. Value − day mean spans the same space as value and day mean, but its result differs (−1.0e-4), because the ridge penalty is not invariant to that change of basis.
- **The gap route to the Score is closed.** Forecasting the gap better does not reach r.

### E4b: calendar effects, stock intercepts and robust targets (`demo/stock_score_calendar.py`)

The always-down baseline lives on the negative mean of r. This round therefore tests day-level calendar effects, which act on the market part of the Score, and stock intercepts, which act on the cross part. The calendar flags are:
- day of week;
- month;
- the last and the first two trading days of a month;
- the futures settlement day (third Wednesday) and the day after it;
- the trading days before and after a break of 4 or more calendar days.

Before 2022 the mean r on the flagged days ranges from −27e-4 (last day of the month) to +21e-4 (February), against −6.6e-4 overall. There are about 450 days per year, so a flag that covers 5% of the days has about 45 days per year.

The flags enter in two ways:
- **Joint:** columns of the reference ridge, under its one shared α.
- **Residual:** a second ridge on the out-of-sample residual of the reference, with its own α chosen on Y − 1 (grid up to 1e10, which means no effect).

Constant columns on the fit rows, such as the one-hot of a stock that is not yet listed, get weight 0. The robust targets clip r at its 1%/99% or 5%/95% training quantiles; α is still chosen on the MSE of the raw r. Walk-forward 2022–2026 (`demo/runs/stock_score_calendar.log`):

| Version | 100 stocks: R² | Score | cross part | vs ref: Score | 50-stock subset: Score | 2025–26: Score |
|---|---|---|---|---|---|---|
| always down (trivial baseline) | — | +7.8e-4 ± 2.4e-4 | 0 | | +3.4e-4 | +11.0e-4 |
| ref: 52 + gap block of ĝ_pre | +0.0097 | **+14.7e-4 ± 2.0e-4** | +3.2e-4 ± 0.8e-4 | 0 | +11.0e-4 | +19.4e-4 |
| + stock one-hot | +0.0080 | +13.8e-4 | +2.6e-4 | −0.9e-4 ± 0.4e-4 | +10.1e-4 | +18.6e-4 |
| + day of week | +0.0094 | +14.9e-4 | +3.6e-4 | +0.2e-4 ± 0.4e-4 | +10.9e-4 | +19.5e-4 |
| + calendar | +0.0102 | +13.3e-4 | +2.5e-4 | −1.4e-4 ± 0.8e-4 | +10.0e-4 | +18.6e-4 |
| + calendar + stock one-hot | +0.0100 | +13.3e-4 | +2.5e-4 | −1.4e-4 ± 0.8e-4 | +9.8e-4 | +19.0e-4 |
| residual ridge on calendar + stock | +0.0101 | +14.7e-4 | +2.9e-4 | +0.1e-4 ± 0.3e-4 | +11.1e-4 | +19.2e-4 |
| residual ridge on calendar | +0.0104 | +14.7e-4 | +2.8e-4 | −0.0e-4 ± 0.3e-4 | +10.9e-4 | +19.2e-4 |
| target winsorized 1% | +0.0094 | +14.6e-4 | +3.3e-4 | −0.1e-4 ± 0.2e-4 | +10.9e-4 | +19.2e-4 |
| target winsorized 5% | +0.0095 | +14.3e-4 | +3.5e-4 | −0.3e-4 ± 0.4e-4 | +10.5e-4 | +19.2e-4 |

- **The calendar raises R² but not the Score.** The residual ridge on the calendar gives R² +0.0008 ± 0.0004. Its α is 1e10 (no effect) in 2022 and 1e5–1e6 later. It flips only 2.8% of the signs, and these flips do not pay. As joint columns the calendar flips 13% of the signs and costs −1.4e-4: under the shared α the day-level flags overfit the market part.
- **Stock intercepts hurt** (−0.9e-4 ± 0.4e-4). The mean open-to-close return of a stock does not persist from one period to the next.
- **Clipping the target does not help.** It shifts weight from the market part to the cross part (cross part +3.2e-4 → +3.5e-4 at 5%), but the total falls.

### E4c: losses aligned with the Score (`demo/stock_score_loss.py`)

Maximizing the Score, mean(sign(f) r), is a cost-sensitive classification: the label is sign(r) and the weight is |r|. The MSE ridge targets E[r], whose sign is the Bayes decision, but it spends its capacity on the size of r. The inputs are those of the reference (52 + gap block of ĝ_pre), standardized on the training rows. The models:
- **Linear models, by full-batch L-BFGS.** The L2 weight is chosen on Y − 1 by the validation loss, then the model is refitted on all years before Y.
  - weighted logistic: |r| · log(1 + exp(−sign(r) f));
  - weighted hinge: |r| · max(0, 1 − sign(r) f);
  - soft Score: −mean(tanh(f) r).
- **MLP:** one hidden layer of 64, GELU, dropout 0.2, AdamW. Epochs are chosen on Y − 1 (patience 3), the model is refitted before Y, and 3 seeds are averaged. It is trained on the MSE or on the weighted logistic loss.

The R² of the classification losses is meaningless (f is not on the scale of r) and is left out. Walk-forward 2022–2026 (`demo/runs/stock_score_loss.log`):

| Model | 100 stocks: Score | cross part | vs ridge: Score | flips | 50-stock subset: Score | vs ridge | 2025–26: Score | up calls |
|---|---|---|---|---|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | 0 | | | +3.4e-4 | | +11.0e-4 | 0% |
| ref: MSE ridge | +14.7e-4 ± 2.0e-4 | +3.2e-4 ± 0.8e-4 | 0 | | +11.0e-4 | 0 | +19.4e-4 | 37.6% |
| weighted logistic | +15.2e-4 | +3.7e-4 | +0.6e-4 ± 0.6e-4 | 8.3% | +11.5e-4 | +0.4e-4 ± 0.7e-4 | +19.7e-4 | 40.1% |
| weighted hinge | +15.3e-4 | +4.2e-4 | +0.6e-4 ± 0.8e-4 | 11.6% | +11.8e-4 | +0.7e-4 ± 0.9e-4 | +19.8e-4 | 38.2% |
| **soft Score** | **+16.1e-4 ± 2.0e-4** | +4.3e-4 | **+1.4e-4 ± 1.1e-4** | 14.6% | +12.7e-4 | +1.7e-4 ± 1.1e-4 | +20.2e-4 | 41.8% |
| MSE MLP | +15.0e-4 | +4.4e-4 | +0.3e-4 ± 1.0e-4 | 16.2% | +12.5e-4 | +1.5e-4 ± 1.0e-4 | +21.2e-4 | 41.8% |
| weighted-logistic MLP | +15.7e-4 ± 1.9e-4 | **+5.3e-4 ± 0.7e-4** | +1.0e-4 ± 1.0e-4 | 13.1% | +12.6e-4 | +1.6e-4 ± 1.1e-4 | **+21.9e-4** | 42.7% |
| mean of ridge and weighted logistic | +14.8e-4 | +3.1e-4 | +0.2e-4 ± 0.3e-4 | 4.1% | +11.0e-4 | −0.0e-4 ± 0.4e-4 | +19.1e-4 | 38.7% |

- **Every Score-aligned loss beats the MSE ridge, in every subset.** The gains are +0.6e-4 to +1.4e-4, and no single one exceeds 1.5 standard errors. They come mostly from the cross part: +4.2e-4 to +5.3e-4, against +3.2e-4 for the ridge.
- **The weighted logistic picks the largest L2 of the grid (0.1) in every fold**, so the grid should be extended upward. The soft Score picks small L2 (0 to 1e-2). The MLPs stop after 1 to 5 epochs.

**Round r2** (`demo/stock_score_loss.py r2`, `demo/runs/stock_score_loss_r2.log`):
- The weighted logistic gets L2 up to 10. It picks 0.1–0.3, so the grid edge of E4c did not limit it.
- The soft Score and the weighted logistic are trained as MLPs with 5 seeds.
- The soft Score is also fitted on the inputs plus their day-demeaned copies.
- The ensemble takes the mean of the soft Score, the soft-Score MLP and the weighted-logistic MLP, each divided by its sd on the fold's test rows.

| Model | 100 stocks: Score | cross part | vs ridge: Score | 50-stock subset: Score | vs ridge | 2025–26: Score | vs ridge | up calls |
|---|---|---|---|---|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | 0 | | +3.4e-4 | | +11.0e-4 | | 0% |
| ref: MSE ridge | +14.7e-4 ± 2.0e-4 | +3.2e-4 | 0 | +11.0e-4 | 0 | +19.4e-4 | 0 | 37.6% |
| soft Score (linear) | +16.1e-4 | +4.3e-4 | +1.4e-4 ± 1.1e-4 | +12.7e-4 | +1.7e-4 ± 1.1e-4 | +20.2e-4 | +0.8e-4 ± 1.6e-4 | 41.8% |
| weighted logistic, L2 up to 10 | +15.1e-4 | +3.8e-4 | +0.4e-4 ± 0.6e-4 | +11.2e-4 | +0.1e-4 ± 0.7e-4 | +19.7e-4 | +0.3e-4 ± 1.3e-4 | 39.4% |
| soft-Score MLP (5 seeds) | +15.8e-4 | **+6.1e-4** | +1.2e-4 ± 1.1e-4 | +13.2e-4 | +2.1e-4 ± 1.2e-4 | +21.2e-4 | +1.8e-4 ± 2.1e-4 | 48.5% |
| weighted-logistic MLP (5 seeds) | +16.0e-4 | +5.4e-4 | +1.3e-4 ± 1.1e-4 | +13.1e-4 | +2.1e-4 ± 1.2e-4 | **+22.7e-4** | +3.3e-4 ± 2.5e-4 | 43.9% |
| soft Score, + day-demeaned inputs | +15.7e-4 | +3.9e-4 | +1.0e-4 ± 1.2e-4 | +12.5e-4 | +1.5e-4 ± 1.2e-4 | +18.8e-4 | −0.5e-4 ± 1.9e-4 | 45.1% |
| **ensemble: soft Score + soft MLP + w-logistic MLP** | **+16.6e-4 ± 1.9e-4** | +5.2e-4 ± 0.7e-4 | **+2.0e-4 ± 0.9e-4** | **+13.8e-4** | **+2.8e-4 ± 0.9e-4** | +22.0e-4 | +2.6e-4 ± 1.5e-4 | 44.2% |

- **The ensemble is the first change since E2e that beats the reference ridge by more than 2 standard errors**: +2.0e-4 ± 0.9e-4 on 100 stocks and +2.8e-4 ± 0.9e-4 on the 50-stock subset. Averaging the three members lowers the standard error of the paired difference (0.9 against 1.1–1.2), since their flips only partly overlap.
- The Score-aligned models call more stocks up (42–49% against 38%). They move weight from the always-down tilt to the cross part (+5.2e-4 to +6.1e-4 against +3.2e-4).
- The day-demeaned copies do not help the linear soft Score.

**Round r3, the ensemble scaled up** (`demo/stock_score_loss.py r3`, `demo/runs/stock_score_loss_r3.log`). Each member set is an ensemble of the soft Score (linear), the soft-Score MLP and the weighted-logistic MLP:
- **base:** the inputs of the reference, MLPs with 10 seeds;
- **rich:** also the 4 raw pre-open features and the 43 own struct news columns, MLPs with 5 seeds;
- **deep:** MLPs with two hidden layers of 128, 5 seeds;
- **all:** the mean of the three ensembles.

| Ensemble | 100 stocks: Score | cross part | vs ridge: Score | 50-stock subset: Score | vs ridge | 2025–26: vs ridge | up calls |
|---|---|---|---|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | 0 | | +3.4e-4 | | | 0% |
| ref: MSE ridge | +14.7e-4 ± 2.0e-4 | +3.2e-4 | 0 | +11.0e-4 | 0 | 0 | 37.6% |
| base (10 seeds) | +16.6e-4 | +5.2e-4 | +1.9e-4 ± 0.9e-4 | +13.6e-4 | +2.6e-4 ± 0.9e-4 | +2.4e-4 ± 1.5e-4 | 44.5% |
| rich | +16.0e-4 | +5.8e-4 | +1.4e-4 ± 0.8e-4 | +13.2e-4 | +2.2e-4 ± 0.9e-4 | +0.5e-4 ± 1.4e-4 | 46.9% |
| deep | +16.7e-4 | +5.2e-4 | +2.1e-4 ± 0.9e-4 | +13.6e-4 | +2.5e-4 ± 0.9e-4 | +2.5e-4 ± 1.5e-4 | 44.0% |
| all three | **+16.7e-4 ± 1.9e-4** | +5.5e-4 | **+2.1e-4 ± 0.8e-4** | **+13.8e-4** | **+2.7e-4 ± 0.9e-4** | +2.0e-4 ± 1.5e-4 | 45.1% |

- **The gain of r2 holds but does not grow.** Ten seeds, a deeper MLP and the mean of the three ensembles all stay at +1.9e-4 to +2.1e-4, about 2.5 standard errors. The limit is the inputs, not the variance of the members.
- **The news and the raw pre-open columns hurt the Score-aligned models**, as they hurt the ridge (E2j, E4a): the linear soft Score falls from +1.4e-4 to −0.0e-4.

**Round r4, wider price inputs** (`demo/stock_score_loss.py r4`, `demo/runs/stock_score_loss_r4.log`). The r3 base ensemble (5 seeds) is fitted on wider inputs:
- **poly230:** the reference inputs plus the other 195 polynomial-fit features of `stock_poly.features` (all 230 instead of the 35 of degree 0);
- **poly230 + extra:** also the volume, flows, technical and global groups of `stock_extra` (73).

A bug in the log: the mean of the three ensembles was written into the row "poly230 + extra: MSE ridge", so the MSE ridge on poly230 + extra was not reported. The row "all: ensemble" is empty (NaN, scored as always down).

| Inputs | 100 stocks: Score | vs ridge: Score | 50-stock subset: vs ridge | 2025–26: vs ridge |
|---|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | | | |
| ref: MSE ridge, reference inputs | +14.7e-4 ± 2.0e-4 | 0 | 0 | 0 |
| MSE ridge, poly230 | +13.8e-4 | −0.9e-4 ± 1.1e-4 | −0.6e-4 ± 1.1e-4 | −0.8e-4 ± 2.2e-4 |
| **ensemble, reference inputs** | **+16.6e-4 ± 1.9e-4** | **+2.0e-4 ± 0.9e-4** | **+2.8e-4 ± 0.9e-4** | +2.6e-4 ± 1.5e-4 |
| ensemble, poly230 | +14.1e-4 | −0.6e-4 ± 1.5e-4 | +1.0e-4 ± 1.5e-4 | +0.3e-4 ± 2.7e-4 |
| ensemble, poly230 + extra | +14.2e-4 | −0.4e-4 ± 1.4e-4 | −0.2e-4 ± 1.4e-4 | −1.6e-4 ± 2.5e-4 |
| mean of the three ensembles | +16.3e-4 | +1.6e-4 ± 1.0e-4 | +2.7e-4 ± 1.1e-4 | +1.4e-4 ± 2.0e-4 |

The wider inputs hurt the Score-aligned models as much as they hurt the ridge: every member loses 1–3e-4 against its version on the reference inputs. The 52 base features (35 of degree 0, 12 range, 5 US) remain the best input set for both losses.

**Round r5, recency weights** (`demo/stock_score_loss.py r5`, `demo/runs/stock_score_loss_r5.log`). Every training row of the members gets the weight 2^(−age / half-life), with the age counted in trading days before the last fit row. The weights are normalized to mean 1.

| Ensemble | 100 stocks: Score | vs ridge: Score | 50-stock subset: vs ridge | 2025–26: vs ridge |
|---|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | | | |
| ref: MSE ridge | +14.7e-4 ± 2.0e-4 | 0 | 0 | 0 |
| **no weights (r3 base, 5 seeds)** | **+16.6e-4** | **+2.0e-4 ± 0.9e-4** | **+2.8e-4 ± 0.9e-4** | +2.6e-4 ± 1.5e-4 |
| half-life 250 days | +15.9e-4 | +1.3e-4 ± 1.0e-4 | +1.7e-4 ± 1.0e-4 | +1.3e-4 ± 1.7e-4 |
| half-life 500 days | +16.4e-4 | +1.7e-4 ± 0.9e-4 | +2.4e-4 ± 0.9e-4 | +2.1e-4 ± 1.6e-4 |
| half-life 1000 days | +16.1e-4 | +1.4e-4 ± 0.9e-4 | +2.3e-4 ± 0.9e-4 | +2.4e-4 ± 1.6e-4 |

Weighting recent years more does not help, even in 2025–26. Shorter half-lives are worse, so the Score-aligned models need all the training years.

**Round r6, the base ensemble by year and the decision rule** (`demo/stock_score_loss.py r6`: the r3 base ensemble alone, 10 seeds, forecasts saved to `demo/runs/e4_r6_pred.npz`; analysis in `demo/runs/e4_rules.log`). Score by year:

| Year | always down (baseline) | MSE ridge | ensemble | ensemble − ridge | 50-stock subset: ensemble − ridge |
|---|---|---|---|---|---|
| 2022 | +6.7e-4 ± 5.5e-4 | +13.9e-4 ± 4.3e-4 | +15.3e-4 ± 4.1e-4 | +1.4e-4 ± 1.6e-4 | +1.4e-4 ± 1.6e-4 |
| 2023 | +3.3e-4 ± 3.9e-4 | +7.8e-4 ± 3.3e-4 | +12.2e-4 ± 2.8e-4 | +4.4e-4 ± 1.7e-4 | +5.4e-4 ± 1.8e-4 |
| 2024 | +8.2e-4 ± 4.7e-4 | +14.4e-4 ± 3.8e-4 | +13.4e-4 ± 3.7e-4 | −0.9e-4 ± 2.1e-4 | −0.2e-4 ± 2.2e-4 |
| 2025 | +9.0e-4 ± 5.3e-4 | +16.7e-4 ± 4.6e-4 | +17.6e-4 ± 4.1e-4 | +0.8e-4 ± 2.0e-4 | +1.3e-4 ± 2.0e-4 |
| 2026 (to Aug) | +13.9e-4 ± 8.8e-4 | +23.5e-4 ± 7.5e-4 | +28.3e-4 ± 7.5e-4 | +4.9e-4 ± 2.3e-4 | +6.4e-4 ± 2.3e-4 |
| 2022–2026 | +7.8e-4 ± 2.4e-4 | +14.7e-4 ± 2.0e-4 | **+16.6e-4 ± 1.9e-4** | **+1.9e-4 ± 0.9e-4** | **+2.6e-4 ± 0.9e-4** |

The ensemble beats the ridge in four of the five years, and its loss in 2024 is within the error.

The decision rule tests whether the ensemble should only rank the stocks within the day while the ridge gives the market tilt. The rule is: up iff rank_t(ensemble) + λ · (day mean of the ridge) / sd + c > 0, with rank_t the ensemble minus its day mean, divided by its sd. (λ, c) is chosen on the earlier evaluation years (expanding window; 2022 keeps the ensemble sign). It picks λ = 1, c = 0 from 2024 on, and it is worse than the ensemble's own sign: −1.2e-4 ± 0.8e-4 on 100 stocks, −1.1e-4 ± 0.7e-4 on the 50-stock subset. The best (λ, c) of the grid in hindsight scores +15.8e-4, still below +16.6e-4. The Score-aligned models therefore learn a better market tilt than the ridge, and their sign is the best decision rule found.

### E4h: a set model over the day's stocks with the Score-aligned losses (`demo/stock_score_set.py`)

The gain of E4c–E4g lies in the ranking of the stocks within a day. In the set model each stock also sees the other stocks of the same day. Per day, the (S, k) reference inputs pass through:
- Linear(k, 64) + GELU;
- one pre-norm Transformer encoder layer over the stocks (4 heads, feed-forward 128, dropout 0.2; stocks not usable that day are masked);
- Linear(64, 1).

It is trained on the soft Score or on the weighted logistic loss, in batches of 16 days. Epochs are chosen on Y − 1 (1–14), the model is refitted before Y, and 5 seeds are averaged. The last row adds the two set models to the three members of the E4 r6 base ensemble. Walk-forward 2022–2026 (`demo/runs/stock_score_set.log`):

| Model | 100 stocks: Score | cross part | vs ridge | vs r6 ensemble | 50-stock subset: Score | vs ridge | vs r6 ensemble | 2025–26: Score |
|---|---|---|---|---|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | 0 | | | +3.4e-4 | | | +11.0e-4 |
| ref: MSE ridge | +14.7e-4 ± 2.0e-4 | +3.2e-4 | 0 | | +11.0e-4 | 0 | | +19.4e-4 |
| E4 r6 base ensemble | +16.6e-4 ± 1.9e-4 | +5.2e-4 | +1.9e-4 ± 0.9e-4 | 0 | +13.6e-4 | +2.6e-4 ± 0.9e-4 | 0 | +21.8e-4 |
| set model, soft Score | **+17.5e-4 ± 2.1e-4** | +5.5e-4 | +2.8e-4 ± 1.2e-4 | +0.9e-4 ± 0.9e-4 | +13.8e-4 | +2.8e-4 ± 1.3e-4 | +0.2e-4 ± 0.9e-4 | **+23.0e-4** |
| set model, weighted logistic | +16.5e-4 | +5.6e-4 | +1.9e-4 ± 1.3e-4 | −0.0e-4 ± 1.0e-4 | +13.6e-4 | +2.6e-4 ± 1.3e-4 | −0.0e-4 ± 1.0e-4 | +22.2e-4 |
| ensemble of the two set models | +17.3e-4 | +5.4e-4 | +2.7e-4 ± 1.2e-4 | +0.8e-4 ± 0.8e-4 | +14.0e-4 | +3.0e-4 ± 1.2e-4 | +0.4e-4 ± 0.8e-4 | +23.0e-4 |
| **r6 base members + set models** | +17.4e-4 ± 2.0e-4 | +5.6e-4 | **+2.7e-4 ± 0.9e-4** | **+0.8e-4 ± 0.4e-4** | **+14.3e-4** | **+3.3e-4 ± 1.0e-4** | +0.7e-4 ± 0.5e-4 | +22.8e-4 |

- **The set models add to the per-stock ensemble.** With them the ensemble beats the ridge by 3.0 standard errors on 100 stocks (+2.7e-4 ± 0.9e-4) and by 3.3 on the 50-stock subset (+3.3e-4 ± 1.0e-4). The gain over the r6 ensemble alone is +0.8e-4 ± 0.4e-4.
- The set model with the soft Score is the best single model so far (+17.5e-4), but its error against the ridge is larger (1.2e-4) than that of the ensembles.

**Round r2 of the set model** (`demo/stock_score_set.py r2`, `demo/runs/stock_score_set_r2.log`, forecasts in `demo/runs/e4h_r2_pred.npz`). Two configurations are compared: one encoder layer with 10 seeds, and two layers with 5 seeds.

| Model | 100 stocks: Score | vs ridge | vs r6 ensemble | 50-stock subset: vs ridge | 2025–26: vs ridge |
|---|---|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | | | | |
| ref: MSE ridge | +14.7e-4 ± 2.0e-4 | 0 | | 0 | 0 |
| E4 r6 base ensemble | +16.6e-4 | +1.9e-4 ± 0.9e-4 | 0 | +2.6e-4 ± 0.9e-4 | +2.4e-4 ± 1.5e-4 |
| 1 layer, 10 seeds: soft Score | **+17.7e-4 ± 2.1e-4** | +3.1e-4 ± 1.2e-4 | +1.2e-4 ± 0.8e-4 | +3.2e-4 ± 1.2e-4 | +4.1e-4 ± 2.2e-4 |
| 1 layer, 10 seeds: weighted logistic | +16.4e-4 | +1.8e-4 ± 1.2e-4 | −0.1e-4 ± 0.9e-4 | +2.3e-4 ± 1.1e-4 | +2.0e-4 ± 2.3e-4 |
| 2 layers, 5 seeds: soft Score | +16.1e-4 | +1.5e-4 ± 1.4e-4 | −0.4e-4 ± 1.1e-4 | +1.5e-4 ± 1.4e-4 | +0.9e-4 ± 2.3e-4 |
| 2 layers, 5 seeds: weighted logistic | +17.5e-4 | +2.9e-4 ± 1.4e-4 | +1.0e-4 ± 1.1e-4 | +3.7e-4 ± 1.4e-4 | +2.6e-4 ± 2.9e-4 |
| **r6 members + 1-layer set models** | +17.5e-4 ± 2.0e-4 | **+2.9e-4 ± 0.9e-4** | **+1.0e-4 ± 0.4e-4** | **+3.4e-4 ± 0.9e-4** | +3.5e-4 ± 1.7e-4 |
| r6 members + 2-layer set models | +17.4e-4 | +2.7e-4 ± 1.0e-4 | +0.8e-4 ± 0.5e-4 | +3.1e-4 ± 1.0e-4 | +2.9e-4 ± 1.9e-4 |
| r6 members + all set models | +17.4e-4 | +2.8e-4 ± 1.0e-4 | +0.9e-4 ± 0.5e-4 | +3.4e-4 ± 1.0e-4 | +2.7e-4 ± 1.9e-4 |

- **The first run reproduces with 10 seeds.** The ensemble with the 1-layer set models beats the ridge by 3.2 standard errors (+2.9e-4 ± 0.9e-4) and the r6 ensemble by 2.5 (+1.0e-4 ± 0.4e-4).
- **A second layer does not help.** The single 2-layer models vary more: the soft Score falls and the weighted logistic rises.
- Whether the gain comes from the attention over the stocks or only from adding a different kind of member to the ensemble is not settled here. A set model without attention, which sees only the day's mean of the stock representations, is the control for that.

### E4i: attention ablation of the set model (`demo/stock_score_target.py attn`)

The encoder layer of the set model is replaced by one of two blocks:
- **day mean (pool):** each stock sees its own representation and the masked mean of the day's representations, Linear(128, 64) + GELU + dropout on [h, mean_day h];
- **own only:** Linear(64, 64) + GELU + dropout on h.

Everything else is unchanged: target r, batches of 16 days, 5 seeds. The models with attention are the 1-layer set models of E4h r2 (10 seeds).

In the paired comparisons below, the ensembles divide each member by its sd over all evaluation rows instead of per fold, so "r6 + attention" reads +17.3e-4 here against +17.5e-4 in E4h r2. Walk-forward 2022–2026 (`demo/runs/stock_score_target_attn.log`, `demo/runs/e4i_attn_pred.npz`):

| Model | 100 stocks: Score | vs ridge | vs the same loss with attention | 50-stock subset: Score | vs the same loss with attention |
|---|---|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | | | +3.4e-4 | |
| ref: MSE ridge | +14.7e-4 ± 2.0e-4 | 0 | | +11.0e-4 | |
| attention, soft Score (E4h r2) | +17.7e-4 | +3.1e-4 ± 1.2e-4 | 0 | +14.2e-4 | 0 |
| day mean, soft Score | +17.0e-4 | +2.3e-4 ± 1.4e-4 | −0.8e-4 ± 0.9e-4 | +13.6e-4 | −0.7e-4 ± 0.9e-4 |
| own only, soft Score | +16.6e-4 | +2.0e-4 ± 1.3e-4 | −1.1e-4 ± 0.9e-4 | +13.9e-4 | −0.3e-4 ± 0.9e-4 |
| attention, weighted logistic (E4h r2) | +16.4e-4 | +1.8e-4 ± 1.2e-4 | 0 | +13.4e-4 | 0 |
| day mean, weighted logistic | +17.4e-4 | +2.7e-4 ± 1.1e-4 | +0.9e-4 ± 1.0e-4 | +14.4e-4 | +1.1e-4 ± 1.0e-4 |
| own only, weighted logistic | +15.5e-4 | +0.8e-4 ± 1.0e-4 | −1.0e-4 ± 1.0e-4 | +13.0e-4 | −0.3e-4 ± 1.0e-4 |
| r6 members + attention set models | +17.3e-4 ± 2.0e-4 | | 0 | +14.2e-4 | 0 |
| r6 members + day-mean set models | +16.9e-4 | | −0.4e-4 ± 0.5e-4 | +14.0e-4 | −0.3e-4 ± 0.5e-4 |
| r6 members + own-only set models | +16.1e-4 | | **−1.1e-4 ± 0.5e-4** | +13.4e-4 | −0.8e-4 ± 0.5e-4 |

- **What helps is that a stock sees the other stocks of its day, not the attention itself.** In the ensemble, the own-only models lose −1.1e-4 ± 0.5e-4 (2.2 standard errors) against the attention models. The day-mean models are within the error of them (−0.4e-4 ± 0.5e-4). The single models are noisier: the day mean is the best weighted-logistic model, and the attention the best soft-Score model.
- So most of the cross-stock information is the day's average state. A stock's position relative to the day's mean is enough, and a learned weighting of particular other stocks adds little.

### E4j: from the E2j iTransformer to the E4h set model, one change at a time (`demo/stock_score_itoken.py`)

The ladder runs from L0 to L7. Each step changes one thing against the step before:
- **L0:** the iT of E2j. The token is Linear(60-day window of [e, r, g, traded]) + Linear(the 55 day features and μ/se), with dropout 0.3, one pre-norm self-attention layer (2 heads, d 16, no mask, no feed-forward) and a zero-started head. The forecast is μ/sd(r) + head, trained with MSE.
- **L1:** the soft Score; the forecast is a·μ/sd(μ) + head.
- **L2:** forecasts r (no μ offset), with the usual head initialization.
- **L3:** e and μ are removed from the token.
- **L4:** the E4h block (GELU, masked encoder layer with feed-forward, dropout 0.2, 4 heads), still at d 16.
- **L5:** d 64.
- **L6:** the day features only (the E4h set model).
- **L7:** the window only.

All steps choose the epochs on Y − 1 (at least 1; E2j allowed 0 steps), refit before Y, and average 5 seeds (`demo/runs/stock_score_itoken.log`, `demo/runs/e4j_pred.npz`).

| Step | 100 stocks: Score | cross part | vs ridge | vs the step before | 50-stock subset: Score | vs the step before |
|---|---|---|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | 0 | | | +3.4e-4 | |
| ref: MSE ridge | +14.7e-4 ± 2.0e-4 | +3.2e-4 | 0 | | +11.0e-4 | |
| L0 iT as in E2j | +14.9e-4 | +3.6e-4 | +0.2e-4 ± 0.4e-4 | | +12.1e-4 | |
| L1 + soft Score | +14.3e-4 | +3.8e-4 | −0.4e-4 ± 1.0e-4 | −0.6e-4 ± 0.9e-4 | +11.9e-4 | −0.2e-4 ± 0.9e-4 |
| L2 + target r | +14.1e-4 | +3.6e-4 | −0.5e-4 ± 1.3e-4 | −0.2e-4 ± 1.0e-4 | +11.5e-4 | −0.4e-4 ± 0.9e-4 |
| L3 + token without e, μ | +14.7e-4 | +3.8e-4 | +0.1e-4 ± 1.3e-4 | +0.6e-4 ± 0.7e-4 | +12.0e-4 | +0.6e-4 ± 0.7e-4 |
| L4 + E4h block, d 16 | +14.6e-4 | +4.5e-4 | −0.1e-4 ± 1.3e-4 | −0.1e-4 ± 0.8e-4 | +12.4e-4 | +0.3e-4 ± 0.9e-4 |
| **L5 + d 64** | **+17.7e-4 ± 2.0e-4** | +5.2e-4 | **+3.1e-4 ± 1.4e-4** | **+3.2e-4 ± 1.0e-4** | **+14.2e-4** | **+1.8e-4 ± 1.0e-4** |
| L6 day features only (E4h set) | +17.5e-4 | +5.5e-4 | +2.8e-4 ± 1.2e-4 | −0.3e-4 ± 1.5e-4 | +13.8e-4 | −0.3e-4 ± 1.5e-4 |
| L7 window only | +8.5e-4 | +1.1e-4 | −6.2e-4 ± 2.2e-4 | −9.0e-4 ± 2.4e-4 | +5.5e-4 | −8.3e-4 ± 2.4e-4 |

- **The step that matters is the width (d 16 → 64): +3.2e-4 ± 1.0e-4.** At d 16 the soft Score, the direct target, the token without e and μ, and the E4h block each change the Score by less than one standard error. With the wider model the iT token (window + day features) does as well as the day features alone.
- **The 60-day window carries little on its own.** The window-only model is far below the ridge (+8.5e-4, 18.5% up calls). With the day features it adds nothing (L5 vs L6).
- **L0, the E2j iT with at least one epoch, is already at the ridge (+0.2e-4 ± 0.4e-4).** E2j fell slightly below the ridge because its selection could pick 0 steps.
- The ladder changes the loss before the width, so it does not show whether the width alone would help under the MSE. The 2 × 2 run (target × loss, d 64) answers that.

### E4i: target (r or the ridge residual) × loss (MSE or Score-aligned) (`demo/stock_score_target.py 2x2`)

Every cell uses the same network, inputs (the reference's 55), protocol (epochs on Y − 1, at least 1; refit; 5 seeds) and decision (sign of the final forecast). The cells are:
- **target r:** the network forecasts r, as r/sd(r) under the MSE and as f under a Score loss;
- **residual:** under the MSE the forecast is μ/sd(r) + network, a fixed offset, so the network fits e; under a Score loss it is a·μ/sd(μ) + network, with a learnable a started at 1. μ is the out-of-sample ridge forecast.

The residual cells are also run with a zero-started output layer, as in E2j. The networks are the MLP of E4c (64) and the 1-layer set model of E4h (d 64). Walk-forward 2022–2026 (`demo/runs/stock_score_target_2x2.log`, `demo/runs/e4i_2x2_pred.npz`):

| Cell | MLP: Score | MLP: vs ridge | set: Score | set: vs ridge | set, 50-stock subset: vs ridge |
|---|---|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | | | | |
| ref: MSE ridge | +14.7e-4 ± 2.0e-4 | 0 | | 0 | 0 |
| target r, MSE | +15.3e-4 | +0.7e-4 ± 1.0e-4 | +17.3e-4 | +2.6e-4 ± 1.5e-4 | +3.5e-4 ± 1.5e-4 |
| residual, MSE | +15.4e-4 | +0.7e-4 ± 0.9e-4 | +17.4e-4 | +2.7e-4 ± 1.3e-4 | +2.7e-4 ± 1.3e-4 |
| residual, MSE, zero head | +15.7e-4 | +1.0e-4 ± 0.5e-4 | +15.8e-4 | +1.1e-4 ± 0.5e-4 | +1.2e-4 ± 0.5e-4 |
| target r, soft Score | +15.8e-4 | +1.2e-4 ± 1.1e-4 | +17.5e-4 | +2.8e-4 ± 1.2e-4 | +2.8e-4 ± 1.3e-4 |
| residual, soft Score | **+16.7e-4** | **+2.1e-4 ± 1.0e-4** | +17.5e-4 | +2.8e-4 ± 1.1e-4 | +2.7e-4 ± 1.2e-4 |
| residual, soft Score, zero head | +16.4e-4 | +1.7e-4 ± 1.0e-4 | **+17.6e-4** | **+3.0e-4 ± 1.1e-4** | +3.3e-4 ± 1.1e-4 |
| target r, w-logistic | +16.0e-4 | +1.3e-4 ± 1.1e-4 | +16.5e-4 | +1.9e-4 ± 1.3e-4 | +2.6e-4 ± 1.3e-4 |
| residual, w-logistic | +16.2e-4 | +1.5e-4 ± 1.1e-4 | +15.5e-4 | +0.8e-4 ± 1.0e-4 | +0.9e-4 ± 1.1e-4 |
| residual, w-logistic, zero head | +15.3e-4 | +0.7e-4 ± 1.0e-4 | +15.1e-4 | +0.4e-4 ± 0.9e-4 | +0.6e-4 ± 1.0e-4 |

Paired contrasts within a network (100 stocks; 50-stock subset in brackets):

| Contrast | MLP | set model |
|---|---|---|
| soft Score − MSE (target r) | +0.5e-4 ± 0.8e-4 (+0.7 ± 0.9) | +0.2e-4 ± 1.2e-4 (−0.7 ± 1.2) |
| weighted logistic − MSE (target r) | +0.7e-4 ± 0.8e-4 (+0.7 ± 0.8) | −0.7e-4 ± 0.8e-4 (−0.9 ± 0.9) |
| residual − target r, MSE | +0.1e-4 ± 0.6e-4 (−0.3 ± 0.7) | +0.1e-4 ± 0.6e-4 (−0.8 ± 0.7) |
| residual − target r, soft Score | +0.9e-4 ± 0.6e-4 (+0.4 ± 0.7) | +0.0e-4 ± 0.6e-4 (−0.1 ± 0.6) |
| zero head − usual head, residual MSE | +0.3e-4 ± 0.7e-4 (+0.1 ± 0.8) | −1.6e-4 ± 1.1e-4 (−1.5 ± 1.1) |

- **Neither the target nor the loss decides the gain of the networks.** At the same network, forecasting r or the residual differs by at most 0.9e-4, and the Score losses differ from the MSE by at most 0.7e-4. None of these contrasts reaches 1.5 standard errors. The set model trained with the plain MSE on r (+2.6e-4) or on the residual (+2.7e-4) is as good as with the soft Score (+2.8e-4).
- **The network matters.** The set model is ahead of the MLP in every MSE cell (+17.3e-4 against +15.3e-4 on r). Together with E4j, where the width d 16 → 64 was the decisive step, and with the attention ablation, the gain comes from a wide enough network that sees the other stocks of the day.
- **The E2j recipe holds the set model back.** Under the MSE a zero-started head stays close to the ridge: 6% flips, −1.6e-4 ± 1.1e-4 against the usual head. E2j/E2f used that head, d 16 and a step selection that could pick 0 steps.
- **The Score losses do help the linear model.** The linear soft Score beats the ridge by +1.4e-4 ± 1.1e-4 (E4c), and all Score-aligned linear models gained in the cross part. For the MLP their gains are +0.5e-4 to +0.7e-4, within the error. The weighted logistic is the weakest loss for the set model.

### E4i: industry means as the pooled context (`demo/stock_score_target.py pool2`)

These are the day-mean models of the attention ablation, with the context changed to [h, day mean, industry mean] or to [h, industry mean]. The industry mean is the masked mean over the stock's TWSE / TPEx industry, the stock itself included (`demo/runs/stock_score_target_pool2.log`):

| Model | 100 stocks: Score | vs day mean, same loss | 50-stock subset: vs day mean |
|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | | |
| ref: MSE ridge | +14.7e-4 ± 2.0e-4 | | |
| day + industry mean, soft Score | +15.6e-4 | −1.4e-4 ± 1.0e-4 | −0.8e-4 ± 1.0e-4 |
| day + industry mean, weighted logistic | +15.5e-4 | −1.9e-4 ± 0.8e-4 | −2.1e-4 ± 0.8e-4 |
| industry mean, soft Score | +15.9e-4 | −1.0e-4 ± 0.9e-4 | −0.2e-4 ± 0.9e-4 |
| industry mean, weighted logistic | +16.2e-4 | −1.2e-4 ± 0.4e-4 | −1.1e-4 ± 0.4e-4 |
| r6 members + day + industry set models | +15.9e-4 | −0.9e-4 ± 0.4e-4 (vs r6 + day mean) | −0.9e-4 ± 0.4e-4 |
| r6 members + industry set models | +16.1e-4 | −0.8e-4 ± 0.4e-4 (vs r6 + day mean) | −0.9e-4 ± 0.4e-4 |

The industry mean hurts in every version. Many industries have only a few of the 100 stocks, so their mean is close to the stock's own representation and adds noise rather than context. The day mean over all stocks remains the useful context, as the industry routing of the news did not hold up either (E2i).

### E4j: width × loss of the set model (`demo/stock_score_itoken.py width`)

The set model uses the day features only, the E4h block and target r, at d 16 / 32 / 64 / 128, under the MSE and under the soft Score (`demo/runs/stock_score_itoken_width.log`, `demo/runs/e4j_width_pred.npz`):

| d | MSE: Score | MSE: vs ridge | soft Score: Score | soft Score: vs ridge | soft Score − MSE | 50-stock subset: soft Score − MSE |
|---|---|---|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | | | | | |
| ref: MSE ridge | +14.7e-4 ± 2.0e-4 | 0 | | 0 | | |
| 16 | +16.2e-4 | +1.6e-4 ± 1.0e-4 | +17.1e-4 | +2.5e-4 ± 1.0e-4 | +0.9e-4 ± 1.0e-4 | +1.4e-4 ± 1.0e-4 |
| 32 | +16.5e-4 | +1.9e-4 ± 1.2e-4 | **+18.1e-4** | **+3.4e-4 ± 1.3e-4** | +1.5e-4 ± 1.2e-4 | +1.4e-4 ± 1.2e-4 |
| 64 | +17.3e-4 | +2.6e-4 ± 1.5e-4 | +17.5e-4 | +2.8e-4 ± 1.2e-4 | +0.2e-4 ± 1.2e-4 | −0.7e-4 ± 1.2e-4 |
| 128 | +17.2e-4 | +2.6e-4 ± 1.4e-4 | +17.9e-4 | +3.3e-4 ± 1.3e-4 | +0.7e-4 ± 1.2e-4 | −0.3e-4 ± 1.2e-4 |

Against d 16 of the same loss the wider models change the Score by −0.6e-4 to +1.5e-4, all within 1.4 standard errors.

**This corrects the reading of the E4j ladder.** With the day features alone, the set model at d 16 already beats the ridge (+2.5e-4 under the soft Score), and the width matters little. In the ladder, the d 16 models at L2–L4 also carried the 60-day window in their token (180 more inputs, plus e and μ up to L2). At d 16 the window drowned the day features, which the window-only model (L7, +8.5e-4) shows to be the useful part. At d 64 the window became harmless. So the decisive differences between the E2j iT and the E4h set model are:
- the 60-day window in a narrow token;
- the zero-started head under the MSE, whose cost is −1.6e-4 in the 2 × 2;
- the selection of 0 steps.

The loss matters less: the soft Score adds +0.2e-4 to +1.5e-4 over the MSE, inconsistently across widths and subsets.

Two checks on the saved forecasts:
- **The window costs the narrow token, paired.** L4 (window + day features, d 16) against the day-only d 16 soft-Score model is −2.5e-4 ± 1.5e-4 on 100 stocks and −2.1e-4 ± 1.4e-4 on the 50-stock subset. L5 against the day-only d 64 model is +0.3e-4 ± 1.5e-4.
- **Larger post-hoc ensembles reach a plateau.** These are chosen after seeing the results, so they are optimistic; each member is divided by its sd per fold.

| Ensemble | 100 stocks: Score | vs ridge | vs r6 + set (E4h r2) | 50-stock subset: vs ridge |
|---|---|---|---|---|
| r6 members + 1-layer set models (E4h r2) | +17.5e-4 ± 2.0e-4 | +2.9e-4 ± 0.9e-4 | 0 | +3.4e-4 ± 0.9e-4 |
| all set models (E4h r2, 8 widths × losses, day mean, set MSE) | +17.9e-4 | +3.3e-4 ± 1.1e-4 | +0.4e-4 ± 0.5e-4 | +3.7e-4 ± 1.1e-4 |
| r6 members + all set models | +17.8e-4 | +3.1e-4 ± 1.0e-4 | +0.3e-4 ± 0.4e-4 | +3.4e-4 ± 1.1e-4 |
| soft-Score set models of all widths | +17.8e-4 | +3.1e-4 ± 1.1e-4 | +0.2e-4 ± 0.7e-4 | +3.4e-4 ± 1.2e-4 |

More members of the same kind do not move the Score beyond about +17.5e-4 to +17.9e-4. As in E4e, the limit is the information in the inputs.

### E4k: the morning's news as tokens of the set model (`demo/stock_score_news.py`)

This test uses the recipe that trains (target r, usual head, d 64, at least one epoch, soft Score). The day's overnight items are added as tokens of the same encoder layer: mean 116 and at most 256 items per day, own and market. Each news token is built as:
- Linear(the 42-dim struct vector: 24 Qwen3.5 fields, summary PCA16, source flags);
- plus a learned type embedding;
- plus, for an item tagged to one of our stocks, Linear(that stock's day features).

The output is read on the stock tokens only. The control gives every day the items of another day of the same year, with the tags moving with the items. Walk-forward 2022–2026, 5 seeds (`demo/runs/stock_score_news.log`, `demo/runs/e4k_pred.npz`):

| Version | 100 stocks: Score | vs ridge | vs no news | vs news of another day | 50-stock subset: vs no news | days with an own item: vs no news | 2025–26: vs no news |
|---|---|---|---|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | | | | | | |
| ref: MSE ridge | +14.7e-4 ± 2.0e-4 | 0 | | | | | |
| set, no news (E4h) | **+17.1e-4 ± 2.1e-4** | +2.5e-4 ± 1.3e-4 | 0 | | 0 | 0 | 0 |
| set + news tokens | +16.8e-4 | +2.1e-4 ± 1.3e-4 | −0.4e-4 ± 0.8e-4 | −0.1e-4 ± 0.9e-4 | +0.1e-4 ± 0.8e-4 | −1.7e-4 ± 1.0e-4 | −2.8e-4 ± 1.4e-4 |
| set + news of another day (control) | +16.9e-4 | +2.2e-4 ± 1.4e-4 | −0.3e-4 ± 0.9e-4 | 0 | −0.9e-4 ± 0.9e-4 | −1.1e-4 ± 1.0e-4 | −2.7e-4 ± 1.6e-4 |
| set + news without the tag term | +16.5e-4 | +1.8e-4 ± 1.2e-4 | −0.6e-4 ± 1.0e-4 | | −0.8e-4 ± 1.0e-4 | −1.9e-4 ± 1.2e-4 | −1.5e-4 ± 1.2e-4 |

The news tokens do not help, even with the recipe that trains the set model. The real news equal the news of another day (−0.1e-4 ± 0.9e-4). Both lose against the model without news on the days with an own item and in 2025–26, so the extra tokens only add noise to the attention. This agrees with the item-level probes: the Taiwan news carry a signal on the direction of the gap (E2g, E3c r6/r10) but none on r after the open.

### E4j: stock identity in the set model (`demo/stock_score_itoken.py ident`)

This is the idea of `StockIdentityAttention`, without the news. The set model (day features, E4h block, d 32, target r, soft Score) adds an identity to each stock's token:
- a learned embedding per stock, started at 0;
- Linear(industry one-hot);
- the industry one-hot given to the wrong stocks (control);
- both the learned embedding and the industry.

Walk-forward 2022–2026, 5 seeds (`demo/runs/stock_score_itoken_ident.log`, `demo/runs/e4j_ident_pred.npz`):

| Identity | 100 stocks: Score | vs ridge | vs no identity | 50-stock subset: vs no identity | 2025–26: vs no identity |
|---|---|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | | | | |
| ref: MSE ridge | +14.7e-4 ± 2.0e-4 | 0 | | | |
| none | **+18.1e-4 ± 2.1e-4** | +3.4e-4 ± 1.3e-4 | 0 | 0 | 0 |
| learned embedding | +16.9e-4 | +2.3e-4 ± 1.2e-4 | −1.1e-4 ± 0.6e-4 | −0.8e-4 ± 0.7e-4 | −2.5e-4 ± 1.7e-4 |
| industry | +17.7e-4 | +3.0e-4 ± 1.2e-4 | −0.4e-4 ± 0.8e-4 | −0.7e-4 ± 0.8e-4 | −0.1e-4 ± 1.3e-4 |
| industry permuted (control) | +17.3e-4 | +2.6e-4 ± 1.2e-4 | −0.8e-4 ± 0.9e-4 | −1.1e-4 ± 0.9e-4 | +1.6e-4 ± 1.2e-4 |
| learned + industry | +18.1e-4 | +3.5e-4 ± 1.3e-4 | +0.0e-4 ± 0.7e-4 | −0.2e-4 ± 0.7e-4 | +1.1e-4 ± 1.2e-4 |

The identity does not help. A learned embedding per stock hurts (−1.1e-4 ± 0.6e-4), like the stock intercepts of the ridge (E4b): what a stock "is" does not persist from the training years to the test year. The industry is no better than its permuted control. The useful structure is the stock's day features relative to the other stocks of the day (E4i), not its identity.

### E4j: an auxiliary gap head, and the raw pre-open features (`demo/stock_score_itoken.py aux`)

The news and the US returns forecast the gap but not r. This round tests whether a shared representation trained on the gap still helps r. The set model (day features, E4h block, d 32, target r, soft Score) gets a second head that forecasts the standardized gap residual (g − ĝ_pre)/sd. Its MSE, weighted by λ, is added to the soft Score, and the epochs are still chosen on the soft Score alone. A separate version, without the head, adds the 4 raw pre-open features to the day features (`demo/runs/stock_score_itoken_aux.log`, `demo/runs/e4j_aux_pred.npz`):

| Version | 100 stocks: Score | vs ridge | vs λ = 0 | 50-stock subset: vs λ = 0 | 2025–26: vs λ = 0 |
|---|---|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | | | | |
| ref: MSE ridge | +14.7e-4 ± 2.0e-4 | 0 | | | |
| λ = 0 (the d 32 set model) | **+18.1e-4 ± 2.1e-4** | +3.4e-4 ± 1.3e-4 | 0 | 0 | 0 |
| λ = 0.3 | +17.0e-4 | +2.3e-4 ± 1.2e-4 | −1.1e-4 ± 0.6e-4 | −0.8e-4 ± 0.7e-4 | −2.8e-4 ± 1.5e-4 |
| λ = 1 | +17.4e-4 | +2.7e-4 ± 1.3e-4 | −0.7e-4 ± 0.7e-4 | −0.7e-4 ± 0.7e-4 | −1.0e-4 ± 1.2e-4 |
| λ = 3 | +17.1e-4 | +2.4e-4 ± 1.3e-4 | −1.0e-4 ± 0.9e-4 | −0.9e-4 ± 0.9e-4 | −1.7e-4 ± 1.3e-4 |
| day features + 4 raw pre-open, no head | +16.4e-4 | +1.7e-4 ± 1.8e-4 | −1.7e-4 ± 1.4e-4 | −1.7e-4 ± 1.3e-4 | −2.1e-4 ± 1.5e-4 |

- **The auxiliary gap head hurts at every weight.** The gap and the return after the open share too little for one representation to serve both. This matches E4a: the forecastable part of the gap is almost unrelated to r.
- **The raw pre-open features hurt as well.** They are market-wide, the same for every stock of a day, and add a day-level signal that the network overfits. The gap block (ĝ_pre, |ĝ_pre|, its day mean) already carries what they offer.

### E4j: training settings of the set model (`demo/stock_score_itoken.py hp`)

The d 32 soft-Score set model (day features, E4h block, target r) is trained with 12 settings: dropout 0.1 / 0.2 / 0.4 × weight decay 1e-2 / 1e-1 × lr 1e-3 / 3e-4. The last row chooses the setting per fold by the mean best validation loss on Y − 1 over the 5 seeds, so no test year enters the choice (`demo/runs/stock_score_itoken_hp.log`, `demo/runs/e4j_hp_pred.npz`):

| dropout, weight decay | lr 1e-3: Score | vs ridge | lr 3e-4: Score | vs ridge |
|---|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | | | |
| ref: MSE ridge | +14.7e-4 ± 2.0e-4 | 0 | | |
| 0.1, 1e-2 | +17.8e-4 | +3.2e-4 ± 1.3e-4 | +16.0e-4 | +1.3e-4 ± 1.1e-4 |
| 0.1, 1e-1 | +17.1e-4 | +2.5e-4 ± 1.1e-4 | +16.4e-4 | +1.7e-4 ± 1.2e-4 |
| **0.2, 1e-2 (the default)** | **+18.1e-4** | **+3.4e-4 ± 1.3e-4** | +16.7e-4 | +2.0e-4 ± 1.2e-4 |
| 0.2, 1e-1 | +17.3e-4 | +2.6e-4 ± 1.2e-4 | +16.3e-4 | +1.7e-4 ± 1.2e-4 |
| 0.4, 1e-2 | +17.8e-4 | +3.2e-4 ± 1.3e-4 | +16.9e-4 | +2.2e-4 ± 1.2e-4 |
| 0.4, 1e-1 | +17.4e-4 | +2.8e-4 ± 1.3e-4 | +16.8e-4 | +2.1e-4 ± 1.2e-4 |
| chosen on Y − 1 | +17.1e-4 | +2.5e-4 ± 1.3e-4 | | |

The choices on Y − 1 differ every fold:

| Fold | dropout | weight decay | lr |
|---|---|---|---|
| 2022 | 0.2 | 1e-1 | 3e-4 |
| 2023 | 0.1 | 1e-1 | 1e-3 |
| 2024 | 0.4 | 1e-2 | 1e-3 |
| 2025 | 0.4 | 1e-1 | 1e-3 |
| 2026 | 0.4 | 1e-1 | 1e-3 |

- **The default is the best setting, and tuning on Y − 1 does not find it.** The chosen setting is −0.7e-4 ± 0.5e-4 below the best fixed one. The validation loss of one year is too noisy to rank settings that differ by about 1e-4.
- **The lower learning rate is worse in every combination** (−0.7e-4 to −1.9e-4). Dropout matters little; the larger weight decay costs −0.4e-4 to −0.8e-4. The setting of E4h stays.

**Learning rate × batch size** (`demo/stock_score_itoken.py hp2`, `demo/runs/stock_score_itoken_hp2.log`). Dropout 0.2 and weight decay 1e-2 are fixed:

| lr | batch 8 days: Score | batch 16 days | batch 32 days |
|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | | |
| ref: MSE ridge | +14.7e-4 ± 2.0e-4 | | |
| 1e-3 | +17.4e-4 | **+18.1e-4** (default; +3.4e-4 ± 1.3e-4 vs ridge) | +17.3e-4 |
| 2e-3 | +15.3e-4 | +17.7e-4 | +17.5e-4 |
| 3e-3 | +15.5e-4 | +14.7e-4 | +17.3e-4 |

The setting chosen on Y − 1 scores +15.9e-4, +1.3e-4 ± 1.3e-4 against the ridge. Its choices are:

| Fold | lr | batch |
|---|---|---|
| 2022 | 3e-3 | 8 days |
| 2023 | 1e-3 | 32 days |
| 2024 | 2e-3 | 32 days |
| 2025 | 3e-3 | 8 days |
| 2026 | 3e-3 | 32 days |

Larger steps, through a higher lr or smaller batches, make the model unstable: lr 3e-3 with 16 days falls to the ridge. The validation loss of one year again picks badly. The training settings of E4h are kept: lr 1e-3, 16 days, dropout 0.2, weight decay 1e-2.

### E4j: inputs as ranks within the day (`demo/stock_score_itoken.py rank`)

Each of the 55 day features is replaced by its rank among the usable stocks of the day, scaled to [−1, 1], or the ranks are appended to the standardized features. The model is the d 32 soft-Score set model (`demo/runs/stock_score_itoken_rank.log`, `demo/runs/e4j_rank_pred.npz`):

| Inputs | 100 stocks: Score | cross part | vs ridge | vs standardized | 50-stock subset: vs standardized | up calls |
|---|---|---|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | 0 | | | | 0% |
| ref: MSE ridge | +14.7e-4 ± 2.0e-4 | +3.2e-4 | 0 | | | 37.6% |
| standardized (the d 32 set model) | **+18.1e-4 ± 2.1e-4** | +5.5e-4 | +3.4e-4 ± 1.3e-4 | 0 | 0 | 46.3% |
| ranks within the day | +11.3e-4 | +9.6e-4 | −3.3e-4 ± 1.9e-4 | −6.7e-4 ± 2.1e-4 | −5.4e-4 ± 1.9e-4 | 41.2% |
| standardized + ranks | +17.4e-4 | +7.9e-4 | +2.8e-4 ± 1.4e-4 | −0.6e-4 ± 1.2e-4 | −0.9e-4 ± 1.2e-4 | |

- **The ranks remove the day's level and with it the market part.** The rank model's total is far below the others, although its cross part is large. Its cross part is not comparable, because a model without a market view spreads its up calls evenly over the days, which raises the cross part mechanically (see the fixed-share rules above).
- **Its ranking is worse, too.** The decision mean_t(f_std)/sd + w · (f_rank − mean_t f_rank)/sd combines the market tilt of the standardized model with the ranking of the rank model. Its Score falls with every w > 0, from +17.8e-4 at w = 0 to +12.4e-4 at w = 4. The same holds when the standardized model's own ranking is kept and w weights the rank model on top. Chosen on the earlier evaluation years, w is 0 in every fold. The sizes of the standardized features carry information that the ranks lose.

### E4j: training on the 50-stock subset only (`demo/stock_score_itoken.py top50`)

The d 32 soft-Score set model is trained and run on the 50-stock subset only. Its loss, its attention context and its choice of epochs are all restricted to those stocks. It is compared on that subset with the same model trained on all 100 stocks (`demo/runs/stock_score_itoken_top50.log`, `demo/runs/e4j_top50_pred.npz`):

| Training universe | 50-stock subset: Score | cross part | vs ridge | vs trained on 100 |
|---|---|---|---|---|
| always down (trivial baseline) | +3.4e-4 ± 2.3e-4 | 0 | | |
| ref: MSE ridge (trained on 100) | +11.0e-4 ± 1.9e-4 | +2.6e-4 | 0 | |
| set model, trained on 100 stocks | +14.8e-4 ± 1.9e-4 | +5.3e-4 | +3.8e-4 ± 1.3e-4 | 0 |
| set model, trained on the 50-stock subset | +14.8e-4 ± 1.9e-4 | +6.7e-4 | +3.8e-4 ± 1.2e-4 | +0.1e-4 ± 0.9e-4 |

The two are equal. Half the training rows and a context of 50 instead of 100 stocks cost nothing on the subset, so the model is not limited by the number of training stock-days.

### E4: summary of the round

Rounds E4a–E4k, walk-forward 2022–2026, all paired against the MSE ridge on the 52 base features + the ĝ_pre gap block (+14.7e-4 on 100 stocks, +11.0e-4 on the 50-stock subset; always down +7.8e-4 / +3.4e-4):

| Model | 100 stocks: Score | vs ridge | 50-stock subset: vs ridge |
|---|---|---|---|
| set model, day features, d 32, soft Score (best single model) | +18.1e-4 ± 2.1e-4 | +3.4e-4 ± 1.3e-4 | +3.8e-4 ± 1.3e-4 |
| r6 members + 1-layer set models (E4h r2) | +17.5e-4 ± 2.0e-4 | +2.9e-4 ± 0.9e-4 | +3.4e-4 ± 0.9e-4 |
| r6 base ensemble (linear soft Score + 2 MLPs) | +16.6e-4 ± 1.9e-4 | +1.9e-4 ± 0.9e-4 | +2.6e-4 ± 0.9e-4 |

**What the gain comes from**
- A network wide enough that sees the other stocks of the day; the day's mean of their representations is enough (E4i).
- The loss and the target matter little for the networks (2 × 2): the MSE on r or on the ridge residual works as well, provided the head is not started at zero.
- The E2j iTransformer failed for three reasons: the 60-day window in a narrow token, the zero-started head under the MSE, and the choice of 0 steps (E4j).

**What did not help**
- the parts of the gap forecast and the raw pre-open features (E4a, aux);
- calendar effects (E4b);
- stock intercepts, learned stock embeddings and the industry (E4b, identity, industry means);
- wider price inputs (r4);
- the news as columns or as tokens (r3 rich, E4k);
- recency weights (r5);
- training settings chosen on Y − 1 (hp, hp2);
- inputs as ranks within the day;
- training on the 50-stock subset only.

**The limit**
Larger ensembles stay near +17.5e-4 to +17.9e-4. The limit is the information in the 55 inputs, not the model.

### E4j: the best single model with 20 seeds (`demo/stock_score_itoken.py final`)

Two models are run with 20 seeds each: the d 32 soft-Score set model and the d 64 MSE set model. Their ensemble divides each by its sd per fold. Forecasts are in `demo/runs/e4j_final_pred.npz`, the log in `demo/runs/stock_score_itoken_final.log`.

| Model | 100 stocks: Score | cross part | vs ridge | 50-stock subset: Score | vs ridge |
|---|---|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | 0 | | +3.4e-4 ± 2.3e-4 | |
| ref: MSE ridge | +14.7e-4 ± 2.0e-4 | +3.2e-4 | 0 | +11.0e-4 ± 1.9e-4 | 0 |
| **d 32, soft Score, 20 seeds** | **+18.7e-4 ± 2.1e-4** | +5.6e-4 | **+4.1e-4 ± 1.2e-4** | **+15.6e-4 ± 2.0e-4** | **+4.6e-4 ± 1.3e-4** |
| d 64, MSE, 20 seeds | +16.9e-4 | +5.8e-4 | +2.2e-4 ± 1.2e-4 | +13.6e-4 | +2.6e-4 ± 1.2e-4 |
| ensemble of the two | +17.7e-4 | +5.6e-4 | +3.1e-4 ± 1.1e-4 | +14.6e-4 | +3.6e-4 ± 1.1e-4 |

By year, 100 stocks (the difference from the ridge in brackets):

| Year | always down | MSE ridge | d 32 soft Score, 20 seeds | d 64 MSE, 20 seeds |
|---|---|---|---|---|
| 2022 | +6.7e-4 ± 5.5e-4 | +13.9e-4 | +19.2e-4 (+5.3e-4 ± 2.8e-4) | +17.3e-4 (+3.4e-4 ± 2.1e-4) |
| 2023 | +3.3e-4 ± 3.9e-4 | +7.8e-4 | +12.7e-4 (+4.9e-4 ± 1.5e-4) | +13.3e-4 (+5.5e-4 ± 1.8e-4) |
| 2024 | +8.2e-4 ± 4.7e-4 | +14.4e-4 | +16.3e-4 (+1.9e-4 ± 2.2e-4) | +12.8e-4 (−1.6e-4 ± 2.8e-4) |
| 2025 | +9.0e-4 ± 5.3e-4 | +16.7e-4 | +17.5e-4 (+0.7e-4 ± 3.4e-4) | +16.5e-4 (−0.3e-4 ± 2.7e-4) |
| 2026 (to Aug) | +13.9e-4 ± 8.8e-4 | +23.5e-4 | +32.8e-4 (+9.4e-4 ± 3.3e-4) | +28.2e-4 (+4.7e-4 ± 3.7e-4) |

- **The d 32 soft-Score set model with 20 seeds is the best model of E4.** It beats the ridge by 3.4 standard errors on 100 stocks and 3.5 on the 50-stock subset, and it is ahead in all five years. Averaging 20 seeds instead of 5 adds +0.6e-4: the seed variance of a single model is still large, unlike that of ensembles of different models.
- **At 20 seeds the soft Score beats the MSE** (−1.9e-4 ± 1.0e-4 for the MSE model; −2.1e-4 ± 1.0e-4 on the 50-stock subset). The 2 × 2 with 5 seeds could not separate them, so the loss does matter a little once the seed noise is averaged out. Mixing in the MSE model only dilutes the soft-Score model.

**40 seeds, and the width at 20 seeds** (`demo/stock_score_itoken.py final2`, `demo/runs/stock_score_itoken_final2.log`, `demo/runs/e4j_final2_pred.npz`):

| Model | 100 stocks: Score | vs ridge | 50-stock subset: vs ridge |
|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | | |
| ref: MSE ridge | +14.7e-4 ± 2.0e-4 | 0 | 0 |
| d 32, soft Score, 20 seeds (above) | +18.7e-4 | +4.1e-4 ± 1.2e-4 | +4.6e-4 ± 1.3e-4 |
| **d 32, soft Score, 40 seeds** | **+18.2e-4 ± 2.1e-4** | **+3.5e-4 ± 1.1e-4** | **+3.7e-4 ± 1.2e-4** |
| d 16, soft Score, 20 seeds | +16.8e-4 | +2.1e-4 ± 1.0e-4 | |
| d 128, soft Score, 20 seeds | +17.8e-4 | +3.1e-4 ± 1.2e-4 | |

The 40 seeds include the 20 of the run above. Against that run they score −0.6e-4 ± 0.3e-4 on 100 stocks and −0.9e-4 ± 0.4e-4 on the 50-stock subset. The 20-seed figure of +18.7e-4 was therefore partly luck of the seeds. The better estimate of the d 32 soft-Score set model is about +18.2e-4, or +3.5e-4 ± 1.1e-4 against the ridge, ahead in all five years:

| Year | vs ridge |
|---|---|
| 2022 | +4.5e-4 |
| 2023 | +4.8e-4 |
| 2024 | +0.7e-4 |
| 2025 | +0.1e-4 |
| 2026 | +9.5e-4 |

d 32 remains the best width at 20 seeds, though within the error of d 128.

Post-hoc ensembles with the 40-seed d 32 soft-Score model (saved forecasts; each member divided by its sd per fold):

| Ensemble | 100 stocks: Score | vs ridge | vs d 32 alone | 50-stock subset: vs d 32 alone |
|---|---|---|---|---|
| d 32 soft Score, 40 seeds | +18.2e-4 ± 2.1e-4 | +3.5e-4 ± 1.1e-4 | 0 | 0 |
| + d 128 (20 seeds) | +18.1e-4 | +3.4e-4 ± 1.2e-4 | −0.1e-4 ± 0.4e-4 | −0.3e-4 ± 0.5e-4 |
| + d 128 + d 16 (20 seeds each) | +17.6e-4 | +3.0e-4 ± 1.1e-4 | −0.5e-4 ± 0.3e-4 | −0.8e-4 ± 0.3e-4 |
| + the three r6 members | +17.4e-4 | +2.8e-4 ± 0.9e-4 | −0.8e-4 ± 0.6e-4 | −0.4e-4 ± 0.6e-4 |

No ensemble beats the 40-seed d 32 soft-Score set model, which stays the E4 candidate.

### E4l: gradient-boosted trees, alone and ensembled with the set model (`demo/stock_score_lgbm.py`, not included)

LightGBM 4.7 (CPU, 8 threads) on the same 55 inputs as the set model (52 base features + gap block of ĝ_pre, not standardized). Variants:
- **binary:** label 1[r > 0], weight |r| (the tree form of the weighted logistic loss), decision sign(p − 0.5);
- **regression:** L2 on r, decision sign(f);
- **binary + day context:** the binary variant plus the day's mean of each input and the stock's deviation from it (165 inputs), a cross-sectional view comparable to what the set model gets from attention.

Selection on Y − 1: grid num_leaves {7, 15, 31} × min_data_in_leaf {200, 1000} (learning rate 0.02, feature fraction 0.5, bagging 0.7, L2 penalty 10), each point trained on the years before Y − 1, tree count by early stopping on Y − 1 (patience 200, 50–3000 trees), lowest validation loss wins; refit on all years before Y with that tree count, 5 seeds averaged. Ensembles: each member divided by its sd over the fold's test rows (p − 0.5 for the binary members), summed; rank averages: ranks within the day centered to [−1/2, 1/2], summed. The recomputed ridge equals the saved reference exactly. Walk-forward 2022–2026, differences paired and day-clustered:

| Model | Score (100) | − ridge | − d 32 | Score (50) | 2022 | 2023 | 2024 | 2025 | 2026 | up share |
|---|---|---|---|---|---|---|---|---|---|---|
| *Always down* | *+7.8e-4 ± 2.4e-4* | *−6.8e-4 ± 2.6e-4* | *−10.4e-4 ± 2.7e-4* | *+3.4e-4* | +6.7e-4 | +3.3e-4 | +8.2e-4 | +9.0e-4 | +13.9e-4 | 0.000 |
| MSE ridge | +14.7e-4 ± 2.0e-4 | 0 | −3.5e-4 ± 1.1e-4 | +11.0e-4 | +13.9e-4 | +7.8e-4 | +14.4e-4 | +16.7e-4 | +23.5e-4 | 0.376 |
| d 32 set model, 40 seeds | +18.2e-4 ± 2.1e-4 | +3.5e-4 ± 1.1e-4 | 0 | +14.7e-4 | +18.3e-4 | +12.6e-4 | +15.1e-4 | +16.9e-4 | +32.9e-4 | 0.453 |
| LGBM binary | +14.9e-4 ± 2.3e-4 | +0.2e-4 ± 1.8e-4 | −3.3e-4 ± 1.7e-4 | +12.0e-4 | +11.3e-4 | +9.9e-4 | +16.3e-4 | +13.2e-4 | +28.3e-4 | 0.435 |
| LGBM regression | +14.7e-4 ± 2.2e-4 | +0.1e-4 ± 1.6e-4 | −3.5e-4 ± 1.6e-4 | +11.0e-4 | +11.6e-4 | +10.9e-4 | +17.1e-4 | +10.7e-4 | +27.9e-4 | 0.404 |
| LGBM binary + day context | +14.5e-4 ± 2.3e-4 | −0.2e-4 ± 1.9e-4 | −3.7e-4 ± 1.8e-4 | +11.7e-4 | +5.7e-4 | +10.6e-4 | +17.4e-4 | +16.3e-4 | +26.8e-4 | 0.458 |
| d 32 + LGBM binary | +17.6e-4 ± 2.2e-4 | +2.9e-4 ± 1.5e-4 | −0.6e-4 ± 1.1e-4 | +14.0e-4 | +13.8e-4 | +11.3e-4 | +17.9e-4 | +16.7e-4 | +33.4e-4 | 0.446 |
| d 32 + LGBM regression | +17.1e-4 ± 2.1e-4 | +2.4e-4 ± 1.1e-4 | −1.1e-4 ± 1.0e-4 | +13.0e-4 | +15.0e-4 | +10.6e-4 | +17.5e-4 | +13.8e-4 | +34.4e-4 | 0.441 |
| d 32 + LGBM binary + context | +18.5e-4 ± 2.1e-4 | +3.8e-4 ± 1.4e-4 | +0.3e-4 ± 1.0e-4 | +14.7e-4 | +14.9e-4 | +13.0e-4 | +17.3e-4 | +18.7e-4 | +33.8e-4 | 0.459 |
| d 32 + mean of the 3 LGBM | +18.1e-4 ± 2.1e-4 | +3.5e-4 ± 1.3e-4 | −0.0e-4 ± 1.0e-4 | +14.2e-4 | +14.8e-4 | +11.6e-4 | +18.3e-4 | +17.5e-4 | +33.9e-4 | 0.447 |
| rank average: d 32 + LGBM binary | +10.3e-4 ± 1.1e-4 | −4.4e-4 ± 1.8e-4 | −7.9e-4 ± 1.9e-4 | +8.8e-4 | +8.8e-4 | +10.2e-4 | +14.8e-4 | +8.4e-4 | +8.9e-4 | 0.497 |
| rank average: d 32 + LGBM regression | +12.3e-4 ± 1.2e-4 | −2.4e-4 ± 1.7e-4 | −5.9e-4 ± 1.8e-4 | +10.0e-4 | +10.9e-4 | +10.1e-4 | +16.6e-4 | +9.7e-4 | +15.1e-4 | 0.502 |
| rank average: d 32 + LGBM binary + context | +10.8e-4 ± 1.2e-4 | −3.8e-4 ± 1.8e-4 | −7.4e-4 ± 1.9e-4 | +9.1e-4 | +10.8e-4 | +10.3e-4 | +13.8e-4 | +7.6e-4 | +11.9e-4 | 0.497 |

On the 50-stock subset vs d 32: binary −2.8e-4 ± 1.7e-4, regression −3.7e-4 ± 1.5e-4, binary + context −3.1e-4 ± 1.7e-4, d 32 + binary + context −0.0e-4 ± 1.0e-4.

- **The trees equal the ridge, not the set model.** All three variants are within 0.2e-4 of the ridge and 3.3–3.7e-4 (about 2 SE) below d 32. Early stopping keeps only 50–170 trees (2023: the 50-tree floor in every grid point); the validation Scores of the grid points differ by 1–5e-4.
- **What the trees use is the day-level part.** The largest gain shares are on inputs equal for all stocks of a day: ADR premium change (0.13–0.17), Nasdaq return, the day mean of ĝ_pre, TAIEX/S&P reversion terms. The day-context inputs do not move the trees toward the set model's within-day ranking.
- **No ensemble helps.** The best, d 32 + binary + context, is +0.3e-4 ± 1.0e-4 over d 32 alone; the others are 0 to −1.1e-4. Rank averages lose 5.9–7.9e-4 because they discard the day's common direction (up share ≈ 0.50), which carries most of the Score.
- The d 32 soft-Score set model stays the candidate.

### E4m: futures-exchange data (single-stock futures, put/call ratio, implied volatility, large traders) (`demo/stock_score_taifex.py`, data from `demo/fetch_taifex.py`)

The data come from the Taiwan Futures Exchange's public download pages (no key, about 200 requests at least 2.5 s apart); its terms forbid redistribution, so they stay in the git-ignored `dataset/taifex/`. Daily reports of all futures and options contracts (both sessions), put/call ratios, large traders' open positions (TX, TXO and per single-stock future); the exchange keeps only three months of its volatility index, so an at-the-money implied volatility of the near-month TXO (straddle settlement / (0.798 F √τ)) replaces it. 92 of the 100 stocks have a single-stock future (SSF); about 80% of the 2020–2026 stock-days are covered. Features, all known before 09:00 of day t, clipped at the 0.5/99.5% quantiles of the pre-2022 rows:
- **market (14):** TX 13:45 close vs TAIEX 13:30 close minus its 20-day median basis (the move after the stock market closes), TX volume and open-interest change, put/call volume and open-interest ratios and their changes, the implied volatility and its change, the top-5/top-10 large traders' net TX position over open interest (all, and specific legal persons) and its change, the top-10 net TXO call minus put position;
- **per-stock SSF (8):** the SSF settlement vs the stock's close minus its 20-day median on the same contract (near month rolled 3 days before expiry), basis change, volume, open-interest change, night-session move (only 2330 and 2303, from 2025), coverage flag, top-10 large traders' net position and its change;
- **SSF open (separate variant):** log(08:45 open of t / settlement of t − 1). A day's "open" can be a trade after 09:00 for an illiquid contract. In the exchange's tick data of 2026-09-23 .. 11-09, 34 contracts have their first trade before 09:00 on at least 95% of the days (and trade on at least 80%); the open is used only for those and only if the near month traded at least half its median volume on day t − 1: 17 stocks, 6.8% of the stock-days. The check covers months after our data end, when liquidity was higher, so some historical opens may still lie after 09:00. The SSF open correlates 0.91 with the stock's gap.

Ridge part (walk-forward 2022–2026; the base ridge reproduces +14.7e-4; differences paired and day-clustered):

| Model | Score (100) | − ridge | Score (50) | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| *Always down* | *+7.8e-4 ± 2.4e-4* | *−6.8e-4 ± 2.6e-4* | *+3.4e-4* | +6.7e-4 | +3.3e-4 | +8.2e-4 | +9.0e-4 | +13.9e-4 |
| MSE ridge | +14.7e-4 ± 2.0e-4 | 0 | +11.0e-4 | +13.9e-4 | +7.8e-4 | +14.4e-4 | +16.7e-4 | +23.5e-4 |
| ridge + market | +12.4e-4 ± 2.1e-4 | −2.3e-4 ± 1.4e-4 | +9.6e-4 | +10.0e-4 | +7.6e-4 | +11.5e-4 | +12.5e-4 | +24.6e-4 |
| ridge + SSF | +15.2e-4 ± 2.0e-4 | +0.5e-4 ± 0.4e-4 | +11.5e-4 | +14.1e-4 | +9.1e-4 | +15.4e-4 | +17.0e-4 | +22.8e-4 |
| ridge + both | +12.7e-4 ± 2.1e-4 | −1.9e-4 ± 1.4e-4 | +9.9e-4 | +9.5e-4 | +7.5e-4 | +12.2e-4 | +13.8e-4 | +24.6e-4 |
| ridge, ĝ_pre with both | +14.8e-4 ± 2.0e-4 | +0.2e-4 ± 0.2e-4 | +11.0e-4 | +14.2e-4 | +7.8e-4 | +14.7e-4 | +16.7e-4 | +23.6e-4 |
| ridge + both + SSF open | +12.8e-4 ± 2.0e-4 | −1.8e-4 ± 1.4e-4 | +9.8e-4 | +9.7e-4 | +7.5e-4 | +12.3e-4 | +14.1e-4 | +24.6e-4 |
| ridge, ĝ_pre with both + SSF open | +14.7e-4 ± 2.0e-4 | +0.1e-4 ± 0.2e-4 | +10.9e-4 | +14.2e-4 | +7.7e-4 | +14.7e-4 | +16.8e-4 | +23.1e-4 |

Walk-forward R² of the gap forecast ĝ_pre: base 0.4779; + market +0.0015 ± 0.0011; + SSF +0.0020 ± 0.0003; + both +0.0035 ± 0.0011; + SSF open +0.0145 ± 0.0017; + both + SSF open +0.0175 ± 0.0022.

- **The market features hurt the return ridge** (−2.3e-4 ± 1.4e-4): they flip about 20% of its calls, mostly moving whole days between up and down, i.e. the ridge overfits day-level series.
- **The SSF features and the SSF open improve the gap forecast, not the return.** The SSF open adds 0.0145 to the gap R², yet the ridge Score with it in ĝ_pre is unchanged (+0.1e-4 ± 0.2e-4), in line with E4a: the predictable part of the gap barely relates to r.

Set-model part: the d 32 soft-Score set model (E4j final2 recipe), 10 seeds 0..9, the same seeds for every variant, so the differences to "set base" are paired (the 40-seed final2 model scored +18.2e-4):

| Model | Score (100) | − ridge | − set base | Score (50) | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|
| set base, 10 seeds | +17.3e-4 ± 2.1e-4 | +2.6e-4 ± 1.1e-4 | 0 | +14.0e-4 | +16.2e-4 | +12.3e-4 | +15.5e-4 | +16.7e-4 | +30.2e-4 |
| set + market | +11.5e-4 ± 2.1e-4 | −3.2e-4 ± 1.7e-4 | −5.8e-4 ± 1.7e-4 | +10.0e-4 | +7.8e-4 | +8.2e-4 | +18.0e-4 | +3.1e-4 | +25.2e-4 |
| set + SSF | +18.5e-4 ± 2.0e-4 | +3.8e-4 ± 1.1e-4 | +1.2e-4 ± 0.7e-4 | +15.7e-4 | +19.2e-4 | +13.2e-4 | +17.8e-4 | +16.4e-4 | +29.5e-4 |
| set + both | +14.7e-4 ± 2.0e-4 | +0.0e-4 ± 1.6e-4 | −2.6e-4 ± 1.4e-4 | +12.7e-4 | +16.7e-4 | +9.7e-4 | +18.2e-4 | +8.7e-4 | +22.8e-4 |
| set, ĝ_pre with both | +18.3e-4 ± 2.1e-4 | +3.6e-4 ± 1.3e-4 | +1.0e-4 ± 0.6e-4 | +15.1e-4 | +18.5e-4 | +12.5e-4 | +15.5e-4 | +17.5e-4 | +32.0e-4 |
| set + both + SSF open | +13.8e-4 ± 2.0e-4 | −0.9e-4 ± 1.7e-4 | −3.5e-4 ± 1.6e-4 | +12.1e-4 | +14.7e-4 | +7.0e-4 | +19.6e-4 | +6.5e-4 | +24.6e-4 |
| set, ĝ_pre with both + SSF open | +18.1e-4 ± 2.1e-4 | +3.4e-4 ± 1.2e-4 | +0.8e-4 ± 0.5e-4 | +14.7e-4 | +18.9e-4 | +12.5e-4 | +15.8e-4 | +17.9e-4 | +28.8e-4 |

- **The market features hurt the set model even more** (−5.8e-4 ± 1.7e-4): its share of up calls rises from 0.46 to 0.63 and about 24% of its calls flip, mostly whole days; most of the loss is in 2025.
- **The per-stock SSF features help a little:** +1.2e-4 ± 0.7e-4 over the same seeds without them (1.7 SE), +1.8e-4 ± 0.7e-4 on the 50-stock subset and +1.5e-4 ± 0.7e-4 on the stocks that have an SSF; the gain comes from 2022–2024 (2025–2026: −0.5e-4 ± 1.2e-4). The features in the gap ridge only give +1.0e-4 ± 0.6e-4.
- The SSF open adds nothing to the Score, in the inputs or in ĝ_pre.
- Round r2 (`stock_score_taifex.py r2`) repeats the SSF-only variants with 40 seeds.
