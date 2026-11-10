# Signalformer

A causal encoder-decoder Transformer for multivariate signal forecasting, written in PyTorch.

Signalformer splits each input signal into multi-scale frequency bands, selects useful bands with a learnable L0 gate, tokenizes them with a low-rank causal convolution, and forecasts future steps with a causal encoder-decoder Transformer. Every output depends only on past inputs, so the model can be trained and evaluated on time-ordered data without look-ahead leakage.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/architecture-dark.svg">
  <img alt="Signalformer architecture" src="docs/architecture.svg">
</picture>

## Features

- **Multi-scale decomposition**: trailing moving averages split each signal into bands from fine to coarse (`x − MA_w1, MA_w1 − MA_w2, …, MA_wk`). The bands sum back to the input exactly.
- **Learned feature selection**: a Hard Concrete L0 gate learns a per-band weight that can be exactly zero.
- **Convolutional tokenizer**: a depthwise separable causal convolution, configurable as a sum of `n_rank` rank-1 convolutions.
- **Patching**: `conv_stride` turns the tokenizer into a causal patch embedding, one token per `conv_stride` input steps.
- **Optional signal independence**: `cross_channel=False` keeps signals separate inside the tokenizer.
- **Causal Transformer**: pre-norm blocks with RMSNorm, rotary position embeddings (RoPE) on queries and keys, and `F.scaled_dot_product_attention`.
- **Non-autoregressive decoder**: learnable queries predict all future steps in one pass.
- **Optional inputs**: calendar or other known-in-advance time features for past and future steps, and cross-attention to embeddings from an external text encoder.
- **Flexible output**: configurable output width (`out_channels`) and a coarser output frequency (`samples_per_output`).

## Requirements

- Python 3.10+
- PyTorch 2.4+ (for `nn.RMSNorm`)

## Quick start

```python
import torch
from model import Signalformer

model = Signalformer(
    in_channels=5,          # number of input signals
    in_len=128,             # input length
    d_model=64,
    kernel_size=3,
    output_length=4,        # number of output steps
    num_heads=4,
    ma_windows=(5, 20, 60), # 4 bands per signal
)

x = torch.randn(32, 5, 128)  # (N, C, L)
y = model(x)                 # (N, out_channels, output_length) = (32, 1, 4)
```

### Patching

```python
model = Signalformer(
    in_channels=5, in_len=128, d_model=80, output_length=4, num_heads=4,
    kernel_size=16,         # patch length
    conv_stride=8,          # one token every 8 steps (overlapping patches), 16 tokens instead of 128
    n_rank=8,               # keep enough detail per patch
    cross_channel=False,    # optional: keep signals separate in the tokenizer (d_model % in_channels == 0)
)
```

### Optional inputs

```python
model = Signalformer(
    in_channels=5, in_len=128, d_model=64, kernel_size=3, output_length=4, num_heads=4,
    time_dim=6,             # width of time features
    text_dim=768,           # width of external text embeddings
    out_channels=3,         # values per output step
    samples_per_output=7,   # e.g. daily input -> weekly output
)

y = model(
    x,                                      # (N, 5, 128)
    time_features=past_time,                # (N, 128, 6)
    future_time_features=future_time,       # (N, 4 * 7, 6), known in advance
    text_memory=text_embeddings,            # (N, L_text, 768)
    text_padding_mask=text_padding_mask,    # (N, L_text) bool, True = padding
)                                           # (N, 3, 4)
```

## Training

Add the gate penalty to the task loss. It is the expected fraction of open gates (0 to 1), so it pushes unused bands toward zero and `lam` means the same thing for any number of signals.

```python
gate_params = [model.tokenizer.gate.log_alpha]
other_params = [p for p in model.parameters() if p is not gate_params[0]]
optimizer = torch.optim.AdamW([
    {"params": other_params, "lr": 1e-3, "weight_decay": 1e-2},
    {"params": gate_params, "lr": 1e-2, "weight_decay": 0.0},
])

with torch.autocast("cuda", dtype=torch.bfloat16):
    pred = model(x)
    loss = task_loss(pred, target) + lam * model.penalty()
loss.backward()
optimizer.step()
```

`lam` around `1e-2` is a reasonable start for a z-scored MSE loss.

## Usage notes

- **Keep the data pipeline causal.** Normalize inputs with statistics from past data only, and validate with time-ordered (walk-forward) splits rather than random splits.
- **Drop the warm-up period.** The first `max(ma_windows) − 1` steps use incomplete moving-average windows. Keep `in_len` well above the largest window.
- **Prefer stationary inputs.** Use differences, log ratios or rates of change instead of drifting levels. RMSNorm does not remove the mean.
- **Encode time features carefully.** Use sin/cos pairs for cyclical fields and 0/1 flags for known events. Avoid absolute year features, which force the model to extrapolate on later test periods.
- **Only use information available at forecast time.** Future time features must be known in advance, and text must have been available before the end of the input window.
- **Raise `n_rank` with long patches.** With `conv_stride > 1`, each channel's patch is compressed to `n_rank` numbers before channels are mixed. `n_rank = 1` with `kernel_size = 16` keeps only one weighted average per patch; `n_rank >= kernel_size` loses nothing.
- **Check gate stability.** Correlated bands may be selected arbitrarily. Trust a selection only if it is stable across seeds and validation folds.

## Stock return models (`stock_model.py`)

`stock_model.py` holds a second model family: forecasts of every stock's open-to-close return on a trading day, from the prices up to the previous close and the news published before the open. It needs only NumPy and PyTorch and reads no data.

- **Stage 1, walk-forward ridge** (`walk_forward`, `Ridge`): every year Y is forecast by a ridge fitted on all years before Y, with its penalty chosen on Y − 1 from the fit on the years before Y − 1, so no forecast or choice uses later data. `gap_block` adds a forecast of the opening gap log(open / previous close).
- **Stage 2, residual network**: forecasts the ridge residual. Its output head starts at zero, so the untrained network is the ridge. `StockTokenEncoder` makes one token per stock from its whole input window (iTransformer style) and mixes the stocks by self-attention.
  - `StockNewsAttention` adds a cross attention between the stock tokens and the day's news items, in four arrangements of query, key and value (`mode`).
  - `StockIdentityAttention`: every news item queries the stocks by their identity (a fixed company embedding such as `comovement_embedding`, plus a learned one), reads their price states, and is pooled back onto the stocks it attended to.
- **`TypedForm`**: a logits processor that makes a local language model fill a fixed form about a news text. The form tokens are forced, and each field is restricted to its choice letters, whose probabilities become the features.

```python
import torch
from stock_model import StockIdentityAttention, comovement_embedding

identity = torch.tensor(comovement_embedding(train_returns, dim=16))   # (S, 16) from a (T, S) training period
net = StockIdentityAttention(n_stocks=100, window_dim=60 * 4, n_features=56, identity=identity, news_dim=20)
e_hat = net(win, cur, news, tags)   # win (B, 100, 240), cur (B, 100, 56), news (B, N, 20), tags (B, N) -> (B, 100)
```

The data pipeline, the experiments and their results are in `demo/` (`demo/README.md`, `demo/EXPERIMENTS.md`). The market data, the news texts and the scores derived from them are not part of this repository.

### Current best model: the d 32 soft-Score set model

The best forecast of the sign of every stock's open-to-close return r = (close − open) / open is a small Transformer over the stocks of one day, trained directly on the sign score. It is defined in `demo/stock_score_itoken.py` (part `final2`), and a faster trainer for it is in `demo/stock_score_fast.py`.

- **Inputs, per stock and day (55), all known before the 09:00 open:**
  - **52 base features** (`stock_recurrent.base_features`):
    - degree-0 multi-window polynomial readouts of the stock's open-to-close returns, overnight gaps and cumulative price path, and of the Nasdaq, S&P 500 and TAIEX;
    - the daily range log(high / low) and its moving averages;
    - 5 US-session features: TSM ADR, SOX and Nasdaq-futures returns, the change of the TSM ADR premium, and the stock's own ADR return for 2330 / 2303 / 3711.
  - **The gap block (3):** ĝ_pre, |ĝ_pre| and its day mean.
    - ĝ_pre is a pooled walk-forward ridge forecast of the opening gap.
    - It uses the 52 features plus 4 pre-open inputs: the Nikkei and KOSPI opening gaps at 08:00, the TAIEX futures gap at 08:45, and that gap times the stock's 60-day gap beta.
- **Network:**
  - Linear(55 → 32) + GELU;
  - one pre-norm Transformer encoder layer over the day's stocks: 4 heads, feed-forward 64, dropout 0.2, with untraded stocks masked as keys;
  - Linear(32 → 1).
  - 10,369 parameters.
- **Loss:** soft score, −Σ tanh(f) r / Σ |r| over the day's usable stocks; a smooth version of the sign score.
- **Training:**
  - AdamW, learning rate 1e-3, weight decay 1e-2, batches of 16 days;
  - the epoch count is chosen on year Y − 1 by early stopping (patience 3, at most 40);
  - the network is then refit from scratch on all years before Y;
  - 40 seeds, whose raw outputs are summed;
  - the forecast is up if the sum is positive.

Walk-forward 2022–2026, 100 stocks (112,853 stock-days). Sign score = mean of sign(forecast) × r; differences are paired and their standard errors clustered by day:

| Model | Sign score (100 stocks) | − MSE ridge | Sign score (50-stock subset) |
|---|---|---|---|
| always down (trivial baseline) | +7.8e-4 ± 2.4e-4 | −6.8e-4 ± 2.6e-4 | +3.4e-4 ± 2.3e-4 |
| MSE ridge on the same 55 inputs | +14.7e-4 ± 2.0e-4 | 0 | +11.0e-4 ± 1.9e-4 |
| **d 32 soft-Score set model, 40 seeds** | **+18.2e-4 ± 2.1e-4** | **+3.5e-4 ± 1.1e-4** | **+14.7e-4 ± 1.9e-4** |

- **Per year:** the set model scores +18.3e-4, +12.6e-4, +15.1e-4, +16.9e-4 and +32.9e-4 in 2022–2026, above the ridge in every year.
- **Checks:**
  - An attention mask built only from pre-open information gives the same result (+0.3e-4 ± 0.2e-4).
  - The fast trainer reproduces it (−0.2e-4 ± 0.2e-4, 1.6 min instead of about 18 min).
- **Tested and not adopted:** news tokens, futures-exchange data, margin / securities-lending / shareholding columns, gradient-boosted trees and post-hoc ensembles. Each was paired with this model, at 40 seeds where it mattered (`demo/EXPERIMENTS.md`, section E4).

### Running it

The scripts expect a git-ignored `dataset/` at the repository root (not distributed):

| Path | Content |
|---|---|
| `dataset/info.json` | `Top100@Y` (the 100 stocks) and `Top50@Y` (the 50-stock subset) |
| `dataset/market/<id>.csv` | daily `date`, `etl:adj_open`, `etl:adj_high`, `etl:adj_low`, `etl:adj_close` (adjusted prices), … per stock |
| `dataset/global/global_data.csv` | daily `^IXIC`, `^GSPC`, `^TWII`, … |
| `dataset/overnight/` | `yahoo_TSM.csv`, `yahoo_UMC.csv`, `yahoo_ASX.csv`, `yahoo_SOX.csv`, `yahoo_NQ_F.csv`, `yahoo_TWD_X.csv`, `yahoo_N225.csv`, `yahoo_KS11.csv` and `taifex_tx.csv`, made by `python demo/fetch_overnight.py` |

```bash
pip install numpy pandas torch
mkdir -p demo/runs
python demo/fetch_overnight.py                 # ADRs, SOX, Nasdaq futures, Nikkei, KOSPI, TWD (Yahoo) and TAIEX futures (TAIFEX)
python demo/stock_score_fast.py check          # the best model, 40 seeds batched on one GPU, about 2 min
python demo/stock_score_itoken.py final2       # the same with one network per seed, plus d 16 / d 128 variants (the original run)
python demo/stock_score_itoken.py premask      # the same with a pre-open attention mask
```

- Each run prints the walk-forward table (the sign score with day-clustered standard errors, paired differences against the MSE ridge, the 50-stock subset, per year).
- Each run saves the forecasts of 2022–2026 to `demo/runs/*.npz`.
- `demo/stock_score_profile.py` times every stage of one fold.

### Earlier models

Walk-forward results of the earlier ridge and residual-network stages, pooled over 2022–2026 for 100 stocks, with standard errors clustered by day:

| Model | Features | R² | Sign score |
|---|---|---|---|
| always down (trivial baseline) | — | — | +7.8e-4 ± 2.4e-4 |
| multi-window polynomial ridge, degrees ≤ 3 (`demo/stock_poly.py`) | 230 | +0.0040 | +10.5e-4 ± 2.0e-4 |
| polynomial ridge, degree 0 + daily range | 47 | +0.0046 | +12.7e-4 ± 2.0e-4 |
| difference equations fitted per window, orders 1–3 (`demo/stock_diffeq.py`) | 91 | −0.0002 | +8.3e-4 ± 2.2e-4 |
| block difference equations, stride 1, order 10, + range (`demo/stock_diffeq_order.py`) | 42 | +0.0070 | +13.9e-4 |
| ridge, 52 base features (degree 0, range, US session) | 52 | +0.0087 | +14.3e-4 ± 2.1e-4 |
| ridge, + the gap block | 55 | +0.0095 | +14.5e-4 ± 2.0e-4 |
| + stock tokens, no news | 55 | +0.0092 | +14.3e-4 ± 2.1e-4 |
| + `StockNewsAttention`, `stockq` | 55 | +0.0090 | +14.3e-4 ± 2.1e-4 |
| + `StockNewsAttention`, `sigval` | 55 | +0.0093 | +14.4e-4 ± 2.0e-4 |

The bilinear model (`demo/stock_bilinear.py`: r̂ = b₀ + Σₖ uₖᵀ Z vₖ over the last L days Z of 6 daily signals, with polynomial lag filters) was evaluated on one fixed split (train 2020–2023, validation 2024, test 2025–2026-08) rather than walk-forward. There, its cubic rank-2 version on 20 lags has 20 parameters and a test score of +16.9e-4, against +16.3e-4 for the 230-feature polynomial ridge and +11.0e-4 for always down.

The rows of this table differ slightly in their sample (the feature sets start on different days, and the gap block of these rows used an earlier gap forecast without the pre-open inputs), so compare them only as rough levels. The residual networks on the ridge did not beat the ridge; the gain came when the network predicts r directly with the soft-score loss (above). `StockIdentityAttention` was evaluated in E2j (`demo/EXPERIMENTS.md`) and did not help.

## Documentation

- [`docs/architecture.html`](docs/architecture.html): detailed design reference for every module
- [`docs/architecture.pdf`](docs/architecture.pdf): the same reference as a PDF
- [`docs/architecture.svg`](docs/architecture.svg) / [`docs/architecture-dark.svg`](docs/architecture-dark.svg): architecture diagram (light / dark)

## License

[MIT](LICENSE)
