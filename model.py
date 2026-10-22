"""Causal encoder-decoder Transformer for multivariate time-series forecasting.

General notes:
- Causality must hold for the whole pipeline, not only the model. Normalize inputs with statistics computed
  from past data only (no full-sample z-score, no centered rolling windows). Never use BatchNorm on the time
  axis; LayerNorm / RMSNorm normalize over features only and are safe.
- Prefer stationary inputs (differences, log ratios, rates of change) over raw levels that drift. RMSNorm does
  not remove the mean, so offsets must be handled before the model.
- Validate with walk-forward (time-ordered) splits, never random splits, or leakage will hide overfitting.
- Mixed precision: prefer bf16 autocast (no GradScaler needed). With fp16 use torch.amp.GradScaler. Keep the
  input data in fp32 and let autocast cast inside the model; small values (e.g. ~1e-4) lose precision in
  fp16/bf16. Compute evaluation metrics in fp32.
- Dropout is still useful alongside normalization: normalization stabilizes training, dropout regularizes.
  Start around 0.1 and raise it if the train/validation gap is large, especially for noisy data with a low
  signal-to-noise ratio.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class Signalformer(nn.Module):
    """Signalformer: multi-scale causal decomposition + convolutional tokenizer + causal encoder-decoder Transformer.

    Pipeline:

    input (N, C, L)
    -> SignalTokenizer   (N, L, d_model)                 decomposition, feature gate, conv, time features
    -> SignalEncoder     (N, L, d_model) = memory        causal self-attention blocks
    -> SignalDecoder     (N, out_channels, output_length) learnable queries, cross-attention, output head

    Training: add the gate penalty to the loss, e.g. loss = task_loss + lam * model.penalty().
    """

    def __init__(
        self,
        in_channels: int,
        in_len: int,
        d_model: int,
        kernel_size: int,
        output_length: int,
        num_heads: int,
        ma_windows: tuple[int, ...] = (5, 20, 60),
        n_rank: int = 1,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
        time_dim: int = 0,
        out_channels: int = 1,
        samples_per_output: int = 1,
        text_dim: int = 0,
    ):
        super().__init__()
        self.tokenizer = SignalTokenizer(in_channels, d_model, kernel_size, ma_windows, n_rank, time_dim)
        self.encoder = SignalEncoder(d_model, num_heads, num_encoder_layers, dropout)
        self.decoder = SignalDecoder(
            d_model,
            num_heads,
            num_decoder_layers,
            output_length,
            dropout,
            out_channels,
            samples_per_output,
            text_dim,
        )

    def forward(
        self,
        input: torch.Tensor,
        time_features: torch.Tensor | None = None,
        future_time_features: torch.Tensor | None = None,
        text_memory: torch.Tensor | None = None,
        text_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # input: (N, C, L), time_features: (N, L, time_dim)
        # future_time_features: (N, output_length * samples_per_output, time_dim)
        # text_memory: (N, L_text, text_dim), text_padding_mask: (N, L_text) bool, True = padding
        # See SignalTokenizer.forward and SignalDecoder.forward for notes on time features and text.
        if self.tokenizer.time_proj is not None:
            assert time_features is not None and future_time_features is not None, (
                "time_features and future_time_features are required when time_dim > 0"
            )
        tokens = self.tokenizer(input, time_features)  # (N, L, d_model)
        memory = self.encoder(tokens)  # (N, L, d_model)
        future_time = self.tokenizer.embed_time(future_time_features) if future_time_features is not None else None
        return self.decoder(memory, future_time, text_memory, text_padding_mask)  # (N, out_channels, output_length)

    def penalty(self) -> torch.Tensor:
        return self.tokenizer.penalty()


class SignalTokenizer(nn.Module):
    """Turns raw signals into a sequence of d_model tokens, one per time step (causal).

    input (N, C, L)
    -> MultiScaleDecomp         (N, B*C, L)     B = len(ma_windows) + 1 frequency bands
    -> L0Gate                   (N, B*C, L)     feature selection
    -> LowRankCausalConv1d+GELU (N, d_model, L)
    -> transpose (+ time features, which bypass the conv)  (N, L, d_model)
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        kernel_size: int,
        ma_windows: tuple[int, ...] = (5, 20, 60),
        n_rank: int = 1,
        time_dim: int = 0,
    ) -> None:
        super().__init__()
        self.decomp = MultiScaleDecomp(ma_windows)
        decomp_channels = self.decomp.num_bands * in_channels  # each signal is split into num_bands bands
        self.gate = L0Gate(decomp_channels)
        self.conv = LowRankCausalConv1d(decomp_channels, d_model, kernel_size, n_rank)
        # Time features (e.g. calendar encodings) bypass the conv; they are projected to d_model and added to
        # the tokens. The projection is shared with the decoder's future time features (via embed_time) so the
        # same date means the same thing on both sides. time_dim=0 disables them.
        self.time_proj = nn.Linear(time_dim, d_model) if time_dim > 0 else None

    def embed_time(self, time_features: torch.Tensor) -> torch.Tensor:  # (N, L, time_dim) -> (N, L, d_model)
        assert self.time_proj is not None, "time features given but time_dim == 0"
        return self.time_proj(time_features)

    def forward(self, input: torch.Tensor, time_features: torch.Tensor | None = None) -> torch.Tensor:
        # input: (N, C, L), time_features: (N, L, time_dim)
        #
        # Time feature notes:
        # - Encode cyclical fields (day of week, month, day of year) with sin/cos pairs, not raw integers,
        #   so that e.g. December and January are close.
        # - Use 0/1 flags for known events (month end, holidays, scheduled events).
        # - Do not encode the year with sin/cos: it is not periodic. Avoid absolute year features altogether,
        #   since walk-forward test years are always outside the training range (extrapolation). Prefer features
        #   that describe the current state or regime of the system instead. If a year is unavoidable, use a linear
        #   scaling such as (year - base) / 10 or Time2Vec.
        # - Future time features (decoder side) may only contain information known in advance; anything only
        #   observed on that day leaks the future.
        x = self.gate(self.decomp(input))  # (N, B*C, L)
        x = F.gelu(self.conv(x))  # (N, d_model, L)
        x = x.transpose(1, 2)  # (N, L, d_model)
        if time_features is not None:
            x = x + self.embed_time(time_features)
        return x

    def penalty(self) -> torch.Tensor:
        return self.gate.penalty()


class SignalEncoder(nn.Module):
    """Stack of causal self-attention blocks followed by a final RMSNorm (pre-norm needs one at the end)."""

    def __init__(self, d_model: int, num_heads: int, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(SelfTransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers))
        self.norm = nn.RMSNorm(d_model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # (N, L, d_model)
        x = input
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class SignalDecoder(nn.Module):
    """Non-autoregressive decoder: learnable future queries -> decoder blocks -> RMSNorm -> output head.

    The decoder runs at the input frequency (output_length * samples_per_output tokens). Self-attention is
    causal; cross-attention may see all of memory since it is entirely past data. Each decoder block optionally
    cross-attends to external text embeddings. Consecutive tokens are concatenated into one output step right
    before the FFN head, so the output frequency can be coarser than the input.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        output_length: int,
        dropout: float = 0.1,
        out_channels: int = 1,
        samples_per_output: int = 1,
        text_dim: int = 0,
    ) -> None:
        super().__init__()
        # One learnable query per future step (non-autoregressive: all steps are predicted at once).
        # The queries also tell the decoder which step it is predicting.
        self.queries = nn.Parameter(torch.randn(output_length * samples_per_output, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            CrossTransformerBlock(d_model, num_heads, dropout, text_dim=text_dim) for _ in range(num_layers)
        )
        self.text_dim = text_dim  # width of external text embeddings; 0 disables text cross-attention
        self.norm = nn.RMSNorm(d_model)
        # Each output step aggregates samples_per_output decoder tokens (e.g. 7 for daily input -> weekly
        # output). The tokens are concatenated (not averaged) so the FFN can still use their order.
        self.samples_per_output = samples_per_output
        self.output_head = nn.Sequential(
            nn.Linear(samples_per_output * d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, out_channels),
        )

    def forward(
        self,
        memory: torch.Tensor,
        future_time: torch.Tensor | None = None,
        text_memory: torch.Tensor | None = None,
        text_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # memory: (N, L, d_model) from the encoder
        # future_time: (N, output_length * samples_per_output, d_model), already embedded future time features
        # text_memory: (N, L_text, text_dim), embeddings from an external (e.g. frozen pretrained) text encoder.
        # text_padding_mask: (N, L_text) bool, True marks padding to ignore (nn.MultiheadAttention convention).
        #
        # Text notes:
        # - Only include text available before the forecast time (the end of the input window). Use the
        #   time the text became available, not the time of the event it describes, and mind time zones.
        # - Text may be omitted (None) even when text_dim > 0, e.g. when no text exists for a sample; the text
        #   cross-attention is then skipped. A sample whose tokens are all padding is also handled.
        # - Text tokens have no time positions aligned with the series, so no RoPE is applied to them.
        # - A pretrained text encoder may have seen data after the training period (look-ahead bias in its
        #   knowledge); prefer encoders whose training cutoff precedes the evaluation period.
        if text_memory is not None:
            assert self.text_dim > 0, "text_memory given but text_dim == 0"
        y = self.queries.expand(memory.size(0), -1, -1)  # (N, output_length * s, d_model)
        if future_time is not None:
            y = y + future_time
        for block in self.blocks:
            # future step k sits at position L + k
            y = block(y, memory, offset=memory.size(1), text_memory=text_memory, text_padding_mask=text_padding_mask)
        y = self.norm(y)
        n, l_dec, d = y.shape
        y = y.reshape(n, l_dec // self.samples_per_output, self.samples_per_output * d)  # (N, output_length, s * d)
        y = self.output_head(y)  # (N, output_length, out_channels)
        return y.transpose(1, 2)  # (N, out_channels, output_length), same layout as the input


class LowRankCausalConv1d(nn.Module):
    """Sum of n_rank rank-1 causal convolutions (depthwise separable).

    Equivalent kernel: W[o, c, j] = sum_r B[o, c, r] * a[c, r, j]
    First convolve each channel along time (a), then mix channels at each time step (B).
    The rank is capped at min(kernel_size, out_channels); n_rank >= kernel_size is as expressive as a full
    Conv1d, so larger values only add parameters.
    Do not insert an activation between the two convs, otherwise it is no longer a low-rank factorization.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_rank: int = 1) -> None:
        super().__init__()
        # groups=in_channels: depthwise, each channel is convolved along time independently, no channel mixing
        self.time_conv = CausalConv1d(
            in_channels=in_channels,
            out_channels=in_channels * n_rank,
            kernel_size=kernel_size,
            groups=in_channels,
        )
        # kernel_size=1: mixes all channels at each time step only (equivalent to a per-step Linear), so it
        # is causal by construction. Transposing to (N, L, C) instead would treat time steps as channels and
        # mix in the future.
        self.channel_conv = nn.Conv1d(in_channels * n_rank, out_channels, kernel_size=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # (N, C_in, L)
        x = self.time_conv(input)  # (N, C_in * n_rank, L)
        return self.channel_conv(x)  # (N, C_out, L)


class MultiScaleDecomp(nn.Module):
    """Hierarchical multi-scale decomposition with trailing moving averages (a causal, wavelet-like split).

    For windows w1 < w2 < ... < wk and MA_w = trailing moving average with window w, each signal x becomes
    k + 1 frequency bands, from fine to coarse:
        x - MA_w1, MA_w1 - MA_w2, ..., MA_w(k-1) - MA_wk, MA_wk
    The bands sum back to x exactly. Like detail / approximation coefficients of a wavelet transform, each
    band holds variation at one time scale, but without an orthogonal basis or downsampling.

    This adds no new information (every band is linear in the input), but it lets the gate select time scales
    separately and keeps small fine-scale bands from being dominated by the coarse band's magnitude.
    - Do not also feed the raw signal: it equals the sum of the bands, so they would be perfectly collinear.
    - Do not use centered moving averages (as in Autoformer's series_decomp); they leak the future.
    - The first max(windows) - 1 steps use incomplete windows; drop them as warm-up when training.
    - For drifting level signals the coarse band is non-stationary; decompose a log or differenced signal.
    - Output channel layout is band-major: [band 0 of all C signals, band 1 of all C signals, ...].
    """

    def __init__(self, windows: tuple[int, ...] | list[int]) -> None:
        super().__init__()
        assert len(windows) > 0, "at least one window is required"
        assert all(w > 1 for w in windows), "windows must be > 1"
        assert list(windows) == sorted(set(windows)), "windows must be strictly increasing"
        self.windows = tuple(windows)
        self.num_bands = len(self.windows) + 1

    @staticmethod
    def _causal_moving_average(input: torch.Tensor, window: int) -> torch.Tensor:
        # Pad on the left only, replicating the first value, so the MA only uses past data
        padded = F.pad(input, (window - 1, 0), mode="replicate")
        return F.avg_pool1d(padded, kernel_size=window, stride=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # (N, C, L) -> (N, num_bands * C, L)
        smooths = [input] + [self._causal_moving_average(input, w) for w in self.windows]
        bands = [fine - coarse for fine, coarse in zip(smooths[:-1], smooths[1:])] + [smooths[-1]]
        return torch.cat(bands, dim=1)


class L0Gate(nn.Module):
    """Hard Concrete gate (Louizos et al., 2018): learns a per-channel weight that can be exactly 0.

    - Standardize inputs first, otherwise gate values reflect feature scale rather than importance.
    - Tune the penalty weight lam: too large closes every gate, too small has no effect. Start with the
      penalty at roughly 1-10% of the task loss.
    - Consider a separate optimizer param group for log_alpha (higher lr, weight_decay=0) so AdamW weight
      decay does not interfere with the penalty.
    - Highly correlated features (e.g. 5-day and 10-day MAs) may be picked arbitrarily; check that the
      selected set is stable across seeds and walk-forward folds before trusting it.
    """

    def __init__(
        self,
        num_features: int,
        init_log_alpha: float = 2.0,
        beta: float = 2 / 3,
        gamma: float = -0.1,
        zeta: float = 1.1,
    ) -> None:
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full((num_features,), init_log_alpha))
        self.beta, self.gamma, self.zeta = beta, gamma, zeta

    def gate_values(self) -> torch.Tensor:
        if self.training:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((u.log() - (1 - u).log() + self.log_alpha) / self.beta)
        else:
            s = torch.sigmoid(self.log_alpha)
        return (s * (self.zeta - self.gamma) + self.gamma).clamp(0, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # (N, C, L)
        return input * self.gate_values().view(1, -1, 1)

    def penalty(self) -> torch.Tensor:
        """Expected number of open channels (differentiable approximation of L0)."""
        shift = self.beta * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.log_alpha - shift).sum()


class RotaryEmbedding(nn.Module):
    """RoPE: rotates q / k by position so attention scores depend only on the relative distance.

    Apply it to q and k inside attention, after the projections. Applying it to the embeddings before the
    q/k projections breaks the relative-position property and turns it into an absolute encoding.
    """

    inv_freq: torch.Tensor  # registered buffer; annotated so type checkers don't infer Tensor | Module

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        assert dim % 2 == 0, "RoPE requires an even dimension"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, input: torch.Tensor, offset: int = 0) -> torch.Tensor:  # (..., L, dim)
        # offset: absolute position of the first element in the sequence
        positions = torch.arange(offset, offset + input.size(-2), device=input.device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq)  # (L, dim / 2)
        cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)  # (L, dim)
        x1, x2 = input.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
        return (input.float() * cos + rotated.float() * sin).type_as(input)


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention: position t only attends to positions <= t.

    - Attention selects time steps, not features; its weights are not a reliable measure of feature
      importance (use the L0 gate or permutation importance instead).
    - Cost is O(L^2) in sequence length.
    - Mask conventions differ across APIs: for a bool mask, nn.MultiheadAttention uses True = blocked while
      F.scaled_dot_product_attention uses True = allowed. Prefer is_causal=True, and avoid finite fill values
      such as -1e9, which overflow in fp16.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.dropout = dropout
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(d_model // num_heads)

    def forward(self, input: torch.Tensor, offset: int = 0) -> torch.Tensor:  # (N, L, D)
        n, l, d = input.shape
        q, k, v = self.qkv_proj(input).chunk(3, dim=-1)
        # (N, L, D) -> (N, H, L, D/H)
        q, k, v = (t.view(n, l, self.num_heads, -1).transpose(1, 2) for t in (q, k, v))
        q, k = self.rope(q, offset), self.rope(k, offset)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).reshape(n, l, d)  # concatenate the heads
        return self.out_proj(y)


class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries from one sequence, keys / values from another (memory).

    With is_causal=False every query sees all of memory, which is only safe when memory contains no future
    information. is_causal=True requires the two sequences to be time-aligned (L_q == L_kv); query position t
    then only sees memory positions <= t.
    use_rope=False disables RoPE, for memory without time positions aligned to the queries (e.g. text tokens).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        kv_dim: int | None = None,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(kv_dim or d_model, 2 * d_model)  # memory may have a different width
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(d_model // num_heads) if use_rope else None

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        is_causal: bool = False,
        query_offset: int = 0,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # query: (N, L_q, D), memory: (N, L_kv, kv_dim)
        # Memory positions are 0..L_kv-1; query_offset is the position of the first query (L_kv when forecasting)
        # key_padding_mask: (N, L_kv) bool, True = padding to ignore
        n, l_q, d = query.shape
        l_kv = memory.size(1)
        if is_causal:
            assert l_q == l_kv and query_offset == 0, "causal cross-attention requires time-aligned sequences"
        q = self.q_proj(query).view(n, l_q, self.num_heads, -1).transpose(1, 2)  # (N, H, L_q, D/H)
        k, v = self.kv_proj(memory).chunk(2, dim=-1)
        k, v = (t.view(n, l_kv, self.num_heads, -1).transpose(1, 2) for t in (k, v))  # (N, H, L_kv, D/H)
        if self.rope is not None:
            q, k = self.rope(q, query_offset), self.rope(k)
        attn_mask = None
        if key_padding_mask is not None:
            assert not is_causal, "key_padding_mask is not supported together with is_causal"
            # F.scaled_dot_product_attention bool masks use True = allowed, the opposite of key_padding_mask
            attn_mask = ~key_padding_mask.view(n, 1, 1, l_kv)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, dropout_p=self.dropout if self.training else 0.0
        )
        y = self.out_proj(y.transpose(1, 2).reshape(n, l_q, d))
        if key_padding_mask is not None:
            # A sample whose keys are all padding gets NaN from softmax; zero its output (after out_proj, so the
            # bias is not added either) so it contributes nothing to the residual
            all_padded = key_padding_mask.all(dim=-1).view(n, 1, 1)
            y = y.masked_fill(all_padded, 0.0)
        return y


class CrossTransformerBlock(nn.Module):
    """Pre-norm decoder block: causal self-attention -> cross-attention -> [text cross-attention] -> FFN.

    Each sub-layer has a residual. The text cross-attention exists only when text_dim > 0 and is skipped when
    no text_memory is passed.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        dim_feedforward: int | None = None,
        kv_dim: int | None = None,
        text_dim: int = 0,
    ) -> None:
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.self_attn_norm = nn.RMSNorm(d_model)
        self.self_attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.cross_attn_norm = nn.RMSNorm(d_model)
        self.memory_norm = nn.RMSNorm(kv_dim or d_model)
        self.cross_attn = CrossAttention(d_model, num_heads, dropout, kv_dim)
        if text_dim > 0:
            self.text_attn_norm = nn.RMSNorm(d_model)
            self.text_memory_norm = nn.RMSNorm(text_dim)
            self.text_attn = CrossAttention(d_model, num_heads, dropout, kv_dim=text_dim, use_rope=False)
        else:
            self.text_attn = None
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input: torch.Tensor,
        memory: torch.Tensor,
        is_causal: bool = False,
        offset: int = 0,
        text_memory: torch.Tensor | None = None,
        text_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # input: (N, L, D), memory: (N, L_kv, kv_dim); offset is the position of the first input element
        # text_memory: (N, L_text, text_dim); text_padding_mask: (N, L_text) bool, True = padding
        x = input + self.dropout(self.self_attn(self.self_attn_norm(input), offset))
        x = x + self.dropout(
            self.cross_attn(
                self.cross_attn_norm(x),
                self.memory_norm(memory),
                is_causal,
                query_offset=offset,
            )
        )
        if self.text_attn is not None and text_memory is not None:
            x = x + self.dropout(
                self.text_attn(
                    self.text_attn_norm(x),
                    self.text_memory_norm(text_memory),
                    key_padding_mask=text_padding_mask,
                )
            )
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class SelfTransformerBlock(nn.Module):
    """Pre-norm Transformer block: RMSNorm -> causal attention -> residual, RMSNorm -> FFN -> residual."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, dim_feedforward: int | None = None) -> None:
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.attn_norm = nn.RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # (N, L, D)
        x = input + self.dropout(self.attn(self.attn_norm(input)))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class CausalConv1d(nn.Conv1d):
    """Conv1d whose output at time t only depends on inputs at times <= t.

    The built-in padding argument pads both sides symmetrically, so padding is fixed to 0 and forward pads
    (kernel_size - 1) * dilation zeros on the left only. Output length equals input length when stride=1.
    Only zero padding is safe: 'reflect' and 'circular' would copy future values into the left padding.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            **kwargs,
        )
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, input):
        x = F.pad(input, (self.left_pad, 0))
        return super().forward(x)
