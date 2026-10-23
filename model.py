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
    -> SignalTokenizer   (N, T, d_model)                 decomposition, feature gate, conv, time features
    -> SignalEncoder     (N, T, d_model) = memory        causal self-attention blocks
    -> SignalDecoder     (N, out_channels, output_length) learnable queries, cross-attention, output head

    T = L when conv_stride = 1. With conv_stride = s > 1, each token summarizes a patch of input steps and
    T = ceil(L / s), which shortens the sequence the attention layers see (patching).

    revin=True wraps the model in reversible instance normalization: each input window is normalized with its
    own per-signal mean and std, and the forecast is mapped back with the same statistics. It only uses the
    input window, so it stays causal, and it lets the model follow level shifts between training and test
    periods. It requires out_channels == in_channels (the forecast targets are the input signals).

    channel_independent=True runs every signal through the whole model as its own univariate series (as in
    PatchTST): (N, C, L) is reshaped to (N * C, 1, L), all weights are shared across signals, and signals never
    interact. This removes the bottleneck of compressing many signals into one d_model token, at the cost of
    ignoring cross-signal information and running C times as many sequences. It requires
    out_channels == in_channels. The L0 gate then selects frequency bands shared by all signals.

    ci_latents = K > 0 (with channel_independent=True) first mixes the C signals into K latent signals with a
    learnable per-step projection (C -> K), runs each latent channel-independently, then concatenates the K
    latents' decoder outputs at every forecast step and maps them back to C signals with the output head. Signals
    can then share information through the projection while the Transformer still sees univariate sequences, and
    K < C runs fewer sequences than plain channel independence. RevIN is applied to the C original signals.

    front_end selects the first stage of the tokenizer: "ma" (fixed moving-average decomposition) or
    "conv_bank" (a learnable bank of causal depthwise convolutions, see CausalConvBank).

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
        conv_stride: int = 1,
        cross_channel: bool = True,
        revin: bool = False,
        channel_independent: bool = False,
        ci_latents: int = 0,
        front_end: str = "ma",
        bank_filters: int = 8,
    ):
        super().__init__()
        if revin:
            assert out_channels == in_channels, "revin requires out_channels == in_channels"
        assert ci_latents == 0 or channel_independent, "ci_latents requires channel_independent=True"
        self.channel_independent = channel_independent
        # Latent projection (C -> K signals per time step); RevIN then normalizes the C original signals
        self.latent_proj = nn.Linear(in_channels, ci_latents, bias=False) if ci_latents > 0 else None
        revin_channels = 1 if channel_independent and ci_latents == 0 else in_channels
        self.revin = RevIN(revin_channels) if revin else None
        if channel_independent:
            assert out_channels == in_channels, "channel_independent requires out_channels == in_channels"
            # every submodule sees one signal at a time; with latents the output head maps K latents back to C
            in_channels = 1
            out_channels = out_channels if ci_latents > 0 else 1
        self.tokenizer = SignalTokenizer(
            in_channels,
            d_model,
            kernel_size,
            ma_windows,
            n_rank,
            time_dim,
            conv_stride,
            cross_channel,
            front_end,
            bank_filters,
        )
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
            latent_signals=max(ci_latents, 1),
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
        batch = input.size(0)
        stats = None
        if self.latent_proj is not None:
            if self.revin is not None:
                input, stats = self.revin.normalize(input)  # normalize the original signals before mixing them
            input = self.latent_proj(input.transpose(1, 2)).transpose(1, 2)  # (N, C, L) -> (N, K, L)
        num_signals = input.size(1)
        if self.channel_independent:
            # (N, C, L) -> (N * C, 1, L); per-sample inputs are repeated for each of its signals
            input = input.reshape(batch * num_signals, 1, -1)
            repeat = lambda t: None if t is None else t.repeat_interleave(num_signals, dim=0)  # noqa: E731
            time_features, future_time_features = repeat(time_features), repeat(future_time_features)
            text_memory, text_padding_mask = repeat(text_memory), repeat(text_padding_mask)
        if self.revin is not None and self.latent_proj is None:
            input, stats = self.revin.normalize(input)
        tokens = self.tokenizer(input, time_features)  # (N, T, d_model)
        memory = self.encoder(tokens)  # (N, T, d_model)
        future_time = self.tokenizer.embed_time(future_time_features) if future_time_features is not None else None
        # Positions in input-time units: token j ends at input step memory_offset + j * memory_stride, and the
        # first forecast step is input step L.
        input_len = input.size(-1)
        memory_offset, memory_stride = self.tokenizer.token_positions(input_len)
        features = self.decoder.features(
            memory,
            future_time,
            text_memory,
            text_padding_mask,
            query_offset=input_len,
            memory_offset=memory_offset,
            memory_stride=memory_stride,
        )  # (N, output_length * samples_per_output, d_model)
        if self.latent_proj is not None:
            # (N * K, L_dec, d) -> (N, L_dec, K * d): the K latents' outputs of each step side by side
            l_dec, d = features.shape[1:]
            features = features.view(batch, num_signals, l_dec, d).transpose(1, 2).reshape(batch, l_dec, -1)
        output = self.decoder.head(features)  # (N, out_channels, output_length)
        if self.revin is not None and stats is not None:
            output = self.revin.denormalize(output, stats)
        if self.channel_independent and self.latent_proj is None:
            output = output.reshape(batch, num_signals, -1)
        return output

    def penalty(self) -> torch.Tensor:
        return self.tokenizer.penalty()


class RevIN(nn.Module):
    """Reversible instance normalization (Kim et al., 2022) over the time axis of each input window.

    normalize() returns the statistics instead of storing them, so the module holds no per-batch state.
    The optional affine weight and bias are learned per signal.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels)) if affine else None
        self.bias = nn.Parameter(torch.zeros(num_channels)) if affine else None

    def normalize(self, input: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # input: (N, C, L); statistics over the window's own time axis only, so no future data is used
        mean = input.mean(dim=-1, keepdim=True)
        std = (input.var(dim=-1, keepdim=True, unbiased=False) + self.eps).sqrt()
        x = (input - mean) / std
        if self.weight is not None and self.bias is not None:
            x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x, (mean, std)

    def denormalize(self, output: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # output: (N, C, H)
        mean, std = stats
        if self.weight is not None and self.bias is not None:
            output = (output - self.bias.view(1, -1, 1)) / (self.weight.view(1, -1, 1) + self.eps)
        return output * std + mean


class SignalTokenizer(nn.Module):
    """Turns raw signals into a sequence of d_model tokens (causal).

    input (N, C, L)
    -> MultiScaleDecomp         (N, B*C, L)     B = len(ma_windows) + 1 frequency bands
       or CausalConvBank                        B = 1 + len(ma_windows) * bank_filters learned filters
    -> L0Gate                   (N, B*C, L)     feature selection
    -> LowRankCausalConv1d+GELU (N, d_model, T)
    -> transpose (+ time features, which bypass the conv)  (N, T, d_model)

    conv_stride = s: one token per s input steps (patching), T = ceil(L / s). Token j covers the input steps
    ending at token_positions(L)[0] + j * s, so the last token always ends at the last input step. kernel_size
    must be >= conv_stride, otherwise some input steps would fall between kernels and never be seen. With long
    patches raise n_rank: each channel's patch is compressed to n_rank numbers before channels are mixed, so
    n_rank = 1 keeps only one weighted average per patch, while n_rank >= kernel_size loses nothing.

    cross_channel = False: the 1x1 channel conv only mixes the bands of the same signal (grouped by signal),
    keeping signals independent inside the tokenizer. Each signal then owns d_model / C dimensions of the
    token, so d_model must be divisible by C. Signals still interact later through attention.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        kernel_size: int,
        ma_windows: tuple[int, ...] = (5, 20, 60),
        n_rank: int = 1,
        time_dim: int = 0,
        conv_stride: int = 1,
        cross_channel: bool = True,
        front_end: str = "ma",
        bank_filters: int = 8,
    ) -> None:
        super().__init__()
        assert front_end in ("ma", "conv_bank"), "front_end must be 'ma' or 'conv_bank'"
        assert conv_stride >= 1, "conv_stride must be >= 1"
        assert kernel_size >= conv_stride, "kernel_size must be >= conv_stride so every input step is covered"
        if not cross_channel:
            assert d_model % in_channels == 0, "d_model must be divisible by in_channels when cross_channel=False"
        self.in_channels = in_channels
        self.conv_stride = conv_stride
        self.cross_channel = cross_channel
        # ma_windows sets the moving-average windows, or the kernel sizes of the conv bank
        if front_end == "ma":
            self.decomp = MultiScaleDecomp(ma_windows)
        else:
            self.decomp = CausalConvBank(in_channels, ma_windows, bank_filters)
        decomp_channels = self.decomp.num_bands * in_channels  # each signal is split into num_bands bands
        self.gate = L0Gate(decomp_channels)
        self.conv = LowRankCausalConv1d(
            decomp_channels,
            d_model,
            kernel_size,
            n_rank,
            stride=conv_stride,
            channel_groups=1 if cross_channel else in_channels,
        )
        # Time features (e.g. calendar encodings) bypass the conv; they are projected to d_model and added to
        # the tokens. The projection is shared with the decoder's future time features (via embed_time) so the
        # same date means the same thing on both sides. time_dim=0 disables them.
        self.time_proj = nn.Linear(time_dim, d_model) if time_dim > 0 else None

    def token_positions(self, length: int) -> tuple[int, int]:
        """(offset, stride) such that token j ends at input step offset + j * stride."""
        return (length - 1) % self.conv_stride, self.conv_stride

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
        x = self.gate(self.decomp(input))  # (N, B*C, L), band-major
        if not self.cross_channel:
            # Grouped channel conv needs each signal's bands to be contiguous: band-major -> signal-major
            n, _, length = x.shape
            x = x.view(n, -1, self.in_channels, length).transpose(1, 2).reshape(n, -1, length)
        x = F.gelu(self.conv(x))  # (N, d_model, T)
        x = x.transpose(1, 2)  # (N, T, d_model)
        if time_features is not None:
            # Use the time features of the last input step each token covers
            offset, stride = self.token_positions(input.size(-1))
            x = x + self.embed_time(time_features[:, offset::stride])
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
        latent_signals: int = 1,
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
        # latent_signals > 1: the head also receives the concatenated features of that many latent signals
        # (Signalformer ci_latents), so its input is latent_signals * samples_per_output * d_model wide.
        self.samples_per_output = samples_per_output
        self.output_head = nn.Sequential(
            nn.Linear(latent_signals * samples_per_output * d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, out_channels),
        )

    def forward(
        self,
        memory: torch.Tensor,
        future_time: torch.Tensor | None = None,
        text_memory: torch.Tensor | None = None,
        text_padding_mask: torch.Tensor | None = None,
        query_offset: int | None = None,
        memory_offset: int = 0,
        memory_stride: int = 1,
    ) -> torch.Tensor:
        features = self.features(
            memory, future_time, text_memory, text_padding_mask, query_offset, memory_offset, memory_stride
        )
        return self.head(features)  # (N, out_channels, output_length)

    def features(
        self,
        memory: torch.Tensor,
        future_time: torch.Tensor | None = None,
        text_memory: torch.Tensor | None = None,
        text_padding_mask: torch.Tensor | None = None,
        query_offset: int | None = None,
        memory_offset: int = 0,
        memory_stride: int = 1,
    ) -> torch.Tensor:
        """Normalized decoder outputs before the head: (N, output_length * samples_per_output, d_model)."""
        # memory: (N, T, d_model) from the encoder
        # Positions for RoPE, in input-time units: memory token j sits at memory_offset + j * memory_stride and
        # decoder query k at query_offset + k. query_offset defaults to T (memory at stride 1).
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
        if query_offset is None:
            query_offset = memory.size(1)
        for block in self.blocks:
            y = block(
                y,
                memory,
                offset=query_offset,
                memory_offset=memory_offset,
                memory_stride=memory_stride,
                text_memory=text_memory,
                text_padding_mask=text_padding_mask,
            )
        return self.norm(y)

    def head(self, features: torch.Tensor) -> torch.Tensor:
        # features: (N, output_length * s, width) with width = latent_signals * d_model
        n, l_dec, d = features.shape
        y = features.reshape(n, l_dec // self.samples_per_output, self.samples_per_output * d)  # (N, output_length, s * d)
        y = self.output_head(y)  # (N, output_length, out_channels)
        return y.transpose(1, 2)  # (N, out_channels, output_length), same layout as the input


class LowRankCausalConv1d(nn.Module):
    """Sum of n_rank rank-1 causal convolutions (depthwise separable).

    Equivalent kernel: W[o, c, j] = sum_r B[o, c, r] * a[c, r, j]
    First convolve each channel along time (a), then mix channels at each time step (B).
    The rank is capped at min(kernel_size, out_channels); n_rank >= kernel_size is as expressive as a full
    Conv1d, so larger values only add parameters.
    Do not insert an activation between the two convs, otherwise it is no longer a low-rank factorization.
    stride > 1 makes the time conv emit one output per stride input steps (see CausalConv1d).
    channel_groups > 1 splits the channel mixing into independent groups of consecutive input channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        n_rank: int = 1,
        stride: int = 1,
        channel_groups: int = 1,
    ) -> None:
        super().__init__()
        assert in_channels % channel_groups == 0 and out_channels % channel_groups == 0, (
            "in_channels and out_channels must be divisible by channel_groups"
        )
        # groups=in_channels: depthwise, each channel is convolved along time independently, no channel mixing
        self.time_conv = CausalConv1d(
            in_channels=in_channels,
            out_channels=in_channels * n_rank,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
        )
        # kernel_size=1: mixes channels at each time step only (equivalent to a per-step Linear), so it
        # is causal by construction. Transposing to (N, L, C) instead would treat time steps as channels and
        # mix in the future.
        self.channel_conv = nn.Conv1d(in_channels * n_rank, out_channels, kernel_size=1, groups=channel_groups)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # (N, C_in, L)
        x = self.time_conv(input)  # (N, C_in * n_rank, T)
        return self.channel_conv(x)  # (N, C_out, T)


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


class CausalConvBank(nn.Module):
    """Learnable alternative to MultiScaleDecomp: a bank of causal depthwise (per-signal) 1-D convolutions.

    For every kernel size k in kernel_sizes, each signal is convolved with filters_per_kernel learned filters of
    length k. The raw signal is kept as band 0, and filter 0 of each size starts as a trailing moving average, so
    at initialization the bank spans the same space as MultiScaleDecomp with the same windows; training is then
    free to turn the other filters into differences, band-passes or any other causal pattern.
    - Filters are separate for each signal (groups = in_channels) and never mix signals.
    - Left zero padding only (see CausalConv1d); the first k - 1 steps see incomplete windows.
    - Output layout is band-major like MultiScaleDecomp: [band 0 of all C signals, band 1 of all C signals, ...].
    """

    def __init__(self, in_channels: int, kernel_sizes: tuple[int, ...] | list[int], filters_per_kernel: int = 8) -> None:
        super().__init__()
        assert len(kernel_sizes) > 0 and filters_per_kernel >= 1
        self.filters_per_kernel = filters_per_kernel
        self.num_bands = 1 + len(kernel_sizes) * filters_per_kernel
        self.convs = nn.ModuleList(
            CausalConv1d(in_channels, in_channels * filters_per_kernel, k, groups=in_channels, bias=False)
            for k in kernel_sizes
        )
        with torch.no_grad():
            for conv, k in zip(self.convs, kernel_sizes):
                assert isinstance(conv, CausalConv1d)
                # weight (C * F, 1, k): output channel c * F + f is filter f of signal c
                conv.weight.view(in_channels, filters_per_kernel, k)[:, 0] = 1.0 / k

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # (N, C, L) -> (N, num_bands * C, L)
        n, c, length = input.shape
        bands = [input]
        for conv in self.convs:
            y = conv(input).view(n, c, self.filters_per_kernel, length)  # signal-major
            bands.append(y.transpose(1, 2).reshape(n, -1, length))  # -> band-major
        return torch.cat(bands, dim=1)


class L0Gate(nn.Module):
    """Hard Concrete gate (Louizos et al., 2018): learns a per-channel weight that can be exactly 0.

    - Standardize inputs first, otherwise gate values reflect feature scale rather than importance.
    - penalty() is the expected fraction of open channels (0 to 1), so lam means the same thing for any number
      of channels. Too large closes every gate, too small has no effect; lam around 1e-2 is a starting point
      for a z-scored MSE loss.
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
        """Expected fraction of open channels (differentiable approximation of L0, normalized by count)."""
        shift = self.beta * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.log_alpha - shift).mean()


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

    def forward(self, input: torch.Tensor, offset: int = 0, stride: int = 1) -> torch.Tensor:  # (..., L, dim)
        # offset: absolute position of the first element; stride: position gap between consecutive elements
        positions = offset + stride * torch.arange(input.size(-2), device=input.device, dtype=torch.float32)
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
        key_offset: int = 0,
        key_stride: int = 1,
    ) -> torch.Tensor:
        # query: (N, L_q, D), memory: (N, L_kv, kv_dim)
        # RoPE positions: query i at query_offset + i, memory j at key_offset + j * key_stride (same time units)
        # key_padding_mask: (N, L_kv) bool, True = padding to ignore
        n, l_q, d = query.shape
        l_kv = memory.size(1)
        if is_causal:
            assert l_q == l_kv and query_offset == key_offset == 0 and key_stride == 1, "causal cross-attention requires time-aligned sequences"
        q = self.q_proj(query).view(n, l_q, self.num_heads, -1).transpose(1, 2)  # (N, H, L_q, D/H)
        k, v = self.kv_proj(memory).chunk(2, dim=-1)
        k, v = (t.view(n, l_kv, self.num_heads, -1).transpose(1, 2) for t in (k, v))  # (N, H, L_kv, D/H)
        if self.rope is not None:
            q, k = self.rope(q, query_offset), self.rope(k, key_offset, key_stride)
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
        memory_offset: int = 0,
        memory_stride: int = 1,
    ) -> torch.Tensor:
        # input: (N, L, D), memory: (N, L_kv, kv_dim); offset is the position of the first input element and
        # memory token j sits at memory_offset + j * memory_stride
        # text_memory: (N, L_text, text_dim); text_padding_mask: (N, L_text) bool, True = padding
        x = input + self.dropout(self.self_attn(self.self_attn_norm(input), offset))
        x = x + self.dropout(
            self.cross_attn(
                self.cross_attn_norm(x),
                self.memory_norm(memory),
                is_causal,
                query_offset=offset,
                key_offset=memory_offset,
                key_stride=memory_stride,
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
    With stride > 1 the windows are aligned so the last output ends exactly at the last input step: output j
    ends at input step (L - 1) % stride + j * stride, and the output length is ceil(L / stride).
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
        self.conv_stride = stride

    def forward(self, input):
        # Shift the stride grid so the last window ends at the last input step (a negative pad crops the left)
        x = F.pad(input, (self.left_pad - (input.size(-1) - 1) % self.conv_stride, 0))
        return super().forward(x)
