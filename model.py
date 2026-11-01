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
    latent_norm=True additionally normalizes each latent over its window (mean 0, std 1, no affine), so the
    latents reach the model at a fixed scale whatever the projection weights are. The forecast is still mapped
    back with the original signals' RevIN statistics.

    front_end selects the first stage of the tokenizer: "ma" (fixed moving-average decomposition), "conv_bank"
    (a learnable bank of causal depthwise convolutions, see CausalConvBank) or "none" (the signal goes straight
    into the tokenizer conv, without decomposition or L0 gate; raise n_rank and kernel_size to compensate).

    freq_tokens=True adds a frequency-domain memory: the rFFT of each (normalized) input window becomes one
    token per frequency bin, and every decoder block cross-attends to it (see FrequencyEmbedding).

    raw_tokens=True adds a raw-signal memory: every (normalized) input step becomes one token (its C values
    projected to d_model), and every decoder block cross-attends to it with RoPE at the step's input position.
    Unlike raw_skip, which adds each patch's raw values to the encoder tokens, this keeps the input at full
    time resolution for the decoder however coarse the patching is.

    cross_decoder=True (the "Cross Transformer", with channel_independent=True) lets the signals exchange
    information in the decoder only, as a complete graph: each signal is tokenized and encoded on its own as
    usual, then the decoder of every signal cross-attends to the encoder memory of every signal, itself
    included, one after another in signal order (see SignalCrossBlock). With raw_tokens the raw-input
    cross-attention comes first. A learnable signal embedding is added to each memory (which signal it comes
    from) and to each signal's decoder queries (which signal is being forecast). Each layer runs C
    cross-attentions over C sequences, so the cost grows with C^2.
    edge_gates="l1" (or True) adds a learnable C x C gate per decoder layer on the edges of that graph, with an
    L1 penalty; edge_gates="l0" adds a Hard Concrete mask to every cross-signal edge, with an L0 penalty that can
    close edges exactly (see SignalCrossBlock). Add lam * model.edge_penalty() to the loss to keep the
    cross-signal edges sparse.
    cross_parallel=True runs the C cross-attentions of each decoder layer in parallel instead of in signal order
    and merges them like the heads of multi-head attention: one concatenation and one C * D -> D output projection
    (see SignalParallelCrossBlock). The result does not depend on the order of the signals.

    output_signals sets the number of forecast signals after the output projection, which maps the C
    per-signal forecasts of every step to output_signals values (with channel_independent=True). 0 or C keeps
    an identity mapping (no parameters, the forecast targets are the input signals); any other value uses a
    per-step FFN across signals (C -> 4 * max(C, output_signals) -> output_signals). It runs after RevIN has
    restored the input scale, since new output signals have no RevIN statistics of their own.

    cycle_len = W > 0 adds residual cycle forecasting (as in CycleNet): a learnable cycle of W steps per signal,
    shared by all windows and indexed by the absolute time step modulo W (forward's phase argument, the absolute
    index of each window's first input step). Each input window is normalized per signal (mean 0, std 1 over the
    window), the cycle values of its steps are subtracted, the rest of the model forecasts the residual, and the
    cycle values of the forecast steps are added back before the window's mean and std are restored. Set W to
    the period of the data in steps (one day: 96 at 15 minutes, 144 at 10 minutes).

    num_classes > 0 turns the model into a classifier of the whole window: the decoder features are averaged
    over the decoder steps, the per-signal averages are concatenated (C * d_model with channel independence),
    and an FFN maps them to num_classes logits. forward then returns (N, num_classes); the forecast head, RevIN
    denormalization and output projection are skipped. output_length sets the number of decoder steps.

    Training: add the gate penalty to the loss, e.g. loss = task_loss + lam * model.penalty().
    """

    def __init__(
        self,
        in_channels: int,
        in_len: int,
        d_model: int,
        kernel_size: int | tuple[int, ...],
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
        raw_skip: bool = False,
        latent_norm: bool = False,
        freq_tokens: bool = False,
        raw_tokens: bool = False,
        cross_decoder: bool = False,
        cross_parallel: bool = False,
        edge_gates: bool | str = False,
        output_signals: int = 0,
        num_classes: int = 0,
        cycle_len: int = 0,
    ):
        super().__init__()
        if revin:
            assert out_channels == in_channels, "revin requires out_channels == in_channels"
        assert ci_latents == 0 or channel_independent, "ci_latents requires channel_independent=True"
        assert not latent_norm or ci_latents > 0, "latent_norm requires ci_latents > 0"
        assert not cross_decoder or (channel_independent and ci_latents == 0 and not freq_tokens and text_dim == 0), (
            "cross_decoder requires channel_independent=True without ci_latents, freq_tokens or text"
        )
        edge_gates = "l1" if edge_gates is True else (edge_gates or "")
        assert edge_gates in ("", "l1", "l0"), "edge_gates must be False, True, 'l1' or 'l0'"
        assert not edge_gates or cross_decoder, "edge_gates requires cross_decoder=True"
        assert not cross_parallel or cross_decoder, "cross_parallel requires cross_decoder=True"
        assert output_signals in (0, in_channels) or channel_independent, "output_signals requires channel_independent"
        assert num_classes == 0 or output_signals in (0, in_channels), "num_classes replaces the output projection"
        assert cycle_len == 0 or (num_classes == 0 and out_channels == in_channels and output_signals in (0, in_channels)), (
            "cycle_len requires a forecast of the input signals"
        )
        self.cycle = nn.Parameter(torch.zeros(cycle_len, in_channels)) if cycle_len > 0 else None
        self.output_length = output_length
        if num_classes > 0:
            if channel_independent:
                width = (ci_latents or in_channels) * d_model
            else:
                width = d_model
            self.class_head: nn.Module | None = nn.Sequential(
                nn.Linear(width, 4 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d_model, num_classes)
            )
        else:
            self.class_head = None
        if output_signals in (0, in_channels):
            self.output_proj: nn.Module = nn.Identity()
        else:
            hidden = 4 * max(in_channels, output_signals)
            self.output_proj = nn.Sequential(
                nn.Linear(in_channels, hidden), nn.GELU(), nn.Linear(hidden, output_signals)
            )
        self.latent_norm = RevIN(ci_latents, affine=False) if latent_norm else None
        self.channel_independent = channel_independent
        # Latent projection (C -> K signals per time step); RevIN then normalizes the C original signals
        self.latent_proj = nn.Linear(in_channels, ci_latents, bias=False) if ci_latents > 0 else None
        revin_channels = 1 if channel_independent and ci_latents == 0 else in_channels
        self.revin = RevIN(revin_channels) if revin else None
        num_signals = in_channels
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
            raw_skip,
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
            freq_tokens=freq_tokens,
            raw_tokens=raw_tokens,
            cross_signals=num_signals if cross_decoder else 0,
            cross_parallel=cross_parallel,
            edge_gates=edge_gates,
        )
        # The frequency and raw tokens describe the signals the tokenizer sees (one signal with channel independence)
        self.freq_embed = FrequencyEmbedding(in_channels, in_len, d_model) if freq_tokens else None
        self.raw_embed = nn.Linear(in_channels, d_model) if raw_tokens else None

    def forward(
        self,
        input: torch.Tensor,
        time_features: torch.Tensor | None = None,
        future_time_features: torch.Tensor | None = None,
        text_memory: torch.Tensor | None = None,
        text_padding_mask: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # phase: (N,) absolute time step of each window's first input step, required with cycle_len > 0
        if self.cycle is None:
            return self.forward_residual(input, time_features, future_time_features, text_memory, text_padding_mask)
        assert phase is not None, "phase is required with cycle_len > 0"
        input_len, output_len = input.size(-1), self.output_length
        steps = torch.arange(input_len + output_len, device=input.device)
        cycle = self.cycle[(phase[:, None] + steps) % self.cycle.size(0)].transpose(1, 2)  # (N, C, L + H)
        mean = input.mean(dim=-1, keepdim=True)
        std = (input.var(dim=-1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        residual = (input - mean) / std - cycle[..., :input_len]
        output = self.forward_residual(residual, time_features, future_time_features, text_memory, text_padding_mask)
        return (output + cycle[..., input_len:]) * std + mean

    def forward_residual(
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
            if self.latent_norm is not None:
                input, _ = self.latent_norm.normalize(input)  # input side only; outputs are C signals, not latents
        num_signals = input.size(1)
        if self.channel_independent:
            # (N, C, L) -> (N * C, 1, L); per-sample inputs are repeated for each of its signals
            input = input.reshape(batch * num_signals, 1, -1)
            repeat = lambda t: None if t is None else t.repeat_interleave(num_signals, dim=0)  # noqa: E731
            time_features, future_time_features = repeat(time_features), repeat(future_time_features)
            text_memory, text_padding_mask = repeat(text_memory), repeat(text_padding_mask)
        if self.revin is not None and self.latent_proj is None:
            input, stats = self.revin.normalize(input)
        freq_memory = self.freq_embed(input) if self.freq_embed is not None else None  # (N, L // 2 + 1, d_model)
        raw_memory = self.raw_embed(input.transpose(1, 2)) if self.raw_embed is not None else None  # (N, L, d_model)
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
            freq_memory=freq_memory,
            raw_memory=raw_memory,
        )  # (N, output_length * samples_per_output, d_model)
        if self.latent_proj is not None:
            # (N * K, L_dec, d) -> (N, L_dec, K * d): the K latents' outputs of each step side by side
            l_dec, d = features.shape[1:]
            features = features.view(batch, num_signals, l_dec, d).transpose(1, 2).reshape(batch, l_dec, -1)
        if self.class_head is not None:
            # mean over decoder steps, then the signals side by side: (N, C * d_model) -> (N, num_classes)
            return self.class_head(features.mean(dim=1).reshape(batch, -1))
        output = self.decoder.head(features)  # (N, out_channels, output_length)
        if self.revin is not None and stats is not None:
            output = self.revin.denormalize(output, stats)
        if self.channel_independent and self.latent_proj is None:
            output = output.reshape(batch, num_signals, -1)
        # (N, C, H) -> (N, output_signals, H), mixing the signals of each step
        return self.output_proj(output.transpose(1, 2)).transpose(1, 2)

    def penalty(self) -> torch.Tensor:
        return self.tokenizer.penalty()

    def edge_penalty(self) -> torch.Tensor:
        """Mean over decoder layers of SignalCrossBlock.edge_penalty; zero without edge gates."""
        blocks = [b for b in self.decoder.blocks if getattr(b, "edge_gate", None) is not None]
        if not blocks:
            return self.decoder.queries.new_zeros(())
        return torch.stack([b.edge_penalty() for b in blocks]).mean()

    def edge_weights(self) -> list[torch.Tensor]:
        """Effective C x C edge weights of every gated decoder layer (masks sampled in training mode)."""
        return [b.edge_weights() for b in self.decoder.blocks if getattr(b, "edge_gate", None) is not None]


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
    [+ raw_skip: Linear over the raw input steps each token covers, bypassing decomposition, gate and conv]

    conv_stride = s: one token per s input steps (patching), T = ceil(L / s). Token j covers the input steps
    ending at token_positions(L)[0] + j * s, so the last token always ends at the last input step. kernel_size
    must be >= conv_stride, otherwise some input steps would fall between kernels and never be seen. With long
    patches raise n_rank: each channel's patch is compressed to n_rank numbers before channels are mixed, so
    n_rank = 1 keeps only one weighted average per patch, while n_rank >= kernel_size loses nothing.
    kernel_size may be a tuple of sizes (multi-kernel, as in Inception / TimesNet): the time conv then runs
    once per size in parallel, and the channel conv mixes all of their outputs.

    cross_channel = False: the 1x1 channel conv only mixes the bands of the same signal (grouped by signal),
    keeping signals independent inside the tokenizer. Each signal then owns d_model / C dimensions of the
    token, so d_model must be divisible by C. Signals still interact later through attention.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        kernel_size: int | tuple[int, ...],
        ma_windows: tuple[int, ...] = (5, 20, 60),
        n_rank: int = 1,
        time_dim: int = 0,
        conv_stride: int = 1,
        cross_channel: bool = True,
        front_end: str = "ma",
        bank_filters: int = 8,
        raw_skip: bool = False,
    ) -> None:
        super().__init__()
        assert front_end in ("ma", "conv_bank", "none"), "front_end must be 'ma', 'conv_bank' or 'none'"
        assert conv_stride >= 1, "conv_stride must be >= 1"
        kernel_sizes = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        assert min(kernel_sizes) >= conv_stride, "kernel_size must be >= conv_stride so every input step is covered"
        if not cross_channel:
            assert d_model % in_channels == 0, "d_model must be divisible by in_channels when cross_channel=False"
        self.in_channels = in_channels
        self.conv_stride = conv_stride
        self.cross_channel = cross_channel
        # ma_windows sets the moving-average windows, or the kernel sizes of the conv bank
        self.decomp: MultiScaleDecomp | CausalConvBank | None
        if front_end == "ma":
            self.decomp = MultiScaleDecomp(ma_windows)
        elif front_end == "conv_bank":
            self.decomp = CausalConvBank(in_channels, ma_windows, bank_filters)
        else:
            self.decomp = None  # raw signal only: a gate over a single band per signal would be all-or-nothing
        num_bands = self.decomp.num_bands if self.decomp is not None else 1
        decomp_channels = num_bands * in_channels  # each signal is split into num_bands bands
        self.gate = L0Gate(decomp_channels) if self.decomp is not None else None
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
        # raw_skip: the (RevIN-normalized) input itself also reaches the Transformer. The conv_stride raw steps of
        # each token's patch, for every signal, are projected to d_model and added to the token, so no
        # information is lost to the gate, the low-rank conv or the GELU. Adding a projection is the same as
        # concatenating the raw values as extra token features and projecting the result.
        self.raw_proj = nn.Linear(in_channels * conv_stride, d_model) if raw_skip else None

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
        x = input
        if self.decomp is not None and self.gate is not None:
            x = self.gate(self.decomp(input))  # (N, B*C, L), band-major
        if not self.cross_channel:
            # Grouped channel conv needs each signal's bands to be contiguous: band-major -> signal-major
            n, _, length = x.shape
            x = x.view(n, -1, self.in_channels, length).transpose(1, 2).reshape(n, -1, length)
        x = F.gelu(self.conv(x))  # (N, d_model, T)
        x = x.transpose(1, 2)  # (N, T, d_model)
        if self.raw_proj is not None:
            # Left zero padding to a multiple of the stride aligns patches with the conv: token j covers the
            # stride steps ending at token_positions(L)[0] + j * stride
            n, c, length = input.shape
            raw = F.pad(input, ((-length) % self.conv_stride, 0))  # (N, C, T * s)
            raw = raw.view(n, c, -1, self.conv_stride).transpose(1, 2).reshape(n, x.size(1), -1)  # (N, T, C * s)
            x = x + self.raw_proj(raw)
        if time_features is not None:
            # Use the time features of the last input step each token covers
            offset, stride = self.token_positions(input.size(-1))
            x = x + self.embed_time(time_features[:, offset::stride])
        return x

    def penalty(self) -> torch.Tensor:
        if self.gate is None:
            return self.conv.channel_conv.weight.new_zeros(())
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
        freq_tokens: bool = False,
        raw_tokens: bool = False,
        cross_signals: int = 0,
        cross_parallel: bool = False,
        edge_gates: str = "",
    ) -> None:
        super().__init__()
        # One learnable query per future step (non-autoregressive: all steps are predicted at once).
        # The queries also tell the decoder which step it is predicting.
        self.queries = nn.Parameter(torch.randn(output_length * samples_per_output, d_model) * 0.02)
        # cross_signals = C > 0: Cross Transformer decoder, a complete graph over C channel-independent signals
        self.cross_signals = cross_signals
        # cross_parallel: the C cross-attentions run in parallel and are merged by one projection
        self.blocks = nn.ModuleList(
            SignalParallelCrossBlock(
                d_model,
                num_heads,
                cross_signals,
                dropout,
                raw_tokens=raw_tokens,
                edge_gate_signals=cross_signals if edge_gates else 0,
                edge_l0=edge_gates == "l0",
            )
            if cross_signals > 0 and cross_parallel
            else SignalCrossBlock(
                d_model,
                num_heads,
                dropout,
                raw_tokens=raw_tokens,
                edge_gate_signals=cross_signals if edge_gates else 0,
                edge_l0=edge_gates == "l0",
            )
            if cross_signals > 0
            else CrossTransformerBlock(
                d_model, num_heads, dropout, text_dim=text_dim, freq_tokens=freq_tokens, raw_tokens=raw_tokens
            )
            for _ in range(num_layers)
        )
        self.signal_embed = nn.Parameter(torch.randn(cross_signals, d_model) * 0.02) if cross_signals > 0 else None
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
        freq_memory: torch.Tensor | None = None,
        raw_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.features(
            memory, future_time, text_memory, text_padding_mask, query_offset, memory_offset, memory_stride,
            freq_memory, raw_memory,
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
        freq_memory: torch.Tensor | None = None,
        raw_memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Normalized decoder outputs before the head: (N, output_length * samples_per_output, d_model)."""
        # memory: (N, T, d_model) from the encoder
        # raw_memory: (N, L, d_model), one token per input step; step i sits at position i
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
        if self.signal_embed is not None:
            # rows of memory and y are (sample, signal) pairs in sample-major order: row r is signal r % C
            num_signals, (rows, length, d) = self.cross_signals, memory.shape
            y = y + self.signal_embed.repeat(rows // num_signals, 1).unsqueeze(1)
            memory = memory.view(-1, num_signals, length, d) + self.signal_embed.view(1, num_signals, 1, d)
        for block in self.blocks:
            y = block(
                y,
                memory,
                offset=query_offset,
                memory_offset=memory_offset,
                memory_stride=memory_stride,
                text_memory=text_memory,
                text_padding_mask=text_padding_mask,
                freq_memory=freq_memory,
                raw_memory=raw_memory,
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
    A tuple kernel_size runs one depthwise time conv per size in parallel (all with the same stride, so their
    outputs stay aligned); the channel conv then mixes n_rank outputs per size and channel.
    channel_groups > 1 splits the channel mixing into independent groups of consecutive input channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        n_rank: int = 1,
        stride: int = 1,
        channel_groups: int = 1,
    ) -> None:
        super().__init__()
        assert in_channels % channel_groups == 0 and out_channels % channel_groups == 0, (
            "in_channels and out_channels must be divisible by channel_groups"
        )
        kernel_sizes = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.n_rank = n_rank
        # groups=in_channels: depthwise, each channel is convolved along time independently, no channel mixing
        self.time_convs = nn.ModuleList(
            CausalConv1d(
                in_channels=in_channels,
                out_channels=in_channels * n_rank,
                kernel_size=k,
                stride=stride,
                groups=in_channels,
            )
            for k in kernel_sizes
        )
        # kernel_size=1: mixes channels at each time step only (equivalent to a per-step Linear), so it
        # is causal by construction. Transposing to (N, L, C) instead would treat time steps as channels and
        # mix in the future.
        self.channel_conv = nn.Conv1d(
            in_channels * n_rank * len(kernel_sizes), out_channels, kernel_size=1, groups=channel_groups
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # (N, C_in, L)
        n, c = input.shape[:2]
        # (N, C_in * n_rank, T) per kernel size; outputs of the same input channel stay contiguous so channel
        # groups still see whole input channels: (N, C_in, K, n_rank, T) -> (N, C_in * K * n_rank, T)
        outs = [conv(input).view(n, c, 1, self.n_rank, -1) for conv in self.time_convs]
        x = torch.cat(outs, dim=2).flatten(1, 3)
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


class FrequencyEmbedding(nn.Module):
    """One token per rFFT bin of the input window: (N, C, L) -> (N, L // 2 + 1, d_model).

    The real and imaginary parts of all C signals at a bin are projected to d_model, and a learnable embedding
    per bin tells the tokens apart (they have no time position, so attention to them uses no RoPE).
    - Causal: the FFT only covers the input window, which is entirely past data. Each bin mixes the whole
      window, so these tokens must never enter the causal encoder; the decoder reads them by cross-attention.
    - Frequency resolution is 1 / L cycles per step: with L = 96 a period of 96 steps sits in bin 1 and longer
      periods are not resolved. Use a longer input window to see slow cycles.
    - norm="ortho" keeps bin magnitudes independent of L; the FFT runs in fp32 even under autocast.
    """

    def __init__(self, in_channels: int, in_len: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * in_channels, d_model)
        self.bin_embed = nn.Parameter(torch.randn(in_len // 2 + 1, d_model) * 0.02)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # (N, C, L)
        spectrum = torch.fft.rfft(input.float(), dim=-1, norm="ortho")  # (N, C, F) complex
        x = torch.cat([spectrum.real, spectrum.imag], dim=1).transpose(1, 2)  # (N, F, 2C)
        return self.proj(x) + self.bin_embed[: x.size(1)]


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

    def open_probability(self) -> torch.Tensor:
        """Probability that each channel's gate is nonzero."""
        shift = self.beta * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.log_alpha - shift)

    def penalty(self) -> torch.Tensor:
        """Expected fraction of open channels (differentiable approximation of L0, normalized by count)."""
        return self.open_probability().mean()


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
    """Pre-norm decoder block: causal self-attention -> cross-attention -> [text cross-attention]
    -> [frequency cross-attention] -> [raw cross-attention] -> FFN.

    Each sub-layer has a residual. The text cross-attention exists only when text_dim > 0 and is skipped when
    no text_memory is passed; the frequency and raw cross-attentions likewise need freq_tokens / raw_tokens and
    their memory.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        dim_feedforward: int | None = None,
        kv_dim: int | None = None,
        text_dim: int = 0,
        freq_tokens: bool = False,
        raw_tokens: bool = False,
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
        if freq_tokens:
            # Frequency bins carry no time position, so no RoPE (like text)
            self.freq_attn_norm = nn.RMSNorm(d_model)
            self.freq_memory_norm = nn.RMSNorm(d_model)
            self.freq_attn = CrossAttention(d_model, num_heads, dropout, use_rope=False)
        else:
            self.freq_attn = None
        if raw_tokens:
            # Raw input steps do have time positions, so RoPE applies (same units as the main memory)
            self.raw_attn_norm = nn.RMSNorm(d_model)
            self.raw_memory_norm = nn.RMSNorm(d_model)
            self.raw_attn = CrossAttention(d_model, num_heads, dropout)
        else:
            self.raw_attn = None
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
        freq_memory: torch.Tensor | None = None,
        raw_memory: torch.Tensor | None = None,
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
        if self.freq_attn is not None and freq_memory is not None:
            x = x + self.dropout(self.freq_attn(self.freq_attn_norm(x), self.freq_memory_norm(freq_memory)))
        if self.raw_attn is not None and raw_memory is not None:
            x = x + self.dropout(
                self.raw_attn(self.raw_attn_norm(x), self.raw_memory_norm(raw_memory), query_offset=offset)
            )
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class SignalCrossBlock(nn.Module):
    """Pre-norm block of the Cross Transformer decoder, a complete graph over the signals: causal self-attention
    -> [raw cross-attention] -> one cross-attention per signal, in signal order -> FFN.

    The queries are the decoder tokens of every (sample, signal) pair, (N * C, L, D); memory holds the encoder
    output of every signal, (N, C, T, D). For j = 0 .. C - 1, every signal's decoder cross-attends to signal j's
    memory, each step with its own residual, so the signals are read sequentially in their dataset order.
    - One cross-attention module (and norm) is shared by all C steps; the signal embedding added to the memory
      tells the sources apart.
    - Each step's residual is scaled by 1 / sqrt(C), so the total cross-attention update keeps the same scale
      whatever the number of signals.
    - edge_gate_signals = C > 0 replaces that scale with a learnable gate g[i, j] on the update signal i receives
      from signal j. The softmax of each step always sums to 1, so without gates no signal can learn to ignore
      another one. The gates start as the identity, so the model starts as plain channel independence (each
      signal reads its own memory at full weight) and opens cross-signal edges only where they help; an L1
      penalty on the off-diagonal gates (edge_penalty) keeps the graph sparse.
    - edge_l0=True multiplies every off-diagonal gate by a Hard Concrete mask (L0Gate) and penalizes the
      expected fraction of open edges instead, so edges can close exactly and the result is a sparse graph.
      The self edges are never masked. The gates keep their sign (a signal may push another one down).
    - Cross-attention sees the whole memory, which is past data only; RoPE positions as in CrossTransformerBlock.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        raw_tokens: bool = False,
        edge_gate_signals: int = 0,
        edge_l0: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn_norm = nn.RMSNorm(d_model)
        self.self_attn = CausalSelfAttention(d_model, num_heads, dropout)
        # edge_gate[i, j]: weight of signal j's update to signal i
        self.edge_gate = nn.Parameter(torch.eye(edge_gate_signals)) if edge_gate_signals > 0 else None
        assert not edge_l0 or edge_gate_signals > 0, "edge_l0 requires edge_gate_signals > 0"
        self.edge_mask = L0Gate(edge_gate_signals**2) if edge_l0 else None
        if raw_tokens:
            self.raw_attn_norm = nn.RMSNorm(d_model)
            self.raw_memory_norm = nn.RMSNorm(d_model)
            self.raw_attn = CrossAttention(d_model, num_heads, dropout)
        else:
            self.raw_attn = None
        self.cross_attn_norm = nn.RMSNorm(d_model)
        self.memory_norm = nn.RMSNorm(d_model)
        self.cross_attn = CrossAttention(d_model, num_heads, dropout)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input: torch.Tensor,
        memory: torch.Tensor,
        offset: int = 0,
        memory_offset: int = 0,
        memory_stride: int = 1,
        raw_memory: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        # input: (N * C, L, D), memory: (N, C, T, D), raw_memory: (N * C, L_in, D)
        x = input + self.dropout(self.self_attn(self.self_attn_norm(input), offset))
        if self.raw_attn is not None and raw_memory is not None:
            x = x + self.dropout(
                self.raw_attn(self.raw_attn_norm(x), self.raw_memory_norm(raw_memory), query_offset=offset)
            )
        n, num_signals, length, d = memory.shape
        memory = self.memory_norm(memory)
        scale = num_signals**-0.5
        if self.edge_gate is not None:
            assert self.edge_gate.size(0) == num_signals, "edge gates were built for a different number of signals"
            gates = self.edge_weights()
        for j in range(num_signals):
            # signal j's memory, repeated for the decoder of every signal of the same sample
            source = memory[:, j : j + 1].expand(-1, num_signals, -1, -1).reshape(n * num_signals, length, d)
            update = self.cross_attn(
                self.cross_attn_norm(x), source, query_offset=offset, key_offset=memory_offset, key_stride=memory_stride
            )
            if self.edge_gate is not None:
                # rows are (sample, signal) pairs in sample-major order: row r receives with gate g[r % C, j]
                weight = gates[:, j].repeat(n).view(-1, 1, 1)
                x = x + weight * self.dropout(update)
            else:
                x = x + scale * self.dropout(update)
        return x + self.dropout(self.ffn(self.ffn_norm(x)))

    def edge_weights(self) -> torch.Tensor:
        """(C, C) weights of the updates, the gates times the off-diagonal masks (sampled in training mode)."""
        if self.edge_mask is None:
            return self.edge_gate
        eye = torch.eye(self.edge_gate.size(0), dtype=torch.bool, device=self.edge_gate.device)
        mask = self.edge_mask.gate_values().view_as(self.edge_gate)
        return self.edge_gate * torch.where(eye, torch.ones_like(mask), mask)

    def edge_penalty(self) -> torch.Tensor:
        """L1: mean |gate| off the diagonal. L0: expected fraction of open off-diagonal edges."""
        eye = torch.eye(self.edge_gate.size(0), dtype=torch.bool, device=self.edge_gate.device)
        if self.edge_mask is None:
            return self.edge_gate.masked_select(~eye).abs().mean()
        return self.edge_mask.open_probability().view_as(self.edge_gate).masked_select(~eye).mean()


class SignalParallelCrossBlock(SignalCrossBlock):
    """SignalCrossBlock with the C cross-attentions in parallel instead of in signal order, merged like the heads
    of multi-head attention: causal self-attention -> [raw cross-attention] -> C parallel cross-attentions ->
    concatenation -> one output projection -> FFN.

    - Every signal's decoder cross-attends to every signal's memory with the same queries (the block input after
      self-attention), so the result does not depend on the order of the signals.
    - Each source signal j is one "head group": its attention has its own softmax, and its output gets its own
      slice of the output projection (C * D -> D), as the heads of multi-head attention do.
    - The queries, keys and values are projected once (keys and values once per signal, not once per pair), and
      all C x C attentions run as one batched call, (N, C, C, H, L, D / H).
    - Edge gates and L0 masks work as in SignalCrossBlock: g[i, j] scales the output of source j for receiver i
      before the concatenation. Without gates every source enters the projection unscaled.
    - The projection has C * D * D weights per block, so this suits tens of signals, not hundreds.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_signals: int,
        dropout: float = 0.1,
        raw_tokens: bool = False,
        edge_gate_signals: int = 0,
        edge_l0: bool = False,
    ) -> None:
        super().__init__(d_model, num_heads, dropout, raw_tokens, edge_gate_signals, edge_l0)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        del self.cross_attn  # replaced by the parallel projections below
        self.num_heads = num_heads
        self.num_signals = num_signals
        self.attn_dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(num_signals * d_model, d_model)
        self.rope = RotaryEmbedding(d_model // num_heads)

    def forward(
        self,
        input: torch.Tensor,
        memory: torch.Tensor,
        offset: int = 0,
        memory_offset: int = 0,
        memory_stride: int = 1,
        raw_memory: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        # input: (N * C, L, D), memory: (N, C, T, D), raw_memory: (N * C, L_in, D)
        x = input + self.dropout(self.self_attn(self.self_attn_norm(input), offset))
        if self.raw_attn is not None and raw_memory is not None:
            x = x + self.dropout(
                self.raw_attn(self.raw_attn_norm(x), self.raw_memory_norm(raw_memory), query_offset=offset)
            )
        n, num_signals, length, d = memory.shape
        assert num_signals == self.num_signals, "the output projection was built for a different number of signals"
        h, l_q = self.num_heads, x.size(1)
        q = self.q_proj(self.cross_attn_norm(x)).view(n, num_signals, l_q, h, -1).transpose(2, 3)  # (N, C, H, L, D/H)
        k, v = self.kv_proj(self.memory_norm(memory)).chunk(2, dim=-1)
        k, v = (t.view(n, num_signals, length, h, -1).transpose(2, 3) for t in (k, v))  # (N, C, H, T, D/H)
        q, k = self.rope(q, offset), self.rope(k, memory_offset, memory_stride)
        # receiver i on dim 1, source j on dim 2
        y = F.scaled_dot_product_attention(
            q.unsqueeze(2).expand(-1, -1, num_signals, -1, -1, -1),
            k.unsqueeze(1).expand(-1, num_signals, -1, -1, -1, -1),
            v.unsqueeze(1).expand(-1, num_signals, -1, -1, -1, -1),
            dropout_p=self.attn_dropout if self.training else 0.0,
        )  # (N, C, C, H, L, D/H)
        if self.edge_gate is not None:
            assert self.edge_gate.size(0) == num_signals, "edge gates were built for a different number of signals"
            y = y * self.edge_weights().view(1, num_signals, num_signals, 1, 1, 1)
        # (N, C_i, L, C_j, H, D/H) -> (N * C, L, C * D): source-major concatenation, heads inside each source
        y = y.permute(0, 1, 4, 2, 3, 5).reshape(n * num_signals, l_q, num_signals * d)
        x = x + self.dropout(self.out_proj(y))
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
