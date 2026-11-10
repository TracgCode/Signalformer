"""PolyConv: a learnable convolution in front of minimum-variance polynomial forecasts (TWStock, fmt run 8).

Every input feature f of a stock goes through M learnable filters of length K (a depthwise convolution shared by all
stocks): z_{f,m} = w_{f,m} * x_f. On each filtered series a polynomial of degree p is fitted by least squares over
the last W steps (the Gauss-Markov estimate under white noise) and read out as three numbers: its value one step
ahead ("next"), next minus its value at the last step ("slope", for p > 0) and next minus the last filtered value
("reversion"). Each readout is a fixed linear filter over the window, so all of them together are one fixed matrix G
(L', Q) applied to the filtered series. A linear head shared by the stocks maps the readouts of every (f, m) to the
forecast of the next return, with a ridge penalty l2 * |head weights|^2 added to the training loss.

The whole model is linear in the input; the convolution can learn what to fit polynomials to, for example
differences of the price levels. The filters are weight-normalized (w / |w|) and every column of G has unit norm, so
on z-scored white-noise input every readout has unit variance whatever the filter, and the learned filters are
shapes that can be compared. Filter 0 of every feature starts as the identity (a delta at the last tap), the others
as the identity plus small noise, and the head starts at zero, so the first forecast is 0 (or the --residual-base
forecast). With sum_penalty > 0 the filters are not weight-normalized; instead
sum_penalty * sum_{f,m} (sum_j w_{f,m,j} - 1)^2 is added to the training loss, which keeps the gain of every filter
at zero frequency (the level it passes) near 1 and sets its scale. readouts selects the readout kinds: "slope" and "revert" sum to zero over the window, so they ignore the
level of the filtered series (which drifts out of the training range for the price features); "next" keeps it.

deconv ("target" or "all") forecasts in the filtered domain and deconvolves back. The last tap of every filter is
fixed at 1 (a monic filter, always invertible without a division), so the next filtered value
z_next = sum_{j<K-1} w_j x_{t-K+2+j} + x_{t+1} gives x_hat_{t+1} = z_next_hat - sum_{j<K-1} w_j x_{t-K+2+j}, one
for every "next" readout (not normalized, so the forecast is in the units of x). As in ARIMA, w = [-1, 1] forecasts
the next change and adds it back to the last value. The head reads the deconvolved forecasts of the target r only
("target") or of every feature ("all"); the free taps start at 0 (the identity) plus small noise for filters m > 0.
deconv_filter chooses the filter form: "monic" as above; "norm" weight-normalizes the whole filter (w / |w|, as
without deconv) and divides by its last tap, x_hat = (z_next_hat - sum_{j<K-1} w_j x_{t-K+2+j}) / w_{K-1} (|w_{K-1}|
kept at least 1e-3); "monic_norm" fixes the last tap at 1 and normalizes the other K - 1 taps to unit norm, which
start random (this filter can never return to the identity).
"""

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


def poly_readouts(length: int, windows: tuple[int, ...], degrees: tuple[int, ...],
                  kinds: tuple[str, ...] = ("next", "slope", "revert")) -> tuple[np.ndarray, list[str]]:
    """G (length, Q): column q is the filter over a series of `length` steps (oldest first) giving one readout."""
    cols, names = [], []
    normalize = kinds != ("raw_next",)
    if not normalize:
        kinds = ("next",)
    for w in windows:
        if w > length:
            continue
        x = (np.arange(w, dtype=np.float64) - (w - 1)) / w  # the last step at 0
        for p in degrees:
            if p > w - 2:  # an exact fit has no residual: skip
                continue
            pinv = np.linalg.pinv(np.vander(x, p + 1, increasing=True))  # (p + 1, w)
            at = lambda t: (t ** np.arange(p + 1)) @ pinv
            nxt, last = np.zeros(length), np.zeros(length)
            nxt[-w:], last[-w:] = at(1 / w), at(0.0)
            delta = np.zeros(length)
            delta[-1] = 1.0
            for kind, col in (("next", nxt), ("slope", nxt - last if p > 0 else None), ("revert", nxt - delta)):
                if kind in kinds and col is not None:
                    cols.append(col / np.linalg.norm(col) if normalize else col); names.append(f"W{w}p{p}:{kind}")
    return np.stack(cols, axis=1), names


class PolyConv(nn.Module):
    uses_phase = False

    def __init__(self, num_signals: int, signal_features: int, in_len: int, out_len: int, kernel_size: int = 8,
                 filters: int = 2, windows: tuple[int, ...] = (5, 10, 20, 40), degrees: tuple[int, ...] = (0, 1, 2),
                 fixed: bool = False, l2: float = 1.0, readouts: tuple[str, ...] = ("next", "slope", "revert"),
                 sum_penalty: float = 0.0, deconv: str = "", deconv_filter: str = "monic") -> None:
        super().__init__()
        self.tokenizer = SimpleNamespace(gate=None)  # the hooks train.py expects
        self.num_signals, self.signal_features, self.filters, self.l2 = num_signals, signal_features, filters, l2
        self.sum_penalty, self.deconv, self.deconv_filter = sum_penalty, deconv, deconv_filter
        self.conv_features = f = 1 if deconv == "target" else signal_features  # the target r is feature 0
        self.conv = nn.Conv1d(f, f * filters, kernel_size, groups=f, bias=False)
        with torch.no_grad():
            w = torch.zeros(f, filters, kernel_size)
            w[:, :, -1] = 1.0
            if filters > 1:
                w[:, 1:] += 0.1 * torch.randn(f, filters - 1, kernel_size) / kernel_size ** 0.5
            if deconv and deconv_filter == "monic_norm":
                w[:, :, :-1] = torch.randn(f, filters, kernel_size - 1)
            self.conv.weight.copy_(w.view(f * filters, 1, kernel_size))
        self.conv.weight.requires_grad_(not fixed)
        if deconv:
            readouts = ("raw_next",)
            mask = torch.ones(1, 1, kernel_size)
            mask[..., -1] = 0.0
            self.register_buffer("free_taps", mask)  # the last tap is fixed at 1
        g, self.readout_names = poly_readouts(in_len - kernel_size + 1, windows, degrees, readouts)
        self.register_buffer("readouts", torch.from_numpy(g).float())  # (L', Q)
        q = g.shape[1]
        self.head = nn.Linear(f * filters * q, out_len)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def penalty(self) -> torch.Tensor:
        return torch.zeros((), device=self.head.weight.device)

    def ridge_penalty(self) -> torch.Tensor:
        penalty = self.l2 * self.head.weight.pow(2).sum()
        if self.sum_penalty > 0:
            penalty = penalty + self.sum_penalty * (self.filters_normalized().sum(dim=-1) - 1).pow(2).sum()
        return penalty

    def filters_normalized(self) -> torch.Tensor:
        w = self.conv.weight  # (F M, 1, K)
        if self.deconv and self.deconv_filter == "monic":
            return w * self.free_taps + (1 - self.free_taps)
        if self.deconv and self.deconv_filter == "monic_norm":
            free = w[..., :-1]
            free = free / free.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return torch.cat([free, torch.ones_like(w[..., -1:])], dim=-1)
        if self.sum_penalty > 0:
            return w  # the sum penalty sets the scale
        return w / w.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def forward(self, x: torch.Tensor, time_features=None, future_time_features=None, phase=None) -> torch.Tensor:
        # x: (N, S * F, L), feature-major within each stock -> (N, S, H)
        n, s = x.size(0), self.num_signals
        x = x.reshape(n * s, self.signal_features, -1)[:, : self.conv_features]
        w = self.filters_normalized()
        z = nn.functional.conv1d(x, w, groups=self.conv_features)  # (N S, F M, L')
        feats = z @ self.readouts  # (N S, F M, Q)
        if self.deconv:
            k = w.size(-1)
            # the known part of the next filtered value: the free taps on the last K - 1 inputs
            known = nn.functional.conv1d(x[..., x.size(-1) - k + 1 :], w[..., :-1], groups=self.conv_features)
            feats = feats - known  # (N S, F M, Q): forecasts of the next x, one per readout
            if self.deconv_filter == "norm":
                last = w[..., -1:]  # (F M, 1, 1)
                last = torch.where(last < 0, -1.0, 1.0) * last.abs().clamp(min=1e-3)
                feats = feats / last.view(1, -1, 1)
        feats = feats.flatten(1)
        return self.head(feats).view(n, s, -1)

    @torch.no_grad()
    def report(self) -> dict:
        """The learned filters (F, M, K) and the head weights (F, M, Q) per readout, for the result file."""
        f, m = self.conv_features, self.filters
        return dict(kernels=self.filters_normalized().view(f, m, -1).tolist(),
                    head=self.head.weight.view(-1, f, m, len(self.readout_names)).tolist(),
                    readouts=self.readout_names)


class PolyMLP(nn.Module):
    """A two-layer MLP forecasting the residual of a fixed polynomial forecast (--residual-base, fmt run 11).

    The inputs hold the base forecast as the last feature (base_feature), so the past residuals e = r - base are
    known up to the day before the forecast day. For every stock the MLP reads the last `window` days of e together
    with the same days of the other features (or their day-to-day changes with diff): D = window * (F - 1) values,
    Linear(D -> D / shrink), GELU, dropout, Linear(D / shrink -> H) (shrink 2 by default). The second layer starts at zero, so the first forecast is
    the base forecast, which train.py adds to the output. Shared by the stocks."""

    uses_phase = False

    def __init__(self, num_signals: int, signal_features: int, out_len: int, window: int = 5, dropout: float = 0.2,
                 diff: bool = False, shrink: int = 2) -> None:
        super().__init__()
        self.tokenizer = SimpleNamespace(gate=None)
        self.num_signals, self.signal_features, self.window, self.diff = num_signals, signal_features, window, diff
        d = window * (signal_features - 1)  # residual + the features other than r and the base
        h = d // shrink
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, out_len))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def penalty(self) -> torch.Tensor:
        return torch.zeros((), device=self.mlp[0].weight.device)

    def forward(self, x: torch.Tensor, time_features=None, future_time_features=None, phase=None) -> torch.Tensor:
        n, s, f = x.size(0), self.num_signals, self.signal_features
        x = x.reshape(n * s, f, -1)
        w = self.window
        resid = (x[:, 0] - x[:, -1])[:, None, -w:]  # (N S, 1, w): r - base
        other = x[:, 1:-1]
        other = (other[..., -w:] - other[..., -w - 1 : -1]) if self.diff else other[..., -w:]
        return self.mlp(torch.cat([resid, other], dim=1).flatten(1)).view(n, s, -1)
