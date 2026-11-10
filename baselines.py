"""Reimplementations of published forecasters, as reference points trained in our own pipeline.

CycleNet (Lin et al., NeurIPS 2024) follows the official code (github.com/ACAT-SCUT/CycleNet, models/CycleNet.py):
residual cycle forecasting with a learnable W x C cycle and a Linear or two-layer MLP backbone shared by the signals.

PAMNet (2026, arXiv 2605.02938) has no public code; it follows the paper's equations, Algorithm 1 and Figure 2.
Choices the paper leaves open are marked "assumption" below. Both models take and return (N, C, L) tensors,
like Signalformer, and get the absolute time step of every window's first input step as `phase`.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn


def _instance_norm(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # x: (N, C, L); per-window, per-signal mean and std without affine parameters, as both papers
    mean = x.mean(dim=-1, keepdim=True)
    std = (x.var(dim=-1, keepdim=True) + 1e-5).sqrt()  # unbiased variance, as torch.var in the CycleNet code
    return (x - mean) / std, mean, std


class _Baseline(nn.Module):
    """The hooks train.py expects from a Signalformer: no tokenizer gate, no penalty, the window phase."""

    uses_phase = True

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = SimpleNamespace(gate=None)

    def penalty(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)


class CycleNet(_Baseline):
    """CycleNet: normalize, subtract the learnable cycle, forecast the residual, add the cycle back, denormalize.

    hidden = 0 is CycleNet/Linear (Linear(L -> H)); hidden > 0 is CycleNet/MLP (L -> hidden -> H with ReLU).
    The official code indexes the cycle by the phase of the first forecast step; indexing by the first input
    step, as here, only shifts which row of the learned cycle holds which phase.
    """

    def __init__(self, num_signals: int, in_len: int, out_len: int, cycle_len: int, hidden: int = 0) -> None:
        super().__init__()
        self.in_len, self.out_len = in_len, out_len
        self.cycle = nn.Parameter(torch.zeros(cycle_len, num_signals))
        self.backbone = (
            nn.Linear(in_len, out_len)
            if hidden == 0
            else nn.Sequential(nn.Linear(in_len, hidden), nn.ReLU(), nn.Linear(hidden, out_len))
        )

    def forward(self, x: torch.Tensor, time_features=None, future_time_features=None, phase=None) -> torch.Tensor:
        # x: (N, C, L), phase: (N,) absolute time step of x[..., 0]
        x, mean, std = _instance_norm(x)
        steps = phase[:, None] + torch.arange(self.in_len + self.out_len, device=x.device)
        cycle = self.cycle[steps % self.cycle.size(0)].transpose(1, 2)  # (N, C, L + H)
        y = self.backbone(x - cycle[..., : self.in_len]) + cycle[..., self.in_len :]
        return y * std + mean


class Modulator(nn.Module):
    """M(E_X, S) = W3(Dropout(SiLU(W1 E_X) * W2 S)) (PAMNet eq. and Figure 2: dropout after the product)."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.w1, self.w2, self.w3 = (nn.Linear(d_model, d_model) for _ in range(3))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, carrier: torch.Tensor) -> torch.Tensor:
        return self.w3(self.dropout(nn.functional.silu(self.w1(x)) * self.w2(carrier)))


class PAMNet(_Baseline):
    """PAMNet: inverted embedding of each signal's window, modulated by a phase and an amplitude carrier.

    E_X = X^T W + b (N x C x d); t = time step of the last input step mod c; E_P = Omega_p[t] (c x d, shared by the
    signals, broadcast to every signal); E_A = Omega_a[t] reshaped to C x d (one carrier per signal). Two separate
    modulators (Figure 2) give M_P and M_A; M_X = M_P + M_A goes through an MLP to the H forecast steps.
    Assumption: the output MLP is Linear(d -> d) -> SiLU -> Dropout -> Linear(d -> H) (the paper only says "MLP").
    signal_features = F > 1 (TWStock) gives every signal F features, feature-major in (N, C * F, L): each signal's
    inverted embedding reads its F normalized feature windows together (Linear(F L -> d)) and the forecast is of
    feature 0, denormalized with its statistics.
    """

    def __init__(self, num_signals: int, in_len: int, out_len: int, cycle_len: int, d_model: int,
                 dropout: float, signal_features: int = 1) -> None:
        super().__init__()
        self.in_len, self.cycle_len, self.d_model, self.signal_features = in_len, cycle_len, d_model, signal_features
        self.embed = nn.Linear(signal_features * in_len, d_model)
        self.phase_embed = nn.Parameter(torch.empty(cycle_len, d_model))
        self.amplitude_embed = nn.Parameter(torch.empty(cycle_len, num_signals * d_model))
        nn.init.xavier_normal_(self.phase_embed)  # Algorithm 1: Xavier normal
        nn.init.xavier_normal_(self.amplitude_embed)
        self.phase_mod = Modulator(d_model, dropout)
        self.amplitude_mod = Modulator(d_model, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_model, out_len)
        )

    def forward(self, x: torch.Tensor, time_features=None, future_time_features=None, phase=None) -> torch.Tensor:
        # x: (N, C, L), phase: (N,) absolute time step of x[..., 0]
        x, mean, std = _instance_norm(x)
        f = self.signal_features
        n, c = x.size(0), x.size(1) // f
        x = x.reshape(n, c, -1)  # (N, C, F * L): the F feature windows of each signal side by side
        mean, std = mean[:, ::f], std[:, ::f]  # the forecast is of feature 0
        t = (phase + self.in_len - 1) % self.cycle_len  # tau_end mod c
        e_x = self.embed(x)  # (N, C, d)
        e_p = self.phase_embed[t].unsqueeze(1)  # (N, 1, d), the same carrier for every signal
        e_a = self.amplitude_embed[t].view(n, c, self.d_model)  # (N, C, d)
        m_x = self.phase_mod(e_x, e_p) + self.amplitude_mod(e_x, e_a)
        return self.head(m_x) * std + mean
