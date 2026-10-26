"""Reversible normalization for PRNet input and forecasts."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


class ReversibleInstanceNorm(nn.Module):
    """Per-window, per-variable instance norm with optional trainable affine scale."""
    def __init__(self, n_channels: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.affine = affine
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(n_channels))
            self.bias = nn.Parameter(torch.zeros(n_channels))

    def normalize(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        mean = x.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(x.var(dim=1, unbiased=False, keepdim=True) + self.eps).detach()
        out = (x - mean) / std
        if self.affine:
            out = out * self.weight.view(1, 1, -1) + self.bias.view(1, 1, -1)
        return out, (mean, std)

    def denormalize(self, x: Tensor, stats: Tuple[Tensor, Tensor]) -> Tensor:
        mean, std = stats
        if self.affine:
            # Preserve sign while avoiding division by a near-zero learned scale.
            w = self.weight.view(1, 1, -1)
            safe_w = torch.where(w.abs() < self.eps,
                                 torch.where(w < 0, -torch.ones_like(w), torch.ones_like(w)) * self.eps,
                                 w)
            x = (x - self.bias.view(1, 1, -1)) / safe_w
        return x * std + mean
