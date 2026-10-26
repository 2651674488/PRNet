"""Three-scale moving-average decomposition for PRNet."""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


def moving_average_same(x: Tensor, kernel_size: int) -> Tensor:
    """Replicate-padded moving average along time, preserving [B, S, C]."""
    left = (kernel_size - 1) // 2
    right = kernel_size - 1 - left
    y = x.transpose(1, 2)           # [B, C, S]
    y = F.pad(y, (left, right), mode="replicate")
    return F.avg_pool1d(y, kernel_size=kernel_size, stride=1).transpose(1, 2)


class MultiScaleDecomposition(nn.Module):
    """Split normalized history into residual, intermediate and trend scales."""

    def __init__(self, short_window, long_window):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window

    def forward(self, x):
        short_trend = moving_average_same(x, self.short_window)
        long_trend = moving_average_same(x, self.long_window)
        return [x - short_trend, short_trend - long_trend, long_trend]
