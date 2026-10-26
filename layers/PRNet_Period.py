"""Scale-specific period reliability and cross-attention bias for PRNet."""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn

from layers.PRNet_Config import ModelConfig


class PeriodReliability(nn.Module):
    """Fixed-form energy * positive autocorrelation * valid-overlap confidence.

    Each scale uses the same candidate frequency bins from the *original* normalized
    input but estimates energy/autocorrelation from the corresponding scale.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        h_idx = torch.arange(cfg.n_history_patches).float()
        f_idx = torch.arange(cfg.n_future_patches).float()
        hist_centers = h_idx * cfg.patch_stride + (cfg.patch_len - 1) / 2
        future_centers = cfg.seq_len + f_idx * cfg.future_patch_len + (cfg.future_patch_len - 1) / 2
        self.register_buffer("delta_time", future_centers[:, None] - hist_centers[None, :], persistent=False)

    def _lag_correlation(self, x: Tensor, lags: Tensor) -> Tuple[Tensor, Tensor]:
        """x [B,C,S]; lags [B,C,K]. Masked, batched Pearson correlation."""
        b, c, s = x.shape
        k = lags.size(-1)
        base = torch.arange(s, device=x.device).view(1, 1, 1, s)
        max_valid = (s - lags).clamp(min=0)
        valid = base < max_valid[..., None]
        count = valid.sum(dim=-1).float()                 # [B,C,K]
        safe_count = count.clamp(min=1.0)
        lagged_indices = (base + lags[..., None]).clamp(0, s - 1).expand(b, c, k, s)
        x0 = x[:, :, None, :].expand(b, c, k, s)
        x1 = x.gather(dim=-1, index=lagged_indices.reshape(b, c, k * s)).reshape(b, c, k, s)
        mask = valid.to(x.dtype)
        mu0 = (x0 * mask).sum(-1) / safe_count
        mu1 = (x1 * mask).sum(-1) / safe_count
        a = (x0 - mu0[..., None]) * mask
        y = (x1 - mu1[..., None]) * mask
        dot = (a * y).sum(-1)
        # Clamp BEFORE sqrt: sqrt(0) has an infinite derivative, so clamping
        # its result still produces NaN gradients for constant/empty overlaps.
        norm = torch.sqrt((a.square().sum(-1) * y.square().sum(-1)).clamp(min=1e-16))
        corr = (dot / norm).clamp(-1, 1)
        has_enough = count >= self.cfg.min_period_overlap
        return torch.where(has_enough, corr, torch.zeros_like(corr)), count / float(s)

    def forward(self, scale_x: Tensor, freq_idx: Tensor, periods: Tensor) -> Tuple[Tensor, Tensor]:
        # FFT-based calculations intentionally use FP32 for accuracy and GPU compatibility.
        xx = scale_x.transpose(1, 2).float()             # [B,C,S]
        power = torch.fft.rfft(xx, dim=-1).abs().square()
        denom = power[..., 1:].sum(-1, keepdim=True).clamp(min=1e-8)
        energy = power.gather(-1, freq_idx) / denom       # [B,C,K]
        lags = periods.round().long()
        corr, overlap = self._lag_correlation(xx, lags)
        valid = (self.cfg.seq_len - lags) >= self.cfg.min_period_overlap
        reliability = energy * corr.clamp(min=0) * overlap * valid.float()
        angle = (2 * math.pi) * self.delta_time[None, None, None, :, :] / periods[..., None, None]
        bias = (reliability[..., None, None] * torch.cos(angle)).sum(dim=2)
        return bias, reliability                         # [B,C,Nf,N], [B,C,K]
