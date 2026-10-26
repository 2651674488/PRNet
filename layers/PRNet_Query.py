"""FFT candidate extraction and soft routing over a learned query library."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from layers.PRNet_Config import ModelConfig


class PeriodQueryLibrary(nn.Module):
    """FFT top-k reconstruction -> input-dependent soft routing -> learned query templates.

    Query templates are shared across variables and are trained by forecasting loss.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.library = nn.Parameter(torch.randn(cfg.num_queries, cfg.n_future_patches, cfg.d_model) * 0.02)
        self.future_position = nn.Parameter(torch.randn(cfg.n_future_patches, cfg.d_model) * 0.02)
        self.router = nn.Sequential(
            nn.Linear(cfg.seq_len, cfg.router_hidden),
            nn.GELU(),
            nn.Linear(cfg.router_hidden, cfg.num_queries),
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    @staticmethod
    def fft_candidates(x: Tensor, top_k: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute FFT candidates for [B,S,C] float input.

        Returns:
            selected frequency indices: [B,C,K], integer
            candidate periods: [B,C,K], float32
            top-k-only time reconstruction: [B,C,S], float32
        """
        _, s, _ = x.shape
        spectrum = torch.fft.rfft(x.transpose(1, 2).float(), dim=-1)
        power = spectrum.abs().square()
        power = power.clone()
        power[..., 0] = float("-inf")  # DC cannot be a period
        selected = power.topk(top_k, dim=-1).indices
        periods = s / selected.float()
        top_values = spectrum.gather(dim=-1, index=selected)
        sparse = torch.zeros_like(spectrum).scatter(dim=-1, index=selected, src=top_values)
        periodic_signal = torch.fft.irfft(sparse, n=s, dim=-1)
        return selected, periods, periodic_signal

    def forward(self, periodic_signal: Tensor) -> Tuple[Tensor, Tensor]:
        b, c, s = periodic_signal.shape
        if s != self.cfg.seq_len:
            raise ValueError("Periodic signal length differs from model seq_len")
        # FFT features are FP32; match explicitly converted model weights as well.
        logits = self.router(periodic_signal.to(self.router[0].weight.dtype))  # [B, C, n]
        weights = torch.softmax(logits, dim=-1)
        mixed_queries = torch.einsum("bcn,nfd->bcfd", weights, self.library)
        queries = self.norm(mixed_queries + self.future_position[None, None, :, :])
        return queries, weights                          # [B,C,Nf,d], [B,C,n]
