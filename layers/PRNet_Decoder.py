"""Shared period-guided cross-attention decoder for PRNet."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from layers.PRNet_Config import ModelConfig


class PeriodGuidedDecoder(nn.Module):
    """One shared decoder applied to 3 separate memories, each with its own bias."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.drop = nn.Dropout(cfg.dropout)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, cfg.d_ff), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, d), nn.Dropout(cfg.dropout),
        )
        self.bias_strength = nn.Parameter(torch.full((3,), cfg.initial_period_bias))

    def _split(self, x: Tensor) -> Tensor:
        b, c, length, d = x.shape
        h = self.cfg.n_heads
        return x.view(b, c, length, h, d // h).permute(0, 1, 3, 2, 4)

    def forward(self, queries: Tensor, memories: list[Tensor], biases: list[Tensor],
                return_attention: bool = False) -> Tuple[list[Tensor], Optional[list[Tensor]]]:
        q = self._split(self.q_proj(queries))
        d_head = self.cfg.d_model // self.cfg.n_heads
        results = []
        attention_maps = [] if return_attention else None
        for m, (h, bias) in enumerate(zip(memories, biases)):
            k = self._split(self.k_proj(h))
            v = self._split(self.v_proj(h))
            logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)
            # [B,C,1,Nf,N] broadcast to all heads.
            logits = logits + self.bias_strength[m] * bias[:, :, None, :, :].to(logits.dtype)
            attention = torch.softmax(logits, dim=-1)
            if return_attention:
                attention_maps.append(attention.detach())
            attention = self.drop(attention)
            out = torch.matmul(attention, v)             # [B,C,h,Nf,d_head]
            b, c, _, n_f, _ = out.shape
            out = out.permute(0, 1, 3, 2, 4).contiguous().view(b, c, n_f, self.cfg.d_model)
            out = self.out_proj(out)
            z = self.norm1(queries + self.drop(out))
            decoded = self.norm2(z + self.ffn(z))
            results.append(decoded)
        return results, attention_maps
