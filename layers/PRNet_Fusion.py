"""Dynamic fusion of the three decoded PRNet scales."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from layers.PRNet_Config import ModelConfig


class DynamicScaleFusion(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, decoded: list[Tensor], queries: Tensor) -> Tuple[Tensor, Tensor]:
        # 3 * [B,C,Nf,d] -> [B,C,Nf,3,d]
        stack = torch.stack(decoded, dim=-2)
        q = queries.unsqueeze(-2).expand_as(stack)
        logits = self.gate(torch.cat([stack, q], dim=-1)).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        combined = (stack * weights.unsqueeze(-1)).sum(dim=-2)
        return combined, weights
