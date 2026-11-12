"""Dynamic fusion of the decoded PRNet components."""
from __future__ import annotations


import torch
from torch import Tensor, nn

from layers.PRNet_Config import ModelConfig


class DynamicScaleFusion(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, decoded: list[Tensor]) -> Tensor:
        # M * [B,C,Nf,d] -> [B,C,Nf,M,d]
        stack = torch.stack(decoded, dim=-2)
        logits = self.gate(stack).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        combined = (stack * weights.unsqueeze(-1)).sum(dim=-2)
        return combined
