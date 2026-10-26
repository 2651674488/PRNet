"""Patch embedding and shared Transformer encoder for PRNet."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from layers.PRNet_Config import ModelConfig


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop(a))
        return self.norm2(x + self.ffn(x))

class SharedPatchEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Linear(cfg.patch_len, cfg.d_model)
        self.history_position = nn.Parameter(torch.zeros(1, 1, cfg.n_history_patches, cfg.d_model))
        self.scale_embedding = nn.Parameter(torch.zeros(3, cfg.d_model))
        self.blocks = nn.ModuleList([
            EncoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.encoder_layers)
        ])
        nn.init.trunc_normal_(self.history_position, std=0.02)
        nn.init.trunc_normal_(self.scale_embedding, std=0.02)

    def forward(self, x: Tensor, scale_id: int) -> Tensor:
        b, s, c = x.shape
        # [B, C, S] -> [B, C, N, patch_len]
        patches = x.transpose(1, 2).unfold(-1, self.cfg.patch_len, self.cfg.patch_stride)
        e = self.patch_embed(patches)
        e = e + self.history_position + self.scale_embedding[scale_id].view(1, 1, 1, -1)
        h = e.reshape(b * c, self.cfg.n_history_patches, self.cfg.d_model)
        for block in self.blocks:
            h = block(h)
        return h.reshape(b, c, self.cfg.n_history_patches, self.cfg.d_model)
