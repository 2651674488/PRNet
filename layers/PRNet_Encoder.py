"""Shared Transformer encoder for embedded PRNet patches."""
from __future__ import annotations

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
        self.blocks = nn.ModuleList([
            EncoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.encoder_layers)
        ])

    def forward(self, embeddings: Tensor) -> Tensor:
        b, c, n, d = embeddings.shape
        h = embeddings.reshape(b * c, n, d)
        for block in self.blocks:
            h = block(h)
        return h.reshape(b, c, n, d)
