"""Single cross-attention decoder: patch-derived Q and encoder-derived K/V."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from layers.PRNet_Config import ModelConfig


class SharedPatchRetrievalDecoder(nn.Module):
    """Use global patch embeddings to query encoder memories directly."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.drop = nn.Dropout(cfg.dropout)
        self.context_projection = nn.Linear(d, d)
        self.future_position = nn.Parameter(torch.empty(1, 1, cfg.n_future_patches, d))
        nn.init.trunc_normal_(self.future_position, std=0.02)
        self.query_norm = nn.LayerNorm(d)

        # The decoder contains only this cross-attention operation.
        self.cross_q_proj = nn.Linear(d, d)
        self.cross_k_proj = nn.Linear(d, d)
        self.cross_v_proj = nn.Linear(d, d)
        self.cross_out_proj = nn.Linear(d, d)
        self.cross_norm1 = nn.LayerNorm(d)
        self.cross_norm2 = nn.LayerNorm(d)
        self.cross_ffn = nn.Sequential(
            nn.Linear(d, cfg.d_ff), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, d), nn.Dropout(cfg.dropout),
        )

    def _split(self, x: Tensor) -> Tensor:
        b, c, length, d = x.shape
        h = self.cfg.n_heads
        return x.view(b, c, length, h, d // h).permute(0, 1, 3, 2, 4)

    def forward(self, embedding: Tensor, memory: Tensor) -> Tensor:
        d_head = self.cfg.d_model // self.cfg.n_heads
        b, c, _, _ = embedding.shape
        # Q comes from patch embeddings; K/V come directly from the encoder.
        global_context = embedding.mean(dim=-2)
        queries = self.query_norm(
            self.context_projection(global_context).unsqueeze(-2)
            + self.future_position)

        # Future Q retrieves history K/V: [B,C,heads,Nf,N].
        query = self._split(self.cross_q_proj(queries))
        key = self._split(self.cross_k_proj(memory))
        value = self._split(self.cross_v_proj(memory))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_head)
        weights = self.drop(torch.softmax(scores, dim=-1))
        retrieved = torch.matmul(weights, value)
        retrieved = retrieved.permute(0, 1, 3, 2, 4).contiguous().view(
            b, c, self.cfg.n_future_patches, self.cfg.d_model)
        retrieved = self.cross_out_proj(retrieved)
        future = self.cross_norm1(queries + self.drop(retrieved))
        future_features = self.cross_norm2(future + self.cross_ffn(future))
        return future_features
