"""PRNet hyperparameters and shape validation."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ModelConfig:
    seq_len: int = 96
    pred_len: int = 192
    enc_in: int = 7
    patch_len: int = 16
    patch_stride: int = 8
    future_patch_len: int = 16
    d_model: int = 64
    n_heads: int = 4
    encoder_layers: int = 1
    d_ff: int = 128
    dropout: float = 0.1
    ma_short: int = 5
    ma_long: int = 25
    top_k: int = 4
    num_queries: int = 5
    router_hidden: int = 64
    min_period_overlap: int = 8
    revin_affine: bool = True
    initial_period_bias: float = 1.0

    def validate(self) -> None:
        if self.seq_len < 4 or self.pred_len <= 0 or self.enc_in <= 0:
            raise ValueError("seq_len>=4, pred_len>0 and enc_in>0 are required")
        if self.patch_len > self.seq_len or min(self.patch_len, self.patch_stride, self.future_patch_len) <= 0:
            raise ValueError("Invalid patch length/stride")
        if self.d_model <= 0 or self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.top_k < 1 or self.top_k > self.seq_len // 2:
            raise ValueError("top_k must be in [1, seq_len//2] (excluding DC bin)")
        if min(self.num_queries, self.router_hidden, self.encoder_layers, self.d_ff) < 1:
            raise ValueError("Query count, router hidden dim, encoder layers and d_ff must be positive")
        if min(self.ma_short, self.ma_long) < 1 or self.ma_short >= self.ma_long:
            raise ValueError("Moving averages require 1 <= ma_short < ma_long")
        if self.min_period_overlap < 2:
            raise ValueError("min_period_overlap must be >=2")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")

    @property
    def n_history_patches(self) -> int:
        return (self.seq_len - self.patch_len) // self.patch_stride + 1

    @property
    def n_future_patches(self) -> int:
        return math.ceil(self.pred_len / self.future_patch_len)
