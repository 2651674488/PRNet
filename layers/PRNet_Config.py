"""PRNet script parameters and derived patch counts."""
from __future__ import annotations

import math


class ModelConfig:
    def __init__(self, configs):
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.patch_len = configs.patch_len
        self.patch_stride = configs.patch_stride
        self.future_patch_len = configs.future_patch_len
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.encoder_layers = configs.e_layers
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        self.ma_short = configs.ma_short
        self.ma_long = configs.ma_long
        self.n_scales = 3
        self.revin_affine = configs.revin_affine

    @property
    def n_history_patches(self) -> int:
        return (self.seq_len - self.patch_len) // self.patch_stride + 1

    @property
    def n_future_patches(self) -> int:
        return math.ceil(self.pred_len / self.future_patch_len)
