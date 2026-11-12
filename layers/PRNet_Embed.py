"""Shared history patch embedding for PRNet."""
import torch
from torch import Tensor, nn


class SharedPatchEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_len = cfg.patch_len
        self.patch_stride = cfg.patch_stride
        self.patch_embed = nn.Linear(cfg.patch_len, cfg.d_model)
        self.history_position = nn.Parameter(torch.empty(1, 1, cfg.n_history_patches, cfg.d_model))
        self.scale_embedding = nn.Parameter(torch.empty(cfg.n_scales, cfg.d_model))
        nn.init.trunc_normal_(self.history_position, std=0.02)
        nn.init.trunc_normal_(self.scale_embedding, std=0.02)

    def forward(self, x: Tensor, scale_id: int) -> Tensor:
        patches = x.transpose(1, 2).unfold(-1, self.patch_len, self.patch_stride)
        return (self.patch_embed(patches) + self.history_position
                + self.scale_embedding[scale_id].view(1, 1, 1, -1))
