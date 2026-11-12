"""PRNet: multi-scale forecasting with global-context patch retrieval."""
import torch.nn as nn

from layers.PRNet_Config import ModelConfig
from layers.PRNet_Decomp import MultiScaleDecomposition
from layers.PRNet_Decoder import SharedPatchRetrievalDecoder
from layers.PRNet_Encoder import SharedPatchEncoder
from layers.PRNet_Embed import SharedPatchEmbedding
from layers.PRNet_Fusion import DynamicScaleFusion
from layers.PRNet_RevIN import ReversibleInstanceNorm


class Model(nn.Module):
    """RevIN -> decomposition -> patch encoder -> patch retrieval decoder -> fusion.

    Multivariate input and output shapes are [B, seq_len, C] and [B, pred_len, C].
    Calendar marks and decoder inputs are accepted to match the experiment API;
    future queries retrieve historical patches using the global window context.
    Each scale has its own encoder and decoder with independent parameters.
    """

    def __init__(self, configs):
        super().__init__()
        cfg = ModelConfig(configs)
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len

        self.revin = ReversibleInstanceNorm(cfg.enc_in, cfg.revin_affine)
        self.decomp = MultiScaleDecomposition(cfg.ma_short, cfg.ma_long)
        self.patch_embedding = SharedPatchEmbedding(cfg)
        self.encoders = nn.ModuleList([SharedPatchEncoder(cfg) for _ in range(cfg.n_scales)])
        self.decoders = nn.ModuleList([SharedPatchRetrievalDecoder(cfg) for _ in range(cfg.n_scales)])
        self.scale_fusion = DynamicScaleFusion(cfg)
        self.prediction_head = nn.Linear(cfg.d_model, cfg.future_patch_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_norm, stats = self.revin.normalize(x_enc)
        scales = self.decomp(x_norm)
        patch_embeddings = [self.patch_embedding(s, m) for m, s in enumerate(scales)]
        memories = [
            encoder(embedding)
            for encoder, embedding in zip(self.encoders, patch_embeddings)
        ]

        decoded = [
            decoder(embedding, memory)
            for decoder, embedding, memory in zip(self.decoders, patch_embeddings, memories)
        ]
        fused = self.scale_fusion(decoded)
        patch_pred = self.prediction_head(fused)         # [B,C,Nf,FuturePatchLen]
        b, c, _, _ = patch_pred.shape
        y_norm = patch_pred.reshape(b, c, -1)[..., :self.cfg.pred_len].transpose(1, 2)
        y_hat = self.revin.denormalize(y_norm, stats)

        return y_hat
