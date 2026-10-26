"""PRNet: period-aware query forecasting with three shared encoder/decoder branches."""
from dataclasses import fields

import torch
import torch.nn as nn

from layers.PRNet_Config import ModelConfig
from layers.PRNet_Decomp import MultiScaleDecomposition
from layers.PRNet_Decoder import PeriodGuidedDecoder
from layers.PRNet_Encoder import SharedPatchEncoder
from layers.PRNet_Fusion import DynamicScaleFusion
from layers.PRNet_Period import PeriodReliability
from layers.PRNet_Query import PeriodQueryLibrary
from layers.PRNet_RevIN import ReversibleInstanceNorm


class Model(nn.Module):
    """RevIN -> decomposition -> patch encoder -> period queries -> decoder -> fusion.

    Input and output shapes are [B, seq_len, enc_in] and [B, pred_len, c_out].
    Calendar marks and decoder inputs are accepted to match the experiment API;
    future queries are generated from the input history alone.
    """

    def __init__(self, configs):
        super().__init__()
        if isinstance(configs, ModelConfig):
            cfg = configs
        else:
            values = {field.name: getattr(configs, field.name)
                      for field in fields(ModelConfig) if hasattr(configs, field.name)}
            if "encoder_layers" not in values and hasattr(configs, "e_layers"):
                values["encoder_layers"] = configs.e_layers
            cfg = ModelConfig(**values)
        cfg.validate()
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        self.output_attention = getattr(configs, "output_attention", False)
        self.c_out = getattr(configs, "c_out", cfg.enc_in)
        self.features = getattr(configs, "features", "M")
        if self.c_out != cfg.enc_in and not (self.features == "MS" and self.c_out == 1):
            raise ValueError("PRNet requires c_out == enc_in, or c_out == 1 for features='MS'")

        self.revin = ReversibleInstanceNorm(cfg.enc_in, cfg.revin_affine)
        self.decomp = MultiScaleDecomposition(cfg.ma_short, cfg.ma_long)
        self.encoder = SharedPatchEncoder(cfg)
        self.query_generator = PeriodQueryLibrary(cfg)
        self.period_reliability = PeriodReliability(cfg)
        self.decoder = PeriodGuidedDecoder(cfg)
        self.scale_fusion = DynamicScaleFusion(cfg)
        self.prediction_head = nn.Linear(cfg.d_model, cfg.future_patch_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                return_aux=False):
        if any(mask is not None for mask in (enc_self_mask, dec_self_mask, dec_enc_mask)):
            raise ValueError("PRNet does not support external attention masks")
        return_aux = return_aux or self.output_attention
        if x_enc.ndim != 3 or x_enc.shape[1] != self.cfg.seq_len or x_enc.shape[2] != self.cfg.enc_in:
            raise ValueError(f"Expected X of [B,{self.cfg.seq_len},{self.cfg.enc_in}], got {tuple(x_enc.shape)}")
        if not torch.is_floating_point(x_enc):
            raise TypeError("Time series input must be floating point")
        x_norm, stats = self.revin.normalize(x_enc)
        scales = self.decomp(x_norm)
        memories = [self.encoder(s, m) for m, s in enumerate(scales)]

        # All FFT calculations use float32. Query router output remains trainable.
        freq_idx, periods, x_period = self.query_generator.fft_candidates(x_norm, self.cfg.top_k)
        query_input, router_weights = self.query_generator(x_period)

        biases = []
        reliability_values = []
        for scale in scales:
            bias, reliability = self.period_reliability(scale, freq_idx, periods)
            biases.append(bias)
            reliability_values.append(reliability)

        decoded, attention_maps = self.decoder(query_input, memories, biases, return_attention=return_aux)
        fused, scale_weights = self.scale_fusion(decoded, query_input)
        patch_pred = self.prediction_head(fused)         # [B,C,Nf,FuturePatchLen]
        b, c, _, _ = patch_pred.shape
        y_norm = patch_pred.reshape(b, c, -1)[..., :self.cfg.pred_len].transpose(1, 2)
        y_hat = self.revin.denormalize(y_norm, stats)

        if self.c_out == 1 and self.cfg.enc_in != 1:
            y_hat = y_hat[..., -1:]

        if not return_aux:
            return y_hat
        aux = {
            "router_weights": router_weights.detach(),
            "selected_periods": periods.detach(),
            "period_reliability": torch.stack(reliability_values, dim=2).detach(),
            "scale_weights": scale_weights.detach(),
            "period_bias": torch.stack(biases, dim=2).detach(),
            "attention_weights": torch.stack(attention_maps, dim=2).detach(),
        }
        return y_hat, aux


# Backward-compatible name for direct history-only callers.
PeriodAwareQueryForecast = Model
