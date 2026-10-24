"""Small fixed-size PyTorch port of the official TSMixer *basic* model.

Upstream: google-research/google-research, ``tsmixer/tsmixer_basic/models/tsmixer.py``
(TensorFlow/Keras), Apache License 2.0, Copyright The Google Research Authors.
Paper: Chen et al., "TSMixer: An All-MLP Architecture for Time Series Forecasting", TMLR 2023.
The exact upstream commit and file SHA-256 are pinned in ``configs/tsmixer_upstream_pin_v25.json``
before any real fit; this module is a port, NOT a reproduction of the paper's tuned configurations.

Port semantics (upstream ``res_block``):
  temporal:  x = LN_(L,C)(inputs) -> [B,C,L] -> Dense(L, relu) -> [B,L,C] -> Dropout -> res = x + inputs
  feature:   x = LN_(L,C)(res) -> Dense(ff, relu) -> Dropout -> Dense(C) -> Dropout -> return x + res
  head:      x[:, :, target_slice] -> [B,C',L] -> Dense(pred_len) -> [B,H,C']
Keras LayerNormalization(axis=[-2,-1]) normalises each sample over its time x channel plane with
eps=1e-3 and learnable (L,C) gamma/beta; Keras Dense uses Glorot-uniform kernels and zero bias.

Documented differences from upstream: PyTorch instead of TensorFlow; RevIN is not used (upstream
optional); a single fixed configuration (2 blocks, ff_dim 32, dropout 0.1) chosen by design, not by
search; only the power slot is projected; an optional information-ablation mask
(``input_mode='power_only'``) replaces auxiliary standardized channels with 0 before any operation.
There is no future covariate, label, turbine embedding, prompt, Wiki or persistence-anchor input.
"""
from __future__ import annotations

import torch
from torch import nn

INPUT_MODES = ("full", "power_only")


def power_only_input(history: torch.Tensor, target_slot: int) -> torch.Tensor:
    """Keep only the standardized target channel; every other channel becomes exactly 0.

    0 is the training-scaler mean, not physical zero. ``zeros_like`` never reads the masked values,
    so arbitrary (even non-finite) auxiliary inputs cannot influence the result.
    """
    masked = torch.zeros_like(history)
    masked[..., target_slot] = history[..., target_slot]
    return masked


class MixerBlock(nn.Module):
    """One upstream ``res_block`` with LayerNorm over (time, channel)."""

    def __init__(self, seq_len: int, channels: int, ff_dim: int, dropout: float, norm_eps: float):
        super().__init__()
        self.temporal_norm = nn.LayerNorm((seq_len, channels), eps=norm_eps, elementwise_affine=True)
        self.temporal = nn.Linear(seq_len, seq_len)
        self.feature_norm = nn.LayerNorm((seq_len, channels), eps=norm_eps, elementwise_affine=True)
        self.feature_in = nn.Linear(channels, ff_dim)
        self.feature_out = nn.Linear(ff_dim, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.temporal_norm(inputs)
        x = torch.relu(self.temporal(x.transpose(1, 2))).transpose(1, 2)
        x = self.dropout(x)
        res = x + inputs
        x = self.feature_norm(res)
        x = torch.relu(self.feature_in(x))
        x = self.dropout(x)
        x = self.feature_out(x)
        x = self.dropout(x)
        return x + res


class Model(nn.Module):
    """Direct multi-step TSMixer-basic forecaster: [B, L, C] standardized history -> [B, H, 1] power."""

    def __init__(self, *, seq_len: int = 24, channels: int = 7, pred_len: int = 4, blocks: int = 2,
                 ff_dim: int = 32, dropout: float = 0.1, norm_eps: float = 1e-3, target_slot: int = 6,
                 input_mode: str = "full"):
        super().__init__()
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}")
        if not 0 <= int(target_slot) < int(channels):
            raise ValueError("target_slot outside the channel axis")
        self.seq_len, self.channels, self.pred_len = int(seq_len), int(channels), int(pred_len)
        self.target_slot, self.input_mode = int(target_slot), input_mode
        # input_mode is a plain attribute, not a buffer: Full and PowerOnly state_dicts are identical.
        self.blocks = nn.ModuleList([MixerBlock(self.seq_len, self.channels, int(ff_dim), float(dropout),
                                                float(norm_eps)) for _ in range(int(blocks))])
        self.head = nn.Linear(self.seq_len, self.pred_len)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or tuple(history.shape[1:]) != (self.seq_len, self.channels):
            raise ValueError(f"Expected [B, {self.seq_len}, {self.channels}] history, got {tuple(history.shape)}")
        x = power_only_input(history, self.target_slot) if self.input_mode == "power_only" else history
        for block in self.blocks:
            x = block(x)
        return self.head(x[..., self.target_slot]).unsqueeze(-1)
