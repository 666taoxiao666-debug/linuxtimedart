"""Flattened linear direct forecaster: one Linear(seq_len * channels -> pred_len) on the standardized history.

The simplest model that can combine channels (unlike channel-independent DLinear). Used in v28 as the low-variance
architecture for the end-of-hour information comparison. Output [B, pred_len, 1] = standardized power. PyTorch default
Linear initialisation; no normalisation, no hidden layer, no dropout. Not a contribution.
"""
from __future__ import annotations

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, *, seq_len: int = 24, channels: int = 7, pred_len: int = 4):
        super().__init__()
        self.seq_len, self.channels, self.pred_len = int(seq_len), int(channels), int(pred_len)
        self.linear = nn.Linear(self.seq_len * self.channels, self.pred_len)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or tuple(history.shape[1:]) != (self.seq_len, self.channels):
            raise ValueError(f"Expected [B, {self.seq_len}, {self.channels}] history, got {tuple(history.shape)}")
        return self.linear(history.reshape(history.shape[0], -1)).unsqueeze(-1)
