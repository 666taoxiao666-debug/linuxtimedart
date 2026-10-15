"""A numerical-only short-horizon forecaster with an exact last-power skip."""
from __future__ import annotations

import torch
from torch import nn
from models.ContextualWindResidual import CausalBlock


class CausalResidualBase(nn.Module):
    def __init__(self, channels=7, width=32, horizon=2, dropout=.1, bound_std=4.):
        super().__init__()
        if channels != 7 or horizon != 2 or bound_std <= 0:
            raise ValueError("Expected the declared seven-channel, two-hour numerical task")
        self.horizon, self.bound_std = horizon, float(bound_std)
        self.encoder = nn.Sequential(CausalBlock(channels, width, 1, dropout),
            CausalBlock(width, width, 2, dropout), CausalBlock(width, width, 4, dropout))
        self.hidden = nn.Sequential(nn.Linear(3 * width + channels, 64), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(64, horizon)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def residual(self, history):
        if history.ndim != 3 or history.shape[-1] != 7 or history.shape[1] < 24:
            raise ValueError("Expected [batch,past hours,seven observed channels]")
        # Only power is centered on its last observation. Absolute standardized
        # wind/control levels remain available; no window-wise wind RevIN.
        relative = history.clone()
        relative[:, :, -1] = relative[:, :, -1] - history[:, -1:, -1]
        z = self.encoder(relative.transpose(1, 2))
        state = torch.cat((z[:, :, -1], z[:, :, -3:].mean(-1), z[:, :, -12:].mean(-1), history[:, -1]), -1)
        return self.bound_std * torch.tanh(self.head(self.hidden(state)))

    def forward(self, history):
        return (history[:, -1:, -1] + self.residual(history))[..., None]
