"""History-conditioned residual expert with optional executable Wiki features.

Numeric/Wiki arms have identical parameter counts, initialization, label budget
and optimization. Wiki features are typed numerical case summaries, not LLM text.
"""
import torch
from torch import nn


class CausalBlock(nn.Module):
    def __init__(self, input_channels, width, dilation, dropout):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(input_channels, width, 3, dilation=dilation, padding=self.padding)
        self.skip = nn.Conv1d(input_channels, width, 1) if input_channels != width else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.skip(x) + self.dropout(torch.nn.functional.gelu(self.conv(x)[..., :-self.padding]))


class ContextualWindResidual(nn.Module):
    def __init__(self, channels=7, width=32, horizon=2, memory_features=20, dropout=.1, bound=.2):
        super().__init__()
        self.horizon, self.memory_features, self.bound = horizon, memory_features, bound
        self.encoder = nn.Sequential(CausalBlock(channels, width, 1, dropout),
            CausalBlock(width, width, 2, dropout), CausalBlock(width, width, 4, dropout))
        self.lead_embedding = nn.Embedding(horizon, 8)
        self.hidden = nn.Sequential(nn.Linear(width * 2 + channels + 2 + 8 + memory_features, 64),
            nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(64, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, history, base_change, anchor_ratio, memory, *, use_memory):
        if history.ndim != 3 or base_change.shape != (len(history), self.horizon):
            raise ValueError("Expected [batch,channel,history] and [batch,horizon]")
        if memory.shape != (len(history), self.horizon, self.memory_features):
            raise ValueError("Typed memory summary does not match horizon/schema")
        z = self.encoder(history)
        state = torch.cat((z[..., -1], z.mean(-1), history[..., -1]), -1)
        state = state[:, None, :].expand(-1, self.horizon, -1)
        lead = self.lead_embedding(torch.arange(self.horizon, device=history.device))[None].expand(len(history), -1, -1)
        tokens = memory if use_memory else torch.zeros_like(memory)
        features = torch.cat((state, base_change[..., None], anchor_ratio[:, None, :].expand(-1, self.horizon, -1), lead, tokens), -1)
        return self.bound * torch.tanh(self.head(self.hidden(features)).squeeze(-1))
