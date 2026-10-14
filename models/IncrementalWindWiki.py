"""A Wiki-only increment above an externally frozen numerical prediction.

This module never owns or updates the numerical forecaster. Unsupported leads
have an exact zero increment, including after training. The zero-memory arm is
a matched ablation of this same module, not a retrained numerical baseline.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


CONTEXT_SCALES = (
    ("power_ratio", 1.), ("wind_mean_mps", 25.), ("wind_std_mps", 10.),
    ("power_change_kw", None), ("wind_change_mps", 10.), ("wind_max_step_mps", 10.),
    ("pitch_mean_div90", 1.), ("pitch_change_div90", 1.),
    ("nacelle_heading_sin_mean", 1.), ("nacelle_heading_cos_mean", 1.),
)


def inputs(pack, capacity):
    """Only observed context and predictions, never outcome values, are read."""
    numeric, anchor, reference = (np.asarray(pack[key], dtype=float) for key in ("numeric", "anchor", "reference"))
    overlap = np.asarray(pack["overlap"], dtype=bool)
    n, horizon = numeric.shape
    events = overlap.shape[-1]
    memory = np.asarray(pack["memory"], dtype=np.float32).reshape(n, horizon, 5, events).copy()
    # Old tokens encode (case - original Trend)/capacity. The executing
    # correction is now explicitly relative to the frozen NumericContext.
    memory[:, :, 0, :] -= ((numeric - reference) / capacity)[..., None]
    memory *= overlap[:, :, None, :]
    context = np.column_stack([np.asarray(pack["context"][name]) / (capacity if scale is None else scale)
                               for name, scale in CONTEXT_SCALES]).astype(np.float32)
    numeric_state = np.stack(((numeric - anchor) / capacity, anchor / capacity), -1).astype(np.float32)
    result = dict(context=context, memory=memory.reshape(n, horizon, -1),
                  numeric_state=numeric_state, supported=overlap.any(-1))
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite incremental Wiki input")
    return result


class IncrementalWindWiki(nn.Module):
    def __init__(self, horizon=2, memory_features=20, width=32, dropout=.1, bound=.2):
        super().__init__()
        self.horizon, self.memory_features, self.bound = horizon, memory_features, bound
        self.lead = nn.Embedding(horizon, 8)
        self.hidden = nn.Sequential(nn.Linear(len(CONTEXT_SCALES) + 2 + 8 + memory_features, width),
                                    nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(width, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, context, numeric_state, memory, supported, *, use_memory=True):
        n = len(context)
        if (context.shape != (n, len(CONTEXT_SCALES)) or numeric_state.shape != (n, self.horizon, 2)
                or memory.shape != (n, self.horizon, self.memory_features)
                or supported.shape != (n, self.horizon)):
            raise ValueError("Incremental Wiki shape/schema mismatch")
        lead = self.lead(torch.arange(self.horizon, device=context.device))[None].expand(n, -1, -1)
        tokens = memory if use_memory else torch.zeros_like(memory)
        features = torch.cat((context[:, None].expand(-1, self.horizon, -1), numeric_state, lead, tokens), -1)
        delta = self.bound * torch.tanh(self.head(self.hidden(features)).squeeze(-1))
        return torch.where(supported.bool(), delta, torch.zeros_like(delta))
