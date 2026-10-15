"""Forecast changes as actual case-weighted outcomes, not a free value head."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ResponseCaseAttention(nn.Module):
    def __init__(self, features=23, horizon=2, width=32, embedding=16, dropout=.1, bound=.2, fixed=False):
        super().__init__()
        self.features, self.horizon, self.bound, self.fixed = features, horizon, bound, fixed
        self.encoder = nn.Sequential(nn.Linear(features, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, embedding))
        self.lead_metric = nn.Parameter(torch.zeros(horizon, embedding))
        # Fitted exactly on the fitting period, never on selection/calibration.
        # Keeping it outside the proposal loss avoids a dead zero-confidence head.
        self.register_buffer("alpha", torch.zeros(horizon))
        if fixed:
            self.requires_grad_(False)

    def set_alpha(self, values):
        values = torch.as_tensor(values, dtype=self.alpha.dtype, device=self.alpha.device)
        if values.shape != self.alpha.shape or not torch.isfinite(values).all() or torch.any((values < 0) | (values > 1)):
            raise ValueError("Invalid fitting-only confidence coefficients")
        self.alpha.copy_(values)

    def components(self, query, neighbors, values, prior, base_change, supported):
        # query[n,event,d], neighbors[n,event,k,d], values[n,h,event,k].
        n, events, count, dimension = neighbors.shape
        if (dimension != self.features or query.shape != (n, events, dimension)
                or values.shape != (n, self.horizon, events, count) or prior.shape != (n, events, count)
                or base_change.shape != (n, self.horizon) or supported.shape != (n, self.horizon, events)):
            raise ValueError("Response attention input schema mismatch")
        if self.fixed:
            distance = (query[:, :, None] - neighbors).square().mean(-1)[:, None].expand(-1, self.horizon, -1, -1)
        else:
            q = self.encoder(query)
            k = self.encoder(neighbors)
            metric = torch.softmax(self.lead_metric, -1) * self.lead_metric.shape[-1]
            distance = ((q[:, None, :, None] - k[:, None]).square() * metric[None, :, None, None]).mean(-1)
        weights = torch.softmax(-distance + prior.clamp_min(1e-30).log()[:, None], -1)
        # No free value projection: only observed historical case increments.
        future_change = (weights * values).sum(-1)
        group_proposal = (future_change - base_change[:, :, None]).clamp(-self.bound, self.bound)
        mask = supported.bool()
        proposal = (group_proposal * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        proposal = torch.where(mask.any(-1), proposal, torch.zeros_like(proposal))
        delta = proposal * self.alpha[None]
        return dict(delta=delta, proposal=proposal, weights=weights, group_proposal=group_proposal)

    def forward(self, *args):
        return self.components(*args)["delta"]


def tensor_inputs(view):
    return tuple(torch.from_numpy(np.ascontiguousarray(view[key])) for key in
                 ("query", "neighbors", "values", "prior", "base_change", "supported"))
