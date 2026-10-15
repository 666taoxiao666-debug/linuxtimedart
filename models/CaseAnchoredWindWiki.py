"""A confidence learner whose numerical proposal remains an executable case.

The head can shrink/abstain, not reverse a case correction or invent one when
the supported case readout is zero. Whether that restriction helps is tested
against the frozen, otherwise matched free-residual and zero-memory controls.
"""
from __future__ import annotations

import torch
from models.IncrementalWindWiki import IncrementalWindWiki


def case_proposal(memory, supported, bound):
    n, horizon, fields = memory.shape
    if fields % 5:
        raise ValueError("Expected five typed fields per event")
    tokens = memory.reshape(n, horizon, 5, fields // 5)
    overlap = tokens[:, :, 4, :] > .5
    count = overlap.sum(-1)
    if not torch.equal(count > 0, supported.bool()):
        raise ValueError("Case token support and executing support disagree")
    return ((tokens[:, :, 0, :] * overlap).sum(-1) / count.clamp_min(1)).clamp(-bound, bound)


class CaseAnchoredWindWiki(IncrementalWindWiki):
    def components(self, context, numeric_state, memory, supported, *, use_memory=True):
        if not use_memory:
            raise ValueError("Case-anchored execution needs actual case values; use the explicit free zero-memory control")
        free = super().forward(context, numeric_state, memory, supported, use_memory=True)
        proposal = case_proposal(memory, supported, self.bound)
        # clamp has a nonzero derivative at zero, unlike ReLU's zero derivative
        # there; exact zero initialization therefore remains trainable.
        confidence = (free / self.bound).clamp(0., 1.)
        delta = torch.where(supported.bool(), confidence * proposal, torch.zeros_like(proposal))
        return dict(delta=delta, confidence=confidence, proposal=proposal)

    def forward(self, context, numeric_state, memory, supported, *, use_memory=True):
        return self.components(context, numeric_state, memory, supported, use_memory=use_memory)["delta"]
