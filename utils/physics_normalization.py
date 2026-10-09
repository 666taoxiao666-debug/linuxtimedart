"""History-only normalization with an invertible absolute-feature contract."""
from __future__ import annotations

import torch


def normalize_history(history, keep_indices=(), *, identity_inverse_for_kept=True):
    """Keep named globally-standardized channels; instance-normalize the rest.

    The returned mean/scale also invert kept channels (mean=0, scale=1).
    No target, future timestamp, or fitted statistic is accepted here.
    """
    mean = history.mean(dim=1, keepdim=True).detach()
    centered = history - mean
    scale = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    normalized = centered / scale
    if keep_indices:
        indices = list(keep_indices)
        normalized = normalized.clone()
        normalized[:, :, indices] = history[:, :, indices]
        if identity_inverse_for_kept:
            mean, scale = mean.clone(), scale.clone()
            mean[:, :, indices] = 0.0
            scale[:, :, indices] = 1.0
    return normalized, mean, scale


def check_normalization_contract(source_args, target_model):
    """Fail before transferring an encoder trained under another convention."""
    source = bool(source_args.get("consistent_physics_norm", False))
    target = bool(getattr(target_model, "consistent_physics_norm", False))
    if source != target:
        raise RuntimeError("Physics normalization checkpoint mismatch: retrain a matched pretrain; "
                           f"checkpoint={source}, target={target}")
    if target:
        if not source_args.get("use_norm") or not source_args.get("revin_keep_wind"):
            raise RuntimeError("Consistent physics normalization checkpoint lacks its wind/scale contract")
        if list(source_args.get("feature_columns", [])) != list(target_model.feature_columns):
            raise RuntimeError("Physics normalization feature order mismatch")
