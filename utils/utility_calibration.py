"""History-only features and chronological calibration for Wiki adapters."""
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader


PHYSICAL_FEATURE_DIM = 45


def physical_history_features(raw_history, feature_columns, rated_power):
    """Last/mean/change/slope/std over 3, 6, 12 observed ten-minute samples.

    Scales are fixed physical units, never window/validation-fitted statistics.
    No forecast labels are accepted by this function.
    """
    pieces = []
    for name, scale in (("Wspd", 25.), ("power", float(rated_power)), ("Pab_mean", 90.)):
        if name not in feature_columns or scale <= 0:
            raise ValueError(f"Required physical feature or scale unavailable: {name}")
        series = raw_history[..., feature_columns.index(name)] / scale
        for length in (3, 6, 12):
            window = series[:, -length:]
            t = torch.arange(window.size(1), device=window.device, dtype=window.dtype)
            t = t - t.mean()
            slope = (window * t).sum(dim=1) / t.square().sum().clamp_min(1.)
            pieces.extend([window[:, -1], window.mean(dim=1),
                           window[:, -1] - window[:, 0], slope,
                           window.std(dim=1, unbiased=False)])
    return torch.stack(pieces, dim=-1).detach()


def split_utility_training(dataset, fraction=0.2):
    """Split ONLY original training windows by global forecast target time.

    Windows with labels on both sides of the cutoff are purged. Calibration
    history may include earlier observations, as at deployment. The existing
    backbone/scaler has seen the original train set: this is adapter/gate
    separation, not full-model out-of-fold fitting.
    """
    if getattr(dataset, "flag", None) != "train":
        raise ValueError("Utility calibration may split only a training dataset")
    if not 0 < float(fraction) < 0.5:
        raise ValueError("Calibration fraction must lie in (0, 0.5)")
    starts = np.asarray(dataset.window_starts)
    dates = np.asarray(dataset.dates)
    first = dates[starts + dataset.seq_len]
    last = dates[starts + dataset.seq_len + dataset.pred_len - 1]
    times = np.unique(first)
    if len(times) < 3:
        raise ValueError("Too few training timestamps for utility calibration")
    cut = times[min(len(times) - 1, max(1, int(len(times) * (1 - fraction))))]
    fit_mask, calibration_mask = last < cut, first >= cut
    if not fit_mask.any() or not calibration_mask.any():
        raise ValueError("Empty adapter or gate split after purging boundary labels")
    fit, calibration = copy.copy(dataset), copy.copy(dataset)
    fit.window_starts = starts[fit_mask].copy()
    calibration.window_starts = starts[calibration_mask].copy()
    audit = {
        "source": "original_train_only", "cutoff": str(cut),
        "adapter_windows": int(fit_mask.sum()), "gate_windows": int(calibration_mask.sum()),
        "purged_windows": int((~(fit_mask | calibration_mask)).sum()),
        "adapter_target_end": str(last[fit_mask].max()),
        "gate_target_start": str(first[calibration_mask].min()),
        "full_model_cross_fitted": False,
    }
    return fit, calibration, audit


def utility_loaders(dataset, args):
    fit, calibration, audit = split_utility_training(dataset, args.utility_calibration_fraction)
    loaders = []
    for offset, part in enumerate((fit, calibration)):
        loaders.append(DataLoader(
            part, batch_size=args.batch_size, shuffle=True, drop_last=False,
            num_workers=0, generator=torch.Generator().manual_seed(args.seed + 730 + offset),
        ))
    return loaders[0], loaders[1], audit


def utility_action_probabilities(utilities, availability, min_gain, temperature):
    """Differentiable abstain/event/composition weights; absent evidence is zero."""
    if temperature <= 0:
        raise ValueError("Utility temperature must be positive")
    available = availability[:, None, :].expand_as(utilities)
    logits = torch.cat([
        torch.full_like(utilities[..., :1], float(min_gain)),
        utilities.masked_fill(~available, float("-inf")),
    ], dim=-1)
    return torch.softmax(logits / float(temperature), dim=-1)
