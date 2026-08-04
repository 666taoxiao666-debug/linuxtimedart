import torch


def compute_regime_pseudo_labels(
    x_patch,
    stable_thresh=0.15,
    ramp_thresh=0.25,
):
    """
    Assign pseudo regime labels from patch-level power dynamics.

    Modes:
        0 - stable (low std & low max diff)
        1 - ramp up (positive mean first-order diff)
        2 - ramp down (negative mean first-order diff)

    :param x_patch: [batch_size, seq_len, patch_len]
    :return: pseudo_labels [batch_size]
    """
    if x_patch.size(-1) < 2:
        return torch.zeros(x_patch.size(0), dtype=torch.long, device=x_patch.device)

    diff = x_patch[..., 1:] - x_patch[..., :-1]
    mean_diff = diff.mean(dim=(-1, -2))
    max_abs_diff = diff.abs().amax(dim=(-1, -2))
    std = x_patch.std(dim=(-1, -2))

    labels = torch.zeros(x_patch.size(0), dtype=torch.long, device=x_patch.device)
    stable_mask = (std < stable_thresh) & (max_abs_diff < ramp_thresh)
    labels[stable_mask] = 0

    active_mask = ~stable_mask
    labels[active_mask & (mean_diff > 0)] = 1
    labels[active_mask & (mean_diff <= 0)] = 2
    return labels


def compute_regime_pseudo_labels_from_series(
    x,
    stable_thresh=0.15,
    ramp_thresh=0.25,
):
    """Build one regime label per sample from a power series.

    ``x`` may be ``[batch, length]`` or ``[batch, length, 1]``.  Multivariate
    input is rejected deliberately: selecting/averaging arbitrary weather and
    identifier channels would no longer describe a wind-power regime.
    """
    if x.ndim == 2:
        power = x.unsqueeze(1)
    elif x.ndim == 3 and x.size(-1) == 1:
        power = x.transpose(1, 2)
    else:
        raise ValueError(
            "Regime pseudo labels require a single power channel with shape "
            "[batch, length] or [batch, length, 1]"
        )
    return compute_regime_pseudo_labels(
        power,
        stable_thresh=stable_thresh,
        ramp_thresh=ramp_thresh,
    )
