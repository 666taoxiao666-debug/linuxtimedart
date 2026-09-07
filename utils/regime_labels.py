"""Leakage-safe pseudo labels and diagnostics for wind-power regimes."""

from __future__ import annotations

import numpy as np
import torch


REGIME_NAMES = ("stable", "ramp_up", "ramp_down")


def _flatten_history(x_patch: torch.Tensor) -> torch.Tensor:
    if x_patch.ndim < 2:
        raise ValueError(
            "Regime histories require a batch dimension and at least one time dimension"
        )
    return x_patch.reshape(x_patch.size(0), -1)


def compute_regime_trend_score(x_patch: torch.Tensor) -> torch.Tensor:
    """Return one least-squares trend score per observed history window."""

    history = _flatten_history(x_patch)
    if history.size(1) < 2:
        return torch.zeros(history.size(0), dtype=history.dtype, device=history.device)
    time_axis = torch.linspace(
        -0.5,
        0.5,
        history.size(1),
        dtype=history.dtype,
        device=history.device,
    )
    centered = history - history.mean(dim=1, keepdim=True)
    return (centered * time_axis).sum(dim=1) / time_axis.square().sum().clamp_min(
        torch.finfo(history.dtype).eps
    )


def compute_regime_pseudo_labels(
    x_patch,
    stable_thresh=0.15,
    ramp_thresh=0.25,
    *,
    method="legacy_volatility",
    down_thresh=None,
    up_thresh=None,
):
    """Assign stable/ramp-up/ramp-down labels from input history only.

    ``trend_quantile`` is the paper protocol. Its thresholds must be fitted on
    training histories and then frozen. The legacy rule remains available only
    for reproducing old checkpoints.
    """

    history = _flatten_history(x_patch)
    if history.size(1) < 2:
        return torch.zeros(history.size(0), dtype=torch.long, device=history.device)

    if method == "trend_quantile":
        if down_thresh is None or up_thresh is None:
            raise ValueError(
                "trend_quantile regime labels require train-fitted down/up thresholds"
            )
        down_value = float(torch.as_tensor(down_thresh).detach().cpu().item())
        up_value = float(torch.as_tensor(up_thresh).detach().cpu().item())
        if not np.isfinite(down_value) or not np.isfinite(up_value):
            raise ValueError("Regime thresholds must be finite")
        if down_value >= up_value:
            raise ValueError(
                f"Regime thresholds must satisfy down < up, got {down_value} >= {up_value}"
            )
        trend = compute_regime_trend_score(history)
        labels = torch.zeros(history.size(0), dtype=torch.long, device=history.device)
        labels[trend > up_value] = 1
        labels[trend < down_value] = 2
        return labels

    if method != "legacy_volatility":
        raise ValueError(f"Unknown regime label method: {method}")
    diff = history[:, 1:] - history[:, :-1]
    mean_diff = diff.mean(dim=1)
    max_abs_diff = diff.abs().amax(dim=1)
    std = history.std(dim=1)
    labels = torch.zeros(history.size(0), dtype=torch.long, device=history.device)
    stable_mask = (std < stable_thresh) & (max_abs_diff < ramp_thresh)
    active_mask = ~stable_mask
    labels[active_mask & (mean_diff > 0)] = 1
    labels[active_mask & (mean_diff <= 0)] = 2
    return labels


def compute_regime_pseudo_labels_from_series(
    x,
    stable_thresh=0.15,
    ramp_thresh=0.25,
    *,
    method="legacy_volatility",
    down_thresh=None,
    up_thresh=None,
):
    """Build one regime label per sample from a single power series."""

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
        method=method,
        down_thresh=down_thresh,
        up_thresh=up_thresh,
    )


def calibrate_regime_thresholds_from_dataset(
    dataset,
    *,
    target_index=-1,
    quantile=1.0 / 3.0,
    max_samples=50000,
    min_class_fraction=0.05,
    chunk_size=2048,
):
    """Fit trend thresholds from deterministic training histories only."""

    if not 0.0 < float(quantile) < 0.5:
        raise ValueError("regime calibration quantile must be between 0 and 0.5")
    if int(max_samples) < 3:
        raise ValueError("regime calibration requires at least three samples")
    sample_count = min(int(max_samples), len(dataset))
    if sample_count < 3:
        raise ValueError("training split is too small to calibrate three regimes")
    selected = np.unique(
        np.linspace(0, len(dataset) - 1, num=sample_count, dtype=np.int64)
    )
    score_parts = []

    if all(hasattr(dataset, name) for name in ("data_x", "window_starts", "seq_len")):
        values = np.asarray(dataset.data_x)
        starts = np.asarray(dataset.window_starts, dtype=np.int64)[selected]
        feature_index = int(target_index) % int(values.shape[1])
        offsets = np.arange(int(dataset.seq_len), dtype=np.int64)
        for offset in range(0, len(starts), int(chunk_size)):
            chunk_starts = starts[offset : offset + int(chunk_size)]
            rows = chunk_starts[:, None] + offsets[None, :]
            histories = torch.as_tensor(
                values[rows, feature_index], dtype=torch.float32
            )
            score_parts.append(compute_regime_trend_score(histories).cpu().numpy())
    else:
        buffer = []
        for index in selected:
            history = np.asarray(dataset[int(index)][0])
            feature_index = int(target_index) % int(history.shape[-1])
            buffer.append(history[:, feature_index])
            if len(buffer) == int(chunk_size):
                tensor = torch.as_tensor(np.stack(buffer), dtype=torch.float32)
                score_parts.append(compute_regime_trend_score(tensor).cpu().numpy())
                buffer.clear()
        if buffer:
            tensor = torch.as_tensor(np.stack(buffer), dtype=torch.float32)
            score_parts.append(compute_regime_trend_score(tensor).cpu().numpy())

    scores = np.concatenate(score_parts).astype(np.float64, copy=False)
    down_thresh, up_thresh = np.quantile(
        scores, [float(quantile), 1.0 - float(quantile)]
    )
    if not np.isfinite(down_thresh) or not np.isfinite(up_thresh):
        raise ValueError("non-finite values encountered during regime calibration")
    if down_thresh >= up_thresh:
        # A large physically stable plateau can put both quantiles at exactly
        # zero. Keep identical histories in the same stable class and place the
        # boundaries halfway toward the nearest observed down/up trend.
        pivot = float(down_thresh)
        lower = scores[scores < pivot]
        upper = scores[scores > pivot]
        if not len(lower) or not len(upper):
            raise ValueError(
                "training histories do not contain both downward and upward "
                "trends required for three leakage-safe regimes"
            )
        down_thresh = 0.5 * (float(lower.max()) + pivot)
        up_thresh = 0.5 * (pivot + float(upper.min()))

    labels = np.zeros(len(scores), dtype=np.int64)
    labels[scores > up_thresh] = 1
    labels[scores < down_thresh] = 2
    counts = np.bincount(labels, minlength=3)
    fractions = counts / max(1, counts.sum())
    if float(fractions.min()) < float(min_class_fraction):
        raise ValueError(
            "train-only regime calibration produced an unsupported class: "
            f"counts={counts.tolist()}, fractions={fractions.tolist()}"
        )
    return {
        "method": "trend_quantile",
        "source_split": "train",
        "sample_count": int(len(scores)),
        "quantile": float(quantile),
        "down_thresh": float(down_thresh),
        "up_thresh": float(up_thresh),
        "class_counts": [int(value) for value in counts],
        "class_fractions": [float(value) for value in fractions],
    }


def update_regime_confusion(confusion, predictions, labels):
    """Accumulate a fixed-size confusion matrix without retaining all batches."""

    matrix = np.asarray(confusion)
    size = int(matrix.shape[0])
    pred = torch.as_tensor(predictions).detach().reshape(-1).cpu().numpy()
    true = torch.as_tensor(labels).detach().reshape(-1).cpu().numpy()
    valid = (true >= 0) & (true < size) & (pred >= 0) & (pred < size)
    encoded = true[valid] * size + pred[valid]
    matrix += np.bincount(encoded, minlength=size * size).reshape(size, size)
    return matrix


def summarize_regime_confusion(confusion):
    """Return imbalance-aware macro-F1, recall and support diagnostics."""

    matrix = np.asarray(confusion, dtype=np.int64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    diagonal = np.diag(matrix).astype(np.float64)
    recall = np.divide(
        diagonal, support, out=np.zeros_like(diagonal), where=support > 0
    )
    precision = np.divide(
        diagonal, predicted, out=np.zeros_like(diagonal), where=predicted > 0
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(diagonal),
        where=(precision + recall) > 0,
    )
    total = int(matrix.sum())
    return {
        "regime_accuracy": float(diagonal.sum() / total) if total else 0.0,
        "regime_macro_f1": float(f1.mean()) if len(f1) else 0.0,
        "regime_counts": ";".join(str(int(value)) for value in support),
        "regime_pred_counts": ";".join(str(int(value)) for value in predicted),
        "regime_recall": ";".join(f"{float(value):.4f}" for value in recall),
        "regime_f1": ";".join(f"{float(value):.4f}" for value in f1),
        "regime_confusion": "/".join(
            ";".join(str(int(value)) for value in row) for row in matrix
        ),
    }
