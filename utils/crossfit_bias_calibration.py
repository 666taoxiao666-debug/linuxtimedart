"""Train-only, horizon-wise bias calibration with a disjoint support gate."""
from __future__ import annotations

import numpy as np


def _validate_pair(prediction, truth):
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("prediction and truth must be aligned 2-D arrays")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("prediction and truth must be finite")
    return prediction, truth


def _gains(reference, corrected, truth):
    ref_error = reference - truth
    corrected_error = corrected - truth
    return {
        "mae_gain_kw": float(np.mean(np.abs(ref_error)) - np.mean(np.abs(corrected_error))),
        "rmse_gain_kw": float(
            np.sqrt(np.mean(ref_error ** 2)) - np.sqrt(np.mean(corrected_error ** 2))
        ),
        "bias_before_kw": float(np.mean(ref_error)),
        "bias_after_kw": float(np.mean(corrected_error)),
    }


def fit_bias_guard(blocks, *, rated_power=1500.0, max_capacity_ratio=0.05,
                   min_windows=64):
    """Fit magnitude on block 1 and use block 2 only to accept/reject it.

    The deployable model contains no labels.  The additive correction is the
    clipped negative median forecast error from the first block, which is the
    empirical MAE-optimal constant.  A horizon is enabled only when that fixed
    correction strictly improves both MAE and RMSE in both training blocks.
    """
    if len(blocks) != 2:
        raise ValueError("Exactly two chronological training blocks required")
    prepared = [_validate_pair(*block) for block in blocks]
    if prepared[0][0].shape[1] != prepared[1][0].shape[1]:
        raise ValueError("Training blocks must have the same horizon")
    if rated_power <= 0 or not 0 < max_capacity_ratio <= 1 or min_windows < 1:
        raise ValueError("Invalid physical bound or support requirement")

    cap_kw = float(rated_power) * float(max_capacity_ratio)
    horizon = prepared[0][0].shape[1]
    corrections = np.zeros(horizon, dtype=float)
    accepted = np.zeros(horizon, dtype=bool)
    evidence = []
    for step in range(horizon):
        fit_prediction, fit_truth = prepared[0]
        delta = float(np.clip(
            np.median(fit_truth[:, step] - fit_prediction[:, step]), -cap_kw, cap_kw
        ))
        rows = []
        for prediction, truth in prepared:
            corrected = np.clip(prediction[:, step] + delta, 0.0, float(rated_power))
            row = {"windows": int(prediction.shape[0]), **_gains(
                prediction[:, step], corrected, truth[:, step]
            )}
            rows.append(row)
        keep = (
            abs(delta) > 0
            and all(row["windows"] >= min_windows for row in rows)
            and all(row["mae_gain_kw"] > 0 and row["rmse_gain_kw"] > 0 for row in rows)
        )
        corrections[step] = delta if keep else 0.0
        accepted[step] = keep
        evidence.append({
            "step": step + 1,
            "proposed_correction_kw": delta,
            "accepted": bool(keep),
            "blocks": rows,
        })
    return {
        "schema_version": 1,
        "rated_power_kw": float(rated_power),
        "max_capacity_ratio": float(max_capacity_ratio),
        "max_abs_correction_kw": cap_kw,
        "min_windows": int(min_windows),
        "correction_kw": corrections.tolist(),
        "accepted": accepted.tolist(),
        "evidence": evidence,
        "magnitude_source": "first chronological training block median residual only",
        "acceptance_source": "first and second chronological training blocks; strict MAE and RMSE improvement",
    }


def apply_bias(model, prediction):
    """Apply a fitted correction without accepting targets or future context."""
    prediction = np.asarray(prediction, dtype=float)
    corrections = np.asarray(model["correction_kw"], dtype=float)
    rated_power = float(model["rated_power_kw"])
    if prediction.ndim != 2 or prediction.shape[1] != corrections.size:
        raise ValueError("Prediction horizon and bias model do not align")
    if not np.isfinite(prediction).all() or not np.isfinite(corrections).all():
        raise ValueError("Prediction and correction must be finite")
    return np.clip(prediction + corrections[None, :], 0.0, rated_power)
