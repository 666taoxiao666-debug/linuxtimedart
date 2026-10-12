"""History-physical-cell bias correction with disjoint train-only support."""
from __future__ import annotations

import numpy as np


def _validate(prediction, truth, group, n_groups):
    prediction = np.asarray(prediction, dtype=float)
    truth = np.asarray(truth, dtype=float)
    group = np.asarray(group)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("prediction and truth must be aligned 2-D arrays")
    if group.shape != (prediction.shape[0],) or not np.isin(group, np.arange(n_groups)).all():
        raise ValueError("history-only groups must align with forecast windows")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("prediction and truth must be finite")
    return prediction, truth, group.astype(int)


def _gain(reference, corrected, truth):
    before = reference - truth
    after = corrected - truth
    return {
        "mae_gain_kw": float(np.abs(before).mean() - np.abs(after).mean()),
        "rmse_gain_kw": float(np.sqrt(np.mean(before ** 2)) - np.sqrt(np.mean(after ** 2))),
        "bias_before_kw": float(before.mean()),
        "bias_after_kw": float(after.mean()),
    }


def fit_conditional_bias(blocks, *, n_groups=20, rated_power=1500.0,
                         max_capacity_ratio=0.05, min_windows=64):
    """Fit block-1 cell medians; block 2 can only accept or reject them."""
    if len(blocks) != 2:
        raise ValueError("Exactly two chronological training blocks required")
    if n_groups < 1 or rated_power <= 0 or not 0 < max_capacity_ratio <= 1 or min_windows < 1:
        raise ValueError("Invalid group count, physical bound, or support")
    prepared = [_validate(*block, n_groups) for block in blocks]
    horizon = prepared[0][0].shape[1]
    if prepared[1][0].shape[1] != horizon:
        raise ValueError("Training blocks must share a forecast horizon")
    cap_kw = float(rated_power) * float(max_capacity_ratio)
    corrections = np.zeros((horizon, n_groups), dtype=float)
    accepted = np.zeros((horizon, n_groups), dtype=bool)
    evidence = []
    fit_prediction, fit_truth, fit_group = prepared[0]
    for step in range(horizon):
        for group in range(n_groups):
            fit_mask = fit_group == group
            proposed = 0.0
            if int(fit_mask.sum()) >= min_windows:
                proposed = float(np.clip(
                    np.median(fit_truth[fit_mask, step] - fit_prediction[fit_mask, step]),
                    -cap_kw,
                    cap_kw,
                ))
            rows = []
            for prediction, truth, groups in prepared:
                mask = groups == group
                count = int(mask.sum())
                row = {"windows": count, "mae_gain_kw": None, "rmse_gain_kw": None,
                       "bias_before_kw": None, "bias_after_kw": None}
                if count:
                    corrected = np.clip(prediction[mask, step] + proposed, 0.0, float(rated_power))
                    row.update(_gain(prediction[mask, step], corrected, truth[mask, step]))
                rows.append(row)
            keep = (
                abs(proposed) > 0
                and all(row["windows"] >= min_windows for row in rows)
                and all(row["mae_gain_kw"] > 0 and row["rmse_gain_kw"] > 0 for row in rows)
            )
            if keep:
                corrections[step, group] = proposed
                accepted[step, group] = True
            evidence.append({"step": step + 1, "physical_bin": group,
                             "proposed_correction_kw": proposed, "accepted": bool(keep),
                             "blocks": rows})
    return {
        "schema_version": 1,
        "rated_power_kw": float(rated_power),
        "n_groups": int(n_groups),
        "max_capacity_ratio": float(max_capacity_ratio),
        "max_abs_correction_kw": cap_kw,
        "min_windows": int(min_windows),
        "correction_kw": corrections.tolist(),
        "accepted": accepted.tolist(),
        "evidence": evidence,
        "group_definition": "frozen history-only mean-power x mean-wind joint bins",
        "magnitude_source": "first chronological training block cell median residual only",
        "acceptance_source": "first and second chronological training blocks; strict MAE and RMSE improvement",
    }


def apply_conditional_bias(model, prediction, group):
    """Apply history-cell corrections; targets are deliberately not accepted."""
    prediction = np.asarray(prediction, dtype=float)
    group = np.asarray(group)
    corrections = np.asarray(model["correction_kw"], dtype=float)
    if prediction.ndim != 2 or corrections.shape[0] != prediction.shape[1]:
        raise ValueError("Prediction horizon and correction table do not align")
    if group.shape != (prediction.shape[0],) or not np.isin(group, np.arange(corrections.shape[1])).all():
        raise ValueError("History-only groups do not align")
    delta = corrections[np.arange(prediction.shape[1])[None, :], group.astype(int)[:, None]]
    return np.clip(prediction + delta, 0.0, float(model["rated_power_kw"]))
