"""Validation-only attribution of a fitted ramp correction.

This compares a checkpoint with its own ramp term removed at inference. It is
not a comparison against a separately trained pure-trend checkpoint and must
not be used to fit or select the gate on the validation labels.
"""

import numpy as np


def summarize_ramp_counterfactual(prediction, without_ramp, truth, eligible, effective):
    prediction = np.asarray(prediction, dtype=np.float64)
    without_ramp = np.asarray(without_ramp, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool).reshape(-1)
    effective = np.asarray(effective, dtype=bool).reshape(-1)
    if prediction.ndim != 2 or prediction.shape != without_ramp.shape or prediction.shape != truth.shape:
        raise ValueError("Ramp counterfactual arrays must have matching [window, horizon] shapes")
    if not prediction.size:
        raise ValueError("Ramp counterfactual arrays must be nonempty")
    if eligible.size != prediction.shape[0] or effective.size != prediction.shape[0]:
        raise ValueError("Ramp masks must have one value per forecast window")
    if not all(np.isfinite(array).all() for array in (prediction, without_ramp, truth)):
        raise ValueError("Ramp counterfactual arrays must be finite")
    if np.any(effective & ~eligible):
        raise ValueError("Effective correction cannot occur outside eligible windows")

    gain = np.abs(without_ramp - truth) - np.abs(prediction - truth)
    report = {
        "windows": int(prediction.shape[0]),
        "horizon": int(prediction.shape[1]),
        "eligible_windows": int(eligible.sum()),
        "eligible_window_pct": float(100.0 * eligible.mean()),
        "effective_windows": int(effective.sum()),
        "effective_window_pct": float(100.0 * effective.mean()),
        "with_ramp_mae_kw": float(np.abs(prediction - truth).mean()),
        "without_ramp_mae_kw": float(np.abs(without_ramp - truth).mean()),
        "gain_from_ramp_kw": float(gain.mean()),
        "gain_by_horizon_kw": gain.mean(axis=0).tolist(),
    }
    if effective.any():
        selected_gain = gain[effective]
        report.update({
            "effective_gain_kw": float(selected_gain.mean()),
            "effective_harm_point_pct": float(100.0 * (selected_gain < 0).mean()),
            "effective_harm_window_pct": float(
                100.0 * (selected_gain.mean(axis=1) < 0).mean()
            ),
        })
    else:
        report.update({
            "effective_gain_kw": None,
            "effective_harm_point_pct": None,
            "effective_harm_window_pct": None,
        })
    return report
