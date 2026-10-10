"""One bounded train-side hypothesis: history-conditioned residual damping."""
from __future__ import annotations

import numpy as np


SCALES = (0., .25, .5, .75, 1.)
MIN_WINDOWS = 64


def chronological_masks(timestamps, start, end):
    times = np.asarray(timestamps, dtype="datetime64[ns]")
    start, end = np.datetime64(start, "ns"), np.datetime64(end, "ns")
    if (times.ndim != 2 or times.shape[1] != 12 or np.isnat(times).any()
            or not start < end or not (times >= start).all() or not (times < end).all()
            or not (np.diff(times, axis=1) == np.timedelta64(10, "m")).all()):
        raise ValueError("Complete h12 train-OOF timestamps are required")
    midpoint = start + (end - start) // 2
    fit, check = times[:, -1] < midpoint, times[:, 0] >= midpoint
    if not fit.any() or not check.any() or (fit & check).any():
        raise ValueError("Two non-overlapping chronological train-period blocks required")
    return fit, check, midpoint


def fit_scale(reference, persistence, truth, trend):
    """Fit only the earlier block. No threshold/seed/epoch search on the check block.

    Choose among five predeclared scales in each of three history bins ×12
    steps, requiring both fit MAE/RMSE no worse than the unchanged predictor.
    There is no intercept, residual amplification or future-derived state.
    """
    reference, persistence, truth = (np.asarray(value, float) for value in (reference, persistence, truth))
    trend = np.asarray(trend)
    if (reference.ndim != 2 or len(reference) == 0 or reference.shape[1] != 12 or persistence.shape != reference.shape
            or truth.shape != reference.shape or trend.shape != (len(reference),)
            or not np.isin(trend, [-1, 0, 1]).all()
            or not all(np.isfinite(value).all() for value in (reference, persistence, truth))):
        raise ValueError("Paired finite h12 forecasts and fixed history bins required")
    model = {"schema_version": 1, "scales": list(SCALES), "min_fit_windows": MIN_WINDOWS,
             "state_source": "history-only fixed 5%-capacity change bins", "states": []}
    for state in (-1, 0, 1):
        selected = trend == state
        count = int(selected.sum())
        weights, cells = [], []
        for step in range(12):
            base = reference[selected, step]
            anchor = persistence[selected, step]
            target = truth[selected, step]
            weight, feasible = 1., []
            if count >= MIN_WINDOWS:
                base_mae = float(np.abs(base - target).mean())
                base_rmse = float(np.sqrt(np.square(base - target).mean()))
                for scale in SCALES:
                    prediction = anchor + scale * (base - anchor)
                    mae = float(np.abs(prediction - target).mean())
                    rmse = float(np.sqrt(np.square(prediction - target).mean()))
                    if mae <= base_mae + 1e-10 and rmse <= base_rmse + 1e-10:
                        score = .5 * (mae / max(base_mae, 1e-12) + rmse / max(base_rmse, 1e-12))
                        feasible.append(dict(scale=scale, mae_kw=mae, rmse_kw=rmse, balanced_ratio=score))
                # Equal scores preserve the original predictor, not fake activity.
                chosen = min(feasible, key=lambda row: (row["balanced_ratio"], -row["scale"]))
                weight = chosen["scale"]
            weights.append(weight)
            cells.append(dict(step=step + 1, fit_windows=count, feasible_candidates=feasible,
                              fallback_insufficient_support=count < MIN_WINDOWS))
        model["states"].append(dict(state=state, weights=weights, cells=cells))
    return model


def apply_scale(model, reference, persistence, trend):
    reference, persistence, trend = np.asarray(reference, float), np.asarray(persistence, float), np.asarray(trend)
    if (reference.ndim != 2 or reference.shape[1] != 12 or persistence.shape != reference.shape
            or not np.isfinite(reference).all() or not np.isfinite(persistence).all()
            or trend.shape != (len(reference),) or not np.isin(trend, [-1, 0, 1]).all()):
        raise ValueError("Forecast/state shapes differ from the calibrated h12 contract")
    states = model["states"]
    if [row["state"] for row in states] != [-1, 0, 1]:
        raise ValueError("Calibration state ordering changed")
    weights = np.asarray([row["weights"] for row in states], float)
    if weights.shape != (3, 12) or not np.isfinite(weights).all() or not ((weights >= 0) & (weights <= 1)).all():
        raise ValueError("Residual scale must stay in [0,1], without amplification")
    return persistence + weights[trend.astype(int) + 1] * (reference - persistence)
