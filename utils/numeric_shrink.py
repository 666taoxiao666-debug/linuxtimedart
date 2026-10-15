"""Chronological, executor-bound joint-loss numerical shrinkage; no Wiki."""
from __future__ import annotations

import hashlib
import numpy as np
from data_provider.kelmarsh_hourly import KelmarshWindows
from utils.numeric_foundation import observed_anchor, ridge_predict


def arrays_digest(arrays):
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[key]))
        digest.update(key.encode())
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def scaler_digest(dataset):
    return arrays_digest(dict(mean=dataset.scaler.mean_, scale=dataset.scaler.scale_))


def nested_partition(series, available_end):
    """90% of the available train prefix fits the expert; last10% calibrates.

    The original scaler was fitted on original outer-train first60%, so the
    earliest calibration must start AFTER that scaler's fitting outcomes.
    """
    start = series.dates.min().astype("datetime64[h]")
    end = np.datetime64(available_end, "h")
    cut = (start + (end - start) * .9).astype("datetime64[h]")
    starts = series.starts
    first = series.dates[starts + series.input_len]
    last = series.dates[starts + series.input_len + series.horizon - 1]
    fit = KelmarshWindows(series, starts[last < cut], "shrink_expert_fit")
    calibration = KelmarshWindows(series, starts[(first >= cut) & (last < end)], "shrink_calibration")
    if not len(fit) or not len(calibration) or end > series.bounds["train_end"]:
        raise ValueError("Invalid nested training interval")
    issue = calibration.target_times()[:, 0].min() - np.timedelta64(1, "h")
    if fit.target_times().max() > issue or series.bounds["base_fit_end"] > issue:
        raise ValueError("Expert/scaler uses outcomes unavailable at first calibration issue")
    fit.flag, calibration.flag = "train", "val"
    fit.train_cutoff = calibration.train_cutoff = cut
    fit.val_cutoff = calibration.val_cutoff = end
    return fit, calibration, dict(expert_cut=str(cut), available_end=str(end),
                                 calibration_first_issue=str(issue), original_scaler_causal=True)


def joint_coefficients(delta, innovation):
    """Minimize MSE on [0,1], constrained by MAE <= exact Persistence.

    Two independent convex one-dimensional problems (one per lead), not a
    threshold/grid search. If the quadratic optimum violates the MAE bound,
    bisect the feasible interval containing zero. All inputs must be calibration
    predictions from the exact already-frozen expert and calibration labels.
    """
    d, y = np.asarray(delta, float), np.asarray(innovation, float)
    if d.ndim != 2 or d.shape != y.shape or not len(d) or not np.isfinite(d).all() or not np.isfinite(y).all():
        raise ValueError("Finite paired calibration delta/innovation required")
    result = []
    for lead in range(d.shape[1]):
        dd, yy = d[:, lead], y[:, lead]
        norm = float(np.sum(dd * dd))
        if not np.isfinite(norm):
            raise ValueError("Nonfinite calibration quadratic norm")
        alpha = float(np.clip(np.sum(dd * yy) / norm, 0., 1.)) if norm > 0 else 0.
        baseline = float(np.mean(np.abs(yy)))
        tolerance = 1e-12 * max(1., baseline)
        if np.mean(np.abs(yy - alpha * dd)) > baseline + tolerance:
            lo, hi = 0., alpha
            for _ in range(64):
                mid = (lo + hi) / 2
                if np.mean(np.abs(yy - mid * dd)) <= baseline + tolerance:
                    lo = mid
                else:
                    hi = mid
            alpha = lo
        # Floating-scale values are treated as exact abstention, not evidence.
        result.append(0. if alpha < 1e-10 else alpha)
    return np.asarray(result)


def make_certificate(model, calibration, checkpoint_sha256):
    prediction = ridge_predict(model, calibration)
    truth, anchor = calibration.truth_and_anchor()
    delta, innovation = prediction - anchor, truth - anchor
    alpha = joint_coefficients(delta, innovation)
    return dict(expert_checkpoint_sha256=checkpoint_sha256, expert_array_sha256=arrays_digest(model),
                scaler_sha256=scaler_digest(calibration), coefficients=alpha.tolist(),
                calibration_arrays_sha256=arrays_digest(dict(prediction=prediction, truth=truth, anchor=anchor,
                    times=calibration.target_times(), starts=calibration.window_starts)),
                calibration_windows=len(calibration), first_target=str(calibration.target_times().min()),
                last_target=str(calibration.target_times().max()),
                calibration_mae_before=np.mean(np.abs(innovation - delta), axis=0).tolist(),
                calibration_mae_after=np.mean(np.abs(innovation - delta * alpha), axis=0).tolist(),
                calibration_persistence_mae=np.mean(np.abs(innovation), axis=0).tolist(),
                calibration_mse_after=np.mean((innovation - delta * alpha) ** 2, axis=0).tolist(),
                calibration_persistence_mse=np.mean(innovation ** 2, axis=0).tolist(),
                expert_refitted_after_calibration=False, wiki_enabled=False)


def predict_bound(model, dataset, certificate, checkpoint_sha256):
    if (certificate["expert_checkpoint_sha256"] != checkpoint_sha256 or
            certificate["expert_array_sha256"] != arrays_digest(model) or
            certificate["scaler_sha256"] != scaler_digest(dataset) or
            certificate["expert_refitted_after_calibration"]):
        raise ValueError("Shrinkage certificate/executing expert/scaler mismatch")
    alpha = np.asarray(certificate["coefficients"], float)
    if alpha.shape != (dataset.pred_len,) or not np.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1).any():
        raise ValueError("Invalid bound shrinkage coefficients")
    anchor = observed_anchor(dataset)
    return anchor + alpha * (ridge_predict(model, dataset) - anchor)
