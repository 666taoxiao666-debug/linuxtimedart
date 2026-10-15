"""Optional frozen residuals: train-only shrinkage and temporal benefit checks.

These checks are empirical deployment rules, NOT statistical coverage guarantees.
Inference reads predictions, clocks and a frozen identity, never target values.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
from utils.wiki_evidence_support_audit import clock_support


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def array_digest(pack):
    h = hashlib.sha256()
    def add(key, value):
        if isinstance(value, dict):
            for child in sorted(value):
                add(key + "/" + child, value[child])
        else:
            array = np.ascontiguousarray(value)
            h.update(json.dumps([key, array.dtype.str, array.shape]).encode())
            h.update(array.tobytes())
    for key in sorted(pack):
        add(key, pack[key])
    return h.hexdigest()


def subset(pack, mask):
    return {key: ({name: array[mask].copy() for name, array in value.items()}
                  if isinstance(value, dict) else value[mask].copy()) for key, value in pack.items()}


def chronological_parts(pack, count, *, train_end):
    """Split distinct target-start clocks, never turbine rows; purge crossings."""
    clock_support(pack["times"], pack["turbines"], train_end=train_end)
    times = pack["times"].astype("datetime64[ns]")
    starts = np.unique(times[:, 0])
    if len(starts) < count:
        raise ValueError("Insufficient distinct training clocks")
    groups = np.array_split(starts, count)
    cuts = [group[0] for group in groups[1:]]
    bounds = [starts[0], *cuts, np.datetime64(train_end, "ns")]
    masks = [(times[:, 0] >= lo) & (times[:, -1] < hi) for lo, hi in zip(bounds, bounds[1:])]
    if any(not mask.any() for mask in masks):
        raise ValueError("Empty purged training partition")
    parts = [subset(pack, mask) for mask in masks]
    for first, second in zip(parts, parts[1:]):
        if first["times"].max() > (second["times"][:, 0] - np.timedelta64(1, "h")).min():
            raise ValueError("Calibration outcomes unavailable at next partition issue")
    return parts, dict(cuts=[str(value) for value in cuts],
                      windows=[int(mask.sum()) for mask in masks],
                      purged_windows=int((~np.logical_or.reduce(masks)).sum()))


def predict_residual(reference, numeric, weights):
    reference, numeric = np.asarray(reference, float), np.asarray(numeric, float)
    weights = np.asarray(weights, float)
    if (reference.ndim != 2 or reference.shape != numeric.shape or weights.shape != (reference.shape[1],)
            or not all(np.isfinite(value).all() for value in (reference, numeric, weights))
            or np.any((weights < 0) | (weights > 1))):
        raise ValueError("Residual prediction shape/range mismatch")
    result = reference + weights * (numeric - reference)
    # Exact fallbacks, not approximate cancellation in floating point.
    result[:, weights == 0] = reference[:, weights == 0]
    result[:, weights == 1] = numeric[:, weights == 1]
    return result


def fit_shrinkage(pack, *, train_end):
    """Fit two convex coefficients on old training selection only, no grid search.

    Minimize mean(MAE/MAE0 + MSE/MSE0)/2 on the jointly non-worse
    interval [0,1]. Bisections are numerical solvers, not searched thresholds.
    """
    support = clock_support(pack["times"], pack["turbines"], train_end=train_end)
    reference, numeric, truth = (np.asarray(pack[key], float) for key in ("reference", "numeric", "truth"))
    if (reference.shape != numeric.shape or reference.shape != truth.shape
            or not all(np.isfinite(value).all() for value in (reference, numeric, truth))):
        raise ValueError("Shrinkage labels/predictions misaligned")
    result = []
    for h in range(truth.shape[1]):
        error, delta = reference[:, h] - truth[:, h], numeric[:, h] - reference[:, h]
        mae0, mse0 = np.abs(error).mean(), np.square(error).mean()
        if mae0 == 0 or mse0 == 0 or not np.any(delta):
            result.append(0.)
            continue
        # Each convex loss has a feasible interval containing zero.
        upper = 1.
        for loss in (lambda a: np.abs(error + a * delta).mean() - mae0,
                     lambda a: np.square(error + a * delta).mean() - mse0):
            if loss(upper) > 0:
                lo, hi = 0., upper
                for _ in range(80):
                    mid = (lo + hi) / 2
                    if loss(mid) <= 0:
                        lo = mid
                    else:
                        hi = mid
                upper = lo
        lo, hi = 0., upper
        for _ in range(80):
            mid = (lo + hi) / 2
            e = error + mid * delta
            derivative = .5 * ((np.sign(e) * delta).mean() / mae0 + 2 * (e * delta).mean() / mse0)
            if derivative < 0:
                lo = mid
            else:
                hi = mid
        choices = [0., (lo + hi) / 2, upper]
        score = lambda a: .5 * (np.abs(error + a * delta).mean() / mae0 + np.square(error + a * delta).mean() / mse0)
        result.append(float(min(choices, key=score)))
    return dict(alpha=result, support=support, fit_arrays_sha256=array_digest(pack))


def certify(pack, shrinkage, execution, settings, *, train_end):
    """Keep a lead only if every fixed training block passes the same rule."""
    clock_support(pack["times"], pack["turbines"], train_end=train_end)
    parts, partition = chronological_parts(pack, settings["temporal_blocks"], train_end=train_end)
    rows = []
    for part in parts:
        prediction = predict_residual(part["reference"], part["numeric"], shrinkage["alpha"])
        gain = np.abs(part["reference"] - part["truth"]) - np.abs(prediction - part["truth"])
        squared = (part["reference"] - part["truth"])**2 - (prediction - part["truth"])**2
        ids = np.unique(part["turbines"])
        macro = np.array([gain[part["turbines"] == unit].mean(0) for unit in ids])
        mse = np.array([squared[part["turbines"] == unit].mean(0) for unit in ids])
        support = clock_support(part["times"], part["turbines"], train_end=train_end)
        enough = (support["windows"] >= settings["min_windows"] and support["turbines"] >= settings["min_turbines"]
                  and support["unique_issue_times"] >= settings["min_issue_times"])
        accepted = enough & (macro.mean(0) > 0) & (mse.mean(0) > 0) & ((macro > 0).mean(0) >= settings["positive_fraction"])
        rows.append(dict(support=support, accepted=accepted.tolist(), macro_gain_kw=macro.mean(0).tolist(),
                         macro_mse_gain=mse.mean(0).tolist(), positive_turbine_fraction=(macro > 0).mean(0).tolist()))
    accepted = np.logical_and.reduce([row["accepted"] for row in rows])
    policy = dict(execution=execution, shrinkage=shrinkage, settings=settings, temporal_partition=partition,
                  blocks=rows, accepted=accepted.tolist(),
                  weights=np.where(accepted, shrinkage["alpha"], 0.).tolist(),
                  calibration_arrays_sha256=array_digest(pack),
                  outcomes_available_ns=int(pack["times"].astype("datetime64[ns]").astype(np.int64).max()),
                  interpretation="Training-period empirical trust rule; correlated blocks, not a distribution-free guarantee")
    policy["gate_id"] = digest(policy)
    return policy


def apply(policy, reference, numeric, times, execution):
    if policy["execution"] != execution or policy["gate_id"] != digest({k: v for k, v in policy.items() if k != "gate_id"}):
        raise ValueError("Residual evidence/executing identity mismatch")
    times = np.asarray(times).astype("datetime64[ns]")
    if times.shape != np.shape(reference) or np.isnat(times).any():
        raise ValueError("Residual inference clock mismatch")
    if (times[:, 0].astype(np.int64) - 3600 * 10**9).min() < policy["outcomes_available_ns"]:
        raise ValueError("Residual calibration outcomes unavailable at issue")
    return predict_residual(reference, numeric, policy["weights"])
