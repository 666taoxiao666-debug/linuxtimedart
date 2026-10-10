"""Training-block-consistent physical support for an existing residual expert."""
from __future__ import annotations
import numpy as np
from utils.residual_scale_calibration import apply_scale, chronological_masks


def three_blocks(times, start, end):
    early, _, midpoint = chronological_masks(times, start, end)
    times = np.asarray(times, dtype="datetime64[ns]")
    third_start = midpoint + (np.datetime64(end, "ns") - midpoint) // 2
    calibration = (times[:, 0] >= midpoint) & (times[:, -1] < third_start)
    check = times[:, 0] >= third_start
    if not calibration.any() or not check.any() or (early & calibration).any() or (calibration & check).any():
        raise ValueError("Three disjoint supported chronological blocks required")
    return early, calibration, check, midpoint, third_start


def fit_guard(model, blocks, *, min_windows=64):
    """Only the first two training blocks enter fitting; never accept a check set."""
    if len(blocks) != 2:
        raise ValueError("Exactly two training blocks required")
    trust = np.zeros((3, 12, 20), bool)
    rows, prepared = [], []
    for arrays, bins in blocks:
        ref, anchor, truth, state = (np.asarray(arrays[k]) for k in ("prediction", "persistence", "truth", "trend"))
        bins = np.asarray(bins)
        if (truth.shape != ref.shape or not np.isfinite(truth).all()
                or bins.shape != state.shape or not np.isin(bins, np.arange(20)).all()):
            raise ValueError("Training labels and physical strata must align")
        candidate = apply_scale(model, ref, anchor, state)
        prepared.append((state, bins, np.abs(ref - truth) - np.abs(candidate - truth),
                         (ref - truth) ** 2 - (candidate - truth) ** 2))
    for state_row in model["states"]:
        state = state_row["state"]
        for step, scale in enumerate(state_row["weights"]):
            if scale == 1:
                continue
            for group in range(20):
                evidence = []
                for states, bins, absolute, squared in prepared:
                    mask = (states == state) & (bins == group)
                    count = int(mask.sum())
                    evidence.append(dict(windows=count, mae_gain_kw=float(absolute[mask, step].mean()) if count else None,
                        mse_gain_kw2=float(squared[mask, step].mean()) if count else None))
                accepted = all(r["windows"] >= min_windows and r["mae_gain_kw"] > 0 and r["mse_gain_kw2"] > 0 for r in evidence)
                trust[state + 1, step, group] = accepted
                rows.append(dict(state=state, step=step + 1, physical_bin=group, accepted=accepted, blocks=evidence))
    return dict(schema_version=1, min_windows=min_windows, frozen_scale_model=model,
        trust=trust.tolist(), evidence=rows, fitting_source="first two original-train chronological blocks only")


def apply_guard(guard, reference, persistence, trend, physical_bin):
    """History bin + original predictions only; no target or timestamp argument."""
    trend, physical_bin = np.asarray(trend), np.asarray(physical_bin)
    trust = np.asarray(guard["trust"])
    if (trust.shape != (3, 12, 20) or not np.isin(trust, [False, True]).all()
            or physical_bin.shape != trend.shape or not np.isin(physical_bin, np.arange(20)).all()):
        raise ValueError("Invalid physical guard shape or bins")
    candidate = apply_scale(guard["frozen_scale_model"], reference, persistence, trend)
    enabled = trust[trend.astype(int)[:, None] + 1, np.arange(12), physical_bin.astype(int)[:, None]]
    return np.where(enabled, candidate, np.asarray(reference, float))
