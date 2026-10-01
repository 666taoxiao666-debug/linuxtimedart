"""Train-only forward-OOF boundaries and event-utility summaries.

The producer trains a *separate* model before the evidence interval.  These
helpers never label in-sample predictions as out of fold.
"""

from __future__ import annotations

import numpy as np


def inner_ratios(outer_train_ratio: float, outer_val_ratio: float,
                 fold: int, n_folds: int) -> tuple[float, float]:
    """Reserve the last 10% of the outer training period for OOF evidence."""
    if n_folds < 1 or not 0 <= fold < n_folds:
        raise ValueError("Invalid outer fold")
    if not 0 < outer_train_ratio < 1 or not 0 < outer_val_ratio < 1:
        raise ValueError("Invalid outer split ratios")
    # rolling_holdout uses expanding training edges within the selection band.
    outer_fraction = outer_train_ratio + fold * outer_val_ratio / n_folds
    inner_train, inner_val = outer_fraction * 0.8, outer_fraction * 0.1
    if not 0 < inner_train < inner_train + inner_val < outer_fraction < 1:
        raise ValueError("No room for a forward OOF interval")
    return inner_train, inner_val


def forward_oof_starts(segments, dates, seq_len: int, pred_len: int,
                       stride: int, evidence_start, outer_train_cutoff) -> np.ndarray:
    """Select complete within-turbine windows, strictly before outer validation."""
    dates = np.asarray(dates, dtype="datetime64[ns]")
    start_time = np.datetime64(evidence_start, "ns")
    end_time = np.datetime64(outer_train_cutoff, "ns")
    if not start_time < end_time:
        raise ValueError("OOF evidence must precede the outer validation boundary")
    if seq_len < 1 or pred_len < 1 or stride < 1:
        raise ValueError("Window lengths and stride must be positive")
    total = seq_len + pred_len
    selected = []
    for first, last in np.asarray(segments, dtype=np.int64):
        if last - first < total:
            continue
        starts = np.arange(first, last - total + 1, stride, dtype=np.int64)
        first_target = dates[starts + seq_len]
        last_target = dates[starts + total - 1]
        keep = (first_target >= start_time) & (last_target < end_time)
        if keep.any():
            selected.append(starts[keep])
    if not selected:
        raise ValueError("No complete forward-OOF windows inside outer training")
    return np.concatenate(selected)


def summarize_event_evidence(base, candidates, target, available, turbines,
                             scenes, *, fold: int, seed: int, as_of_step: int) -> dict:
    """Aggregate paired candidate-minus-reference gain by turbine and horizon.

    The source model's OOF status is verified by the caller.  A turbine is one
    transfer unit; window-level standard deviations are recorded for audit but
    are not treated as independent transfer replications.
    """
    base = np.asarray(base, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    turbines = np.asarray(turbines)
    if base.ndim != 2 or target.shape != base.shape:
        raise ValueError("base and target must have [windows, horizon] shape")
    count, horizon = base.shape
    factors = len(scenes)
    if (candidates.shape != (count, horizon, factors)
            or available.shape != candidates.shape
            or turbines.shape != (count,)):
        raise ValueError("Candidate, availability, or turbine shapes do not align")
    if count == 0 or not np.isfinite(base).all() or not np.isfinite(target).all():
        raise ValueError("OOF predictions and targets must be nonempty and finite")
    if not np.isfinite(candidates).all():
        raise ValueError("Candidate predictions must be finite")
    if as_of_step < 0:
        raise ValueError("as_of_step must be nonnegative")

    base_error = np.abs(base - target)
    candidate_error = np.abs(candidates - target[:, :, None])
    evidence = []
    source_turbines = set()
    for factor_index, scene in enumerate(scenes):
        rows = []
        for turbine in np.unique(turbines):
            turbine_windows = turbines == turbine
            mask = available[turbine_windows, :, factor_index]
            active_window = mask.any(axis=1)
            if int(active_window.sum()) < 2:
                continue
            reference = base_error[turbine_windows]
            proposal = candidate_error[turbine_windows, :, factor_index]
            reference_total = reference[mask].mean()
            if reference_total <= 0:
                # A perfect reference has no positive correction utility.
                macro_values = np.zeros(int(active_window.sum()), dtype=np.float64)
            else:
                delta_window = ((reference - proposal) * mask).sum(axis=1)
                delta_window = delta_window[active_window] / mask.sum(axis=1)[active_window]
                macro_values = delta_window / reference_total
            horizon_mean, horizon_std = [], []
            for step in range(horizon):
                supported = mask[:, step]
                if int(supported.sum()) < 2:
                    horizon_mean.append(0.0)
                    horizon_std.append(0.0)
                    continue
                step_reference = reference[supported, step]
                scale = step_reference.mean()
                values = ((step_reference - proposal[supported, step]) / scale
                          if scale > 0 else np.zeros(int(supported.sum())))
                horizon_mean.append(float(values.mean()))
                horizon_std.append(float(values.std(ddof=1)))
            row = {
                "source_split": "train_oof",
                "turbine_id": int(turbine),
                "n_windows": int(active_window.sum()),
                "mean_utility": float(macro_values.mean()),
                "std_utility": float(macro_values.std(ddof=1)),
                "horizon_mean_utility": horizon_mean,
                "horizon_std_utility": horizon_std,
            }
            rows.append(row)
            source_turbines.add(int(turbine))
        if rows:
            evidence.append({
                "id": f"forward_oof_{scene['id']}_f{fold}_s{seed}",
                "factor_id": scene["id"],
                "title": scene.get("title", scene["id"]),
                "prompt": scene["prompt"],
                "created_step": int(as_of_step),
                "last_evidence_step": int(as_of_step),
                "turbine_evidence": rows,
            })
    if not evidence:
        raise ValueError("No event has at least two supported OOF windows per turbine")
    return {
        "schema_version": 1,
        "source_split": "train_oof",
        "utility_definition": "paired absolute-error reduction divided by mean reference absolute error",
        "sdwpf_fold": int(fold),
        "pred_len": int(horizon),
        "as_of_step": int(as_of_step),
        "source_turbines": sorted(source_turbines),
        "held_out_turbines": [],
        "candidates": evidence,
    }
