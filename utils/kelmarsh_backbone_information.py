"""Torch-free helpers for the v25 Kelmarsh numeric-backbone information experiment.

Everything here is a pure function of explicit inputs (records, arrays, hashes), so gates, statistics,
alignment and device decisions can be unit tested without data, GPUs or a trained model.
Gate decisions are budget-allocation rules on development evidence, not significance tests.
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils.paired_forecast_statistics import moving_block_bootstrap_ratio, newey_west_mean_test

PRIMARY = "TSMixerFull"
INFO_CONTROL = "TSMixerPowerOnly"
BASELINE = "DLinear"
TSMIXER_ARMS = (PRIMARY, INFO_CONTROL)
ALL_ARMS = (PRIMARY, INFO_CONTROL, BASELINE)
METRIC_KEYS = ("mae", "rmse")
TIE_TOLERANCE_KW = 1e-6
STAGE_SUBSETS = {"A0": {"fit", "selection"}, "A1": {"fit", "selection"}, "A2": {"fit", "selection"},
                 "A3": {"fit", "selection", "validation"}}


# --------------------------------------------------------------------------- serialisation / hashing
def safe(value):
    """JSON-safe conversion; non-finite floats become None (never silently 0)."""
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.datetime64, pd.Timestamp, _dt.datetime)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    """Atomic JSON write (temporary file then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(safe(value), indent=2, allow_nan=False))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def text_sha256_lf(path):
    """Checkout-independent hash: Windows clones may be CRLF, the training host LF."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_registered_files(directory, hashes):
    """Fail closed on missing/changed registered artifacts, including completed resumes."""
    directory = Path(directory).resolve()
    for name, digest in hashes.items():
        path = (directory / name).resolve()
        if directory not in path.parents or not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Registered artifact changed or missing: {name}")


def verify_training_completed(directory):
    record = read_json(Path(directory) / "training_completed.json")
    verify_registered_files(directory, record["hashes"])
    return record


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(safe(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def array_sha256(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def window_hash(starts):
    """Identical to the v19-v24 contract hash: raw int64 bytes of window starts."""
    return hashlib.sha256(np.asarray(starts).tobytes()).hexdigest()


def append_event(path, event, **fields):
    """Append-only JSON-lines event log; never rewritten."""
    record = {"utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "event": event, **safe(fields)}
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def contract_mismatches(saved, expected):
    keys = sorted(set(saved) | set(expected))
    return [key for key in keys if canonical_sha256(saved.get(key)) != canonical_sha256(expected.get(key))]


def acquire_exclusive_lock(path):
    """Non-blocking exclusive flock. Raises BlockingIOError when another controller holds it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise
    return handle


# --------------------------------------------------------------------------- protocol self-consistency
DESIGN_VALUES = {
    ("input_len",): 24, ("pred_len",): 4, ("train_stride",): 3, ("eval_stride",): 3, ("folds",): [0, 1, 2],
    ("rated_power_kw",): 2050.0, ("target_slot",): 6,
    ("feature_columns",): ["Wspd", "Etmp", "Ndir_sin", "Ndir_cos", "Pab_mean", "rotor_speed", "power"],
    ("candidate", "blocks"): 2, ("candidate", "feature_hidden_dim"): 32, ("candidate", "norm_shape"): [24, 7],
    ("candidate", "norm_eps"): 0.001, ("candidate", "dropout"): 0.1, ("candidate", "expected_trainable_parameters"): 3618,
    ("candidate", "revin"): False, ("candidate", "pretraining"): False, ("candidate", "prompt"): False,
    ("candidate", "wiki"): False, ("candidate", "persistence_residual_anchor"): False,
    ("training", "optimizer"): "AdamW", ("training", "learning_rate"): 0.001, ("training", "weight_decay"): 0.0001,
    ("training", "batch_size"): 64, ("training", "eval_batch_size"): 128, ("training", "max_epochs"): 20,
    ("training", "patience"): 5, ("training", "grad_clip_norm"): 1.0, ("training", "scheduler_t_max"): 20,
    ("training", "scheduler_eta_min"): 0.0, ("training", "loss_mae_weight"): 0.8, ("training", "loss_mse_weight"): 0.2,
    ("training", "amp"): False, ("training", "num_workers"): 0, ("training", "refit_after_selection"): False,
    ("budget", "new_fits_max"): 24, ("budget", "new_epochs_max"): 480, ("seeds", "A1"): [2024],
    ("seeds", "A2"): [2025, 2026], ("gates", "numeric_tie_tolerance_kw"): 1e-6, ("wiki_enabled",): False,
    ("sealed_test_access",): False, ("statistics", "block_length"): 8, ("statistics", "replicates"): 5000,
    ("statistics", "seed"): 2024, ("statistics", "estimand"): "window_uniform", ("statistics", "hac_max_lag"): 7,
}


def validate_protocol(protocol):
    """Refuse an executable protocol that silently departs from the delivered design values."""
    for path, expected in DESIGN_VALUES.items():
        value = protocol
        for key in path:
            value = value[key]
        if value != expected:
            raise ValueError(f"Protocol value {'.'.join(path)}={value!r} departs from design {expected!r}")
    arms = protocol["arms"]
    if tuple(arms) != ALL_ARMS:
        raise ValueError("Arms must be exactly TSMixerFull, TSMixerPowerOnly, DLinear")
    if arms[PRIMARY]["input_mode"] != "full" or arms[INFO_CONTROL]["input_mode"] != "power_only":
        raise ValueError("PowerOnly must be the masked information ablation of the Full arm")
    if protocol["gates"]["primary_candidate"] != PRIMARY:
        raise ValueError("Primary candidate is fixed to TSMixerFull")
    if protocol["statistics"]["hac_max_lag"] != protocol["statistics"]["block_length"] - 1:
        raise ValueError("HAC lag must equal block_length-1 as in utils.paired_forecast_statistics")
    budget = planned_new_fits(protocol)
    if budget["max_new_fits"] > protocol["budget"]["new_fits_max"]:
        raise ValueError("Declared matrix exceeds the 24-fit budget")
    if budget["max_new_fits"] * protocol["training"]["max_epochs"] > protocol["budget"]["new_epochs_max"]:
        raise ValueError("Declared matrix exceeds the 480-epoch budget")
    return True


def planned_new_fits(protocol):
    folds = len(protocol["folds"])
    a1 = folds * len(protocol["seeds"]["A1"]) * len(TSMIXER_ARMS)
    a2 = folds * len(protocol["seeds"]["A2"]) * len(ALL_ARMS)
    return {"A1": a1, "A2_max": a2, "max_new_fits": a1 + a2}


# --------------------------------------------------------------------------- gates
def _finite(value, name):
    if value is None or not np.isfinite(float(value)):
        raise ValueError(f"Missing or non-finite metric: {name}")
    return float(value)


def improves(reference, candidate, tolerance=TIE_TOLERANCE_KW):
    """reference_error - candidate_error > tolerance; |difference| <= tolerance is a numeric tie."""
    return _finite(reference, "reference") - _finite(candidate, "candidate") > float(tolerance)


def joint_improvement(reference, candidate, tolerance=TIE_TOLERANCE_KW):
    return all(improves(reference[key], candidate[key], tolerance) for key in METRIC_KEYS)


def _index(records):
    index = {}
    for record in records:
        key = (int(record["fold"]), int(record["seed"]), str(record["arm"]))
        if key in index:
            raise ValueError(f"Duplicate inner record {key}")
        index[key] = {metric: record[metric] for metric in METRIC_KEYS}
    return index


def _mean(values):
    return float(np.mean([_finite(v, "aggregate") for v in values]))


def _missing(index, folds, seeds, arms):
    return [list(key) for key in ((f, s, a) for f in folds for s in seeds for a in arms) if key not in index]


def _persistence(persistence, folds):
    return {f: {key: _finite(persistence[f][key], f"persistence f{f} {key}") for key in METRIC_KEYS} for f in folds}


def gate_g1(records, persistence, settings, folds, tolerance=TIE_TOLERANCE_KW):
    """Seed-2024 screen on internal selection: Full beats DLinear and Persistence in macro MAE AND RMSE,
    and jointly beats DLinear in at least ``min_folds`` folds."""
    seed = int(settings["seed"])
    index = _index(records)
    missing = _missing(index, folds, [seed], (PRIMARY, BASELINE))
    if missing:
        return {"gate": "G1", "passed": False, "complete": False, "missing": missing,
                "reason": "incomplete records are a failure, never a pass"}
    pers = _persistence(persistence, folds)
    rows = []
    for f in folds:
        cand, ref = index[(f, seed, PRIMARY)], index[(f, seed, BASELINE)]
        rows.append({"fold": f, "candidate": cand, "dlinear": ref, "persistence": pers[f],
                     "mae_gain_vs_dlinear_kw": ref["mae"] - cand["mae"],
                     "rmse_gain_vs_dlinear_kw": ref["rmse"] - cand["rmse"],
                     "joint_vs_dlinear": joint_improvement(ref, cand, tolerance)})
    macro = {name: {k: _mean([r[name][k] for r in rows]) for k in METRIC_KEYS}
             for name in ("candidate", "dlinear", "persistence")}
    joint_folds = int(sum(r["joint_vs_dlinear"] for r in rows))
    checks = {"macro_mae_rmse_better_than_dlinear": joint_improvement(macro["dlinear"], macro["candidate"], tolerance),
              "macro_mae_rmse_better_than_persistence": joint_improvement(macro["persistence"], macro["candidate"], tolerance),
              "jointly_better_folds_vs_dlinear": joint_folds,
              "min_jointly_better_folds_required": int(settings["min_folds_jointly_better_than_DLinear"]),
              "fold_requirement_met": joint_folds >= int(settings["min_folds_jointly_better_than_DLinear"])}
    passed = (checks["macro_mae_rmse_better_than_dlinear"] and checks["macro_mae_rmse_better_than_persistence"]
              and checks["fold_requirement_met"])
    return {"gate": "G1", "passed": bool(passed), "complete": True, "seed": seed, "rows": rows, "macro": macro,
            "checks": checks, "tolerance_kw": tolerance, "comparison_space": "internal_selection_only",
            "interpretation": "operational budget gate on development evidence; not a significance test"}


def info_i1(records, settings, folds, tolerance=TIE_TOLERANCE_KW):
    seed = int(settings["seed"])
    index = _index(records)
    missing = _missing(index, folds, [seed], (PRIMARY, INFO_CONTROL))
    if missing:
        return {"evidence": "I1", "supported": False, "complete": False, "missing": missing}
    rows = [{"fold": f, "full": index[(f, seed, PRIMARY)], "power_only": index[(f, seed, INFO_CONTROL)],
             "joint_full_better": joint_improvement(index[(f, seed, INFO_CONTROL)], index[(f, seed, PRIMARY)], tolerance)}
            for f in folds]
    macro = {name: {k: _mean([r[name][k] for r in rows]) for k in METRIC_KEYS} for name in ("full", "power_only")}
    joint = int(sum(r["joint_full_better"] for r in rows))
    supported = joint_improvement(macro["power_only"], macro["full"], tolerance) and joint >= 2
    return {"evidence": "I1", "supported": bool(supported), "complete": True, "seed": seed, "rows": rows,
            "macro": macro, "jointly_better_folds": joint,
            "interpretation": "auxiliary-channel package diagnostic only; never changes the primary candidate "
                              "and never authorises an outer fallback"}


def _seed_fold_means(index, arm, folds, seeds):
    per_fold = {f: {k: _mean([index[(f, s, arm)][k] for s in seeds]) for k in METRIC_KEYS} for f in folds}
    per_seed = {s: {k: _mean([index[(f, s, arm)][k] for f in folds]) for k in METRIC_KEYS} for s in seeds}
    macro = {k: _mean([per_fold[f][k] for f in folds]) for k in METRIC_KEYS}
    return per_fold, per_seed, macro


def gate_g2(records, persistence, settings, folds, tolerance=TIE_TOLERANCE_KW):
    """Three-seed screen: mean over seeds within fold, then equal fold mean; plus per-fold and per-seed
    joint-improvement counts so a seed average cannot hide widespread damage."""
    seeds = [int(s) for s in settings["seeds"]]
    index = _index(records)
    missing = _missing(index, folds, seeds, (PRIMARY, BASELINE))
    if missing:
        return {"gate": "G2", "passed": False, "complete": False, "missing": missing,
                "reason": "incomplete records are a failure, never a pass"}
    pers = _persistence(persistence, folds)
    cand_fold, cand_seed, cand_macro = _seed_fold_means(index, PRIMARY, folds, seeds)
    ref_fold, ref_seed, ref_macro = _seed_fold_means(index, BASELINE, folds, seeds)
    pers_macro = {k: _mean([pers[f][k] for f in folds]) for k in METRIC_KEYS}
    fold_joint = {f: joint_improvement(ref_fold[f], cand_fold[f], tolerance) for f in folds}
    seed_joint = {s: joint_improvement(ref_seed[s], cand_seed[s], tolerance) for s in seeds}
    checks = {"macro_mae_rmse_better_than_dlinear": joint_improvement(ref_macro, cand_macro, tolerance),
              "macro_mae_rmse_better_than_persistence": joint_improvement(pers_macro, cand_macro, tolerance),
              "seed_averaged_folds_jointly_better": int(sum(fold_joint.values())),
              "min_seed_averaged_folds": int(settings["min_seed_averaged_folds_jointly_better_than_DLinear"]),
              "fold_averaged_seeds_jointly_better": int(sum(seed_joint.values())),
              "min_fold_averaged_seeds": int(settings["min_fold_averaged_seeds_jointly_better_than_DLinear"])}
    passed = (checks["macro_mae_rmse_better_than_dlinear"] and checks["macro_mae_rmse_better_than_persistence"]
              and checks["seed_averaged_folds_jointly_better"] >= checks["min_seed_averaged_folds"]
              and checks["fold_averaged_seeds_jointly_better"] >= checks["min_fold_averaged_seeds"])
    return {"gate": "G2", "passed": bool(passed), "complete": True, "seeds": seeds,
            "aggregation": "equal mean of seeds within fold, then equal mean of folds",
            "candidate": {"per_fold_seed_mean": cand_fold, "per_seed_fold_mean": cand_seed, "macro": cand_macro},
            "dlinear": {"per_fold_seed_mean": ref_fold, "per_seed_fold_mean": ref_seed, "macro": ref_macro},
            "persistence": {"per_fold": pers, "macro": pers_macro},
            "fold_joint_vs_dlinear": fold_joint, "seed_joint_vs_dlinear": seed_joint, "checks": checks,
            "tolerance_kw": tolerance, "comparison_space": "internal_selection_only",
            "interpretation": "operational budget gate on development evidence; not a significance test"}


def info_i2(records, settings, folds, tolerance=TIE_TOLERANCE_KW):
    seeds = [int(s) for s in settings["seeds"]]
    index = _index(records)
    missing = _missing(index, folds, seeds, (PRIMARY, INFO_CONTROL))
    if missing:
        return {"evidence": "I2", "supported": False, "complete": False, "missing": missing}
    full_fold, _, full_macro = _seed_fold_means(index, PRIMARY, folds, seeds)
    po_fold, _, po_macro = _seed_fold_means(index, INFO_CONTROL, folds, seeds)
    fold_joint = {f: joint_improvement(po_fold[f], full_fold[f], tolerance) for f in folds}
    supported = joint_improvement(po_macro, full_macro, tolerance) and sum(fold_joint.values()) >= 2
    return {"evidence": "I2", "supported": bool(supported), "complete": True, "seeds": seeds,
            "full": {"per_fold_seed_mean": full_fold, "macro": full_macro},
            "power_only": {"per_fold_seed_mean": po_fold, "macro": po_macro},
            "fold_joint_full_better": fold_joint, "jointly_better_folds": int(sum(fold_joint.values())),
            "interpretation": "auxiliary-channel package diagnostic only; not a per-channel causal effect"}


# --------------------------------------------------------------------------- statistics
def holm_adjust(pvalues):
    """Holm step-down adjusted p-values in the original order (raw values kept by the caller)."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Holm requires finite p-values in [0, 1]")
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[index]))
        adjusted[index] = running
    return adjusted.tolist()


def _clusters(window_values, forecast_start):
    values = np.asarray(window_values, dtype=np.float64)
    frame = pd.DataFrame({"forecast_start": pd.to_datetime(pd.Series(forecast_start), errors="raise"), "value": values})
    if frame.forecast_start.isna().any() or not np.isfinite(values).all():
        raise ValueError("Finite losses and issue times required")
    return frame.groupby("forecast_start", sort=True, observed=True).value.agg(["sum", "count"])


def paired_loss_statistics(candidate_loss, reference_loss, forecast_start, *, block_length, replicates, seed,
                           units="kW", loss_name="absolute_error"):
    """Window-uniform paired test on per-point losses (e.g. seed-averaged |error| or squared error).

    Reproduces the window_uniform branch of utils.paired_forecast_statistics exactly (cluster ratio
    bootstrap + Newey-West ratio-mean influence test, lag = block_length-1), but accepts losses so that
    losses of several seeds on the SAME windows can be averaged first. This averages losses; it is not a
    prediction ensemble, and seed x window rows are never treated as independent samples.
    """
    cand = np.asarray(candidate_loss, dtype=np.float64)
    ref = np.asarray(reference_loss, dtype=np.float64)
    if cand.shape != ref.shape or cand.ndim not in (1, 2) or not (np.isfinite(cand).all() and np.isfinite(ref).all()):
        raise ValueError("Aligned finite candidate/reference losses required")
    if cand.ndim == 2:
        cand, ref = cand.mean(1), ref.mean(1)
    if len(forecast_start) != len(cand):
        raise ValueError("forecast_start must align one-to-one with windows")
    gain = ref - cand
    by_time = _clusters(gain, forecast_start)
    bootstrap = moving_block_bootstrap_ratio(by_time["sum"].to_numpy(), by_time["count"].to_numpy(),
                                             block_length=block_length, replicates=replicates, seed=seed)
    primary = float(gain.mean())
    influence = (by_time["sum"].to_numpy() - primary * by_time["count"].to_numpy()) / by_time["count"].mean()
    dm = newey_west_mean_test(influence + primary, max_lag=block_length - 1)
    return {"loss": loss_name, "units": units, "gain_definition": "mean reference loss - mean candidate loss; positive favours candidate",
            "estimand": "window_uniform", "candidate_mean_loss": float(cand.mean()), "reference_mean_loss": float(ref.mean()),
            "gain": primary, "ci95_low": bootstrap["ci95_low_kw"], "ci95_high": bootstrap["ci95_high_kw"],
            "probability_gain_positive": bootstrap["probability_gain_positive"],
            "bootstrap": {"block_length_issue_times": bootstrap["block_length_time_points"],
                          "replicates": bootstrap["replicates"], "seed": bootstrap["seed"]},
            "newey_west": {"statistic": dm["statistic"], "p_value_two_sided": dm["p_value_two_sided_normal"],
                           "lag": dm["newey_west_lag"], "time_points": dm["time_points"]},
            "windows": int(len(gain)), "issue_time_clusters": int(len(by_time)),
            "candidate_better_window_pct": float(100 * np.mean(cand < ref))}


def rmse_difference_bootstrap(candidate_squared, reference_squared, forecast_start, *, block_length, replicates, seed):
    """RMSE(reference) - RMSE(candidate) with an issue-time moving-block CI.

    Each replicate resamples whole issue-time clusters of squared-error sums and point counts, takes the
    square root per arm, then differences. This is NOT the MAE-difference CI.
    """
    cand = np.asarray(candidate_squared, dtype=np.float64)
    ref = np.asarray(reference_squared, dtype=np.float64)
    if cand.shape != ref.shape or cand.ndim != 2 or np.any(cand < 0) or np.any(ref < 0) \
            or not (np.isfinite(cand).all() and np.isfinite(ref).all()):
        raise ValueError("Aligned finite non-negative [window, lead] squared errors required")
    if len(forecast_start) != len(cand):
        raise ValueError("forecast_start must align one-to-one with windows")
    frame = pd.DataFrame({"forecast_start": pd.to_datetime(pd.Series(forecast_start), errors="raise"),
                          "cand": cand.sum(1), "ref": ref.sum(1), "points": np.full(len(cand), cand.shape[1], float)})
    by_time = frame.groupby("forecast_start", sort=True).agg(cand=("cand", "sum"), ref=("ref", "sum"), points=("points", "sum"))
    c, r, n = (by_time[k].to_numpy() for k in ("cand", "ref", "points"))
    reps = int(replicates)
    if reps < 100 or len(n) < 2:
        raise ValueError("At least 100 replicates and two issue-time clusters required")
    block = min(max(1, int(block_length)), len(n))
    starts = np.arange(len(n) - block + 1)
    needed = int(math.ceil(len(n) / block))
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(reps)
    for i in range(reps):
        chosen = rng.choice(starts, size=needed, replace=True)
        index = np.concatenate([np.arange(start, start + block) for start in chosen])[:len(n)]
        total = n[index].sum()
        estimates[i] = math.sqrt(r[index].sum() / total) - math.sqrt(c[index].sum() / total)
    low, high = np.quantile(estimates, [.025, .975])
    rmse_c, rmse_r = math.sqrt(c.sum() / n.sum()), math.sqrt(r.sum() / n.sum())
    return {"units": "kW", "rmse_candidate_kw": rmse_c, "rmse_reference_kw": rmse_r, "rmse_gain_kw": rmse_r - rmse_c,
            "ci95_low_kw": float(low), "ci95_high_kw": float(high),
            "probability_gain_positive": float(np.mean(estimates > 0)),
            "block_length_issue_times": block, "replicates": reps, "seed": int(seed),
            "method": "issue-time moving-block bootstrap of squared-error sums; sqrt per arm per replicate; "
                      "NOT a reuse of the MAE-difference interval"}


def issue_time_coverage(forecast_start, stride_hours):
    times = pd.Series(pd.to_datetime(pd.Series(forecast_start)).drop_duplicates().sort_values().to_numpy())
    gaps = times.diff().dropna().dt.total_seconds() / 3600
    return {"issue_times": int(len(times)), "covered_utc_days": int(times.dt.floor("D").nunique()),
            "gaps_above_stride": int((gaps > stride_hours).sum()),
            "max_gap_hours": float(gaps.max()) if len(gaps) else None,
            "note": "8 issue-time blocks are 24 clock hours only when there is no gap"}


def assert_aligned(reference, other, keys=("times", "turbines", "window_starts", "truth", "persistence"), label=""):
    """Paired comparisons require byte-identical cohort metadata, truth and Persistence."""
    for key in keys:
        if key not in reference or key not in other:
            raise ValueError(f"{label}: missing aligned field {key}")
        a, b = np.asarray(reference[key]), np.asarray(other[key])
        if a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError(f"{label}: cohort field {key} differs; refuse to pair")
    return True


# --------------------------------------------------------------------------- independent point metrics (A4 recomputation)
def point_metrics(prediction, truth, persistence, capacity):
    """Plain numpy re-implementation used to recompute registered metrics independently."""
    p, y, b = (np.asarray(v, dtype=np.float64) for v in (prediction, truth, persistence))
    if p.shape != y.shape or b.shape != y.shape or not (np.isfinite(p).all() and np.isfinite(y).all() and np.isfinite(b).all()):
        raise ValueError("Aligned finite arrays required")
    e = p - y
    mae, mse = float(np.abs(e).mean()), float((e ** 2).mean())
    sst = float(((y - y.mean()) ** 2).sum())
    eb = b - y
    mae_b, rmse_b = float(np.abs(eb).mean()), math.sqrt(float((eb ** 2).mean()))
    rmse = math.sqrt(mse)
    return {"mae": mae, "mse": mse, "rmse": rmse,
            "r2": (1 - float((e ** 2).sum()) / sst) if sst > np.finfo(float).eps else None,
            "r2_defined": bool(sst > np.finfo(float).eps), "mbe": float(e.mean()),
            "mae_skill_pct": 100 * (1 - mae / mae_b) if mae_b > 0 else None,
            "rmse_skill_pct": 100 * (1 - rmse / rmse_b) if rmse_b > 0 else None,
            "within_5pct_capacity_pct": float(100 * np.mean(np.abs(e) <= .05 * capacity)),
            "within_10pct_capacity_pct": float(100 * np.mean(np.abs(e) <= .10 * capacity)),
            "points": int(e.size)}


# --------------------------------------------------------------------------- figure rule
def select_trace_segment(issue_times, turbines, stride_hours=3, issues_required=24):
    """Pre-declared forecast_trace rule: smallest turbine ID; time-ordered; the FIRST run of
    ``issues_required`` consecutive issue times spaced exactly ``stride_hours`` (72 h of issues);
    if none exists, the longest run, ties -> earliest. Returns row indices into the cohort arrays."""
    issue_times = pd.to_datetime(pd.Series(np.asarray(issue_times))).to_numpy()
    turbines = np.asarray(turbines)
    unit = int(turbines.min())
    rows = np.flatnonzero(turbines == unit)
    rows = rows[np.argsort(issue_times[rows], kind="mergesort")]
    times = issue_times[rows]
    step = np.timedelta64(int(stride_hours), "h")
    breaks = np.flatnonzero(np.diff(times) != step) + 1
    runs = np.split(np.arange(len(rows)), breaks)
    complete = [run for run in runs if len(run) >= issues_required]
    if complete:
        chosen, is_complete = complete[0][:issues_required], True
    else:
        longest = max(len(run) for run in runs)
        chosen, is_complete = next(run for run in runs if len(run) == longest), False
    return {"turbine": unit, "rows": rows[chosen].tolist(), "complete": bool(is_complete),
            "first_issue": str(times[chosen[0]]), "last_issue": str(times[chosen[-1]]),
            "rule": f"smallest turbine; first run of {issues_required} issue times spaced {stride_hours}h; "
                    "else longest run, ties earliest; leads 1 and 4 shown; never chosen by error"}


# --------------------------------------------------------------------------- device policy
def choose_pair_device(gpu, settings):
    """Decide the device for a whole fold/seed pair BEFORE it starts.

    ``gpu``: None (no CUDA) or {"total_mib", "free_mib", "other_compute_pids"}. A memory fraction caps
    only this process's PyTorch allocator; it is not hardware isolation nor a compute share.
    """
    if not gpu:
        return {"device": "cpu", "reason": "CUDA unavailable", "cpu_threads": settings["cpu_threads"]}
    free_gib, total_mib = gpu["free_mib"] / 1024., float(gpu["total_mib"])
    if free_gib < settings["gpu_free_gib_min_for_new_pair"]:
        return {"device": "cpu", "reason": f"only {free_gib:.2f} GiB free < {settings['gpu_free_gib_min_for_new_pair']} GiB",
                "cpu_threads": settings["cpu_threads"], "gpu_snapshot": gpu}
    shared = bool(gpu.get("other_compute_pids"))
    fraction = (min(1.0, settings["shared_gpu_pytorch_allocator_cap_gib"] * 1024. / total_mib) if shared
                else settings["idle_gpu_allocator_fraction_max"])
    return {"device": "cuda", "memory_fraction": float(fraction), "shared_with_other_processes": shared,
            "allocator_cap_gib": fraction * total_mib / 1024., "cpu_threads": settings["cpu_threads"],
            "gpu_snapshot": gpu, "compute_share_guaranteed": False,
            "reason": "shared GPU, 4 GiB allocator cap" if shared else "idle GPU, 90% allocator cap"}


# --------------------------------------------------------------------------- data access audit
class DataAccessLog:
    """Records every subset request; validation only in A3 after a passed G2, never sealed test."""

    def __init__(self, path):
        self.path = Path(path)
        self.requests = []

    def request(self, stage, subset, fold, *, g2_passed=False):
        if subset in ("test", "sealed_test"):
            raise PermissionError("The sealed test suffix is never requested in v25")
        allowed = STAGE_SUBSETS.get(stage)
        if allowed is None or subset not in allowed:
            raise PermissionError(f"Stage {stage} may not request subset {subset!r}")
        if subset == "validation" and not g2_passed:
            raise PermissionError("Outer validation requires a registered passed G2")
        key = (stage, subset, int(fold))
        if key not in self.requests:
            append_event(self.path, "data_subset_requested", stage=stage, subset=subset, fold=int(fold))
        self.requests.append(key)


def inner_datasets(series):
    """v20 internal allocation: fit = targets ending before calibration_early_end (90% of outer train);
    selection = targets fully inside [calibration_early_end, train_end). History may precede the stage.
    Outer validation is NOT instantiated here."""
    from data_provider.kelmarsh_hourly import KelmarshWindows

    first = series.dates[series.starts + series.input_len]
    last = series.dates[series.starts + series.input_len + series.horizon - 1]
    cutoff, end = series.bounds["calibration_early_end"], series.bounds["train_end"]
    masks = {"fit": last < cutoff, "selection": (first >= cutoff) & (last < end)}
    data = {}
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"Empty chronological stage {name}")
        dataset = KelmarshWindows(series, series.starts[mask], name)
        dataset.train_cutoff, dataset.val_cutoff = cutoff, end
        data[name] = dataset
    return data
