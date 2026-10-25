"""Torch-free helpers for v27: training-side forward-block diagnostic of the v26 anchored residual (Wiki off).

Pure functions of explicit inputs: block edges, step masks, per-step scalers, the protocol "views" consumed by the
unchanged v25/v26 trainers, metric tables, pooled paired statistics and the predeclared decision rules. Generic hashing,
I/O and statistics are reused from ``utils.kelmarsh_backbone_information`` (v25), whose bytes enter the fingerprint.
"""
from __future__ import annotations

import copy

import numpy as np

from utils.kelmarsh_backbone_information import (  # noqa: F401  (re-exported for the v27 controller/report)
    acquire_exclusive_lock, append_event, array_sha256, assert_aligned, canonical_sha256, choose_pair_device,
    contract_mismatches, file_sha256, joint_improvement, paired_loss_statistics, point_metrics, read_json,
    rmse_difference_bootstrap, safe, text_sha256_lf, verify_registered_files, verify_training_completed, window_hash,
    write_json)

PROTOCOL_ID = "kelmarsh_forward_block_residual_diagnostic_v27"
FULL, POWER, ANCHOR = "AnchoredFull", "AnchoredPowerOnly", "DLinear"
RESIDUAL_ARMS = (FULL, POWER)
HOUR = np.timedelta64(1, "h")

DESIGN_VALUES = {
    ("protocol_id",): PROTOCOL_ID, ("input_len",): 24, ("pred_len",): 4, ("train_stride",): 3, ("target_slot",): 6,
    ("rated_power_kw",): 2050.0,
    ("feature_columns",): ["Wspd", "Etmp", "Ndir_sin", "Ndir_cos", "Pab_mean", "rotor_speed", "power"],
    ("timeline", "source_fold"): 0, ("timeline", "blocks"): 6, ("timeline", "first_eval_block"): 3,
    ("timeline", "eval_blocks"): [3, 4, 5, 6],
    ("timeline", "fold0_fit_window_hash"): "9c843ac52086be031af7e2e635116a872047fffab76d8ed22a260888cd5e0922",
    ("anchor", "expected_parameters"): 200, ("anchor", "seed"): 2024, ("anchor", "training", "epoch0_included"): True,
    ("candidate", "expected_trainable_parameters"): 3618, ("candidate", "expected_frozen_parameters"): 200,
    ("candidate", "training", "epoch0_selectable"): False, ("candidate", "training", "first_selectable_epoch"): 1,
    ("residual_seeds",): [2024, 2025, 2026, 2027, 2028],
    ("budget", "new_fits_max"): 44, ("budget", "new_epochs_max"): 880,
    ("internal_selection_access",): False, ("outer_validation_access",): False, ("sealed_test_access",): False,
    ("wiki_enabled",): False, ("labels_modified",): False, ("prediction_clipping",): False,
    ("evaluation", "pooled_statistics", "block_length"): 8, ("evaluation", "pooled_statistics", "replicates"): 5000,
    ("evaluation", "pooled_statistics", "seed"): 2024, ("evaluation", "pooled_statistics", "hac_max_lag"): 7,
    ("decision", "numeric_tie_tolerance_kw"): 1e-6,
}
RECIPE_KEYS = ("learning_rate", "weight_decay", "batch_size", "eval_batch_size", "max_epochs", "patience", "grad_clip_norm",
               "scheduler_t_max", "scheduler_eta_min", "loss_mae_weight", "loss_mse_weight")


def validate_protocol(protocol):
    for path, expected in DESIGN_VALUES.items():
        value = protocol
        for key in path:
            value = value[key]
        if value != expected:
            raise ValueError(f"Protocol value {'.'.join(path)}={value!r} departs from design {expected!r}")
    a, c = protocol["anchor"]["training"], protocol["candidate"]["training"]
    if {k: a[k] for k in RECIPE_KEYS} != {k: c[k] for k in RECIPE_KEYS}:
        raise ValueError("Anchor and residual must share the v20/v26 optimiser recipe")
    if not (protocol["anchor"]["model_args"] == protocol["candidate"]["anchor"]["model_args"]
            == protocol["arms"][ANCHOR]["model_args"]):
        raise ValueError("The trained anchor and the embedded anchor must be the same DLinear")
    if protocol["arms"][FULL]["input_mode"] != "full" or protocol["arms"][POWER]["input_mode"] != "power_only":
        raise ValueError("AnchoredPowerOnly must be the masked ablation of AnchoredFull")
    fits = planned_fits(protocol)
    if fits["total"] > protocol["budget"]["new_fits_max"] or \
            fits["total"] * a["max_epochs"] > protocol["budget"]["new_epochs_max"]:
        raise ValueError("Declared matrix exceeds the budget")
    return True


def planned_fits(protocol):
    steps = len(protocol["timeline"]["eval_blocks"])
    per_step = 1 + len(RESIDUAL_ARMS) * len(protocol["residual_seeds"])
    return {"steps": steps, "per_step": per_step, "total": steps * per_step}


# --------------------------------------------------------------------------- views for the unchanged trainers
def anchor_view(protocol):
    """Protocol dict in the shape ``scripts.train_kelmarsh_backbone_information.fit_arm`` (v25) expects for DLinear."""
    a = protocol["anchor"]
    return {"rated_power_kw": protocol["rated_power_kw"], "training": copy.deepcopy(a["training"]),
            "arms": {ANCHOR: {"family": "DLinear", "model_args": dict(a["model_args"]),
                              "expected_parameters": a["expected_parameters"]}}}


def residual_view(protocol):
    """Protocol dict in the shape ``scripts.train_kelmarsh_anchored_residual`` (v26) build_model/fit_arm expect."""
    return {"rated_power_kw": protocol["rated_power_kw"], "input_len": protocol["input_len"], "pred_len": protocol["pred_len"],
            "feature_columns": list(protocol["feature_columns"]), "target_slot": protocol["target_slot"],
            "training": copy.deepcopy(protocol["candidate"]["training"]), "candidate": copy.deepcopy(protocol["candidate"]),
            "arms": copy.deepcopy(protocol["arms"])}


# --------------------------------------------------------------------------- blocks, steps, scalers
def block_edges(t_lo, t_hi, blocks):
    """Equal integer-hour blocks of [t_lo, t_hi): edges[i] = t_lo + floor(total * i / blocks)."""
    t_lo, t_hi = np.datetime64(t_lo, "h"), np.datetime64(t_hi, "h")
    total = int((t_hi - t_lo) / HOUR)
    if total < blocks or blocks < 2:
        raise ValueError("Timeline too short for the declared blocks")
    return [t_lo + np.timedelta64(total * i // blocks, "h") for i in range(blocks + 1)]


def step_bounds(edges, step):
    """Hour bounds of 1-based step ``step``: training span [edges[0], edges[step-1]) split 90/10, evaluation block."""
    t_lo, train_end, eval_end = edges[0], edges[step - 1], edges[step]
    span = int((train_end - t_lo) / HOUR)
    cut = t_lo + np.timedelta64(int(np.floor(0.9 * span)), "h")
    if not (t_lo < cut < train_end < eval_end):
        raise ValueError(f"Degenerate step {step}")
    return {"step": int(step), "start": t_lo, "fit_cut": cut, "train_end": train_end, "eval_end": eval_end}


def step_masks(first_target, last_target, bounds):
    """Boolean masks over candidate windows: fit / selection / evaluation (all four targets inside the part)."""
    first = np.asarray(first_target).astype("datetime64[h]")
    last = np.asarray(last_target).astype("datetime64[h]")
    fit = (first >= bounds["start"]) & (last < bounds["fit_cut"])
    selection = (first >= bounds["fit_cut"]) & (last < bounds["train_end"])
    evaluation = (first >= bounds["train_end"]) & (last < bounds["eval_end"])
    if (fit & selection).any() or (fit & evaluation).any() or (selection & evaluation).any():
        raise AssertionError("Step parts overlap")
    for name, mask in (("fit", fit), ("selection", selection), ("evaluation", evaluation)):
        if not mask.any():
            raise ValueError(f"Step {bounds['step']} has no {name} windows")
    return {"fit": fit, "selection": selection, "evaluation": evaluation}


def scaler_stats(raw, dates, cut):
    """Per-channel mean / std (ddof 0, NaN ignored) over rows with hour-end < cut. sklearn StandardScaler semantics,
    including scale 1.0 for a constant channel. Returns (mean, scale, rows_used, constant_channels)."""
    raw = np.asarray(raw, dtype=np.float64)
    rows = np.asarray(dates).astype("datetime64[h]") < np.datetime64(cut, "h")
    if not rows.any():
        raise ValueError("No rows before the scaler cut")
    sub = raw[rows]
    with np.errstate(invalid="ignore"):
        mean, std = np.nanmean(sub, axis=0), np.nanstd(sub, axis=0)
    if not (np.isfinite(mean).all() and np.isfinite(std).all()):
        raise ValueError("A channel has no finite observation before the cut")
    constant = np.flatnonzero(std == 0).tolist()
    scale = np.where(std == 0, 1.0, std)
    return mean, scale, int(rows.sum()), constant


# --------------------------------------------------------------------------- metrics and decisions
def basic_metrics(prediction, truth):
    prediction, truth = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape or truth.ndim != 2 or not truth.size:
        raise ValueError("Aligned nonempty [window, lead] arrays required; broadcasting is forbidden")
    e = prediction - truth
    if not np.isfinite(e).all():
        raise ValueError("Finite [window, lead] arrays required")
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())), "mbe": float(e.mean()),
            "median_error": float(np.median(e)),
            "lead_mae": np.abs(e).mean(0).tolist(), "lead_rmse": np.sqrt((e ** 2).mean(0)).tolist(), "lead_mbe": e.mean(0).tolist()}


def step_record(arrays):
    """Metrics of one step from its registered evaluation arrays (anchor, residual seeds, seed means, persistence)."""
    y = arrays["truth"]
    full, power = np.asarray(arrays["full_best"]), np.asarray(arrays["power_best"])
    if full.shape != power.shape or full.ndim != 3 or full.shape[1:] != np.asarray(y).shape or len(full) == 0:
        raise ValueError("Aligned [seed, window, lead] forecasts required")
    record = {"persistence": basic_metrics(arrays["persistence"], y), "anchor": basic_metrics(arrays["anchor"], y),
              "full_seed_mean": basic_metrics(full.mean(0), y), "power_seed_mean": basic_metrics(power.mean(0), y),
              "full_single": [basic_metrics(p, y) for p in full], "power_single": [basic_metrics(p, y) for p in power]}
    for name in ("full_single", "power_single"):
        record[name + "_mean"] = {k: float(np.mean([m[k] for m in record[name]])) for k in ("mae", "rmse", "mbe")}
    return record


def _joint(reference, candidate, tol):
    return bool(joint_improvement({"mae": reference["mae"], "rmse": reference["rmse"]},
                                  {"mae": candidate["mae"], "rmse": candidate["rmse"]}, tol))


def pooled_statistics(steps_arrays, settings):
    """Pooled paired statistics over all evaluation windows of all steps (issue-time clusters)."""
    def cat(key, fn=None):
        values = [fn(a) if fn else a[key] for a in steps_arrays]
        return np.concatenate(values, axis=0)
    y = cat("truth")
    starts = cat(None, lambda a: a["times"][:, 0] - HOUR)
    anchor = cat("anchor")
    full = cat(None, lambda a: np.asarray(a["full_best"]).mean(0))
    power = cat(None, lambda a: np.asarray(a["power_best"]).mean(0))
    fixed = cat(None, lambda a: np.asarray(a["full_best"])[0])
    kwargs = {"block_length": settings["block_length"], "replicates": settings["replicates"], "seed": settings["seed"]}

    def pair(candidate, reference, name):
        return {"reference": name,
                "absolute_error": paired_loss_statistics(np.abs(candidate - y), np.abs(reference - y), starts, **kwargs),
                "rmse_difference": rmse_difference_bootstrap((candidate - y) ** 2, (reference - y) ** 2, starts, **kwargs),
                "lead1_mae_gain_kw": float(np.abs(reference - y)[:, 0].mean() - np.abs(candidate - y)[:, 0].mean()),
                "lead1_rmse_gain_kw": float(np.sqrt(((reference - y)[:, 0] ** 2).mean()) - np.sqrt(((candidate - y)[:, 0] ** 2).mean()))}
    return {"windows": int(len(y)), "issue_times": int(len(np.unique(starts))),
            "full_mean_vs_anchor": pair(full, anchor, "anchor"),
            "full_fixed_seed_vs_anchor": pair(fixed, anchor, "anchor"),
            "full_mean_vs_power_mean": pair(full, power, "power_seed_mean"),
            "power_mean_vs_anchor": pair(power, anchor, "anchor")}


def decide(step_records, pooled, protocol):
    """Predeclared decision rules (protocol['decision']); returns rule flags, counts and the outcome."""
    tol = float(protocol["decision"]["numeric_tie_tolerance_kw"])
    n = len(step_records)
    need = 3 if n == 4 else int(np.ceil(0.75 * n))
    r1_steps = sum(_joint(r["anchor"], r["full_seed_mean"], tol) for r in step_records)
    r2_steps = sum(r["full_single_mean"]["mae"] - r["full_seed_mean"]["mae"] > tol for r in step_records)
    r3_steps = sum(_joint(r["power_seed_mean"], r["full_seed_mean"], tol) for r in step_records)
    r5_steps = sum(_joint(r["anchor"], r["full_single"][0], tol) for r in step_records)
    fa, fp, single = pooled["full_mean_vs_anchor"], pooled["full_mean_vs_power_mean"], pooled["full_fixed_seed_vs_anchor"]
    flags = {"R1_residual_transfer": bool(r1_steps >= need and fa["absolute_error"]["ci95_low"] > 0),
             "R2_averaging_benefit": bool(r2_steps >= need),
             "R3_auxiliary_attribution": bool(r3_steps >= need and fp["absolute_error"]["ci95_low"] > 0),
             "R4_near_term": bool(fa["lead1_mae_gain_kw"] >= -tol and fa["lead1_rmse_gain_kw"] >= -tol),
             "R5_fixed_seed_transfer": bool(r5_steps >= need and single["absolute_error"]["ci95_low"] > 0
                                            and single["lead1_mae_gain_kw"] >= -tol
                                            and single["lead1_rmse_gain_kw"] >= -tol)}
    if flags["R1_residual_transfer"] and flags["R4_near_term"] and flags["R2_averaging_benefit"]:
        outcome = "BAGGED_RESIDUAL_MAY_ENTER_CONFIRMATION"
    elif flags["R1_residual_transfer"] and flags["R4_near_term"] and flags["R5_fixed_seed_transfer"]:
        outcome = "RESIDUAL_MAY_ENTER_CONFIRMATION_WITHOUT_BAGGING"
    else:
        outcome = "STOP_RESIDUAL_LINE_ON_THIS_DATA"
    return {"steps": n, "steps_required": need, "counts": {"R1": r1_steps, "R2": r2_steps, "R3": r3_steps, "R5": r5_steps},
            "pooled_full_mean_vs_anchor_mae_ci95": [fa["absolute_error"]["ci95_low"], fa["absolute_error"]["ci95_high"]],
            "pooled_full_vs_power_mae_ci95": [fp["absolute_error"]["ci95_low"], fp["absolute_error"]["ci95_high"]],
            "flags": flags, "outcome": outcome, "auxiliary_claim_allowed": flags["R3_auxiliary_attribution"],
            "interpretation": "training-side diagnostic on a summer-only period; authorises at most the design of one later "
                              "confirmation protocol, never a forecasting claim. R2 measures cancellation benefit, not "
                              "evidence that seed variance is the main cause (MAE is convex). The nonbagged route requires "
                              "the prespecified seed 2024 itself to pass R5; no seed is selected after evaluation."}


# --------------------------------------------------------------------------- descriptive diagnostics
POWER_EDGES = (0.1, 0.3, 0.6, 0.9)
POWER_LABELS = ("<10%", "10-30%", "30-60%", "60-90%", ">=90%")


def correction_diagnostics(arrays):
    """Per arm: mean / mean-abs correction and corr(correction, anchor error) per lead; cross-seed correlation."""
    y, b = arrays["truth"], arrays["anchor"]
    target = y - b
    out = {}
    for name, key in (("full", "full_best"), ("power", "power_best")):
        preds = np.asarray(arrays[key])
        corrections = preds - b[None]
        mean_corr = corrections.mean(0)
        per_lead_corr = []
        for h in range(y.shape[1]):
            x = mean_corr[:, h]
            per_lead_corr.append(float(np.corrcoef(x, target[:, h])[0, 1]) if x.std() > 0 else None)
        flat = corrections.reshape(len(preds), -1)
        pairs = [float(np.corrcoef(flat[i], flat[j])[0, 1]) for i in range(len(flat)) for j in range(i + 1, len(flat))
                 if flat[i].std() > 0 and flat[j].std() > 0]
        out[name] = {"seed_mean_correction_kw_by_lead": mean_corr.mean(0).tolist(),
                     "seed_mean_abs_correction_kw_by_lead": np.abs(mean_corr).mean(0).tolist(),
                     "single_seed_abs_correction_kw": float(np.abs(corrections).mean()),
                     "corr_seed_mean_correction_vs_anchor_error_by_lead": per_lead_corr,
                     "cross_seed_correction_corr_mean": float(np.mean(pairs)) if pairs else None,
                     "cross_seed_correction_corr_min": float(np.min(pairs)) if pairs else None}
    return out


def regime_diagnostics(arrays, rated_power):
    """MAE of anchor / Full seed mean by physical regime of the last observed hour and by turbine, with support."""
    y, b = arrays["truth"], arrays["anchor"]
    f = np.asarray(arrays["full_best"]).mean(0)
    last = arrays["persistence"][:, 0]
    bins = np.digitize(last / float(rated_power), POWER_EDGES)
    regimes, turbines = {}, {}
    for i, label in enumerate(POWER_LABELS):
        m = bins == i
        regimes[label] = ({"windows": int(m.sum()), "anchor_mae": float(np.abs(b - y)[m].mean()),
                           "full_mean_mae": float(np.abs(f - y)[m].mean()), "anchor_mbe": float((b - y)[m].mean())}
                          if m.any() else {"windows": 0})
    for unit in np.unique(arrays["turbines"]):
        m = arrays["turbines"] == unit
        turbines[str(int(unit))] = {"windows": int(m.sum()), "anchor_mae": float(np.abs(b - y)[m].mean()),
                                    "full_mean_mae": float(np.abs(f - y)[m].mean())}
    return {"power_regime": regimes, "turbine": turbines}
