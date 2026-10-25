"""Torch-free helpers for v26 (frozen DLinear anchor + zero-initialised TSMixerBasicSmall residual), Wiki off.

Generic hashing, I/O, statistics, alignment, metric and device helpers are re-used unchanged from the closed v25
module ``utils.kelmarsh_backbone_information`` (its bytes enter the v26 source fingerprint). Everything v26-specific
lives here: arm names, protocol self-consistency, the epoch-0 rule, the lead-1 non-inferiority gate and the
information-attribution evidence. Gate decisions are budget rules on development evidence, not significance tests.
"""
from __future__ import annotations

import numpy as np

from utils.kelmarsh_backbone_information import (  # noqa: F401  (re-exported for the v26 controller/report)
    DataAccessLog as _V25DataAccessLog, acquire_exclusive_lock, append_event, array_sha256, assert_aligned,
    canonical_sha256, choose_pair_device, contract_mismatches, file_sha256, holm_adjust, improves, inner_datasets,
    issue_time_coverage, joint_improvement, paired_loss_statistics, point_metrics, read_json, rmse_difference_bootstrap,
    safe, select_trace_segment, text_sha256_lf, verify_registered_files, verify_training_completed, window_hash,
    write_json)

PROTOCOL_ID = "kelmarsh_anchored_residual_v26"
PRIMARY = "AnchoredFull"
INFO_CONTROL = "AnchoredPowerOnly"
BASELINE = "DLinear"
CANDIDATE_ARMS = (PRIMARY, INFO_CONTROL)
ALL_ARMS = (PRIMARY, INFO_CONTROL, BASELINE)
METRIC_KEYS = ("mae", "rmse")
LEAD1_KEYS = ("lead1_mae", "lead1_rmse")
TIE_TOLERANCE_KW = 1e-6
STAGE_SUBSETS = {"A0": {"fit", "selection"}, "A1": {"fit", "selection"}, "A2": {"fit", "selection"},
                 "A3": {"fit", "selection", "validation"}}


# --------------------------------------------------------------------------- protocol self-consistency
DESIGN_VALUES = {
    ("protocol_id",): PROTOCOL_ID,
    ("input_len",): 24, ("pred_len",): 4, ("train_stride",): 3, ("eval_stride",): 3, ("folds",): [0, 1, 2],
    ("rated_power_kw",): 2050.0, ("target_slot",): 6,
    ("feature_columns",): ["Wspd", "Etmp", "Ndir_sin", "Ndir_cos", "Pab_mean", "rotor_speed", "power"],
    ("candidate", "family"): "DLinearAnchoredResidual",
    ("candidate", "residual", "blocks"): 2, ("candidate", "residual", "feature_hidden_dim"): 32,
    ("candidate", "residual", "norm_shape"): [24, 7], ("candidate", "residual", "norm_eps"): 0.001,
    ("candidate", "residual", "dropout"): 0.1, ("candidate", "residual", "trainable_parameters"): 3618,
    ("candidate", "expected_trainable_parameters"): 3618, ("candidate", "expected_frozen_parameters"): 200,
    ("candidate", "anchor", "parameters"): 200, ("candidate", "anchor", "frozen"): True,
    ("candidate", "anchor", "requires_grad"): False, ("candidate", "anchor", "in_optimizer"): False,
    ("candidate", "gate"): False, ("candidate", "revin"): False, ("candidate", "pretraining"): False,
    ("candidate", "prompt"): False, ("candidate", "wiki"): False, ("candidate", "persistence_residual_anchor"): False,
    ("training", "optimizer"): "AdamW", ("training", "learning_rate"): 0.001, ("training", "weight_decay"): 0.0001,
    ("training", "batch_size"): 64, ("training", "eval_batch_size"): 128, ("training", "max_epochs"): 20,
    ("training", "patience"): 5, ("training", "grad_clip_norm"): 1.0, ("training", "scheduler_t_max"): 20,
    ("training", "scheduler_eta_min"): 0.0, ("training", "loss_mae_weight"): 0.8, ("training", "loss_mse_weight"): 0.2,
    ("training", "amp"): False, ("training", "num_workers"): 0, ("training", "refit_after_selection"): False,
    ("training", "epoch0_recorded"): True, ("training", "epoch0_selectable"): False,
    ("training", "first_selectable_epoch"): 1,
    ("budget", "new_fits_max"): 18, ("budget", "new_epochs_max"): 360, ("budget", "dlinear_retraining"): False,
    ("seeds", "A1"): [2024], ("seeds", "A2"): [2025, 2026], ("seeds", "anchor_seed"): 2024,
    ("reference", "seed"): 2024, ("reference", "expected_parameters"): 200,
    ("gates", "numeric_tie_tolerance_kw"): 1e-6, ("gates", "primary_candidate"): PRIMARY,
    ("gates", "G1", "min_folds_jointly_better_than_DLinear"): 2,
    ("gates", "G1", "lead1_macro_mae_rmse_not_worse_than"): BASELINE,
    ("gates", "G2", "lead1_macro_mae_rmse_not_worse_than"): BASELINE,
    ("wiki_enabled",): False, ("sealed_test_access",): False, ("prediction_clipping",): False,
    ("target_clipping",): False, ("labels_modified",): False,
    ("statistics", "block_length"): 8, ("statistics", "replicates"): 5000, ("statistics", "seed"): 2024,
    ("statistics", "estimand"): "window_uniform", ("statistics", "hac_max_lag"): 7,
}


def validate_protocol(protocol):
    """Refuse an executable protocol that silently departs from the reviewed v26 design."""
    for path, expected in DESIGN_VALUES.items():
        value = protocol
        for key in path:
            value = value[key]
        if value != expected:
            raise ValueError(f"Protocol value {'.'.join(path)}={value!r} departs from design {expected!r}")
    arms = protocol["arms"]
    if tuple(arms) != ALL_ARMS:
        raise ValueError("Arms must be exactly AnchoredFull, AnchoredPowerOnly, DLinear")
    if arms[PRIMARY]["input_mode"] != "full" or arms[INFO_CONTROL]["input_mode"] != "power_only":
        raise ValueError("AnchoredPowerOnly must be the masked information ablation of AnchoredFull")
    if arms[BASELINE]["model_args"] != protocol["candidate"]["anchor"]["model_args"]:
        raise ValueError("The reference DLinear and the anchor must be the same model")
    if [c["fold"] for c in protocol["reference"]["dlinear_checkpoints"]] != list(protocol["folds"]):
        raise ValueError("One frozen anchor checkpoint per declared fold is required")
    if protocol["statistics"]["hac_max_lag"] != protocol["statistics"]["block_length"] - 1:
        raise ValueError("HAC lag must equal block_length-1 as in utils.paired_forecast_statistics")
    budget = planned_new_fits(protocol)
    if budget["max_new_fits"] > protocol["budget"]["new_fits_max"]:
        raise ValueError("Declared matrix exceeds the fit budget")
    if budget["max_new_fits"] * protocol["training"]["max_epochs"] > protocol["budget"]["new_epochs_max"]:
        raise ValueError("Declared matrix exceeds the epoch budget")
    return True


def planned_new_fits(protocol):
    """Only the two residual arms are ever trained; the anchor/reference is never refitted."""
    folds = len(protocol["folds"])
    a1 = folds * len(protocol["seeds"]["A1"]) * len(CANDIDATE_ARMS)
    a2 = folds * len(protocol["seeds"]["A2"]) * len(CANDIDATE_ARMS)
    return {"A1": a1, "A2_max": a2, "max_new_fits": a1 + a2}


# --------------------------------------------------------------------------- epoch selection rule
def select_epoch(history, first_selectable=1):
    """Return (best_epoch, best_mae) under the v26 rule: epoch 0 is recorded but not selectable, a strictly lower
    selection MAE updates, ties keep the earlier epoch. ``history`` is the list written by the controller."""
    best_epoch, best = None, np.inf
    for entry in history:
        epoch = int(entry["epoch"])
        if epoch < int(first_selectable):
            continue
        score = float(entry["selection"]["mae"])
        if not np.isfinite(score):
            raise ValueError(f"Non-finite selection MAE at epoch {epoch}")
        if score < best:
            best_epoch, best = epoch, score
    if best_epoch is None:
        raise ValueError("No selectable epoch in the history")
    return best_epoch, best


# --------------------------------------------------------------------------- per-lead summaries
def lead_summary(prediction, truth, persistence, capacity):
    """Per-lead point metrics (plain numpy, same definitions as v25 ``point_metrics``)."""
    prediction, truth, persistence = (np.asarray(v, dtype=np.float64) for v in (prediction, truth, persistence))
    if prediction.ndim != 2 or prediction.shape != truth.shape or persistence.shape != truth.shape:
        raise ValueError("Aligned [window, lead] arrays required")
    return [{"lead_hours": h + 1, **point_metrics(prediction[:, h:h + 1], truth[:, h:h + 1],
                                                  persistence[:, h:h + 1], capacity)} for h in range(truth.shape[1])]


def record_from_metrics(fold, seed, arm, overall, leads, best_epoch=None, source=""):
    """Gate record: overall MAE/RMSE plus lead-1 MAE/RMSE (and the selected epoch for candidate arms)."""
    lead1 = next(row for row in leads if int(row["lead_hours"]) == 1)
    return {"fold": int(fold), "seed": int(seed), "arm": str(arm), "mae": float(overall["mae"]),
            "rmse": float(overall["rmse"]), "lead1_mae": float(lead1["mae"]), "lead1_rmse": float(lead1["rmse"]),
            "best_epoch": None if best_epoch is None else int(best_epoch), "source": source}


def correction_summary(prediction, anchor, truth):
    """How far the candidate moved from the frozen anchor, and whether the move helped (kW)."""
    prediction, anchor, truth = (np.asarray(v, dtype=np.float64) for v in (prediction, anchor, truth))
    if prediction.shape != anchor.shape or truth.shape != anchor.shape:
        raise ValueError("Aligned arrays required")
    moved = np.abs(prediction - anchor)
    gain = np.abs(anchor - truth) - np.abs(prediction - truth)
    changed = moved > 1e-9
    return {"mean_abs_correction_kw": float(moved.mean()), "max_abs_correction_kw": float(moved.max()),
            "mean_correction_kw": float((prediction - anchor).mean()),
            "share_points_changed_pct": float(100 * changed.mean()),
            "share_changed_points_helped_pct": float(100 * (gain[changed] > 0).mean()) if changed.any() else None,
            "mean_abs_error_gain_kw": float(gain.mean()),
            "per_lead_mean_abs_correction_kw": moved.mean(0).tolist() if moved.ndim == 2 else None}


# --------------------------------------------------------------------------- gates
def _finite(value, name):
    if value is None or not np.isfinite(float(value)):
        raise ValueError(f"Missing or non-finite metric: {name}")
    return float(value)


def _index(records):
    index = {}
    for record in records:
        key = (int(record["fold"]), int(record["seed"]), str(record["arm"]))
        if key in index:
            raise ValueError(f"Duplicate inner record {key}")
        index[key] = {**{k: record[k] for k in METRIC_KEYS + LEAD1_KEYS}, "best_epoch": record.get("best_epoch")}
    return index


def _mean(values):
    return float(np.mean([_finite(v, "aggregate") for v in values]))


def _missing(index, folds, seeds, arms):
    return [list(key) for key in ((f, s, a) for f in folds for s in seeds for a in arms) if key not in index]


def _persistence(persistence, folds):
    return {f: {key: _finite(persistence[f][key], f"persistence f{f} {key}") for key in METRIC_KEYS} for f in folds}


def _dlinear(index, folds, seed):
    """The frozen anchor is a single seed-2024 model; every residual seed is compared with it."""
    return {f: index[(f, seed, BASELINE)] for f in folds}


def is_trained_selection(record):
    """True only for a candidate whose selected state is a trained epoch (>= 1). Epoch 0 is the anchor itself."""
    epoch = record.get("best_epoch")
    return epoch is not None and int(epoch) >= 1


def not_worse(reference, candidate, tolerance=TIE_TOLERANCE_KW):
    """candidate_error - reference_error <= tolerance (non-inferiority up to a numeric tie)."""
    return _finite(candidate, "candidate") - _finite(reference, "reference") <= float(tolerance)


def _lead1_checks(candidate_macro, dlinear_macro, tolerance):
    return {"lead1_macro_mae_not_worse_than_dlinear": not_worse(dlinear_macro["lead1_mae"], candidate_macro["lead1_mae"], tolerance),
            "lead1_macro_rmse_not_worse_than_dlinear": not_worse(dlinear_macro["lead1_rmse"], candidate_macro["lead1_rmse"], tolerance),
            "lead1_mae_difference_kw": candidate_macro["lead1_mae"] - dlinear_macro["lead1_mae"],
            "lead1_rmse_difference_kw": candidate_macro["lead1_rmse"] - dlinear_macro["lead1_rmse"]}


def gate_g1(records, persistence, settings, folds, tolerance=TIE_TOLERANCE_KW):
    """Seed-2024 screen on internal selection. All must hold:
    macro MAE AND RMSE better than DLinear and Persistence; >= min folds jointly better than DLinear where a fold
    only counts if its selected state is a trained epoch; lead-1 macro MAE and RMSE not worse than DLinear."""
    seed = int(settings["seed"])
    index = _index(records)
    missing = _missing(index, folds, [seed], (PRIMARY, BASELINE))
    if missing:
        return {"gate": "G1", "passed": False, "complete": False, "missing": missing,
                "reason": "incomplete records are a failure, never a pass"}
    pers = _persistence(persistence, folds)
    dl = _dlinear(index, folds, seed)
    rows = []
    for f in folds:
        cand = index[(f, seed, PRIMARY)]
        trained = is_trained_selection(cand)
        rows.append({"fold": f, "candidate": cand, "dlinear": dl[f], "persistence": pers[f],
                     "selected_trained_epoch": trained,
                     "mae_gain_vs_dlinear_kw": dl[f]["mae"] - cand["mae"],
                     "rmse_gain_vs_dlinear_kw": dl[f]["rmse"] - cand["rmse"],
                     "lead1_mae_gain_vs_dlinear_kw": dl[f]["lead1_mae"] - cand["lead1_mae"],
                     "joint_vs_dlinear": bool(trained and joint_improvement(dl[f], cand, tolerance))})
    keys = METRIC_KEYS + LEAD1_KEYS
    macro = {"candidate": {k: _mean([r["candidate"][k] for r in rows]) for k in keys},
             "dlinear": {k: _mean([r["dlinear"][k] for r in rows]) for k in keys},
             "persistence": {k: _mean([r["persistence"][k] for r in rows]) for k in METRIC_KEYS}}
    joint_folds = int(sum(r["joint_vs_dlinear"] for r in rows))
    checks = {"macro_mae_rmse_better_than_dlinear": joint_improvement(macro["dlinear"], macro["candidate"], tolerance),
              "macro_mae_rmse_better_than_persistence": joint_improvement(macro["persistence"], macro["candidate"], tolerance),
              "jointly_better_folds_vs_dlinear": joint_folds,
              "min_jointly_better_folds_required": int(settings["min_folds_jointly_better_than_DLinear"]),
              "fold_requirement_met": joint_folds >= int(settings["min_folds_jointly_better_than_DLinear"]),
              **_lead1_checks(macro["candidate"], macro["dlinear"], tolerance),
              "folds_selecting_epoch0": [r["fold"] for r in rows if not r["selected_trained_epoch"]]}
    passed = (checks["macro_mae_rmse_better_than_dlinear"] and checks["macro_mae_rmse_better_than_persistence"]
              and checks["fold_requirement_met"] and checks["lead1_macro_mae_not_worse_than_dlinear"]
              and checks["lead1_macro_rmse_not_worse_than_dlinear"])
    return {"gate": "G1", "passed": bool(passed), "complete": True, "seed": seed, "rows": rows, "macro": macro,
            "checks": checks, "tolerance_kw": tolerance, "comparison_space": "internal_selection_only",
            "interpretation": "operational budget gate on development evidence; not a significance test"}


def info_i1(records, settings, folds, tolerance=TIE_TOLERANCE_KW):
    """Attribution evidence: AnchoredFull vs the same-parameter AnchoredPowerOnly (seed 2024)."""
    seed = int(settings["seed"])
    index = _index(records)
    missing = _missing(index, folds, [seed], (PRIMARY, INFO_CONTROL))
    if missing:
        return {"evidence": "I1", "supported": False, "complete": False, "missing": missing}
    rows = []
    for f in folds:
        full, power = index[(f, seed, PRIMARY)], index[(f, seed, INFO_CONTROL)]
        rows.append({"fold": f, "full": full, "power_only": power,
                     "joint_full_better": bool(is_trained_selection(full) and joint_improvement(power, full, tolerance))})
    macro = {name: {k: _mean([r[name][k] for r in rows]) for k in METRIC_KEYS} for name in ("full", "power_only")}
    joint = int(sum(r["joint_full_better"] for r in rows))
    need = int(settings.get("min_folds_jointly_better", 2))
    supported = joint_improvement(macro["power_only"], macro["full"], tolerance) and joint >= need
    return {"evidence": "I1", "supported": bool(supported), "complete": True, "seed": seed, "rows": rows,
            "macro": macro, "jointly_better_folds": joint, "min_folds_required": need,
            "interpretation": "auxiliary-channel package attribution only; never changes the primary candidate, never "
                              "authorises an outer fallback, not a per-channel causal effect (masking also changes "
                              "the LayerNorm input statistics)"}


def _seed_fold_means(index, arm, folds, seeds, keys):
    per_fold = {f: {k: _mean([index[(f, s, arm)][k] for s in seeds]) for k in keys} for f in folds}
    per_seed = {s: {k: _mean([index[(f, s, arm)][k] for f in folds]) for k in keys} for s in seeds}
    macro = {k: _mean([per_fold[f][k] for f in folds]) for k in keys}
    return per_fold, per_seed, macro


def gate_g2(records, persistence, settings, folds, anchor_seed, tolerance=TIE_TOLERANCE_KW):
    """Three residual seeds against the single frozen anchor: seed mean within fold, then equal fold mean; per-fold
    and per-seed joint counts (a fold/seed only counts if every contributing selection is a trained epoch); lead-1
    non-inferiority on the same aggregation."""
    seeds = [int(s) for s in settings["seeds"]]
    index = _index(records)
    missing = _missing(index, folds, seeds, (PRIMARY,)) + _missing(index, folds, [int(anchor_seed)], (BASELINE,))
    if missing:
        return {"gate": "G2", "passed": False, "complete": False, "missing": missing,
                "reason": "incomplete records are a failure, never a pass"}
    keys = METRIC_KEYS + LEAD1_KEYS
    pers = _persistence(persistence, folds)
    dl = _dlinear(index, folds, int(anchor_seed))
    cand_fold, cand_seed, cand_macro = _seed_fold_means(index, PRIMARY, folds, seeds, keys)
    ref_fold = {f: {k: dl[f][k] for k in keys} for f in folds}
    ref_macro = {k: _mean([ref_fold[f][k] for f in folds]) for k in keys}
    pers_macro = {k: _mean([pers[f][k] for f in folds]) for k in METRIC_KEYS}
    trained = {(f, s): is_trained_selection(index[(f, s, PRIMARY)]) for f in folds for s in seeds}
    fold_joint = {f: bool(all(trained[(f, s)] for s in seeds) and joint_improvement(ref_fold[f], cand_fold[f], tolerance))
                  for f in folds}
    seed_joint = {s: bool(all(trained[(f, s)] for f in folds) and joint_improvement(ref_macro, cand_seed[s], tolerance))
                  for s in seeds}
    checks = {"macro_mae_rmse_better_than_dlinear": joint_improvement(ref_macro, cand_macro, tolerance),
              "macro_mae_rmse_better_than_persistence": joint_improvement(pers_macro, cand_macro, tolerance),
              "seed_averaged_folds_jointly_better": int(sum(fold_joint.values())),
              "min_seed_averaged_folds": int(settings["min_seed_averaged_folds_jointly_better_than_DLinear"]),
              "fold_averaged_seeds_jointly_better": int(sum(seed_joint.values())),
              "min_fold_averaged_seeds": int(settings["min_fold_averaged_seeds_jointly_better_than_DLinear"]),
              **_lead1_checks(cand_macro, ref_macro, tolerance),
              "fold_seed_units_selecting_epoch0": [list(k) for k, v in trained.items() if not v]}
    passed = (checks["macro_mae_rmse_better_than_dlinear"] and checks["macro_mae_rmse_better_than_persistence"]
              and checks["seed_averaged_folds_jointly_better"] >= checks["min_seed_averaged_folds"]
              and checks["fold_averaged_seeds_jointly_better"] >= checks["min_fold_averaged_seeds"]
              and checks["lead1_macro_mae_not_worse_than_dlinear"] and checks["lead1_macro_rmse_not_worse_than_dlinear"])
    return {"gate": "G2", "passed": bool(passed), "complete": True, "seeds": seeds, "anchor_seed": int(anchor_seed),
            "aggregation": "equal mean of residual seeds within fold, then equal mean of folds; the anchor is one model",
            "candidate": {"per_fold_seed_mean": cand_fold, "per_seed_fold_mean": cand_seed, "macro": cand_macro},
            "dlinear": {"per_fold": ref_fold, "macro": ref_macro},
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
    full_fold, _, full_macro = _seed_fold_means(index, PRIMARY, folds, seeds, METRIC_KEYS)
    po_fold, _, po_macro = _seed_fold_means(index, INFO_CONTROL, folds, seeds, METRIC_KEYS)
    fold_joint = {f: bool(any(is_trained_selection(index[(f, s, PRIMARY)]) for s in seeds)
                          and joint_improvement(po_fold[f], full_fold[f], tolerance)) for f in folds}
    need = int(settings.get("min_seed_averaged_folds_jointly_better", 2))
    supported = joint_improvement(po_macro, full_macro, tolerance) and sum(fold_joint.values()) >= need
    return {"evidence": "I2", "supported": bool(supported), "complete": True, "seeds": seeds,
            "full": {"per_fold_seed_mean": full_fold, "macro": full_macro},
            "power_only": {"per_fold_seed_mean": po_fold, "macro": po_macro},
            "fold_joint_full_better": fold_joint, "jointly_better_folds": int(sum(fold_joint.values())),
            "min_folds_required": need,
            "interpretation": "auxiliary-channel package attribution only; not a per-channel causal effect"}


# --------------------------------------------------------------------------- data access audit
class DataAccessLog(_V25DataAccessLog):
    """v26 stage map; validation only in A3 after a registered passed G2, never the sealed test."""

    def request(self, stage, subset, fold, *, g2_passed=False):
        if subset in ("test", "sealed_test"):
            raise PermissionError("The sealed test suffix is never requested in v26")
        allowed = STAGE_SUBSETS.get(stage)
        if allowed is None or subset not in allowed:
            raise PermissionError(f"Stage {stage} may not request subset {subset!r}")
        if subset == "validation" and not g2_passed:
            raise PermissionError("Outer validation requires a registered passed G2")
        key = (stage, subset, int(fold))
        if key not in self.requests:
            append_event(self.path, "data_subset_requested", stage=stage, subset=subset, fold=int(fold))
        self.requests.append(key)
