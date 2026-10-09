"""Train-only objective/epoch selection for a bounded SDWPF search."""
from __future__ import annotations

import math
import numpy as np


def verify_inner_manifest(manifest, plan, expected_stage):
    args = manifest["args"]
    if manifest.get("stage") != expected_stage or manifest.get("extra", {}).get("status") != "complete":
        raise ValueError("Inner stage is not complete")
    expected = {"data": "SDWPF", "model": "PromptTimeDART", "prompt_router": "trend",
                "sdwpf_split": "time_ratio", "sdwpf_fold": plan["sdwpf_fold"],
                "seed": plan["seed"]}
    for key, value in expected.items():
        if args.get(key) != value:
            raise ValueError(f"Inner provenance mismatch: {key}")
    if args.get("utility_wiki") or args.get("evaluate_test_after_train"):
        raise ValueError("Inner search requires plain trend and no test evaluation")
    if manifest.get("data_file", {}).get("sha256") != plan["outer_data_sha256"]:
        raise ValueError("Inner and outer dataset hashes differ")
    for name in ("train", "val"):
        if not math.isclose(float(args[f"sdwpf_{name}_ratio"]), float(plan[f"inner_{name}_ratio"]), abs_tol=1e-12):
            raise ValueError("Inner split ratios changed")
    boundaries = manifest["datasets"]["train"]
    # Dataset manifests use NumPy's nine-digit nanosecond timestamps; Python
    # 3.10 datetime.fromisoformat only accepts three or six fractional digits.
    fit = np.datetime64(boundaries["train_cutoff"], "ns")
    selection = np.datetime64(boundaries["val_cutoff"], "ns")
    outer = np.datetime64(plan["outer_train_cutoff"], "ns")
    if any(np.isnat(value) for value in (fit, selection, outer)):
        raise ValueError("Inner boundaries must be real timestamps")
    if not fit < selection < outer:
        raise ValueError("Inner selection touches outer validation")


def choose_candidate(histories, candidate_order, schedule_epochs):
    """Constrain MAE and RMSE jointly, then rank on train-only holdout error.

    ``histories`` contains the inner histories only, never an outer result.
    Each row is the numeric training_history.csv record for an epoch.
    """
    if set(histories) != set(candidate_order) or not candidate_order or candidate_order[0] != "reference":
        raise ValueError("The complete frozen candidate matrix is required")
    rows = {}
    for name in candidate_order:
        eligible = []
        for raw in histories[name]:
            epoch = int(raw["epoch"])
            if epoch == 0:
                continue
            if not 1 <= epoch <= schedule_epochs:
                raise ValueError("Candidate epoch is outside the frozen schedule")
            row = {"candidate_id": name, "epoch": epoch}
            for key in ("val_mae_kw", "val_rmse_kw", "val_r2_original", "val_persistence_mae_kw", "val_persistence_rmse_kw"):
                value = float(raw[key])
                if not math.isfinite(value) or (key != "val_r2_original" and value <= 0):
                    raise ValueError("Candidate metrics must be finite and errors positive")
                row[key] = value
            row["balanced_ratio"] = 0.5 * (
                row["val_mae_kw"] / row["val_persistence_mae_kw"]
                + row["val_rmse_kw"] / row["val_persistence_rmse_kw"])
            eligible.append(row)
        if len(eligible) != schedule_epochs or len({row["epoch"] for row in eligible}) != schedule_epochs:
            raise ValueError(f"Candidate {name} has incomplete or duplicate epochs")
        rows[name] = eligible
    reference = min(rows["reference"], key=lambda row: (row["val_mae_kw"], row["epoch"]))
    reference_persistence = (reference["val_persistence_mae_kw"], reference["val_persistence_rmse_kw"])
    feasible = []
    for name in candidate_order:
        for row in rows[name]:
            if any(not math.isclose(row[key], value, rel_tol=1e-8) for key, value in zip(
                    ("val_persistence_mae_kw", "val_persistence_rmse_kw"), reference_persistence)):
                raise ValueError("Candidates do not share the same inner evaluation targets")
            if (row["val_mae_kw"] <= reference["val_mae_kw"]
                    and row["val_rmse_kw"] <= reference["val_rmse_kw"]):
                feasible.append(row)
    chosen = min(feasible, key=lambda row: (
        row["balanced_ratio"], candidate_order.index(row["candidate_id"]), row["epoch"]))
    return {
        "selection_source": "inner_train_period_only",
        "selected": chosen,
        "reference": reference,
        "inner_improves_both": (
            chosen["val_mae_kw"] < reference["val_mae_kw"]
            and chosen["val_rmse_kw"] < reference["val_rmse_kw"]),
        "candidate_count": len(candidate_order),
        "outer_validation_used_for_selection": False,
        "sealed_test_accessed": False,
    }
