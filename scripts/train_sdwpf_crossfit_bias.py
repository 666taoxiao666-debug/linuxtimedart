#!/usr/bin/env python3
"""One train-only, three-block horizon bias pilot; never reads outer validation/test."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from scripts.tune_sdwpf_accuracy import read_json, write_json, sha256
from scripts.train_sdwpf_residual_scale import metrics
from utils.crossfit_bias_calibration import fit_bias_guard, apply_bias
from utils.physical_scale_guard import three_blocks
from utils.sdwpf_logging import allocate_log_directory, finish_log_directory


def run(source: Path):
    drift = read_json(source / "protocol.json")
    completed = read_json(source / "completed.json")
    calibration_dir = Path(drift["source_dir"])
    scale_protocol = read_json(calibration_dir / "protocol.json")
    predictions = Path(scale_protocol["source_audit"]) / "reference_predictions.npz"
    checkpoint = Path(scale_protocol["source_checkpoint"]["checkpoint"])
    if (
        drift["protocol_id"] != "sdwpf_train_only_fixed_residual_drift_audit"
        or drift["outer_validation_used"]
        or drift["sealed_test_evaluated"]
        or sha256(predictions) != drift["source_predictions_sha256"]
        or sha256(source / "history_context.npz") != completed["context_sha256"]
        or sha256(source / "report.json") != completed["report_sha256"]
        or sha256(calibration_dir / "protocol.json") != drift["source_protocol_sha256"]
        or sha256(checkpoint) != drift["source_checkpoint_sha256"]
    ):
        raise ValueError("Training-only source identity changed")

    protocol = {
        "protocol_id": "sdwpf_train_crossfit_horizon_bias_pilot_v9",
        "source_dir": str(source),
        "source_protocol_sha256": sha256(source / "protocol.json"),
        "source_predictions_sha256": drift["source_predictions_sha256"],
        "source_checkpoint_sha256": drift["source_checkpoint_sha256"],
        "split": "first 50pct fit magnitude; next 25pct accept/reject; last 25pct one-time check; purge crossing targets",
        "correction": "per-horizon additive negative median error from first block",
        "max_abs_correction": "5pct rated power = 75kW",
        "min_windows_per_training_block": 64,
        "acceptance": "same fixed correction has strictly positive MAE and RMSE gain in first and second blocks",
        "prediction_bounds_kw": [0.0, 1500.0],
        "backbone_retrained": False,
        "wiki_modified": False,
        "new_candidates_or_seeds": False,
        "outer_validation_used": False,
        "sealed_test_evaluated": False,
        "evidence_status": "exploratory original-train development; final check block not used for fitting or selection",
        "no_outer_followup_automatically": True,
    }
    pointer = ROOT / "outputs/logs/SDWPF/crossfit_bias_latest.txt"
    if pointer.exists():
        previous = Path(pointer.read_text(encoding="utf-8").strip())
        if read_json(previous / "protocol.json") != protocol:
            raise ValueError("Existing pilot uses another protocol; no silent rerun")
        if not (previous / "completed.json").exists():
            raise ValueError("Incomplete pilot; inspect before retry")
        print(f"[CROSSFIT BIAS] Already complete: {previous}; nothing refitted", flush=True)
        return previous

    directory = allocate_log_directory(
        task="crossfit_bias",
        parameters="h12_f1_s2024_train_50-25-25_cap75_min64_cpu",
        run_id="crossfit_horizon_bias_v9",
    )
    write_json(directory / "protocol.json", protocol)
    pointer.write_text(str(directory) + "\n", encoding="utf-8")
    print(f"[CROSSFIT BIAS] RESULT_DIR={directory}", flush=True)
    try:
        with np.load(predictions, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        early, calibration, check, midpoint, check_start = three_blocks(
            arrays["target_timestamps"], drift["selection_end"], drift["end_exclusive"]
        )
        model = fit_bias_guard(
            [(arrays["prediction"][mask], arrays["truth"][mask])
             for mask in (early, calibration)],
            rated_power=1500.0,
            max_capacity_ratio=0.05,
            min_windows=64,
        )
        write_json(directory / "bias_model.json", model)
        reference = arrays["prediction"][check]
        truth = arrays["truth"][check]
        persistence = arrays["persistence"][check]
        prediction = apply_bias(model, reference)
        scores = {
            name: metrics(value, truth, persistence)
            for name, value in (
                ("reference", reference),
                ("crossfit_bias", prediction),
                ("persistence", persistence),
            )
        }
        changed = np.abs(prediction - reference) > 1e-10
        changed_windows = changed.any(axis=1)
        window_harm = np.abs(prediction - truth).mean(axis=1) > np.abs(reference - truth).mean(axis=1)
        report = {
            "protocol": protocol,
            "midpoint": str(midpoint),
            "check_start": str(check_start),
            "early_windows": int(early.sum()),
            "calibration_windows": int(calibration.sum()),
            "check_windows": int(check.sum()),
            "purged_windows": int((~(early | calibration | check)).sum()),
            "check_target_min": str(arrays["target_timestamps"][check].min()),
            "check_target_max": str(arrays["target_timestamps"][check].max()),
            "accepted_horizons": int(np.sum(model["accepted"])),
            "check": scores,
            "changed_point_pct": float(100.0 * changed.mean()),
            "changed_window_pct": float(100.0 * changed_windows.mean()),
            "abstain_window_pct": float(100.0 * (~changed_windows).mean()),
            "harm_changed_window_pct": (
                float(100.0 * window_harm[changed_windows].mean()) if changed_windows.any() else None
            ),
            "selected_point_gain_kw": (
                float((np.abs(reference - truth) - np.abs(prediction - truth))[changed].mean())
                if changed.any() else None
            ),
            "gain_mae_vs_reference_kw": scores["reference"]["mae"] - scores["crossfit_bias"]["mae"],
            "gain_rmse_vs_reference_kw": scores["reference"]["rmse"] - scores["crossfit_bias"]["rmse"],
            "joint_improvement_vs_reference": all(
                scores["crossfit_bias"][key] < scores["reference"][key]
                for key in ("mae", "rmse")
            ),
            "next_action": "no automatic outer evaluation; stop if joint benefit is absent or negligible",
        }
        np.savez_compressed(
            directory / "check_predictions.npz",
            prediction=prediction,
            reference=reference,
            persistence=persistence,
            truth=truth,
            target_timestamps=arrays["target_timestamps"][check],
        )
        from utils.forecast_report import _plot_accuracy_overview
        _plot_accuracy_overview(
            prediction,
            truth,
            str(directory),
            persistence=persistence,
            reference=reference,
            rated_power=1500,
            model_name="Cross-fit horizon bias",
            scope_caption=(
                f"Exploratory original-train final block only; NOT outer validation or test; {int(check.sum())} h12 windows\n"
                f"Magnitude fit on first 50%, support-gated on next 25%; same frozen checkpoint {drift['source_checkpoint_sha256'][:10]}"
            ),
        )
        write_json(directory / "result.json", report)
        write_json(directory / "completed.json", {
            "result_sha256": sha256(directory / "result.json"),
            "model_sha256": sha256(directory / "bias_model.json"),
            "predictions_sha256": sha256(directory / "check_predictions.npz"),
        })
        finish_log_directory(directory, 0)
        print("[CROSSFIT BIAS] " + str({
            key: report[key] for key in (
                "accepted_horizons", "check_windows", "gain_mae_vs_reference_kw",
                "gain_rmse_vs_reference_kw", "joint_improvement_vs_reference",
            )
        }), flush=True)
    except Exception:
        finish_log_directory(directory, 1)
        raise
    return directory


def main():
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drift-dir", required=True, type=Path)
    args = parser.parse_args()
    lock_path = ROOT / "outputs/logs/SDWPF/crossfit_bias.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.drift_dir.resolve())


if __name__ == "__main__":
    main()
