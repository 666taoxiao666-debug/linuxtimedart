#!/usr/bin/env python3
"""One predeclared training-period physical guard; no new neural training or outer eval."""
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
from utils.physical_scale_guard import three_blocks, fit_guard, apply_guard
from utils.residual_scale_calibration import apply_scale
from utils.sdwpf_logging import allocate_log_directory, finish_log_directory


def run(source):
    drift = read_json(source / "protocol.json")
    completed = read_json(source / "completed.json")
    calibration_dir = Path(drift["source_dir"])
    scale_protocol = read_json(calibration_dir / "protocol.json")
    model = read_json(calibration_dir / "calibration.json")
    predictions = Path(scale_protocol["source_audit"]) / "reference_predictions.npz"
    checkpoint = Path(scale_protocol["source_checkpoint"]["checkpoint"])
    if (drift["protocol_id"] != "sdwpf_train_only_fixed_residual_drift_audit"
            or drift["outer_validation_used"] or drift["sealed_test_evaluated"]
            or sha256(predictions) != drift["source_predictions_sha256"]
            or sha256(source / "history_context.npz") != completed["context_sha256"]
            or sha256(source / "report.json") != completed["report_sha256"]
            or sha256(calibration_dir / "calibration.json") != drift["calibration_sha256"]
            or sha256(calibration_dir / "protocol.json") != drift["source_protocol_sha256"]
            or sha256(checkpoint) != drift["source_checkpoint_sha256"]):
        raise ValueError("Training-only diagnostic/calibration identity changed")
    protocol = dict(protocol_id="sdwpf_train_physical_scale_guard_pilot_v8", source_dir=str(source),
        source_protocol_sha256=sha256(source / "protocol.json"), context_sha256=completed["context_sha256"],
        source_predictions_sha256=drift["source_predictions_sha256"],
        source_checkpoint_sha256=drift["source_checkpoint_sha256"], calibration_sha256=drift["calibration_sha256"],
        split="first 50pct calendar alpha fit; next 25pct guard calibration; last 25pct check; purge crossing targets",
        min_windows_per_training_block=64, acceptance="both training blocks have strictly positive MAE and MSE gains",
        alpha_reselected=False, backbone_retrained=False, new_candidates_or_seeds=False,
        history_bins="frozen drift diagnostic power/wind bins", fallback="unchanged source predictor",
        outer_validation_used=False, sealed_test_evaluated=False, utility_wiki=False,
        evidence_status="exploratory original-train development; previous full summaries inspected",
        no_outer_followup_automatically=True)
    pointer = ROOT / "outputs/logs/SDWPF/physical_scale_guard_latest.txt"
    if pointer.exists():
        previous = Path(pointer.read_text(encoding="utf-8").strip())
        if read_json(previous / "protocol.json") != protocol:
            raise ValueError("Existing pilot uses another protocol; no silent rerun")
        if not (previous / "result.json").exists():
            raise ValueError("Incomplete pilot; inspect before retry")
        print(f"[PHYSICAL GUARD] Already complete: {previous}; nothing refitted", flush=True)
        return
    directory = allocate_log_directory(task="physical_scale_guard",
        parameters="h12_f1_s2024_train_50-25-25_min64_fixed_scale_cpu", run_id="physical_scale_guard_train_v8")
    write_json(directory / "protocol.json", protocol)
    pointer.write_text(str(directory) + "\n", encoding="utf-8")
    print(f"[PHYSICAL GUARD] RESULT_DIR={directory}", flush=True)
    try:
        with np.load(predictions, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        with np.load(source / "history_context.npz", allow_pickle=False) as saved:
            bins = saved["joint_bin"]
            np.testing.assert_array_equal(saved["window_starts"], arrays["window_starts"])
        early, calibration, check, midpoint, check_start = three_blocks(arrays["target_timestamps"],
            drift["selection_end"], drift["end_exclusive"])
        keys = ("prediction", "persistence", "truth", "trend")
        guard = fit_guard(model, [({key: arrays[key][mask] for key in keys}, bins[mask])
                                   for mask in (early, calibration)])
        write_json(directory / "guard.json", guard)
        ref, anchor, truth, state = (arrays[k][check] for k in keys)
        static = apply_scale(model, ref, anchor, state)
        prediction = apply_guard(guard, ref, anchor, state, bins[check])
        scores = {name: metrics(value, truth, anchor) for name, value in
                  (("reference", ref), ("static_scale", static), ("physical_guard", prediction), ("persistence", anchor))}
        changed = np.abs(prediction - ref) > 1e-10
        window_harm = np.abs(prediction - truth).mean(axis=1) > np.abs(ref - truth).mean(axis=1)
        altered_windows = changed.any(axis=1)
        report = dict(protocol=protocol, midpoint=str(midpoint), check_start=str(check_start),
            early_windows=int(early.sum()), calibration_windows=int(calibration.sum()), check_windows=int(check.sum()),
            purged_windows=int((~(early | calibration | check)).sum()),
            check_target_min=str(arrays["target_timestamps"][check].min()),
            check_target_max=str(arrays["target_timestamps"][check].max()),
            accepted_physical_cells=int(np.asarray(guard["trust"]).sum()), check=scores,
            changed_point_pct=float(100 * changed.mean()), changed_window_pct=float(100 * altered_windows.mean()),
            abstain_window_pct=float(100 * (~altered_windows).mean()),
            harm_changed_window_pct=float(100 * window_harm[altered_windows].mean()) if altered_windows.any() else None,
            selected_point_gain_kw=float((np.abs(ref - truth) - np.abs(prediction - truth))[changed].mean()) if changed.any() else None,
            gain_mae_vs_reference_kw=scores["reference"]["mae"] - scores["physical_guard"]["mae"],
            gain_rmse_vs_reference_kw=scores["reference"]["rmse"] - scores["physical_guard"]["rmse"],
            joint_improvement_vs_reference=all(scores["physical_guard"][k] < scores["reference"][k] for k in ("mae", "rmse")),
            next_action="no automatic outer evaluation; assess this single training-side mechanism")
        np.savez_compressed(directory / "check_predictions.npz", prediction=prediction, reference=ref,
            static_scale=static, persistence=anchor, truth=truth, target_timestamps=arrays["target_timestamps"][check])
        from utils.forecast_report import _plot_accuracy_overview
        _plot_accuracy_overview(prediction, truth, str(directory), persistence=anchor, reference=ref,
            rated_power=1500, model_name="Physical support guard", scope_caption=
            f"Exploratory original-train final block only; NOT outer validation or test; {int(check.sum())} h12 windows\n"
            f"Same frozen checkpoint {drift['source_checkpoint_sha256'][:10]}; no neural refit; fixed original scale + two-block physical support")
        write_json(directory / "result.json", report)
        write_json(directory / "completed.json", dict(result_sha256=sha256(directory / "result.json"),
            guard_sha256=sha256(directory / "guard.json"), predictions_sha256=sha256(directory / "check_predictions.npz")))
        finish_log_directory(directory, 0)
        print("[PHYSICAL GUARD] " + str({k: report[k] for k in
            ("accepted_physical_cells", "check_windows", "gain_mae_vs_reference_kw", "gain_rmse_vs_reference_kw", "joint_improvement_vs_reference")}), flush=True)
    except Exception:
        finish_log_directory(directory, 1)
        raise


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drift-dir", required=True, type=Path)
    args = parser.parse_args()
    with (ROOT / "outputs/logs/SDWPF/physical_scale_guard.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.drift_dir.resolve())


if __name__ == "__main__":
    main()
