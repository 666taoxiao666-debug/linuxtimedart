#!/usr/bin/env python3
"""Frozen-checkpoint CPU calibration pilot; never modify the diagnostic source."""
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
from utils.residual_scale_calibration import chronological_masks, fit_scale, apply_scale, SCALES, MIN_WINDOWS
from utils.metrics import forecast_metrics
from utils.sdwpf_logging import allocate_log_directory, finish_log_directory


def metrics(prediction, truth, persistence):
    result = forecast_metrics(prediction, truth, rated_power=1500)
    benchmark = forecast_metrics(persistence, truth)
    result["mae_skill_pct"] = 100 * (1 - result["mae"] / benchmark["mae"]) if benchmark["mae"] > 0 else None
    result["rmse_skill_pct"] = 100 * (1 - result["rmse"] / benchmark["rmse"]) if benchmark["rmse"] > 0 else None
    return {key: (None if isinstance(value, float) and not np.isfinite(value) else value)
            for key, value in result.items()}


def render_result(directory, source):
    """Reproduce the saved plot, with numerical checks and absolutely no fit."""
    protocol, model, report = (read_json(directory / name) for name in
                               ("protocol.json", "calibration.json", "result.json"))
    audit = read_json(source / "protocol.json")
    predictions = source / "reference_predictions.npz"
    if (sha256(predictions) != protocol["source_predictions_sha256"]
            or sha256(source / "protocol.json") != protocol["source_audit_protocol_sha256"]
            or model["source_checkpoint_sha256"] != protocol["source_checkpoint"]["checkpoint_sha256"]):
        raise ValueError("Saved calibration/plot source identity changed")
    with np.load(predictions, allow_pickle=False) as arrays:
        reference, persistence, truth, trend, times = (arrays[key] for key in
            ("prediction", "persistence", "truth", "trend", "target_timestamps"))
    _, check, _ = chronological_masks(times, audit["selection_end"], audit["oof_end_exclusive"])
    prediction = apply_scale(model, reference[check], persistence[check], trend[check])
    actual = metrics(prediction, truth[check], persistence[check])
    calibrated = report["check"]["calibrated"]
    for key in ("mae", "rmse", "r2", "mae_skill_pct", "rmse_skill_pct",
                "within_5pct_capacity_pct", "within_10pct_capacity_pct"):
        np.testing.assert_allclose(actual[key], calibrated[key], rtol=1e-10, atol=1e-10)
    if int(check.sum()) != report["check_windows"]:
        raise ValueError("Plot check windows differ from the saved experiment")
    from utils.forecast_report import _plot_accuracy_overview
    check_times = times[check]
    caption = (f"Source: frozen checkpoint {model['source_checkpoint_sha256'][:10]}; "
        f"original-train OOF later block, {str(check_times.min())[:16]} to {str(check_times.max())[:16]}\n"
        f"Exploratory, NOT outer validation or test; {int(check.sum())} windows, all h12 points. "
        f"Capacity tolerance hits: ±5% {calibrated['within_5pct_capacity_pct']:.2f}%, "
        f"±10% {calibrated['within_10pct_capacity_pct']:.2f}%")
    _plot_accuracy_overview(prediction, truth[check], str(directory),
        persistence=persistence[check], rated_power=1500, model_name="Residual scale calibration",
        scope_caption=caption)
    print("[SCALE] Saved metrics reproduced; plot only, no fitting", flush=True)


def run(source):
    protocol = read_json(source / "protocol.json")
    if protocol["protocol_id"] != "sdwpf_f1_s2024_train_tail_error_audit_v1":
        raise ValueError("Only the frozen train-tail error audit is supported")
    source_state = read_json(source / "reference_completed.json")
    predictions = source / "reference_predictions.npz"
    if sha256(predictions) != source_state["predictions_sha256"]:
        raise ValueError("Source predictions changed")
    checkpoint = Path(source_state["source"]["checkpoint"])
    if sha256(checkpoint) != source_state["source"]["checkpoint_sha256"]:
        raise ValueError("Source forecast identity changed; calibration cannot be transferred")
    pointer = ROOT / "outputs/logs/SDWPF/residual_scale_latest.txt"
    if pointer.exists():
        previous = Path(pointer.read_text(encoding="utf-8").strip())
        old = read_json(previous / "protocol.json")
        if old["source_predictions_sha256"] != sha256(predictions):
            raise ValueError("Existing pilot uses another source; do not silently rerun")
        if not (previous / "result.json").exists():
            raise ValueError("Existing incomplete pilot; inspect before any retry")
        print(f"[SCALE] Already complete: {previous}; no fit repeated", flush=True)
        return previous
    directory = allocate_log_directory(task="residual_scale_pilot",
        parameters="h12_f1_s2024_train_midpoint_grid5_cpu", run_id="train_tail_residual_scale_v7")
    # Preserve the exact plan before fitting or looking at check-block scores.
    write_json(directory / "protocol.json", dict(protocol_id="sdwpf_train_tail_residual_scale_pilot_v7",
        source_audit=str(source), source_audit_protocol_sha256=sha256(source / "protocol.json"),
        source_predictions_sha256=sha256(predictions), source_checkpoint=source_state["source"],
        scales=list(SCALES), min_fit_windows=MIN_WINDOWS, states=[-1, 0, 1],
        history_steps=protocol["event_thresholds"]["recent_steps"], history_change_threshold_capacity_ratio=.05,
        split="fixed_time_midpoint; purge crossing target windows; earlier fit/later check",
        selection_rule="fit-only MAE and RMSE both no worse; minimize mean normalized error ratio",
        bias_intercept=False, amplify_residual=False, backbone_frozen=True, utility_wiki=False,
        outer_validation_used=False, sealed_test_evaluated=False,
        evidence_status="exploratory original-train development; full-audit summaries already inspected",
        no_additional_candidates=True, no_reselection_on_check_block=True))
    pointer.write_text(str(directory) + "\n", encoding="utf-8")
    print(f"[SCALE] RESULT_DIR={directory}", flush=True)
    try:
        with np.load(predictions, allow_pickle=False) as arrays:
            reference, persistence, truth, trend, times = (arrays[key] for key in
                ("prediction", "persistence", "truth", "trend", "target_timestamps"))
        fit, check, midpoint = chronological_masks(times, protocol["selection_end"], protocol["oof_end_exclusive"])
        model = fit_scale(reference[fit], persistence[fit], truth[fit], trend[fit])
        model["source_checkpoint_sha256"] = source_state["source"]["checkpoint_sha256"]
        write_json(directory / "calibration.json", model)
        prediction = apply_scale(model, reference, persistence, trend)
        report = dict(protocol_id="sdwpf_train_tail_residual_scale_pilot_v7", midpoint=str(midpoint),
            fit_windows=int(fit.sum()), check_windows=int(check.sum()), purged_crossing_windows=int((~(fit | check)).sum()),
            outer_validation_used=False, sealed_test_evaluated=False, utility_wiki=False,
            source_checkpoint_sha256=model["source_checkpoint_sha256"], fitting_source="earlier_train_oof_only")
        for name, selected in (("fit", fit), ("check", check)):
            report[name] = {"reference": metrics(reference[selected], truth[selected], persistence[selected]),
                "calibrated": metrics(prediction[selected], truth[selected], persistence[selected]),
                "persistence": metrics(persistence[selected], truth[selected], persistence[selected])}
        ref, calibrated = report["check"]["reference"], report["check"]["calibrated"]
        report["check_gain_mae_kw"], report["check_gain_rmse_kw"] = ref["mae"] - calibrated["mae"], ref["rmse"] - calibrated["rmse"]
        report["check_joint_improvement"] = calibrated["mae"] < ref["mae"] and calibrated["rmse"] < ref["rmse"]
        report["next_action"] = "request_matched_independent_confirmation" if report["check_joint_improvement"] else "stop_mechanism"
        report["do_not_transfer_to_another_backbone"] = True
        np.savez_compressed(directory / "check_predictions.npz", prediction=prediction[check],
            reference=reference[check], persistence=persistence[check], truth=truth[check],
            target_timestamps=times[check], trend=trend[check])
        write_json(directory / "result.json", report)
        render_result(directory, source)
        finish_log_directory(directory, 0)
        print("[SCALE] check: " + str({key: report[key] for key in
            ("fit_windows", "check_windows", "check_gain_mae_kw", "check_gain_rmse_kw", "check_joint_improvement", "next_action")}), flush=True)
    except Exception:
        finish_log_directory(directory, 1)
        raise
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, help="Replot an existing completed pilot; never train")
    args = parser.parse_args()
    if args.render_dir is not None:
        render_result(args.render_dir.resolve(), args.audit_dir.resolve())
        return
    import fcntl
    lock_path = ROOT / "outputs/logs/SDWPF/residual_scale_pilot.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.audit_dir.resolve())


if __name__ == "__main__":
    main()
