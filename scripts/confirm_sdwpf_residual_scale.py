#!/usr/bin/env python3
"""One frozen-checkpoint outer-val follow-up; no fitting or sealed-test loader."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd

from scripts.audit_sdwpf_train_errors import restore_cpu, infer
from scripts.tune_sdwpf_accuracy import read_json, write_json, sha256
from scripts.train_sdwpf_residual_scale import metrics
from utils.residual_scale_calibration import apply_scale
from utils.paired_forecast_statistics import paired_forecast_statistics
from utils.wind_wiki_oof import forward_oof_starts
from utils.sdwpf_logging import finish_log_directory


def frozen_sources(source):
    """Bind the follow-up to the exact saved calibration, forecast and data."""
    scale_protocol = read_json(source / "protocol.json")
    model = read_json(source / "calibration.json")
    result = read_json(source / "result.json")
    if (scale_protocol["protocol_id"] != "sdwpf_train_tail_residual_scale_pilot_v7"
            or not result["check_joint_improvement"] or result["outer_validation_used"]
            or result["sealed_test_evaluated"] or result["utility_wiki"]):
        raise ValueError("Only the completed train-only v7 calibration is supported")
    audit_dir = Path(scale_protocol["source_audit"])
    audit = read_json(audit_dir / "protocol.json")
    if (sha256(audit_dir / "protocol.json") != scale_protocol["source_audit_protocol_sha256"]
            or sha256(audit_dir / "reference_predictions.npz") != scale_protocol["source_predictions_sha256"]):
        raise ValueError("Calibration training evidence changed")
    checkpoint = Path(scale_protocol["source_checkpoint"]["checkpoint"])
    expected_hash = scale_protocol["source_checkpoint"]["checkpoint_sha256"]
    if (sha256(checkpoint) != expected_hash or model["source_checkpoint_sha256"] != expected_hash
            or result["source_checkpoint_sha256"] != expected_hash
            or sha256(checkpoint.parent / "run_manifest.json") != scale_protocol["source_checkpoint"]["manifest_sha256"]):
        raise ValueError("Forecast/checkpoint identity changed; no transplant allowed")
    manifest = read_json(checkpoint.parent / "run_manifest.json")
    outer_checkpoint = Path(audit["source_plan"]["outer_trend_checkpoint"])
    outer_path = outer_checkpoint.parent / "run_manifest.json"
    outer = read_json(outer_path)
    args = manifest["args"]
    if (args.get("prompt_router") != "trend" or args.get("utility_wiki")
            or (args.get("sdwpf_fold"), args.get("seed"), args.get("seq_len"), args.get("pred_len")) != (1, 2024, 336, 12)
            or args.get("sdwpf_split") != "time_ratio" or manifest["extra"]["best_epoch"] != 8
            or outer["args"].get("sdwpf_split") != "rolling_holdout"
            or outer["args"].get("sdwpf_fold") != 1 or outer["args"].get("seed") != 2024
            or manifest["data_file"]["sha256"] != outer["data_file"]["sha256"]
            or manifest["data_file"]["sha256"] != audit["source_plan"]["outer_data_sha256"]):
        raise ValueError("Source is not the frozen f1 s2024 plain-trend pair")
    # Reconstruction must retain the inner scaler. The outer manifest supplies
    # validation boundaries and window identities, NOT forecast parameters.
    data_keys = ("feature_columns", "rated_power", "sdwpf_clip_power", "sdwpf_filter_abnormal",
                 "sdwpf_causal_fill", "sdwpf_keep_curtailment", "sdwpf_robust_pitch",
                 "sdwpf_physics_features", "sdwpf_drop_weak", "sdwpf_eval_stride")
    for key in data_keys:
        # The historical manifest predates the opt-in pitch repair flag.
        # Its missing field means the established default False, not unknown.
        default = False if key == "sdwpf_robust_pitch" else None
        if args.get(key, default) != outer["args"].get(key, default):
            raise ValueError(f"Outer window convention differs: {key}")
    boundaries = outer["datasets"]["val"]
    start, end, sealed = (np.datetime64(boundaries[key], "ns") for key in
                           ("train_cutoff", "val_cutoff", "test_start_cutoff"))
    if (any(np.isnat(t) for t in (start, end, sealed)) or not start < end <= sealed
            or start != np.datetime64(audit["oof_end_exclusive"], "ns")):
        raise ValueError("Outer validation overlaps calibration or sealed test")
    apply_scale(model, np.ones((1, 12)), np.zeros((1, 12)), np.array([0]))
    protocol = dict(protocol_id="sdwpf_frozen_residual_scale_outer_followup_v7",
        source_calibration_dir=str(source), calibration_sha256=sha256(source / "calibration.json"),
        source_protocol_sha256=sha256(source / "protocol.json"),
        source_checkpoint=str(checkpoint), source_checkpoint_sha256=expected_hash,
        outer_boundary_manifest=str(outer_path), outer_boundary_manifest_sha256=sha256(outer_path),
        data_sha256=manifest["data_file"]["sha256"], val_start=str(start), val_end_exclusive=str(end),
        sealed_test_start=str(sealed), event_thresholds=audit["event_thresholds"], capacity_kw=1500.,
        forecast_clip=[0., 1500.], stride=12, expected_windows=boundaries["windows"],
        target_date_min=boundaries["target_date_min"], target_date_max=boundaries["target_date_max"],
        inference_device="cpu", inference_threads=2, fitting_performed=False, utility_wiki=False,
        outer_validation_used_for_selection=False, sealed_test_evaluated=False,
        evidence_status="exploratory follow-up; outer validation has prior development exposure",
        comparison="same inner-fit checkpoint, scaler and windows; fixed calibration vs unchanged prediction",
        bootstrap_block_time_points=144, bootstrap_replicates=5000, bootstrap_seed=2024,
        statistics_estimand="mean MAE gain per forecast issue time, averaged across turbines",
        no_grid_threshold_epoch_or_seed_reselection=True)
    return model, manifest, protocol


def followup_dataset(train_data, manifest, protocol):
    for key, values in (("mean", train_data.scaler.mean_), ("scale", train_data.scaler.scale_)):
        np.testing.assert_allclose(values, manifest["datasets"]["train"]["scaler"][key], rtol=1e-12, atol=1e-12)
    if train_data.feature_columns != manifest["args"]["feature_columns"]:
        raise ValueError("Inner scaler feature order changed")
    for key in ("train_cutoff", "val_cutoff"):
        if np.datetime64(getattr(train_data, key), "ns") != np.datetime64(manifest["datasets"]["train"][key], "ns"):
            raise ValueError("Inner scaler fit/selection boundary changed")
    starts = forward_oof_starts(train_data.segments, train_data.dates, 336, 12, 12,
                               protocol["val_start"], protocol["val_end_exclusive"])
    dataset = copy.copy(train_data)
    dataset.flag, dataset.window_starts = "outer_val_frozen_readonly", starts
    rows = starts[:, None] + 336 + np.arange(12)
    times = np.asarray(dataset.dates, dtype="datetime64[ns]")[rows]
    if (len(starts) != protocol["expected_windows"]
            or times.min() != np.datetime64(protocol["target_date_min"], "ns")
            or times.max() != np.datetime64(protocol["target_date_max"], "ns")
            or not (times >= np.datetime64(protocol["val_start"], "ns")).all()
            or not (times < np.datetime64(protocol["val_end_exclusive"], "ns")).all()
            or not (times < np.datetime64(protocol["sealed_test_start"], "ns")).all()):
        raise ValueError("Validation window identity/boundaries differ from the frozen fold")
    return dataset, times, np.asarray(dataset.turbines)[starts + 336], np.asarray(dataset.available_mask)[rows]


def summarize(model, arrays, protocol):
    reference, persistence, truth, trend = (arrays[key] for key in ("prediction", "persistence", "truth", "trend"))
    calibrated = apply_scale(model, reference, persistence, trend)
    changed = np.abs(calibrated - reference) > 1e-8
    active = changed.any(axis=1)
    gain = np.abs(reference - truth) - np.abs(calibrated - truth)
    window_gain = gain.mean(axis=1)
    report = dict(protocol_id=protocol["protocol_id"], evaluation_scope="outer_val_development_followup",
        fitting_performed=False, sealed_test_evaluated=False, utility_wiki=False,
        source_checkpoint_sha256=protocol["source_checkpoint_sha256"],
        metrics={name: metrics(value, truth, persistence) for name, value in
                 (("reference", reference), ("calibrated", calibrated), ("persistence", persistence))},
        correction=dict(changed_point_pct=float(100 * changed.mean()),
            changed_window_pct=float(100 * active.mean()), unchanged_window_pct=float(100 * (~active).mean()),
            selected_point_gain_kw=float(gain[changed].mean()) if changed.any() else None,
            harm_changed_window_pct=float(100 * (window_gain[active] < -1e-8).mean()) if active.any() else None,
            definition="residual calibration changes vs reference; NOT Wiki coverage/harm"),
        horizon=[], historical_state=[], paired_statistics={})
    for step in range(12):
        report["horizon"].append(dict(step=step + 1, metrics={name: metrics(value[:, step], truth[:, step], persistence[:, step])
            for name, value in (("reference", reference), ("calibrated", calibrated), ("persistence", persistence))}))
    for state in (-1, 0, 1):
        mask = trend == state
        report["historical_state"].append(dict(state=state, windows=int(mask.sum()), metrics={name: metrics(value[mask], truth[mask], persistence[mask])
            for name, value in (("reference", reference), ("calibrated", calibrated), ("persistence", persistence))} if mask.any() else None))
    metadata = pd.DataFrame(dict(forecast_start=arrays["target_timestamps"][:, 0]))
    for name, baseline in (("reference", reference), ("persistence", persistence)):
        stats = paired_forecast_statistics(calibrated, baseline, truth, metadata,
            block_length=protocol["bootstrap_block_time_points"], replicates=protocol["bootstrap_replicates"],
            seed=protocol["bootstrap_seed"])
        stats["interpretation"] = ("Exploratory outer-validation follow-up, not sealed confirmation. "
            "CI/DM refer to time-balanced gains, whereas headline MAE is window-weighted. "
            "144 observed issue-time points; gaps may make duration differ from one day.")
        report["paired_statistics"][name] = stats
    ref, new, base = (report["metrics"][name] for name in ("reference", "calibrated", "persistence"))
    report.update(gain_mae_vs_reference_kw=ref["mae"] - new["mae"], gain_rmse_vs_reference_kw=ref["rmse"] - new["rmse"],
        joint_improvement_vs_reference=new["mae"] < ref["mae"] and new["rmse"] < ref["rmse"],
        joint_improvement_vs_persistence=new["mae"] < base["mae"] and new["rmse"] < base["rmse"],
        next_action="stop_same_fold_search; no automated refit or calibration transplant")
    return calibrated, report


def run(source, directory):
    import fcntl
    import torch
    from run import load_finetuned_model
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with (directory / "followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model, manifest, protocol = frozen_sources(source)
        saved = directory / "protocol.json"
        if saved.exists() and read_json(saved) != protocol:
            raise ValueError("Frozen follow-up protocol/source changed on resume")
        write_json(saved, protocol)  # Persist exact plan before accessing outer targets.
        cache = directory / "predictions.npz"
        marker = directory / "predictions_completed.json"
        if marker.exists():
            if sha256(cache) != read_json(marker)["sha256"]:
                raise ValueError("Completed validation predictions changed")
            print("[FOLLOWUP] Reusing completed CPU inference", flush=True)
            with np.load(cache, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
        else:
            exp = restore_cpu(manifest["args"])
            train_data, _ = exp._get_data("train")
            if sha256(Path(exp.args.root_path) / exp.args.data_path) != protocol["data_sha256"]:
                raise ValueError("Data file hash changed")
            dataset, times, turbines, availability = followup_dataset(train_data, manifest, protocol)
            load_finetuned_model(exp, protocol["source_checkpoint"])
            arrays = infer(exp, dataset, directory, "frozen_reference", protocol,
                           progress_stage="cpu_outer_val_inference", log_prefix="FOLLOWUP")
            arrays.update(target_timestamps=times, turbines=turbines, availability=availability, window_starts=dataset.window_starts)
            temporary = cache.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.replace(cache)
            write_json(marker, dict(sha256=sha256(cache)))
            del exp
        if (directory / "result.json").exists() and (directory / "forecast_accuracy_overview.png").exists():
            print("[FOLLOWUP] Already complete; no fitting or inference repeated", flush=True)
            finish_log_directory(directory, 0)
            return
        calibrated, report = summarize(model, arrays, protocol)
        report["data_audit"] = dict(windows=len(calibrated), points=int(calibrated.size),
            turbine_count=int(len(np.unique(arrays["turbines"]))),
            target_min=str(arrays["target_timestamps"].min()), target_max=str(arrays["target_timestamps"].max()),
            unavailable_point_pct=float(100 * (~arrays["availability"]).mean()),
            availability_policy="retained unchanged actual-power protocol", predictions_sha256=sha256(cache))
        write_json(directory / "result.json", report)
        from utils.forecast_report import _plot_accuracy_overview
        score = report["metrics"]["calibrated"]
        caption = (f"Frozen checkpoint {protocol['source_checkpoint_sha256'][:10]} and saved v7 scales; f1 s2024 outer validation\n"
            f"Previously exposed development segment, not sealed test; {len(calibrated)} h12 windows. "
            f"Capacity hits: ±5% {score['within_5pct_capacity_pct']:.2f}%, ±10% {score['within_10pct_capacity_pct']:.2f}%")
        _plot_accuracy_overview(calibrated, arrays["truth"], str(directory), persistence=arrays["persistence"],
            rated_power=1500, model_name="Frozen residual scale follow-up", scope_caption=caption)
        write_json(directory / "progress.json", dict(stage="completed", pid=os.getpid()))
        finish_log_directory(directory, 0)
        print("[FOLLOWUP] Completed: " + str(directory / "result.json"), flush=True)
        print("[FOLLOWUP] " + str(report["metrics"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.calibration_dir.resolve(), args.result_dir.resolve())
    except BlockingIOError:
        print("Existing follow-up owns this directory; nothing started", file=sys.stderr)
        return 3
    except Exception:
        finish_log_directory(args.result_dir, 1)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
