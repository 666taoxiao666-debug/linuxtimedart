#!/usr/bin/env python3
"""Reconstruct historical context; reuse train-OOF forecasts without model inference."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from scripts.tune_sdwpf_accuracy import read_json, write_json, sha256
from utils.residual_scale_calibration import chronological_masks
from utils.residual_utility_drift import history_context, drift_report, POWER_EDGES, WIND_EDGES
from utils.train_error_audit import observable_groups
from utils.wind_wiki_oof import forward_oof_starts
from utils.sdwpf_logging import finish_log_directory


def train_dataset(args):
    """The SDWPF data_factory mapping, without importing any neural models."""
    from data_provider.data_loader import Dataset_SDWPF
    if args["data"] != "SDWPF":
        raise ValueError("SDWPF training source required")
    fields = dict(train_ratio="sdwpf_train_ratio", val_ratio="sdwpf_val_ratio",
        expected_freq="sdwpf_expected_freq", window_stride="sdwpf_train_stride",
        filter_abnormal="sdwpf_filter_abnormal", rated_power="rated_power",
        clip_power="sdwpf_clip_power", circular_wind="sdwpf_circular_wind",
        collapse_pitch="sdwpf_collapse_pitch", keep_curtailment="sdwpf_keep_curtailment",
        causal_fill="sdwpf_causal_fill", split="sdwpf_split", fold="sdwpf_fold",
        n_folds="sdwpf_n_folds", physics_features="sdwpf_physics_features", drop_weak_features="sdwpf_drop_weak")
    return Dataset_SDWPF(root_path=args["root_path"], data_path=args["data_path"], flag="train",
        size=[args["seq_len"], args["label_len"], args["pred_len"]], features=args["features"],
        target=args["target"], timeenc=1 if args["embed"] == "timeF" else 0, freq=args["freq"],
        robust_pitch=args.get("sdwpf_robust_pitch", False), **{k: args[v] for k, v in fields.items()})


def reconstruct_context(manifest, audit, arrays):
    """Construct train data only, with the exact saved scaler; no model created."""
    args = manifest["args"]
    data = train_dataset(args)
    if sha256(Path(args["root_path"]) / args["data_path"]) != audit["source_plan"]["outer_data_sha256"]:
        raise ValueError("Raw training data changed")
    if data.feature_columns != manifest["args"]["feature_columns"]:
        raise ValueError("Historical feature order changed")
    for key, actual in (("mean", data.scaler.mean_), ("scale", data.scaler.scale_)):
        np.testing.assert_allclose(actual, manifest["datasets"]["train"]["scaler"][key], rtol=1e-12, atol=1e-12)
    if (np.datetime64(data.train_cutoff, "ns") != np.datetime64(audit["fit_cutoff"], "ns")
            or np.datetime64(data.val_cutoff, "ns") != np.datetime64(audit["selection_end"], "ns")):
        raise ValueError("Scaler fit/selection boundary changed")
    starts = forward_oof_starts(data.segments, data.dates, 336, 12, 12,
                               audit["selection_end"], audit["oof_end_exclusive"])
    np.testing.assert_array_equal(starts, arrays["window_starts"])
    target_rows = starts[:, None] + 336 + np.arange(12)
    dates = np.asarray(data.dates, dtype="datetime64[ns]")
    np.testing.assert_array_equal(dates[target_rows], arrays["target_timestamps"])
    np.testing.assert_array_equal(np.asarray(data.turbines)[starts + 336], arrays["turbines"])
    rows = starts[:, None] + 336 - 12 + np.arange(12)
    if not (dates[rows] < arrays["target_timestamps"][:, :1]).all():
        raise ValueError("Context contains future observations")
    # Identical inverse transformation to the original inference, without
    # reading data_y/target values or constructing validation/test loaders.
    wi, pi = data.feature_columns.index("Wspd"), data.feature_columns.index(data.target)
    wind = data.data_x[rows, wi].astype(float) * data.scaler.scale_[wi] + data.scaler.mean_[wi]
    power = data.data_x[rows, pi].astype(float) * data.scaler.scale_[pi] + data.scaler.mean_[pi]
    events, codes, state = observable_groups(wind, power, rated_power=1500., thresholds=audit["event_thresholds"])
    for key, value in (("events", events), ("codes", codes), ("trend", state)):
        np.testing.assert_array_equal(value, arrays[key])
    np.testing.assert_allclose(np.clip(power[:, -1], 0, 1500), arrays["persistence"][:, 0], rtol=0, atol=1e-8)
    context = history_context(wind, power)
    context.update(window_starts=starts, history_last_time=dates[rows[:, -1]])
    return context


def run(source, directory):
    import fcntl
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with (directory / "audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scale_protocol = read_json(source / "protocol.json")
        model = read_json(source / "calibration.json")
        audit_dir = Path(scale_protocol["source_audit"])
        audit = read_json(audit_dir / "protocol.json")
        checkpoint = Path(scale_protocol["source_checkpoint"]["checkpoint"])
        manifest = read_json(checkpoint.parent / "run_manifest.json")
        if (scale_protocol["protocol_id"] != "sdwpf_train_tail_residual_scale_pilot_v7"
                or scale_protocol["outer_validation_used"] or scale_protocol["sealed_test_evaluated"]
                or sha256(audit_dir / "protocol.json") != scale_protocol["source_audit_protocol_sha256"]
                or sha256(audit_dir / "reference_predictions.npz") != scale_protocol["source_predictions_sha256"]
                or sha256(checkpoint) != model["source_checkpoint_sha256"]
                or model["source_checkpoint_sha256"] != scale_protocol["source_checkpoint"]["checkpoint_sha256"]
                or sha256(checkpoint.parent / "run_manifest.json") != scale_protocol["source_checkpoint"]["manifest_sha256"]):
            raise ValueError("Frozen calibration/prediction/checkpoint identity changed")
        protocol = dict(protocol_id="sdwpf_train_only_fixed_residual_drift_audit",
            source_dir=str(source), calibration_sha256=sha256(source / "calibration.json"),
            source_protocol_sha256=sha256(source / "protocol.json"),
            source_predictions_sha256=scale_protocol["source_predictions_sha256"],
            source_checkpoint_sha256=model["source_checkpoint_sha256"],
            data_sha256=audit["source_plan"]["outer_data_sha256"],
            selection_end=audit["selection_end"], end_exclusive=audit["oof_end_exclusive"],
            history_steps=12, power_ratio_edges=list(POWER_EDGES), wind_mps_edges=list(WIND_EDGES),
            split="existing fixed train-period calendar midpoint, crossing targets purged",
            bin_boundary_rule="edge value belongs to upper bin",
            analysis="existing nonidentity calibration cells only; exact symmetric mix/conditional decomposition",
            model_inference_performed=False, fitting_performed=False, new_parameters_selected=False,
            outer_validation_used=False, sealed_test_evaluated=False, utility_wiki=False,
            evidence_status="exploratory original-train development; previous summaries inspected")
        saved = directory / "protocol.json"
        if saved.exists() and read_json(saved) != protocol:
            raise ValueError("Diagnostic protocol changed on resume")
        if (directory / "report.json").exists():
            state = read_json(directory / "completed.json")
            if (sha256(directory / "report.json") != state["report_sha256"]
                    or sha256(directory / "history_context.npz") != state["context_sha256"]):
                raise ValueError("Completed diagnostic changed")
            print("[DRIFT] Already complete; no extraction or inference repeated", flush=True)
            return
        # Freeze context thresholds before loading per-window targets.
        write_json(saved, protocol)
        (directory / "status.env").write_text("STATUS=RUNNING\nSTARTED_AT=" +
            datetime.now().astimezone().isoformat(timespec="seconds") + "\n", encoding="utf-8")
        write_json(directory / "progress.json", dict(stage="history_context_reconstruction", pid=os.getpid()))
        with np.load(audit_dir / "reference_predictions.npz", allow_pickle=False) as saved_arrays:
            arrays = {key: saved_arrays[key] for key in saved_arrays.files}
        cache = directory / "history_context.npz"
        if cache.exists():
            marker = read_json(directory / "context_completed.json")
            if sha256(cache) != marker["context_sha256"] or marker["protocol_sha256"] != sha256(saved):
                raise ValueError("Historical context cache/protocol changed")
            with np.load(cache, allow_pickle=False) as content:
                context = {key: content[key] for key in content.files}
            np.testing.assert_array_equal(context["window_starts"], arrays["window_starts"])
        else:
            context = reconstruct_context(manifest, audit, arrays)
            np.savez_compressed(cache, **context)
            write_json(directory / "context_completed.json", dict(context_sha256=sha256(cache), protocol_sha256=sha256(saved)))
        early, late, midpoint = chronological_masks(arrays["target_timestamps"], audit["selection_end"], audit["oof_end_exclusive"])
        features = {key: value for key, value in context.items() if key not in ("window_starts", "history_last_time")}
        report = drift_report(model, arrays, features, early, late)
        report.update(protocol=protocol, midpoint=str(midpoint),
            purged_crossing_windows=int((~(early | late)).sum()), context_sha256=sha256(cache),
            next_action="inspect training-only conditional utility and support; no automatic refit or outer evaluation")
        write_json(directory / "report.json", report)
        write_json(directory / "completed.json", dict(report_sha256=sha256(directory / "report.json"), context_sha256=sha256(cache)))
        write_json(directory / "progress.json", dict(stage="completed", pid=os.getpid()))
        finish_log_directory(directory, 0)
        for row in report["changed_cells"]:
            print("[DRIFT] " + str({key: row[key] for key in ("state", "step", "scale")}) +
                  " " + str({k: row["absolute_error"][k] for k in ("early_gain", "late_gain", "composition_shift", "conditional_utility_shift")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.calibration_dir.resolve(), args.result_dir.resolve())
    except BlockingIOError:
        print("Existing diagnostic owns this directory; nothing started", file=sys.stderr)
        return 3
    except Exception:
        finish_log_directory(args.result_dir, 1)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
