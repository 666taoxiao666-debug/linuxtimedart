#!/usr/bin/env python3
"""Fixed-epoch, train-period pilot correcting SDWPF relative-angle semantics."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.tune_sdwpf_accuracy import read_json, write_json, sha256, train_stage, assert_gpu_idle
from utils.accuracy_tuning import verify_inner_manifest, compare_fixed_epoch
from utils.sdwpf_logging import finish_log_directory

PROTOCOL = ROOT / "configs/sdwpf_angle_semantics_protocol_v14.json"


def read_history(checkpoint):
    with (checkpoint.parent / "training_history.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selected_window_fingerprint(dataset):
    """Hash only this split's windows/targets, never the dataset's sealed suffix."""
    starts = np.asarray(dataset.window_starts, dtype=np.int64)
    offsets = np.arange(dataset.pred_len, dtype=np.int64)
    indices = starts[:, None] + dataset.seq_len + offsets[None, :]
    digest = hashlib.sha256()
    arrays = (
        starts, np.asarray(dataset.dates)[indices].astype("datetime64[ns]").view("i8"),
        np.asarray(dataset.turbines)[indices], dataset.data_y[indices, -1],
        np.asarray(dataset.available_mask)[indices],
        np.asarray(dataset.scaler.mean_[-1:], dtype=np.float64),
        np.asarray(dataset.scaler.scale_[-1:], dtype=np.float64),
    )
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str((value.shape, value.dtype.str)).encode("ascii"))
        digest.update(value.tobytes())
    return dict(windows=len(starts), sha256=digest.hexdigest(),
                first_target=str(np.asarray(dataset.dates)[indices].min()),
                last_target=str(np.asarray(dataset.dates)[indices].max()))


def audit_matching_windows(reference_manifest, plan):
    from data_provider.data_factory import data_provider
    from data_provider.data_loader import Dataset_SDWPF
    fingerprints = {}
    for mode in ("legacy_difference", "relative_wdir"):
        args = SimpleNamespace(**reference_manifest["args"])
        args.sdwpf_angle_mode = mode
        args.num_workers = 0
        args.use_gpu = False
        fingerprints[mode] = {}
        for split in ("train", "val"):
            dataset, loader = data_provider(args, split)
            fingerprint = selected_window_fingerprint(dataset)
            if np.datetime64(fingerprint["last_target"]) >= np.datetime64(plan["outer_train_cutoff"]):
                raise ValueError("Window audit touched the outer validation period")
            fingerprints[mode][split] = fingerprint
            del dataset, loader
        Dataset_SDWPF._cache.clear()
        gc.collect()
    if fingerprints["legacy_difference"] != fingerprints["relative_wdir"]:
        raise ValueError("Angle correction changed windows, target labels, or target scaling")
    return dict(selected_windows_and_labels_identical=True, fingerprints=fingerprints,
                outer_validation_used=False, sealed_test_accessed=False)


def check_candidate(manifest, plan, stage, fixed_epoch=None):
    verify_inner_manifest(manifest, plan, stage)
    args = manifest["args"]
    expected_features = ["Wspd", "Etmp", "Ndir_sin", "Ndir_cos", "yaw_sin", "yaw_cos", "Pab_mean", "power"]
    if (args.get("sdwpf_angle_mode") != "relative_wdir"
            or args.get("feature_columns") != expected_features
            or args.get("consistent_physics_norm", False)):
        raise ValueError("Candidate uses the wrong angle/normalization convention")
    if fixed_epoch is not None and (
            manifest["extra"]["best_epoch"] != fixed_epoch
            or manifest["extra"].get("checkpoint_selection_rule") != "predeclared_train_only_epoch"):
        raise ValueError("Candidate did not select the predeclared fixed epoch")


def internal_prediction_report(directory, checkpoint, manifest, plan):
    """Evaluate the fixed checkpoint on its original internal holdout only."""
    import torch
    from run import build_parser, authorize_forecast_report_split, load_finetuned_model
    from exp.exp_timedart import Exp_TimeDART
    from data_provider.data_factory import data_provider
    from utils.experiment_audit import dataset_summary
    from utils.forecast_report import _plot_accuracy_overview

    report = directory / "internal_validation" / "artifacts"
    audit_path = report.parent / "report_audit.json"
    if (report / "metrics.json").is_file() and audit_path.is_file():
        previous = read_json(audit_path)
        if (previous["checkpoint_sha256"] != sha256(checkpoint)
                or previous["plan"] != plan):
            raise ValueError("Saved prediction report has different provenance")
        return read_json(report / "metrics.json")["original"], report
    assert_gpu_idle()
    args = build_parser().parse_args(["--task_name", "finetune", "--model_id", "SDWPF",
                                    "--model", "PromptTimeDART", "--data", "SDWPF"])
    vars(args).update(manifest["args"])
    args.is_training = 0
    args.load_checkpoints = None
    args.overlay_checkpoint = None
    args.utility_init_checkpoint = None
    args.report_split = "val"
    args.report_output_dir = str(report)
    args.run_id = f"angles_v14_{directory.name}_internal_report"
    args.num_workers = 0
    args.use_gpu = torch.cuda.is_available()
    authorize_forecast_report_split(args)
    dataset, loader = data_provider(args, "val")
    summary = dataset_summary(dataset, "val")
    if (np.datetime64(summary["target_date_max"]) >= np.datetime64(plan["outer_train_cutoff"])
            or summary["windows"] != manifest["datasets"]["val"]["windows"]):
        raise ValueError("Prediction report is not the matched internal holdout")
    expected = read_json(directory / "window_audit.json")["fingerprints"]["relative_wdir"]["val"]
    if selected_window_fingerprint(dataset) != expected:
        raise ValueError("Prediction report targets changed after training")
    del loader, dataset
    exp = Exp_TimeDART(args)
    try:
        load_finetuned_model(exp, str(checkpoint))
        exp.test()
    finally:
        exp.writer.close()
    # The overview explicitly states the evaluated period; R2 is not an accuracy percentage.
    with np.load(report / "predictions.npz") as arrays:
        _plot_accuracy_overview(arrays["prediction_original"], arrays["truth_original"], str(report),
            persistence=arrays["persistence_original"], rated_power=args.rated_power,
            model_name="Corrected-angle trend",
            scope_caption="Training-period internal holdout; fixed epoch 8; not outer validation/test")
    write_json(audit_path, dict(checkpoint_sha256=sha256(checkpoint), plan=plan,
                              dataset=summary, outer_validation_used=False, sealed_test_accessed=False))
    return read_json(report / "metrics.json")["original"], report


def run(directory, source, protocol_path):
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = read_json(protocol_path)
        if protocol != read_json(PROTOCOL):
            raise ValueError("Unsupported corrected-angle protocol")
        protocol_hash = sha256(protocol_path)
        saved = directory / "protocol.json"
        if saved.exists() and sha256(saved) != protocol_hash:
            raise ValueError("Protocol changed on resume")
        if not saved.exists():
            saved.write_bytes(protocol_path.read_bytes())
        if (directory / "result.json").exists():
            print("[ANGLES] Already complete; no training repeated", flush=True)
            return
        if (sha256(source / "protocol.json") != protocol["reference_protocol_sha256"]
                or not (source / "result.json").is_file()):
            raise ValueError("Frozen v5 reference is mismatched or incomplete")
        plan = read_json(source / "tuning_plan.json")
        if plan["sdwpf_fold"] != protocol["fold"] or plan["seed"] != protocol["seed"]:
            raise ValueError("Reference fold/seed mismatch")
        reference_state = read_json(source / "candidates/reference/completed.json")
        reference_checkpoint = Path(reference_state["checkpoint"])
        if (sha256(reference_checkpoint) != reference_state["checkpoint_sha256"]
                or reference_state["checkpoint_sha256"] != protocol["reference_checkpoint_sha256"]):
            raise ValueError("Frozen reference checkpoint changed")
        reference_manifest = read_json(reference_state["manifest"])
        verify_inner_manifest(reference_manifest, plan, "finetune")
        constraints = {"mix_mse_weight": .2, "horizon_weight_end": 1., "power_weight_alpha": 0.,
            "train_epochs": 8, "learning_rate": 1e-6, "new_module_learning_rate": 5e-6,
            "revin_keep_wind": True, "consistent_physics_norm": False,
            "sdwpf_robust_pitch": False, "ramp_residual": False, "channel_prior": True, "op_context": True}
        for key, value in constraints.items():
            actual = reference_manifest["args"].get(key, False)
            if isinstance(value, bool):
                actual = bool(actual)  # Legacy manifests predate some optional switches.
            if actual != value:
                raise ValueError(f"Reference method differs: {key}")
        if reference_manifest["args"].get("sdwpf_angle_mode", "legacy_difference") != "legacy_difference":
            raise ValueError("Reference angle convention differs")
        provenance = dict(protocol_sha256=protocol_hash, source_v5=str(source), plan=plan,
            reference_checkpoint_sha256=reference_state["checkpoint_sha256"],
            reference_manifest_sha256=sha256(reference_state["manifest"]),
            reference_history_sha256=sha256(reference_checkpoint.parent / "training_history.csv"),
            data_sha256=plan["outer_data_sha256"])
        if (directory / "provenance.json").exists() and read_json(directory / "provenance.json") != provenance:
            raise ValueError("Source evidence changed on resume")
        write_json(directory / "provenance.json", provenance)
        (directory / "status.env").write_text("STATUS=RUNNING\nSTARTED_AT="
            + datetime.now().astimezone().isoformat(timespec="seconds") + "\n", encoding="utf-8")
        if sha256(Path(reference_manifest["args"]["root_path"]) / reference_manifest["args"]["data_path"]) != plan["outer_data_sha256"]:
            raise ValueError("Dataset changed")
        if not (directory / "window_audit.json").is_file():
            write_json(directory / "progress.json", {"stage": "matched_window_audit"})
            write_json(directory / "window_audit.json", audit_matching_windows(reference_manifest, plan))
        common = dict(os.environ, FOLD="1", SEED="2024", N_FOLDS="3", PRED_LEN="12",
            MODEL="PromptTimeDART", PROMPT_ROUTER="trend", REGIME_LABEL_METHOD="trend_quantile",
            SCENE_WIKI_CONFIG="configs/wind_regime_wiki.json",
            SCENE_WIKI_EMBEDDINGS="outputs/wiki/wind_regime_wiki_qwen.npz",
            HF_HUB_OFFLINE="1", UTILITY_WIKI="0", UTILITY_FACTORIZED="0", UTILITY_ADAPTER_MODE="legacy",
            UTILITY_HARM_VETO="0", UTILITY_HARM_CLASSIFIER_WEIGHT="0", UTILITY_HARM_LOSS_WEIGHT="0",
            UTILITY_DOWNSIDE_GUARD="0", UTILITY_DOWNSIDE_LOSS_WEIGHT="0", FREEZE_NON_UTILITY="0",
            OVERLAY_CHECKPOINT="", UTILITY_INIT_CHECKPOINT="", UTILITY_INIT_CHECKPOINT_SHA256="",
            RAMP_RESIDUAL="0", RAMP_LEARNING_RATE="0",
            ROBUST_PITCH="0", CHANNEL_PRIOR="1", OP_CONTEXT="1", REVIN_KEEP_WIND="1",
            CONSISTENT_PHYSICS_NORM="0", SDWPF_ANGLE_MODE="relative_wdir", REGIME_PROMPT="1",
            ALLOW_RANDOM="0", RESIDUAL_GATE_INIT="-2.2", SPLIT="time_ratio",
            SDWPF_TRAIN_RATIO=str(plan["inner_train_ratio"]), SDWPF_VAL_RATIO=str(plan["inner_val_ratio"]),
            TRAIN_STRIDE="6", EVAL_STRIDE="12")
        pretrain_id = f"angles_v14_{directory.name}_inner_pretrain"
        common["PRETRAIN_RUN_ID"] = pretrain_id
        write_json(directory / "progress.json", {"stage": "matched_inner_pretrain"})
        _, manifest = train_stage("scripts/pretrain/SDWPF.sh", directory / "inner_pretrain",
            dict(common, RUN_ID=pretrain_id, PRED_LEN="24", TRAIN_EPOCHS="20", PATIENCE="3",
                 LEARNING_RATE="0.0001", EVAL_STRIDE="6"), "pretrain")
        check_candidate(manifest, plan, "pretrain")
        write_json(directory / "progress.json", {"stage": "fixed_epoch_inner_finetune", "epoch": 8})
        candidate, manifest = train_stage("scripts/finetune/SDWPF_ablation_prompt.sh", directory / "candidate",
            dict(common, RUN_ID=f"angles_v14_{directory.name}_inner", LOSS="MIXED",
                MIX_MSE_WEIGHT="0.2", HORIZON_WEIGHT_END="1.0", POWER_WEIGHT_ALPHA="0.0",
                LEARNING_RATE="0.000001", NEW_MODULE_LEARNING_RATE="0.000005", PCT_START="0.2",
                EARLY_STOP_METRIC="original_mae_rmse_ratio", TRAIN_EPOCHS="8", PATIENCE="9",
                FIXED_FINETUNE_EPOCH="8"), "finetune")
        check_candidate(manifest, plan, "finetune", fixed_epoch=8)
        result = compare_fixed_epoch(read_history(reference_checkpoint), read_history(candidate), 8)
        write_json(directory / "progress.json", {"stage": "internal_holdout_prediction_report"})
        original, report = internal_prediction_report(directory, candidate, manifest, plan)
        for history_key, report_key in (("val_mae_kw", "mae"), ("val_rmse_kw", "rmse")):
            if not np.isclose(result["candidate"][history_key], original[report_key], rtol=1e-6, atol=1e-4):
                raise ValueError("Prediction report does not reproduce the fixed-epoch metrics")
        result.update(protocol_id=protocol["protocol_id"], protocol_sha256=protocol_hash,
            checkpoint=str(candidate), checkpoint_sha256=sha256(candidate), original=original,
            report_dir=str(report), outer_validation_used=False, sealed_test_accessed=False,
            utility_wiki=False, next_action="request_matched_outer_confirmation" if result["joint_improvement"] else "stop_mechanism")
        write_json(directory / "result.json", result)
        write_json(directory / "progress.json", {"stage": "completed"})
        finish_log_directory(directory, 0)
        print("[ANGLES] Completed: " + str(directory / "result.json"), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-v5-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    try:
        run(args.result_dir.resolve(), args.source_v5_dir.resolve(), args.protocol.resolve())
    except BlockingIOError:
        print("Another process owns this pilot; nothing started", file=sys.stderr)
        return 3
    except Exception:
        finish_log_directory(args.result_dir, 1)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
