#!/usr/bin/env python3
"""One matched-normalization train-period pilot, with audited v5 reuse."""
from __future__ import annotations
import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tune_sdwpf_accuracy import (read_json, write_json, sha256, train_stage)
from scripts.audit_sdwpf_normalization import audit
from utils.accuracy_tuning import verify_inner_manifest, compare_fixed_epoch
from utils.sdwpf_logging import finish_log_directory


def read_history(checkpoint):
    with (checkpoint.parent / "training_history.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(directory, source, protocol_path):
    import fcntl
    with (directory / "pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = read_json(protocol_path)
        # The supported protocol is a single fixed mechanism, not a search.
        expected = read_json(ROOT / "configs/sdwpf_physics_norm_protocol_v6.json")
        if protocol != expected:
            raise ValueError("Unsupported normalization pilot protocol")
        protocol_hash = sha256(protocol_path)
        saved = directory / "protocol.json"
        if saved.exists() and sha256(saved) != protocol_hash:
            raise ValueError("Protocol changed on resume")
        if not saved.exists():
            saved.write_bytes(protocol_path.read_bytes())
        if (directory / "result.json").exists():
            print("[NORM] Already complete; no training repeated", flush=True)
            return
        if sha256(source / "protocol.json") != protocol["reference_protocol_sha256"]:
            raise ValueError("Reference protocol mismatch")
        if not (source / "result.json").is_file():
            raise ValueError("Reference v5 run is incomplete")
        plan = read_json(source / "tuning_plan.json")
        if plan["sdwpf_fold"] != 1 or plan["seed"] != 2024:
            raise ValueError("Reference fold/seed mismatch")
        reference_state = read_json(source / "candidates/reference/completed.json")
        reference_checkpoint = Path(reference_state["checkpoint"])
        if sha256(reference_checkpoint) != reference_state["checkpoint_sha256"]:
            raise ValueError("Reference checkpoint changed")
        reference_manifest = read_json(reference_state["manifest"])
        verify_inner_manifest(reference_manifest, plan, "finetune")
        for key, value in {"consistent_physics_norm": False, "mix_mse_weight": .2,
                "horizon_weight_end": 1., "power_weight_alpha": 0., "train_epochs": 8,
                "learning_rate": 1e-6, "new_module_learning_rate": 5e-6,
                "revin_keep_wind": True, "sdwpf_robust_pitch": False, "ramp_residual": False}.items():
            if reference_manifest["args"].get(key, False) != value:
                raise ValueError(f"Reference method differs: {key}")
        # Preserve the source plan/hash, not a mutable latest-pointer lookup.
        provenance = {"protocol_sha256": protocol_hash, "source_v5": str(source),
            "reference_checkpoint_sha256": reference_state["checkpoint_sha256"],
            "reference_history_sha256": sha256(reference_checkpoint.parent / "training_history.csv"),
            "plan": plan}
        if (directory / "provenance.json").exists() and read_json(directory / "provenance.json") != provenance:
            raise ValueError("Source evidence changed on resume")
        write_json(directory / "provenance.json", provenance)
        (directory / "status.env").write_text("STATUS=RUNNING\nSTARTED_AT="
            + datetime.now().astimezone().isoformat(timespec="seconds") + "\n", encoding="utf-8")
        if not (directory / "train_audit.json").exists():
            write_json(directory / "progress.json", {"stage": "training_data_audit"})
            report = audit(ROOT / "datasets", "sdwpf_fixed.csv", plan["inner_train_ratio"], plan["inner_val_ratio"])
            if report["data_sha256"] != plan["outer_data_sha256"]:
                raise ValueError("Training data changed")
            write_json(directory / "train_audit.json", report)
            # Free the large data cache before spawning a training process.
            from data_provider.data_loader import Dataset_SDWPF
            Dataset_SDWPF._cache.clear()
        common = dict(os.environ, FOLD="1", SEED="2024", N_FOLDS="3", PRED_LEN="12",
            MODEL="PromptTimeDART", PROMPT_ROUTER="trend", REGIME_LABEL_METHOD="trend_quantile",
            SCENE_WIKI_CONFIG="configs/wind_regime_wiki.json",
            SCENE_WIKI_EMBEDDINGS="outputs/wiki/wind_regime_wiki_qwen.npz",
            HF_HUB_OFFLINE="1", UTILITY_WIKI="0", UTILITY_FACTORIZED="0", UTILITY_ADAPTER_MODE="legacy",
            UTILITY_HARM_VETO="0", UTILITY_HARM_CLASSIFIER_WEIGHT="0", UTILITY_HARM_LOSS_WEIGHT="0",
            UTILITY_DOWNSIDE_GUARD="0", UTILITY_DOWNSIDE_LOSS_WEIGHT="0", FREEZE_NON_UTILITY="0",
            OVERLAY_CHECKPOINT="", UTILITY_INIT_CHECKPOINT="", RAMP_RESIDUAL="0", RAMP_LEARNING_RATE="0",
            ROBUST_PITCH="0", CHANNEL_PRIOR="1", OP_CONTEXT="1", REVIN_KEEP_WIND="1",
            CONSISTENT_PHYSICS_NORM="1", REGIME_PROMPT="1", ALLOW_RANDOM="0", RESIDUAL_GATE_INIT="-2.2",
            SPLIT="time_ratio", SDWPF_TRAIN_RATIO=str(plan["inner_train_ratio"]),
            SDWPF_VAL_RATIO=str(plan["inner_val_ratio"]), TRAIN_STRIDE="6", EVAL_STRIDE="12")
        pretrain_id = f"physics_norm_v6_{directory.name}_inner_pretrain"
        common["PRETRAIN_RUN_ID"] = pretrain_id
        write_json(directory / "progress.json", {"stage": "matched_inner_pretrain"})
        checkpoint, manifest = train_stage("scripts/pretrain/SDWPF.sh", directory / "inner_pretrain",
            dict(common, RUN_ID=pretrain_id, PRED_LEN="24", TRAIN_EPOCHS="20", PATIENCE="3",
                 LEARNING_RATE="0.0001", EVAL_STRIDE="6"), "pretrain")
        verify_inner_manifest(manifest, plan, "pretrain")
        if not manifest["args"].get("consistent_physics_norm"):
            raise ValueError("New pretrain is not physics-normalization matched")
        write_json(directory / "progress.json", {"stage": "fixed_epoch_inner_finetune", "epoch": 8})
        candidate, manifest = train_stage("scripts/finetune/SDWPF_ablation_prompt.sh", directory / "candidate",
            dict(common, RUN_ID=f"physics_norm_v6_{directory.name}_inner", LOSS="MIXED",
                MIX_MSE_WEIGHT="0.2", HORIZON_WEIGHT_END="1.0", POWER_WEIGHT_ALPHA="0.0",
                LEARNING_RATE="0.000001", NEW_MODULE_LEARNING_RATE="0.000005", PCT_START="0.2",
                EARLY_STOP_METRIC="original_mae_rmse_ratio", TRAIN_EPOCHS="8", PATIENCE="9",
                FIXED_FINETUNE_EPOCH="8"), "finetune")
        verify_inner_manifest(manifest, plan, "finetune")
        if (not manifest["args"].get("consistent_physics_norm") or manifest["extra"]["best_epoch"] != 8
                or manifest["extra"].get("checkpoint_selection_rule") != "predeclared_train_only_epoch"):
            raise ValueError("Candidate selection did not follow the predeclared epoch")
        result = compare_fixed_epoch(read_history(reference_checkpoint), read_history(candidate), 8)
        result.update(protocol_id=protocol["protocol_id"], protocol_sha256=protocol_hash,
            checkpoint=str(candidate), checkpoint_sha256=sha256(candidate),
            outer_validation_used=False, sealed_test_accessed=False, utility_wiki=False,
            next_action="design_matched_outer_confirmation" if result["joint_improvement"] else "stop_mechanism")
        write_json(directory / "result.json", result)
        write_json(directory / "progress.json", {"stage": "completed"})
        finish_log_directory(directory, 0)
        print("[NORM] Completed: " + str(directory / "result.json"), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-v5-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/sdwpf_physics_norm_protocol_v6.json")
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
