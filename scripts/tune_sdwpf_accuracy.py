#!/usr/bin/env python3
"""Run a bounded train-only objective search and one fixed-epoch outer refit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_wind_wiki_oof_evidence import make_plan, read_pipeline_value
from utils.accuracy_tuning import choose_candidate, verify_inner_manifest
from utils.sdwpf_logging import finish_log_directory


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_gpu_idle():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True)
    if result.stdout.strip():
        raise RuntimeError("GPU is occupied; resume this run after the other process finishes")


def completed_training(directory, stage):
    marker = directory / "completed.json"
    if marker.is_file():
        state = read_json(marker)
        if state["stage"] != stage or sha256(state["checkpoint"]) != state["checkpoint_sha256"]:
            raise ValueError("Completed checkpoint changed")
        return Path(state["checkpoint"]), read_json(state["manifest"])
    log = directory / "launcher.log"
    if not log.exists():
        return None
    paths = re.findall(r"^\[AUDIT\] Run manifest: (.+)$", log.read_text(encoding="utf-8"), re.MULTILINE)
    if not paths:
        return None
    path = Path(paths[-1].strip())
    manifest = read_json(path)
    if manifest.get("extra", {}).get("status") != "complete" or manifest.get("stage") != stage:
        return None
    name = "ckpt_best.pth" if stage == "pretrain" else "checkpoint.pth"
    checkpoint = path.parent / name
    if not checkpoint.is_file():
        return None
    write_json(marker, {"stage": stage, "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256(checkpoint), "manifest": str(path)})
    return checkpoint, manifest


def train_stage(script, directory, environment, stage):
    directory.mkdir(parents=True, exist_ok=True)
    previous = completed_training(directory, stage)
    if previous:
        print(f"[TUNE] Reusing completed {stage}: {directory}", flush=True)
        return previous
    assert_gpu_idle()
    environment = dict(environment, SDWPF_LOG_DIR=str(directory), SDWPF_LOG_FILE="")
    with (directory / "launcher.log").open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(["bash", script], cwd=ROOT, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        write_json(directory / "process.json", {"pid": process.pid, "stage": stage})
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            if line.startswith(("Epoch:", "[AUDIT]", "Using ForecastLoss", "Traceback", "Pretrain early")):
                print(line, end="", flush=True)
        if process.wait() != 0:
            raise RuntimeError(f"{stage} failed; see {directory / 'launcher.log'}")
    result = completed_training(directory, stage)
    if result is None:
        raise RuntimeError("Training exited without a complete checkpoint manifest")
    return result


def objective_environment(common, candidate, schedule_epochs):
    return dict(common, LOSS="MIXED", MIX_MSE_WEIGHT=str(candidate["mix_mse_weight"]),
                HORIZON_WEIGHT_END=str(candidate["horizon_weight_end"]),
                POWER_WEIGHT_ALPHA=str(candidate["power_weight_alpha"]),
                EARLY_STOP_METRIC="original_mae_rmse_ratio", TRAIN_EPOCHS=str(schedule_epochs),
                PATIENCE=str(schedule_epochs + 1), FIXED_FINETUNE_EPOCH="0")


def run(directory, protocol_path, trend_cv):
    # Advisory lock is released even if SSH/automation exits unexpectedly.
    import fcntl
    with (directory / "tuning.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = read_json(protocol_path)
        if (protocol["inner_fit_fraction_of_outer_train"] != 0.8
                or protocol["inner_selection_fraction_of_outer_train"] != 0.1
                or protocol["inner_split"] != "time_ratio"
                or protocol["input_len"] != 336
                or protocol["model"] != "PromptTimeDART"
                or protocol["prompt_router"] != "trend" or protocol["utility_wiki"]):
            raise ValueError("Protocol is not supported by this frozen launcher")
        protocol_hash = sha256(protocol_path)
        saved_protocol = directory / "protocol.json"
        if saved_protocol.exists() and sha256(saved_protocol) != protocol_hash:
            raise ValueError("Resume protocol differs from the original")
        if not saved_protocol.exists():
            saved_protocol.write_bytes(protocol_path.read_bytes())
        result_file = directory / "result.json"
        if result_file.exists():
            print("[TUNE] Entire run already complete; no training repeated", flush=True)
            finish_log_directory(directory, 0)
            return
        (directory / "status.env").write_text(
            "STATUS=RUNNING\nSTARTED_AT=" + datetime.now().astimezone().isoformat(timespec="seconds") + "\n",
            encoding="utf-8")

        fold, seed = protocol["fold"], protocol["seed"]
        plan = make_plan(trend_cv, fold, seed)
        plan.update(protocol_id=protocol["protocol_id"], protocol_sha256=protocol_hash)
        plan_file = directory / "tuning_plan.json"
        if plan_file.exists() and read_json(plan_file) != plan:
            raise ValueError("Matched source plan changed since launch")
        write_json(plan_file, plan)
        outer_pretrain = read_pipeline_value(
            trend_cv / f"runs/f{fold}_s{seed}/finetune.env", "PRETRAIN_RUN_ID")
        common = dict(os.environ,
            FOLD=str(fold), SEED=str(seed), N_FOLDS="3", PRED_LEN=str(protocol["pred_len"]),
            MODEL="PromptTimeDART", PROMPT_ROUTER="trend", REGIME_LABEL_METHOD="trend_quantile",
            SCENE_WIKI_CONFIG="configs/wind_regime_wiki.json",
            SCENE_WIKI_EMBEDDINGS="outputs/wiki/wind_regime_wiki_qwen.npz",
            HF_HUB_OFFLINE="1", UTILITY_WIKI="0", UTILITY_FACTORIZED="0",
            UTILITY_ADAPTER_MODE="legacy", UTILITY_HARM_VETO="0", UTILITY_HARM_LOSS_WEIGHT="0",
            UTILITY_HARM_CLASSIFIER_WEIGHT="0", UTILITY_DOWNSIDE_GUARD="0",
            UTILITY_DOWNSIDE_LOSS_WEIGHT="0", FREEZE_NON_UTILITY="0", OVERLAY_CHECKPOINT="",
            UTILITY_INIT_CHECKPOINT="", RAMP_RESIDUAL="0", RAMP_LEARNING_RATE="0",
            ROBUST_PITCH="0", CHANNEL_PRIOR="1", OP_CONTEXT="1", REVIN_KEEP_WIND="1",
            REGIME_PROMPT="1", ALLOW_RANDOM="0", RESIDUAL_GATE_INIT="-2.2",
            LEARNING_RATE=str(protocol["finetuning"]["learning_rate"]),
            NEW_MODULE_LEARNING_RATE=str(protocol["finetuning"]["new_module_learning_rate"]),
            PCT_START=str(protocol["finetuning"]["pct_start"]), TRAIN_STRIDE="6", EVAL_STRIDE="12")
        inner_pretrain = f"accuracy_v5_{directory.name}_inner_pretrain"
        inner = dict(common, SPLIT="time_ratio", SDWPF_TRAIN_RATIO=str(plan["inner_train_ratio"]),
                     SDWPF_VAL_RATIO=str(plan["inner_val_ratio"]), PRETRAIN_RUN_ID=inner_pretrain)
        pretraining = protocol["pretraining"]
        print("[TUNE] Stage 1: pretrain on the inner fit prefix", flush=True)
        write_json(directory / "progress.json", {"stage": "inner_pretrain"})
        checkpoint, manifest = train_stage("scripts/pretrain/SDWPF.sh", directory / "inner_pretrain",
            dict(inner, RUN_ID=inner_pretrain, PRED_LEN=str(pretraining["pred_len"]),
                 TRAIN_EPOCHS=str(pretraining["epochs"]), PATIENCE=str(pretraining["patience"]),
                 LEARNING_RATE=str(pretraining["learning_rate"]), EVAL_STRIDE="6"), "pretrain")
        verify_inner_manifest(manifest, plan, "pretrain")

        schedule_epochs = protocol["finetuning"]["schedule_epochs"]
        histories, provenance = {}, {}
        for candidate in protocol["candidates"]:
            name = candidate["id"]
            print(f"[TUNE] Stage 2: fixed candidate {name}", flush=True)
            write_json(directory / "progress.json", {"stage": "inner_candidates", "candidate": name})
            environment = objective_environment(inner, candidate, schedule_epochs)
            environment["RUN_ID"] = f"accuracy_v5_{directory.name}_inner_{name}"
            checkpoint, manifest = train_stage("scripts/finetune/SDWPF_ablation_prompt.sh",
                directory / "candidates" / name, environment, "finetune")
            verify_inner_manifest(manifest, plan, "finetune")
            with (checkpoint.parent / "training_history.csv").open(encoding="utf-8") as handle:
                histories[name] = list(csv.DictReader(handle))
            provenance[name] = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                                "training_history": str(checkpoint.parent / "training_history.csv")}
        selection = choose_candidate(histories, [c["id"] for c in protocol["candidates"]], schedule_epochs)
        selection.update(protocol_sha256=protocol_hash, candidate_provenance=provenance)
        selected_file = directory / "selected.json"
        if selected_file.exists() and read_json(selected_file) != selection:
            raise ValueError("Train-only parameter selection changed on resume")
        write_json(selected_file, selection)
        selected = selection["selected"]
        candidate = next(c for c in protocol["candidates"] if c["id"] == selected["candidate_id"])
        print(f"[TUNE] Chosen from inner training: {candidate['id']} epoch={selected['epoch']}", flush=True)

        print("[TUNE] Stage 3: one outer fixed-epoch refit", flush=True)
        write_json(directory / "progress.json", {"stage": "outer_refit", "candidate": candidate["id"],
                                                "fixed_epoch": selected["epoch"]})
        environment = objective_environment(common, candidate, schedule_epochs)
        environment.update(SPLIT="rolling_holdout", SDWPF_TRAIN_RATIO="0.7", SDWPF_VAL_RATIO="0.1",
            PRETRAIN_RUN_ID=outer_pretrain, FIXED_FINETUNE_EPOCH=str(selected["epoch"]),
            RUN_ID=f"accuracy_v5_{directory.name}_outer_{candidate['id']}_ep{selected['epoch']}")
        checkpoint, manifest = train_stage("scripts/finetune/SDWPF_ablation_prompt.sh",
            directory / "refit", environment, "finetune")
        if (manifest["extra"].get("checkpoint_selection_rule") != "predeclared_train_only_epoch"
                or manifest["extra"]["best_epoch"] != selected["epoch"]):
            raise ValueError("Outer checkpoint was not selected by the fixed train-only epoch")
        outer_args = manifest["args"]
        if (manifest["data_file"]["sha256"] != plan["outer_data_sha256"]
                or outer_args["sdwpf_split"] != "rolling_holdout"
                or outer_args["sdwpf_fold"] != fold or outer_args["seed"] != seed
                or outer_args["prompt_router"] != "trend" or outer_args["utility_wiki"]
                or outer_args["evaluate_test_after_train"]):
            raise ValueError("Outer refit provenance mismatch")

        print("[TUNE] Stage 4: read-only outer validation report", flush=True)
        write_json(directory / "progress.json", {"stage": "outer_validation_report"})
        report = directory / "validation" / "artifacts"
        if not (report / "metrics.json").exists():
            assert_gpu_idle()
            validation_dir = directory / "validation"
            validation_dir.mkdir(exist_ok=True)
            environment = dict(common, SPLIT="rolling_holdout", FINETUNE_CHECKPOINT=str(checkpoint),
                REPORT_OUTPUT_DIR=str(report), SDWPF_LOG_DIR=str(validation_dir),
                SDWPF_LOG_FILE=str(validation_dir / "validation.log"),
                FORECAST_PLOT_POINTS="300", RUN_ID=f"accuracy_v5_{directory.name}_validation")
            with (validation_dir / "launcher.log").open("a", encoding="utf-8") as handle:
                subprocess.run(["bash", "scripts/eval/SDWPF_logged_validation_plot.sh"],
                    cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
        write_json(result_file, {"protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash,
            "selection": selection, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "original": read_json(report / "metrics.json")["original"], "sealed_test_accessed": False})
        write_json(directory / "progress.json", {"stage": "completed"})
        finish_log_directory(directory, 0)
        print(f"[TUNE] Completed: {result_file}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/sdwpf_accuracy_tuning_protocol_v5.json")
    parser.add_argument("--trend-cv-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.result_dir.resolve()
    try:
        run(directory, args.protocol.resolve(), args.trend_cv_dir.resolve())
    except BlockingIOError:
        # A duplicate observer must not overwrite the actual owner's status.
        print("[TUNE] Another process owns this run; nothing started", file=sys.stderr)
        return 3
    except Exception:
        finish_log_directory(directory, 1)
        raise


if __name__ == "__main__":
    sys.exit(main())
