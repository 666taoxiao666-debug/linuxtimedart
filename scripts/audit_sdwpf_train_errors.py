#!/usr/bin/env python3
"""CPU-only paired error audit on the unused tail of original outer training."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Set this before importing the training modules, even for a direct invocation.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from scripts.tune_sdwpf_accuracy import read_json, write_json, sha256
from utils.train_error_audit import observable_groups, error_report, verify_audit_sources
from utils.wind_wiki_oof import forward_oof_starts
from utils.sdwpf_logging import finish_log_directory


def sources(v5, v6):
    if not (v5 / "result.json").is_file() or not (v6 / "result.json").is_file():
        raise ValueError("Both fixed source runs must be complete")
    plan = read_json(v5 / "tuning_plan.json")
    reference = read_json(v5 / "candidates/reference/completed.json")
    candidate = read_json(v6 / "result.json")
    provenance = read_json(v6 / "provenance.json")
    if (provenance["plan"] != plan or candidate["fixed_epoch"] != 8
            or reference["stage"] != "finetune"
            or sha256(v5 / "protocol.json") != plan["protocol_sha256"]
            or sha256(v6 / "protocol.json") != candidate["protocol_sha256"]
            or provenance["reference_checkpoint_sha256"] != reference["checkpoint_sha256"]):
        raise ValueError("Frozen source protocols/provenance differ")
    records = []
    for name, state in (("reference", reference), ("physics_norm_v6", candidate)):
        checkpoint = Path(state["checkpoint"]).resolve()
        if sha256(checkpoint) != state["checkpoint_sha256"]:
            raise ValueError(f"Source checkpoint changed: {name}")
        manifest_path = checkpoint.parent / "run_manifest.json"
        manifest = read_json(manifest_path)
        records.append(dict(name=name, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                            manifest=manifest, manifest_sha256=sha256(manifest_path)))
    fit, selection = verify_audit_sources([r["manifest"] for r in records], plan)
    config_path = ROOT / "configs/wind_event_factor_wiki.json"
    config = read_json(config_path)
    from utils.train_error_audit import EVENT_IDS
    if tuple(scene["id"] for scene in config["scenes"]) != EVENT_IDS:
        raise ValueError("Frozen event ordering changed")
    protocol = dict(schema_version=1, protocol_id="sdwpf_f1_s2024_train_tail_error_audit_v1",
        source_v5=str(v5), source_v6=str(v6), source_plan=plan,
        source_protocol_hashes=[sha256(v5 / "protocol.json"), sha256(v6 / "protocol.json")],
        source_checkpoints=[{k: value for k, value in r.items() if k != "manifest"} for r in records],
        fit_cutoff=fit, selection_end=selection, oof_end_exclusive=plan["outer_train_cutoff"],
        event_config_sha256=sha256(config_path), event_thresholds=config["rule_defaults"],
        inference_device="cpu", inference_threads=2, eval_batch_size=32, stride=12,
        capacity_kw=1500., forecast_clip=[0., 1500.], physical_labels_unchanged=True,
        outer_validation_used=False, sealed_test_evaluated=False,
        epoch_selection="existing_fixed_epoch8_checkpoints_only", fitting_performed=False)
    return records, protocol


def restore_cpu(args_source):
    import torch
    from run import build_parser, configure_args
    from exp.exp_timedart import Exp_TimeDART
    args = build_parser().parse_args(["--task_name", "finetune", "--model_id", "SDWPF",
                                     "--model", "PromptTimeDART", "--data", "SDWPF"])
    vars(args).update(copy.deepcopy(args_source))
    args.is_training, args.use_gpu, args.use_multi_gpu = 0, False, False
    args.evaluate_test_after_train, args.wiki_diagnostic = False, False
    args.freeze_non_utility, args.overlay_checkpoint = False, None
    args.pretrain_init, args.load_checkpoints, args.allow_random_init = "none", None, True
    args.use_amp, args.num_workers, args.eval_batch_size = False, 0, 32
    args.run_id = "readonly_train_tail_error_audit"
    args = configure_args(args)
    if args.use_gpu:
        raise RuntimeError("CPU audit must never allocate GPU")
    torch.manual_seed(args.seed)
    return Exp_TimeDART(args)


def infer(exp, dataset, directory, name, protocol):
    import torch
    from torch.utils.data import DataLoader
    from utils.forecast_report import inverse_transform_target
    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0)
    arrays = {key: [] for key in ("prediction", "truth", "persistence", "events", "codes", "trend")}
    wind_index = list(dataset.feature_columns).index("Wspd")
    target_index = list(dataset.feature_columns).index(dataset.target)
    exp.model.eval().requires_grad_(False)
    with torch.inference_mode():
        for batch, (x, y, _, _) in enumerate(loader):
            output = exp.model(x.float().to(exp.device))
            prediction = output[:, -dataset.pred_len:, -1].float().cpu().numpy()
            truth = y[:, -dataset.pred_len:, -1].numpy()
            history_power = inverse_transform_target(dataset, x[..., target_index].numpy())
            history_wind = (x[..., wind_index].numpy().astype(float) * dataset.scaler.scale_[wind_index]
                            + dataset.scaler.mean_[wind_index])
            events, codes, trend = observable_groups(history_wind, history_power,
                rated_power=protocol["capacity_kw"], thresholds=protocol["event_thresholds"])
            values = dict(prediction=np.clip(inverse_transform_target(dataset, prediction), 0, 1500),
                truth=np.clip(inverse_transform_target(dataset, truth), 0, 1500),
                persistence=np.repeat(np.clip(history_power[:, -1:], 0, 1500), dataset.pred_len, axis=1),
                events=events, codes=codes, trend=trend)
            for key, value in values.items():
                arrays[key].append(value)
            if batch % 20 == 0 or batch + 1 == len(loader):
                print(f"[ERROR AUDIT] {name} CPU batch {batch + 1}/{len(loader)}", flush=True)
                write_json(directory / "progress.json", dict(stage="cpu_oof_inference", model=name,
                    batches_completed=batch + 1, batches_total=len(loader), pid=os.getpid()))
    return {key: np.concatenate(value) for key, value in arrays.items()}


def run(directory, v5, v6):
    import fcntl
    import torch
    from run import load_finetuned_model
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with (directory / "audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records, protocol = sources(v5, v6)
        saved = directory / "protocol.json"
        if saved.exists() and read_json(saved) != protocol:
            raise ValueError("Read-only audit source/protocol changed on resume")
        if (directory / "report.json").exists():
            print("[ERROR AUDIT] Already complete; no inference repeated", flush=True)
            return
        write_json(saved, protocol)
        (directory / "status.env").write_text("STATUS=RUNNING\nSTARTED_AT=" +
            datetime.now().astimezone().isoformat(timespec="seconds") + "\n", encoding="utf-8")
        write_json(directory / "progress.json", dict(stage="dataset_and_boundary_checks", pid=os.getpid()))
        results, dataset = {}, None
        for record in records:
            name = record["name"]
            exp = restore_cpu(record["manifest"]["args"])
            if dataset is None:
                train_data, _ = exp._get_data("train")
                data_path = Path(exp.args.root_path) / exp.args.data_path
                if sha256(data_path) != protocol["source_plan"]["outer_data_sha256"]:
                    raise ValueError("Dataset hash differs from source training")
                if (np.datetime64(train_data.train_cutoff, "ns") != np.datetime64(protocol["fit_cutoff"], "ns")
                        or np.datetime64(train_data.val_cutoff, "ns") != np.datetime64(protocol["selection_end"], "ns")):
                    raise ValueError("Reconstructed train/selection boundaries differ")
                for key, values in (("mean", train_data.scaler.mean_), ("scale", train_data.scaler.scale_)):
                    np.testing.assert_allclose(values, record["manifest"]["datasets"]["train"]["scaler"][key],
                                               rtol=1e-12, atol=1e-12)
                if train_data.feature_columns != record["manifest"]["args"]["feature_columns"]:
                    raise ValueError("Reconstructed feature order differs")
                starts = forward_oof_starts(train_data.segments, train_data.dates, 336, 12, 12,
                    protocol["selection_end"], protocol["oof_end_exclusive"])
                dataset = copy.copy(train_data)
                dataset.flag, dataset.window_starts = "train_forward_oof_readonly", starts
                target_rows = starts + dataset.seq_len
                dates = np.asarray(dataset.dates, dtype="datetime64[ns]")
                timestamps = dates[target_rows[:, None] + np.arange(dataset.pred_len)]
                turbines = np.asarray(dataset.turbines)[target_rows]
                availability = np.asarray(dataset.available_mask)[target_rows[:, None] + np.arange(dataset.pred_len)]
                data_audit = dict(window_count=len(starts), forecast_point_count=int(timestamps.size),
                    target_min=str(timestamps.min()), target_max=str(timestamps.max()),
                    outer_train_end_exclusive=protocol["oof_end_exclusive"],
                    all_targets_after_inner_selection=bool((timestamps >= np.datetime64(protocol["selection_end"], "ns")).all()),
                    all_targets_strictly_before_outer_val=bool((timestamps < np.datetime64(protocol["oof_end_exclusive"], "ns")).all()),
                    turbine_count=len(np.unique(turbines)), unavailable_point_pct=float(100 * (~availability).mean()),
                    unavailable_policy="included, unchanged from existing actual-power evaluation; no removal/relabeling",
                    feature_columns=dataset.feature_columns, data_sha256=sha256(data_path))
                write_json(directory / "data_audit.json", data_audit)
                print(f"[ERROR AUDIT] {len(starts)} complete forward-OOF windows, CPU only", flush=True)
            cache = directory / f"{name}_predictions.npz"
            marker = directory / f"{name}_completed.json"
            if marker.exists():
                state = read_json(marker)
                if state["source"] != protocol["source_checkpoints"][records.index(record)] or sha256(cache) != state["predictions_sha256"]:
                    raise ValueError("Completed inference/source changed")
                with np.load(cache, allow_pickle=False) as arrays:
                    if not np.array_equal(arrays["window_starts"], dataset.window_starts):
                        raise ValueError("Reused predictions have different OOF windows")
                    values = {key: arrays[key] for key in ("prediction", "truth", "persistence", "events", "codes", "trend")}
                print(f"[ERROR AUDIT] Reusing completed {name} inference", flush=True)
            else:
                load_finetuned_model(exp, record["checkpoint"])
                values = infer(exp, dataset, directory, name, protocol)
                temporary = cache.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **values, window_starts=dataset.window_starts,
                    target_timestamps=timestamps, turbines=turbines, availability=availability)
                temporary.replace(cache)
                write_json(marker, dict(source=protocol["source_checkpoints"][records.index(record)],
                                        predictions_sha256=sha256(cache)))
            results[name] = values
            del exp
        reference, candidate = results["reference"], results["physics_norm_v6"]
        for key in ("truth", "persistence", "events", "codes", "trend"):
            np.testing.assert_array_equal(reference[key], candidate[key])
        report = error_report({name: value["prediction"] for name, value in results.items()},
            reference["truth"], reference["persistence"], turbines, reference["events"], reference["codes"],
            reference["trend"], rated_power=protocol["capacity_kw"])
        report.update(protocol=protocol, data_audit=data_audit,
            artifact_hashes={name: sha256(directory / f"{name}_predictions.npz") for name in results})
        write_json(directory / "report.json", report)
        write_json(directory / "progress.json", dict(stage="completed", pid=os.getpid()))
        finish_log_directory(directory, 0)
        print("[ERROR AUDIT] Completed: " + str(directory / "report.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-v5-dir", type=Path, required=True)
    parser.add_argument("--source-v6-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        run(args.result_dir.resolve(), args.source_v5_dir.resolve(), args.source_v6_dir.resolve())
    except BlockingIOError:
        print("Existing audit owns this directory; nothing started", file=sys.stderr)
        return 3
    except Exception:
        finish_log_directory(args.result_dir, 1)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
