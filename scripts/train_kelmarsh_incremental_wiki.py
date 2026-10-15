#!/usr/bin/env python3
"""Frozen-numeric Wiki increment: strict replay, matched fit, fresh evidence."""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.IncrementalWindWiki import IncrementalWindWiki, inputs
from scripts import train_kelmarsh_wiki_quality as previous
from utils.wiki_training_replay import verify_replay, replay_diagnostic
from utils.wiki_evidence_support_audit import certificate_support_audit

old, quality = previous.old, previous.quality


def save_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_pack(path, pack):
    arrays = {}
    for key, value in pack.items():
        if isinstance(value, dict):
            arrays.update({key + "__" + field: array for field, array in value.items()})
        else:
            arrays[key] = value
    np.savez_compressed(path, **arrays)


def load_pack(path):
    result = {}
    with np.load(path, allow_pickle=False) as file:
        for key in file.files:
            if "__" in key:
                parent, child = key.split("__", 1)
                result.setdefault(parent, {})[child] = file[key].copy()
            else:
                result[key] = file[key].copy()
    return result


def wait_for_gpu(output, hours):
    begin, last_log = time.monotonic(), -1
    while True:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no silent CPU substitution")
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True)
        others = [int(line.strip()) for line in result.stdout.splitlines()
                  if line.strip().isdigit() and int(line.strip()) != os.getpid()]
        if not others:
            return
        elapsed = time.monotonic() - begin
        old.write(output / "status.json", dict(status="WAITING_FOR_GPU", pid=os.getpid(),
                                              other_gpu_pids=others, waited_seconds=int(elapsed)))
        if int(elapsed // 600) != last_log:
            print(f"[WAIT_GPU] other_pids={others}; no GPU allocated by this run", flush=True)
            last_log = int(elapsed // 600)
        if elapsed >= hours * 3600:
            raise TimeoutError("GPU queue wait limit reached; no other process was modified")
        time.sleep(30)


def source_configuration(spec):
    source = ROOT / spec["source_directory"]
    if old.sha(source / "protocol.json") != spec["source_protocol_sha256"]:
        raise ValueError("Frozen v5 source protocol changed")
    v5 = old.read(source / "protocol.json")
    v4_root = ROOT / v5["source_directory"]
    if old.sha(v4_root / "protocol.json") != v5["source_protocol_sha256"]:
        raise ValueError("Frozen v4 protocol changed")
    v4 = old.read(v4_root / "protocol.json")
    base_root = ROOT / v4["source_directory"]
    if old.sha(base_root / "protocol.json") != v4["source_protocol_sha256"]:
        raise ValueError("Original hourly protocol changed")
    protocol = old.read(base_root / "protocol.json")
    if old.sha(ROOT / protocol["data_path"]) != protocol["data_sha256"]:
        raise ValueError("Original data file changed")
    return source, v5, v4_root, v4, base_root, protocol


def prepare(spec, target, fold, device, protocol_sha256):
    source, v5, v4_root, v4, base_root, p = source_configuration(spec)
    old_run = source / "runs" / f"f{fold}_s{spec['seed']}"
    numeric_run = v4_root / "runs" / f"f{fold}_s{spec['seed']}"
    base_run = base_root / "runs" / f"f{fold}_s{spec['seed']}"
    previous.verify_completed(old_run)
    previous.verify_completed(numeric_run)
    expert = old.read(old_run / "quality_expert.json")
    certificate = old.read(old_run / "benefit_certificate.json")
    execution = certificate["execution"]
    if (execution["quality_expert_id"] != expert["expert_id"]
            or certificate["policy_id"] != quality.legacy.history.digest({k: v for k, v in certificate.items() if k != "policy_id"})):
        raise ValueError("Frozen v5 expert/certificate identity mismatch")
    checkpoints = dict(base=base_run / "PromptTimeDART/checkpoint.pt",
                       numeric=numeric_run / "NumericContext/checkpoint.pt",
                       old_wiki=old_run / "WikiQualityContext/checkpoint.pt")
    expected = dict(base=execution["base"]["base_checkpoint_sha256"],
                    numeric=execution["numerical_checkpoint_sha256"], old_wiki=execution["wiki_checkpoint_sha256"])
    if any(old.sha(path) != expected[key] for key, path in checkpoints.items()):
        raise ValueError("A frozen executing checkpoint changed")
    identity = dict(protocol_sha256=protocol_sha256,
                    source_certificate_sha256=old.sha(old_run / "benefit_certificate.json"),
                    checkpoints=expected, quality_expert_id=expert["expert_id"],
                    execution_source_sha256=hashlib.sha256("\n".join(inspect.getsource(function) for function in
                        (prepare, train_arm, infer, tensor_inputs, save_pack, load_pack, save_torch, inputs)).encode()).hexdigest(),
                    frozen_numeric_runner_sha256=old.sha(ROOT / "scripts/train_kelmarsh_contextual_expert.py"),
                    model_source_sha256=old.sha(ROOT / "models/IncrementalWindWiki.py"),
                    replay_source_sha256=old.sha(ROOT / "utils/wiki_training_replay.py"),
                    runtime=dict(torch=str(torch.__version__), cuda=str(torch.version.cuda), device=torch.cuda.get_device_name(device)))
    series = old.KelmarshSeries(ROOT / p["data_path"], fold=fold, input_len=p["input_len"],
        horizon=p["pred_len"], stride=p["train_stride"], start=p["start_utc"], end=p["end_utc_exclusive"])
    meta = pd.read_csv(ROOT / p["data_path"], usecols=["timestamp_hour_end_utc", "turbine_id",
        "pitch_a_deg", "pitch_b_deg", "pitch_c_deg", "available_capacity_kw"])
    ledger = quality.ObservedQuality(meta, series, v5["quality"])
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "prepared.json"
    if marker.exists():
        done = old.read(marker)
        if done["identity"] != identity or any(old.sha(target / name) != digest for name, digest in done["artifacts"].items()):
            raise ValueError("Completed training replay or its source changed")
        packs = {name: load_pack(target / f"train_{name}.npz") for name in ("memory_fit", "early_fit", "selection", "calibration")}
        print(f"[REUSE_PREPARED] f{fold}; no repeated inference", flush=True)
        return packs, identity, (series, ledger, expert, execution["base"], p, old_run, numeric_run)
    old.gpu_available_to_this_process()
    base = old.PromptTimeDART.Model(old.model_args("PromptTimeDART", series.subset("fit"), p, device)).to(device)
    base.load_state_dict(torch.load(checkpoints["base"], map_location=device, weights_only=False)["model"], strict=True)
    numeric = previous.load_contextual(checkpoints["numeric"], v4["training"], device)
    wiki = previous.load_contextual(checkpoints["old_wiki"], v4["training"], device)
    previous_expert = old.read(numeric_run / "case_expert.json")
    (selection, calibration), _ = old.split_late(series)
    datasets = dict(memory_fit=series.subset("memory"), early_fit=series.subset("calibration_early"),
                    selection=selection, calibration=calibration)
    packs = {}
    for name, dataset in datasets.items():
        reference = old.predict(base, dataset, spec["seed"], device)
        pack, _ = previous.quality_pack(dataset, reference, expert, execution["base"], p["rules"],
            ledger, previous_expert, ready=name != "memory_fit")
        pack["numeric"] = old.infer(numeric, pack, p["rated_power_kw"], device, False)
        packs[name] = pack
    cal = packs["calibration"]
    candidate = old.clip_relative(old.infer(wiki, cal, p["rated_power_kw"], device, True),
                                  cal["numeric"], p["rated_power_kw"], v4["training"]["correction_bound_capacity"])
    actual_hash = verify_replay(cal, cal["numeric"], candidate, certificate, old.digest_arrays)
    diagnostic = replay_diagnostic(cal, cal["numeric"], candidate, certificate, train_end=series.bounds["train_end"])
    diagnostic.update(actual_outputs_sha256=actual_hash, bit_identical_replay=True,
                      support_audit=certificate_support_audit(cal, certificate, train_end=series.bounds["train_end"]))
    old.write(target / "training_replay_audit.json", diagnostic)
    np.savez_compressed(target / "frozen_calibration_predictions.npz", numeric=cal["numeric"], candidate=candidate,
                        truth=cal["truth"], times=cal["times"], turbines=cal["turbines"], overlap=cal["overlap"])
    if diagnostic["oracle_gain_all_points_kw"] <= 0:
        raise ValueError("No training-side supported candidate oracle headroom; do not fit a new gate")
    fit = old.concatenate_packs(packs["memory_fit"], packs["early_fit"])
    if fit["times"].max() >= packs["selection"]["times"].min() or packs["selection"]["times"].max() >= cal["times"].min():
        raise ValueError("Incremental fit/selection/calibration target intervals overlap")
    if packs["memory_fit"]["overlap"].any():
        raise ValueError("Memory-period queries cannot use the not-yet-mature bank")
    for name, pack in packs.items():
        if np.any(pack["turbines"] == 6) or np.any(pack["times"] >= series.bounds["train_end"]):
            raise ValueError("Forbidden training label partition")
        save_pack(target / f"train_{name}.npz", pack)
    files = [f"train_{name}.npz" for name in packs] + ["training_replay_audit.json", "frozen_calibration_predictions.npz"]
    old.write(marker, dict(identity=identity, runner_sha256_at_preparation=old.sha(Path(__file__)),
                           artifacts={name: old.sha(target / name) for name in files}))
    print(f"[STRICT_REPLAY_OK] f{fold} sha={actual_hash}; supported oracle headroom={diagnostic['oracle_gain_all_points_kw']:.6f} kW (retrospective)", flush=True)
    del base, numeric, wiki
    gc.collect()
    torch.cuda.empty_cache()
    return packs, identity, (series, ledger, expert, execution["base"], p, old_run, numeric_run)


def tensor_inputs(pack, capacity):
    return tuple(torch.from_numpy(np.ascontiguousarray(value)) for value in inputs(pack, capacity).values())


@torch.no_grad()
def infer(model, pack, capacity, device, use_memory):
    model.eval()
    chunks = []
    for batch in DataLoader(TensorDataset(*tensor_inputs(pack, capacity)), batch_size=256, shuffle=False):
        context, memory, state, supported = [value.to(device) for value in batch]
        chunks.append(model(context, state, memory, supported, use_memory=use_memory).cpu().numpy().astype(float))
    return pack["numeric"] + capacity * np.concatenate(chunks)


def train_arm(name, fit, selection, capacity, spec, directory, identity, device, *, resource_check=None, after_step=None):
    cfg, use_memory = spec["training"], name == "WikiIncrement"
    old.seed_all(spec["seed"])
    model = IncrementalWindWiki(width=cfg["width"], dropout=cfg["dropout"], bound=cfg["correction_bound_capacity"]).to(device)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "trained.json"
    if marker.exists():
        done = old.read(marker)
        if done["identity"] != identity or any(old.sha(directory / key) != value for key, value in done["artifacts"].items()):
            raise ValueError("Completed incremental expert identity changed")
        model.load_state_dict(torch.load(directory / "checkpoint.pt", map_location=device, weights_only=False)["model"])
        print(f"[REUSE_TRAINED] {name} best_epoch={done['best_epoch']}", flush=True)
        return model, done
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"])
    baseline = old.model_metrics(selection["numeric"], selection["truth"], selection["anchor"], capacity)
    history, start, best_epoch, best = [dict(epoch=0, selection=baseline)], 0, 0, 1.
    if (directory / "last.pt").exists():
        last = torch.load(directory / "last.pt", map_location=device, weights_only=False)
        if last["identity"] != identity:
            raise ValueError("Interrupted training identity changed")
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        schedule.load_state_dict(last["scheduler"])
        start, history, best_epoch, best = last["epoch"], last["history"], last["best_epoch"], last["best"]
        torch.set_rng_state(last["rng_cpu"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in last["rng_cuda"]])
        print(f"[RESUME_ARM] {name} epoch={start}", flush=True)
    else:
        save_torch(directory / "checkpoint.pt", dict(model=model.state_dict(), identity=identity))
    targets = torch.from_numpy(((fit["truth"] - fit["numeric"]) / capacity).astype(np.float32))
    data = TensorDataset(*tensor_inputs(fit, capacity), targets)
    for epoch in range(start + 1, cfg["epochs"] + 1):
        if device.type == "cuda":
            (resource_check or old.gpu_available_to_this_process)()
        model.train()
        generator = torch.Generator().manual_seed(spec["seed"] + epoch)
        total, count = 0., 0
        for batch in DataLoader(data, batch_size=cfg["batch_size"], shuffle=True, generator=generator):
            context, memory, state, supported, target = [value.to(device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            error = model(context, state, memory, supported, use_memory=use_memory) - target
            loss = (1 - cfg["mix_mse_weight"]) * error.abs().mean() + cfg["mix_mse_weight"] * error.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite incremental loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            if after_step is not None:
                after_step()
            total, count = total + float(loss.detach()) * len(context), count + len(context)
        metrics = old.model_metrics(infer(model, selection, capacity, device, use_memory), selection["truth"], selection["anchor"], capacity)
        score = old.joint_score(metrics, baseline)
        history.append(dict(epoch=epoch, train_loss=total / count, selection=metrics))
        if score < best:
            best, best_epoch = score, epoch
            save_torch(directory / "checkpoint.pt", dict(model=model.state_dict(), identity=identity))
        schedule.step()
        save_torch(directory / "last.pt", dict(model=model.state_dict(), identity=identity, optimizer=optimizer.state_dict(),
            scheduler=schedule.state_dict(), epoch=epoch, history=history, best_epoch=best_epoch, best=best,
            rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else []))
        old.write(directory / "training_history.json", history)
        print(f"[EPOCH] {name} {epoch}/{cfg['epochs']} selection MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f} best={best_epoch}", flush=True)
    model.load_state_dict(torch.load(directory / "checkpoint.pt", map_location=device, weights_only=False)["model"])
    old.write(directory / "manifest.json", dict(identity=identity, best_epoch=best_epoch, training=cfg,
        parameters=sum(parameter.numel() for parameter in model.parameters()), use_memory=use_memory,
        selection_scope="Original donor-only first half of late training calibration; zero-increment fallback",
        numerical_forecaster_in_optimizer=False, unsupported_increment_is_exact_zero=True))
    done = dict(identity=identity, best_epoch=best_epoch, epochs_completed=cfg["epochs"],
                artifacts={name: old.sha(directory / name) for name in ("checkpoint.pt", "manifest.json", "training_history.json")})
    old.write(marker, done)
    return model, done


def evaluate(spec, target, packs, trained_arms, information, device):
    series, ledger, expert, contract, p, old_run, numeric_run = information
    validation = series.subset("validation")
    truth, _ = validation.truth_and_anchor()
    cached = {}
    for name in ("OriginalTrend", "NoWikiUnchanged", "WikiQualitySelective"):
        with np.load(old_run / name / "raw_predictions.npz", allow_pickle=False) as file:
            np.testing.assert_array_equal(file["truth"], truth)
            np.testing.assert_array_equal(file["times"], validation.target_times())
            np.testing.assert_array_equal(file["turbines"], series.turbines[validation.window_starts])
            cached[name] = file["prediction"].copy()
    previous_expert = old.read(numeric_run / "case_expert.json")
    evaluation, _ = previous.quality_pack(validation, cached["OriginalTrend"], expert, contract, p["rules"],
        ledger, previous_expert, donors=False)
    evaluation["numeric"] = cached["NoWikiUnchanged"]
    predictions, intervention = dict(cached), {}
    for name, (model, trained, execution) in trained_arms.items():
        raw = infer(model, evaluation, p["rated_power_kw"], device, name == "WikiIncrement")
        supported = evaluation["overlap"].any(-1)
        np.testing.assert_array_equal(raw[~supported], evaluation["numeric"][~supported])
        policy = old.read(target / name / "benefit_certificate.json")
        routed = old.route(policy, evaluation, evaluation["numeric"], raw, execution)
        predictions[name + "Raw"], predictions[name + "Selective"] = raw, routed["prediction"]
        intervention[name + "Raw"] = np.where(supported & (np.abs(raw - evaluation["numeric"]) > 1e-9), 1, 0)
        intervention[name + "Selective"] = routed["action"]
    rows = []
    for name, prediction in predictions.items():
        if name in cached:
            result = old.read(old_run / name / "result.json")
        else:
            result = old.report(name, prediction, validation, target / "reports" / name, p["rated_power_kw"])
            arm = name.removesuffix("Raw").removesuffix("Selective")
            result["best_epoch"] = trained_arms[arm][1]["best_epoch"]
            active = intervention[name] > 0
            windows = active.any(1)
            gain = np.abs(evaluation["numeric"] - truth) - np.abs(prediction - truth)
            result["wiki"] = dict(coverage_window_pct=float(100 * windows.mean()),
                abstention_window_pct=float(100 * (~windows).mean()),
                quality_supported_window_pct=float(100 * evaluation["overlap"].any((1, 2)).mean()),
                selected_gain_kw=float(gain[active].mean()) if active.any() else None,
                harm_window_pct=float(100 * (gain.mean(1)[windows] < 0).mean()) if windows.any() else None)
            result["paired"] = {reference: old.paired_forecast_statistics(prediction, predictions[reference], truth,
                pd.DataFrame({"forecast_start": evaluation["times"][:, 0]}), **spec["statistics"])
                for reference in ("NoWikiUnchanged", "WikiQualitySelective", "ZeroMemoryIncrementSelective")}
            old.write(target / "reports" / name / "result.json", result)
            np.savez_compressed(target / "reports" / name / "interventions.npz", action=intervention[name])
        rows.append(dict(model=name, best_epoch=result.get("best_epoch"), **result["metrics"], **result.get("wiki", {})))
    results = {row["model"]: row for row in rows}
    method = results["WikiIncrementSelective"]
    success = all(method[key] < results[reference][key] for reference in ("NoWikiUnchanged", "ZeroMemoryIncrementSelective")
                  for key in ("mae", "rmse"))
    old.write(target / "result.json", dict(results=rows, joint_improvement_over_numeric_and_zero_memory=success,
        numerical_prediction_arrays_reused_unchanged=True, sealed_test_evaluated=False,
        interpretation="Single-fold development pilot; full supervised memory/early training, no claim of independent confirmation"))
    pd.DataFrame(rows).to_csv(target / "metrics_all_runs.csv", index=False)
    return rows


def run(protocol, output, gpu_wait_hours):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / "suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = old.read(protocol)
        if spec["protocol_id"] != "kelmarsh_h2_frozen_numeric_incremental_wiki_v6":
            raise ValueError("Unknown incremental protocol")
        saved = output / "protocol.json"
        if saved.exists() and old.sha(saved) != old.sha(protocol):
            raise ValueError("Protocol changed on resume")
        if not saved.exists():
            saved.write_bytes(protocol.read_bytes())
        if (output / "completed.json").exists():
            done = old.read(output / "completed.json")
            if done["protocol_sha256"] != old.sha(saved) or any(old.sha(output / key) != value for key, value in done["artifacts"].items()):
                raise ValueError("Completed suite changed")
            print("[REUSE_SUITE] All declared training/report stages already completed", flush=True)
            return
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
        old.write(output / "progress.json", dict(stage="waiting_for_gpu"))
        wait_for_gpu(output, gpu_wait_hours)
        device = torch.device("cuda:0")
        old.write(output / "status.json", dict(status="RUNNING", pid=os.getpid()))
        rows = []
        for fold in spec["folds"]:
            target = output / "runs" / f"f{fold}_s{spec['seed']}"
            old.write(output / "progress.json", dict(stage="strict_training_replay", fold=fold))
            packs, identity, information = prepare(spec, target, fold, device, old.sha(saved))
            fit = old.concatenate_packs(packs["memory_fit"], packs["early_fit"])
            capacity = information[4]["rated_power_kw"]
            trained_arms = {}
            for name in spec["arms"]:
                old.write(output / "progress.json", dict(stage="train_incremental", fold=fold, arm=name))
                arm_identity = dict(identity, arm=name,
                    fit_arrays_sha256=old.digest_arrays(fit),
                    numeric_fit_sha256=hashlib.sha256(np.ascontiguousarray(fit["numeric"]).tobytes()).hexdigest(),
                    prepared_sha256=old.sha(target / "prepared.json"))
                model, trained = train_arm(name, fit, packs["selection"], capacity, spec, target / name, arm_identity, device)
                cal = packs["calibration"]
                candidate = infer(model, cal, capacity, device, name == "WikiIncrement")
                execution = dict(arm_identity, wiki_checkpoint_sha256=old.sha(target / name / "checkpoint.pt"),
                                 numerical_checkpoint_sha256=identity["checkpoints"]["numeric"],
                                 quality_expert_id=identity["quality_expert_id"], base=information[3])
                certificate = old.fit_certificate(cal, cal["numeric"], candidate, execution, spec["utility"])
                old.write(target / name / "benefit_certificate.json", certificate)
                old.write(target / name / "training_benefit_by_day.json", replay_diagnostic(cal, cal["numeric"], candidate,
                    certificate, train_end=information[0].bounds["train_end"]))
                trained_arms[name] = (model, trained, execution)
            # Outer target values are not read for training or epoch selection.
            old.write(output / "progress.json", dict(stage="frozen_outer_report", fold=fold))
            fold_rows = evaluate(spec, target, packs, trained_arms, information, device)
            rows.extend(dict(fold=fold, seed=spec["seed"], **row) for row in fold_rows)
            del trained_arms
            gc.collect()
            torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(output / "metrics_all_runs.csv", index=False)
        retained = {"checkpoint.pt", "trained.json", "manifest.json", "training_history.json", "prepared.json",
                    "result.json", "metrics_all_runs.csv", "benefit_certificate.json", "training_benefit_by_day.json",
                    "training_replay_audit.json", "frozen_calibration_predictions.npz", "raw_predictions.npz", "interventions.npz"}
        artifacts = {str(path.relative_to(output)): old.sha(path) for path in output.rglob("*")
                     if path.is_file() and (path.name in retained or path.name.startswith("train_") and path.suffix == ".npz")}
        old.write(output / "completed.json", dict(protocol_sha256=old.sha(saved), artifacts=artifacts))
        old.write(output / "status.json", dict(status="COMPLETED", exit_code=0, pid=os.getpid()))
        old.write(output / "progress.json", dict(stage="COMPLETED"))
        print("[SUITE_COMPLETE] Frozen numerical path unchanged; inspect real paired results", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/kelmarsh_incremental_wiki_protocol_v6.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-wait-hours", type=float, default=24.)
    args = parser.parse_args()
    try:
        run(args.protocol.resolve(), args.output.resolve(), args.gpu_wait_hours)
    except BlockingIOError:
        print("[BUSY] An existing suite is running; not modified", flush=True)
    except Exception as error:
        old.write(args.output / "status.json", dict(status="FAILED", error=f"{type(error).__name__}: {error}", pid=os.getpid()))
        raise
