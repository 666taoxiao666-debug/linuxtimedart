#!/usr/bin/env python3
"""One frozen v7 pilot: optional numerical residual plus matched Wiki increments."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np
import pandas as pd
import torch
from scripts import train_kelmarsh_incremental_wiki as frozen
from utils import certified_numeric_residual as guard
from utils.gpu_runtime_policy import GPURuntimePolicy

old, previous = frozen.old, frozen.previous
PROTOCOL_ID = "kelmarsh_h2_optional_numeric_residual_v7"


def verify_directory(directory, protocol_sha=None):
    done = old.read(directory / "completed.json")
    if protocol_sha is not None and done["protocol_sha256"] != protocol_sha:
        raise ValueError("Completed source protocol mismatch")
    for name, expected in done["artifacts"].items():
        if old.sha(directory / name) != expected:
            raise ValueError(f"Completed source artifact changed: {name}")


def prepare(spec, output, protocol_sha):
    source = ROOT / spec["source_directory"]
    if old.sha(source / "protocol.json") != spec["source_protocol_sha256"]:
        raise ValueError("Frozen v6 protocol changed")
    verify_directory(source, spec["source_protocol_sha256"])
    prior = old.read(source / "protocol.json")
    v5_root, v5, v4_root, v4, base_root, p = frozen.source_configuration(prior)
    fold, seed = spec["folds"][0], spec["seed"]
    suffix = Path("runs") / f"f{fold}_s{seed}"
    source_run, v5_run, v4_run, base_run = [root / suffix for root in (source, v5_root, v4_root, base_root)]
    previous.verify_completed(v5_run)
    previous.verify_completed(v4_run)
    prepared = old.read(source_run / "prepared.json")
    for name, expected in prepared["artifacts"].items():
        if old.sha(source_run / name) != expected:
            raise ValueError("Frozen training replay pack changed")
    paths = dict(base=base_run / "PromptTimeDART/checkpoint.pt", numeric=v4_run / "NumericContext/checkpoint.pt",
                 old_wiki=v5_run / "WikiQualityContext/checkpoint.pt")
    if any(old.sha(path) != prepared["identity"]["checkpoints"][name] for name, path in paths.items()):
        raise ValueError("Frozen base/numeric executor checkpoint changed")
    contract = old.read(v5_run / "benefit_certificate.json")["execution"]["base"]
    bank = old.read(v5_run / "quality_expert.json")
    if bank["expert_id"] != prepared["identity"]["quality_expert_id"]:
        raise ValueError("Frozen quality bank changed")
    series = old.KelmarshSeries(ROOT / p["data_path"], fold=fold, input_len=p["input_len"], horizon=p["pred_len"],
                               stride=p["train_stride"], start=p["start_utc"], end=p["end_utc_exclusive"])
    cutoff = series.bounds["train_end"]
    packs = {name: frozen.load_pack(source_run / f"train_{name}.npz")
             for name in ("memory_fit", "early_fit", "selection", "calibration")}
    for pack in packs.values():
        guard.clock_support(pack["times"], pack["turbines"], train_end=cutoff)
    if packs["selection"]["times"].max() > (packs["calibration"]["times"][:, 0] - np.timedelta64(1, "h")).min():
        raise ValueError("Numerical shrinkage fit overlaps later calibration")
    identity = dict(protocol_sha256=protocol_sha, source_prepared_sha256=old.sha(source_run / "prepared.json"),
        source_completed_sha256=old.sha(source / "completed.json"), checkpoints=prepared["identity"]["checkpoints"],
        quality_expert_id=bank["expert_id"], base=contract, data_sha256=p["data_sha256"],
        source_pack_hashes={name: guard.array_digest(pack) for name, pack in packs.items()},
        code_sha256={name: old.sha(ROOT / name) for name in ("scripts/train_kelmarsh_certified_fallback.py",
            "utils/certified_numeric_residual.py", "scripts/train_kelmarsh_incremental_wiki.py",
            "models/IncrementalWindWiki.py", "scripts/train_kelmarsh_contextual_expert.py",
            "scripts/train_kelmarsh_wiki_quality.py", "data_provider/kelmarsh_hourly.py",
            "utils/kelmarsh_wiki_quality.py", "utils/paired_forecast_statistics.py", "utils/gpu_runtime_policy.py")})
    if (output / "prepared.json").exists() and old.read(output / "prepared.json")["identity"] != identity:
        raise ValueError("v7 replay/protocol/executor changed on resume")
    parts, partition = guard.chronological_parts(packs["calibration"], 3, train_end=cutoff)
    numeric_evidence, selection, calibration = parts
    shrinkage = guard.fit_shrinkage(packs["selection"], train_end=cutoff)
    numerical = guard.certify(numeric_evidence, shrinkage, identity, spec["numeric_guard"], train_end=cutoff)
    # No outer labels are loaded here. This is the final base used in ALL arms.
    transformed = {}
    for name, pack in dict(memory_fit=packs["memory_fit"], early_fit=packs["early_fit"],
                           selection=selection, calibration=calibration).items():
        copy = guard.subset(pack, np.ones(len(pack["truth"]), bool))
        copy["raw_numeric"] = copy["numeric"].copy()
        if name in ("memory_fit", "early_fit"):
            # Offline fitting view of an already frozen model, NOT an online
            # historical backtest. Epoch selection/evaluation enforce clocks.
            copy["numeric"] = guard.predict_residual(copy["reference"], copy["raw_numeric"], numerical["weights"])
        else:
            copy["numeric"] = guard.apply(numerical, copy["reference"], copy["raw_numeric"], copy["times"], identity)
        transformed[name] = copy
    record = dict(identity=identity, partition=partition, numerical_gate_id=numerical["gate_id"],
                  train_end=str(cutoff), fit_view="Offline training view; not a historical online replay",
                  pack_hashes={name: guard.array_digest(pack) for name, pack in transformed.items()})
    if (output / "prepared.json").exists() and old.read(output / "prepared.json") != record:
        raise ValueError("v7 partition or numerical execution changed")
    old.write(output / "prepared.json", record)
    old.write(output / "numerical_gate.json", numerical)
    for name, pack in transformed.items():
        frozen.save_pack(output / f"train_{name}.npz", pack)
    print(f"[NUMERIC_GATE] alpha={shrinkage['alpha']} accepted={numerical['accepted']} weights={numerical['weights']}", flush=True)
    print(f"[TRAIN_PARTITION] {partition}", flush=True)
    return transformed, identity, numerical, (series, bank, contract, p, v5_run, v4_run, source_run, v5)


def interventions(prediction, baseline, truth, supported):
    active = np.abs(prediction - baseline) > 1e-9
    windows = active.any(1)
    gain = np.abs(baseline - truth) - np.abs(prediction - truth)
    return dict(coverage_window_pct=float(100 * windows.mean()), coverage_point_pct=float(100 * active.mean()),
        abstention_window_pct=float(100 * (~windows).mean()),
        selected_gain_kw=float(gain[active].mean()) if active.any() else None,
        harm_window_pct=float(100 * (gain.mean(1)[windows] < 0).mean()) if windows.any() else None,
        quality_supported_window_pct=float(100 * supported.any(1).mean()))


def evaluate(spec, output, trained, identity, numerical, information, device):
    # This function is reached only after ALL training/selection/certificates freeze.
    series, bank, contract, p, v5_run, v4_run, source_run, v5 = information
    validation = series.subset("validation")
    truth, anchor = validation.truth_and_anchor()
    predictions = dict(Persistence=anchor)
    for name, directory in (("OriginalTrend", v5_run / "OriginalTrend"),
                            ("NumericContextFrozen", v5_run / "NoWikiUnchanged"),
                            ("NumericCalibratedV4", v4_run / "NumericCalibrated"),
                            ("OldWikiIncrementV6", source_run / "reports/WikiIncrementSelective")):
        with np.load(directory / "raw_predictions.npz", allow_pickle=False) as data:
            for key, expected in dict(truth=truth, times=validation.target_times(),
                                       turbines=series.turbines[validation.window_starts]).items():
                np.testing.assert_array_equal(data[key], expected)
            predictions[name] = data["prediction"].copy()
    predictions["CertifiedNumeric"] = guard.apply(numerical, predictions["OriginalTrend"],
        predictions["NumericContextFrozen"], validation.target_times(), identity)
    meta = pd.read_csv(ROOT / p["data_path"], usecols=["timestamp_hour_end_utc", "turbine_id", "pitch_a_deg",
        "pitch_b_deg", "pitch_c_deg", "available_capacity_kw"])
    ledger = frozen.quality.ObservedQuality(meta, series, v5["quality"])
    pack, _ = previous.quality_pack(validation, predictions["OriginalTrend"], bank, contract, p["rules"], ledger,
        old.read(v4_run / "case_expert.json"), donors=False)
    pack["numeric"] = predictions["CertifiedNumeric"]
    supported = pack["overlap"].any(-1)
    actions = {}
    for name, (model, done, execution) in trained.items():
        raw = frozen.infer(model, pack, p["rated_power_kw"], device, name == "WikiIncrement")
        np.testing.assert_array_equal(raw[~supported], pack["numeric"][~supported])
        certificate = old.read(output / name / "benefit_certificate.json")
        routed = old.route(certificate, pack, pack["numeric"], raw, execution)
        predictions[name + "Raw"], predictions[name + "Selective"] = raw, routed["prediction"]
        actions[name + "Selective"] = routed["action"]
        actions[name + "Raw"] = (supported & (np.abs(raw - pack["numeric"]) > 1e-9)).astype(int)
    rows = []
    references = ("Persistence", "OriginalTrend", "NumericContextFrozen", "CertifiedNumeric", "ZeroMemoryIncrementSelective")
    for name, prediction in predictions.items():
        result = old.report(name, prediction, validation, output / "reports" / name, p["rated_power_kw"])
        if name in actions:
            arm = name.removesuffix("Raw").removesuffix("Selective")
            result["best_epoch"] = trained[arm][1]["best_epoch"]
            result["wiki"] = interventions(prediction, pack["numeric"], truth, supported)
            np.savez_compressed(output / "reports" / name / "interventions.npz", action=actions[name])
        result["paired"] = {reference: old.paired_forecast_statistics(prediction, predictions[reference], truth,
            pd.DataFrame({"forecast_start": pack["times"][:, 0]}), **spec["statistics"])
            for reference in references if reference != name}
        old.write(output / "reports" / name / "result.json", result)
        rows.append(dict(model=name, best_epoch=result.get("best_epoch"), **result["metrics"], **result.get("wiki", {})))
    values = {row["model"]: row for row in rows}
    method = values["WikiIncrementSelective"]
    success = all(method[key] < values[reference][key] for reference in
                  ("CertifiedNumeric", "ZeroMemoryIncrementSelective", "OriginalTrend", "Persistence") for key in ("mae", "rmse"))
    result = dict(results=rows, primary_joint_success=success, numerical_gate_weights=numerical["weights"],
        sealed_test_evaluated=False, outer_used_for_fitting=False,
        interpretation="Single historically inspected development fold; numerical fallback change is NOT Wiki contribution",
        numerical_correction=interventions(pack["numeric"], predictions["OriginalTrend"], truth, np.ones_like(supported)))
    old.write(output / "result.json", result)
    pd.DataFrame(rows).to_csv(output / "metrics_all_runs.csv", index=False)


def run(protocol, output, hours, *, shared_gpu=False):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / "suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = old.read(protocol)
        if spec["protocol_id"] != PROTOCOL_ID or spec["folds"] != [1] or spec["seed"] != 2024:
            raise ValueError("Not the frozen single-pilot protocol")
        saved = output / "protocol.json"
        if saved.exists() and old.sha(saved) != old.sha(protocol):
            raise ValueError("Protocol changed on resume")
        if not saved.exists():
            saved.write_bytes(protocol.read_bytes())
        if (output / "completed.json").exists():
            verify_directory(output, old.sha(saved))
            print("[REUSE_SUITE] completed, no repeated training", flush=True)
            return
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
        def record_resources(runtime):
            runtime.update(training_protocol_sha256=old.sha(saved), training_hyperparameters_changed=False,
                           shared_execution_overrides_only_protocol_idle_requirement=shared_gpu)
            old.write(output / "resource_policy.json", runtime)
        resources = GPURuntimePolicy(shared=shared_gpu, on_change=record_resources)
        if not shared_gpu:
            frozen.wait_for_gpu(output, hours)
        runtime = resources.configure()
        print(f"[RESOURCE_POLICY] {runtime}", flush=True)
        resources.check()
        old.write(output / "status.json", dict(status="RUNNING", pid=os.getpid()))
        old.write(output / "progress.json", dict(stage="training_only_numerical_guard"))
        packs, identity, numerical, information = prepare(spec, output, old.sha(saved))
        fit = old.concatenate_packs(packs["memory_fit"], packs["early_fit"])
        device, capacity = torch.device("cuda:0"), information[3]["rated_power_kw"]
        trained = {}
        for name in spec["arms"]:
            resources.check()
            old.write(output / "progress.json", dict(stage="train_incremental", arm=name))
            execution = dict(identity, numerical_gate_id=numerical["gate_id"], arm=name,
                             fit_sha256=guard.array_digest(fit), selection_sha256=guard.array_digest(packs["selection"]))
            model, done = frozen.train_arm(name, fit, packs["selection"], capacity, spec, output / name, execution, device,
                                          resource_check=resources.check, after_step=resources.after_step)
            cal = packs["calibration"]
            candidate = frozen.infer(model, cal, capacity, device, name == "WikiIncrement")
            execution = dict(execution, wiki_checkpoint_sha256=old.sha(output / name / "checkpoint.pt"))
            certificate = old.fit_certificate(cal, cal["numeric"], candidate, execution, spec["utility"])
            old.write(output / name / "benefit_certificate.json", certificate)
            old.write(output / name / "training_benefit_by_day.json", frozen.replay_diagnostic(cal, cal["numeric"], candidate,
                certificate, train_end=information[0].bounds["train_end"]))
            trained[name] = model, done, execution
        old.write(output / "progress.json", dict(stage="frozen_outer_report"))
        resources.check()
        evaluate(spec, output, trained, identity, numerical, information, device)
        artifacts = {str(path.relative_to(output)): old.sha(path) for path in output.rglob("*")
                     if path.is_file() and path.name not in ("suite.lock", "launch.log", "launcher.pid", "source_commits.log",
                                                            "status.json", "progress.json", "completed.json")}
        old.write(output / "completed.json", dict(protocol_sha256=old.sha(saved), artifacts=artifacts))
        old.write(output / "status.json", dict(status="COMPLETED", exit_code=0, pid=os.getpid()))
        old.write(output / "progress.json", dict(stage="COMPLETED"))
        print("[SUITE_COMPLETE] Inspect paired results; no automatic expansion", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/kelmarsh_certified_fallback_protocol_v7.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-wait-hours", type=float, default=24.)
    parser.add_argument("--allow-shared-gpu", action="store_true",
                        help="Adaptive: unrestricted when alone; 2 GiB allocator cap/headroom guard/yielding only with other GPU jobs")
    args = parser.parse_args()
    try:
        run(args.protocol.resolve(), args.output.resolve(), args.gpu_wait_hours, shared_gpu=args.allow_shared_gpu)
    except BlockingIOError:
        print("[BUSY] Existing v7 suite not modified", flush=True)
    except Exception as error:
        old.write(args.output / "status.json", dict(status="FAILED", error=f"{type(error).__name__}: {error}", pid=os.getpid()))
        raise
