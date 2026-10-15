#!/usr/bin/env python3
"""One matched Wiki-key experiment. No repeated control training or outer fit."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "2"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import torch
from data_provider.kelmarsh_hourly import KelmarshWindows
from models.CaseAnchoredWindWiki import CaseAnchoredWindWiki
from scripts import train_kelmarsh_case_anchored as v8
from utils import kelmarsh_short_dynamics_wiki as dynamics
from utils.gpu_runtime_policy import GPURuntimePolicy

old, frozen, previous, guard = v8.old, v8.frozen, v8.previous, v8.guard
PROTOCOL_ID = "kelmarsh_h2_short_dynamics_case_keys_v9"


def source_series(p):
    return old.KelmarshSeries(ROOT / p["data_path"], fold=1, input_len=p["input_len"], horizon=p["pred_len"],
        stride=p["train_stride"], start=p["start_utc"], end=p["end_utc_exclusive"])


def quality_ledger(series, p, v5):
    meta = pd.read_csv(ROOT / p["data_path"], usecols=["timestamp_hour_end_utc", "turbine_id", "pitch_a_deg",
        "pitch_b_deg", "pitch_c_deg", "available_capacity_kw"])
    return frozen.quality.ObservedQuality(meta, series, v5["quality"])


def training_info(series, ledger, pack):
    """Recover only the saved training windows, with exact label/clock checks."""
    starts = series.starts
    lookup = {(int(series.turbines[i]), int(series.dates[i + series.input_len].astype(np.int64))): int(i) for i in starts}
    indices = [lookup[(int(unit), int(clock))] for unit, clock in
               zip(pack["turbines"], pack["times"][:, 0].astype("datetime64[ns]").astype(np.int64))]
    dataset = KelmarshWindows(series, np.asarray(indices), "saved_train_only")
    np.testing.assert_array_equal(dataset.target_times(), pack["times"])
    truth, anchor = dataset.truth_and_anchor()
    np.testing.assert_array_equal(truth, pack["truth"])
    np.testing.assert_array_equal(anchor, pack["anchor"])
    # Numerical histories themselves must not be changed by a Wiki-key patch.
    rows = dataset.window_starts[:, None] + np.arange(dataset.seq_len)[None]
    np.testing.assert_array_equal(series.values[rows].transpose(0, 2, 1), pack["history"])
    return ledger.context(dataset)


def control_reports(source, controls):
    inherited = dict(OriginalTrend="OriginalTrend", CertifiedNumeric="CertifiedNumeric", NumericContextFrozen="NumericContextFrozen",
        FreeWikiRaw="WikiIncrementRaw", FreeWikiSelective="WikiIncrementSelective",
        ZeroMemoryRaw="ZeroMemoryIncrementRaw", ZeroMemorySelective="ZeroMemoryIncrementSelective")
    result = {key: source / "reports" / value for key, value in inherited.items()}
    result.update({key: controls / "reports" / key for key in
                   ("CaseAnchoredWikiRaw", "CaseAnchoredWikiSelective", "DirectCaseSelective")})
    return result


def prepare(spec, output, protocol_sha):
    controls = ROOT / spec["anchored_control_directory"]
    if old.sha(controls / "protocol.json") != spec["anchored_control_protocol_sha256"]:
        raise ValueError("Completed anchored control protocol changed")
    previous.verify_directory(controls, spec["anchored_control_protocol_sha256"])
    # This reuses immutable old packs and checkpoints; it does NOT retrain v8.
    packs, identity, numerical, information = v8.prepare(spec, output / "source_check", protocol_sha)
    source, v5_run, v4_run, original_bank, p, v5, contract = information
    if old.sha(ROOT / p["data_path"]) != p["data_sha256"]:
        raise ValueError("Frozen input data changed")
    series = source_series(p)
    ledger = quality_ledger(series, p, v5)
    infos = {name: training_info(series, ledger, pack) for name, pack in packs.items()}
    bank = dynamics.fit_memory(original_bank, packs["memory_fit"], infos["memory_fit"], contract)
    transformed = {name: (pack if name == "memory_fit" else dynamics.transform(pack, infos[name], bank, contract))
                   for name, pack in packs.items()}
    for name, pack in transformed.items():
        for key in ("truth", "anchor", "reference", "numeric", "history", "times", "turbines"):
            np.testing.assert_array_equal(pack[key], packs[name][key])
    identity = dict(identity, short_case_expert_id=bank["expert_id"],
        anchored_controls_completed_sha256=old.sha(controls / "completed.json"),
        transformed_pack_hashes={name: guard.array_digest(pack) for name, pack in transformed.items()},
        changed_source_sha256={name: old.sha(ROOT / name) for name in (
            "utils/kelmarsh_short_dynamics_wiki.py", "scripts/train_kelmarsh_short_dynamics.py",
            "data_provider/kelmarsh_hourly.py", "utils/kelmarsh_wiki_quality.py", "utils/kelmarsh_episode_wiki.py")})
    record = dict(identity=identity, train_end=str(series.bounds["train_end"]),
        windows={name: len(pack["truth"]) for name, pack in packs.items()},
        added_case_labels=0, old_training_or_numerical_gate_repeated=False,
        coordinate_names=bank["short_names"], source_case_expert_id=original_bank["expert_id"])
    if (output / "prepared.json").exists() and old.read(output / "prepared.json") != record:
        raise ValueError("Short-dynamics execution identity changed on resume")
    old.write(output / "prepared.json", record)
    old.write(output / "short_case_expert.json", bank)
    for name, pack in transformed.items():
        frozen.save_pack(output / f"train_{name}.npz", pack)
    old.write(output / "training_input_audit.json", dict(
        original_case_windows=len(packs["memory_fit"]["truth"]),
        active_groups=[{key: group[key] for key in ("event", "state", "windows", "turbines", "available_ns")}
                       for group in bank["groups"] if group["active"]],
        support_before={name: float(pack["overlap"].any(-1).mean()) for name, pack in packs.items()},
        support_after={name: float(pack["overlap"].any(-1).mean()) for name, pack in transformed.items()},
        non_wiki_inputs_bit_identical=True, labels_bit_identical=True,
        bank_maturation_not_changed=True, outer_labels_read_for_fit=False))
    print(f"[KEY_AUDIT] same {len(packs['memory_fit']['truth'])} donor rows; new expert={bank['expert_id']}", flush=True)
    return transformed, identity, numerical, information, bank


def evaluate(spec, output, model, trained, executions, numerical, information, bank, device):
    source, v5_run, v4_run, _, p, v5, contract = information
    validation = source_series(p).subset("validation")
    truth, anchor = validation.truth_and_anchor()
    times, units = validation.target_times(), validation.series.turbines[validation.window_starts]
    controls = ROOT / spec["anchored_control_directory"]
    predictions = dict(Persistence=anchor)
    reports = control_reports(source, controls)
    for name, directory in reports.items():
        with np.load(directory / "raw_predictions.npz", allow_pickle=False) as saved:
            for key, expected in dict(truth=truth, times=times, turbines=units).items():
                np.testing.assert_array_equal(saved[key], expected)
            predictions[name] = saved["prediction"].copy()
    exact_base = guard.apply(numerical, predictions["OriginalTrend"], predictions["NumericContextFrozen"],
                             times, numerical["execution"])
    np.testing.assert_array_equal(exact_base, predictions["CertifiedNumeric"])
    ledger = quality_ledger(validation.series, p, v5)
    pack = old.pack_data(validation, predictions["OriginalTrend"], old.read(v4_run / "case_expert.json"),
                         contract, p["rules"], False)
    pack["numeric"] = exact_base
    pack = dynamics.transform(pack, ledger.context(validation), bank, contract)
    actual = v8.anchored_output(model, pack, p["rated_power_kw"], device)
    candidates = dict(DynamicsWiki=actual["prediction"],
        DirectDynamicsCase=v8.direct_prediction(pack, p["rated_power_kw"], spec["training"]["correction_bound_capacity"]))
    np.savez_compressed(output / "executed_case_trace.npz", confidence=actual["confidence"], proposal=actual["proposal"],
        delta=actual["delta"], times=times, turbines=units, overlap=pack["overlap"])
    supported, actions = pack["overlap"].any(-1), {}
    for name, candidate in candidates.items():
        np.testing.assert_array_equal(candidate[~supported], exact_base[~supported])
        certificate = old.read(output / name / "benefit_certificate.json")
        routed = old.route(certificate, pack, exact_base, candidate, executions[name])
        predictions[name + "Raw"], predictions[name + "Selective"] = candidate, routed["prediction"]
        actions[name + "Raw"] = (abs(candidate - exact_base) > 1e-9).astype(int)
        actions[name + "Selective"] = routed["action"]
    refs = ("Persistence", "CertifiedNumeric", "CaseAnchoredWikiSelective", "FreeWikiSelective", "ZeroMemorySelective",
            "DirectDynamicsCaseSelective")
    rows = []
    for name, prediction in predictions.items():
        if name in reports:
            result = old.read(reports[name] / "result.json")
        else:
            result = old.report(name, prediction, validation, output / "reports" / name, p["rated_power_kw"])
            if name in actions:
                result["wiki"] = previous.interventions(prediction, exact_base, truth, supported)
                result["best_epoch"] = trained["best_epoch"] if name.startswith("DynamicsWiki") else None
                np.savez_compressed(output / "reports" / name / "interventions.npz", action=actions[name])
            result["paired"] = {ref: old.paired_forecast_statistics(prediction, predictions[ref], truth,
                pd.DataFrame({"forecast_start": times[:, 0]}), **spec["statistics"]) for ref in refs if ref != name}
            old.write(output / "reports" / name / "result.json", result)
        rows.append(dict(model=name, best_epoch=result.get("best_epoch"), **result["metrics"], **result.get("wiki", {})))
    values = {row["model"]: row for row in rows}
    method = values["DynamicsWikiSelective"]
    success = all(method[key] < values[ref][key] for ref in refs for key in ("mae", "rmse"))
    old.write(output / "result.json", dict(results=rows, primary_joint_success=success,
        parameters=sum(parameter.numel() for parameter in model.parameters()), old_controls_retrained=False,
        numerical_inputs_modified=False, added_case_labels=0, sealed_test_evaluated=False,
        interpretation="One historically inspected development fold; short dynamics may help or hurt case transport"))
    pd.DataFrame(rows).to_csv(output / "metrics_all_runs.csv", index=False)


def run(protocol, output, hours, shared_gpu=True):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / "suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = old.read(protocol)
        if spec["protocol_id"] != PROTOCOL_ID or spec["folds"] != [1] or spec["seed"] != 2024:
            raise ValueError("Unknown single-pilot protocol")
        saved = output / "protocol.json"
        if saved.exists() and old.sha(saved) != old.sha(protocol):
            raise ValueError("Frozen protocol changed")
        if not saved.exists():
            saved.write_bytes(protocol.read_bytes())
        if (output / "completed.json").exists():
            previous.verify_directory(output, old.sha(saved))
            print("[REUSE_SUITE] completed; no repeated training", flush=True)
            return
        resources = GPURuntimePolicy(shared=shared_gpu,
            on_change=lambda state: old.write(output / "resource_policy.json", dict(state, training_protocol_sha256=old.sha(saved))))
        if not shared_gpu:
            frozen.wait_for_gpu(output, hours)
        resources.configure()
        old.write(output / "status.json", dict(status="RUNNING", pid=os.getpid()))
        old.write(output / "progress.json", dict(stage="same_case_new_key_preflight"))
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
        packs, identity, numerical, information, bank = prepare(spec, output, old.sha(saved))
        capacity, device = information[4]["rated_power_kw"], torch.device("cuda:0")
        fit = old.concatenate_packs(packs["memory_fit"], packs["early_fit"])
        old.write(output / "progress.json", dict(stage="train_short_dynamics"))
        resources.check()
        model, trained = frozen.train_arm("WikiIncrement", fit, packs["selection"], capacity, spec, output / "DynamicsWiki",
            identity, device, resource_check=resources.check, after_step=resources.after_step, model_factory=CaseAnchoredWindWiki)
        cal = packs["calibration"]
        actual = v8.anchored_output(model, cal, capacity, device)
        candidates = dict(DynamicsWiki=actual["prediction"],
            DirectDynamicsCase=v8.direct_prediction(cal, capacity, spec["training"]["correction_bound_capacity"]))
        np.savez_compressed(output / "training_calibration_predictions.npz", numeric=cal["numeric"],
            anchored=candidates["DynamicsWiki"], direct=candidates["DirectDynamicsCase"], truth=cal["truth"],
            times=cal["times"], turbines=cal["turbines"], overlap=cal["overlap"],
            confidence=actual["confidence"], proposal=actual["proposal"], delta=actual["delta"])
        executions = {}
        for name, candidate in candidates.items():
            execution = dict(identity, executor=name, checkpoint_sha256=old.sha(output / "DynamicsWiki/checkpoint.pt")
                if name == "DynamicsWiki" else None, fixed_case_confidence=1. if name == "DirectDynamicsCase" else None)
            certificate = old.fit_certificate(cal, cal["numeric"], candidate, execution, spec["utility"])
            old.write(output / name / "benefit_certificate.json", certificate)
            old.write(output / name / "training_benefit_by_day.json", frozen.replay_diagnostic(cal, cal["numeric"], candidate,
                certificate, train_end=old.read(output / "prepared.json")["train_end"]))
            executions[name] = execution
        old.write(output / "progress.json", dict(stage="frozen_outer_report"))
        resources.check()
        evaluate(spec, output, model, trained, executions, numerical, information, bank, device)
        artifacts = {str(path.relative_to(output)): old.sha(path) for path in output.rglob("*") if path.is_file() and path.name
            not in ("suite.lock", "launch.log", "launcher.pid", "source_commits.log", "status.json", "progress.json", "completed.json")}
        old.write(output / "completed.json", dict(protocol_sha256=old.sha(saved), artifacts=artifacts))
        old.write(output / "status.json", dict(status="COMPLETED", exit_code=0, pid=os.getpid()))
        old.write(output / "progress.json", dict(stage="COMPLETED"))
        print("[SUITE_COMPLETE] same cases; one new confidence fit; frozen controls reused", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/kelmarsh_short_dynamics_protocol_v9.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-wait-hours", type=float, default=24.)
    parser.add_argument("--idle-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args.protocol.resolve(), args.output.resolve(), args.gpu_wait_hours, not args.idle_only)
    except BlockingIOError:
        print("[BUSY] same experiment already running", flush=True)
    except Exception as error:
        old.write(args.output / "status.json", dict(status="FAILED", error=f"{type(error).__name__}: {error}", pid=os.getpid()))
        raise
