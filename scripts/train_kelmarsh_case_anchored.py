#!/usr/bin/env python3
"""One case-anchored confidence pilot; reuse all completed v7 controls."""
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
from torch.utils.data import DataLoader, TensorDataset
from models.CaseAnchoredWindWiki import CaseAnchoredWindWiki, case_proposal
from scripts import train_kelmarsh_certified_fallback as previous
from utils.gpu_runtime_policy import GPURuntimePolicy

old, frozen, guard = previous.old, previous.frozen, previous.guard
PROTOCOL_ID = "kelmarsh_h2_case_anchored_confidence_v8"


def prepare(spec, output, protocol_sha):
    source = ROOT / spec["source_directory"]
    if old.sha(source / "protocol.json") != spec["source_protocol_sha256"]:
        raise ValueError("Frozen v7 protocol changed")
    previous.verify_directory(source, spec["source_protocol_sha256"])
    prepared = old.read(source / "prepared.json")
    numerical = old.read(source / "numerical_gate.json")
    if numerical["gate_id"] != prepared["numerical_gate_id"] or numerical["gate_id"] != guard.digest(
            {key: value for key, value in numerical.items() if key != "gate_id"}):
        raise ValueError("Frozen numerical execution identity changed")
    if prepared["identity"]["code_sha256"]["utils/certified_numeric_residual.py"] != old.sha(ROOT / "utils/certified_numeric_residual.py"):
        raise ValueError("Numerical gate implementation changed; no unassessed evidence reuse")
    prior = old.read(source / "protocol.json")
    if spec["training"] != prior["training"] or spec["utility"] != prior["utility"]:
        raise ValueError("Not a single-mechanism comparison: training or utility settings changed")
    source_v6 = ROOT / prior["source_directory"]
    if old.sha(source_v6 / "protocol.json") != prior["source_protocol_sha256"]:
        raise ValueError("Source v6 protocol changed")
    v5_root, v5, v4_root, v4, base_root, p = frozen.source_configuration(old.read(source_v6 / "protocol.json"))
    suffix = Path("runs") / "f1_s2024"
    v5_run, v4_run, base_run = [path / suffix for path in (v5_root, v4_root, base_root)]
    for name, path in dict(base=base_run / "PromptTimeDART/checkpoint.pt", numeric=v4_run / "NumericContext/checkpoint.pt",
                           old_wiki=v5_run / "WikiQualityContext/checkpoint.pt").items():
        if old.sha(path) != prepared["identity"]["checkpoints"][name]:
            raise ValueError("Actual executing frozen checkpoint changed")
    bank = old.read(v5_run / "quality_expert.json")
    if bank["expert_id"] != prepared["identity"]["quality_expert_id"]:
        raise ValueError("Quality case bank changed")
    packs = {name: frozen.load_pack(source / f"train_{name}.npz")
             for name in ("memory_fit", "early_fit", "selection", "calibration")}
    for name, pack in packs.items():
        guard.clock_support(pack["times"], pack["turbines"], train_end=prepared["train_end"])
        if guard.array_digest(pack) != prepared["pack_hashes"][name]:
            raise ValueError("Frozen fitting/selection/calibration data changed")
    if packs["selection"]["times"].max() > (packs["calibration"]["times"][:, 0] - np.timedelta64(1, "h")).min():
        raise ValueError("Epoch selection and new certificate periods overlap")
    identity = dict(protocol_sha256=protocol_sha, source_completed_sha256=old.sha(source / "completed.json"),
        source_prepared_sha256=old.sha(source / "prepared.json"), numerical_gate_id=numerical["gate_id"],
        checkpoints=prepared["identity"]["checkpoints"], base=prepared["identity"]["base"],
        quality_expert_id=bank["expert_id"], pack_hashes=prepared["pack_hashes"],
        source_sha256={name: old.sha(ROOT / name) for name in ("models/CaseAnchoredWindWiki.py",
            "models/IncrementalWindWiki.py", "scripts/train_kelmarsh_case_anchored.py", "scripts/train_kelmarsh_incremental_wiki.py",
            "scripts/train_kelmarsh_contextual_expert.py", "scripts/train_kelmarsh_wiki_quality.py", "utils/gpu_runtime_policy.py")})
    record = dict(identity=identity, source_run=spec["source_directory"], train_end=prepared["train_end"],
                  windows={name: len(pack["truth"]) for name, pack in packs.items()},
                  old_training_or_gate_repeated=False)
    if (output / "prepared.json").exists() and old.read(output / "prepared.json") != record:
        raise ValueError("Case-anchored execution changed on resume")
    old.write(output / "prepared.json", record)
    return packs, identity, numerical, (source, v5_run, v4_run, bank, p, v5, prepared["identity"]["base"])


def direct_prediction(pack, capacity, bound):
    inputs = frozen.inputs(pack, capacity)
    proposal = case_proposal(torch.from_numpy(inputs["memory"]), torch.from_numpy(inputs["supported"]), bound)
    return pack["numeric"] + capacity * proposal.numpy().astype(float)


@torch.no_grad()
def anchored_output(model, pack, capacity, device):
    model.eval()
    chunks = {key: [] for key in ("delta", "confidence", "proposal")}
    for batch in DataLoader(TensorDataset(*frozen.tensor_inputs(pack, capacity)), batch_size=256, shuffle=False):
        context, memory, state, supported = [value.to(device) for value in batch]
        out = model.components(context, state, memory, supported)
        for key in chunks:
            chunks[key].append(out[key].cpu().numpy().astype(float))
    result = {key: np.concatenate(value) for key, value in chunks.items()}
    if (np.any((result["confidence"] < 0) | (result["confidence"] > 1))
            or np.any(result["delta"] * result["proposal"] < -1e-12)
            or np.any(abs(result["delta"]) > abs(result["proposal"]) + 1e-8)):
        raise ValueError("Actual executing confidence/case constraint violated")
    result["prediction"] = pack["numeric"] + capacity * result["delta"]
    return result


def evaluate(spec, output, model, trained, executions, identity, numerical, information, device):
    source, v5_run, v4_run, bank, p, v5, contract = information
    series = old.KelmarshSeries(ROOT / p["data_path"], fold=1, input_len=p["input_len"], horizon=p["pred_len"],
        stride=p["train_stride"], start=p["start_utc"], end=p["end_utc_exclusive"])
    validation = series.subset("validation")
    truth, anchor = validation.truth_and_anchor()
    predictions = dict(Persistence=anchor)
    controls = dict(OriginalTrend="OriginalTrend", CertifiedNumeric="CertifiedNumeric", NumericContextFrozen="NumericContextFrozen",
        FreeWikiRaw="WikiIncrementRaw", FreeWikiSelective="WikiIncrementSelective",
        ZeroMemoryRaw="ZeroMemoryIncrementRaw", ZeroMemorySelective="ZeroMemoryIncrementSelective")
    for name, prior in controls.items():
        with np.load(source / "reports" / prior / "raw_predictions.npz", allow_pickle=False) as file:
            for key, expected in dict(truth=truth, times=validation.target_times(), turbines=series.turbines[validation.window_starts]).items():
                np.testing.assert_array_equal(file[key], expected)
            predictions[name] = file["prediction"].copy()
    from_gate = guard.apply(numerical, predictions["OriginalTrend"], predictions["NumericContextFrozen"],
                            validation.target_times(), numerical["execution"])
    np.testing.assert_array_equal(from_gate, predictions["CertifiedNumeric"])
    meta = pd.read_csv(ROOT / p["data_path"], usecols=["timestamp_hour_end_utc", "turbine_id", "pitch_a_deg", "pitch_b_deg",
                                                    "pitch_c_deg", "available_capacity_kw"])
    ledger = frozen.quality.ObservedQuality(meta, series, v5["quality"])
    pack, _ = frozen.previous.quality_pack(validation, predictions["OriginalTrend"], bank, contract, p["rules"], ledger,
                                          old.read(v4_run / "case_expert.json"), donors=False)
    pack["numeric"] = predictions["CertifiedNumeric"]
    actual = anchored_output(model, pack, p["rated_power_kw"], device)
    proposals = dict(CaseAnchoredWiki=actual["prediction"],
                     DirectCase=direct_prediction(pack, p["rated_power_kw"], spec["training"]["correction_bound_capacity"]))
    np.savez_compressed(output / "executed_case_trace.npz", confidence=actual["confidence"], proposal=actual["proposal"],
        delta=actual["delta"], times=pack["times"], turbines=pack["turbines"], overlap=pack["overlap"])
    supported = pack["overlap"].any(-1)
    actions = {}
    for name, candidate in proposals.items():
        np.testing.assert_array_equal(candidate[~supported], pack["numeric"][~supported])
        certificate = old.read(output / name / "benefit_certificate.json")
        routed = old.route(certificate, pack, pack["numeric"], candidate, executions[name])
        predictions[name + "Raw"], predictions[name + "Selective"] = candidate, routed["prediction"]
        actions[name + "Raw"] = (abs(candidate - pack["numeric"]) > 1e-9).astype(int)
        actions[name + "Selective"] = routed["action"]
    rows = []
    references = ("Persistence", "CertifiedNumeric", "FreeWikiSelective", "ZeroMemorySelective", "DirectCaseSelective")
    for name, prediction in predictions.items():
        if name in controls:
            result = old.read(source / "reports" / controls[name] / "result.json")
        else:
            result = old.report(name, prediction, validation, output / "reports" / name, p["rated_power_kw"])
            if name in actions:
                result["wiki"] = previous.interventions(prediction, pack["numeric"], truth, supported)
                result["best_epoch"] = trained["best_epoch"] if name.startswith("CaseAnchoredWiki") else None
                np.savez_compressed(output / "reports" / name / "interventions.npz", action=actions[name])
            result["paired"] = {ref: old.paired_forecast_statistics(prediction, predictions[ref], truth,
                pd.DataFrame({"forecast_start": pack["times"][:, 0]}), **spec["statistics"]) for ref in references if ref != name}
            old.write(output / "reports" / name / "result.json", result)
        rows.append(dict(model=name, best_epoch=result.get("best_epoch"), **result["metrics"], **result.get("wiki", {})))
    values = {row["model"]: row for row in rows}
    method = values["CaseAnchoredWikiSelective"]
    improved = all(method[key] < values[ref][key] for ref in references for key in ("mae", "rmse"))
    old.write(output / "result.json", dict(results=rows, primary_joint_success=improved,
        parameters=sum(parameter.numel() for parameter in model.parameters()), old_controls_retrained=False, sealed_test_evaluated=False,
        interpretation="One historically inspected development fold; confidence-direction constraint is a hypothesis, not guaranteed transfer"))
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
        def record_resources(state):
            old.write(output / "resource_policy.json", dict(state, training_protocol_sha256=old.sha(saved)))
        resources = GPURuntimePolicy(shared=shared_gpu, on_change=record_resources)
        if not shared_gpu:
            frozen.wait_for_gpu(output, hours)
        resources.configure()
        old.write(output / "status.json", dict(status="RUNNING", pid=os.getpid()))
        old.write(output / "progress.json", dict(stage="frozen_training_pack_verification"))
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
        packs, identity, numerical, information = prepare(spec, output, old.sha(saved))
        capacity, device = information[4]["rated_power_kw"], torch.device("cuda:0")
        fit = old.concatenate_packs(packs["memory_fit"], packs["early_fit"])
        old.write(output / "progress.json", dict(stage="train_case_anchored"))
        resources.check()
        model, trained = frozen.train_arm("WikiIncrement", fit, packs["selection"], capacity, spec, output / "CaseAnchoredWiki",
            identity, device, resource_check=resources.check, after_step=resources.after_step, model_factory=CaseAnchoredWindWiki)
        cal = packs["calibration"]
        actual = anchored_output(model, cal, capacity, device)
        candidates = dict(CaseAnchoredWiki=actual["prediction"],
                          DirectCase=direct_prediction(cal, capacity, spec["training"]["correction_bound_capacity"]))
        np.savez_compressed(output / "training_calibration_predictions.npz", numeric=cal["numeric"],
            anchored=candidates["CaseAnchoredWiki"], direct=candidates["DirectCase"], truth=cal["truth"],
            times=cal["times"], turbines=cal["turbines"], overlap=cal["overlap"],
            confidence=actual["confidence"], proposal=actual["proposal"], delta=actual["delta"])
        executions = {}
        for name, candidate in candidates.items():
            execution = dict(identity, executor=name, checkpoint_sha256=old.sha(output / "CaseAnchoredWiki/checkpoint.pt")
                             if name == "CaseAnchoredWiki" else None, fixed_case_confidence=1. if name == "DirectCase" else None)
            certificate = old.fit_certificate(cal, cal["numeric"], candidate, execution, spec["utility"])
            old.write(output / name / "benefit_certificate.json", certificate)
            old.write(output / name / "training_benefit_by_day.json", frozen.replay_diagnostic(cal, cal["numeric"], candidate,
                certificate, train_end=old.read(output / "prepared.json")["train_end"]))
            executions[name] = execution
        old.write(output / "progress.json", dict(stage="frozen_outer_report"))
        resources.check()
        evaluate(spec, output, model, trained, executions, identity, numerical, information, device)
        artifacts = {str(path.relative_to(output)): old.sha(path) for path in output.rglob("*") if path.is_file() and path.name
                     not in ("suite.lock", "launch.log", "launcher.pid", "source_commits.log", "status.json", "progress.json", "completed.json")}
        old.write(output / "completed.json", dict(protocol_sha256=old.sha(saved), artifacts=artifacts))
        old.write(output / "status.json", dict(status="COMPLETED", exit_code=0, pid=os.getpid()))
        old.write(output / "progress.json", dict(stage="COMPLETED"))
        print("[SUITE_COMPLETE] one anchored fit; frozen controls reused", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/kelmarsh_case_anchored_protocol_v8.json")
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
