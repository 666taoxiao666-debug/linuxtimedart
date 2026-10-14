#!/usr/bin/env python3
"""Audit v5 training certificate support, without neural inference or training."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scripts import train_kelmarsh_wiki_quality as runner
from utils import kelmarsh_wiki_quality as quality
from utils.wiki_evidence_support_audit import clock_support, certificate_support_audit

old = runner.old


def metadata_overlap(expert, query, info, contract):
    """Reproduce only v5's label-free neighbour mask, not candidate predictions."""
    legacy = quality.legacy
    if (expert["expert_id"] != legacy.history.digest({k: v for k, v in expert.items() if k != "expert_id"})
            or expert["contract"] != contract
            or expert["implementation_sha256"] != quality.implementation_digest()
            or expert["readout_sha256"] != quality.episode.implementation_digest()
            or expert["coordinate_sha256"] != legacy.implementation_digest()
            or expert["neighbor_sha256"] != legacy.neighbors.implementation_digest()):
        raise ValueError("Frozen quality expert implementation/identity changed")
    times = np.asarray(query["times"]).astype("datetime64[ns]")
    if np.isnat(times).any():
        raise ValueError("Invalid target clock")
    origins = times[:, 0].astype(np.int64) - 3600 * 10**9
    if np.any(info["last_observed_ns"] > origins):
        raise ValueError("Future operating metadata")
    cfg = expert["settings"]
    keys = legacy.keys(info["context"], query["anchor"], contract["rated_power_kw"], cfg["use_controls"])
    overlap = np.zeros((*times.shape, len(legacy.EVENTS)), bool)
    for group in expert["groups"]:
        event = group["event"]
        mask = query["events"][:, event] & (info["state"] == group["state"])
        if not group["active"] or not mask.any():
            continue
        if np.any(origins[mask] < group["available_ns"]):
            raise ValueError("Source knowledge outcomes are not mature")
        mean, scale = np.asarray(group["mean"]), np.asarray(group["scale"])
        _, distance, diversity = legacy.neighbors._neighbors(
            (keys[mask] - mean) / scale, (np.asarray(group["keys"]) - mean) / scale,
            np.asarray(group["units"]), cfg["neighbors"])
        overlap[mask, :, event] = ((distance <= group["radius"]) &
                                    (diversity >= cfg["min_neighbor_turbines"]))[:, None]
    return overlap


def run(suite, output):
    # A separate output prevents mutation of the completed experiment's manifest.
    suite, output = suite.resolve(), output.resolve()
    if suite == output or suite in output.parents:
        raise ValueError("Audit output must be outside the completed suite")
    spec = old.read(suite / "protocol.json")
    if spec["protocol_id"] != "kelmarsh_h2_wiki_only_observed_quality_v5":
        raise ValueError("This audit is only for the frozen v5 suite")
    source = ROOT / spec["source_directory"]
    if old.sha(source / "protocol.json") != spec["source_protocol_sha256"]:
        raise ValueError("v4 source protocol changed")
    v4 = old.read(source / "protocol.json")
    base_root = ROOT / v4["source_directory"]
    if old.sha(base_root / "protocol.json") != v4["source_protocol_sha256"]:
        raise ValueError("Original protocol changed")
    protocol = old.read(base_root / "protocol.json")
    if old.sha(ROOT / protocol["data_path"]) != protocol["data_sha256"]:
        raise ValueError("Frozen data file changed")
    meta = pd.read_csv(ROOT / protocol["data_path"], usecols=[
        "timestamp_hour_end_utc", "turbine_id", "pitch_a_deg", "pitch_b_deg", "pitch_c_deg", "available_capacity_kw"])
    report = dict(suite=str(suite), protocol_sha256=old.sha(suite / "protocol.json"),
                  audit_script_sha256=old.sha(Path(__file__)),
                  audit_utility_sha256=old.sha(ROOT / "utils/wiki_evidence_support_audit.py"),
                  neural_inference=False, training=False, outer_target_values_used=False,
                  sealed_test_evaluated=False, folds={})
    for fold in spec["folds"]:
        target = suite / "runs" / f"f{fold}_s{spec['seed']}"
        runner.verify_completed(target)
        completed = old.read(target / "completed.json")
        if (completed["protocol_sha256"] != old.sha(suite / "protocol.json")
                or completed["runner_sha256"] != old.sha(ROOT / "scripts/train_kelmarsh_wiki_quality.py")
                or completed["quality_source_sha256"] != quality.implementation_digest()):
            raise ValueError("Completed suite protocol or implementation changed")
        certificate = old.read(target / "benefit_certificate.json")
        expert = old.read(target / "quality_expert.json")
        execution = certificate["execution"]
        if execution["protocol_sha256"] != old.sha(suite / "protocol.json"):
            raise ValueError("Certificate belongs to a different protocol")
        original = source / "runs" / f"f{fold}_s{spec['seed']}"
        contract = execution["base"]
        checks = [(target / "WikiQualityContext/checkpoint.pt", execution["wiki_checkpoint_sha256"]),
                  (original / "NumericContext/checkpoint.pt", execution["numerical_checkpoint_sha256"]),
                  (base_root / "runs" / f"f{fold}_s{spec['seed']}" / "PromptTimeDART/checkpoint.pt", contract["base_checkpoint_sha256"])]
        if any(old.sha(path) != expected for path, expected in checks):
            raise ValueError("Executing expert/base checkpoint identity changed")
        if expert["expert_id"] != execution["quality_expert_id"]:
            raise ValueError("Certificate describes a different knowledge expert")
        if certificate["policy_id"] != quality.legacy.history.digest({k: v for k, v in certificate.items() if k != "policy_id"}):
            raise ValueError("Benefit certificate changed")
        series = old.KelmarshSeries(ROOT / protocol["data_path"], fold=fold,
            input_len=protocol["input_len"], horizon=protocol["pred_len"], stride=protocol["train_stride"],
            start=protocol["start_utc"], end=protocol["end_utc_exclusive"])
        (selection, calibration), _ = old.split_late(series)
        ledger = quality.ObservedQuality(meta, series, spec["quality"])
        datasets = {}
        for name, dataset in (("selection", selection), ("calibration", calibration)):
            ids = series.turbines[dataset.window_starts]
            donors = ids != 6
            times = dataset.target_times()
            context = dataset.physical_context()
            last = series.raw[dataset.window_starts + dataset.seq_len - 1, -1]
            anchor = np.repeat(last[:, None], dataset.pred_len, axis=1)
            query = dict(times=times, anchor=anchor,
                         events=quality.legacy.physical_events(context, protocol["rules"]))
            overlap = metadata_overlap(expert, query, ledger.context(dataset), contract)
            pack = dict(times=times[donors], turbines=ids[donors], overlap=overlap[donors],
                        context={k: v[donors] for k, v in context.items()})
            datasets[name] = pack
        result = certificate_support_audit(datasets["calibration"], certificate, train_end=series.bounds["train_end"])
        result["selection_partition"] = clock_support(datasets["selection"]["times"], datasets["selection"]["turbines"], train_end=series.bounds["train_end"])
        result["certificate_sha256"] = old.sha(target / "benefit_certificate.json")
        result["actual_outputs_sha256_recorded_not_recomputed"] = certificate["actual_outputs_sha256"]
        report["folds"][str(fold)] = result
        print(f"[AUDIT] f{fold}: {result['partition']['windows']} windows, "
              f"{result['partition']['unique_issue_times']} issue clocks, {result['partition']['utc_day_blocks']} UTC day blocks", flush=True)
    old.write(output, report)
    print(f"[COMPLETE] {output}; no neural inference/training or policy changes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.suite, args.output)
