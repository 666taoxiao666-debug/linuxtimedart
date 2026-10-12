#!/usr/bin/env python3
"""Fit physical event experts, bind their evidence, and check time/turbine transfer."""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd

from scripts.tune_sdwpf_accuracy import read_json, write_json, sha256
from scripts.train_sdwpf_residual_scale import metrics
from utils.physical_scale_guard import three_blocks
from utils.transport_event_wiki import (fit_experts, fit_transfer_policy, apply_transfer,
    candidates, calibration_blocks, validate_identity)
from utils.train_error_audit import EVENT_IDS
from utils.paired_forecast_statistics import paired_forecast_statistics
from utils.sdwpf_logging import finish_log_directory


def slice_context(context, mask):
    return {key:value[mask] for key,value in context.items()}


def verify_source(source):
    drift = read_json(source / "protocol.json")
    completed = read_json(source / "completed.json")
    scale_dir = Path(drift["source_dir"])
    scale = read_json(scale_dir / "protocol.json")
    audit_dir = Path(scale["source_audit"])
    audit = read_json(audit_dir / "protocol.json")
    pred = audit_dir / "reference_predictions.npz"
    checkpoint = Path(scale["source_checkpoint"]["checkpoint"])
    manifest = checkpoint.parent / "run_manifest.json"
    manifest_data = read_json(manifest)
    data = Path(manifest_data["args"]["root_path"]) / manifest_data["args"]["data_path"]
    config = ROOT / "configs/wind_event_factor_wiki.json"
    if (drift["protocol_id"] != "sdwpf_train_only_fixed_residual_drift_audit"
            or drift["outer_validation_used"] or drift["sealed_test_evaluated"]
            or sha256(pred) != drift["source_predictions_sha256"]
            or sha256(source/"history_context.npz") != completed["context_sha256"]
            or sha256(source/"report.json") != completed["report_sha256"]
            or sha256(scale_dir/"protocol.json") != drift["source_protocol_sha256"]
            or sha256(audit_dir/"protocol.json") != scale["source_audit_protocol_sha256"]
            or sha256(checkpoint) != drift["source_checkpoint_sha256"]
            or sha256(manifest) != scale["source_checkpoint"]["manifest_sha256"]
            or sha256(data) != drift["data_sha256"]
            or sha256(config) != audit["event_config_sha256"]):
        raise ValueError("Original train OOF, base forecast, history or event-rule identity changed")
    return drift, audit, pred, manifest


def evaluate(result, reference, truth, persistence, times):
    prediction = result["prediction"]
    changed = result["action"] > 0
    windows = changed.any(1)
    gain = np.abs(reference-truth)-np.abs(prediction-truth)
    window_gain = gain.mean(1)
    return dict(metrics=metrics(prediction, truth, persistence),
        gain_mae_kw=float(gain.mean()),
        gain_rmse_kw=float(np.sqrt(np.mean((reference-truth)**2))-np.sqrt(np.mean((prediction-truth)**2))),
        coverage_window_pct=float(100*windows.mean()), coverage_point_pct=float(100*changed.mean()),
        abstention_pct=float(100*(~windows).mean()),
        harm_window_pct=float(100*(window_gain[windows]<0).mean()) if windows.any() else None,
        selected_point_gain_kw=float(gain[changed].mean()) if changed.any() else None,
        history_overlap_window_pct=float(100*result["history_overlap"].any((1,2)).mean()),
        physical_candidate_window_pct=float(100*result["physical_available"].any((1,2)).mean()),
        evidence_supported_window_pct=float(100*result["evidence_supported"].any((1,2)).mean()),
        paired=paired_forecast_statistics(prediction, reference, truth,
            pd.DataFrame({"forecast_start":times[:, 0]}), block_length=12, replicates=5000, seed=2024))


def run(directory, source, protocol_path):
    import fcntl
    with (directory/"transport.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = read_json(protocol_path)
        if protocol["protocol_id"] != "sdwpf_expert_bound_physical_knowledge_transport_v12":
            raise ValueError("Unsupported transport protocol")
        saved = directory/"protocol.json"
        if saved.exists() and sha256(saved) != sha256(protocol_path):
            raise ValueError("Protocol changed on resume")
        if not saved.exists():
            saved.write_bytes(protocol_path.read_bytes())
        drift, audit, predictions, manifest = verify_source(source)
        if (protocol["transport"]["physical_power_edges"] != drift["power_ratio_edges"]
                or protocol["transport"]["physical_wind_edges_mps"] != drift["wind_mps_edges"]):
            raise ValueError("Frozen physical strata differ from the audited history")
        if (directory/"completed.json").exists():
            completed = read_json(directory/"completed.json")
            for name, expected in completed.items():
                if sha256(directory/name) != expected:
                    raise ValueError("Completed transport artifact changed")
            model = read_json(directory/"expert_model.json")
            validate_identity(model, model["contract"], read_json(directory/"utility_policy.json"))
            print("[TRANSPORT] Already complete; no training repeated", flush=True)
            finish_log_directory(directory, 0)
            return
        (directory/"status.env").write_text("STATUS=RUNNING\n", encoding="utf-8")
        with np.load(predictions, allow_pickle=False) as saved_arrays:
            arrays = {key:saved_arrays[key] for key in saved_arrays.files}
        with np.load(source/"history_context.npz", allow_pickle=False) as saved_arrays:
            context = {key:saved_arrays[key] for key in saved_arrays.files}
        times = arrays["target_timestamps"]
        np.testing.assert_array_equal(context["window_starts"], arrays["window_starts"])
        if not ((context["history_last_time"] < times[:, 0]).all()
                and (times[:, 0] >= np.datetime64(drift["selection_end"], "ns")).all()
                and (times[:, -1] < np.datetime64(drift["end_exclusive"], "ns")).all()):
            raise ValueError("History/target training-time boundary violated")
        early, calibration, check, midpoint, check_start = three_blocks(
            times, drift["selection_end"], drift["end_exclusive"])
        first, second, utility_cutoff = calibration_blocks(times, calibration)
        turbine = arrays["turbines"]
        held_out = turbine % 5 == 0
        source_mask = ~held_out
        contract = dict(base_checkpoint_sha256=drift["source_checkpoint_sha256"],
            base_manifest_sha256=sha256(manifest), event_config_sha256=audit["event_config_sha256"],
            data_sha256=drift["data_sha256"], input_len=336, pred_len=12, rated_power_kw=1500.,
            physical_coordinate_version="observed_last12_v1", feature_basis="physical_ridge_v1",
            event_ids=list(EVENT_IDS), source_turbines=sorted(int(v) for v in np.unique(turbine[source_mask])),
            residual_held_out_turbines=sorted(int(v) for v in np.unique(turbine[held_out])),
            backbone_seen_held_out_turbines=True)
        provenance = dict(contract=contract, protocol_sha256=sha256(protocol_path),
            source_predictions_sha256=sha256(predictions), context_sha256=sha256(source/"history_context.npz"),
            expert_fit_windows=int((early & source_mask).sum()),
            evidence_block_windows=[int((m & source_mask).sum()) for m in (first, second)],
            held_out_labels_used_for_fit=False, held_out_labels_used_for_calibration=False,
            source_check_windows=int((check & source_mask).sum()),
            residual_held_out_check_windows=int((check & held_out).sum()),
            midpoint=str(midpoint), utility_cutoff=str(utility_cutoff), check_start=str(check_start),
            check_target_min=str(times[check].min()), check_target_max=str(times[check].max()),
            neural_backbone_retrained=False, sealed_test_evaluated=False, outer_validation_used=False)
        write_json(directory/"provenance.json", provenance)
        fit = early & source_mask
        write_json(directory/"progress.json", dict(stage="expert_fit", pid=os.getpid()))
        model_path = directory/"expert_model.json"
        if model_path.exists():
            model = read_json(model_path); validate_identity(model, contract)
            print("[TRANSPORT] Reusing frozen fitted expert bundle", flush=True)
        else:
            model = fit_experts(slice_context(context, fit), arrays["prediction"][fit],
                arrays["persistence"][fit], arrays["events"][fit], arrays["truth"][fit],
                turbine[fit], contract, **{k:v for k,v in protocol["expert"].items()
                    if k not in ("type", "uses_hidden_backbone_features", "uses_turbine_id_as_predictor")})
            write_json(model_path, model)
        write_json(directory/"progress.json", dict(stage="bound_utility_calibration", pid=os.getpid(),
                   expert_bundle_id=model["expert_bundle_id"]))
        blocks = []
        for mask in (first & source_mask, second & source_mask):
            blocks.append(dict(context=slice_context(context, mask), reference=arrays["prediction"][mask],
                anchor=arrays["persistence"][mask], events=arrays["events"][mask],
                truth=arrays["truth"][mask], turbines=turbine[mask], times=times[mask]))
        policy_path = directory/"utility_policy.json"
        if policy_path.exists():
            policy = read_json(policy_path); validate_identity(model, contract, policy)
        else:
            settings = protocol["transport"]
            policy = fit_transfer_policy(model, blocks, contract,
                min_windows=settings["min_windows_per_calibration_block"],
                min_turbines=settings["min_turbines_per_calibration_block"],
                min_positive_fraction=settings["min_positive_turbine_fraction"])
            write_json(policy_path, policy)
        write_json(directory/"progress.json", dict(stage="frozen_time_turbine_check", pid=os.getpid()))
        report = dict(protocol_id=protocol["protocol_id"], provenance=provenance,
            expert_bundle_id=model["expert_bundle_id"], policy_id=policy["policy_id"],
            active_experts=[e["event_id"] for e in model["experts"] if e["active"]],
            accepted_evidence_cells=sum(row["accepted"] for row in policy["entries"].values()),
            evidence_status="exploratory_train_period_only", checks={})
        for name, mask in (("future_source_turbines", check & source_mask),
                           ("future_residual_held_out_turbines", check & held_out)):
            if not mask.any():
                raise ValueError("Both temporal and residual-turbine checks require observations")
            ctx = slice_context(context, mask)
            reference, truth, anchor = (arrays[k][mask] for k in ("prediction", "truth", "persistence"))
            guarded = apply_transfer(model, policy, ctx, reference, anchor, arrays["events"][mask], contract)
            coarse = apply_transfer(model, policy, ctx, reference, anchor, arrays["events"][mask], contract,
                                    event_only=True)
            report["checks"][name] = dict(windows=int(mask.sum()), turbines=int(len(np.unique(turbine[mask]))),
                reference=metrics(reference, truth, anchor), persistence=metrics(anchor, truth, anchor),
                event_only=evaluate(coarse, reference, truth, anchor, times[mask]),
                physical_transport=evaluate(guarded, reference, truth, anchor, times[mask]))
            np.savez_compressed(directory/(name+"_predictions.npz"), prediction=guarded["prediction"],
                event_only=coarse["prediction"], reference=reference, persistence=anchor, truth=truth,
                action=guarded["action"], target_timestamps=times[mask], turbines=turbine[mask],
                expert_bundle_id=np.asarray(model["expert_bundle_id"]), policy_id=np.asarray(policy["policy_id"]))
            print("[TRANSPORT] " + name + " " + str({
                k:report["checks"][name]["physical_transport"][k] for k in
                ("gain_mae_kw", "gain_rmse_kw", "coverage_window_pct", "harm_window_pct")}), flush=True)
        report["joint_improvement_in_both_checks"] = all(
            block["physical_transport"]["gain_mae_kw"] > 0 and block["physical_transport"]["gain_rmse_kw"] > 0
            for block in report["checks"].values())
        report["next_action"] = "review_train_only_evidence_no_automatic_outer_or_new_seed"
        write_json(directory/"result.json", report)
        write_json(directory/"completed.json", {name:sha256(directory/name) for name in
            ("result.json", "expert_model.json", "utility_policy.json", "provenance.json",
             "future_source_turbines_predictions.npz", "future_residual_held_out_turbines_predictions.npz")})
        write_json(directory/"progress.json", dict(stage="completed", pid=os.getpid()))
        finish_log_directory(directory, 0)
        print("[TRANSPORT] Complete. joint_improvement_in_both_checks="+
              str(report["joint_improvement_in_both_checks"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--drift-dir", required=True, type=Path)
    parser.add_argument("--protocol", default=ROOT/"configs/sdwpf_knowledge_transport_protocol_v12.json", type=Path)
    args = parser.parse_args()
    try:
        run(args.result_dir.resolve(), args.drift_dir.resolve(), args.protocol.resolve())
    except BlockingIOError:
        print("[TRANSPORT] Existing process owns the run; no duplicate started", flush=True)
        return
    except Exception:
        finish_log_directory(args.result_dir.resolve(), 1)
        raise


if __name__ == "__main__":
    main()
