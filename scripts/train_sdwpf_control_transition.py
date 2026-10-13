#!/usr/bin/env python3
"""Opt-in, CPU-only control/transition factorial pilot on original-train OOF."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
import pandas as pd

from scripts.tune_sdwpf_accuracy import read_json, write_json, sha256
from scripts.train_sdwpf_knowledge_transport import verify_source, slice_context
from scripts.audit_sdwpf_residual_drift import train_dataset
from scripts.train_sdwpf_residual_scale import metrics
from utils import control_transition_wiki as wiki
from utils.physical_scale_guard import three_blocks
from utils.transport_event_wiki import calibration_blocks
from utils.paired_forecast_statistics import paired_forecast_statistics
from utils.sdwpf_logging import finish_log_directory


def reconstruct_controls(manifest, audit, arrays):
    """Recover the original-unit controls using the saved training-only scaler."""
    args = manifest["args"]
    if not args.get("sdwpf_causal_fill", True):
        raise ValueError("Noncausal sensor filling cannot supply transfer evidence")
    data = train_dataset(args)
    if data.feature_columns != args["feature_columns"]:
        raise ValueError("Source feature order/angle convention changed")
    for key, actual in (("mean", data.scaler.mean_), ("scale", data.scaler.scale_)):
        np.testing.assert_allclose(actual, manifest["datasets"]["train"]["scaler"][key], rtol=1e-12, atol=1e-12)
    if (np.datetime64(data.train_cutoff, "ns") != np.datetime64(audit["fit_cutoff"], "ns")
            or np.datetime64(data.val_cutoff, "ns") != np.datetime64(audit["selection_end"], "ns")):
        raise ValueError("Source scaler fit/selection boundaries changed")
    starts = arrays["window_starts"]
    history_rows = starts[:, None]+336-12+np.arange(12)
    target_rows = starts[:, None]+336+np.arange(12)
    dates = np.asarray(data.dates, dtype="datetime64[ns]")
    np.testing.assert_array_equal(dates[target_rows], arrays["target_timestamps"])
    np.testing.assert_array_equal(np.asarray(data.turbines)[starts+336], arrays["turbines"])
    if not (dates[history_rows] < arrays["target_timestamps"][:, :1]).all():
        raise ValueError("Control context includes future observations")
    def inverse(name):
        i = data.feature_columns.index(name)
        return data.data_x[history_rows, i].astype(float)*data.scaler.scale_[i]+data.scaler.mean_[i]
    # Legacy Wdir_cos is the original relative sensor, NOT legacy yaw_cos
    # (Wdir-Ndir). Corrected-frame sources must use their yaw_cos instead.
    cosine_name = "yaw_cos" if args.get("sdwpf_angle_mode", "legacy_difference") == "relative_wdir" else "Wdir_cos"
    return wiki.control_context(inverse("Pab_mean"), inverse(cosine_name))


def control_audit(context, arrays, masks):
    """Descriptive only: fixed 5-degree bins are NOT used by the model."""
    rows = []
    for block, mask in masks.items():
        for k, event in enumerate(("gust", "high_wind_low_power", "rated", "low_wind_idle")):
            for label, group in (("pitch_le5", context["pitch_mean_div90"] <= 5/90),
                                 ("pitch_gt5", context["pitch_mean_div90"] > 5/90)):
                chosen = mask & arrays["events"][:, k] & group
                if chosen.any():
                    rows.append(dict(block=block, event=event, diagnostic_bin=label,
                        windows=int(chosen.sum()), pitch_mean_degrees=float(context["pitch_mean_div90"][chosen].mean()*90),
                        relative_yaw_cos_mean=float(context["relative_yaw_cos_mean"][chosen].mean()),
                        residual_kw=float((arrays["truth"]-arrays["prediction"])[chosen].mean()),
                        power_transition_kw=float((arrays["truth"]-arrays["persistence"])[chosen].mean())))
    return dict(rows=rows, uses_check_labels=False, uses_outer_labels=False,
                bin_scope="diagnostic only, not fitted or used in retrieval")


def preflight(source, protocol_path):
    """Read-only source/data-flow check: no fitting or check-label statistics."""
    protocol = read_json(protocol_path)
    drift, audit, predictions, manifest = verify_source(source)
    if (protocol["protocol_id"] != "sdwpf_control_conditioned_transition_memory_v15"
            or sha256(predictions) != protocol["source_predictions_sha256"]
            or drift["source_checkpoint_sha256"] != protocol["source_base_checkpoint_sha256"]):
        raise ValueError("Preflight source does not match the declared protocol")
    with np.load(predictions, allow_pickle=False) as content:
        arrays = {key: content[key] for key in content.files}
    controls = reconstruct_controls(read_json(manifest), audit, arrays)
    early, cal, _, _, _ = three_blocks(arrays["target_timestamps"], drift["selection_end"], drift["end_exclusive"])
    first, second, _ = calibration_blocks(arrays["target_timestamps"], cal)
    source_mask = arrays["turbines"] % 5 != 0
    return dict(protocol_id=protocol["protocol_id"], protocol_sha256=sha256(protocol_path),
        source_predictions_sha256=sha256(predictions), control_keys=list(controls),
        windows=len(arrays["truth"]), model_fitting_performed=False, check_labels_aggregated=False,
        outer_validation_used=False, sealed_test_evaluated=False,
        train_control_audit=control_audit(controls, arrays, dict(source_fit=early & source_mask,
            calibration_early=first & source_mask, calibration_late=second & source_mask)))


def evaluate(result, reference, truth, anchor, times, statistics):
    active = result["action"] > 0; windows = active.any(1)
    gain = np.abs(reference-truth)-np.abs(result["prediction"]-truth)
    paired = paired_forecast_statistics(result["prediction"], reference, truth,
        pd.DataFrame({"forecast_start": times[:, 0]}), **statistics)
    paired["interpretation"] = ("Exploratory original-train check. Window-uniform ratio of issue-time cluster sums; "
        "CI and HAC test concern the same primary MAE gain. Not independent confirmation.")
    return dict(metrics=metrics(result["prediction"], truth, anchor),
        gain_mae_kw=float(gain.mean()),
        gain_rmse_kw=float(np.sqrt(np.mean((reference-truth)**2))-
                           np.sqrt(np.mean((result["prediction"]-truth)**2))),
        coverage_window_pct=float(100*windows.mean()), coverage_point_pct=float(100*active.mean()),
        abstention_pct=float(100*(~windows).mean()),
        harm_window_pct=float(100*(gain.mean(1)[windows] < 0).mean()) if windows.any() else None,
        selected_point_gain_kw=float(gain[active].mean()) if active.any() else None,
        physical_candidate_window_pct=float(100*result["physical_available"].any((1, 2)).mean()),
        history_overlap_window_pct=float(100*result["history_overlap"].any((1, 2)).mean()),
        evidence_supported_window_pct=float(100*result["evidence_supported"].any((1, 2)).mean()), paired=paired)


def verify_completed(directory, contract):
    state = read_json(directory/"completed.json")
    for name, expected in state.items():
        if sha256(directory/name) != expected:
            raise ValueError("Completed control/transition artifact changed: "+name)
    model = read_json(directory/"expert_model.json")
    wiki.validate_identity(model, contract, read_json(directory/"utility_policy.json"))
    return read_json(directory/"result.json")


def candidate_audit(model, blocks, contract):
    report = {}
    for name, block in blocks.items():
        prediction, physical, overlap = wiki.candidates(model, block["context"], block["reference"],
            block["anchor"], block["events"], contract)
        reference, truth = block["reference"], block["truth"]
        rows = []
        for k, event in enumerate(contract["event_ids"]+["composition"]):
            mask = physical[..., k]; supported = overlap[..., k]
            gain = np.abs(reference-truth)-np.abs(prediction[..., k]-truth)
            rows.append(dict(event=event, physical_windows=int(mask.any(1).sum()),
                supported_windows=int(supported.any(1).sum()),
                correction_mean_kw=float((prediction[..., k]-reference)[mask].mean()) if mask.any() else None,
                candidate_mae_gain_kw=float(gain[mask].mean()) if mask.any() else None,
                candidate_rmse_gain_kw=float(np.sqrt(np.mean((reference-truth)[mask]**2))-
                    np.sqrt(np.mean((prediction[..., k]-truth)[mask]**2))) if mask.any() else None,
                supported_mae_gain_kw=float(gain[supported].mean()) if supported.any() else None))
        report[name] = rows
    return dict(expert_bundle_id=model["expert_bundle_id"], blocks=report,
                uses_check_labels=False, uses_outer_labels=False)


def run(directory, source, protocol_path):
    import fcntl
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    with (directory/"pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = read_json(protocol_path)
        if protocol["protocol_id"] != "sdwpf_control_conditioned_transition_memory_v15":
            raise ValueError("Unsupported declared pilot")
        expected_arms = [("residual_no_control", "residual", False), ("residual_control", "residual", True),
                         ("transition_no_control", "transition", False), ("transition_control", "transition", True)]
        if ([(a["name"], a["readout"], a["use_controls"]) for a in protocol["arms"]] != expected_arms
                or (protocol["fold"], protocol["seed"], protocol["input_len"], protocol["pred_len"],
                    protocol["rated_power_kw"]) != (1, 2024, 336, 12, 1500)
                or protocol["primary_arm"] != "transition_control"
                or protocol["outer_validation_used"] or protocol["sealed_test_evaluated"]):
            raise ValueError("Declared fixed arms or training-only pilot scope changed")
        saved = directory/"protocol.json"
        if saved.exists() and sha256(saved) != sha256(protocol_path):
            raise ValueError("Protocol changed on resume")
        if not saved.exists():
            saved.write_bytes(protocol_path.read_bytes())
        drift, audit, predictions, manifest_path = verify_source(source)
        if (sha256(predictions) != protocol["source_predictions_sha256"]
                or drift["source_checkpoint_sha256"] != protocol["source_base_checkpoint_sha256"]):
            raise ValueError("Pilot source differs from declared fold/seed/OOF base")
        with np.load(predictions, allow_pickle=False) as content:
            arrays = {key: content[key] for key in content.files}
        with np.load(source/"history_context.npz", allow_pickle=False) as content:
            context = {key: content[key] for key in content.files}
        times, ids = arrays["target_timestamps"], arrays["turbines"]
        np.testing.assert_array_equal(context["window_starts"], arrays["window_starts"])
        if not ((context["history_last_time"] < times[:, 0]).all()
                and (times[:, 0] >= np.datetime64(drift["selection_end"], "ns")).all()
                and (times[:, -1] < np.datetime64(drift["end_exclusive"], "ns")).all()):
            raise ValueError("Original-train history/target boundaries violated")
        early, calibration, check, midpoint, check_start = three_blocks(times, drift["selection_end"], drift["end_exclusive"])
        first, second, utility_cutoff = calibration_blocks(times, calibration)
        source_mask = ids % 5 != 0; fit = early & source_mask
        contract = dict(base_checkpoint_sha256=drift["source_checkpoint_sha256"],
            base_manifest_sha256=sha256(manifest_path), data_sha256=drift["data_sha256"],
            event_config_sha256=audit["event_config_sha256"], input_len=336, pred_len=12, rated_power_kw=1500.,
            physical_coordinate_version="last12_control_relative_wdir_v1",
            event_ids=list(("gust_or_turbulent", "high_wind_low_power", "rated_saturation", "low_wind_idle")),
            source_turbines=sorted(int(v) for v in np.unique(ids[source_mask])),
            residual_held_out_turbines=sorted(int(v) for v in np.unique(ids[~source_mask])),
            backbone_seen_held_out_turbines=True)
        # Resume cannot silently change executable, context extraction or source.
        provenance = dict(contract=contract, protocol_sha256=sha256(protocol_path),
            source_predictions_sha256=sha256(predictions), source_context_sha256=sha256(source/"history_context.npz"),
            runner_sha256=hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest(),
            statistics_implementation_sha256=hashlib.sha256((ROOT/"utils/paired_forecast_statistics.py").read_text(encoding="utf-8").encode()).hexdigest(),
            expert_implementation_sha256=wiki.implementation_digest(),
            expert_fit_windows=int(fit.sum()), evidence_block_windows=[int((m & source_mask).sum()) for m in (first, second)],
            midpoint=str(midpoint), utility_cutoff=str(utility_cutoff), check_start=str(check_start),
            outer_validation_used=False, sealed_test_evaluated=False, held_out_labels_used_for_fit=False,
            held_out_labels_used_for_calibration=False, neural_backbone_retrained=False)
        prov_path = directory/"provenance.json"
        if prov_path.exists() and read_json(prov_path) != provenance:
            raise ValueError("Pilot provenance changed on resume")
        if not prov_path.exists():
            write_json(prov_path, provenance)
        if (directory/"completed.json").exists():
            for name, expected in read_json(directory/"completed.json").items():
                if sha256(directory/name) != expected:
                    raise ValueError("Completed pilot changed: "+name)
            for arm in protocol["arms"]:
                verify_completed(directory/arm["name"], contract)
            finish_log_directory(directory, 0)
            print("[CONTROL] Already complete; no fitting repeated", flush=True)
            return
        (directory/"status.env").write_text("STATUS=RUNNING\n", encoding="utf-8")
        context_path = directory/"control_context.npz"
        if context_path.exists():
            marker = read_json(directory/"context_completed.json")
            if marker != dict(context_sha256=sha256(context_path), provenance_sha256=sha256(prov_path)):
                raise ValueError("Historical control cache changed")
            with np.load(context_path, allow_pickle=False) as content:
                controls = {key: content[key] for key in content.files}
        else:
            write_json(directory/"progress.json", dict(stage="history_controls", pid=os.getpid()))
            controls = reconstruct_controls(read_json(manifest_path), audit, arrays)
            np.savez_compressed(context_path, **controls)
            write_json(directory/"context_completed.json", dict(context_sha256=sha256(context_path), provenance_sha256=sha256(prov_path)))
        context.update(controls)
        write_json(directory/"train_control_audit.json", control_audit(context, arrays,
            dict(source_fit=fit, calibration_early=first & source_mask, calibration_late=second & source_mask)))
        blocks = []
        for mask in (first & source_mask, second & source_mask):
            blocks.append(dict(context=slice_context(context, mask), reference=arrays["prediction"][mask],
                anchor=arrays["persistence"][mask], events=arrays["events"][mask], truth=arrays["truth"][mask],
                turbines=ids[mask], times=times[mask]))
        report = dict(protocol_id=protocol["protocol_id"], primary_arm=protocol["primary_arm"],
            provenance=provenance, no_winner_selection_on_check=True, arms={})
        for arm in protocol["arms"]:
            name = arm["name"]; arm_dir = directory/name; arm_dir.mkdir(exist_ok=True)
            write_json(directory/"progress.json", dict(stage="fixed_arm", arm=name, pid=os.getpid()))
            if (arm_dir/"completed.json").exists():
                report["arms"][name] = verify_completed(arm_dir, contract)
                print("[CONTROL] Reusing completed "+name, flush=True)
                continue
            model_path, policy_path = arm_dir/"expert_model.json", arm_dir/"utility_policy.json"
            if model_path.exists():
                model = read_json(model_path); wiki.validate_identity(model, contract)
            else:
                model = wiki.fit_experts(slice_context(context, fit), arrays["prediction"][fit], arrays["persistence"][fit],
                    arrays["events"][fit], arrays["truth"][fit], ids[fit], contract,
                    readout=arm["readout"], use_controls=arm["use_controls"], **protocol["expert"])
                write_json(model_path, model)
            if policy_path.exists():
                policy = read_json(policy_path); wiki.validate_identity(model, contract, policy)
            else:
                policy = wiki.fit_transfer_policy(model, blocks, contract,
                    **{k: v for k, v in protocol["transport"].items() if k != "decision"})
                write_json(policy_path, policy)
            arm_report = dict(expert_bundle_id=model["expert_bundle_id"], policy_id=policy["policy_id"],
                readout=model["readout"], use_controls=model["use_controls"],
                accepted_evidence_cells=sum(row["accepted"] for row in policy["entries"].values()), checks={})
            write_json(arm_dir/"train_candidate_audit.json", candidate_audit(model,
                dict(calibration_early=blocks[0], calibration_late=blocks[1]), contract))
            for scope, mask in (("future_source_turbines", check & source_mask),
                                ("future_residual_held_out_turbines", check & ~source_mask)):
                ref, truth, anchor = (arrays[key][mask] for key in ("prediction", "truth", "persistence"))
                result = wiki.apply_transfer(model, policy, slice_context(context, mask), ref, anchor,
                    arrays["events"][mask], contract)
                arm_report["checks"][scope] = dict(windows=int(mask.sum()), turbines=int(len(np.unique(ids[mask]))),
                    reference=metrics(ref, truth, anchor), physical_transport=evaluate(result, ref, truth, anchor,
                    times[mask], protocol["statistics"]))
                np.savez_compressed(arm_dir/(scope+"_predictions.npz"), prediction=result["prediction"],
                    reference=ref, persistence=anchor, truth=truth, action=result["action"],
                    target_timestamps=times[mask], turbines=ids[mask])
            write_json(arm_dir/"result.json", arm_report)
            files = ("result.json", "expert_model.json", "utility_policy.json", "train_candidate_audit.json",
                     "future_source_turbines_predictions.npz", "future_residual_held_out_turbines_predictions.npz")
            write_json(arm_dir/"completed.json", {f: sha256(arm_dir/f) for f in files})
            report["arms"][name] = arm_report
            print("[CONTROL] Complete "+name+" accepted_cells="+str(arm_report["accepted_evidence_cells"]), flush=True)
        report["next_action"] = "review_all_declared_arms_no_automatic_outer_or_seed_followup"
        # Render only the predeclared primary arm, never pick a plot by which
        # ablation happens to look strongest. If plotting fails, all saved arm
        # completion receipts remain reusable; no model is refitted on resume.
        from utils.forecast_report import _plot_accuracy_overview
        primary_dir = directory/protocol["primary_arm"]
        plot_files = []
        for scope in ("future_source_turbines", "future_residual_held_out_turbines"):
            with np.load(primary_dir/(scope+"_predictions.npz"), allow_pickle=False) as values:
                plot_dir = primary_dir/(scope+"_artifacts"); plot_dir.mkdir(exist_ok=True)
                _plot_accuracy_overview(values["prediction"], values["truth"], str(plot_dir),
                    persistence=values["persistence"], reference=values["reference"], rated_power=1500,
                    model_name="Control-conditioned transition memory",
                    scope_caption="Original-train development only, NOT outer validation/test; "+scope+
                                  "; held-out refers to residual knowledge, not the frozen backbone.")
                plot_files.append(str((plot_dir/"forecast_accuracy_overview.png").relative_to(directory)))
        write_json(directory/"result.json", report)
        summary = ["SCOPE=exploratory_original_train_only", "PRIMARY_ARM="+protocol["primary_arm"]]
        for name, arm in report["arms"].items():
            for scope, block in arm["checks"].items():
                value = block["physical_transport"]; m = value["metrics"]
                r2_text = "undefined" if m["r2"] is None else f"{m['r2']:.6f}"
                summary.append(f"{name} {scope}: MAE={m['mae']:.6f} RMSE={m['rmse']:.6f} "
                    f"R2={r2_text} gain_kw={value['gain_mae_kw']:.6f} coverage={value['coverage_window_pct']:.3f}% "
                    f"harm={value['harm_window_pct']} abstain={value['abstention_pct']:.3f}%")
        (directory/"summary.txt").write_text("\n".join(summary)+"\n", encoding="utf-8")
        names = ["result.json", "protocol.json", "provenance.json", "control_context.npz",
                 "context_completed.json", "train_control_audit.json", "summary.txt"]
        names += [arm["name"]+"/completed.json" for arm in protocol["arms"]]
        names += plot_files
        write_json(directory/"completed.json", {f: sha256(directory/f) for f in names})
        write_json(directory/"progress.json", dict(stage="completed", pid=os.getpid()))
        finish_log_directory(directory, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--drift-dir", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=ROOT/"configs/sdwpf_control_transition_protocol_v15.json")
    parser.add_argument("--preflight", action="store_true", help="Read-only input audit; no expert/policy fitting")
    args = parser.parse_args()
    if args.preflight:
        import json
        print("[PREFLIGHT] "+json.dumps(preflight(args.drift_dir.resolve(), args.protocol.resolve())), flush=True)
        return
    if args.result_dir is None:
        parser.error("--result-dir is required unless --preflight is used")
    try:
        run(args.result_dir.resolve(), args.drift_dir.resolve(), args.protocol.resolve())
    except BlockingIOError:
        print("[CONTROL] Another process owns the pilot; no duplicate started", flush=True)
        return
    except Exception:
        finish_log_directory(args.result_dir.resolve(), 1)
        raise


if __name__ == "__main__":
    main()
