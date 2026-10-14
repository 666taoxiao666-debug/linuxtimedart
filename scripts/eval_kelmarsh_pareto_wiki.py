#!/usr/bin/env python3
"""Train-utility refit and paired evaluation of a frozen-base Wiki readout."""
from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch

from data_provider.kelmarsh_hourly import KelmarshSeries
from models import PromptTimeDART
from scripts.train_kelmarsh_hourly import (read, write, sha, model_args, predict, block,
    report, model_metrics, wiki_diagnostics, gpu_available_to_this_process, verify_stage)
from utils import kelmarsh_hourly_wiki as old
from utils import kelmarsh_pareto_wiki as new
from utils.paired_forecast_statistics import paired_forecast_statistics


def train_audit(model, policy, blocks, contract):
    results = []
    for stage, data in blocks.items():
        output = new.candidates(model, data, contract)
        applied = new.apply(model, policy, data, contract)
        active = applied["action"] > 0
        alpha = np.take_along_axis(applied["alpha"], np.maximum(applied["action"] - 1, 0)[..., None], -1).squeeze(-1)
        results.append(dict(stage=stage, fit_scope="outer_train_donors_only; also used for policy fit, not independent evidence",
            windows=len(data["truth"]), supported_points=int(output["overlap"].sum()),
            locally_joint_profitable_points=int(output["support"].sum()),
            metrics=model_metrics(applied["prediction"], data["truth"], data["anchor"], contract["rated_power_kw"]),
            wiki=wiki_diagnostics(applied, data, contract),
            selected_alpha_mean=float(alpha[active].mean()) if active.any() else None,
            selected_alpha_lt_one_pct=float(100 * (alpha[active] < 1 - 1e-8).mean()) if active.any() else None))
    return results


def run(protocol_path, output, outer_validation, device):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / "suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = read(protocol_path)
        if spec["protocol_id"] != "kelmarsh_hourly_forecast_relative_pareto_wiki_v2" or spec["folds"] != [0, 1, 2] or spec["seed"] != 2024:
            raise ValueError("Undeclared pilot matrix")
        saved = output / "protocol.json"
        if saved.exists() and sha(saved) != sha(protocol_path):
            raise ValueError("Protocol changed on resume")
        if not saved.exists():
            saved.write_bytes(protocol_path.read_bytes())
        source = ROOT / spec["source_directory"]
        if sha(source / "protocol.json") != spec["source_protocol_sha256"]:
            raise ValueError("Source frozen protocol identity mismatch")
        protocol = read(source / "protocol.json")
        if sha(ROOT / protocol["data_path"]) != protocol["data_sha256"]:
            raise ValueError("Prepared source data changed")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
        (output / "status.env").write_text("STATUS=RUNNING\n", encoding="utf-8")
        records = []
        for fold in spec["folds"]:
            run_name = f"f{fold}_s{spec['seed']}"
            target, original = output / "runs" / run_name, source / "runs" / run_name
            target.mkdir(parents=True, exist_ok=True)
            done_path = target / ("outer_completed.json" if outer_validation else "train_completed.json")
            if done_path.exists():
                done = read(done_path)
                if done["protocol_sha256"] != sha(saved) or done["implementation_sha256"] != new.implementation_digest():
                    raise ValueError("Completed readout protocol or implementation changed")
                for path, expected in done["artifacts"].items():
                    if sha(target / path) != expected:
                        raise ValueError("Completed readout artifact changed")
                print(f"[REUSE] {run_name} completed outer={outer_validation}", flush=True)
                if outer_validation:
                    records.extend(read(target / name / "result.json") | {"fold": fold, "seed": spec["seed"]}
                                   for name in ("Trend", "LegacyWiki", "NumericControl", *spec["variants"]))
                continue
            gpu_available_to_this_process()
            write(output / "progress.json", dict(fold=fold, seed=spec["seed"], stage="train_utility_refit"))
            series = KelmarshSeries(ROOT / protocol["data_path"], fold=fold, input_len=protocol["input_len"],
                horizon=protocol["pred_len"], stride=protocol["train_stride"], start=protocol["start_utc"], end=protocol["end_utc_exclusive"])
            state = read(original / "PromptTimeDART/trained.json")
            verify_stage(original / "PromptTimeDART", state["contract"])
            fit = series.subset("fit")
            assert state["contract"]["fit_window_hash"] == hashlib.sha256(fit.window_starts.tobytes()).hexdigest()
            np.testing.assert_array_equal(state["contract"]["scaler_mean"], series.scaler.mean_)
            np.testing.assert_array_equal(state["contract"]["scaler_scale"], series.scaler.scale_)
            contract = read(original / "Wiki_full/utility_policy.json")["contract"]
            if contract["base_checkpoint_sha256"] != sha(original / "PromptTimeDART/checkpoint.pt") or contract["base_manifest_sha256"] != sha(original / "PromptTimeDART/manifest.json"):
                raise ValueError("Actual frozen base checkpoint mismatch")
            memory = read(original / "Wiki_full/expert.json")
            old.validate(memory, contract)
            args = model_args("PromptTimeDART", fit, protocol, device)
            base = PromptTimeDART.Model(args).to(device)
            base.load_state_dict(torch.load(original / "PromptTimeDART/checkpoint.pt", map_location=device, weights_only=False)["model"], strict=True)
            blocks = {name: block(series.subset(name), predict(base, series.subset(name), spec["seed"], device), protocol["rules"], donors=True)
                      for name in ("calibration_early", "calibration_late")}
            trained = {}
            for name, mode in spec["variants"].items():
                directory = target / name
                model = new.make_model(memory, contract, mode=mode, protocol_sha256=sha(saved),
                                       positive_fraction=spec["local_positive_donor_fraction"])
                policy = new.fit_policy(model, list(blocks.values()), contract, protocol["utility"])
                write(directory / "expert.json", model)
                write(directory / "utility_policy.json", policy)
                write(directory / "train_audit.json", train_audit(model, policy, blocks, contract))
                trained[name] = model, policy
                print(f"[TRAIN_FIT] {run_name} {name} accepted_cells={sum(row['accepted'] for row in policy['entries'].values())}", flush=True)
            write(target / "provenance.json", dict(protocol_sha256=sha(saved), source_contract=contract,
                  implementation_sha256=new.implementation_digest(), source_memory_expert_id=memory["expert_id"],
                  source_best_epoch=state["best_epoch"], backbone_retrained=False, memory_retrained=False,
                  outer_labels_used_for_policy=False, sealed_test_evaluated=False))
            if outer_validation:
                write(output / "progress.json", dict(fold=fold, seed=spec["seed"], stage="declared_outer_evaluation"))
                dataset = series.subset("validation")
                reference = predict(base, dataset, spec["seed"], device)
                with np.load(original / "PromptTimeDART/raw_predictions.npz") as cache:
                    np.testing.assert_allclose(reference, cache["prediction"], rtol=1e-8, atol=1e-7)
                    np.testing.assert_array_equal(dataset.target_times(), cache["times"])
                    np.testing.assert_array_equal(dataset.truth_and_anchor()[0], cache["truth"])
                evaluation = block(dataset, reference, protocol["rules"])
                for name, source_name in (("Trend", "PromptTimeDART"), ("LegacyWiki", "Wiki_no_composition"),
                                           ("NumericControl", "PromptTimeDART_numeric_control")):
                    result = read(original / source_name / "result.json")
                    result["model"] = name
                    write(target / name / "result.json", result)
                    records.append(result | {"fold": fold, "seed": spec["seed"]})
                for name, (model, policy) in trained.items():
                    applied = new.apply(model, policy, evaluation, contract)
                    result = report(name, applied["prediction"], dataset, target / name, protocol["rated_power_kw"])
                    result["best_epoch"] = state["best_epoch"]
                    result["wiki"] = wiki_diagnostics(applied, evaluation, contract)
                    result["paired_vs_trend"] = paired_forecast_statistics(applied["prediction"], reference, evaluation["truth"],
                        pd.DataFrame({"forecast_start": evaluation["times"][:, 0]}), **protocol["statistics"])
                    result["development_evidence"] = "outer validation already inspected historically; not independent confirmation"
                    write(target / name / "result.json", result)
                    np.savez_compressed(target / name / "interventions.npz", action=applied["action"], alpha=applied["alpha"],
                                        local_gain_kw=applied["local_gain_kw"], variance_kw2=applied["variance_kw2"])
                    records.append(result | {"fold": fold, "seed": spec["seed"]})
            retained = {"expert.json", "utility_policy.json", "train_audit.json", "provenance.json", "result.json", "raw_predictions.npz", "interventions.npz"}
            write(done_path, dict(protocol_sha256=sha(saved), implementation_sha256=new.implementation_digest(),
                artifacts={str(path.relative_to(target)): sha(path) for path in target.rglob("*") if path.name in retained}))
            if records:
                pd.DataFrame([{k: value[k] for k in ("model", "fold", "seed", "best_epoch")} | value["metrics"] | value.get("wiki", {})
                              for value in records]).drop(columns=["by_event", "by_granularity", "knowledge_seen_turbines", "knowledge_held_out_turbine6"], errors="ignore").to_csv(output / "metrics_all_runs.csv", index=False)
            del base
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[FOLD_COMPLETE] {run_name} outer={outer_validation}", flush=True)
        write(output / "progress.json", dict(stage="COMPLETED", outer_validation=outer_validation, folds=3))
        (output / "status.env").write_text("STATUS=COMPLETED\nEXIT_CODE=0\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/kelmarsh_pareto_protocol_v2.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-validation", action="store_true", help="One frozen, exploratory outer evaluation; never tuning")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    try:
        run(args.protocol.resolve(), args.output.resolve(), args.outer_validation, torch.device(f"cuda:{args.gpu}"))
    except BlockingIOError:
        print("[BUSY] Existing controller not modified", flush=True)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "status.env").write_text(f"STATUS=FAILED\nERROR={type(error).__name__}: {error}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
