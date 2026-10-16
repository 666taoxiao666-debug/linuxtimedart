#!/usr/bin/env python3
"""Single-variable, internally gated frozen-representation transfer experiment."""
from __future__ import annotations

import argparse
import gc
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
import numpy as np
import pandas as pd
import torch

from models.FrozenTransferTimeDART import Model
from scripts import train_kelmarsh_power_anchor as anchor

shared, prior, recent = anchor.shared, anchor.prior, anchor.recent
PROTOCOL = ROOT / "configs/kelmarsh_frozen_transfer_protocol_v22.json"
PROTOCOL_SHA256 = "13d36296551a3e744badd491955a5ca44247abe57fad71d1ee212275229ce594"
POINTER = ROOT / "outputs/logs/Kelmarsh/frozen_transfer_latest.txt"
SOURCE_FILES = tuple(dict.fromkeys(("scripts/train_kelmarsh_frozen_transfer.py",) + anchor.SOURCE_FILES))


def protocol():
    recipe = shared.read(PROTOCOL)
    if hashlib.sha256(PROTOCOL.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen v22 recipe changed")
    base = ROOT / recipe["base_protocol"]
    if hashlib.sha256(base.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != recipe["base_protocol_sha256"]:
        raise ValueError("Frozen v21 source recipe changed")
    old_recipe, effective = anchor.protocol()
    effective.update(protocol_id=recipe["protocol_id"], executed_models=[recipe["executed_model"]])
    return recipe, effective, old_recipe


def decision(rows, settings):
    jointly = sum(all(r["candidate"][k] < r["reference"][k] for k in ("mae", "rmse")) for r in rows)
    names = ("candidate", *settings["macro_jointly_better_than"])
    means = {name: {k: float(np.mean([r[name][k] for r in rows])) for k in ("mae", "rmse")} for name in names}
    passed = (jointly >= settings["minimum_jointly_improved_folds_vs_v21"] and
        all(means["candidate"][k] < means[name][k] for name in settings["macro_jointly_better_than"] for k in ("mae", "rmse")))
    return dict(passed=bool(passed), jointly_improved_folds=int(jointly), macro_inner=means, rows=rows,
                selection_scope="chronological internal selection only; includes frozen DLinear")


def audit_frozen_state(model, initial):
    """Reject missing/changed frozen tensors, including accidental wrong pretraining."""
    initial_state = torch.load(initial, map_location="cpu", weights_only=False)["model"]
    current = model.state_dict()
    audit = model.trainability_audit()
    for name in audit["frozen_names"]:
        if name not in initial_state or current[name].shape != initial_state[name].shape:
            raise ValueError(f"Frozen representation missing from matched pretrain: {name}")
        if not torch.equal(current[name].detach().cpu(), initial_state[name]):
            raise ValueError(f"Frozen representation changed: {name}")
    audit.update(initial_checkpoint_sha256=shared.sha(initial), frozen_tensors_bitwise_unchanged=True)
    return audit


def internal_selected_metrics(directory):
    state = shared.read(directory / "trained.json")
    return next(r["selection"] for r in shared.read(directory / "training_history.json") if r["epoch"] == state["best_epoch"])


def run(directory, gpu=0):
    import fcntl
    recipe, p, old = protocol()
    directory.mkdir(parents=True, exist_ok=True)
    POINTER.parent.mkdir(parents=True, exist_ok=True)
    with (POINTER.parent / "frozen_transfer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = {"recipe": recipe, "effective": p}
        for name, value in (("protocol.json", record), ("source_hashes.json", {n: shared.sha(ROOT / n) for n in SOURCE_FILES})):
            path = directory / name
            if path.exists() and shared.read(path) != value:
                raise ValueError(f"Immutable identity changed: {name}")
            shared.write(path, value)
        reference, numeric = ROOT / recipe["reference_dir"], ROOT / old["reference_dir"]
        for path, expected in ((reference, recipe["reference_completed_sha256"]), (numeric, old["reference_completed_sha256"])):
            if shared.sha(path / "completed.json") != expected:
                raise ValueError("Completed frozen reference changed")
            prior.verify_completed_suite(path)
        if shared.sha(ROOT / p["data_path"]) != p["data_sha256"]:
            raise ValueError("Prepared data changed")
        POINTER.write_text(str(directory), encoding="utf-8")
        if (directory / "completed.json").exists():
            prior.verify_completed_suite(directory)
            print("[REUSE] Completed v22; no retraining", flush=True)
            return
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        (directory / "status.env").write_text("STATUS=RUNNING\n", encoding="utf-8")
        device, rows, prepared = torch.device(f"cuda:{gpu}"), [], {}
        for fold in recipe["folds"]:
            seed = recipe["seeds"][0]
            series = shared.KelmarshSeries(ROOT / p["data_path"], fold=fold, input_len=24, horizon=4,
                stride=3, start=p["start_utc"], end=p["end_utc_exclusive"])
            data = anchor.train_datasets_only(series)
            prior_run = numeric / "runs" / f"f{fold}_s{seed}"
            pretrain = prior_run / "pretrain/checkpoint.pt"
            if shared.sha(pretrain) != old["pretrain_checkpoint_sha256"][str(fold)]:
                raise ValueError("Cached pretraining checkpoint changed")
            contract = shared.read(prior_run / "pretrain/trained.json")["contract"]
            for key, subset in (("fit_window_hash", "fit"), ("selection_window_hash", "selection")):
                if hashlib.sha256(data[subset].window_starts.tobytes()).hexdigest() != contract[key]:
                    raise ValueError("Pretrain training queue changed")
            np.testing.assert_array_equal(series.scaler.mean_, contract["scaler_mean"])
            np.testing.assert_array_equal(series.scaler.scale_, contract["scaler_scale"])
            output = directory / "runs" / f"f{fold}_s{seed}" / recipe["executed_model"]
            prior.wait_for_gpu(directory, fold, seed, "INNER_FROZEN_TRANSFER")
            model, state = shared.train_stage("PromptTimeDART", data["fit"], data["selection"], p,
                output, seed, device, initial=pretrain, model_factory=Model)
            trainability = audit_frozen_state(model, pretrain)
            shared.write(output / "trainability.json", trainability)
            truth, persistence = data["selection"].truth_and_anchor()
            measured = shared.model_metrics(shared.predict(model, data["selection"], seed, device), truth, persistence, p["rated_power_kw"])
            ref = reference / "runs" / f"f{fold}_s{seed}/PowerAnchoredTimeDART"
            old_count = shared.read(ref / "manifest.json")["parameters"]
            if trainability["total_parameters"] != old_count:
                raise ValueError("Total parameter count changed")
            row = dict(fold=fold, best_epoch=state["best_epoch"], candidate=measured,
                reference=internal_selected_metrics(ref), dlinear=internal_selected_metrics(prior_run / "DLinear"),
                persistence=shared.model_metrics(persistence, truth, persistence, p["rated_power_kw"]), trainability=trainability)
            rows.append(row)
            shared.write(output / "inner_metrics.json", row)
            shared.write(directory / "inner_records.json", rows)
            prepared[fold] = series, data, output, state
            print(f"[INNER] f{fold} MAE={measured['mae']:.6f} RMSE={measured['rmse']:.6f} frozen={trainability['frozen_parameters']}", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        gate = decision(rows, recipe["outer_gate"])
        shared.write(directory / "gate.json", gate)
        print(f"[TRAIN_ONLY_GATE] passed={gate['passed']} jointly_improved_folds={gate['jointly_improved_folds']}", flush=True)
        if gate["passed"]:
            for fold, (series, data, output, state) in prepared.items():
                seed = recipe["seeds"][0]
                prior.wait_for_gpu(directory, fold, seed, "OUTER_EVALUATION")
                data["validation"] = series.subset("validation")
                recent.assert_reference_cohort(series, data, numeric)
                model = Model(shared.model_args("PromptTimeDART", data["fit"], p, device)).to(device)
                model.load_state_dict(torch.load(output / "checkpoint.pt", map_location=device, weights_only=False)["model"], strict=True)
                prediction = shared.predict(model, data["validation"], seed, device)
                result = prior.write_report(recipe["executed_model"], prediction, data["validation"], output, p, state)
                for name, path in (("v21", reference / "runs" / f"f{fold}_s{seed}/PowerAnchoredTimeDART"),
                                   ("dlinear", numeric / "runs" / f"f{fold}_s{seed}/DLinear")):
                    with np.load(path / "raw_predictions.npz", allow_pickle=False) as old_prediction:
                        truth, _ = data["validation"].truth_and_anchor()
                        np.testing.assert_array_equal(truth, old_prediction["truth"])
                        result[f"paired_vs_{name}"] = prior.paired_forecast_statistics(prediction, old_prediction["prediction"], truth,
                            pd.DataFrame({"forecast_start": data["validation"].target_times()[:, 0]}), **p["statistics"])
                shared.write(output / "result.json", result)
                shared.write(output / "completed.json", {str(f.relative_to(output)): shared.sha(f)
                    for f in output.rglob("*") if f.is_file() and f.name != "completed.json"})
                prior.update_summary(directory)
                del model
                gc.collect()
                torch.cuda.empty_cache()
        shared.write(directory / "result.json", dict(outer_evaluated=gate["passed"], gate=gate,
            reference=str(reference), strong_baseline=str(numeric), wiki_enabled=False, labels_modified=False))
        shared.write(directory / "progress.json", dict(stage="COMPLETED", outer_evaluated=gate["passed"]))
        (directory / "status.env").write_text("STATUS=COMPLETED\nEXIT_CODE=0\n", encoding="utf-8")
        shared.write(directory / "completed.json", {str(f.relative_to(directory)): shared.sha(f)
            for f in directory.rglob("*") if f.is_file() and f.name not in
            ("suite.lock", "launch.log", "launcher.pid", "status.env", "progress.json", "last.pt")})
        print(f"[COMPLETED] {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        if not POINTER.exists():
            print("STATUS=NOT_STARTED")
            return
        directory = Path(POINTER.read_text().strip())
        print(f"RESULT_DIR={directory}")
        for name in ("status.env", "progress.json"):
            if (directory / name).exists():
                print((directory / name).read_text())
        return
    if args.output is None:
        raise ValueError("Independent output directory required")
    try:
        run(args.output.resolve(), args.gpu)
    except BlockingIOError:
        print("[BUSY] Existing v22 task left unchanged", flush=True)
    except Exception as error:
        if args.output.exists():
            (args.output / "status.env").write_text(f"STATUS=FAILED\nERROR={type(error).__name__}: {error}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
