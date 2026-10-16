#!/usr/bin/env python3
"""Matched early/late channel-interaction trial, with training-only acceptance."""
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

from models.ChannelTimingTimeDART import EarlyModel, LateModel
from scripts import train_kelmarsh_power_anchor as anchor
from scripts.train_kelmarsh_frozen_transfer import internal_selected_metrics
from utils.runtime_snapshot import capture_at_launch

shared, prior, recent = anchor.shared, anchor.prior, anchor.recent
PROTOCOL = ROOT / "configs/kelmarsh_channel_timing_protocol_v23.json"
PROTOCOL_SHA256 = "43f17d453a2b52eda6c1d102a72d9d89f7daba9d78c49378a0afb2ae8e58b6a4"
POINTER = ROOT / "outputs/logs/Kelmarsh/channel_timing_latest.txt"
FACTORIES = {"EarlyChannelTimeDART": EarlyModel, "LateChannelTimeDART": LateModel}
SOURCE_FILES = tuple(dict.fromkeys(("scripts/train_kelmarsh_channel_timing.py", "utils/runtime_snapshot.py",
    "scripts/train_kelmarsh_frozen_transfer.py") + anchor.SOURCE_FILES))


def protocol():
    recipe = shared.read(PROTOCOL)
    if hashlib.sha256(PROTOCOL.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen v23 recipe changed")
    base = ROOT / recipe["base_recipe"]
    if hashlib.sha256(base.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != recipe["base_recipe_sha256"]:
        raise ValueError("Frozen v21 recipe changed")
    old, p = anchor.protocol()
    p.update(protocol_id=recipe["protocol_id"], executed_models=list(FACTORIES))
    return recipe, p, old


def decision(rows, settings):
    jointly = sum(all(r["early"][k] < r[name][k] for name in ("late", "reference") for k in ("mae", "rmse")) for r in rows)
    names = ("early", *settings["macro_jointly_better_than"])
    means = {name: {k: float(np.mean([r[name][k] for r in rows])) for k in ("mae", "rmse")} for name in names}
    passed = jointly >= settings["minimum_jointly_improved_folds_vs_late_and_v21"] and all(
        means["early"][k] < means[name][k] for name in settings["macro_jointly_better_than"] for k in ("mae", "rmse"))
    return dict(passed=bool(passed), jointly_improved_folds=int(jointly), macro_inner=means,
        rows=rows, selection_scope="chronological internal selection; not an independent confirmation")


def history_power_partition(data, prediction, rated):
    truth, persistence = data.truth_and_anchor()
    fraction = persistence[:, 0] / rated
    labels = np.where(fraction < .05, "low_observed_power", np.where(fraction >= .9, "near_rated_observed_power", "middle_observed_power"))
    result = {}
    for name in ("low_observed_power", "middle_observed_power", "near_rated_observed_power"):
        keep = labels == name
        result[name] = dict(windows=int(keep.sum()), metrics=shared.model_metrics(prediction[keep], truth[keep], persistence[keep], rated) if keep.any() else None)
    assert sum(row["windows"] for row in result.values()) == len(truth)
    return result


def run(directory, gpu=0):
    import fcntl
    recipe, p, old = protocol()
    directory.mkdir(parents=True, exist_ok=True)
    POINTER.parent.mkdir(parents=True, exist_ok=True)
    with (POINTER.parent / "channel_timing.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = dict(recipe=recipe, effective=p)
        for name, value in (("protocol.json", record), ("source_hashes.json", {n: shared.sha(ROOT / n) for n in SOURCE_FILES})):
            path = directory / name
            if path.exists() and shared.read(path) != value:
                raise ValueError(f"Immutable identity changed: {name}")
            shared.write(path, value)
        reference, numeric = ROOT / recipe["reference_dir"], ROOT / old["reference_dir"]
        for path, expected in ((reference, recipe["reference_completed_sha256"]), (numeric, old["reference_completed_sha256"])):
            if shared.sha(path / "completed.json") != expected:
                raise ValueError("Completed reference identity changed")
            prior.verify_completed_suite(path)
        if shared.sha(ROOT / p["data_path"]) != p["data_sha256"]:
            raise ValueError("Prepared data changed")
        POINTER.write_text(str(directory), encoding="utf-8")
        if (directory / "completed.json").exists():
            prior.verify_completed_suite(directory)
            print("[REUSE] Completed v23; no retraining", flush=True)
            return
        capture_at_launch(directory, ROOT)
        runtime_identity = dict(factories={k:f"{v.__module__}.{v.__qualname__}" for k,v in FACTORIES.items()},
            external_llm_checkpoint=None, wiki_enabled=False, zero_residual_initialization_preserved=True,
            torch_version=torch.__version__, cuda_runtime_build=torch.version.cuda)
        if (directory / "runtime_identity.json").exists() and shared.read(directory / "runtime_identity.json") != runtime_identity:
            raise ValueError("Runtime build changed on resume")
        shared.write(directory / "runtime_identity.json", runtime_identity)
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
            num_run = numeric / "runs" / f"f{fold}_s{seed}"
            pretrain = num_run / "pretrain/checkpoint.pt"
            if shared.sha(pretrain) != old["pretrain_checkpoint_sha256"][str(fold)]:
                raise ValueError("Matched pretraining checkpoint changed")
            contract = shared.read(num_run / "pretrain/trained.json")["contract"]
            for key, stage in (("fit_window_hash", "fit"), ("selection_window_hash", "selection")):
                if hashlib.sha256(data[stage].window_starts.tobytes()).hexdigest() != contract[key]:
                    raise ValueError("Training queue changed")
            np.testing.assert_array_equal(series.scaler.mean_, contract["scaler_mean"])
            np.testing.assert_array_equal(series.scaler.scale_, contract["scaler_scale"])
            ref_run = reference / "runs" / f"f{fold}_s{seed}/PowerAnchoredTimeDART"
            row = dict(fold=fold, reference=internal_selected_metrics(ref_run),
                dlinear=internal_selected_metrics(num_run / "DLinear"), epochs={}, parameters={})
            truth, persistence = data["selection"].truth_and_anchor()
            row["persistence"] = shared.model_metrics(persistence, truth, persistence, p["rated_power_kw"])
            for name, factory in FACTORIES.items():
                output = directory / "runs" / f"f{fold}_s{seed}" / name
                prior.wait_for_gpu(directory, fold, seed, name)
                model, state = shared.train_stage("PromptTimeDART", data["fit"], data["selection"], p,
                    output, seed, device, initial=pretrain, model_factory=factory)
                predicted = shared.predict(model, data["selection"], seed, device)
                arm = recipe["arms"][name]
                row[arm] = shared.model_metrics(predicted, truth, persistence, p["rated_power_kw"])
                row["epochs"][arm] = state["best_epoch"]
                row["parameters"][arm] = sum(parameter.numel() for parameter in model.parameters())
                np.savez_compressed(output / "inner_predictions.npz", prediction=predicted, truth=truth, persistence=persistence,
                    times=data["selection"].target_times(), turbines=series.turbines[data["selection"].window_starts], window_starts=data["selection"].window_starts)
                shared.write(output / "inner_by_history_power.json", history_power_partition(data["selection"], predicted, p["rated_power_kw"]))
                prepared[fold, name] = series, data, output, state
                print(f"[INNER] f{fold} {arm} epoch={state['best_epoch']} MAE={row[arm]['mae']:.6f} RMSE={row[arm]['rmse']:.6f}", flush=True)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            if row["parameters"]["early"] != row["parameters"]["late"]:
                raise ValueError("Early/late parameter budgets differ")
            rows.append(row)
            shared.write(directory / "inner_records.json", rows)
        gate = decision(rows, recipe["outer_gate"])
        shared.write(directory / "gate.json", gate)
        print(f"[TRAIN_ONLY_GATE] passed={gate['passed']} joint_folds={gate['jointly_improved_folds']}", flush=True)
        if gate["passed"]:
            for (fold, name), (series, data, output, state) in prepared.items():
                seed = recipe["seeds"][0]
                prior.wait_for_gpu(directory, fold, seed, "OUTER_" + name)
                data["validation"] = series.subset("validation")
                recent.assert_reference_cohort(series, data, numeric)
                model = FACTORIES[name](shared.model_args("PromptTimeDART", data["fit"], p, device)).to(device)
                model.load_state_dict(torch.load(output / "checkpoint.pt", map_location=device, weights_only=False)["model"], strict=True)
                predicted = shared.predict(model, data["validation"], seed, device)
                result = prior.write_report(name, predicted, data["validation"], output, p, state)
                for label, other in (("v21", reference / "runs" / f"f{fold}_s{seed}/PowerAnchoredTimeDART"),
                                     ("dlinear", numeric / "runs" / f"f{fold}_s{seed}/DLinear")):
                    with np.load(other / "raw_predictions.npz", allow_pickle=False) as old_prediction:
                        truth, _ = data["validation"].truth_and_anchor()
                        np.testing.assert_array_equal(truth, old_prediction["truth"])
                        result[f"paired_vs_{label}"] = prior.paired_forecast_statistics(predicted, old_prediction["prediction"], truth,
                            pd.DataFrame({"forecast_start": data["validation"].target_times()[:, 0]}), **p["statistics"])
                shared.write(output / "result.json", result)
                shared.write(output / "by_history_power.json", history_power_partition(data["validation"], predicted, p["rated_power_kw"]))
                del model
                gc.collect()
                torch.cuda.empty_cache()
            for fold in recipe["folds"]:
                first = prepared[fold, "EarlyChannelTimeDART"][2]
                second = prepared[fold, "LateChannelTimeDART"][2]
                with np.load(first / "raw_predictions.npz", allow_pickle=False) as early, np.load(second / "raw_predictions.npz", allow_pickle=False) as late:
                    np.testing.assert_array_equal(early["truth"], late["truth"])
                    shared.write(first / "paired_vs_matched_late.json", prior.paired_forecast_statistics(early["prediction"], late["prediction"], early["truth"],
                        pd.DataFrame({"forecast_start": early["times"][:, 0]}), **p["statistics"]))
            for _, _, output, _ in prepared.values():
                shared.write(output / "completed.json", {str(f.relative_to(output)): shared.sha(f) for f in output.rglob("*") if f.is_file() and f.name != "completed.json"})
            prior.update_summary(directory)
        shared.write(directory / "result.json", dict(outer_evaluated=gate["passed"], gate=gate, wiki_enabled=False,
            labels_modified=False, data_cleaning=False, claim_scope=recipe["claim_scope"]))
        shared.write(directory / "progress.json", dict(stage="COMPLETED", outer_evaluated=gate["passed"]))
        (directory / "status.env").write_text("STATUS=COMPLETED\nEXIT_CODE=0\n", encoding="utf-8")
        shared.write(directory / "completed.json", {str(f.relative_to(directory)): shared.sha(f) for f in directory.rglob("*")
            if f.is_file() and f.name not in ("launch.log", "launcher.pid", "status.env", "progress.json", "last.pt")})
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
        root = Path(POINTER.read_text().strip())
        print(f"RESULT_DIR={root}")
        for name in ("status.env", "progress.json"):
            if (root / name).exists():
                print((root / name).read_text())
        return
    if args.output is None:
        raise ValueError("Independent output directory required")
    try:
        run(args.output.resolve(), args.gpu)
    except BlockingIOError:
        print("[BUSY] Existing v23 left unchanged", flush=True)
    except Exception as error:
        if args.output.exists():
            (args.output / "status.env").write_text(f"STATUS=FAILED\nERROR={type(error).__name__}: {error}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
