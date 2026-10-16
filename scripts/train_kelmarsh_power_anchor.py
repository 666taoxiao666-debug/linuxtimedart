#!/usr/bin/env python3
"""Training-gated power-feature transfer experiment, with no label cleaning."""
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

from data_provider.kelmarsh_hourly import KelmarshWindows
from models.PowerAnchoredTimeDART import Model
from scripts import train_kelmarsh_24h4h as prior
from scripts import train_kelmarsh_recent_numeric as recent

shared = prior.shared
PROTOCOL = ROOT / "configs/kelmarsh_power_anchor_protocol_v21.json"
PROTOCOL_SHA256 = "b0c83ac5bf96179865d635d1b1f846756b0d148ab278490b63c13e6c5bdca17b"
POINTER = ROOT / "outputs/logs/Kelmarsh/power_anchor_latest.txt"
SOURCE_FILES = tuple(dict.fromkeys(("scripts/train_kelmarsh_power_anchor.py",
    "scripts/train_kelmarsh_recent_numeric.py") + prior.SOURCE_FILES))


def protocol():
    recipe = shared.read(PROTOCOL)
    if hashlib.sha256(PROTOCOL.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen v21 recipe changed")
    base = ROOT / recipe["base_protocol"]
    if hashlib.sha256(base.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != recipe["base_protocol_sha256"]:
        raise ValueError("v20 training allocation changed")
    effective = shared.read(base)
    effective.update(protocol_id=recipe["protocol_id"], old_checkpoint_reuse=True,
        reused_checkpoint_scope="exact same-fold pretrain only", executed_models=[recipe["executed_model"]])
    return recipe, effective


def train_datasets_only(series):
    first = series.dates[series.starts + 24]
    last = series.dates[series.starts + 27]
    cutoff, end = series.bounds["calibration_early_end"], series.bounds["train_end"]
    masks = {"fit": last < cutoff, "selection": (first >= cutoff) & (last < end)}
    data = {name: KelmarshWindows(series, series.starts[mask], name) for name, mask in masks.items()}
    for dataset in data.values():
        dataset.train_cutoff, dataset.val_cutoff = cutoff, end
    return data


def decision(rows, settings):
    jointly = [r["candidate"]["mae"] < r["reference"]["mae"] and
               r["candidate"]["rmse"] < r["reference"]["rmse"] for r in rows]
    means = {name: {key: float(np.mean([r[name][key] for r in rows])) for key in ("mae", "rmse")}
             for name in ("candidate", "reference", "persistence")}
    passed = (sum(jointly) >= settings["minimum_jointly_improved_folds_vs_v20"] and
              all(means["candidate"][key] < means[name][key]
                  for name in ("reference", "persistence") for key in ("mae", "rmse")))
    return {"passed": bool(passed), "jointly_improved_folds": int(sum(jointly)), "macro_inner": means,
            "rows": rows, "selection_scope": "chronological internal selection only"}


def run(directory, gpu=0):
    import fcntl
    recipe, p = protocol()
    directory.mkdir(parents=True, exist_ok=True)
    POINTER.parent.mkdir(parents=True, exist_ok=True)
    with (POINTER.parent / "power_anchor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = {"recipe": recipe, "effective": p}
        if (directory / "protocol.json").exists() and shared.read(directory / "protocol.json") != record:
            raise ValueError("Frozen protocol changed on resume")
        shared.write(directory / "protocol.json", record)
        source = {name: shared.sha(ROOT / name) for name in SOURCE_FILES}
        if (directory / "source_hashes.json").exists() and shared.read(directory / "source_hashes.json") != source:
            raise ValueError("Implementation changed; never loosen the source lock")
        shared.write(directory / "source_hashes.json", source)
        reference = ROOT / recipe["reference_dir"]
        if shared.sha(reference / "completed.json") != recipe["reference_completed_sha256"]:
            raise ValueError("Completed reference changed")
        prior.verify_completed_suite(reference)
        if shared.sha(ROOT / p["data_path"]) != p["data_sha256"]:
            raise ValueError("Prepared source changed")
        POINTER.write_text(str(directory), encoding="utf-8")
        if (directory / "completed.json").exists():
            prior.verify_completed_suite(directory)
            print("[REUSE] Completed v21, no retraining", flush=True)
            return
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        (directory / "status.env").write_text("STATUS=RUNNING\n", encoding="utf-8")
        device, prepared, rows = torch.device(f"cuda:{gpu}"), {}, []
        for fold in recipe["folds"]:
            seed = recipe["seeds"][0]
            series = shared.KelmarshSeries(ROOT / p["data_path"], fold=fold, input_len=24, horizon=4,
                stride=3, start=p["start_utc"], end=p["end_utc_exclusive"])
            data = train_datasets_only(series)
            ref_run = reference / "runs" / f"f{fold}_s{seed}"
            pretrain = ref_run / "pretrain/checkpoint.pt"
            if shared.sha(pretrain) != recipe["pretrain_checkpoint_sha256"][str(fold)]:
                raise ValueError("Pretrained expert changed")
            old_contract = shared.read(ref_run / "pretrain/trained.json")["contract"]
            for key, stage in (("fit_window_hash", "fit"), ("selection_window_hash", "selection")):
                if hashlib.sha256(data[stage].window_starts.tobytes()).hexdigest() != old_contract[key]:
                    raise ValueError("Pretrain and candidate training queue differ")
            np.testing.assert_array_equal(series.scaler.mean_, old_contract["scaler_mean"])
            np.testing.assert_array_equal(series.scaler.scale_, old_contract["scaler_scale"])
            output = directory / "runs" / f"f{fold}_s{seed}" / recipe["executed_model"]
            prior.wait_for_gpu(directory, fold, seed, "INNER_POWER_ANCHOR")
            model, state = shared.train_stage("PromptTimeDART", data["fit"], data["selection"], p,
                output, seed, device, initial=pretrain, model_factory=Model)
            truth, anchor = data["selection"].truth_and_anchor()
            metrics = shared.model_metrics(shared.predict(model, data["selection"], seed, device), truth,
                                            anchor, p["rated_power_kw"])
            history = shared.read(ref_run / "PromptTimeDART/training_history.json")
            ref_epoch = shared.read(ref_run / "PromptTimeDART/trained.json")["best_epoch"]
            ref_metrics = next(r["selection"] for r in history if r["epoch"] == ref_epoch)
            parameter_count = shared.read(output / "manifest.json")["parameters"]
            reference_count = shared.read(ref_run / "PromptTimeDART/manifest.json")["parameters"]
            if parameter_count != reference_count:
                raise ValueError("Candidate changed the parameter budget")
            row = {"fold": fold, "best_epoch": state["best_epoch"], "candidate": metrics,
                   "reference": ref_metrics, "persistence": shared.model_metrics(anchor, truth, anchor, p["rated_power_kw"]),
                   "parameter_count": parameter_count, "reference_parameter_count": reference_count}
            rows.append(row)
            shared.write(output / "inner_metrics.json", row)
            shared.write(directory / "inner_records.json", rows)
            prepared[fold] = (series, data, output, state)
            print(f"[INNER] f{fold} candidate MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f} "
                  f"reference MAE={ref_metrics['mae']:.6f} RMSE={ref_metrics['rmse']:.6f}", flush=True)
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
                recent.assert_reference_cohort(series, data, reference)
                args = shared.model_args("PromptTimeDART", data["fit"], p, device)
                model = Model(args).to(device)
                model.load_state_dict(torch.load(output / "checkpoint.pt", map_location=device, weights_only=False)["model"], strict=True)
                prediction = shared.predict(model, data["validation"], seed, device)
                result = prior.write_report(recipe["executed_model"], prediction, data["validation"], output, p, state)
                with np.load(reference / "runs" / f"f{fold}_s{seed}/PromptTimeDART/raw_predictions.npz", allow_pickle=False) as old:
                    truth, _ = data["validation"].truth_and_anchor()
                    np.testing.assert_array_equal(truth, old["truth"])
                    result["paired_vs_v20"] = prior.paired_forecast_statistics(prediction, old["prediction"], truth,
                        pd.DataFrame({"forecast_start": data["validation"].target_times()[:, 0]}), **p["statistics"])
                shared.write(output / "result.json", result)
                shared.write(output / "completed.json", {str(f.relative_to(output)): shared.sha(f)
                    for f in output.rglob("*") if f.is_file() and f.name != "completed.json"})
                prior.update_summary(directory)
                del model
                gc.collect()
                torch.cuda.empty_cache()
        shared.write(directory / "result.json", {"outer_evaluated": gate["passed"], "gate": gate,
            "baseline_reference": str(reference), "wiki_enabled": False, "labels_modified": False,
            "note": "A failed inner gate is a negative result, not permission for another threshold or seed."})
        shared.write(directory / "progress.json", {"stage": "COMPLETED", "outer_evaluated": gate["passed"]})
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
        print("[BUSY] Existing v21 task left unchanged", flush=True)
    except Exception as error:
        if args.output.exists():
            (args.output / "status.env").write_text(f"STATUS=FAILED\nERROR={type(error).__name__}: {error}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
