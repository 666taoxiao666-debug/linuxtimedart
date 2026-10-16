#!/usr/bin/env python3
"""Use recent chronological train data while keeping v19 evaluation unchanged."""
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
from sklearn.linear_model import Ridge

from data_provider.kelmarsh_hourly import KelmarshWindows
from scripts import train_kelmarsh_24h4h as prior

shared = prior.shared
PROTOCOL = ROOT / "configs/kelmarsh_recent_train_protocol_v20.json"
PROTOCOL_SHA256 = "600ef0bf4bc21ae59dc1a6701c6948ab0ac234936e205e3035208e384d9ebc9a"
POINTER = ROOT / "outputs/logs/Kelmarsh/hourly_recent_latest.txt"
SOURCE_FILES = tuple(dict.fromkeys(("scripts/train_kelmarsh_recent_numeric.py",) + prior.SOURCE_FILES))


def validate_protocol(p):
    digest = hashlib.sha256(PROTOCOL.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if digest != PROTOCOL_SHA256 or p != shared.read(PROTOCOL):
        raise ValueError("Only exact frozen v20 configuration is supported")
    old = shared.read(prior.PROTOCOL)
    for key in ("data_path", "data_sha256", "input_len", "pred_len", "train_stride", "eval_stride",
                "folds", "seeds", "models", "classical", "training", "pretraining", "feature_columns"):
        if p[key] != old[key]:
            raise ValueError("An unrelated mechanism changed: " + key)
    if p["wiki_enabled"] or p["old_checkpoint_reuse"]:
        raise ValueError("No Wiki or stale neural checkpoint reuse in this experiment")


def recent_datasets(series):
    """Expand fitting, but do NOT refit scaler or modify raw inputs/targets."""
    first = series.dates[series.starts + series.input_len]
    last = series.dates[series.starts + series.input_len + series.horizon - 1]
    cutoff, end = series.bounds["calibration_early_end"], series.bounds["train_end"]
    masks = {"fit": last < cutoff, "selection": (first >= cutoff) & (last < end)}
    data = {name: KelmarshWindows(series, series.starts[mask], name) for name, mask in masks.items()}
    for dataset in data.values():
        dataset.train_cutoff, dataset.val_cutoff = cutoff, end
        if not len(dataset):
            raise ValueError("Empty chronological numeric stage")
    data["validation"] = series.subset("validation")
    return data


def assert_reference_cohort(series, datasets, reference):
    audit = shared.read(reference / f"data_audit_f{series.audit['fold']}.json")
    np.testing.assert_array_equal(series.scaler.mean_, audit["scaler_mean"])
    np.testing.assert_array_equal(series.scaler.scale_, audit["scaler_scale"])
    validation = datasets["validation"]
    truth, anchor = validation.truth_and_anchor()
    with np.load(reference / "runs" / f"f{series.audit['fold']}_s2024/Persistence/raw_predictions.npz", allow_pickle=False) as saved:
        for key, value in (("truth", truth), ("persistence", anchor), ("times", validation.target_times()),
                           ("window_starts", validation.window_starts),
                           ("turbines", validation.turbines[validation.window_starts])):
            np.testing.assert_array_equal(value, saved[key], err_msg="v19 evaluation queue changed: " + key)


def train_audit(series, datasets):
    old_fit, old_select = series.subset("fit"), series.subset("selection")
    end = series.bounds["train_end"]
    raw = series.raw[series.dates < end]
    quality = {}
    for k, name in enumerate(shared.FEATURES):
        v = raw[:, k]
        quality[name] = {"missing_rows": int((~np.isfinite(v)).sum()), "minimum": np.nanmin(v),
                         "median": np.nanmedian(v), "p99": np.nanquantile(v, .99), "maximum": np.nanmax(v)}
    result = prior.audit_dataset(series, datasets, shared.read(PROTOCOL))
    result.update({"diagnostic_label_scope": "outer-train only; outer labels compared for queue identity, never for fitting",
        "numeric_fit_end_exclusive": str(series.bounds["calibration_early_end"]),
        "numeric_selection_end_exclusive": str(end), "scaler_unchanged_from_v19": True,
        "old_fit_windows": len(old_fit), "old_selection_windows": len(old_select),
        "fit_window_increase_pct": 100 * (len(datasets["fit"]) / len(old_fit) - 1),
        "old_fit_staleness_days": (end - old_fit.target_times().max()) / np.timedelta64(1, "D"),
        "new_fit_staleness_days": (end - datasets["fit"].target_times().max()) / np.timedelta64(1, "D"),
        "old_selection_staleness_days": (end - old_select.target_times().max()) / np.timedelta64(1, "D"),
        "new_selection_staleness_days": (end - datasets["selection"].target_times().max()) / np.timedelta64(1, "D"),
        "train_only_quality": quality, "negative_power_rows": int((raw[:, -1] < 0).sum()),
        "negative_power_clipped": False, "input_values_changed": False, "target_values_changed": False,
        "pitch_extremes_note": "Flagged for source-sensor consistency investigation, not silently removed or repaired."})
    if len(datasets["fit"]) <= len(old_fit):
        raise ValueError("Proposed allocation did not expand the fitting queue")
    return result


def write_report(name, prediction, data, output, p, reference, fold, seed, state):
    result = prior.write_report(name, prediction, data, output, p, state)
    with np.load(reference / "runs" / f"f{fold}_s{seed}" / name / "raw_predictions.npz", allow_pickle=False) as old:
        truth, _ = data.truth_and_anchor()
        np.testing.assert_array_equal(truth, old["truth"])
        metadata = pd.DataFrame({"forecast_start": data.target_times()[:, 0]})
        result["paired_vs_v19_same_model"] = prior.paired_forecast_statistics(
            prediction, old["prediction"], truth, metadata, **p["statistics"])
        old_metrics = shared.model_metrics(old["prediction"], truth, old["persistence"], p["rated_power_kw"])
        result["vs_v19"] = {"mae_gain_kw": old_metrics["mae"] - result["metrics"]["mae"],
            "rmse_gain_kw": old_metrics["rmse"] - result["metrics"]["rmse"],
            "joint_improvement": bool(result["metrics"]["mae"] < old_metrics["mae"] and result["metrics"]["rmse"] < old_metrics["rmse"])}
    shared.write(output / "result.json", result)


def run(directory, gpu=0, audit_only=False):
    import fcntl
    p = shared.read(PROTOCOL)
    validate_protocol(p)
    directory.mkdir(parents=True, exist_ok=True)
    POINTER.parent.mkdir(parents=True, exist_ok=True)
    with (POINTER.parent / "hourly_recent.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = directory / "protocol.json"
        if saved.exists() and shared.sha(saved) != shared.sha(PROTOCOL):
            raise ValueError("Frozen protocol changed on resume")
        if not saved.exists():
            saved.write_bytes(PROTOCOL.read_bytes())
        source = {name: shared.sha(ROOT / name) for name in SOURCE_FILES}
        if (directory / "source_hashes.json").exists() and shared.read(directory / "source_hashes.json") != source:
            raise ValueError("Implementation changed across stages")
        shared.write(directory / "source_hashes.json", source)
        reference = ROOT / p["reference_dir"]
        if (shared.sha(reference / "completed.json") != p["reference_completed_sha256"] or
                shared.sha(reference / "protocol.json") != p["reference_protocol_sha256"]):
            raise ValueError("Frozen v19 reference identity changed")
        prior.verify_completed_suite(reference)
        if shared.sha(ROOT / p["data_path"]) != p["data_sha256"]:
            raise ValueError("Prepared source data fingerprint changed")
        POINTER.write_text(str(directory), encoding="utf-8")
        if (directory / "completed.json").exists():
            prior.verify_completed_suite(directory)
            print("[REUSE] Completed v20; no retraining", flush=True)
            return
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        (directory / "status.env").write_text("STATUS=RUNNING\n", encoding="utf-8")
        prepared = {}
        for fold in p["folds"]:
            series = shared.KelmarshSeries(ROOT / p["data_path"], fold=fold, input_len=24, horizon=4,
                stride=3, start=p["start_utc"], end=p["end_utc_exclusive"])
            data = recent_datasets(series)
            assert_reference_cohort(series, data, reference)
            audit = train_audit(series, data)
            shared.write(directory / f"data_audit_f{fold}.json", audit)
            prepared[fold] = data
            print(f"[AUDIT] f{fold} fit {audit['old_fit_windows']} -> {len(data['fit'])}, "
                  f"selection {len(data['selection'])}; v19 scaler and evaluation queue EXACT", flush=True)
        if audit_only:
            (directory / "status.env").write_text("STATUS=AUDIT_COMPLETED\nEXIT_CODE=0\n", encoding="utf-8")
            return
        for fold, data in prepared.items():
            fit, selection, validation = (data[k] for k in ("fit", "selection", "validation"))
            for seed in p["seeds"]:
                run_dir = directory / "runs" / f"f{fold}_s{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                _, anchor = validation.truth_and_anchor()
                for name in p["classical"] + p["models"]:
                    output = run_dir / name
                    if (output / "completed.json").exists():
                        prior.verify_hashes(output, shared.read(output / "completed.json"))
                        print(f"[REUSE] f{fold} {name}", flush=True)
                        continue
                    shared.write(directory / "progress.json", {"fold": fold, "seed": seed, "stage": name})
                    model, state = None, None
                    if name == "Persistence":
                        prediction = anchor
                    elif name == "DailyPersistence":
                        prediction = prior.daily_prediction(validation)
                    elif name == "Ridge":
                        x = np.stack([fit[i][0] for i in range(len(fit))]).reshape(len(fit), -1)
                        xx = np.stack([validation[i][0] for i in range(len(validation))]).reshape(len(validation), -1)
                        truth, _ = fit.truth_and_anchor()
                        model = Ridge(alpha=p["ridge_alpha"]).fit(x, truth)
                        prediction = model.predict(xx)
                        shared.write(output / "ridge.json", {"coefficients": model.coef_, "intercepts": model.intercept_,
                            "input_shape": [24, 7], "alpha": p["ridge_alpha"], "fit_scope": "outer-train first90%"})
                    else:
                        prior.wait_for_gpu(directory, fold, seed, name)
                        device, initial = torch.device(f"cuda:{gpu}"), None
                        if name == "PromptTimeDART":
                            pre, _ = shared.train_stage(name, fit, selection, p, run_dir / "pretrain", seed, device, pretrain=True)
                            del pre
                            gc.collect()
                            torch.cuda.empty_cache()
                            initial = run_dir / "pretrain/checkpoint.pt"
                            prior.wait_for_gpu(directory, fold, seed, name)
                        model, state = shared.train_stage(name, fit, selection, p, output, seed, device, initial=initial)
                        prediction = shared.predict(model, validation, seed, device)
                    write_report(name, prediction, validation, output, p, reference, fold, seed, state)
                    shared.write(output / "completed.json", {str(f.relative_to(output)): shared.sha(f)
                        for f in output.rglob("*") if f.is_file() and f.name != "completed.json"})
                    del model
                    gc.collect()
                    if torch.cuda.is_initialized():
                        torch.cuda.empty_cache()
                    prior.update_summary(directory)
                    print(f"[MODEL_COMPLETE] f{fold} {name}", flush=True)
        shared.write(directory / "progress.json", {"stage": "COMPLETED"})
        (directory / "status.env").write_text("STATUS=COMPLETED\nEXIT_CODE=0\n", encoding="utf-8")
        shared.write(directory / "completed.json", {str(f.relative_to(directory)): shared.sha(f)
            for f in directory.rglob("*") if f.is_file() and (f.name in
            ("completed.json", "protocol.json", "source_hashes.json", "metrics_all_runs.csv", "metrics_macro.csv",
             "metrics_by_hour_all_runs.csv", "source_commit.txt", "trained.json", "checkpoint.pt", "manifest.json",
             "training_history.json", "regime_calibration.json") or f.name.startswith("data_audit_f"))})
        print(f"[COMPLETED] {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
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
        run(args.output.resolve(), args.gpu, args.audit_only)
    except BlockingIOError:
        print("[BUSY] Existing experiment left unchanged", flush=True)
    except Exception as error:
        if args.output.exists():
            (args.output / "status.env").write_text(f"STATUS=FAILED\nERROR={type(error).__name__}: {error}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
