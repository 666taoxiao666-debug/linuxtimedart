#!/usr/bin/env python3
"""Frozen 24-hour history / four-hour hourly numerical foundation benchmark.

Reuse the real neural implementations, not the endpoint-only v15/v18 experts.
Wiki is deliberately off: four-lead knowledge needs new executor-bound evidence.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "2"
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from scripts import train_kelmarsh_hourly as shared
from utils.paired_forecast_statistics import paired_forecast_statistics

PROTOCOL = ROOT / "configs/kelmarsh_24h4h_protocol_v19.json"
PROTOCOL_SHA256 = "2330f209f6af0fdba123244c8d8a997471cff5d2c0ac538da866f0aa24fd3474"
POINTER = ROOT / "outputs/logs/Kelmarsh/hourly_24h4h_latest.txt"
SOURCE_FILES = ("scripts/train_kelmarsh_24h4h.py", "scripts/train_kelmarsh_hourly.py", "run.py") + tuple(
    sorted(str(path.relative_to(ROOT)).replace("\\", "/")
           for folder in ("data_provider", "models", "layers", "utils")
           for path in (ROOT / folder).rglob("*.py")))


def validate_protocol(p):
    # Git checks out CRLF on Windows and LF on the training host.
    digest = hashlib.sha256(PROTOCOL.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if digest != PROTOCOL_SHA256:
        raise ValueError("Predeclared v19 protocol bytes changed")
    expected = shared.read(PROTOCOL)
    if p != expected:
        raise ValueError("Only the exact predeclared v19 protocol is supported")
    if (p["input_len"], p["pred_len"], p["frequency"]) != (24, 4, "1h"):
        raise ValueError("Expected 24 hourly input points and four hourly targets")
    if p["wiki_enabled"] or p["old_checkpoint_reuse"] or p["old_wiki_certificate_reuse"]:
        raise ValueError("No legacy two-lead model or Wiki evidence may be reused")


def audit_dataset(series, datasets, p):
    result = dict(series.audit)
    result["input_shape"] = [24, len(shared.FEATURES)]
    result["target_shape"] = [4, len(shared.FEATURES)]
    result["stages"] = {}
    for name, data in datasets.items():
        starts = data.window_starts
        issue = series.dates[starts + 23]
        times = data.target_times()
        np.testing.assert_array_equal(times - issue[:, None], np.broadcast_to(
            np.arange(1, 5).astype("timedelta64[h]"), times.shape))
        if np.any(times[:, -1] >= series.bounds["sealed_test_start"]):
            raise ValueError("Sealed suffix reached")
        x, y, _, _ = data[0]
        if x.shape != (24, 7) or y.shape != (4, 7):
            raise ValueError("Actual dataset tensor does not implement 24 -> 4")
        result["stages"][name] = {"windows": len(data), "first_target": str(times.min()),
            "last_target": str(times.max()), "window_starts_sha256": hashlib.sha256(starts.tobytes()).hexdigest(),
            "target_times_sha256": hashlib.sha256(times.tobytes()).hexdigest()}
    result.update({"forecast_issue": "end of last observed hourly bin",
                   "future_covariates": False, "labels_modified": False,
                   "actual_history_hours": p["input_len"], "actual_forecast_hours": p["pred_len"]})
    return result


def daily_prediction(data):
    # At issue t, target t+h uses observed power t+h-24, h=1..4.
    rows = data.window_starts[:, None] + data.seq_len - 24 + np.arange(data.pred_len)
    if np.any(rows > (data.window_starts + data.seq_len - 1)[:, None]):
        raise ValueError("Daily persistence attempted to use future observations")
    return data.series.raw[rows, -1]


def wait_for_gpu(directory, fold, seed, stage):
    began, notified = time.monotonic(), False
    while True:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True)
        others = [int(v) for v in result.stdout.splitlines() if v.strip().isdigit() and int(v) != os.getpid()]
        if not others:
            shared.write(directory / "progress.json", {"stage": stage, "fold": fold, "seed": seed})
            return
        shared.write(directory / "progress.json", {"stage": "WAITING_FOR_GPU", "next_stage": stage,
                                                    "fold": fold, "seed": seed, "other_pids": others})
        if not notified:
            print(f"[WAIT] Other compute PIDs {others}; no parallel training or process changes", flush=True)
            notified = True
        if time.monotonic() - began > 86400:
            raise TimeoutError("24-hour GPU wait elapsed; resume same directory when safe")
        time.sleep(60)


def write_report(name, prediction, data, directory, p, state=None):
    result = shared.report(name, prediction, data, directory, p["rated_power_kw"])
    truth, anchor = data.truth_and_anchor()
    metadata = pd.DataFrame({"forecast_start": data.target_times()[:, 0]})
    result["history_hours"], result["forecast_hours"] = 24, 4
    result["best_epoch"] = None if state is None else state["best_epoch"]
    result["per_hour"] = [{"lead_hours": h + 1, **shared.model_metrics(
        prediction[:, h:h+1], truth[:, h:h+1], anchor[:, h:h+1], p["rated_power_kw"])} for h in range(4)]
    result["paired_vs_persistence"] = paired_forecast_statistics(
        prediction, anchor, truth, metadata, **p["statistics"])
    gaps = metadata.forecast_start.drop_duplicates().sort_values().diff().dropna().dt.total_seconds() / 3600
    result["issue_time_gap_hours"] = {"median": gaps.median(), "maximum": gaps.max(),
        "pct_above_nominal_stride": 100 * (gaps > p["eval_stride"]).mean()}
    result["wiki_enabled"] = False
    shared.write(directory / "result.json", result)
    return result


def update_summary(directory):
    shared.update_summary(directory)
    rows = []
    for path in sorted((directory / "runs").glob("f*_s*/*/result.json")):
        fold, seed = path.parts[-3].split("_")
        result = shared.read(path)
        for row in result["per_hour"]:
            rows.append({"fold": int(fold[1:]), "seed": int(seed[1:]), "model": result["model"], **row})
    if rows:
        pd.DataFrame(rows).to_csv(directory / "metrics_by_hour_all_runs.csv", index=False)


def verify_hashes(directory, hashes):
    for name, expected in hashes.items():
        if shared.sha(directory / name) != expected:
            raise ValueError("Registered completed artifact changed: " + name)


def verify_completed_suite(directory):
    verify_hashes(directory, shared.read(directory / "completed.json"))
    for marker in (directory / "runs").glob("f*_s*/*/completed.json"):
        verify_hashes(marker.parent, shared.read(marker))


def run(directory, protocol_path=PROTOCOL, gpu=0, audit_only=False):
    import fcntl
    p = shared.read(protocol_path)
    validate_protocol(p)
    directory.mkdir(parents=True, exist_ok=True)
    # One lock across directories prevents duplicate launches of this task.
    lock_path = ROOT / "outputs/logs/Kelmarsh/hourly_24h4h.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = directory / "protocol.json"
        if saved.exists() and shared.sha(saved) != shared.sha(protocol_path):
            raise ValueError("Frozen protocol changed on resume")
        if not saved.exists():
            saved.write_bytes(Path(protocol_path).read_bytes())
        identity = {name: shared.sha(ROOT / name) for name in SOURCE_FILES}
        if (directory / "source_hashes.json").exists():
            if shared.read(directory / "source_hashes.json") != identity:
                raise ValueError("Implementation changed; do not silently mix completed stages")
        else:
            shared.write(directory / "source_hashes.json", identity)
        if shared.sha(ROOT / p["data_path"]) != p["data_sha256"]:
            raise ValueError("Prepared source data fingerprint changed")
        POINTER.write_text(str(directory), encoding="utf-8")
        if (directory / "completed.json").exists():
            verify_completed_suite(directory)
            print("[REUSE] Entire 24 -> 4 benchmark already completed", flush=True)
            return
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        (directory / "status.env").write_text("STATUS=RUNNING\n", encoding="utf-8")
        for fold in p["folds"]:
            series = shared.KelmarshSeries(ROOT / p["data_path"], fold=fold, input_len=24, horizon=4,
                stride=p["train_stride"], start=p["start_utc"], end=p["end_utc_exclusive"])
            data = {name: series.subset(name) for name in ("fit", "selection", "validation")}
            shared.write(directory / f"data_audit_f{fold}.json", audit_dataset(series, data, p))
            if audit_only:
                continue
            fit, selection, validation = (data[k] for k in ("fit", "selection", "validation"))
            for seed in p["seeds"]:
                run_dir = directory / "runs" / f"f{fold}_s{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                _, anchor = validation.truth_and_anchor()
                for name in p["classical"] + p["models"]:
                    output = run_dir / name
                    if (output / "completed.json").exists():
                        verify_hashes(output, shared.read(output / "completed.json"))
                        print(f"[REUSE] f{fold} s{seed} {name}", flush=True)
                        continue
                    shared.write(directory / "progress.json", {"fold": fold, "seed": seed, "stage": name})
                    state, model = None, None
                    if name == "Persistence":
                        prediction = anchor
                    elif name == "DailyPersistence":
                        prediction = daily_prediction(validation)
                    elif name == "Ridge":
                        x = np.stack([fit[i][0] for i in range(len(fit))]).reshape(len(fit), 24 * 7)
                        xx = np.stack([validation[i][0] for i in range(len(validation))]).reshape(len(validation), 24 * 7)
                        truth, _ = fit.truth_and_anchor()
                        model = Ridge(alpha=p["ridge_alpha"]).fit(x, truth)
                        prediction = model.predict(xx)
                        shared.write(output / "ridge.json", {"coefficients": model.coef_, "intercepts": model.intercept_,
                            "input_shape": [24, 7], "output_hours": 4, "alpha": p["ridge_alpha"], "fit_scope": "base_fit only"})
                    else:
                        wait_for_gpu(directory, fold, seed, name)
                        device = torch.device(f"cuda:{gpu}")
                        initial = None
                        if name == "PromptTimeDART":
                            pre, _ = shared.train_stage(name, fit, selection, p, run_dir / "pretrain", seed, device, pretrain=True)
                            del pre
                            gc.collect()
                            torch.cuda.empty_cache()
                            initial = run_dir / "pretrain/checkpoint.pt"
                            wait_for_gpu(directory, fold, seed, name)
                        model, state = shared.train_stage(name, fit, selection, p, output, seed, device, initial=initial)
                        prediction = shared.predict(model, validation, seed, device)
                    write_report(name, prediction, validation, output, p, state)
                    hashes = {str(path.relative_to(output)): shared.sha(path) for path in output.rglob("*")
                              if path.is_file() and path.name != "completed.json"}
                    shared.write(output / "completed.json", hashes)
                    del model
                    gc.collect()
                    if torch.cuda.is_initialized():
                        torch.cuda.empty_cache()
                    update_summary(directory)
                    print(f"[MODEL_COMPLETE] f{fold} s{seed} {name}", flush=True)
        status = "AUDIT_COMPLETED" if audit_only else "COMPLETED"
        shared.write(directory / "progress.json", {"stage": status})
        (directory / "status.env").write_text(f"STATUS={status}\nEXIT_CODE=0\n", encoding="utf-8")
        if not audit_only:
            shared.write(directory / "completed.json", {str(path.relative_to(directory)): shared.sha(path)
                for path in directory.rglob("*") if path.is_file() and (path.name in
                ("completed.json", "protocol.json", "source_hashes.json", "metrics_all_runs.csv",
                 "metrics_macro.csv", "metrics_by_hour_all_runs.csv", "source_commit.txt",
                 "trained.json", "checkpoint.pt", "manifest.json", "training_history.json",
                 "regime_calibration.json") or path.name.startswith("data_audit_f"))})
        print(f"[{status}] {directory}", flush=True)


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
        raise ValueError("Explicit independent result directory required")
    try:
        run(args.output.resolve(), gpu=args.gpu, audit_only=args.audit_only)
    except BlockingIOError:
        print("[BUSY] Existing 24 -> 4 task left unchanged", flush=True)
    except Exception as error:
        if args.output.exists():
            (args.output / "status.env").write_text(f"STATUS=FAILED\nERROR={type(error).__name__}: {error}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
