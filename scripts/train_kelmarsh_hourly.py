#!/usr/bin/env python3
"""Sequential, resumable hourly multi-model and expert-bound Wiki experiment."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
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
from torch.utils.data import DataLoader

from data_provider.kelmarsh_hourly import FEATURES, KelmarshSeries
from models import DLinear, PatchTST, PromptTimeDART
from run import build_parser
from utils import kelmarsh_hourly_wiki as wiki
from utils.forecast_report import build_forecast_report
from utils.metrics import forecast_metrics
from utils.paired_forecast_statistics import paired_forecast_statistics
from utils.regime_labels import calibrate_regime_thresholds_from_dataset


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for part in iter(lambda: file.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def safe(value):
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(safe(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(dataset, seed, shuffle=False, batch=64):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=0,
                      drop_last=False, generator=generator, pin_memory=torch.cuda.is_available())


def model_args(name, dataset, protocol, device, task="finetune"):
    args = build_parser().parse_args(["--task_name", task, "--downstream_task", "forecast", "--is_training", "1",
        "--model_id", "Kelmarsh_hourly", "--model", name, "--data", "KelmarshHourly"])
    args.device, args.features, args.target, args.freq = device, "MS", "power", "h"
    args.root_path, args.data_path = ".", protocol["data_path"]
    args.feature_columns = FEATURES
    args.input_len = args.seq_len = protocol["input_len"]
    args.pred_len = protocol["pred_len"]
    args.enc_in = args.dec_in = len(FEATURES)
    args.c_out = 1 if name == "PromptTimeDART" and task == "finetune" else len(FEATURES)
    cfg = protocol["training"]
    for key in ("d_model", "d_ff", "n_heads", "e_layers", "dropout"):
        setattr(args, key, cfg[key])
    args.patch_len, args.stride = cfg["patch_len"], cfg["patch_stride"]
    args.use_norm = 1
    args.mix_channels = task == "finetune" and name == "PromptTimeDART"
    args.channel_prior = args.op_context = args.mix_channels
    args.revin_keep_wind = name == "PromptTimeDART"
    args.consistent_physics_norm = name == "PromptTimeDART"
    args.residual_forecast = name == "PromptTimeDART"
    args.residual_gate_init = -2.2
    args.zero_init_residual_head = name == "PromptTimeDART"
    args.ramp_residual = args.utility_wiki = False
    args.prompt_router = "trend"
    args.rated_power = protocol["rated_power_kw"]
    args.regime_target_index, args.regime_label_method = -1, "trend_quantile"
    train_cfg = protocol["pretraining"] if task == "pretrain" else cfg
    args.train_epochs, args.patience, args.batch_size = train_cfg["epochs"], train_cfg["patience"], cfg["batch_size"]
    args.loss = "MSE" if task == "pretrain" else "MIXED"
    args.mix_mse_weight = cfg["mix_mse_weight"]
    args.lambda_ce = protocol["pretraining"]["lambda_ce"]
    args.learning_rate = (train_cfg["learning_rate"] if task == "pretrain" else
                          cfg["backbone_lr"] if name == "PromptTimeDART" else cfg[name + "_lr"])
    args.new_module_learning_rate = cfg["new_module_lr"] if name == "PromptTimeDART" and task == "finetune" else 0.
    return args


def gpu_available_to_this_process():
    if not torch.cuda.is_available():
        raise RuntimeError("This full experiment requires CUDA; no silent CPU/model substitution")
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    other = [int(line) for line in result.stdout.splitlines() if line.strip().isdigit() and int(line) != os.getpid()]
    if other:
        raise RuntimeError(f"GPU occupied by other processes {other}; defer, do not run in parallel")


def extract(output):
    if isinstance(output, dict):
        return output["pred"]
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def predict(model, dataset, seed, device):
    model.eval()
    chunks = []
    for x, _, _, _ in loader(dataset, seed, batch=128):
        chunks.append(extract(model(x.float().to(device)))[..., -1].cpu().numpy())
    scaled = np.concatenate(chunks).astype(float)
    return scaled * dataset.scaler.scale_[-1] + dataset.scaler.mean_[-1]


@torch.no_grad()
def reconstruction(model, dataset, seed, device):
    model.eval()
    total, count = 0., 0
    # Validation noise is reproducible and cannot perturb the training RNG.
    with torch.random.fork_rng(devices=[device.index or 0]):
        torch.manual_seed(seed + 9000)
        for x, _, _, _ in loader(dataset, seed, batch=128):
            x = x.float().to(device)
            output = extract(model(x))
            total += torch.square(output - x).sum().item()
            count += x.numel()
    return total / count


def verify_stage(directory, contract):
    state = read(directory / "trained.json")
    if state["contract"] != contract:
        raise ValueError("Completed stage configuration/data identity changed")
    for name, expected in state["artifacts"].items():
        if sha(directory / name) != expected:
            raise ValueError("Completed stage artifact changed: " + name)
    return state


def train_stage(name, fit, selection, protocol, directory, seed, device, *, pretrain=False, initial=None):
    directory.mkdir(parents=True, exist_ok=True)
    gpu_available_to_this_process()
    contract = {"protocol_sha256": sha(directory.parents[2] / "protocol.json"), "data_sha256": protocol["data_sha256"],
                "name": name, "seed": seed, "pretrain": pretrain, "scaler_mean": fit.scaler.mean_.tolist(),
                "scaler_scale": fit.scaler.scale_.tolist(), "fit_window_hash": hashlib.sha256(fit.window_starts.tobytes()).hexdigest(),
                "selection_window_hash": hashlib.sha256(selection.window_starts.tobytes()).hexdigest(),
                "initial_checkpoint_sha256": sha(initial) if initial else None}
    seed_all(seed)
    args = model_args(name, fit, protocol, device, "pretrain" if pretrain else "finetune")
    model = {"DLinear": DLinear, "PatchTST": PatchTST, "PromptTimeDART": PromptTimeDART}[name].Model(args).to(device)
    if name == "PromptTimeDART":
        audit = calibrate_regime_thresholds_from_dataset(fit)
        model.regime_down_thresh.fill_(audit["down_thresh"])
        model.regime_up_thresh.fill_(audit["up_thresh"])
        write(directory / "regime_calibration.json", audit)
    loaded = []
    if initial:
        state = torch.load(initial, map_location="cpu", weights_only=False)["model"]
        destination = model.state_dict()
        loaded = [key for key in state if key in destination and state[key].shape == destination[key].shape]
        destination.update({key: state[key] for key in loaded})
        model.load_state_dict(destination, strict=True)
    if (directory / "trained.json").exists():
        state = verify_stage(directory, contract)
        model.load_state_dict(torch.load(directory / "checkpoint.pt", map_location=device, weights_only=False)["model"], strict=True)
        print(f"[REUSE] {name} pretrain={pretrain} completed epoch={state['best_epoch']}", flush=True)
        return model, state
    cfg = protocol["pretraining"] if pretrain else protocol["training"]
    epochs, patience = cfg["epochs"], cfg["patience"]
    if pretrain:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=protocol["training"]["weight_decay"])
    elif name == "PromptTimeDART":
        new, base = [], []
        for key, parameter in model.named_parameters():
            (new if key.startswith(("head.", "channel_mixer.", "residual_gate")) else base).append(parameter)
        optimizer = torch.optim.AdamW([{"params": base, "lr": cfg["backbone_lr"]},
                                      {"params": new, "lr": cfg["new_module_lr"]}], weight_decay=cfg["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg[name + "_lr"], weight_decay=cfg["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    truth, _ = selection.truth_and_anchor()
    def assess():
        if pretrain:
            value = reconstruction(model, selection, seed, device)
            return value, {"reconstruction_mse": value}
        metrics = forecast_metrics(predict(model, selection, seed, device), truth, rated_power=protocol["rated_power_kw"])
        return metrics["mae"], metrics
    history, start_epoch, bad_epochs = [], 0, 0
    last_path = directory / "last.pt"
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint["contract"] != contract:
            raise ValueError("Interrupted stage identity changed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        schedule.load_state_dict(checkpoint["scheduler"])
        history, start_epoch = checkpoint["history"], checkpoint["epoch"]
        best, best_epoch, bad_epochs = checkpoint["best"], checkpoint["best_epoch"], checkpoint["bad_epochs"]
        torch.set_rng_state(checkpoint["rng_cpu"])
        torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
        print(f"[RESUME] {name} after epoch {start_epoch}", flush=True)
    else:
        best, initial_metrics = assess()
        best_epoch = 0
        history.append({"epoch": 0, "selection": initial_metrics})
        torch.save({"model": model.state_dict(), "contract": contract}, directory / "checkpoint.pt")
    for epoch in range(start_epoch + 1, epochs + 1):
        if bad_epochs >= patience:
            break
        model.train()
        total, batches, begin = 0., 0, time.monotonic()
        for x, y, _, _ in loader(fit, seed + epoch, shuffle=True, batch=protocol["training"]["batch_size"]):
            x, y = x.float().to(device), y.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            if pretrain:
                loss = torch.square(extract(output) - x).mean()
                logits, labels = (output[1], output[2]) if isinstance(output, tuple) else (
                    output.get("regime_logits"), output.get("pseudo_labels"))
                if logits is not None:
                    loss = loss + cfg["lambda_ce"] * torch.nn.functional.cross_entropy(logits, labels)
            else:
                difference = extract(output)[..., -1] - y[..., -1]
                weight = cfg["mix_mse_weight"]
                loss = (1 - weight) * difference.abs().mean() + weight * difference.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; never continue silently")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["training"]["grad_clip"])
            optimizer.step()
            total += loss.item()
            batches += 1
        score, metrics = assess()
        history.append({"epoch": epoch, "train_loss": total / batches, "selection": metrics,
                        "elapsed_seconds": time.monotonic() - begin})
        if score < best:
            best, best_epoch, bad_epochs = score, epoch, 0
            torch.save({"model": model.state_dict(), "contract": contract}, directory / "checkpoint.pt")
        else:
            bad_epochs += 1
        schedule.step()
        # Epoch-boundary resume includes optimizer, schedule and stochastic state.
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": schedule.state_dict(),
                    "contract": contract, "epoch": epoch, "best": best, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
                    "history": history, "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()}, last_path)
        write(directory / "training_history.json", history)
        print(f"[EPOCH] {name} pretrain={pretrain} epoch={epoch} train={total/batches:.6f} selection={score:.6f} best_epoch={best_epoch} seconds={time.monotonic()-begin:.1f}", flush=True)
    model.load_state_dict(torch.load(directory / "checkpoint.pt", map_location=device, weights_only=False)["model"])
    write(directory / "training_history.json", history)
    active_names = ["task_name", "downstream_task", "data", "root_path", "data_path", "device", "features", "target", "freq",
        "input_len", "seq_len", "pred_len", "enc_in", "dec_in", "c_out", "d_model", "d_ff", "n_heads", "e_layers", "d_layers",
        "dropout", "patch_len", "stride", "use_norm", "mix_channels", "channel_prior", "op_context", "revin_keep_wind",
        "consistent_physics_norm", "residual_forecast", "residual_gate_init", "zero_init_residual_head", "num_modes",
        "prompt_router", "regime_label_method", "regime_target_index", "rated_power", "train_epochs", "patience", "batch_size",
        "loss", "mix_mse_weight", "lambda_ce", "learning_rate", "new_module_learning_rate"]
    active_args = {key: getattr(args, key) for key in active_names}
    active_args["device"] = str(device)
    write(directory / "manifest.json", {"contract": contract, "feature_columns": FEATURES, "model_args": safe(active_args),
          "parameters": sum(p.numel() for p in model.parameters()), "transferred_state_keys": loaded,
          "selection_scope": "internal outer-train only; never outer validation", "best_epoch": best_epoch,
          "scaler_fit_scope": "base_fit prefix only", "targets_clipped": False})
    result = {"contract": contract, "best_epoch": best_epoch, "epochs_completed": len(history) - 1,
              "artifacts": {key: sha(directory / key) for key in ("checkpoint.pt", "training_history.json", "manifest.json")}}
    write(directory / "trained.json", result)
    return model, result


def model_metrics(prediction, truth, anchor, capacity):
    result = forecast_metrics(prediction, truth, rated_power=capacity)
    baseline = forecast_metrics(anchor, truth)
    result["mae_skill_pct"] = 100 * (1 - result["mae"] / baseline["mae"]) if baseline["mae"] else None
    result["rmse_skill_pct"] = 100 * (1 - result["rmse"] / baseline["rmse"]) if baseline["rmse"] else None
    return safe(result)


def report(name, prediction, dataset, directory, capacity):
    directory.mkdir(parents=True, exist_ok=True)
    # sklearn may emit float32. Normalize in float64 before the report's
    # inverse transform, otherwise the roundtrip adds an extra quantization.
    prediction = np.asarray(prediction, dtype=np.float64)
    truth, anchor = dataset.truth_and_anchor()
    metrics = model_metrics(prediction, truth, anchor, capacity)
    mean, scale = dataset.scaler.mean_[-1], dataset.scaler.scale_[-1]
    reported = build_forecast_report((prediction - mean) / scale, (truth - mean) / scale, dataset, str(directory / "artifacts"),
        last_observation_scaled=(anchor[:, 0] - mean) / scale, step_minutes=60,
        rated_power=capacity, model_name=name, clip_to_capacity=False)
    for key in ("mae", "mse", "rmse", "r2"):
        np.testing.assert_allclose(reported[key], metrics[key], rtol=1e-10, atol=1e-9)
    np.savez_compressed(directory / "raw_predictions.npz", prediction=prediction, truth=truth, persistence=anchor,
                        times=dataset.target_times(), turbines=dataset.turbines[dataset.window_starts], window_starts=dataset.window_starts)
    first = model_metrics(prediction[:, :2], truth[:, :2], anchor[:, :2], capacity)
    result = {"model": name, "metrics": metrics, "first_two_hours": first, "evaluation_split": "outer_validation",
              "windows": len(dataset), "power_units": "kW", "step_minutes": 60, "target_clipping": False, "sealed_test_evaluated": False}
    write(directory / "result.json", result)
    print(f"[RESULT] {name} MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f} R2={metrics['r2']:.6f} MAE_SKILL={metrics['mae_skill_pct']:.3f}%", flush=True)
    return result


def block(dataset, reference, rules, donors=False):
    truth, anchor = dataset.truth_and_anchor()
    context = dataset.physical_context()
    result = {"reference": reference, "truth": truth, "anchor": anchor, "context": context,
              "events": wiki.physical_events(context, rules), "times": dataset.target_times(),
              "turbines": dataset.turbines[dataset.window_starts]}
    if donors:
        mask = result["turbines"] != 6
        result = {key: ({k: v[mask] for k, v in value.items()} if isinstance(value, dict) else value[mask]) for key, value in result.items()}
    return result


def wiki_diagnostics(applied, block, contract):
    active = applied["action"] > 0
    windows = active.any(1)
    gain = np.abs(block["reference"] - block["truth"]) - np.abs(applied["prediction"] - block["truth"])
    result = {"coverage_window_pct": 100 * windows.mean(), "coverage_point_pct": 100 * active.mean(),
              "abstention_window_pct": 100 * (~windows).mean(),
              "physical_candidate_window_pct": 100 * applied["physical_available"].any((1, 2)).mean(),
              "supported_candidate_window_pct": 100 * applied["history_overlap"].any((1, 2)).mean(),
              "selected_point_gain_kw": gain[active].mean() if active.any() else None,
              "harm_point_pct": 100 * (gain[active] < 0).mean() if active.any() else None,
              "harm_window_pct": 100 * (gain.mean(1)[windows] < 0).mean() if windows.any() else None,
              "overall_gain_kw": gain.mean(), "expert_id": applied["expert_id"], "policy_id": applied["policy_id"]}
    result["by_event"] = []
    for k, name in enumerate(wiki.EVENTS + ["composition"]):
        selected = applied["action"] == k + 1
        result["by_event"].append({"event": name, "selected_points": int(selected.sum()),
                                  "selected_gain_kw": gain[selected].mean() if selected.any() else None,
                                  "harm_point_pct": 100 * (gain[selected] < 0).mean() if selected.any() else None})
    result["by_granularity"] = [{"level": level, "selected_points": int((active & (np.take_along_axis(applied["granularity"],
                                np.maximum(applied["action"] - 1, 0)[..., None], -1).squeeze(-1) == code)).sum())}
                                for code, level in enumerate(("event", "power", "joint"))]
    for name, mask in (("knowledge_seen_turbines", block["turbines"] != 6), ("knowledge_held_out_turbine6", block["turbines"] == 6)):
        result[name] = {"windows": int(mask.sum()), "gain_kw": gain[mask].mean() if mask.any() else None,
                        "coverage_pct": 100 * windows[mask].mean() if mask.any() else None,
                        "backbone_seen_this_turbine": True}
    return safe(result)


def numeric_residual_control(blocks, evaluation, contract, protocol):
    """Additional-label control, with no events or retrieval/knowledge hierarchy."""
    capacity = contract["rated_power_kw"]
    settings = protocol["utility"]
    def coordinates(block):
        fixed = wiki.keys(block["context"], block["anchor"], capacity, True)
        repeated = np.repeat(fixed[:, None, :], contract["pred_len"], axis=1)
        return np.concatenate((repeated, ((block["reference"] - block["anchor"]) / capacity)[..., None]), axis=-1)
    fit = blocks["memory"]
    wiki.history._source_ids(fit["turbines"], contract)
    x = coordinates(fit)
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-6, scale, 1.)
    target = (fit["truth"] - fit["reference"]) / capacity
    ids, counts = np.unique(fit["turbines"], return_counts=True)
    weights = np.array([1. / counts[np.searchsorted(ids, unit)] for unit in fit["turbines"]])
    regressors = [Ridge(alpha=protocol["numeric_residual_ridge_alpha"]).fit(
        (x[:, h] - mean[h]) / scale[h], target[:, h], sample_weight=weights)
        for h in range(contract["pred_len"])]
    def candidates(block):
        xx = (coordinates(block) - mean) / scale
        correction = np.column_stack([regressor.predict(xx[:, h]) for h, regressor in enumerate(regressors)])
        correction = np.clip(correction, -protocol["wiki"]["max_scale"], protocol["wiki"]["max_scale"]) * capacity
        return block["reference"] + correction
    utility = []
    for stage in ("calibration_early", "calibration_late"):
        block = blocks[stage]
        wiki.history._source_ids(block["turbines"], contract)
        prediction = candidates(block)
        gain = np.abs(block["reference"] - block["truth"]) - np.abs(prediction - block["truth"])
        mse = (block["reference"] - block["truth"]) ** 2 - (prediction - block["truth"]) ** 2
        unit_ids = np.unique(block["turbines"])
        unit_gain = np.stack([gain[block["turbines"] == unit].mean(0) for unit in unit_ids])
        unit_mse = np.stack([mse[block["turbines"] == unit].mean(0) for unit in unit_ids])
        utility.append({"stage": stage, "windows": len(gain), "turbines": len(unit_ids),
                        "gain_kw": unit_gain.mean(0), "mse_gain": unit_mse.mean(0),
                        "positive_fraction": (unit_gain > 0).mean(0)})
    accepted = np.ones(contract["pred_len"], bool)
    for evidence in utility:
        accepted &= (evidence["windows"] >= settings["min_windows"] and evidence["turbines"] >= settings["min_turbines"])
        accepted &= (evidence["gain_kw"] > 0) & (evidence["mse_gain"] > 0) & (evidence["positive_fraction"] >= settings["positive_fraction"])
    prediction = np.where(accepted[None], candidates(evaluation), evaluation["reference"])
    artifact = {"contract": contract, "alpha": protocol["numeric_residual_ridge_alpha"],
                "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
                "coefficients": [regressor.coef_.tolist() for regressor in regressors],
                "intercepts": [float(regressor.intercept_) for regressor in regressors],
                "accepted_horizons": accepted.tolist(), "utility_blocks": safe(utility),
                "events_used": False, "retrieval_used": False, "outer_labels_used_for_fit": False}
    return prediction, artifact


def update_summary(directory):
    records = []
    for path in sorted((directory / "runs").glob("f*_s*/*/result.json")):
        result = read(path)
        fold, seed = path.parts[-3].split("_")
        records.append({"fold": int(fold[1:]), "seed": int(seed[1:]), "model": result["model"],
                        "best_epoch": result.get("best_epoch"), **result["metrics"],
                        **{key: result.get("wiki", {}).get(key) for key in (
                            "coverage_window_pct", "abstention_window_pct", "harm_window_pct", "selected_point_gain_kw", "overall_gain_kw")},
                        "result_path": str(path)})
    if records:
        frame = pd.DataFrame(records)
        frame.to_csv(directory / "metrics_all_runs.csv", index=False)
        keys = ["mae", "mse", "rmse", "r2", "mae_skill_pct", "rmse_skill_pct", "within_5pct_capacity_pct", "within_10pct_capacity_pct",
                "coverage_window_pct", "abstention_window_pct", "harm_window_pct", "selected_point_gain_kw", "overall_gain_kw"]
        macro = frame.groupby("model", sort=False)[keys].agg(["mean", "std", "count"])
        macro.columns = ["_".join(column) for column in macro.columns]
        macro.to_csv(directory / "metrics_macro.csv")


def run(directory, protocol_path, device):
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = read(protocol_path)
        if (protocol["protocol_id"] != "kelmarsh_hourly_physical_transition_wiki_v1" or
                protocol["folds"] != [0, 1, 2] or protocol["seeds"] != [2024, 2025, 2026]):
            raise ValueError("Undeclared experiment matrix")
        saved = directory / "protocol.json"
        if saved.exists() and sha(saved) != sha(protocol_path):
            raise ValueError("Frozen protocol changed on resume")
        if not saved.exists():
            saved.write_bytes(Path(protocol_path).read_bytes())
        if sha(ROOT / protocol["data_path"]) != protocol["data_sha256"]:
            raise ValueError("Prepared source data fingerprint changed")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        (directory / "status.env").write_text("STATUS=RUNNING\n", encoding="utf-8")
        for fold in protocol["folds"]:
            series = KelmarshSeries(ROOT / protocol["data_path"], fold=fold, input_len=protocol["input_len"],
                horizon=protocol["pred_len"], stride=protocol["train_stride"], start=protocol["start_utc"], end=protocol["end_utc_exclusive"])
            datasets = {stage: series.subset(stage) for stage in ("fit", "selection", "memory", "calibration_early", "calibration_late", "validation")}
            write(directory / f"data_audit_f{fold}.json", series.audit | {"stage_windows": {k: len(v) for k, v in datasets.items()}})
            fit, selection, validation = datasets["fit"], datasets["selection"], datasets["validation"]
            for seed in protocol["seeds"]:
                run_dir = directory / "runs" / f"f{fold}_s{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                if (run_dir / "completed.json").exists():
                    for name, digest in read(run_dir / "completed.json").items():
                        if sha(run_dir / name) != digest:
                            raise ValueError("Completed run changed")
                    print(f"[REUSE] Entire run fold={fold} seed={seed}", flush=True)
                    continue
                write(directory / "progress.json", {"fold": fold, "seed": seed, "stage": "classical"})
                truth, anchor = validation.truth_and_anchor()
                persistence = anchor
                starts = validation.window_starts
                day_rows = starts[:, None] + protocol["input_len"] - 24 + np.arange(protocol["pred_len"])
                daily = series.raw[day_rows, -1]
                fit_x = np.stack([fit[i][0] for i in range(len(fit))]).reshape(len(fit), -1)
                val_x = np.stack([validation[i][0] for i in range(len(validation))]).reshape(len(validation), -1)
                fit_y, _ = fit.truth_and_anchor()
                ridge = Ridge(alpha=1.).fit(fit_x, fit_y).predict(val_x)
                for name, predictions in (("Persistence", persistence), ("DailyPersistence", daily), ("Ridge", ridge)):
                    if not (run_dir / name / "result.json").exists():
                        report(name, predictions, validation, run_dir / name, protocol["rated_power_kw"])
                for name in protocol["models"]:
                    write(directory / "progress.json", {"fold": fold, "seed": seed, "stage": name})
                    initial = None
                    if name == "PromptTimeDART":
                        pre, _ = train_stage(name, fit, selection, protocol, run_dir / "pretrain", seed, device, pretrain=True)
                        del pre
                        gc.collect()
                        torch.cuda.empty_cache()
                        initial = run_dir / "pretrain" / "checkpoint.pt"
                    model, state = train_stage(name, fit, selection, protocol, run_dir / name, seed, device, initial=initial)
                    prediction = predict(model, validation, seed, device)
                    result = report(name, prediction, validation, run_dir / name, protocol["rated_power_kw"])
                    result["best_epoch"] = state["best_epoch"]
                    write(run_dir / name / "result.json", result)
                    contract = {"data_sha256": protocol["data_sha256"], "base_checkpoint_sha256": sha(run_dir / name / "checkpoint.pt"),
                            "base_manifest_sha256": sha(run_dir / name / "manifest.json"), "protocol_sha256": sha(saved),
                            "scaler_sha256": wiki.history.digest({"mean": series.scaler.mean_.tolist(), "scale": series.scaler.scale_.tolist()}),
                            "rules_sha256": wiki.history.digest(protocol["rules"]), "feature_columns": FEATURES,
                            "pred_len": protocol["pred_len"], "input_len": protocol["input_len"], "step_minutes": 60,
                            "physical_coordinate_version": "kelmarsh_hourly_measured_heading_pitch_v1",
                            "rated_power_kw": protocol["rated_power_kw"], "event_ids": wiki.EVENTS,
                            "source_turbines": protocol["knowledge_source_turbines"], "held_out_turbines": [6]}
                    blocks = {stage: block(datasets[stage], predict(model, datasets[stage], seed, device), protocol["rules"], donors=True)
                                  for stage in ("memory", "calibration_early", "calibration_late")}
                    evaluation = block(validation, prediction, protocol["rules"])
                    numeric, numeric_artifact = numeric_residual_control(blocks, evaluation, contract, protocol)
                    numeric_name = name + "_numeric_control"
                    numeric_result = report(numeric_name, numeric, validation, run_dir / numeric_name, protocol["rated_power_kw"])
                    numeric_result["best_epoch"] = state["best_epoch"]
                    numeric_result["accepted_horizons"] = numeric_artifact["accepted_horizons"]
                    write(run_dir / numeric_name / "result.json", numeric_result)
                    write(run_dir / numeric_name / "calibration.json", numeric_artifact)
                    if name == "PromptTimeDART":
                        write(directory / "progress.json", {"fold": fold, "seed": seed, "stage": "expert_bound_wiki"})
                        for no_controls in (False, True):
                            settings = protocol["wiki"] | {"use_controls": not no_controls}
                            expert = wiki.fit_experts(blocks["memory"], contract, settings)
                            policy = wiki.fit_policy(expert, [blocks["calibration_early"], blocks["calibration_late"]], contract, protocol["utility"])
                            variants = [("Wiki_no_controls", {})] if no_controls else [
                                ("Wiki_full", {}), ("Wiki_event_only", {"event_only": True}),
                                ("Wiki_no_composition", {"allow_composition": False})]
                            for variant, options in variants:
                                output = run_dir / variant
                                output.mkdir(exist_ok=True)
                                write(output / "expert.json", expert)
                                write(output / "utility_policy.json", policy)
                                applied = wiki.apply(expert, policy, evaluation, contract, **options)
                                result = report(variant, applied["prediction"], validation, output, protocol["rated_power_kw"])
                                result["best_epoch"] = state["best_epoch"]
                                result["wiki"] = wiki_diagnostics(applied, evaluation, contract)
                                metadata = pd.DataFrame({"forecast_start": evaluation["times"][:, 0]})
                                result["paired_vs_trend"] = paired_forecast_statistics(applied["prediction"], prediction, truth,
                                                                                     metadata, **protocol["statistics"])
                                gaps = metadata.forecast_start.drop_duplicates().sort_values().diff().dropna().dt.total_seconds() / 3600
                                result["paired_vs_trend"]["issue_time_gap_hours"] = {"median": gaps.median(),
                                    "maximum": gaps.max(), "pct_larger_than_nominal_stride": 100 * (gaps > protocol["eval_stride"]).mean()}
                                write(output / "result.json", result)
                                np.savez_compressed(output / "interventions.npz", action=applied["action"], granularity=applied["granularity"],
                                                     events=evaluation["events"], times=evaluation["times"], turbines=evaluation["turbines"])
                        if sha(run_dir / name / "checkpoint.pt") != contract["base_checkpoint_sha256"]:
                            raise ValueError("Base changed after knowledge evidence generation")
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    update_summary(directory)
                retained = {"result.json", "checkpoint.pt", "trained.json", "expert.json", "utility_policy.json",
                            "raw_predictions.npz", "interventions.npz", "manifest.json", "training_history.json", "calibration.json"}
                hashes = {str(path.relative_to(run_dir)): sha(path) for path in run_dir.rglob("*") if path.name in retained}
                write(run_dir / "completed.json", hashes)
                update_summary(directory)
                print(f"[RUN_COMPLETE] fold={fold} seed={seed}", flush=True)
        write(directory / "progress.json", {"stage": "COMPLETED", "runs": 9})
        (directory / "status.env").write_text("STATUS=COMPLETED\nEXIT_CODE=0\n", encoding="utf-8")
        print(f"[COMPLETE] Hourly suite: {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/kelmarsh_hourly_protocol_v1.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    pointer = ROOT / "outputs/logs/Kelmarsh/hourly_system_latest.txt"
    if args.status:
        if not pointer.exists():
            print("STATUS=NOT_STARTED")
            return
        directory = Path(pointer.read_text().strip())
        print(f"RESULT_DIR={directory}")
        for name in ("status.env", "progress.json", "metrics_macro.csv"):
            if (directory / name).exists():
                print((directory / name).read_text())
        return
    if args.audit_only:
        p = read(args.protocol)
        for fold in p["folds"]:
            series = KelmarshSeries(ROOT / p["data_path"], fold=fold, input_len=p["input_len"], horizon=p["pred_len"],
                                    stride=p["train_stride"], start=p["start_utc"], end=p["end_utc_exclusive"])
            print(json.dumps(series.audit | {"stage_windows": {name: len(series.subset(name)) for name in
                            ("fit", "selection", "memory", "calibration_early", "calibration_late", "validation")}}, indent=2))
        return
    if args.output is None:
        raise ValueError("Explicit independent output directory required")
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(args.output.resolve()), encoding="utf-8")
    try:
        run(args.output.resolve(), args.protocol, torch.device(f"cuda:{args.gpu}"))
    except BlockingIOError:
        print("[BUSY] The suite is already running; existing state was not modified", flush=True)
        return
    except Exception as error:
        if args.output.exists():
            (args.output / "status.env").write_text(f"STATUS=FAILED\nERROR={type(error).__name__}: {error}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
