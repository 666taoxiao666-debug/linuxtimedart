#!/usr/bin/env python3
"""Bounded train-only numerical optimization; Wiki is entirely disabled."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "2"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import torch
from models.CausalResidualBase import CausalResidualBase
from scripts import train_kelmarsh_hourly as old
from scripts.train_kelmarsh_incremental_wiki import save_torch
from utils import certified_numeric_residual as identity_tools
from utils import numeric_foundation as foundation
from utils.gpu_runtime_policy import GPURuntimePolicy

PROTOCOL_ID = "kelmarsh_h2_numeric_foundation_trainonly_v11"


def source_configuration(spec):
    source = ROOT / spec["source_directory"]
    if old.sha(source / "protocol.json") != spec["source_protocol_sha256"]:
        raise ValueError("Original numerical protocol changed")
    p = old.read(source / "protocol.json")
    run = source / f"runs/f{spec['fold']}_s{spec['seed']}"
    complete = old.read(run / "completed.json")
    for name, expected in complete.items():
        if old.sha(run / name) != expected:
            raise ValueError("Completed original run changed: " + name)
    if old.sha(run / "PromptTimeDART/checkpoint.pt") != spec["source_base_checkpoint_sha256"]:
        raise ValueError("Reference forecast checkpoint changed")
    if old.sha(ROOT / p["data_path"]) != p["data_sha256"]:
        raise ValueError("Raw hourly data changed")
    return run, p


def stage_identity(dataset):
    return dict(windows=len(dataset), starts_sha256=identity_tools.array_digest(dict(starts=dataset.window_starts)),
                last_target=str(dataset.target_times().max()), first_target=str(dataset.target_times().min()))


def training_plot(history, directory, name):
    import matplotlib.pyplot as plt
    epochs = [row["epoch"] for row in history if "train_loss" in row]
    losses = [row["train_loss"] for row in history if "train_loss" in row]
    selected = [row for row in history if row.get("selection") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, losses, marker=".")
    axes[0].set(xlabel="Epoch", ylabel="Fit loss", title=name + " fitting")
    for key in ("mae", "rmse"):
        if selected:
            axes[1].plot([row["epoch"] for row in selected], [row["selection"][key] for row in selected], label=key.upper())
    if selected:
        axes[1].legend()
    else:
        axes[1].text(.5, .5, "Fixed refit: no outer metric selection", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set(xlabel="Epoch", ylabel="Internal error (kW)", title="Internal holdout only")
    fig.tight_layout()
    fig.savefig(directory / "training_curves.png", dpi=140)
    plt.close(fig)


def make_model(name, fit, p, spec, device, initial):
    old.seed_all(spec["seed"])
    if name == "ResidualTCN":
        return CausalResidualBase(width=spec["training"]["width"], dropout=spec["training"]["dropout"],
            bound_std=p["rated_power_kw"] * spec["training"]["bound_capacity"] / fit.scaler.scale_[-1]).to(device)
    if name != "PromptExpanded":
        raise ValueError("Unknown numerical architecture")
    args = old.model_args("PromptTimeDART", fit, p, device)
    if args.utility_wiki or args.ramp_residual or args.prompt_router != "trend":
        raise ValueError("Wiki and event/ramp experts must be disabled")
    model = old.PromptTimeDART.Model(args).to(device)
    src = torch.load(initial, map_location="cpu", weights_only=False)["model"]
    dst = model.state_dict()
    compatible = {key: value for key, value in src.items() if key in dst and value.shape == dst[key].shape}
    dst.update(compatible)
    model.load_state_dict(dst, strict=True)
    calibrated = old.calibrate_regime_thresholds_from_dataset(fit)
    model.regime_down_thresh.fill_(calibrated["down_thresh"])
    model.regime_up_thresh.fill_(calibrated["up_thresh"])
    return model


@torch.no_grad()
def predict(model, dataset, device, seed, resources):
    model.eval()
    chunks = []
    for x, _, _, _ in old.loader(dataset, seed, batch=128):
        resources.check()
        x = x.float().to(device)
        if isinstance(model, CausalResidualBase):
            chunks.append(model.residual(x).cpu().numpy().astype(float))
        else:
            chunks.append(old.extract(model(x))[..., -1].cpu().numpy().astype(float))
    z = np.concatenate(chunks)
    if isinstance(model, CausalResidualBase):
        return foundation.observed_anchor(dataset) + dataset.scaler.scale_[-1] * z
    return z * dataset.scaler.scale_[-1] + dataset.scaler.mean_[-1]


def neural_fit(name, fit, selection, p, spec, directory, identity, initial, device, resources, *, fixed_epochs=None):
    directory.mkdir(parents=True, exist_ok=True)
    contract = dict(identity, model=name, fit=stage_identity(fit),
                    selection=stage_identity(selection) if selection is not None else None,
                    fixed_refit_epochs=fixed_epochs, original_pretrain_sha256=old.sha(initial) if name == "PromptExpanded" else None)
    model = make_model(name, fit, p, spec, device, initial)
    marker = directory / "trained.json"
    if marker.exists():
        done = old.read(marker)
        if done["contract"] != contract or any(old.sha(directory / k) != v for k, v in done["artifacts"].items()):
            raise ValueError("Completed numerical stage changed")
        model.load_state_dict(torch.load(directory / "checkpoint.pt", map_location=device, weights_only=False)["model"], strict=True)
        print(f"[REUSE_STAGE] {name} fixed_epochs={fixed_epochs}", flush=True)
        return model, done
    cfg = spec["training"]
    if name == "PromptExpanded":
        new, base = [], []
        for key, parameter in model.named_parameters():
            (new if key.startswith(("head.", "channel_mixer.", "residual_gate")) else base).append(parameter)
        groups = [{"params": base, "lr": p["training"]["backbone_lr"]}, {"params": new, "lr": p["training"]["new_module_lr"]}]
    else:
        groups = [{"params": model.parameters(), "lr": cfg["residual_learning_rate"]}]
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"])
    fit_truth, fit_anchor = fit.truth_and_anchor()
    error_std = (fit_truth - fit_anchor) / fit.scaler.scale_[-1]
    loss_scale = dict(mae=max(float(abs(error_std).mean()), 1e-8), mse=max(float((error_std**2).mean()), 1e-8))
    if selection is not None:
        truth, anchor = selection.truth_and_anchor()
        baseline = old.model_metrics(anchor, truth, anchor, p["rated_power_kw"])
    else:
        truth = anchor = baseline = None
    history, start, best, best_epoch = [], 0, 1., 0
    if (directory / "last.pt").exists():
        last = torch.load(directory / "last.pt", map_location=device, weights_only=False)
        if last["contract"] != contract:
            raise ValueError("Interrupted numerical stage identity changed")
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        history, start, best, best_epoch = last["history"], last["epoch"], last["best"], last["best_epoch"]
        torch.set_rng_state(last["rng_cpu"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in last["rng_cuda"]])
    else:
        initial_metrics = (old.model_metrics(predict(model, selection, device, spec["seed"], resources), truth, anchor, p["rated_power_kw"])
                           if selection is not None else None)
        history.append(dict(epoch=0, selection=initial_metrics))
        save_torch(directory / "checkpoint.pt", dict(model=model.state_dict(), contract=contract))
    epochs = cfg["epochs"] if fixed_epochs is None else fixed_epochs
    for epoch in range(start + 1, epochs + 1):
        model.train()
        total, count = 0., 0
        for x, y, _, _ in old.loader(fit, spec["seed"] + epoch, shuffle=True, batch=cfg["batch_size"]):
            resources.check()
            x, y = x.float().to(device), y.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            if name == "ResidualTCN":
                error = model.residual(x) - (y[..., -1] - x[:, -1:, -1])
                loss = .5 * error.abs().mean() / loss_scale["mae"] + .5 * error.square().mean() / loss_scale["mse"]
            else:
                error = old.extract(model(x))[..., -1] - y[..., -1]
                weight = p["training"]["mix_mse_weight"]
                loss = (1 - weight) * error.abs().mean() + weight * error.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite numerical loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            resources.after_step()
            total += len(x) * float(loss.detach())
            count += len(x)
        if selection is not None:
            metrics = old.model_metrics(predict(model, selection, device, spec["seed"], resources), truth, anchor, p["rated_power_kw"])
            score = foundation.joint_score(metrics, baseline)
            if score < best - 1e-8:
                best, best_epoch = score, epoch
                save_torch(directory / "checkpoint.pt", dict(model=model.state_dict(), contract=contract))
        else:
            metrics, best_epoch = None, epoch
            # Fixed refit never evaluates or selects using outer validation.
            save_torch(directory / "checkpoint.pt", dict(model=model.state_dict(), contract=contract))
        history.append(dict(epoch=epoch, train_loss=total / count, selection=metrics))
        scheduler.step()
        save_torch(directory / "last.pt", dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            contract=contract, epoch=epoch, history=history, best=best, best_epoch=best_epoch,
            rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else []))
        old.write(directory / "training_history.json", history)
        print(f"[EPOCH] {name} {epoch}/{epochs} scope={'inner' if selection is not None else 'fixed_refit'} best={best_epoch}" +
              (f" MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f}" if metrics is not None else " no_outer_selection"), flush=True)
    model.load_state_dict(torch.load(directory / "checkpoint.pt", map_location=device, weights_only=False)["model"])
    if selection is not None:
        metrics = old.model_metrics(predict(model, selection, device, spec["seed"], resources), truth, anchor, p["rated_power_kw"])
    else:
        metrics = baseline
    old.write(directory / "training_history.json", history)
    training_plot(history, directory, name)
    old.write(directory / "manifest.json", dict(contract=contract, model=name, best_epoch=best_epoch,
        epochs_completed=epochs, training=cfg, loss_normalizers=loss_scale,
        original_scaler_mean=fit.scaler.mean_.tolist(), original_scaler_scale=fit.scaler.scale_.tolist(),
        wiki_enabled=False, numerical_units=[1, 2, 3, 4, 5, 6], selected_on_outer=False,
        parameters=sum(x.numel() for x in model.parameters())))
    done = dict(contract=contract, model=name, best_epoch=best_epoch, metrics=metrics,
        parameters=sum(x.numel() for x in model.parameters()), epochs_completed=epochs,
        qualifying_trained_epoch=bool(selection is not None and best_epoch > 0),
        metric_semantics="Actual saved model, not substituted Persistence; epoch0 cannot be selected for refit",
        artifacts={key: old.sha(directory / key) for key in ("checkpoint.pt", "manifest.json", "training_history.json", "training_curves.png")})
    old.write(marker, done)
    return model, done


def ridge_fit(name, fit, selection, alpha, p, directory, identity):
    directory.mkdir(parents=True, exist_ok=True)
    contract = dict(identity, model=name, alpha=alpha, fit=stage_identity(fit),
                    selection=stage_identity(selection) if selection is not None else None)
    marker = directory / "trained.json"
    if marker.exists():
        done = old.read(marker)
        if done["contract"] != contract or old.sha(directory / "model.npz") != done["model_sha256"]:
            raise ValueError("Completed Ridge stage changed")
        with np.load(directory / "model.npz", allow_pickle=False) as saved:
            return {key: saved[key].copy() for key in saved.files}, done
    model = foundation.fit_ridge(fit, alpha)
    np.savez_compressed(directory / "model.npz", **model)
    metrics = None
    if selection is not None:
        truth, anchor = selection.truth_and_anchor()
        metrics = old.model_metrics(foundation.ridge_predict(model, selection), truth, anchor, p["rated_power_kw"])
    done = dict(contract=contract, model=name, alpha=alpha, best_epoch=None, metrics=metrics,
        parameters=int(np.asarray(model["coefficients"]).size + np.asarray(model["intercept"]).size),
        model_sha256=old.sha(directory / "model.npz"))
    old.write(marker, done)
    return model, done


def outer_report(spec, output, series, prediction, original_run, selected):
    # Called only after selected.json and the fixed refit checkpoint freeze.
    validation = series.subset("validation")
    truth, anchor = validation.truth_and_anchor()
    forecasts = dict(Persistence=anchor, SelectedNumeric=prediction)
    for name in ("PromptTimeDART", "DLinear", "PatchTST", "Ridge"):
        with np.load(original_run / name / "raw_predictions.npz", allow_pickle=False) as saved:
            for key, expected in dict(truth=truth, times=validation.target_times(), turbines=series.turbines[validation.window_starts]).items():
                np.testing.assert_array_equal(saved[key], expected)
            forecasts["Historical" + name] = saved["prediction"].copy()
    rows, results = [], {}
    for name, value in forecasts.items():
        result = old.report(name, value, validation, output / "reports" / name, 2050.)
        result["paired"] = {ref: old.paired_forecast_statistics(value, forecasts[ref], truth,
            pd.DataFrame({"forecast_start": validation.target_times()[:, 0]}), **spec["statistics"])
            for ref in ("Persistence", "HistoricalPromptTimeDART") if ref != name}
        old.write(output / "reports" / name / "result.json", result)
        results[name] = result
        rows.append(dict(model=name, **result["metrics"]))
    method = results["SelectedNumeric"]["metrics"]
    success = all(method[key] < results[ref]["metrics"][key] for ref in ("Persistence", "HistoricalPromptTimeDART") for key in ("mae", "rmse"))
    old.write(output / "result.json", dict(results=rows, selected=selected, primary_joint_success=success,
        wiki_enabled=False, old_models_retrained=False, old_new_label_budget_identical=False, sealed_test_evaluated=False,
        interpretation="Train-only numerical optimization on one historically inspected development fold; not a Wiki contribution or independent confirmation"))
    pd.DataFrame(rows).to_csv(output / "metrics_all_runs.csv", index=False)


def run(protocol, output, shared=True):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / "suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = old.read(protocol)
        if spec["protocol_id"] != PROTOCOL_ID or spec["wiki_enabled"] or spec["fold"] != 1 or spec["seed"] != 2024:
            raise ValueError("Unknown numerical-only pilot")
        saved = output / "protocol.json"
        if saved.exists() and old.sha(saved) != old.sha(protocol):
            raise ValueError("Frozen optimization protocol changed")
        if not saved.exists():
            saved.write_bytes(protocol.read_bytes())
        if (output / "completed.json").exists():
            complete = old.read(output / "completed.json")
            if complete["protocol_sha256"] != old.sha(saved) or any(old.sha(output / k) != v for k, v in complete["artifacts"].items()):
                raise ValueError("Completed numerical result changed")
            print("[REUSE_SUITE] no repeated optimization", flush=True)
            return
        resources = GPURuntimePolicy(shared=shared,
            on_change=lambda value: old.write(output / "resource_policy.json", dict(value, protocol_sha256=old.sha(saved))))
        resources.configure()
        old.write(output / "status.json", dict(status="RUNNING", pid=os.getpid()))
        old.write(output / "progress.json", dict(stage="numerical_data_preflight"))
        original_run, p = source_configuration(spec)
        series = old.KelmarshSeries(ROOT / p["data_path"], fold=1, input_len=72, horizon=2, stride=3, start=p["start_utc"], end=p["end_utc_exclusive"])
        datasets, audit = foundation.partitions(series, spec["inner_fit_fraction_of_outer_train"])
        original_manifest = old.read(original_run / "PromptTimeDART/manifest.json")
        np.testing.assert_array_equal(series.scaler.mean_, original_manifest["contract"]["scaler_mean"])
        np.testing.assert_array_equal(series.scaler.scale_, original_manifest["contract"]["scaler_scale"])
        initial = original_run / "pretrain/checkpoint.pt"
        code = ("scripts/train_kelmarsh_numeric_foundation.py", "models/CausalResidualBase.py", "utils/numeric_foundation.py",
                "scripts/train_kelmarsh_hourly.py", "models/PromptTimeDART.py", "models/ContextualWindResidual.py",
                "models/TimeDART.py", "data_provider/kelmarsh_hourly.py", "utils/gpu_runtime_policy.py")
        identity = dict(protocol_sha256=old.sha(saved), source_completed_sha256=old.sha(original_run / "completed.json"),
            initial_pretrain_sha256=old.sha(initial), data_sha256=p["data_sha256"],
            source_sha256={key: old.sha(ROOT / key) for key in code}, partitions={key: stage_identity(value) for key, value in datasets.items()})
        if (output / "prepared.json").exists() and old.read(output / "prepared.json")["identity"] != identity:
            raise ValueError("Numerical data/implementation changed on resume")
        old.write(output / "prepared.json", dict(identity=identity, audit=audit, source_data_audit=series.audit,
            legacy_base_fit=stage_identity(series.subset("fit")), legacy_base_selection=stage_identity(series.subset("selection")), wiki_enabled=False))
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
        fit, selection = datasets["fit"], datasets["selection"]
        rows = []
        for alpha in spec["ridge_alphas"]:
            name = f"DeltaRidge_a{alpha:g}"
            old.write(output / "progress.json", dict(stage="inner_" + name))
            _, done = ridge_fit(name, fit, selection, alpha, p, output / "inner" / name, identity)
            rows.append({key: done[key] for key in ("model", "alpha", "best_epoch", "metrics", "parameters")})
        for name in ("PromptExpanded", "ResidualTCN"):
            old.write(output / "progress.json", dict(stage="inner_" + name))
            model, done = neural_fit(name, fit, selection, p, spec, output / "inner" / name, identity, initial, device, resources)
            rows.append({key: done[key] for key in ("model", "best_epoch", "metrics", "parameters")})
            del model
            gc.collect()
            torch.cuda.empty_cache()
        truth, anchor = selection.truth_and_anchor()
        baseline = old.model_metrics(anchor, truth, anchor, p["rated_power_kw"])
        selected = foundation.select_candidate(rows, baseline)
        selected.update(selection_arrays_sha256=identity_tools.array_digest(dict(times=selection.target_times(), truth=truth, anchor=anchor)),
                        protocol_sha256=old.sha(saved), outer_used_for_selection=False)
        if (output / "selected.json").exists() and old.read(output / "selected.json") != selected:
            raise ValueError("Frozen inner selection changed")
        old.write(output / "inner_candidates.json", dict(results=rows, persistence=baseline))
        old.write(output / "selected.json", selected)
        old.write(output / "progress.json", dict(stage="fixed_outer_refit", selected=selected["model"]))
        all_train, name = datasets["outer_refit"], selected["model"]
        # No outer prediction/metric is computed before the chosen refit freezes.
        if name.startswith("DeltaRidge"):
            model, _ = ridge_fit(name, all_train, None, selected["alpha"], p, output / "outer_refit", identity)
            validation = series.subset("validation")
            prediction = foundation.ridge_predict(model, validation)
        elif name != "Persistence":
            model, _ = neural_fit(name, all_train, None, p, spec, output / "outer_refit", identity, initial, device, resources,
                                  fixed_epochs=selected["best_epoch"])
            validation = series.subset("validation")
            prediction = predict(model, validation, device, spec["seed"], resources)
        else:
            validation = series.subset("validation")
            prediction = foundation.observed_anchor(validation)
        old.write(output / "progress.json", dict(stage="outer_report_no_selection"))
        outer_report(spec, output, series, prediction, original_run, selected)
        artifacts = {str(path.relative_to(output)): old.sha(path) for path in output.rglob("*") if path.is_file() and path.name
            not in ("suite.lock", "launch.log", "launcher.pid", "source_commits.log", "status.json", "progress.json", "completed.json")}
        old.write(output / "completed.json", dict(protocol_sha256=old.sha(saved), artifacts=artifacts))
        old.write(output / "status.json", dict(status="COMPLETED", exit_code=0, pid=os.getpid()))
        old.write(output / "progress.json", dict(stage="COMPLETED"))
        print("[SUITE_COMPLETE] numerical-only selected refit; no Wiki trained", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/kelmarsh_numeric_foundation_protocol_v11.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--idle-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args.protocol.resolve(), args.output.resolve(), not args.idle_only)
    except BlockingIOError:
        print("[BUSY] numerical suite already running", flush=True)
    except Exception as error:
        old.write(args.output / "status.json", dict(status="FAILED", error=f"{type(error).__name__}: {error}", pid=os.getpid()))
        raise
