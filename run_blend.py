"""Validation-selected Persistence + PromptTimeDART forecast blending.

This script is intentionally separate from training.  It loads a frozen
fine-tuned checkpoint, generates validation forecasts, chooses two blending
parameters using validation data only, freezes the resulting horizon weights,
and evaluates those weights once on the test split.

The horizon-dependent blend is

    blend[h] = (1 - w[h]) * persistence[h] + w[h] * model[h]

where

    w[h] = 0                                      for h <= cutoff
    w[h] = beta * (h - cutoff) / (H - cutoff)    for h > cutoff.

The script reuses the project's run.py parser, data loader, model definition,
checkpoint loader, and target inverse transform.  No existing source file is
modified and no model is trained.
"""

import hashlib
import json
import math
import os
import random
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = np.finfo(np.float64).eps


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, ensure_ascii=False, indent=2)


def _parse_int_grid(text):
    values = sorted({int(item.strip()) for item in str(text).split(",") if item.strip()})
    if not values:
        raise ValueError("blend_cutoffs must contain at least one integer")
    return values


def _parse_float_grid(text):
    values = sorted(
        {round(float(item.strip()), 10) for item in str(text).split(",") if item.strip()}
    )
    if not values:
        raise ValueError("blend_betas must contain at least one number")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Every blend beta must be in [0, 1]")
    if 0.0 not in values:
        values.insert(0, 0.0)
    return values


def _target_2d(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(
            f"{name} must have shape [window, horizon] or "
            f"[window, horizon, 1], got {values.shape}"
        )
    if values.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return values


def forecast_metrics(prediction, truth):
    prediction = _target_2d(prediction, "prediction")
    truth = _target_2d(truth, "truth")
    if prediction.shape != truth.shape:
        raise ValueError(
            f"Prediction/truth shapes differ: {prediction.shape} != {truth.shape}"
        )

    pred_flat = prediction.reshape(-1)
    true_flat = truth.reshape(-1)
    error = pred_flat - true_flat
    absolute_error = np.abs(error)
    mse = float(np.mean(error**2))
    mae = float(np.mean(absolute_error))
    true_centered = true_flat - np.mean(true_flat)
    total_sum_squares = float(np.sum(true_centered**2))
    r2 = (
        1.0 - float(np.sum(error**2)) / total_sum_squares
        if total_sum_squares > EPS
        else float("nan")
    )

    pred_std = float(np.std(pred_flat))
    true_std = float(np.std(true_flat))
    pearson = (
        float(np.corrcoef(pred_flat, true_flat)[0, 1])
        if pred_std > EPS and true_std > EPS
        else float("nan")
    )
    smape = float(
        200.0
        * np.mean(
            absolute_error
            / np.maximum(np.abs(pred_flat) + np.abs(true_flat), EPS)
        )
    )
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "median_ae": float(np.median(absolute_error)),
        "r2": float(r2),
        "pearson_r": pearson,
        "smape_pct": smape,
        "mbe": float(np.mean(error)),
        "overestimate_pct": float(100.0 * np.mean(error > 0.0)),
        "point_count": int(error.size),
    }


def _with_skill(metrics, persistence_metrics):
    result = dict(metrics)
    result["mae_skill_vs_persistence_pct"] = float(
        100.0 * (1.0 - metrics["mae"] / max(persistence_metrics["mae"], EPS))
    )
    result["rmse_skill_vs_persistence_pct"] = float(
        100.0 * (1.0 - metrics["rmse"] / max(persistence_metrics["rmse"], EPS))
    )
    return result


def blend_weights(horizon, cutoff, beta):
    horizon = int(horizon)
    cutoff = int(cutoff)
    beta = float(beta)
    if horizon < 2:
        raise ValueError("Forecast horizon must be at least 2")
    if cutoff < 0 or cutoff >= horizon:
        raise ValueError(f"cutoff must be in [0, {horizon - 1}], got {cutoff}")
    if beta < 0.0 or beta > 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")

    steps = np.arange(1, horizon + 1, dtype=np.float64)
    weights = np.zeros(horizon, dtype=np.float64)
    after = steps > cutoff
    weights[after] = beta * (steps[after] - cutoff) / (horizon - cutoff)
    return weights


def apply_blend(model_prediction, persistence, weights):
    model_prediction = _target_2d(model_prediction, "model_prediction")
    persistence = _target_2d(persistence, "persistence")
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if model_prediction.shape != persistence.shape:
        raise ValueError(
            "Model/persistence shapes differ: "
            f"{model_prediction.shape} != {persistence.shape}"
        )
    if model_prediction.shape[1] != len(weights):
        raise ValueError(
            f"Weight count {len(weights)} does not match horizon "
            f"{model_prediction.shape[1]}"
        )
    if np.any(weights < 0.0) or np.any(weights > 1.0):
        raise ValueError("Blend weights must be in [0, 1]")
    return persistence * (1.0 - weights[None, :]) + model_prediction * weights[None, :]


def search_validation_weights(
    model_prediction,
    truth,
    persistence,
    cutoffs,
    betas,
    mae_tolerance_pct=1.0,
):
    model_prediction = _target_2d(model_prediction, "validation model prediction")
    truth = _target_2d(truth, "validation truth")
    persistence = _target_2d(persistence, "validation persistence")
    if not (model_prediction.shape == truth.shape == persistence.shape):
        raise ValueError("Validation model, truth, and persistence shapes must match")

    horizon = model_prediction.shape[1]
    valid_cutoffs = [int(value) for value in cutoffs if 0 <= int(value) < horizon]
    if not valid_cutoffs:
        raise ValueError(f"No cutoff is valid for horizon={horizon}: {cutoffs}")
    if mae_tolerance_pct < 0.0:
        raise ValueError("mae_tolerance_pct must be non-negative")

    persistence_metrics = forecast_metrics(persistence, truth)
    maximum_mae = persistence_metrics["mae"] * (1.0 + mae_tolerance_pct / 100.0)
    rows = []
    for cutoff in valid_cutoffs:
        for beta in betas:
            weights = blend_weights(horizon, cutoff, beta)
            blended = apply_blend(model_prediction, persistence, weights)
            metrics = forecast_metrics(blended, truth)
            accepted = bool(
                metrics["rmse"] <= persistence_metrics["rmse"] + 1e-12
                and metrics["mae"] <= maximum_mae + 1e-12
            )
            rows.append(
                {
                    "cutoff": int(cutoff),
                    "beta": float(beta),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "mae_skill_vs_persistence_pct": 100.0
                    * (1.0 - metrics["mae"] / max(persistence_metrics["mae"], EPS)),
                    "rmse_skill_vs_persistence_pct": 100.0
                    * (1.0 - metrics["rmse"] / max(persistence_metrics["rmse"], EPS)),
                    "accepted": accepted,
                }
            )

    table = pd.DataFrame(rows)
    accepted = table.loc[table["accepted"]].copy()
    if accepted.empty:
        raise RuntimeError(
            "No validation candidate passed the RMSE and MAE constraints. "
            "This should not happen because beta=0 is inserted automatically."
        )

    # RMSE is the primary objective.  The remaining columns provide stable,
    # conservative tie-breaking: lower MAE, less model weight, later cutoff.
    accepted = accepted.sort_values(
        ["rmse", "mae", "beta", "cutoff"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )
    best = accepted.iloc[0].to_dict()
    weights = blend_weights(horizon, int(best["cutoff"]), float(best["beta"]))
    return best, weights, table, persistence_metrics


def _unwrap_model_output(output):
    import torch

    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        if torch.is_tensor(output[0]):
            return output[0]
    if isinstance(output, dict):
        for key in ("pred", "prediction", "output"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    raise TypeError(f"Unsupported forecast model output type: {type(output).__name__}")


def collect_split_predictions(exp, split):
    import torch
    from tqdm import tqdm

    from utils.forecast_report import inverse_transform_target

    dataset, loader = exp._get_data(flag=split)
    predictions = []
    truths = []
    last_observations = []
    exp.model.eval()

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(
            loader, desc=f"Blend inference ({split})"
        ):
            del batch_x_mark, batch_y_mark
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            with exp._autocast():
                output = _unwrap_model_output(exp.model(batch_x))
                feature_start = -1 if exp.args.features == "MS" else 0
                output = output[:, -exp.args.pred_len :, feature_start:]
                target = batch_y[:, -exp.args.pred_len :, feature_start:]

            if output.shape[-1] != 1 or target.shape[-1] != 1:
                raise ValueError(
                    "Blending is target-only. Use features=MS or a single-variable "
                    f"dataset; got output={tuple(output.shape)}, target={tuple(target.shape)}"
                )

            predictions.append(output.detach().cpu())
            truths.append(target.detach().cpu())
            if exp.args.features == "MS":
                last_observations.append(batch_x[:, -1:, -1:].detach().cpu())
            elif batch_x.shape[-1] == 1:
                last_observations.append(batch_x[:, -1:, :].detach().cpu())
            else:
                raise ValueError(
                    "Cannot identify the target channel for the Persistence baseline"
                )

    if not predictions:
        raise RuntimeError(f"The {split} loader produced no forecast windows")

    prediction_scaled = torch.cat(predictions, dim=0).numpy()
    truth_scaled = torch.cat(truths, dim=0).numpy()
    last_scaled = torch.cat(last_observations, dim=0).numpy()
    prediction = _target_2d(
        inverse_transform_target(dataset, prediction_scaled),
        f"{split} model prediction",
    )
    truth = _target_2d(
        inverse_transform_target(dataset, truth_scaled), f"{split} truth"
    )
    last = np.asarray(
        inverse_transform_target(dataset, last_scaled), dtype=np.float64
    ).reshape(-1, 1)
    persistence = np.repeat(last, prediction.shape[1], axis=1)
    if not (prediction.shape == truth.shape == persistence.shape):
        raise ValueError(
            f"Collected {split} shapes differ: prediction={prediction.shape}, "
            f"truth={truth.shape}, persistence={persistence.shape}"
        )
    return {
        "prediction_original": prediction,
        "truth_original": truth,
        "persistence_original": persistence,
        "prediction_normalized": _target_2d(
            prediction_scaled, f"{split} normalized prediction"
        ),
        "truth_normalized": _target_2d(
            truth_scaled, f"{split} normalized truth"
        ),
    }


def _save_arrays(path, arrays, blended=None, weights=None):
    payload = {
        name: np.asarray(value, dtype=np.float32) for name, value in arrays.items()
    }
    if blended is not None:
        payload["blend_original"] = np.asarray(blended, dtype=np.float32)
    if weights is not None:
        payload["weight_by_horizon"] = np.asarray(weights, dtype=np.float32)
    np.savez_compressed(path, **payload)


def _comparison(split, arrays, blended):
    truth = arrays["truth_original"]
    persistence_metrics = forecast_metrics(arrays["persistence_original"], truth)
    model_metrics = _with_skill(
        forecast_metrics(arrays["prediction_original"], truth), persistence_metrics
    )
    blend_metrics = _with_skill(forecast_metrics(blended, truth), persistence_metrics)
    return {
        "split": split,
        "persistence": persistence_metrics,
        "PromptTimeDART": model_metrics,
        "blend": blend_metrics,
    }


def _summary_table(validation_comparison, test_comparison):
    rows = []
    for comparison in (validation_comparison, test_comparison):
        for method in ("persistence", "PromptTimeDART", "blend"):
            metrics = comparison[method]
            row = {"split": comparison["split"], "method": method}
            for key in (
                "mae",
                "rmse",
                "r2",
                "pearson_r",
                "smape_pct",
                "mbe",
                "overestimate_pct",
                "mae_skill_vs_persistence_pct",
                "rmse_skill_vs_persistence_pct",
            ):
                if key in metrics:
                    row[key] = metrics[key]
            rows.append(row)
    return pd.DataFrame(rows)


def _horizon_table(arrays, blended):
    model = arrays["prediction_original"]
    truth = arrays["truth_original"]
    persistence = arrays["persistence_original"]
    rows = []
    for index in range(model.shape[1]):
        base = forecast_metrics(persistence[:, index : index + 1], truth[:, index : index + 1])
        model_metrics = forecast_metrics(model[:, index : index + 1], truth[:, index : index + 1])
        blend_metrics = forecast_metrics(blended[:, index : index + 1], truth[:, index : index + 1])
        rows.append(
            {
                "horizon_step": index + 1,
                "persistence_mae": base["mae"],
                "model_mae": model_metrics["mae"],
                "blend_mae": blend_metrics["mae"],
                "persistence_rmse": base["rmse"],
                "model_rmse": model_metrics["rmse"],
                "blend_rmse": blend_metrics["rmse"],
                "model_rmse_skill_pct": 100.0
                * (1.0 - model_metrics["rmse"] / max(base["rmse"], EPS)),
                "blend_rmse_skill_pct": 100.0
                * (1.0 - blend_metrics["rmse"] / max(base["rmse"], EPS)),
            }
        )
    return pd.DataFrame(rows)


def _segment_bounds(horizon):
    candidates = [
        ("short_1_12", 1, 12),
        ("mid_short_13_24", 13, 24),
        ("medium_25_48", 25, 48),
        (f"long_49_{horizon}", 49, horizon),
    ]
    return [(name, start, min(end, horizon)) for name, start, end in candidates if start <= horizon]


def _segment_table(arrays, blended):
    model = arrays["prediction_original"]
    truth = arrays["truth_original"]
    persistence = arrays["persistence_original"]
    rows = []
    for name, start, end in _segment_bounds(model.shape[1]):
        region = slice(start - 1, end)
        base = forecast_metrics(persistence[:, region], truth[:, region])
        model_metrics = forecast_metrics(model[:, region], truth[:, region])
        blend_metrics = forecast_metrics(blended[:, region], truth[:, region])
        rows.append(
            {
                "segment": name,
                "start_step": start,
                "end_step": end,
                "persistence_mae": base["mae"],
                "model_mae": model_metrics["mae"],
                "blend_mae": blend_metrics["mae"],
                "persistence_rmse": base["rmse"],
                "model_rmse": model_metrics["rmse"],
                "blend_rmse": blend_metrics["rmse"],
                "model_rmse_skill_pct": 100.0
                * (1.0 - model_metrics["rmse"] / max(base["rmse"], EPS)),
                "blend_rmse_skill_pct": 100.0
                * (1.0 - blend_metrics["rmse"] / max(base["rmse"], EPS)),
            }
        )
    return pd.DataFrame(rows)


def _plot_weights(weights, cutoff, beta, output_dir):
    steps = np.arange(1, len(weights) + 1)
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(steps, weights, linewidth=2.3, color="tab:purple")
    axis.axvline(cutoff, color="black", linestyle="--", linewidth=1.2)
    axis.set(
        xlabel="Forecast horizon step",
        ylabel="PromptTimeDART weight",
        title=f"Frozen validation-selected weights (cutoff={cutoff}, beta={beta:.2f})",
        xlim=(1, len(weights)),
        ylim=(-0.02, 1.02),
    )
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "blend_weights.png"), dpi=180)
    plt.close(fig)


def _plot_horizon(table, output_dir):
    steps = table["horizon_step"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    styles = {
        "persistence": {"color": "tab:orange", "linestyle": "--"},
        "model": {"color": "tab:blue", "linestyle": "-"},
        "blend": {"color": "tab:green", "linestyle": "-"},
    }
    for method, style in styles.items():
        axes[0].plot(steps, table[f"{method}_mae"], label=method, **style)
        axes[1].plot(steps, table[f"{method}_rmse"], label=method, **style)
    axes[0].set_ylabel("MAE (kW)")
    axes[1].set_ylabel("RMSE (kW)")
    axes[2].plot(
        steps,
        table["model_rmse_skill_pct"],
        label="PromptTimeDART",
        color="tab:blue",
    )
    axes[2].plot(
        steps,
        table["blend_rmse_skill_pct"],
        label="Blend",
        color="tab:green",
    )
    axes[2].axhline(0.0, color="black", linewidth=1.0)
    axes[2].set_ylabel("RMSE skill vs Persistence (%)")
    axes[2].set_xlabel("Forecast horizon step")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "error_by_horizon.png"), dpi=180)
    plt.close(fig)


def _plot_examples(arrays, blended, output_dir, count=6):
    model = arrays["prediction_original"]
    truth = arrays["truth_original"]
    persistence = arrays["persistence_original"]
    window_rmse = np.sqrt(np.mean((blended - truth) ** 2, axis=1))
    order = np.argsort(window_rmse)
    count = min(int(count), len(order))
    positions = np.linspace(0, len(order) - 1, count).round().astype(int)
    selected = order[positions]
    steps = np.arange(1, model.shape[1] + 1)
    fig, axes = plt.subplots(count, 1, figsize=(12, 2.8 * count), squeeze=False)
    for axis, window_index in zip(axes[:, 0], selected):
        axis.plot(steps, truth[window_index], color="black", linewidth=2.0, label="Truth")
        axis.plot(
            steps,
            persistence[window_index],
            color="tab:orange",
            linestyle="--",
            linewidth=1.4,
            label="Persistence",
        )
        axis.plot(
            steps,
            model[window_index],
            color="tab:blue",
            linewidth=1.4,
            label="PromptTimeDART",
        )
        axis.plot(
            steps,
            blended[window_index],
            color="tab:green",
            linewidth=1.9,
            label="Blend",
        )
        axis.set_title(
            f"window={window_index} | blend RMSE={window_rmse[window_index]:.2f} kW"
        )
        axis.grid(alpha=0.2)
    axes[0, 0].legend(ncol=4, fontsize=9)
    axes[-1, 0].set_xlabel("Forecast horizon step")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "forecast_examples.png"), dpi=180)
    plt.close(fig)


def _write_score(path, best, validation_comparison, test_comparison):
    lines = [
        "Persistence + PromptTimeDART validation-selected blend",
        "=" * 72,
        f"cutoff: {int(best['cutoff'])}",
        f"beta: {float(best['beta']):.4f}",
        "",
    ]
    for comparison in (validation_comparison, test_comparison):
        lines.append(f"[{comparison['split']}]")
        for method in ("persistence", "PromptTimeDART", "blend"):
            values = comparison[method]
            text = (
                f"{method:14s} MAE={values['mae']:.4f} kW  "
                f"RMSE={values['rmse']:.4f} kW  R2={values['r2']:.6f}"
            )
            if "rmse_skill_vs_persistence_pct" in values:
                text += (
                    f"  RMSE_skill={values['rmse_skill_vs_persistence_pct']:+.3f}%"
                )
            lines.append(text)
        lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_blend_arguments(parser):
    parser.description = "Validation-selected Persistence + PromptTimeDART blending"
    parser.add_argument(
        "--blend_cutoffs",
        default="24,36,48,60",
        help="comma-separated candidate cutoff steps",
    )
    parser.add_argument(
        "--blend_betas",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="comma-separated maximum PromptTimeDART weights",
    )
    parser.add_argument(
        "--blend_mae_tolerance_pct",
        type=float,
        default=1.0,
        help="maximum validation MAE degradation versus Persistence",
    )
    parser.add_argument(
        "--blend_output_dir",
        default=None,
        help="output directory; defaults to outputs/blend_results/.../idTIMESTAMP",
    )
    parser.add_argument(
        "--blend_run_id",
        default=None,
        help="optional stable identifier; defaults to YYYYMMDD_HHMMSS",
    )
    return parser


def main():
    # These imports are lazy so the pure blending functions remain testable
    # without importing the complete TimeDART project.
    from exp.exp_timedart import Exp_TimeDART
    from run import build_parser, configure_args, load_finetuned_model
    import torch

    parser = add_blend_arguments(build_parser())
    args = configure_args(parser.parse_args())
    if args.task_name != "finetune" or args.downstream_task != "forecast":
        raise ValueError("run_blend.py requires --task_name finetune --downstream_task forecast")
    if not args.finetune_checkpoint:
        raise ValueError("--finetune_checkpoint is required; use the frozen best checkpoint")

    checkpoint_path = os.path.abspath(os.path.expanduser(args.finetune_checkpoint))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {checkpoint_path}")

    run_id = args.blend_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.blend_output_dir or os.path.join(
        "./outputs/blend_results", args.model, args.data, f"id{run_id}"
    )
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cutoffs = _parse_int_grid(args.blend_cutoffs)
    betas = _parse_float_grid(args.blend_betas)

    # Prevent Exp_TimeDART from also loading a pretraining checkpoint.  The
    # complete frozen fine-tuned state is loaded immediately afterwards.
    args.load_checkpoints = None
    exp = Exp_TimeDART(args)
    load_finetuned_model(exp, checkpoint_path)

    print("Generating validation predictions (used for parameter selection only)...")
    validation = collect_split_predictions(exp, "val")
    best, weights, grid, validation_persistence = search_validation_weights(
        validation["prediction_original"],
        validation["truth_original"],
        validation["persistence_original"],
        cutoffs,
        betas,
        mae_tolerance_pct=args.blend_mae_tolerance_pct,
    )
    validation_blend = apply_blend(
        validation["prediction_original"],
        validation["persistence_original"],
        weights,
    )
    _save_arrays(
        os.path.join(output_dir, "validation_predictions.npz"),
        validation,
        validation_blend,
        weights,
    )
    grid.sort_values(
        ["accepted", "rmse", "mae"], ascending=[False, True, True]
    ).to_csv(os.path.join(output_dir, "validation_search.csv"), index=False)

    # The first access to test targets occurs only after cutoff/beta are frozen.
    print(
        "Frozen validation choice: "
        f"cutoff={int(best['cutoff'])}, beta={float(best['beta']):.3f}"
    )
    print("Generating test predictions and applying the frozen weights once...")
    test = collect_split_predictions(exp, "test")
    if test["prediction_original"].shape[1] != len(weights):
        raise ValueError(
            "Validation/test horizons differ: "
            f"{len(weights)} != {test['prediction_original'].shape[1]}"
        )
    test_blend = apply_blend(
        test["prediction_original"], test["persistence_original"], weights
    )
    _save_arrays(
        os.path.join(output_dir, "test_predictions.npz"),
        test,
        test_blend,
        weights,
    )

    validation_comparison = _comparison("validation", validation, validation_blend)
    test_comparison = _comparison("test", test, test_blend)
    _write_json(
        os.path.join(output_dir, "validation_metrics.json"), validation_comparison
    )
    _write_json(os.path.join(output_dir, "test_metrics.json"), test_comparison)

    configuration = {
        "method": "linear_horizon_blend",
        "formula": "(1-w_h)*persistence + w_h*PromptTimeDART",
        "selection_split": "validation",
        "evaluation_split": "test",
        "selection_primary_metric": "rmse",
        "validation_mae_tolerance_pct": args.blend_mae_tolerance_pct,
        "candidate_cutoffs": cutoffs,
        "candidate_betas": betas,
        "selected_cutoff": int(best["cutoff"]),
        "selected_beta": float(best["beta"]),
        "selected_validation_mae": float(best["mae"]),
        "selected_validation_rmse": float(best["rmse"]),
        "validation_persistence_mae": validation_persistence["mae"],
        "validation_persistence_rmse": validation_persistence["rmse"],
        "weight_by_horizon": weights,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "seed": args.seed,
        "run_id": run_id,
    }
    _write_json(os.path.join(output_dir, "blend_config.json"), configuration)

    _summary_table(validation_comparison, test_comparison).to_csv(
        os.path.join(output_dir, "metrics_summary.csv"), index=False
    )
    horizon = _horizon_table(test, test_blend)
    horizon["model_weight"] = weights
    horizon.to_csv(os.path.join(output_dir, "metrics_by_horizon.csv"), index=False)
    _segment_table(test, test_blend).to_csv(
        os.path.join(output_dir, "metrics_by_horizon_segment.csv"), index=False
    )
    _write_score(
        os.path.join(output_dir, "score.txt"),
        best,
        validation_comparison,
        test_comparison,
    )
    _plot_weights(weights, int(best["cutoff"]), float(best["beta"]), output_dir)
    _plot_horizon(horizon, output_dir)
    _plot_examples(test, test_blend, output_dir)

    if hasattr(exp, "writer"):
        exp.writer.flush()
        exp.writer.close()

    blend_metrics = test_comparison["blend"]
    model_metrics = test_comparison["PromptTimeDART"]
    persistence_metrics = test_comparison["persistence"]
    print("=" * 72)
    print(f"Output: {output_dir}")
    print(
        f"Test Persistence : MAE={persistence_metrics['mae']:.3f} kW, "
        f"RMSE={persistence_metrics['rmse']:.3f} kW"
    )
    print(
        f"Test PromptTimeDART: MAE={model_metrics['mae']:.3f} kW, "
        f"RMSE={model_metrics['rmse']:.3f} kW"
    )
    print(
        f"Test Blend       : MAE={blend_metrics['mae']:.3f} kW, "
        f"RMSE={blend_metrics['rmse']:.3f} kW, "
        f"RMSE skill={blend_metrics['rmse_skill_vs_persistence_pct']:+.3f}%"
    )
    if float(best["beta"]) == 0.0:
        print(
            "Validation selected beta=0: the defensible result is Persistence; "
            "no non-zero blend passed the validation objective."
        )


if __name__ == "__main__":
    main()