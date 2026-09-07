import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.metrics import forecast_metrics


def _target_2d(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(
            "Forecast reporting expects [window, horizon] or "
            f"[window, horizon, 1], got {values.shape}"
        )
    return values


def inverse_transform_target(dataset, values):
    """Inverse-transform only the target channel of an M/MS/S dataset.

    sklearn's StandardScaler cannot inverse-transform an MS prediction with
    one output channel after it was fitted on all input channels.  Applying the
    fitted target mean and scale directly is the correct inverse operation.
    """
    values = np.asarray(values, dtype=np.float64)
    if not getattr(dataset, "scale", True):
        return values.copy()

    scaler = getattr(dataset, "scaler", None)
    columns = list(getattr(dataset, "feature_columns", []))
    target = getattr(dataset, "target", None)
    target_index = columns.index(target) if target in columns else -1

    if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean = np.asarray(scaler.mean_, dtype=np.float64).reshape(-1)[target_index]
        scale = np.asarray(scaler.scale_, dtype=np.float64).reshape(-1)[target_index]
        return values * scale + mean

    if scaler is not None and hasattr(scaler, "mean") and hasattr(scaler, "std"):
        mean = np.asarray(scaler.mean, dtype=np.float64).reshape(-1)[target_index]
        scale = np.asarray(scaler.std, dtype=np.float64).reshape(-1)[target_index]
        return values * scale + mean

    # This fallback is valid for genuinely single-variable datasets.
    if hasattr(dataset, "inverse_transform"):
        restored = dataset.inverse_transform(values.reshape(-1, 1))
        return np.asarray(restored).reshape(values.shape)
    raise AttributeError("Dataset has no usable scaler or inverse_transform method")


def _window_metadata(dataset, window_count):
    starts = getattr(dataset, "window_starts", None)
    dates = getattr(dataset, "dates", None)
    turbines = getattr(dataset, "turbines", None)
    seq_len = getattr(dataset, "seq_len", None)
    if starts is None or dates is None or seq_len is None:
        return pd.DataFrame({"window_index": np.arange(window_count)})

    starts = np.asarray(starts, dtype=np.int64)
    if len(starts) != window_count:
        return pd.DataFrame({"window_index": np.arange(window_count)})
    target_starts = starts + int(seq_len)
    result = pd.DataFrame(
        {
            "window_index": np.arange(window_count),
            "forecast_start": pd.to_datetime(np.asarray(dates)[target_starts]),
        }
    )
    if turbines is not None:
        result["TurbID"] = np.asarray(turbines)[target_starts]
    return result


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _add_persistence_metrics(result, pred, true, persistence):
    persistence_metrics = forecast_metrics(persistence, true)
    result["persistence_mae"] = persistence_metrics["mae"]
    result["persistence_rmse"] = persistence_metrics["rmse"]
    result["mae_skill_vs_persistence_pct"] = 100.0 * (
        1.0 - result["mae"] / max(persistence_metrics["mae"], np.finfo(float).eps)
    )
    result["rmse_skill_vs_persistence_pct"] = 100.0 * (
        1.0 - result["rmse"] / max(persistence_metrics["rmse"], np.finfo(float).eps)
    )
    result["mae_ratio_to_persistence"] = result["mae"] / max(
        persistence_metrics["mae"], np.finfo(float).eps
    )
    if "direction_accuracy_pct" in persistence_metrics:
        result["persistence_direction_accuracy_pct"] = persistence_metrics[
            "direction_accuracy_pct"
        ]


def _save_summary(output_dir, normalized_metrics, original_metrics):
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(
            _json_safe({"normalized": normalized_metrics, "original": original_metrics}),
            handle,
            ensure_ascii=False,
            indent=2,
        )

    rows = []
    for scale_name, values in (
        ("normalized", normalized_metrics),
        ("original", original_metrics),
    ):
        for name, value in values.items():
            rows.append({"scale": scale_name, "metric": name, "value": value})
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "metrics_summary.csv"), index=False
    )

    lines = [
        "Forecast metrics (primary values are in original power units)",
        "=" * 72,
        "Task split: 0-4 h is the SCADA-only claim; 4-16 h without NWP is",
        "an extrapolation ceiling, not a SOTA comparison.",
        "",
    ]
    for name, value in original_metrics.items():
        if isinstance(value, (int, np.integer)):
            lines.append(f"{name:36s}: {int(value)}")
        elif value is None or not np.isfinite(value):
            lines.append(f"{name:36s}: NA")
        else:
            lines.append(f"{name:36s}: {float(value):.8f}")
    lines.extend(["", "Normalized-space metrics", "-" * 72])
    for name, value in normalized_metrics.items():
        if isinstance(value, (int, np.integer)):
            lines.append(f"{name:36s}: {int(value)}")
        elif value is None or not np.isfinite(value):
            lines.append(f"{name:36s}: NA")
        else:
            lines.append(f"{name:36s}: {float(value):.8f}")
    with open(os.path.join(output_dir, "score.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _horizon_table(
    pred, true, step_minutes, rated_power, persistence=None
):
    rows = []
    for index in range(pred.shape[1]):
        values = forecast_metrics(
            pred[:, index], true[:, index], rated_power=rated_power
        )
        row = {
                "horizon_step": index + 1,
                "lead_minutes": (index + 1) * step_minutes,
                "mae": values["mae"],
                "rmse": values["rmse"],
                "mape_masked_pct": values["mape_masked_pct"],
                "smape_pct": values["smape_pct"],
                "wape_pct": values["wape_pct"],
                "r2": values["r2"],
                "pearson_r": values["pearson_r"],
                "bias_mbe": values["bias_mbe"],
            }
        if persistence is not None:
            baseline = forecast_metrics(
                persistence[:, index], true[:, index], rated_power=rated_power
            )
            eps = np.finfo(float).eps
            row.update(
                {
                    "persistence_mae": baseline["mae"],
                    "persistence_rmse": baseline["rmse"],
                    "persistence_r2": baseline["r2"],
                    "mae_skill_vs_persistence_pct": 100.0
                    * (1.0 - values["mae"] / max(baseline["mae"], eps)),
                    "rmse_skill_vs_persistence_pct": 100.0
                    * (1.0 - values["rmse"] / max(baseline["rmse"], eps)),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _horizon_task_bands(horizon, step_minutes):
    """Primary 0-4 h / 4-16 h bands, plus finer diagnostic slices."""
    four_hour_steps = max(1, int(round(240 / max(int(step_minutes), 1))))
    requested = [
        ("0_4h", 1, min(four_hour_steps, horizon), True),
        ("4_16h", four_hour_steps + 1, horizon, True),
        ("step_1_12", 1, min(12, horizon), False),
        ("step_13_24", 13, min(24, horizon), False),
        ("step_25_48", 25, min(48, horizon), False),
        (f"step_49_{horizon}", 49, horizon, False),
    ]
    bands = []
    for name, start, end, primary in requested:
        if start > horizon or start > end:
            continue
        bands.append((name, start, min(end, horizon), primary))
    return bands


def _horizon_segment_table(
    pred, true, persistence, step_minutes, rated_power
):
    """Report 0-4 h and 4-16 h as primary bands (10-min SDWPF steps)."""
    horizon = pred.shape[1]
    rows = []
    eps = np.finfo(float).eps
    for name, start, end, primary in _horizon_task_bands(horizon, step_minutes):
        model_values = forecast_metrics(
            pred[:, start - 1 : end],
            true[:, start - 1 : end],
            rated_power=rated_power,
        )
        row = {
            "segment": name,
            "primary_task": bool(primary),
            "start_step": start,
            "end_step": end,
            "start_minutes": start * step_minutes,
            "end_minutes": end * step_minutes,
            "point_count": model_values["point_count"],
            "mae": model_values["mae"],
            "rmse": model_values["rmse"],
            "r2": model_values["r2"],
            "bias_mbe": model_values["bias_mbe"],
            "smape_pct": model_values["smape_pct"],
        }
        if persistence is not None:
            baseline = forecast_metrics(
                persistence[:, start - 1 : end],
                true[:, start - 1 : end],
                rated_power=rated_power,
            )
            row.update(
                {
                    "persistence_mae": baseline["mae"],
                    "persistence_rmse": baseline["rmse"],
                    "persistence_r2": baseline["r2"],
                    "mae_skill_vs_persistence_pct": 100.0
                    * (1.0 - model_values["mae"] / max(baseline["mae"], eps)),
                    "rmse_skill_vs_persistence_pct": 100.0
                    * (1.0 - model_values["rmse"] / max(baseline["rmse"], eps)),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _per_turbine_table(metadata, pred, true, rated_power):
    if "TurbID" not in metadata:
        return None
    rows = []
    turbine_values = metadata["TurbID"].to_numpy()
    for turbine in np.unique(turbine_values):
        keep = turbine_values == turbine
        values = forecast_metrics(pred[keep], true[keep], rated_power=rated_power)
        rows.append(
            {
                "TurbID": int(turbine),
                "window_count": int(keep.sum()),
                "mae": values["mae"],
                "rmse": values["rmse"],
                "smape_pct": values["smape_pct"],
                "wape_pct": values["wape_pct"],
                "r2": values["r2"],
                "bias_mbe": values["bias_mbe"],
            }
        )
    return pd.DataFrame(rows).sort_values("TurbID").reset_index(drop=True)


def _power_bin_table(pred, true, bins=10):
    p = pred.reshape(-1)
    y = true.reshape(-1)
    edges = np.unique(np.quantile(y, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return None
    bucket = np.digitize(y, edges[1:-1], right=True)
    rows = []
    for index in range(len(edges) - 1):
        keep = bucket == index
        if not keep.any():
            continue
        values = forecast_metrics(p[keep], y[keep])
        rows.append(
            {
                "bin": index + 1,
                "true_min": float(edges[index]),
                "true_max": float(edges[index + 1]),
                "point_count": int(keep.sum()),
                "mae": values["mae"],
                "rmse": values["rmse"],
                "bias_mbe": values["bias_mbe"],
            }
        )
    return pd.DataFrame(rows)


def _lead_time_bands(horizon):
    """Return the standard short-to-long lead-time bands in 1-based steps."""
    requested = [
        (1, 12, "Short", "tab:blue"),
        (13, 24, "Mid-short", "tab:orange"),
        (25, 48, "Medium", "tab:green"),
        (49, horizon, "Long", "tab:purple"),
    ]
    return [
        (start, min(end, horizon), name, color)
        for start, end, name, color in requested
        if start <= horizon and start <= min(end, horizon)
    ]


def _shade_lead_time_bands(axis, horizon, show_labels=False):
    """Shade the four forecast ranges without obscuring the plotted lines."""
    for start, end, name, color in _lead_time_bands(horizon):
        axis.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.055, zorder=0)
        if show_labels:
            axis.text(
                (start + end) / 2.0,
                0.985,
                name,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
                color=color,
                fontweight="bold",
            )


def _plot_horizon(table, output_dir):
    """Plot continuous 1-horizon model/baseline error curves."""
    if table is None or table.empty:
        return

    steps = table["horizon_step"].to_numpy(dtype=float)
    horizon = int(steps.max())
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)

    model_style = {
        "color": "tab:blue",
        "linewidth": 2.2,
        "label": "PromptTimeDART",
    }
    baseline_style = {
        "color": "tab:orange",
        "linewidth": 1.9,
        "linestyle": "--",
        "label": "Persistence",
    }

    axes[0].plot(steps, table["mae"], **model_style)
    if "persistence_mae" in table:
        axes[0].plot(steps, table["persistence_mae"], **baseline_style)
    axes[0].set_ylabel("MAE (power units)")
    axes[0].set_title("MAE by forecast step")
    axes[0].legend(ncol=2)

    axes[1].plot(steps, table["rmse"], **model_style)
    if "persistence_rmse" in table:
        axes[1].plot(steps, table["persistence_rmse"], **baseline_style)
    axes[1].set_ylabel("RMSE (power units)")
    axes[1].set_title("RMSE by forecast step")
    axes[1].legend(ncol=2)

    if {
        "mae_skill_vs_persistence_pct",
        "rmse_skill_vs_persistence_pct",
    }.issubset(table.columns):
        axes[2].plot(
            steps,
            table["mae_skill_vs_persistence_pct"],
            color="tab:green",
            linewidth=2.0,
            label="MAE skill",
        )
        axes[2].plot(
            steps,
            table["rmse_skill_vs_persistence_pct"],
            color="tab:red",
            linewidth=2.0,
            label="RMSE skill",
        )
        axes[2].set_ylabel("Skill vs persistence (%)")
        axes[2].set_title("Positive values mean the model beats persistence")
        axes[2].legend(ncol=2)
    else:
        axes[2].plot(
            steps,
            table["r2"],
            color="tab:green",
            linewidth=2.0,
            label="R2",
        )
        axes[2].set_ylabel("R2")
        axes[2].set_title("R2 by forecast step")
        axes[2].legend()
    axes[2].axhline(0.0, color="black", linewidth=1.0)

    for index, axis in enumerate(axes):
        _shade_lead_time_bands(axis, horizon, show_labels=index == 0)
        axis.grid(alpha=0.22)
        axis.set_xlim(0.5, horizon + 0.5)

    major_ticks = np.unique(
        np.clip(
            np.concatenate(([1], np.arange(12, horizon + 1, 12), [horizon])),
            1,
            horizon,
        )
    )
    axes[2].set_xticks(major_ticks)
    axes[2].set_xlabel("Forecast step (10 minutes per step)")
    fig.suptitle("PromptTimeDART vs Persistence across the forecast horizon")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "error_by_horizon.png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_horizon_segments(table, output_dir):
    """Compare model and persistence across the standard lead-time bands."""
    if table is None or table.empty:
        return

    band_names = ["0-4h", "4-16h"]
    labels = []
    for index, row in table.reset_index(drop=True).iterrows():
        name = row["segment"] if "segment" in table else (
            band_names[index] if index < len(band_names) else f"Band {index + 1}"
        )
        labels.append(
            f"{name}\nstep {int(row['start_step'])}-{int(row['end_step'])}\n"
            f"{int(row['start_minutes'])}-{int(row['end_minutes'])} min"
        )

    x = np.arange(len(table), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    def draw_error_bars(axis, metric, title):
        model = table[metric].to_numpy(dtype=float)
        model_bars = axis.bar(
            x - width / 2,
            model,
            width,
            label="PromptTimeDART",
        )
        bars = [model_bars]
        baseline_key = f"persistence_{metric}"
        if baseline_key in table:
            baseline = table[baseline_key].to_numpy(dtype=float)
            baseline_bars = axis.bar(
                x + width / 2,
                baseline,
                width,
                label="Persistence",
            )
            bars.append(baseline_bars)
        for group in bars:
            axis.bar_label(group, fmt="%.1f", padding=3, fontsize=8)
        axis.set_ylabel(f"{metric.upper()} (power units)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()

    draw_error_bars(axes[0], "mae", "MAE: model vs persistence")
    draw_error_bars(axes[1], "rmse", "RMSE: model vs persistence")

    if "rmse_skill_vs_persistence_pct" in table:
        skill = table["rmse_skill_vs_persistence_pct"].to_numpy(dtype=float)
        colors = np.where(skill >= 0.0, "tab:green", "tab:red")
        skill_bars = axes[2].bar(x, skill, width=0.58, color=colors)
        axes[2].bar_label(skill_bars, fmt="%+.1f%%", padding=3, fontsize=9)
        axes[2].set_ylabel("RMSE skill vs persistence (%)")
        axes[2].set_title("Positive values mean the model beats persistence")
    else:
        r2_bars = axes[2].bar(x, table["r2"].to_numpy(dtype=float), width=0.58)
        axes[2].bar_label(r2_bars, fmt="%.3f", padding=3, fontsize=9)
        axes[2].set_ylabel("R2")
        axes[2].set_title("R2 by lead-time segment")
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].set_xticks(x, labels)

    fig.suptitle("Forecast performance by lead-time segment")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "error_by_horizon_segment.png"),
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_examples(
    pred, true, metadata, output_dir, persistence=None, count=6
):
    count = min(int(count), len(pred))
    if count <= 0:
        return
    window_rmse = np.sqrt(np.mean((pred - true) ** 2, axis=1))
    order = np.argsort(window_rmse)
    positions = np.linspace(0, len(order) - 1, count).round().astype(int)
    selected = order[positions]
    fig, axes = plt.subplots(count, 1, figsize=(12, 2.8 * count), squeeze=False)
    horizon = pred.shape[1]
    steps = np.arange(1, horizon + 1)
    for axis_index, (axis, sample_index) in enumerate(zip(axes[:, 0], selected)):
        _shade_lead_time_bands(axis, horizon, show_labels=axis_index == 0)
        axis.plot(
            steps,
            true[sample_index],
            label="Truth",
            color="black",
            linewidth=2.1,
        )
        axis.plot(
            steps,
            pred[sample_index],
            label="PromptTimeDART",
            color="tab:blue",
            linewidth=1.8,
        )
        if persistence is not None:
            axis.plot(
                steps,
                persistence[sample_index],
                "--",
                label="Persistence",
                color="tab:orange",
                linewidth=1.3,
            )
        details = [f"window={sample_index}", f"RMSE={window_rmse[sample_index]:.3f}"]
        if "TurbID" in metadata:
            details.append(f"TurbID={metadata.iloc[sample_index]['TurbID']}")
        if "forecast_start" in metadata:
            details.append(str(metadata.iloc[sample_index]["forecast_start"]))
        axis.set_title(" | ".join(details), fontsize=10)
        axis.set_ylabel("Power")
        axis.grid(alpha=0.2)
        axis.set_xlim(0.5, horizon + 0.5)
    axes[0, 0].legend(ncol=3)
    major_ticks = np.unique(
        np.clip(
            np.concatenate(([1], np.arange(12, horizon + 1, 12), [horizon])),
            1,
            horizon,
        )
    )
    axes[-1, 0].set_xticks(major_ticks)
    axes[-1, 0].set_xlabel("Forecast step (10 minutes per step)")
    fig.suptitle("Representative forecast windows by error quantile", y=1.002)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "forecast_examples.png"), dpi=180)
    plt.close(fig)


def _continuous_forecast_trace(
    pred,
    true,
    metadata,
    *,
    persistence=None,
    step_minutes=10,
    max_points=150,
    turbine_id=None,
    requested_start=None,
):
    """Build a deterministic, non-overlapping continuous trace for one turbine."""

    horizon = int(pred.shape[1])
    max_points = max(horizon, int(max_points))
    if {"TurbID", "forecast_start"}.issubset(metadata.columns):
        candidates = metadata[["window_index", "TurbID", "forecast_start"]].copy()
        candidates["forecast_start"] = pd.to_datetime(candidates["forecast_start"])
        if turbine_id is not None:
            candidates = candidates[candidates["TurbID"] == int(turbine_id)]
            if candidates.empty:
                raise ValueError(f"Requested forecast plot TurbID={turbine_id} is absent")
        requested = pd.Timestamp(requested_start) if requested_start else None
        expected = pd.Timedelta(minutes=int(step_minutes) * horizon)
        blocks = []
        for current_turbine, group in candidates.groupby("TurbID", sort=True):
            group = group.sort_values("forecast_start", kind="mergesort")
            indices = group.index.to_numpy(dtype=np.int64)
            starts = group["forecast_start"].to_numpy()
            split_points = np.flatnonzero(
                np.diff(starts).astype("timedelta64[m]")
                != np.timedelta64(int(expected.total_seconds() // 60), "m")
            ) + 1
            for block in np.split(indices, split_points):
                if len(block):
                    blocks.append((int(current_turbine), block))
        if not blocks:
            return None, None
        if requested is not None:
            trimmed = []
            for current_turbine, block in blocks:
                times = candidates.loc[block, "forecast_start"]
                keep = np.flatnonzero(times.to_numpy() >= requested.to_datetime64())
                if len(keep):
                    trimmed.append((current_turbine, block[keep[0] :]))
            if not trimmed:
                raise ValueError(
                    f"No continuous prediction trace starts at or after {requested.isoformat()}"
                )
            blocks = trimmed
            blocks.sort(
                key=lambda item: candidates.loc[item[1][0], "forecast_start"]
            )
        else:
            blocks.sort(
                key=lambda item: (
                    -len(item[1]),
                    item[0],
                    candidates.loc[item[1][0], "forecast_start"],
                )
            )
        selected_turbine, selected_rows = blocks[0]
        window_indices = metadata.loc[selected_rows, "window_index"].to_numpy(dtype=np.int64)
        window_starts = pd.to_datetime(
            metadata.loc[selected_rows, "forecast_start"]
        ).to_numpy()
    else:
        selected_turbine = None
        window_indices = np.asarray([0], dtype=np.int64)
        window_starts = np.asarray([np.datetime64("NaT")])

    records = []
    for window_index, forecast_start in zip(window_indices, window_starts):
        for horizon_index in range(horizon):
            if len(records) >= max_points:
                break
            timestamp = (
                pd.Timestamp(forecast_start)
                + pd.Timedelta(minutes=int(step_minutes) * horizon_index)
                if not pd.isna(forecast_start)
                else pd.NaT
            )
            row = {
                "point_index": len(records),
                "timestamp": timestamp,
                "TurbID": selected_turbine,
                "source_window": int(window_index),
                "horizon_step": horizon_index + 1,
                "truth_kw": float(true[window_index, horizon_index]),
                "prediction_kw": float(pred[window_index, horizon_index]),
            }
            if persistence is not None:
                row["persistence_kw"] = float(
                    persistence[window_index, horizon_index]
                )
            records.append(row)
        if len(records) >= max_points:
            break
    if not records:
        return None, None
    trace = pd.DataFrame.from_records(records)
    selection = {
        "selection_rule": (
            "requested turbine/start, then earliest continuous block"
            if turbine_id is not None or requested_start
            else "longest continuous block; tie-break by TurbID and start time"
        ),
        "TurbID": selected_turbine,
        "forecast_start": (
            trace["timestamp"].iloc[0].isoformat()
            if pd.notna(trace["timestamp"].iloc[0])
            else None
        ),
        "point_count": int(len(trace)),
        "step_minutes": int(step_minutes),
    }
    return trace, selection


def _plot_continuous_forecast(
    pred,
    true,
    metadata,
    output_dir,
    *,
    persistence=None,
    step_minutes=10,
    max_points=150,
    turbine_id=None,
    requested_start=None,
    model_name="Forecast model",
):
    """Save a paper-ready truth/prediction trace with deterministic zooms."""

    trace, selection = _continuous_forecast_trace(
        pred,
        true,
        metadata,
        persistence=persistence,
        step_minutes=step_minutes,
        max_points=max_points,
        turbine_id=turbine_id,
        requested_start=requested_start,
    )
    if trace is None or trace.empty:
        return
    trace.to_csv(os.path.join(output_dir, "forecast_trace.csv"), index=False)
    with open(
        os.path.join(output_dir, "forecast_trace_selection.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(_json_safe(selection), handle, ensure_ascii=False, indent=2)

    x = trace["point_index"].to_numpy()
    fig, axis = plt.subplots(figsize=(14, 6.4))
    line_specs = [
        ("truth_kw", "True", "black", "--", 1.8),
        ("prediction_kw", str(model_name), "tab:blue", "-", 1.6),
    ]
    if "persistence_kw" in trace:
        line_specs.append(("persistence_kw", "Persistence", "tab:orange", "-", 1.25))
    for column, label, color, style, width in line_specs:
        axis.plot(
            x,
            trace[column],
            label=label,
            color=color,
            linestyle=style,
            linewidth=width,
        )
    axis.set_xlabel(f"Time step ({int(step_minutes)} min)")
    axis.set_ylabel("Wind power / kW")
    title_parts = ["Continuous wind-power forecast"]
    if selection["TurbID"] is not None:
        title_parts.append(f"TurbID={selection['TurbID']}")
    if selection["forecast_start"] is not None:
        title_parts.append(f"start={selection['forecast_start']}")
    axis.set_title(" | ".join(title_parts), fontsize=11)
    axis.grid(alpha=0.18)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=len(line_specs))
    axis.margins(x=0.01)

    point_count = len(trace)
    if point_count >= 24:
        zoom_ranges = [
            (max(0, int(point_count * 0.12)), max(12, int(point_count * 0.30))),
            (max(0, int(point_count * 0.56)), max(12, int(point_count * 0.76))),
        ]
        placements = [(0.10, 0.55, 0.27, 0.36), (0.60, 0.55, 0.27, 0.36)]
        for (start, end), placement in zip(zoom_ranges, placements):
            end = min(point_count - 1, max(start + 2, end))
            inset = axis.inset_axes(placement)
            for column, _, color, style, width in line_specs:
                inset.plot(
                    x[start : end + 1],
                    trace[column].to_numpy()[start : end + 1],
                    color=color,
                    linestyle=style,
                    linewidth=max(1.0, width - 0.2),
                )
            inset.set_xlim(start, end)
            inset.grid(alpha=0.12)
            inset.tick_params(labelsize=7)
            axis.indicate_inset_zoom(inset, edgecolor="0.35", alpha=0.65)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "forecast_trace.png"),
        dpi=240,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(output_dir, "forecast_trace.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_scatter(pred, true, output_dir, max_points=150000):
    p = pred.reshape(-1)
    y = true.reshape(-1)
    if len(p) > max_points:
        rng = np.random.default_rng(2024)
        selected = rng.choice(len(p), size=max_points, replace=False)
        p, y = p[selected], y[selected]
    low = float(min(p.min(), y.min()))
    high = float(max(p.max(), y.max()))
    fig, axis = plt.subplots(figsize=(7.5, 7))
    image = axis.hexbin(y, p, gridsize=80, bins="log", mincnt=1, cmap="viridis")
    axis.plot([low, high], [low, high], "r--", linewidth=1.5, label="Ideal")
    axis.set_xlabel("True power")
    axis.set_ylabel("Predicted power")
    axis.set_title("Prediction versus truth density")
    axis.legend()
    fig.colorbar(image, ax=axis, label="log10(count)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "prediction_scatter.png"), dpi=180)
    plt.close(fig)


def _plot_residuals(pred, true, output_dir, max_points=200000):
    residual = (pred - true).reshape(-1)
    if len(residual) > max_points:
        rng = np.random.default_rng(2024)
        residual = residual[rng.choice(len(residual), size=max_points, replace=False)]
    low, high = np.percentile(residual, [1, 99])
    shown = residual[(residual >= low) & (residual <= high)]
    abs_sorted = np.sort(np.abs(residual))
    cumulative = np.arange(1, len(abs_sorted) + 1) / len(abs_sorted) * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(shown, bins=80, alpha=0.85)
    axes[0].axvline(0.0, color="black", linewidth=1)
    axes[0].set_title("Residual distribution (1st-99th percentile)")
    axes[0].set_xlabel("Prediction - truth")
    axes[0].set_ylabel("Count")
    axes[1].plot(abs_sorted, cumulative, linewidth=2)
    axes[1].set_title("Absolute-error cumulative distribution")
    axes[1].set_xlabel("Absolute error")
    axes[1].set_ylabel("Points within error (%)")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "residual_distribution.png"), dpi=180)
    plt.close(fig)


def _plot_per_turbine(table, output_dir, top_n=20):
    if table is None or table.empty:
        return
    worst = table.nlargest(min(top_n, len(table)), "mae").sort_values("mae")
    fig, axis = plt.subplots(figsize=(9, max(5, 0.35 * len(worst))))
    axis.barh(worst["TurbID"].astype(str), worst["mae"])
    axis.set_xlabel("MAE")
    axis.set_ylabel("TurbID")
    axis.set_title(f"Worst {len(worst)} turbines by MAE")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "worst_turbines_mae.png"), dpi=180)
    plt.close(fig)


def _plot_power_bins(table, output_dir):
    if table is None or table.empty:
        return
    labels = [f"Q{value}" for value in table["bin"]]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(labels, table["mae"], marker="o", label="MAE")
    axes[0].plot(labels, table["rmse"], marker="o", label="RMSE")
    axes[0].set_ylabel("Power error")
    axes[0].set_title("Error by true-power quantile")
    axes[0].legend()
    axes[1].bar(labels, table["bias_mbe"])
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("Bias")
    axes[1].set_xlabel("True-power quantile (low to high)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "error_by_power_bin.png"), dpi=180)
    plt.close(fig)


def save_training_history(records, output_dir, title="Training history"):
    """Save loss/LR history and its plot after every epoch."""
    if not records:
        return
    os.makedirs(output_dir, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    has_validation_metrics = {"val_mse", "val_mae"}.issubset(table.columns)
    panel_count = 3 if has_validation_metrics else 2
    fig, axes = plt.subplots(
        panel_count, 1, figsize=(9, 4 * panel_count), sharex=True
    )
    axes[0].plot(table["epoch"], table["train_loss"], marker="o", label="Train")
    axes[0].plot(table["epoch"], table["val_loss"], marker="o", label="Validation")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    lr_axis = 1
    if has_validation_metrics:
        axes[1].plot(table["epoch"], table["val_mse"], marker="o", label="Val MSE")
        axes[1].plot(table["epoch"], table["val_mae"], marker="o", label="Val MAE")
        axes[1].set_ylabel("Validation metric")
        axes[1].legend()
        axes[1].grid(alpha=0.2)
        lr_axis = 2
    axes[lr_axis].plot(table["epoch"], table["learning_rate"], marker="o")
    axes[lr_axis].set_xlabel("Epoch")
    axes[lr_axis].set_ylabel("Learning rate")
    axes[lr_axis].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=180)
    plt.close(fig)


def build_forecast_report(
    preds_scaled,
    trues_scaled,
    dataset,
    output_dir,
    last_observation_scaled=None,
    step_minutes=10,
    rated_power=None,
    trace_points=150,
    trace_turbine_id=None,
    trace_start=None,
    model_name="Forecast model",
):
    """Save original-scale metrics, breakdown tables, arrays, and PNG charts."""
    os.makedirs(output_dir, exist_ok=True)
    pred_scaled = _target_2d(preds_scaled)
    true_scaled = _target_2d(trues_scaled)
    if pred_scaled.shape != true_scaled.shape:
        raise ValueError(f"Prediction/target shapes differ: {pred_scaled.shape} != {true_scaled.shape}")

    pred = inverse_transform_target(dataset, pred_scaled)
    true = inverse_transform_target(dataset, true_scaled)
    if rated_power is not None and float(rated_power) > 0:
        pred = np.clip(pred, 0.0, float(rated_power))
        true = np.clip(true, 0.0, float(rated_power))
    normalized_metrics = forecast_metrics(pred_scaled, true_scaled)
    original_metrics = forecast_metrics(pred, true, rated_power=rated_power)

    persistence = None
    if last_observation_scaled is not None:
        last_scaled = np.asarray(last_observation_scaled, dtype=np.float64).reshape(-1, 1)
        if len(last_scaled) == len(pred):
            last = inverse_transform_target(dataset, last_scaled)
            if rated_power is not None and float(rated_power) > 0:
                last = np.clip(last, 0.0, float(rated_power))
            persistence = np.repeat(last, pred.shape[1], axis=1)
            _add_persistence_metrics(original_metrics, pred, true, persistence)

    available_mask = None
    if hasattr(dataset, "target_available_mask"):
        mask = np.asarray(dataset.target_available_mask(), dtype=bool)
        if mask.shape == pred.shape:
            available_mask = mask
            original_metrics["available_point_pct"] = float(mask.mean() * 100.0)
            if mask.any():
                available_metrics = forecast_metrics(
                    pred[mask], true[mask], rated_power=rated_power
                )
                original_metrics["available_mae"] = available_metrics["mae"]
                original_metrics["available_rmse"] = available_metrics["rmse"]
                original_metrics["available_r2"] = available_metrics["r2"]
                if persistence is not None:
                    available_base = forecast_metrics(
                        persistence[mask], true[mask], rated_power=rated_power
                    )
                    eps = np.finfo(float).eps
                    original_metrics["available_mae_skill_vs_persistence_pct"] = (
                        100.0 * (1.0 - available_metrics["mae"] / max(available_base["mae"], eps))
                    )
                    original_metrics["available_rmse_skill_vs_persistence_pct"] = (
                        100.0 * (1.0 - available_metrics["rmse"] / max(available_base["rmse"], eps))
                    )

    metadata = _window_metadata(dataset, len(pred))
    horizon = _horizon_table(
        pred,
        true,
        int(step_minutes),
        rated_power,
        persistence=persistence,
    )
    horizon.to_csv(os.path.join(output_dir, "metrics_by_horizon.csv"), index=False)

    segments = _horizon_segment_table(
        pred,
        true,
        persistence,
        int(step_minutes),
        rated_power,
    )
    segments.to_csv(
        os.path.join(output_dir, "metrics_by_horizon_segment.csv"), index=False
    )
    for _, row in segments.iterrows():
        if not bool(row.get("primary_task", False)):
            continue
        key = str(row["segment"])
        original_metrics[f"{key}_mae"] = float(row["mae"])
        original_metrics[f"{key}_rmse"] = float(row["rmse"])
        original_metrics[f"{key}_r2"] = float(row["r2"])
        if "mae_skill_vs_persistence_pct" in row:
            original_metrics[f"{key}_mae_skill_vs_persistence_pct"] = float(
                row["mae_skill_vs_persistence_pct"]
            )
            original_metrics[f"{key}_rmse_skill_vs_persistence_pct"] = float(
                row["rmse_skill_vs_persistence_pct"]
            )

    _save_summary(output_dir, normalized_metrics, original_metrics)

    turbine = _per_turbine_table(metadata, pred, true, rated_power)
    if turbine is not None:
        turbine.to_csv(os.path.join(output_dir, "metrics_by_turbine.csv"), index=False)

    power_bins = _power_bin_table(pred, true)
    if power_bins is not None:
        power_bins.to_csv(os.path.join(output_dir, "metrics_by_power_bin.csv"), index=False)

    window_rmse = np.sqrt(np.mean((pred - true) ** 2, axis=1))
    window_mae = np.mean(np.abs(pred - true), axis=1)
    metadata = metadata.copy()
    metadata["mae"] = window_mae
    metadata["rmse"] = window_rmse
    metadata.to_csv(os.path.join(output_dir, "metrics_by_window.csv"), index=False)

    arrays = {
        "prediction_original": pred.astype(np.float32),
        "truth_original": true.astype(np.float32),
        "prediction_normalized": pred_scaled.astype(np.float32),
        "truth_normalized": true_scaled.astype(np.float32),
    }
    if persistence is not None:
        arrays["persistence_original"] = persistence.astype(np.float32)
        arrays["correction_original"] = (pred - persistence).astype(np.float32)
    if available_mask is not None:
        arrays["available_mask"] = available_mask.astype(np.uint8)
    np.savez_compressed(os.path.join(output_dir, "predictions.npz"), **arrays)

    _plot_horizon(horizon, output_dir)
    _plot_horizon_segments(segments, output_dir)
    _plot_examples(
        pred,
        true,
        metadata,
        output_dir,
        persistence=persistence,
    )
    _plot_continuous_forecast(
        pred,
        true,
        metadata,
        output_dir,
        persistence=persistence,
        step_minutes=int(step_minutes),
        max_points=int(trace_points),
        turbine_id=trace_turbine_id,
        requested_start=trace_start,
        model_name=model_name,
    )
    _plot_scatter(pred, true, output_dir)
    _plot_residuals(pred, true, output_dir)
    _plot_per_turbine(turbine, output_dir)
    _plot_power_bins(power_bins, output_dir)
    return original_metrics
