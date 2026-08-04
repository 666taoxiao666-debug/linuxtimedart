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
    one output channel after it was fitted on all input channels. Applying the
    fitted target mean and scale directly is the correct inverse operation.
    """
    values = np.asarray(values, dtype=np.float64)
    if not getattr(dataset, "scale", True):
        return values.copy()

    scaler = getattr(dataset, "scaler", None)
    columns = list(getattr(dataset, "feature_columns", []))
    target = getattr(dataset, "target", None)
    target_index = (
        columns.index(target)
        if target in columns
        else -1
    )

    if (
        scaler is not None
        and hasattr(scaler, "mean_")
        and hasattr(scaler, "scale_")
    ):
        mean = np.asarray(
            scaler.mean_,
            dtype=np.float64,
        ).reshape(-1)[target_index]

        scale = np.asarray(
            scaler.scale_,
            dtype=np.float64,
        ).reshape(-1)[target_index]

        return values * scale + mean

    if (
        scaler is not None
        and hasattr(scaler, "mean")
        and hasattr(scaler, "std")
    ):
        mean = np.asarray(
            scaler.mean,
            dtype=np.float64,
        ).reshape(-1)[target_index]

        scale = np.asarray(
            scaler.std,
            dtype=np.float64,
        ).reshape(-1)[target_index]

        return values * scale + mean

    # This fallback is valid for genuinely
    # single-variable datasets.
    if hasattr(dataset, "inverse_transform"):
        restored = dataset.inverse_transform(
            values.reshape(-1, 1)
        )
        return np.asarray(restored).reshape(
            values.shape
        )

    raise AttributeError(
        "Dataset has no usable scaler or "
        "inverse_transform method"
    )


def _window_metadata(
    dataset,
    window_count,
):
    starts = getattr(
        dataset,
        "window_starts",
        None,
    )
    dates = getattr(
        dataset,
        "dates",
        None,
    )
    turbines = getattr(
        dataset,
        "turbines",
        None,
    )
    seq_len = getattr(
        dataset,
        "seq_len",
        None,
    )

    if (
        starts is None
        or dates is None
        or seq_len is None
    ):
        return pd.DataFrame(
            {
                "window_index": np.arange(
                    window_count
                )
            }
        )

    starts = np.asarray(
        starts,
        dtype=np.int64,
    )

    if len(starts) != window_count:
        return pd.DataFrame(
            {
                "window_index": np.arange(
                    window_count
                )
            }
        )

    target_starts = starts + int(seq_len)

    result = pd.DataFrame(
        {
            "window_index": np.arange(
                window_count
            ),
            "forecast_start": pd.to_datetime(
                np.asarray(dates)[target_starts]
            ),
        }
    )

    if turbines is not None:
        result["TurbID"] = np.asarray(
            turbines
        )[target_starts]

    return result


def _json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(
        value,
        (np.floating, float),
    ):
        return (
            float(value)
            if np.isfinite(value)
            else None
        )

    return value


def _add_persistence_metrics(
    result,
    pred,
    true,
    persistence,
):
    persistence_metrics = forecast_metrics(
        persistence,
        true,
    )

    eps = np.finfo(float).eps

    result["persistence_mae"] = (
        persistence_metrics["mae"]
    )
    result["persistence_rmse"] = (
        persistence_metrics["rmse"]
    )

    result[
        "mae_skill_vs_persistence_pct"
    ] = 100.0 * (
        1.0
        - result["mae"]
        / max(
            persistence_metrics["mae"],
            eps,
        )
    )

    result[
        "rmse_skill_vs_persistence_pct"
    ] = 100.0 * (
        1.0
        - result["rmse"]
        / max(
            persistence_metrics["rmse"],
            eps,
        )
    )

    result[
        "mae_ratio_to_persistence"
    ] = (
        result["mae"]
        / max(
            persistence_metrics["mae"],
            eps,
        )
    )

    if (
        "direction_accuracy_pct"
        in persistence_metrics
    ):
        result[
            "persistence_direction_accuracy_pct"
        ] = persistence_metrics[
            "direction_accuracy_pct"
        ]


def _save_summary(
    output_dir,
    normalized_metrics,
    original_metrics,
):
    with open(
        os.path.join(
            output_dir,
            "metrics.json",
        ),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            _json_safe(
                {
                    "normalized": (
                        normalized_metrics
                    ),
                    "original": (
                        original_metrics
                    ),
                }
            ),
            handle,
            ensure_ascii=False,
            indent=2,
        )

    rows = []

    for scale_name, values in (
        (
            "normalized",
            normalized_metrics,
        ),
        (
            "original",
            original_metrics,
        ),
    ):
        for name, value in values.items():
            rows.append(
                {
                    "scale": scale_name,
                    "metric": name,
                    "value": value,
                }
            )

    pd.DataFrame(rows).to_csv(
        os.path.join(
            output_dir,
            "metrics_summary.csv",
        ),
        index=False,
    )

    lines = [
        (
            "Forecast metrics "
            "(primary values are in "
            "original power units)"
        ),
        "=" * 72,
    ]

    for name, value in original_metrics.items():
        if isinstance(
            value,
            (int, np.integer),
        ):
            lines.append(
                f"{name:36s}: {int(value)}"
            )
        elif (
            value is None
            or not np.isfinite(value)
        ):
            lines.append(
                f"{name:36s}: NA"
            )
        else:
            lines.append(
                f"{name:36s}: "
                f"{float(value):.8f}"
            )

    lines.extend(
        [
            "",
            "Normalized-space metrics",
            "-" * 72,
        ]
    )

    for name, value in normalized_metrics.items():
        if isinstance(
            value,
            (int, np.integer),
        ):
            lines.append(
                f"{name:36s}: {int(value)}"
            )
        elif (
            value is None
            or not np.isfinite(value)
        ):
            lines.append(
                f"{name:36s}: NA"
            )
        else:
            lines.append(
                f"{name:36s}: "
                f"{float(value):.8f}"
            )

    with open(
        os.path.join(
            output_dir,
            "score.txt",
        ),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "\n".join(lines) + "\n"
        )


def _horizon_table(
    pred,
    true,
    step_minutes,
    rated_power,
    persistence=None,
):
    rows = []

    for index in range(pred.shape[1]):
        values = forecast_metrics(
            pred[:, index],
            true[:, index],
            rated_power=rated_power,
        )

        row = {
            "horizon_step": index + 1,
            "lead_minutes": (
                (index + 1) * step_minutes
            ),
            "mae": values["mae"],
            "rmse": values["rmse"],
            "mape_masked_pct": (
                values["mape_masked_pct"]
            ),
            "smape_pct": values["smape_pct"],
            "wape_pct": values["wape_pct"],
            "r2": values["r2"],
            "pearson_r": values["pearson_r"],
            "bias_mbe": values["bias_mbe"],
        }

        if persistence is not None:
            baseline = forecast_metrics(
                persistence[:, index],
                true[:, index],
                rated_power=rated_power,
            )

            eps = np.finfo(float).eps

            row.update(
                {
                    "persistence_mae": (
                        baseline["mae"]
                    ),
                    "persistence_rmse": (
                        baseline["rmse"]
                    ),
                    "persistence_r2": (
                        baseline["r2"]
                    ),
                    "mae_skill_vs_persistence_pct": (
                        100.0
                        * (
                            1.0
                            - values["mae"]
                            / max(
                                baseline["mae"],
                                eps,
                            )
                        )
                    ),
                    "rmse_skill_vs_persistence_pct": (
                        100.0
                        * (
                            1.0
                            - values["rmse"]
                            / max(
                                baseline["rmse"],
                                eps,
                            )
                        )
                    ),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def _horizon_segment_table(
    pred,
    true,
    persistence,
    step_minutes,
    rated_power,
):
    """Aggregate standard forecast-lead bands."""
    horizon = pred.shape[1]

    requested = [
        (1, 12),
        (13, 24),
        (25, 48),
        (49, horizon),
    ]

    rows = []
    eps = np.finfo(float).eps

    for start, end in requested:
        if start > horizon:
            continue

        end = min(
            end,
            horizon,
        )

        model_values = forecast_metrics(
            pred[:, start - 1:end],
            true[:, start - 1:end],
            rated_power=rated_power,
        )

        row = {
            "segment": (
                f"step_{start}_{end}"
            ),
            "start_step": start,
            "end_step": end,
            "start_minutes": (
                start * step_minutes
            ),
            "end_minutes": (
                end * step_minutes
            ),
            "point_count": (
                model_values["point_count"]
            ),
            "mae": model_values["mae"],
            "rmse": model_values["rmse"],
            "r2": model_values["r2"],
            "bias_mbe": (
                model_values["bias_mbe"]
            ),
            "smape_pct": (
                model_values["smape_pct"]
            ),
        }

        if persistence is not None:
            baseline = forecast_metrics(
                persistence[:, start - 1:end],
                true[:, start - 1:end],
                rated_power=rated_power,
            )

            row.update(
                {
                    "persistence_mae": (
                        baseline["mae"]
                    ),
                    "persistence_rmse": (
                        baseline["rmse"]
                    ),
                    "persistence_r2": (
                        baseline["r2"]
                    ),
                    "mae_skill_vs_persistence_pct": (
                        100.0
                        * (
                            1.0
                            - model_values["mae"]
                            / max(
                                baseline["mae"],
                                eps,
                            )
                        )
                    ),
                    "rmse_skill_vs_persistence_pct": (
                        100.0
                        * (
                            1.0
                            - model_values["rmse"]
                            / max(
                                baseline["rmse"],
                                eps,
                            )
                        )
                    ),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def _per_turbine_table(
    metadata,
    pred,
    true,
    rated_power,
):
    if "TurbID" not in metadata:
        return None

    rows = []
    turbine_values = metadata[
        "TurbID"
    ].to_numpy()

    for turbine in np.unique(
        turbine_values
    ):
        keep = turbine_values == turbine

        values = forecast_metrics(
            pred[keep],
            true[keep],
            rated_power=rated_power,
        )

        rows.append(
            {
                "TurbID": int(turbine),
                "window_count": int(
                    keep.sum()
                ),
                "mae": values["mae"],
                "rmse": values["rmse"],
                "smape_pct": (
                    values["smape_pct"]
                ),
                "wape_pct": (
                    values["wape_pct"]
                ),
                "r2": values["r2"],
                "bias_mbe": (
                    values["bias_mbe"]
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("TurbID")
        .reset_index(drop=True)
    )


def _power_bin_table(
    pred,
    true,
    bins=10,
):
    p = pred.reshape(-1)
    y = true.reshape(-1)

    edges = np.unique(
        np.quantile(
            y,
            np.linspace(
                0.0,
                1.0,
                bins + 1,
            ),
        )
    )

    if len(edges) < 3:
        return None

    bucket = np.digitize(
        y,
        edges[1:-1],
        right=True,
    )

    rows = []

    for index in range(
        len(edges) - 1
    ):
        keep = bucket == index

        if not keep.any():
            continue

        values = forecast_metrics(
            p[keep],
            y[keep],
        )

        rows.append(
            {
                "bin": index + 1,
                "true_min": float(
                    edges[index]
                ),
                "true_max": float(
                    edges[index + 1]
                ),
                "point_count": int(
                    keep.sum()
                ),
                "mae": values["mae"],
                "rmse": values["rmse"],
                "bias_mbe": (
                    values["bias_mbe"]
                ),
            }
        )

    return pd.DataFrame(rows)


def _plot_horizon(
    table,
    output_dir,
):
    x = (
        table["lead_minutes"].to_numpy()
        / 60.0
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 11),
        sharex=True,
    )

    axes[0].plot(
        x,
        table["mae"],
        label="MAE",
        linewidth=2,
    )
    axes[0].plot(
        x,
        table["rmse"],
        label="RMSE",
        linewidth=2,
    )

    if "persistence_mae" in table:
        axes[0].plot(
            x,
            table["persistence_mae"],
            "--",
            label="Persistence MAE",
            linewidth=1.6,
        )
        axes[0].plot(
            x,
            table["persistence_rmse"],
            "--",
            label="Persistence RMSE",
            linewidth=1.6,
        )

    axes[0].set_ylabel("Power error")
    axes[0].legend()

    axes[1].plot(
        x,
        table["bias_mbe"],
        linewidth=2,
    )
    axes[1].axhline(
        0.0,
        color="black",
        linewidth=1,
    )
    axes[1].set_ylabel(
        "Bias (prediction - truth)"
    )

    if (
        "mae_skill_vs_persistence_pct"
        in table
    ):
        axes[2].plot(
            x,
            table[
                "mae_skill_vs_persistence_pct"
            ],
            linewidth=2,
        )
        axes[2].set_ylabel(
            "MAE skill vs persistence (%)"
        )
    else:
        axes[2].plot(
            x,
            table["r2"],
            linewidth=2,
        )
        axes[2].set_ylabel("R2")

    axes[2].axhline(
        0.0,
        color="black",
        linewidth=1,
    )
    axes[2].set_xlabel(
        "Forecast lead time (hours)"
    )

    fig.suptitle(
        "Error by forecast horizon"
    )
    fig.tight_layout()

    fig.savefig(
        os.path.join(
            output_dir,
            "error_by_horizon.png",
        ),
        dpi=180,
    )

    plt.close(fig)


def _plot_examples(
    pred,
    true,
    metadata,
    output_dir,
    persistence=None,
    count=6,
):
    count = min(
        int(count),
        len(pred),
    )

    if count <= 0:
        return

    window_rmse = np.sqrt(
        np.mean(
            (pred - true) ** 2,
            axis=1,
        )
    )

    order = np.argsort(window_rmse)

    positions = np.linspace(
        0,
        len(order) - 1,
        count,
    ).round().astype(int)

    selected = order[positions]

    fig, axes = plt.subplots(
        count,
        1,
        figsize=(12, 2.8 * count),
        squeeze=False,
    )

    for axis, sample_index in zip(
        axes[:, 0],
        selected,
    ):
        axis.plot(
            true[sample_index],
            label="Truth",
            linewidth=2,
        )
        axis.plot(
            pred[sample_index],
            label="Prediction",
            linewidth=1.6,
        )

        if persistence is not None:
            axis.plot(
                persistence[sample_index],
                "--",
                label="Persistence",
                linewidth=1.3,
            )

        details = [
            f"window={sample_index}",
            (
                f"RMSE="
                f"{window_rmse[sample_index]:.3f}"
            ),
        ]

        if "TurbID" in metadata:
            details.append(
                "TurbID="
                f"{metadata.iloc[sample_index]['TurbID']}"
            )

        if "forecast_start" in metadata:
            details.append(
                str(
                    metadata.iloc[
                        sample_index
                    ]["forecast_start"]
                )
            )

        axis.set_title(
            " | ".join(details),
            fontsize=10,
        )
        axis.set_ylabel("Power")
        axis.grid(alpha=0.2)

    axes[0, 0].legend(ncol=3)
    axes[-1, 0].set_xlabel(
        "Forecast step"
    )

    fig.suptitle(
        "Representative forecast windows "
        "by error quantile",
        y=1.002,
    )
    fig.tight_layout()

    fig.savefig(
        os.path.join(
            output_dir,
            "forecast_examples.png",
        ),
        dpi=180,
    )

    plt.close(fig)


def _plot_scatter(
    pred,
    true,
    output_dir,
    max_points=150000,
):
    p = pred.reshape(-1)
    y = true.reshape(-1)

    if len(p) > max_points:
        rng = np.random.default_rng(2024)
        selected = rng.choice(
            len(p),
            size=max_points,
            replace=False,
        )
        p = p[selected]
        y = y[selected]

    low = float(
        min(
            p.min(),
            y.min(),
        )
    )
    high = float(
        max(
            p.max(),
            y.max(),
        )
    )

    fig, axis = plt.subplots(
        figsize=(7.5, 7)
    )

    image = axis.hexbin(
        y,
        p,
        gridsize=80,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )

    axis.plot(
        [low, high],
        [low, high],
        "r--",
        linewidth=1.5,
        label="Ideal",
    )

    axis.set_xlabel("True power")
    axis.set_ylabel("Predicted power")
    axis.set_title(
        "Prediction versus truth density"
    )
    axis.legend()

    fig.colorbar(
        image,
        ax=axis,
        label="log10(count)",
    )
    fig.tight_layout()

    fig.savefig(
        os.path.join(
            output_dir,
            "prediction_scatter.png",
        ),
        dpi=180,
    )

    plt.close(fig)


def _plot_residuals(
    pred,
    true,
    output_dir,
    max_points=200000,
):
    residual = (
        pred - true
    ).reshape(-1)

    if len(residual) > max_points:
        rng = np.random.default_rng(2024)
        residual = residual[
            rng.choice(
                len(residual),
                size=max_points,
                replace=False,
            )
        ]

    low, high = np.percentile(
        residual,
        [1, 99],
    )

    shown = residual[
        (residual >= low)
        & (residual <= high)
    ]

    abs_sorted = np.sort(
        np.abs(residual)
    )

    cumulative = (
        np.arange(
            1,
            len(abs_sorted) + 1,
        )
        / len(abs_sorted)
        * 100.0
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    axes[0].hist(
        shown,
        bins=80,
        alpha=0.85,
    )
    axes[0].axvline(
        0.0,
        color="black",
        linewidth=1,
    )
    axes[0].set_title(
        "Residual distribution "
        "(1st-99th percentile)"
    )
    axes[0].set_xlabel(
        "Prediction - truth"
    )
    axes[0].set_ylabel("Count")

    axes[1].plot(
        abs_sorted,
        cumulative,
        linewidth=2,
    )
    axes[1].set_title(
        "Absolute-error "
        "cumulative distribution"
    )
    axes[1].set_xlabel(
        "Absolute error"
    )
    axes[1].set_ylabel(
        "Points within error (%)"
    )
    axes[1].grid(alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            output_dir,
            "residual_distribution.png",
        ),
        dpi=180,
    )

    plt.close(fig)


def _plot_per_turbine(
    table,
    output_dir,
    top_n=20,
):
    if table is None or table.empty:
        return

    worst = (
        table.nlargest(
            min(
                top_n,
                len(table),
            ),
            "mae",
        )
        .sort_values("mae")
    )

    fig, axis = plt.subplots(
        figsize=(
            9,
            max(
                5,
                0.35 * len(worst),
            ),
        )
    )

    axis.barh(
        worst["TurbID"].astype(str),
        worst["mae"],
    )
    axis.set_xlabel("MAE")
    axis.set_ylabel("TurbID")
    axis.set_title(
        f"Worst {len(worst)} "
        "turbines by MAE"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            output_dir,
            "worst_turbines_mae.png",
        ),
        dpi=180,
    )

    plt.close(fig)


def _plot_power_bins(
    table,
    output_dir,
):
    if table is None or table.empty:
        return

    labels = [
        f"Q{value}"
        for value in table["bin"]
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    axes[0].plot(
        labels,
        table["mae"],
        marker="o",
        label="MAE",
    )
    axes[0].plot(
        labels,
        table["rmse"],
        marker="o",
        label="RMSE",
    )
    axes[0].set_ylabel("Power error")
    axes[0].set_title(
        "Error by true-power quantile"
    )
    axes[0].legend()

    axes[1].bar(
        labels,
        table["bias_mbe"],
    )
    axes[1].axhline(
        0.0,
        color="black",
        linewidth=1,
    )
    axes[1].set_ylabel("Bias")
    axes[1].set_xlabel(
        "True-power quantile "
        "(low to high)"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            output_dir,
            "error_by_power_bin.png",
        ),
        dpi=180,
    )

    plt.close(fig)


def save_training_history(
    records,
    output_dir,
    title="Training history",
):
    """Save loss/LR history and plot."""
    if not records:
        return

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    table = pd.DataFrame(records)

    table.to_csv(
        os.path.join(
            output_dir,
            "training_history.csv",
        ),
        index=False,
    )

    has_validation_metrics = {
        "val_mse",
        "val_mae",
    }.issubset(table.columns)

    panel_count = (
        3
        if has_validation_metrics
        else 2
    )

    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(
            9,
            4 * panel_count,
        ),
        sharex=True,
    )

    axes[0].plot(
        table["epoch"],
        table["train_loss"],
        marker="o",
        label="Train",
    )
    axes[0].plot(
        table["epoch"],
        table["val_loss"],
        marker="o",
        label="Validation",
    )
    axes[0].set_ylabel("Loss")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    lr_axis = 1

    if has_validation_metrics:
        axes[1].plot(
            table["epoch"],
            table["val_mse"],
            marker="o",
            label="Val MSE",
        )
        axes[1].plot(
            table["epoch"],
            table["val_mae"],
            marker="o",
            label="Val MAE",
        )
        axes[1].set_ylabel(
            "Validation metric"
        )
        axes[1].legend()
        axes[1].grid(alpha=0.2)

        lr_axis = 2

    axes[lr_axis].plot(
        table["epoch"],
        table["learning_rate"],
        marker="o",
    )
    axes[lr_axis].set_xlabel("Epoch")
    axes[lr_axis].set_ylabel(
        "Learning rate"
    )
    axes[lr_axis].grid(alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            output_dir,
            "training_curves.png",
        ),
        dpi=180,
    )

    plt.close(fig)


def build_forecast_report(
    preds_scaled,
    trues_scaled,
    dataset,
    output_dir,
    last_observation_scaled=None,
    step_minutes=10,
    rated_power=None,
):
    """Save metrics, arrays, and charts."""
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    pred_scaled = _target_2d(
        preds_scaled
    )
    true_scaled = _target_2d(
        trues_scaled
    )

    if pred_scaled.shape != true_scaled.shape:
        raise ValueError(
            "Prediction/target shapes differ: "
            f"{pred_scaled.shape} != "
            f"{true_scaled.shape}"
        )

    pred = inverse_transform_target(
        dataset,
        pred_scaled,
    )
    true = inverse_transform_target(
        dataset,
        true_scaled,
    )

    normalized_metrics = forecast_metrics(
        pred_scaled,
        true_scaled,
    )
    original_metrics = forecast_metrics(
        pred,
        true,
        rated_power=rated_power,
    )

    persistence = None

    if last_observation_scaled is not None:
        last_scaled = np.asarray(
            last_observation_scaled,
            dtype=np.float64,
        ).reshape(-1, 1)

        if len(last_scaled) == len(pred):
            last = inverse_transform_target(
                dataset,
                last_scaled,
            )

            persistence = np.repeat(
                last,
                pred.shape[1],
                axis=1,
            )

            _add_persistence_metrics(
                original_metrics,
                pred,
                true,
                persistence,
            )

    _save_summary(
        output_dir,
        normalized_metrics,
        original_metrics,
    )

    metadata = _window_metadata(
        dataset,
        len(pred),
    )

    horizon = _horizon_table(
        pred,
        true,
        int(step_minutes),
        rated_power,
        persistence=persistence,
    )

    horizon.to_csv(
        os.path.join(
            output_dir,
            "metrics_by_horizon.csv",
        ),
        index=False,
    )

    segments = _horizon_segment_table(
        pred,
        true,
        persistence,
        int(step_minutes),
        rated_power,
    )

    segments.to_csv(
        os.path.join(
            output_dir,
            "metrics_by_horizon_segment.csv",
        ),
        index=False,
    )

    turbine = _per_turbine_table(
        metadata,
        pred,
        true,
        rated_power,
    )

    if turbine is not None:
        turbine.to_csv(
            os.path.join(
                output_dir,
                "metrics_by_turbine.csv",
            ),
            index=False,
        )

    power_bins = _power_bin_table(
        pred,
        true,
    )

    if power_bins is not None:
        power_bins.to_csv(
            os.path.join(
                output_dir,
                "metrics_by_power_bin.csv",
            ),
            index=False,
        )

    window_rmse = np.sqrt(
        np.mean(
            (pred - true) ** 2,
            axis=1,
        )
    )
    window_mae = np.mean(
        np.abs(pred - true),
        axis=1,
    )

    metadata = metadata.copy()
    metadata["mae"] = window_mae
    metadata["rmse"] = window_rmse

    metadata.to_csv(
        os.path.join(
            output_dir,
            "metrics_by_window.csv",
        ),
        index=False,
    )

    arrays = {
        "prediction_original": (
            pred.astype(np.float32)
        ),
        "truth_original": (
            true.astype(np.float32)
        ),
        "prediction_normalized": (
            pred_scaled.astype(np.float32)
        ),
        "truth_normalized": (
            true_scaled.astype(np.float32)
        ),
    }

    if persistence is not None:
        arrays["persistence_original"] = (
            persistence.astype(np.float32)
        )
        arrays["correction_original"] = (
            pred - persistence
        ).astype(np.float32)

    np.savez_compressed(
        os.path.join(
            output_dir,
            "predictions.npz",
        ),
        **arrays,
    )

    _plot_horizon(
        horizon,
        output_dir,
    )
    _plot_examples(
        pred,
        true,
        metadata,
        output_dir,
        persistence=persistence,
    )
    _plot_scatter(
        pred,
        true,
        output_dir,
    )
    _plot_residuals(
        pred,
        true,
        output_dir,
    )
    _plot_per_turbine(
        turbine,
        output_dir,
    )
    _plot_power_bins(
        power_bins,
        output_dir,
    )

    return original_metrics