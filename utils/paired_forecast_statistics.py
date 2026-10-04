"""Leakage-neutral paired inference for ordered wind forecast errors."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _as_2d(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite [window, horizon] array")
    return array


def _normal_two_sided_p(statistic):
    return float(math.erfc(abs(float(statistic)) / math.sqrt(2.0)))


def newey_west_mean_test(series, max_lag):
    """Approximate DM test for a zero mean loss differential.

    The loss differential is first aggregated by forecast issue time, then a
    Newey-West long-run variance accounts for remaining serial dependence.
    Positive values favor the candidate forecast.
    """
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    if values.size < 3 or not np.isfinite(values).all():
        raise ValueError("DM series must contain at least three finite values")
    lag = min(max(0, int(max_lag)), values.size - 2)
    centered = values - values.mean()
    gamma0 = float(np.dot(centered, centered) / values.size)
    long_run = gamma0
    for step in range(1, lag + 1):
        covariance = float(np.dot(centered[step:], centered[:-step]) / values.size)
        long_run += 2.0 * (1.0 - step / (lag + 1.0)) * covariance
    variance_mean = max(long_run / values.size, np.finfo(float).eps)
    statistic = float(values.mean() / math.sqrt(variance_mean))
    return {
        "statistic": statistic,
        "p_value_two_sided_normal": _normal_two_sided_p(statistic),
        "newey_west_lag": lag,
        "time_points": int(values.size),
    }


def moving_block_bootstrap_mean(series, block_length=12, replicates=5000, seed=2024):
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Bootstrap series must contain at least two finite values")
    block = min(max(1, int(block_length)), values.size)
    reps = int(replicates)
    if reps < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    starts = np.arange(values.size - block + 1)
    blocks_needed = int(math.ceil(values.size / block))
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[start : start + block] for start in chosen])
        estimates[i] = sample[: values.size].mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "mean_gain_kw": float(values.mean()),
        "ci95_low_kw": float(low),
        "ci95_high_kw": float(high),
        "probability_gain_positive": float(np.mean(estimates > 0.0)),
        "block_length_time_points": block,
        "replicates": reps,
        "seed": int(seed),
    }


def paired_forecast_statistics(
    candidate,
    reference,
    truth,
    metadata,
    block_length=12,
    replicates=5000,
    seed=2024,
):
    candidate = _as_2d(candidate, "candidate")
    reference = _as_2d(reference, "reference")
    truth = _as_2d(truth, "truth")
    if candidate.shape != reference.shape or candidate.shape != truth.shape:
        raise ValueError("Candidate, reference and truth shapes must match exactly")
    metadata = metadata.reset_index(drop=True).copy()
    if len(metadata) != len(truth) or "forecast_start" not in metadata:
        raise ValueError("Metadata must align one-to-one and include forecast_start")
    candidate_loss = np.abs(candidate - truth).mean(axis=1)
    reference_loss = np.abs(reference - truth).mean(axis=1)
    gain = reference_loss - candidate_loss
    frame = pd.DataFrame(
        {
            "forecast_start": pd.to_datetime(metadata["forecast_start"], errors="raise"),
            "gain_kw": gain,
        }
    )
    by_time = frame.groupby("forecast_start", sort=True, observed=True).gain_kw.mean()
    bootstrap = moving_block_bootstrap_mean(
        by_time.to_numpy(), block_length=block_length, replicates=replicates, seed=seed
    )
    dm = newey_west_mean_test(by_time.to_numpy(), max_lag=block_length - 1)
    return {
        "gain_definition": "MAE(reference)-MAE(candidate); positive favors candidate",
        "candidate_mae_kw": float(candidate_loss.mean()),
        "reference_mae_kw": float(reference_loss.mean()),
        "paired_gain_kw": float(gain.mean()),
        "paired_gain_pct": float(100.0 * gain.mean() / max(reference_loss.mean(), np.finfo(float).eps)),
        "candidate_win_window_pct": float(100.0 * np.mean(candidate_loss < reference_loss)),
        "windows": int(len(gain)),
        "horizon": int(candidate.shape[1]),
        "unique_forecast_times": int(len(by_time)),
        "time_block_bootstrap": bootstrap,
        "dm_newey_west": dm,
        "interpretation": (
            "Exploratory validation inference; freeze the method before using a sealed test. "
            "Windows sharing a forecast time are averaged before temporal inference."
        ),
    }


def compare_report_directories(
    candidate_dir,
    reference_dir,
    output_dir,
    block_length=12,
    replicates=5000,
    seed=2024,
):
    candidate_dir = Path(candidate_dir)
    reference_dir = Path(reference_dir)
    output_dir = Path(output_dir)
    with np.load(candidate_dir / "predictions.npz") as candidate_npz:
        candidate = candidate_npz["prediction_original"]
        candidate_truth = candidate_npz["truth_original"]
    with np.load(reference_dir / "predictions.npz") as reference_npz:
        reference = reference_npz["prediction_original"]
        reference_truth = reference_npz["truth_original"]
    if not np.array_equal(candidate_truth, reference_truth):
        raise ValueError("Paired reports do not contain byte-identical truth arrays")
    candidate_meta = pd.read_csv(candidate_dir / "metrics_by_window.csv")
    reference_meta = pd.read_csv(reference_dir / "metrics_by_window.csv")
    identity = ["TurbID", "forecast_start"]
    if not set(identity).issubset(candidate_meta) or not set(identity).issubset(reference_meta):
        raise ValueError("Both reports must include matching identity metadata")
    if not candidate_meta[identity].equals(reference_meta[identity]):
        raise ValueError("Paired report windows are not identically ordered")
    report = paired_forecast_statistics(
        candidate,
        reference,
        candidate_truth,
        candidate_meta,
        block_length=block_length,
        replicates=replicates,
        seed=seed,
    )
    report.update(
        {
            "candidate_report": str(candidate_dir.resolve()),
            "reference_report": str(reference_dir.resolve()),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paired_statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame([{
        "candidate_mae_kw": report["candidate_mae_kw"],
        "reference_mae_kw": report["reference_mae_kw"],
        "gain_kw": report["paired_gain_kw"],
        "gain_pct": report["paired_gain_pct"],
        "window_win_pct": report["candidate_win_window_pct"],
        "bootstrap_low_kw": report["time_block_bootstrap"]["ci95_low_kw"],
        "bootstrap_high_kw": report["time_block_bootstrap"]["ci95_high_kw"],
        "dm_statistic": report["dm_newey_west"]["statistic"],
        "dm_p_value": report["dm_newey_west"]["p_value_two_sided_normal"],
    }]).to_csv(output_dir / "paired_statistics.csv", index=False)
    return report
