from collections import OrderedDict

import numpy as np


def _paired_finite(pred, true):
    """Return shape-checked finite prediction/target arrays."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"pred and true must have the same shape: {pred.shape} != {true.shape}")
    finite = np.isfinite(pred) & np.isfinite(true)
    if not finite.any():
        raise ValueError("pred and true contain no paired finite values")
    return pred, true, finite


def RSE(pred, true):
    pred, true, finite = _paired_finite(pred, true)
    pred, true = pred[finite], true[finite]
    denominator = np.sum((true - true.mean()) ** 2)
    return np.sqrt(np.sum((true - pred) ** 2) / denominator) if denominator > 0 else np.nan


def CORR(pred, true):
    pred, true, finite = _paired_finite(pred, true)
    pred, true = pred[finite], true[finite]
    if pred.size < 2 or np.std(pred) == 0 or np.std(true) == 0:
        return np.nan
    return np.corrcoef(pred, true)[0, 1]


def MAE(pred, true):
    pred, true, finite = _paired_finite(pred, true)
    return np.mean(np.abs(pred[finite] - true[finite]))


def MSE(pred, true):
    pred, true, finite = _paired_finite(pred, true)
    return np.mean((pred[finite] - true[finite]) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def _mape_threshold(true, threshold=None):
    finite_true = np.abs(np.asarray(true, dtype=np.float64))
    finite_true = finite_true[np.isfinite(finite_true)]
    if finite_true.size == 0:
        return np.finfo(np.float64).eps
    if threshold is not None:
        return max(float(threshold), np.finfo(np.float64).eps)
    # Wind power frequently contains zeros.  Ignore targets below 1% of the
    # 95th-percentile magnitude instead of allowing them to explode MAPE.
    return max(0.01 * float(np.percentile(finite_true, 95)), np.finfo(np.float64).eps)


def MAPE(pred, true, threshold=None):
    pred, true, finite = _paired_finite(pred, true)
    threshold = _mape_threshold(true[finite], threshold)
    valid = finite & (np.abs(true) >= threshold)
    if not valid.any():
        return np.nan
    return np.mean(np.abs((pred[valid] - true[valid]) / true[valid]))


def MSPE(pred, true, threshold=None):
    pred, true, finite = _paired_finite(pred, true)
    threshold = _mape_threshold(true[finite], threshold)
    valid = finite & (np.abs(true) >= threshold)
    if not valid.any():
        return np.nan
    return np.mean(np.square((pred[valid] - true[valid]) / true[valid]))


def forecast_metrics(pred, true, mape_threshold=None, rated_power=None):
    """Calculate a broad, zero-safe set of deterministic forecast metrics.

    Percentage metrics are returned in percentage points.  ``MAPE`` is masked
    around zero and its retained-data percentage is reported alongside it.
    ``rated_power`` is optional because it must come from turbine metadata, not
    be guessed from the test set.
    """
    pred, true, finite = _paired_finite(pred, true)
    p = pred[finite]
    y = true[finite]
    error = p - y
    abs_error = np.abs(error)
    squared_error = error ** 2
    eps = np.finfo(np.float64).eps

    mae = float(abs_error.mean())
    mse = float(squared_error.mean())
    rmse = float(np.sqrt(mse))
    mean_abs_true = max(float(np.mean(np.abs(y))), eps)
    true_range = max(float(np.max(y) - np.min(y)), eps)
    threshold = _mape_threshold(y, mape_threshold)
    mape_mask = np.abs(y) >= threshold
    mape = (
        float(np.mean(abs_error[mape_mask] / np.abs(y[mape_mask])) * 100.0)
        if mape_mask.any()
        else np.nan
    )
    smape = float(
        np.mean(2.0 * abs_error / np.maximum(np.abs(p) + np.abs(y), eps)) * 100.0
    )
    wape = float(abs_error.sum() / max(np.abs(y).sum(), eps) * 100.0)

    sse = float(squared_error.sum())
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - sse / sst) if sst > eps else np.nan
    explained_variance = (
        float(1.0 - np.var(error) / np.var(y)) if np.var(y) > eps else np.nan
    )
    pearson = (
        float(np.corrcoef(p, y)[0, 1])
        if p.size > 1 and np.std(p) > eps and np.std(y) > eps
        else np.nan
    )

    result = OrderedDict(
        point_count=int(p.size),
        mean_true=float(np.mean(y)),
        std_true=float(np.std(y)),
        mse=mse,
        rmse=rmse,
        mae=mae,
        median_ae=float(np.median(abs_error)),
        mape_masked_pct=mape,
        mape_coverage_pct=float(mape_mask.mean() * 100.0),
        mape_threshold=float(threshold),
        smape_pct=smape,
        wape_pct=wape,
        nmae_mean_abs_pct=float(mae / mean_abs_true * 100.0),
        nrmse_mean_abs_pct=float(rmse / mean_abs_true * 100.0),
        nrmse_range_pct=float(rmse / true_range * 100.0),
        r2=r2,
        explained_variance=explained_variance,
        pearson_r=pearson,
        bias_mbe=float(np.mean(error)),
        error_std=float(np.std(error)),
        max_ae=float(np.max(abs_error)),
        p50_ae=float(np.percentile(abs_error, 50)),
        p90_ae=float(np.percentile(abs_error, 90)),
        p95_ae=float(np.percentile(abs_error, 95)),
        p99_ae=float(np.percentile(abs_error, 99)),
        negative_prediction_pct=float(np.mean(p < 0) * 100.0),
        over_prediction_pct=float(np.mean(error > 0) * 100.0),
        under_prediction_pct=float(np.mean(error < 0) * 100.0),
    )

    if rated_power is not None and float(rated_power) > 0:
        capacity = float(rated_power)
        result["rated_power"] = capacity
        result["nmae_capacity_pct"] = float(mae / capacity * 100.0)
        result["nrmse_capacity_pct"] = float(rmse / capacity * 100.0)
        result["within_5pct_capacity_pct"] = float(np.mean(abs_error <= 0.05 * capacity) * 100.0)
        result["within_10pct_capacity_pct"] = float(np.mean(abs_error <= 0.10 * capacity) * 100.0)
        result["above_rated_prediction_pct"] = float(np.mean(p > capacity) * 100.0)

    # Direction accuracy must not compare the end of one forecast window with
    # the beginning of another, so preserve the window dimension here.
    if pred.ndim >= 2 and pred.shape[1] > 1:
        valid_pairs = (
            finite[:, 1:, ...] & finite[:, :-1, ...]
            if finite.ndim >= 2
            else None
        )
        pred_delta = np.diff(pred, axis=1)
        true_delta = np.diff(true, axis=1)
        if valid_pairs is not None and valid_pairs.any():
            result["direction_accuracy_pct"] = float(
                np.mean(
                    np.sign(pred_delta[valid_pairs])
                    == np.sign(true_delta[valid_pairs])
                )
                * 100.0
            )

    return result


def metric(pred, true):
    """Backward-compatible five-value interface used by older experiments."""
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = np.sqrt(mse)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    return mae, mse, rmse, mape, mspe
