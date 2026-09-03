"""Train-only Persistence, wind-power curve, and tree baselines for SDWPF.

All models see only history available at the forecast issue time.  The power
curve maps persisted last wind speed to power.  The tree model never receives
future SCADA or NWP.
"""

import os
from collections import OrderedDict

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from utils.forecast_report import inverse_transform_target
from utils.metrics import forecast_metrics


EPS = np.finfo(np.float64).eps


def authorize_eval_split(eval_split, environ=None):
    """Keep the test split sealed until the user explicitly confirms final use."""
    environ = os.environ if environ is None else environ
    if eval_split == "test" and environ.get("CONFIRM_FINAL_EVAL") != "1":
        raise PermissionError(
            "Test evaluation is locked. Select baselines on --eval_split val. "
            "For the one-time final test only, set CONFIRM_FINAL_EVAL=1."
        )
    return eval_split


def _target_index(dataset):
    columns = list(dataset.feature_columns)
    target = dataset.target
    return columns.index(target) if target in columns else -1


def original_target_series(dataset):
    index = _target_index(dataset)
    scaled = np.asarray(dataset.data_x[:, index], dtype=np.float64)
    mean = float(np.asarray(dataset.scaler.mean_).reshape(-1)[index])
    scale = float(np.asarray(dataset.scaler.scale_).reshape(-1)[index])
    return scaled * scale + mean


def _window_truth(dataset, window_ids=None):
    index = _target_index(dataset)
    starts = np.asarray(dataset.window_starts, dtype=np.int64)
    if window_ids is not None:
        starts = starts[np.asarray(window_ids, dtype=np.int64)]
    rows = starts[:, None] + int(dataset.seq_len) + np.arange(int(dataset.pred_len))
    scaled = dataset.data_x[rows][:, :, index]
    return inverse_transform_target(dataset, scaled)


def _history_stats(dataset):
    index = _target_index(dataset)
    starts = np.asarray(dataset.window_starts, dtype=np.int64)
    seq = int(dataset.seq_len)
    hist = starts[:, None] + np.arange(seq)
    power = original_target_series(dataset)[hist]
    wspd = np.asarray(dataset.wspd, dtype=np.float64)[hist]
    hour = (
        np.asarray(dataset.data_stamp, dtype=np.float64)[starts + seq - 1, 3]
        if dataset.data_stamp.shape[1] > 3
        else np.zeros(len(starts))
    )
    return np.column_stack(
        [
            power[:, -1],
            wspd[:, -1],
            power.mean(axis=1),
            wspd.mean(axis=1),
            power.std(axis=1),
            wspd.std(axis=1),
            hour,
        ]
    )


def persistence_forecast(dataset):
    last = original_target_series(dataset)
    starts = np.asarray(dataset.window_starts, dtype=np.int64)
    last_power = last[starts + int(dataset.seq_len) - 1]
    return np.repeat(last_power.reshape(-1, 1), int(dataset.pred_len), axis=1)


def seasonal_persistence_forecast(dataset, seasonal_lag=144):
    """Repeat the power observed one daily cycle before each target step."""
    lag = int(seasonal_lag)
    horizon = int(dataset.pred_len)
    if lag < horizon or int(dataset.seq_len) < lag:
        raise ValueError("seasonal_lag must be within history and >= pred_len")
    power = original_target_series(dataset)
    starts = np.asarray(dataset.window_starts, dtype=np.int64)
    rows = (
        starts[:, None]
        + int(dataset.seq_len)
        + np.arange(horizon, dtype=np.int64)[None, :]
        - lag
    )
    return power[rows]


def fit_power_curve(train_dataset):
    """Isotonic P=f(Wspd) on training timestamps that are still generating."""
    power = original_target_series(train_dataset)
    wspd = np.asarray(train_dataset.wspd, dtype=np.float64)
    available = np.asarray(train_dataset.available_mask, dtype=bool)
    train_time = train_dataset.dates < train_dataset.train_cutoff
    keep = train_time & available & np.isfinite(wspd) & np.isfinite(power)
    if keep.sum() < 50:
        keep = train_time & np.isfinite(wspd) & np.isfinite(power)
    curve = IsotonicRegression(out_of_bounds="clip")
    curve.fit(wspd[keep], power[keep])
    return curve


def power_curve_forecast(dataset, curve):
    last_wspd = np.asarray(dataset.last_history_wspd(), dtype=np.float64)
    last_wspd = np.nan_to_num(last_wspd, nan=0.0)
    power = curve.predict(last_wspd)
    return np.repeat(power.reshape(-1, 1), int(dataset.pred_len), axis=1)


def fit_tree_baseline(train_dataset, max_samples=500000, random_state=2024):
    """Balanced direct HGB over every lead, using history-only statistics."""
    features = _history_stats(train_dataset)
    n_windows = len(features)
    horizon = int(train_dataset.pred_len)
    rng = np.random.default_rng(random_state)
    window_count = min(n_windows, max(1, int(max_samples) // horizon))
    window_ids = rng.choice(n_windows, size=window_count, replace=False)
    truth = _window_truth(train_dataset, window_ids=window_ids)
    rows = np.column_stack(
        [
            np.repeat(features[window_ids], horizon, axis=0),
            np.tile(np.arange(horizon, dtype=np.float64), window_count),
        ]
    )
    targets = truth.reshape(-1)
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        random_state=random_state,
    )
    model.fit(np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64))
    model.sdwpf_training_samples_ = int(len(targets))
    model.sdwpf_windows_sampled_ = int(window_count)
    return model


def tree_forecast(dataset, model):
    features = _history_stats(dataset)
    horizon = int(dataset.pred_len)
    predictions = np.empty((len(features), horizon), dtype=np.float64)
    for lead in range(horizon):
        lead_features = np.column_stack(
            [features, np.full(len(features), float(lead), dtype=np.float64)]
        )
        predictions[:, lead] = model.predict(lead_features)
    return predictions


def _skill(metrics, baseline):
    result = dict(metrics)
    result["mae_skill_vs_persistence_pct"] = 100.0 * (
        1.0 - metrics["mae"] / max(baseline["mae"], EPS)
    )
    result["rmse_skill_vs_persistence_pct"] = 100.0 * (
        1.0 - metrics["rmse"] / max(baseline["rmse"], EPS)
    )
    return result


def evaluate_methods(truth, methods, rated_power, persistence):
    persistence_metrics = forecast_metrics(persistence, truth, rated_power=rated_power)
    rows = OrderedDict()
    rows["persistence"] = persistence_metrics
    for name, prediction in methods.items():
        rows[name] = _skill(
            forecast_metrics(prediction, truth, rated_power=rated_power),
            persistence_metrics,
        )
    return rows
