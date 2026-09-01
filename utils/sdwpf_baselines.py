"""Train-only Persistence, wind-power curve, and tree baselines for SDWPF.

All models see only history available at the forecast issue time.  The power
curve maps persisted last wind speed to power.  The tree model never receives
future SCADA or NWP.
"""

from collections import OrderedDict

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from utils.forecast_report import inverse_transform_target
from utils.metrics import forecast_metrics


EPS = np.finfo(np.float64).eps


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


def _window_truth(dataset):
    index = _target_index(dataset)
    starts = np.asarray(dataset.window_starts, dtype=np.int64)
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


def fit_tree_baseline(train_dataset, max_windows=20000, random_state=2024):
    """Single HGB with horizon-step as a feature; history only."""
    features = _history_stats(train_dataset)
    truth = _window_truth(train_dataset)
    n_windows, horizon = truth.shape
    rng = np.random.default_rng(random_state)
    window_count = min(n_windows, max_windows)
    window_ids = rng.choice(n_windows, size=window_count, replace=False)
    # Two samples per window keep the matrix small while covering short/long leads.
    horizon_ids = rng.integers(0, horizon, size=(window_count, 2))
    rows = []
    targets = []
    for window_id, leads in zip(window_ids, horizon_ids):
        for lead in leads:
            rows.append(np.concatenate([features[window_id], [float(lead)]]))
            targets.append(truth[window_id, int(lead)])
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        random_state=random_state,
    )
    model.fit(np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64))
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
