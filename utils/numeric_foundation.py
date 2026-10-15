"""Numerical-only partitions, causal lag features and train-only selection."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from data_provider.kelmarsh_hourly import KelmarshWindows

LAGS = (1, 2, 3, 6, 12, 24)
MEANS = (3, 6, 12, 24)


def partitions(series, fraction=.9):
    if not 0 < fraction < 1:
        raise ValueError("Internal fitting fraction must be inside (0,1)")
    origin = series.dates.min().astype("datetime64[h]")
    train_end = series.bounds["train_end"].astype("datetime64[h]")
    cut = (origin + (train_end - origin) * fraction).astype("datetime64[h]")
    starts = series.starts
    first = series.dates[starts + series.input_len]
    last = series.dates[starts + series.input_len + series.horizon - 1]
    masks = dict(fit=last < cut, selection=(first >= cut) & (last < train_end), outer_refit=last < train_end)
    datasets = {key: KelmarshWindows(series, starts[mask], "numeric_" + key) for key, mask in masks.items()}
    if any(len(value) == 0 for value in datasets.values()):
        raise ValueError("Empty numerical fitting/selection period")
    if datasets["fit"].target_times().max() > datasets["selection"].target_times()[:, 0].min() - np.timedelta64(1, "h"):
        raise ValueError("Fitting outcomes unavailable at first internal selection issue")
    return datasets, dict(inner_cut=str(cut), outer_train_end=str(train_end),
        windows={key: len(value) for key, value in datasets.items()},
        raw_labels_modified=False, original_input_scaler_retained=True,
        wiki_memory_or_labels_reserved=False, numerical_training_units=sorted(np.unique(series.turbines).tolist()))


def lag_features(history):
    x = np.asarray(history, float)
    if x.ndim != 3 or x.shape[-1] != 7 or x.shape[1] < 25 or not np.isfinite(x).all():
        raise ValueError("Finite [window,past hours,seven channels] history required")
    current = x[:, -1]
    return np.concatenate([current, *[current - x[:, -1 - lag] for lag in LAGS],
                           *[x[:, -span:].mean(1) - current for span in MEANS]], -1)


def dataset_features(dataset):
    rows = dataset.window_starts[:, None] + np.arange(dataset.seq_len)[None]
    history = dataset.series.values[rows]
    return lag_features(history)


def observed_anchor(dataset):
    last = dataset.series.raw[dataset.window_starts + dataset.seq_len - 1, -1]
    return np.repeat(last[:, None], dataset.pred_len, axis=1)


def dataset_arrays(dataset):
    truth, anchor = dataset.truth_and_anchor()
    return dataset_features(dataset), (truth - anchor) / dataset.scaler.scale_[-1]


def fit_ridge(dataset, alpha):
    if alpha <= 0:
        raise ValueError("Positive Ridge penalty required")
    x, target = dataset_arrays(dataset)
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=float(alpha)).fit(scaler.transform(x), target)
    return dict(alpha=float(alpha), mean=scaler.mean_, scale=scaler.scale_,
                coefficients=model.coef_, intercept=model.intercept_)


def ridge_predict(model, dataset):
    x = dataset_features(dataset)
    delta = ((x - model["mean"]) / model["scale"]) @ np.asarray(model["coefficients"]).T + model["intercept"]
    return observed_anchor(dataset) + dataset.scaler.scale_[-1] * delta


def joint_score(metrics, baseline):
    # Numerical tolerance only, not a searched utility threshold.
    if any(not np.isfinite(metrics[key]) for key in ("mae", "rmse")):
        return float("inf")
    if any(metrics[key] > baseline[key] + 1e-8 for key in ("mae", "rmse")):
        return float("inf")
    return .5 * sum(metrics[key] / baseline[key] for key in ("mae", "rmse"))


def select_candidate(rows, baseline):
    valid = [row for row in rows if row.get("best_epoch") != 0 and joint_score(row["metrics"], baseline) < 1 - 1e-8]
    if not valid:
        return dict(model="Persistence", best_epoch=0, inner_score=1., reason="No joint internal improvement")
    selected = min(valid, key=lambda row: (joint_score(row["metrics"], baseline), row["parameters"], row["model"]))
    return dict(selected, inner_score=joint_score(selected["metrics"], baseline))
