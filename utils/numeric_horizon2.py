"""Direct second-hour history residual; the first-hour forecast is immutable."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from utils.numeric_subhour import predict as endpoint_predict

LAGS = (1, 2, 3, 6, 12, 24, 48, 71)
MEANS = (3, 6, 12, 24, 48, 72)


def history_features(dataset):
    if dataset.seq_len != 72 or dataset.pred_len != 2:
        raise ValueError("This protocol requires 72 observed hours and 2 direct forecast leads")
    rows = dataset.window_starts[:, None] + np.arange(72)[None]
    x = np.asarray(dataset.series.values[rows], float)
    if x.ndim != 3 or x.shape[-1] != 7 or not np.isfinite(x).all():
        raise ValueError("Finite seven-channel past-only history required")
    current = x[:, -1]
    return np.concatenate([current, *[current - x[:, -1 - lag] for lag in LAGS],
                           *[x[:, -span:].mean(1) - current for span in MEANS]], axis=1)


def fit(dataset, tables, beta):
    x = history_features(dataset)
    truth, _ = dataset.truth_and_anchor()
    baseline = endpoint_predict(beta, dataset, tables)
    # Only the second-hour outcome trains this head. Global target scale is the
    # frozen original prefix scaler, not fitted on any query/outer labels.
    y = (truth[:, 1] - baseline[:, 1]) / dataset.scaler.scale_[-1]
    scaler = StandardScaler(with_mean=False).fit(x)
    head = Ridge(alpha=100., fit_intercept=False).fit(scaler.transform(x), y)
    return dict(scale=scaler.scale_, coefficients=head.coef_, alpha=np.array(100.))


def predict(model, dataset, tables, beta):
    x = history_features(dataset)
    scale, coefficients = np.asarray(model["scale"]), np.asarray(model["coefficients"])
    if (scale.shape != (105,) or coefficients.shape != (105,) or np.any(scale <= 0)
            or not np.isfinite(scale).all() or not np.isfinite(coefficients).all()
            or float(model["alpha"]) != 100.):
        raise ValueError("Frozen 105-feature Ridge100 head required")
    baseline = endpoint_predict(beta, dataset, tables)
    prediction = baseline.copy()
    prediction[:, 1] += dataset.scaler.scale_[-1] * ((x / scale) @ coefficients)
    np.testing.assert_array_equal(prediction[:, 0], baseline[:, 0])
    return prediction


def choose(records, references):
    if set(records) != {"o70", "o80", "o90"} or set(references) != set(records):
        raise ValueError("Exactly three declared TRAIN query blocks required")
    scores, failures = [], []
    for key, row in records.items():
        if set(references[key]) != {"v15", "persistence"}:
            raise ValueError("Both frozen v15 and Persistence references required")
        for name, ref in references[key].items():
            for scope in ("horizon2", "overall"):
                for metric in ("mae", "rmse"):
                    value, reference = row[scope][metric], ref[scope][metric]
                    if (not np.isfinite(value) or not np.isfinite(reference) or reference <= 0
                            or value > reference + 1e-8):
                        failures.append(dict(origin=key, reference=name, scope=scope, metric=metric))
        denominator = [references[key]["v15"]["horizon2"][m] for m in ("mae", "rmse")]
        if all(np.isfinite(value) and value > 0 for value in denominator):
            scores.append(.5 * sum(row["horizon2"][m] / ref for m, ref in zip(("mae", "rmse"), denominator)))
    if failures or len(scores) != 3 or not np.isfinite(scores).all() or np.mean(scores) >= 1 - 1e-8:
        return dict(model="SubhourEndState", reused_v15=True, failed_checks=failures,
                    reason="No stable second-hour joint TRAIN gain; no new outer evaluation")
    return dict(model="Horizon2HistoryRidge", reused_v15=False, parameters=105,
                inner_horizon2_score=float(np.mean(scores)), first_hour_unchanged=True)
