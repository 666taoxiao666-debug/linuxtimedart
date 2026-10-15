"""Causal non-self farm context and fixed numerical residual learners (no Wiki)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from utils.numeric_foundation import dataset_features, observed_anchor

LAGS = (0, 1, 2, 3, 6, 12, 24)
HGB = dict(loss="squared_error", learning_rate=.05, max_iter=200,
           max_leaf_nodes=15, min_samples_leaf=64, l2_regularization=10.,
           early_stopping=False, random_state=2024)
NAMES = ("OwnDeltaRidge_a100", "FarmDeltaRidge_a100", "OwnDeltaHGB", "FarmDeltaHGB")


class FarmContext:
    """Index clocks/units only; fetch values lazily at requested past clocks.

    Hourly records are end-stamped by the frozen dataset pipeline. lag=0 is
    the last fully observed hour, not an observation from a forecast horizon.
    """

    def __init__(self, series):
        self.series = series
        self.clocks = np.unique(series.dates)
        self.units = np.unique(series.turbines)
        if len(self.units) < 2:
            raise ValueError("Farm context requires at least two turbines")
        clock = np.searchsorted(self.clocks, series.dates)
        unit = np.searchsorted(self.units, series.turbines)
        flat = clock * len(self.units) + unit
        if len(np.unique(flat)) != len(flat):
            raise ValueError("Duplicate timestamp/turbine observation")
        self.rows = np.full((len(self.clocks), len(self.units)), -1, dtype=np.int64)
        self.rows[clock, unit] = np.arange(len(flat))

    def features(self, dataset):
        if dataset.series is not self.series:
            raise ValueError("Farm context belongs to a different series")
        last = dataset.window_starts + dataset.seq_len - 1
        issue = self.series.dates[last]
        own_unit = np.searchsorted(self.units, self.series.turbines[last])
        pieces = []
        for lag in LAGS:
            clocks = issue - np.timedelta64(lag, "h")
            positions = np.searchsorted(self.clocks, clocks)
            if ((positions >= len(self.clocks)).any() or
                    not np.array_equal(self.clocks[positions], clocks) or (clocks > issue).any()):
                raise ValueError("Exact causal hourly context unavailable")
            rows = self.rows[positions]
            # No future labels are read: every gathered row has exactly this
            # requested historical clock, including partially missing neighbors.
            values = self.series.values[np.maximum(rows, 0)][:, :, [0, 6]].astype(float)
            values[rows < 0] = np.nan
            own = values[np.arange(len(last)), own_unit].copy()
            if not np.isfinite(own).all():
                raise ValueError("Own historical context unavailable")
            values[np.arange(len(last)), own_unit] = np.nan
            finite = np.isfinite(values)
            count = finite.sum(axis=1)
            mean = np.divide(np.nansum(values, axis=1), count,
                             out=np.zeros_like(own), where=count > 0)
            variance = np.divide(np.nansum((values - mean[:, None]) ** 2, axis=1), count,
                                 out=np.zeros_like(own), where=count > 0)
            difference = np.where(count > 0, mean - own, 0.)
            pieces.extend((difference, np.sqrt(variance), count / (len(self.units) - 1)))
        result = np.concatenate(pieces, axis=1)
        if result.shape != (len(last), 42) or not np.isfinite(result).all():
            raise ValueError("Invalid causal farm context")
        return result


def features(dataset, name, context):
    if name not in NAMES:
        raise ValueError("Undeclared numerical candidate")
    own = dataset_features(dataset)
    return np.concatenate((own, context.features(dataset)), axis=1) if name.startswith("Farm") else own


def fit(dataset, name, context):
    x = features(dataset, name, context)
    truth, anchor = dataset.truth_and_anchor()
    y = (truth - anchor) / dataset.scaler.scale_[-1]
    if name.endswith("Ridge_a100"):
        scale = StandardScaler().fit(x)
        predictor = Ridge(alpha=100.).fit(scale.transform(x), y)
    else:
        scale = None
        predictor = [HistGradientBoostingRegressor(**HGB).fit(x, y[:, lead]) for lead in range(y.shape[1])]
    return dict(model=name, scale=scale, predictor=predictor)


def predict(model, dataset, context):
    x = features(dataset, model["model"], context)
    if model["scale"] is not None:
        delta = model["predictor"].predict(model["scale"].transform(x))
    else:
        delta = np.column_stack([head.predict(x) for head in model["predictor"]])
    # Exact observed raw-kW anchor; no normalization round-trip or clipping.
    return observed_anchor(dataset) + dataset.scaler.scale_[-1] * delta


def capacity(name, horizon=2):
    """Fixed declared complexity proxy, NOT a claim about neural parameters."""
    if name not in NAMES:
        raise ValueError("Undeclared numerical candidate")
    return horizon * ((119 if name.startswith("Farm") else 77) + 1) if "Ridge" in name else horizon * 200 * 29
