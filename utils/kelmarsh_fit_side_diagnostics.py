"""Torch-free helpers for read-only fit-side diagnostics of frozen Kelmarsh forecasters.

Regimes are frozen BEFORE looking at any error: power regimes use physical fractions of the rated power, ramp regimes
use terciles of the historical 3-hour power change computed on the FIT windows only. Nothing here selects, refits,
clips, deletes or re-weights samples; it only describes errors of already frozen models.
"""
from __future__ import annotations

import numpy as np

POWER_EDGES_FRACTION = (0.1, 0.3, 0.6, 0.9)
POWER_LABELS = ("<10%", "10-30%", "30-60%", "60-90%", ">=90%")
RAMP_LABELS = ("ramp_low", "ramp_mid", "ramp_high")


def last_power_and_ramp(raw_power, window_starts, input_len, ramp_hours=3):
    """Last observed hourly power and |power[t] - power[t-ramp_hours]| at the issue time (history only)."""
    starts = np.asarray(window_starts, dtype=np.int64)
    last = np.asarray(raw_power, dtype=np.float64)[starts + input_len - 1]
    before = np.asarray(raw_power, dtype=np.float64)[starts + input_len - 1 - int(ramp_hours)]
    return last, np.abs(last - before)


def ramp_edges_from_fit(fit_ramp):
    """Terciles of the fit-window historical ramp; frozen and then applied unchanged to every split."""
    values = np.asarray(fit_ramp, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or values.size < 3:
        raise ValueError("Finite fit-window ramps required")
    return tuple(float(q) for q in np.quantile(values, [1 / 3, 2 / 3]))


def regime_labels(last_power, ramp, rated_power, ramp_edges):
    power_bin = np.digitize(np.asarray(last_power, dtype=np.float64) / float(rated_power), POWER_EDGES_FRACTION)
    ramp_bin = np.digitize(np.asarray(ramp, dtype=np.float64), ramp_edges)
    return {"power": np.asarray(POWER_LABELS)[power_bin], "ramp": np.asarray(RAMP_LABELS)[ramp_bin]}


def error_summary(prediction, truth):
    """MAE / RMSE / MBE overall and per lead (kW) for [window, lead] arrays."""
    prediction, truth = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 2 or not (np.isfinite(prediction).all() and np.isfinite(truth).all()):
        raise ValueError("Aligned finite [window, lead] arrays required")
    error = prediction - truth
    return {"windows": int(len(error)), "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt((error ** 2).mean())),
            "mbe": float(error.mean()),
            "per_lead": [{"lead_hours": h + 1, "mae": float(np.abs(error[:, h]).mean()),
                          "rmse": float(np.sqrt((error[:, h] ** 2).mean())), "mbe": float(error[:, h].mean())}
                         for h in range(error.shape[1])]}


def by_group(prediction, truth, labels, order):
    """Error summary per label value, in a fixed order, with the support (windows) always reported."""
    labels = np.asarray(labels)
    out = {}
    for label in order:
        mask = labels == label
        out[str(label)] = error_summary(np.asarray(prediction)[mask], np.asarray(truth)[mask]) if mask.any() \
            else {"windows": 0, "note": "no support"}
    return out
