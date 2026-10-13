"""Hourly Kelmarsh windows, with target-only future validity and causal inputs.

This deliberately does not reuse SDWPF clipping, angle semantics or minute
defaults. Future covariates are neither required nor supplied to the forecaster.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


FEATURES = ["Wspd", "Etmp", "Ndir_sin", "Ndir_cos", "Pab_mean", "rotor_speed", "power"]
SOURCE_COLUMNS = ["timestamp_hour_end_utc", "turbine_id", "wind_speed_mps", "ambient_temp_c",
                  "nacelle_position_deg", "pitch_a_deg", "pitch_b_deg", "pitch_c_deg",
                  "rotor_speed_rpm", "power_kw"]


def time_bounds(timeline, fold):
    if fold not in (0, 1, 2) or len(timeline) < 100:
        raise ValueError("Three declared rolling folds and a nonempty timeline required")
    n = len(timeline)
    train = int(n * (6 + fold) / 10)
    validation = int(n * (7 + fold) / 10)
    parts = {"base_fit_end": int(train * .6), "base_selection_end": int(train * .7),
             "memory_fit_end": int(train * .8), "calibration_early_end": int(train * .9),
             "train_end": train, "validation_end": validation, "sealed_test_start": int(n * .9)}
    return {key: timeline[index] for key, index in parts.items()}


def valid_window_starts(values, dates, turbines, input_len, horizon, stride):
    """Require finite history and power targets; never filter on future weather."""
    values = np.asarray(values)
    starts = np.arange(0, len(values) - input_len - horizon + 1, stride, dtype=np.int64)
    hbad = np.concatenate(([0], np.cumsum(~np.isfinite(values).all(1))))
    ybad = np.concatenate(([0], np.cumsum(~np.isfinite(values[:, -1]))))
    end = starts + input_len + horizon
    keep = ((hbad[starts + input_len] - hbad[starts]) == 0)
    keep &= (ybad[end] - ybad[starts + input_len]) == 0
    keep &= np.asarray(turbines)[starts] == np.asarray(turbines)[end - 1]
    keep &= (np.asarray(dates)[end - 1] - np.asarray(dates)[starts]) == np.timedelta64(input_len + horizon - 1, "h")
    gaps = np.concatenate(([0], np.cumsum(np.diff(np.asarray(dates)) != np.timedelta64(1, "h"))))
    keep &= gaps[end - 1] == gaps[starts]
    return starts[keep], len(starts)


class KelmarshSeries:
    def __init__(self, path, *, fold, input_len=72, horizon=12, stride=3,
                 start="2016-05-01T00:00:00Z", end="2017-01-01T00:00:00Z"):
        frame = pd.read_csv(path, usecols=SOURCE_COLUMNS)
        frame["date"] = pd.to_datetime(frame.timestamp_hour_end_utc, utc=True).dt.tz_localize(None)
        lo, hi = pd.Timestamp(start).tz_convert(None), pd.Timestamp(end).tz_convert(None)
        frame = frame[(frame.date >= lo) & (frame.date < hi)].copy()
        if frame.duplicated(["turbine_id", "date"]).any():
            raise ValueError("Duplicate turbine/hour rows")
        timeline = pd.date_range(lo, hi, freq="1h", inclusive="left").to_numpy()
        self.bounds = time_bounds(timeline, fold)
        # Do not engineer, scale, window, aggregate or infer any sealed suffix.
        frame = frame[frame.date < self.bounds["sealed_test_start"]].copy()
        ids = sorted(frame.turbine_id.unique())
        if list(ids) != [1, 2, 3, 4, 5, 6]:
            raise ValueError("Expected the six registered Kelmarsh turbines")
        grid = timeline[timeline < self.bounds["sealed_test_start"]]
        frames = []
        for turbine in ids:
            unit = frame[frame.turbine_id == turbine].set_index("date").reindex(grid)
            unit["turbine_id"] = turbine
            unit.index.name = "date"
            frames.append(unit.reset_index())
        frame = pd.concat(frames, ignore_index=True)
        angles = np.deg2rad(frame.nacelle_position_deg)
        # Nacelle heading is an observed compass orientation, NOT yaw error.
        pitch = frame[["pitch_a_deg", "pitch_b_deg", "pitch_c_deg"]].mean(1, skipna=False)
        raw = np.column_stack((frame.wind_speed_mps, frame.ambient_temp_c, np.sin(angles),
                               np.cos(angles), pitch, frame.rotor_speed_rpm, frame.power_kw)).astype(float)
        raw[~np.isfinite(raw)] = np.nan
        self.dates = frame.date.to_numpy(dtype="datetime64[ns]")
        self.turbines = frame.turbine_id.to_numpy(dtype=np.int16)
        fit = self.dates < self.bounds["base_fit_end"]
        self.scaler = StandardScaler().fit(raw[fit])
        if not (np.isfinite(self.scaler.mean_).all() and np.isfinite(self.scaler.scale_).all()):
            raise ValueError("A required input channel has no training-fit observations")
        self.raw = raw
        self.values = self.scaler.transform(raw).astype(np.float32)
        self.input_len, self.horizon = int(input_len), int(horizon)
        self.starts, potential = valid_window_starts(raw, self.dates, self.turbines, input_len, horizon, stride)
        self.audit = {"fold": fold, "rows_before_sealed_test": len(frame),
                      "candidate_starts_before_checks": potential, "eligible_starts": len(self.starts),
                      "boundaries_utc": {key: str(value) for key, value in self.bounds.items()},
                      "feature_columns": FEATURES, "scaler_fit_end_exclusive": str(self.bounds["base_fit_end"]),
                      "scaler_mean": self.scaler.mean_.tolist(), "scaler_scale": self.scaler.scale_.tolist(),
                      "future_weather_validity_required": False, "labels_clipped": False,
                      "input_imputation": False, "sealed_test_evaluated": False,
                      "angle_semantics": "nacelle compass heading sin/cos; no assumed wind/yaw misalignment"}

    def subset(self, name):
        b = self.bounds
        intervals = {
            "fit": (None, b["base_fit_end"]),
            "selection": (b["base_fit_end"], b["base_selection_end"]),
            "memory": (b["base_selection_end"], b["memory_fit_end"]),
            "calibration_early": (b["memory_fit_end"], b["calibration_early_end"]),
            "calibration_late": (b["calibration_early_end"], b["train_end"]),
            "validation": (b["train_end"], b["validation_end"]),
        }
        if name not in intervals:
            raise ValueError("Unknown/forbidden data stage")
        lo, hi = intervals[name]
        first = self.dates[self.starts + self.input_len]
        last = self.dates[self.starts + self.input_len + self.horizon - 1]
        mask = last < hi
        if lo is not None:
            mask &= first >= lo
        if not mask.any():
            raise ValueError(f"No eligible windows for {name}; inspect input missingness")
        return KelmarshWindows(self, self.starts[mask], name)


class KelmarshWindows(Dataset):
    def __init__(self, series, starts, stage):
        self.series, self.window_starts, self.stage = series, np.asarray(starts), stage
        self.data_x = self.data_y = series.values
        self.dates, self.turbines = series.dates, series.turbines
        self.scaler = series.scaler
        self.seq_len, self.pred_len, self.label_len = series.input_len, series.horizon, 0
        self.feature_columns, self.target, self.scale, self.freq = FEATURES, "power", True, "h"
        self.flag = "train" if stage == "fit" else "val"
        self.train_cutoff, self.val_cutoff = series.bounds["base_fit_end"], series.bounds["base_selection_end"]

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, index):
        start = self.window_starts[index]
        end = start + self.seq_len
        return (self.data_x[start:end], self.data_y[end:end + self.pred_len],
                np.zeros((self.seq_len, 1), np.float32), np.zeros((self.pred_len, 1), np.float32))

    def truth_and_anchor(self):
        rows = self.window_starts[:, None] + self.seq_len + np.arange(self.pred_len)
        truth = self.series.raw[rows, -1]
        last = self.series.raw[self.window_starts + self.seq_len - 1, -1]
        return truth, np.repeat(last[:, None], self.pred_len, axis=1)

    def target_times(self):
        rows = self.window_starts[:, None] + self.seq_len + np.arange(self.pred_len)
        return self.dates[rows]

    def physical_context(self, capacity=2050.):
        rows = self.window_starts[:, None] + self.seq_len - 12 + np.arange(12)
        raw = self.series.raw[rows]
        wind, pitch, power = raw[..., 0], raw[..., 4], raw[..., -1]
        ratio = power.mean(1) / capacity
        power_bin = np.digitize(ratio, [.1, .3, .6, .9])
        wind_bin = np.digitize(wind.mean(1), [3., 6., 12.])
        return {"power_ratio": ratio, "wind_mean_mps": wind.mean(1), "wind_std_mps": wind.std(1),
                "power_change_kw": power[:, 6:].mean(1) - power[:, :6].mean(1),
                "wind_change_mps": wind[:, 6:].mean(1) - wind[:, :6].mean(1),
                "wind_max_step_mps": np.abs(np.diff(wind, axis=1)).max(1),
                "pitch_mean_div90": pitch.mean(1) / 90., "pitch_last_div90": pitch[:, -1] / 90.,
                "pitch_change_div90": (pitch[:, 6:].mean(1) - pitch[:, :6].mean(1)) / 90.,
                "nacelle_heading_cos_mean": raw[..., 3].mean(1),
                "nacelle_heading_cos_last": raw[:, -1, 3],
                "nacelle_heading_sin_mean": raw[..., 2].mean(1),
                "nacelle_heading_sin_last": raw[:, -1, 2],
                "power_bin": power_bin, "joint_bin": power_bin * 4 + wind_bin}
