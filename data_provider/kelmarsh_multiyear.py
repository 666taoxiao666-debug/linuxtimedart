"""Multi-year Kelmarsh windows on the enriched hourly CSV (scripts/prepare_kelmarsh_multiyear.py).

* Year-based splits (default train 2016-2019, validation 2020, test 2021); a window belongs to a split only if all of
  its target hours fall in that split's years; its history may reach back into the previous year (observed data).
* Channel order keeps power LAST (target slot -1), so the existing DLinear / TSMixer / anchored models work unchanged.
  Optional extra channels are all observable at the issue time: end-of-hour tick, within-hour spread, available
  capacity, stopped-tick share, yaw misalignment, hour-of-day / day-of-year, farm-mean power and wind.
* Two cohorts per split: ``normal`` (ex-post conditional cohort using all four TARGET hours' operating state) and
  ``all`` (default, every eligible window with original labels). Only the latter describes deployment accuracy.
* Scaler: mean/std (ddof 0, NaN ignored, constant channel -> 1) of training-year rows only.
* Preparation schema/hash and curve-fit years are checked against the requested training years.
Labels are never modified, clipped or imputed here.
"""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

try:  # the training host has torch; the pure logic stays importable without it
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    Dataset = object

from utils.kelmarsh_enrichment import window_cohort

BASE_FEATURES = ["Wspd", "Etmp", "Ndir_sin", "Ndir_cos", "Pab_mean", "rotor_speed"]
EXTRA_FEATURES = ("power_last10", "wind_last10", "power_std10", "wind_std10", "available_capacity_frac", "stopped_frac",
                  "yaw_error_sin", "yaw_error_cos", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
                  "farm_power_mean", "farm_wind_mean")
# Proposed split only; a training protocol must freeze this before any real fit/evaluation.
DEFAULT_SPLITS = {"train": (2016, 2017, 2018, 2019), "validation": (2020,), "test": (2021,)}
COHORTS = ("normal", "all")
FARM_MIN_TURBINES = 3


def _naive_utc(value):
    stamp = pd.Timestamp(value)
    return stamp.tz_convert("UTC").tz_localize(None) if stamp.tzinfo is not None else stamp


def valid_window_starts(values, dates, turbines, input_len, horizon, stride):
    """Same rule as data_provider.kelmarsh_hourly.valid_window_starts: finite history in every channel, finite power
    targets, one turbine, consecutive hours; never filters on future inputs."""
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


def _extra_column(name, unit, farm):
    if name == "power_last10":
        return unit["power_last10_kw"]
    if name == "wind_last10":
        return unit["wind_last10_mps"]
    if name == "power_std10":
        return unit["power_std10_kw"]
    if name == "wind_std10":
        return unit["wind_std10_mps"]
    if name == "available_capacity_frac":
        return unit["available_capacity_frac"]
    if name == "stopped_frac":
        return unit["n_stopped_ticks"] / 6.0
    if name in ("yaw_error_sin", "yaw_error_cos"):
        return unit[name]
    hours = unit.index.hour.to_numpy()
    days = unit.index.dayofyear.to_numpy()
    if name == "hour_sin":
        return pd.Series(np.sin(2 * np.pi * hours / 24), index=unit.index)
    if name == "hour_cos":
        return pd.Series(np.cos(2 * np.pi * hours / 24), index=unit.index)
    if name == "doy_sin":
        return pd.Series(np.sin(2 * np.pi * days / 366), index=unit.index)
    if name == "doy_cos":
        return pd.Series(np.cos(2 * np.pi * days / 366), index=unit.index)
    if name == "farm_power_mean":
        return farm["power"].reindex(unit.index)
    if name == "farm_wind_mean":
        return farm["wind"].reindex(unit.index)
    raise ValueError(f"Unknown extra feature {name!r}")


class KelmarshMultiYearSeries:
    def __init__(self, path_or_frame, *, splits=None, input_len=24, horizon=4, stride=3, extras=(), start=None, end=None,
                 preparation=None):
        if any(int(v) != v or int(v) <= 0 for v in (input_len, horizon, stride)):
            raise ValueError("input_len, horizon and stride must be positive integers")
        input_len, horizon, stride = int(input_len), int(horizon), int(stride)
        splits = {k: tuple(int(y) for y in v) for k, v in (splits or DEFAULT_SPLITS).items()}
        if "train" not in splits or any(not ys for ys in splits.values()):
            raise ValueError("A train split is required (the scaler is fitted on it)")
        years = [y for v in splits.values() for y in v]
        if len(years) != len(set(years)):
            raise ValueError("Split years overlap")
        heldout = [y for k, ys in splits.items() if k != "train" for y in ys]
        if heldout and max(splits["train"]) >= min(heldout):
            raise ValueError("Training years must precede every held-out year")
        if "validation" in splits and "test" in splits and max(splits["validation"]) >= min(splits["test"]):
            raise ValueError("Validation years must precede test years")
        unknown = [e for e in extras if e not in EXTRA_FEATURES]
        if unknown:
            raise ValueError(f"Unknown extras {unknown}")
        if len(extras) != len(set(extras)):
            raise ValueError("Duplicate extra channels")
        is_frame = isinstance(path_or_frame, pd.DataFrame)
        if preparation is None and not is_frame:
            sidecar = Path(path_or_frame).parent.parent / "metadata/preparation.json"
            if sidecar.exists():
                preparation = json.loads(sidecar.read_text(encoding="utf-8"))
        if not preparation or preparation.get("schema_version") != 2:
            raise ValueError("Audited preparation metadata (schema 2) is required")
        curve_years = set(preparation.get("curve_years", []))
        if not curve_years or not curve_years <= set(splits["train"]):
            raise ValueError("Cohort power-curve fit years must all belong to train")
        if not is_frame:
            digest = hashlib.sha256()
            with Path(path_or_frame).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != preparation.get("prepared", {}).get("sha256"):
                raise ValueError("Prepared CSV hash does not match its provenance")
        frame = path_or_frame.copy() if isinstance(path_or_frame, pd.DataFrame) else pd.read_csv(path_or_frame)
        frame["date"] = pd.to_datetime(frame["timestamp_hour_end_utc"], utc=True).dt.tz_localize(None)
        if frame["date"].isna().any() or not (frame["date"] == frame["date"].dt.floor("h")).all():
            raise ValueError("Valid exact hourly labels are required")
        # Do not let undeclared years affect grid extent, stride phase or spatial features.
        # The history preceding an explicitly requested year remains observed input.
        allowed = pd.Series(False, index=frame.index)
        for year in years:
            lo = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(hours=1 - input_len)
            hi = pd.Timestamp(year=year + 1, month=1, day=1)
            allowed |= frame["date"].between(lo, hi)
        frame = frame[allowed].copy()
        if start is not None:
            frame = frame[frame["date"] >= _naive_utc(start)]
        if end is not None:
            frame = frame[frame["date"] < _naive_utc(end)]
        if frame.empty:
            raise ValueError("No rows in the declared periods")
        frame["turbine_id"] = frame["turbine_id"].astype(int)
        if frame.duplicated(["turbine_id", "date"]).any():
            raise ValueError("Duplicate turbine/hour rows")
        grid = pd.date_range(frame["date"].min(), frame["date"].max(), freq="1h")
        by_date = frame.groupby("date")
        counts = by_date["power_kw"].count()
        farm = {"power": by_date["power_kw"].mean().where(counts >= FARM_MIN_TURBINES),
                "wind": by_date["wind_speed_mps"].mean().where(by_date["wind_speed_mps"].count() >= FARM_MIN_TURBINES)}
        blocks, dates, turbines, normal, last10 = [], [], [], [], []
        self.channels = BASE_FEATURES + list(extras) + ["power"]
        for unit_id in sorted(frame["turbine_id"].unique()):
            unit = frame[frame["turbine_id"] == unit_id].set_index("date").reindex(grid)
            angle = np.deg2rad(unit["nacelle_position_deg"])
            pitch = unit[["pitch_a_deg", "pitch_b_deg", "pitch_c_deg"]].mean(axis=1, skipna=False)
            columns = [unit["wind_speed_mps"], unit["ambient_temp_c"], np.sin(angle), np.cos(angle), pitch, unit["rotor_speed_rpm"]]
            columns += [_extra_column(name, unit, farm) for name in extras]
            columns.append(unit["power_kw"])
            blocks.append(np.column_stack([np.asarray(c, dtype=float) for c in columns]))
            dates.append(grid.to_numpy())
            turbines.append(np.full(len(grid), unit_id, dtype=np.int16))
            normal.append(unit["normal_operation"].map(lambda v: v is True or v == 1 or str(v).lower() == "true").to_numpy(bool))
            last10.append(unit["power_last10_kw"].to_numpy(dtype=float) if "power_last10_kw" in unit
                          else np.full(len(grid), np.nan))
        raw = np.concatenate(blocks)
        raw[~np.isfinite(raw)] = np.nan
        self.raw = raw
        self.power_last10 = np.concatenate(last10)
        self.dates = np.concatenate(dates).astype("datetime64[ns]")
        self.turbines = np.concatenate(turbines)
        self.normal_hour = np.concatenate(normal)
        self.label_year = (pd.DatetimeIndex(self.dates) - pd.Timedelta(hours=1)).year.to_numpy()
        train_rows = np.isin(self.label_year, splits["train"])
        if not train_rows.any():
            raise ValueError("No training-year rows")
        with np.errstate(invalid="ignore"):
            mean, std = np.nanmean(raw[train_rows], axis=0), np.nanstd(raw[train_rows], axis=0)
        if not (np.isfinite(mean).all() and np.isfinite(std).all()):
            raise ValueError("A channel has no finite training-year observation")
        self.scaler_mean, self.scaler_scale = mean, np.where(std == 0, 1.0, std)
        self.values = ((raw - mean) / self.scaler_scale).astype(np.float32)
        self.input_len, self.horizon, self.splits = int(input_len), int(horizon), splits
        self.starts, potential = valid_window_starts(raw, self.dates, self.turbines, input_len, horizon, stride)
        self.split_of = np.full(len(self.starts), "", dtype=object)
        for name, ys in splits.items():
            outside = np.concatenate(([0], np.cumsum(~np.isin(self.label_year, ys))))
            inside = outside[self.starts + input_len + horizon] == outside[self.starts + input_len]
            self.split_of[inside] = name
        self.normal_window = window_cohort(self.normal_hour, self.starts, input_len, horizon)
        self.audit = {"channels": self.channels, "extras": list(extras), "splits": {k: list(v) for k, v in splits.items()},
                      "candidate_starts": int(potential), "eligible_starts": int(len(self.starts)),
                      "windows": {name: {"all": int((self.split_of == name).sum()),
                                         "normal": int(((self.split_of == name) & self.normal_window).sum())}
                                  for name in splits},
                      "scaler_rows": int(train_rows.sum()), "scaler_mean": mean.tolist(), "scaler_scale": self.scaler_scale.tolist(),
                      "normal_hour_share_pct": float(100 * self.normal_hour.mean()),
                      "preparation_schema": preparation["schema_version"], "curve_years": sorted(curve_years),
                      "prepared_hash_verified": not is_frame,
                      "normal_cohort_uses_future_target_state": True,
                      "all_cohort_required_for_deployment_accuracy": True,
                      "labels_modified": False, "imputation": False}

    def windows(self, split, cohort="all"):
        if split not in self.splits:
            raise ValueError(f"Unknown split {split!r}")
        if cohort not in COHORTS:
            raise ValueError(f"Cohort must be one of {COHORTS}")
        mask = self.split_of == split
        if cohort == "normal":
            mask &= self.normal_window
        if not mask.any():
            raise ValueError(f"No {cohort} windows in {split}")
        return MultiYearWindows(self, self.starts[mask], split, cohort)


class MultiYearWindows(Dataset):
    """KelmarshWindows-compatible view: __getitem__ -> (x, y, mark, mark); truth_and_anchor(); target_times()."""

    def __init__(self, series, starts, split, cohort):
        self.series, self.window_starts, self.split, self.cohort = series, np.asarray(starts, dtype=np.int64), split, cohort
        self.data_x = self.data_y = series.values
        self.dates, self.turbines = series.dates, series.turbines
        self.seq_len, self.pred_len, self.label_len = series.input_len, series.horizon, 0
        self.feature_columns, self.target, self.scale, self.freq = series.channels, "power", True, "h"
        self.scaler = SimpleNamespace(mean_=series.scaler_mean, scale_=series.scaler_scale)  # sklearn-like attributes

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, index):
        start = self.window_starts[index]
        end = start + self.seq_len
        return (self.data_x[start:end], self.data_y[end:end + self.pred_len],
                np.zeros((self.seq_len, 1), np.float32), np.zeros((self.pred_len, 1), np.float32))

    def truth_and_anchor(self):
        """Original-scale targets and the classic Persistence (last hourly mean power)."""
        rows = self.window_starts[:, None] + self.seq_len + np.arange(self.pred_len)
        truth = self.series.raw[rows, -1]
        last = self.series.raw[self.window_starts + self.seq_len - 1, -1]
        return truth, np.repeat(last[:, None], self.pred_len, axis=1)

    def last10_anchor(self):
        """Stronger naive baseline: the last observed 10-minute power at the issue time (NaN where that tick is absent)."""
        last = self.series.power_last10[self.window_starts + self.seq_len - 1]
        return np.repeat(last[:, None], self.pred_len, axis=1)

    def target_times(self):
        return self.dates[self.window_starts[:, None] + self.seq_len + np.arange(self.pred_len)]
