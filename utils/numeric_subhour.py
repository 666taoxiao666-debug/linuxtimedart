"""Causal end-of-hour state for forecasting unchanged hourly power averages.

The source tick at issue-10min is conservatively available at issue (even if
source timestamps denote interval starts). No tick from the target hour is used.
"""
from __future__ import annotations

import csv
import re
import zipfile

import numpy as np
import pandas as pd

from utils.numeric_foundation import observed_anchor


def load_power_ticks(path, *, start, stop):
    """Read only the declared prefix [start,stop); never construct suffix features."""
    lo, hi = pd.Timestamp(start), pd.Timestamp(stop)
    if lo.tzinfo is not None:
        lo = lo.tz_convert("UTC").tz_localize(None)
    if hi.tzinfo is not None:
        hi = hi.tz_convert("UTC").tz_localize(None)
    if not lo < hi:
        raise ValueError("Nonempty raw history prefix required")
    tables = {}
    with zipfile.ZipFile(path) as archive:
        members = sorted(n for n in archive.namelist() if n.startswith("Turbine_Data_"))
        if len(members) != 6:
            raise ValueError("Expected exactly six source turbine members")
        for name in members:
            match = re.search(r"Kelmarsh_(\d+)_", name)
            if match is None:
                raise ValueError("Unexpected source turbine identity")
            unit = int(match.group(1))
            if unit in tables:
                raise ValueError("Duplicate source turbine identity")
            with archive.open(name) as stream:
                for line in stream:
                    text = line.decode("utf-8-sig")
                    if text.startswith("# Date and time,"):
                        columns = next(csv.reader([text[2:].strip()]))
                        break
                else:
                    raise ValueError("Source header absent")
                chunks, previous = [], None
                for chunk in pd.read_csv(stream, names=columns, usecols=["Date and time", "Power (kW)"],
                                         dtype={"Date and time": "string", "Power (kW)": "string"}, chunksize=10000):
                    times = pd.to_datetime(chunk["Date and time"], utc=True).dt.tz_localize(None)
                    if times.isna().any() or not times.is_monotonic_increasing or times.duplicated().any():
                        raise ValueError("Raw times must be strictly increasing")
                    if previous is not None and times.iloc[0] <= previous:
                        raise ValueError("Overlapping raw chunks")
                    previous = times.iloc[-1]
                    keep = (times >= lo) & (times < hi)
                    # Parse power values only after restricting to the permitted prefix.
                    chunks.append(pd.Series(pd.to_numeric(chunk.loc[keep, "Power (kW)"], errors="raise").to_numpy(dtype=float, na_value=np.nan),
                                            index=pd.DatetimeIndex(times[keep])))
                    if times.iloc[-1] >= hi:
                        break
            values = pd.concat(chunks) if chunks else pd.Series(dtype=float)
            if len(values) and not ((values.index.minute % 10 == 0) & (values.index.second == 0)
                                    & (values.index.microsecond == 0) & (values.index.nanosecond == 0)).all():
                raise ValueError("Off-grid raw observations")
            tables[unit] = values
    if set(tables) != set(range(1, 7)):
        raise ValueError("Wrong source turbine set")
    return tables


def end_state_delta(dataset, tables):
    """One scalar/window, aligned by BOTH turbine and clock; no target access."""
    rows = dataset.window_starts + dataset.seq_len - 1
    issues = dataset.dates[rows].astype("datetime64[ns]")
    units = dataset.turbines[rows]
    anchor = observed_anchor(dataset)[:, 0]
    last, max_error = np.empty(len(rows)), 0.
    for unit in np.unique(units):
        mask = units == unit
        ticks = issues[mask, None] - np.arange(60, 0, -10)[None] * np.timedelta64(1, "m")
        values = tables[int(unit)].reindex(pd.DatetimeIndex(ticks.ravel())).to_numpy().reshape(-1, 6)
        if not np.isfinite(values).all():
            raise ValueError("Missing raw tick on frozen cohort; do not drop or impute windows")
        error = np.abs(values.mean(1) - anchor[mask])
        if np.any(error > 1e-8):
            raise ValueError("Raw hourly reconstruction differs from frozen anchor")
        max_error = max(max_error, float(error.max(initial=0.)))
        last[mask] = values[:, -1]
    return last - anchor, dict(windows=len(rows), reconstruction_max_error_kw=max_error,
        first_issue=str(issues.min()), last_issue=str(issues.max()),
        max_source_tick=str(issues.max() - np.timedelta64(10, "m")),
        availability_rule="source_tick+10min<=issue; no tick from target hour",
        inputs_added=["last_10min_power_minus_last_hour_mean_kw"], labels_modified=False)


def fit_coefficients(dataset, tables):
    x, audit = end_state_delta(dataset, tables)
    truth, anchor = dataset.truth_and_anchor()
    denominator = float(x @ x)
    coefficients = np.zeros(dataset.pred_len) if denominator == 0 else np.clip(x @ (truth - anchor) / denominator, 0., 1.)
    return coefficients, audit


def predict(coefficients, dataset, tables):
    coefficients = np.asarray(coefficients, float)
    if coefficients.shape != (dataset.pred_len,) or not np.isfinite(coefficients).all() or np.any((coefficients < 0) | (coefficients > 1)):
        raise ValueError("One finite [0,1] coefficient per lead required")
    x, _ = end_state_delta(dataset, tables)
    return observed_anchor(dataset) + x[:, None] * coefficients
