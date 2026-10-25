"""Pure (torch-free) Kelmarsh 10-minute -> hourly enrichment and operating-state flags.

Two separate products, never mixed:

1. Extra INPUT features, all observable at the hour-end label time (the forecast issue time):
   end-of-hour tick (minute 50) power / wind / rotor / pitch, within-hour std/min/max, yaw-misalignment sin/cos,
   available-capacity fraction, stopped/derated tick counts.
2. Operating-state FLAGS used only to define the "normal operation" cohort (training samples and the primary evaluation
   cohort). They use the hour's own operational signals, never forecast errors, and every threshold is declared in
   ``RULES`` before any model sees the data. The power curve behind the underperformance flag is fitted only on the
   declared curve years and then applied unchanged to every year.

Hour convention matches ``scripts/acquire_hourly_wind_datasets.hourly_scada``: bins [t, t+1h) labelled t+1h; the
minute-50 tick is inside the bin and available at the label under either tick-start or tick-end timestamp semantics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RULES = {
    "stop_wind_mps": 5.0,                 # nacelle wind at/above this (well above the 3 m/s cut-in, robust to lulls) ...
    "stop_power_frac": 0.01,              # ... with power at/below 1% of rated = stopped tick
    "derate_frac": 0.98,                  # available capacity below 98% of rated = derated/limited tick
    "curve_bin_mps": 0.5,
    "curve_max_mps": 30.0,
    "curve_min_samples": 20,
    "underperf_ratio": 0.5,               # hourly power below 50% of the curve ...
    "underperf_min_expected_frac": 0.10,  # ... where the curve expects at least 10% of rated
    "end_tick_minute": 50,
}

REQUIRED_TICK_COLUMNS = ("wind_speed_mps", "power_kw", "nacelle_position_deg", "ambient_temp_c", "rotor_speed_rpm",
                         "pitch_a_deg", "pitch_b_deg", "pitch_c_deg")
OPTIONAL_TICK_COLUMNS = ("wind_direction_deg", "yaw_bearing_deg", "power_setpoint_kw", "available_capacity_kw",
                         "data_availability_source")
FLAG_COLUMNS = ("flag_incomplete_power", "flag_incomplete_state", "flag_stopped", "flag_derated", "flag_underperformance")


def validate_tick_index(index):
    """Refuse off-grid/duplicate/naive ticks before either hourly aggregation."""
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or index.hasnans:
        raise ValueError("Ticks require valid timezone-aware timestamps")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Ticks must be unique and sorted")
    if not (index == index.floor("10min")).all():
        raise ValueError("Off-grid SCADA timestamps require source inspection")


def hour_label(index):
    """Hour-end label of a left-closed hourly bin for each tick timestamp."""
    return index.floor("h") + pd.Timedelta(hours=1)


def available_capacity_share(ticks):
    power_valid = ticks["power_kw"].notna()
    if "available_capacity_kw" not in ticks or not power_valid.any():
        return 0.0
    return float((ticks["available_capacity_kw"].notna() & power_valid).sum() / power_valid.sum())


def tick_states(ticks, rated_kw, rules=RULES, use_available_capacity=True):
    """Per-tick derived values: pitch mean, stopped/derated indicators (NaN-aware), yaw-misalignment sin/cos."""
    out = pd.DataFrame(index=ticks.index)
    pitch = ticks[["pitch_a_deg", "pitch_b_deg", "pitch_c_deg"]]
    out["pitch_mean_deg"] = pitch.mean(axis=1, skipna=False)
    wind, power = ticks["wind_speed_mps"], ticks["power_kw"]
    valid = wind.notna() & power.notna()
    out["stopped"] = (valid & (wind >= rules["stop_wind_mps"]) & (power <= rules["stop_power_frac"] * rated_kw)).astype(float)
    out.loc[~valid, "stopped"] = np.nan
    if use_available_capacity and "available_capacity_kw" in ticks:
        capacity = ticks["available_capacity_kw"]
        out["derated"] = (capacity < rules["derate_frac"] * rated_kw).astype(float)
        out.loc[capacity.isna(), "derated"] = np.nan
        out["available_capacity_frac"] = capacity / rated_kw
    else:
        out["derated"] = np.nan
        out["available_capacity_frac"] = np.nan
    if "wind_direction_deg" in ticks:
        delta = np.deg2rad(ticks["wind_direction_deg"] - ticks["nacelle_position_deg"])
        out["yaw_error_sin"], out["yaw_error_cos"] = np.sin(delta), np.cos(delta)
    else:
        out["yaw_error_sin"] = out["yaw_error_cos"] = np.nan
    return out


def hourly_enrichment(ticks, rated_kw, rules=RULES, use_available_capacity=True):
    """Extra hourly columns (indexed by hour-end label). ``ticks`` must be on the 10-minute grid, indexed by UTC time."""
    validate_tick_index(ticks.index)
    missing = [c for c in REQUIRED_TICK_COLUMNS if c not in ticks]
    if missing:
        raise ValueError(f"Missing required tick columns: {missing}")
    states = tick_states(ticks, rated_kw, rules, use_available_capacity)
    frame = pd.concat([ticks[["power_kw", "wind_speed_mps", "rotor_speed_rpm"]], states], axis=1)
    groups = frame.groupby(hour_label(frame.index))
    out = pd.DataFrame({
        "power_std10_kw": groups["power_kw"].std(ddof=0),
        "power_min10_kw": groups["power_kw"].min(),
        "power_max10_kw": groups["power_kw"].max(),
        "wind_std10_mps": groups["wind_speed_mps"].std(ddof=0),
        "yaw_error_sin": groups["yaw_error_sin"].mean(),
        "yaw_error_cos": groups["yaw_error_cos"].mean(),
        "available_capacity_frac": groups["available_capacity_frac"].mean(),
        "n_stopped_ticks": groups["stopped"].sum(min_count=1),
        "n_derated_ticks": groups["derated"].sum(min_count=1),
        "n_state_ticks": groups["stopped"].count(),
        "n_capacity_ticks": groups["derated"].count(),
    })
    end = frame[frame.index.minute == int(rules["end_tick_minute"])]
    end_hour = pd.DataFrame({"power_last10_kw": end["power_kw"].to_numpy(), "wind_last10_mps": end["wind_speed_mps"].to_numpy(),
                             "rotor_last10_rpm": end["rotor_speed_rpm"].to_numpy(), "pitch_last10_deg": end["pitch_mean_deg"].to_numpy()},
                            index=hour_label(end.index))
    if end_hour.index.has_duplicates:
        raise ValueError("Duplicate end-of-hour ticks")
    out = out.join(end_hour, how="outer")
    out.index.name = "timestamp_hour_end_utc"
    return out


def state_flags(hourly):
    """Hour-level flags from tick counts (no curve yet). ``hourly`` needs power_kw, n_stopped_ticks, n_derated_ticks."""
    out = pd.DataFrame(index=hourly.index)
    out["flag_incomplete_power"] = hourly["power_kw"].isna()
    # Missing wind/state must not silently become confirmed normal operation.
    out["flag_incomplete_state"] = hourly["n_state_ticks"].fillna(0) != 6
    out["flag_stopped"] = hourly["n_stopped_ticks"].fillna(0) >= 1
    out["flag_derated"] = hourly["n_derated_ticks"].fillna(0) >= 1
    return out


def label_year(timestamps):
    """Calendar year of the hour that a label closes (label - 1h), so 00:00 on 1 January belongs to the old year."""
    return (pd.to_datetime(timestamps) - pd.Timedelta(hours=1)).dt.year


def fit_power_curve(wind, power, years, curve_years, stopped, derated, rated_kw, rules=RULES):
    """Median hourly power per wind bin on curve years only, excluding stopped/derated hours; monotone, capped at rated."""
    wind, power = np.asarray(wind, dtype=float), np.asarray(power, dtype=float)
    years, stopped, derated = np.asarray(years), np.asarray(stopped, dtype=bool), np.asarray(derated, dtype=bool)
    use = np.isin(years, list(curve_years)) & np.isfinite(wind) & np.isfinite(power) & ~stopped & ~derated
    if use.sum() < 10 * rules["curve_min_samples"]:
        raise ValueError("Too few curve-year hours to fit a power curve")
    edges = np.arange(0.0, rules["curve_max_mps"] + rules["curve_bin_mps"], rules["curve_bin_mps"])
    centers = (edges[:-1] + edges[1:]) / 2
    index = np.digitize(wind[use], edges) - 1
    medians, counts = np.full(len(centers), np.nan), np.zeros(len(centers), dtype=int)
    for b in range(len(centers)):
        values = power[use][index == b]
        counts[b] = len(values)
        if len(values) >= rules["curve_min_samples"]:
            medians[b] = np.median(values)
    known = np.isfinite(medians)
    if known.sum() < 5:
        raise ValueError("Power curve has too few populated wind bins")
    filled = np.interp(centers, centers[known], medians[known])
    filled[centers > centers[known].max()] = medians[known][-1]
    monotone = np.minimum(np.maximum.accumulate(np.maximum(filled, 0.0)), rated_kw)
    return {"centers_mps": centers.tolist(), "median_kw": [None if not np.isfinite(v) else float(v) for v in medians],
            "curve_kw": monotone.tolist(), "counts": counts.tolist(), "curve_years": sorted(int(y) for y in curve_years),
            "hours_used": int(use.sum()), "rule": "median hourly power per bin on curve years, stopped/derated excluded, "
            "linear fill, cumulative max, capped at rated"}


def expected_power(curve, wind):
    return np.interp(np.asarray(wind, dtype=float), curve["centers_mps"], curve["curve_kw"], left=0.0,
                     right=curve["curve_kw"][-1])


def underperformance_flag(wind, power, curve, rated_kw, rules=RULES):
    wind, power = np.asarray(wind, dtype=float), np.asarray(power, dtype=float)
    expected = expected_power(curve, wind)
    valid = np.isfinite(wind) & np.isfinite(power)
    return valid & (expected >= rules["underperf_min_expected_frac"] * rated_kw) & (power < rules["underperf_ratio"] * expected)


def normal_operation(flags):
    """Normal-operation hour: complete power label and none of stopped / derated / underperformance."""
    return ~flags[list(FLAG_COLUMNS)].any(axis=1)


def window_cohort(normal_hour, starts, input_len, horizon):
    """True for windows whose four TARGET hours are all normal-operation hours (history may contain abnormal hours)."""
    normal = np.asarray(normal_hour, dtype=bool)
    cumulative = np.concatenate(([0], np.cumsum(~normal)))
    starts = np.asarray(starts, dtype=np.int64)
    first, end = starts + input_len, starts + input_len + horizon
    return (cumulative[end] - cumulative[first]) == 0
