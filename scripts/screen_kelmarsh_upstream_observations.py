#!/usr/bin/env python3
"""Screen external station observations (Met Office MIDAS Open hourly weather, OGL v3) for residual signal.

Read-only and torch-free: never trains a forecaster, never selects an arm or epoch, never opens Kelmarsh or station data
after 2022 (2023 is reserved for a future frozen confirmation, 2024 stays sealed). For each opened year the residual of
the registered validation-chosen forecast (seed mean) is corrected by a linear model of causal station features. The
correction is fitted on one half-year and scored on the other, in both directions (retrospective development only):
  2021: v28 TSMixer_last10            2022: v29 Guard_last10_normal
L2 fits score RMSE, L1 fits (IRLS) score MAE. Primary gains use the SAME intercept + issue-power-regime control;
intercept-only results are descriptive. The farm-context group (other turbines at the same issue time) is
scored identically as a SCADA-internal control. PASS_RULE only decides whether a frozen v30 protocol is worth writing.

Station rule, fixed before any residual is seen: every station listed in the MIDAS Open hourly-weather station metadata
within RADIUS_KM of Kelmarsh whose hourly wind-speed completeness is >= MIN_COMPLETENESS in every year 2016-2022.
Features use the latest observation with ob_time <= issue time - lag (primary lag 1 h; lag 0 h as a sensitivity).

  python scripts/screen_kelmarsh_upstream_observations.py --midas-root <dir with BADC-CSV files> \
      --metadata <station-metadata.csv> --eval-2021 <v28 run>/evaluation --eval-2022 <v29 run>/evaluation \
      --output outputs/audits/upstream_screen/screen.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

KELMARSH = (52.4013, -0.9428)          # centre of the provider static table (52.3988-52.4038 N, 0.9495-0.9361 W)
RATED_KW = 2050.0
YEARS = tuple(range(2016, 2023))
RADIUS_KM = 150.0
MIN_COMPLETENESS = 0.85
MIN_UPWIND_KM = 20.0
UPWIND_HALF_ANGLE = 60.0
LAGS = (1, 0)
ARMS = {2021: "TSMixer_last10", 2022: "Guard_last10_normal"}
SEEDS = (2024, 2025, 2026)
PASS_RULE = {"lag_hours": 1, "years": [2021, 2022], "leads": [2, 3, 4], "min_rmse_gain_pct": 2.0, "min_mae_gain_pct": 2.0,
             "definition": "in BOTH years, the mean over leads 2-4 of the held-out gain over the SAME intercept + "
                           "issue-power-regime correction must reach both thresholds for the upstream group"}
SCREEN_VERSION = "upstream_observation_screen_reviewed_v1"
DATASET = {'name': 'MIDAS Open UK hourly weather observations', 'version': '202607',
           'doi': '10.5285/d04207b551674c07801b7d4e6d883e50', 'licence': 'Open Government Licence v3'}
SOURCES = {
    "catalogue": "https://catalogue.ceda.ac.uk/uuid/d04207b551674c07801b7d4e6d883e50/",
    "format": "https://help.ceda.ac.uk/article/105-badc-csv",
    "qc": "https://help.ceda.ac.uk/article/4982-midas-open-user-guide",
    "variables": "https://artefacts.ceda.ac.uk/badc_datadocs/ukmo-midas/WH_Table.html",
}


# --------------------------------------------------------------------------- MIDAS Open parsing
def read_badc_csv(path):
    """BADC-CSV: metadata lines, a line 'data', the column header, rows, then 'end data'."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    def marker(line):
        return line.strip().strip(',').strip().lower()
    start = next((i for i, line in enumerate(lines) if marker(line) == "data"), None)
    if start is None:
        out = pd.read_csv(path, low_memory=False)
    else:
        end = next((i for i in range(start + 1, len(lines)) if marker(lines[i]) == "end data"), None)
        if end is None:
            raise ValueError(f"{path}: truncated BADC-CSV (no end data)")
        out = pd.read_csv(io.StringIO("\n".join(lines[start + 1:end])), low_memory=False)
        out.attrs['units'] = {row[1].strip().lower(): row[2].strip().lower()
                              for row in csv.reader(lines[:start])
                              if len(row) >= 3 and row[0].strip().lower() == 'unit'}
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def column(frame, *candidates, required=True):
    lower = {str(c).strip().lower(): c for c in frame.columns}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    if required:
        raise ValueError(f"None of {candidates} among columns {list(frame.columns)[:25]}")
    return None


def distance_bearing(lat, lon):
    lat0, lon0, lat1, lon1 = map(np.radians, (KELMARSH[0], KELMARSH[1], np.asarray(lat, float), np.asarray(lon, float)))
    dlat, dlon = lat1 - lat0, lon1 - lon0
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2) ** 2
    distance = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    bearing = np.degrees(np.arctan2(np.sin(dlon) * np.cos(lat1),
                                    np.cos(lat0) * np.sin(lat1) - np.sin(lat0) * np.cos(lat1) * np.cos(dlon))) % 360
    return distance, bearing


def load_metadata(path):
    meta = read_badc_csv(path)
    out = pd.DataFrame({"src_id": pd.to_numeric(meta[column(meta, "src_id")], errors="coerce"),
                        "name": meta[column(meta, "station_name", "src_name", "name", required=False) or column(meta, "src_id")].astype(str),
                        "lat": pd.to_numeric(meta[column(meta, "station_latitude", "latitude", "lat")], errors="coerce"),
                        "lon": pd.to_numeric(meta[column(meta, "station_longitude", "longitude", "lon")], errors="coerce")})
    out = out.dropna(subset=["src_id", "lat", "lon"])
    if ((out.lat.abs() > 90) | (out.lon.abs() > 180)).any():
        raise ValueError('Invalid station coordinates')
    if (out.groupby('src_id')[['lat', 'lon']].nunique() > 1).any().any():
        raise ValueError('Station metadata has conflicting locations for one src_id')
    out = out.drop_duplicates("src_id")
    out["src_id"] = out["src_id"].astype(int)
    out["distance_km"], out["bearing_deg"] = distance_bearing(out["lat"], out["lon"])
    return out[out["distance_km"] <= RADIUS_KM].sort_values("distance_km").reset_index(drop=True)


def load_observations(root, candidates):
    """{src_id: hourly frame (wind_speed, wind_direction, msl_pressure)} from 2016-2022 files only."""
    parts, read, skipped = {}, [], []
    for path in sorted(Path(root).rglob("*.csv")):
        match = re.search(r"_(\d{4})\.csv$", path.name)
        if match is None or "metadata" in path.name.lower():
            continue
        year = int(match.group(1))
        if year not in YEARS or 'sealed' in [x.lower() for x in path.resolve().parts] or 'qc-version-0' in path.parts:
            skipped.append(str(path))                      # never opened
            continue
        release = re.search(r'dv-(\d{6})', path.name)
        if release and release.group(1) != DATASET['version']:
            raise ValueError('Observation release differs from the reviewed MIDAS version')
        data = read_badc_csv(path)
        version = column(data, 'version_num', required=False)
        if version is None:
            raise ValueError(f'{path}: no MIDAS version_num; refusing unidentified observation versions')
        data = data[pd.to_numeric(data[version], errors='coerce') == 1].copy()
        speed_col = column(data, "wind_speed", "mean_wind_speed")
        units = data.attrs.get('units', {})
        unit = units.get(speed_col)
        if unit in ('knots', 'knot', 'kt'):
            speed_factor = 1852 / 3600
        elif unit in ('m/s', 'm s-1', 'ms-1'):
            speed_factor = 1.0
        else:
            raise ValueError(f'{path}: missing or unsupported declared wind-speed unit {unit!r}')
        sid = pd.to_numeric(data[column(data, "src_id")], errors="coerce")
        keep = sid.isin(candidates)
        if not keep.any():
            continue
        frame = pd.DataFrame({"src_id": sid[keep].astype(int),
                              "time": pd.to_datetime(data.loc[keep, column(data, "ob_time", "ob_end_time")], utc=True,
                                                     errors="coerce").dt.tz_localize(None),
                              "wind_speed": pd.to_numeric(data.loc[keep, speed_col], errors="coerce") * speed_factor,
                              "wind_direction": pd.to_numeric(data.loc[keep, column(data, "wind_direction", "mean_wind_dir")],
                                                              errors="coerce")})
        pressure = column(data, "msl_pressure", required=False)
        if pressure and units.get(pressure) not in ('hpa', 'mb', 'millibars'):
            raise ValueError(f'{path}: unsupported or missing pressure unit')
        direction = column(data, 'wind_direction', 'mean_wind_dir')
        if units.get(direction) not in ('degrees', 'degree', 'degrees true', 'degrees_true'):
            raise ValueError(f'{path}: unsupported or missing wind-direction unit')
        frame["msl_pressure"] = pd.to_numeric(data.loc[keep, pressure], errors="coerce") if pressure else np.nan
        if (frame['time'].dropna().dt.year != year).any():
            raise ValueError(f'{path}: row year differs from the filename; no silent filtering')
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame.loc[frame.wind_speed < 0, 'wind_speed'] = np.nan
        frame.loc[(frame.wind_direction < 0) | (frame.wind_direction > 360) | (frame.wind_speed == 0), 'wind_direction'] = np.nan
        frame.loc[frame.msl_pressure <= 0, 'msl_pressure'] = np.nan
        frame = frame[frame["time"].notna() & (frame["time"].dt.minute == 0) & (frame['time'].dt.second == 0)]
        for sid_value, group in frame.groupby("src_id"):
            parts.setdefault(int(sid_value), []).append(group.drop(columns="src_id"))
        read.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     'wind_unit_raw': unit, 'wind_unit_output': 'm/s', 'version_num': 1,
                     'quality_flags': {c: data[c].astype(str).value_counts().to_dict()
                                       for c in ('wind_speed_q', 'wind_direction_q', 'msl_pressure_q') if c in data}})
    out = {}
    for sid_value, groups in parts.items():
        all_rows = pd.concat(groups)
        values = ['wind_speed', 'wind_direction', 'msl_pressure']
        if (all_rows.groupby('time')[values].nunique(dropna=False) > 1).any().any():
            raise ValueError(f'Station {sid_value}: conflicting duplicate observations; no cross-row field mixing')
        frame = all_rows.drop_duplicates('time').set_index('time').sort_index()
        out[sid_value] = frame
    return out, read, skipped


def qualified_stations(meta, observations):
    rows = []
    for _, station in meta.iterrows():
        frame = observations.get(int(station["src_id"]))
        completeness = {}
        for year in YEARS:
            hours = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left")
            completeness[year] = (0.0 if frame is None else
                                  float(frame["wind_speed"].reindex(hours).notna().mean()))
        rows.append({**station.to_dict(), "completeness": completeness,
                     "qualified": all(v >= MIN_COMPLETENESS for v in completeness.values())})
    return rows


# --------------------------------------------------------------------------- features
def station_matrix(observations, stations, times, name):
    return np.column_stack([observations[s["src_id"]][name].reindex(times).to_numpy(dtype=float) for s in stations])


def upstream_features(issue_times, observations, stations, lag):
    """Causal issue-level features from observations at t0 = issue - lag and t0 - 3 h."""
    t0 = pd.DatetimeIndex(issue_times) - pd.Timedelta(hours=lag)
    t3 = t0 - pd.Timedelta(hours=3)
    ws0, ws3 = station_matrix(observations, stations, t0, "wind_speed"), station_matrix(observations, stations, t3, "wind_speed")
    dir0 = station_matrix(observations, stations, t0, "wind_direction")
    p0, p3 = station_matrix(observations, stations, t0, "msl_pressure"), station_matrix(observations, stations, t3, "msl_pressure")
    distance = np.array([s["distance_km"] for s in stations])
    bearing = np.array([s["bearing_deg"] for s in stations])
    x = distance * np.sin(np.radians(bearing))
    y = distance * np.cos(np.radians(bearing))
    nearest = int(np.argmin(distance))
    radians = np.radians(dir0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        regional_from = np.degrees(np.arctan2(np.nanmean(np.sin(radians), 1), np.nanmean(np.cos(radians), 1))) % 360
    wind_from = np.where(np.isfinite(dir0[:, nearest]), dir0[:, nearest], regional_from)
    angle = np.abs(((bearing[None, :] - wind_from[:, None] + 180) % 360) - 180)
    upwind = (angle <= UPWIND_HALF_ANGLE) & (distance[None, :] >= MIN_UPWIND_KM)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        up_ws = np.nanmean(np.where(upwind, ws0, np.nan), 1)
        up_dws = np.nanmean(np.where(upwind, ws0 - ws3, np.nan), 1)
        regional_ws, regional_dws = np.nanmean(ws0, 1), np.nanmean(ws0 - ws3, 1)
    gradient = np.full((len(t0), 3), np.nan)
    for i in range(len(t0)):
        ok = np.isfinite(p0[i]) & np.isfinite(p3[i])
        if ok.sum() >= 5:
            design = np.c_[np.ones(ok.sum()), x[ok], y[ok]]
            now = np.linalg.lstsq(design, p0[i, ok], rcond=None)[0]
            before = np.linalg.lstsq(design, p3[i, ok], rcond=None)[0]
            gradient[i] = [now[1], now[2], np.hypot(now[1], now[2]) - np.hypot(before[1], before[2])]
    features = np.column_stack([up_ws, up_dws, up_ws - ws0[:, nearest], ws0[:, nearest], ws0[:, nearest] - ws3[:, nearest],
                                regional_ws, regional_dws, p0[:, nearest] - p3[:, nearest], gradient])
    names = ["upwind_ws", "upwind_dws3", "upwind_minus_local", "local_ws", "local_dws3", "regional_ws", "regional_dws3",
             "local_dp3", "grad_east", "grad_north", "grad_magnitude_change3"]
    return features, names


def farm_features(issue_key, turbines, last10, last_hour):
    unique, inverse = np.unique(issue_key, return_inverse=True)
    grid_l, grid_h = np.full((len(unique), 7), np.nan), np.full((len(unique), 7), np.nan)
    grid_l[inverse, turbines], grid_h[inverse, turbines] = last10, last_hour
    count = np.isfinite(grid_l[:, 1:]).sum(1)[inverse]
    others = (np.nansum(grid_l[:, 1:], 1)[inverse] - last10) / np.maximum(count - 1, 1)
    features = np.column_stack([others - last10, np.nanmean(grid_l[:, 1:], 1)[inverse] - np.nanmean(grid_h[:, 1:], 1)[inverse], others])
    features[count < 4] = np.nan
    return features, ["others_minus_own_last10", "farm_last10_minus_last_hour", "others_last10"]


# --------------------------------------------------------------------------- out-of-sample correction
def l1_fit(design, target, iterations=60):
    coef = np.linalg.lstsq(design, target, rcond=None)[0]
    for _ in range(iterations):
        weight = 1.0 / np.maximum(np.abs(target - design @ coef), 1e-3)
        root = np.sqrt(weight)
        coef = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)[0]
    return coef


def out_of_sample(design, error, halves, loss):
    if loss not in ('l1', 'l2'):
        raise ValueError('Unknown correction loss')
    if set(np.unique(halves)) != {0, 1}:
        raise ValueError('Both chronological half-years must have usable rows')
    corrected = np.empty_like(error)
    for train_half in (0, 1):
        fit, score = halves == train_half, halves != train_half
        if fit.sum() <= design.shape[1]:
            raise ValueError('Too few fit rows for the specified linear correction')
        # Only training-half features determine numerical normalization.
        centre, scale = design[fit].mean(0), design[fit].std(0)
        centre[0], scale[0] = 0, 1   # retain the intercept
        scale[scale < 1e-12] = 1
        fit_x, score_x = (design[fit]-centre)/scale, (design[score]-centre)/scale
        coef = l1_fit(fit_x, error[fit]) if loss == "l1" else np.linalg.lstsq(fit_x, error[fit], rcond=None)[0]
        corrected[score] = error[score] - score_x @ coef
    return corrected


def score_group(features, error, anchor, halves, issue_times=None):
    """Matched power-regime control, conditional coverage and forward/reverse half-year scores.

    This is a bidirectional DEVELOPMENT screen, not a deployable forward validation result.
    MAE and RMSE use different fitted correction coefficients; passing does not establish that
    one final forecaster jointly improves both losses.
    """
    features, error, anchor, halves = map(np.asarray, (features, error, anchor, halves))
    if features.ndim != 2 or error.ndim != 2 or error.shape[1] != 4 or not len(error):
        raise ValueError('Expected nonempty features and four-lead errors')
    if not (len(features) == len(error) == len(anchor) == len(halves)) or not np.isfinite(error).all() or not np.isfinite(anchor).all():
        raise ValueError('Misaligned or nonfinite screening inputs')
    if anchor.ndim != 1 or halves.ndim != 1 or not np.isin(halves, (0, 1)).all():
        raise ValueError('Expected scalar issue-power anchor and binary half-year labels')
    eligible = np.ones(len(error), bool)
    if issue_times is not None:
        issue = pd.DatetimeIndex(issue_times)
        if len(issue) != len(error) or len(set(issue.year)) != 1:
            raise ValueError('Issue times must align and belong to one development year')
        boundary = pd.Timestamp(year=int(issue.year[0]), month=7, day=1)
        # Drop crossing targets so the halves never reuse the same target timestamps.
        eligible = (issue + pd.Timedelta(hours=4) < boundary) | (issue >= boundary)
    ok = np.isfinite(features).all(1) & eligible
    mid = ((anchor >= .1 * RATED_KW) & (anchor < .9 * RATED_KW)).astype(float)[:, None]
    design = np.c_[np.ones(len(error)), mid, features, features * mid][ok]
    matched = np.c_[np.ones(len(error)), mid][ok]
    intercept = np.ones((ok.sum(), 1))
    rows = []
    for lead in range(error.shape[1]):
        e = error[ok, lead]
        base_l2 = out_of_sample(intercept, e, halves[ok], "l2")
        base_l1 = out_of_sample(intercept, e, halves[ok], "l1")
        matched_l2 = out_of_sample(matched, e, halves[ok], 'l2')
        matched_l1 = out_of_sample(matched, e, halves[ok], 'l1')
        new_l2 = out_of_sample(design, e, halves[ok], "l2")
        new_l1 = out_of_sample(design, e, halves[ok], "l1")
        rmse = lambda r: float(np.sqrt(np.mean(r ** 2)))  # noqa: E731
        mae = lambda r: float(np.mean(np.abs(r)))         # noqa: E731
        def gain(candidate, reference, metric):
            denom = metric(reference)
            return float(100*(1-metric(candidate)/denom)) if denom > 0 else None
        rows.append({"lead": lead + 1, "windows": int(ok.sum()), "rmse_raw": rmse(e), "mae_raw": mae(e),
                     "rmse_bias_only": rmse(base_l2), "mae_bias_only": mae(base_l1),
                     'rmse_matched_control': rmse(matched_l2), 'mae_matched_control': mae(matched_l1),
                     "rmse_corrected": rmse(new_l2), "mae_corrected": mae(new_l1),
                     'rmse_gain_vs_bias_only_pct': gain(new_l2, base_l2, rmse),
                     'mae_gain_vs_bias_only_pct': gain(new_l1, base_l1, mae),
                     "rmse_gain_pct": gain(new_l2, matched_l2, rmse),
                     "mae_gain_pct": gain(new_l1, matched_l1, mae),
                     'directions': {name: {'windows': int((halves[ok] == half).sum()),
                                           'rmse_gain_pct': gain(new_l2[halves[ok] == half], matched_l2[halves[ok] == half], rmse),
                                           'mae_gain_pct': gain(new_l1[halves[ok] == half], matched_l1[halves[ok] == half], mae)}
                                    for half, name in ((1, 'early_fit_late_score_forward'), (0, 'late_fit_early_score_reverse'))}})
    return {"coverage": float(ok.mean()), 'boundary_purged_windows': int((~eligible).sum()),
            'scope': 'complete-feature windows only; both half-years pooled, not independent confirmation',
            'rows_per_half': [int(((halves == h) & ok).sum()) for h in (0, 1)], 'leads': rows}


def load_year(folder, arm, year=None):
    folder = Path(folder)
    if (year is not None and year not in (2021, 2022)) or 'sealed' in [p.lower() for p in folder.resolve().parts]:
        raise ValueError('Only registered 2021/2022 development evaluations may be opened')
    completed = json.loads((folder.parent/'completed.json').read_text())
    def checked(name):
        path = folder/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != completed.get(f'evaluation/{name}'):
            raise ValueError(f'Unregistered or modified prediction input: {path}')
        return path
    with np.load(checked('test_cohort.npz'), allow_pickle=False) as saved:
        cohort = {k: saved[k] for k in ("truth", "times", "turbines", "PersistenceLast10", "PersistenceHour", 'window_starts')}
    preds = []
    for seed in SEEDS:
        with np.load(checked(f'{arm}_s{seed}.npz'), allow_pickle=False) as saved:
            if not np.array_equal(saved['window_starts'], cohort['window_starts']):
                raise ValueError('Prediction/cohort window identity differs')
            if saved['prediction'].shape != cohort['truth'].shape or not np.isfinite(saved['prediction']).all():
                raise ValueError('Prediction dimensions or finiteness differ')
            preds.append(saved['prediction'])
    prediction = np.mean(preds, axis=0)
    if cohort['truth'].ndim != 2 or cohort['truth'].shape[1] != 4 or not np.isfinite(cohort['truth']).all():
        raise ValueError('Expected finite four-lead truth')
    if not np.isin(cohort['turbines'], np.arange(1,7)).all() or len(np.unique(cohort['window_starts'])) != len(prediction):
        raise ValueError('Invalid turbine or duplicate window identity')
    n = len(prediction)
    if (cohort['times'].shape != (n, 4) or cohort['turbines'].shape != (n,)
            or any(cohort[key].shape != (n, 4) or not np.isfinite(cohort[key]).all()
                   for key in ('PersistenceLast10', 'PersistenceHour'))):
        raise ValueError('Cohort history, timestamps or turbine dimensions differ')
    issue = pd.DatetimeIndex(cohort["times"][:, 0]) - pd.Timedelta(hours=1)
    if issue.year.min() < min(YEARS) or issue.year.max() > max(YEARS):
        raise ValueError(f"{folder}: issue times outside {YEARS[0]}-{YEARS[-1]}")
    if year is not None and (not (issue.year == year).all() or not (pd.DatetimeIndex(cohort['times'].ravel()).year == year).all()):
        raise ValueError('Prediction year differs from the requested development year')
    expected_times = issue.to_numpy()[:, None] + np.arange(1,5)[None, :]*np.timedelta64(1,'h')
    if not np.array_equal(expected_times, cohort['times']):
        raise ValueError('Target lead timestamps are misaligned')
    return cohort, prediction, issue


def screen(midas_root, metadata, evaluations):
    meta = load_metadata(metadata)
    observations, files_read, files_skipped = load_observations(midas_root, set(meta["src_id"]))
    station_rows = qualified_stations(meta, observations)
    stations = [s for s in station_rows if s["qualified"]]
    if len(stations) < 5:
        raise ValueError(f"Only {len(stations)} qualified stations within {RADIUS_KM} km; inspect the download")
    result = {'screen_version': SCREEN_VERSION, 'sources': SOURCES, 'dataset': DATASET,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'metadata_sha256': hashlib.sha256(Path(metadata).read_bytes()).hexdigest(),
              'availability_limit': 'archival qc-version-1 values, NOT a verified real-time vintage; 1h observation lag does not establish publication availability',
              'interpretation': 'retrospective bidirectional development screen; L1 and L2 corrections differ; no accuracy/causal information ceiling claim',
              "rule": {"radius_km": RADIUS_KM, "min_completeness": MIN_COMPLETENESS, "years": list(YEARS),
                       "upwind_half_angle_deg": UPWIND_HALF_ANGLE, "min_upwind_km": MIN_UPWIND_KM, "lags_hours": list(LAGS),
                       "pass_rule": PASS_RULE, "arms": {str(k): v for k, v in ARMS.items()}},
              "stations": station_rows, "files_read": files_read, "files_skipped_unopened": files_skipped, "years": {}}
    for year, folder in evaluations.items():
        cohort, prediction, issue = load_year(folder, ARMS[year], year)
        error = prediction - cohort["truth"]
        anchor = cohort["PersistenceLast10"][:, 0]
        halves = np.asarray(issue.month >= 7, dtype=int)
        unique, inverse = np.unique(issue.asi8, return_inverse=True)
        groups = {}
        farm, farm_names = farm_features(issue.asi8, cohort["turbines"], anchor, cohort["PersistenceHour"][:, 0])
        groups["farm_control"] = score_group(farm, error, anchor, halves, issue) | {"features": farm_names}
        for lag in LAGS:
            issue_level, names = upstream_features(pd.DatetimeIndex(unique), observations, stations, lag)
            weather = issue_level[inverse]
            groups[f"upstream_lag{lag}h"] = score_group(weather, error, anchor, halves, issue) | {"features": names}
            matched_farm = farm.copy()
            matched_farm[~np.isfinite(weather).all(1)] = np.nan
            groups[f'farm_control_weather_lag{lag}h_queue'] = score_group(matched_farm, error, anchor, halves, issue) | {'features': farm_names}
        result["years"][str(year)] = {"arm": ARMS[year], "windows": int(len(error)), "groups": groups}
        result['years'][str(year)]['input_identity'] = {
            'completed_sha256': hashlib.sha256((Path(folder).parent/'completed.json').read_bytes()).hexdigest(),
            'registered_files': {name: value for name, value in json.loads((Path(folder).parent/'completed.json').read_text()).items()
                                 if name.startswith('evaluation/') and (name.endswith('test_cohort.npz') or any(f'{ARMS[year]}_s{s}.npz' in name for s in SEEDS))}}
    passed = []
    for year in PASS_RULE["years"]:
        leads = [r for r in result["years"][str(year)]["groups"][f"upstream_lag{PASS_RULE['lag_hours']}h"]["leads"]
                 if r["lead"] in PASS_RULE["leads"]]
        if len(leads) != len(PASS_RULE['leads']) or any(r[key] is None for r in leads for key in ('rmse_gain_pct', 'mae_gain_pct')):
            raise ValueError('Primary gain undefined; screen is INVALID, not a pass or negative result')
        rmse_gain, mae_gain = np.mean([r["rmse_gain_pct"] for r in leads]), np.mean([r["mae_gain_pct"] for r in leads])
        passed.append(bool(np.isfinite(rmse_gain) and np.isfinite(mae_gain) and rmse_gain >= PASS_RULE["min_rmse_gain_pct"] and mae_gain >= PASS_RULE["min_mae_gain_pct"]))
        result["years"][str(year)]["primary_mean_gain_pct"] = {"rmse": float(rmse_gain), "mae": float(mae_gain)}
    result["passed"] = bool(all(passed))
    result['status'] = 'SCREENED'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--midas-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--eval-2021", type=Path, required=True, help="v28 run evaluation/ directory")
    parser.add_argument("--eval-2022", type=Path, required=True, help="v29 run evaluation/ directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = screen(args.midas_root, args.metadata, {2021: args.eval_2021, 2022: args.eval_2022})
    except (ValueError, FileNotFoundError) as exc:
        result = {'status': 'INVALID_INPUT', 'passed': None, 'reason': str(exc), 'screen_version': SCREEN_VERSION}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix+'.tmp')
    temp.write_text(json.dumps(result, indent=2, default=float, allow_nan=False), encoding="utf-8")
    temp.replace(args.output)
    if result['status'] != 'SCREENED':
        print(json.dumps(result))
        return 2
    print(f"qualified stations: {sum(s['qualified'] for s in result['stations'])} of {len(result['stations'])} within {RADIUS_KM} km")
    for year, block in result["years"].items():
        for group, scores in block["groups"].items():
            def percentage(value):
                return 'undefined' if value is None else f'{value:+.1f}%'
            cells = " | ".join(f"h{r['lead']} RMSE {percentage(r['rmse_gain_pct'])} MAE {percentage(r['mae_gain_pct'])}" for r in scores["leads"])
            print(f"{year} {group:16s} coverage {scores['coverage']:.3f} | {cells}")
    print(f"PASS_RULE {PASS_RULE['definition']}: passed={result['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
