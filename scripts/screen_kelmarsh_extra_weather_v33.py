#!/usr/bin/env python3
"""v33 stage S: does unused MIDAS weather (air temperature, dewpoint, total cloud) add residual signal BEYOND the
11 existing upstream features?  Proposal section 14.3 (second revision).

Read-only and torch-free. No forecaster is (re)trained; only the linear / IRLS residual corrections of the original
upstream screen are fitted, so call this a "linear correction screen without retraining the base", not zero fitting.

Fixed before any residual is seen (frozen in this file):
  * year 2021 only (the D1 validation year); residual = v28 TSMixer_last10 seed-mean prediction - truth
    (that model was fitted on 2016-2019, so 2021 is out of sample). No 2022-2024 residual is opened, and nothing
    learnt here may be fed back into D2/D3 evaluation.
  * the 36 stations registered in the completed upstream screen (no re-screening); every opened file must match
    the SHA256 recorded by that screen.
  * fields: air_temperature, dewpoint, cld_ttl_amt_id (A0 coverage 99.5 / 99.1 / 85.3 %). Gust (3.5 %), present-weather
    codes (22 %, manual/automatic code tables differ) and relative humidity (derived from T and Td, no QC column)
    are NOT used. QC rule identical to the existing wind/pressure features: version_num == 1 finite values only.
  * control  = intercept + power regime + the 11 existing upstream features (+ regime interactions) + night flag;
    candidate = control + 7 pre-registered new features (+ regime interactions). Gains are the INCREMENT over the
    control, scored half-year forward and reverse exactly like the original screen.
  * PASS_RULE below. Thermal-only and cloud-only groups are descriptive and never change the decision.

  python scripts/screen_kelmarsh_extra_weather_v33.py --midas-root <MIDAS dir> \
      --screen-json <completed upstream screen.json> --eval-2021 <v28 run>/evaluation --output <json>
  python scripts/screen_kelmarsh_extra_weather_v33.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import screen_kelmarsh_upstream_observations as base  # noqa: E402

SCREEN_VERSION = 'extra_weather_screen_v33_S_r2'
YEAR = 2021
LAG_HOURS = 1
READ_YEARS = (2020, 2021)            # 2020 only for the 3 h look-back of early-January issue times
FIELDS = {'air_temperature': ('degc',), 'dewpoint': ('degc',), 'cld_ttl_amt_id': ('oktas',)}
NEW_FEATURES = ['upwind_dT3', 'local_dT3', 'upwind_minus_local_T', 'local_dewpoint_depression',
                'regional_dewpoint_depression', 'regional_cloud_oktas', 'night_x_regional_cloud']
GROUPS = {'thermal_cloud': NEW_FEATURES, 'thermal': NEW_FEATURES[:5], 'cloud': NEW_FEATURES[5:]}
PRIMARY_GROUP = 'thermal_cloud'
PASS_RULE = {'year': YEAR, 'lag_hours': LAG_HOURS, 'group': PRIMARY_GROUP, 'leads': [1, 2, 3, 4],
             'min_rmse_gain_pct': 1.0, 'min_mae_gain_pct': 1.0, 'min_coverage': 0.80,
             'definition': 'mean over leads 1-4 of the held-out INCREMENTAL gain of control+new over control must reach '
                           'both thresholds, and the lead-mean gain must be > 0 for both metrics in BOTH half-year '
                           'directions; coverage below min_coverage gives INSUFFICIENT_COVERAGE (not a pass)'}


# --------------------------------------------------------------------------- calendar control
def solar_elevation_deg(times, lat=base.KELMARSH[0], lon=base.KELMARSH[1]):
    """NOAA low-precision solar elevation (about 1 degree), UTC input; used only for a night flag."""
    t = pd.DatetimeIndex(times)
    gamma = 2 * np.pi / 365.0 * (t.dayofyear.to_numpy() - 1 + (t.hour.to_numpy() - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) - 0.006758 * np.cos(2 * gamma)
            + 0.000907 * np.sin(2 * gamma) - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    minutes = t.hour.to_numpy() * 60 + t.minute.to_numpy() + eqtime + 4 * lon
    hour_angle = np.radians(minutes / 4.0 - 180.0)
    phi = np.radians(lat)
    cos_zenith = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
    return 90.0 - np.degrees(np.arccos(np.clip(cos_zenith, -1, 1)))


# --------------------------------------------------------------------------- extra MIDAS fields
def clean_extra(data, path):
    """Return a frame (time, air_temperature, dewpoint, cloud) plus cleaning counts for one BADC file."""
    units = data.attrs.get('units', {})
    for name, allowed in FIELDS.items():
        col = base.column(data, name)
        if units.get(col) not in allowed:
            raise ValueError(f'{path}: unsupported or missing unit for {name}: {units.get(col)!r}')
    version = base.column(data, 'version_num', required=False)
    if version is None:
        raise ValueError(f'{path}: no version_num; refusing unidentified observation versions')
    data = data[pd.to_numeric(data[version], errors='coerce') == 1]
    frame = pd.DataFrame({
        'src_id': pd.to_numeric(data[base.column(data, 'src_id')], errors='coerce'),
        'time': pd.to_datetime(data[base.column(data, 'ob_time', 'ob_end_time')], utc=True, errors='coerce').dt.tz_localize(None),
        'air_temperature': pd.to_numeric(data[base.column(data, 'air_temperature')], errors='coerce'),
        'dewpoint': pd.to_numeric(data[base.column(data, 'dewpoint')], errors='coerce'),
        'cloud': pd.to_numeric(data[base.column(data, 'cld_ttl_amt_id')], errors='coerce')})
    frame = frame.replace([np.inf, -np.inf], np.nan)
    counts = {'rows_version1': int(len(frame))}
    bad_t = ~frame.air_temperature.between(-40, 45) & frame.air_temperature.notna()
    bad_td = ~frame.dewpoint.between(-50, 40) & frame.dewpoint.notna()
    frame.loc[bad_t, 'air_temperature'] = np.nan
    frame.loc[bad_td, 'dewpoint'] = np.nan
    supersaturated = (frame.dewpoint > frame.air_temperature + 1.0)
    frame.loc[supersaturated, 'dewpoint'] = np.nan          # physically inconsistent pair; temperature kept
    obscured = frame.cloud == 9
    invalid_cloud = frame.cloud.notna() & ~frame.cloud.isin(range(10))
    frame.loc[obscured | invalid_cloud, 'cloud'] = np.nan  # 9 = sky obscured: missing, NOT overcast
    counts.update({'temperature_out_of_range': int(bad_t.sum()), 'dewpoint_out_of_range': int(bad_td.sum()),
                   'dewpoint_above_temperature_plus_1': int(supersaturated.sum()),
                   'cloud_obscured_9': int(obscured.sum()), 'cloud_invalid_code': int(invalid_cloud.sum())})
    frame = frame[frame.time.notna() & (frame.time.dt.minute == 0) & (frame.time.dt.second == 0)]
    return frame, counts


def load_extra(paths, station_ids):
    """{src_id: hourly frame (air_temperature, dewpoint, cloud)} for the listed (already hash-verified) files."""
    parts, report = {}, []
    for path in paths:
        data = base.read_badc_csv(path)
        frame, counts = clean_extra(data, path)
        frame = frame[frame.src_id.isin(station_ids)]
        for sid, group in frame.groupby('src_id'):
            parts.setdefault(int(sid), []).append(group.drop(columns='src_id'))
        report.append({'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(), **counts})
    out = {}
    for sid, groups in parts.items():
        rows = pd.concat(groups)
        values = ['air_temperature', 'dewpoint', 'cloud']
        if (rows.groupby('time')[values].nunique(dropna=False) > 1).any().any():
            raise ValueError(f'Station {sid}: conflicting duplicate extra-field observations')
        out[sid] = rows.drop_duplicates('time').set_index('time').sort_index()
    return out, report


# --------------------------------------------------------------------------- features
def upwind_mask(observations, stations, t0):
    """Same wind-from / upwind rule as base.upstream_features (nearest-station direction, regional circular fallback)."""
    dir0 = base.station_matrix(observations, stations, t0, 'wind_direction')
    distance = np.array([s['distance_km'] for s in stations])
    bearing = np.array([s['bearing_deg'] for s in stations])
    nearest = int(np.argmin(distance))
    radians = np.radians(dir0)
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        regional_from = np.degrees(np.arctan2(np.nanmean(np.sin(radians), 1), np.nanmean(np.cos(radians), 1))) % 360
    wind_from = np.where(np.isfinite(dir0[:, nearest]), dir0[:, nearest], regional_from)
    angle = np.abs(((bearing[None, :] - wind_from[:, None] + 180) % 360) - 180)
    return (angle <= base.UPWIND_HALF_ANGLE) & (distance[None, :] >= base.MIN_UPWIND_KM), nearest


def extra_features(issue_times, observations, extra, stations, lag):
    issue = pd.DatetimeIndex(issue_times)
    t0 = issue - pd.Timedelta(hours=lag)
    t3 = t0 - pd.Timedelta(hours=3)
    upwind, nearest = upwind_mask(observations, stations, t0)

    def matrix(times, name):
        return np.column_stack([extra[s['src_id']][name].reindex(times).to_numpy(dtype=float)
                                if s['src_id'] in extra else np.full(len(times), np.nan) for s in stations])
    temp0, temp3 = matrix(t0, 'air_temperature'), matrix(t3, 'air_temperature')
    dew0, cloud0 = matrix(t0, 'dewpoint'), matrix(t0, 'cloud')
    night = (solar_elevation_deg(t0) < 0).astype(float)
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        up_dt3 = np.nanmean(np.where(upwind, temp0 - temp3, np.nan), 1)
        up_t0 = np.nanmean(np.where(upwind, temp0, np.nan), 1)
        regional_dpd = np.nanmean(temp0 - dew0, 1)
        regional_cloud = np.nanmean(cloud0, 1)
    features = np.column_stack([up_dt3, temp0[:, nearest] - temp3[:, nearest], up_t0 - temp0[:, nearest],
                                temp0[:, nearest] - dew0[:, nearest], regional_dpd, regional_cloud,
                                night * regional_cloud])
    return features, night, upwind


# --------------------------------------------------------------------------- incremental scoring
def score_increment(control, new, error, anchor, halves, issue_times):
    """Held-out gain of [control + new] over [control], same rows, both half-year directions (development only)."""
    control, new, error, anchor, halves = map(np.asarray, (control, new, error, anchor, halves))
    if not (len(control) == len(new) == len(error) == len(anchor) == len(halves)) or error.shape[1:] != (4,):
        raise ValueError('Misaligned incremental screening inputs')
    issue = pd.DatetimeIndex(issue_times)
    boundary = pd.Timestamp(year=int(issue.year[0]), month=7, day=1)
    eligible = np.asarray((issue + pd.Timedelta(hours=4) < boundary) | (issue >= boundary))
    ok = np.isfinite(control).all(1) & np.isfinite(new).all(1) & eligible
    mid = ((anchor >= .1 * base.RATED_KW) & (anchor < .9 * base.RATED_KW)).astype(float)[:, None]
    matched = np.c_[np.ones(len(error)), mid][ok]
    ctrl = np.c_[np.ones(len(error)), mid, control, control * mid][ok]
    cand = np.c_[np.ones(len(error)), mid, control, control * mid, new, new * mid][ok]
    h = halves[ok]
    rmse = lambda r: float(np.sqrt(np.mean(r ** 2)))  # noqa: E731
    mae = lambda r: float(np.mean(np.abs(r)))         # noqa: E731

    def gain(candidate, reference, metric):
        denom = metric(reference)
        return float(100 * (1 - metric(candidate) / denom)) if denom > 0 else None
    rows = []
    for lead in range(4):
        e = error[ok, lead]
        fits = {name: {loss: base.out_of_sample(design, e, h, loss) for loss in ('l2', 'l1')}
                for name, design in (('matched', matched), ('control', ctrl), ('candidate', cand))}
        rows.append({'lead': lead + 1, 'windows': int(ok.sum()),
                     'rmse_control': rmse(fits['control']['l2']), 'mae_control': mae(fits['control']['l1']),
                     'rmse_candidate': rmse(fits['candidate']['l2']), 'mae_candidate': mae(fits['candidate']['l1']),
                     'rmse_gain_pct': gain(fits['candidate']['l2'], fits['control']['l2'], rmse),
                     'mae_gain_pct': gain(fits['candidate']['l1'], fits['control']['l1'], mae),
                     'control_vs_matched': {'rmse_gain_pct': gain(fits['control']['l2'], fits['matched']['l2'], rmse),
                                            'mae_gain_pct': gain(fits['control']['l1'], fits['matched']['l1'], mae)},
                     'directions': {name: {'windows': int((h == half).sum()),
                                           'rmse_gain_pct': gain(fits['candidate']['l2'][h == half], fits['control']['l2'][h == half], rmse),
                                           'mae_gain_pct': gain(fits['candidate']['l1'][h == half], fits['control']['l1'][h == half], mae)}
                                    for half, name in ((1, 'early_fit_late_score_forward'), (0, 'late_fit_early_score_reverse'))}})
    return {'coverage': float(ok.mean()), 'eligible_coverage': float(ok.sum() / max(eligible.sum(), 1)),
            'boundary_purged_windows': int((~eligible).sum()), 'leads': rows}


def decide(group):
    if group['eligible_coverage'] < PASS_RULE['min_coverage']:
        return 'INSUFFICIENT_COVERAGE', None
    leads = [r for r in group['leads'] if r['lead'] in PASS_RULE['leads']]
    values = [r[k] for r in leads for k in ('rmse_gain_pct', 'mae_gain_pct')]
    values += [r['directions'][d][k] for r in leads for d in r['directions'] for k in ('rmse_gain_pct', 'mae_gain_pct')]
    if any(v is None or not np.isfinite(v) for v in values):
        raise ValueError('Primary gain undefined; screen is INVALID, not a pass or negative result')
    summary = {'rmse': float(np.mean([r['rmse_gain_pct'] for r in leads])),
               'mae': float(np.mean([r['mae_gain_pct'] for r in leads]))}
    for direction in leads[0]['directions']:
        summary[direction] = {k: float(np.mean([r['directions'][direction][f'{k}_gain_pct'] for r in leads]))
                              for k in ('rmse', 'mae')}
    passed = (summary['rmse'] >= PASS_RULE['min_rmse_gain_pct'] and summary['mae'] >= PASS_RULE['min_mae_gain_pct']
              and all(summary[d][k] > 0 for d in leads[0]['directions'] for k in ('rmse', 'mae')))
    return ('PASS' if passed else 'NOT_PASSED'), summary


# --------------------------------------------------------------------------- run
def registered_stations(screen_json):
    screen = json.loads(Path(screen_json).read_text(encoding='utf-8'))
    if screen.get('status') != 'SCREENED' or screen.get('passed') is not True:
        raise ValueError('The upstream screen is not a completed passed screen')
    stations = [s for s in screen['stations'] if s['qualified']]
    if len(stations) != 36:
        raise ValueError(f'Expected the 36 registered stations, found {len(stations)}')
    hashes = {}
    for item in screen['files_read']:
        name = Path(item['path']).name
        if name in hashes and hashes[name] != item['sha256']:
            raise ValueError(f'Ambiguous registered hash for {name}')
        hashes[name] = item['sha256']
    return screen, stations, hashes


def run(midas_root, screen_json, eval_2021):
    screen, stations, hashes = registered_stations(screen_json)
    ids = {int(s['src_id']) for s in stations}
    observations, read, _ = base.load_observations(midas_root, ids)
    for item in read:                                     # every opened wind/pressure file is the screened byte stream
        if hashes.get(Path(item['path']).name) != item['sha256']:
            raise ValueError(f'Unregistered or modified MIDAS file: {item["path"]}')
    if set(observations) != ids:
        raise ValueError('Not every registered station has observations')
    extra_paths = [Path(item['path']) for item in read if int(Path(item['path']).stem[-4:]) in READ_YEARS]
    extra, extra_report = load_extra(extra_paths, ids)
    for item in extra_report:
        if hashes.get(Path(item['path']).name) != item['sha256']:
            raise ValueError(f'Extra-field file changed after hash verification: {item["path"]}')

    cohort, prediction, issue = base.load_year(eval_2021, base.ARMS[YEAR], YEAR)
    error = prediction - cohort['truth']
    anchor = cohort['PersistenceLast10'][:, 0]
    halves = np.asarray(issue.month >= 7, dtype=int)
    unique, inverse = np.unique(issue.asi8, return_inverse=True)
    issue_unique = pd.DatetimeIndex(unique)
    existing, existing_names = base.upstream_features(issue_unique, observations, stations, LAG_HOURS)
    new, night, upwind = extra_features(issue_unique, observations, extra, stations, LAG_HOURS)
    # Identity check: our upwind mask reproduces the frozen upwind wind-speed feature exactly.
    ws0 = base.station_matrix(observations, stations, issue_unique - pd.Timedelta(hours=LAG_HOURS), 'wind_speed')
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        recomputed = np.nanmean(np.where(upwind, ws0, np.nan), 1)
    if not np.allclose(recomputed, existing[:, existing_names.index('upwind_ws')], equal_nan=True):
        raise ValueError('Upwind mask differs from the frozen upstream_features rule')
    control = np.column_stack([existing, night])[inverse]
    groups = {}
    for name, columns in GROUPS.items():
        idx = [NEW_FEATURES.index(c) for c in columns]
        groups[name] = score_increment(control, new[inverse][:, idx], error, anchor, halves, issue) | {'features': columns}
    status, summary = decide(groups[PRIMARY_GROUP])
    completed = Path(eval_2021).parent / 'completed.json'
    return {'screen_version': SCREEN_VERSION, 'status': status, 'passed': status == 'PASS',
            'primary_mean_gain_pct': summary, 'pass_rule': PASS_RULE,
            'scope': 'retrospective bidirectional development screen on the D1 validation year; linear/IRLS correction '
                     'only, no forecaster retraining; incremental over the 11 existing upstream features + night flag; '
                     'archival qc-version-1 values with an assumed 1 h availability lag',
            'control_features': existing_names + ['night'], 'new_features': NEW_FEATURES,
            'reference_original_screen_2021_lead2_4_gain_pct': screen['years'][str(YEAR)].get('primary_mean_gain_pct'),
            'groups': groups,
            'input_identity': {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               'base_screen_script_sha256': hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
                               'screen_json_sha256': hashlib.sha256(Path(screen_json).read_bytes()).hexdigest(),
                               'v28_completed_sha256': hashlib.sha256(completed.read_bytes()).hexdigest(),
                               'wind_pressure_files': len(read), 'extra_field_files': extra_report},
            'stations': [{k: s[k] for k in ('src_id', 'name', 'distance_km', 'bearing_deg')} for s in stations],
            'windows': int(len(error))}


# --------------------------------------------------------------------------- self-test (synthetic only)
def self_test():
    noon, midnight = solar_elevation_deg(['2021-06-21 12:00']), solar_elevation_deg(['2021-12-21 00:00'])
    assert noon[0] > 55 and midnight[0] < -50, (noon, midnight)

    data = pd.DataFrame({'src_id': [1, 1, 1, 1], 'ob_time': ['2021-01-01 00:00', '2021-01-01 01:00', '2021-01-01 02:00', '2021-01-01 03:30'],
                         'version_num': [1, 1, 1, 1], 'air_temperature': [5.0, 60.0, 3.0, 4.0], 'dewpoint': [2.0, 1.0, 6.0, 1.0],
                         'cld_ttl_amt_id': [9, 8, 3, 10]})
    data.attrs['units'] = {'air_temperature': 'degc', 'dewpoint': 'degc', 'cld_ttl_amt_id': 'oktas'}
    frame, counts = clean_extra(data, 'synthetic')
    assert len(frame) == 3 and np.isnan(frame.cloud.iloc[0]) and frame.cloud.iloc[1] == 8
    assert np.isnan(frame.air_temperature.iloc[1]) and np.isnan(frame.dewpoint.iloc[2]) and counts['cloud_obscured_9'] == 1
    bad = data.copy()
    bad.attrs['units'] = {'air_temperature': 'degf', 'dewpoint': 'degc', 'cld_ttl_amt_id': 'oktas'}
    try:
        clean_extra(bad, 'synthetic')
        raise AssertionError('unit mismatch accepted')
    except ValueError:
        pass

    times = pd.date_range('2021-01-01', periods=48, freq='h')
    stations = [{'src_id': 1, 'distance_km': 30.0, 'bearing_deg': 270.0}, {'src_id': 2, 'distance_km': 25.0, 'bearing_deg': 90.0},
                {'src_id': 3, 'distance_km': 60.0, 'bearing_deg': 260.0}]
    rng = np.random.default_rng(0)
    obs = {s['src_id']: pd.DataFrame({'wind_speed': rng.uniform(2, 10, len(times)), 'wind_direction': 270.0,
                                      'msl_pressure': 1010 + rng.normal(0, 1, len(times))}, index=times) for s in stations}
    issue = times[6:]
    existing, names = base.upstream_features(issue, obs, stations, 1)
    upwind, nearest = upwind_mask(obs, stations, issue - pd.Timedelta(hours=1))
    ws0 = base.station_matrix(obs, stations, issue - pd.Timedelta(hours=1), 'wind_speed')
    assert nearest == 1 and upwind[:, 0].all() and upwind[:, 2].all() and not upwind[:, 1].any()
    assert np.allclose(np.nanmean(np.where(upwind, ws0, np.nan), 1), existing[:, names.index('upwind_ws')])

    n = 6000
    t = pd.date_range('2021-01-01', periods=n, freq='h')
    halves = np.asarray(t.month >= 7, dtype=int)
    control, new = rng.normal(size=(n, 3)), rng.normal(size=(n, 2))
    anchor = rng.uniform(0, 2050, n)
    noise = rng.normal(0, 50, size=(n, 4))
    signal = noise + 40 * new[:, :1] + 30 * control[:, :1]
    hit = score_increment(control, new, signal, anchor, halves, t)
    null = score_increment(control, new, noise + 30 * control[:, :1], anchor, halves, t)
    assert decide(hit)[0] == 'PASS' and hit['leads'][0]['rmse_gain_pct'] > 5
    assert decide(null)[0] == 'NOT_PASSED' and abs(null['leads'][0]['rmse_gain_pct']) < 1
    sparse = new.copy()
    sparse[: n // 3 * 2] = np.nan
    assert decide(score_increment(control, sparse, signal, anchor, halves, t))[0] == 'INSUFFICIENT_COVERAGE'
    print('S self-test PASS: solar night flag, field cleaning/units, upwind-mask identity, incremental gain, decision, coverage')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--midas-root', type=Path)
    parser.add_argument('--screen-json', type=Path)
    parser.add_argument('--eval-2021', type=Path, help='v28 run evaluation/ directory')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not all((args.midas_root, args.screen_json, args.eval_2021, args.output)):
        parser.error('--midas-root, --screen-json, --eval-2021 and --output are required')
    try:
        result = run(args.midas_root, args.screen_json, args.eval_2021)
    except (ValueError, FileNotFoundError) as exc:
        result = {'status': 'INVALID_INPUT', 'passed': None, 'reason': str(exc), 'screen_version': SCREEN_VERSION}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + '.tmp')
    temp.write_text(json.dumps(result, indent=2, default=float, allow_nan=False), encoding='utf-8')
    temp.replace(args.output)
    print(json.dumps({k: result.get(k) for k in ('status', 'passed', 'primary_mean_gain_pct', 'reason') if k in result}, indent=2))
    for name, group in result.get('groups', {}).items():
        cells = ' | '.join(f"h{r['lead']} RMSE {r['rmse_gain_pct']:+.2f}% MAE {r['mae_gain_pct']:+.2f}%" for r in group['leads'])
        print(f"{name:14s} coverage {group['eligible_coverage']:.3f} | {cells}")
    return 0 if result['status'] in ('PASS', 'NOT_PASSED', 'INSUFFICIENT_COVERAGE') else 2


if __name__ == '__main__':
    raise SystemExit(main())
