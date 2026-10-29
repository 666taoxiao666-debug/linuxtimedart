#!/usr/bin/env python3
"""Approved A0: fixed 36-station, 2016--2022 input-only audit. No training/network.

Reads only registered MIDAS weather files, metadata and the existing screen register.
It never imports the training/screening implementation or reads power/SCADA labels.
Coverage is on a calendar grid, NOT the unknown label-eligible forecast windows.
Outputs are diagnostics; this script cannot authorize A1 or change a protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

YEARS = tuple(range(2016, 2023))
FIELDS = ('air_temperature', 'dewpoint', 'q10mnt_mxgst_spd',
          'cld_ttl_amt_id', 'prst_wx_id', 'rltv_hum')
WIND = ('wind_speed', 'wind_direction')
ROOT_SUFFIX = ('datasets', 'upstream_observations', 'midas_open_v202607', 'development')
META = 'midas-open_uk-hourly-weather-obs_dv-202607_station-metadata.csv'
PLAN_SHA = 'a4d083627893bb8cbc76095e809b819b091047f94778a221ccae0b696fa3c31f'
META_SHA = '3c98d20c4b9a8aba3270280c8692b72a54d66df05d178c5fed27eb70e6fd3acc'
QUANTILES = [0, .01, .05, .25, .5, .75, .95, .99, 1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def guarded(path, root):
    resolved = path.resolve(strict=True)
    if 'sealed' in (p.lower() for p in resolved.parts) or not resolved.is_relative_to(root):
        raise ValueError('Refusing path outside the development root or inside sealed data')
    return resolved


def read_badc(path):
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    marker = lambda s: s.strip().strip(',').strip().lower()
    start = next(i for i, line in enumerate(lines) if marker(line) == 'data')
    end = next(i for i in range(start + 1, len(lines)) if marker(lines[i]) == 'end data')
    headers = list(csv.reader(lines[:start]))
    units, descriptions = {}, {}
    for row in headers:
        if len(row) >= 4 and row[0].strip().lower() == 'long_name':
            units[row[1].strip().lower()] = row[3].strip()
            descriptions[row[1].strip().lower()] = row[2].strip()
        elif len(row) >= 3 and row[0].strip().lower() == 'unit':
            key, unit = row[1].strip().lower(), row[2].strip()
            if key in units and units[key].lower() != unit.lower():
                raise ValueError('Conflicting unit declarations')
            units[key] = unit
    # All columns in these files are weather metadata/observations, never target labels.
    frame = pd.read_csv(io.StringIO('\n'.join(lines[start + 1:end])), low_memory=False)
    frame.columns = frame.columns.str.strip().str.lower()
    return frame, units, descriptions


def numeric(frame, name):
    if name not in frame:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[name], errors='coerce').replace([np.inf, -np.inf], np.nan)


def counts(series):
    return {str(k): int(v) for k, v in series.fillna('MISSING').astype(str).value_counts().items()}


def distribution(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return {'n': int(values.size), 'quantiles': dict(zip(map(str, QUANTILES),
            np.quantile(values, QUANTILES).tolist())) if values.size else {}}


def speed_factor(unit):
    if unit.lower() in ('knots', 'knot', 'kt'):
        return 1852 / 3600
    if unit.lower() in ('m/s', 'm s-1', 'ms-1'):
        return 1.0
    raise ValueError(f'Unsupported wind unit: {unit}')


def physics(speed, direction, distance, bearing, kappa):
    """Own observation-time wind-from; regional circular fallback, no future input."""
    valid_speed = np.isfinite(speed) & (speed >= 0)
    valid_dir = np.isfinite(direction) & (direction >= 0) & (direction <= 360) & (speed > 0)
    radians = np.deg2rad(np.where(valid_dir, direction, np.nan))
    n = valid_dir.sum(axis=1)
    sin_sum, cos_sum = np.nansum(np.sin(radians), axis=1), np.nansum(np.cos(radians), axis=1)
    resultant = np.hypot(sin_sum, cos_sum)
    regional_dir = np.where((n > 0) & (resultant > 1e-10),
                            np.rad2deg(np.arctan2(sin_sum, cos_sum)) % 360, np.nan)
    fallback = ~valid_dir & valid_speed & (speed > 0) & np.isfinite(regional_dir[:, None])
    used_dir = np.where(fallback, regional_dir[:, None], np.where(valid_dir, direction, np.nan))
    cosine = np.cos(np.deg2rad(bearing[None, :] - used_dir))
    parallel = distance[None, :] * cosine
    # Clause 4.2: d_parallel>0; no additional 60-degree or distance cutoff introduced.
    upwind = (parallel > 1e-10) & valid_speed
    count = upwind.sum(axis=1)
    regional_n = valid_speed.sum(axis=1)
    regional_speed = np.divide(np.where(valid_speed, speed, 0).sum(axis=1), regional_n,
                               out=np.full(speed.shape[0], np.nan), where=regional_n > 0)
    v = np.divide(np.where(upwind, speed, 0).sum(axis=1), count,
                  out=regional_speed.copy(), where=count > 0)
    # Literal 14.3 formula: kappa * max(v, 2); alternative floor order remains for Claude.
    t_adv = parallel / (3.6 * kappa * np.maximum(v[:, None], 2))
    active = upwind & (speed >= 2) & np.isfinite(t_adv)
    t_adv = np.where(active, t_adv, np.nan)
    perpendicular = distance[None, :] * np.abs(np.sin(np.deg2rad(bearing[None, :] - used_dir)))
    geometry = np.where(active, np.maximum(cosine, 0) * np.exp(-.5 * (perpendicular / 30) ** 2), 0)
    return t_adv, geometry, {'direction_fallback_station_hours': int(fallback.sum()),
                            'calm_or_slow_station_hours': int((valid_speed & (speed < 2)).sum()),
                            'regional_speed_fallback_hours': int((count == 0).sum())}, valid_speed & (speed > 0) & np.isfinite(used_dir)


def support(t_adv, geometry, token_valid, lag, calendar, window):
    # Common denominator excludes initial 29h (largest age in any scenario) and final4h.
    start, stop = 29, len(calendar) - 4
    issue_count = stop - start
    ages = range(lag, 25) if window == 'fixed_past_24h' else range(lag, lag + 24)
    centers = np.zeros((issue_count, 4), dtype=bool)
    qsum = np.zeros((issue_count, 4))
    available_n = np.zeros(issue_count, dtype=int)
    active_n = np.zeros(issue_count, dtype=int)
    r_bins = [-np.inf, -24, -12, -6, -1, 0, 1, 2, 3, 4, 6, 12, 24, np.inf]
    histogram = np.zeros(len(r_bins) - 1, dtype=np.int64)
    remaining_n = inside_n = 0
    latest_centers = None
    for age in ages:
        ta = t_adv[start - age:stop - age]
        geo = geometry[start - age:stop - age]
        r = ta - age
        finite = np.isfinite(r)
        available_n += token_valid[start - age:stop - age].sum(axis=1)
        active_n += finite.sum(axis=1)
        remaining_n += int(finite.sum())
        inside_n += int(((r >= 0) & (r <= 4)).sum())
        histogram += np.histogram(r[finite], bins=r_bins)[0]
        sigma = np.maximum(1, .5 * ta)
        latest = []
        for h in range(1, 5):
            exact = ((r >= h - 1) & (r <= h)).any(axis=1)
            centers[:, h - 1] |= exact
            latest.append(float(exact.mean()))
            # Stable CDF difference on positive tails; mathematical nonzero != useful support.
            lo, hi = (h - 1 - r) / sigma, (h - r) / sigma
            mass = np.where(lo > 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))
            qsum[:, h - 1] += np.nansum(geo * mass, axis=1)
        if age == lag:
            latest_centers = latest
    # Reference M has content scores fixed to zero; NOT learned attention or calibrated probability.
    qmean = np.divide(qsum, available_n[:, None], out=np.zeros_like(qsum), where=available_n[:, None] > 0)
    mass = qmean / (1 + qmean)
    overall = centers.mean(axis=0).tolist()
    return {'lag_hours': lag, 'window': window, 'ages_hours': list(ages),
            'calendar_issue_count': issue_count, 'first_issue_utc': str(calendar[start]),
            'last_issue_utc': str(calendar[stop - 1]),
            'center_in_target_hour_fraction': overall,
            'center_20pct_diagnostic_pass_each_h': [v >= .2 for v in overall],
            'latest_only_center_fraction': latest_centers,
            'no_physically_active_token_fraction': float((active_n == 0).mean()),
            'no_valid_wind_token_fraction': float((available_n == 0).mean()),
            'remaining_in_0_4_fraction_of_active_tokens': inside_n / remaining_n if remaining_n else None,
            'remaining_active_tokens': remaining_n,
            'remaining_histogram_edges': [str(x) for x in r_bins],
            'remaining_histogram_counts': histogram.tolist(),
            'reference_null_mass_M_by_h': [distribution(mass[:, h]) for h in range(4)],
            'center_fraction_by_year': {str(y): centers[calendar[start:stop].year == y].mean(axis=0).tolist()
                                        for y in sorted(set(calendar[start:stop].year))}}


def run(args):
    repo = Path(__file__).resolve().parents[1]
    root = (repo / Path(*ROOT_SUFFIX)).resolve(strict=True)
    if tuple(root.parts[-4:]) != ROOT_SUFFIX or 'sealed' in (x.lower() for x in root.parts):
        raise ValueError('Unexpected development root')
    audit = repo / 'outputs/audits/upstream_screen/midas_v202607'
    screen_path, plan_path = audit / 'screen.json', audit / 'download_plan.json'
    if sha(plan_path) != PLAN_SHA:
        raise ValueError('Download plan identity changed')
    screen = json.loads(screen_path.read_text())
    stations = sorted((s for s in screen['stations'] if s['qualified']), key=lambda s: int(s['src_id']))
    if len(stations) != 36 or len({s['src_id'] for s in stations}) != 36:
        raise ValueError('Expected registered 36 unique stations; no reselection')
    metadata = guarded(root / META, root)
    if sha(metadata) != META_SHA or screen['metadata_sha256'] != META_SHA:
        raise ValueError('Metadata identity changed')
    registered = {Path(f['path']).name: f['sha256'] for f in screen['files_read']}
    calendar = pd.date_range('2016-01-01', '2023-01-01', freq='h', inclusive='left', tz='UTC')
    speed = np.full((len(calendar), 36), np.nan)
    direction = np.full_like(speed, np.nan)
    records, files = [], []
    totals = {f: Counter() for f in FIELDS}
    for j, station in enumerate(stations):
        sid = int(station['src_id'])
        folder = guarded(root / str(sid), root)
        for year in YEARS:
            # Enumerates only this approved station directory; never traverses sealed trees.
            matches = list(folder.glob(f'*_qcv-1_{year}.csv'))
            if len(matches) != 1:
                raise ValueError(f'{sid}/{year}: expected exactly one registered file')
            path = guarded(matches[0], root)
            if f'dv-202607_' not in path.name or path.name not in registered:
                raise ValueError('Unregistered weather source')
            digest = sha(path)
            if digest != registered[path.name]:
                raise ValueError('Weather SHA256 differs from original screening register')
            frame, units, descriptions = read_badc(path)
            times = pd.to_datetime(frame['ob_time'], utc=True, errors='coerce')
            if times.isna().any() or (times.dt.year != year).any():
                raise ValueError('Invalid/out-of-scope observation timestamp')
            if not (numeric(frame, 'src_id') == sid).all() or not (numeric(frame, 'version_num') == 1).all():
                raise ValueError('Station/version identity differs from filename')
            whole = (times.dt.minute == 0) & (times.dt.second == 0)
            hourly = frame.loc[whole].copy()
            hourly.index = pd.DatetimeIndex(times[whole])
            columns = [f for f in (*FIELDS, *WIND) if f in hourly]
            if (hourly.groupby(level=0)[columns].nunique(dropna=False) > 1).any().any():
                raise ValueError('Conflicting duplicate weather records; no silent mixing')
            duplicates = int(hourly.index.duplicated().sum())
            hourly = hourly.loc[~hourly.index.duplicated()]
            year_hours = int((calendar.year == year).sum())
            field_rows = {}
            for field in FIELDS:
                nonnull = int(numeric(hourly, field).notna().sum())
                qfield = field + '_q'
                row = {'column_present': field in frame, 'unit': units.get(field),
                       'description': descriptions.get(field), 'nonnull_hourly': nonnull,
                       'calendar_hours': year_hours, 'calendar_completeness': nonnull / year_hours,
                       'qc_column_present': qfield in frame,
                       'qc_counts_all_rows': counts(frame[qfield]) if qfield in frame else {'COLUMN_ABSENT': len(frame)},
                       'qc_counts_when_value_present': counts(frame.loc[numeric(frame, field).notna(), qfield])
                            if qfield in frame else {'COLUMN_ABSENT': int(numeric(frame, field).notna().sum())}}
                if field in ('cld_ttl_amt_id', 'prst_wx_id'):
                    row['code_counts_all_rows'] = counts(frame[field]) if field in frame else {}
                field_rows[field] = row
                totals[field].update({'nonnull_hourly': nonnull, 'calendar_hours': year_hours,
                                      'files_with_column': int(field in frame)})
            factor = speed_factor(units.get('wind_speed', ''))
            if units.get('wind_direction', '').lower() not in ('deg true', 'degrees', 'degree', 'degrees true'):
                raise ValueError('Unsupported direction units')
            indices = calendar.get_indexer(hourly.index)
            if (indices < 0).any():
                raise ValueError('Weather timestamp outside audit calendar')
            s, d = numeric(hourly, 'wind_speed').to_numpy() * factor, numeric(hourly, 'wind_direction').to_numpy()
            s[s < 0] = np.nan
            d[(d < 0) | (d > 360) | (s == 0)] = np.nan
            speed[indices, j], direction[indices, j] = s, d
            records.append({'src_id': sid, 'year': year, 'raw_rows': len(frame),
                            'non_hourly_rows': int((~whole).sum()), 'duplicate_hourly_rows': duplicates,
                            'unique_hourly_rows': len(hourly), 'fields': field_rows,
                            'wind_units': {k: units.get(k) for k in WIND},
                            'wind_qc': {k: counts(frame[k + '_q']) if k + '_q' in frame else {} for k in WIND},
                            'src_opr_type_counts': counts(frame['src_opr_type']) if 'src_opr_type' in frame else {}})
            files.append({'path': str(path.relative_to(repo)), 'bytes': path.stat().st_size, 'sha256': digest})
        print(f'A0 weather audit: station {j + 1}/36 completed', flush=True)
    summaries = {f: {**dict(v), 'calendar_completeness': v['nonnull_hourly'] / v['calendar_hours']} for f, v in totals.items()}
    distance = np.array([s['distance_km'] for s in stations])
    bearing = np.array([s['bearing_deg'] for s in stations])
    physical = []
    for kappa in (1., .5, 4.):
        ta, geo, diag, token_valid = physics(speed, direction, distance, bearing, kappa)
        # Include direction-fallback tokens in the reference denominator, even if slow/downwind.
        for window in ('fixed_past_24h', 'latest_available_24h'):
            for lag in (1, 3, 6):
                result = support(ta, geo, token_valid, lag, calendar, window)
                physical.append({'kappa': kappa, 't_adv_hours': distribution(ta), **diag, **result})
        print(f'A0 support diagnostics: kappa={kappa} completed', flush=True)
    out = Path(args.output).resolve()
    # Only new audit artifacts; never overwrite an input, checkpoint, protocol or prior run.
    if not out.is_relative_to((repo / 'outputs/audits/station_history_v33').resolve()):
        raise ValueError('Output must be inside outputs/audits/station_history_v33')
    if out.exists():
        raise ValueError('Output exists; refusing to overwrite an audit run')
    result = {'schema': 'station_history_a0_v1', 'status': 'INPUT_AUDIT_COMPLETED_NOT_A1_APPROVAL',
              'years': list(YEARS), 'station_count': len(stations), 'file_count': len(files),
              'source_bytes': sum(f['bytes'] for f in files), 'script_sha256': sha(Path(__file__)),
              'screen_sha256': sha(screen_path), 'plan_sha256': sha(plan_path), 'metadata_sha256': sha(metadata),
              'station_order': [s['src_id'] for s in stations], 'stations': stations,
              'field_summary': summaries, 'station_years': records, 'files_read': files, 'physical': physical,
              'definitions': {
                  'availability': 'Simulated ob_time + lag, not actual release/receipt timestamp; archive QC version1 is revised.',
                  'calendar_denominator': 'All UTC hourly issues 2016-2022 minus initial29/final4h; no label eligibility examined.',
                  'support': 'Center r=T_adv-age in [h-1,h], at least one active token; also report soft null mass distribution.',
                  'threshold': '20% applies only descriptively to center support; no soft-mass cutoff or label cohort is approved.',
                  'direction': 'Own observation-time wind-from; missing direction uses same-time circular regional mean if defined.',
                  'speed': 'Mean of same-time upwind station speeds; regional mean if no upwind speeds; kappa*max(v,2).',
                  'calm': 'Own speed <2m/s supplies no physical prior. Direction fallback is audit-only input rule, not deployed.',
                  'quality': 'Report raw nonmissing coverage and QC codes; do not label all qcv1 values QC-passed or change frozen masks.',
                  'soft_prior': 'Normal CDF integral, sigma=max(1,0.5*T_adv), cos(direction difference)*exp(-0.5*(crosswind/30)^2).',
                  'reference_M': 'Content scores zero, q summed/number wind tokens then qmean/(1+qmean); not trained attention.',
                  'window': 'Two interpretations of 24h reported; neither chosen using outcomes.',
                  'remaining_histogram': 'Half-open bins except final bin; [0,4] fraction computed directly, includes endpoints.',
                  'no_labels': True, 'no_training': True, 'no_network': True},
              'unresolved_for_claude': ['Soft effective-support threshold and label-eligible denominator',
                  '24h window anchoring; kappa speed-floor order; invalid-direction fallback policy',
                  'Full-text literature review is recorded separately; no automatic novelty or feasibility pass']}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': str(out.relative_to(repo)), 'sha256': sha(out), 'fields': summaries}))


def self_test():
    # Hand-check: north station36km, north10m/s =>1h; south station is downwind.
    s = np.full((70, 2), 10.)
    d = np.zeros_like(s)
    ta, geo, _, _ = physics(s, d, np.array([36., 36.]), np.array([0., 180.]), 1.)
    assert np.allclose(ta[:, 0], 1) and np.isnan(ta[:, 1]).all()
    d[:, 0] = 360
    assert np.allclose(physics(s, d, np.array([36., 36.]), np.array([0., 180.]), 1)[0][:, 0], 1)
    assert np.isclose(speed_factor('knots') * 3600, 1852)
    slow = s * 0 + 1
    assert np.isnan(physics(slow, d, np.array([36., 36.]), np.array([0., 180.]), 1)[0]).all()
    cal = pd.date_range('2016-01-01', periods=70, freq='h', tz='UTC')
    # r=0 at age1 is first-hour endpoint; age3 has already passed, so no center support.
    one = support(ta, geo, np.isfinite(ta), 1, cal, 'fixed_past_24h')
    three = support(ta, geo, np.isfinite(ta), 3, cal, 'fixed_past_24h')
    assert one['center_in_target_hour_fraction'] == [1, 0, 0, 0]
    assert three['center_in_target_hour_fraction'] == [0, 0, 0, 0]
    empty = support(ta * np.nan, geo * 0, np.zeros_like(ta, dtype=bool), 1, cal, 'fixed_past_24h')
    assert empty['no_physically_active_token_fraction'] == 1
    assert empty['reference_null_mass_M_by_h'][0]['quantiles']['1'] == 0
    # Invalid paths rejected before opening any content.
    try:
        guarded(Path(__file__), Path(__file__).parent / 'nonexistent_development')
    except ValueError:
        pass
    else:
        raise AssertionError('Path escape accepted')
    print('A0 self-test PASS: units, wind-from geometry, wraparound, calm, age, empty support, path boundary')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', default='outputs/audits/station_history_v33/feasibility.json')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)
