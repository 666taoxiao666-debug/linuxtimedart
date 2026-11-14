"""G2a: read-only HoT 2016-2023 preparation; no training or sealed-year opens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_g1_batches as g1
from scripts.acquire_hourly_wind_datasets import hourly_scada
from scripts.audit_hill_of_towie_e1 import development_loader, sha
from scripts import screen_kelmarsh_upstream_observations as screen
from utils import kelmarsh_enrichment as enrich
from data_provider.kelmarsh_multiyear import valid_window_starts, BASE_FEATURES

YEARS = tuple(range(2016, 2024))
CUTOFF = pd.Timestamp('2024-01-01', tz='UTC')
QUALIFIED_SHA = '5909e1436559dd4334649c41c66881853ace75bc68a80f4fc9bfb30fcc60a979'
RECORD_SHA = '7064b838681db88aa70eb7c9783e3383b074204a41e621bd62681c8abe50b5d0'
MAPPING = {
    'tblSCTurbine': {'wtc_AcWindSp_mean': 'wind_speed_mps',
        'wtc_NacelPos_mean': 'nacelle_position_deg', 'wtc_MainSRpm_mean': 'rotor_speed_rpm',
        'wtc_PitcPosA_mean': 'pitch_a_deg', 'wtc_PitcPosB_mean': 'pitch_b_deg',
        'wtc_PitcPosC_mean': 'pitch_c_deg'},
    'tblSCTurGrid': {'wtc_ActPower_mean': 'power_kw'},
    'tblSCTurTemp': {'wtc_AmbieTmp_mean': 'ambient_temp_c'}}


def development_year(year):
    if year not in YEARS:
        raise ValueError('Protected year refused before opening input')
    return year


def works_bounds(first, last):
    a, b = pd.Timestamp(first, tz='UTC'), pd.Timestamp(last, tz='UTC') + pd.Timedelta(days=1)
    if a != a.normalize() or b != b.normalize() or not a < b < CUTOFF:
        raise ValueError('Invalid construction calendar dates')
    return a, b


def window_works(issue, first, end):
    t = pd.DatetimeIndex(issue)
    # Hour-end labels represent [label-1h,label); input+target span [issue-24h,issue+4h).
    excluded = (t - pd.Timedelta(hours=24) < end) & (t + pd.Timedelta(hours=4) > first)
    return np.asarray(excluded), np.asarray(t >= end, dtype=np.float32)


def read_month(z, year, month, station_ids):
    development_year(year)
    lower = pd.Timestamp(year=year, month=month, day=1, tz='UTC')
    upper = lower + pd.offsets.MonthBegin(1)
    tables = []
    for prefix, mapping in MAPPING.items():
        name = f'{prefix}_{year}_{month:02d}.csv'
        with z.open(name) as f:
            t = pd.read_csv(f, usecols=['TimeStamp', 'StationId'] + list(mapping),
                            encoding='cp1252', dtype=str)
        t['TimeStamp'] = pd.to_datetime(t.TimeStamp.str.strip(), utc=True, errors='raise')
        if t.TimeStamp.isna().any() or not t.TimeStamp.between(lower, upper).all():
            raise ValueError('SCADA timestamp outside registered month and endpoint')
        # Inspect time keys first; protected rows never reach numeric conversion.
        t = t.loc[t.TimeStamp < CUTOFF].copy()
        t['StationId'] = pd.to_numeric(t.StationId, errors='raise')
        if (not set(t.StationId) <= set(station_ids) or t.duplicated(['TimeStamp', 'StationId']).any()):
            raise ValueError('SCADA year/month/unit/duplicate identity mismatch')
        for c in mapping:
            t[c] = pd.to_numeric(t[c], errors='raise')
        tables.append(t.rename(columns=mapping))
    ticks = tables[0]
    for t in tables[1:]:
        ticks = ticks.merge(t, on=['TimeStamp', 'StationId'], how='outer', validate='one_to_one')
    return ticks.rename(columns={'TimeStamp': 'timestamp'})


def merge_boundary(previous, current):
    """Exact duplicate observations collapse; never average conflicting labels."""
    merged = pd.concat([previous, current], ignore_index=True) if previous is not None else current.copy()
    keys = ['timestamp', 'StationId']
    duplicates = merged.duplicated(keys, keep=False)
    if duplicates.any():
        values = merged.loc[duplicates].groupby(keys, dropna=False).nunique(dropna=False)
        if (values > 1).any().any():
            raise ValueError('Conflicting SCADA boundary observations')
        merged = merged.drop_duplicates(keys, keep='first')
    return merged.sort_values(keys).reset_index(drop=True)


def aggregate(ticks):
    ticks = ticks.sort_values('timestamp').copy()
    # Official data dictionary: SCADA TimeStamp is period END (UTC).
    ticks['timestamp'] = ticks.timestamp - pd.Timedelta(minutes=10)
    enrich.validate_tick_index(pd.DatetimeIndex(ticks.timestamp))
    base = hourly_scada(ticks.copy(), ['nacelle_position_deg'])
    for c in enrich.OPTIONAL_TICK_COLUMNS:
        ticks[c] = np.nan
    extra = enrich.hourly_enrichment(ticks.set_index('timestamp'), 2300)
    base = base.merge(extra[['power_last10_kw', 'wind_last10_mps']].reset_index(),
                      on='timestamp_hour_end_utc', validate='one_to_one')
    check = hourly_scada(ticks[['timestamp', 'power_kw']].copy())
    if not np.array_equal(base.power_kw, check.power_kw, equal_nan=True):
        raise ValueError('Power label altered during enrichment')
    return base[base.timestamp_hour_end_utc < CUTOFF]


def verified(path, expected, identities):
    actual = g1.digest(path)
    if actual != expected:
        raise ValueError('Input hash mismatch: ' + str(path))
    identities[str(path)] = actual
    return path


def raw_inventory(root, identities):
    directory = root/'datasets/external_validation/sealed/hill_of_towie'
    record = root/'output/external_acquisition/hill_of_towie/record_22662930.json'
    verified(record, RECORD_SHA, identities)
    registered = {f['key']: f for f in json.loads(record.read_text())['files']}
    manifest = root/'output/external_acquisition/hill_of_towie/manifest.json'
    identities[str(manifest)] = g1.digest(manifest)
    entries = {f['name']: f for f in json.loads(manifest.read_text())['files']}
    names = [f'{y}.zip' for y in YEARS] + ['Hill_of_Towie_turbine_metadata.csv',
                                           'Hill_of_Towie_AeroUp_install_dates.csv',
                                           'Hill_of_Towie_data_dictionary.md']
    for name in names:
        f, official, p = entries[name], registered[name], directory/name
        if not (p.stat().st_size == f['actual_bytes'] == official['size']
                and sha(p, 'md5') == f['official_md5'] == official['checksum'].removeprefix('md5:')):
            raise ValueError('Official size/MD5 mismatch: ' + name)
        verified(p, f['sha256'], identities)
    return directory


def prepare_scada(directory, out):
    dictionary = (directory/'Hill_of_Towie_data_dictionary.md').read_text(encoding='utf-8')
    if 'TimeStamp` giving the end of the' not in dictionary or 'subtract 10 minutes' not in dictionary:
        raise ValueError('Official SCADA time semantics require reinspection')
    meta = pd.read_csv(directory/'Hill_of_Towie_turbine_metadata.csv', encoding='utf-8-sig')
    works = pd.read_csv(directory/'Hill_of_Towie_AeroUp_install_dates.csv').set_index('Turbine')
    if (len(meta) != 21 or meta['Station ID'].nunique() != 21 or meta['Turbine Name'].nunique() != 21
            or not meta['Rated power (kW)'].eq(2300).all()
            or set(meta['Turbine Name']) != set(works.index) or not works.index.is_unique):
        raise ValueError('Expected 21 unique 2300-kW turbines and construction records')
    bounds = {r['Turbine Name']: works_bounds(*works.loc[r['Turbine Name']].tolist())
              for _, r in meta.iterrows()}
    sid_to_unit = dict(zip(meta['Station ID'], meta['Turbine Name']))
    chunks, carry = [], None
    for year in YEARS:
        with zipfile.ZipFile(directory/f'{development_year(year)}.zip') as z:
            for month in range(1, 13):
                ticks = merge_boundary(carry, read_month(z, year, month, sid_to_unit))
                boundary = pd.Timestamp(year=year, month=month, day=1, tz='UTC') + pd.offsets.MonthBegin(1)
                # Retain the entire final physical hour until the adjacent file
                # has supplied its duplicate endpoint for exact-value checking.
                carry = ticks.loc[ticks.timestamp > boundary-pd.Timedelta(hours=1)].copy()
                ticks = ticks.loc[ticks.timestamp <= boundary-pd.Timedelta(hours=1)]
                for sid, t in ticks.groupby('StationId'):
                    h = aggregate(t.drop(columns='StationId'))
                    h['turbine_id'] = sid_to_unit[sid]
                    chunks.append(h)
        print(f'[SCADA] {year} complete', flush=True)
    if carry is not None and not carry.empty:
        for sid, t in carry.groupby('StationId'):
            h = aggregate(t.drop(columns='StationId'))
            h['turbine_id'] = sid_to_unit[sid]
            chunks.append(h)
    frame = pd.concat(chunks, ignore_index=True)
    if frame.duplicated(['turbine_id', 'timestamp_hour_end_utc']).any():
        raise ValueError('Duplicate hourly labels across monthly archives')
    frame.to_csv(out/'hourly_scada.csv', index=False)
    return frame, meta, bounds


def prepare_midas(root, hours, identities):
    from utils.kelmarsh_upstream_v31 import hourly_upstream
    audit = ROOT/'output/hill_of_towie_e1/l0b_r2'
    qpath = verified(audit/'qualified_stations.json', QUALIFIED_SHA, identities)
    q = json.loads(qpath.read_text())
    stations = q['stations']
    ids = q['qualified_station_ids']
    if len(ids) != 12 or sorted(s['src_id'] for s in stations) != ids:
        raise ValueError('Frozen twelve-station identity differs')
    manifest = json.loads((audit/'download_manifest.json').read_text())
    plan = audit/'download_plan.json'
    verified(plan, manifest['plan_sha256'], identities)
    identities[str(audit/'download_manifest.json')] = g1.digest(audit/'download_manifest.json')
    dev = root/'datasets/upstream_observations/hill_of_towie_midas_v202607/development'
    allowed = {}
    for f in manifest['files']:
        if f['src_id'] not in ids or f['year'] not in YEARS:
            continue
        if f['sealed']:
            raise ValueError('Development entry incorrectly sealed')
        path = dev/str(f['src_id'])/Path(f['path']).name
        if path.stat().st_size != f['bytes']:
            raise ValueError('MIDAS size mismatch')
        verified(path, f['sha256'], identities)
        allowed[str(path.resolve())] = f
    conflicts, obs, coverage = {}, {}, {}
    loader = development_loader(screen, conflicts)
    for sid in ids:
        if any(str(p.resolve()) not in allowed for p in (dev/str(sid)).glob('*.csv')):
            raise ValueError('Unregistered file in approved station directory')
        frames, parsed, _ = loader(dev/str(sid), [sid])
        if any(str(Path(r['path']).resolve()) not in allowed for r in parsed):
            raise ValueError('Unregistered MIDAS parse')
        obs.update(frames)
        coverage[str(sid)] = {}
        for y in YEARS:
            grid = pd.date_range(f'{y}-01-01', f'{y+1}-01-01', freq='h', inclusive='left')
            coverage[str(sid)][str(y)] = {c: float(frames[sid][c].reindex(grid).notna().mean())
                                         for c in ('wind_speed', 'wind_direction', 'msl_pressure')}
    upstream = hourly_upstream(obs, stations, hours, 1)
    return obs, stations, upstream, dict(coverage=coverage, conflicts=conflicts)


def scada_values(unit):
    a = np.deg2rad(unit.nacelle_position_deg)
    pitch = unit[['pitch_a_deg', 'pitch_b_deg', 'pitch_c_deg']].mean(axis=1, skipna=False)
    return np.column_stack([unit.wind_speed_mps, unit.ambient_temp_c, np.sin(a), np.cos(a),
        pitch, unit.rotor_speed_rpm, unit.power_last10_kw, unit.wind_last10_mps, unit.power_kw])


def descriptive_curve(unit, first, end):
    """Fixed .5 m/s bins from existing rules, raw means only; no smoothing or label edits."""
    output = {}
    for name, mask in [('before', unit.index <= first), ('after', unit.index - pd.Timedelta(hours=1) >= end)]:
        rows = unit.loc[mask, ['wind_speed_mps', 'power_kw']].dropna()
        edges = np.arange(0, enrich.RULES['curve_max_mps'] + .5, .5)
        bins = pd.cut(rows.wind_speed_mps, edges, right=False)
        grouped = rows.groupby(bins, observed=False).power_kw.agg(['count', 'mean', 'median'])
        output[name] = [dict(wind_lower=float(edges[i]), n=int(r['count']),
                            mean_kw=None if pd.isna(r['mean']) else float(r['mean']),
                            median_kw=None if pd.isna(r['median']) else float(r['median']))
                        for i, (_, r) in enumerate(grouped.iterrows())]
    return output


def boundary_examples(first, end):
    points = sorted(set([first-pd.Timedelta(hours=1), first, first+pd.Timedelta(hours=1),
                         end-pd.Timedelta(hours=1), end, end+pd.Timedelta(hours=1),
                         first-pd.Timedelta(hours=4), end+pd.Timedelta(hours=24)]))
    excluded, flag = window_works(points, first, end)
    return [dict(issue=str(t), excluded=bool(e), retrofit=int(f)) for t, e, f in zip(points, excluded, flag)]


def prepare_windows(frame, hours, bounds, out):
    blocks, annual, curves, examples = [], {}, {}, {}
    units = sorted(bounds)
    utc_hours = hours.tz_localize('UTC')
    for name in units:
        u = frame[frame.turbine_id == name].set_index('timestamp_hour_end_utc').reindex(utc_hours)
        x = scada_values(u)
        blocks.append(x)
        first, end = bounds[name]
        curves[name], examples[name] = descriptive_curve(u, first, end), boundary_examples(first, end)
        annual[name] = {}
        label_year = (hours - pd.Timedelta(hours=1)).year
        for y in YEARS:
            take = label_year == y
            expected = len(pd.date_range(f'{y}-01-01', f'{y+1}-01-01', freq='h', inclusive='left'))
            annual[name][str(y)] = dict(calendar_hours=expected, retained_hours=int(take.sum()),
                power_complete=float(np.isfinite(x[take, -1]).sum()/expected),
                all_scada_complete=float(np.isfinite(x[take]).all(1).sum()/expected))
    raw = np.concatenate(blocks)
    times = np.tile(hours.to_numpy(), len(units))
    turbines = np.repeat(np.arange(len(units)), len(hours))
    starts, potential = valid_window_starts(raw, times, turbines, 24, 4, 3)
    issue = pd.DatetimeIndex(times[starts+23]).tz_localize('UTC')
    excluded, retrofit = np.zeros(len(starts), bool), np.zeros(len(starts), np.float32)
    for i, name in enumerate(units):
        take = turbines[starts] == i
        excluded[take], retrofit[take] = window_works(issue[take], *bounds[name])
    targets = times[starts[:, None]+np.arange(24, 28)]
    fit = g1.period_mask(targets, ['2016-01-01', '2023-01-01'], hour_end=True).all(1)
    val = g1.period_mask(targets, ['2023-01-01', '2024-01-01'], hour_end=True).all(1)
    counts = {}
    for name, take in [('train_provisional', fit), ('val_provisional', val)]:
        keep = take & ~excluded
        counts[name] = dict(before_works=int(take.sum()), excluded_works=int((take & excluded).sum()),
            windows=int(keep.sum()), after_retrofit=int(retrofit[keep].sum()),
            after_retrofit_fraction=float(retrofit[keep].mean()) if keep.any() else None)
        np.savez(out/(name+'_windows.npz'), starts=starts[keep], turbine=turbines[starts[keep]],
                 issue=times[starts[keep]+23].astype('datetime64[ns]').astype(np.int64), retrofit=retrofit[keep])
    np.savez(out/'scada_rows.npz', raw=raw, hours=hours.to_numpy().astype('datetime64[ns]').astype(np.int64),
             units=np.asarray(units), channels=np.asarray(BASE_FEATURES+['power_last10', 'wind_last10', 'power']))
    g1.save_json(out/'curves_descriptive.json', dict(rule='raw bin means/medians; no causal retrofit claim', units=curves))
    g1.save_json(out/'construction_boundaries.json', examples)
    return dict(counts=counts, potential=potential, valid_scada_windows=len(starts),
                excluded_works_total=int(excluded.sum()), yearly_completeness=annual)


def prepare_nwp(root, hours, frame, meta, observations, stations, out, identities):
    center = meta[['Latitude', 'Longitude']].mean().to_numpy()
    points = np.vstack([center, [[s['latitude'], s['longitude']] for s in stations]])
    if not ((points[:, 0] >= 49.5) & (points[:, 0] <= 60) &
            (points[:, 1] >= -8) & (points[:, 1] <= 2.5)).all():
        raise ValueError('HoT point outside registered GFS region')
    pairs = [(ROOT/'output/gfs_g0/b2/audit_b2.json', ROOT/'output/gfs_g0/b2/acceptance.json'),
             (ROOT/'output/gfs_g0/b1/audit_b1_corrected.json', ROOT/'output/gfs_g0/b1/acceptance.json')]
    stores = []
    for j, (audit, acceptance) in enumerate(pairs):
        store = g1.PointForecasts(root, points, audit, json.loads(acceptance.read_text()))
        store.bridge(audit, acceptance, out/f'points_{j}.npz', 2023)
        stores.append(store)
        identities[str(audit)], identities[str(acceptance)] = g1.digest(audit), g1.digest(acceptance)
    union = g1.ForecastUnion(stores)
    class DevelopmentForecasts:
        def get(self, cycle):
            # Initial hours request Dec 2015: outside approved inputs, explicitly missing, no file open.
            if np.datetime64(cycle, 'h') < np.datetime64('2016-01-01', 'h'):
                return np.full((6, len(points)), np.nan)
            return union.get(cycle)
    wind = frame.groupby('timestamp_hour_end_utc').wind_speed_mps.mean().reindex(hours.tz_localize('UTC')).to_numpy()
    ws = np.column_stack([observations[s['src_id']].wind_speed.reindex(hours) for s in stations])
    directions = np.column_stack([observations[s['src_id']].wind_direction.reindex(hours) for s in stations])
    ws = np.where(ws >= 0, ws, np.nan)
    directions = np.where(ws > 0, directions, np.nan)
    hourly = g1.prepare_hourly(hours, DevelopmentForecasts(), ws, wind, directions, ['2016-01-01', '2023-01-01'])
    lead = g1.validate_lead_audit(hourly['audit'])
    np.savez(out/'nwp_hourly.npz', **{k: v for k, v in hourly.items() if isinstance(v, np.ndarray)})
    g1.save_json(out/'lead_audit.json', hourly['audit'])
    rules = hourly['calibration']
    g1.save_json(out/'calibration.json', dict(rules=rules, fit_bounds=hourly['fit_bounds'],
        points=points.tolist(), nwp_mean=hourly['nwp_mean'], nwp_scale=hourly['nwp_scale']))
    for store in stores:
        identities.update(store.identities)
    return dict(points=points.tolist(), lead_audit=lead, calibration=rules,
                cycle_missing_hours=int((hourly['cycle_ok'] == 0).sum()))


def run(root, out):
    out = out.resolve()
    if out.exists() or out.is_relative_to((root/'datasets').resolve()):
        raise ValueError('Require a fresh output outside raw datasets')
    from utils.kelmarsh_upstream_v31 import UPSTREAM_FEATURES
    out.mkdir(parents=True)
    identities = {}
    g1.save_json(out/'launch.json', dict(source_sha256=g1.digest(__file__),
        development_years=list(YEARS), training=False, automatic_restart=False,
        provisional_split={'train': '2016-2022', 'validation': '2023'},
        cutoff='strict valid time < 2024-01-01; no protected file opens'))
    try:
        directory = raw_inventory(root, identities)
        frame, meta, bounds = prepare_scada(directory, out)
        hours = pd.date_range('2016-01-01T01:00', '2024-01-01', freq='h', inclusive='left')
        windows = prepare_windows(frame, hours, bounds, out)
        obs, stations, upstream, midas_audit = prepare_midas(root, hours, identities)
        fit = g1.period_mask(hours, ['2016-01-01', '2023-01-01'], hour_end=True)
        source = upstream.to_numpy()
        if not np.isfinite(source[fit]).any(0).all():
            raise ValueError('Upstream feature has no finite provisional fit values')
        fill = np.nanmean(np.where(np.isfinite(source[fit]), source[fit], np.nan), axis=0)
        missing = ~np.isfinite(source).all(1)
        np.savez(out/'upstream_hourly.npz', raw=source, filled=np.where(np.isfinite(source), source, fill),
                 missing=missing, fit_fill=fill, hours=hours.to_numpy().astype('datetime64[ns]').astype(np.int64))
        nwp = prepare_nwp(root, hours, frame, meta, obs, stations, out, identities)
        g1.save_json(out/'input_hashes.json', identities)
        for p, expected in identities.items():
            if g1.digest(p) != expected:
                raise ValueError('Input changed during preparation: ' + p)
        g1.save_json(out/'summary.json', dict(status='PREPARED_DEVELOPMENT_ONLY', training=False,
            split_frozen=False, labels='unchanged six-tick hourly means; no imputation/clipping',
            mapping=MAPPING, base_channels=BASE_FEATURES+['power_last10', 'wind_last10', 'power'],
            upstream_channels=UPSTREAM_FEATURES+['upstream_missing'], retrofit='per-window issue>=works_end',
            representation='Unscaled SCADA rows + hourly upstream/NWP arrays + common window indices; model fitting prohibited',
            hour_convention='Official UTC period-end TimeStamp minus 10min, then [t,t+1h) label t+1h; last10 normalized minute 50, same aggregation as G1',
            timestamp_source='Hill_of_Towie_data_dictionary.md: TimeStamp giving the end of the period; subtract 10 minutes for the start',
            upstream_fill_provisional=fill.tolist(), **windows, midas=midas_audit, nwp=nwp,
            limitations=['Split awaits Claude freeze; all 2023 observations are development.',
                'First 2015 GFS cycles are masked without opening; final 2023 cycles crossing 2024 are not decoded.',
                'Retrofit curves are descriptive; seasonal/year effects are not controlled.']))
        artifacts = {p.name: g1.digest(p) for p in out.iterdir() if p.is_file()}
        g1.save_json(out/'completed.json', dict(status='COMPLETED', artifacts=artifacts,
                    source_sha256=g1.digest(__file__), input_count=len(identities)))
        print('[COMPLETED] HoT development preparation', flush=True)
    except Exception as exc:
        g1.save_json(out/'failed.json', dict(error_type=type(exc).__name__, error=str(exc), restart=False))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args.data_root.resolve(), args.output)
