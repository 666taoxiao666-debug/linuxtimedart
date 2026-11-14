"""G2b development fits and immutable freeze; sealed evaluation requires a separate authorization."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_hot_dev as hot
from scripts import prepare_g1_batches as prep
from scripts import train_g1 as runner


def load_arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def case_bounds(case):
    if case not in ('A', 'B'):
        raise ValueError('Diagnostic did not authorize fitting')
    cut = '2023-07-01' if case == 'A' else '2023-01-01'
    return {'train': ['2016-01-01', cut], 'val': [cut, '2024-01-01']}


def input_state(prepared, root, case):
    complete = json.loads((prepared/'completed.json').read_text())
    if complete['status'] != 'COMPLETED' or (prepared/'failed.json').exists():
        raise ValueError('Development preparation incomplete')
    for name, sha in complete['artifacts'].items():
        if prep.digest(prepared/name) != sha:
            raise ValueError('Prepared artifact changed: '+name)
    rows = load_arrays(prepared/'scada_rows.npz')
    upstream = load_arrays(prepared/'upstream_hourly.npz')
    hourly = load_arrays(prepared/'nwp_hourly.npz')
    hours = rows['hours'].astype('datetime64[ns]')
    if np.any(hours >= np.datetime64('2024-01-01')):
        raise ValueError('Protected hour in development input')
    units, raw = rows['units'], rows['raw']
    if len(raw) != len(hours)*len(units):
        raise ValueError('SCADA shape mismatch')
    times = np.tile(hours, len(units))
    turbine = np.repeat(np.arange(len(units)), len(hours))
    bounds = case_bounds(case)
    fit_hour = prep.period_mask(hours, bounds['train'], hour_end=True)
    fit_rows = np.tile(fit_hour, len(units))
    up = upstream['raw']
    if not np.array_equal(upstream['hours'].astype('datetime64[ns]'), hours):
        raise ValueError('Upstream hours not aligned')
    fill = np.nanmean(np.where(np.isfinite(up[fit_hour]), up[fit_hour], np.nan), axis=0)
    if not np.isfinite(fill).all():
        raise ValueError('No fit data for upstream feature')
    up_filled = np.where(np.isfinite(up), up, fill)
    features = np.column_stack([up_filled, (~np.isfinite(up).all(1)).astype(float)])
    hraw = np.column_stack([raw[:, :-1], np.tile(features, (len(units), 1)), raw[:, -1]])
    mean, scale = np.nanmean(hraw[fit_rows], 0), np.nanstd(hraw[fit_rows], 0)
    scale = np.where(scale == 0, 1., scale)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError('Invalid fit-only scaler')
    # Refit calibration from original development observations, not the old fitted transform.
    identities = {}
    obs, stations, _, _ = hot.prepare_midas(root, pd.DatetimeIndex(hours), identities)
    wind = np.nanmean(raw.reshape(len(units), len(hours), -1)[:, :, 0], axis=0)
    observed = np.column_stack([wind]+[obs[s['src_id']].wind_speed.reindex(hours).to_numpy() for s in stations])
    observed[:, 1:] = np.where(observed[:, 1:] >= 0, observed[:, 1:], np.nan)
    cal = []
    for j in range(13):
        rule = prep.fit_affine_calibration(observed[fit_hour, j], hourly['historical'][fit_hour, j])
        hourly['calibrated'][:, j], hourly['calibrated_mask'][:, j] = prep.apply_calibration(
            observed[:, j], np.isfinite(observed[:, j]), rule)
        cal.append(rule)
    hourly['nwp_mean'] = float(np.nanmean(hourly['historical'][fit_hour, 0]))
    hourly['nwp_scale'] = max(float(np.nanstd(hourly['historical'][fit_hour, 0])), 1e-3)
    stats = dict(case=case, bounds=bounds, upstream_fill=fill.tolist(), mean=mean.tolist(), scale=scale.tolist(),
        nwp_mean=hourly['nwp_mean'], nwp_scale=hourly['nwp_scale'], calibration=cal,
        retrofit_channel=case == 'A', preparation_sha256=prep.digest(prepared/'completed.json'),
        additional_inputs=identities)
    works_path = root/'datasets/external_validation/sealed/hill_of_towie/Hill_of_Towie_AeroUp_install_dates.csv'
    expected_inputs = json.loads((prepared/'input_hashes.json').read_text())
    if prep.digest(works_path) != expected_inputs[str(works_path)]:
        raise ValueError('Construction metadata changed')
    stats['works_sha256'] = prep.digest(works_path)
    return common_packs(raw, hours, units, up, hourly, stats, works_path, bounds, '2024-01-01'), stats


def common_packs(raw, hours, units, up, hourly, stats, works_path, bounds, cutoff):
    times = np.tile(hours, len(units))
    turbine = np.repeat(np.arange(len(units)), len(hours))
    fill, mean, scale = (np.asarray(stats[k]) for k in ('upstream_fill', 'mean', 'scale'))
    features = np.column_stack([np.where(np.isfinite(up), up, fill), (~np.isfinite(up).all(1)).astype(float)])
    hraw = np.column_stack([raw[:, :-1], np.tile(features, (len(units), 1)), raw[:, -1]])
    values = ((hraw-mean)/scale).astype(np.float32)
    starts, _ = hot.valid_window_starts(raw, times, turbine, 24, 4, 3)
    works = pd.read_csv(works_path).set_index('Turbine')
    excluded = np.zeros(len(starts), bool)
    retrofit = np.zeros(len(starts), np.float32)
    issue = pd.DatetimeIndex(times[starts+23]).tz_localize('UTC')
    for i, unit in enumerate(units):
        take = turbine[starts] == i
        excluded[take], retrofit[take] = hot.window_works(issue[take], *hot.works_bounds(*works.loc[unit].tolist()))
    idx = starts[:, None]+np.arange(24)
    targets = starts[:, None]+np.arange(24, 28)
    hidx = idx % len(hours)
    common = ~excluded & (hidx >= 3).all(1)
    common &= prep.available_cycle(hours[hidx[:, -1]])+np.timedelta64(18, 'h') < np.datetime64(cutoff)
    packs = {}
    for split, limits in bounds.items():
        keep = common & prep.period_mask(times[targets], limits, hour_end=True).all(1)
        if not keep.any():
            raise ValueError('Empty common split')
        x = values[idx[keep]]
        y = raw[targets[keep], -1].astype(np.float32)
        pack = dict(x21=x, x50=prep.nwp_input(x, hourly, hidx[keep]), truth=y,
            issue=times[starts[keep]+23].astype(np.int64), turbines=turbine[starts[keep]],
            times=times[targets[keep]].astype(np.int64), retrofit=retrofit[keep],
            anchor=raw[starts[keep]+23, -1].astype(np.float32),
            power_mean=np.array(mean[-1]), power_scale=np.array(scale[-1]),
            **{k:np.asarray(v, np.float32) for k,v in prep.station_batch(hourly, hidx[keep, -1]).items()})
        if not np.isfinite(x).all() or not np.isfinite(pack['x50']).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite common inputs or targets')
        packs[split] = pack
    return packs


def arm_pack(pack, arm, retrofit_channel):
    result = dict(pack)
    if arm in ('B0', 'N0'):
        result['x21'] = pack['x21'][..., [*range(8), 20]]
        result['x50'] = pack['x50'][..., [*range(8), *range(20, 50)]]
    if retrofit_channel:
        flag = np.broadcast_to(pack['retrofit'][:, None, None], (len(pack['truth']), 24, 1))
        for key in ('x21', 'x50'):
            x = result[key]
            result[key] = np.concatenate([x[..., :-1], flag, x[..., -1:]], -1).astype(np.float32)
    return result


def make_model(arm, pack, config, device):
    base_arm = 'B' if arm in ('B', 'B0') else 'N'
    key = 'x21' if base_arm == 'B' else 'x50'
    return prep.nd.G1TSMixerModel(base_arm, pack[key].shape[-1], None,
        blocks=config['blocks'], ff_dim=config['ff_dim'], dropout=config['dropout'],
        norm_eps=config['norm_eps']).to(device)


def sources():
    names = ['scripts/train_g2b.py', 'scripts/prepare_hot_dev.py', 'scripts/train_g1.py',
             'scripts/prepare_g1_batches.py', 'models/NWPDisplacement.py', 'models/TSMixerBasicSmall.py',
             'utils/kelmarsh_upstream_v31.py', 'data_provider/kelmarsh_multiyear.py',
             'scripts/screen_kelmarsh_upstream_observations.py', 'scripts/audit_hill_of_towie_e1.py',
             'scripts/acquire_hourly_wind_datasets.py', 'utils/kelmarsh_enrichment.py',
             'utils/kelmarsh_backbone_information.py', 'scripts/ramp_diagnostics.py']
    return {n: prep.digest(ROOT/n) for n in names}


def wait_for_idle_gpu(directory):
    import subprocess
    import time
    while True:
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if not occupied:
            prep.save_json(directory/'resource_state.json', dict(status='GPU_AVAILABLE', checked_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
            return
        prep.save_json(directory/'resource_state.json', dict(status='WAITING_FOR_GPU', existing_pids=occupied.splitlines(),
            checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(), interrupted_other_tasks=False))
        time.sleep(30)


def fit_development(root, prepared, diagnostic, directory, protocol_sha, device):
    import fcntl
    protocol_path = ROOT/'configs/g2b_protocol.json'
    if prep.digest(protocol_path) != protocol_sha:
        raise ValueError('Protocol hash mismatch')
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol)
    done = json.loads((diagnostic/'completed.json').read_text())
    for name, sha in done['artifacts'].items():
        if prep.digest(diagnostic/name) != sha:
            raise ValueError('Diagnostic changed')
    decision = done['decision']
    if decision['case'] not in ('A', 'B'):
        raise ValueError('Registered diagnostic stop')
    if directory.exists():
        raise ValueError('No resume or repeat in existing directory')
    directory.parent.mkdir(parents=True, exist_ok=True)
    with (directory.parent/'g2b.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory.mkdir()
        snapshot = sources()
        prep.save_json(directory/'launch.json', dict(protocol_sha256=protocol_sha,
            decision=decision, source_code=snapshot, formal_fits=14, sealed_access=False,
            diagnostic_sha256=prep.digest(diagnostic/'completed.json'), device=device))
        try:
            packs, stats = input_state(prepared, root, decision['case'])
            prep.save_json(directory/'fit_statistics.json', stats)
            prep.save_json(directory/'cohorts.json', {k:dict(windows=len(v['truth']),
                first_target=str(v['times'].min().astype('datetime64[ns]')),
                last_target=str(v['times'].max().astype('datetime64[ns]')),
                retrofit_fraction=float(v['retrofit'].mean())) for k,v in packs.items()})
            if device.startswith('cuda'):
                wait_for_idle_gpu(directory)
            if sources() != snapshot or prep.digest(protocol_path) != protocol_sha:
                raise ValueError('Source or protocol changed before fitting')
            fits = []
            original = runner.make_model
            runner.make_model = make_model
            try:
                for arm, seeds in protocol['seeds'].items():
                    train = arm_pack(packs['train'], arm, decision['case'] == 'A')
                    val = arm_pack(packs['val'], arm, decision['case'] == 'A')
                    for seed in seeds:
                        key = f'{arm}_{seed}'
                        result = runner.fit_one(arm, seed, train, val, protocol['fit'], directory/key, device)
                        fits.append(dict(key=key, checkpoint=f'{key}/selected.pt', **result))
                        print(f'[FIT {len(fits)}/14] {key} val_mae={result["validation_mae"]:.6f}', flush=True)
                    del train, val
            finally:
                runner.make_model = original
            if sources() != snapshot or prep.digest(protocol_path) != protocol_sha:
                raise ValueError('Source or protocol changed during fitting')
            freeze = dict(status='FROZEN_DEVELOPMENT_ONLY', fits=fits, decision=decision,
                protocol_sha256=protocol_sha, source_code=snapshot,
                diagnostic_sha256=prep.digest(diagnostic/'completed.json'),
                preparation_sha256=prep.digest(prepared/'completed.json'),
                fit_statistics_sha256=prep.digest(directory/'fit_statistics.json'),
                cohorts_sha256=prep.digest(directory/'cohorts.json'), sealed_access=False,
                sealed_script='scripts/train_g2b.py', sealed_script_sha256=snapshot['scripts/train_g2b.py'])
            runner.check_freeze(freeze, 14, directory)
            prep.save_json(directory/'selection_freeze.json', freeze)
            prep.save_json(directory/'completed.json', dict(status='FROZEN_DEVELOPMENT_ONLY',
                fits=14, freeze_sha256=prep.digest(directory/'selection_freeze.json'),
                sealed_access=False, wait_for_claude=True))
            print('[FROZEN] 14 fits; stop before sealed access', flush=True)
        except Exception as exc:
            prep.save_json(directory/'failed.json', dict(error_type=type(exc).__name__, error=str(exc), restart=False))
            raise


def validate_protocol(protocol):
    parent = json.loads((ROOT/'configs/g1_protocol.json').read_text())
    expected = {a: list(range(2024, 2029 if a == 'N' else 2027)) for a in ('B', 'N0', 'B0', 'N')}
    if protocol['seeds'] != expected or protocol['formal_fits'] != 14 or protocol['fit'] != parent['fit']:
        raise ValueError('Registered fitting budget or recipe changed')
    if protocol['test_years'] != [2024, 2025] or protocol['protected_years'] != [2026]:
        raise ValueError('Registered sealed scope changed')


def scoped_functions(module, **changes):
    """Clone function globals without modifying the already frozen shared modules."""
    import types
    namespace = {**vars(module), **changes}
    for name, value in vars(module).items():
        if isinstance(value, types.FunctionType) and value.__module__ == module.__name__:
            namespace[name] = types.FunctionType(value.__code__, namespace, name, value.__defaults__, value.__closure__)
            namespace[name].__kwdefaults__ = value.__kwdefaults__
    return types.SimpleNamespace(**namespace)


def frozen_gate(directory, expected_sha):
    path = directory/'selection_freeze.json'
    if prep.digest(path) != expected_sha or (directory/'failed.json').exists():
        raise ValueError('Invalid freeze identity')
    frozen = json.loads(path.read_text())
    runner.check_freeze(frozen, 14, directory)
    if frozen['source_code'] != sources():
        raise ValueError('Frozen source changed')
    for name, key in [('fit_statistics.json', 'fit_statistics_sha256'), ('cohorts.json', 'cohorts_sha256')]:
        if prep.digest(directory/name) != frozen[key]:
            raise ValueError('Frozen fit metadata changed')
    protocol_path = ROOT/'configs/g2b_protocol.json'
    if prep.digest(protocol_path) != frozen['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol)
    return frozen, protocol


def sealed_midas(root, hours, identities):
    """Future authorized test preparation: only registered 2023 history and 2024/25."""
    from utils.kelmarsh_upstream_v31 import hourly_upstream
    audit = ROOT/'output/hill_of_towie_e1/l0b_r2'
    q = json.loads(hot.verified(audit/'qualified_stations.json', hot.QUALIFIED_SHA, identities).read_text())
    manifest_path = audit/'download_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    hot.verified(audit/'download_plan.json', manifest['plan_sha256'], identities)
    identities[str(manifest_path)] = prep.digest(manifest_path)
    parts = {sid: [] for sid in q['qualified_station_ids']}
    for item in manifest['files']:
        sid, year = item['src_id'], item['year']
        if sid not in parts or year not in (2023, 2024, 2025):
            continue
        if item['sealed'] != (year >= 2024):
            raise ValueError('MIDAS sealed routing differs')
        base = ('datasets/upstream_observations/sealed/hill_of_towie_midas_v202607' if year >= 2024
                else 'datasets/upstream_observations/hill_of_towie_midas_v202607/development')
        path = root/base/str(sid)/Path(item['path']).name
        if path.stat().st_size != item['bytes'] or hot.sha(path, 'md5') != item['md5']:
            raise ValueError('MIDAS official identity differs')
        hot.verified(path, item['sha256'], identities)
        data = hot.screen.read_badc_csv(path)
        col = hot.screen.column
        data = data[pd.to_numeric(data[col(data, 'version_num')], errors='coerce') == 1].copy()
        time = pd.to_datetime(data[col(data, 'ob_time', 'ob_end_time')], utc=True, errors='coerce')
        if (time.dropna().dt.year != year).any():
            raise ValueError('MIDAS row year differs')
        units = data.attrs.get('units', {})
        speed, direction = col(data, 'wind_speed', 'mean_wind_speed'), col(data, 'wind_direction', 'mean_wind_dir')
        factor = {'knots': 1852/3600, 'knot': 1852/3600, 'kt': 1852/3600,
                  'm/s': 1., 'm s-1': 1., 'ms-1': 1.}.get(units.get(speed))
        if factor is None or units.get(direction) not in ('degrees', 'degree', 'degrees true', 'degrees_true', 'deg true'):
            raise ValueError('MIDAS units differ')
        pressure = col(data, 'msl_pressure', required=False)
        if pressure and units.get(pressure) not in ('hpa', 'mb', 'millibars'):
            raise ValueError('MIDAS pressure units differ')
        frame = pd.DataFrame(dict(time=time.dt.tz_localize(None),
            wind_speed=pd.to_numeric(data[speed], errors='coerce')*factor,
            wind_direction=pd.to_numeric(data[direction], errors='coerce'),
            msl_pressure=pd.to_numeric(data[pressure], errors='coerce') if pressure else np.nan))
        frame = frame[pd.to_numeric(data[col(data, 'src_id')], errors='coerce') == sid].replace([np.inf, -np.inf], np.nan)
        frame.loc[frame.wind_speed < 0, 'wind_speed'] = np.nan
        frame.loc[(frame.wind_direction < 0) | (frame.wind_direction > 360) | (frame.wind_speed == 0), 'wind_direction'] = np.nan
        frame.loc[frame.msl_pressure <= 0, 'msl_pressure'] = np.nan
        frame = frame[frame.time.notna() & (frame.time.dt.minute == 0) & (frame.time.dt.second == 0)]
        parts[sid].append(frame)
    observations = {}
    for sid, frames in parts.items():
        if not frames:
            raise ValueError('No registered MIDAS observations for frozen station')
        frame = pd.concat(frames)
        if (frame.groupby('time')[['wind_speed', 'wind_direction', 'msl_pressure']].nunique(dropna=False) > 1).any().any():
            raise ValueError('Conflicting test observations; no reconciliation authorized')
        observations[sid] = frame.drop_duplicates('time').set_index('time').sort_index()
    return observations, q['stations'], hourly_upstream(observations, q['stations'], hours, 1)


def decode_test_points(root, manifest_path, expected_sha, points, output):
    """Accept an explicitly approved byte inventory; never discover latest files."""
    import xarray as xr
    if prep.digest(manifest_path) != expected_sha:
        raise ValueError('Test GFS inventory hash differs')
    manifest = json.loads(manifest_path.read_text())
    if not manifest['status'].startswith('PASS'):
        raise ValueError('Test GFS inventory not accepted')
    cache, identities = {}, {}
    for record in manifest['records']:
        key = record['cycle']
        if int(key[:4]) not in (2023, 2024, 2025) or not record['complete']:
            continue
        cycle = np.datetime64(dt.datetime.strptime(key, '%Y%m%d%H'), 'h')
        if cycle+np.timedelta64(18, 'h') >= np.datetime64('2026-01-01'):
            continue
        path = (root/record['path']).resolve()
        if not path.is_relative_to((root/'datasets/external/gfs_g0').resolve()) or path.name != f'gfs_{key}_cropped.nc':
            raise ValueError('Unregistered GFS location')
        if path.stat().st_size != record['bytes'] or prep.digest(path) != record['sha256']:
            raise ValueError('Test GFS byte identity changed')
        with xr.open_dataset(path, cache=False, decode_timedelta=True) as ds:
            step = ds.step.values/np.timedelta64(1, 'h')
            if not np.array_equal(step, prep.LEADS) or not np.array_equal(ds.valid_time.values, cycle+prep.LEADS.astype('timedelta64[h]')):
                raise ValueError('Test GFS lead/valid-time identity differs')
            uv = [prep.bilinear(ds[n].transpose('step', 'latitude', 'longitude').values,
                               ds.latitude.values, ds.longitude.values, points) for n in ('u10', 'v10')]
            speed = np.hypot(*uv)
            if not np.isfinite(speed).all() or key in cache:
                raise ValueError('Nonfinite or duplicate accepted GFS cycle')
            cache[key] = speed
            identities[str(path)] = record['sha256']
    if not cache:
        raise ValueError('Empty approved GFS inventory')
    np.savez(output, **cache)
    prep.save_json(Path(output).with_suffix('.identities.json'), identities)


def prepare_sealed(root, freeze_dir, expected_freeze, directory, gfs_manifest, gfs_sha, authorization):
    """Not called by development mode. Authorization and one-shot marker precede labels."""
    import subprocess
    from collections import Counter
    if authorization != 'G2b-2024-2025-once':
        raise ValueError('Separate sealed evaluation authorization required')
    frozen, _ = frozen_gate(freeze_dir, expected_freeze)
    # Exclusive marker in the frozen run, not a replaceable destination, prevents another evaluation.
    with (freeze_dir/'test_evaluation_started.json').open('x') as stream:
        json.dump(dict(freeze_sha256=expected_freeze, output=str(directory), years=[2024, 2025],
                       started_utc=dt.datetime.now(dt.timezone.utc).isoformat()), stream)
    if directory.exists() or directory.resolve().is_relative_to((root/'datasets').resolve()):
        raise ValueError('Fresh evaluation directory outside raw inputs required')
    directory.mkdir(parents=True)
    try:
        stats = json.loads((freeze_dir/'fit_statistics.json').read_text())
        identities = {}
        h = scoped_functions(hot, YEARS=(2023, 2024, 2025), CUTOFF=pd.Timestamp('2026-01-01', tz='UTC'))
        raw_dir = h.raw_inventory(root, identities)
        works_path = raw_dir/'Hill_of_Towie_AeroUp_install_dates.csv'
        if prep.digest(works_path) != stats['works_sha256']:
            raise ValueError('Frozen construction identity changed')
        boundary = dict(cells=Counter(), masked_hours=set(), duplicate_keys=0,
                        diagnostic_sha256=frozen['diagnostic_sha256'])
        frame, meta, works = h.prepare_scada(raw_dir, directory, boundary)
        hours = pd.date_range('2023-01-01T01:00', '2026-01-01', freq='h', inclusive='left')
        units = sorted(works)
        raw = np.concatenate([h.scada_values(frame[frame.turbine_id == unit]
            .set_index('timestamp_hour_end_utc').reindex(hours.tz_localize('UTC'))) for unit in units])
        obs, stations, upstream = sealed_midas(root, hours, identities)
        points = np.vstack([meta[['Latitude', 'Longitude']].mean().to_numpy(),
                            [[s['latitude'], s['longitude']] for s in stations]])
        prep.save_json(directory/'points.json', points.tolist())
        # Existing decoding runtime only; no package installation or changes to its environment.
        subprocess.run(['/home/XT/miniconda3/envs/gfs_g0/bin/python', str(Path(__file__).resolve()),
            '--mode', 'decode-test-points', '--data-root', str(root), '--gfs-manifest', str(gfs_manifest),
            '--gfs-sha256', gfs_sha, '--points', str(directory/'points.json'),
            '--output', str(directory/'points.npz'), '--authorization', authorization], check=True)
        points_cache = load_arrays(directory/'points.npz')
        identities.update(json.loads((directory/'points.identities.json').read_text()))
        identities[str(gfs_manifest)] = gfs_sha
        hourly = frozen_hourly(hours.to_numpy(), points_cache, stats)
        bounds = {str(y): [f'{y}-01-01', f'{y+1}-01-01'] for y in (2024, 2025)}
        packs = common_packs(raw, hours.to_numpy(), units, upstream.to_numpy(), hourly, stats,
                             works_path, bounds, '2026-01-01')
        for year, pack in packs.items():
            np.savez(directory/f'{year}.npz', **pack)
        for path, sha in identities.items():
            if prep.digest(path) != sha:
                raise ValueError('Input identity changed during test preparation')
        prep.save_json(directory/'input_hashes.json', identities)
        prep.save_json(directory/'prepared_complete.json', dict(freeze_sha256=expected_freeze,
            packs={y: prep.digest(directory/f'{y}.npz') for y in packs},
            source_code=sources(), no_refit=True, years=[2024, 2025], protected_years=[2026]))
        return packs
    except Exception as exc:
        prep.save_json(directory/'failed.json', dict(error_type=type(exc).__name__, error=str(exc), restart=False))
        raise


def frozen_hourly(hours, cache, stats):
    """N/B need only GFS inputs; station-block arrays are unused, fixed finite zeros."""
    hours = np.asarray(hours, dtype='datetime64[h]')
    current = np.zeros((len(hours), 13, 13), np.float32)
    masks = np.zeros_like(current)
    historical = np.full((len(hours), 13), np.nan)
    cycles = prep.available_cycle(hours)
    age = ((hours-cycles)/np.timedelta64(1, 'h')).astype(int)
    ok = np.zeros(len(hours))
    for i, (cycle, a) in enumerate(zip(cycles, age)):
        if cycle+np.timedelta64(18, 'h') >= np.datetime64('2026-01-01'):
            continue
        key = pd.Timestamp(cycle).strftime('%Y%m%d%H')
        values = cache.get(key, np.full((6, 13), np.nan))
        hist, valid, _ = prep.interpolate_leads(values, np.array([a]))
        historical[i] = np.where(valid[0], hist[0], np.nan)
        current[i], masks[i], _ = prep.interpolate_leads(values, a+np.arange(-5, 8))
        ok[i] = np.isfinite(values).all()
    return dict(hours=hours, historical=historical, current=current, masks=masks, age=age, cycle_ok=ok,
        calibrated=np.zeros((len(hours), 13)), calibrated_mask=np.zeros((len(hours), 13)),
        directions=np.full((len(hours), 12), np.nan), nwp_mean=stats['nwp_mean'], nwp_scale=stats['nwp_scale'])


def joint_test_decisions(reports, alpha=.05):
    from utils.kelmarsh_backbone_information import holm_adjust
    questions = ('Q1h', 'Qe')
    p = [max(reports[str(y)]['primary'][q]['newey_west']['p_value_two_sided'] for y in (2024, 2025)) for q in questions]
    adjusted = holm_adjust(p)
    return {q: dict(max_annual_p=p[i], holm_p=adjusted[i], passed=bool(adjusted[i] < alpha and
        all(reports[str(y)]['primary'][q]['gain'] > 0 and reports[str(y)]['primary'][q]['ci95_low'] > 0
            for y in (2024, 2025)))) for i, q in enumerate(questions)}


def evaluate_sealed(freeze_dir, expected_freeze, directory, packs, device):
    import torch
    frozen, protocol = frozen_gate(freeze_dir, expected_freeze)
    if not (freeze_dir/'test_evaluation_started.json').exists() or (directory/'completed.json').exists():
        raise ValueError('Missing one-shot gate or already completed')
    reports, hashes = {}, {}
    try:
        for year, pack in packs.items():
            predictions = {}
            for arm, seeds in protocol['seeds'].items():
                arm_data = arm_pack(pack, arm, frozen['decision']['case'] == 'A')
                predictions[arm] = []
                for seed in seeds:
                    model = make_model(arm, arm_data, protocol['fit'], device)
                    model.load_state_dict(torch.load(freeze_dir/f'{arm}_{seed}/selected.pt', map_location=device, weights_only=True))
                    pred = runner.predict(model, arm_data, device)
                    if pred.shape != pack['truth'].shape or not np.isfinite(pred).all():
                        raise ValueError('Invalid or misaligned test prediction')
                    path = directory/f'{year}_{arm}_{seed}.npy'
                    np.save(path, pred)
                    hashes[path.name] = prep.digest(path)
                    predictions[arm].append(pred)
                    del model
            predictions['N3'] = predictions['N'][:3]
            predictions['N_ensemble'] = [np.mean(predictions['N'], axis=0)]
            reports[year] = runner.evaluate_arrays(predictions, pack['truth'], pack['issue'].astype('datetime64[ns]'),
                None, protocol['statistics'], pack['anchor'], 2300,
                dict(Q1h=('N3', 'B'), Qe=('N_ensemble', 'N'), N_vs_N0=('N3', 'N0'),
                     B_vs_B0=('B', 'B0'), N0_vs_B0=('N0', 'B0')))
            for arm, metrics in reports[year]['metrics'].items():
                for metric, pred in zip(metrics, predictions[arm]):
                    metric['one_minus_nrmse'] = 1-metric['rmse']/2300
                    metric['mean_bias_kw'] = float((pred-pack['truth']).mean())
        frozen_gate(freeze_dir, expected_freeze)
        prep.save_json(directory/'results.json', dict(years=reports, joint=joint_test_decisions(reports),
            secondary=['N_vs_N0'], descriptive=['B_vs_B0', 'N0_vs_B0', 'ramps', 'mean_bias_kw', 'one_minus_nrmse'],
            cross_farm_pooling='Descriptive only, requires previously saved farm estimates; no extra test access'))
        prep.save_json(directory/'completed.json', dict(status='COMPLETED', freeze_sha256=expected_freeze,
            predictions=hashes, results_sha256=prep.digest(directory/'results.json'), new_fits=0))
    except Exception as exc:
        prep.save_json(directory/'failed.json', dict(error_type=type(exc).__name__, error=str(exc), restart=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['development', 'sealed', 'decode-test-points'], default='development')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--diagnostic', type=Path)
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--freeze-sha256')
    parser.add_argument('--gfs-manifest', type=Path)
    parser.add_argument('--gfs-sha256')
    parser.add_argument('--points', type=Path)
    parser.add_argument('--authorization')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.mode == 'development':
        if not all((args.prepared, args.diagnostic, args.protocol_sha256)):
            parser.error('Development requires prepared, diagnostic and protocol-sha256')
        fit_development(args.data_root, args.prepared, args.diagnostic, args.output, args.protocol_sha256, args.device)
    else:
        if args.authorization != 'G2b-2024-2025-once':
            parser.error('Separate sealed authorization required')
        if args.mode == 'decode-test-points':
            decode_test_points(args.data_root, args.gfs_manifest, args.gfs_sha256,
                               np.asarray(json.loads(args.points.read_text())), args.output)
        else:
            packs = prepare_sealed(args.data_root, args.freeze, args.freeze_sha256, args.output,
                                   args.gfs_manifest, args.gfs_sha256, args.authorization)
            evaluate_sealed(args.freeze, args.freeze_sha256, args.output, packs, args.device)


if __name__ == '__main__':
    main()
