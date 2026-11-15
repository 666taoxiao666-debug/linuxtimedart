"""G2b #84: 2023 rehearsal, then sealed preparation only; no sealed inference CLI."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_g2b as train
from scripts.train_g2b import (frozen_gate, common_packs, frozen_hourly, decode_test_points,
    arm_pack, make_model, evaluate_sealed, joint_test_decisions, pooled_descriptive)
from scripts.audit_hill_of_towie_e1 import conflict_missing

hot, prep = train.hot, train.prep
FREEZE_SHA = '981f3ae27e6792f5da71c177af1c24a0e91c4dfcd880322773fb4f9fe34607f6'
GFS_PYTHON = '/home/XT/miniconda3/envs/gfs_g0/bin/python'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sealed_midas(root, hours, identities, years):
    from utils.kelmarsh_upstream_v31 import hourly_upstream
    if tuple(years) not in ((2023,), (2023, 2024, 2025)):
        raise ValueError('Unapproved MIDAS years')
    audit = ROOT/'output/hill_of_towie_e1/l0b_r2'
    q = read_json(hot.verified(audit/'qualified_stations.json', hot.QUALIFIED_SHA, identities))
    manifest = read_json(audit/'download_manifest.json')
    hot.verified(audit/'download_plan.json', manifest['plan_sha256'], identities)
    identities[str(audit/'download_manifest.json')] = prep.digest(audit/'download_manifest.json')
    if len(q['qualified_station_ids']) != 12 or sorted(s['src_id'] for s in q['stations']) != q['qualified_station_ids']:
        raise ValueError('Frozen station identity differs')
    parts = {sid: [] for sid in q['qualified_station_ids']}
    conflicts, seen = {}, set()
    for item in manifest['files']:
        sid, year = item['src_id'], item['year']
        if sid not in parts or year not in years:
            continue
        if item['sealed'] != (year >= 2024) or (sid, year) in seen:
            raise ValueError('MIDAS routing or duplicate file differs')
        seen.add((sid, year))
        base = ('datasets/upstream_observations/sealed/hill_of_towie_midas_v202607' if year >= 2024
                else 'datasets/upstream_observations/hill_of_towie_midas_v202607/development')
        path = root/base/str(sid)/Path(item['path']).name
        if path.stat().st_size != item['bytes'] or hot.sha(path, 'md5') != item['md5']:
            raise ValueError('MIDAS official identity differs')
        hot.verified(path, item['sha256'], identities)
        data, counts = conflict_missing(hot.screen.read_badc_csv(path), year)
        conflicts[f'{sid}/{year}'] = counts
        col = hot.screen.column
        data = data[pd.to_numeric(data[col(data, 'version_num')], errors='coerce') == 1].copy()
        time = pd.to_datetime(data[col(data, 'ob_time', 'ob_end_time')], utc=True, errors='coerce')
        units = data.attrs.get('units', {})
        speed = col(data, 'wind_speed', 'mean_wind_speed')
        direction = col(data, 'wind_direction', 'mean_wind_dir')
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
    if seen != {(sid, year) for sid in parts for year in years}:
        raise ValueError('Missing registered station/year')
    obs = {sid: pd.concat(frames).drop_duplicates('time').set_index('time').sort_index()
           for sid, frames in parts.items()}
    return hourly_upstream(obs, q['stations'], hours, 1), conflicts


def merge_boundary_cycle(dev, extra):
    import xarray as xr
    if set(dev.data_vars) != set(extra.data_vars):
        raise ValueError('GFS variable structure differs')
    for coord in ('latitude', 'longitude'):
        if not np.array_equal(dev[coord], extra[coord]):
            raise ValueError('GFS grid differs')
    if np.intersect1d(dev.step, extra.step).size:
        raise ValueError('Overlapping boundary leads')
    return xr.concat([dev, extra], dim='step', data_vars='minimal', coords='minimal', compat='equals').sortby('step')


def cycle_complete(ds, cycle, points, template):
    for coord in ('latitude', 'longitude'):
        if not np.array_equal(ds[coord], template[coord]):
            raise ValueError('GFS grid differs from development')
    if set(ds.data_vars) != set(template.data_vars):
        raise ValueError('GFS variables differ from development')
    c = np.datetime64(dt.datetime.strptime(cycle, '%Y%m%d%H'), 'h')
    if (not np.array_equal(ds.step.values / np.timedelta64(1, 'h'), prep.LEADS)
            or not np.array_equal(ds.valid_time.values, c + prep.LEADS.astype('timedelta64[h]'))):
        return False
    return all(np.isfinite(prep.bilinear(ds[name].transpose('step', 'latitude', 'longitude').values,
        ds.latitude.values, ds.longitude.values, points)).all() for name in ('u10', 'v10'))


def gfs_inventory(root, directory, points, sealed):
    """Runs in the existing decoding environment. Only enumerated <=2025 paths open."""
    import xarray as xr
    acceptance_path = ROOT/'output/gfs_g0/b1/acceptance.json'
    audit_path = ROOT/'output/gfs_g0/b1/audit_b1_corrected.json'
    acceptance = read_json(acceptance_path)
    if prep.digest(audit_path) != acceptance['source_audit_sha256']:
        raise ValueError('B1 corrected audit identity differs')
    audit = read_json(audit_path)
    dev_records = {r['cycle']: r for r in audit['records'] if r['cycle'].startswith('2023')}
    identities = {str(audit_path): prep.digest(audit_path), str(acceptance_path): prep.digest(acceptance_path)}
    gfs = root/'datasets/external/gfs_g0'
    messages, crop = {}, None
    if sealed:
        manifest_path = root/'output/external_acquisition/gfs_g0/manifest_b1.json'
        hot.verified(manifest_path, audit['manifest_sha256'], identities)
        manifest = read_json(manifest_path)
        if manifest['task'] != 'G0_GFS_B1':
            raise ValueError('Wrong GFS manifest')
        for entry in manifest['messages'].values():
            if entry['sealed'] and 2023 <= int(entry['cycle'][:4]) <= 2025:
                messages.setdefault(entry['cycle'], []).append(entry)
        crop_path = root/'scripts/acquire_gfs_g0.py'
        hot.verified(crop_path, '2ada9f7374689f17bd61c82b4e881d938eb2db2cec6551620c7dbbf9c8b3932f', identities)
        spec = importlib.util.spec_from_file_location('registered_crop', crop_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        crop = module.crop_grib_to_netcdf
    first = gfs/'dev/2023/gfs_2023010100_cropped.nc'
    hot.verified(first, dev_records['2023010100']['sha256'], identities)
    with xr.open_dataset(first, decode_timedelta=True) as f:
        template = f.load()
    records = []
    cutoff = pd.Timestamp('2026-01-01' if sealed else '2024-01-01')
    for stamp in pd.date_range('2023-01-01', cutoff, freq='6h', inclusive='left'):
        key = stamp.strftime('%Y%m%d%H')
        row = dict(cycle=key, complete=False)
        records.append(row)
        if stamp + pd.Timedelta(hours=18) >= cutoff:
            row['reason'] = 'valid_time_cutoff'
            continue
        dev = None
        if stamp.year == 2023:
            record = dev_records[key]
            path = gfs/f'dev/2023/gfs_{key}_cropped.nc'
            if path.stat().st_size != record['bytes']:
                raise ValueError('Development GFS size differs')
            hot.verified(path, record['sha256'], identities)
            if key not in acceptance['complete_cycles']:
                row['reason'] = 'development_quarantine'
                continue
            with xr.open_dataset(path, decode_timedelta=True) as f:
                dev = f.load()
        if key > '2023123100':
            entries = sorted(messages.get(key, []), key=lambda e: (e['lead'], int(e['idx_line'].split(':')[0])))
            payload, available = [], set()
            for entry in entries:
                lead = int(entry['lead'][1:])
                valid = stamp + pd.Timedelta(hours=lead)
                if (lead not in prep.LEADS or valid != pd.Timestamp(entry['valid_time'])
                        or valid < pd.Timestamp('2024-01-01') or valid >= cutoff):
                    raise ValueError('GFS message routing differs')
                name = f"{key}_{entry['lead']}_{entry['var']}_{entry['level'].replace(' ', '_')}.grib2"
                raw_path = gfs/'sealed_raw'/key[:4]/name
                if not raw_path.is_file():
                    continue
                if raw_path.stat().st_size != entry['bytes']:
                    raise ValueError('GFS message size differs')
                hot.verified(raw_path, entry['sha256'], identities)
                payload.append(raw_path.read_bytes())
                available.add((lead, entry['var'], entry['level']))
            needed = [int(x) for x in prep.LEADS if stamp + pd.Timedelta(hours=int(x)) >= pd.Timestamp('2024-01-01')]
            required = {(lead, var, level) for lead in needed for var, level in
                        [('UGRD', '10 m above ground'), ('VGRD', '10 m above ground'), ('PRMSL', 'mean sea level')]}
            if not required <= available:
                row['reason'] = 'incomplete_required_messages'
                continue
            path = gfs/f'test/{key[:4]}/gfs_{key}_cropped.nc'
            if path.exists():
                raise ValueError('Refuse overwriting existing converted GFS: '+str(path))
            crop(b''.join(payload), str(path))
            if dev is not None:
                with xr.open_dataset(path, decode_timedelta=True) as f:
                    merged = merge_boundary_cycle(dev, f.load())
                temporary = path.with_suffix('.merged.nc')
                merged.to_netcdf(temporary)
                temporary.replace(path)
        with xr.open_dataset(path, decode_timedelta=True) as f:
            row['complete'] = bool(cycle_complete(f, key, points, template))
        row.update(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=prep.digest(path))
        if len(records) % 200 == 0:
            print(f'[GFS] checked {len(records)} cycles', flush=True)
    annual = {}
    for year in ((2024, 2025) if sealed else (2023,)):
        rows = [r for r in records if r['cycle'].startswith(str(year))]
        count = sum(r['complete'] for r in rows)
        annual[str(year)] = dict(complete=count, denominator=len(rows), passes_95pct=count/len(rows) >= .95)
    result = dict(status='PASS' if all(x['passes_95pct'] for x in annual.values()) else 'FAIL',
                  records=records, annual=annual, source_identities=identities, protected_years=[2026])
    inventory = directory/'gfs_test_inventory.json'
    prep.save_json(inventory, result)
    if result['status'] != 'PASS':
        raise ValueError('GFS completeness below 95%; SCADA must remain unopened')
    decode_test_points(root, inventory, prep.digest(inventory), points, directory/'points.npz')


def preparation_config(sealed):
    if sealed:
        return ((2023, 2024, 2025), ('2023-01-01T01:00', '2026-01-01'),
                {str(y): [f'{y}-01-01', f'{y+1}-01-01'] for y in (2024, 2025)}, '2026-01-01')
    return ((2023,), ('2023-01-01T01:00', '2024-01-01'),
            {'val': ['2023-07-01', '2024-01-01']}, '2024-01-01')


def start_preparation(root, directory, freeze_dir, years):
    if directory.exists() or directory.resolve().is_relative_to((root/'datasets').resolve()):
        raise ValueError('Fresh output directory outside datasets required')
    frozen, protocol = frozen_gate(freeze_dir, FREEZE_SHA)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory/'test_preparation_started.json').open('x') as stream:
        json.dump(dict(freeze_sha256=FREEZE_SHA, years=years, sealed_inference=False,
            started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            entrypoint_sha256=prep.digest(Path(__file__))), stream)
    return frozen, protocol


def prepare_sealed(root, freeze_dir, directory, years, hour_axis, bounds, cutoff):
    """One reconstruction function for rehearsal and sealed preparation, without models."""
    sealed = tuple(years) == (2023, 2024, 2025)
    if (tuple(years), tuple(hour_axis), bounds, cutoff) != preparation_config(sealed):
        raise ValueError('Unapproved preparation configuration')
    frozen, _ = start_preparation(root, directory, freeze_dir, years)
    try:
        stats = read_json(freeze_dir/'fit_statistics.json')
        identities = {}
        # Only turbine/station coordinates, no SCADA archive inspection before GFS/MIDAS gates.
        metadata = root/'datasets/external_validation/sealed/hill_of_towie/Hill_of_Towie_turbine_metadata.csv'
        record = read_json(hot.verified(root/'output/external_acquisition/hill_of_towie/record_22662930.json',
                                       hot.RECORD_SHA, identities))
        official = next(f for f in record['files'] if f['key'] == metadata.name)
        if metadata.stat().st_size != official['size'] or hot.sha(metadata, 'md5') != official['checksum'].removeprefix('md5:'):
            raise ValueError('Turbine metadata identity differs')
        identities[str(metadata)] = prep.digest(metadata)
        qpath = ROOT/'output/hill_of_towie_e1/l0b_r2/qualified_stations.json'
        q = read_json(hot.verified(qpath, hot.QUALIFIED_SHA, identities))
        meta = pd.read_csv(metadata, encoding='utf-8-sig')
        points = np.vstack([meta[['Latitude', 'Longitude']].mean().to_numpy(),
                            [[s['latitude'], s['longitude']] for s in q['stations']]])
        prep.save_json(directory/'points.json', points.tolist())
        subprocess.run([GFS_PYTHON, str(Path(__file__).resolve()), '--mode', 'gfs', '--data-root', str(root),
            '--output', str(directory), '--scope', 'sealed' if sealed else 'rehearsal'], check=True)
        inventory = read_json(directory/'gfs_test_inventory.json')
        identities.update(inventory['source_identities'])
        identities.update(read_json(directory/'points.identities.json'))
        identities[str(directory/'gfs_test_inventory.json')] = prep.digest(directory/'gfs_test_inventory.json')
        hours = pd.date_range(*hour_axis, freq='h', inclusive='left')
        upstream, conflicts = sealed_midas(root, hours, identities, years)
        prep.save_json(directory/'midas_conflicts.json', conflicts)
        # Only now may selected SCADA archives be opened.
        h = train.scoped_functions(hot, YEARS=tuple(years), CUTOFF=pd.Timestamp(cutoff, tz='UTC'))
        raw_dir = h.raw_inventory(root, identities)
        works_path = raw_dir/'Hill_of_Towie_AeroUp_install_dates.csv'
        if prep.digest(works_path) != stats['works_sha256']:
            raise ValueError('Frozen construction identity differs')
        boundary = dict(cells=Counter(), masked_hours=set(), duplicate_keys=0,
                        diagnostic_sha256=frozen['diagnostic_sha256'])
        frame, _, works = h.prepare_scada(raw_dir, directory, boundary)
        units = sorted(works)
        raw = np.concatenate([h.scada_values(frame[frame.turbine_id == unit]
            .set_index('timestamp_hour_end_utc').reindex(hours.tz_localize('UTC'))) for unit in units])
        hourly = frozen_hourly(hours.to_numpy(), train.load_arrays(directory/'points.npz'), stats)
        packs = common_packs(raw, hours.to_numpy(), units, upstream.to_numpy(), hourly, stats,
                             works_path, bounds, cutoff)
        counts = {}
        for name, pack in packs.items():
            np.savez(directory/f'{name}.npz', **pack)
            counts[name] = dict(windows=len(pack['truth']), retrofit_fraction=float(pack['retrofit'].mean()))
        for path, expected in identities.items():
            if prep.digest(path) != expected:
                raise ValueError('Input changed during preparation: '+path)
        frozen_gate(freeze_dir, FREEZE_SHA)
        prep.save_json(directory/'input_hashes.json', identities)
        prep.save_json(directory/'prepared_complete.json', dict(status='PREPARED_ONLY',
            freeze_sha256=FREEZE_SHA, packs={k: prep.digest(directory/f'{k}.npz') for k in packs},
            counts=counts, annual_gfs=inventory['annual'], scada_masked_hours=len(boundary['masked_hours']),
            upstream_missing_fraction=float((~np.isfinite(upstream.to_numpy()).all(1)).mean()),
            entrypoint_sha256=prep.digest(Path(__file__)), frozen_source_code=train.sources(),
            no_refit=True, model_loaded=False, label_statistics=False, years=list(years), protected_years=[2026]))
        return packs
    except Exception as exc:
        prep.save_json(directory/'failed.json', dict(error_type=type(exc).__name__, error=str(exc), restart=False))
        raise


def rehearse(root, freeze_dir, directory):
    import torch
    packs = prepare_sealed(root, freeze_dir, directory, *preparation_config(False))
    try:
        frozen, protocol = frozen_gate(freeze_dir, FREEZE_SHA)
        pack = packs['val']
        if len(pack['truth']) != 30056:
            raise ValueError('Rehearsal window count differs from 30056')
        reports = []
        # CPU only; no interference with other GPU jobs.
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        for fit in frozen['fits']:
            data = arm_pack(pack, fit['arm'], True)
            model = make_model(fit['arm'], data, protocol['fit'], 'cpu')
            model.load_state_dict(torch.load(freeze_dir/fit['checkpoint'], map_location='cpu', weights_only=True))
            prediction = train.runner.predict(model, data, 'cpu')
            mae = float(np.abs(prediction-pack['truth']).mean())
            trained = read_json(freeze_dir/fit['key']/'trained.json')
            delta = abs(mae-trained['validation_mae'])
            reports.append(dict(key=fit['key'], mae=mae, frozen_mae=trained['validation_mae'], delta_kw=delta))
            prep.save_json(directory/'rehearsal_metrics.json', reports)
            if not np.isfinite(mae) or delta > .01:
                raise ValueError('Rehearsal MAE gate failed: '+fit['key'])
            del model, prediction
        frozen_gate(freeze_dir, FREEZE_SHA)
        prep.save_json(directory/'rehearsal_passed.json', dict(status='PASS', windows=30056, models=14,
            max_delta_kw=max(r['delta_kw'] for r in reports), freeze_sha256=FREEZE_SHA,
            entrypoint_sha256=prep.digest(Path(__file__)), prepared_sha256=prep.digest(directory/'prepared_complete.json'),
            metrics_sha256=prep.digest(directory/'rehearsal_metrics.json')))
    except Exception as exc:
        prep.save_json(directory/'failed.json', dict(error_type=type(exc).__name__, error=str(exc), restart=False))
        raise


def rehearsal_gate(directory):
    passed = read_json(directory/'rehearsal_passed.json')
    if (passed['status'] != 'PASS' or passed['models'] != 14 or passed['windows'] != 30056
            or passed['max_delta_kw'] > .01 or passed['freeze_sha256'] != FREEZE_SHA
            or passed['entrypoint_sha256'] != prep.digest(Path(__file__)) or (directory/'failed.json').exists()
            or passed['prepared_sha256'] != prep.digest(directory/'prepared_complete.json')
            or passed['metrics_sha256'] != prep.digest(directory/'rehearsal_metrics.json')):
        raise ValueError('Rehearsal evidence gate failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['rehearsal', 'prepare', 'gfs'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--rehearsal', type=Path)
    parser.add_argument('--scope', choices=['rehearsal', 'sealed'], default='rehearsal')
    args = parser.parse_args()
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    if args.mode == 'gfs':
        started = read_json(args.output/'test_preparation_started.json')
        if started['years'] != ([2023, 2024, 2025] if args.scope == 'sealed' else [2023]):
            raise ValueError('GFS scope differs from preparation marker')
        gfs_inventory(args.data_root, args.output, np.asarray(read_json(args.output/'points.json')), args.scope == 'sealed')
    elif args.mode == 'rehearsal':
        rehearse(args.data_root, args.freeze, args.output)
    else:
        rehearsal_gate(args.rehearsal)
        prepare_sealed(args.data_root, args.freeze, args.output, *preparation_config(True))


PROTOCOL_SHA = 'da809d691e93b7bb284fc7ce282f1c16a59ecd0fe5342f68ceab0e8530966a53'
PREPARATION_SHA = 'cbdc1b980c652095978366b7f3f2ccdfa5160a52d59f930cb448fb7c9aee97fe'
EVALUATION_PACKS = {
    '2024': ('a4dfc88037a7b14fecccf6fbcd56e1ed65b9a3d9f996d43d0446e7a65c942420', 59587),
    '2025': ('07f34631dece57d7fafb52cfe647c58e3745fddd58e2f0909dc8018462952c02', 56857)}
REHEARSAL_PACK_SHA = 'c8395144cad7dcee83602efb5c5e8d6156d3ed5318fc4c924925956a8b266a4e'
CANONICAL_FREEZE = Path('/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/G2/g2b_4471499_shared_r1/formal')


def evaluation_inputs(prepared, smoke=False):
    complete = read_json(prepared/'prepared_complete.json')
    if (complete['status'] != 'PREPARED_ONLY' or complete['entrypoint_sha256'] != PREPARATION_SHA
            or complete['freeze_sha256'] != FREEZE_SHA or (prepared/'failed.json').exists()):
        raise ValueError('Prepared evidence differs')
    expected = {'val': (REHEARSAL_PACK_SHA, 30056)} if smoke else EVALUATION_PACKS
    if complete['packs'] != {k: v[0] for k, v in expected.items()}:
        raise ValueError('Registered pack hashes differ')
    for key, (sha, count) in expected.items():
        if complete['counts'][key]['windows'] != count or prep.digest(prepared/f'{key}.npz') != sha:
            raise ValueError('Data package hash/count differs: '+key)
    return expected


def reserve_evaluation(freeze_dir, directory, marker):
    if directory.exists():
        raise ValueError('Evaluation output already exists')
    if (freeze_dir/'test_evaluation_started.json').exists():
        raise ValueError('Evaluation already attempted; retry forbidden')
    directory.mkdir(parents=True, exist_ok=False)
    # x is the authoritative race-safe gate, including simultaneous launches.
    with (freeze_dir/'test_evaluation_started.json').open('x') as stream:
        json.dump(marker, stream, indent=2)
    prep.save_json(directory/'evaluation_launch.json', marker)


def evaluate_once(prepared, freeze_dir, directory, smoke=False, smoke_result=None):
    frozen, _ = frozen_gate(freeze_dir, FREEZE_SHA)
    if frozen['protocol_sha256'] != PROTOCOL_SHA:
        raise ValueError('Registered protocol differs')
    expected = evaluation_inputs(prepared, smoke)
    if smoke:
        origin = read_json(freeze_dir/'smoke_copy_origin.json')
        if (freeze_dir.resolve() == CANONICAL_FREEZE.resolve()
                or origin['source'] != str(CANONICAL_FREEZE) or origin['freeze_sha256'] != FREEZE_SHA):
            raise ValueError('Smoke requires a separate verified freeze copy')
    else:
        if freeze_dir.resolve() != CANONICAL_FREEZE.resolve():
            raise ValueError('Formal evaluation requires the canonical freeze directory')
        evidence = read_json(smoke_result/'smoke_passed.json')
        if (evidence['status'] != 'PASS' or evidence['predictions'] != 28
                or evidence['entrypoint_sha256'] != prep.digest(Path(__file__))
                or (smoke_result/'failed.json').exists()
                or evidence['completed_sha256'] != prep.digest(smoke_result/'completed.json')):
            raise ValueError('Smoke evidence gate failed')
    marker = dict(freeze_sha256=FREEZE_SHA, protocol_sha256=PROTOCOL_SHA, smoke=smoke,
        output=str(directory.resolve()), prepared=str(prepared.resolve()),
        entrypoint_sha256=prep.digest(Path(__file__)), package_sha256={k:v[0] for k,v in expected.items()},
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), device='cpu', threads=2, retry=False)
    reserve_evaluation(freeze_dir, directory, marker)
    try:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if hasattr(os, 'sched_getaffinity'):
            os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
        packs = {k: train.load_arrays(prepared/f'{k}.npz') for k in expected}
        evaluation_inputs(prepared, smoke)
        for key, pack in packs.items():
            if len(pack['truth']) != expected[key][1]:
                raise ValueError('Actual window count differs')
        if smoke:
            packs = {'2024': packs['val'], '2025': packs['val']}
        evaluate_sealed(freeze_dir, FREEZE_SHA, directory, packs, 'cpu')
        completed = read_json(directory/'completed.json')
        if completed['status'] != 'COMPLETED' or len(completed['predictions']) != 28:
            raise ValueError('Incomplete evaluation')
        for name, sha in completed['predictions'].items():
            if prep.digest(directory/name) != sha or not np.isfinite(np.load(directory/name, allow_pickle=False)).all():
                raise ValueError('Prediction verification failed')
        if smoke:
            prep.save_json(directory/'smoke_passed.json', dict(status='PASS', predictions=28,
                entrypoint_sha256=prep.digest(Path(__file__)), completed_sha256=prep.digest(directory/'completed.json'),
                development_only=True, interpret_results=False))
        else:
            exports = {}
            for year, pack in packs.items():
                small = {k: pack[k] for k in ('truth', 'issue', 'turbines', 'times', 'anchor')}
                for fit in frozen['fits']:
                    small['pred_'+fit['key']] = np.load(directory/f"{year}_{fit['key']}.npy", allow_pickle=False)
                path = directory/f'{year}_arrays.npz'
                np.savez_compressed(path, **small)
                exports[year] = dict(sha256=prep.digest(path), bytes=path.stat().st_size, keys=list(small))
            prep.save_json(directory/'array_exports.json', exports)
    except Exception as exc:
        prep.save_json(directory/'failed.json', dict(error_type=type(exc).__name__, error=str(exc), restart=False))
        raise


def evaluation_main():
    parser = argparse.ArgumentParser(description='G2c one-shot CPU evaluation')
    parser.add_argument('--mode', choices=['evaluate'], required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-result', type=Path)
    args = parser.parse_args()
    if not args.smoke and args.smoke_result is None:
        parser.error('Formal evaluation requires --smoke-result')
    evaluate_once(args.prepared, args.freeze, args.output, args.smoke, args.smoke_result)


if __name__ == '__main__':
    if '--mode=evaluate' in sys.argv or any(sys.argv[i:i+2] == ['--mode', 'evaluate'] for i in range(len(sys.argv))):
        evaluation_main()
    else:
        main()
