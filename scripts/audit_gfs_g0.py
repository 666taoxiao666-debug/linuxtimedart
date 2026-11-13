"""Read-only B2 NetCDF audit. Never downloads, trains, or decodes sealed values."""
import argparse
import calendar
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

LEADS = (3, 6, 9, 12, 15, 18)
REQUIRED = ('u10', 'v10', 'prmsl')
OPTIONAL = ('u100', 'v100')
CUTOFF = np.datetime64('2024-01-01')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def cycles(year):
    start = dt.datetime(year, 1, 1)
    return [(start + dt.timedelta(hours=6 * i)).strftime('%Y%m%d%H')
            for i in range((366 if calendar.isleap(year) else 365) * 4)]


def inspect_dataset(ds, cycle, mode='B2'):
    errors, missing, optional_issues = [], [], []
    result = dict(errors=errors, missing_messages=missing, optional_issues=optional_issues,
                  complete=False, optional_complete=False, optional_level_verified=False, ranges={})
    expected_time = np.datetime64(dt.datetime.strptime(cycle, '%Y%m%d%H'))
    required_leads = tuple(h for h in LEADS if expected_time + np.timedelta64(h, 'h') < CUTOFF) if mode == 'B1' else LEADS
    for key in ('time', 'step', 'valid_time', 'latitude', 'longitude'):
        if key not in ds.coords:
            errors.append('missing_coordinate:' + key)
    if errors:
        return result
    # cfgrib squeezes a single lead into a scalar coordinate. Restore only
    # its dimension; the sealed-time guard below still precedes value access.
    if mode == 'B1' and ds.step.ndim == 0 and ds.valid_time.ndim == 0:
        ds = ds.expand_dims('step')
        ds = ds.assign_coords(valid_time=('step', np.atleast_1d(ds.valid_time.values)))
    time = ds.time.values
    step = ds.step.values
    valid = ds.valid_time.values
    if time.size != 1 or not np.issubdtype(time.dtype, np.datetime64):
        errors.append('invalid_cycle_coordinate')
        return result
    if not np.issubdtype(step.dtype, np.timedelta64) or step.ndim != 1:
        errors.append('invalid_step_coordinate')
        return result
    if not np.issubdtype(valid.dtype, np.datetime64) or valid.shape != step.shape:
        errors.append('invalid_valid_time_coordinate')
        return result
    # Coordinates only so far: stop before touching data variables for sealed/f000 inputs.
    if np.any(valid >= CUTOFF) or np.any(time + step >= CUTOFF):
        errors.append('sealed_valid_time_refused')
        return result
    hours = step / np.timedelta64(1, 'h')
    if np.any(hours == 0):
        errors.append('f000_refused')
        return result
    if time.reshape(-1)[0] != expected_time:
        errors.append('cycle_mismatch')
    if not np.array_equal(valid, time + step):
        errors.append('valid_time_mismatch')
    if len(set(hours.tolist())) != len(hours) or not set(hours.tolist()) <= set(LEADS):
        errors.append('unexpected_or_duplicate_leads')
    for key, expected in [('latitude', np.arange(49.5, 60.01, .25)),
                          ('longitude', np.arange(-8, 2.51, .25))]:
        values = ds[key].values
        canonical = (values + 180) % 360 - 180 if key == 'longitude' else values
        if values.ndim != 1 or len(np.unique(canonical)) != 43 or values.size != 43:
            errors.append('grid_shape_or_duplicate:' + key)
        elif not np.allclose(np.sort(canonical), expected, rtol=0, atol=1e-8):
            errors.append('grid_extent:' + key)
    if errors:
        return result
    level_verified, optional_complete = True, True
    for name in REQUIRED + OPTIONAL:
        issues = errors if name in REQUIRED else optional_issues
        if name not in ds.data_vars:
            if name in REQUIRED:
                missing.extend(f'{name}:f{h:03}' for h in LEADS)
            else:
                optional_complete = False
                level_verified = False
            continue
        var = ds[name]
        if set(var.dims) != {'step', 'latitude', 'longitude'} or var.ndim != 3:
            issues.append('variable_shape:' + name)
            optional_complete = False
            continue
        pressure = name == 'prmsl'
        units = 'Pa' if pressure else 'm s**-1'
        level_type = 'meanSea' if pressure else 'heightAboveGround'
        param = 260074 if pressure else ((165 if name == 'u10' else 166) if mode == 'B1' and name in REQUIRED else (131 if name.startswith('u') else 132))
        if (var.attrs.get('units') != units or var.attrs.get('GRIB_typeOfLevel') != level_type
                or var.attrs.get('GRIB_paramId') != param):
            issues.append('variable_identity:' + name)
        level = var.attrs.get('GRIB_level')
        target = 0 if pressure else (100 if name in OPTIONAL else 10)
        if level is None and level_type in ds.coords and ds[level_type].size == 1:
            level = float(ds[level_type].values)
        if level != target:
            if name in OPTIONAL:
                level_verified = False
            else:
                issues.append('level_identity:' + name)
        arr = var.transpose('step', 'latitude', 'longitude').values
        finite = arr[np.isfinite(arr)]
        result['ranges'][name] = [float(finite.min()), float(finite.max())] if finite.size else None
        for h in required_leads:
            indices = np.flatnonzero(hours == h)
            values = arr[indices[0]] if len(indices) == 1 else np.array([np.nan])
            if np.isnan(values).all():
                if name in REQUIRED:
                    missing.append(f'{name}:f{h:03}')
                else:
                    optional_complete = False
                continue
            if not np.isfinite(values).all():
                issues.append(f'nonfinite:{name}:f{h:03}')
            elif (np.any((values < 90000) | (values > 110000)) if pressure
                  else np.any(np.abs(values) >= 80)):
                issues.append(f'physical_range:{name}:f{h:03}')
    result.update(complete=not errors and not missing,
                  optional_complete=optional_complete and not optional_issues,
                  optional_level_verified=level_verified)
    return result


def audit_file(path, cycle, entry):
    record = dict(cycle=cycle, file=str(path), complete=False, errors=[])
    if not path.is_file():
        record['missing_file'] = True
        if entry is not None:
            record['errors'].append('manifest_file_missing')
        return record
    before = path.stat()
    record.update(bytes=before.st_size, sha256=sha256(path))
    if entry is None:
        record['errors'].append('file_not_in_manifest')
    elif record['sha256'] != entry.get('nc_sha256') or before.st_size != entry.get('dev_bytes'):
        record['errors'].append('manifest_hash_or_size_mismatch')
    if record['errors']:
        return record
    try:
        with xr.open_dataset(path, cache=False, decode_timedelta=True) as ds:
            record.update(inspect_dataset(ds, cycle))
    except Exception as exc:
        record['errors'].append('decode_error:' + type(exc).__name__ + ':' + str(exc))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        record['errors'].append('file_changed_during_audit')
    if bool(entry.get('complete_18')) != record['complete']:
        record['errors'].append('manifest_completeness_mismatch')
    record['complete'] = record['complete'] and not record['errors']
    return record


def audit_b1(root, manifest_path):
    """Independent development decode and sealed byte-only validation; no repairs."""
    root = Path(root).resolve()
    before = sha256(manifest_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if manifest.get('task') != 'G0_GFS_B1':
        raise ValueError('Expected G0_GFS_B1 manifest')
    entries, messages = manifest['cycles'], manifest['messages']
    records, sealed_failures, sealed_count, sealed_bytes = [], [], 0, 0
    for year in range(2021, 2024):
        for cycle in cycles(year):
            path = root/'dev'/str(year)/f'gfs_{cycle}_cropped.nc'
            row = dict(cycle=cycle, file=str(path), complete=False, errors=[])
            entry = entries.get(cycle)
            if not path.is_file():
                row['missing_file'] = True
                if cycle not in manifest.get('user_accepted_missing_cycles', []):
                    row['errors'].append('unapproved_missing_cycle')
            else:
                stat = path.stat()
                row.update(bytes=stat.st_size, sha256=sha256(path))
                if entry is None or stat.st_size != entry.get('dev_bytes'):
                    row['errors'].append('manifest_byte_mismatch')
                else:
                    try:
                        with xr.open_dataset(path, cache=False, decode_timedelta=True) as ds:
                            result = inspect_dataset(ds, cycle, mode='B1')
                        row.update(result)
                        if row['missing_messages']:
                            row['errors'].append('incomplete_required_fields')
                    except Exception as exc:
                        row['errors'].append('decode_error:'+type(exc).__name__+':'+str(exc))
                if (stat.st_size, stat.st_mtime_ns) != (path.stat().st_size, path.stat().st_mtime_ns):
                    row['errors'].append('changed_during_audit')
                row['complete'] = row['complete'] and not row['errors']
            records.append(row)
        print(f'B1 development {year} checked', flush=True)
    expected_paths = set()
    for key, entry in messages.items():
        valid = np.datetime64(entry['valid_time'])
        if np.isnat(valid) or bool(entry['sealed']) != bool(valid >= CUTOFF):
            sealed_failures.append(dict(key=key, error='valid_time_sealed_flag_mismatch'))
            continue
        if not entry['sealed']:
            continue
        cycle, lead, var, level = entry['cycle'], entry['lead'], entry['var'], entry['level']
        expected_valid = np.datetime64(dt.datetime.strptime(cycle, '%Y%m%d%H')) + np.timedelta64(int(lead[1:]), 'h')
        if valid != expected_valid or int(lead[1:]) not in LEADS or var not in ('UGRD', 'VGRD', 'PRMSL'):
            sealed_failures.append(dict(key=key, error='message_identity_mismatch'))
            continue
        path = root/'sealed_raw'/cycle[:4]/f'{cycle}_{lead}_{var}_{level.replace(" ", "_")}.grib2'
        if not path.resolve().is_relative_to(root/'sealed_raw'):
            raise ValueError('Sealed path outside approved root')
        expected_paths.add(path.resolve())
        if not path.is_file():
            sealed_failures.append(dict(key=key, error='missing_file'))
            continue
        stat = path.stat()
        value = sha256(path)  # bytes only; never open_dataset/cfgrib for this tree
        sealed_count += 1
        sealed_bytes += stat.st_size
        if stat.st_size != entry['bytes'] or value != entry['sha256']:
            sealed_failures.append(dict(key=key, error='byte_or_hash_mismatch'))
        if (stat.st_size, stat.st_mtime_ns) != (path.stat().st_size, path.stat().st_mtime_ns):
            sealed_failures.append(dict(key=key, error='changed_during_hash'))
        if sealed_count % 10000 == 0:
            print(f'B1 sealed bytes hashed: {sealed_count}', flush=True)
    actual_paths = {p.resolve() for p in (root/'sealed_raw').rglob('*') if p.is_file()}
    extras = sorted(str(p) for p in actual_paths-expected_paths)
    annual = {}
    for year in range(2021, 2024):
        rows = [r for r in records if r['cycle'].startswith(str(year))]
        count = sum(r['complete'] for r in rows)
        annual[str(year)] = dict(denominator=len(rows), complete=count, completeness_pct=100*count/len(rows), passes_95pct=count/len(rows) >= .95)
    dev_bytes = sum(p.stat().st_size for p in (root/'dev').rglob('*') if p.is_file())
    actual_sealed_bytes = sum(p.stat().st_size for p in actual_paths)
    dl = manifest['stats'].get('successful_message_download_bytes')
    budget = dict(download_successful_payload_bytes=dl, download_total_bytes_verified=False,
        download_note=manifest['stats'].get('download_measurement_note'),
        dev_storage_bytes=dev_bytes, sealed_storage_bytes=actual_sealed_bytes,
        dev_storage_pass=dev_bytes <= 20_000_000_000, sealed_storage_pass=actual_sealed_bytes <= 110_000_000_000,
        successful_payload_under_limit=dl is not None and dl <= 300_000_000_000)
    failures = [r['cycle'] for r in records if r['errors']]
    stable = before == sha256(manifest_path)
    data_pass = not failures and not sealed_failures and not extras and stable and all(a['passes_95pct'] for a in annual.values())
    passed = data_pass and budget['dev_storage_pass'] and budget['sealed_storage_pass'] and budget['download_total_bytes_verified']
    return dict(status='PASS_REQUIRED' if passed else 'FAIL', task='G0_GFS_B1',
        completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), script_sha256=sha256(__file__),
        manifest_sha256=before, manifest_unchanged=stable, annual=annual, failures=failures,
        sealed_failures=sealed_failures, unexpected_sealed_files=extras, budget=budget,
        sealed_files_hashed=sealed_count, sealed_bytes_hashed=sealed_bytes, records=records,
        limitations=['Development NetCDF hashes computed here: manifest supplies sizes but no registered NetCDF hashes.',
                    'Unique successful download payload excludes unmetered historical failed/repeated traffic.',
                    'Sealed content was hashed only; no GRIB decode or numerical interpretation.'])


def audit(root, manifest_path):
    root = Path(root).resolve()
    digest = sha256(manifest_path)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('task') != 'G0_GFS_B2':
        raise ValueError('This version accepts only B2 2016-2020 NetCDF manifests')
    expected = [c for y in range(2016, 2021) for c in cycles(y)]
    entries = manifest['cycles']
    unexpected_keys = sorted(set(entries) - set(expected))
    records = []
    for i, c in enumerate(expected):
        path = root / c[:4] / f'gfs_{c}_cropped.nc'
        if not path.resolve().is_relative_to(root):
            raise ValueError('Data path leaves approved B2 dev root')
        records.append(audit_file(path, c, entries.get(c)))
        if (i + 1) % 500 == 0:
            print(f'Checked {i + 1}/{len(expected)}', flush=True)
    annual = {}
    for year in range(2016, 2021):
        rows = [r for r in records if r['cycle'].startswith(str(year))]
        count = sum(r['complete'] for r in rows)
        annual[str(year)] = dict(denominator=len(rows), complete=count,
            completeness_pct=100 * count / len(rows), passes_95pct=count / len(rows) >= .95,
            missing_cycles=[r['cycle'] for r in rows if r.get('missing_file')],
            incomplete_cycles=[r['cycle'] for r in rows if not r['complete'] and not r.get('missing_file')],
            optional_complete=sum(r.get('optional_complete', False) for r in rows),
            optional_level_verified=sum(r.get('optional_level_verified', False) for r in rows))
    known_paths = {str((root / c[:4] / f'gfs_{c}_cropped.nc').resolve()) for c in expected}
    extras = [str(p) for year in range(2016, 2021) for p in (root / str(year)).glob('*.nc')
              if str(p.resolve()) not in known_paths]
    failures = [r['cycle'] for r in records if r['errors']]
    stable = digest == sha256(manifest_path)
    passed = (not failures and not extras and not unexpected_keys and stable
              and all(r['passes_95pct'] for r in annual.values()))
    return dict(status='PASS_REQUIRED' if passed else 'FAIL', scope='B2 dev 2016-2020 only',
        completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        script_sha256=sha256(__file__), manifest_sha256=digest, manifest_unchanged=stable,
        annual=annual, failures=failures, unexpected_files=extras, unexpected_manifest_keys=unexpected_keys,
        files_hashed=sum('sha256' in r for r in records), bytes_hashed=sum(r.get('bytes', 0) for r in records),
        optional_100m_approved=all(r['optional_level_verified'] / r['denominator'] >= .95
                                  and r['optional_complete'] / r['denominator'] >= .95 for r in annual.values()),
        limitations=['Raw monthly tar hashes are manifest claims only: source archives were purged.',
                     'No B1 GRIB or sealed values decoded; forecast publication timing not proven by this audit.'],
        records=records)


def derive_b2_acceptance(audit_path):
    """Apply the approved R2 quarantine without changing the original audit."""
    expected_sha = 'ebea2c9877bc3995456d7e7ed76041ea7c5a4f82559b9bcab294b5d0de4c861d'
    if sha256(audit_path) != expected_sha:
        raise ValueError('R2 decision applies only to the reviewed B2 audit')
    report = json.loads(Path(audit_path).read_text(encoding='utf-8'))
    rows = report['records']
    accepted = [r['cycle'] for r in rows if r['complete']]
    excluded = [dict(cycle=r['cycle'], cycle_ok=0,
                     reason=(['missing_file'] if r.get('missing_file') else
                             r['errors'] + r.get('missing_messages', [])))
                for r in rows if not r['complete']]
    assert len(rows) == 7308 and len(accepted) == 7300 and len(excluded) == 8
    assert all(v['passes_95pct'] for v in report['annual'].values())
    return dict(status='PASS_REQUIRED_WITH_QUARANTINE', conclusion='B2 必需字段通过',
                scope='B2 2016-2020; only listed complete cycles are usable',
                source_audit_sha256=expected_sha, source_audit_status=report['status'],
                decision='HANDOFF 第72次 / DECISIONS 阶段G0c R1-B1采纳隔离',
                complete_cycles=accepted, excluded_cycles=excluded, annual=report['annual'],
                denominator=7308, complete_count=7300, incomplete_count=7, missing_count=1,
                **{'100m_used': False},
                all_arms_cycle_ok_zero_for_excluded=True, previous_cycle_fallback=False,
                limitations=report['limitations'])


def derive_b1_acceptance(audit_path, traffic_estimate):
    """Apply only the #77 approved quarantine to the immutable independent audit."""
    expected = '6f2dbba37d8d256a97bd4cc2761173b72409edd9c529a370a735e91ce279d2a7'
    if sha256(audit_path) != expected:
        raise ValueError('R2 decision applies only to the reviewed B1 audit')
    report = json.loads(Path(audit_path).read_text(encoding='utf-8'))
    rows = report['records']
    accepted = [r['cycle'] for r in rows if r['complete'] and not r['errors']]
    excluded = [dict(cycle=r['cycle'], cycle_ok=0,
        reason=['missing_file'] if r.get('missing_file') else r['errors'] + r.get('missing_messages', []))
        for r in rows if not r['complete'] or r['errors']]
    quarantine = {'2021020200', '2021020312', '2021020318', '2021020412',
                  '2021020612', '2021020618', '2021020712', '2022112918', '2022113006'}
    if (len(rows) != 4380 or len(set(accepted)) != 4371 or {r['cycle'] for r in excluded} != quarantine
            or report['sealed_failures'] or report['unexpected_sealed_files']
            or not report['manifest_unchanged'] or not all(v['passes_95pct'] for v in report['annual'].values())
            or not report['budget']['dev_storage_pass'] or not report['budget']['sealed_storage_pass']):
        raise ValueError('Audit differs from approved quarantine or storage gates')
    return dict(status='PASS_REQUIRED_WITH_QUARANTINE', scope='B1 development 2021-2023; sealed bytes only',
        source_audit_sha256=expected, source_audit_status=report['status'],
        decision='HANDOFF #77 / DECISIONS G1c R1-B1/B2', complete_cycles=accepted,
        excluded_cycles=excluded, annual=report['annual'], complete_count=4371,
        denominator=4380, missing_count=7, incomplete_count=2,
        all_arms_cycle_ok_zero_for_excluded=True, previous_cycle_fallback=False,
        budget={**report['budget'], 'download_total_verified': False, 'user_accepted_gap': '2026-11-13',
                'log_based_estimate': traffic_estimate}, limitations=report['limitations'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('B2', 'B1'), default='B2')
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.root.resolve()):
        raise ValueError('Report must be outside the input data tree')
    if args.out.exists():
        raise FileExistsError('Use a fresh report path; no overwrites')
    report = audit_b1(args.root, args.manifest) if args.mode == 'B1' else audit(args.root, args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))
    return 0 if report['status'] == 'PASS_REQUIRED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
