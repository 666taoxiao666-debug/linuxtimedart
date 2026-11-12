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


def inspect_dataset(ds, cycle):
    errors, missing, optional_issues = [], [], []
    result = dict(errors=errors, missing_messages=missing, optional_issues=optional_issues,
                  complete=False, optional_complete=False, optional_level_verified=False, ranges={})
    expected_time = np.datetime64(dt.datetime.strptime(cycle, '%Y%m%d%H'))
    for key in ('time', 'step', 'valid_time', 'latitude', 'longitude'):
        if key not in ds.coords:
            errors.append('missing_coordinate:' + key)
    if errors:
        return result
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
        param = 260074 if pressure else (131 if name.startswith('u') else 132)
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
        for h in LEADS:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.root.resolve()):
        raise ValueError('Report must be outside the input data tree')
    if args.out.exists():
        raise FileExistsError('Use a fresh report path; no overwrites')
    report = audit(args.root, args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))
    return 0 if report['status'] == 'PASS_REQUIRED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
