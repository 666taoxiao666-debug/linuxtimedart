"""Acquire pinned MIDAS Open observations with private Bearer authentication.

Development CSVs are 2016-2022 only. 2023/2024 are stored separately and only
hashed, never parsed. Public directory size/MD5 metadata pins every file.
No credentials enter command arguments, manifests, exceptions or stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.acquire_hourly_wind_datasets import digest, save_json
from scripts.screen_kelmarsh_upstream_observations import ARMS, DATASET, RADIUS_KM, load_metadata, load_year, read_badc_csv, screen

BASE = '/badc/ukmo-midas-open/data/uk-hourly-weather-obs/dataset-version-202607'
META = 'midas-open_uk-hourly-weather-obs_dv-202607_station-metadata.csv'
YEARS = set(range(2016, 2025))


class AuthorizationError(RuntimeError):
    pass


def safe_path(path):
    if not path.startswith(BASE+'/') and path != BASE:
        raise ValueError('Unrecognized CEDA dataset path')
    if any(x in ('.', '..') for x in path.split('/')) or any(x in path for x in ('?', '#', '\\')):
        raise ValueError('Invalid provider path')
    return path


def checked_roots(development, sealed):
    development, sealed = Path(development).resolve(), Path(sealed).resolve()
    if ('sealed' not in [p.lower() for p in sealed.parts]
            or 'sealed' in [p.lower() for p in development.parts]
            or development == sealed or development in sealed.parents or sealed in development.parents):
        raise ValueError('Development and explicitly sealed roots must be separate')
    return development, sealed


class Archive:
    def __init__(self, token_file):
        self.token_file = Path(token_file).expanduser()

    def headers(self):
        info = self.token_file.stat()
        if os.name != 'nt' and (stat.S_IMODE(info.st_mode) & 0o077 or info.st_uid != os.getuid()):
            raise AuthorizationError('Token file must be private and owned by this user')
        value = self.token_file.read_text().strip()
        if len(value.split('.')) != 3 or '\n' in value or '\r' in value:
            raise AuthorizationError('Invalid token file format')
        return {'Authorization': 'Bearer '+value}

    def get(self, path, *, listing=False, offset=0):
        path = safe_path(path)
        url = ('https://data.ceda.ac.uk' if listing else 'https://dap.ceda.ac.uk')+path
        if listing:
            url = url.rstrip('/')+'/?json'
        headers = {} if listing else self.headers()
        if offset:
            headers['Range'] = f'bytes={offset}-'
        for attempt in range(6):
            try:
                response = requests.get(url, headers=headers, timeout=(30, 120),
                                        allow_redirects=False, stream=not listing)
            except requests.RequestException as exc:
                if attempt == 5:
                    raise RuntimeError('CEDA network request failed: '+type(exc).__name__) from None
                time.sleep(min(5*2**attempt, 60))
                continue
            if response.status_code in (429, 503):
                delay = response.headers.get('Retry-After', '30')
                response.close()
                if not delay.isdigit() or int(delay) > 300:
                    raise RuntimeError('Provider requested deferred acquisition; resume later')
                time.sleep(int(delay))
                continue
            if response.status_code in (301, 302, 303, 307, 308, 401, 403):
                response.close()
                raise AuthorizationError('CEDA authorization/redirect failure; renew token if expired')
            if response.status_code not in (200, 206, 404):
                status = response.status_code
                response.close()
                raise RuntimeError(f'CEDA HTTP {status}')
            return response
        raise RuntimeError('Provider remains unavailable; byte fragments preserved')

    def listing(self, path):
        with self.get(path, listing=True) as response:
            if response.status_code == 404:
                return []
            result = response.json()
        # Discard all preview/content fields. Only provider file identities are retained.
        items = []
        for source in result['items']:
            item = {k: source[k] for k in ('path', 'name', 'type', 'size', 'md5') if k in source}
            safe_path(item['path'])
            if str(Path(item['path']).parent).replace('\\', '/') != path or Path(item['path']).name != item['name']:
                raise ValueError('Provider listing escaped its requested directory')
            items.append(item)
        return items

    def fetch(self, item, target):
        target = Path(target)
        size, md5 = item['size'], item['md5']
        safe_path(item['path'])
        if not isinstance(size, int) or size < 0 or not re.fullmatch('[0-9a-fA-F]{32}', md5):
            raise ValueError('Missing official file size or MD5')
        if target.exists():
            if target.stat().st_size != size or digest(target, 'md5') != md5.lower():
                raise ValueError('Existing file differs from its pinned provider identity; preserved')
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name+'.part')
        for attempt in range(12):
            offset = part.stat().st_size if part.exists() else 0
            if offset > size:
                raise ValueError('Partial file exceeds pinned size; preserved')
            if offset == size:
                break
            try:
                with self.get(item['path'], offset=offset) as response:
                    if 'html' in response.headers.get('Content-Type', '').lower() or response.status_code == 404:
                        raise ValueError('Provider did not return a dataset file')
                    if offset and (response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                        raise ValueError('Provider ignored resume range; fragment preserved')
                    with part.open('ab') as stream:
                        for block in response.iter_content(256*1024):
                            stream.write(block)
                if part.stat().st_size == size:
                    break
            except requests.RequestException as exc:
                if attempt == 11:
                    raise RuntimeError('CEDA body interrupted; resume preserved bytes: '+type(exc).__name__) from None
                time.sleep(min(5*2**attempt, 60))
        if not part.exists() or part.stat().st_size != size or digest(part, 'md5') != md5.lower():
            raise ValueError('Downloaded bytes failed pinned size/MD5; preserved')
        if target.exists():
            raise FileExistsError('Another task published this file; not overwritten')
        part.rename(target)


def make_plan(archive, metadata, root_entries):
    frame = read_badc_csv(metadata)
    ids = set(load_metadata(metadata).src_id)
    candidates = frame[frame.src_id.isin(ids)]
    active = candidates[(candidates.first_year <= 2024) & (candidates.last_year >= 2016)]
    files, missing = [], []
    for _, station in active.iterrows():
        sid = int(station.src_id)
        county, name = str(station.historic_county), str(station.station_file_name)
        if any('/' in x or x in ('', '.', '..') for x in (county, name)):
            raise ValueError('Invalid station directory components')
        directory = f'{BASE}/{county}/{sid:05d}_{name}/qc-version-1'
        found = []
        for entry in archive.listing(directory):
            match = re.search(r'_(\d{4})\.csv$', entry['name'])
            if entry.get('type') != 'file' or not match or int(match.group(1)) not in YEARS:
                continue
            year = int(match.group(1))
            found.append(year)
            files.append({**entry, 'src_id': sid, 'year': year, 'sealed': year >= 2023})
        missing.append({'src_id': sid, 'missing_years': sorted(YEARS-set(found))})
        print(f'[PLAN] station={sid}, available_years={len(found)}', flush=True)
    if len({x['path'] for x in files}) != len(files):
        raise ValueError('Duplicate planned file identity')
    return {'dataset': DATASET, 'radius_km': RADIUS_KM, 'candidate_stations': len(candidates),
            'stations_overlapping_requested_years': len(active),
            'metadata_sha256': digest(metadata), 'metadata_provider': root_entries,
            'missing_by_station': missing,
            'files': sorted(files, key=lambda x: (x['sealed'], x['src_id'], x['year']))}


def validate_plan(plan):
    if plan['dataset'] != DATASET or plan['radius_km'] != RADIUS_KM:
        raise ValueError('Pinned dataset or radius differs')
    paths = set()
    for item in plan['files']:
        safe_path(item['path'])
        match = re.search(r'_(\d{4})\.csv$', item['name'])
        if (not match or int(match.group(1)) != item['year'] or item['year'] not in YEARS
                or item['sealed'] != (item['year'] >= 2023)
                or '/qc-version-1/' not in item['path']
                or Path(item['path']).name != item['name'] or item['path'] in paths
                or not isinstance(item['size'], int) or item['size'] < 0
                or not re.fullmatch('[0-9a-fA-F]{32}', item['md5'])):
            raise ValueError('Invalid pinned file identity/year/sealed assignment')
        paths.add(item['path'])


def acquire(archive, development, sealed, audit, evaluations=None):
    development, sealed = checked_roots(development, sealed)
    audit = Path(audit)
    audit.mkdir(parents=True, exist_ok=True)
    metadata = development/META
    root_entries = [x for x in archive.listing(BASE) if x['name'] == META]
    if len(root_entries) != 1:
        raise ValueError('Official station metadata entry missing/ambiguous')
    archive.fetch(root_entries[0], metadata)
    plan_path = audit/'download_plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan['dataset'] != DATASET or plan['radius_km'] != RADIUS_KM or plan['metadata_sha256'] != digest(metadata):
            raise ValueError('Existing acquisition plan identity differs; no overwrite')
    else:
        plan = make_plan(archive, metadata, root_entries[0])
        validate_plan(plan)
        save_json(plan_path, plan)
    validate_plan(plan)
    manifest = {'status': 'DOWNLOADING', 'dataset': DATASET, 'plan_sha256': digest(plan_path),
                'candidate_stations': plan['candidate_stations'], 'planned_files': len(plan['files']),
                'planned_bytes': sum(x['size'] for x in plan['files']), 'files': []}
    manifest_path = audit/'download_manifest.json'
    save_json(manifest_path, manifest)
    screened = evaluations is None
    def receive(item):
        target = (sealed if item['sealed'] else development)/str(item['src_id'])/item['name']
        archive.fetch(item, target)
        return {'src_id': item['src_id'], 'year': item['year'], 'sealed': item['sealed'],
                'path': str(target), 'provider_path': item['path'], 'bytes': item['size'],
                'md5': item['md5'], 'sha256': digest(target)}
    try:
        for sealed_phase in (False, True):
            pending_items = iter(x for x in plan['files'] if x['sealed'] == sealed_phase)
            # Bounded submission: at most two requests, and no queued flood after an auth/network failure.
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {pool.submit(receive, item) for item in [next(pending_items, None), next(pending_items, None)] if item}
                try:
                    while futures:
                        completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for future in completed:
                            entry = future.result()
                            manifest['files'].append(entry)
                            manifest['files'].sort(key=lambda x: (x['sealed'], x['src_id'], x['year']))
                            save_json(manifest_path, manifest)
                            print(f"[VERIFIED] {len(manifest['files'])}/{manifest['planned_files']} station={entry['src_id']} year={entry['year']}", flush=True)
                            item = next(pending_items, None)
                            if item:
                                futures.add(pool.submit(receive, item))
                except Exception:
                    for future in futures:
                        future.cancel()
                    raise
            if not sealed_phase and not screened:
                run_screen(development, metadata, evaluations, audit)
                screened = True
        manifest['status'] = 'DOWNLOADED_VERIFIED'
    except Exception as exc:
        manifest.update(status='INTERRUPTED', error_type=type(exc).__name__)
        save_json(manifest_path, manifest)
        raise
    save_json(manifest_path, manifest)
    return manifest


def run_screen(development, metadata, evaluations, audit):
    path = Path(audit)/'screen.json'
    identity = {'metadata_sha256': digest(metadata),
                'screen_source_sha256': digest(ROOT/'scripts/screen_kelmarsh_upstream_observations.py'),
                'development_files': {str(p.relative_to(development)): digest(p) for p in sorted(Path(development).rglob('*.csv'))},
                'evaluation_manifests': {str(year): digest(Path(folder).parent/'completed.json') for year,folder in evaluations.items()}}
    # Verify registered prediction bytes as well as the manifests before reusing any result.
    for year, folder in evaluations.items():
        load_year(folder, ARMS[year], year)
    completion = Path(audit)/'screen_completed.json'
    if path.exists() or completion.exists():
        if not path.exists() or not completion.exists():
            raise ValueError('Incomplete screen publication; inspect before any repeated fitting')
        saved = json.loads(completion.read_text())
        if saved['identity'] != identity or saved['screen_sha256'] != digest(path):
            raise ValueError('Existing screen identity differs; no repeated fitting')
        print('[SCREEN] verified existing result reused; no fitting repeated', flush=True)
        return
    try:
        result = screen(development, metadata, evaluations)
    except (ValueError, FileNotFoundError) as exc:
        result = {'status': 'INVALID_INPUT', 'passed': None, 'reason': str(exc)}
    save_json(path, result)
    save_json(completion, {'identity': identity, 'screen_sha256': digest(path)})
    print('[SCREEN] '+result['status']+', passed='+str(result['passed']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token-file', type=Path, default=Path.home()/'.ceda_token')
    parser.add_argument('--development-root', type=Path, required=True)
    parser.add_argument('--sealed-root', type=Path, required=True)
    parser.add_argument('--audit-root', type=Path, required=True)
    parser.add_argument('--eval-2021', type=Path)
    parser.add_argument('--eval-2022', type=Path)
    args = parser.parse_args()
    if bool(args.eval_2021) != bool(args.eval_2022):
        parser.error('Both development evaluation queues are required together')
    evaluations = {2021: args.eval_2021, 2022: args.eval_2022} if args.eval_2021 else None
    import fcntl
    args.audit_root.mkdir(parents=True, exist_ok=True)
    with (args.audit_root/'.acquire.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquire(Archive(args.token_file), args.development_root, args.sealed_root, args.audit_root, evaluations)


if __name__ == '__main__':
    main()
