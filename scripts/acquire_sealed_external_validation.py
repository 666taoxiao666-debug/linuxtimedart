"""Queue complete official external datasets behind verified MIDAS/Kelmarsh downloads.

Only provider JSON and raw file bytes are read. No archive extraction, CSV/parquet
parsing, model import, training, or choice of external evaluation protocol occurs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import unquote, urlparse
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.acquire_hourly_wind_datasets import checked_download, digest, save_json
from scripts.acquire_kelmarsh_sealed_years import FILES, OFFICIAL
from scripts.acquire_midas_upstream_observations import META, validate_plan

FIGSHARE_ID, FIGSHARE_VERSION = 24798654, 2
PENMANSHIEL_ID, PENMANSHIEL_PARENT = 16807304, 5946807
POLICY = 'sealed raw bytes only; no extraction, data values, statistics, plots or model use'


def filename(name):
    if (not isinstance(name, str) or name in ('', '.', '..') or '/' in name or '\\' in name
            or any(ord(c) < 32 for c in name) or ':' in name):
        raise ValueError('Unsafe provider filename')
    return name


def contained(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if target == root or root not in target.parents:
        raise ValueError('Artifact path escaped its registered root')
    return target


def sealed_root(path):
    path = Path(path).resolve()
    if 'sealed' not in [x.lower() for x in path.parts]:
        raise ValueError('External data must be stored in an explicitly sealed directory')
    return path


def validate_inventory(plan):
    if plan.get('policy') != POLICY or len(plan.get('datasets', [])) != 2:
        raise ValueError('Unexpected acquisition scope')
    expected = [('sdwpf', FIGSHARE_ID), ('penmanshiel', PENMANSHIEL_ID)]
    paths = set()
    for dataset, (kind, record_id) in zip(plan['datasets'], expected):
        if dataset['name'] != kind or dataset['record_id'] != record_id or not dataset['files']:
            raise ValueError('Dataset identity or complete file list missing')
        if dataset['license']['id'] != 'cc-by-4.0':
            raise ValueError('Unreviewed dataset license')
        ids = set()
        for item in dataset['files']:
            name = filename(item['name'])
            file_id = str(item['id'])
            if not re.fullmatch(r'[0-9a-fA-F-]+', file_id) or file_id in ids:
                raise ValueError('Duplicate or invalid provider file ID')
            ids.add(file_id)
            relative = f'{kind}_{record_id}/{file_id}/{name}'
            if item['relative_path'] != relative or relative in paths:
                raise ValueError('File identity/path collision')
            paths.add(relative)
            if (not isinstance(item['bytes'], int) or item['bytes'] < 0
                    or not re.fullmatch('[0-9a-f]{32}', item['md5'])):
                raise ValueError('Missing official byte size or MD5')
            url = urlparse(item['url'])
            wanted = (f'/files/{file_id}' if kind == 'sdwpf'
                      else f'/api/records/{record_id}/files/{name}/content')
            if (url.scheme != 'https' or url.query or url.fragment or url.username
                    or url.netloc != ('ndownloader.figshare.com' if kind == 'sdwpf' else 'zenodo.org')
                    or unquote(url.path) != wanted):
                raise ValueError('Unrecognized official download endpoint')


def build_inventory(figshare_path, penmanshiel_path):
    fig = json.loads(Path(figshare_path).read_text(encoding='utf-8-sig'))
    pen = json.loads(Path(penmanshiel_path).read_text(encoding='utf-8-sig'))
    if (fig['id'], fig['version']) != (FIGSHARE_ID, FIGSHARE_VERSION):
        raise ValueError('Figshare version differs from reviewed official v2')
    if (int(pen['id']), int(pen['conceptrecid'])) != (PENMANSHIEL_ID, PENMANSHIEL_PARENT):
        raise ValueError('Penmanshiel record is not the reviewed latest member of 5946808')
    if (fig['license']['url'].rstrip('/') != 'https://creativecommons.org/licenses/by/4.0'
            or pen['metadata']['license']['id'] != 'cc-by-4.0'):
        raise ValueError('Unreviewed provider license')
    datasets = []
    for kind, record, source in [('sdwpf', fig, figshare_path), ('penmanshiel', pen, penmanshiel_path)]:
        files = []
        for item in record['files']:
            if kind == 'sdwpf':
                if item.get('is_link_only'):
                    raise ValueError('Link-only Figshare file requires separate review')
                name, md5, url = item['name'], item['computed_md5'], item['download_url']
                if item.get('supplied_md5') and item['supplied_md5'].lower() != md5.lower():
                    raise ValueError('Figshare official checksum declarations disagree')
            else:
                name, url = item['key'], item['links']['self']
                algorithm, md5 = item['checksum'].split(':', 1)
                if algorithm != 'md5':
                    raise ValueError('Unreviewed provider checksum algorithm')
            files.append({'id': str(item['id']), 'name': name, 'bytes': item['size'],
                          'md5': md5.lower(), 'url': url,
                          'relative_path': f"{kind}_{record['id']}/{item['id']}/{name}"})
        datasets.append({'name': kind, 'record_id': int(record['id']),
                         'doi': record['doi'], 'license': {'id': 'cc-by-4.0',
                         'url': 'https://creativecommons.org/licenses/by/4.0/'},
                         'provider_json_sha256': digest(source),
                         'metadata_url': ('https://api.figshare.com/v2/articles/24798654/versions/2'
                                          if kind == 'sdwpf' else 'https://zenodo.org/api/records/16807304'),
                         'resolved_from': ('24798654.v2' if kind == 'sdwpf' else '5946808/versions/latest'),
                         'files': sorted(files, key=lambda x: x['id'])})
    plan = {'policy': POLICY, 'datasets': datasets}
    validate_inventory(plan)
    return plan


def statuses(midas_audit, kelmarsh):
    paths = [Path(midas_audit)/'download_manifest.json', Path(kelmarsh)/'download_manifest.json']
    values = [json.loads(p.read_text()) if p.exists() else {} for p in paths]
    ready = values[0].get('status') == 'DOWNLOADED_VERIFIED' and values[1].get('status') == 'SEALED_VERIFIED'
    return ready, values, [x.get('status', 'NOT_YET_REGISTERED') for x in values]


def check_bytes(path, size, md5, sha256):
    if (not re.fullmatch('[0-9a-f]{64}', sha256) or not path.is_file()
            or path.stat().st_size != size or digest(path, 'md5') != md5 or digest(path) != sha256):
        raise ValueError('Registered raw bytes do not match size/MD5/SHA256: '+str(path))


def verify_prerequisites(midas_audit, midas_development, midas_sealed, kelmarsh):
    ready, manifests, _ = statuses(midas_audit, kelmarsh)
    if not ready:
        raise ValueError('Predecessor downloads have not both completed')
    midas, kel = manifests
    plan_path = Path(midas_audit)/'download_plan.json'
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    if not plan['files'] or plan['metadata_sha256'] != digest(Path(midas_development)/META):
        raise ValueError('MIDAS station metadata/plan missing or changed')
    if midas['plan_sha256'] != digest(plan_path) or len(midas['files']) != len(plan['files']):
        raise ValueError('MIDAS complete manifest/plan identity mismatch')
    expected = {x['path']: x for x in plan['files']}
    if (len(expected) != len(plan['files'])
            or len({x['provider_path'] for x in midas['files']}) != len(expected)
            or midas['planned_files'] != len(expected)
            or midas['planned_bytes'] != sum(x['size'] for x in expected.values())):
        raise ValueError('MIDAS complete-file accounting mismatch')
    for item in midas['files']:
        source = expected[item['provider_path']]
        if (source['year'] not in range(2016, 2025) or source['sealed'] != (source['year'] >= 2023)
                or any(item[k] != source[k] for k in ['src_id', 'year', 'sealed', 'md5'])
                or item['bytes'] != source['size']):
            raise ValueError('MIDAS year/sealed/byte identity mismatch')
        root = midas_sealed if source['sealed'] else midas_development
        path = contained(root, f"{source['src_id']}/{filename(source['name'])}")
        if Path(item['path']).resolve() != path:
            raise ValueError('MIDAS registered path mismatch')
        check_bytes(path, source['size'], source['md5'], item['sha256'])
    if kel['record_id'] != 16807551 or len(kel['files']) != 2 or {x['year'] for x in kel['files']} != set(FILES):
        raise ValueError('Kelmarsh predecessor identity/inventory mismatch')
    for item in kel['files']:
        size, md5 = OFFICIAL[item['year']]
        if item['file'] != FILES[item['year']] or (item['bytes'], item['md5']) != (size, md5):
            raise ValueError('Kelmarsh official predecessor identity mismatch')
        check_bytes(contained(kelmarsh, item['file']), size, md5, item['sha256'])
    return {'midas_manifest_sha256': digest(Path(midas_audit)/'download_manifest.json'),
            'midas_plan_sha256': digest(plan_path),
            'kelmarsh_manifest_sha256': digest(Path(kelmarsh)/'download_manifest.json')}


def acquire(plan, root, audit, predecessor_evidence, downloader=checked_download, sleep=time.sleep):
    validate_inventory(plan)
    root = sealed_root(root)
    audit = Path(audit)
    plan_sha = digest(audit/'inventory.json')
    if json.loads((audit/'inventory.json').read_text()) != plan:
        raise ValueError('In-memory inventory differs from registered inventory')
    path = audit/'download_manifest.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous.get('inventory_sha256', plan_sha) != plan_sha:
        raise ValueError('Acquisition inventory changed; no overwrite')
    previous_files = {x['relative_path']: x for x in previous.get('files', [])}
    if previous.get('status') == 'SEALED_VERIFIED':
        expected = {x['relative_path']: x for d in plan['datasets'] for x in d['files']}
        if set(previous_files) != set(expected) or len(previous['files']) != len(expected):
            raise ValueError('Completed external inventory is incomplete or duplicated')
        for relative, item in expected.items():
            old = previous_files[relative]
            if any(old[k] != item[k] for k in item):
                raise ValueError('Completed external file registration changed')
            check_bytes(contained(root, relative), item['bytes'], item['md5'], old['sha256'])
        return previous
    manifest = {'status': 'DOWNLOADING', 'policy': POLICY, 'inventory_sha256': plan_sha,
                'predecessors': predecessor_evidence, 'files': []}
    save_json(path, manifest)
    try:
        for dataset in plan['datasets']:
            for item in dataset['files']:
                target = contained(root, item['relative_path'])
                for attempt in range(12):
                    try:
                        downloader(item['url'], target, item['bytes'], item['md5'], workers=2)
                        break
                    except requests.HTTPError:
                        # Transport retries must not hammer a provider's denial response.
                        raise
                    except requests.RequestException as exc:
                        if attempt == 11:
                            raise
                        # Existing byte-range fragments remain intact. Never retry identity errors.
                        print('[RETRY] '+type(exc).__name__+'; retaining bytes', flush=True)
                        sleep(min(60*2**attempt, 300))
                sha = digest(target)
                check_bytes(target, item['bytes'], item['md5'], sha)
                old = previous_files.get(item['relative_path'])
                if old and old['sha256'] != sha:
                    raise ValueError('Previously verified external bytes changed')
                manifest['files'].append({**item, 'dataset': dataset['name'],
                                          'license': dataset['license'], 'sha256': sha})
                save_json(path, manifest)
                print(f"[VERIFIED] {dataset['name']} {item['id']} {item['bytes']} bytes", flush=True)
        manifest['status'] = 'SEALED_VERIFIED'
    except Exception as exc:
        manifest.update(status='INTERRUPTED', error_type=type(exc).__name__)
        save_json(path, manifest)
        raise
    save_json(path, manifest)
    return manifest


def run_queue(args, *, sleep=time.sleep, downloader=checked_download):
    import fcntl
    root = sealed_root(args.sealed_root)
    audit = Path(args.audit_root)
    audit.mkdir(parents=True, exist_ok=True)
    with (audit/'.queue.lock').open('a') as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = audit/'inventory.json'
        plan = json.loads(plan_path.read_text())
        validate_inventory(plan)
        while True:
            ready, _, state = statuses(args.midas_audit, args.kelmarsh_sealed)
            save_json(audit/'queue_status.json', {'status': 'VERIFYING_PREDECESSORS' if ready else 'WAITING_FOR_PREDECESSORS',
                      'predecessor_status': state, 'updated_utc': datetime.now(timezone.utc).isoformat()})
            if ready:
                with ExitStack() as locks:
                    try:
                        for directory in [args.midas_audit, args.kelmarsh_sealed]:
                            lock = locks.enter_context((Path(directory)/'.acquire.lock').open('a'))
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        ready = False
                    if ready:
                        evidence = verify_prerequisites(args.midas_audit, args.midas_development,
                                                        args.midas_sealed, args.kelmarsh_sealed)
                if ready:
                    break
            sleep(args.poll_seconds)
        save_json(audit/'queue_status.json', {'status': 'DOWNLOADING_EXTERNAL', 'predecessors': evidence})
        result = acquire(plan, root, audit, evidence, downloader)
        save_json(audit/'queue_status.json', {'status': result['status'], 'files': len(result['files'])})
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-root', type=Path, default=Path('outputs/audits/external_validation_download'))
    p.add_argument('--sealed-root', type=Path, default=Path('datasets/external_validation/sealed'))
    p.add_argument('--figshare-record', type=Path)
    p.add_argument('--penmanshiel-record', type=Path)
    p.add_argument('--prepare-only', action='store_true', help='Register provider JSON inventory, no data transfer')
    p.add_argument('--midas-audit', type=Path, default=Path('outputs/audits/upstream_screen/midas_v202607'))
    p.add_argument('--midas-development', type=Path, default=Path('datasets/upstream_observations/midas_open_v202607/development'))
    p.add_argument('--midas-sealed', type=Path, default=Path('datasets/upstream_observations/sealed/midas_open_v202607'))
    p.add_argument('--kelmarsh-sealed', type=Path, default=Path('datasets/upstream_observations/sealed/kelmarsh'))
    p.add_argument('--poll-seconds', type=int, default=60)
    args = p.parse_args()
    if args.poll_seconds < 30:
        p.error('Predecessor poll interval must be at least 30 seconds')
    if args.prepare_only:
        if not args.figshare_record or not args.penmanshiel_record:
            p.error('Both provider metadata records are required')
        plan = build_inventory(args.figshare_record, args.penmanshiel_record)
        target = args.audit_root/'inventory.json'
        if target.exists() and json.loads(target.read_text()) != plan:
            raise ValueError('Existing inventory differs; no overwrite')
        save_json(target, plan)
        print(json.dumps({'inventory_sha256': digest(target), 'files': sum(len(d['files']) for d in plan['datasets']),
                          'bytes': sum(x['bytes'] for d in plan['datasets'] for x in d['files'])}))
    else:
        try:
            run_queue(args)
        except BlockingIOError:
            # A second queue must not overwrite the live queue's status.
            raise
        except Exception as exc:
            save_json(args.audit_root/'queue_status.json', {'status': 'FAILED', 'error_type': type(exc).__name__})
            raise


if __name__ == '__main__':
    main()
