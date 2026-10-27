"""Download only the authorized 2023/2024 SCADA archives; NEVER open their ZIP contents.

Uses the existing resumable exact-range downloader with official MD5 validation.
No training/preprocessing entry point is called. One lock protects the whole batch.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.acquire_hourly_wind_datasets import checked_download, digest, save_json

FILES = {2023: 'Kelmarsh_SCADA_2023_5961.zip', 2024: 'Kelmarsh_SCADA_2024_5962.zip'}
OFFICIAL = {2023: (716099870, '685d9589347d1866f6f2cfe45b9af4f4'),
            2024: (704598245, 'e9e3318f5b393d8596ff1801f8da0414')}


def acquire(record, directory, downloader=checked_download):
    directory = Path(directory).resolve()
    if 'sealed' not in [p.lower() for p in directory.parts]:
        raise ValueError('An explicitly separated sealed directory is required')
    if int(record['id']) != 16807551:
        raise ValueError('Unexpected Kelmarsh provider record')
    entries = {}
    for year, key in FILES.items():
        item = next(f for f in record['files'] if f['key'] == key)
        size, md5 = OFFICIAL[year]
        if (item['size'], item['checksum']) != (size, 'md5:'+md5):
            raise ValueError('Provider identity differs from the reviewed download plan')
        if not item['links']['self'].startswith('https://zenodo.org/api/records/16807551/files/'):
            raise ValueError('Unexpected provider download endpoint')
        entries[year] = item
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {'record_id': 16807551, 'doi': record.get('doi'), 'status': 'DOWNLOADING',
                'provider_record_sha256': hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
                'content_policy': 'raw sealed bytes only: size, provider MD5 and SHA256; no extraction or parsing',
                'files': []}
    save_json(directory/'download_manifest.json', manifest)
    for year, key in FILES.items():
        item = entries[year]
        size, md5 = OFFICIAL[year]
        target = directory/key
        try:
            downloader(item['links']['self'], target, size, md5, workers=4)
            if target.stat().st_size != size or digest(target, 'md5') != md5:
                raise ValueError('Sealed archive failed official byte integrity')
        except Exception as exc:
            manifest.update(status='INTERRUPTED', failed_year=year, error_type=type(exc).__name__)
            save_json(directory/'download_manifest.json', manifest)
            raise
        manifest['files'].append({'year': year, 'file': key, 'url': item['links']['self'],
                                  'bytes': size, 'md5': md5, 'sha256': digest(target),
                                  'verified_utc': datetime.now(timezone.utc).isoformat()})
        save_json(directory/'download_manifest.json', manifest)
    manifest['status'] = 'SEALED_VERIFIED'
    save_json(directory/'download_manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True, help='Previously registered Zenodo record JSON')
    parser.add_argument('--sealed-root', type=Path, required=True)
    args = parser.parse_args()
    # Linux deployment only. Import here so torch-free Windows tests can inspect acquire().
    import fcntl
    args.sealed_root.mkdir(parents=True, exist_ok=True)
    with (args.sealed_root/'.acquire.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = acquire(json.loads(args.record.read_text()), args.sealed_root)
        print(json.dumps({'status': result['status'], 'years': list(FILES)}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
