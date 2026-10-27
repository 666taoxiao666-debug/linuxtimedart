"""Resume the unchanged sealed Kelmarsh byte downloader after transport failures only."""
import argparse
import json
from pathlib import Path
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.acquire_kelmarsh_sealed_years import acquire


def resume_transport(record, root, *, operation=acquire, sleep=time.sleep, attempts=24):
    for attempt in range(attempts):
        try:
            return operation(record, root)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
            if attempt + 1 == attempts:
                raise
            # Original acquire() records the interruption and keeps all exact-range fragments.
            print(f'[TRANSPORT_RESUME] {type(exc).__name__}; attempt {attempt+1}/{attempts}; '
                  'retain fragments and wait 300 seconds', flush=True)
            sleep(300)
    raise ValueError('Positive attempt budget required')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--sealed-root', type=Path, required=True)
    args = p.parse_args()
    if 'sealed' not in [x.lower() for x in args.sealed_root.resolve().parts]:
        p.error('An explicitly sealed root is required')
    import fcntl
    args.sealed_root.mkdir(parents=True, exist_ok=True)
    with (args.sealed_root/'.acquire.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = resume_transport(json.loads(args.record.read_text()), args.sealed_root)
        print('[COMPLETE] '+result['status'], flush=True)


if __name__ == '__main__':
    main()
