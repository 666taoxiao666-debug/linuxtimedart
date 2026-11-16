"""Read-only identity/context preflight; no fit, prediction or test loss."""
import argparse
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import run_x1 as x


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--logs', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    args = p.parse_args()
    receipt = x.verify_chronos(args.receipt, args.manifest)
    print('IDENTITY_OK', receipt['revision'], receipt['manifest_sha256'], flush=True)
    protocol = x.read(ROOT/'configs/x1_protocol.json')
    for farm, item in {v['farm']:v for v in protocol['datasets'].values()}.items():
        stats, hourly = x.chronos_history(args.logs, item)
        if len(stats['mean']) != 21 or len(stats['scale']) != 21:
            raise ValueError('Unexpected Chronos scale layout')
        train, val = x.development(args.logs, item)
        packs = [train, val]
        for other in protocol['datasets'].values():
            if other['farm'] == farm:
                packs.append(x.load_pack(x.paths(args.logs, other)[0]))
        grid = x.history_grid(packs)
        hours = hourly['hours'].astype('datetime64[ns]')
        for pack in packs:
            times = pack['issue'].astype('datetime64[ns]')[:, None]-np.arange(23,-1,-1)*np.timedelta64(1,'h')
            pos = np.searchsorted(hours, times)
            if (pos >= len(hours)).any() or not np.array_equal(hours[pos], times):
                raise ValueError('Missing historical GFS hours: '+farm)
        sample = x.longer_context(val, grid, np.arange(min(32,len(val['issue']))))
        print('CONTEXT_OK', farm, 'windows', [len(p['issue']) for p in packs],
              'history_rows', len(grid['grid_hour']), 'sample_older_missing_fraction',
              float(np.isnan(sample[:,:-24,-1]).mean()), flush=True)


if __name__ == '__main__':
    main()
