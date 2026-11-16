"""X1 baselines on hash-locked cohorts. Only reviewed, ready arms run."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def checked(path, sha):
    if digest(path) != sha:
        raise ValueError('Hash mismatch: '+str(path))
    return Path(path)


def persistence(truth, anchor):
    if truth.ndim != 2 or truth.shape[1] != 4 or anchor.shape != truth.shape[:1]:
        raise ValueError('Invalid aligned cohort')
    if not np.isfinite(truth).all() or not np.isfinite(anchor).all():
        raise ValueError('Nonfinite cohort, no repair permitted')
    return np.broadcast_to(anchor[:, None], truth.shape).copy()


def paths(base, item):
    if item['family'] == 'G1':
        parent = base/'G1/g1c_27312d7_r2/formal'
        pack = parent.parent/'prepared'/item['farm']/'test.npz'
        receipt = ROOT/'output/g1/results_78/formal/completed.json'
        names = [f"{item['farm']}/N/{s}/prediction.npy" for s in (2024, 2025, 2026)]
    else:
        parent = base/'G2/g2c_2eeb23f_r1/formal'
        pack = base/'G2/g2b84_bc7730e_r1/sealed_prep'/f"{item['year']}.npz"
        receipt = ROOT/'output/g2b/sealed_eval/server/completed.json'
        names = [f"{item['year']}_N_{s}.npy" for s in (2024, 2025, 2026)]
    frozen = read(receipt)
    if read(parent/'completed.json') != frozen or (parent/'failed.json').exists():
        raise ValueError('Parent receipt mismatch or failed run')
    checked(pack, item['pack_sha256'])
    preds = [checked(parent/n, frozen['predictions'][n]) for n in names]
    return pack, preds


def run(args):
    from scripts.train_g1 import evaluate_arrays
    checked(ROOT/'configs/x1_protocol.json', args.protocol_sha256)
    protocol = read(ROOT/'configs/x1_protocol.json')
    if args.arm not in protocol['authorized_ready_arms']:
        raise ValueError('Arm not ready or not reviewed')
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'launch.json', dict(arm=args.arm, protocol_sha256=args.protocol_sha256,
        script_sha256=digest(__file__), started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        cpu_threads=2, retry=False, new_fits=0))
    try:
        reports, hashes = {}, {}
        for name, item in protocol['datasets'].items():
            path, pred_paths = paths(args.logs, item)
            with np.load(path, allow_pickle=False) as z:
                truth, anchor, issue = (z[k] for k in ('truth', 'anchor', 'issue'))
                if len(truth) != item['count']:
                    raise ValueError('Cohort size mismatch')
                years = (z['times'].astype('datetime64[ns]')-np.timedelta64(1, 'ns')).astype('datetime64[Y]').astype(int)+1970
                if not np.all(years == item['year']):
                    raise ValueError('Target year mismatch')
            baseline = persistence(truth, anchor)
            preds = [np.load(p, allow_pickle=False) for p in pred_paths]
            if any(p.shape != truth.shape or not np.isfinite(p).all() for p in preds):
                raise ValueError('Invalid saved N predictions')
            report = evaluate_arrays({'N': preds, args.arm: [baseline]}, truth,
                issue.astype('datetime64[ns]'), None, protocol['statistics'],
                comparison_pairs={'N_vs_'+args.arm: ('N', args.arm)})
            for rows in report['metrics'].values():
                for row in rows:
                    row['one_minus_nrmse'] = 1-row['rmse']/item['rated_kw']
            np.save(args.output/(name+'_'+args.arm+'.npy'), baseline)
            hashes[name] = digest(args.output/(name+'_'+args.arm+'.npy'))
            reports[name] = report
            print('COMPLETED', name, flush=True)
        passed = all(r['primary']['N_vs_'+args.arm]['gain'] > 0 and
            r['primary']['N_vs_'+args.arm]['ci95_low'] > 0 for r in reports.values())
        save(args.output/'results.json', dict(arm=args.arm, datasets=reports, all_four_passed=passed))
        save(args.output/'completed.json', dict(status='COMPLETED', new_fits=0,
            predictions=hashes, results_sha256=digest(args.output/'results.json')))
    except Exception as exc:
        save(args.output/'failed.json', dict(type=type(exc).__name__, message=str(exc), retry=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--logs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--protocol-sha256', required=True)
    p.add_argument('--arm', choices=['A0'], default='A0')
    run(p.parse_args())
