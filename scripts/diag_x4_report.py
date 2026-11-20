"""X4 smoke-report diagnosis (READ-ONLY): which development comparison made the report gate raise
'Technical statistical failure (degenerate CI)' in x4_efbd0af/smoke_report, and why.

For A (C_T1 vs L_1X), B1 (FT-B vs LUG on V2), B2 (ensemble vs LUG, ensemble vs LUG + N_v1) and every farm it reports:
windows, issue-time clusters, non-finite predictions, whether candidate and reference predictions are identical,
the largest prediction difference, the V1 weights w / control w, the per-farm CI width and the pooled status.
It reuses the registered smoke arrays only (hash-checked through each stage's registry) and the run_x1/x2/x3/x4 code
of the worktree it is run from.  Writes ONLY <out>/diagnosis.json into a NEW directory.  Author: Claude (2026-11-20).

  cd <clean worktree of efbd0af> && PYTHONPATH=<worktree> python <this file> --logs <logs> \\
      --base <logs>/X4/x4_efbd0af --out <new directory>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def compare(x3, st, cand, ref, truth, issue):
    cand, ref, truth = (np.asarray(v, float) for v in (cand, ref, truth))
    d = dict(windows=int(len(truth)), issue_clusters=int(len(np.unique(np.asarray(issue)))),
             nonfinite_candidate=int((~np.isfinite(cand)).sum()), nonfinite_reference=int((~np.isfinite(ref)).sum()),
             identical=bool(np.array_equal(cand, ref)))
    if np.isfinite(cand).all() and np.isfinite(ref).all():
        d['max_abs_prediction_diff'] = float(np.abs(cand - ref).max())
        s = x3.paired(st, cand, ref, truth, issue)
        d.update(gain=s['gain'], ci95_low=s['ci95_low'], ci95_high=s['ci95_high'],
                 ci_width=s['ci95_high'] - s['ci95_low'], reference_mean_loss=s['reference_mean_loss'])
        return s, d
    return None, d


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--logs', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path.cwd()
    sys.path.insert(0, str(root))
    from scripts import run_x1 as x1, run_x2 as x2, run_x3 as x3, run_x4 as x4
    protocol, x1p = x1.read(x4.PROTOCOL), x1.read(root / 'configs/x1_protocol.json')
    run = SimpleNamespace(base=args.base, mode='report', smoke=True)
    a, b = x4.upstream(run, 'a'), x4.upstream(run, 'b')
    refs, x3r = x3.Refs(args.logs, protocol), x4.X3Run(args.logs, protocol)
    st, z = protocol['statistics_smoke'], float(protocol['family_z'])
    res = dict(read_only=True, base=str(args.base), statistics=st, A={}, B1={}, B2_vs_LUG={}, B2_vs_control={},
               weights={}, pooled={})
    stats = {q: {} for q in ('A', 'B1', 'B2_vs_LUG', 'B2_vs_control')}
    for T in x4.FARMS:
        keys = dict(x1.load_pack(x1.checked(a['root'] / f'{T}/v1_keys.npz', a['registry'][f'{T}/v1_keys.npz'])))
        full = refs.keys(T, 'V1')
        c = x3.load_registered(a, f'{T}/pred/C_T1_V1.npy')[:, 1]
        r = x4.aligned(x3r.array('q4', f'{T}_L_1X.npy'), full, keys)
        stats['A'][T], res['A'][T] = compare(x3, st, c, r, keys['truth'], keys['issue'])
    for f in x4.FARMS:
        keys = {s: refs.keys(f, s) for s in ('V1', 'V2')}
        sub = {s: dict(x1.load_pack(x1.checked(b['root'] / f'{f}/{s}_keys.npz', b['registry'][f'{f}/{s}_keys.npz'])))
               for s in ('V1', 'V2')}
        ftb = {s: x3.load_registered(b, f'{f}/FTB_{s}.npy') for s in ('V1', 'V2')}
        lug = {s: x4.aligned(refs.lug(f, s), keys[s], sub[s]) for s in ('V1', 'V2')}
        nv1 = {s: x4.aligned(refs.nv1(f, s), keys[s], sub[s]) for s in ('V1', 'V2')}
        y1, y2, i2 = sub['V1']['truth'], sub['V2']['truth'], sub['V2']['issue']
        stats['B1'][f], res['B1'][f] = compare(x3, st, ftb['V2'][:, 1], lug['V2'], y2, i2)
        w = x3.ensemble_search(lug['V1'], ftb['V1'][:, 1], y1, protocol['b']['w_step'])
        cw = x3.ensemble_search(lug['V1'], nv1['V1'], y1, protocol['b']['w_step'])
        res['weights'][f] = dict(w=w[0], w_v1_mae=w[1], control_w=cw[0], control_v1_mae=cw[1],
                                 ftb_source=x1.read(b['root'] / 'b_meta.json')['farms'][f]['selected'])
        ens = w[0] * lug['V2'] + (1 - w[0]) * ftb['V2'][:, 1]
        ctrl = cw[0] * lug['V2'] + (1 - cw[0]) * nv1['V2']
        stats['B2_vs_LUG'][f], res['B2_vs_LUG'][f] = compare(x3, st, ens, lug['V2'], y2, i2)
        stats['B2_vs_control'][f], res['B2_vs_control'][f] = compare(x3, st, ens, ctrl, y2, i2)
    for q, s in stats.items():
        res['pooled'][q] = (x2.pooled_effect(s, z) if all(v is not None for v in s.values())
                            else dict(status='NOT_COMPUTED', reason='non-finite predictions'))
    res['degenerate'] = [dict(question=q, farm=f, **d) for q in ('A', 'B1', 'B2_vs_LUG', 'B2_vs_control')
                         for f, d in res[q].items() if d['identical'] or not d.get('ci_width', 0) > 0]
    (args.out / 'diagnosis.json').write_text(json.dumps(res, indent=2, allow_nan=False, default=float) + '\n',
                                             encoding='utf-8')
    print('[X4-diag] degenerate:', json.dumps(res['degenerate'], default=float)[:2000], flush=True)


if __name__ == '__main__':
    main()
