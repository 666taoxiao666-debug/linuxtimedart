"""Read-only verification of the completed, frozen M6 O2 run; no inference."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path('/home/XT/nuist/pythoncode/TimeDARTFirst_M6_ba7bca7')
RUN = Path('/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/LLM_L1/l1_ba7bca7_o2')
sys.path.insert(0, str(ROOT))
from scripts.llm_encoder_search_l1 import calc_metrics, decide_l1
from utils import kelmarsh_backbone_information as kbi


def read(name):
    return json.loads((RUN / name).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def same(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            same(actual[key], expected[key])
    elif isinstance(expected, (float, int)):
        assert np.isclose(actual, expected, rtol=1e-10, atol=1e-10), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


s = read('l1_summary.json')
freeze = read('formal_selection_freeze.json')
start = read('test_evaluation_started.json')
completed = read('audit/completed.json')
launch = read('audit/launch.json')
assert not (RUN / 'audit/failed.json').exists()
assert completed['decision'] == s['decision']
assert freeze['total_fits'] == 24
assert len(list((RUN / 'training').rglob('*.json'))) == 24
assert len(s['artifact_sha256']) == 50 and len(freeze['checkpoints']) == 24
for name, digest in s['artifact_sha256'].items():
    assert sha(RUN / name) == digest, name
for name, digest in s['input_hashes'].items():
    assert sha(name) == digest, name
paths = {'protocol_sha256': 'configs/l1_llm_encoder_protocol.json',
         'script_sha256': 'scripts/llm_encoder_search_l1.py',
         'program_encoder_sha256': 'models/ProgramEncoder.py'}
for key, name in paths.items():
    assert sha(ROOT / name) == s['code_hashes'][key], name
same(s['code_hashes'], launch['code_hashes'])
stamp = lambda value: dt.datetime.fromisoformat(value).timestamp()
frozen_at = stamp(freeze['frozen_at_utc'])
started_at = stamp(start['started_at_utc'])
assert stamp(read('search_selection_freeze.json')['frozen_at_utc']) < frozen_at
assert frozen_at <= started_at < stamp(completed['completed_at_utc'])
for name, digest in freeze['checkpoints'].items():
    assert s['artifact_sha256'][name] == digest
    assert (RUN / name).stat().st_mtime <= frozen_at
for name in s['artifact_sha256']:
    if name.startswith('predictions/'):
        assert started_at <= (RUN / name).stat().st_mtime
for arm in ('R_star', 'L_star'):
    assert freeze['sources'][arm] == s['search'][arm]['source']

recomputed = {}
for farm in ('Kelmarsh', 'Penmanshiel'):
    with np.load(RUN / f'predictions/{farm}/cohort.npz', allow_pickle=False) as c:
        truth, issue = c['truth'], c['issue']
    arms, losses = {}, {}
    for arm, saved in s['formal'][farm]['arms'].items():
        metrics, preds = [], []
        for row in saved['seeds']:
            seed = row['seed']
            record = read(f'training/{farm}/{arm}_s{seed}.json')
            assert record['checkpoint_sha256'] == freeze['checkpoints'][record['checkpoint']]
            assert record['best_epoch'] == row['best_epoch']
            same(record['validation_mae'], row['val_mae'])
            with np.load(RUN / f'predictions/{farm}/{arm}_s{seed}.npz', allow_pickle=False) as p:
                prediction = p['prediction']
            m = calc_metrics(prediction, truth)
            same(m, {k: row[k] for k in ('mae', 'rmse', 'r2')})
            metrics.append(m)
            preds.append(prediction)
        means = {'mean_' + k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
        same(means, {k: saved[k] for k in means})
        arms[arm] = means
        errors = np.stack(preds).astype(np.float64) - truth.astype(np.float64)
        losses[arm] = np.abs(errors).mean(0), np.square(errors).mean(0)
    comparisons = {}
    for ref in ('H', 'P', 'R_star'):
        kw = dict(block_length=8, replicates=5000, seed=2024)
        comparisons['L_star_vs_' + ref] = {
            'mae': kbi.paired_loss_statistics(losses['L_star'][0], losses[ref][0], issue, **kw),
            'rmse': kbi.rmse_difference_bootstrap(losses['L_star'][1], losses[ref][1], issue, **kw)}
    same(comparisons, s['formal'][farm]['comparisons'])
    recomputed[farm] = {'arms': arms, 'comparisons': comparisons}
decision = decide_l1(recomputed)
assert decision == s['decision']
for farm in recomputed:
    same(recomputed[farm]['relative_gains_pct'], s['formal'][farm]['relative_gains_pct'])
metadata = {str(p.relative_to(RUN)): sha(p) for p in sorted(RUN.rglob('*.json'))}
print(json.dumps({'verified_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'status': 'PASS', 'run': str(RUN), 'decision': decision,
    'artifact_hashes_verified': len(s['artifact_sha256']),
    'input_hashes_verified': len(s['input_hashes']), 'code_hashes_verified': 3,
    'formal_fits': 24, 'seed_metric_sets_recomputed': 24, 'comparison_sets_recomputed': 6,
    'checkpoint_freeze_before_evaluation': True,
    'formal': recomputed, 'metadata_sha256': metadata,
    'statistics_utility_sha256': sha(ROOT / 'utils/kelmarsh_backbone_information.py'),
    'verifier_scope': 'Existing arrays and hashes only; no training or prediction generation.'}, indent=2))
