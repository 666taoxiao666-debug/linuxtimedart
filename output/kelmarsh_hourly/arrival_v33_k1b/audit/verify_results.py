"""Read-only replay of K1b metrics and checkpoint selection; no model fitting."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/XT/nuist/pythoncode/TimeDARTFirst')
WORK = Path('/home/XT/nuist/pythoncode/TimeDARTFirst_K1b_20261101T003221Z')
AUDIT = ROOT / 'outputs/audits/arrival_v33_k1b/20261101T003221Z'
RUN = ROOT / 'outputs/logs/Kelmarsh/v33_k1b_20261101T003221Z'
os.chdir(WORK)
sys.path.insert(0, str(WORK))
import numpy as np
from scripts import train_kelmarsh_arrival_v33_k1b as k

begin = time.monotonic()
completion = json.loads((AUDIT / 'completion.json').read_text())
assert completion['exit_code'] == 0
launch = json.loads((AUDIT / 'launch.json').read_text())
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == launch['commit']
assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'exp/K1b'
for name, digest in launch['source_sha256_lf'].items():
    assert hashlib.sha256((WORK / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == digest, name
report = json.loads((RUN / 'k1b_decision.json').read_text())
manifest = json.loads((RUN / 'data_manifest.json').read_text())
assert report['smoke'] is False and manifest['smoke'] is False
cfg = json.loads(k.k1.CONFIG.read_text())
p, series, observations, stations, identity = k.k1.load_real(cfg)
assert manifest['identity'] == identity
windows = {s: series.windows(s, 'all') for s in ('train', 'validation', 'test')}
truth = {s: w.truth_and_anchor()[0] for s, w in windows.items()}
assert len(identity['station_files']) == 252 and len(stations) == 36
for s, label in [('train', 'train'), ('validation', 'validation'), ('test', 'development_eval')]:
    assert manifest['windows'][label] == len(windows[s])
    ys = set(np.unique(series.label_year[windows[s].window_starts[:, None] + series.input_len + np.arange(series.horizon)]).tolist())
    assert ys <= set(series.splits[s]), (s, ys)
    assert np.array_equal(truth[s], series.base.windows(s, 'all').truth_and_anchor()[0])
    assert np.isfinite(truth[s]).all()

persist = {}
for s, label in [('validation', 'validation'), ('test', 'development_eval')]:
    persist[label] = k.k1.scores(windows[s].truth_and_anchor()[1], truth[s], k.RATED)
    for metric in ('mae', 'rmse', 'r2'):
        assert abs(persist[label][metric] - report['metrics']['Persistence'][label][metric]) < 1e-9

rows, all_metrics = [], {}
assert set(d.name for d in (RUN / 'arms').iterdir() if d.is_dir()) == set(k.K1B['arms'])
for arm in k.K1B['arms']:
    folder = RUN / 'arms' / arm
    trained = json.loads((folder / 'trained.json').read_text())
    history = json.loads((folder / 'history.json').read_text())
    assert trained['contract']['run_contract'] == manifest['run_contract']
    assert 1 <= len(history) - 1 <= 20
    assert [v['epoch'] for v in history] == list(range(len(history)))
    assert all(np.isfinite(v[n]) for v in history for n in ('mae', 'rmse', 'r2', 'J_p'))
    best = min(history, key=lambda v: v['J_p'])
    assert trained['best_epoch'] == best['epoch'] == report['metrics'][arm]['best_epoch']
    assert abs(trained['validation_J_p'] - best['J_p']) < 1e-9
    for h in history:
        assert abs(k.j_p(h, persist['validation']) - h['J_p']) < 1e-9
    pred = np.load(folder / 'predictions.npz')
    pooled, by_h = {}, {}
    for s, label, key in [('validation', 'validation', 'validation_starts'), ('test', 'development_eval', 'eval_starts')]:
        assert np.array_equal(pred[key], windows[s].window_starts)
        pooled[label] = k.k1.scores(pred[label], truth[s], k.RATED)
        by_h[label] = [k.k1.scores(pred[label][:, h], truth[s][:, h], k.RATED) for h in range(4)]
        for metric in ('mae', 'rmse', 'r2'):
            assert abs(pooled[label][metric] - trained[label][metric]) < 1e-9
            assert abs(pooled[label][metric] - report['metrics'][arm][label][metric]) < 1e-9
    all_metrics[arm] = pooled['development_eval']
    rows.append({'arm': arm, 'metrics': pooled, 'by_h': by_h, 'best_epoch': best['epoch'],
                 'epochs_run': len(history) - 1, 'kappa': trained['kappa'], 'parameters': trained['trainable_parameters']})

def rmse_gain(a, b):
    return 100 * (all_metrics[b]['rmse'] - all_metrics[a]['rmse']) / all_metrics[b]['rmse']

def mae_gain(a, b):
    return all_metrics[b]['mae'] - all_metrics[a]['mae']

prior = [b for b in ('T9', 'T21') if rmse_gain(b+'+A', b+'+B2') >= 1 and mae_gain(b+'+A', b+'+B2') >= -1
         and rmse_gain(b+'+A', b) >= 1 and mae_gain(b+'+A', b) >= 0]
fusion = [b+'+'+a for b in ('T9', 'T21') for a in ('B2', 'A')
          if rmse_gain(b+'+'+a, b) >= 1 and mae_gain(b+'+'+a, b) >= 0]
independent_branch = 'E1' if prior else 'E2' if fusion else 'E3'
assert independent_branch == report['decision']['branch']
assert k.k1b_decision(all_metrics) == report['decision']
result = {'passed': True, 'commit': launch['commit'], 'decision_recomputed': independent_branch,
          'source_files_verified': len(launch['source_sha256_lf']), 'station_files_verified': len(identity['station_files']),
          'stations': len(stations), 'windows': manifest['windows'], 'labels_equal_frozen_provider': True,
          'predictions_aligned_all_cohort': True, 'metrics_max_abs_tolerance': 1e-9,
          'checkpoint_selection': 'earliest minimum validation J_p including epoch 0', 'arms': rows,
          'real_training_seconds': completion['seconds'], 'verification_seconds': time.monotonic()-begin,
          'completed_utc': k.k1.now()}
(AUDIT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')

# Export only reviewable records, not weights or prediction arrays. Preserve raw-byte hashes.
files = [(f, f.relative_to(RUN).as_posix()) for f in RUN.rglob('*') if f.is_file() and f.name in
         {'k1b_decision.json', 'k1b_summary.md', 'data_manifest.json', 'trained.json', 'history.json'}]
for name in ['launch.json', 'process.json', 'completion.json', 'preflight.json', 'verification.json',
             'tests.log', 'smoke.log', 'smoke.exit', 'train.log', 'launch_once.py', 'verify_results.py']:
    files.append((AUDIT / name, 'audit/' + name))
records = {target: {'sha256': hashlib.sha256(src.read_bytes()).hexdigest(), 'bytes': src.stat().st_size} for src, target in files}
(AUDIT / 'export_sha256.json').write_text(json.dumps(records, indent=2) + '\n')
import tarfile
with tarfile.open(AUDIT / 'reviewable_results.tar.gz', 'w:gz') as archive:
    for src, target in files:
        archive.add(src, arcname=target)
    archive.add(AUDIT / 'export_sha256.json', arcname='audit/export_sha256.json')
print(json.dumps({'verified': True, 'decision': independent_branch, 'seconds': completion['seconds'],
                  'arms': [{k: r[k] for k in ('arm', 'best_epoch', 'epochs_run', 'kappa')} | r['metrics']['development_eval'] for r in rows]}))
