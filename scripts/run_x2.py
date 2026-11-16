"""X2: LightGBM information ablation + ANI (part A) and base screening (part B).

Author: Claude (2026-11-16).  Under the user's split, Claude writes all code, GPT reviews it and Cursor launches it.
Plan and gates: docs/STAGE_PLAN.md (X2); protocol: configs/x2_protocol.json (SHA passed on the command line).

Modes (each writes into a FRESH --output directory and leaves failed.json on any error; no retries):
  smoke   tiny real slices end-to-end (part A, part B, report) with a minimal budget; results are never reported.
  a       part A on CPU (timedart, LightGBM 4.5.0): 6 arms x 3 farms; V1/V2/test predictions + per-farm dev freeze.
  b       part B on GPU: N_v1, TCNGRU, DATCNGRU, CNN, BILSTM; same layout.
  report  CPU: dev decisions (gates, base selection, fusion) are written FIRST, then exploratory test statistics.

Selection uses V1 only, gates/selection use V2 only; the test sets were opened before, so every test number is
exploratory.  A per-farm dev_freeze.json (with prediction SHAs) is written before any test prediction of that farm.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_x1 as x1  # noqa: E402  (numpy-only at import time)

digest, read, save, checked = x1.digest, x1.read, x1.save, x1.checked
PROTOCOL = ROOT / 'configs/x2_protocol.json'
Z95 = 1.959963984540054

# x50 last-step channel layout (prepare_g1_batches.nwp_input): 0-7 local SCADA, 8-18 upstream features,
# 19 upstream-missing flag, 20 GFS history, 21-33 GFS current cycle, 34-46 GFS masks, 47 age, 48 cycle missing, 49 power.
LOCAL = list(range(8)) + [49]
UPSTREAM = list(range(8, 20))
GFS = list(range(20, 49))
INFO = {'L0': LOCAL, 'LU': LOCAL + UPSTREAM, 'LG': LOCAL + GFS, 'LUG': list(range(50))}


def legacy():
    """models/LegacyBases.py by file path (models/__init__ imports torch-only modules)."""
    spec = importlib.util.spec_from_file_location('LegacyBases', ROOT / 'models/LegacyBases.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------- data helpers (numpy)
def period(times, bounds):
    """Hour-end labels in [lo, hi) after shifting back one hour (prepare_g1_batches.period_mask)."""
    t = np.asarray(times).astype('datetime64[ns]') - np.timedelta64(1, 'h')
    return (t >= np.datetime64(bounds[0])) & (t < np.datetime64(bounds[1]))


def split_masks(pack, v1, v2):
    times = np.asarray(pack['times'])
    if times.ndim != 2 or times.shape != (len(pack['truth']), 4):
        raise ValueError('Target times [n, 4] required')
    m1, m2 = period(times, v1).all(1), period(times, v2).all(1)
    if (m1 & m2).any() or not m1.any() or not m2.any():
        raise ValueError('V1/V2 must be disjoint and non-empty')
    return m1, m2


def subset(pack, mask):
    n = len(pack['truth'])
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == n else v) for k, v in pack.items()}


def head(pack, count):
    return subset(pack, np.arange(len(pack['truth'])) < count)


def key_bytes(pack):
    return {k: hashlib.sha256(np.ascontiguousarray(pack[k]).tobytes()).hexdigest() for k in ('issue', 'turbines', 'truth')}


def tree_matrix(pack, info, hot):
    """run_x1.tree_features restricted to the last-step x50 columns of one information set."""
    x = pack['x50']
    if x.ndim != 3 or x.shape[1:] != (24, 50):
        raise ValueError('Unexpected frozen feature dimensions')
    parts = [x[:, -1, INFO[info]], x[:, :, -1], x[:, :, 0]]
    if hot:
        parts.append(pack['retrofit'].reshape(-1, 1))
    parts.append(pack['turbines'].reshape(-1, 1))
    result = np.concatenate(parts, axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite tree inputs')
    return result


# ----------------------------------------------------------------------------- ANI / PCL (numpy)
def fit_power_curves(train, width=.5, minimum=30, min_bins=2):
    """Per-turbine GFS-forecast-wind -> power curve from the FIT period only (target cells h1-h4, gF cols 6-9).
    Turbines with < min_bins valid bins use the farm-pooled curve; a pooled curve with < min_bins is None."""
    w = train['gF'][:, 6:10].astype(float)
    y = train['truth'].astype(float)
    valid = train['gF_mask'][:, 6:10] > 0
    pooled = x1.curve_points(w, y, valid, width, minimum)
    pooled = pooled if len(pooled) >= min_bins else None
    curves = {}
    for u in np.unique(train['turbines']):
        take = train['turbines'] == u
        own = x1.curve_points(w[take], y[take], valid[take], width, minimum)
        if len(own) >= min_bins:
            curves[int(u)] = dict(points=own, source='turbine')
        else:
            curves[int(u)] = dict(points=pooled, source='pooled' if pooled is not None else 'none')
    return curves


def curve_values(pack, curves, columns):
    """PC(gF[:, c]) for each requested gF column; NaN where the mask/value is invalid or no curve exists."""
    w = pack['gF'][:, columns].astype(float)
    ok = (pack['gF_mask'][:, columns] > 0) & np.isfinite(w) & (w >= 0)
    out = np.full(w.shape, np.nan)
    for u in np.unique(pack['turbines']):
        points = curves.get(int(u), dict(points=None))['points']
        if points is None:
            continue
        rows = pack['turbines'] == u
        cell = rows[:, None] & ok
        out[cell] = np.interp(w[cell], points[:, 0], points[:, 1])
    return out


def ani_features(pack, curves, rated, kind):
    """PCL: [PC(g(t0)), PC(g(t0+1..4))] (5).  ANI: [dP_1..4, clip(P(t0) + dP_h, 0, rated)] (8).
    g(t0) = gF col 5 (current cycle at lead age = the GFS history channel at the issue hour)."""
    pc = curve_values(pack, curves, [5, 6, 7, 8, 9])
    if kind == 'PCL':
        return pc.astype(np.float32)
    if kind != 'ANI':
        raise ValueError('kind must be PCL or ANI')
    delta = pc[:, 1:] - pc[:, :1]
    anchored = np.clip(pack['anchor'].astype(float)[:, None] + delta, 0, rated)
    anchored[~np.isfinite(delta)] = np.nan
    return np.concatenate([delta, anchored], axis=1).astype(np.float32)


# ----------------------------------------------------------------------------- decisions (numpy)
def pooled_effect(stats):
    """Inverse-variance pooled RELATIVE gain (STAGE_PLAN X2, R1-B2).  stats: farm -> paired_loss_statistics dict."""
    rows = {}
    for farm, s in stats.items():
        width = s['ci95_high'] - s['ci95_low']
        ref = s['reference_mean_loss']
        if not (np.isfinite(width) and width > 0 and np.isfinite(ref) and ref > 0):
            return dict(status='TECHNICAL_FAILURE', reason='zero or nonfinite CI width / reference loss', farm=farm)
        se = width / (2 * Z95)
        rows[farm] = dict(relative_gain=s['gain'] / ref, relative_se=se / ref)
    w = {f: 1 / r['relative_se'] ** 2 for f, r in rows.items()}
    total = sum(w.values())
    est = sum(w[f] * rows[f]['relative_gain'] for f in rows) / total
    se = (1 / total) ** .5
    return dict(status='OK', farms=rows, pooled_relative_gain=est, pooled_se=se,
                ci95_low=est - Z95 * se, ci95_high=est + Z95 * se)


def gate_decisions(ani_vs_lug, ani_vs_pcl, min_gain=.01, min_positive=2):
    p1, p2 = pooled_effect(ani_vs_lug), pooled_effect(ani_vs_pcl)
    if p1['status'] != 'OK' or p2['status'] != 'OK':
        return dict(gate1=None, gate2=None, pooled_ani_vs_lug=p1, pooled_ani_vs_pcl=p2, outcome='TECHNICAL_FAILURE')
    g1 = bool(all(r['relative_gain'] >= min_gain for r in p1['farms'].values()) and p1['ci95_low'] > 0)
    g2 = bool(p2['ci95_low'] > 0 and sum(r['relative_gain'] > 0 for r in p2['farms'].values()) >= min_positive)
    near = bool(not g1 and all(r['relative_gain'] > 0 for r in p1['farms'].values()))
    outcome = ('GATES_1_2_PASS_RUN_GATE3' if g1 and g2 else 'GATE1_ONLY_FEATURE_ENGINEERING' if g1
               else 'NEAR_PASS_REGISTER_BEFORE_ANY_ROUND' if near else 'NEGATIVE')
    return dict(gate1=g1, gate2=g2, near_pass=near, outcome=outcome, pooled_ani_vs_lug=p1, pooled_ani_vs_pcl=p2,
                attribution='whole ANI representation (increment + anchor), never the anchor alone (R1-B1)')


def select_base(v2_mae, reference='LUG', margin=.01):
    """v2_mae: base -> farm -> MAE.  Lowest mean relative MAE vs reference; within `margin` keeps the reference."""
    farms = sorted(v2_mae[reference])
    rel = {b: float(np.mean([v2_mae[b][f] / v2_mae[reference][f] - 1 for f in farms]))
           for b in v2_mae if all(f in v2_mae[b] for f in farms)}
    best = min(sorted(rel), key=lambda b: rel[b])
    chosen = best if rel[best] <= -margin else reference
    return dict(relative_mae=rel, best=best, selected=chosen, margin=margin, reference=reference)


# ----------------------------------------------------------------------------- run bookkeeping
def git_identity():
    try:
        commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, dirty


def start_run(args, mode, extra=None):
    """Protocol SHA lock, clean committed worktree, fresh output directory, launch record."""
    checked(PROTOCOL, args.protocol_sha256)
    commit, dirty = git_identity()
    if not args.allow_dirty and (commit is None or dirty):
        raise ValueError('Run from a clean worktree of the reviewed commit (git status not clean or no git)')
    if mode in ('A', 'B'):  # formal fits only after a passed smoke run of the SAME commit
        smoke = read(Path(args.smoke_result) / 'smoke_passed.json') if args.smoke_result else {}
        if smoke.get('status') != 'PASS' or smoke.get('commit') != commit:
            raise ValueError('Formal A/B runs need --smoke-result of a passed smoke run of this commit')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / '.gitignore').write_text('*.npy\n*.npz\n*.pt\nmodels/\n', encoding='utf-8')
    save(out / 'launch.json', dict(mode=mode, commit=commit, script_sha256=digest(__file__),
         protocol_sha256=args.protocol_sha256, python=sys.executable, pid=os.getpid(),
         started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), args={k: str(v) for k, v in vars(args).items()},
         **(extra or {})))
    return out, read(PROTOCOL), read(ROOT / 'configs/x1_protocol.json')


def beat(out, message):
    save(out / 'heartbeat.json', dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), message=message))
    print('[X2]', message, flush=True)


def fail(out, exc):
    save(out / 'failed.json', dict(type=type(exc).__name__, message=str(exc), retry=False,
                                   utc=dt.datetime.now(dt.timezone.utc).isoformat()))


class Store:
    """Saves arrays under the run directory and records their SHA-256 (relative path -> sha)."""

    def __init__(self, root):
        self.root, self.registry = Path(root), {}

    def put(self, rel, array):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(array))
        self.registry[str(rel)] = digest(path)
        return path


def farms(x1p):
    """farm -> {'items': {name: item}, 'rated': kW} from the X1 dataset registry (same 4 test sets)."""
    out = {}
    for name, item in x1p['datasets'].items():
        out.setdefault(item['farm'], dict(items={}, rated=float(item['rated_kw'])))['items'][name] = item
    return out


def budget(protocol, smoke):
    b = dict(protocol['budget']['smoke'] if smoke else protocol['budget']['formal'])
    b['smoke'] = smoke
    return b


def load_farm(args, farm, spec, protocol, bud):
    """Development train/V1/V2 (V1/V2 = registered halves of the frozen validation segment) + exposed test packs."""
    first = next(iter(spec['items'].values()))
    train, val = x1.development(Path(args.logs), first)
    m1, m2 = split_masks(val, protocol['splits'][farm]['V1'], protocol['splits'][farm]['V2'])
    v1, v2 = subset(val, m1), subset(val, m2)
    dropped = int(len(val['truth']) - m1.sum() - m2.sum())
    tests = {}
    for name, item in spec['items'].items():
        path, _ = x1.paths(Path(args.logs), item)
        tests[name] = x1.load_pack(path)
        if len(tests[name]['truth']) != item['count']:
            raise ValueError('Test cohort size differs: ' + name)
    if bud['smoke']:
        train, v1, v2 = head(train, bud['train_rows']), head(v1, bud['eval_rows']), head(v2, bud['eval_rows'])
        tests = {k: head(v, bud['eval_rows']) for k, v in tests.items()}
    return train, v1, v2, tests, dict(V1=int(m1.sum()), V2=int(m2.sum()), crossing_dropped=dropped)


def save_keys(store, part, farm, packs):
    """Labels/keys for every evaluated cohort (V1, V2, tests) so the report and Claude can recompute."""
    for split, pack in packs.items():
        rel = f'{part}/keys/{farm}_{split}.npz'
        path = store.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{k: pack[k] for k in ('truth', 'issue', 'turbines', 'times', 'anchor')})
        store.registry[rel] = digest(path)
    return {split: key_bytes(p) for split, p in packs.items()}


# ----------------------------------------------------------------------------- part A: LightGBM ablation + ANI
def lgb_params(cfg, leaves, child, seed, threads, bud):
    return dict(objective='regression_l1', num_leaves=leaves, min_child_samples=child,
                learning_rate=cfg['learning_rate'], n_estimators=bud.get('n_estimators', cfg['n_estimators']),
                colsample_bytree=cfg['feature_fraction'], subsample=cfg['bagging_fraction'], subsample_freq=1,
                random_state=seed, n_jobs=threads, verbosity=-1, deterministic=True, force_col_wise=True)


def arm_matrix(pack, spec, hot, curves, rated):
    """tree_matrix of the arm's information set; ANI/PCL columns inserted before the categorical turbine id."""
    base = tree_matrix(pack, spec['info'], hot)
    if not spec.get('extra'):
        return base
    extra = ani_features(pack, curves, rated, spec['extra'])
    return np.concatenate([base[:, :-1], extra, base[:, -1:]], axis=1).astype(np.float32)


def fit_horizons(lgb, X, Y, Xv, Yv, params, rounds):
    models = []
    for h in range(Y.shape[1]):
        m = lgb.LGBMRegressor(**params)
        m.fit(X, Y[:, h], eval_set=[(Xv, Yv[:, h])], eval_metric='l1', categorical_feature=[X.shape[1] - 1],
              callbacks=[lgb.early_stopping(rounds, verbose=False)])
        models.append(m)
    return models


def save_boosters(models, folder):
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for h, m in enumerate(models):
        path = folder / f'h{h + 1}.txt'
        best = getattr(m, 'best_iteration_', None)
        m.booster_.save_model(str(path), num_iteration=best if best else None)
        out[path.name] = digest(path)
    return out


def booster_predict(lgb, folder, hashes, X):
    cols = []
    for name in sorted(hashes):
        cols.append(lgb.Booster(model_file=str(checked(folder / name, hashes[name]))).predict(X))
    return np.stack(cols, axis=1)


def run_a_farm(lgb, farm, rated, train, v1, v2, tests, protocol, x1p, store, bud, threads, out):
    hot = farm == 'HoT'
    cfg = x1p['lightgbm']
    rounds = bud.get('early_stopping_rounds', cfg['early_stopping_rounds'])
    (out / 'A' / farm).mkdir(parents=True, exist_ok=True)
    probe = head(train, 1000)
    if not np.array_equal(tree_matrix(probe, 'LUG', hot), x1.tree_features(probe, hot)):
        raise ValueError('LUG features differ from the A1 feature set')
    curves = fit_power_curves(train, **protocol['part_a']['curve'])
    save(out / 'A' / farm / 'curves.json', {str(u): dict(source=c['source'], points=None if c['points'] is None
                                                       else c['points'].tolist()) for u, c in curves.items()})
    freeze = dict(curves_sha256=digest(out / 'A' / farm / 'curves.json'), arms={})
    grid = [(a, b) for a in cfg['num_leaves'] for b in cfg['min_child_samples']][:bud['grid_cells']]
    for arm, spec in protocol['part_a']['arms'].items():
        seeds = spec['seeds'][:bud['seeds']]
        X = {k: arm_matrix(p, spec, hot, curves, rated) for k, p in (('train', train), ('V1', v1), ('V2', v2))}
        rows, best = [], None
        for leaves, child in grid:
            models = fit_horizons(lgb, X['train'], train['truth'], X['V1'], v1['truth'],
                                  lgb_params(cfg, leaves, child, seeds[0], threads, bud), rounds)
            pv1 = np.stack([m.predict(X['V1']) for m in models], axis=1)
            row = dict(num_leaves=leaves, min_child_samples=child, v1_mae=float(np.abs(pv1 - v1['truth']).mean()),
                       best_iterations=[int(m.best_iteration_ or 0) for m in models])
            rows.append(row)
            beat(out, f'A {farm} {arm} grid {row}')
            if best is None or row['v1_mae'] < best[0]['v1_mae']:  # ties keep the earlier grid cell
                best = (row, models, pv1)
        selected, record = best[0], dict(grid=rows, selected=best[0], seeds={})
        for i, seed in enumerate(seeds):
            if i == 0:
                models, pv1_mem = best[1], best[2]
            else:
                models = fit_horizons(lgb, X['train'], train['truth'], X['V1'], v1['truth'],
                                      lgb_params(cfg, selected['num_leaves'], selected['min_child_samples'], seed,
                                                 threads, bud), rounds)
                pv1_mem = np.stack([m.predict(X['V1']) for m in models], axis=1)
            folder = out / 'A' / farm / 'models' / arm / str(seed)
            hashes = save_boosters(models, folder)
            pv1 = booster_predict(lgb, folder, hashes, X['V1'])
            if not np.allclose(pv1, pv1_mem, rtol=1e-6, atol=1e-3):
                raise ValueError('Saved booster does not reproduce in-memory predictions')
            pv2 = booster_predict(lgb, folder, hashes, X['V2'])
            store.put(f'A/{farm}/{arm}/{seed}/V1.npy', pv1)
            store.put(f'A/{farm}/{arm}/{seed}/V2.npy', pv2)
            record['seeds'][str(seed)] = dict(boosters=hashes, v1_mae=float(np.abs(pv1 - v1['truth']).mean()),
                                              best_iterations=[int(m.best_iteration_ or 0) for m in models])
            beat(out, f'A {farm} {arm} seed {seed} V1 MAE {record["seeds"][str(seed)]["v1_mae"]:.3f}')
        freeze['arms'][arm] = record
        del X
    freeze['predictions'] = {k: v for k, v in store.registry.items() if k.startswith(f'A/{farm}/')}
    save(out / 'A' / farm / 'dev_freeze.json', freeze)          # frozen BEFORE any test prediction of this farm
    for name, pack in tests.items():
        for arm, spec in protocol['part_a']['arms'].items():
            Xt = arm_matrix(pack, spec, hot, curves, rated)
            for seed, rec in freeze['arms'][arm]['seeds'].items():
                pred = booster_predict(lgb, out / 'A' / farm / 'models' / arm / seed, rec['boosters'], Xt)
                store.put(f'A/{farm}/{arm}/{seed}/{name}.npy', pred)
        beat(out, f'A {farm} test {name} predicted')
    save(out / 'A' / farm / 'test_predictions.json',
         {k: v for k, v in store.registry.items() if k.startswith(f'A/{farm}/') and k not in freeze['predictions']})


def run_a(args, out=None, protocol=None, x1p=None):
    import lightgbm as lgb
    if lgb.__version__ != '4.5.0':
        raise ValueError('LightGBM fits require the timedart environment with lightgbm 4.5.0')
    if out is None:
        out, protocol, x1p = start_run(args, 'A', dict(lightgbm=lgb.__version__, threads=args.threads))
    bud = budget(protocol, args.mode == 'smoke')
    store = Store(out)
    (out / 'A').mkdir(parents=True, exist_ok=True)
    try:
        for farm, spec in farms(x1p).items():
            train, v1, v2, tests, counts = load_farm(args, farm, spec, protocol, bud)
            keys = save_keys(store, 'A', farm, dict(V1=v1, V2=v2, **tests))
            save(out / 'A' / f'{farm}_cohorts.json', dict(counts=counts, train=len(train['truth']), keys=keys))
            run_a_farm(lgb, farm, spec['rated'], train, v1, v2, tests, protocol, x1p, store, bud, args.threads, out)
            del train, v1, v2, tests
        save(out / 'A' / 'registry.json', store.registry)
        save(out / 'A' / 'completed.json', dict(status='COMPLETED', registry_sha256=digest(out / 'A' / 'registry.json')))
    except Exception as exc:
        fail(out, exc)
        raise
    return out


# ----------------------------------------------------------------------------- part B: neural bases
def predict_nn(model, pack, device, batch=512):
    """Same as train_g1.predict for x50 models (no station block): normalised output -> kW."""
    import torch
    model.eval()
    x, outs = pack['x50'], []
    with torch.no_grad():
        for lo in range(0, len(x), batch):
            xb = torch.as_tensor(x[lo:lo + batch], dtype=torch.float32, device=device)
            outs.append(model(xb, None).detach().cpu().numpy().reshape(-1, 4))
    return np.concatenate(outs) * float(pack['power_scale']) + float(pack['power_mean'])


def fit_nn(factory, train, val, config, seed, folder, device, domains=None, lam_end=None, min_epochs=0):
    """train_g1.fit_one, unchanged except: plain x50 input, optional domain-adversarial term (standard GRL,
    LegacyBases.da_loss) and no early stop before `min_epochs`.  `val` is V1 (early stopping only)."""
    import random
    import torch
    lb = legacy()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = factory(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_epochs'], eta_min=config['eta_min'])
    best = float(np.abs(predict_nn(model, val, device) - val['truth']).mean())
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epoch_best, stale, history = 0, 0, [dict(epoch=0, val_mae=best)]
    xs = train['x50']
    y = (train['truth'] - float(train['power_mean'])) / float(train['power_scale'])
    dom = None if domains is None else torch.as_tensor(np.asarray(domains), dtype=torch.long, device=device)
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        lam = 0.0 if dom is None else lb.lambda_schedule(epoch, config['max_epochs'], lam_end)
        order = np.random.default_rng(seed + epoch).permutation(len(xs))
        for lo in range(0, len(xs), config['batch_size']):
            ix = order[lo:lo + config['batch_size']]
            xb = torch.as_tensor(xs[ix], dtype=torch.float32, device=device)
            target = torch.as_tensor(y[ix], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            if dom is None:
                error = model(xb, None).reshape(-1, 4) - target
                loss = config['mae_weight'] * error.abs().mean() + config['mse_weight'] * error.square().mean()
            else:
                loss, _, _ = lb.da_loss(model, xb, target, dom[torch.as_tensor(ix, device=device)], lam,
                                        config['mae_weight'], config['mse_weight'])
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
        scheduler.step()
        score = float(np.abs(predict_nn(model, val, device) - val['truth']).mean())
        if not np.isfinite(score):
            raise ValueError('Non-finite validation score')
        history.append(dict(epoch=epoch, val_mae=score, lam=lam))
        if score < best:
            best, epoch_best, stale = score, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config['patience'] and epoch >= min_epochs:
            break
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    torch.save(state, folder / 'selected.pt')
    result = dict(seed=seed, v1_mae=best, best_epoch=epoch_best, epochs=len(history) - 1, lam_end=lam_end,
                  min_epochs=min_epochs, sha256=digest(folder / 'selected.pt'), history=history)
    save(folder / 'trained.json', result)
    model.load_state_dict(state)
    return model, result


def nn_factory(arm, farm, train, config, lb):
    from scripts import train_g1 as g1, train_g2b as g2
    if arm == 'N_v1':
        if farm == 'HoT':
            return lambda device: g2.make_model('N', train, config, device)
        return lambda device: g1.make_model('N', train, config, device)
    c_in = train['x50'].shape[-1]
    return lambda device: lb.build_legacy(arm, c_in).to(device)


def load_nn(factory, folder, sha, device):
    import torch
    model = factory(device)
    model.load_state_dict(torch.load(checked(Path(folder) / 'selected.pt', sha), map_location=device, weights_only=True))
    return model


def run_b(args, out=None, protocol=None, x1p=None):
    import torch
    from scripts import train_g2b as g2
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and args.gpu_fraction:
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    if out is None:
        out, protocol, x1p = start_run(args, 'B', dict(device=device, torch=torch.__version__, threads=args.threads,
                                                       gpu_fraction=args.gpu_fraction))
    bud = budget(protocol, args.mode == 'smoke')
    lb = legacy()
    store = Store(out)
    (out / 'B').mkdir(parents=True, exist_ok=True)
    try:
        for farm, spec in farms(x1p).items():
            hot = farm == 'HoT'
            config = dict(read(ROOT / ('configs/g2b_protocol.json' if hot else 'configs/g1_protocol.json'))['fit'])
            config['max_epochs'] = bud.get('max_epochs', config['max_epochs'])
            train, v1, v2, tests, counts = load_farm(args, farm, spec, protocol, bud)
            keys = save_keys(store, 'B', farm, dict(V1=v1, V2=v2, **tests))
            save(out / 'B' / f'{farm}_cohorts.json', dict(counts=counts, train=len(train['truth']), keys=keys))
            if hot:  # same retrofit channel as the frozen HoT N (train_g2b.arm_pack)
                train, v1, v2 = (g2.arm_pack(p, 'N', True) for p in (train, v1, v2))
                tests = {k: g2.arm_pack(p, 'N', True) for k, p in tests.items()}
            domains = lb.time_domains(np.asarray(train['issue']).astype('datetime64[ns]'), protocol['part_b']['domains'])
            (out / 'B' / farm).mkdir(parents=True, exist_ok=True)
            freeze = dict(config=config, arms={})
            for arm, aspec in protocol['part_b']['arms'].items():
                factory = nn_factory(arm, farm, train, config, lb)
                lams = aspec.get('lambda_end', [None])[:bud['lambda_cells']]
                runs = {}
                for lam in lams:
                    tag = 'none' if lam is None else str(lam)
                    runs[tag] = {}
                    for seed in aspec['seeds'][:bud['seeds']]:
                        folder = out / 'B' / farm / 'models' / arm / tag / str(seed)
                        model, result = fit_nn(factory, train, v1, config, seed, folder, device,
                                               domains if lam is not None else None, lam, aspec.get('min_epochs', 0))
                        store.put(f'B/{farm}/{arm}/{tag}/{seed}/V1.npy', predict_nn(model, v1, device))
                        store.put(f'B/{farm}/{arm}/{tag}/{seed}/V2.npy', predict_nn(model, v2, device))
                        runs[tag][str(seed)] = {k: result[k] for k in ('v1_mae', 'best_epoch', 'epochs', 'sha256')}
                        beat(out, f'B {farm} {arm} lam={tag} seed {seed} V1 MAE {result["v1_mae"]:.3f}')
                        del model
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                chosen = min(runs, key=lambda t: np.mean([r['v1_mae'] for r in runs[t].values()]))
                freeze['arms'][arm] = dict(runs=runs, selected=chosen, selection='mean V1 MAE over seeds')
            freeze['predictions'] = {k: v for k, v in store.registry.items() if k.startswith(f'B/{farm}/')}
            save(out / 'B' / farm / 'dev_freeze.json', freeze)   # frozen BEFORE any test prediction of this farm
            for name, pack in tests.items():
                for arm, rec in freeze['arms'].items():
                    factory = nn_factory(arm, farm, train, config, lb)
                    tag = rec['selected']
                    for seed, r in rec['runs'][tag].items():
                        model = load_nn(factory, out / 'B' / farm / 'models' / arm / tag / seed, r['sha256'], device)
                        store.put(f'B/{farm}/{arm}/{tag}/{seed}/{name}.npy', predict_nn(model, pack, device))
                        del model
                beat(out, f'B {farm} test {name} predicted')
            save(out / 'B' / farm / 'test_predictions.json',
                 {k: v for k, v in store.registry.items() if k.startswith(f'B/{farm}/') and k not in freeze['predictions']})
            del train, v1, v2, tests
        save(out / 'B' / 'registry.json', store.registry)
        save(out / 'B' / 'completed.json', dict(status='COMPLETED', registry_sha256=digest(out / 'B' / 'registry.json')))
    except Exception as exc:
        fail(out, exc)
        raise
    return out


# ----------------------------------------------------------------------------- report
def metrics(preds, truth, rated):
    """Seed-mean per-seed metrics plus the seed-ensemble (mean prediction)."""
    denom = float(np.square(truth - truth.mean()).sum())

    def one(p):
        e = p - truth
        rmse = float(np.sqrt(np.square(e).mean()))
        return dict(mae=float(np.abs(e).mean()), rmse=rmse, r2=float(1 - np.square(e).sum() / denom),
                    accuracy=1 - rmse / rated, bias=float(e.mean()), lead_mae=np.abs(e).mean(0).tolist())
    per = [one(p) for p in preds]
    mean = {k: (np.mean([m[k] for m in per], axis=0).tolist() if k == 'lead_mae' else float(np.mean([m[k] for m in per])))
            for k in per[0]}
    return dict(seeds=len(preds), seed_mean=mean, ensemble=one(np.mean(preds, axis=0)))


def loss(preds, truth):
    return np.mean([np.abs(p - truth) for p in preds], axis=0)


class Saved:
    """SHA-checked access to the predictions of a finished A or B run."""

    def __init__(self, root, part):
        self.root, self.part = Path(root), part
        done = read(self.root / part / 'completed.json')
        if done['status'] != 'COMPLETED' or (self.root / 'failed.json').exists():
            raise ValueError(f'Part {part} run is not complete')
        self.registry = read(checked(self.root / part / 'registry.json', done['registry_sha256']))
        self.freeze = {}

    def dev(self, farm):
        if farm not in self.freeze:
            self.freeze[farm] = read(self.root / self.part / farm / 'dev_freeze.json')
        return self.freeze[farm]

    def preds(self, farm, arm, split, seeds=None, tag=None):
        if self.part == 'A':
            seeds = seeds or list(self.dev(farm)['arms'][arm]['seeds'])
            rels = [f'A/{farm}/{arm}/{s}/{split}.npy' for s in seeds]
        else:
            rec = self.dev(farm)['arms'][arm]
            tag = tag or rec['selected']
            rels = [f'B/{farm}/{arm}/{tag}/{s}/{split}.npy' for s in (seeds or list(rec['runs'][tag]))]
        return [np.load(checked(self.root / r, self.registry[r])) for r in rels]

    def keys(self, farm, split):
        rel = f'{self.part}/keys/{farm}_{split}.npz'
        with np.load(checked(self.root / rel, self.registry[rel])) as z:
            return {k: z[k] for k in z.files}


def check_part_identity(root, part, expected, smoke=False):
    """Reject mixed implementations/protocols and smoke arrays in formal reports."""
    launch = read(Path(root) / 'launch.json')
    for key in ('commit', 'script_sha256', 'protocol_sha256'):
        if not expected.get(key) or launch.get(key) != expected[key]:
            raise ValueError(f'Part {part} launch identity differs: {key}')
    if launch.get('mode') != ('SMOKE' if smoke else part):
        raise ValueError(f'Part {part} launch mode differs')


def run_report(args, out=None, protocol=None, x1p=None):
    from utils.kelmarsh_backbone_information import paired_loss_statistics
    lb = legacy()
    if out is None:
        out, protocol, x1p = start_run(args, 'REPORT')
    smoke = args.mode == 'smoke'
    expected = read(out / 'launch.json')
    check_part_identity(args.a_results, 'A', expected, smoke)
    if args.b_results:
        check_part_identity(args.b_results, 'B', expected, smoke)
    st = protocol['statistics_smoke' if smoke else 'statistics']
    A = Saved(args.a_results, 'A')
    B = Saved(args.b_results, 'B') if args.b_results else None

    def paired(cand, ref, issue):
        return paired_loss_statistics(cand, ref, np.asarray(issue).astype('datetime64[ns]'),
                                      block_length=st['block_length'], replicates=st['replicates'], seed=st['seed'])
    try:
        decisions = dict(part_a={}, part_b={}, exploratory=False)
        ani_lug, ani_pcl, v2_mae, stacks = {}, {}, {}, {}
        for farm, spec in farms(x1p).items():
            k2 = A.keys(farm, 'V2')
            if B is not None:
                for split in ['V1', 'V2'] + list(spec['items']):
                    a, b = A.keys(farm, split), B.keys(farm, split)
                    if any(not np.array_equal(a[k], b[k]) for k in ('truth', 'issue', 'turbines')):
                        raise ValueError(f'A/B cohort keys differ: {farm} {split}')
            y2, i2 = k2['truth'], k2['issue']
            L = {arm: loss(A.preds(farm, arm, 'V2'), y2) for arm in protocol['part_a']['arms']}
            one = {arm: loss(A.preds(farm, arm, 'V2', seeds=['2024']), y2) for arm in ('L0', 'LU', 'LG', 'LUG')}
            ani_lug[farm] = paired(L['LUG_ANI'], L['LUG'], i2)
            ani_pcl[farm] = paired(L['LUG_ANI'], L['LUG_PCL'], i2)
            decisions['part_a'][farm] = dict(
                v2_metrics={arm: metrics(A.preds(farm, arm, 'V2'), y2, spec['rated']) for arm in protocol['part_a']['arms']},
                ANI_vs_LUG=ani_lug[farm], ANI_vs_PCL=ani_pcl[farm],
                information_seed2024=dict(U=paired(one['LU'], one['L0'], i2), G=paired(one['LG'], one['L0'], i2),
                                          U_given_G=paired(one['LUG'], one['LG'], i2),
                                          G_given_U=paired(one['LUG'], one['LU'], i2), descriptive_only=True))
            v2_mae.setdefault('LUG', {})[farm] = float(L['LUG'].mean())
            if B is not None:
                k1 = A.keys(farm, 'V1')
                for arm in protocol['part_b']['arms']:
                    v2_mae.setdefault(arm, {})[farm] = float(loss(B.preds(farm, arm, 'V2'), y2).mean())
                fus = protocol['part_b']['fusion']
                member = lambda split: np.stack([np.mean(A.preds(farm, m, split), 0) if m == 'LUG'
                                                 else np.mean(B.preds(farm, m, split), 0) for m in fus['members']], 1)
                stacks[farm] = lb.fit_stacking(member('V1'), k1['truth'], np.asarray(k1['issue']).astype('datetime64[ns]'),
                                               spec['rated'], tuple(fus['alphas']), fus['blocks'])
                f2 = lb.apply_stacking(member('V2'), stacks[farm])
                v2_mae.setdefault('FUS', {})[farm] = float(np.abs(f2 - y2).mean())
                best_member = min(fus['members'], key=lambda m: v2_mae[m][farm])
                ref = (L['LUG'] if best_member == 'LUG' else loss(B.preds(farm, best_member, 'V2'), y2))
                decisions['part_b'][farm] = dict(
                    DA_vs_TCN=paired(loss(B.preds(farm, 'DATCNGRU', 'V2'), y2), loss(B.preds(farm, 'TCNGRU', 'V2'), y2), i2),
                    FUS_vs_best_member=dict(member=best_member, stats=paired(np.abs(f2 - y2), ref, i2)),
                    DA_lambda=B.dev(farm)['arms']['DATCNGRU']['selected'], stacking=stacks[farm])
        decisions['part_a']['gates'] = gate_decisions(ani_lug, ani_pcl, **protocol['part_a']['gates'])
        decisions['v2_mae'] = v2_mae
        if B is not None:
            decisions['part_b']['selection'] = select_base(v2_mae, **protocol['part_b']['selection'])
        decisions['note'] = 'V2 was used before for A1/N early stopping: development-stage evidence only (R1-S3)'
        save(out / 'dev_decisions.json', decisions)          # written BEFORE any test statistic
        beat(out, 'dev decisions written: ' + decisions['part_a']['gates']['outcome'])

        tests = {}
        for farm, spec in farms(x1p).items():
            for name, item in spec['items'].items():
                k = A.keys(farm, name)
                y, issue = k['truth'], k['issue']
                P = {arm: A.preds(farm, arm, name) for arm in protocol['part_a']['arms']}
                if B is not None:
                    P.update({arm: B.preds(farm, arm, name) for arm in protocol['part_b']['arms']})
                    members = np.stack([np.mean(P[m], 0) for m in protocol['part_b']['fusion']['members']], 1)
                    P['FUS'] = [lb.apply_stacking(members, stacks[farm])]
                if not (args.skip_frozen_n or smoke):
                    _, npaths = x1.paths(Path(args.logs), item)
                    P['N_frozen'] = [np.load(p) for p in npaths]
                    if len(P['N_frozen'][0]) != len(y):
                        raise ValueError('Frozen N cohort differs (smoke runs must use --skip-frozen-n)')
                ref = loss(P['LUG'], y)
                tests[name] = dict(
                    metrics={arm: metrics(p, y, spec['rated']) for arm, p in P.items()},
                    vs_LUG={arm: paired(loss(p, y), ref, issue) for arm, p in P.items() if arm != 'LUG'},
                    ANI_vs_PCL=paired(loss(P['LUG_ANI'], y), loss(P['LUG_PCL'], y), issue))
                if 'N_frozen' in P:
                    tests[name]['LUG_vs_N_frozen'] = paired(ref, loss(P['N_frozen'], y), issue)
                beat(out, f'test {name} done')
        save(out / 'results.json', dict(exploratory_only=True, dev_decisions_sha256=digest(out / 'dev_decisions.json'),
                                        tests=tests))
        save(out / 'completed.json', dict(status='COMPLETED', results_sha256=digest(out / 'results.json'),
                                          dev_decisions_sha256=digest(out / 'dev_decisions.json')))
    except Exception as exc:
        fail(out, exc)
        raise
    return out


# ----------------------------------------------------------------------------- entry point
def run_smoke(args):
    """Tiny real slices through A -> B -> report in ONE fresh directory (budget['smoke']); nothing is reported."""
    out, protocol, x1p = start_run(args, 'SMOKE')
    run_a(args, out, protocol, x1p)
    run_b(args, out, protocol, x1p)
    args.a_results, args.b_results, args.skip_frozen_n = out, out, True
    run_report(args, out, protocol, x1p)
    save(out / 'smoke_passed.json', dict(status='PASS', interpret_results=False, commit=git_identity()[0]))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['smoke', 'a', 'b', 'report'], required=True)
    p.add_argument('--logs', type=Path, required=True, help='<data root>/outputs/logs')
    p.add_argument('--output', type=Path, required=True, help='fresh directory (must not exist)')
    p.add_argument('--protocol-sha256', required=True, help='sha256 of configs/x2_protocol.json')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--gpu-fraction', type=float, default=.35)
    p.add_argument('--a-results', type=Path, help='report: finished part A run directory')
    p.add_argument('--b-results', type=Path, help='report: finished part B run directory (optional)')
    p.add_argument('--skip-frozen-n', action='store_true', help='report: no comparison with the frozen N predictions')
    p.add_argument('--smoke-result', type=Path, help='a/b: passed smoke run directory of the same commit')
    p.add_argument('--allow-dirty', action='store_true', help='tests only; formal runs need a clean worktree')
    args = p.parse_args(argv)
    if args.mode == 'smoke':
        run_smoke(args)
    elif args.mode == 'a':
        run_a(args)
    elif args.mode == 'b':
        run_b(args)
    else:
        if args.a_results is None:
            p.error('report needs --a-results')
        run_report(args)


if __name__ == '__main__':
    main()
