"""X4: two-stage transfer adaptation of Chronos-2 for cold start (A), stronger in-farm fine-tuning (B), and the CPU
feasibility diagnostic D0 for event-conditioned transfer value (P1).  Plan: docs/STAGE_PLAN.md (X4, approved after R2
and the user ruling, eab3276); decisions docs/review/DECISIONS.md stage X4.

Author: Claude (2026-11-20).  GPT reviews, Cursor runs.  X3's formal run f8af1b6 is a hash-pinned reference (Q1
selection, Q2-FT20k windows and predictions, Q4 arms and meta); nothing of X3 is rerun.  Every mode writes
<base>/<mode> (smoke: <base>/smoke_<mode>) and leaves failed.json on any error; no retries.

Modes:  a       GPU  per target: stage 1 on the two source farms (L_1X source pool, 20k windows per farm, lr grid,
                     selection on the target's k=1 early-stopping set), stage 2 target adaptation for k = 1 and 3 (step 0
                     is a candidate), direct target adaptation C_D1, V1 predictions of C_T0 / C_T1 / C_T3 / C_D1
        b       GPU  per farm: fine-tuning from the base model on X3's Q2-FT20k windows, lr grid; V1 selection among
                     {X3 Q2-FT20k, new runs} -> FT-B; FT-B predictions on V1, V2 and the test sets
        d0      CPU  LightGBM proxy of event-group transfer value; groups frozen on the first block (user ruling R2-B1);
                     every target training and source label strictly before the block's evaluation cut (R2-B2)
        report  CPU  development decisions A / B1 / B2 (Bonferroni z) and the D0 outcome first, then exploratory tests
"""
from __future__ import annotations

import argparse
import datetime as dt
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
from scripts import run_x2 as x2  # noqa: E402
from scripts import run_x3 as x3  # noqa: E402

digest, read, save, checked = x1.digest, x1.read, x1.save, x1.checked
PROTOCOL = ROOT / 'configs/x4_protocol.json'
HOUR = x3.HOUR
FARMS = ('Kelmarsh', 'Penmanshiel', 'HoT')
VARIATES = x3.VARIATES
PER_WINDOW_EXCLUDE = ('mean', 'scale')


# ============================================================================ numpy core (tested locally)
def label_times_of(issue):
    """Hour-end label times issue+1 h .. issue+4 h ([n, 4], datetime64[ns]); the frozen packs use exactly these."""
    issue = np.asarray(issue).astype('datetime64[ns]')
    return issue[:, None] + np.arange(1, 5) * HOUR


def context_index(issues_by_name):
    """One window index over named contexts in a fixed order: context id, row, issue and label times."""
    names = list(issues_by_name)
    nid = np.concatenate([np.full(len(issues_by_name[n]), i) for i, n in enumerate(names)]).astype(np.int64)
    row = np.concatenate([np.arange(len(issues_by_name[n])) for n in names]).astype(np.int64)
    issue = np.concatenate([np.asarray(issues_by_name[n]).astype('datetime64[ns]') for n in names])
    return dict(names=names, nid=nid, row=row, issue=issue, times=label_times_of(issue))


def source_pool(index, cut):
    """L_kX source rule (run_x3.select_rows(rows, '1900-01-01', cut)): every h1-h4 label strictly before the cut."""
    return np.flatnonzero((index['times'] < np.datetime64(cut, 'ns')).all(1))


def target_split(index, deployment, k, frac):
    """Target k-month windows exactly as X3 Q4 (budget_mask + es_split): fit (labels < cut), es (issue >= cut)."""
    lo, hi = x3.months_before(deployment, k), np.datetime64(deployment, 'ns')
    m = np.flatnonzero(x3.budget_mask(dict(issue=index['issue'], times=index['times']), lo, hi))
    fit, es, cut = x3.es_split(index['issue'][m], index['times'][m], frac)
    return dict(fit=m[fit], es=m[es], cut=cut, lo=lo)


def rows_by_context(index, positions):
    """{context name: sorted rows} for index positions (array building is done per context)."""
    out = {}
    for p in np.sort(np.asarray(positions)):
        out.setdefault(index['names'][index['nid'][p]], []).append(int(index['row'][p]))
    return {n: np.asarray(r, np.int64) for n, r in out.items()}


def draw(n_avail, n, key):
    """Uniform draw without replacement, sorted; key is a fixed integer list (seed, farm codes, ...)."""
    return np.sort(np.random.default_rng(key).choice(n_avail, size=min(int(n), int(n_avail)), replace=False))


def budget_hours(issue, L, lo):
    """[n, L] True where an input hour lies before the observation budget start lo (masked to NaN)."""
    return x3.window_hours(issue, L) < np.datetime64(lo, 'ns').astype(np.int64)


def window_arrays(ctx, table, hist_src, idx, L, truth, lo=None, masked=True, chunk=2000):
    """Raw-unit fine-tuning/evaluation arrays (run_x3.training_arrays layout) for windows idx of a context.
    masked: upstream channels NaN on flagged hours (no fit-segment fill, as C_zs);  lo: every channel and the GFS
    history NaN before the observation budget start (target windows only).  Order: unmask -> budget mask."""
    if 'path' in hist_src:
        hist_src = x3.checked_npz(hist_src)
    idx = np.asarray(idx)
    target = np.empty((len(idx), L), np.float32)
    cov = np.empty((len(idx), len(x3.COVARIATE_NAMES), L), np.float32)
    mean, scale = np.asarray(ctx['mean'], float), np.asarray(ctx['scale'], float)
    for a in range(0, len(idx), chunk):
        part = idx[a:a + chunk]
        x = x3.context_window(ctx, table, part, L)
        if masked:
            x = x3.mask_flagged_upstream(x, mean, scale)
        raw = x.astype(float) * scale + mean
        hist = x3.gfs_history(ctx, hist_src, part, L)
        if lo is not None:
            out = budget_hours(np.asarray(ctx['issue'])[part], L, lo)
            raw[out] = np.nan
            hist[out] = np.nan
        target[a:a + len(part)] = raw[:, :, -1]
        cov[a:a + len(part)] = np.stack([raw[:, :, 0], raw[:, :, 19], hist] + [raw[:, :, 8 + k] for k in range(11)], 1)
    future = np.where(np.asarray(ctx['gF_mask'])[idx, 6:10] > 0, np.asarray(ctx['gF'])[idx, 6:10], np.nan)
    return dict(target=target, covariates=cov, gfs_future=future.astype(np.float32),
                truth=np.asarray(truth, np.float32), covariate_names=np.array(x3.COVARIATE_NAMES),
                issue=np.asarray(ctx['issue'])[idx], turbines=np.asarray(ctx['turbines'])[idx])


def concat_arrays(parts):
    keys = ('target', 'covariates', 'gfs_future', 'truth', 'issue', 'turbines')
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out['covariate_names'] = parts[0]['covariate_names']
    return out


def arrays_inputs(arr):
    """Chronos predict inputs from raw arrays, identical to run_x1.chronos_inputs: past target, past covariates (GFS
    history as 'gfs'), and the issue-time GFS forecast as the future 'gfs'."""
    names = [str(n) for n in arr['covariate_names']]
    t, c, f = (np.asarray(arr[k], float) for k in ('target', 'covariates', 'gfs_future'))
    return [dict(target=t[i], past_covariates={n: c[i, j] for j, n in enumerate(names)},
                 future_covariates={'gfs': f[i]}) for i in range(len(t))]


def subset_rows(ctx, positions):
    """Per-window fields of a context at positions (in that order); mean/scale and other constants unchanged."""
    n, pos = len(ctx['issue']), np.asarray(positions)
    return {k: (v[pos] if (np.ndim(v) and len(v) == n and k not in PER_WINDOW_EXCLUDE) else v) for k, v in ctx.items()}


def key_subset(keys, n_issues=None):
    """Smoke: keep the windows of the first n issue hours (whole issue groups); None = all."""
    if n_issues is None:
        return keys
    issue = np.asarray(keys['issue'])
    keep = np.isin(issue, np.unique(issue)[:int(n_issues)])
    return {k: np.asarray(v)[keep] for k, v in keys.items()}


def pick_lr(results, lrs):
    """Stage-1 / B selection over learning rates: lowest MAE; ties -> smaller (earlier listed) lr."""
    return min(results, key=lambda r: (round(r['mae'], 9), list(lrs).index(r['learning_rate'])))


def pick_b(candidates):
    """FT-B: lowest V1 MAE; ties -> the earlier candidate (X3 Q2-FT20k first, then the smaller lr)."""
    return min(range(len(candidates)), key=lambda i: (round(candidates[i]['v1_mae'], 9), i))


def eval_curve(issue, preds, truth, steps):
    """R1-S1 report for a selection set: MAE per evaluated step, per issue day, effective issue hours, best-vs-second."""
    issue = np.asarray(issue).astype('datetime64[ns]')
    err = np.abs(np.asarray(preds, float) - np.asarray(truth, float)[None])       # [E, n, 4]
    mae = err.mean((1, 2))
    days = issue.astype('datetime64[D]')
    daily = {str(d): err[:, days == d].mean((1, 2)).round(4).tolist() for d in np.unique(days)}
    order = np.sort(mae)
    return dict(steps=list(map(int, steps)), mae=mae.round(4).tolist(), daily=daily,
                effective_issue_hours=int(len(np.unique(issue))), windows=int(len(issue)),
                best_minus_second=float(order[1] - order[0]) if len(order) > 1 else None)


# ---------------------------------------------------------------------------- D0 core
D0_EVENT = ('gfs_future_mean', 'gfs_future_change', 'power_pu', 'up_missing')


def month_blocks(deployment, back=(5, 4, 3, 2)):
    """Calendar-month blocks [D-b months, D-(b-1) months) for b in back (default: the 4 months before D-1 month)."""
    return [(x3.months_before(deployment, b), x3.months_before(deployment, b - 1)) for b in back]


def event_features(X):
    """Issue-time knowable event state from the Q4 physical columns: mean issue-cycle GFS over h1-h4 (gF columns 6-9),
    GFS change from the issue hour (column 5) to h4, current power / rated (p_hist_23), upstream-missing flag."""
    col = {n: i for i, n in enumerate(x3.PHYS_NAMES)}
    g = np.asarray(X, float)[:, [col[f'gfs_{i}'] for i in range(13)]]
    with np.errstate(invalid='ignore'):
        fut = np.nanmean(g[:, 6:10], 1) if g.size else np.zeros(0)
    fut = np.where(np.isfinite(g[:, 6:10]).any(1), fut, np.nan)
    return np.column_stack([fut, g[:, 9] - g[:, 5], np.asarray(X, float)[:, col['p_hist_23']],
                            np.asarray(X, float)[:, col['up_missing']]])


def fit_standardizer(Z):
    mean, std = Z.mean(0), Z.std(0)
    return mean, np.where(std > 0, std, 1.0)


def kmeans(Z, k, seed, n_init=10, iters=300):
    """Lloyd's k-means with k-means++ starts (numpy only); best inertia over n_init; deterministic for a seed."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        c = [Z[rng.integers(len(Z))]]
        for _ in range(1, k):
            d = np.min(((Z[:, None, :] - np.asarray(c)[None]) ** 2).sum(2), 1)
            c.append(Z[rng.choice(len(Z), p=d / d.sum())] if d.sum() > 0 else Z[rng.integers(len(Z))])
        c = np.asarray(c, float)
        for _ in range(iters):
            lab = np.argmin(((Z[:, None, :] - c[None]) ** 2).sum(2), 1)
            new = np.array([Z[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = float(((Z - c[np.argmin(((Z[:, None, :] - c[None]) ** 2).sum(2), 1)]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, c)
    return best[1]


def assign(Z, mean, std, centers):
    z = (np.asarray(Z, float) - mean) / std
    return np.argmin(((z[:, None, :] - centers[None]) ** 2).sum(2), 1)


def spearman(a, b):
    """Rank correlation with average ranks; None when either vector is constant (undefined, R2-S3)."""
    def ranks(v):
        v = np.asarray(v, float)
        order = np.argsort(v, kind='mergesort')
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for u in np.unique(v):
            r[v == u] = r[v == u].mean()
        return r
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def top_groups(U_past, m=2):
    """Groups with the highest mean U over the earlier blocks; ties -> lower group id."""
    mean = np.asarray(U_past, float).mean(0)
    return sorted(range(len(mean)), key=lambda g: (-round(mean[g], 12), g))[:m]


def d0_farm(U, wins, cfg):
    """Per pseudo-target: condition 1 (mean adjacent-block Spearman), condition 2 (top-2 beat random)."""
    rho = [spearman(U[e], U[e + 1]) for e in range(len(U) - 1)]
    if any(r is None for r in rho):
        return dict(status='UNDETERMINED', reason='constant U vector (Spearman undefined)', spearman=rho, wins=wins)
    mean = float(np.mean(rho))
    return dict(status='DETERMINED', spearman=rho, spearman_mean=mean, cond1=bool(mean >= cfg['spearman_min']),
                wins=wins, cond2=bool(sum(wins) >= cfg['wins_min']))


def d0_outcome(farms, cfg):
    if any(f['status'] != 'DETERMINED' for f in farms.values()):
        return 'INCOMPLETE'
    c1 = sum(f['cond1'] for f in farms.values())
    c2 = sum(f['cond2'] for f in farms.values())
    return 'GO_P1' if c1 >= cfg['farms_min'] and c2 >= cfg['farms_min'] else 'STOP_P1'


# ============================================================================ Chronos worker (tsfm env, GPU)
def median_predict(pipe, inputs, L, chunk=2048, batch=256):
    f = x3.chronos_predict_fn(pipe, L, False, x3.quantile_index(pipe))
    return np.concatenate([np.stack(f(inputs[i:i + chunk], batch)) for i in range(0, len(inputs), chunk)])


def checked_model(init):
    """Verify every file of a saved model against its freeze record; returns the model directory."""
    root = Path(init['root'])
    for rel, sha in init['model_sha256'].items():
        checked(root / rel, sha)
    return root / 'selected'


def eval_set(spec):
    e = spec['eval']
    if e['kind'] == 'arrays':
        arr = x3.checked_npz(e['npz'])
        return arrays_inputs(arr), np.asarray(arr['truth'], float)
    ctx = x3.checked_npz(e['context'])
    hist = x3.checked_npz(e['context']['history'])
    table = x3.HourTable(x3.checked_npz(e['grid']))
    inputs, _ = x3.build_inputs(ctx, table, hist, np.arange(len(ctx['issue'])), int(spec['context_length']),
                                bool(e.get('masked', False)))
    return inputs, np.asarray(x3.checked_npz(e['keys'])['truth'], float)


def worker_fit(spec, out):
    """Fine-tune (from the base model or a verified checkpoint) with the frozen preflight candidate; median MAE on the
    selection set every eval_every steps (and at step 0 when registered); lowest MAE kept, ties -> earlier step."""
    import torch
    from transformers import TrainerCallback
    from chronos import Chronos2Pipeline
    model_dir = checked_model(spec['init']) if spec.get('init') else None
    pipe = x3.load_pipe(spec, model_dir=model_dir)
    rows = x3.training_series(x3.checked_npz(spec['train']))
    if len(rows) != int(spec['budget']['train_windows']):
        raise ValueError('Training windows differ from the registered budget')
    inputs, truth = eval_set(spec)
    L, every, steps = int(spec['context_length']), int(spec['eval_every']), int(spec['num_steps'])
    selected = out / 'selected'
    state = dict(best=float('inf'), rows=[], preds=[], trainer={})

    def evaluate(p, step):
        with torch.inference_mode():
            q = median_predict(p, inputs, L)
        mae = float(np.abs(q[:, 1] - truth).mean())
        if not np.isfinite(mae):
            raise ValueError('Non-finite selection MAE')
        state['rows'].append(dict(step=int(step), mae=mae))
        state['preds'].append(q[:, 1].astype(np.float32))
        if mae < state['best']:
            state['best'] = mae
            p.save_pretrained(selected)
            save(out / 'selection.json', dict(step=int(step), mae=mae))
        save(out / 'eval_history.json', state['rows'])

    class Select(TrainerCallback):
        def on_train_begin(self, args, st, control, **kwargs):
            state['trainer'] = {k: getattr(args, k, None) for k in (
                'per_device_train_batch_size', 'gradient_accumulation_steps', 'max_steps', 'learning_rate',
                'n_gpu', 'world_size', 'seed', 'bf16', 'fp16')}
            state['trainer'] = {k: (v if isinstance(v, (int, float, bool, str)) or v is None else str(v))
                                for k, v in state['trainer'].items()}
            c = spec['candidate']
            if (state['trainer']['gradient_accumulation_steps'] != c['accumulation']
                    or state['trainer']['per_device_train_batch_size'] != c['microbatch']):
                raise ValueError('Trainer batch settings differ from the frozen preflight candidate')
            return control

        def on_step_end(self, args, st, control, model=None, **kwargs):
            if st.global_step % every:
                return control
            was = model.training
            model.eval()
            evaluate(Chronos2Pipeline(model=model), st.global_step)
            model.train(was)
            return control

    if spec['eval_step0']:
        pipe.model.eval()
        evaluate(pipe, 0)
    x3.run_fit(pipe, rows, out / 'fit', spec, spec['candidate'], steps, callbacks=[Select()])
    expected = steps // every + (1 if spec['eval_step0'] else 0)
    if len(state['rows']) != expected or not selected.is_dir():
        raise ValueError('Selection did not run at every registered evaluation')
    np.save(out / 'eval_preds.npy', np.stack(state['preds']))
    hashes = {str(p.relative_to(out)): digest(p) for p in selected.rglob('*') if p.is_file()}
    save(out / 'selection_freeze.json', dict(model_sha256=hashes, selected=read(out / 'selection.json'),
         history=state['rows'], candidate=spec['candidate'], learning_rate=spec['learning_rate'], steps=steps,
         eval_every=every, eval_step0=bool(spec['eval_step0']), init=spec.get('init'), trainer=state['trainer'],
         train_windows=len(rows), budget=spec['budget'], selection='median MAE on the registered selection set'))


def worker_predict(spec, out):
    """Median/quantile predictions of each registered model on each context (single series)."""
    import torch
    tables = {}
    for m in spec['models']:
        pipe = x3.load_pipe(spec, model_dir=checked_model(m['init']) if m['init'] else None)
        for rec in spec['contexts']:
            ctx, hist = x3.checked_npz(rec), x3.checked_npz(rec['history'])
            if rec['grid']['sha256'] not in tables:
                tables[rec['grid']['sha256']] = x3.HourTable(x3.checked_npz(rec['grid']))
            pred, _ = x3.predict_context(pipe, ctx, tables[rec['grid']['sha256']], hist, int(spec['context_length']),
                                         'single', masked=bool(spec['masked']))
            with (out / f"{m['name']}_{rec['name']}.npy").open('xb') as fh:
                np.save(fh, pred)
        del pipe
        torch.cuda.empty_cache()


def worker_main(args):
    spec = read(checked(args.spec, args.spec_sha))
    out = Path(spec['output'])
    {'preflight': x3.worker_preflight, 'fit': worker_fit, 'predict': worker_predict}[spec['task']](spec, out)


# ============================================================================ parent side (timedart env)
def run_dir(base, mode, smoke):
    return Path(base) / (('smoke_' if smoke else '') + mode)


def start_run(args, mode):
    """Same gate as X3: protocol SHA, clean worktree, formal modes need a completed smoke report of this commit."""
    checked(PROTOCOL, args.protocol_sha256)
    commit, dirty = x2.git_identity()
    if not args.allow_dirty and (commit is None or dirty):
        raise ValueError('Run from a clean worktree of the reviewed commit')
    if not args.smoke:
        smoke = run_dir(args.base, 'report', True)
        done = read(smoke / 'completed.json') if (smoke / 'completed.json').exists() else {}
        launch = read(smoke / 'launch.json') if (smoke / 'launch.json').exists() else {}
        if (done.get('status') != 'COMPLETED' or launch.get('commit') != commit or not launch.get('smoke')
                or launch.get('mode') != 'report' or launch.get('script_sha256') != digest(__file__)
                or launch.get('protocol_sha256') != args.protocol_sha256 or (smoke / 'failed.json').exists()):
            raise ValueError('Formal X4 modes need a completed smoke chain of this commit')
        for name, field in (('registry.json', 'registry_sha256'), ('results.json', 'results_sha256'),
                            ('dev_decisions.json', 'dev_decisions_sha256')):
            checked(smoke / name, done[field])
    out = run_dir(args.base, mode, args.smoke)
    out.mkdir(parents=True, exist_ok=False)
    (out / '.gitignore').write_text('*.npy\n*.npz\nselected/\nfit/\n', encoding='utf-8')
    save(out / 'launch.json', dict(mode=mode, smoke=bool(args.smoke), commit=commit, script_sha256=digest(__file__),
         protocol_sha256=args.protocol_sha256, pid=os.getpid(), started_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    return out, read(PROTOCOL), read(ROOT / 'configs/x1_protocol.json')


UPSTREAM_FILES = {'a': (('a_meta.json', 'meta_sha256'),), 'b': (('b_meta.json', 'meta_sha256'),),
                  'd0': (('d0.json', 'd0_sha256'),)}


def upstream(args, mode):
    """A finished X4 run of the same commit, script, protocol and smoke flag; receipts and metadata hash-checked."""
    root = run_dir(args.base, mode, args.smoke)
    if not (root / 'completed.json').exists():
        raise ValueError(f'Upstream X4 {mode} not completed')
    x3.checked_launch(args, root)
    done = read(root / 'completed.json')
    if done.get('status') != 'COMPLETED' or (root / 'failed.json').exists():
        raise ValueError(f'Upstream {mode} is failed or not completed')
    registry = read(checked(root / 'registry.json', done['registry_sha256']))
    for name, field in UPSTREAM_FILES[mode]:
        checked(root / name, done[field])
    return dict(root=root, done=done, registry=registry)


class X3Run:
    """Hash-pinned X3 formal run (f8af1b6): completed receipts, registries, Q1 selection, Q4 meta, Q2 specs."""

    def __init__(self, logs, protocol):
        r = protocol['references']['x3']
        self.root, self.cfg, self.stage = Path(logs) / r['dir'], r, {}
        for mode, sha in r['completed_sha256'].items():
            d = self.root / mode
            done = read(checked(d / 'completed.json', sha))
            if done.get('status') != 'COMPLETED' or (d / 'failed.json').exists():
                raise ValueError('X3 reference stage not completed: ' + mode)
            launch = read(d / 'launch.json')
            if launch['commit'] != r['commit'] or launch.get('smoke'):
                raise ValueError('X3 reference identity differs: ' + mode)
            self.stage[mode] = dict(root=d, done=done, registry=read(checked(d / 'registry.json', done['registry_sha256'])))
        q1, q4 = self.stage['q1'], self.stage['q4']
        self.selection = read(checked(q1['root'] / 'selection.json', q1['done']['selection_sha256']))
        self.q4_meta = read(checked(q4['root'] / 'q4_meta.json', q4['done']['meta_sha256']))

    def array(self, mode, rel):
        st = self.stage[mode]
        return np.load(checked(st['root'] / rel, st['registry'][rel]))

    def q2_spec(self, farm):
        return read(checked(self.root / 'q2' / farm / 'fit_run' / 'q2_spec.json', self.cfg['q2_spec_sha256'][farm]))


def call_worker(args, spec, out, name):
    path = out / (name + '_spec.json')
    save(path, spec)
    subprocess.run([args.chronos_python, '-u', str(Path(__file__).resolve()), '--mode', 'worker',
                    '--spec', str(path), '--spec-sha', digest(path)], check=True)


def put_npz(store, rel, arrays):
    path = store.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    store.registry[str(rel)] = digest(path)
    return dict(path=str(path), sha256=store.registry[str(rel)])


def bind_contexts(args, refs, farm, spec, names, out, cache):
    """X3-HS binding of every used context (full frozen pack, every window and hour) + label-time check; returns the
    pack truth (kW) per context, a hash-pinned history record per context and the binding report."""
    srcs, grid = x3.history_sources(args, refs, farm, names)
    nwp = x3.scaler(args, farm)[2:]
    packs = x3.frozen_packs(args, spec, names)
    truth, hist, report = {}, {}, {}
    for n in names:
        rec = refs.context(n)
        report[n] = x3.bind_source(packs[n], srcs[n], grid, *nwp, rec)
        times = np.asarray(packs[n]['times']).astype('datetime64[ns]')
        if not np.array_equal(times, label_times_of(packs[n]['issue'])):
            raise ValueError('Pack label times are not issue + 1..4 h: ' + n)
        truth[n] = np.asarray(packs[n]['truth'], np.float32)
        hist[n] = x3.save_source(out, farm, srcs[n], cache)
    del packs
    return truth, hist, report


def register_sources(out, store, cache):
    for rec in cache.values():
        store.registry[str(Path(rec['path']).relative_to(out))] = rec['sha256']


def subset_context(store, rel, rec, positions, name, history, grid):
    ctx = x1.load_pack(checked(Path(rec['path']), rec['sha256']))
    out = put_npz(store, rel, subset_rows(ctx, positions))
    return dict(out, name=name, history=history, grid=grid)


def worker_spec(protocol, args, task, out, **kw):
    return dict(x3.worker_base(protocol, args, task, out), **kw)


def preflight(args, protocol, cfg, out, trains, L, lr):
    """X3-Q2fix rule: first candidate (full only) passing on every farm, one fresh process per (candidate, farm)."""
    candidates = x3.q2_candidates(cfg)

    def probe(cand, farm):
        d = out / farm / f"pre_{cand['name']}"
        d.mkdir(parents=True)
        call_worker(args, worker_spec(protocol, args, 'preflight', d, candidate=cand, train=trains[farm],
                    preflight_windows=cfg['preflight_windows_smoke' if args.smoke else 'preflight_windows'],
                    preflight_steps=cfg['preflight_steps_smoke' if args.smoke else 'preflight_steps'],
                    context_length=L, learning_rate=lr, seed=cfg['seed']), d, 'preflight')
        return read(d / 'preflight.json')
    chosen, tried = x3.preflight_search(candidates, list(trains), probe)
    save(out / 'preflight_decision.json', dict(chosen=chosen, tried=tried, candidates=candidates))
    if chosen is None:
        raise ValueError('Preflight failed for every registered setting (technical, not a quality result)')
    return chosen


def fit_run(args, protocol, cfg, store, d, **kw):
    """One registered fine-tune in a fresh worker; registers its freeze, history and selection predictions."""
    d.mkdir(parents=True)
    call_worker(args, worker_spec(protocol, args, 'fit', d, seed=cfg['seed'], **kw), d, 'fit')
    rel = d.relative_to(store.root)
    for name in ('selection_freeze.json', 'eval_history.json', 'eval_preds.npy'):
        store.registry[str(rel / name)] = digest(d / name)
    return read(d / 'selection_freeze.json')


def mode_a(args, protocol, x1p, out):
    refs, x3r, store, cfg = x3.Refs(args.logs, protocol), X3Run(args.logs, protocol), x2.Store(out), protocol['a']
    farms, L, smoke = x2.farms(x1p), int(cfg['context_length']), args.smoke
    steps1, every1 = cfg['stage1_steps_smoke' if smoke else 'stage1_steps'], cfg['stage1_eval_smoke' if smoke else 'stage1_eval']
    steps2, every2 = cfg['stage2_steps_smoke' if smoke else 'stage2_steps'], cfg['stage2_eval_smoke' if smoke else 'stage2_eval']
    names = {f: [f + '_train', f + '_validation'] + list(farms[f]['items']) for f in FARMS}
    index, recs = {}, {}
    for f in FARMS:
        recs[f] = {n: refs.context(n) for n in names[f]}
        issues = {}
        for n, r in recs[f].items():
            with np.load(checked(Path(r['path']), r['sha256']), allow_pickle=False) as z:
                issues[n] = z['issue']
        index[f] = context_index(issues)
    # --- window plan and binding to the frozen L_kX sets (q4_meta)
    plan = {}
    for T in FARMS:
        D, arms = cfg['deployment'][T], x3r.q4_meta[T]['arms']
        p = dict(split={}, pool={}, check={})
        for k in cfg['k_months']:
            sp = target_split(index[T], D, k, cfg['es_fraction'])
            pools = {S: source_pool(index[S], sp['cut']) for S in FARMS if S != T}
            ref = arms[f'L_{k}X']
            got = dict(target_windows=int(len(sp['fit'])), es_windows=int(len(sp['es'])), es_cut=str(sp['cut']),
                       fit_windows=int(sum(len(v) for v in pools.values()) + len(sp['fit'])))
            if any(got[key] != ref[key] for key in got):
                raise ValueError(f'Target/source windows differ from frozen L_{k}X: {T} {got} {ref}')
            p['split'][k], p['pool'][k], p['check'][k] = sp, pools, got
        n_src = cfg['source_windows_smoke' if smoke else 'source_windows']
        p['source'] = {S: p['pool'][1][S][draw(len(p['pool'][1][S]), n_src,
                                               [cfg['seed'], x3.FARM_CODE[T], x3.FARM_CODE[S]])] for S in p['pool'][1]}
        keys = key_subset(refs.keys(T, 'V1'), protocol['smoke']['issues'] if smoke else None)
        earliest = np.asarray(keys['issue']).astype('datetime64[ns]').min() - (L - 1) * HOUR
        if earliest < x3.months_before(D, 1):
            raise ValueError('V1 context reaches before the k=1 observation budget: ' + T)
        p['keys'] = keys
        plan[T] = p
    # --- arrays, built farm by farm (each farm's contexts loaded once)
    cache, binding, recs_out, budget = {}, {}, {T: {} for T in FARMS}, {T: {} for T in FARMS}
    for f in FARMS:
        need = set()
        for T in FARMS:
            if T != f:
                need |= set(rows_by_context(index[f], plan[T]['source'][f]))
        for k in cfg['k_months']:
            sp = plan[f]['split'][k]
            need |= set(rows_by_context(index[f], np.r_[sp['fit'], sp['es']]))
        need.add(f + '_validation')
        need = [n for n in names[f] if n in need]
        truth, hist, binding[f] = bind_contexts(args, refs, f, farms[f], need, out, cache)
        g = refs.grid(f)
        table = x3.HourTable(x1.load_pack(checked(Path(g['path']), g['sha256'])))
        ctxs = {n: x1.load_pack(checked(Path(recs[f][n]['path']), recs[f][n]['sha256'])) for n in need}
        for T in FARMS:
            if T != f:
                parts = [window_arrays(ctxs[n], table, hist[n], rows, L, truth[n][rows])
                         for n, rows in rows_by_context(index[f], plan[T]['source'][f]).items()]
                recs_out[T][f'src_{f}'] = put_npz(store, f'{T}/src_{f}.npz', concat_arrays(parts))
        for k in cfg['k_months']:
            sp = plan[f]['split'][k]
            for part in ('fit', 'es'):
                parts = [window_arrays(ctxs[n], table, hist[n], rows, L, truth[n][rows], lo=sp['lo'])
                         for n, rows in rows_by_context(index[f], sp[part]).items()]
                arr = concat_arrays(parts)
                budget[f][f'{part}{k}'] = dict(windows=int(len(arr['truth'])), labels=int(arr['truth'].size),
                                              finite_target_history_fraction=float(np.isfinite(arr['target']).mean()))
                recs_out[f][f'{part}{k}'] = put_npz(store, f'{f}/{part}_k{k}.npz', arr)
        vname = f + '_validation'
        keys = plan[f]['keys']
        pos = x3.key_positions(ctxs[vname]['issue'], ctxs[vname]['turbines'], keys)
        recs_out[f]['v1'] = subset_context(store, f'{f}/v1_context.npz', recs[f][vname], pos, 'V1', hist[vname], g)
        recs_out[f]['v1_keys'] = put_npz(store, f'{f}/v1_keys.npz', {k: np.asarray(v) for k, v in keys.items()})
        del ctxs, table
        print('[X4] A arrays', f, flush=True)
    trains = {}
    for T in FARMS:
        parts = [x1.load_pack(checked(Path(recs_out[T][f'src_{S}']['path']), recs_out[T][f'src_{S}']['sha256']))
                 for S in FARMS if S != T]
        trains[T] = put_npz(store, f'{T}/stage1_train.npz', concat_arrays(parts))
    chosen = preflight(args, protocol, cfg, out, trains, L, cfg['stage1_lrs'][0])
    meta = dict(plan={T: dict(check={str(k): v for k, v in plan[T]['check'].items()},
                               source_pool={S: int(len(v)) for S, v in plan[T]['pool'][1].items()},
                               source_drawn={S: int(len(v)) for S, v in plan[T]['source'].items()},
                               v1_windows=int(len(plan[T]['keys']['issue']))) for T in FARMS},
                observation_budget=budget, history_binding=binding, setting=chosen, targets={})
    for T in FARMS:
        r, m = recs_out[T], dict(stage1=[], curves={})
        n_train = int(sum(len(v) for v in plan[T]['source'].values()))
        base = dict(candidate=chosen, context_length=L)
        for i, lr in enumerate(cfg['stage1_lrs']):
            fr = fit_run(args, protocol, cfg, store, out / T / f'stage1_lr{i}', train=trains[T],
                         eval=dict(kind='arrays', npz=r['es1']), init=None, learning_rate=lr, num_steps=steps1,
                         eval_every=every1, eval_step0=False, budget=dict(train_windows=n_train), **base)
            m['stage1'].append(dict(learning_rate=lr, mae=fr['selected']['mae'], step=fr['selected']['step'],
                                    root=str(out / T / f'stage1_lr{i}'), model_sha256=fr['model_sha256']))
        s1 = pick_lr(m['stage1'], cfg['stage1_lrs'])
        init1 = dict(root=s1['root'], model_sha256=s1['model_sha256'])
        m['stage1_selected'] = dict(learning_rate=s1['learning_rate'], step=s1['step'], mae=s1['mae'])
        runs = {'C_T0': init1}
        for k, arm in ((1, 'C_T1'), (3, 'C_T3')):
            d = out / T / f'stage2_k{k}'
            fr = fit_run(args, protocol, cfg, store, d, train=r[f'fit{k}'], eval=dict(kind='arrays', npz=r[f'es{k}']),
                         init=init1, learning_rate=s1['learning_rate'], num_steps=steps2, eval_every=every2,
                         eval_step0=True, budget=dict(train_windows=int(len(plan[T]['split'][k]['fit']))), **base)
            runs[arm] = dict(root=str(d), model_sha256=fr['model_sha256'])
            m[arm] = fr['selected']
        d = out / T / 'direct_k1'
        fr = fit_run(args, protocol, cfg, store, d, train=r['fit1'], eval=dict(kind='arrays', npz=r['es1']),
                     init=None, learning_rate=s1['learning_rate'], num_steps=steps2, eval_every=every2, eval_step0=True,
                     budget=dict(train_windows=int(len(plan[T]['split'][1]['fit']))), **base)
        runs['C_D1'] = dict(root=str(d), model_sha256=fr['model_sha256'])
        m['C_D1'] = fr['selected']
        m['C_D1_rule'] = 'learning rate shared with the stage-1 selection of C_T1 (R2-S2); target adaptation budget only'
        for name, d, es in [(f'stage1_lr{i}', out / T / f'stage1_lr{i}', 'es1') for i in range(len(cfg['stage1_lrs']))] + \
                [('stage2_k1', out / T / 'stage2_k1', 'es1'), ('stage2_k3', out / T / 'stage2_k3', 'es3'),
                 ('direct_k1', out / T / 'direct_k1', 'es1')]:
            arr = x1.load_pack(checked(Path(r[es]['path']), r[es]['sha256']))
            hist_rows = read(d / 'eval_history.json')
            m['curves'][name] = eval_curve(arr['issue'], np.load(d / 'eval_preds.npy'), arr['truth'],
                                           [h['step'] for h in hist_rows])
        pd = out / T / 'pred'
        pd.mkdir()
        call_worker(args, worker_spec(protocol, args, 'predict', pd, context_length=L, masked=True,
                    models=[dict(name=a, init=runs[a]) for a in ('C_T0', 'C_T1', 'C_T3', 'C_D1')],
                    contexts=[r['v1']]), pd, 'predict')
        for a in ('C_T0', 'C_T1', 'C_T3', 'C_D1'):
            store.registry[f'{T}/pred/{a}_V1.npy'] = digest(pd / f'{a}_V1.npy')
        meta['targets'][T] = m
        print('[X4] A done', T, m['stage1_selected'], flush=True)
    register_sources(out, store, cache)
    save(out / 'a_meta.json', meta)
    x3.finish(out, store, dict(meta_sha256=digest(out / 'a_meta.json')))


def mode_b(args, protocol, x1p, out):
    refs, x3r, store, cfg = x3.Refs(args.logs, protocol), X3Run(args.logs, protocol), x2.Store(out), protocol['b']
    farms, smoke = x2.farms(x1p), args.smoke
    steps, every = cfg['steps_smoke' if smoke else 'steps'], cfg['eval_smoke' if smoke else 'eval']
    trains, ctx_plan, cache, binding = {}, {}, {}, {}
    Ls = {f: int(x3r.selection['selection'][f]['context_length']) for f in FARMS}
    if len(set(Ls.values())) != 1:
        raise ValueError('B preflight assumes one context length (X3 Q1 selected 1024 h everywhere)')
    for f in FARMS:
        spec = x3r.q2_spec(f)
        if int(spec['context_length']) != Ls[f]:
            raise ValueError('X3 Q2 context differs from the Q1 selection: ' + f)
        tr = dict(path=spec['train']['path'], sha256=spec['train']['sha256'])
        if smoke:
            arr = x3.checked_npz(tr)
            n = cfg['train_windows_smoke']
            tr = put_npz(store, f'{f}/train_smoke.npz', {k: (v[:n] if k != 'covariate_names' else v) for k, v in arr.items()})
        trains[f] = tr
        splits = ['V1', 'V2'] + list(farms[f]['items'])
        cnames = {s: (f + '_validation' if s in ('V1', 'V2') else s) for s in splits}
        truth, hist, binding[f] = bind_contexts(args, refs, f, farms[f], sorted(set(cnames.values())), out, cache)
        g = refs.grid(f)
        p = {}
        for s in splits:
            full = refs.context(cnames[s])
            keys = key_subset(refs.keys(f, s), protocol['smoke']['issues'] if smoke else None)
            with np.load(checked(Path(full['path']), full['sha256']), allow_pickle=False) as z:
                pos = x3.key_positions(z['issue'], z['turbines'], keys)
            p[s] = dict(keys=keys, full_positions=pos,
                        ctx=subset_context(store, f'{f}/{s}_context.npz', full, pos, s, hist[cnames[s]], g),
                        keys_rec=put_npz(store, f'{f}/{s}_keys.npz', {k: np.asarray(v) for k, v in keys.items()}))
        ctx_plan[f] = p
    L = Ls['HoT']
    chosen = preflight(args, protocol, cfg, out, trains, L, cfg['lrs'][0])
    n_train = {}
    for f in FARMS:
        with np.load(checked(Path(trains[f]['path']), trains[f]['sha256']), allow_pickle=False) as z:
            n_train[f] = int(len(z['truth']))
    meta = dict(history_binding=binding, setting=chosen, farms={})
    for f in FARMS:
        p, m = ctx_plan[f], dict(candidates=[])
        x3v = x3r.array('q2', f'{f}/fit_run/validation_q2.npy')
        v1 = x3v[p['V1']['full_positions']]
        m['candidates'].append(dict(name='X3_Q2_FT20k', learning_rate=1e-6,
                                    v1_mae=float(np.abs(v1[:, 1] - p['V1']['keys']['truth']).mean())))
        runs = {}
        for i, lr in enumerate(cfg['lrs']):
            d = out / f / f'lr{i}'
            fr = fit_run(args, protocol, cfg, store, d, train=trains[f], init=None, learning_rate=lr, num_steps=steps,
                         eval=dict(kind='context', context=p['V1']['ctx'], grid=p['V1']['ctx']['grid'],
                                   keys=p['V1']['keys_rec'], masked=False),
                         eval_every=every, eval_step0=False, candidate=chosen, context_length=L,
                         budget=dict(train_windows=n_train[f]))
            runs[f'lr{i}'] = dict(root=str(d), model_sha256=fr['model_sha256'])
            pd = d / 'pred_v1'
            pd.mkdir()
            call_worker(args, worker_spec(protocol, args, 'predict', pd, context_length=L, masked=False,
                        models=[dict(name='m', init=runs[f'lr{i}'])], contexts=[p['V1']['ctx']]), pd, 'predict')
            store.registry[str((pd / 'm_V1.npy').relative_to(out))] = digest(pd / 'm_V1.npy')
            pv = np.load(pd / 'm_V1.npy')
            m['candidates'].append(dict(name=f'lr{i}', learning_rate=lr, step=fr['selected']['step'],
                                        v1_mae=float(np.abs(pv[:, 1] - p['V1']['keys']['truth']).mean())))
        best = pick_b(m['candidates'])
        m['selected'] = m['candidates'][best]
        splits = ['V1', 'V2'] + list(farms[f]['items'])
        if best == 0:
            for s in splits:
                rel = (f'{f}/fit_run/validation_q2.npy' if s in ('V1', 'V2') else f'{f}/fit_run/{s}_q2.npy')
                store.put(f'{f}/FTB_{s}.npy', x3r.array('q2', rel)[p[s]['full_positions']])
        else:                                       # V1 already predicted by the selected run; the rest now
            name = m['selected']['name']
            store.put(f'{f}/FTB_V1.npy', np.load(out / f / name / 'pred_v1' / 'm_V1.npy'))
            pd = out / f / 'pred_ftb'
            pd.mkdir()
            call_worker(args, worker_spec(protocol, args, 'predict', pd, context_length=L, masked=False,
                        models=[dict(name='FTB', init=runs[name])], contexts=[p[s]['ctx'] for s in splits[1:]]),
                        pd, 'predict')
            for s in splits[1:]:
                store.put(f'{f}/FTB_{s}.npy', np.load(pd / f'FTB_{s}.npy'))
        meta['farms'][f] = m
        print('[X4] B done', f, m['selected'], flush=True)
    register_sources(out, store, cache)
    save(out / 'b_meta.json', meta)
    x3.finish(out, store, dict(meta_sha256=digest(out / 'b_meta.json')))


def proxy_mae(lgb, cfg, threads, trees, tr, ev):
    """LightGBM proxy (fixed trees, no early stopping, L1), one model per horizon; MAE in per-unit power."""
    pred = np.empty_like(ev['y'], dtype=float)
    for h in range(4):
        m = lgb.LGBMRegressor(objective='regression_l1', num_leaves=cfg['num_leaves'],
                              min_child_samples=cfg['min_child_samples'], learning_rate=cfg['learning_rate'],
                              n_estimators=trees, random_state=cfg['seed'], n_jobs=threads, verbosity=-1,
                              deterministic=True, force_col_wise=True)
        m.fit(tr['X'], tr['y'][:, h], categorical_feature=list(x3.CATEGORICAL))
        pred[:, h] = m.predict(ev['X'])
    return float(np.abs(pred - ev['y']).mean())


def stack_rows(rows, masks):
    return {k: np.concatenate([r[k][m] for r, m in zip(rows, masks)]) for k in ('X', 'y', 'issue', 'times')}


def d0_target(lgb, cfg, threads, T, tables, deadline, smoke):
    """One pseudo-target: 4 calendar-month blocks, frozen first-block groups, U(g, e), random control, top-2 test."""
    trees = cfg['trees_smoke' if smoke else 'trees']
    rows_t = tables[T]
    src_rows = [r for S in FARMS if S != T for r in tables[S]]
    blocks, U, R, rec = month_blocks(cfg['deployment'][T]), [], [], dict(blocks=[])
    frozen = None
    for b, (lo, hi) in enumerate(month_blocks(cfg['deployment'][T])):
        if time.monotonic() > deadline:
            raise TimeoutError('D0 exceeded its registered time budget')
        tgt = stack_rows(rows_t, [x3.budget_mask(dict(issue=r['issue'], times=r['times']), lo, hi) for r in rows_t])
        if not len(tgt['issue']):
            return dict(status='UNDETERMINED', reason=f'empty target block {b}', blocks=rec['blocks'])
        fit, es, cut = x3.es_split(tgt['issue'], tgt['times'], cfg['es_fraction'])
        if not fit.any() or not es.any():
            return dict(status='UNDETERMINED', reason=f'empty target block {b}', blocks=rec['blocks'])
        tr = {k: v[fit] for k, v in tgt.items()}
        ev = {k: v[es] for k, v in tgt.items()}
        assert (tr['times'].astype('datetime64[ns]') < cut).all()            # R2-B2: no training label at/after c
        pool = stack_rows(src_rows, [(np.asarray(r['times']).astype('datetime64[ns]') < cut).all(1) for r in src_rows])
        Z = event_features(pool['X'])
        ok = np.isfinite(Z).all(1)
        pool, Z = {k: v[ok] for k, v in pool.items()}, Z[ok]
        if not len(Z):
            return dict(status='UNDETERMINED', reason=f'empty finite source pool in block {b}',
                        dropped_nonfinite=int((~ok).sum()), blocks=rec['blocks'])
        if smoke and len(Z) > cfg['pool_smoke']:
            keep = draw(len(Z), cfg['pool_smoke'], [cfg['seed'], x3.FARM_CODE[T], b, 5])
            pool, Z = {k: v[keep] for k, v in pool.items()}, Z[keep]
        assert (pool['times'].astype('datetime64[ns]') < cut).all()
        if frozen is None:                                                     # R2-B1: first block only
            mean, std = fit_standardizer(Z)
            frozen = dict(mean=mean, std=std, centers=kmeans((Z - mean) / std, cfg['groups'], cfg['seed'],
                                                             cfg['kmeans_n_init'], cfg['kmeans_iters']))
        group = assign(Z, frozen['mean'], frozen['std'], frozen['centers'])
        counts = np.bincount(group, minlength=cfg['groups'])
        n = int(min(cfg['n_cap_smoke' if smoke else 'n_cap'], counts.min()))
        info = dict(block=[str(lo), str(hi)], cut=str(cut), target_fit=int(fit.sum()), target_eval=int(es.sum()),
                    target_dropped_cross_cut=int((~fit & ~es).sum()), pool=int(len(Z)), dropped_nonfinite=int((~ok).sum()),
                    group_counts=counts.tolist(), n=n)
        rec['blocks'].append(info)
        if n == 0:
            return dict(status='UNDETERMINED', reason=f'empty source group in block {b}', **rec)

        def with_src(sel):
            return {k: np.concatenate([tr[k], pool[k][sel]]) for k in ('X', 'y')}

        def timed(train):
            if time.monotonic() > deadline:
                raise TimeoutError('D0 exceeded its registered time budget')
            loss = proxy_mae(lgb, cfg, threads, trees, train, ev)
            if time.monotonic() > deadline:
                raise TimeoutError('D0 exceeded its registered time budget')
            if not np.isfinite(loss):
                raise ValueError('Non-finite D0 proxy MAE')
            return loss
        base = timed({k: tr[k] for k in ('X', 'y')})
        u = []
        for g in range(cfg['groups']):
            members = np.flatnonzero(group == g)
            sel = members[draw(len(members), n, [cfg['seed'], x3.FARM_CODE[T], b, g])]
            u.append(base - timed(with_src(sel)))
        rand = draw(len(Z), n, [cfg['seed'], x3.FARM_CODE[T], b, 99])
        r_loss = timed(with_src(rand))
        info.update(base_mae_pu=base, U=u, R=base - r_loss)
        if b:
            top = top_groups(U, 2)
            members = np.flatnonzero(np.isin(group, top))
            sel = members[draw(len(members), n, [cfg['seed'], x3.FARM_CODE[T], b, 77])]
            info.update(top2=top, top2_beats_random=bool(timed(with_src(sel)) < r_loss))
        U.append(u)
        R.append(base - r_loss)
    if time.monotonic() > deadline:
        raise TimeoutError('D0 exceeded its registered time budget')
    wins = [blk['top2_beats_random'] for blk in rec['blocks'][1:]]
    res = d0_farm(np.asarray(U), wins, cfg)
    res.update(rec, frozen=dict(mean=frozen['mean'].tolist(), std=frozen['std'].tolist(),
                                centers=frozen['centers'].tolist()), U=np.asarray(U).tolist(), R=R)
    return res


def mode_d0(args, protocol, x1p, out):
    import lightgbm as lgb
    if lgb.__version__ != '4.5.0':
        raise ValueError('D0 requires lightgbm 4.5.0 (timedart)')
    refs, cfg = x3.Refs(args.logs, protocol), protocol['d0']
    t0 = time.monotonic()
    tables, _, binding = x3.farm_tables(args, protocol, x1p, refs)
    prep = time.monotonic() - t0
    deadline = time.monotonic() + 60 * cfg['timeout_min']                    # proxy fits only (preparation reported)
    farms, outcome = {}, None
    try:
        for T in FARMS:
            farms[T] = d0_target(lgb, cfg, args.threads, T, tables, deadline, args.smoke)
            print('[X4] D0', T, farms[T].get('status'), farms[T].get('spearman_mean'), flush=True)
        outcome = d0_outcome(farms, cfg)
    except TimeoutError as exc:
        outcome, farms['timeout'] = 'INCOMPLETE', str(exc)
    wording = {'GO_P1': 'proxy reached the registered continuation threshold; a Chronos version may be registered',
               'STOP_P1': 'under this LightGBM proxy, six-group partition and budget the continuation threshold was '
                          'not reached (R2-S1); not evidence that Chronos or all event-value learning is infeasible',
               'INCOMPLETE': 'diagnostic incomplete or undeterminable (empty group, constant U or time budget); '
                             'not a negative result, no rerun (R2-S3)'}[outcome]
    save(out / 'd0.json', dict(outcome=outcome, wording=wording, farms=farms, preparation_seconds=prep,
                               fit_seconds=time.monotonic() - t0 - prep, history_binding=binding, rule=cfg))
    x3.finish(out, x2.Store(out), dict(d0_sha256=digest(out / 'd0.json'), outcome=outcome))


def aligned(arr, full_keys, keys):
    return x3.restrict(np.asarray(arr), full_keys, keys)


def mode_report(args, protocol, x1p, out):
    refs, x3r = x3.Refs(args.logs, protocol), X3Run(args.logs, protocol)
    z, st = float(protocol['family_z']), protocol['statistics_smoke' if args.smoke else 'statistics']
    a, b, d0 = upstream(args, 'a'), upstream(args, 'b'), upstream(args, 'd0')
    farms = x2.farms(x1p)
    dec = dict(family=['A', 'B1', 'B2'], z=z, note='development-stage evidence; V1/V2/tests were used in X3 (R1-S3)')
    sa, desc_a = {}, {}
    for T in FARMS:
        keys = {k: v for k, v in x1.load_pack(checked(a['root'] / f'{T}/v1_keys.npz',
                                                         a['registry'][f'{T}/v1_keys.npz'])).items()}
        full = refs.keys(T, 'V1')
        y, issue = keys['truth'], keys['issue']
        arms = {n: x3.load_registered(a, f'{T}/pred/{n}_V1.npy')[:, 1] for n in ('C_T0', 'C_T1', 'C_T3', 'C_D1')}
        arms['C_zs'] = aligned(x3r.array('q4', f'{T}_C_zs.npy'), full, keys)[:, 1]
        for n in ('L_1', 'L_1X', 'L_3', 'L_3X'):
            arms[n] = aligned(x3r.array('q4', f'{T}_{n}.npy'), full, keys)
        sa[T] = x3.paired(st, arms['C_T1'], arms['L_1X'], y, issue)
        desc_a[T] = dict(mae={n: float(np.abs(p - y).mean()) for n, p in arms.items()},
                         C_T1_vs_C_D1=x3.paired(st, arms['C_T1'], arms['C_D1'], y, issue),
                         C_T1_vs_C_zs=x3.paired(st, arms['C_T1'], arms['C_zs'], y, issue),
                         C_D1_vs_C_zs=x3.paired(st, arms['C_D1'], arms['C_zs'], y, issue))
    dec['A'] = dict(**x3.all_positive_pooled(sa, z), comparison='C_T1 vs L_1X on target V1 (k = 1)',
                    wording='the tested two-stage adaptation pipeline; no causal attribution to source transfer (R1-S4)',
                    descriptive=desc_a, a_meta_sha256=a['done']['meta_sha256'])
    s1, s3, s3c, prob = {}, {}, {}, {}
    for f in FARMS:
        keys = {s: refs.keys(f, s) for s in ('V1', 'V2')}
        sub = {s: dict(x1.load_pack(checked(b['root'] / f'{f}/{s}_keys.npz', b['registry'][f'{f}/{s}_keys.npz'])))
               for s in ('V1', 'V2')}
        ftb = {s: x3.load_registered(b, f'{f}/FTB_{s}.npy') for s in ('V1', 'V2')}
        lug = {s: aligned(refs.lug(f, s), keys[s], sub[s]) for s in ('V1', 'V2')}
        nv1 = {s: aligned(refs.nv1(f, s), keys[s], sub[s]) for s in ('V1', 'V2')}
        y1, y2, i2 = sub['V1']['truth'], sub['V2']['truth'], sub['V2']['issue']
        s1[f] = x3.paired(st, ftb['V2'][:, 1], lug['V2'], y2, i2)
        w = x3.ensemble_search(lug['V1'], ftb['V1'][:, 1], y1, protocol['b']['w_step'])[0]
        cw = x3.ensemble_search(lug['V1'], nv1['V1'], y1, protocol['b']['w_step'])[0]
        ens = w * lug['V2'] + (1 - w) * ftb['V2'][:, 1]
        ctrl = cw * lug['V2'] + (1 - cw) * nv1['V2']
        s3[f], s3c[f] = x3.paired(st, ens, lug['V2'], y2, i2), x3.paired(st, ens, ctrl, y2, i2)
        prob[f] = dict(w=w, control_w=cw, pinball=x3.pinball(ftb['V2'], y2, rated=farms[f]['rated']))
    dec['B1'] = x3.q2_decision(s1, z, protocol['b']['margin'])
    g3, g3c = x3.all_positive_pooled(s3, z, protocol['b']['min_gain']), x3.all_positive_pooled(s3c, z)
    dec['B2'] = dict(vs_LUG=g3, vs_LUG_plus_Nv1=dict(pooled=g3c['pooled']),
                     passed=None if g3['passed'] is None or g3c['pooled']['status'] != 'OK'
                     else bool(g3['passed'] and g3c['pooled']['ci95_low'] > 0), weights=prob,
                     b_meta_sha256=b['done']['meta_sha256'])
    dec['D0'] = dict(outcome=d0['done']['outcome'], d0_sha256=d0['done']['d0_sha256'])
    if dec['B1'].get('outcome') == 'TECHNICAL_FAILURE' or dec['A']['passed'] is None or dec['B2']['passed'] is None:
        raise ValueError('Technical statistical failure (degenerate CI); no outcome written')
    save(out / 'dev_decisions.json', dec)                     # BEFORE any test statistic
    tests = {}
    for f in FARMS:
        for name in farms[f]['items']:
            full = refs.keys(f, name)
            sub = dict(x1.load_pack(checked(b['root'] / f'{f}/{name}_keys.npz', b['registry'][f'{f}/{name}_keys.npz'])))
            y, issue = sub['truth'], sub['issue']
            lug = aligned(refs.lug(f, name), full, sub)
            ftb = x3.load_registered(b, f'{f}/FTB_{name}.npy')[:, 1]
            w = dec['B2']['weights'][f]['w']
            arms = dict(FT_B=ftb, ensemble=w * lug + (1 - w) * ftb)
            tests[name] = dict(LUG_mae=float(np.abs(lug - y).mean()),
                               mae={n: float(np.abs(p - y).mean()) for n, p in arms.items()},
                               vs_LUG={n: x3.paired(st, p, lug, y, issue) for n, p in arms.items()})
    save(out / 'results.json', dict(exploratory_only=True, dev_decisions_sha256=digest(out / 'dev_decisions.json'),
                                    tests=tests))
    x3.finish(out, x2.Store(out), dict(dev_decisions_sha256=digest(out / 'dev_decisions.json'),
                                       results_sha256=digest(out / 'results.json')))


MODES = {'a': mode_a, 'b': mode_b, 'd0': mode_d0, 'report': mode_report}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=list(MODES) + ['worker'], required=True)
    p.add_argument('--logs', type=Path, help='<data root>/outputs/logs')
    p.add_argument('--base', type=Path, help='<logs>/X4/x4_<commit>')
    p.add_argument('--protocol-sha256')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--chronos-python', default='/home/XT/miniconda3/envs/tsfm/bin/python')
    p.add_argument('--chronos-receipt', type=Path)
    p.add_argument('--chronos-manifest', type=Path)
    p.add_argument('--spec', type=Path)
    p.add_argument('--spec-sha')
    p.add_argument('--allow-dirty', action='store_true', help='tests only')
    args = p.parse_args(argv)
    if args.mode == 'worker':
        return worker_main(args)
    out, protocol, x1p = start_run(args, args.mode)
    try:
        MODES[args.mode](args, protocol, x1p, out)
    except Exception as exc:
        x2.fail(out, exc)
        raise


if __name__ == '__main__':
    main()
