"""X3: application value of a time-series foundation model (Chronos-2) — Q1 context/grouping, Q2 fine-tuning,
Q3 ensembling with LightGBM, Q4 cold start.  Plan: docs/STAGE_PLAN.md (X3, R2 approved 30df3b9).

Author: Claude (2026-11-18).  GPT reviews, Cursor runs.  Every mode writes to <base>/<mode> (smoke: <base>/smoke_<mode>)
and leaves failed.json on any error; no retries.  Parent modes run in the `timedart` env; Chronos runs in a
`tsfm` worker subprocess that only reads hash-checked npz/json specs.

Modes:  q1check  grouping isolation/permutation check (GPU worker, small dev subset)
        q1       zero-shot Chronos-2, context {168, 512, 1024} x {single, group}; V1 selection per farm, frozen first
        q2       resource preflight (all farms) -> one frozen fine-tune mode -> fine-tune per farm, V1-MAE checkpoint
        q3       ensembles w*LUG + (1-w)*M (M = Chronos member; control M = N_v1), w and member chosen on V1
        q4       cold start: C_zs (512 h, single) vs LightGBM L_kX / L_k trained only on data before D_T
        report   development decisions (Q2/Q3/Q4 family, Bonferroni z) written FIRST, then exploratory test statistics
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_x1 as x1  # noqa: E402  (numpy-only at import time)
from scripts import run_x2 as x2  # noqa: E402  (numpy-only at import time)

digest, read, save, checked = x1.digest, x1.read, x1.save, x1.checked
PROTOCOL = ROOT / 'configs/x3_protocol.json'
HOUR = np.timedelta64(1, 'h')
FARM_CODE = {'Kelmarsh': 0, 'Penmanshiel': 1, 'HoT': 2}


# ============================================================================ numpy core (tested locally)
class HourTable:
    """Dense (unit, hour) -> grid row table built from run_x1.history_grid output (one value per turbine-hour,
    conflicts were rejected when the grid was built).  Vectorised replacement of run_x1.longer_context's dict."""

    def __init__(self, grid):
        hours = np.asarray(grid['grid_hour'], np.int64)
        units = np.asarray(grid['grid_unit'], np.int64)
        self.h0 = int(hours.min())
        idx = (hours - self.h0) // 3_600_000_000_000
        if np.any((hours - self.h0) % 3_600_000_000_000):
            raise ValueError('Grid hours are not on the hour')
        self.units = {int(u): k for k, u in enumerate(np.unique(units))}
        self.table = np.full((len(self.units), int(idx.max()) + 1), -1, np.int64)
        self.table[[self.units[int(u)] for u in units], idx] = np.arange(len(hours))
        self.values = np.asarray(grid['grid_value'], np.float32)

    def rows(self, units, hours):
        """grid rows for unit [n] at hour labels [n, L] (int64 ns); -1 where absent."""
        idx = (np.asarray(hours, np.int64) - self.h0) // 3_600_000_000_000
        u = np.array([self.units.get(int(v), -1) for v in units])
        ok = (idx >= 0) & (idx < self.table.shape[1]) & (u[:, None] >= 0)
        out = np.full(idx.shape, -1, np.int64)
        out[ok] = self.table[np.broadcast_to(u[:, None], idx.shape)[ok], idx[ok]]
        return out


def window_hours(issue, L):
    """Hour labels issue-(L-1) .. issue (int64 ns); never later than the issue hour."""
    issue = np.asarray(issue).astype('datetime64[ns]').astype(np.int64)
    return issue[:, None] - (L - 1 - np.arange(L))[None, :] * 3_600_000_000_000


def context_window(ctx, table, idx, L):
    """[n, L, 21] normalised x21 history ending at the issue hour.  Older hours come from the turbine-hour table
    (NaN where absent); the last 24 hours are the frozen window itself and must agree with the table."""
    idx = np.asarray(idx)
    x = np.full((len(idx), L, 21), np.nan, np.float32)
    rows = table.rows(ctx['turbines'][idx], window_hours(ctx['issue'][idx], L))
    have = rows >= 0
    x[have] = table.values[rows[have]]
    own = np.asarray(ctx['x21'][idx], np.float32)
    seen = have[:, -24:]
    if not np.allclose(x[:, -24:][seen], own[seen], rtol=0, atol=1e-6, equal_nan=True):
        raise ValueError('Turbine-hour table disagrees with the frozen window')
    x[:, -24:] = own
    return x


def gfs_history(ctx, grid, idx, L):
    """Hourly GFS history (raw m/s) at the L input hours from the grid's hourly axis; NaN where absent."""
    times = window_hours(ctx['issue'][np.asarray(idx)], L)
    hours = np.asarray(grid['hours']).astype('datetime64[ns]').astype(np.int64)
    pos = np.clip(np.searchsorted(hours, times), 0, len(hours) - 1)
    valid = hours[pos] == times
    out = np.full(times.shape, np.nan)
    out[valid] = np.asarray(grid['historical'])[pos[valid], 0]
    return out


def coverage(x):
    """Share of finite power values (last x21 channel) in each context."""
    return np.isfinite(x[:, :, -1]).mean(1)


def mask_flagged_upstream(x, mean, scale):
    """Strict Q4 budget (X3-code R1-B4): on hours whose raw upstream-missing flag is set, the 11 upstream channels
    hold the frozen fill value (a mean over the multi-year fit segment).  Set them to NaN.  x: normalised [..., 21]."""
    x = np.array(x, np.float32, copy=True)
    flag = x[..., 19].astype(float) * float(scale[19]) + float(mean[19]) > .5
    up = x[..., 8:19]
    up[flag] = np.nan
    return x


def build_inputs(ctx, table, grid, idx, L, masked=False):
    """Chronos input dicts (run_x1.chronos_inputs) with an L-hour context; only issue-time information.
    masked=True: upstream fill values removed (Q4 C_zs only)."""
    x = context_window(ctx, table, idx, L)
    if masked:
        x = mask_flagged_upstream(x, ctx['mean'], ctx['scale'])
    hist = gfs_history(ctx, grid, idx, L)
    part = dict(x21=x, gF=ctx['gF'][idx], gF_mask=ctx['gF_mask'][idx])
    return x1.chronos_inputs(part, ctx['mean'], ctx['scale'], hist, np.arange(len(idx))), coverage(x)


def issue_groups(issue):
    """One group per issue time (= same farm, same issue; every member's history ends at that hour)."""
    issue = np.asarray(issue).astype(np.int64)
    order = np.argsort(issue, kind='stable')
    cuts = np.flatnonzero(np.diff(issue[order])) + 1
    return np.split(order, cuts)


def run_groups(predict, inputs, groups):
    """Group mode: exactly one issue group per predict call (cross-learning never spans issue times)."""
    out = [None] * len(inputs)
    for g in groups:
        res = predict([inputs[i] for i in g], len(g))
        for i, r in zip(g, res):
            out[i] = r
    return out


def key_positions(issue, turbines, keys):
    """Positions in a context (issue, turbines) of the windows listed by an X2 keys file; every key must exist."""
    lookup = {(int(t), int(u)): i for i, (t, u) in enumerate(zip(np.asarray(issue).astype(np.int64), turbines))}
    pos = np.array([lookup.get((int(t), int(u)), -1) for t, u in
                    zip(np.asarray(keys['issue']).astype(np.int64), keys['turbines'])], np.int64)
    if (pos < 0).any():
        raise ValueError('X2 key missing from the Chronos context')
    return pos


def select_config(v1_mae):
    """v1_mae {(L, mode): MAE}; lowest wins, ties -> shorter context, then single (registered tie rule)."""
    return min(v1_mae, key=lambda c: (round(v1_mae[c], 9), c[0], c[1] != 'single'))


def ensemble_search(ref, member, truth, step=.05):
    """w*ref + (1-w)*member, w on a fixed grid; lowest MAE, ties -> larger w (closer to no ensemble)."""
    best = None
    for w in np.round(np.arange(0, 1 + 1e-9, step), 10):
        mae = float(np.abs(w * ref + (1 - w) * member - truth).mean())
        if best is None or mae < best[1] - 1e-12 or (abs(mae - best[1]) <= 1e-12 and w > best[0]):
            best = (float(w), mae)
    return best


def q2_decision(stats, z, margin):
    """Fixed sequence: non-inferiority (pooled lower > -margin and no farm with its own 95% CI entirely below
    -margin), then superiority (pooled lower > 0).  stats: farm -> paired_loss_statistics vs LUG."""
    pooled = x2.pooled_effect(stats, z)
    if pooled['status'] != 'OK':
        return dict(outcome='TECHNICAL_FAILURE', pooled=pooled)
    worse = [f for f, s in stats.items() if s['ci95_high'] / s['reference_mean_loss'] < -margin]
    ni = pooled['ci95_low'] > -margin and not worse
    sup = bool(ni and pooled['ci95_low'] > 0)
    return dict(outcome='SUPERIOR' if sup else 'NON_INFERIOR' if ni else 'NEGATIVE', pooled=pooled,
                farms_significantly_worse_than_margin=worse, margin=margin,
                wording='pooled non-inferiority only; not shown separately per farm (R2-S1)')


def all_positive_pooled(stats, z, per_farm_min=0.0):
    pooled = x2.pooled_effect(stats, z)
    if pooled['status'] != 'OK':
        return dict(passed=None, pooled=pooled)
    rel = {f: s['gain'] / s['reference_mean_loss'] for f, s in stats.items()}
    return dict(passed=bool(pooled['ci95_low'] > 0 and all(r >= per_farm_min if per_farm_min else r > 0
                                                          for r in rel.values())), pooled=pooled, farm_relative=rel)


def pinball(q_pred, truth, levels=(.1, .5, .9), rated=1.0):
    """Mean pinball loss per quantile and horizon, normalised by rated power; plus 80% coverage/width/crossing."""
    q = np.asarray(q_pred, float)
    y = np.asarray(truth, float)
    out = {}
    for j, a in enumerate(levels):
        d = y - q[:, j]
        out[str(a)] = (np.maximum(a * d, (a - 1) * d).mean(0) / rated).tolist()
    lo, hi = q[:, 0], q[:, 2]
    return dict(pinball=out, coverage80=float(((y >= lo) & (y <= hi)).mean()),
                width80_kw=float((hi - lo).mean()), crossing_rate=float(((q[:, 0] > q[:, 1]) | (q[:, 1] > q[:, 2])).mean()))


# ---------------------------------------------------------------------------- Q4: physical features & time budget
def months_before(day, k):
    d = np.datetime64(day, 'M') - np.timedelta64(k, 'M')
    return np.datetime64(d, 'D').astype('datetime64[ns]')


def label_times(pack):
    return np.asarray(pack['times']).astype('datetime64[ns]')


def budget_mask(pack, lo, hi, input_hours=24):
    """Window inside the observation budget [lo, hi): every input hour (issue-23 h .. issue) >= lo, the same rule as
    the C_zs history bound, and every label < hi by its actual time (a label stamped hi is not available before hi)."""
    t = label_times(pack)
    issue = np.asarray(pack['issue']).astype('datetime64[ns]')
    lo, hi = np.datetime64(lo, 'ns'), np.datetime64(hi, 'ns')
    return (issue - (input_hours - 1) * HOUR >= lo) & (t >= lo).all(1) & (t < hi).all(1)


def es_split(issue, times, frac=.2):
    """Time-based split of the TARGET's k-month windows: early-stopping set = issue >= the (1-frac) quantile of
    issue times; fit windows need every label before that cut (no fit label inside the early-stopping period)."""
    issue = np.asarray(issue).astype('datetime64[ns]').astype(np.int64)
    cut = np.quantile(np.unique(issue), 1 - frac, method='lower')
    labels = np.asarray(times).astype('datetime64[ns]').astype(np.int64)
    return (labels < cut).all(1), issue >= cut, np.datetime64(int(cut), 'ns')


PHYS_NAMES = (['wspd', 'temp', 'ndir_sin', 'ndir_cos', 'pitch', 'rotor', 'power_last10_pu', 'wind_last10']
              + [f'up_{i}' for i in range(11)] + ['up_missing'] + [f'gfs_{i}' for i in range(13)]
              + ['gfs_age', 'gfs_cycle_missing'] + [f'p_hist_{i}' for i in range(24)] + [f'w_hist_{i}' for i in range(24)]
              + [f'gfs_hist_{i}' for i in range(24)] + ['retrofit', 'retrofit_applicable', 'farm', 'farm_turbine'])
CATEGORICAL = (len(PHYS_NAMES) - 2, len(PHYS_NAMES) - 1)


def gfs_binding(x50, gfs_hist, nwp_mean, nwp_scale, atol=1e-4):
    """The hourly-array GFS history must be what the frozen pack saw: x50 channel 20 = (hist - m)/s where the
    history is finite and exactly 0 where it is missing.  m, s are used for this check only, never as features."""
    ch = np.asarray(x50)[:, :, 20].astype(float)
    h = np.asarray(gfs_hist, float)
    if h.shape != ch.shape:
        raise ValueError('GFS history shape differs from the pack')
    if not np.isfinite(ch).all():
        raise ValueError('Non-finite frozen GFS channel 20')
    if not np.isfinite([nwp_mean, nwp_scale]).all() or float(nwp_scale) <= 0:
        raise ValueError('Invalid GFS binding normalisation')
    fin = np.isfinite(h)
    bad = np.zeros(h.shape, bool)
    bad[fin] = np.abs((h[fin] - float(nwp_mean)) / float(nwp_scale) - ch[fin]) > atol
    bad[~fin] = ch[~fin] != 0
    if bad.any():
        raise ValueError(f'GFS history differs from frozen x50 channel 20 at {int(bad.sum())} of {bad.size} hours')
    return dict(hours=int(bad.size), missing=int((~fin).sum()))


def physical_features(pack, farm, mean, scale, rated, gfs_hist):
    """Fixed-order physical columns from one farm's frozen normalisation (inverse transform only, nothing fitted).
    Power columns are divided by rated power; upstream features are NaN on flagged rows (no fill values); GFS
    comes from gF with its own mask; GFS history (raw m/s, NaN = missing) from the hourly arrays (bound to x50
    channel 20 by gfs_binding); categorical farm and farm*1000+turbine codes."""
    x = np.asarray(pack['x50'])
    mean, scale = np.asarray(mean, float), np.asarray(scale, float)
    if x.ndim != 3 or x.shape[1:] != (24, 50) or mean.shape != (21,) or scale.shape != (21,):
        raise ValueError('Unexpected frozen layout')
    n = len(x)
    gh = np.asarray(gfs_hist, float)
    if gh.shape != (n, 24):
        raise ValueError('GFS history must be [n, 24]')
    last = x[:, -1, :20].astype(float) * scale[:20] + mean[:20]
    last[:, 6] /= rated
    flag = last[:, 19] > .5
    last[flag, 8:19] = np.nan
    last[:, 19] = flag
    gfs = np.where(np.asarray(pack['gF_mask']) > 0, np.asarray(pack['gF'], float), np.nan)
    age, missing = x[:, -1, 47].astype(float), x[:, -1, 48].astype(float)
    p_hist = (x[:, :, 49].astype(float) * scale[20] + mean[20]) / rated
    w_hist = x[:, :, 0].astype(float) * scale[0] + mean[0]
    retro = np.asarray(pack['retrofit'], float) if farm == 'HoT' else np.zeros(n)
    applicable = np.full(n, 1.0 if farm == 'HoT' else 0.0)
    code = np.full(n, FARM_CODE[farm], float)
    ft = FARM_CODE[farm] * 1000 + np.asarray(pack['turbines'], float)
    out = np.column_stack([last, gfs, age, missing, p_hist, w_hist, gh, retro, applicable, code, ft]).astype(np.float32)
    if out.shape[1] != len(PHYS_NAMES):
        raise ValueError('Physical feature count differs')
    return out


# ============================================================================ Chronos worker (tsfm env, GPU)
def chronos_predict_fn(pipe, L, cross_learning, qidx):
    def predict(inputs, batch_size):
        if cross_learning:
            # Chronos counts variates, not turbine dictionaries. Keep this issue group in ONE SDK batch.
            batch_size = sum((1 if np.asarray(row['target']).ndim == 1 else np.asarray(row['target']).shape[0])
                             + len(row.get('past_covariates', {})) for row in inputs)
        res = pipe.predict(inputs, prediction_length=4, context_length=L, batch_size=batch_size,
                           cross_learning=cross_learning)
        return [p[0, qidx, :].detach().float().cpu().numpy() for p in res]
    return predict


def quantile_index(pipe):
    return [next(i for i, q in enumerate(pipe.quantiles) if abs(q - v) < 1e-5) for v in (.1, .5, .9)]


def predict_context(pipe, ctx, table, grid, L, mode, batch_size=1024, chunk=2048, masked=False):
    """[n, 3, 4] quantiles (0.1/0.5/0.9) for every window of a context, plus per-window history coverage."""
    qidx = quantile_index(pipe)
    n = len(ctx['issue'])
    out = np.full((n, 3, 4), np.nan, np.float32)
    cov = np.zeros(n, np.float32)
    if mode == 'single':
        f = chronos_predict_fn(pipe, L, False, qidx)
        for lo in range(0, n, chunk):
            ix = np.arange(lo, min(lo + chunk, n))
            inputs, c = build_inputs(ctx, table, grid, ix, L, masked)
            out[ix], cov[ix] = np.stack(f(inputs, batch_size)), c
    else:
        f = chronos_predict_fn(pipe, L, True, qidx)
        for g in issue_groups(ctx['issue']):
            inputs, c = build_inputs(ctx, table, grid, g, L, masked)
            out[g], cov[g] = np.stack(run_groups(f, inputs, [np.arange(len(g))])), c
    if not np.isfinite(out).all():
        raise ValueError('Non-finite Chronos prediction')
    return out, cov


def load_pipe(spec, model_dir=None):
    from importlib.metadata import version
    import torch
    from chronos import Chronos2Pipeline
    if version('chronos-forecasting') != '2.3.2':
        raise ValueError('X3 requires chronos-forecasting 2.3.2')
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(spec['gpu_fraction']))
    receipt = x1.verify_chronos(Path(spec['receipt']), Path(spec['manifest']))
    return Chronos2Pipeline.from_pretrained(model_dir or receipt['model_dir'], device_map='cuda', local_files_only=True)


def checked_npz(record):
    return x1.load_pack(checked(Path(record['path']), record['sha256']))


def worker_check(spec, out):
    """Grouping check on a small dev subset: member order must not matter; removing a member is reported; two
    issue groups in one call vs separately is reported (shows why one call must hold exactly one group)."""
    pipe = load_pipe(spec)
    ctx, grid = checked_npz(spec['context']), checked_npz(spec['grid'])
    table, L = HourTable(grid), int(spec['context_length'])
    qidx = quantile_index(pipe)
    f = chronos_predict_fn(pipe, L, True, qidx)
    groups = [g for g in issue_groups(ctx['issue']) if len(g) >= 3][:int(spec['groups'])]
    perm, removal, mixed = [], [], []
    for a, b in zip(groups[::2], groups[1::2]):
        for g in (a, b):
            inputs, _ = build_inputs(ctx, table, grid, g, L)
            base = np.stack(f(inputs, len(g)))[:, 1]
            rev = np.stack(f(inputs[::-1], len(g)))[::-1][:, 1]
            perm.append(float(np.abs(base - rev).max()))
            drop = np.stack(f(inputs[1:], len(g) - 1))[:, 1]
            removal.append(float(np.abs(base[1:] - drop).max()))
        ia, _ = build_inputs(ctx, table, grid, a, L)
        ib, _ = build_inputs(ctx, table, grid, b, L)
        joint = np.stack(f(ia + ib, len(ia) + len(ib)))[:len(ia), 1]
        mixed.append(float(np.abs(np.stack(f(ia, len(ia)))[:, 1] - joint).max()))
    tol = float(spec['permutation_tolerance_kw'])
    save(out / 'check.json', dict(status='PASS' if max(perm) <= tol else 'GROUP_MODE_EXCLUDED', groups=len(groups),
         context_length=L, permutation_max_abs_kw=max(perm), tolerance_kw=tol,
         member_removal_max_abs_kw=max(removal), member_removal_median_kw=float(np.median(removal)),
         two_groups_one_call_max_abs_kw=max(mixed),
         note='one issue group per call is enforced by run_groups; mixed-call difference is descriptive only'))


def q1_jobs(spec, name):
    """(L, mode, masked, file stem) for one context: every Q1 candidate; plus the registered Q4 C_zs variant with
    upstream fill removed, on the validation context only (never a Q1 selection candidate)."""
    jobs = [(int(L), m, False, f'{name}_{L}_{m}') for L, m in spec['configs']]
    if name == spec.get('masked_context'):
        jobs += [(int(L), m, True, f'{name}_{L}_{m}_masked') for L, m in spec['masked_configs']]
    return jobs


def worker_q1(spec, out):
    pipe = load_pipe(spec)
    grid = checked_npz(spec['grid'])
    table = HourTable(grid)
    coverage_report = {}
    for rec in spec['contexts']:
        ctx = checked_npz(rec)
        hist24 = gfs_history(ctx, grid, np.arange(len(ctx['issue'])), 24)
        both = np.isfinite(hist24) & np.isfinite(ctx['historical'])
        if not np.allclose(hist24[both], np.asarray(ctx['historical'])[both], rtol=0, atol=1e-6):
            raise ValueError('Grid GFS history differs from the A3 context: ' + rec['name'])
        for L, mode, masked, stem in q1_jobs(spec, rec['name']):
            pred, cov = predict_context(pipe, ctx, table, grid, L, mode, masked=masked)
            with (out / f'{stem}.npy').open('xb') as fh:
                np.save(fh, pred)
            coverage_report[stem] = dict(mean=float(cov.mean()), min=float(cov.min()))
            print('[X3] Q1', stem, 'coverage', round(float(cov.mean()), 4), flush=True)
    save(out / f"coverage_{spec['farm']}.json", coverage_report)     # one file per farm (all farms share out)


def training_series(train_npz):
    """Fine-tuning series in the run_x1.chronos_training_inputs layout: past target + 4 labels; past covariates
    with NaN future except GFS, whose future part is the issue-time forecast."""
    rows = []
    t, c, f, y = (train_npz[k] for k in ('target', 'covariates', 'gfs_future', 'truth'))
    names = [str(n) for n in train_npz['covariate_names']]
    for i in range(len(t)):
        cov = {}
        for j, name in enumerate(names):
            future = f[i] if name == 'gfs' else np.full(4, np.nan)
            cov[name] = np.r_[c[i, j], future]
        rows.append(dict(target=np.r_[t[i], y[i]], past_covariates=cov, future_covariates={'gfs': None}))
    return rows


VARIATES = 1 + len(['wind', 'up_missing', 'gfs'] + [f'up_{k}' for k in range(11)])   # = 15 rows per window


def sdk_windows(rows_per_batch, variates=VARIATES):
    """Chronos2Dataset (2.3.2, _generate_train_batches) adds whole windows until the row count reaches batch_size."""
    return -(-int(rows_per_batch) // int(variates))


def q2_candidates(cfg, variates=VARIATES):
    """Preflight order (X3-Q2fix, smoke 2026-11-18): full fine-tune with the registered batch first, then smaller
    microbatches with gradient accumulation keeping the SAME windows per optimizer step; LoRA only afterwards."""
    per_step = sdk_windows(cfg['batch_size'], variates)
    out = []
    for mode in cfg['modes']:
        for micro, accum in cfg['microbatch_plan']:
            if sdk_windows(micro, variates) * int(accum) != per_step:
                raise ValueError(f'Microbatch {micro} x {accum} changes the windows per optimizer step')
            out.append(dict(name=f'{mode}_mb{micro}x{accum}', mode=mode, microbatch=int(micro),
                            accumulation=int(accum), windows_per_step=per_step))
    return out


def preflight_search(candidates, farms, probe):
    """First candidate that passes on EVERY farm; a candidate is abandoned at its first failing farm (it can no
    longer be chosen).  probe(candidate, farm) -> report dict with 'ok'.  Returns (chosen or None, tried)."""
    tried = []
    for cand in candidates:
        reports = {}
        for farm in farms:
            reports[farm] = probe(cand, farm)
            if not reports[farm].get('ok'):
                break
        tried.append(dict(candidate=cand, farms=reports))
        if len(reports) == len(farms) and all(r.get('ok') for r in reports.values()):
            return cand, tried
    return None, tried


def fit_kwargs(pipe, spec, cand, steps):
    import importlib.util
    import inspect
    params = inspect.signature(pipe.fit).parameters
    mode = cand['mode']
    wanted = dict(prediction_length=4, context_length=int(spec['context_length']),
                  min_past=int(spec['context_length']), learning_rate=float(spec['learning_rate']),
                  num_steps=int(steps), batch_size=int(cand['microbatch']),
                  gradient_accumulation_steps=int(cand['accumulation']), finetune_mode=mode, seed=int(spec['seed']))
    accepts_any = any(p.kind == p.VAR_KEYWORD for p in params.values())
    missing = [k for k in ('num_steps', 'learning_rate', 'batch_size', 'gradient_accumulation_steps')
               if k not in params and not accepts_any]
    if missing:
        raise ValueError('Chronos fit lacks registered arguments: ' + ', '.join(missing))
    if mode != 'full' and 'finetune_mode' not in params:
        raise ValueError('finetune_mode not supported by the locked chronos-forecasting version')
    if mode == 'lora' and importlib.util.find_spec('peft') is None:
        # 2.3.2 would warn and silently fall back to full fine-tuning (smoke 2026-11-18): never allow that.
        raise ValueError('peft is not installed: LoRA unavailable')
    if mode == 'full' and 'finetune_mode' not in params:
        wanted.pop('finetune_mode')
    return wanted


def run_fit(pipe, rows, output_dir, spec, cand, steps, callbacks=None):
    """pipe.fit with the registered candidate; the SDK's LoRA -> full fallback warning becomes an error."""
    import warnings
    kwargs = fit_kwargs(pipe, spec, cand, steps)
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='.*peft.*required')
        pipe.fit(rows, output_dir=str(output_dir), callbacks=callbacks, **kwargs)


def worker_preflight(spec, out):
    """Resource probe on the fit set for ONE candidate in a fresh process; never a model-quality result."""
    import torch
    cand = spec['candidate']
    rows = training_series(checked_npz(spec['train']))[:int(spec['preflight_windows'])]
    report = dict(candidate=cand, context_length=int(spec['context_length']), free_gb_before=None)
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        report.update(free_gb_before=free / 2 ** 30, total_gb=total / 2 ** 30,
                      limit_gb=total * float(spec['gpu_fraction']) / 2 ** 30)
    try:
        pipe = load_pipe(spec)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        run_fit(pipe, rows, out / 'fit', spec, cand, spec['preflight_steps'])
        report.update(ok=True, peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None)
    except Exception as exc:  # resource or interface failure is recorded, not a quality result
        report.update(ok=False, error=f'{type(exc).__name__}: {str(exc)[:300]}')
    save(out / 'preflight.json', report)


def worker_q2(spec, out):
    import torch
    from transformers import TrainerCallback
    from chronos import Chronos2Pipeline
    pipe = load_pipe(spec)
    rows = training_series(checked_npz(spec['train']))
    if len(rows) != int(spec['budget']['train_windows']):
        raise ValueError('Q2 training windows differ from the registered budget')
    val, grid = checked_npz(spec['validation']), checked_npz(spec['grid'])
    table, L = HourTable(grid), int(spec['context_length'])
    v1 = np.asarray(spec['v1_positions'], np.int64)
    v1_truth = np.asarray(spec['v1_truth'], float)
    v1_inputs, _ = build_inputs(val, table, grid, v1, L)
    selected = out / 'selected'
    every, steps = int(spec['eval_every']), int(spec['num_steps'])

    class SelectByMAE(TrainerCallback):
        def __init__(self):
            self.best, self.rows, self.trainer = float('inf'), [], {}

        def on_train_begin(self, training_args, state, control, **kwargs):
            # R1-S1: record what the trainer actually used (getattr: recording must never stop the run)
            self.trainer = {k: getattr(training_args, k, None) for k in (
                'per_device_train_batch_size', 'gradient_accumulation_steps', 'max_steps', 'learning_rate',
                'n_gpu', 'world_size', 'seed', 'bf16', 'fp16')}
            self.trainer = {k: (v if isinstance(v, (int, float, bool, str)) or v is None else str(v))
                            for k, v in self.trainer.items()}
            cand = spec['candidate']
            if (self.trainer['gradient_accumulation_steps'] != cand['accumulation']
                    or self.trainer['per_device_train_batch_size'] != cand['microbatch']):
                raise ValueError('Trainer batch settings differ from the frozen preflight candidate')
            return control

        def on_step_end(self, training_args, state, control, model=None, **kwargs):
            if state.global_step % every:
                return control
            was = model.training
            model.eval()
            p = Chronos2Pipeline(model=model)
            with torch.inference_mode():
                f = chronos_predict_fn(p, L, False, quantile_index(p))
                med = np.concatenate([np.stack(f(v1_inputs[i:i + 2048], 256)) for i in range(0, len(v1_inputs), 2048)])[:, 1]
            score = float(np.abs(med - v1_truth).mean())
            if not np.isfinite(score):
                raise ValueError('Non-finite V1 MAE during fine-tuning')
            self.rows.append(dict(step=int(state.global_step), v1_mae=score))
            if score < self.best:
                self.best = score
                p.save_pretrained(selected)
                save(out / 'selection.json', dict(step=int(state.global_step), v1_mae=score))
            save(out / 'v1_history.json', self.rows)
            model.train(was)
            return control

    sel = SelectByMAE()
    run_fit(pipe, rows, out / 'fit', spec, spec['candidate'], steps, callbacks=[sel])
    if len(sel.rows) != steps // every or not selected.is_dir():
        raise ValueError('Checkpoint selection did not run at every registered evaluation')
    model_hashes = {str(p.relative_to(out)): digest(p) for p in selected.rglob('*') if p.is_file()}
    save(out / 'selection_freeze.json', dict(model_sha256=model_hashes, selected=read(out / 'selection.json'),
         mode=spec['mode'], candidate=spec['candidate'], steps=steps,
         selection='V1 median MAE (custom callback, not eval_loss)',
         train_windows=len(rows), budget=spec['budget'], trainer=sel.trainer))
    del pipe
    torch.cuda.empty_cache()
    for name, sha in model_hashes.items():
        checked(out / name, sha)
    tuned = load_pipe(spec, model_dir=selected)
    for rec in [dict(spec['validation'], name='validation')] + spec['tests']:
        ctx = checked_npz(rec)
        pred, _ = predict_context(tuned, ctx, table, grid, L, 'single')
        with (out / f"{rec['name']}_q2.npy").open('xb') as fh:
            np.save(fh, pred)


def worker_main(args):
    spec = read(checked(args.spec, args.spec_sha))
    out = Path(spec['output'])
    {'check': worker_check, 'q1': worker_q1, 'preflight': worker_preflight, 'q2': worker_q2}[spec['task']](spec, out)


# ============================================================================ parent side (timedart env)
def git_identity():
    return x2.git_identity()


def run_dir(base, mode, smoke):
    return Path(base) / (('smoke_' if smoke else '') + mode)


def start_run(args, mode):
    checked(PROTOCOL, args.protocol_sha256)
    commit, dirty = git_identity()
    if not args.allow_dirty and (commit is None or dirty):
        raise ValueError('Run from a clean worktree of the reviewed commit')
    if not args.smoke:
        smoke = run_dir(args.base, 'report', True)
        done = read(smoke / 'completed.json') if (smoke / 'completed.json').exists() else {}
        launch = read(smoke / 'launch.json') if (smoke / 'launch.json').exists() else {}
        if (done.get('status') != 'COMPLETED' or launch.get('commit') != commit or not launch.get('smoke')
                or launch.get('mode') != 'report' or launch.get('script_sha256') != digest(__file__)
                or launch.get('protocol_sha256') != args.protocol_sha256 or (smoke / 'failed.json').exists()):
            raise ValueError('Formal X3 modes need a completed smoke chain of this commit')
        for name, field in (('registry.json', 'registry_sha256'), ('results.json', 'results_sha256'),
                            ('dev_decisions.json', 'dev_decisions_sha256')):
            checked(smoke / name, done[field])
    out = run_dir(args.base, mode, args.smoke)
    out.mkdir(parents=True, exist_ok=False)
    (out / '.gitignore').write_text('*.npy\n*.npz\nselected/\nfit/\npre_*/\n', encoding='utf-8')
    save(out / 'launch.json', dict(mode=mode, smoke=bool(args.smoke), commit=commit, script_sha256=digest(__file__),
         protocol_sha256=args.protocol_sha256, pid=os.getpid(), started_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    return out, read(PROTOCOL), read(ROOT / 'configs/x1_protocol.json')


def checked_launch(args, root):
    me = read(run_dir(args.base, args.mode, args.smoke) / 'launch.json')
    other = read(root / 'launch.json')
    for key in ('commit', 'script_sha256', 'protocol_sha256', 'smoke'):
        if me[key] != other[key]:
            raise ValueError(f'Upstream identity differs: {key}')
    expected = root.name.removeprefix('smoke_')
    if other.get('mode') != expected:
        raise ValueError('Upstream mode differs')
    return other


def upstream(args, mode, required=True):
    """A finished X3 run of the same commit, script, protocol and smoke flag."""
    root = run_dir(args.base, mode, args.smoke)
    if not (root / 'completed.json').exists():
        if required:
            raise ValueError(f'Upstream X3 {mode} not completed')
        return None
    checked_launch(args, root)
    done = read(root / 'completed.json')
    if done.get('status') != 'COMPLETED' or (root / 'failed.json').exists():
        raise ValueError(f'Upstream {mode} is failed or not completed')
    registry = read(checked(root / 'registry.json', done['registry_sha256'])) if 'registry_sha256' in done else {}
    for name, field in (('selection.json', 'selection_sha256'), ('q4_meta.json', 'meta_sha256'),
                        ('dev_decisions.json', 'dev_decisions_sha256'), ('results.json', 'results_sha256')):
        if field in done:
            checked(root / name, done[field])
    return dict(root=root, done=done, registry=registry)


def load_registered(up, rel):
    return np.load(checked(up['root'] / rel, up['registry'][rel]))


class Refs:
    """Hash-pinned earlier runs reused by X3 (A3 contexts, X2 LightGBM / TSMixer predictions and keys)."""

    def __init__(self, logs, protocol):
        r = protocol['references']
        self.logs = Path(logs)
        self.a3 = self.logs / r['a3']['dir']
        self.a3_done = read(checked(self.a3 / 'completed.json', r['a3']['completed_sha256']))
        for part, key in (('A', 'x2_a'), ('B', 'x2_b')):
            root = self.logs / r[key]['dir']
            checked(root / part / 'completed.json', r[key]['completed_sha256'])
            if read(root / 'launch.json')['commit'] != r[key]['commit']:
                raise ValueError('X2 reference commit differs: ' + key)
        self.A = x2.Saved(self.logs / r['x2_a']['dir'], 'A')
        self.B = x2.Saved(self.logs / r['x2_b']['dir'], 'B')

    def context(self, key):
        name = key + '_context.npz'
        keys = read(checked(self.a3 / (key + '_keys.json'), self.a3_done['artifacts'][key + '_keys.json']))
        if keys['input_sha256'] != self.a3_done['artifacts'][name]:
            raise ValueError('A3 context identity differs: ' + key)
        return dict(name=key, path=str(self.a3 / name), sha256=keys['input_sha256'])

    def grid(self, farm):
        name = farm + '_history.npz'
        return dict(path=str(self.a3 / name), sha256=self.a3_done['artifacts'][name])

    def keys(self, farm, split):
        a, b = self.A.keys(farm, split), self.B.keys(farm, split)
        if any(not np.array_equal(a[k], b[k]) for k in ('truth', 'issue', 'turbines')):
            raise ValueError(f'X2 A/B keys differ: {farm} {split}')
        return a

    def lug(self, farm, split):
        return np.mean(self.A.preds(farm, 'LUG', split, seeds=[2024, 2025, 2026]), axis=0)

    def nv1(self, farm, split):
        return np.mean(self.B.preds(farm, 'N_v1', split, seeds=[2024, 2025, 2026], tag='none'), axis=0)


def shrink_context(record, out, n_issues, splits=None):
    """Smoke only: keep whole issue groups, separately in V1/V2 for validation contexts."""
    ctx = x1.load_pack(checked(Path(record['path']), record['sha256']))
    if splits is None:
        keep = np.isin(ctx['issue'], np.unique(ctx['issue'])[:n_issues])
    else:
        issue = np.asarray(ctx['issue']).astype('datetime64[ns]')
        times = issue[:, None] + np.arange(1, 5) * HOUR
        masks = [x2.period(times, splits[s]).all(1) for s in ('V1', 'V2')]
        if not all(m.any() for m in masks) or (masks[0] & masks[1]).any():
            raise ValueError('Smoke V1/V2 must be disjoint and non-empty')
        keep = np.zeros(len(issue), bool)
        for mask in masks:
            chosen = np.unique(ctx['issue'][mask])[:n_issues]
            keep |= mask & np.isin(ctx['issue'], chosen)
    path = out / ('smoke_' + Path(record['path']).name)
    np.savez(path, **{k: (v[keep] if v.ndim and len(v) == len(keep) else v) for k, v in ctx.items()})
    return dict(record, path=str(path), sha256=digest(path))


def call_worker(args, spec, out, name):
    path = out / (name + '_spec.json')
    save(path, spec)
    subprocess.run([args.chronos_python, '-u', str(Path(__file__).resolve()), '--mode', 'worker',
                    '--spec', str(path), '--spec-sha', digest(path)], check=True)


def worker_base(protocol, args, task, out):
    return dict(task=task, output=str(out), receipt=str(args.chronos_receipt), manifest=str(args.chronos_manifest),
                gpu_fraction=protocol['gpu_fraction'])


def positions(ctx_record, keys, smoke):
    ctx = x1.load_pack(checked(Path(ctx_record['path']), ctx_record['sha256']))
    if smoke:
        have = np.isin(np.asarray(keys['issue']).astype(np.int64), np.asarray(ctx['issue']).astype(np.int64))
        keys = {k: v[have] for k, v in keys.items()}
    return key_positions(ctx['issue'], ctx['turbines'], keys), keys


def paired(st, cand, ref, truth, issue):
    from utils.kelmarsh_backbone_information import paired_loss_statistics
    return paired_loss_statistics(np.abs(cand - truth), np.abs(ref - truth), np.asarray(issue).astype('datetime64[ns]'),
                                  block_length=st['block_length'], replicates=st['replicates'], seed=st['seed'])


def finish(out, store, extra=None):
    save(out / 'registry.json', store.registry)
    save(out / 'completed.json', dict(status='COMPLETED', registry_sha256=digest(out / 'registry.json'), **(extra or {})))


def mode_q1check(args, protocol, x1p, out):
    refs = Refs(args.logs, protocol)
    cfg = protocol['q1']['check']
    spec = dict(worker_base(protocol, args, 'check', out), context=refs.context(cfg['farm'] + '_validation'),
                grid=refs.grid(cfg['farm']), context_length=cfg['context_length'],
                groups=cfg['groups_smoke' if args.smoke else 'groups'], permutation_tolerance_kw=cfg['tolerance_kw'])
    call_worker(args, spec, out, 'check')
    finish(out, x2.Store(out), dict(check=read(out / 'check.json')))


def mode_q1(args, protocol, x1p, out):
    refs, store = Refs(args.logs, protocol), x2.Store(out)
    check = upstream(args, 'q1check')['done']['check']
    configs = [[L, 'single'] for L in protocol['q1']['context_lengths']]
    if check['status'] == 'PASS':
        configs += [[L, 'group'] for L in protocol['q1']['context_lengths']]
    masked = [[int(protocol['q4']['zs_context']), 'single']]     # Q4 C_zs, upstream fill removed (R1-B4)
    selection, records = {}, {}
    for farm, spec in x2.farms(x1p).items():
        ctxs = [refs.context(farm + '_validation')] + [refs.context(n) for n in spec['items']]
        if args.smoke:
            ctxs = [shrink_context(c, out, protocol['smoke']['issues'],
                                  protocol['splits'][farm] if c['name'] == farm + '_validation' else None)
                    for c in ctxs]
        wspec = dict(worker_base(protocol, args, 'q1', out), farm=farm, contexts=ctxs, grid=refs.grid(farm),
                     configs=configs, masked_configs=masked, masked_context=farm + '_validation')
        call_worker(args, wspec, out, 'q1_' + farm)
        for c in ctxs:
            for *_, stem in q1_jobs(wspec, c['name']):
                store.registry[stem + '.npy'] = digest(out / (stem + '.npy'))
        store.registry[f'coverage_{farm}.json'] = digest(out / f'coverage_{farm}.json')
        records[farm] = ctxs
        pos, keys = positions(ctxs[0], refs.keys(farm, 'V1'), args.smoke)
        v1 = {(L, m): float(np.abs(load_registered(dict(root=out, registry=store.registry),
                                                   f"{farm}_validation_{L}_{m}.npy")[pos, 1] - keys['truth']).mean())
              for L, m in configs}
        L, m = select_config(v1)
        selection[farm] = dict(context_length=L, mode=m, v1_mae={f'{a}_{b}': v for (a, b), v in v1.items()})
    save(out / 'selection.json', dict(selection=selection, configs=configs, check_status=check['status'],
         rule='per farm lowest V1 MAE of the median; ties -> shorter, then single', contexts=records,
         q4_masked=masked, q4_masked_note='validation context only, upstream NaN on flagged hours; not a Q1 candidate'))
    finish(out, store, dict(selection_sha256=digest(out / 'selection.json')))


COVARIATE_NAMES = ['wind', 'up_missing', 'gfs'] + [f'up_{k}' for k in range(11)]


def training_arrays(ctx, table, grid, idx, L, truth, chunk=2000):
    """Raw-unit fine-tuning arrays for windows idx: target [n, L] kW, covariates [n, 14, L], GFS future [n, 4].
    Built in chunks (long contexts of large farms)."""
    idx = np.asarray(idx)
    target = np.empty((len(idx), L), np.float32)
    cov = np.empty((len(idx), len(COVARIATE_NAMES), L), np.float32)
    for lo in range(0, len(idx), chunk):
        part = idx[lo:lo + chunk]
        x = context_window(ctx, table, part, L).astype(float) * np.asarray(ctx['scale']) + np.asarray(ctx['mean'])
        hist = gfs_history(ctx, grid, part, L)
        target[lo:lo + len(part)] = x[:, :, -1]
        cov[lo:lo + len(part)] = np.stack([x[:, :, 0], x[:, :, 19], hist] + [x[:, :, 8 + k] for k in range(11)], axis=1)
    future = np.where(np.asarray(ctx['gF_mask'])[idx, 6:10] > 0, np.asarray(ctx['gF'])[idx, 6:10], np.nan)
    return dict(target=target, covariates=cov, gfs_future=future.astype(np.float32),
                truth=np.asarray(truth, np.float32), covariate_names=np.array(COVARIATE_NAMES))


def mode_q2(args, protocol, x1p, out):
    refs, store, cfg = Refs(args.logs, protocol), x2.Store(out), protocol['q2']
    q1 = upstream(args, 'q1')
    sel = read(q1['root'] / 'selection.json')
    specs, budgets = {}, {}
    variates = 1 + len(COVARIATE_NAMES)                     # target + covariates = SDK batch units per window
    if variates != VARIATES:
        raise ValueError('Covariate layout differs from the SDK batch accounting')
    candidates = q2_candidates(cfg)
    for farm, spec in x2.farms(x1p).items():
        item = next(iter(spec['items'].values()))
        L = int(sel['selection'][farm]['context_length'])
        rec = refs.context(farm + '_train')
        ctx = x1.load_pack(checked(Path(rec['path']), rec['sha256']))
        train, _ = x1.development(Path(args.logs), item)
        if not (np.array_equal(ctx['issue'], train['issue']) and np.array_equal(ctx['turbines'], train['turbines'])):
            raise ValueError('A3 train context and development train pack differ: ' + farm)
        n = cfg['train_windows_smoke' if args.smoke else 'train_windows']
        # Registered fixed budget Q2-FT20k (R1-B5): uniform random subset of the fit segment, not the full segment.
        idx = np.sort(np.random.default_rng(cfg['seed']).choice(len(ctx['issue']), size=min(n, len(ctx['issue'])),
                                                               replace=False))
        steps = cfg['num_steps_smoke' if args.smoke else 'num_steps']
        budgets[farm] = dict(arm=cfg['arm'], fit_windows_available=int(len(ctx['issue'])), train_windows=int(len(idx)),
                             sampling=f"uniform without replacement, seed {cfg['seed']}", context_length=L,
                             variates_per_window=variates, sdk_batch_size_rows=int(cfg['batch_size']),
                             windows_per_step_by_sdk_rule=candidates[0]['windows_per_step'],
                             window_draws_by_sdk_rule=steps * candidates[0]['windows_per_step'],
                             note='derived from the 2.3.2 batching rule (windows drawn uniformly WITH replacement; not '
                                  'every window is seen); trainer settings are recorded in fit_run/selection_freeze.json')
        grid = x1.load_pack(checked(Path(refs.grid(farm)['path']), refs.grid(farm)['sha256']))
        path = out / f'{farm}_train.npz'
        np.savez(path, **training_arrays(ctx, HourTable(grid), grid, idx, L, train['truth'][idx]))
        del ctx, train, grid
        val = sel['contexts'][farm][0]
        pos, keys = positions(val, refs.keys(farm, 'V1'), args.smoke)
        specs[farm] = dict(worker_base(protocol, args, 'preflight', out / farm), context_length=L,
                           train=dict(path=str(path), sha256=digest(path)), validation=val,
                           tests=sel['contexts'][farm][1:], grid=refs.grid(farm),
                           v1_positions=pos.tolist(), v1_truth=keys['truth'].tolist(),
                           learning_rate=cfg['learning_rate'], batch_size=cfg['batch_size'], seed=cfg['seed'],
                           preflight_windows=cfg['preflight_windows'],
                           preflight_steps=cfg['preflight_steps_smoke' if args.smoke else 'preflight_steps'],
                           num_steps=steps, eval_every=cfg['eval_every_smoke' if args.smoke else 'eval_every'],
                           budget=budgets[farm])
        (out / farm).mkdir()

    def probe(cand, farm):                               # one fresh worker process per (candidate, farm)
        d = out / farm / f"pre_{cand['name']}"
        d.mkdir()
        call_worker(args, dict(specs[farm], candidate=cand, output=str(d)), d, 'preflight')
        return read(d / 'preflight.json')

    # every farm's preflight BEFORE any formal fine-tune; one setting for all farms
    chosen, tried = preflight_search(candidates, list(specs), probe)
    save(out / 'preflight_decision.json', dict(chosen=chosen, tried=tried, candidates=candidates,
         rule='first candidate (full first, then smaller microbatch with accumulation, then LoRA) passing on every farm'))
    if chosen is None:
        raise ValueError('Q2 resource preflight failed for every registered setting (technical, not a quality result)')
    mode = chosen['mode']
    for farm, spec in specs.items():
        run = out / farm / 'fit_run'
        run.mkdir()
        call_worker(args, dict(spec, task='q2', mode=mode, candidate=chosen, output=str(run)), run, 'q2')
        for rec in [dict(name='validation')] + spec['tests']:
            rel = f"{farm}/fit_run/{rec['name']}_q2.npy"
            store.registry[rel] = digest(out / rel)
        for rel in ('selection_freeze.json', 'v1_history.json'):
            store.registry[f'{farm}/fit_run/{rel}'] = digest(run / rel)
    store.registry['preflight_decision.json'] = digest(out / 'preflight_decision.json')
    finish(out, store, dict(mode=mode, setting=chosen, arm=cfg['arm'], budgets=budgets))


def q2_state(args):
    """Q2 must have finished: completed -> its run; failed -> None (technical, zero-shot member only); still
    running or never started -> error (q3/report must not silently skip the fine-tuned member)."""
    root = run_dir(args.base, 'q2', args.smoke)
    if (root / 'completed.json').exists():
        return upstream(args, 'q2')
    if (root / 'failed.json').exists():
        checked_launch(args, root)
        return None
    raise ValueError('Q2 has not finished (neither completed.json nor failed.json)')


def chronos_members(args, refs, farm, split, x1p):
    """Chronos predictions mapped to the X2 key order of a split: zero-shot winner of Q1 and, if Q2 completed,
    the fine-tuned model.  Every file is checked against its run registry."""
    q1 = upstream(args, 'q1')
    sel = read(q1['root'] / 'selection.json')
    win = sel['selection'][farm]
    name = farm + '_validation' if split in ('V1', 'V2') else split
    rec = next(c for c in sel['contexts'][farm] if c['name'] == name)
    pos, keys = positions(rec, refs.keys(farm, split), args.smoke)
    out = {'zs': load_registered(q1, f"{name}_{win['context_length']}_{win['mode']}.npy")[pos]}
    q2 = q2_state(args)
    if q2 is not None:
        tag = 'validation' if split in ('V1', 'V2') else split
        out['ft'] = load_registered(q2, f'{farm}/fit_run/{tag}_q2.npy')[pos]
    return out, keys


def restrict(arr, keys_full, keys):
    """Smoke: align a full-split array (X2 key order) to the subset of keys present in the shrunk context."""
    if len(keys['issue']) == len(keys_full['issue']):
        return arr
    pos = key_positions(keys_full['issue'], keys_full['turbines'], keys)
    return arr[pos]


def mode_q3(args, protocol, x1p, out):
    refs, store, step = Refs(args.logs, protocol), x2.Store(out), protocol['q3']['w_step']
    selection = {}
    for farm, spec in x2.farms(x1p).items():
        members, keys = chronos_members(args, refs, farm, 'V1', x1p)
        full = refs.keys(farm, 'V1')
        lug, nv1 = restrict(refs.lug(farm, 'V1'), full, keys), restrict(refs.nv1(farm, 'V1'), full, keys)
        y = keys['truth']
        cands = {m: ensemble_search(lug, p[:, 1], y, step) for m, p in members.items()}
        member = min(sorted(cands), key=lambda m: (round(cands[m][1], 9), m != 'zs'))
        selection[farm] = dict(member=member, w=cands[member][0], v1_mae=cands[member][1],
                               candidates={m: dict(w=c[0], v1_mae=c[1]) for m, c in cands.items()},
                               control_w=ensemble_search(lug, nv1, y, step)[0])
    save(out / 'selection.json', dict(selection=selection, rule='V1 MAE; ties -> zero-shot member; w ties -> larger w'))
    for farm, spec in x2.farms(x1p).items():
        s = selection[farm]
        for split in ['V1', 'V2'] + list(spec['items']):
            members, keys = chronos_members(args, refs, farm, split, x1p)
            full = refs.keys(farm, split)
            lug, nv1 = restrict(refs.lug(farm, split), full, keys), restrict(refs.nv1(farm, split), full, keys)
            for tag, pred in (('ens', s['w'] * lug + (1 - s['w']) * members[s['member']][:, 1]),
                              ('ctrl', s['control_w'] * lug + (1 - s['control_w']) * nv1)):
                store.put(f'{farm}_{split}_{tag}.npy', pred.astype(np.float32))
    finish(out, store, dict(selection_sha256=digest(out / 'selection.json')))


def scaler(args, farm):
    """Frozen x21 normalisation (inverse transform only) and the GFS normalisation (binding check only)."""
    if farm == 'HoT':
        st = read(Path(args.logs) / 'G2/g2b_4471499_shared_r1/formal/fit_statistics.json')
        return st['mean'], st['scale'], st['nwp_mean'], st['nwp_scale']
    meta = read(ROOT / 'output/g1/preparation_78/prepared' / farm / 'prepared.json')
    return meta['scaler']['mean'], meta['scaler']['scale'], meta['nwp_mean'], meta['nwp_scale']


def hourly_gfs(record):
    """Hash-checked hourly GFS axis of an A3 history grid (hours, historical) without loading the turbine table."""
    with np.load(checked(Path(record['path']), record['sha256']), allow_pickle=False) as z:
        return dict(hours=z['hours'], historical=z['historical'])


def farm_tables(args, protocol, x1p, refs):
    """Physical features, normalised labels and times of every window of every farm (train, val, tests)."""
    tables, v1, binding = {}, {}, {}
    for farm, spec in x2.farms(x1p).items():
        item = next(iter(spec['items'].values()))
        mean, scale, nwp_mean, nwp_scale = scaler(args, farm)
        rec = refs.context(farm + '_validation')
        with np.load(checked(Path(rec['path']), rec['sha256']), allow_pickle=False) as z:
            if not (np.array_equal(np.asarray(z['mean'], float), np.asarray(mean, float))
                    and np.array_equal(np.asarray(z['scale'], float), np.asarray(scale, float))):
                raise ValueError('Q4 normalisation differs from the A3/Chronos context: ' + farm)
        hourly = hourly_gfs(refs.grid(farm))
        train, val = x1.development(Path(args.logs), item)
        packs = dict(train=train, val=val, **{n: x1.load_pack(x1.paths(Path(args.logs), it)[0]) for n, it in spec['items'].items()})
        rows, binding[farm] = [], {}
        for name, p in packs.items():
            gh = gfs_history(p, hourly, np.arange(len(p['issue'])), 24)
            binding[farm][name] = gfs_binding(p['x50'], gh, nwp_mean, nwp_scale)
            rows.append(dict(X=physical_features(p, farm, mean, scale, spec['rated'], gh), y=p['truth'] / spec['rated'],
                             issue=np.asarray(p['issue']), times=np.asarray(p['times']), source=name))
        m1, _ = x2.split_masks(val, *(protocol['splits'][farm][k] for k in ('V1', 'V2')))
        v1[farm] = dict(X=rows[1]['X'][m1], issue=np.asarray(val['issue'])[m1], turbines=np.asarray(val['turbines'])[m1],
                        truth=val['truth'][m1])
        tables[farm] = rows
        del packs, train, val
    return tables, v1, binding


def select_rows(rows, lo, hi):
    keep = [budget_mask(dict(issue=r['issue'], times=r['times']), lo, hi) for r in rows]
    return {k: np.concatenate([r[k][m] for r, m in zip(rows, keep)]) for k in ('X', 'y', 'issue', 'times')}


def fit_lgb(lgb, fit, es, cfg, threads, n_estimators):
    models = []
    for h in range(4):
        m = lgb.LGBMRegressor(objective='regression_l1', num_leaves=cfg['num_leaves'],
                              min_child_samples=cfg['min_child_samples'], learning_rate=cfg['learning_rate'],
                              n_estimators=n_estimators, colsample_bytree=cfg['feature_fraction'],
                              subsample=cfg['bagging_fraction'], subsample_freq=1, random_state=cfg['seed'],
                              n_jobs=threads, verbosity=-1, deterministic=True, force_col_wise=True)
        m.fit(fit['X'], fit['y'][:, h], eval_set=[(es['X'], es['y'][:, h])], eval_metric='l1',
              categorical_feature=list(CATEGORICAL),
              callbacks=[lgb.early_stopping(cfg['early_stopping_rounds'], verbose=False)])
        models.append(m)
    return models


def mode_q4(args, protocol, x1p, out):
    import lightgbm as lgb
    if lgb.__version__ != '4.5.0':
        raise ValueError('Q4 LightGBM fits require lightgbm 4.5.0 (timedart)')
    refs, store, cfg = Refs(args.logs, protocol), x2.Store(out), protocol['q4']
    q1 = upstream(args, 'q1')
    sel = read(q1['root'] / 'selection.json')
    if [int(cfg['zs_context']), 'single'] not in sel.get('q4_masked', []):
        raise ValueError('Registered C_zs (masked upstream) configuration was not produced by Q1')
    tables, v1, binding = farm_tables(args, protocol, x1p, refs)
    n_est = cfg['n_estimators_smoke' if args.smoke else 'n_estimators']
    meta = dict(gfs_binding=binding, features=PHYS_NAMES, categorical=[PHYS_NAMES[i] for i in CATEGORICAL])
    for target, spec in x2.farms(x1p).items():
        D = np.datetime64(cfg['deployment'][target], 'ns')
        rec = next(c for c in sel['contexts'][target] if c['name'] == target + '_validation')
        pos, keys = positions(rec, refs.keys(target, 'V1'), args.smoke)
        zs = load_registered(q1, f"{target}_validation_{cfg['zs_context']}_single_masked.npy")[pos]
        tpos = key_positions(v1[target]['issue'], v1[target]['turbines'], keys)
        Xv = v1[target]['X'][tpos]
        if not np.allclose(v1[target]['truth'][tpos], keys['truth']):
            raise ValueError('Q4 V1 rows differ from the X2 keys')
        earliest = np.asarray(keys['issue']).astype('datetime64[ns]').min() - (cfg['zs_context'] - 1) * HOUR
        store.put(f'{target}_C_zs.npy', zs.astype(np.float32))
        meta[target] = dict(deployment=str(D), zs_history_lower_bound=str(earliest), arms={},
                            zs_input='512 h single, upstream NaN on flagged hours (no fit-segment fill)',
                            zs_v1_mae=float(np.abs(zs[:, 1] - keys['truth']).mean()))
        fill_rel = f"{target}_validation_{cfg['zs_context']}_single.npy"
        if fill_rel in q1['registry']:                     # descriptive only: same arm with the frozen fill values
            zf = load_registered(q1, fill_rel)[pos]
            meta[target]['zs_with_fill_v1_mae_descriptive'] = float(np.abs(zf[:, 1] - keys['truth']).mean())
        for k in cfg['k_months']:
            lo = months_before(D, k)
            if k == 1 and earliest < lo:
                raise ValueError('C_zs context reaches before the k=1 observation budget')
            tgt = select_rows(tables[target], lo, D)
            fit_t, es_t, cut = es_split(tgt['issue'], tgt['times'], cfg['es_fraction'])
            tfit = {key: v[fit_t] for key, v in tgt.items()}
            es = {key: v[es_t] for key, v in tgt.items()}
            src = [select_rows(tables[s], '1900-01-01', cut) for s in x2.farms(x1p) if s != target]
            arms = {f'L_{k}': tfit,
                    f'L_{k}X': {key: np.concatenate([s[key] for s in src] + [tfit[key]]) for key in tfit}}
            for arm, fit in arms.items():
                if (fit['times'].astype('datetime64[ns]') >= cut).any() or not len(es['y']):
                    raise ValueError('Q4 time budget violated: ' + arm)
                models = fit_lgb(lgb, fit, es, cfg['lightgbm'], args.threads, n_est)
                pred = np.stack([m.predict(Xv) for m in models], 1) * spec['rated']
                store.put(f'{target}_{arm}.npy', pred.astype(np.float32))
                meta[target]['arms'][arm] = dict(fit_windows=int(len(fit['y'])), es_windows=int(len(es['y'])),
                    es_cut=str(cut), target_windows=int(fit_t.sum()), best_iterations=[int(m.best_iteration_ or 0) for m in models])
                print('[X3] Q4', target, arm, meta[target]['arms'][arm], flush=True)
    save(out / 'q4_meta.json', meta)
    finish(out, store, dict(meta_sha256=digest(out / 'q4_meta.json')))


def mode_report(args, protocol, x1p, out):
    refs, z, st = Refs(args.logs, protocol), float(protocol['family_z']), protocol['statistics_smoke' if args.smoke else 'statistics']
    q1, q3, q4 = upstream(args, 'q1'), upstream(args, 'q3'), upstream(args, 'q4')
    q2 = q2_state(args)
    sel = read(q1['root'] / 'selection.json')
    farms = x2.farms(x1p)
    dec = dict(family=['Q2', 'Q3', 'Q4'], z=z, q2_available=q2 is not None,
               note='V1/V2 were used before: development-stage evidence; Q4 is an exploratory cold-start replay')
    q1d, q2s, q3s, q3c, q4s, prob = {}, {}, {}, {}, {}, {}
    for farm, spec in farms.items():
        members, keys = chronos_members(args, refs, farm, 'V2', x1p)
        full = refs.keys(farm, 'V2')
        y, issue = keys['truth'], keys['issue']
        lug = restrict(refs.lug(farm, 'V2'), full, keys)
        rec = sel['contexts'][farm][0]
        pos, _ = positions(rec, full, args.smoke)
        q1d[farm] = {f'{L}_{m}': float(np.abs(load_registered(q1, f'{farm}_validation_{L}_{m}.npy')[pos, 1] - y).mean())
                     for L, m in sel['configs']}
        q1d[farm]['LUG'] = float(np.abs(lug - y).mean())
        prob[farm] = {m: pinball(p, y, rated=spec['rated']) for m, p in members.items()}
        if 'ft' in members:
            q2s[farm] = paired(st, members['ft'][:, 1], lug, y, issue)
        ens, ctrl = load_registered(q3, f'{farm}_V2_ens.npy'), load_registered(q3, f'{farm}_V2_ctrl.npy')
        q3s[farm], q3c[farm] = paired(st, ens, lug, y, issue), paired(st, ens, ctrl, y, issue)
        k1 = protocol['q4']['k_months'][0]
        _, kv1 = positions(rec, refs.keys(farm, 'V1'), args.smoke)
        q4s[farm] = paired(st, load_registered(q4, f'{farm}_C_zs.npy')[:, 1], load_registered(q4, f'{farm}_L_{k1}X.npy'),
                           kv1['truth'], kv1['issue'])
    dec['Q1_descriptive'] = dict(v2_mae=q1d, selection=sel['selection'], coverage_note='see q1 coverage.json per farm')
    dec['Q2'] = dict(arm=protocol['q2']['arm'], budget=protocol['q2']['budget'],
                     **(q2_decision(q2s, z, protocol['q2']['margin']) if q2s else
                        dict(outcome='NOT_RUN_TECHNICAL',
                             note='Q2 run failed technically (resource/interface); not a quality result')))
    if q2 is not None:
        dec['Q2']['training_budget'] = q2['done'].get('budgets')
        dec['Q2']['setting'] = q2['done'].get('setting')
    g3 = all_positive_pooled(q3s, z, protocol['q3']['min_gain'])
    g3c = all_positive_pooled(q3c, z)
    dec['Q3'] = dict(vs_LUG=g3, vs_LUG_plus_Nv1=dict(pooled=g3c['pooled']),
                     passed=None if g3['passed'] is None or g3c['pooled']['status'] != 'OK'
                     else bool(g3['passed'] and g3c['pooled']['ci95_low'] > 0),
                     wording='better than LUG and than the tested ordinary ensemble control; no pre-training attribution')
    dec['Q4'] = dict(k=protocol['q4']['k_months'][0], **all_positive_pooled(q4s, z),
                     wording='exploratory cold-start replay on target V1; C_zs also uses 512 h of observed history; '
                             'upstream fill values removed on both arms (no target fit-segment statistic)')
    dec['probabilistic_descriptive'] = prob
    for name in ('Q2', 'Q3', 'Q4'):
        status = dec[name].get('outcome') if name == 'Q2' else dec[name].get('passed')
        if status in ('TECHNICAL_FAILURE',) or (name != 'Q2' and status is None):
            raise ValueError(f'{name}: technical statistical failure (degenerate CI), no outcome written')
    save(out / 'dev_decisions.json', dec)                    # BEFORE any test statistic
    tests = {}
    for farm, spec in farms.items():
        for name in spec['items']:
            members, keys = chronos_members(args, refs, farm, name, x1p)
            full = refs.keys(farm, name)
            y, issue, lug = keys['truth'], keys['issue'], restrict(refs.lug(farm, name), full, keys)
            arms = {f'chronos_{m}': p[:, 1] for m, p in members.items()}
            arms['ensemble'] = load_registered(q3, f'{farm}_{name}_ens.npy')
            arms['control_ensemble'] = load_registered(q3, f'{farm}_{name}_ctrl.npy')
            tests[name] = dict(LUG_mae=float(np.abs(lug - y).mean()),
                               vs_LUG={a: paired(st, p, lug, y, issue) for a, p in arms.items()},
                               mae={a: float(np.abs(p - y).mean()) for a, p in arms.items()})
    save(out / 'results.json', dict(exploratory_only=True, dev_decisions_sha256=digest(out / 'dev_decisions.json'),
                                    tests=tests))
    finish(out, x2.Store(out), dict(dev_decisions_sha256=digest(out / 'dev_decisions.json'),
                                    results_sha256=digest(out / 'results.json')))


MODES = {'q1check': mode_q1check, 'q1': mode_q1, 'q2': mode_q2, 'q3': mode_q3, 'q4': mode_q4, 'report': mode_report}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=list(MODES) + ['worker'], required=True)
    p.add_argument('--logs', type=Path, help='<data root>/outputs/logs')
    p.add_argument('--base', type=Path, help='<logs>/X3/x3_<commit>')
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
