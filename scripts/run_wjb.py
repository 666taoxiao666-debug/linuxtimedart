"""wind-journal-baselines (wjb): three wind-power journal models on the three farms, development data only.

Author: Claude (2026-11-24).  User split: Claude writes the code, Codex reviews, Cursor launches.
Plan: docs/STAGE_PLAN.md (stage wind-journal-baselines, approved at db32df0; line 84 mapping corrected by Codex);
protocol: configs/wjb_protocol.json (its SHA-256 is passed on the command line).

Lines (one mode each; every mode writes a FRESH <base>/<mode> or <base>/smoke_<mode> and leaves failed.json on any
error, no retries):
  krr    Nystroem kernel ridge regression of direct power (Bouche et al., Renewable Energy 2023, adapted), CPU.
         {LG, LUG} x GFS delay {6, 18} h, GFS columns rebuilt with run_x5; compared with X5's LightGBM of that delay.
  lgbh   LightGBM on the flattened 24 h history that Cabin sees ({L0, LU}), CPU.
  tsg    learned-partition adaptive robust regression with linear decision rules (Stratigakos & Andrianesis, IEEE TSG;
         author code in models/vendor_tsg), {LG, LUG} trained on the X5 mixed delays, evaluated at 6 h and 18 h; CPU pool.
  cabin  Cabin (Wu et al., Energy 2025; author code in models/vendor_cabin) with the sample-axis softmax switched off by
         the author's own flag (use_sample_softmax=False, causal adaptation), {L0, LU}, GPU.
  report paired comparisons on V2 (descriptive, no gate).
Chains (scripts/launch_x2.sh): wjbsmoke (every mode with --smoke), wjbcpu (krr lgbh tsg), wjbgpu (cabin), wjbfinal
(report).  Formal modes need a completed smoke report of the same commit, script and protocol.

Data: only train / V1 / V2 of the frozen development preparations (run_x1.development + run_x2.split_masks).  The test
and sealed loaders (run_x1.paths, run_x2.load_farm, run_x5.farm_caches) are disabled by dev_only() before any file is
opened.  Every line selects on V1 (4-horizon mean MAE, ties -> earlier candidate), writes v1_freeze.json for all three
farms, and only then predicts V2.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_x1 as x1  # noqa: E402  (numpy-only at import time)
from scripts import run_x2 as x2  # noqa: E402
from scripts import run_x5 as x5  # noqa: E402

digest, read, save, checked = x1.digest, x1.read, x1.save, x1.checked
PROTOCOL = ROOT / 'configs/wjb_protocol.json'
SEED = 2024
VENDOR_CABIN = ROOT / 'models/vendor_cabin'
VENDOR_TSG = ROOT / 'models/vendor_tsg'
VALUE = list(range(21, 34))          # x50 last step: GFS current-cycle values at offsets t0-5 .. t0+7
VALID = list(range(34, 47))          # their raw validity bits (1 = valid), same order
CABIN_DYN = {'L0': list(range(8)), 'LU': list(range(20))}   # e channels: local 0-7 (+ upstream 8-18 and flag 19)
FORBIDDEN_PARTS = ('sealed_prep', 'test.npz')


class TechnicalLimit(RuntimeError):
    """Registered stop of one line (plan: projected time > 3 x the estimate); the other lines are still reported."""


# ============================================================================ vendored author code
def blob_sha(path):
    """git blob SHA-1 of a file, CRLF normalised to LF (a Windows checkout must give the upstream blob)."""
    data = Path(path).read_bytes().replace(b'\r\n', b'\n')
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def vendor_fingerprint(protocol):
    got = {}
    for group in protocol['vendor'].values():
        for rel, rec in group['files'].items():
            sha = blob_sha(ROOT / rel)
            if sha != rec['git_blob']:
                raise ValueError(f'Vendored file differs from the registered copy: {rel}')
            got[rel] = sha
    return got


def load_cabin():
    """models/vendor_cabin as package 'vendor_cabin' (models/__init__ imports unrelated torch models)."""
    import importlib.util
    if 'vendor_cabin' not in sys.modules:
        spec = importlib.util.spec_from_file_location('vendor_cabin', VENDOR_CABIN / '__init__.py',
                                                      submodule_search_locations=[str(VENDOR_CABIN)])
        module = importlib.util.module_from_spec(spec)
        sys.modules['vendor_cabin'] = module
        spec.loader.exec_module(module)
    return sys.modules['vendor_cabin'].Cabin


def load_tsg():
    """The author modules by their own names (they use `from clean_torch_custom_layers import *`).  The author code
    calls torch.math.sqrt; if this torch build has no torch.math the standard math module is attached (no code edit)."""
    import math
    import torch
    if not hasattr(torch, 'math'):
        torch.math = math
    if str(VENDOR_TSG) not in sys.path:
        sys.path.insert(0, str(VENDOR_TSG))
    import clean_finite_adaptability_functions as fa
    if Path(fa.__file__).resolve().parent != VENDOR_TSG.resolve():
        raise ValueError('TSG author module resolved outside models/vendor_tsg')
    return fa


# ============================================================================ development-only data
@contextlib.contextmanager
def dev_only():
    """Disable every loader that can reach a test or sealed pack; refuse such paths in run_x1.load_pack."""
    orig = dict(paths=x1.paths, load_pack=x1.load_pack, load_farm=x2.load_farm, farm_caches=x5.farm_caches)

    def refuse(name):
        def stop(*_a, **_k):
            raise RuntimeError(f'dev-only stage: {name} must not be called')
        return stop

    def guarded(path):
        text = str(path).replace('\\', '/')
        if any(part in text for part in FORBIDDEN_PARTS):
            raise RuntimeError(f'dev-only stage: refusing to open {text}')
        return orig['load_pack'](path)

    x1.paths, x1.load_pack = refuse('run_x1.paths'), guarded
    x2.load_farm, x5.farm_caches = refuse('run_x2.load_farm'), refuse('run_x5.farm_caches')
    try:
        yield
    finally:
        x1.paths, x1.load_pack = orig['paths'], orig['load_pack']
        x2.load_farm, x5.farm_caches = orig['load_farm'], orig['farm_caches']


def guard_controls():
    """Negative controls of dev_only(), run inside it: each forbidden loader must be refused."""
    calls = {'run_x1.paths': lambda: x1.paths(Path('.'), {}),
             'run_x2.load_farm': lambda: x2.load_farm(None, None, None, None, None),
             'run_x5.farm_caches': lambda: x5.farm_caches(Path('.'), 'HoT'),
             'load_pack(sealed_prep)': lambda: x1.load_pack(Path('G2/x/sealed_prep/points.npz'))}
    out = {}
    for name, call in calls.items():
        try:
            call()
        except RuntimeError as exc:
            if 'dev-only stage' in str(exc):
                out[name] = 'refused'
                continue
            raise
        raise ValueError(f'dev_only did not refuse {name}')
    return out


def take(pack, rows):
    return x2.subset(pack, np.asarray(rows))


def label_hours(pack):
    return np.asarray(pack['times']).astype('datetime64[ns]') - np.timedelta64(1, 'h')


def check_cohorts(farm, v1, v2, train_count, protocol):
    ref = protocol['references']['x2_cohorts']
    cohorts = read(checked(ROOT / ref['dir'] / f'{farm}_cohorts.json', ref['sha256'][farm]))
    got = {k: x2.key_bytes(p) for k, p in (('V1', v1), ('V2', v2))}
    if got != {k: cohorts['keys'][k] for k in ('V1', 'V2')} or train_count != cohorts['train']:
        raise ValueError(f'{farm}: development windows differ from X2 part A (V1/V2 keys or train count)')
    return got


def load_dev(args, farm, spec, protocol, bud):
    """train / V1 / V2 of one farm (full cohort checks BEFORE any smoke slicing) and the farm's turbine ids."""
    first = next(iter(spec['items'].values()))
    train, val = x1.development(Path(args.logs), first)
    b = protocol['splits'][farm]
    m1, m2 = x2.split_masks(val, b['V1'], b['V2'])
    v1, v2 = x2.subset(val, m1), x2.subset(val, m2)
    del val
    end = np.datetime64(b['V2'][1])
    for name, p in (('train', train), ('V1', v1), ('V2', v2)):
        if not (label_hours(p) < end).all():
            raise ValueError(f'{farm} {name}: window beyond the development end {end}')
    keys = check_cohorts(farm, v1, v2, len(train['truth']), protocol)
    try:                                   # negative control: one shifted window must be detected
        check_cohorts(farm, take(v1, np.arange(1, len(v1['truth']))), v2, len(train['truth']), protocol)
    except ValueError:
        pass
    else:
        raise ValueError('Cohort check did not detect a shifted V1')
    units = np.unique(train['turbines'])
    for p in (v1, v2):
        if not np.isin(p['turbines'], units).all():
            raise ValueError(f'{farm}: evaluation turbine outside the training turbines')
    full = dict(train=len(train['truth']), V1=len(v1['truth']), V2=len(v2['truth']), keys=keys,
                train_last_label=str(label_hours(train).max()), V1_start=b['V1'][0], units=units.tolist())
    if bud['smoke']:
        train, v1, v2 = x2.head(train, bud['train_rows']), x2.head(v1, bud['eval_rows']), x2.head(v2, bud['eval_rows'])
    return dict(train=train, V1=v1, V2=v2), full, units


def dev_caches(logs, farm, end):
    """The development GFS point caches of run_x5.farm_caches (never its HoT sealed branch); cycles at or after the
    farm's development end are dropped before use (a dev window never reads them: cycle <= issue < end)."""
    if farm in ('Kelmarsh', 'Penmanshiel'):
        folder = logs / 'G1/g1c_27312d7_r2/prepared' / farm
        meta = read(folder / 'prepared.json')
        if meta != read(ROOT / 'output/g1/preparation_78/prepared' / farm / 'prepared.json'):
            raise ValueError('Preparation identity differs')
        hashes = {n: meta['inputs'][str(folder / n)] for n in ('points_0.npz', 'points_1.npz')}
        mean, scale = float(meta['nwp_mean']), float(meta['nwp_scale'])
    else:
        stats = read(logs / 'G2/g2b_4471499_shared_r1/formal/fit_statistics.json')
        folder = logs / 'G2/hot_dev_f58f2ce_r5/prepared'
        art = read(ROOT / 'output/g2/hot_dev/results82/completed.json')['artifacts']
        hashes = {n: art[n] for n in ('points_0.npz', 'points_1.npz')}
        mean, scale = float(stats['nwp_mean']), float(stats['nwp_scale'])
    kept, cycles, ident = {}, set(), {}
    end_hour = np.datetime64(end, 'h')
    for name, sha in hashes.items():
        ident[str(folder / name)] = sha
        with np.load(checked(folder / name, sha), allow_pickle=False) as archive:
            for key in archive.files:
                if len(key) != 10 or not key.isdigit():
                    raise ValueError(f'Unexpected GFS cache key {key!r}')
                cycle = np.datetime64(f'{key[:4]}-{key[4:6]}-{key[6:8]}T{key[8:]}', 'h')
                cycles.add(key)
                if cycle < end_hour:
                    kept[key] = archive[key]
    return kept, mean, scale, '2024-01-01', dict(identities=ident, cycles=len(cycles), cycles_kept=len(kept),
                                                  kept_before=str(end))


def gfs_blocks(args, farm, packs, wanted, protocol, out):
    """{split: {delay: [n, 29]}} from run_x5.gfs_block; at the nominal delay the rebuilt block must equal the frozen x50
    of every window (X5 nominal binding, atol 1e-5) and the frozen block itself is used."""
    nominal = protocol['references']['x5']['nominal_delay_hours']
    cache, mean, scale, cutoff, ident = dev_caches(Path(args.logs), farm, protocol['splits'][farm]['V2'][1])
    nsites = next(iter(cache.values())).shape[-1]
    blocks, audit = {}, {}
    for split, delays in wanted.items():
        pack = packs[split]
        blocks[split] = {}
        for d in sorted(set(delays) | {nominal}):
            blocks[split][d], audit[f'{split}@{d}'] = x5.gfs_block(pack['issue'], cache, d, mean, scale, cutoff, nsites)
        frozen = pack['x50'][:, -1, x5.GFS_COLS]
        diff = float(np.abs(blocks[split][nominal] - frozen).max())
        audit[f'{split}@binding'] = dict(max_abs_diff=diff, exact_fraction=float(np.mean(blocks[split][nominal] == frozen)))
        if not diff <= 1e-5:
            raise ValueError(f'{farm} {split}: rebuilt nominal GFS columns differ from the frozen x50 ({diff})')
        blocks[split][nominal] = frozen.copy()
    save(out / f'{farm}_gfs.json', dict(caches=ident, mean=mean, scale=scale, cutoff=cutoff, audit=audit))
    return blocks


# ============================================================================ features (numpy)
def one_hot(turbines, units):
    turbines = np.asarray(turbines)
    idx = np.searchsorted(units, turbines)
    if not np.array_equal(units[np.minimum(idx, len(units) - 1)], turbines):
        raise ValueError('Turbine outside the registered units')
    out = np.zeros((len(turbines), len(units)), np.float32)
    out[np.arange(len(turbines)), idx] = 1.0
    return out


def krr_view(pack, info, hot, units):
    """run_x2.tree_matrix (the X5 LightGBM view) with the turbine id column replaced by a one-hot block."""
    m = x2.tree_matrix(pack, info, hot)
    if not np.array_equal(m[:, -1], np.asarray(pack['turbines'], np.float32)):
        raise ValueError('tree_matrix no longer ends with the turbine id')
    return np.concatenate([m[:, :-1], one_hot(pack['turbines'], units)], axis=1)


def tree_columns(info, hot):
    cols = [('last', c) for c in x2.INFO[info]] + [('power', t) for t in range(24)] + [('wind', t) for t in range(24)]
    return cols + ([('retrofit', 0)] if hot else []) + [('turbine', 0)]


def tsg_layout(info, hot):
    """Kept tree_matrix columns (validity bits 34..46 removed) and the input position of value channel 21 + k."""
    cols = tree_columns(info, hot)
    keep = [i for i, c in enumerate(cols) if not (c[0] == 'last' and c[1] in VALID)]
    kept = [cols[i] for i in keep]
    if kept[-1] != ('turbine', 0):
        raise ValueError('turbine id must be the last tree column')
    return keep, [kept.index(('last', c)) for c in VALUE]


def tsg_view(pack, info, hot, units, start):
    """(X raw [n, p], valid [n, 13] bool, value positions).  Positions must be start..start+12 (LG 10, LUG 21); the
    columns there must be the x50 values 21..33 and the validity bits are read from x50 34..46 (0/1 only)."""
    keep, pos = tsg_layout(info, hot)
    if pos != list(range(start, start + 13)):
        raise ValueError(f'{info}: GFS value columns at {pos}, registered {start}..{start + 12}')
    m = x2.tree_matrix(pack, info, hot)
    X = np.concatenate([m[:, keep[:-1]], one_hot(pack['turbines'], units)], axis=1)
    last = pack['x50'][:, -1]
    if not np.array_equal(X[:, pos], last[:, VALUE].astype(np.float32)):
        raise ValueError(f'{info}: mapped columns are not the GFS values 21..33')
    valid = last[:, VALID]
    if not np.isin(valid, (0.0, 1.0)).all():
        raise ValueError('GFS validity bits are not 0/1')
    return X, valid.astype(bool), pos


def tsg_scaler(X, valid, pos):
    """MinMax on the fit windows; value columns use their valid cells only; constant columns keep scale 1."""
    lo, hi = X.min(0).astype(np.float64), X.max(0).astype(np.float64)
    for k, j in enumerate(pos):
        v = X[valid[:, k], j].astype(np.float64)
        lo[j], hi[j] = (v.min(), v.max()) if len(v) else (0.0, 1.0)
    scale = hi - lo
    return lo, np.where(scale > 0, scale, 1.0)


def tsg_inputs(X, valid, pos, lo, scale):
    """Scaled inputs with every truly missing value cell set back to 0, and the full mask a (1 only at a value column
    whose raw validity bit is 0)."""
    Z = (X.astype(np.float64) - lo) / scale
    A = np.zeros(X.shape, np.float32)
    for k, j in enumerate(pos):
        Z[~valid[:, k], j] = 0.0
        A[:, j] = ~valid[:, k]
    return Z.astype(np.float32), A


def cabin_raw(pack, info, hot, units, rows=slice(None)):
    """Cabin inputs before scaling: x = power history (x50 ch 49) [n, 24, 1]; e = channels CABIN_DYN[info] over 24 h
    plus two constant channels (HoT retrofit flag, 0 elsewhere; turbine index / number of turbines) [n, 24, C]."""
    x50 = pack['x50'][rows]
    n = len(x50)
    turbines = np.asarray(pack['turbines'])[rows]
    idx = np.searchsorted(units, turbines)
    if not np.array_equal(units[np.minimum(idx, len(units) - 1)], turbines):
        raise ValueError('Turbine outside the registered units')
    retro = np.asarray(pack['retrofit'], np.float32).reshape(-1)[rows] if hot else np.zeros(n, np.float32)
    const = np.stack([retro, (idx / len(units)).astype(np.float32)], axis=1)
    x = np.ascontiguousarray(x50[:, :, 49:50], dtype=np.float32)
    e = np.concatenate([x50[:, :, CABIN_DYN[info]], np.broadcast_to(const[:, None, :], (n, 24, 2))],
                       axis=2).astype(np.float32)
    if not (np.isfinite(x).all() and np.isfinite(e).all()):
        raise ValueError('Nonfinite Cabin inputs')
    return x, e


def hist_matrix(x, e, n_dyn):
    """LightGBM-hist row: e dynamic channels (time-major), power history, then the two constants once."""
    n = len(x)
    return np.concatenate([e[:, :, :n_dyn].reshape(n, -1), x[:, :, 0], e[:, 0, n_dyn:]], axis=1)


def hist_restore(H, n_dyn):
    n, k = len(H), 24 * n_dyn
    const = H[:, k + 24:]
    e = np.concatenate([H[:, :k].reshape(n, 24, n_dyn), np.broadcast_to(const[:, None, :], (n, 24, const.shape[1]))],
                       axis=2)
    return H[:, k:k + 24, None], e


def fit_history_scaler(train, info, hot, units, chunk=16384):
    """Per-channel MinMax bounds across all fit windows and all 24 hours, shared by both learners."""
    minima, maxima = [], []
    for lo in range(0, len(train['truth']), chunk):
        x, e = cabin_raw(train, info, hot, units, slice(lo, lo + chunk))
        minima.append(np.concatenate((x.min(axis=(0, 1)), e.min(axis=(0, 1)))))
        maxima.append(np.concatenate((x.max(axis=(0, 1)), e.max(axis=(0, 1)))))
    low = np.min(minima, axis=0)
    span = np.max(maxima, axis=0) - low
    span[span == 0] = 1
    return (low[:1], span[:1]), (low[1:], span[1:])


def paired_inputs(pack, info, hot, units, cols, cabin, scaler, chunk=16384):
    """Cabin arrays (cabin=True) or the LightGBM-hist matrix, built chunk by chunk from the SAME cabin_raw; every chunk
    is scaled with the fit-only scaler and checked against its flatten/restore round trip and column count. The
    digests identify the scaled x/e arrays supplied to the two models."""
    n, n_dyn = len(pack['truth']), len(CABIN_DYN[info])
    (lx, sx), (le, se) = scaler
    H = None if cabin else np.empty((n, cols), np.float32)
    if cabin:
        xs = np.empty((n, 24, 1), np.float32)
        es = np.empty((n, 24, n_dyn + 2), np.float32)
    hx, he = hashlib.sha256(), hashlib.sha256()
    for lo in range(0, n, chunk):
        x, e = cabin_raw(pack, info, hot, units, slice(lo, lo + chunk))
        x, e = (x - lx) / sx, (e - le) / se
        h = hist_matrix(x, e, n_dyn)
        if h.shape[1] != cols:
            raise ValueError(f'{info}: {h.shape[1]} history columns, registered {cols}')
        rx, re_ = hist_restore(h, n_dyn)
        if not (np.array_equal(rx, x) and np.array_equal(re_, e)):
            raise ValueError('LightGBM-hist and Cabin inputs differ')
        hx.update(np.ascontiguousarray(x).tobytes())
        he.update(np.ascontiguousarray(e).tobytes())
        if cabin:
            xs[lo:lo + len(x)] = x
            es[lo:lo + len(x)] = e
        else:
            H[lo:lo + len(x)] = h
    digests = dict(x=hx.hexdigest(), e=he.hexdigest(), windows=int(n), channels=n_dyn + 2)
    return ((xs, es) if cabin else H), digests


# ============================================================================ KRR (numpy)
def standardizer(Z):
    mean = Z.mean(0, dtype=np.float64)
    std = Z.std(0, dtype=np.float64)
    return mean, np.where(std > 1e-12, std, 1.0)       # constant (e.g. all-missing future GFS at 18 h) -> scale 1


def sq_dist(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T, 0.0)


def sigma_median(C):
    d = np.sqrt(sq_dist(C, C)[np.triu_indices(len(C), 1)])
    med = float(np.median(d))
    if not med > 0:
        raise ValueError('sigma_med must be > 0')
    return med


def krr_fit(Z, Y, C, sigmas, lams, chunk, jitter):
    """alpha = (K_nm' K_nm + n lambda K_mm + jitter I)^-1 K_nm' Y for every (sigma, lambda); K_nm' K_nm and K_nm' Y are
    accumulated over row chunks in float64 (one distance pass serves both sigmas)."""
    n, m = len(Z), len(C)
    G = {s: np.zeros((m, m)) for s in sigmas}
    B = {s: np.zeros((m, Y.shape[1])) for s in sigmas}
    for lo in range(0, n, chunk):
        D = sq_dist(Z[lo:lo + chunk], C)
        y = np.asarray(Y[lo:lo + chunk], np.float64)
        for s in sigmas:
            K = np.exp(-D / (2.0 * s * s))
            G[s] += K.T @ K
            B[s] += K.T @ y
    Dmm = sq_dist(C, C)
    out = {}
    for s in sigmas:
        Kmm = np.exp(-Dmm / (2.0 * s * s))
        for lam in lams:
            out[(s, lam)] = np.linalg.solve(G[s] + n * lam * Kmm + jitter * np.eye(m), B[s])
    return out


def krr_predict(Z, C, sigma, alpha, chunk):
    return np.concatenate([np.exp(-sq_dist(Z[lo:lo + chunk], C) / (2.0 * sigma * sigma)) @ alpha
                           for lo in range(0, len(Z), chunk)])


def kernel_summary(Z, C, sigmas, rows, cfg):
    """Quantiles of the fit-window kernel values on a seeded row sample; 'saturated' is only a flag."""
    sat, D = cfg['saturation'], sq_dist(Z[rows], C)
    out = {}
    for s in sigmas:
        K = np.exp(-D / (2.0 * s * s))
        below, above = float((K < sat['below']).mean()), float((K > 1.0 - sat['above_gap']).mean())
        out[s] = dict(quantiles=dict(zip([str(q) for q in cfg['quantiles']], np.quantile(K, cfg['quantiles']).tolist())),
                      below=below, above=above, saturated=bool(below >= sat['fraction'] or above >= sat['fraction']))
    return out


# ============================================================================ run bookkeeping
def budget(protocol, smoke):
    b = dict(protocol['budget']['smoke' if smoke else 'formal'])
    b['smoke'] = smoke
    return b


def run_dir(base, mode, smoke):
    return Path(base) / (('smoke_' if smoke else '') + mode)


def start_run(args, mode):
    """Protocol SHA, vendored files, clean worktree; formal modes need a completed smoke report of this commit,
    script and protocol (X4 rule); fresh output directory."""
    checked(PROTOCOL, args.protocol_sha256)
    protocol = read(PROTOCOL)
    vendor = vendor_fingerprint(protocol)
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
            raise ValueError('Formal wjb modes need a completed smoke chain (smoke_report) of this commit')
        for name, field in (('registry.json', 'registry_sha256'), ('results.json', 'results_sha256')):
            checked(smoke / name, done[field])
    out = run_dir(args.base, mode, args.smoke)
    out.mkdir(parents=True, exist_ok=False)
    (out / '.gitignore').write_text('*.npy\n*.npz\n*.pt\n*.pkl\nmodels/\njobs/\n', encoding='utf-8')
    save(out / 'launch.json', dict(mode=mode, smoke=bool(args.smoke), commit=commit, script_sha256=digest(__file__),
         protocol_sha256=args.protocol_sha256, vendor=vendor, python=sys.executable, pid=os.getpid(),
         started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), args={k: str(v) for k, v in vars(args).items()}))
    x1p = read(ROOT / 'configs/x1_protocol.json')
    if protocol['splits'] != read(ROOT / 'configs/x2_protocol.json')['splits']:
        raise ValueError('wjb splits differ from the X2 splits')
    return out, protocol, x1p


def beat(out, message):
    save(out / 'heartbeat.json', dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), message=message))
    print('[WJB]', message, flush=True)


def finish(out, store, extra=None):
    save(out / 'registry.json', store.registry)
    save(out / 'completed.json', dict(status='COMPLETED', registry_sha256=digest(out / 'registry.json'), **(extra or {})))


def save_keys(store, farm, packs):
    for split in ('V1', 'V2'):
        rel = f'keys/{farm}_{split}.npz'
        path = store.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{k: packs[split][k] for k in ('truth', 'issue', 'turbines', 'times')})
        store.registry[rel] = digest(path)


def v1_mae(pred, truth):
    return float(np.abs(np.asarray(pred, np.float64) - truth).mean())


def choose(rows):
    """Lowest V1 MAE; ties keep the earlier candidate."""
    return min(range(len(rows)), key=lambda i: (rows[i]['v1_mae'], i))


def write_freeze(out, freeze):
    """All three farms' V1 selections, written BEFORE any V2 prediction of the line."""
    freeze['written_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    save(out / 'v1_freeze.json', freeze)
    beat(out, 'V1 selections of all farms frozen; predicting V2')
    return digest(out / 'v1_freeze.json')


# ============================================================================ line A: Nystroem KRR (CPU)
def mode_krr(args, protocol, x1p, out):
    cfg, bud = protocol['krr'], budget(protocol, args.smoke)
    store, pending = x2.Store(out), {}
    freeze = dict(line='krr', guard=guard_controls(), farms={})
    cands = [dict(c=c, lam=lam) for c in cfg['c'] for lam in cfg['lambda']][:bud['krr_candidates']]
    for farm, spec in x2.farms(x1p).items():
        hot, rated = farm == 'HoT', float(spec['rated'])
        packs, full, units = load_dev(args, farm, spec, protocol, bud)
        save_keys(store, farm, packs)
        blocks = gfs_blocks(args, farm, packs, {s: cfg['delays'] for s in packs}, protocol, out)
        tr = packs['train']
        n = len(tr['truth'])
        centres = np.sort(np.random.default_rng(SEED).choice(n, min(cfg['centres'], n), replace=False))
        qrows = np.sort(np.random.default_rng(SEED).choice(n, min(cfg['saturation']['sample_rows'], n), replace=False))
        ckeys = hashlib.sha256(np.ascontiguousarray(tr['issue'][centres]).tobytes()
                               + np.ascontiguousarray(tr['turbines'][centres]).tobytes()).hexdigest()
        rec, pending[farm] = dict(full=full, centres=len(centres), centre_keys_sha256=ckeys, conditions={}), {}
        y = tr['truth'] / rated
        for d in cfg['delays']:
            x5.set_gfs(packs, {s: blocks[s][d] for s in packs})
            for info in cfg['infos']:
                cond = f'{info}_d{d}'
                X = {s: krr_view(packs[s], info, hot, units) for s in packs}
                mean, scale = standardizer(X['train'])
                Z = {s: ((X[s] - mean) / scale).astype(np.float32) for s in X}
                del X
                C = Z['train'][centres].astype(np.float64)
                smed = sigma_median(C)
                sigmas = sorted({c['c'] * smed for c in cands})
                t0 = time.perf_counter()
                alphas = krr_fit(Z['train'], y, C, sigmas, sorted({c['lam'] for c in cands}), cfg['chunk_rows'],
                                 cfg['jitter'])
                seconds = time.perf_counter() - t0
                kern = kernel_summary(Z['train'], C, sigmas, qrows, cfg)
                rows, preds = [], []
                for c in cands:
                    s = c['c'] * smed
                    pv1 = krr_predict(Z['V1'], C, s, alphas[(s, c['lam'])], cfg['chunk_rows']) * rated
                    preds.append(pv1)
                    rows.append(dict(c=c['c'], lam=c['lam'], sigma=s, v1_mae=v1_mae(pv1, packs['V1']['truth']),
                                     saturated=kern[s]['saturated']))
                i = choose(rows)
                s, a = rows[i]['sigma'], alphas[(rows[i]['sigma'], rows[i]['lam'])]
                path = out / 'models' / farm / f'{cond}.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, centres=C, alpha=a, mean=mean, scale=scale, sigma=np.float64(s))
                store.registry[f'models/{farm}/{cond}.npz'] = digest(path)
                store.put(f'{farm}/{cond}/V1.npy', preds[i])
                pending[farm][cond] = dict(Z=Z['V2'], C=C, sigma=s, alpha=a, rated=rated)
                rec['conditions'][cond] = dict(sigma_med=smed, features=int(Z['train'].shape[1]), candidates=rows,
                                               selected=i, fit_seconds_all_candidates=seconds,
                                               kernel={str(k): v for k, v in kern.items()},
                                               model_sha256=store.registry[f'models/{farm}/{cond}.npz'])
                beat(out, f'krr {farm} {cond}: selected {rows[i]}')
                del Z
        freeze['farms'][farm] = rec
        del packs, blocks
    sha = write_freeze(out, freeze)
    for farm, conds in pending.items():
        for cond, m in conds.items():
            store.put(f'{farm}/{cond}/V2.npy', krr_predict(m['Z'], m['C'], m['sigma'], m['alpha'], cfg['chunk_rows'])
                      * m['rated'])
    finish(out, store, dict(v1_freeze_sha256=sha))


# ============================================================================ line B1: LightGBM-hist (CPU)
def fit_plain(lgb, X, Y, Xv, Yv, params, rounds):
    """run_x2.fit_horizons without a categorical column (the turbine enters as index / count, as for Cabin)."""
    models = []
    for h in range(Y.shape[1]):
        m = lgb.LGBMRegressor(**params)
        m.fit(X, Y[:, h], eval_set=[(Xv, Yv[:, h])], eval_metric='l1',
              callbacks=[lgb.early_stopping(rounds, verbose=False)])
        models.append(m)
    return models


def mode_lgbh(args, protocol, x1p, out):
    import lightgbm as lgb
    if lgb.__version__ != '4.5.0':
        raise ValueError('LightGBM fits require the timedart environment with lightgbm 4.5.0')
    cfg, bud, lcfg = protocol['lgbh'], budget(protocol, args.smoke), x1p['lightgbm']
    rounds = bud.get('early_stopping_rounds', lcfg['early_stopping_rounds'])
    grid = [(a, b) for a in lcfg['num_leaves'] for b in lcfg['min_child_samples']][:bud['grid_cells']]
    store, pending = x2.Store(out), {}
    freeze = dict(line='lgbh', guard=guard_controls(), farms={})
    for farm, spec in x2.farms(x1p).items():
        hot = farm == 'HoT'
        packs, full, units = load_dev(args, farm, spec, protocol, bud)
        save_keys(store, farm, packs)
        rec, pending[farm] = dict(full=full, infos={}), {}
        for info in cfg['infos']:
            H, dig = {}, {}
            scaler = fit_history_scaler(packs['train'], info, hot, units)
            for s in packs:
                H[s], dig[s] = paired_inputs(packs[s], info, hot, units, cfg['columns'][info], cabin=False, scaler=scaler)
            rows, best = [], None
            for leaves, child in grid:
                t0 = time.perf_counter()
                models = fit_plain(lgb, H['train'], packs['train']['truth'], H['V1'], packs['V1']['truth'],
                                   x2.lgb_params(lcfg, leaves, child, SEED, args.threads, bud), rounds)
                pv1 = np.stack([m.predict(H['V1']) for m in models], axis=1)
                row = dict(num_leaves=leaves, min_child_samples=child, v1_mae=v1_mae(pv1, packs['V1']['truth']),
                           best_iterations=[int(m.best_iteration_ or 0) for m in models],
                           seconds=time.perf_counter() - t0)
                rows.append(row)
                beat(out, f'lgbh {farm} {info} grid {row}')
                if best is None or row['v1_mae'] < best[1]['v1_mae']:      # ties keep the earlier cell
                    best = (len(rows) - 1, row, models, pv1)
            folder = out / 'models' / farm / info
            hashes = x2.save_boosters(best[2], folder)
            pv1 = x2.booster_predict(lgb, folder, hashes, H['V1'])
            if not np.allclose(pv1, best[3], rtol=1e-6, atol=1e-3):
                raise ValueError('Saved booster does not reproduce in-memory predictions')
            store.put(f'{farm}/{info}/V1.npy', pv1)
            pending[farm][info] = dict(folder=folder, hashes=hashes, H=H['V2'])
            rec['infos'][info] = dict(columns=cfg['columns'][info], candidates=rows, selected=best[0], boosters=hashes,
                                      input_digests=dig)
            del H
        freeze['farms'][farm] = rec
        del packs
    sha = write_freeze(out, freeze)
    for farm, infos in pending.items():
        for info, m in infos.items():
            store.put(f'{farm}/{info}/V2.npy', x2.booster_predict(lgb, m['folder'], m['hashes'], m['H']))
    finish(out, store, dict(v1_freeze_sha256=sha))


# ============================================================================ line B2: Cabin (GPU)
def make_cabin(Cabin, n_ex, hidden, sample_softmax=False):
    return Cabin(input_window_size=24, input_vars=1, output_window_size=4, output_vars=1, ex_vars=n_ex,
                 hidden_size=hidden, mode='data_first', use_sample_softmax=sample_softmax)


def cabin_predict(torch, model, x, e, device, batch=4096):
    """Normalised [n, 4] predictions (power / rated)."""
    model.eval()
    outs = []
    with torch.no_grad():
        for lo in range(0, len(x), batch):
            xb = torch.as_tensor(x[lo:lo + batch], device=device)
            eb = torch.as_tensor(e[lo:lo + batch], device=device)
            outs.append(model(xb, eb)[..., 0].float().cpu().numpy())
    return np.concatenate(outs)


def independence_gap(torch, model, x, e, issue, cfg):
    """R2-B1 acceptance: the earliest `windows` windows predicted alone vs inside a batch with `later_windows` windows
    of strictly later issue time (placed before and after them); max |difference| of the normalised outputs, on a CPU
    copy of the model."""
    k, late = cfg['windows'], cfg['later_windows']
    t = np.asarray(issue).astype('datetime64[ns]')
    order = np.argsort(t, kind='stable')
    small, later = order[:k], order[-late:]
    if len(later) < late or not t[later].min() > t[small].max():
        raise ValueError('Independence check needs later issue times')
    cpu = copy.deepcopy(model).to('cpu').eval()

    def run(rows):
        with torch.no_grad():
            return cpu(torch.as_tensor(x[rows]), torch.as_tensor(e[rows]))[..., 0].numpy()

    mixed = run(np.concatenate([later[:late // 2], small, later[late // 2:]]))[late // 2:late // 2 + k]
    singletons = np.concatenate([run([row]) for row in small], axis=0)
    return float(np.abs(singletons - mixed).max())


def cabin_fit(torch, Cabin, data, n_ex, cand, cfg, bud, device, folder, tag):
    """Author loss (MSE) and optimiser (Adam); shuffled batches; early stopping on the V1 MSE (patience); the best
    epoch's state is kept and saved."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = make_cabin(Cabin, n_ex, cand['hidden']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cand['lr'])
    xs, es, ys = (torch.as_tensor(data['train'][k], device=device) for k in ('x', 'e', 'y'))
    v1 = data['V1']
    epochs, patience = bud.get('cabin_max_epochs', cfg['max_epochs']), bud.get('cabin_patience', cfg['patience'])
    best, best_epoch, stale, state, history = float('inf'), 0, 0, None, []
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        order = torch.as_tensor(np.random.default_rng(SEED + epoch).permutation(len(xs)), device=device)
        bad = torch.zeros((), dtype=torch.bool, device=device)
        for lo in range(0, len(xs), cfg['batch_size']):
            ix = order[lo:lo + cfg['batch_size']]
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((model(xs[ix], es[ix])[..., 0] - ys[ix]) ** 2)
            bad |= ~torch.isfinite(loss)
            loss.backward()
            opt.step()
        if bool(bad):
            raise ValueError('Non-finite Cabin training loss')
        mse = float(np.mean((cabin_predict(torch, model, v1['x'], v1['e'], device) - v1['y']) ** 2))
        if not np.isfinite(mse):
            raise ValueError('Non-finite Cabin V1 loss')
        history.append(dict(epoch=epoch, v1_mse=mse, seconds=time.perf_counter() - t0))
        print(f'[WJB] {tag} epoch {epoch} V1 MSE {mse:.6f} {history[-1]["seconds"]:.1f}s', flush=True)
        if mse < best:
            best, best_epoch, stale = mse, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    folder.mkdir(parents=True, exist_ok=False)
    torch.save(state, folder / 'selected.pt')
    model.load_state_dict(state)
    return model, dict(best_epoch=best_epoch, epochs=len(history), v1_mse=best, sha256=digest(folder / 'selected.pt'),
                       seconds_per_epoch=float(np.mean([h['seconds'] for h in history])), history=history)


def mode_cabin(args, protocol, x1p, out):
    import torch
    Cabin = load_cabin()
    torch.set_num_threads(args.threads)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    cfg, bud = protocol['cabin'], budget(protocol, args.smoke)
    fraction = args.gpu_fraction if args.gpu_fraction is not None else cfg['gpu_fraction']
    if device == 'cuda' and fraction:
        torch.cuda.set_per_process_memory_fraction(fraction)
    cands, tol = cfg['candidates'][:bud['cabin_candidates']], cfg['independence']['tolerance']
    store, pending = x2.Store(out), {}
    freeze = dict(line='cabin', guard=guard_controls(), device=device, torch=torch.__version__, farms={})
    for farm, spec in x2.farms(x1p).items():
        hot, rated = farm == 'HoT', float(spec['rated'])
        packs, full, units = load_dev(args, farm, spec, protocol, bud)
        save_keys(store, farm, packs)
        rec, pending[farm] = dict(full=full, infos={}), {}
        for info in cfg['infos']:
            data, dig = {}, {}
            scaler = fit_history_scaler(packs['train'], info, hot, units)
            (lx, sx), (le, se) = scaler
            for s in packs:
                (x, e), dig[s] = paired_inputs(packs[s], info, hot, units, protocol['lgbh']['columns'][info],
                                               cabin=True, scaler=scaler)
                data[s] = dict(x=x, e=e, y=(packs[s]['truth'] / rated).astype(np.float32))
            n_ex, issue = data['train']['e'].shape[2], packs['V1']['issue']
            if 'negative_control' not in freeze:          # the check must see the author's sample-axis softmax
                torch.manual_seed(SEED)
                gap = independence_gap(torch, make_cabin(Cabin, n_ex, cands[0]['hidden'], True), data['V1']['x'],
                                       data['V1']['e'], issue, cfg['independence'])
                if not gap > tol:
                    raise ValueError(f'Independence check has no power (sample softmax gap {gap})')
                freeze['negative_control'] = dict(farm=farm, info=info, use_sample_softmax=True, gap=gap)
            rows, best = [], None
            for i, cand in enumerate(cands):
                folder = out / 'models' / farm / info / f'cand{i}'
                model, r = cabin_fit(torch, Cabin, data, n_ex, cand, cfg, bud, device, folder, f'cabin {farm} {info} {cand}')
                gap = independence_gap(torch, model, data['V1']['x'], data['V1']['e'], issue, cfg['independence'])
                if not gap <= tol:
                    raise ValueError(f'Cabin prediction depends on other windows of the batch ({gap})')
                pv1 = cabin_predict(torch, model, data['V1']['x'], data['V1']['e'], device) * rated
                store.put(f'{farm}/{info}/cand{i}/V1.npy', pv1)
                rows.append(dict(cand, v1_mae=v1_mae(pv1, packs['V1']['truth']), independence_gap=gap,
                                 **{k: v for k, v in r.items() if k != 'history'}))
                save(folder / 'trained.json', dict(rows[-1], history=r['history']))
                beat(out, f'cabin {farm} {info} cand {i}: V1 MAE {rows[-1]["v1_mae"]:.3f} best epoch {r["best_epoch"]}')
                del model
                if device == 'cuda':
                    torch.cuda.empty_cache()
            i = choose(rows)
            pending[farm][info] = dict(x=data['V2']['x'], e=data['V2']['e'], n_ex=n_ex, hidden=rows[i]['hidden'],
                                       path=out / 'models' / farm / info / f'cand{i}' / 'selected.pt',
                                       sha=rows[i]['sha256'], rated=rated)
            rec['infos'][info] = dict(channels=n_ex, candidates=rows, selected=i, input_digests=dig,
                                      scaling=dict(x_min=lx.tolist(), x_scale=sx.tolist(), e_min=le.tolist(),
                                                   e_scale=se.tolist()))
            del data
        freeze['farms'][farm] = rec
        del packs
    sha = write_freeze(out, freeze)
    for farm, infos in pending.items():
        for info, m in infos.items():
            model = make_cabin(Cabin, m['n_ex'], m['hidden']).to(device)
            model.load_state_dict(torch.load(checked(m['path'], m['sha']), map_location=device, weights_only=True))
            store.put(f'{farm}/{info}/V2.npy', cabin_predict(torch, model, m['x'], m['e'], device) * m['rated'])
    finish(out, store, dict(v1_freeze_sha256=sha))


# ============================================================================ line C: TSG adaptive robust (CPU pool)
def tree_record(model):
    return dict(total_models=int(model.total_models), nodes=len(model.Node_id),
                leaves=int(sum(v == -1 for v in model.children_left_dict.values())),
                split_columns=[int(f) for f in model.feature if f != -1],
                lower_loss=[float(v) for v in model.LB_Loss], upper_loss=[float(v) for v in model.UB_Loss])


def routing_check(fa, Z, A, pos):
    """R2-B3: the author's apply/predict on a hand-built tree that splits on one value column f (no fit).  Author
    rules: a window whose mask equals a node's pattern stops there (all-valid windows stay at the root); otherwise
    f valid -> left (node 1), f missing -> right (node 2); at the reached node the nominal model is used when the mask
    equals that node's pattern, the robust (worst-case) model otherwise."""
    frac = A[:, pos].mean(0)
    f = int(pos[int(np.argmax(np.minimum(frac, 1 - frac)))])
    g = int(next(j for j in pos if j != f))
    synth = np.zeros((4, Z.shape[1]), np.float32)
    synth[1, g] = synth[2, f] = synth[3, f] = synth[3, g] = 1.0        # none, other only, f only, f + other
    Zc, Ac = np.concatenate([Z, Z[:4]]), np.concatenate([A, synth])
    p = Z.shape[1]
    tree = fa.Learn_FiniteAdapt_Robust_Reg.__new__(fa.Learn_FiniteAdapt_Robust_Reg)
    e_f = np.zeros(p)
    e_f[f] = 1.0
    tree.Node_id, tree.feature = [0, 1, 2], [f, -1, -1]
    tree.children_left_dict, tree.children_right_dict = {0: 1, 1: -1, 2: -1}, {0: 2, 1: -1, 2: -1}
    tree.missing_pattern = [np.zeros(p), np.zeros(p), e_f]

    class Tag:
        def __init__(self, v):
            self.v = v

        def predict(self, x0, m0):
            return np.array([[self.v]])

    tree.node_model_ = [Tag(0.0), Tag(1.0), Tag(2.0)]
    tree.wc_node_model_ = [Tag(10.0), Tag(11.0), Tag(12.0)]
    miss_f, none = Ac[:, f] == 1, ~Ac.astype(bool).any(1)
    only_f = miss_f & (Ac.sum(1) == 1)
    node = tree.apply(Zc, Ac)[:, 0]
    want_node = np.where(miss_f, 2, np.where(none, 0, 1))
    pred = tree.predict(Zc, Ac)[:, 0]
    want_pred = np.select([none, ~miss_f, only_f], [0.0, 11.0, 2.0], default=12.0)
    if not (np.array_equal(node, want_node) and np.array_equal(pred, want_pred)):
        raise ValueError('Author routing differs from the registered valid-left / missing-right rule')
    return dict(split_column=f, other_column=g, rows=int(len(Zc)), root=int(none.sum()), left=int((want_node == 1).sum()),
                right_nominal=int(only_f.sum()), right_robust=int((miss_f & ~only_f).sum()))


def tsg_acceptance(fa, v2, pos):
    """Masks/bits of the 6 h and 18 h V2 inputs must differ; mask = 1 - validity at the 13 positions only; missing cells
    are 0 after scaling; controlled-tree routing on real (and 4 synthetic) masks."""
    (Z6, A6, V6), (Z18, A18, V18) = v2[6], v2[18]
    if np.array_equal(V6, V18) or np.array_equal(A6, A18):
        raise ValueError('6 h and 18 h inputs have identical validity bits or masks')
    for Z, A, V in ((Z6, A6, V6), (Z18, A18, V18)):
        other = np.delete(A, pos, axis=1)
        if other.any() or not np.array_equal(A[:, pos].astype(bool), ~V) or np.any(Z[:, pos][~V] != 0):
            raise ValueError('Mask or re-zeroed cells do not follow the registered mapping')
    rows = slice(0, 2000)
    route = routing_check(fa, np.concatenate([Z6[rows], Z18[rows]]), np.concatenate([A6[rows], A18[rows]]), pos)
    return dict(mask_cells_6=int(A6.sum()), mask_cells_18=int(A18.sum()), windows_differing=int((A6 != A18).any(1).sum()),
                routing=route)


def worker_init(threads):
    import torch
    torch.set_num_threads(threads)


def tsg_fit_job(job):
    """One author tree (ARR(learn)) for one horizon; V1 predictions with the author's predict (clipped to [0, 1])."""
    import torch
    fa = load_tsg()
    folder = Path(job['folder'])
    folder.mkdir(parents=True, exist_ok=False)
    begin = mark_started(folder)
    with np.load(checked(job['data'], job['data_sha256'])) as z:
        X, Y = z['X_train'], z['Y_train'][:, [job['horizon']]]
        Xv, Av, pos = z['X_V1'], z['A_V1'], z['value_pos'].tolist()
    p, c, t = X.shape[1], job['candidate'], job['train']
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = fa.Learn_FiniteAdapt_Robust_Reg(pos, [i for i in range(p) if i not in pos], D=t['D'],
                                            Max_splits=c['max_splits'], red_threshold=t['red_threshold'],
                                            error_metric='mse', input_size=p, hidden_sizes=list(c['hidden']),
                                            output_size=1, budget_constraint='inequality', attack_type='greedy',
                                            apply_LDR=True)
    with open(folder / 'author_stdout.txt', 'w', encoding='utf-8') as fh, contextlib.redirect_stdout(fh):
        t0 = time.perf_counter()
        model.fit(X, Y, tree_grow_algo='leaf-wise', max_gap_threshold=t['max_gap_threshold'], val_split=t['val_split'],
                  epochs=job['epochs'], patience=job['patience'], verbose=0, lr=t['lr'], batch_size=t['batch_size'],
                  weight_decay=c['weight_decay'])
        t1 = time.perf_counter()
        pv1 = model.predict(Xv, Av)[:, 0].astype(np.float32)
        t2 = time.perf_counter()
    with open(folder / 'model.pkl', 'wb') as fh:
        pickle.dump(model, fh, protocol=4)
    np.save(folder / 'V1.npy', pv1)
    res = dict(name=job['name'], fit_seconds=t1 - t0, v1_predict_seconds=t2 - t1, tree=tree_record(model),
               torch_threads=torch.get_num_threads(), model_sha256=digest(folder / 'model.pkl'),
               v1_sha256=digest(folder / 'V1.npy'), wall_seconds=time.time() - begin)
    save(folder / 'result.json', res)
    return res


def tsg_predict_job(job):
    """V2 predictions of one selected tree at each evaluation delay, plus the leaf each window reaches."""
    load_tsg()
    folder = Path(job['folder'])
    folder.mkdir(parents=True, exist_ok=False)
    with open(checked(job['model'], job['model_sha256']), 'rb') as fh:
        model = pickle.load(fh)
    res = dict(name=job['name'], delays={})
    with np.load(checked(job['data'], job['data_sha256'])) as z:
        for d in job['delays']:
            X, A = z[f'X_V2_{d}'], z[f'A_V2_{d}']
            pred = model.predict(X, A)[:, 0].astype(np.float32)
            leaf, count = np.unique(model.apply(X, A)[:, 0], return_counts=True)
            np.save(folder / f'V2_{d}.npy', pred)
            res['delays'][str(d)] = dict(sha256=digest(folder / f'V2_{d}.npy'),
                                         leaves={str(int(a)): int(b) for a, b in zip(leaf, count)})
    return res


def mark_started(folder):
    """Start time of a job, written atomically so the parent never reads a half-written file."""
    begin = time.time()
    tmp = Path(folder) / 'started.json.tmp'
    save(tmp, dict(epoch=begin))
    os.replace(tmp, Path(folder) / 'started.json')
    return begin


def remaining_lower_bound(now, kinds, finished, begun, workers):
    """R1-B4: a LOWER bound of the wall time the TSG pool still needs, in seconds.
    Each unfinished job is expected to take the mean duration of the finished jobs of its kind (its candidate), minus
    the time it has already run (begun: job -> start epoch); a kind with no finished job yet counts 0.
    Bound = max(total remaining work / workers, longest remaining job): perfect packing, unseen kinds free.  So a fast
    kind queued behind slow ones never makes the run look longer than it is, and a stop means that even this
    optimistic estimate is over the limit."""
    by_kind = {}
    for name, seconds in finished.items():
        by_kind.setdefault(kinds[name], []).append(float(seconds))
    mean = {k: float(np.mean(v)) for k, v in by_kind.items()}
    left = [max(mean.get(kind, 0.0) - (now - begun[name] if name in begun else 0.0), 0.0)
            for name, kind in kinds.items() if name not in finished]
    return max(sum(left) / workers, max(left, default=0.0))


def check_time_limit(out, limit_s, elapsed, remaining, done, total):
    """Registered stop of the TSG line: elapsed line time + remaining lower bound > limit (three farms together)."""
    projected = elapsed + remaining
    if projected > limit_s:
        save(out / 'technical_limit.json', dict(
            projected_hours=projected / 3600, elapsed_hours=elapsed / 3600, remaining_lower_bound_hours=remaining / 3600,
            limit_hours=limit_s / 3600, done=done, jobs=total,
            rule='elapsed since the start of mode tsg + remaining_lower_bound > tsg.time_limit_hours (all farms)'))
        raise TechnicalLimit(f'TSG projected at least {projected / 3600:.1f} h > limit {limit_s / 3600:.1f} h')
    return projected


def run_pool(args, jobs, fn, out, limit_s=None, started=None):
    """Spawned worker processes (args.workers x args.worker_threads).  With a limit, after every finished job the
    elapsed line time plus remaining_lower_bound is checked against it (check_time_limit)."""
    import concurrent.futures as cf
    import multiprocessing as mp
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[var] = str(args.worker_threads)                 # inherited by the spawned workers
    results, finished = {}, {}
    kinds = {job['name']: job.get('kind', job['name']) for job in jobs}
    pool = cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                                  initializer=worker_init, initargs=(args.worker_threads,))
    try:
        futures = {pool.submit(fn, job): job['name'] for job in jobs}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            name = futures[fut]
            results[name] = fut.result()
            beat(out, f'{fn.__name__} {i}/{len(jobs)} done: {name}')
            if limit_s:
                finished[name] = float(results[name]['wall_seconds'])
                now = time.time()
                begun = {}
                for job in jobs:
                    mark = Path(job['folder']) / 'started.json'
                    if job['name'] not in finished and mark.exists():
                        begun[job['name']] = float(read(mark)['epoch'])
                check_time_limit(out, limit_s, now - started,
                                 remaining_lower_bound(now, kinds, finished, begun, args.workers), i, len(jobs))
    except BaseException:
        for proc in list((getattr(pool, '_processes', None) or {}).values()):
            proc.terminate()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return results


def mode_tsg(args, protocol, x1p, out):
    fa = load_tsg()
    cfg, bud, ref = protocol['tsg'], budget(protocol, args.smoke), protocol['references']['x5']
    delays, nominal, evals = ref['delays'], ref['nominal_delay_hours'], cfg['eval_delays']
    cands = cfg['candidates'][:bud['tsg_candidates']]
    started, store = time.time(), x2.Store(out)
    freeze = dict(line='tsg', guard=guard_controls(), farms={})
    jobs, truth, rated_of = [], {}, {}
    for farm, spec in x2.farms(x1p).items():
        hot, rated = farm == 'HoT', float(spec['rated'])
        rated_of[farm] = rated
        packs, full, units = load_dev(args, farm, spec, protocol, bud)
        save_keys(store, farm, packs)
        # X5's mixed-delay assignment, rebuilt on the FULL train then V1 (same generator and call order as run_x5)
        rng = np.random.default_rng(SEED)
        assign = {s: x5.mix_assignment(full[s], delays, rng) for s in ('train', 'V1')}
        counts = {s: np.bincount(np.searchsorted(delays, a), minlength=len(delays)).tolist() for s, a in assign.items()}
        if counts != read(checked(ROOT / ref['local_dir'] / 'X5' / f'{farm}_mix.json', ref['mix_sha256'][farm])):
            raise ValueError(f'{farm}: rebuilt delay assignment differs from the X5 mix counts')
        n = len(packs['train']['truth'])
        a_tr, a_v1 = assign['train'][:n], assign['V1'][:len(packs['V1']['truth'])]
        rows = np.sort(np.random.default_rng(SEED).choice(n, min(bud['tsg_sample'], n), replace=False))
        rows = rows[np.argsort(np.asarray(packs['train']['issue']).astype('datetime64[ns]')[rows], kind='stable')]
        packs['train'], a_tr = take(packs['train'], rows), a_tr[rows]
        rec = dict(full=full, assignment_counts=counts, sample=len(rows),
                   assignment_sha256={s: hashlib.sha256(a.astype(np.int64).tobytes()).hexdigest() for s, a in assign.items()},
                   sample_keys_sha256=hashlib.sha256(np.ascontiguousarray(packs['train']['issue']).tobytes()
                                                     + np.ascontiguousarray(packs['train']['turbines']).tobytes()).hexdigest(),
                   infos={})
        blocks = gfs_blocks(args, farm, packs, dict(train=delays, V1=delays, V2=evals), protocol, out)
        x5.set_gfs(packs, {'train': x5.mixed_block(blocks['train'], a_tr), 'V1': x5.mixed_block(blocks['V1'], a_v1)})
        truth[farm] = dict(V1=packs['V1']['truth'], V2=packs['V2']['truth'])
        for info in cfg['infos']:
            start = cfg['value_start'][info]
            Xtr, vtr, pos = tsg_view(packs['train'], info, hot, units, start)
            lo, scale = tsg_scaler(Xtr, vtr, pos)
            Ztr, _ = tsg_inputs(Xtr, vtr, pos, lo, scale)
            Zv1, Av1 = tsg_inputs(*tsg_view(packs['V1'], info, hot, units, start)[:2], pos, lo, scale)
            v2 = {}
            for d in evals:
                x5.set_gfs(packs, {'V2': blocks['V2'][d]})
                X2, V2, _ = tsg_view(packs['V2'], info, hot, units, start)
                v2[d] = (*tsg_inputs(X2, V2, pos, lo, scale), V2)
            x5.set_gfs(packs, {'V2': blocks['V2'][nominal]})
            accept = tsg_acceptance(fa, v2, pos)
            rel = f'data/{farm}_{info}.npz'
            path = out / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, X_train=Ztr, Y_train=(packs['train']['truth'] / rated).astype(np.float32), X_V1=Zv1, A_V1=Av1,
                     value_pos=np.asarray(pos), **{f'X_V2_{d}': v2[d][0] for d in evals},
                     **{f'A_V2_{d}': v2[d][1] for d in evals})
            store.registry[rel] = digest(path)
            rec['infos'][info] = dict(features=int(Ztr.shape[1]), value_positions=pos, acceptance=accept,
                                      data_sha256=store.registry[rel])
            beat(out, f'tsg {farm} {info}: data ready, acceptance {accept}')
            for ci, cand in enumerate(cands):
                for h in range(4):
                    jobs.append(dict(name=f'{farm}/{info}/cand{ci}/h{h + 1}', folder=str(out / 'jobs' / farm / info /
                                     f'cand{ci}' / f'h{h + 1}'), data=str(path), data_sha256=store.registry[rel],
                                     horizon=h, candidate=cand, kind=cand['name'], train=cfg['train'],
                                     epochs=bud.get('tsg_epochs', cfg['train']['epochs']),
                                     patience=bud.get('tsg_patience', cfg['train']['patience'])))
        freeze['farms'][farm] = rec
        del packs, blocks
    jobs.sort(key=lambda j: (not j['candidate']['hidden'], j['name']))          # NN trees first (longest)
    limit = None if args.smoke else cfg['time_limit_hours'] * 3600              # all three farms together (R1-B4)
    res = run_pool(args, jobs, tsg_fit_job, out, limit, started)
    pending = []
    for farm, rec in freeze['farms'].items():
        for info, irec in rec['infos'].items():
            rows = []
            for ci, cand in enumerate(cands):
                names = [f'{farm}/{info}/cand{ci}/h{h + 1}' for h in range(4)]
                pv1 = np.stack([np.load(checked(Path(out / 'jobs' / n / 'V1.npy'), res[n]['v1_sha256'])) for n in names],
                               axis=1).astype(np.float64) * rated_of[farm]
                store.put(f'{farm}/{info}/cand{ci}/V1.npy', pv1)
                rows.append(dict(cand, v1_mae=v1_mae(pv1, truth[farm]['V1']), trees={n: res[n] for n in names},
                                 fit_seconds=float(sum(res[n]['fit_seconds'] for n in names))))
            i = choose(rows)
            irec.update(candidates=rows, selected=i)
            for h in range(4):
                name = f'{farm}/{info}/cand{i}/h{h + 1}'
                pending.append(dict(name=name, folder=str(out / 'jobs' / name / 'v2'), delays=evals,
                                    model=str(out / 'jobs' / name / 'model.pkl'), model_sha256=res[name]['model_sha256'],
                                    data=str(out / f'data/{farm}_{info}.npz'), data_sha256=irec['data_sha256']))
    for r in res.values():
        store.registry[f"jobs/{r['name']}/model.pkl"] = r['model_sha256']
    sha = write_freeze(out, freeze)
    pred = run_pool(args, pending, tsg_predict_job, out)
    natural = {}
    for farm, rec in freeze['farms'].items():
        natural[farm] = {}
        for info, irec in rec['infos'].items():
            names = [f'{farm}/{info}/cand{irec["selected"]}/h{h + 1}' for h in range(4)]
            by_d = {}
            for d in evals:
                by_d[d] = np.stack([np.load(checked(out / 'jobs' / n / 'v2' / f'V2_{d}.npy',
                                                    pred[n]['delays'][str(d)]['sha256'])) for n in names],
                                   axis=1).astype(np.float64) * rated_of[farm]
                store.put(f'{farm}/{info}/V2@{d}.npy', by_d[d])
            diff = np.abs(by_d[evals[0]] - by_d[evals[-1]])
            natural[farm][info] = dict(mean_abs_kw=float(diff.mean()), identical_windows=float(np.mean(diff.max(1) == 0)),
                                       leaves={n: pred[n]['delays'] for n in names}, note='logged only (R2-B3)')
    save(out / 'natural_difference.json', natural)
    finish(out, store, dict(v1_freeze_sha256=sha, natural_difference_sha256=digest(out / 'natural_difference.json')))


# ============================================================================ report
def metrics(pred, truth):
    pred, truth = np.asarray(pred, np.float64), np.asarray(truth, np.float64)

    def one(p, t):
        e = p - t
        return dict(mae=float(np.abs(e).mean()), rmse=float(np.sqrt(np.square(e).mean())),
                    r2=float(1 - np.square(e).sum() / np.square(t - t.mean()).sum()))
    return dict(overall=one(pred, truth), by_horizon=[one(pred[:, h], truth[:, h]) for h in range(truth.shape[1])],
                windows=int(len(truth)))


def compare(st, cand, ref, truth, issue):
    """Paired issue-time block bootstrap of the MAE difference; gain = MAE(reference) - MAE(candidate), positive
    favours the candidate.  Identical predictions give gain 0 without a bootstrap (X4 rule)."""
    from utils.kelmarsh_backbone_information import paired_loss_statistics
    cand, ref, truth = (np.asarray(v, np.float64) for v in (cand, ref, truth))
    if not (cand.shape == ref.shape == truth.shape) or cand.ndim != 2:
        raise ValueError('Aligned [window, horizon] predictions required')
    if not (np.isfinite(cand).all() and np.isfinite(ref).all() and np.isfinite(truth).all()):
        raise ValueError('Nonfinite predictions or truth')
    if np.array_equal(cand, ref):
        m = float(np.abs(ref - truth).mean())
        return dict(identical=True, gain=0.0, ci95_low=0.0, ci95_high=0.0, candidate_mean_loss=m, reference_mean_loss=m,
                    windows=int(len(truth)))
    s = paired_loss_statistics(np.abs(cand - truth), np.abs(ref - truth), np.asarray(issue).astype('datetime64[ns]'),
                               block_length=st['block_length'], replicates=st['replicates'], seed=st['seed'])
    return dict(identical=False, p_value=s['newey_west']['p_value_two_sided'],
                **{k: s[k] for k in ('gain', 'ci95_low', 'ci95_high', 'candidate_mean_loss', 'reference_mean_loss',
                                     'windows', 'issue_time_clusters', 'probability_gain_positive')})


def upstream_gain(st, lg, lug, truth, issue):
    """U = MAE(LG) - MAE(LUG) (or MAE(L0) - MAE(LU)): LUG / LU as candidate."""
    return compare(st, lug, lg, truth, issue)


def rise(st, lg6, lug6, lg18, lug18, truth, issue):
    """U(18) - U(6) on the same windows (run_x5.rise_losses)."""
    from utils.kelmarsh_backbone_information import paired_loss_statistics
    e = [np.abs(np.asarray(p, np.float64) - truth) for p in (lg6, lug6, lg18, lug18)]
    cand, ref = x5.rise_losses(*e)
    s = paired_loss_statistics(cand, ref, np.asarray(issue).astype('datetime64[ns]'),
                               block_length=st['block_length'], replicates=st['replicates'], seed=st['seed'])
    return {k: s[k] for k in ('gain', 'ci95_low', 'ci95_high', 'windows', 'issue_time_clusters')}


def upstream(args, mode):
    """A finished wjb line of the same commit, script, protocol and smoke flag; None for a registered TSG technical
    limit (the other lines are still reported)."""
    root = run_dir(args.base, mode, args.smoke)
    me, other = read(run_dir(args.base, 'report', args.smoke) / 'launch.json'), read(root / 'launch.json')
    for key in ('commit', 'script_sha256', 'protocol_sha256', 'smoke'):
        if me[key] != other[key]:
            raise ValueError(f'{mode}: launch identity differs ({key})')
    if other.get('mode') != mode:
        raise ValueError(f'{mode}: launch mode differs')
    if (root / 'failed.json').exists():
        failed = read(root / 'failed.json')
        if mode == 'tsg' and failed.get('type') == 'TechnicalLimit' and (root / 'technical_limit.json').exists():
            return None
        raise ValueError(f'{mode} failed: {failed.get("message")}')
    done = read(root / 'completed.json')
    if done.get('status') != 'COMPLETED':
        raise ValueError(f'{mode} not completed')
    registry = read(checked(root / 'registry.json', done['registry_sha256']))
    return dict(root=root, registry=registry, freeze=read(checked(root / 'v1_freeze.json', done['v1_freeze_sha256'])))


def line_array(line, rel):
    return np.load(checked(line['root'] / rel, line['registry'][rel]))


def line_keys(line, farm):
    rel = f'keys/{farm}_V2.npz'
    with np.load(checked(line['root'] / rel, line['registry'][rel])) as z:
        return {k: z[k] for k in z.files}


class References:
    """Hash-pinned X5 (e01cc6b) and X2 part A (0f24875) V2 predictions; smoke uses their first rows."""

    def __init__(self, args, protocol):
        r5, r2 = protocol['references']['x5'], protocol['references']['x2_a']
        local = ROOT / r5['local_dir']
        done = read(checked(local / 'completed.json', r5['completed_sha256']))
        self.x5_registry = read(checked(local / 'X5' / 'registry.json', done['registry_sha256']))
        self.x5_root = Path(args.logs) / r5['server_dir']
        self.x2_root = Path(args.logs) / r2['server_dir']
        checked(self.x2_root / 'A' / 'completed.json', r2['completed_sha256'])
        self.x2 = x2.Saved(self.x2_root, 'A')
        self.seed = r2['seed']

    def x5(self, rel, n):
        a = np.load(checked(self.x5_root / rel, self.x5_registry[rel]))
        return a[:n]

    def x2a(self, farm, arm, n):
        return self.x2.preds(farm, arm, 'V2', seeds=[self.seed])[0][:n]


def mode_report(args, protocol, x1p, out):
    st = protocol['statistics_smoke' if args.smoke else 'statistics']
    lines = {m: upstream(args, m) for m in ('krr', 'lgbh', 'cabin', 'tsg')}
    if any(lines[m] is None for m in ('krr', 'lgbh', 'cabin')):
        raise ValueError('Only the TSG line may stop as a technical limit')
    refs = References(args, protocol)
    full = {f: lines['krr']['freeze']['farms'][f]['full'] for f in lines['krr']['freeze']['farms']}
    res = dict(note='development-stage comparison on V2 (V1/V2 used in earlier rounds); descriptive, no gate; '
                    'gain = MAE(reference) - MAE(candidate) in kW, positive favours the candidate; single seed 2024',
               statistics=st, farms={}, input_identity={})
    for farm in x2.farms(x1p):
        keys = line_keys(lines['krr'], farm)
        for m, line in lines.items():
            if line is not None and x2.key_bytes(line_keys(line, farm)) != x2.key_bytes(keys):
                raise ValueError(f'{farm}: V2 keys of {m} differ from krr')
        y, issue, n = keys['truth'].astype(np.float64), keys['issue'], len(keys['truth'])
        if not args.smoke and x2.key_bytes(keys) != full[farm]['keys']['V2']:
            raise ValueError(f'{farm}: V2 keys differ from X2 part A')
        r = res['farms'][farm] = dict(windows=n)
        # ---- KRR vs X5 matched-delay LightGBM
        kp = {(i, d): line_array(lines['krr'], f'{farm}/{i}_d{d}/V2.npy') for i in ('LG', 'LUG') for d in (6, 18)}
        xp = {(i, d): refs.x5(f'D{d}/{farm}/{i}_d{d}/2024/V2.npy', n) for i in ('LG', 'LUG') for d in (6, 18)}
        r['krr'] = {f'd{d}': dict(
            **{i: dict(krr=metrics(kp[i, d], y), x5_lightgbm=metrics(xp[i, d], y),
                       krr_vs_x5_lightgbm=compare(st, kp[i, d], xp[i, d], y, issue)) for i in ('LG', 'LUG')},
            U_krr=upstream_gain(st, kp['LG', d], kp['LUG', d], y, issue),
            U_x5_lightgbm=upstream_gain(st, xp['LG', d], xp['LUG', d], y, issue)) for d in (6, 18)}
        r['krr']['rise_krr'] = rise(st, kp['LG', 6], kp['LUG', 6], kp['LG', 18], kp['LUG', 18], y, issue)
        r['krr']['rise_x5_lightgbm'] = rise(st, xp['LG', 6], xp['LUG', 6], xp['LG', 18], xp['LUG', 18], y, issue)
        # ---- Cabin vs LightGBM-hist (same arrays), X2 LightGBM as business reference
        cp = {i: line_array(lines['cabin'], f'{farm}/{i}/V2.npy') for i in ('L0', 'LU')}
        hp = {i: line_array(lines['lgbh'], f'{farm}/{i}/V2.npy') for i in ('L0', 'LU')}
        bp = {i: refs.x2a(farm, protocol['references']['x2_a']['arms'][i], n) for i in ('L0', 'LU')}
        for i in ('L0', 'LU'):
            for split in ('train', 'V1', 'V2'):
                a = lines['cabin']['freeze']['farms'][farm]['infos'][i]['input_digests'][split]
                b = lines['lgbh']['freeze']['farms'][farm]['infos'][i]['input_digests'][split]
                if a != b:
                    raise ValueError(f'{farm} {i} {split}: Cabin and LightGBM-hist inputs differ')
            res['input_identity'][f'{farm}/{i}'] = 'cabin == lgbh (train, V1, V2 x/e digests)'
        r['cabin'] = {i: dict(cabin=metrics(cp[i], y), lightgbm_hist=metrics(hp[i], y), x2_lightgbm=metrics(bp[i], y),
                              cabin_vs_lightgbm_hist=compare(st, cp[i], hp[i], y, issue),
                              lightgbm_hist_vs_x2_lightgbm=compare(st, hp[i], bp[i], y, issue)) for i in ('L0', 'LU')}
        r['cabin'].update(U_cabin=upstream_gain(st, cp['L0'], cp['LU'], y, issue),
                          U_lightgbm_hist=upstream_gain(st, hp['L0'], hp['LU'], y, issue),
                          U_x2_lightgbm=upstream_gain(st, bp['L0'], bp['LU'], y, issue))
        # ---- TSG vs X5 mixed-delay LightGBM
        if lines['tsg'] is None:
            r['tsg'] = dict(status='TECHNICAL_LIMIT', record=read(run_dir(args.base, 'tsg', args.smoke) / 'technical_limit.json'))
            continue
        tp = {(i, d): line_array(lines['tsg'], f'{farm}/{i}/V2@{d}.npy') for i in ('LG', 'LUG') for d in (6, 18)}
        mp = {(i, d): refs.x5(f'X5/{farm}/{i}_mix@{d}/V2.npy', n) for i in ('LG', 'LUG') for d in (6, 18)}
        r['tsg'] = {f'd{d}': dict(
            **{i: dict(tsg=metrics(tp[i, d], y), x5_mix_lightgbm=metrics(mp[i, d], y),
                       tsg_vs_x5_mix_lightgbm=compare(st, tp[i, d], mp[i, d], y, issue)) for i in ('LG', 'LUG')},
            U_tsg=upstream_gain(st, tp['LG', d], tp['LUG', d], y, issue),
            U_x5_mix_lightgbm=upstream_gain(st, mp['LG', d], mp['LUG', d], y, issue)) for d in (6, 18)}
        r['tsg']['rise_tsg'] = rise(st, tp['LG', 6], tp['LUG', 6], tp['LG', 18], tp['LUG', 18], y, issue)
        r['tsg']['rise_x5_mix_lightgbm'] = rise(st, mp['LG', 6], mp['LUG', 6], mp['LG', 18], mp['LUG', 18], y, issue)
        beat(out, f'report {farm} done')
    res['selections'] = {m: (None if line is None else line['freeze']) for m, line in lines.items()}
    save(out / 'results.json', res)
    (out / 'SUMMARY.md').write_text(summary_md(res), encoding='utf-8')
    finish(out, x2.Store(out), dict(results_sha256=digest(out / 'results.json')))


def summary_md(res):
    def ci(c):
        return f"{c['gain']:.2f} [{c['ci95_low']:.2f}, {c['ci95_high']:.2f}]"
    lines = ['# wind-journal-baselines (development stage, V2, seed 2024)', '', res['note'], '',
             '## Nystroem KRR vs X5 LightGBM (same GFS delay)', '',
             '| farm | delay | MAE KRR LG / LUG | MAE LGBM LG / LUG | KRR vs LGBM LG | KRR vs LGBM LUG | U KRR | U LGBM |',
             '|---|---|---|---|---|---|---|---|']
    for farm, r in res['farms'].items():
        for d in (6, 18):
            k = r['krr'][f'd{d}']
            lines.append(f"| {farm} | {d} h | {k['LG']['krr']['overall']['mae']:.1f} / {k['LUG']['krr']['overall']['mae']:.1f}"
                         f" | {k['LG']['x5_lightgbm']['overall']['mae']:.1f} / {k['LUG']['x5_lightgbm']['overall']['mae']:.1f}"
                         f" | {ci(k['LG']['krr_vs_x5_lightgbm'])} | {ci(k['LUG']['krr_vs_x5_lightgbm'])}"
                         f" | {ci(k['U_krr'])} | {ci(k['U_x5_lightgbm'])} |")
        lines.append(f"| {farm} | rise | | | | | {ci(r['krr']['rise_krr'])} | {ci(r['krr']['rise_x5_lightgbm'])} |")
    lines += ['', '## Cabin vs LightGBM-hist (identical inputs); X2 LightGBM as business reference', '',
              '| farm | info | MAE Cabin / LGBM-hist / X2 | Cabin vs LGBM-hist | LGBM-hist vs X2 |', '|---|---|---|---|---|']
    for farm, r in res['farms'].items():
        for i in ('L0', 'LU'):
            c = r['cabin'][i]
            lines.append(f"| {farm} | {i} | {c['cabin']['overall']['mae']:.1f} / {c['lightgbm_hist']['overall']['mae']:.1f}"
                         f" / {c['x2_lightgbm']['overall']['mae']:.1f} | {ci(c['cabin_vs_lightgbm_hist'])}"
                         f" | {ci(c['lightgbm_hist_vs_x2_lightgbm'])} |")
        c = r['cabin']
        lines.append(f"| {farm} | U (L0 - LU) | Cabin {ci(c['U_cabin'])}; LGBM-hist {ci(c['U_lightgbm_hist'])}; "
                     f"X2 {ci(c['U_x2_lightgbm'])} | | |")
    lines += ['', '## TSG adaptive robust vs X5 mixed-delay LightGBM', '',
              '| farm | delay | MAE TSG LG / LUG | MAE LGBM LG / LUG | TSG vs LGBM LG | TSG vs LGBM LUG | U TSG | U LGBM |',
              '|---|---|---|---|---|---|---|---|']
    for farm, r in res['farms'].items():
        if r['tsg'].get('status') == 'TECHNICAL_LIMIT':
            lines.append(f'| {farm} | technical limit | | | | | | |')
            continue
        for d in (6, 18):
            t = r['tsg'][f'd{d}']
            lines.append(f"| {farm} | {d} h | {t['LG']['tsg']['overall']['mae']:.1f} / {t['LUG']['tsg']['overall']['mae']:.1f}"
                         f" | {t['LG']['x5_mix_lightgbm']['overall']['mae']:.1f} / {t['LUG']['x5_mix_lightgbm']['overall']['mae']:.1f}"
                         f" | {ci(t['LG']['tsg_vs_x5_mix_lightgbm'])} | {ci(t['LUG']['tsg_vs_x5_mix_lightgbm'])}"
                         f" | {ci(t['U_tsg'])} | {ci(t['U_x5_mix_lightgbm'])} |")
        lines.append(f"| {farm} | rise | | | | | {ci(r['tsg']['rise_tsg'])} | {ci(r['tsg']['rise_x5_mix_lightgbm'])} |")
    return '\n'.join(lines) + '\n'


# ============================================================================ main
MODES = {'krr': mode_krr, 'lgbh': mode_lgbh, 'tsg': mode_tsg, 'cabin': mode_cabin, 'report': mode_report}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=list(MODES), required=True)
    p.add_argument('--logs', type=Path, required=True, help='<data root>/outputs/logs')
    p.add_argument('--base', type=Path, required=True, help='<logs>/WJB/wjb_<commit>')
    p.add_argument('--protocol-sha256', required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--threads', type=int, default=8, help='LightGBM / torch threads of the parent process')
    p.add_argument('--workers', type=int, default=8, help='TSG worker processes')
    p.add_argument('--worker-threads', type=int, default=1, help='torch / BLAS threads per TSG worker')
    p.add_argument('--device', default='auto')
    p.add_argument('--gpu-fraction', type=float, default=None, help='default: protocol cabin.gpu_fraction')
    p.add_argument('--allow-dirty', action='store_true', help='tests only')
    args = p.parse_args(argv)
    out, protocol, x1p = start_run(args, args.mode)
    try:
        with dev_only():
            MODES[args.mode](args, protocol, x1p, out)
    except BaseException as exc:
        x2.fail(out, exc)
        raise


if __name__ == '__main__':
    main()
