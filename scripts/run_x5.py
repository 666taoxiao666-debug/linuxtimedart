"""X5 block 3: how much are upstream observations worth on top of GFS when the GFS feed is late?  (LightGBM, CPU)

Author: Claude (2026-11-21).  User split: Claude writes the code, GPT reviews, Cursor launches.
Plan: docs/STAGE_PLAN.md (X5); protocol: configs/x5_protocol.json (its SHA256 is passed on the command line).

For a release delay d the forecast used at issue t0 is the latest cycle C with C + d <= t0 (cycles every 6 h), so
age = t0 - C lies in [d, d + 5].  Only the last-step GFS columns 20..48 of the frozen x50 (the only GFS inputs the
trees read: run_x2.tree_matrix) are rebuilt, from the SAME point caches the frozen preparations used, through the
SAME prepare_g1_batches.interpolate_leads / nwp_input code; leads outside [3, 18] h stay masked.  For d = 6 the
rebuilt columns must reproduce the frozen x50 of every window before anything is fitted (nominal binding).

Modes: smoke (tiny slices, 2 delays, minimal budget, never reported) and run (formal; needs a passed smoke of the
same commit).  Every run writes into a FRESH --output directory and leaves failed.json on any error (no retries).
Windows, labels and splits are exactly X2 part A's; nothing is repaired; the test years are exploratory.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_x1 as x1  # noqa: E402
from scripts import run_x2 as x2  # noqa: E402
from scripts.run_x4 import spearman  # registered average ranks, including ties

digest, read, save, checked = x1.digest, x1.read, x1.save, x1.checked
PROTOCOL = ROOT / 'configs/x5_protocol.json'
GFS_COLS = slice(20, 49)                     # x50 last-step GFS block: history, 13 current, 13 masks, age, missing
SEED = 2024
X2_COHORTS = ROOT / 'output/x2/x2_0f24875/a/A'


def prep():
    from scripts import prepare_g1_batches as p
    return p


# ----------------------------------------------------------------------------- delayed GFS columns (numpy)
def delayed_cycle(t0, delay):
    """Latest 6-hourly cycle C with C + delay <= t0 (t0 in whole hours); delay 6 = prepare_g1_batches.available_cycle."""
    h = np.asarray(t0, dtype='datetime64[h]').astype(np.int64)
    return ((h - int(delay)) // 6 * 6).astype('datetime64[h]')


def cycle_key(cycle):
    return str(np.datetime64(cycle, 'h')).replace('-', '').replace('T', '')


def delayed_hourly(issue_hours, cache, delay, mean, scale, cutoff, nsites):
    """Hourly GFS arrays (prepare_g1_batches/run_x1.cached_hourly layout) for the unique issue hours at a delay.
    A cycle whose 18 h container reaches `cutoff` is never read (treated as missing, as the frozen preparations do)."""
    p = prep()
    hours = np.asarray(issue_hours, dtype='datetime64[h]')
    cycles = delayed_cycle(hours, delay)
    age = ((hours - cycles) / np.timedelta64(1, 'h')).astype(int)
    current = np.zeros((len(hours), 13, nsites), np.float32)
    masks = np.zeros_like(current)
    historical = np.full((len(hours), nsites), np.nan)
    ok = np.zeros(len(hours))
    limit = np.datetime64(cutoff, 'h')
    reads = []
    for i, (cycle, a) in enumerate(zip(cycles, age)):
        if cycle + np.timedelta64(18, 'h') >= limit:
            continue
        values = cache.get(cycle_key(cycle))
        if values is None:
            continue
        hist, valid, _ = p.interpolate_leads(values, np.array([a]))
        historical[i] = np.where(valid[0], hist[0], np.nan)
        current[i], masks[i], audit = p.interpolate_leads(values, a + np.arange(-5, 8))
        ok[i] = np.isfinite(values).all()
        if audit['min_read_lead'] is not None:
            reads.append((audit['min_read_lead'], audit['max_read_lead']))
    if reads and (min(r[0] for r in reads) < 3 or max(r[1] for r in reads) > 18):
        raise ValueError('GFS read outside [3, 18] h')
    return dict(historical=historical, current=current, masks=masks, age=age, cycle_ok=ok,
                nwp_mean=mean, nwp_scale=scale), dict(delay=int(delay), age_min=int(age.min()), age_max=int(age.max()),
                                                     cycles_used=int(ok.sum()), issues=int(len(hours)))


def gfs_block(issue_ns, cache, delay, mean, scale, cutoff, nsites):
    """[n_windows, 29] float32 = x50[:, -1, 20:49] at the given delay, built by prepare_g1_batches.nwp_input itself."""
    p = prep()
    issue = np.asarray(issue_ns).astype('datetime64[ns]').astype('datetime64[h]')
    uniq, inv = np.unique(issue, return_inverse=True)
    hourly, info = delayed_hourly(uniq, cache, delay, mean, scale, cutoff, nsites)
    dummy = np.zeros((len(uniq), 1, 2), np.float32)
    block = p.nwp_input(dummy, hourly, np.arange(len(uniq))[:, None])[:, 0, 1:30]
    return block[inv], info


def set_gfs(packs, blocks):
    """Overwrite, IN PLACE, the x50 last-step GFS columns of each named pack (HoT train x50 is ~2 GB, so no copies);
    every other input is untouched.  The frozen block is kept separately and restored with the same call."""
    for split, block in blocks.items():
        x = packs[split]['x50']
        if block.shape != (len(x), 29):
            raise ValueError('GFS block does not align with the pack')
        x[:, -1, GFS_COLS] = block


def mix_assignment(n, delays, rng):
    return np.asarray(delays)[rng.integers(0, len(delays), n)]


def mixed_block(blocks, assignment):
    out = np.empty_like(next(iter(blocks.values())))
    for d, b in blocks.items():
        take = assignment == d
        out[take] = b[take]
    return out


def rise_losses(e_lg_a, e_lug_a, e_lg_b, e_lug_b):
    """Paired losses whose 'gain' (reference - candidate) equals U(b) - U(a), U(d) = |e_LG(d)| - |e_LUG(d)|."""
    return e_lug_b + e_lg_a, e_lg_b + e_lug_a


# ----------------------------------------------------------------------------- caches per farm
def farm_caches(logs, farm):
    """(dev cache, test cache, mean, scale, dev cutoff, test cutoff, identities) from the frozen preparations."""
    ident = {}
    if farm in ('Kelmarsh', 'Penmanshiel'):
        folder = logs / 'G1/g1c_27312d7_r2/prepared' / farm
        meta = read(folder / 'prepared.json')
        if meta != read(ROOT / 'output/g1/preparation_78/prepared' / farm / 'prepared.json'):
            raise ValueError('Preparation identity differs')
        cache = {}
        for name in ('points_0.npz', 'points_1.npz'):
            path = folder / name
            ident[str(path)] = meta['inputs'][str(path)]
            cache.update(x1.load_pack(checked(path, ident[str(path)])))
        return cache, cache, float(meta['nwp_mean']), float(meta['nwp_scale']), '2024-01-01', '2024-01-01', ident
    stats = read(logs / 'G2/g2b_4471499_shared_r1/formal/fit_statistics.json')
    folder = logs / 'G2/hot_dev_f58f2ce_r5/prepared'
    dev = {}
    hashes = read(ROOT / 'output/g2/hot_dev/results82/completed.json')['artifacts']
    for name in ('points_0.npz', 'points_1.npz'):
        path = folder / name
        ident[str(path)] = hashes[name]
        dev.update(x1.load_pack(checked(path, hashes[name])))
    sealed = logs / 'G2/g2b84_bc7730e_r1/sealed_prep/points.npz'
    identities = read(ROOT / 'output/x3/x3_f8af1b6/q1/history_binding.json')['farms']['HoT']['HoT_2024']['identities']
    ident[str(sealed)] = identities[str(sealed)]
    test = x1.load_pack(checked(sealed, ident[str(sealed)]))
    return dev, test, float(stats['nwp_mean']), float(stats['nwp_scale']), '2024-01-01', '2026-01-01', ident


# ----------------------------------------------------------------------------- run
def git_identity():
    return x2.git_identity()


def start_run(args, mode):
    checked(PROTOCOL, args.protocol_sha256)
    commit, dirty = git_identity()
    if not args.allow_dirty and (commit is None or dirty):
        raise ValueError('Run from a clean worktree of the reviewed commit')
    if mode == 'RUN':
        smoke = read(Path(args.smoke_result) / 'smoke_passed.json') if args.smoke_result else {}
        if smoke.get('status') != 'PASS' or smoke.get('commit') != commit:
            raise ValueError('Formal run needs --smoke-result of a passed smoke run of this commit')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / '.gitignore').write_text('*.npy\n*.npz\nmodels/\n', encoding='utf-8')
    save(out / 'launch.json', dict(mode=mode, commit=commit, script_sha256=digest(__file__),
         protocol_sha256=args.protocol_sha256, python=sys.executable, pid=os.getpid(),
         started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), args={k: str(v) for k, v in vars(args).items()}))
    return out


def beat(out, message):
    save(out / 'heartbeat.json', dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), message=message))
    print('[X5]', message, flush=True)


def arm(info):
    return {'info': info, 'extra': None, 'seeds': [SEED]}


def prepare_farm(args, out, farm, spec, x2p, protocol, bud, delays, audit_suffix):
    logs = Path(args.logs)
    train, v1, v2, tests, counts = x2.load_farm(args, farm, spec, x2p, bud)
    if not bud['smoke']:
        expected = read(X2_COHORTS / f'{farm}_cohorts.json')['keys']
        got = {k: x2.key_bytes(p) for k, p in dict(V1=v1, V2=v2, **tests).items()}
        if got != expected:
            raise ValueError(f'{farm}: cohorts differ from X2 part A')
    dev, test, mean, scale, dev_cut, test_cut, ident = farm_caches(logs, farm)
    nsites = next(iter(dev.values())).shape[-1]
    packs = dict(train=train, V1=v1, V2=v2, **tests)
    nominal = protocol['nominal_delay_hours']
    blocks, audit = {}, {}
    for split, pack in packs.items():
        cache, cut = (test, test_cut) if split in tests else (dev, dev_cut)
        blocks[split] = {}
        for d in delays:
            blocks[split][d], audit[f'{split}@{d}'] = gfs_block(pack['issue'], cache, d, mean, scale, cut, nsites)
        frozen = pack['x50'][:, -1, GFS_COLS]
        diff = float(np.abs(blocks[split][nominal] - frozen).max())
        exact = float(np.mean(blocks[split][nominal] == frozen))
        audit[f'{split}@binding'] = dict(max_abs_diff=diff, exact_fraction=exact)
        if not diff <= 1e-5:
            raise ValueError(f'{farm} {split}: rebuilt nominal GFS columns differ from the frozen x50 ({diff})')
        blocks[split][nominal] = frozen.copy()          # use the frozen values themselves at the nominal delay
    save(out / 'X5' / f'{farm}_{audit_suffix}.json', dict(counts=counts, caches=ident, mean=mean, scale=scale, audit=audit))
    beat(out, f'{farm}: nominal binding passed, delays {delays}')
    return packs, list(tests), blocks


def run_farm(lgb, args, out, store, farm, spec, x1p, x2p, protocol, bud, delays):
    packs, tests, blocks = prepare_farm(args, out, farm, spec, x2p, protocol, bud, delays, 'audit')
    nominal = protocol['nominal_delay_hours']
    hot = farm == 'HoT'
    curves = {}
    for d in delays:
        arms = {f'LG_d{d}': arm('LG'), f'LUG_d{d}': arm('LUG')}
        if d == nominal:
            arms['LU'] = arm('LU')
        set_gfs(packs, {s: blocks[s][d] for s in packs})
        x2.run_a_farm(lgb, farm, spec['rated'], packs['train'], packs['V1'], packs['V2'], {k: packs[k] for k in tests},
                      x2p, x1p, store, bud, args.threads, out, curves, {}, part=f'D{d}', arms=arms)
        beat(out, f'{farm}: delay {d} fitted')
    rng = np.random.default_rng(SEED)
    assign = {s: mix_assignment(len(packs[s]['truth']), delays, rng) for s in ('train', 'V1')}
    set_gfs(packs, {s: mixed_block(blocks[s], assign[s]) for s in ('train', 'V1')})
    set_gfs(packs, {'V2': blocks['V2'][nominal]})
    x2.run_a_farm(lgb, farm, spec['rated'], packs['train'], packs['V1'], packs['V2'], {}, x2p, x1p, store, bud,
                  args.threads, out, curves, {}, part='MIX', arms={'LG_mix': arm('LG'), 'LUG_mix': arm('LUG')})
    save(out / 'X5' / f'{farm}_mix.json', {s: np.bincount(np.searchsorted(delays, a), minlength=len(delays)).tolist()
                                            for s, a in assign.items()})
    del packs['train']
    beat(out, f'{farm}: mix fitted')
    # cross-delay predictions: nominal models at other delays (mismatch) and the mix models at every delay
    models = {name: (part, read(out / part / farm / 'dev_freeze.json')['arms'][name]['seeds'][str(SEED)]['boosters'])
              for part, name in ((f'D{nominal}', f'LG_d{nominal}'), (f'D{nominal}', f'LUG_d{nominal}'),
                                 ('MIX', 'LG_mix'), ('MIX', 'LUG_mix'))}
    for split in ['V2'] + list(tests):
        for d in delays:
            set_gfs(packs, {split: blocks[split][d]})
            for name, (part, hashes) in models.items():
                X = x2.tree_matrix(packs[split], 'LUG' if name.startswith('LUG') else 'LG', hot)
                pred = x2.booster_predict(lgb, out / part / farm / 'models' / name / str(SEED), hashes, X)
                store.put(f'X5/{farm}/{name}@{d}/{split}.npy', pred)
        set_gfs(packs, {split: blocks[split][nominal]})
    beat(out, f'{farm}: cross-delay predictions done')
    return {s: dict(truth=packs[s]['truth'], issue=packs[s]['issue']) for s in ['V2'] + list(tests)}


def report(out, keys, delays, protocol, farms_order):
    from utils.kelmarsh_backbone_information import paired_loss_statistics
    st = protocol['statistics']
    kw = dict(block_length=st['block_length'], replicates=st['replicates'], seed=st['seed'])
    nominal, longest = protocol['nominal_delay_hours'], max(delays)
    res = {'delays': delays, 'farms': {}}
    for farm in farms_order:
        res['farms'][farm] = {}
        for split, k in keys[farm].items():
            truth, start = k['truth'].astype(np.float64), k['issue'].astype('datetime64[ns]')
            load = lambda rel: np.load(out / rel).astype(np.float64)  # noqa: E731
            err = lambda pred: np.abs(truth - pred)  # noqa: E731
            e_lu = err(load(f'D{nominal}/{farm}/LU/{SEED}/{split}.npy'))
            rows, e = {}, {}
            for d in delays:
                e[d] = {n: err(load(f'D{d}/{farm}/{n}_d{d}/{SEED}/{split}.npy')) for n in ('LG', 'LUG')}
                cross = {n: err(load(f'X5/{farm}/{n}@{d}/{split}.npy'))
                         for n in (f'LG_d{nominal}', f'LUG_d{nominal}', 'LG_mix', 'LUG_mix')}
                rows[str(d)] = dict(
                    mae=dict(LU=float(e_lu.mean()), LG=float(e[d]['LG'].mean()), LUG=float(e[d]['LUG'].mean()),
                             **{f'{n}@{d}': float(v.mean()) for n, v in cross.items()}),
                    U_matched=paired_loss_statistics(e[d]['LUG'], e[d]['LG'], start, **kw),
                    G_matched=paired_loss_statistics(e[d]['LUG'], e_lu, start, **kw),
                    U_mismatch=paired_loss_statistics(cross[f'LUG_d{nominal}'], cross[f'LG_d{nominal}'], start, **kw),
                    U_mix=paired_loss_statistics(cross['LUG_mix'], cross['LG_mix'], start, **kw),
                    mix_vs_matched=paired_loss_statistics(cross['LUG_mix'], e[d]['LUG'], start, **kw),
                    U_by_h=(e[d]['LG'] - e[d]['LUG']).mean(0).tolist())
            cand, ref = rise_losses(e[nominal]['LG'], e[nominal]['LUG'], e[longest]['LG'], e[longest]['LUG'])
            u = [rows[str(d)]['U_matched']['gain'] for d in delays]
            res['farms'][farm][split] = dict(role='primary' if split == 'V2' else 'exploratory', delays=rows,
                                             U_rise=paired_loss_statistics(cand, ref, start, **kw),
                                             spearman_U_vs_delay=spearman(delays, u) if len(delays) > 2 else None)
    v2 = [res['farms'][f]['V2']['U_rise'] for f in farms_order]
    passed = sum(r['gain'] > 0 and r['ci95_low'] > 0 for r in v2)
    res['H_latency'] = dict(farms_supported_on_V2=passed, of=len(v2),
                            outcome='SUPPORTED' if passed >= 2 else 'NOT_SUPPORTED', descriptive=True)
    return res


def summary_md(res):
    lines = ['# X5 block 3: upstream value under GFS latency (LightGBM, seed 2024)', '',
             f"H_latency on V2: {res['H_latency']['outcome']} ({res['H_latency']['farms_supported_on_V2']}/"
             f"{res['H_latency']['of']} farms). V2 = development; test years exploratory.", '',
             '| farm | split | delay | MAE LU / LG / LUG | U matched [CI] | G matched | U mismatch | U mix | LUG_mix - LUG |',
             '|---|---|---|---|---|---|---|---|---|']
    for farm, splits in res['farms'].items():
        for split, r in splits.items():
            for d, row in r['delays'].items():
                m, u = row['mae'], row['U_matched']
                lines.append(f"| {farm} | {split} | {d} | {m['LU']:.1f} / {m['LG']:.1f} / {m['LUG']:.1f} | {u['gain']:.2f} "
                             f"[{u['ci95_low']:.2f}, {u['ci95_high']:.2f}] | {row['G_matched']['gain']:.2f} | "
                             f"{row['U_mismatch']['gain']:.2f} | {row['U_mix']['gain']:.2f} | {-row['mix_vs_matched']['gain']:.2f} |")
            ur = r['U_rise']
            lines.append(f"| {farm} | {split} | rise | | {ur['gain']:.2f} [{ur['ci95_low']:.2f}, {ur['ci95_high']:.2f}] | "
                         f"spearman {r['spearman_U_vs_delay']} | | | |")
    return '\n'.join(lines) + '\n'


def run(args, mode):
    import lightgbm as lgb
    if lgb.__version__ != '4.5.0':
        raise ValueError('LightGBM fits require the timedart environment with lightgbm 4.5.0')
    out = start_run(args, mode)
    try:
        protocol, x1p, x2p = read(PROTOCOL), read(ROOT / 'configs/x1_protocol.json'), read(ROOT / 'configs/x2_protocol.json')
        smoke = mode == 'SMOKE'
        bud = dict(protocol['budget']['smoke' if smoke else 'formal'], smoke=smoke)
        delays = sorted(bud.get('delays_hours', protocol['delays_hours']))
        if protocol['nominal_delay_hours'] not in delays:
            raise ValueError('Nominal delay must be evaluated')
        store = x2.Store(out)
        (out / 'X5').mkdir(parents=True, exist_ok=True)
        keys, order = {}, []
        farms = x2.farms(x1p)
        # Discard each farm's packs after checking: the HoT train pack alone is ~2 GB.
        for farm, spec in farms.items():
            prepare_farm(args, out, farm, spec, x2p, protocol, bud, [protocol['nominal_delay_hours']], 'preflight')
        for farm, spec in farms.items():
            keys[farm] = run_farm(lgb, args, out, store, farm, spec, x1p, x2p, protocol, bud, delays)
            order.append(farm)
        save(out / 'X5' / 'registry.json', store.registry)
        res = report(out, keys, delays, protocol, order)
        res.update(commit=git_identity()[0], protocol_sha256=args.protocol_sha256, smoke=smoke)
        save(out / 'X5' / 'results.json', res)
        (out / 'X5' / 'SUMMARY.md').write_text(summary_md(res), encoding='utf-8')
        save(out / 'completed.json', dict(status='COMPLETED', results_sha256=digest(out / 'X5' / 'results.json'),
                                          registry_sha256=digest(out / 'X5' / 'registry.json')))
        if smoke:
            save(out / 'smoke_passed.json', dict(status='PASS', interpret_results=False, commit=git_identity()[0]))
        beat(out, 'completed')
    except Exception as exc:
        x2.fail(out, exc)
        raise
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['smoke', 'run'], required=True)
    p.add_argument('--logs', type=Path, required=True, help='<data root>/outputs/logs')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--protocol-sha256', required=True)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--smoke-result', type=Path)
    p.add_argument('--allow-dirty', action='store_true')
    args = p.parse_args(argv)
    run(args, 'SMOKE' if args.mode == 'smoke' else 'RUN')


if __name__ == '__main__':
    main()
