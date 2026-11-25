"""X5 block 3: how much are upstream observations worth on top of GFS when the GFS feed is late?  (LightGBM, CPU)

Author: Claude (2026-11-21).  User split: Claude writes the code, GPT reviews, Cursor launches.
Plan: docs/STAGE_PLAN.md (X5); protocol: configs/x5_protocol.json (its SHA256 is passed on the command line).

For a release delay d the forecast used at issue t0 is the latest cycle C with C + d <= t0 (cycles every 6 h), so
age = t0 - C lies in [d, d + 5].  Only the last-step GFS columns 20..48 of the frozen x50 (the only GFS inputs the
trees read: run_x2.tree_matrix) are rebuilt, from the SAME point caches the frozen preparations used, through the
SAME prepare_g1_batches.interpolate_leads / nwp_input code; leads outside [3, 18] h stay masked.  For d = 6 the
rebuilt columns must reproduce the frozen x50 of every window before anything is fitted (nominal binding).

Modes: smoke (tiny slices, 2 delays, minimal budget, never reported) and run (formal; needs a passed smoke of the
same commit).  X5b (2026-11-23, configs/x5b_protocol.json): msmoke / mrun, the matched-coverage check of a fresh
(6 h) against an old (15 h) cycle, see the X5b section below; mrun needs a passed msmoke of the same commit.  Every run writes into a FRESH --output directory and leaves failed.json on any error (no retries).
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


def start_run(args, mode, protocol_path=PROTOCOL):
    checked(protocol_path, args.protocol_sha256)
    commit, dirty = git_identity()
    if not args.allow_dirty and (commit is None or dirty):
        raise ValueError('Run from a clean worktree of the reviewed commit')
    if mode in ('RUN', 'MRUN'):
        smoke = read(Path(args.smoke_result) / 'smoke_passed.json') if args.smoke_result else {}
        kind = 'match' if mode == 'MRUN' else None          # X5 smoke receipts carry no kind
        if smoke.get('status') != 'PASS' or smoke.get('commit') != commit or smoke.get('kind') != kind:
            raise ValueError('Formal run needs --smoke-result of a passed smoke run of the same kind and commit')
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


# ----------------------------------------------------------------------------- X5b: matched coverage, 6 h vs 15 h
# Claude 2026-11-23 (GPT chose option B; post-hoc development check, 15 h picked after seeing X5).
# States per window (same issue, same turbine, same targets), built only from blocks X5 already rebuilds:
#   A  = fresh cycle (d = 6) as is            C  = old cycle (d = 15) as is
#   D  = A restricted to the positions that are STRUCTURALLY available for C (lead = age + offset in [3, 18] h)
#   Cc = C restricted to the positions that are structurally available for A
# D and Cc therefore share one structural coverage; each keeps its own age, cycle-missing flag and natural gaps.
# R1 review (2026-11-23): natural gaps may still differ between D and Cc -> every split is audited window by window,
# the cycle contrast is also given on agreeing windows only and the labels use that contrast (R1-B1); the labels are
# neutral descriptions and "within bound" needs the WHOLE CI inside +-0.5 x |truncation point estimate| (R1-B2);
# validity bits outside the registered structure stop the run (R1-S1).  R2-B1: the agreeing-window contrast and its
# label are NOT_ESTIMABLE below classification.min_issue_clusters (40 = 5 block lengths) issue-time clusters.
PROTOCOL_B = ROOT / 'configs/x5b_protocol.json'
OFFSETS = np.arange(-5, 8)                     # current-cycle offsets t0-5 .. t0+7 (nwp_input)
TARGETS = slice(6, 10)                         # offsets +1 .. +4 inside OFFSETS
B_HIST, B_FC, B_MASK, B_AGE, B_MISS = 0, slice(1, 14), slice(14, 27), 27, 28
STATES = ('A', 'C', 'Cc', 'D')


def structural_masks(block):
    """Availability implied by the issue age alone, ignoring missing files: [n, 13] for the current-cycle offsets
    (lead = age + offset) and [n] for the history value (lead = age); both require 3 <= lead <= 18 h."""
    age = np.asarray(block[:, B_AGE], dtype=np.float64)
    if not np.array_equal(age, np.round(age)):
        raise ValueError('Age column is not in whole hours')
    lead = age[:, None] + OFFSETS[None, :]
    return (lead >= 3) & (lead <= 18), (age >= 3) & (age <= 18)


def match_coverage(own, other):
    """`own` restricted to the positions structurally available for `other` in the SAME window.  Forecast values
    and validity bits are zeroed together; the history value is zeroed where `other` has no admissible history lead;
    age and the cycle-missing flag stay `own`'s, and own natural gaps (missing files) stay as they are."""
    own, other = np.asarray(own), np.asarray(other)
    if own.shape != other.shape or own.ndim != 2 or own.shape[1] != 29:
        raise ValueError('Blocks must be aligned [n, 29]')
    fc, hist = structural_masks(other)
    out = own.copy()
    out[:, B_FC] = np.where(fc, own[:, B_FC], 0)
    out[:, B_MASK] = np.where(fc, own[:, B_MASK], 0)
    out[:, B_HIST] = np.where(hist, own[:, B_HIST], 0)
    return out


def matched_states(fresh_block, old_block):
    return {'A': fresh_block, 'C': old_block, 'Cc': match_coverage(old_block, fresh_block),
            'D': match_coverage(fresh_block, old_block)}


def natural_gap_agreement(states):
    """[n] bool: D and Cc have the SAME actual availability inside their common structure (every current-cycle
    validity bit and the cycle-missing flag agree).  Outside the common structure both are 0 by construction, so a
    disagreement can only come from a natural gap (missing file / cycle) on one side.  The history value has no
    validity bit; its natural gaps follow the cycle-missing flag, which is compared."""
    vd, vc = states['D'][:, B_MASK] > 0, states['Cc'][:, B_MASK] > 0
    return (vd == vc).all(1) & (states['D'][:, B_MISS] == states['Cc'][:, B_MISS])


def coverage_audit(states):
    """Per state: share of (window, target hour +1..+4) with a valid forecast, share of windows with no valid target,
    mean age; natural gaps (structurally admissible but invalid) for A and C; how often Cc / D differ from C / A;
    window-level D-vs-Cc disagreement of the actual availability inside the common structure (X5b R1-B1)."""
    out = {}
    for name, b in states.items():
        valid = b[:, B_MASK] > 0
        row = dict(target_valid=float(valid[:, TARGETS].mean()), windows_no_target=float((~valid[:, TARGETS]).all(1).mean()),
                   age_mean=float(b[:, B_AGE].mean()), cycle_missing=float(b[:, B_MISS].mean()))
        if name in ('A', 'C'):
            fc, _ = structural_masks(b)
            row['natural_gap_targets'] = float((fc & ~valid)[:, TARGETS].mean())
        out[name] = row
    common = structural_masks(states['A'])[0] & structural_masks(states['C'])[0]
    valid = {n: states[n][:, B_MASK] > 0 for n in STATES}
    out['Cc_windows_differ_from_C'] = float((states['Cc'] != states['C']).any(1).mean())
    out['D_windows_differ_from_A'] = float((states['D'] != states['A']).any(1).mean())
    out['masks_within_own_structure'] = bool(all((valid[n] <= structural_masks(states[n])[0]).all() for n in STATES))
    out['matched_within_common_structure'] = bool((valid['D'] <= common).all() and (valid['Cc'] <= common).all())
    agree = natural_gap_agreement(states)
    out['natural_gap'] = dict(
        windows=int(len(agree)), windows_disagree=int((~agree).sum()),
        windows_mask_disagree=int((valid['D'] != valid['Cc']).any(1).sum()),
        windows_target_mask_disagree=int((valid['D'][:, TARGETS] != valid['Cc'][:, TARGETS]).any(1).sum()),
        windows_cycle_status_disagree=int((states['D'][:, B_MISS] != states['Cc'][:, B_MISS]).sum()),
        cycle_missing_D_only=int(((states['D'][:, B_MISS] == 1) & (states['Cc'][:, B_MISS] == 0)).sum()),
        cycle_missing_Cc_only=int(((states['D'][:, B_MISS] == 0) & (states['Cc'][:, B_MISS] == 1)).sum()))
    return out


def check_structure(audit, farm, split):
    """X5b R1-S1: a validity bit outside its own or the common structure is an input error, not a finding."""
    if not (audit['masks_within_own_structure'] and audit['matched_within_common_structure']):
        raise ValueError(f'{farm} {split}: validity bits outside the registered structure')


def run_farm_match(lgb, args, out, store, farm, spec, x1p, x2p, protocol, bud):
    fresh, old = protocol['fresh_delay_hours'], protocol['old_delay_hours']
    packs, tests, blocks = prepare_farm(args, out, farm, spec, x2p, protocol, bud, [fresh, old], 'audit')
    states = {s: matched_states(blocks[s][fresh], blocks[s][old]) for s in packs}
    audits = {s: coverage_audit(v) for s, v in states.items()}
    save(out / 'X5' / f'{farm}_coverage.json', audits)
    for s, a in audits.items():
        check_structure(a, farm, s)
    for s in ['V2'] + list(tests):
        store.put(f'X5/{farm}/agree/{s}.npy', natural_gap_agreement(states[s]))
    curves = {}
    for name in STATES:
        set_gfs(packs, {s: states[s][name] for s in packs})
        x2.run_a_farm(lgb, farm, spec['rated'], packs['train'], packs['V1'], packs['V2'], {k: packs[k] for k in tests},
                      x2p, x1p, store, bud, args.threads, out, curves, {}, part=f'S_{name}',
                      arms={f'LG_{name}': arm('LG'), f'LUG_{name}': arm('LUG')})
        beat(out, f'{farm}: state {name} fitted')
    return {s: dict(truth=packs[s]['truth'], issue=packs[s]['issue']) for s in ['V2'] + list(tests)}


def bounded_contrast(e, a, b, start, rows, kw, min_clusters):
    """U(b) - U(a) on the windows selected by the bool mask `rows`.  With fewer than `min_clusters` issue-time clusters
    the moving-block bootstrap has too few candidate blocks (X5b R2-B1: 8 clusters, block 8 -> one block, zero-width CI)
    and the DM test needs >= 3 values, so only the window count, cluster count and the point estimate are reported."""
    from utils.kelmarsh_backbone_information import paired_loss_statistics
    rows = np.asarray(rows, dtype=bool)
    cand, ref = rise_losses(e[a]['LG'][rows], e[a]['LUG'][rows], e[b]['LG'][rows], e[b]['LUG'][rows])
    clusters = int(len(np.unique(start[rows])))
    if clusters >= min_clusters:
        return paired_loss_statistics(cand, ref, start[rows], **kw)
    gain = float((ref.mean(1) - cand.mean(1)).mean()) if len(cand) else None
    return dict(status='NOT_ESTIMABLE', gain=gain, ci95_low=None, ci95_high=None, windows=int(len(cand)),
                issue_time_clusters=clusters, min_issue_clusters=int(min_clusters))


def classify_match(truncation, cycle, ratio):
    """Pre-registered descriptive labels for one farm on V2 (X5b R1-B2: the WHOLE interval must lie in the bound).
    bound = ratio x |U(D) - U(A)| point estimate.  cycle labels: CI_ABOVE_0 / CI_BELOW_0 / CI_WITHIN_BOUND (the whole
    95% CI inside [-bound, +bound]) / CI_WIDE.  `cycle` is the contrast on windows where D and Cc agree (R1-B1)."""
    bound = ratio * abs(truncation['gain'])
    if cycle is None or cycle.get('status') == 'NOT_ESTIMABLE':
        label = 'NOT_ESTIMABLE'
    elif cycle['ci95_low'] > 0:
        label = 'CI_ABOVE_0'
    elif cycle['ci95_high'] < 0:
        label = 'CI_BELOW_0'
    elif -bound <= cycle['ci95_low'] and cycle['ci95_high'] <= bound:
        label = 'CI_WITHIN_BOUND'
    else:
        label = 'CI_WIDE'
    return dict(truncation_ci_above_0=bool(truncation['gain'] > 0 and truncation['ci95_low'] > 0), cycle=label,
                bound_kw=float(bound), bound_relative_to=f'{ratio} x |U(D) - U(A)| point estimate')


def report_match(out, keys, protocol, farms_order):
    from utils.kelmarsh_backbone_information import paired_loss_statistics
    st = protocol['statistics']
    kw = dict(block_length=st['block_length'], replicates=st['replicates'], seed=st['seed'])
    res = {'states': list(STATES), 'farms': {}}
    for farm in farms_order:
        res['farms'][farm] = {}
        for split, k in keys[farm].items():
            truth, start = k['truth'].astype(np.float64), k['issue'].astype('datetime64[ns]')
            e = {n: {a: np.abs(truth - np.load(out / f'S_{n}/{farm}/{a}_{n}/{SEED}/{split}.npy').astype(np.float64))
                     for a in ('LG', 'LUG')} for n in STATES}
            agree = np.load(out / f'X5/{farm}/agree/{split}.npy')
            if agree.shape != (len(truth),) or agree.dtype != bool:
                raise ValueError('Agreement mask does not align with the cohort')

            def delta(a, b, rows=slice(None)):   # U(b) - U(a) on the same windows
                cand, ref = rise_losses(e[a]['LG'][rows], e[a]['LUG'][rows], e[b]['LG'][rows], e[b]['LUG'][rows])
                return paired_loss_statistics(cand, ref, start[rows], **kw)
            deltas = dict(truncation_fresh=delta('A', 'D'), cycle_common=delta('D', 'Cc'),
                          old_side_restriction=delta('Cc', 'C'), total=delta('A', 'C'))
            parts = sum(deltas[n]['gain'] for n in ('truncation_fresh', 'cycle_common', 'old_side_restriction'))
            if not np.isclose(parts, deltas['total']['gain'], rtol=0, atol=1e-6):
                raise ValueError('Decomposition U(C) - U(A) does not add up')
            gaps = bool((~agree).any())
            # X5b R2-B1 (user ruling on Claude's proposal): the label contrast needs >= min_issue_clusters clusters
            clean = bounded_contrast(e, 'D', 'Cc', start, agree, kw, protocol['classification']['min_issue_clusters'])
            row = dict(role='primary' if split == 'V2' else 'exploratory',
                       mae={n: {a: float(e[n][a].mean()) for a in ('LG', 'LUG')} for n in STATES},
                       U={n: paired_loss_statistics(e[n]['LUG'], e[n]['LG'], start, **kw) for n in STATES},
                       U_by_h={n: (e[n]['LG'] - e[n]['LUG']).mean(0).tolist() for n in STATES}, deltas=deltas,
                       natural_gap=dict(windows_disagree=int((~agree).sum()), share_agree=float(agree.mean()),
                                        cycle_common_includes_natural_gaps=gaps),
                       cycle_common_agreeing_windows=clean)
            if split == 'V2':
                row['reading'] = dict(classify_match(deltas['truncation_fresh'], clean,
                                                     protocol['classification']['near_zero_ratio']),
                                      natural_gap_difference=gaps)
            res['farms'][farm][split] = row
    readings = [res['farms'][f]['V2']['reading'] for f in farms_order]
    need = protocol['classification']['min_farms']
    if sum(r['truncation_ci_above_0'] and r['cycle'] == 'CI_WITHIN_BOUND' for r in readings) >= need:
        outcome = 'TRUNCATION_CI_ABOVE_0_AND_CYCLE_CI_WITHIN_BOUND'
    elif sum(r['cycle'] == 'CI_ABOVE_0' for r in readings) >= need:
        outcome = 'CYCLE_CI_ABOVE_0'
    else:
        outcome = 'NO_COMMON_PATTERN'
    res['outcome'] = dict(outcome=outcome, min_farms=need, descriptive=True, post_hoc=True,
                          farms_with_natural_gap_difference_on_V2=[f for f, r in zip(farms_order, readings)
                                                                   if r['natural_gap_difference']])
    return res


def summary_match_md(res):
    lines = ['# X5b: matched coverage, fresh (6 h) vs old (15 h) GFS cycle (LightGBM, seed 2024)', '',
             f"Label on V2: {res['outcome']['outcome']} (post-hoc development check; descriptive labels, not causal "
             "claims; test years exploratory). Farms whose V2 cycle contrast includes natural-gap differences: "
             f"{res['outcome']['farms_with_natural_gap_difference_on_V2'] or 'none'}.",
             'U = MAE(LG) - MAE(LUG) in kW. A fresh as is, C old as is, D fresh with C\'s coverage, '
             'Cc old with A\'s coverage. U(C) - U(A) = [U(D) - U(A)] + [U(Cc) - U(D)] + [U(C) - U(Cc)].',
             'cycle (all) includes any natural-gap difference between D and Cc; cycle (agreeing) uses only windows '
             'where D and Cc have identical actual availability; labels use the agreeing contrast and a bound of '
             'ratio x |truncation point estimate|.', '',
             '| farm | split | U A / C / Cc / D | truncation U(D)-U(A) [CI] | cycle all [CI] | cycle agreeing [CI] | '
             'agree share | old-side | total | label |', '|---|---|---|---|---|---|---|---|---|---|']

    def ci(x):
        if x is None:
            return 'n/a'
        if x.get('status') == 'NOT_ESTIMABLE':
            point = 'n/a' if x['gain'] is None else f"{x['gain']:.2f}"
            return f"{point} [not estimable: {x['issue_time_clusters']} clusters]"
        return f"{x['gain']:.2f} [{x['ci95_low']:.2f}, {x['ci95_high']:.2f}]"
    for farm, splits in res['farms'].items():
        for split, r in splits.items():
            d = r['deltas']
            us = ' / '.join('%.2f' % r['U'][n]['gain'] for n in STATES)
            reading = r.get('reading')
            tag = '' if reading is None else (f"{reading['cycle']} (bound {reading['bound_kw']:.2f})"
                                              + ('; truncation CI > 0' if reading['truncation_ci_above_0'] else ''))
            lines.append(f"| {farm} | {split} | {us} | {ci(d['truncation_fresh'])} | {ci(d['cycle_common'])} | "
                         f"{ci(r['cycle_common_agreeing_windows'])} | {r['natural_gap']['share_agree']:.4f} | "
                         f"{d['old_side_restriction']['gain']:.2f} | {d['total']['gain']:.2f} | {tag} |")
    return '\n'.join(lines) + '\n'


def run(args, mode):
    import lightgbm as lgb
    if lgb.__version__ != '4.5.0':
        raise ValueError('LightGBM fits require the timedart environment with lightgbm 4.5.0')
    match = mode in ('MSMOKE', 'MRUN')
    out = start_run(args, mode, PROTOCOL_B if match else PROTOCOL)
    try:
        protocol = read(PROTOCOL_B if match else PROTOCOL)
        x1p, x2p = read(ROOT / 'configs/x1_protocol.json'), read(ROOT / 'configs/x2_protocol.json')
        smoke = mode in ('SMOKE', 'MSMOKE')
        bud = dict(protocol['budget']['smoke' if smoke else 'formal'], smoke=smoke)
        if match:
            delays = sorted({protocol['fresh_delay_hours'], protocol['old_delay_hours']})
        else:
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
            if match:
                keys[farm] = run_farm_match(lgb, args, out, store, farm, spec, x1p, x2p, protocol, bud)
            else:
                keys[farm] = run_farm(lgb, args, out, store, farm, spec, x1p, x2p, protocol, bud, delays)
            order.append(farm)
        save(out / 'X5' / 'registry.json', store.registry)
        res = report_match(out, keys, protocol, order) if match else report(out, keys, delays, protocol, order)
        res.update(commit=git_identity()[0], protocol_sha256=args.protocol_sha256, smoke=smoke)
        save(out / 'X5' / 'results.json', res)
        (out / 'X5' / 'SUMMARY.md').write_text(summary_match_md(res) if match else summary_md(res), encoding='utf-8')
        save(out / 'completed.json', dict(status='COMPLETED', results_sha256=digest(out / 'X5' / 'results.json'),
                                          registry_sha256=digest(out / 'X5' / 'registry.json')))
        if smoke:
            passed = dict(status='PASS', interpret_results=False, commit=git_identity()[0])
            if match:
                passed['kind'] = 'match'
            save(out / 'smoke_passed.json', passed)
        beat(out, 'completed')
    except Exception as exc:
        x2.fail(out, exc)
        raise
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['smoke', 'run', 'msmoke', 'mrun'], required=True)
    p.add_argument('--logs', type=Path, required=True, help='<data root>/outputs/logs')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--protocol-sha256', required=True)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--smoke-result', type=Path)
    p.add_argument('--allow-dirty', action='store_true')
    args = p.parse_args(argv)
    run(args, args.mode.upper())


if __name__ == '__main__':
    main()
