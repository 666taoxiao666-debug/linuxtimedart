"""X3-HS0: READ-ONLY evidence for the GFS-history source conflict behind the X3 smoke Q4 failure (fb27194).

Question: does the hourly GFS history that X1 (A3) handed to Chronos agree with the GFS-history channel the
frozen packs were built with (x50[:, :, 20])?  For every frozen pack (train, validation, tests) of every farm it
compares x50 channel 20 with three hourly sources at offsets -6/-3/0/+3/+6 h:
  a3_grid     the A3 history grid (run_x1.chronos_history: for HoT, sealed-prep values first, then the
              development values written over the overlap, including NaN)
  hot_dev     HoT development preparation nwp_hourly.npz (G2 hot_dev_f58f2ce_r5)
  hot_sealed  HoT sealed-preparation point cache (run_x2.hot_test_history, as X2-A2 used for the test packs)
It also compares the sources with each other and counts, per A3 context and context length, the windows whose
input span contains an hour where a3_grid differs from the pack's own ("native") source.

Writes ONLY <out>/report.json and <out>/mismatch_hours.csv into a NEW directory.  Modifies nothing, trains and
predicts nothing.  Author: Claude (2026-11-18).  Run from a clean worktree of the reviewed commit, e.g.
  PYTHONPATH=<worktree> python diag_x3_gfs_source.py --logs <data>/outputs/logs --out <new dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HOUR = np.timedelta64(1, 'h')
OFFSETS = (-6, -3, 0, 3, 6)
LENGTHS = (24, 168, 512, 1024)
CUTOFF = np.datetime64('2024-01-01T00', 'ns')        # prepare_g1_batches.CUTOFF (development preparations)


def available_cycle(hours):
    """prepare_g1_batches.available_cycle, in ns."""
    h = np.asarray(hours).astype('datetime64[h]').astype(np.int64)
    return ((h - 6) // 6 * 6).astype('datetime64[h]').astype('datetime64[ns]')


def outside_development(hours):
    """Hours the development preparations left NaN on purpose: the cycle's 18 h container reaches the cutoff."""
    return available_cycle(hours) + 18 * HOUR >= CUTOFF


def lookup(src, times):
    """Source values at hour labels `times` (any shape); NaN where the hour is not on the source axis."""
    hours = np.asarray(src['hours']).astype('datetime64[ns]').astype(np.int64)
    t = np.asarray(times).astype('datetime64[ns]').astype(np.int64)
    pos = np.clip(np.searchsorted(hours, t), 0, len(hours) - 1)
    on = hours[pos] == t
    out = np.full(t.shape, np.nan)
    out[on] = np.asarray(src['historical'], float).reshape(len(hours), -1)[pos[on], 0]
    return out, on


def classify(ch, h, m, s, atol=1e-4):
    """Same rule as run_x3.gfs_binding: finite h -> ch == (h-m)/s; NaN h -> ch == 0 exactly."""
    fin = np.isfinite(h)
    nan_vs_value = ~fin & (ch != 0)
    value_vs_missing = fin & (ch == 0) & (np.abs((h - m) / s) > atol)
    differs = fin & ~value_vs_missing & (np.abs((np.where(fin, h, 0) - m) / s - ch) > atol)
    return dict(source_nan_pack_value=nan_vs_value, source_value_pack_missing=value_vs_missing,
                value_differs=differs)


def window_times(issue, L=24):
    issue = np.asarray(issue).astype('datetime64[ns]')
    return issue[:, None] - (L - 1 - np.arange(L)) * HOUR


def compare_sources(a, b, name_a, name_b, tol=1e-4):
    """Hours on both axes where the two sources disagree (NaN-aware; finite pairs differing by > tol m/s)."""
    ha = np.asarray(a['hours']).astype('datetime64[ns]')
    hb = np.asarray(b['hours']).astype('datetime64[ns]')
    common = np.intersect1d(ha, hb)
    va, _ = lookup(a, common)
    vb, _ = lookup(b, common)
    fa, fb = np.isfinite(va), np.isfinite(vb)
    both = fa & fb
    gap = np.abs(np.where(both, va - vb, 0))
    first_nan, second_nan, value = ~fa & fb, fa & ~fb, both & (gap > tol)
    diff = first_nan | second_nan | value
    od = outside_development(common[diff])
    rows = [dict(hour=str(t), **{name_a: None if not np.isfinite(x) else float(x)},
                 **{name_b: None if not np.isfinite(y) else float(y)}, outside_development=bool(o))
            for t, x, y, o in zip(common[diff][:200], va[diff][:200], vb[diff][:200], od[:200])]
    return dict(common_hours=int(len(common)), first=str(common.min()) if len(common) else None,
                last=str(common.max()) if len(common) else None, differing_hours=int(diff.sum()),
                only_first_nan=int(first_nan.sum()), only_second_nan=int(second_nan.sum()),
                values_differ=int(value.sum()), max_abs_diff_finite=float(gap[both].max()) if both.any() else None,
                all_outside_development=bool(od.all()), rows=rows), common[diff]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--logs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)                     # new directory only
    root = Path.cwd()
    sys.path.insert(0, str(root))
    from scripts import run_x1 as x1, run_x2 as x2, run_x3 as x3
    from scripts import train_g2b as g2, evaluate_g2b_sealed as sealed
    protocol, x1p = x3.read(x3.PROTOCOL), x3.read(root / 'configs/x1_protocol.json')
    refs = x3.Refs(args.logs, protocol)
    report = dict(stage='X3-HS0', read_only=True, offsets_h=OFFSETS, packs={}, sources={}, contexts={},
                  identities={})
    csv_rows = []
    for farm, spec in x2.farms(x1p).items():
        mean, scale, m, s = x3.scaler(SimpleNamespace(logs=args.logs), farm)
        g = refs.grid(farm)
        srcs = {'a3_grid': x3.hourly_gfs(g)}
        report['identities'][g['path']] = g['sha256']
        if farm == 'HoT':
            freeze = args.logs / 'G2/g2b_4471499_shared_r1/formal'
            frozen, _ = g2.frozen_gate(freeze, sealed.FREEZE_SHA)
            folder = args.logs / 'G2/hot_dev_f58f2ce_r5/prepared'
            x1.checked(folder / 'completed.json', frozen['preparation_sha256'])
            meta = x1.read(folder / 'completed.json')
            dev = x1.load_pack(x1.checked(folder / 'nwp_hourly.npz', meta['artifacts']['nwp_hourly.npz']))
            srcs['hot_dev'] = dict(hours=dev['hours'], historical=np.asarray(dev['historical'], float)[:, :1])
            h = x2.hot_test_history(args.logs)
            srcs['hot_sealed'] = dict(hours=h['hours'], historical=np.asarray(h['hist'], float)[:, None])
            report['identities'].update(h['identities'])
            report['identities'][str(folder / 'nwp_hourly.npz')] = meta['artifacts']['nwp_hourly.npz']
            report['sources'][farm] = {}
            for a, b in (('a3_grid', 'hot_dev'), ('a3_grid', 'hot_sealed'), ('hot_dev', 'hot_sealed')):
                res, _ = compare_sources(srcs[a], srcs[b], a, b)
                report['sources'][farm][f'{a}__vs__{b}'] = res
        item = next(iter(spec['items'].values()))
        train, val = x1.development(args.logs, item)
        packs = dict(train=train, validation=val,
                     **{n: x1.load_pack(x1.paths(args.logs, it)[0]) for n, it in spec['items'].items()})
        del train, val
        for name, pack in packs.items():
            native = 'hot_sealed' if (farm == 'HoT' and name not in ('train', 'validation')) else (
                'hot_dev' if farm == 'HoT' else 'a3_grid')
            ch = np.asarray(pack['x50'])[:, :, 20].astype(float)
            rec = dict(windows=int(len(ch)), native_source=native, nonfinite_channel=int((~np.isfinite(ch)).sum()),
                       by_source={})
            for sname, src in srcs.items():
                rec['by_source'][sname] = {}
                for d in OFFSETS:
                    t = window_times(pack['issue']) + d * HOUR
                    v, on = lookup(src, t)
                    c = classify(ch, v, m, s)
                    entry = {k: int(x.sum()) for k, x in c.items()}
                    entry.update(total=int(sum(entry.values())), off_axis=int((~on).sum()))
                    rec['by_source'][sname][str(d)] = entry
                    if d == 0:
                        bad = np.zeros(ch.shape, bool)
                        for x in c.values():
                            bad |= x
                        if bad.any():
                            hrs, cnt = np.unique(t[bad], return_counts=True)
                            entry['distinct_hours'] = int(len(hrs))
                            for hr, k in zip(hrs[:300], cnt[:300]):                 # detail capped at 300 hours
                                sel = bad & (t == hr)
                                raw = ch[sel] * s + m
                                csv_rows.append(dict(farm=farm, pack=name, source=sname, hour=str(hr), occurrences=int(k),
                                    types='|'.join(sorted(n for n, x in c.items() if (x & sel).any())),
                                    pack_raw_min=float(raw.min()), pack_raw_max=float(raw.max()),
                                    pack_value_zero=bool((ch[sel] == 0).all()),
                                    **{f'{q}_value': (lambda z: None if not np.isfinite(z) else float(z))(lookup(srcs[q], [hr])[0][0])
                                       for q in srcs},
                                    available_cycle=str(available_cycle([hr])[0]),
                                    outside_development=bool(outside_development([hr])[0])))
            report['packs'][f'{farm}/{name}'] = rec
        # windows of each A3 context whose span touches an hour where a3_grid differs from the native source
        for name in [farm + '_train', farm + '_validation'] + list(spec['items']):
            key = {farm + '_train': 'train', farm + '_validation': 'validation'}.get(name, name)
            native = report['packs'][f'{farm}/{key}']['native_source']
            ctx_rec = refs.context(name)
            with np.load(x1.checked(Path(ctx_rec['path']), ctx_rec['sha256']), allow_pickle=False) as z:
                issue, turbines = z['issue'], z['turbines']
            pk = packs[key]
            same = bool(np.array_equal(issue, pk['issue']) and np.array_equal(turbines, pk['turbines']))
            drift = np.array([], 'datetime64[ns]')
            if native != 'a3_grid':
                _, drift = compare_sources(srcs['a3_grid'], srcs[native], 'a3_grid', native)
            it = np.asarray(issue).astype('datetime64[ns]')
            counts = {}
            for L in LENGTHS:
                lo = it - (L - 1) * HOUR
                if len(drift):
                    k = np.searchsorted(drift, lo)                 # first drift hour >= window start
                    hit = (k < len(drift)) & (drift[np.minimum(k, len(drift) - 1)] <= it)
                else:
                    hit = np.zeros(len(it), bool)
                counts[str(L)] = dict(windows=int(hit.sum()), fraction=float(hit.mean()) if len(hit) else 0.0,
                                      first_issue=str(it[hit].min()) if hit.any() else None,
                                      last_issue=str(it[hit].max()) if hit.any() else None)
            report['contexts'][name] = dict(native_source=native, a3_context_equals_pack=same,
                                            drift_hours=[str(x) for x in drift], affected_windows=counts)
        del packs
        print('[X3-HS0] done', farm, flush=True)
    # pre-registered predicates
    zero = {k: v['by_source'][v['native_source']]['0']['total'] for k, v in report['packs'].items()}
    p1_rows = [r for r in csv_rows if r['source'] == 'a3_grid']
    P1 = bool(p1_rows) and all(r['farm'] == 'HoT' and r['pack'] not in ('train', 'validation')
                                and r['types'] == 'source_nan_pack_value' and r['outside_development'] for r in p1_rows)
    P2 = all(v == 0 for v in zero.values())
    P3 = all(all(v['by_source'][v['native_source']][str(d)]['total'] > 0 for d in OFFSETS if d)
             for v in report['packs'].values())
    hs = report['sources'].get('HoT', {})

    def only_dev_nan_at_boundary(r):            # every disagreement: first source NaN, second finite, boundary hour
        return r['differing_hours'] == r['only_first_nan'] and r['all_outside_development']
    P4 = bool(hs) and only_dev_nan_at_boundary(hs['hot_dev__vs__hot_sealed']) and \
        only_dev_nan_at_boundary(hs['a3_grid__vs__hot_sealed']) and hs['a3_grid__vs__hot_dev']['differing_hours'] == 0
    report['predicates'] = dict(
        P1_cause=P1, P2_native_source_binds_every_pack=P2, P3_no_time_offset_explains_it=P3,
        P4_sources_differ_only_at_outside_development_hours=P4,
        a3_grid_mismatch_occurrences={k: v['by_source']['a3_grid']['0']['total'] for k, v in report['packs'].items()},
        native_mismatch_occurrences=zero)
    report['status'] = 'CONFIRMED' if all((P1, P2, P3, P4)) else 'NOT_CONFIRMED'
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (args.out / 'mismatch_hours.csv').open('w', newline='', encoding='utf-8') as fh:
        if csv_rows:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
    print('[X3-HS0]', report['status'], json.dumps(report['predicates']), flush=True)


if __name__ == '__main__':
    main()
