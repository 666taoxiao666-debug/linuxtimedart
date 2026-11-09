"""M1 C0b: one frozen-predictor, past-only selection/evaluation run.

F is an empirical finite-radius rule, not a coverage guarantee. No point-model
training, label edits, downloads, or retries. The original C0 stays unchanged.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import heapq
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import pilot_conformal_c0 as c0

VARIANTS = ('S', 'A', 'S+A')
ALPHA = .1


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finite(value, name):
    if not np.isfinite(value).all():
        raise ValueError(f'Nonfinite {name}: technical failure')


def causal_check(times, issue, year):
    times, issue = np.asarray(times), np.asarray(issue)
    cutoff = np.datetime64(f'{year-1}-12-31T23:00', 'ns')
    if (not times.size or not issue.size or np.isnat(times).any() or np.isnat(issue).any()
            or times.max() > cutoff or np.any(issue < times.max())):
        raise ValueError('Calibration labels unavailable at forecast issue')


def finite_sorted_quantile(errors, cumulative, own, alpha):
    """F: preserve the C0 test-point mass; cap only its infinite outcome."""
    q = c0.sorted_quantile(errors, cumulative, own, alpha)
    capped = bool(np.isposinf(q))
    if capped:
        q = float(errors[-1])
    finite(q, 'quantile')
    return q, capped


def finite_quantile(errors, weights, own, alpha):
    errors, weights = np.asarray(errors), np.asarray(weights)
    finite(errors, 'calibration errors'); finite(weights, 'weights')
    if not len(errors) or errors.shape != weights.shape or np.any(weights <= 0):
        raise ValueError('Empty/invalid calibration group')
    order = np.argsort(errors, kind='stable')
    return finite_sorted_quantile(errors[order], np.cumsum(weights[order]), own, alpha)


def scales(srcerr, tarerr):
    denominator, numerator = np.median(srcerr, axis=0), np.median(tarerr, axis=0)
    finite(denominator, 'kappa denominator'); finite(numerator, 'kappa numerator')
    if np.any(denominator <= 0):
        raise ValueError('Zero kappa denominator: preparation failure')
    kappa = numerator / denominator
    finite(kappa, 'kappa')
    return kappa


def checked_metrics(radius, error, capped, mask=None):
    if mask is not None:
        radius, error, capped = radius[mask], error[mask], capped[mask]
    finite(radius, 'radius'); finite(error, 'error')
    if np.any(radius < 0):
        raise ValueError('Negative radius')
    finite(2 * radius, 'width')
    finite(2 * radius + 2 / ALPHA * np.maximum(error-radius, 0), 'Winkler')
    result = c0.metrics(radius, error, ALPHA)
    if error.size:
        result['cap_fraction'] = float(np.mean(capped))
    return result


def intervals(source, target, test, weights, year, variants=VARIANTS):
    causal_check(source['times'], test['issue'], year)
    causal_check(target['times'], test['issue'], year)
    for pack in (source, target, test):
        finite(pack['prediction'], 'point prediction'); finite(pack['truth'], 'truth')
    srcerr = np.abs(source['prediction']-source['truth'])
    tarerr = np.abs(target['prediction']-target['truth'])
    error = np.abs(test['prediction']-test['truth'])
    kappa = scales(srcerr, tarerr)
    ws, wt = weights
    finite(ws, 'source weights'); finite(wt, 'test weights')
    if np.any(ws <= 0) or np.any(wt <= 0):
        raise ValueError('Nonpositive density weights')
    methods = ('B0', 'B1', 'B2-A') + tuple(f'{base}-{v}' for v in variants for base in ('M', 'B3'))
    adaptive = tuple(m for m in methods if m == 'B2-A' or m.endswith('-A') or m.endswith('S+A'))
    radii = {m: np.empty_like(error, dtype=float) for m in methods}
    caps = {m: np.zeros_like(error, dtype=bool) for m in methods}
    order = np.lexsort((test['turbines'], test['issue']))
    for h in range(error.shape[1]):
        es = np.sort(srcerr[:, h]); cs = np.arange(1, len(es)+1)
        q0, cap0 = finite_sorted_quantile(es, cs, 1., ALPHA)
        q1, cap1 = finite_quantile(tarerr[:, h], np.ones(len(tarerr)), 1., ALPHA)
        edges = np.unique(np.quantile(source['prediction'][:, h], [.2, .4, .6, .8]))
        bins = np.searchsorted(edges, source['prediction'][:, h], side='right')
        groups = {}
        for b in np.unique(bins):
            ix = np.flatnonzero(bins == b)
            ix = ix[np.argsort(srcerr[ix, h], kind='stable')]
            groups[int(b)] = (srcerr[ix, h], np.cumsum(ws[ix]))
        sorted_power = np.argsort(source['prediction'][:, h], kind='stable')
        power = source['prediction'][sorted_power, h]
        states, queues = {}, {}
        for i in order:
            turbine, now = int(test['turbines'][i]), test['issue'][i]
            if turbine not in states:
                states[turbine] = {m: ALPHA for m in adaptive}; queues[turbine] = []
            while queues[turbine] and queues[turbine][0][0] <= now:
                _, _, misses = heapq.heappop(queues[turbine])
                for m, miss in zip(adaptive, misses):
                    states[turbine][m] = float(np.clip(states[turbine][m] + .005*(ALPHA-miss), .005, .5))
            predicted = test['prediction'][i, h]
            b = int(np.searchsorted(edges, predicted, side='right'))
            b = min(groups, key=lambda g: (abs(g-b), g))
            eg, cg = groups[b]
            j = np.searchsorted(power, predicted)
            pool = sorted_power[max(0, j-500):min(len(power), j+500)]
            nearest = pool[np.argsort(np.abs(source['prediction'][pool, h]-predicted), kind='stable')[:500]]
            en = np.sort(srcerr[nearest, h]); cn = np.arange(1, len(en)+1)
            radii['B0'][i, h], caps['B0'][i, h] = q0, cap0
            radii['B1'][i, h], caps['B1'][i, h] = q1, cap1
            for m in methods[2:]:
                a = states[turbine][m] if m in adaptive else ALPHA
                e, c, own = (es, cs, 1.) if m == 'B2-A' else ((eg, cg, wt[i]) if m.startswith('M-') else (en, cn, 1.))
                q, capped = finite_sorted_quantile(e, c, own, a)
                if m.endswith('-S') or m.endswith('S+A'):
                    q *= kappa[h]
                finite(q, 'emitted radius')
                radii[m][i, h], caps[m][i, h] = q, capped
            arrival = test['times'][i, h]
            if arrival != now + np.timedelta64(h+1, 'h'):
                raise ValueError('ACI label delay is not the registered horizon')
            misses = tuple(float(error[i, h] > radii[m][i, h]) for m in adaptive)
            heapq.heappush(queues[turbine], (arrival, int(i), misses))
    report = {}
    for m in methods:
        report[m] = {'pooled': checked_metrics(radii[m], error, caps[m]),
                     'horizons': [checked_metrics(radii[m][:, h], error[:, h], caps[m][:, h]) for h in range(error.shape[1])],
                     'missing': checked_metrics(radii[m], error, caps[m], test['missing']),
                     'observed': checked_metrics(radii[m], error, caps[m], ~test['missing'])}
    return report, radii, caps, kappa


def select_variant(means):
    for v in VARIANTS:
        finite(list(means[f'M-{v}'].values()), 'selection metrics')
    eligible = [v for v in VARIANTS if .88 <= means[f'M-{v}']['coverage'] <= .92]
    if eligible:
        return min(eligible, key=lambda v: (means[f'M-{v}']['winkler_kw'], VARIANTS.index(v)))
    return min(VARIANTS, key=lambda v: (abs(means[f'M-{v}']['coverage']-.9), VARIANTS.index(v)))


def finite_leq(a, b):
    return bool(np.isfinite(a) and np.isfinite(b) and a <= b)


def verdict(means, selected):
    comparators = ['B0', 'B1', 'B2-A', f'B3-{selected}']
    eligible = [m for m in comparators if means[m]['coverage'] >= .88] or comparators
    scores = [means[m]['winkler_kw'] for m in eligible]
    best = min(scores) if np.isfinite(scores).all() else float('nan')
    m = means[f'M-{selected}']
    flags = {'C1': bool(.88 <= m['coverage'] <= .92),
             'C2': finite_leq(m['winkler_kw'], .95*best),
             'C3': finite_leq(m['winkler_kw'], .97*means[f'B3-{selected}']['winkler_kw'])}
    return {**flags, 'decision': 'continue' if all(flags.values()) else 'stop',
            'eligible_comparators': eligible, 'best_comparator_winkler': best if np.isfinite(best) else None,
            'scope': 'development evidence only; empirical coverage, no finite-sample guarantee; no second optimization'}


def require_selection(out):
    path = out/'selection_freeze.json'
    if not path.is_file():
        raise ValueError('2023 evaluation requires selection_freeze.json')
    frozen = json.loads(path.read_text(encoding='utf-8'))
    if frozen['selected'] not in VARIANTS:
        raise ValueError('Invalid frozen selection')
    for name, digest in frozen['source_code'].items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'Code changed after selection: {name}')
    if sha(out/'selection_2022/result.json') != frozen['selection_result_sha256']:
        raise ValueError('Selection result changed')
    for name, digest in frozen['input_hashes'].items():
        if sha(name) != digest:
            raise ValueError(f'Input changed after selection: {name}')
    return frozen


def phase(data_root, out, year, variants, device):
    import pandas as pd
    import torch
    from scripts import train_penmanshiel_p2_v35 as p2
    from scripts import train_n1ph as n1
    u, trainer = p2.u, p2.trainer
    out.mkdir(parents=True, exist_ok=False)
    recipe = u.read_json(p2.PROTOCOL); old = u.read_json(ROOT/'configs/n1ph_protocol.json')
    identities = {}
    pf, pup, meta, kf, kup, km = n1.other_inputs(data_root, old, recipe, identities)
    splits = {'train': [2016, 2017, 2018, 2019], 'validation': [2020], 'calibration': [year-1], 'test': [year]}
    # Source has observations through 2022 only; target uses opened years <=2023.
    kel = u.KelmarshUpstreamSeries(kf, kup, splits=splits, start='2016-05-01', end='2023-01-01T01:00:00Z', preparation=km)
    pen = u.KelmarshUpstreamSeries(pf[pf.label_year <= year], pup[pup.index < pd.Timestamp(f'{year+1}-01-01T01:00:00')],
                                  splits=splits, start=meta['start'], end=f'{year+1}-01-01T01:00:00Z', preparation=meta)
    records = [u.read_json(ROOT/f'output/penmanshiel_hourly/p2_v35/fits/K-upstream/s{s}/trained.json') for s in p2.SEEDS]
    for path in [p2.PROTOCOL, ROOT/'configs/n1ph_protocol.json', ROOT/'docs/STAGE_PLAN.md',
                 ROOT/p2.farm.D2_SUBDIR/'part_A.json'] + [ROOT/f'output/penmanshiel_hourly/p2_v35/fits/K-upstream/s{s}/trained.json' for s in p2.SEEDS]:
        identities[str(path)] = sha(path)
    state = records[0]['contract']['statistics']
    if any(r['contract']['statistics'] != state for r in records):
        raise ValueError('P2 seeds have different preprocessing states')
    kel = p2.adapt_statistics(kel, kup, state=state); pen = p2.adapt_statistics(pen, pup, state=state)

    def windows(series, lo, hi):
        s = series.starts; times = series.dates[s[:, None]+24+np.arange(4)]
        keep = (times[:, 0] >= np.datetime64(lo)) & (times[:, -1] <= np.datetime64(hi))
        return p2.MultiYearWindows(series, s[keep], 'C0b', 'all')
    source = windows(kel, f'{year-1}-01-01', f'{year-1}-12-31T23:00')
    target = windows(pen, f'{year-1}-12-01', f'{year-1}-12-31T23:00')
    test = pen.windows('test')
    direction_cache = {}

    def nearest_direction(is_pen, labels):
        if is_pen not in direction_cache:
            if is_pen:
                part = u.read_json(ROOT/p2.farm.D2_SUBDIR/'part_A.json')
                stations = [s for s in part['midas']['stations'] if s['src_id'] in p2.farm.STATIONS]
            else:
                stations = recipe['data']['upstream']['stations']
            sid = int(min(stations, key=lambda s: (s['distance_km'], s['src_id']))['src_id'])
            registry = u.read_json(data_root/old['p3_run']/'prepared/input_hashes.json') if is_pen else None
            files = []
            for raw_year in range(year-2, year+1 if is_pen else year):
                if is_pen:
                    matches = [(p, e) for p, e in registry.items() if Path(p).parent.name == str(sid) and Path(p).name.endswith(f'_qcv-1_{raw_year}.csv')]
                    if len(matches) != 1:
                        raise ValueError('Ambiguous nearest-station file')
                    name, entry = matches[0]; path = data_root/'datasets'/name.split('/datasets/', 1)[1]
                    n1.checked(path, entry['sha256'], identities)
                    files.append({'path': path, 'year': raw_year})
                else:
                    spec = recipe['data']['upstream']; hashes = u.manifest_hashes(u.read_json(data_root/spec['download_manifest']))
                    for root in spec['development_roots']:
                        if raw_year in root['years']:
                            files += u.station_files(data_root/root['root'], [sid], [raw_year], hashes, True)[0]
            direction_cache[is_pen] = u.load_station_observations(files, [sid])[sid]
            for entry in files:
                identities[str(entry['path'])] = u.file_sha256(entry['path'])
        return direction_cache[is_pen].reindex(pd.DatetimeIndex(labels)-pd.Timedelta(hours=1)).wind_direction.to_numpy()

    def context(frame, is_pen):
        labels = pd.date_range(f'{year-1}-01-01', f'{year-1}-12-31T23:00', freq='h')
        return c0.direction_features(frame.reindex(labels).to_numpy(float), nearest_direction(is_pen, labels))
    ratio, prior = c0.ratio_fit(context(kup, False), context(pup, True))

    def pack(w, frame, is_pen):
        issue = w.target_times()[:, 0]-np.timedelta64(1, 'h')
        raw = frame.reindex(pd.DatetimeIndex(issue)).to_numpy(float)
        return {'truth': w.truth_and_anchor()[0], 'times': w.target_times(), 'issue': issue,
                'turbines': w.turbines[w.window_starts], 'missing': ~np.isfinite(raw).all(1),
                'features': c0.direction_features(raw, nearest_direction(is_pen, issue)), 'starts': w.window_starts}
    packs = [pack(source, kup, False), pack(target, pup, True), pack(test, pup, True)]
    causal_check(packs[0]['times'], packs[2]['issue'], year); causal_check(packs[1]['times'], packs[2]['issue'], year)
    ws, wt = c0.density(ratio, prior, packs[0]['features']), c0.density(ratio, prior, packs[2]['features'])
    audit = {'year': year, 'calibration_cutoff': f'{year-1}-12-31T23:00',
             'windows': dict(zip(('source', 'target_month', 'evaluation'), [len(w) for w in (source, target, test)])),
             'input_hashes': identities, 'checkpoint_hashes': {}, 'source_code': p2.source_code(),
             'point_predictor_training': False, 'density_year': year-1, 'variants': list(variants)}
    for seed, record in zip(p2.SEEDS, records):
        folder = data_root/old['p2_run']/f'fits/K-upstream/s{seed}'
        trainer.verify_trained(folder, record['contract'])
        if u.canonical_sha256(u.read_json(folder/'trained.json')) != u.canonical_sha256(record):
            raise ValueError('P2 registration changed')
        for name, digest in record['artifacts'].items():
            n1.checked(folder/name, digest, identities)
        audit['checkpoint_hashes'][str(seed)] = record['artifacts']['best.pt']
    save(out/'calibration_freeze.json', audit)
    with (out/'evaluation_started.json').open('x') as stream:
        json.dump({'utc': p2.now(), 'year': year}, stream)
    per_seed = []
    for seed in p2.SEEDS:
        model, _ = trainer.build_model(p2.configuration('K-upstream', recipe))
        checkpoint = torch.load(data_root/old['p2_run']/f'fits/K-upstream/s{seed}/best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True); model.to(device)
        for name, w, p in zip(('source', 'target_month', 'evaluation'), (source, target, test), packs):
            p['prediction'] = trainer.v26run.predict(model, p2.view(w, 'K-upstream'), seed, device, 512)
            finite(p['prediction'], 'point prediction')
            np.savez_compressed(out/f'{name}_s{seed}.npz', **p)
        report, radii, caps, kappa = intervals(*packs, (ws, wt), year, variants)
        np.savez_compressed(out/f'intervals_s{seed}.npz', **radii)
        np.savez_compressed(out/f'caps_s{seed}.npz', **caps)
        result = {'seed': seed, 'kappa': kappa.tolist(), 'methods': report}
        save(out/f'seed_{seed}.json', result); per_seed.append(result)
        print(f'[C0b] {year} seed {seed} complete', flush=True)
    means = {m: {k: float(np.mean([r['methods'][m]['pooled'][k] for r in per_seed]))
                 for k in ('coverage', 'width_kw', 'winkler_kw', 'cap_fraction')} for m in per_seed[0]['methods']}
    result = {'status': 'COMPLETED', 'year': year, 'seeds': list(p2.SEEDS), 'mean_90': means,
              'artifacts': {p.name: sha(p) for p in out.glob('*.npz')}}
    save(out/'result.json', result)
    return result, audit


def run(data_root, out, device):
    selection, audit = phase(data_root, out/'selection_2022', 2022, VARIANTS, device)
    selected = select_variant(selection['mean_90'])
    save(out/'selection_freeze.json', {'selected': selected, 'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
         'selection_result_sha256': sha(out/'selection_2022/result.json'),
         'source_code': audit['source_code'], 'input_hashes': audit['input_hashes']})
    frozen = require_selection(out)  # guard precedes ALL preparation/prediction/calculation for 2023
    save(out/'test_evaluation_started.json', {'year': 2023, 'selection_freeze_sha256': sha(out/'selection_freeze.json'),
                                            'utc': dt.datetime.now(dt.timezone.utc).isoformat()})
    evaluation, _ = phase(data_root, out/'evaluation_2023', 2023, (frozen['selected'],), device)
    result = {'status': 'COMPLETED', 'selected': selected, 'mean_90': evaluation['mean_90'],
              **verdict(evaluation['mean_90'], selected), 'selection_freeze_sha256': sha(out/'selection_freeze.json')}
    save(out/'result.json', result)
    return result


def smoke(out):
    rng = np.random.default_rng(2024)
    def fake(n, start):
        issue = np.datetime64(start, 'ns') + np.arange(n)*np.timedelta64(3, 'h')
        return {'prediction': rng.normal(size=(n, 4)), 'truth': rng.normal(size=(n, 4)),
                'issue': issue, 'times': issue[:, None]+np.arange(1, 5)*np.timedelta64(1, 'h'),
                'turbines': np.ones(n, int), 'missing': np.zeros(n, bool)}
    report, _, _, _ = intervals(fake(600, '2021-01-01'), fake(100, '2021-12-01'), fake(50, '2022-01-01'),
                                 (np.ones(600), np.ones(50)), 2022)
    result = {'status': 'PASSED', 'smoke': True, 'methods': list(report)}
    save(out/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    save(args.out/'launch.json', {'argv': sys.argv, 'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
         'script_sha256': sha(__file__), 'c0_sha256': sha(ROOT/'scripts/pilot_conformal_c0.py'),
         'plan_sha256_lf': hashlib.sha256((ROOT/'docs/STAGE_PLAN.md').read_bytes().replace(b'\r\n', b'\n')).hexdigest()})
    try:
        if args.smoke:
            result = smoke(args.out)
        else:
            import torch
            torch.set_num_threads(2)
            result = run(args.data_root.resolve(), args.out, torch.device('cpu'))
        save(args.out/'completed.json', {'status': result['status'], 'seconds': time.monotonic()-started,
                                        'utc': dt.datetime.now(dt.timezone.utc).isoformat()})
    except Exception as exc:
        save(args.out/'failed.json', {'type': type(exc).__name__, 'error': str(exc), 'seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
