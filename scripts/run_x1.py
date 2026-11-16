"""X1 baselines on hash-locked cohorts. Only reviewed, ready arms run."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def checked(path, sha):
    if digest(path) != sha:
        raise ValueError('Hash mismatch: '+str(path))
    return Path(path)


def persistence(truth, anchor):
    if truth.ndim != 2 or truth.shape[1] != 4 or anchor.shape != truth.shape[:1]:
        raise ValueError('Invalid aligned cohort')
    if not np.isfinite(truth).all() or not np.isfinite(anchor).all():
        raise ValueError('Nonfinite cohort, no repair permitted')
    return np.broadcast_to(anchor[:, None], truth.shape).copy()


def paths(base, item):
    if item['family'] == 'G1':
        parent = base/'G1/g1c_27312d7_r2/formal'
        pack = parent.parent/'prepared'/item['farm']/'test.npz'
        receipt = ROOT/'output/g1/results_78/formal/completed.json'
        names = [f"{item['farm']}/N/{s}/prediction.npy" for s in (2024, 2025, 2026)]
    else:
        parent = base/'G2/g2c_2eeb23f_r1/formal'
        pack = base/'G2/g2b84_bc7730e_r1/sealed_prep'/f"{item['year']}.npz"
        receipt = ROOT/'output/g2b/sealed_eval/server/completed.json'
        names = [f"{item['year']}_N_{s}.npy" for s in (2024, 2025, 2026)]
    frozen = read(receipt)
    if read(parent/'completed.json') != frozen or (parent/'failed.json').exists():
        raise ValueError('Parent receipt mismatch or failed run')
    checked(pack, item['pack_sha256'])
    preds = [checked(parent/n, frozen['predictions'][n]) for n in names]
    return pack, preds


def curve_points(wind, power, valid, width=.5, minimum=30):
    valid = valid & np.isfinite(wind) & np.isfinite(power) & (wind >= 0)
    w, y = wind[valid], power[valid]
    bins = np.floor(w/width).astype(int)
    points = [(float((b+.5)*width), float(np.median(y[bins == b])))
              for b in np.unique(bins) if (bins == b).sum() >= minimum]
    return np.asarray(points, float).reshape(-1, 2)


def fit_curves(packs, post_only=False):
    w = np.concatenate([p['gF'][:, 6:10] for p in packs])
    y = np.concatenate([p['truth'] for p in packs])
    valid = np.concatenate([p['gF_mask'][:, 6:10] > 0 for p in packs])
    unit = np.concatenate([p['turbines'] for p in packs])
    if post_only:
        valid &= np.concatenate([p['retrofit'] > 0 for p in packs])[:, None]
    pooled = curve_points(w, y, valid)
    curves = {}
    for u in np.unique(unit):
        take = unit == u
        points = curve_points(w[take], y[take], valid[take])
        curves[str(u)] = dict(points=points.tolist() if len(points) else pooled.tolist(),
                              pooled_fallback=not bool(len(points)))
    return curves


def curve_predict(pack, curves):
    pred = persistence(pack['truth'], pack['anchor'])
    w, mask = pack['gF'][:, 6:10], pack['gF_mask'][:, 6:10] > 0
    valid = mask & np.isfinite(w) & (w >= 0)
    used = np.zeros_like(valid)
    for u in np.unique(pack['turbines']):
        points = np.asarray(curves[str(u)]['points']).reshape(-1, 2)
        take = (pack['turbines'] == u)[:, None] & valid
        if len(points):
            pred[take] = np.interp(w[take], points[:, 0], points[:, 1])
            used[take] = True
    return pred, dict(missing_gfs_cells=int((~mask).sum()),
                     total_fallback_cells=int((~used).sum()), total_cells=int(mask.size))


def load_pack(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def development(logs, item):
    if item['family'] == 'G1':
        folder = logs/'G1/g1c_27312d7_r2/prepared'/item['farm']
        local = ROOT/'output/g1/preparation_78/prepared'/item['farm']/'prepared.json'
        meta = read(local)
        if read(folder/'prepared.json') != meta:
            raise ValueError('Development metadata differs')
        return [load_pack(checked(folder/(s+'.npz'), meta['artifacts'][s+'.npz']))
                for s in ('train', 'val')]
    from scripts import train_g2b as g2, evaluate_g2b_sealed as sealed
    freeze = logs/'G2/g2b_4471499_shared_r1/formal'
    frozen, _ = g2.frozen_gate(freeze, sealed.FREEZE_SHA)
    stats = read(freeze/'fit_statistics.json')
    folder = logs/'G2/hot_dev_f58f2ce_r5/prepared'
    checked(folder/'completed.json', frozen['preparation_sha256'])
    meta = read(folder/'completed.json')
    arrays = {n: load_pack(checked(folder/n, meta['artifacts'][n]))
              for n in ('scada_rows.npz', 'upstream_hourly.npz', 'nwp_hourly.npz')}
    rows, up, hourly = [arrays[n] for n in ('scada_rows.npz', 'upstream_hourly.npz', 'nwp_hourly.npz')]
    hours = rows['hours'].astype('datetime64[ns]')
    if np.any(hours >= np.datetime64('2024-01-01')):
        raise ValueError('Non-development inputs')
    hourly.update(nwp_mean=stats['nwp_mean'], nwp_scale=stats['nwp_scale'])
    root = logs.parents[1]
    works = root/'datasets/external_validation/sealed/hill_of_towie/Hill_of_Towie_AeroUp_install_dates.csv'
    checked(works, stats['works_sha256'])
    packs = g2.common_packs(rows['raw'], hours, rows['units'], up['raw'], hourly, stats,
                           works, stats['bounds'], '2024-01-01')
    return [packs[s] for s in ('train', 'val')]


def cached_hourly(hours, cache, stats):
    from scripts import prepare_g1_batches as prep
    import pandas as pd
    hours = np.asarray(hours, dtype='datetime64[h]')
    nsites = next(iter(cache.values())).shape[-1]
    current = np.zeros((len(hours), 13, nsites), np.float32)
    masks = np.zeros_like(current)
    historical = np.full((len(hours), nsites), np.nan)
    cycles = prep.available_cycle(hours)
    age = ((hours-cycles)/np.timedelta64(1, 'h')).astype(int)
    ok = np.zeros(len(hours))
    for i, (cycle, a) in enumerate(zip(cycles, age)):
        values = cache.get(pd.Timestamp(cycle).strftime('%Y%m%d%H'))
        if values is None:
            continue
        h, valid, _ = prep.interpolate_leads(values, np.array([a]))
        historical[i] = np.where(valid[0], h[0], np.nan)
        current[i], masks[i], _ = prep.interpolate_leads(values, a+np.arange(-5, 8))
        ok[i] = np.isfinite(values).all()
    return dict(hours=hours, historical=historical, current=current, masks=masks, age=age,
        cycle_ok=ok, calibrated=np.zeros((len(hours), nsites)),
        calibrated_mask=np.zeros((len(hours), nsites)), directions=np.full((len(hours), nsites-1), np.nan),
        nwp_mean=stats['nwp_mean'], nwp_scale=stats['nwp_scale'])


def dense_pack(logs, item):
    """Read frozen preparation sources; only change stride in a local function clone."""
    import pandas as pd
    from scripts import prepare_g1_batches as prep, train_g2b as g2
    from data_provider.kelmarsh_multiyear import valid_window_starts
    root = logs.parents[1]
    if item['family'] == 'G1':
        folder = logs/'G1/g1c_27312d7_r2/prepared'/item['farm']
        meta = read(folder/'prepared.json')
        expected = read(ROOT/'output/g1/preparation_78/prepared'/item['farm']/'prepared.json')
        if meta != expected:
            raise ValueError('Preparation identity differs')
        series, _, _, _, hours, _, upstream = prep.load_development(root, item['farm'], meta['bounds'], {})
        stat = meta['scaler']
        features = upstream.reindex(pd.DatetimeIndex(series.dates)).to_numpy(float)
        raw = np.column_stack([series.base.raw[:, :-1],
            np.where(np.isfinite(features), features, np.array(stat['fill'])),
            (~np.isfinite(features).all(1)).astype(float), series.base.raw[:, -1]])
        series.raw = raw
        series.values = ((raw-np.array(stat['mean']))/np.array(stat['scale'])).astype(np.float32)
        series.scaler_mean, series.scaler_scale = np.array(stat['mean']), np.array(stat['scale'])
        series.starts, _ = valid_window_starts(series.base.raw, series.dates, series.turbines, 24, 4, 1)
        cache = {}
        for name in ('points_0.npz', 'points_1.npz'):
            # Point files are reconstructed caches; exact old-window checks below bind their effects.
            cache.update(load_pack(folder/name))
        hourly = cached_hourly(hours, cache, meta)
        pack = prep.build_packs(series, hours, hourly, {'test': meta['bounds']['test']})['test']
        old = load_pack(folder/'test.npz')
        pack.update(geo=old['geo'], ws_scale=old['ws_scale'])
        return pack
    from scripts import evaluate_g2b_sealed as sealed
    folder = logs/'G2/g2b84_bc7730e_r1/sealed_prep'
    freeze = logs/'G2/g2b_4471499_shared_r1/formal'
    g2.frozen_gate(freeze, sealed.FREEZE_SHA)
    stats = read(freeze/'fit_statistics.json')
    frame = pd.read_csv(folder/'hourly_scada.csv')
    frame['timestamp_hour_end_utc'] = pd.to_datetime(frame['timestamp_hour_end_utc'], utc=True)
    if (frame['timestamp_hour_end_utc'].dt.year >= 2026).any():
        raise ValueError('Protected SCADA timestamp')
    hours = pd.date_range('2023-01-01T01:00', '2026-01-01', freq='h', inclusive='left')
    works = root/'datasets/external_validation/sealed/hill_of_towie/Hill_of_Towie_AeroUp_install_dates.csv'
    checked(works, stats['works_sha256'])
    units = sorted(pd.read_csv(works)['Turbine'].tolist())
    raw = np.concatenate([g2.hot.scada_values(frame[frame.turbine_id == u]
        .set_index('timestamp_hour_end_utc').reindex(hours.tz_localize('UTC'))) for u in units])
    upstream, _ = sealed.sealed_midas(root, hours, {}, [2023, 2024, 2025])
    hourly = cached_hourly(hours, load_pack(folder/'points.npz'), stats)
    class StrideOne:
        def __getattr__(self, name):
            if name == 'valid_window_starts':
                return lambda values, dates, turbines, a, b, stride: g2.hot.valid_window_starts(values, dates, turbines, a, b, 1)
            return getattr(g2.hot, name)
    clone = g2.scoped_functions(g2, hot=StrideOne())
    return clone.common_packs(raw, hours.to_numpy(), units, upstream.to_numpy(), hourly,
        stats, works, {'test': [f"{item['year']}-01-01", f"{item['year']+1}-01-01"]}, '2026-01-01')['test']


def farm_metrics(pred, truth, issue, turbines, rated, unit_count):
    if len(set(zip(issue.tolist(), turbines.tolist()))) != len(issue):
        raise ValueError('Duplicate turbine/issue')
    _, inverse, counts = np.unique(issue, return_inverse=True, return_counts=True)
    keep = counts >= np.ceil(.8*unit_count)
    p, y = np.zeros((len(counts), 4)), np.zeros((len(counts), 4))
    np.add.at(p, inverse, pred)
    np.add.at(y, inverse, truth)
    error = (p[keep]-y[keep])/(counts[keep, None]*rated)
    if not len(error):
        raise ValueError('No issue with 80 percent turbines')
    return dict(issue_count=int(keep.sum()), total_issue_count=len(counts),
        nmae=float(np.abs(error).mean()), one_minus_nrmse=float(1-np.sqrt(np.square(error).mean())),
        lead_nmae=np.abs(error).mean(0).tolist(),
        lead_one_minus_nrmse=(1-np.sqrt(np.square(error).mean(0))).tolist())


def frozen_predictions(logs, item, pack, arm):
    import torch
    from scripts import train_g1 as g1, train_g2b as g2
    if item['family'] == 'G1':
        if arm == 'N0':
            parent = logs/'G2/g2a_e72e462_r1/formal'
            local = ROOT/'output/g2a/results81'
            data = g1.g2a_pack(pack)
        else:
            parent = logs/'G1/g1c_27312d7_r2/formal'
            local = ROOT/'output/g1/results_78/formal'
            data = pack
        protocol = read(ROOT/'configs/g1_protocol.json')
        factory = g1.make_model
        keys = [f"{item['farm']}/{arm}/{s}" for s in (2024, 2025, 2026)]
    else:
        parent = logs/'G2/g2b_4471499_shared_r1/formal'
        local = ROOT/'output/g2b/freeze/formal'
        protocol = read(ROOT/'configs/g2b_protocol.json')
        factory = g2.make_model
        data = g2.arm_pack(pack, arm, True)
        keys = [f'{arm}_{s}' for s in (2024, 2025, 2026)]
    frozen = read(parent/'selection_freeze.json')
    if frozen != read(local/'selection_freeze.json'):
        raise ValueError('Checkpoint manifest mismatch')
    fits = {f['key']: f for f in frozen['fits']}
    predictions = []
    for key in keys:
        f = fits[key]
        model = factory(arm, data, protocol['fit'], 'cpu')
        model.load_state_dict(torch.load(checked(parent/f['checkpoint'], f['sha256']),
                                         map_location='cpu', weights_only=True))
        predictions.append(g1.predict(model, data, 'cpu'))
        del model
    return predictions


def run_e1(args):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    checked(ROOT/'configs/x1_protocol.json', args.protocol_sha256)
    protocol = read(ROOT/'configs/x1_protocol.json')
    if hasattr(os, 'sched_getaffinity'):
        cores = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, cores[2:4] or cores[:2])
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'launch.json', dict(mode='E1', script_sha256=digest(__file__),
        protocol_sha256=args.protocol_sha256, cpu_threads=2, stride=1, new_fits=0))
    try:
        results = {}
        for name, item in protocol['datasets'].items():
            path, _ = paths(args.logs, item)
            old = load_pack(path)
            pack = dense_pack(args.logs, item)
            index = {(int(t), int(u)): i for i, (t, u) in enumerate(zip(pack['issue'], pack['turbines']))}
            overlap = np.array([index[(int(t), int(u))] for t, u in zip(old['issue'], old['turbines'])])
            for k in ('truth', 'anchor', 'x21', 'x50', 'gF', 'gF_mask'):
                if not np.allclose(pack[k][overlap], old[k], rtol=0, atol=1e-5):
                    raise ValueError('Dense input equivalence failed: '+name+'/'+k)
            report = dict(windows=len(pack['truth']), overlap_windows=len(old['truth']), metrics={}, equivalence={})
            for arm in ('N', 'N0', 'B'):
                predictions = frozen_predictions(args.logs, item, pack, arm)
                if item['family'] == 'G1':
                    parent = args.logs/('G2/g2a_e72e462_r1/formal' if arm == 'N0' else 'G1/g1c_27312d7_r2/formal')
                    names = [f"{item['farm']}/{arm}/{s}/prediction.npy" for s in (2024, 2025, 2026)]
                else:
                    parent = args.logs/'G2/g2c_2eeb23f_r1/formal'
                    names = [f"{item['year']}_{arm}_{s}.npy" for s in (2024, 2025, 2026)]
                receipt = read(parent/'completed.json')
                deltas = [float(np.max(np.abs(p[overlap]-np.load(checked(parent/n, receipt['predictions'][n])))))
                          for p, n in zip(predictions, names)]
                if max(deltas) > .01:
                    raise ValueError('Dense prediction equivalence failed: '+name+'/'+arm+': '+str(deltas))
                report['equivalence'][arm] = deltas
                report['metrics'][arm] = [farm_metrics(p, pack['truth'], pack['issue'], pack['turbines'],
                    item['rated_kw'], len(np.unique(pack['turbines']))) for p in predictions]
                report['metrics'][arm+'_ensemble'] = farm_metrics(np.mean(predictions, 0), pack['truth'],
                    pack['issue'], pack['turbines'], item['rated_kw'], len(np.unique(pack['turbines'])))
                del predictions
            curves = read(args.a2_results/(item['farm']+'_curves.json'))
            for arm, pred in [('A0', persistence(pack['truth'], pack['anchor'])), ('A2', curve_predict(pack, curves)[0])]:
                report['metrics'][arm] = farm_metrics(pred, pack['truth'], pack['issue'], pack['turbines'],
                    item['rated_kw'], len(np.unique(pack['turbines'])))
            results[name] = report
            save(args.output/(name+'.json'), report)
            print('COMPLETED', name, flush=True)
            del old, pack
        save(args.output/'results.json', dict(descriptive_only=True, datasets=results))
        save(args.output/'completed.json', dict(status='COMPLETED', results_sha256=digest(args.output/'results.json')))
    except Exception as exc:
        save(args.output/'failed.json', dict(type=type(exc).__name__, message=str(exc), stop_E1_only=True))
        raise


def raw_timeline(logs, item):
    import pandas as pd
    from scripts import prepare_g1_batches as prep
    if item['family'] == 'G1':
        meta = read(ROOT/'output/g1/preparation_78/prepared'/item['farm']/'prepared.json')
        series, *_ = prep.load_development(logs.parents[1], item['farm'], meta['bounds'], {})
        return series.dates, series.turbines, series.base.raw[:, 0], series.base.raw[:, -1], meta['bounds']['train']
    dev = logs/'G2/hot_dev_f58f2ce_r5/prepared'
    meta = read(dev/'completed.json')
    rows = load_pack(checked(dev/'scada_rows.npz', meta['artifacts']['scada_rows.npz']))
    hours = rows['hours'].astype('datetime64[ns]')
    dates = np.tile(hours, len(rows['units']))
    units = np.repeat(np.arange(len(rows['units'])), len(hours))
    winds, power = rows['raw'][:, 0], rows['raw'][:, -1]
    frame = pd.read_csv(logs/'G2/g2b84_bc7730e_r1/sealed_prep/hourly_scada.csv')
    stamp = pd.to_datetime(frame['timestamp_hour_end_utc'], utc=True).dt.tz_localize(None)
    take = stamp >= pd.Timestamp('2024-01-01')
    frame, stamp = frame[take], stamp[take]
    if (stamp >= pd.Timestamp('2026-01-01')).any():
        raise ValueError('Protected timestamp')
    mapping = {u: i for i, u in enumerate(rows['units'])}
    return (np.r_[dates, stamp.to_numpy()], np.r_[units, frame.turbine_id.map(mapping).to_numpy()],
            np.r_[winds, frame.wind_speed_mps.to_numpy()], np.r_[power, frame.power_kw.to_numpy()],
            ['2016-01-01', '2023-07-01'])


def normal_cells(wind, truth, curves, turbines, rated):
    valid = np.isfinite(wind)
    stop = (wind >= 4) & (truth <= .01*rated)
    derate = np.zeros_like(valid)
    for u in np.unique(turbines):
        take = turbines == u
        points = np.asarray(curves[str(u)]).reshape(-1, 2)
        if len(points):
            expected = np.interp(wind[take], points[:, 0], points[:, 1])
            derate[take] = (wind[take] >= 6) & (wind[take] <= 25) & (truth[take] < .5*expected)
    return valid & ~stop & ~derate, dict(missing_wind_cells=int((~valid).sum()),
        stop_cells=int(stop.sum()), derated_cells=int(derate.sum()), removed_cells=int((~(valid & ~stop & ~derate)).sum()))


def run_e2(args):
    from scripts import prepare_g1_batches as prep
    checked(ROOT/'configs/x1_protocol.json', args.protocol_sha256)
    protocol = read(ROOT/'configs/x1_protocol.json')
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'launch.json', dict(mode='E2', protocol_sha256=args.protocol_sha256,
         script_sha256=digest(__file__), target_wind_used_only_for_descriptive_mask=True))
    try:
        reports = {}
        for name, item in protocol['datasets'].items():
            path, npaths = paths(args.logs, item)
            pack = load_pack(path)
            dates, units, wind, power, bounds = raw_timeline(args.logs, item)
            fit = prep.period_mask(dates, bounds, hour_end=True)
            curves = {str(u): curve_points(wind, power, fit & (units == u)).tolist() for u in np.unique(pack['turbines'])}
            index = {(int(t), int(u)): i for i, (t, u) in enumerate(zip(dates.astype('datetime64[ns]').astype(np.int64), units))}
            target = np.array([[index.get((int(t), int(u)), -1) for t in ts]
                               for ts, u in zip(pack['times'], pack['turbines'])])
            if (target < 0).any() or not np.allclose(power[target], pack['truth'], atol=.001, rtol=0):
                raise ValueError('Target observations do not align with frozen truth: '+name)
            mask, audit = normal_cells(wind[target], pack['truth'], curves, pack['turbines'], item['rated_kw'])
            predictions = {'N': [np.load(p) for p in npaths], 'A0': [persistence(pack['truth'], pack['anchor'])]}
            a2 = read(args.a2_results/'completed.json')
            predictions['A2'] = [np.load(checked(args.a2_results/(name+'_A2.npy'), a2['predictions'][name]))]
            metrics = {}
            for arm, preds in predictions.items():
                metrics[arm] = {}
                for scope, use in [('all', np.ones_like(mask)), ('normal', mask)]:
                    if not use.any():
                        metrics[arm][scope] = None
                        continue
                    mse = float(np.mean([np.square(p-pack['truth'])[use].mean() for p in preds]))
                    mae = float(np.mean([np.abs(p-pack['truth'])[use].mean() for p in preds]))
                    variance = float(np.var(pack['truth'][use]))
                    metrics[arm][scope] = dict(mae=mae, rmse=float(np.sqrt(mse)),
                        r2=1-mse/variance if variance else None, cells=int(use.sum()))
            reports[name] = dict(descriptive_only=True, audit=audit, metrics=metrics,
                removed_fraction=float((~mask).mean()), curves_fit_bounds=bounds)
            np.save(args.output/(name+'_normal_mask.npy'), mask)
            save(args.output/(name+'.json'), reports[name])
            print('COMPLETED', name, flush=True)
        save(args.output/'results.json', reports)
        save(args.output/'completed.json', dict(status='COMPLETED', results_sha256=digest(args.output/'results.json')))
    except Exception as exc:
        save(args.output/'failed.json', dict(type=type(exc).__name__, message=str(exc), retry=False))
        raise


def run(args):
    from scripts.train_g1 import evaluate_arrays
    checked(ROOT/'configs/x1_protocol.json', args.protocol_sha256)
    protocol = read(ROOT/'configs/x1_protocol.json')
    if args.arm not in protocol['authorized_ready_arms']:
        raise ValueError('Arm not ready or not reviewed')
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'launch.json', dict(arm=args.arm, protocol_sha256=args.protocol_sha256,
        script_sha256=digest(__file__), started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        cpu_threads=2, retry=False, new_fits=0))
    try:
        reports, hashes, fitted = {}, {}, {}
        for name, item in protocol['datasets'].items():
            path, pred_paths = paths(args.logs, item)
            with np.load(path, allow_pickle=False) as z:
                truth, anchor, issue = (z[k] for k in ('truth', 'anchor', 'issue'))
                if len(truth) != item['count']:
                    raise ValueError('Cohort size mismatch')
                years = (z['times'].astype('datetime64[ns]')-np.timedelta64(1, 'ns')).astype('datetime64[Y]').astype(int)+1970
                if not np.all(years == item['year']):
                    raise ValueError('Target year mismatch')
            if args.arm == 'A0':
                baseline = persistence(truth, anchor)
                audit = {}
            elif args.arm == 'A2':
                if item['farm'] not in fitted:
                    packs = development(args.logs, item)
                    fitted[item['farm']] = fit_curves(packs, item['family'] == 'G2')
                    save(args.output/(item['farm']+'_curves.json'), fitted[item['farm']])
                    del packs
                pack = load_pack(path)
                baseline, audit = curve_predict(pack, fitted[item['farm']])
                del pack
            else:
                raise ValueError('Mode implementation not ready')
            preds = [np.load(p, allow_pickle=False) for p in pred_paths]
            if any(p.shape != truth.shape or not np.isfinite(p).all() for p in preds):
                raise ValueError('Invalid saved N predictions')
            report = evaluate_arrays({'N': preds, args.arm: [baseline]}, truth,
                issue.astype('datetime64[ns]'), None, protocol['statistics'],
                comparison_pairs={'N_vs_'+args.arm: ('N', args.arm)})
            for rows in report['metrics'].values():
                for row in rows:
                    row['one_minus_nrmse'] = 1-row['rmse']/item['rated_kw']
            np.save(args.output/(name+'_'+args.arm+'.npy'), baseline)
            hashes[name] = digest(args.output/(name+'_'+args.arm+'.npy'))
            reports[name] = report
            report['baseline_audit'] = audit
            print('COMPLETED', name, flush=True)
        passed = all(r['primary']['N_vs_'+args.arm]['gain'] > 0 and
            r['primary']['N_vs_'+args.arm]['ci95_low'] > 0 for r in reports.values())
        save(args.output/'results.json', dict(arm=args.arm, datasets=reports, all_four_passed=passed))
        save(args.output/'completed.json', dict(status='COMPLETED', new_fits=0,
            predictions=hashes, results_sha256=digest(args.output/'results.json')))
    except Exception as exc:
        save(args.output/'failed.json', dict(type=type(exc).__name__, message=str(exc), retry=False))
        raise


def tree_features(pack, hot=False):
    x = pack['x50']
    if x.ndim != 3 or x.shape[1:] != (24, 50):
        raise ValueError('Unexpected frozen feature dimensions')
    parts = [x[:, -1, :], x[:, :, -1], x[:, :, 0]]
    if hot:
        parts.append(pack['retrofit'].reshape(-1, 1))
    parts.append(pack['turbines'].reshape(-1, 1))
    result = np.concatenate(parts, axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite tree inputs')
    return result


def run_a1(args):
    import lightgbm as lgb
    from scripts.train_g1 import evaluate_arrays
    checked(ROOT/'configs/x1_protocol.json', args.protocol_sha256)
    protocol = read(ROOT/'configs/x1_protocol.json')
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'launch.json', dict(mode='A1', protocol_sha256=args.protocol_sha256,
         source_sha256=digest(__file__), lightgbm_version=lgb.__version__, threads=2))
    try:
        models, reports, hashes = {}, {}, {}
        cfg = protocol['lightgbm']
        for name, item in protocol['datasets'].items():
            farm = item['farm']
            if farm not in models:
                train, val = development(args.logs, item)
                xt, xv = [tree_features(p, item['family']=='G2') for p in (train, val)]
                yt, yv = train['truth'], val['truth']
                best, rows, selected = float('inf'), [], None
                for leaves in cfg['num_leaves']:
                    for child in cfg['min_child_samples']:
                        candidates, pv = [], []
                        for h in range(4):
                            m = lgb.LGBMRegressor(objective='regression_l1', num_leaves=leaves,
                                min_child_samples=child, learning_rate=cfg['learning_rate'],
                                n_estimators=cfg['n_estimators'], colsample_bytree=cfg['feature_fraction'],
                                subsample=cfg['bagging_fraction'], subsample_freq=1,
                                random_state=cfg['seed'], n_jobs=2, verbosity=-1,
                                deterministic=True, force_col_wise=True)
                            m.fit(xt, yt[:, h], eval_set=[(xv, yv[:, h])], eval_metric='l1',
                                  categorical_feature=[xt.shape[1]-1],
                                  callbacks=[lgb.early_stopping(cfg['early_stopping_rounds'], verbose=False)])
                            pv.append(m.predict(xv)); candidates.append(m)
                        pv = np.stack(pv, axis=1)
                        score = float(np.abs(pv-yv).mean())
                        row = dict(num_leaves=leaves, min_child_samples=child, validation_mae=score,
                                   best_iterations=[int(m.best_iteration_) for m in candidates])
                        rows.append(row)
                        print('GRID', farm, row, flush=True)
                        if score < best:
                            best, selected = score, row
                            models[farm] = candidates
                            np.save(args.output/(farm+'_validation.npy'), pv)
                checkpoint_hashes = {}
                for h, m in enumerate(models[farm]):
                    target = args.output/(farm+'_h'+str(h+1)+'.txt')
                    m.booster_.save_model(str(target))
                    checkpoint_hashes[target.name] = digest(target)
                save(args.output/(farm+'_selection.json'), dict(grid=rows, selected=selected,
                     model_sha256=checkpoint_hashes, selection='validation_mae_only',
                     validation_prediction_sha256=digest(args.output/(farm+'_validation.npy'))))
                del train, val, xt, xv, yt, yv
            path, npaths = paths(args.logs, item)
            pack = load_pack(path)
            features = tree_features(pack, item['family']=='G2')
            prediction = np.stack([m.predict(features) for m in models[farm]], axis=1)
            target = args.output/(name+'_A1.npy')
            np.save(target, prediction); hashes[name] = digest(target)
            reports[name] = evaluate_arrays({'N': [np.load(p) for p in npaths], 'A1': [prediction]},
                pack['truth'], pack['issue'].astype('datetime64[ns]'), None, protocol['statistics'],
                comparison_pairs={'N_vs_A1': ('N', 'A1')})
            save(args.output/(name+'.json'), reports[name])
            print('COMPLETED', name, flush=True)
            del pack, features, prediction
        save(args.output/'results.json', dict(arm='A1', datasets=reports))
        save(args.output/'completed.json', dict(status='COMPLETED', predictions=hashes,
             results_sha256=digest(args.output/'results.json')))
    except Exception as exc:
        save(args.output/'failed.json', dict(type=type(exc).__name__, message=str(exc), retry=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--logs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--protocol-sha256', required=True)
    p.add_argument('--arm', choices=['A0', 'A1', 'A2', 'E1', 'E2'], required=True)
    p.add_argument('--a2-results', type=Path)
    args = p.parse_args()
    if args.arm == 'A1':
        run_a1(args)
    elif args.arm == 'E1':
        run_e1(args)
    elif args.arm == 'E2':
        run_e2(args)
    else:
        run(args)
