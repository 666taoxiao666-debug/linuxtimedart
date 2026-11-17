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


def frozen_predictions(logs, item, pack, arm, seeds=(2024, 2025, 2026)):
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
        keys = [f"{item['farm']}/{arm}/{s}" for s in seeds]
    else:
        parent = logs/'G2/g2b_4471499_shared_r1/formal'
        local = ROOT/'output/g2b/freeze/formal'
        protocol = read(ROOT/'configs/g2b_protocol.json')
        factory = g2.make_model
        data = g2.arm_pack(pack, arm, True)
        keys = [f'{arm}_{s}' for s in seeds]
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


def b2_pack(pack, quantiles, variant, hot=False):
    from scripts import train_g2b as g2
    result = g2.arm_pack(pack, 'N', True) if hot else dict(pack)
    features = quantiles.reshape(len(quantiles), 12) if variant == 'quantiles' else quantiles[:, 1]
    features = (features-float(pack['power_mean']))/float(pack['power_scale'])
    x = result['x50']
    result['x50'] = np.concatenate([x[..., :-1], np.broadcast_to(features[:,None,:],
                                   (len(x),24,features.shape[1])), x[..., -1:]], -1).astype(np.float32)
    return result


def b2_model(pack, config, variant, device):
    from models.NWPDisplacement import G1TSMixerModel
    class ResidualModel(G1TSMixerModel):
        def forward(self, x, batch=None):
            return super().forward(x, batch)+x[:, -1, -5:-1].unsqueeze(-1)
    cls = ResidualModel if variant == 'residual' else G1TSMixerModel
    return cls('N', pack['x50'].shape[-1], None, blocks=config['blocks'], ff_dim=config['ff_dim'],
               dropout=config['dropout'], norm_eps=config['norm_eps']).to(device)


def run_b2(args):
    import types
    import torch
    from scripts import train_g1 as g1
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.2)
    checked(ROOT/'configs/x1_protocol.json', args.protocol_sha256)
    protocol = read(ROOT/'configs/x1_protocol.json')
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'launch.json', dict(mode='B2', script_sha256=digest(__file__),
        protocol_sha256=args.protocol_sha256, fits=27, selection='validation_only'))
    try:
        fits, scores, n_scores = [], {}, {}
        items = {item['farm']:item for item in protocol['datasets'].values()}
        for farm, item in items.items():
            config = read(ROOT/('configs/g2b_protocol.json' if farm=='HoT' else 'configs/g1_protocol.json'))['fit']
            train, val = development(args.logs, item)
            qt = checked_a3(args.a3_results, farm+'_train', train)
            qv = checked_a3(args.a3_results, farm+'_validation', val)
            n_scores[farm] = float(np.mean([np.abs(p-val['truth']).mean()
                                          for p in frozen_predictions(args.logs,item,val,'N')]))
            scores[farm] = {}
            for variant in protocol['B2']['variants']:
                tr = b2_pack(train, qt, variant, farm=='HoT')
                va = b2_pack(val, qv, variant, farm=='HoT')
                factory = lambda arm, pack, conf, device: b2_model(pack,conf,variant,device)
                fit = types.FunctionType(g1.fit_one.__code__, {**vars(g1), 'make_model':factory}, 'fit_one')
                scores[farm][variant] = []
                for seed in protocol['B2']['seeds']:
                    folder = args.output/farm/variant/str(seed)
                    result = fit('N', seed, tr, va, config, folder, 'cuda')
                    fits.append(dict(farm=farm,variant=variant,seed=seed,checkpoint=str(folder/'selected.pt'),
                                     sha256=result['sha256'],validation_mae=result['validation_mae']))
                    scores[farm][variant].append(result['validation_mae'])
                    print('FIT_COMPLETED',farm,variant,seed,flush=True)
                del tr,va
            del train,val,qt,qv
        gain = {v: {f:1-float(np.mean(scores[f][v]))/n_scores[f] for f in items}
                for v in protocol['B2']['variants']}
        winner = max(protocol['B2']['variants'],key=lambda v:np.mean(list(gain[v].values())))
        passed = all(x>=protocol['B2']['validation_gain_required'] for x in gain[winner].values())
        freeze = dict(fits=fits,validation=scores,reference=n_scores,relative_gain=gain,
                      selected=winner,passed=passed,test_started=False)
        if len(fits)!=27:
            raise ValueError('Incomplete fit budget')
        for record in fits:
            checked(record['checkpoint'],record['sha256'])
        save(args.output/'selection_freeze.json',freeze)
        reports, hashes = {}, {}
        if passed:
            save(args.output/'test_started.json',dict(freeze_sha256=digest(args.output/'selection_freeze.json')))
            for name,item in protocol['datasets'].items():
                farm=item['farm']
                path,npaths=paths(args.logs,item)
                pack=load_pack(path)
                q=checked_a3(args.a3_results,name,pack)
                data=b2_pack(pack,q,winner,farm=='HoT')
                config=read(ROOT/('configs/g2b_protocol.json' if farm=='HoT' else 'configs/g1_protocol.json'))['fit']
                predictions=[]
                for record in [r for r in fits if r['farm']==farm and r['variant']==winner]:
                    model=b2_model(data,config,winner,'cuda')
                    model.load_state_dict(torch.load(checked(record['checkpoint'],record['sha256']),map_location='cuda',weights_only=True))
                    pred=g1.predict(model,data,'cuda')
                    target=args.output/(name+'_'+str(record['seed'])+'.npy')
                    np.save(target,pred); hashes[target.name]=digest(target)
                    predictions.append(pred)
                    del model
                reports[name]=g1.evaluate_arrays({'N':[np.load(p) for p in npaths],'B2':predictions},pack['truth'],
                    pack['issue'].astype('datetime64[ns]'),None,protocol['statistics'],comparison_pairs={'B2_vs_N':('B2','N')})
        save(args.output/'results.json',dict(exploratory_only=True,passed=passed,selected=winner,datasets=reports))
        save(args.output/'completed.json',dict(status='COMPLETED',fits=27,predictions=hashes,
            results_sha256=digest(args.output/'results.json')))
    except Exception as exc:
        save(args.output/'failed.json',dict(type=type(exc).__name__,message=str(exc)))
        raise


def select_weights(predictions, truth):
    best, weights = float('inf'), None
    for a in range(11):
        for b in range(11-a):
            w = np.array([a, b, 10-a-b])/10
            score = float(np.abs(np.einsum('i,inj->nj', w, predictions)-truth).mean())
            if score < best:
                best, weights = score, w
    return weights, best


def checked_a3(directory, key, pack):
    completed = read(directory/'completed.json')
    if completed['status'] != 'COMPLETED' or (directory/'failed.json').exists():
        raise ValueError('A3 parent incomplete')
    name = key+'_quantiles.npy'
    pred = np.load(checked(directory/name, completed['predictions'][name]))
    key_file = key+'_keys.json'
    keys = read(checked(directory/key_file, completed['artifacts'][key_file]))
    for field in ('issue', 'turbines'):
        tag = 'issue_sha256' if field == 'issue' else 'turbine_sha256'
        if hashlib.sha256(pack[field].tobytes()).hexdigest() != keys[tag]:
            raise ValueError('A3 sample keys mismatch')
    if pred.shape != (len(pack['truth']), 3, 4):
        raise ValueError('A3 shape mismatch')
    return pred


def run_c1(args):
    import torch
    from scripts.train_g1 import evaluate_arrays
    torch.set_num_threads(2)
    checked(ROOT/'configs/x1_protocol.json', args.protocol_sha256)
    protocol = read(ROOT/'configs/x1_protocol.json')
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'launch.json', dict(mode='C1', new_fits=0, script_sha256=digest(__file__),
        protocol_sha256=args.protocol_sha256))
    try:
        selected, reports, hashes = {}, {}, {}
        parent = read(args.a1_results/'completed.json')
        if parent['status'] != 'COMPLETED':
            raise ValueError('A1 incomplete')
        for name, item in protocol['datasets'].items():
            farm = item['farm']
            seeds = (2024,2025,2026,2027,2028) if farm == 'HoT' else (2024,2025,2026)
            if farm not in selected:
                _, val = development(args.logs, item)
                n = np.mean(frozen_predictions(args.logs, item, val, 'N', seeds), 0)
                sel = read(args.a1_results/(farm+'_selection.json'))
                a1 = np.load(checked(args.a1_results/(farm+'_validation.npy'), sel['validation_prediction_sha256']))
                a3 = checked_a3(args.a3_results, farm+'_validation', val)[:,1]
                w, score = select_weights(np.stack([n,a1,a3]), val['truth'])
                selected[farm] = w
                save(args.output/(farm+'_selection.json'), dict(weights=w.tolist(), validation_mae=score,
                    order=['N_ensemble','A1','A3a'], seeds=seeds, selection='validation_only'))
                del val
            path, _ = paths(args.logs, item)
            pack = load_pack(path)
            nparent = args.logs/('G2/g2c_2eeb23f_r1/formal' if farm == 'HoT' else 'G1/g1c_27312d7_r2/formal')
            nr = read(nparent/'completed.json')
            names = ([f"{item['year']}_N_{s}.npy" for s in seeds] if farm=='HoT'
                     else [f'{farm}/N/{s}/prediction.npy' for s in seeds])
            n = np.mean([np.load(checked(nparent/nm, nr['predictions'][nm])) for nm in names], 0)
            a1 = np.load(checked(args.a1_results/(name+'_A1.npy'), parent['predictions'][name]))
            a3 = checked_a3(args.a3_results, name, pack)[:,1]
            pred = np.einsum('i,inj->nj', selected[farm], np.stack([n,a1,a3]))
            target = args.output/(name+'_C1.npy')
            np.save(target, pred); hashes[target.name] = digest(target)
            reports[name] = evaluate_arrays({'N_ensemble':[n], 'C1':[pred]}, pack['truth'],
                pack['issue'].astype('datetime64[ns]'), None, protocol['statistics'],
                comparison_pairs={'C1_vs_N':('C1','N_ensemble')})
            save(args.output/(name+'.json'), reports[name])
        save(args.output/'results.json', dict(descriptive_only=True,datasets=reports))
        save(args.output/'completed.json', dict(status='COMPLETED', predictions=hashes,
            results_sha256=digest(args.output/'results.json')))
    except Exception as exc:
        save(args.output/'failed.json', dict(type=type(exc).__name__,message=str(exc)))
        raise


def verify_chronos(receipt_path, manifest_path):
    receipt = read(receipt_path)
    if receipt['revision'] != '29ec3766d36d6f73f0696f85560a422f50e8498c' or receipt['manifest_sha256'] != '758b5c252d2a0bbe0237554d3c1587b4bb5165b73d96ccdcec0cf98e31d0de5e':
        raise ValueError('Chronos identity differs from user-approved receipt')
    if receipt['status'] != 'READY':
        raise ValueError('Chronos receipt is not READY')
    checked(manifest_path, receipt['manifest_sha256'])
    manifest = read(manifest_path)
    if manifest['revision'] != receipt['revision'] or len(manifest['files']) != receipt['n_files']:
        raise ValueError('Chronos identity mismatch')
    root = Path(receipt['model_dir']).resolve()
    for record in manifest['files']:
        path = (root/record['path']).resolve()
        if not path.is_relative_to(root) or path.stat().st_size != record['bytes']:
            raise ValueError('Invalid model file')
        checked(path, record['sha256'])
    return receipt


def chronos_inputs(pack, mean, scale, historical, indices):
    """Only historical inputs and issue-time GFS; never read forecast labels."""
    raw = pack['x21'][indices].astype(float)*np.asarray(scale)+np.asarray(mean)
    result = []
    for j, i in enumerate(indices):
        cov = {'wind': raw[j, :, 0], 'up_missing': raw[j, :, 19], 'gfs': historical[j]}
        cov.update({'up_'+str(k): raw[j, :, 8+k] for k in range(11)})
        future = np.where(pack['gF_mask'][i, 6:10] > 0, pack['gF'][i, 6:10], np.nan)
        result.append(dict(target=raw[j, :, -1], past_covariates=cov,
                           future_covariates={'gfs': future}))
    return result


def chronos_history(logs, item):
    if item['family'] == 'G1':
        folder = logs/'G1/g1c_27312d7_r2/prepared'/item['farm']
        meta = read(folder/'prepared.json')
        if meta != read(ROOT/'output/g1/preparation_78/prepared'/item['farm']/'prepared.json'):
            raise ValueError('Scaler metadata mismatch')
        cache = {}
        for name in ('points_0.npz', 'points_1.npz'):
            cache.update(load_pack(folder/name))
        import pandas as pd
        hours = pd.date_range(meta['bounds']['train'][0], meta['bounds']['test'][1], freq='h', inclusive='left').to_numpy()
        hourly = cached_hourly(hours, cache, meta)
        stats = meta['scaler']
    else:
        freeze = logs/'G2/g2b_4471499_shared_r1/formal'
        from scripts import train_g2b as g2, evaluate_g2b_sealed as sealed
        frozen, _ = g2.frozen_gate(freeze, sealed.FREEZE_SHA)
        stats = read(freeze/'fit_statistics.json')
        folder = logs/'G2/hot_dev_f58f2ce_r5/prepared'
        checked(folder/'completed.json', frozen['preparation_sha256'])
        meta = read(folder/'completed.json')
        hourly = load_pack(checked(folder/'nwp_hourly.npz', meta['artifacts']['nwp_hourly.npz']))
        # Sealed years use the already approved 2023-2025 point cache, never 2026.
        import pandas as pd
        hours = pd.date_range('2023-01-01T01:00', '2026-01-01', freq='h', inclusive='left').to_numpy()
        extended = cached_hourly(hours, load_pack(logs/'G2/g2b84_bc7730e_r1/sealed_prep/points.npz'), stats)
        old_hours = hourly['hours'].astype('datetime64[ns]')
        all_hours = np.union1d(old_hours, hours)
        hist = np.full((len(all_hours), 1), np.nan)
        hist[np.searchsorted(all_hours, hours), 0] = extended['historical'][:, 0]
        hist[np.searchsorted(all_hours, old_hours), 0] = hourly['historical'][:, 0]
        hourly = dict(hours=all_hours, historical=hist)
    return stats, hourly


def history_grid(packs):
    """Recover timestamped historical features only; no forecast labels enter the grid."""
    grid = {}
    for pack in packs:
        issue = pack['issue'].astype('datetime64[ns]').astype(np.int64)
        for offset in range(24):
            hours = issue - (23-offset)*3600000000000
            for i, (unit, hour) in enumerate(zip(pack['turbines'], hours)):
                key = (int(unit), int(hour))
                value = pack['x21'][i, offset]
                if key in grid and not np.array_equal(grid[key], value):
                    raise ValueError('Conflicting historical context at '+str(key))
                grid[key] = value.copy()
    keys = sorted(grid)
    return dict(grid_unit=np.array([k[0] for k in keys]),
                grid_hour=np.array([k[1] for k in keys]),
                grid_value=np.stack([grid[k] for k in keys]))


def longer_context(pack, grid, indices, lookup=None):
    lookup = lookup if lookup is not None else {(int(u), int(t)): i for i, (u, t) in
                                               enumerate(zip(grid['grid_unit'], grid['grid_hour']))}
    out = np.full((len(indices), 168, 21), np.nan, np.float32)
    for j, i in enumerate(indices):
        issue = int(pack['issue'][i].astype('datetime64[ns]').astype(np.int64))
        for offset in range(144):
            k = lookup.get((int(pack['turbines'][i]), issue-(167-offset)*3600000000000))
            if k is not None:
                out[j, offset] = grid['grid_value'][k]
        out[j, -24:] = pack['x21'][i]
    return out


def infer_chronos(pipe, pack, context_length=24, grid=None, batch_size=1024):
    qidx = [next(i for i, q in enumerate(pipe.quantiles) if abs(q-v)<1e-5) for v in (.1, .5, .9)]
    output = np.empty((len(pack['x21']), 3, 4), np.float32)
    lookup = None if grid is None else {(int(u), int(t)): i for i, (u, t) in
                                     enumerate(zip(grid['grid_unit'], grid['grid_hour']))}
    for lo in range(0, len(output), 2048):
        ix = np.arange(lo, min(lo+2048, len(output)))
        if context_length == 24:
            inputs = chronos_inputs(pack, pack['mean'], pack['scale'], pack['historical'][ix], ix)
        else:
            part = dict(x21=longer_context(pack, grid, ix, lookup), gF=pack['gF'][ix],
                        gF_mask=pack['gF_mask'][ix])
            times = pack['issue'][ix].astype('datetime64[ns]')[:, None] - np.arange(167,-1,-1)*np.timedelta64(1,'h')
            hours = grid['hours'].astype('datetime64[ns]')
            pos = np.searchsorted(hours, times)
            valid = (pos < len(hours)) & (hours[np.minimum(pos, len(hours)-1)] == times)
            hist = np.full(times.shape, np.nan)
            hist[valid] = grid['historical'][pos[valid], 0]
            inputs = chronos_inputs(part, pack['mean'], pack['scale'], hist, np.arange(len(ix)))
        prediction = pipe.predict(inputs, prediction_length=4, context_length=context_length,
                                  batch_size=batch_size, cross_learning=False)
        output[ix] = np.stack([p[0, qidx, :].detach().cpu().numpy() for p in prediction])
        if not np.isfinite(output[ix]).all():
            raise ValueError('Non-finite Chronos prediction')
        if lo % 16384 == 0:
            print('INFERENCE', context_length, lo, '/', len(output), flush=True)
    return output


def chronos_worker(args):
    import torch
    from chronos import Chronos2Pipeline
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.20)
    receipt = verify_chronos(args.chronos_receipt, args.chronos_manifest)
    pack = load_pack(checked(args.worker_input, args.worker_input_sha))
    pipe = Chronos2Pipeline.from_pretrained(receipt['model_dir'], device_map='cuda', local_files_only=True)
    output = infer_chronos(pipe, pack)
    with args.output.open('xb') as f:
        np.save(f, output)
    grid = load_pack(checked(args.history_grid, args.history_grid_sha))
    output = infer_chronos(pipe, pack, 168, grid)
    with args.output.with_name(args.output.stem+'_168.npy').open('xb') as f:
        np.save(f, output)


def chronos_training_inputs(pack):
    """A fixed 24/4 fit window: unknown future covariates remain missing."""
    rows = chronos_inputs(pack, pack['mean'], pack['scale'], pack['historical'],
                          np.arange(len(pack['x21'])))
    for i, row in enumerate(rows):
        row['target'] = np.r_[row['target'], pack['truth'][i]]
        for key, values in row['past_covariates'].items():
            future = row['future_covariates']['gfs'] if key == 'gfs' else np.full(4, np.nan)
            row['past_covariates'][key] = np.r_[values, future]
        row['future_covariates'] = {'gfs': None}
    return rows


def b1_worker(args):
    import torch
    from chronos import Chronos2Pipeline
    from transformers import TrainerCallback
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.20)
    receipt = verify_chronos(args.chronos_receipt, args.chronos_manifest)
    spec = read(checked(args.worker_input, args.worker_input_sha))
    train = load_pack(checked(spec['train'], spec['train_sha256']))
    val = load_pack(checked(spec['validation'], spec['validation_sha256']))
    args.output.mkdir(parents=True, exist_ok=False)
    selected = args.output/'selected'

    class SelectByMAE(TrainerCallback):
        def __init__(self):
            self.best = float('inf')
            self.rows = []

        def on_step_end(self, training_args, state, control, model=None, **kwargs):
            if state.global_step % 100:
                return control
            was_training = model.training
            model.eval()
            pipeline = Chronos2Pipeline(model=model)
            with torch.inference_mode():
                prediction = infer_chronos(pipeline, val, batch_size=256)[:, 1]
            score = float(np.abs(prediction-val['truth']).mean())
            if not np.isfinite(score):
                raise ValueError('Nonfinite B1 validation MAE')
            self.rows.append(dict(step=int(state.global_step), validation_mae=score))
            if score < self.best:
                self.best = score
                pipeline.save_pretrained(selected)
                np.save(args.output/'selected_validation.npy', prediction)
                save(args.output/'selection.json', dict(step=int(state.global_step), validation_mae=score))
            model.train(was_training)
            save(args.output/'validation_history.json', self.rows)
            return control

    selector = SelectByMAE()
    pipe = Chronos2Pipeline.from_pretrained(receipt['model_dir'], device_map='cuda', local_files_only=True)
    fitted = pipe.fit(chronos_training_inputs(train), prediction_length=4,
                      context_length=24, min_past=24, output_dir=args.output/'fit',
                      callbacks=[selector], seed=2024)
    if len(selector.rows) != 10 or not selected.is_dir():
        raise ValueError('B1 checkpoint selection did not complete all ten scheduled evaluations')
    model_hashes = {str(p.relative_to(args.output)): digest(p) for p in selected.rglob('*') if p.is_file()}
    save(args.output/'selection_freeze.json', dict(model_sha256=model_hashes,
         selected=read(args.output/'selection.json'), fit_steps=1000, selection='validation_median_mae',
         train_sha256=spec['train_sha256'], validation_sha256=spec['validation_sha256']))
    del fitted, pipe, train, val
    torch.cuda.empty_cache()
    for name, sha in model_hashes.items():
        checked(args.output/name, sha)
    pipe = Chronos2Pipeline.from_pretrained(selected, device_map='cuda', local_files_only=True)
    save(args.output/'test_started.json', dict(freeze_sha256=digest(args.output/'selection_freeze.json')))
    hashes = {}
    for record in spec['tests']:
        pack = load_pack(checked(record['path'], record['sha256']))
        pred = infer_chronos(pipe, pack)
        target = args.output/(record['name']+'_quantiles.npy')
        np.save(target, pred)
        hashes[target.name] = digest(target)
    save(args.output/'completed.json', dict(status='COMPLETED', predictions=hashes, fits=1,
         freeze_sha256=digest(args.output/'selection_freeze.json')))


def run_b1(args):
    import subprocess
    from scripts.train_g1 import evaluate_arrays
    checked(ROOT/'configs/x1_protocol.json', args.protocol_sha256)
    protocol = read(ROOT/'configs/x1_protocol.json')
    verify_chronos(args.chronos_receipt, args.chronos_manifest)
    parent = read(args.a3_results/'completed.json')
    if parent['status'] != 'COMPLETED':
        raise ValueError('A3 incomplete')
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'launch.json', dict(mode='B1', source_sha256=digest(__file__),
         protocol_sha256=args.protocol_sha256, fits=3, checkpoint_selection='validation_mae'))
    try:
        reports, hashes = {}, {}
        items = {item['farm']: item for item in protocol['datasets'].values()}
        for farm, item in items.items():
            train, val = development(args.logs, item)
            prepared = []
            for split, data in [('train', train), ('validation', val)]:
                # Validate A3 sample order and context identity before adding development labels.
                checked_a3(args.a3_results, farm+'_'+split, data)
                keys = read(args.a3_results/(farm+'_'+split+'_keys.json'))
                context = load_pack(checked(args.a3_results/(farm+'_'+split+'_context.npz'), keys['input_sha256']))
                context['truth'] = data['truth']
                target = args.output/(farm+'_'+split+'.npz')
                np.savez(target, **context)
                prepared.append((str(target), digest(target)))
            tests = []
            for name, test_item in protocol['datasets'].items():
                if test_item['farm'] == farm:
                    keys = read(args.a3_results/(name+'_keys.json'))
                    path = checked(args.a3_results/(name+'_context.npz'), keys['input_sha256'])
                    tests.append(dict(name=name, path=str(path), sha256=keys['input_sha256']))
            spec = args.output/(farm+'_worker.json')
            save(spec, dict(train=prepared[0][0], train_sha256=prepared[0][1],
                 validation=prepared[1][0], validation_sha256=prepared[1][1], tests=tests))
            subprocess.run([args.chronos_python, '-u', str(Path(__file__).resolve()), '--arm', 'B1-worker',
                '--logs', str(args.logs), '--output', str(args.output/farm),
                '--protocol-sha256', args.protocol_sha256, '--chronos-receipt', str(args.chronos_receipt),
                '--chronos-manifest', str(args.chronos_manifest), '--worker-input', str(spec),
                '--worker-input-sha', digest(spec)], check=True)
            done = read(args.output/farm/'completed.json')
            for rec in tests:
                name = rec['name']
                path, npaths = paths(args.logs, protocol['datasets'][name])
                pack = load_pack(path)
                file = name+'_quantiles.npy'
                prediction = np.load(checked(args.output/farm/file, done['predictions'][file]))[:, 1]
                a3 = checked_a3(args.a3_results, name, pack)[:, 1]
                hashes[farm+'/'+file] = done['predictions'][file]
                reports[name] = evaluate_arrays({'N': [np.load(p) for p in npaths], 'B1': [prediction],
                    'A3a': [a3]}, pack['truth'], pack['issue'].astype('datetime64[ns]'), None,
                    protocol['statistics'], comparison_pairs={'B1_vs_N': ('B1', 'N'), 'B1_vs_A3a': ('B1', 'A3a')})
                save(args.output/(name+'.json'), reports[name])
        save(args.output/'results.json', dict(exploratory_only=True, datasets=reports))
        save(args.output/'completed.json', dict(status='COMPLETED', fits=3, predictions=hashes,
             results_sha256=digest(args.output/'results.json')))
    except Exception as exc:
        save(args.output/'failed.json', dict(type=type(exc).__name__, message=str(exc), retry=False))
        raise


def run_a3(args):
    import subprocess
    checked(ROOT/'configs/x1_protocol.json', args.protocol_sha256)
    protocol = read(ROOT/'configs/x1_protocol.json')
    receipt = verify_chronos(args.chronos_receipt, args.chronos_manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'launch.json', dict(mode='A3a/A3b', script_sha256=digest(__file__),
        protocol_sha256=args.protocol_sha256, model=receipt, context_lengths=[24,168], cross_learning=False))
    try:
        done, hashes, reports, grids, artifacts = set(), {}, {}, {}, {}
        for name, item in protocol['datasets'].items():
            stats, hourly = chronos_history(args.logs, item)
            hours = hourly['hours'].astype('datetime64[ns]')
            tasks = []
            if item['farm'] not in done:
                tr, va = development(args.logs, item)
                tasks.extend([(item['farm']+'_train', tr, None), (item['farm']+'_validation', va, None)])
                # Frozen historical windows provide an existing hourly series. Missing older hours
                # remain NaN; all cohorts and the last 24 hours remain exactly unchanged.
                source_packs = [tr, va]
                for other in protocol['datasets'].values():
                    if other['farm'] == item['farm']:
                        source_packs.append(load_pack(paths(args.logs, other)[0]))
                grid = history_grid(source_packs)
                grid.update(hours=hours, historical=hourly['historical'])
                gridpath = args.output/(item['farm']+'_history.npz')
                np.savez(gridpath, **grid)
                grids[item['farm']] = (gridpath, digest(gridpath))
                artifacts[gridpath.name] = digest(gridpath)
                del grid, source_packs
                done.add(item['farm'])
            path, npaths = paths(args.logs, item)
            tasks.append((name, load_pack(path), npaths))
            for key, pack, reference in tasks:
                times = pack['issue'].astype('datetime64[ns]')[:, None]-np.arange(23, -1, -1)*np.timedelta64(1, 'h')
                pos = np.searchsorted(hours, times)
                if (pos >= len(hours)).any() or not np.array_equal(hours[pos], times):
                    raise ValueError('Missing historical GFS hour')
                context = args.output/(key+'_context.npz')
                np.savez(context, x21=pack['x21'], gF=pack['gF'], gF_mask=pack['gF_mask'],
                         historical=hourly['historical'][pos, 0], mean=stats['mean'], scale=stats['scale'],
                         issue=pack['issue'], turbines=pack['turbines'])
                target = args.output/(key+'_quantiles.npy')
                input_sha = digest(context)
                subprocess.run([args.chronos_python, '-u', str(Path(__file__).resolve()), '--arm', 'A3-worker',
                    '--logs', str(args.logs), '--output', str(target), '--protocol-sha256', args.protocol_sha256,
                    '--chronos-receipt', str(args.chronos_receipt), '--chronos-manifest', str(args.chronos_manifest),
                    '--worker-input', str(context), '--worker-input-sha', input_sha,
                    '--history-grid', str(grids[item['farm']][0]), '--history-grid-sha', grids[item['farm']][1]], check=True)
                output = np.load(target)
                hashes[target.name] = digest(target)
                target168 = target.with_name(target.stem+'_168.npy')
                hashes[target168.name] = digest(target168)
                save(args.output/(key+'_keys.json'), dict(windows=len(output),
                    input_sha256=input_sha,
                    issue_sha256=hashlib.sha256(pack['issue'].tobytes()).hexdigest(),
                    turbine_sha256=hashlib.sha256(pack['turbines'].tobytes()).hexdigest()))
                artifacts[key+'_keys.json'] = digest(args.output/(key+'_keys.json'))
                artifacts[context.name] = input_sha
                if reference is not None:
                    from scripts.train_g1 import evaluate_arrays
                    reports[key] = evaluate_arrays({'N': [np.load(p) for p in reference], 'A3a': [output[:, 1]],
                        'A3b': [np.load(target168)[:, 1]]},
                        pack['truth'], pack['issue'].astype('datetime64[ns]'), None, protocol['statistics'],
                        comparison_pairs={'N_vs_A3a': ('N', 'A3a'), 'N_vs_A3b': ('N', 'A3b')})
                    save(args.output/(key+'.json'), reports[key])
                print('COMPLETED', key, flush=True)
            del tasks
        save(args.output/'results.json', reports)
        save(args.output/'completed.json', dict(status='COMPLETED', predictions=hashes, artifacts=artifacts,
            results_sha256=digest(args.output/'results.json'), new_fits=0))
    except Exception as exc:
        save(args.output/'failed.json', dict(type=type(exc).__name__, message=str(exc)))
        raise


def dense_equivalence(pack, old, overlap, tolerance_kw):
    """truth and anchor are raw hourly power in kW from the same source (registered 0.01 kW, float32 CSV
    round trip, X1-E1 decision 2026-11-17); normalised model inputs stay at 1e-5."""
    deltas = {}
    for key in ('truth', 'anchor', 'x21', 'x50', 'gF', 'gF_mask'):
        current, previous = pack[key][overlap], old[key]
        tolerance = tolerance_kw if key in ('truth', 'anchor') else 1e-5
        if current.shape != previous.shape or not np.allclose(current, previous, rtol=0, atol=tolerance):
            raise ValueError('Dense input equivalence failed: '+key)
        deltas[key] = float(np.max(np.abs(current-previous)))
    return deltas


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
            if args.e1_hot_only and item['farm'] != 'HoT':
                continue
            path, _ = paths(args.logs, item)
            old = load_pack(path)
            pack = dense_pack(args.logs, item)
            index = {(int(t), int(u)): i for i, (t, u) in enumerate(zip(pack['issue'], pack['turbines']))}
            overlap = np.array([index[(int(t), int(u))] for t, u in zip(old['issue'], old['turbines'])])
            input_deltas = dense_equivalence(pack, old, overlap, protocol['E1']['equivalence_kw'])
            report = dict(windows=len(pack['truth']), overlap_windows=len(old['truth']), metrics={}, equivalence={})
            report['input_max_abs_difference'] = input_deltas
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


def run_e2_saved(args):
    checked(ROOT/'configs/x1_protocol.json', args.protocol_sha256)
    protocol = read(ROOT/'configs/x1_protocol.json')
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    original = read(checked(args.e2_results/'results.json', read(args.e2_results/'completed.json')['results_sha256']))
    a1 = read(args.a1_results/'completed.json')
    checked(args.a1_results/'results.json', a1['results_sha256'])
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'launch.json', dict(mode='E2-saved', new_predictions=0, new_fits=0,
         protocol_sha256=args.protocol_sha256, script_sha256=digest(__file__)))
    try:
        reports = {}
        for name, item in protocol['datasets'].items():
            path, _ = paths(args.logs, item)
            pack = load_pack(path)
            mask_path = args.e2_results/(name+'_normal_mask.npy')
            mask = np.load(mask_path)
            if mask.dtype != bool or mask.shape != pack['truth'].shape:
                raise ValueError('Invalid saved normal mask')
            report = original[name]
            if int(mask.sum()) != report['metrics']['N']['normal']['cells']:
                raise ValueError('Saved mask count mismatch')
            report['mask_sha256'] = digest(mask_path)
            report['prediction_sha256'] = {}
            predictions = {'A1': [checked(args.a1_results/(name+'_A1.npy'), a1['predictions'][name])]}
            for arm in ('B', 'B0', 'N0'):
                if item['family'] == 'G1':
                    parent = args.logs/('G1/g1c_27312d7_r2/formal' if arm=='B' else 'G2/g2a_e72e462_r1/formal')
                    local = ROOT/('output/g1/results_78/formal' if arm=='B' else 'output/g2a/results81')
                    names = [f"{item['farm']}/{arm}/{s}/prediction.npy" for s in (2024, 2025, 2026)]
                else:
                    parent = args.logs/'G2/g2c_2eeb23f_r1/formal'
                    local = ROOT/'output/g2b/sealed_eval/server'
                    names = [f"{item['year']}_{arm}_{s}.npy" for s in (2024, 2025, 2026)]
                receipt = read(local/'completed.json')
                if receipt != read(parent/'completed.json'):
                    raise ValueError('Parent receipt mismatch')
                predictions[arm] = [checked(parent/n, receipt['predictions'][n]) for n in names]
            for arm, files in predictions.items():
                preds = [np.load(p) for p in files]
                if any(p.shape != mask.shape or not np.isfinite(p).all() for p in preds):
                    raise ValueError('Invalid saved prediction')
                report['prediction_sha256'][arm] = [digest(p) for p in files]
                report['metrics'][arm] = {}
                for scope, use in [('all', np.ones_like(mask)), ('normal', mask)]:
                    mse = float(np.mean([np.square(p-pack['truth'])[use].mean() for p in preds]))
                    variance = float(np.var(pack['truth'][use]))
                    report['metrics'][arm][scope] = dict(
                        mae=float(np.mean([np.abs(p-pack['truth'])[use].mean() for p in preds])),
                        rmse=float(np.sqrt(mse)), r2=1-mse/variance if variance else None,
                        cells=int(use.sum()))
            reports[name] = report
            print('COMPLETED', name, flush=True)
        save(args.output/'results.json', reports)
        save(args.output/'completed.json', dict(status='COMPLETED', results_sha256=digest(args.output/'results.json')))
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
    if lgb.__version__ != '4.5.0' or Path(sys.prefix).name != 'timedart':
        raise ValueError('LightGBM fits require timedart / 4.5.0')
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
    p.add_argument('--arm', choices=['A0', 'A1', 'A2', 'A3', 'A3-worker', 'B1', 'B1-worker', 'B2', 'C1', 'E1', 'E2', 'E2-saved'], required=True)
    p.add_argument('--a2-results', type=Path)
    p.add_argument('--a1-results', type=Path)
    p.add_argument('--a3-results', type=Path)
    p.add_argument('--e2-results', type=Path)
    p.add_argument('--e1-hot-only', action='store_true')
    p.add_argument('--chronos-receipt', type=Path)
    p.add_argument('--chronos-manifest', type=Path)
    p.add_argument('--chronos-python', default='/home/XT/miniconda3/envs/tsfm/bin/python')
    p.add_argument('--worker-input', type=Path)
    p.add_argument('--worker-input-sha')
    p.add_argument('--history-grid', type=Path)
    p.add_argument('--history-grid-sha')
    args = p.parse_args()
    if args.arm == 'B1':
        run_b1(args)
    elif args.arm == 'B1-worker':
        b1_worker(args)
    elif args.arm == 'B2':
        run_b2(args)
    elif args.arm == 'C1':
        run_c1(args)
    elif args.arm == 'A3-worker':
        chronos_worker(args)
    elif args.arm == 'A3':
        run_a3(args)
    elif args.arm == 'E2-saved':
        run_e2_saved(args)
    elif args.arm == 'A1':
        run_a1(args)
    elif args.arm == 'E1':
        run_e1(args)
    elif args.arm == 'E2':
        run_e2(args)
    else:
        run(args)
