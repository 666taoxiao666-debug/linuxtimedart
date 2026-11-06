"""C0: frozen P2 predictors, past-only calibration, one descriptive evaluation.

No training of point predictors, downloads, label edits, or repeat evaluation.
Density-ratio fitting consumes only deduplicated hourly H11/direction features.
"""
from __future__ import annotations
import argparse
import hashlib
import heapq
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CUTOFF = np.datetime64('2021-12-31T23:00', 'ns')
METHODS = ('B0', 'B1', 'B2', 'B3', 'M', 'M+')


def causal_check(calibration_times, issue):
    times = np.asarray(calibration_times)
    if times.size == 0 or np.isnat(times).any() or times.max() > CUTOFF or np.any(issue < times.max()):
        raise ValueError('Calibration labels unavailable at forecast issue')


def weighted_quantile(errors, weights, own_weight, alpha):
    """Positive-infinity test mass, including finite sample correction."""
    if not 0 < alpha < 1:
        return np.inf if alpha <= 0 else 0.
    order = np.argsort(errors, kind='stable')
    return sorted_quantile(np.asarray(errors)[order], np.cumsum(np.asarray(weights)[order]), own_weight, alpha)


def sorted_quantile(errors, cumulative, own_weight, alpha):
    if alpha <= 0:
        return np.inf
    if alpha >= 1:
        return 0.
    threshold = (1-alpha) * (cumulative[-1]+own_weight)
    index = np.searchsorted(cumulative, threshold, side='left')
    return float(errors[index]) if index < len(errors) else np.inf


def ratio_fit(source, target):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    x = np.concatenate([source, target])
    if x.shape[1] != 20:
        raise ValueError('H11 plus eight sectors and missing category required')
    model = make_pipeline(SimpleImputer(strategy='mean', keep_empty_features=True), StandardScaler(),
                          LogisticRegression(C=1., penalty='l2', solver='lbfgs', max_iter=2000, random_state=2024))
    y = np.r_[np.zeros(len(source)), np.ones(len(target))]
    model.fit(x, y)
    if model[-1].n_iter_.max() >= 2000:
        raise ValueError('Density ratio did not converge')
    return model, len(source)/len(target)


def density(model, prior, x):
    p = model.predict_proba(x)[:, 1]
    return np.clip(prior*p/np.maximum(1-p, 1e-12), .05, 20.)


def direction_features(h, directions):
    direction = np.asarray(directions)
    sectors = np.full(len(direction), 8, int)
    finite = np.isfinite(direction)
    sectors[finite] = (np.floor((direction[finite] % 360)/45).astype(int))
    return np.c_[h, np.eye(9)[sectors]]


def metrics(radius, error, alpha, mask=None):
    if mask is not None:
        radius, error = radius[mask], error[mask]
    if not error.size:
        return {'n': 0}
    score = 2*radius + 2/alpha*np.maximum(error-radius, 0)
    return {'n': int(error.size), 'coverage': float(np.mean(error <= radius)),
            'width_kw': float(np.mean(2*radius)), 'winkler_kw': float(np.mean(score)),
            'infinite_intervals': int(np.isinf(radius).sum())}


def intervals(source, target_cal, test, weights, alpha):
    """Each turbine/horizon has its own ACI state; same-issue order invariant.

    Feedback queues contain errors of the interval actually issued. They are
    consumed only when that interval's label timestamp is <= current issue.
    """
    causal_check(source['times'], test['issue'])
    causal_check(target_cal['times'], test['issue'])
    pred, truth = test['prediction'], test['truth']
    error = np.abs(pred-truth)
    n, nh = pred.shape
    radii = {m: np.empty((n, nh)) for m in METHODS}
    srcerr = np.abs(source['prediction']-source['truth'])
    tarerr = np.abs(target_cal['prediction']-target_cal['truth'])
    ws, wt = weights
    order = np.lexsort((test['turbines'], test['issue']))
    for h in range(nh):
        edges = np.unique(np.quantile(source['prediction'][:, h], [.2,.4,.6,.8]))
        bins = np.searchsorted(edges, source['prediction'][:, h], side='right')
        groups = {}
        for b in np.unique(bins):
            ix = np.flatnonzero(bins == b)
            ix = ix[np.argsort(srcerr[ix, h], kind='stable')]
            groups[int(b)] = (srcerr[ix, h], np.cumsum(ws[ix]))
        es = np.sort(srcerr[:, h]); cs = np.arange(1, len(es)+1)
        q0 = sorted_quantile(es, cs, 1., alpha)
        q1 = weighted_quantile(tarerr[:, h], np.ones(len(tarerr)), 1., alpha)
        sorted_power = np.argsort(source['prediction'][:, h], kind='stable')
        power = source['prediction'][sorted_power, h]
        states, queues = {}, {}
        for i in order:
            turbine = int(test['turbines'][i]); now = test['issue'][i]
            key = turbine
            if key not in states:
                states[key] = [alpha, alpha]; queues[key] = []
            while queues[key] and queues[key][0][0] <= now:
                _, _, miss0, missm = heapq.heappop(queues[key])
                states[key][0] += .005*(alpha-miss0)
                states[key][1] += .005*(alpha-missm)
            b = int(np.searchsorted(edges, pred[i, h], side='right'))
            # Empty quantile bins merge into the closest nonempty bin; ties lower.
            b = min(groups, key=lambda g: (abs(g-b), g))
            eg, cg = groups[b]
            qm = sorted_quantile(eg, cg, wt[i], alpha)
            q2 = sorted_quantile(es, cs, 1., states[key][0])
            qplus = sorted_quantile(eg, cg, wt[i], states[key][1])
            j = np.searchsorted(power, pred[i, h]); lo=max(0,j-500); hi=min(len(power),j+500)
            pool = sorted_power[lo:hi]
            nearest = pool[np.argsort(np.abs(source['prediction'][pool,h]-pred[i,h]), kind='stable')[:500]]
            q3 = weighted_quantile(srcerr[nearest,h], np.ones(len(nearest)), 1., alpha)
            for m, q in zip(METHODS, (q0,q1,q2,q3,qm,qplus)):
                radii[m][i,h] = q
            arrival = test['times'][i,h]
            if arrival != now + np.timedelta64(h+1, 'h'):
                raise ValueError('ACI label delay is not the registered horizon')
            heapq.heappush(queues[key], (arrival, int(i), float(error[i,h]>q2), float(error[i,h]>qplus)))
    reports = {}
    for m, radius in radii.items():
        reports[m] = {'pooled': metrics(radius,error,alpha),
                      'horizons': [metrics(radius[:,h],error[:,h],alpha) for h in range(nh)],
                      'missing': metrics(radius,error,alpha,test['missing']),
                      'observed': metrics(radius,error,alpha,~test['missing'])}
    return reports, radii


def verdict(means):
    eligible = [m for m in ('B0','B1','B3') if means[m]['coverage'] >= .88]
    eligible = eligible or ['B0','B1','B3']
    best = min(means[m]['winkler_kw'] for m in eligible)
    m = means['M']
    go = .88 <= m['coverage'] <= .92 and np.isfinite(m['winkler_kw']) and m['winkler_kw'] <= .95*best
    return {'decision': 'continue' if go else 'stop', 'eligible_comparators': eligible,
            'best_comparator_score': best, 'score_threshold': .95*best,
            'scope': 'development only; previously opened 2022; no coverage guarantee'}


def run(data_root, out, device):
    import pandas as pd
    import torch
    from scripts import train_penmanshiel_p2_v35 as p2
    from scripts import train_n1ph as n1
    u, trainer = p2.u, p2.trainer
    recipe=u.read_json(p2.PROTOCOL); old=u.read_json(ROOT/'configs/n1ph_protocol.json')
    identities={}
    pf,pup,meta,kf,kup,km=n1.other_inputs(data_root,old,recipe,identities)
    splits={'train':[2016,2017,2018,2019], 'validation':[2020], 'calibration':[2021], 'test':[2022]}
    kel=u.KelmarshUpstreamSeries(kf,kup,splits=splits,start='2016-05-01',end='2023-01-01T01:00:00Z',preparation=km)
    pen=u.KelmarshUpstreamSeries(pf,pup,splits=splits,start=meta['start'],end='2023-01-01T01:00:00Z',preparation=meta)
    # Frozen source statistics are taken from the committed training contract.
    records=[u.read_json(ROOT/f'output/penmanshiel_hourly/p2_v35/fits/K-upstream/s{s}/trained.json') for s in p2.SEEDS]
    state=records[0]['contract']['statistics']
    for r in records:
        if r['contract']['statistics'] != state:
            raise ValueError('P2 seeds have different preprocessing states')
    kel=p2.adapt_statistics(kel,kup,state=state); pen=p2.adapt_statistics(pen,pup,state=state)

    def windows(series,lo,hi):
        s=series.starts; times=series.dates[s[:,None]+24+np.arange(4)]
        keep=(times[:,0]>=np.datetime64(lo)) & (times[:,-1]<=np.datetime64(hi))
        return p2.MultiYearWindows(series,s[keep],'C0','all')
    source=windows(kel,'2021-01-01','2021-12-31T23:00')
    target=windows(pen,'2021-12-01','2021-12-31T23:00')
    test=pen.windows('test')  # same original 2022 queue; hour-ending labels unchanged

    def nearest_direction(is_pen, labels):
        if is_pen:
            part=u.read_json(ROOT/p2.farm.D2_SUBDIR/'part_A.json')
            stations=[s for s in part['midas']['stations'] if s['src_id'] in p2.farm.STATIONS]
        else:
            stations=recipe['data']['upstream']['stations']
        station=min(stations,key=lambda s:(s['distance_km'],s['src_id']))
        sid=int(station['src_id'])
        registry=u.read_json(data_root/old['p3_run']/'prepared/input_hashes.json') if is_pen else None
        files=[]
        for year in (2020,2021,2022):
            if is_pen:
                candidates=[(p,e) for p,e in registry.items() if Path(p).parent.name==str(sid) and Path(p).name.endswith(f'_qcv-1_{year}.csv')]
                if len(candidates)!=1: raise ValueError('Ambiguous nearest-station file')
                name,e=candidates[0]; path=data_root/'datasets'/name.split('/datasets/',1)[1]
                n1.checked(path,e['sha256'],identities)
                files.append({'path':path,'year':year})
            else:
                spec=recipe['data']['upstream']; hashes=u.manifest_hashes(u.read_json(data_root/spec['download_manifest']))
                for root in spec['development_roots']:
                    if year in root['years']:
                        files+=u.station_files(data_root/root['root'],[sid],[year],hashes,True)[0]
        obs=u.load_station_observations(files,[sid])[sid]
        for e in files: identities[str(e['path'])]=u.file_sha256(e['path'])
        return obs.reindex(pd.DatetimeIndex(labels)-pd.Timedelta(hours=1)).wind_direction.to_numpy()

    # One row per hour, never one repeated row per turbine. No power/labels enter ratio fitting.
    def context(frame,is_pen):
        labels=pd.date_range('2021-01-01','2021-12-31T23:00',freq='h')
        return direction_features(frame.reindex(labels).to_numpy(float),nearest_direction(is_pen,labels))
    ratio,prior=ratio_fit(context(kup,False),context(pup,True))
    def pack(w,frame,is_pen):
        issue=w.target_times()[:,0]-np.timedelta64(1,'h')
        raw=frame.reindex(pd.DatetimeIndex(issue)).to_numpy(float)
        return {'truth':w.truth_and_anchor()[0], 'times':w.target_times(), 'issue':issue,
                'turbines':w.turbines[w.window_starts], 'missing':~np.isfinite(raw).all(1),
                'features':direction_features(raw,nearest_direction(is_pen,issue)), 'starts':w.window_starts}
    packs=[pack(source,kup,False),pack(target,pup,True),pack(test,pup,True)]
    causal_check(packs[0]['times'],packs[2]['issue']); causal_check(packs[1]['times'],packs[2]['issue'])
    ws,wt=density(ratio,prior,packs[0]['features']),density(ratio,prior,packs[2]['features'])
    audit={'protocol':'STAGE_PLAN at launch commit', 'calibration_cutoff':str(CUTOFF),
           'windows':dict(zip(('source','target_month','evaluation'),[len(w) for w in (source,target,test)])),
           'aci':'independent state per turbine/horizon; observed feedback only; same-issue order invariant',
           'density':'unique hourly 2021 rows; H11+9 sector categories; mean fill from pooled unlabeled rows',
           'input_hashes':identities, 'checkpoint_hashes':{}, 'point_predictor_training':False}
    for seed,r in zip(p2.SEEDS,records):
        folder=data_root/old['p2_run']/f'fits/K-upstream/s{seed}'
        trainer.verify_trained(folder,r['contract'])
        if u.canonical_sha256(u.read_json(folder/'trained.json'))!=u.canonical_sha256(r):
            raise ValueError('P2 registration changed')
        for name,digest in r['artifacts'].items(): n1.checked(folder/name,digest,identities)
        audit['checkpoint_hashes'][str(seed)]=r['artifacts']['best.pt']
    p2.save(out/'calibration_freeze.json',audit)
    with (out/'evaluation_started.json').open('x') as stream: json.dump({'utc':p2.now()},stream)
    per_seed=[]
    for seed,r in zip(p2.SEEDS,records):
        model,_=trainer.build_model(p2.configuration('K-upstream',recipe))
        state=torch.load(data_root/old['p2_run']/f'fits/K-upstream/s{seed}/best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'],strict=True); model.to(device)
        for name,w,p in zip(('source','target_month','evaluation'),(source,target,test),packs):
            p['prediction']=trainer.v26run.predict(model,p2.view(w,'K-upstream'),seed,device,512)
            if not np.isfinite(p['prediction']).all(): raise ValueError('Nonfinite point prediction')
            np.savez_compressed(out/f'{name}_s{seed}.npz',**p)
        result={'seed':seed}
        for alpha in (.1,.2):
            report,radii=intervals(*packs,(ws,wt),alpha)
            result[str(alpha)]=report
            np.savez_compressed(out/f'intervals_{alpha}_s{seed}.npz',**radii)
        p2.save(out/f'seed_{seed}.json',result); per_seed.append(result)
        print(f'[C0] seed {seed} completed',flush=True)
    means={m:{k:float(np.mean([r['0.1'][m]['pooled'][k] for r in per_seed]))
              for k in ('coverage','width_kw','winkler_kw')} for m in METHODS}
    result={'status':'COMPLETED','seeds':list(p2.SEEDS),'mean_90':means,**verdict(means),
            'M_plus_minus_B2':{k:means['M+'][k]-means['B2'][k] for k in means['M+']},
            'artifacts':{p.name:u.file_sha256(p) for p in out.glob('*.npz')}}
    p2.save(out/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data-root',type=Path); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cpu'); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    from scripts import train_penmanshiel_p2_v35 as p2
    import torch
    torch.set_num_threads(2)
    started=time.monotonic()
    p2.save(args.out/'launch.json',{'argv':sys.argv,'utc':p2.now(),'source_code':p2.source_code(),
                                  'script_sha256':p2.u.file_sha256(__file__), 'plan_sha256_lf':p2.u.text_sha256_lf(ROOT/'docs/STAGE_PLAN.md')})
    try:
        if args.smoke:
            rng=np.random.default_rng(2024)
            def fake(n,start):
                issue=np.datetime64(start,'ns')+np.arange(n)*np.timedelta64(3,'h')
                return {'prediction':rng.normal(size=(n,4)), 'truth':rng.normal(size=(n,4)),
                        'issue':issue,'times':issue[:,None]+np.arange(1,5)*np.timedelta64(1,'h'),
                        'turbines':np.ones(n,int),'missing':np.zeros(n,bool)}
            s,t,e=fake(600,'2021-01-01'),fake(100,'2021-12-01'),fake(50,'2022-01-01')
            report,_=intervals(s,t,e,(np.ones(600),np.ones(50)),.1)
            result={'smoke':True,'status':'PASSED','methods':list(report)}
            p2.save(args.out/'result.json',result)
        else: result=run(args.data_root,args.out,torch.device(args.device))
        p2.save(args.out/'completed.json',{'utc':p2.now(),'seconds':time.monotonic()-started,'status':result['status']})
    except Exception as exc:
        p2.save(args.out/'failed.json',{'utc':p2.now(),'error':str(exc),'type':type(exc).__name__})
        raise


if __name__=='__main__': main()
