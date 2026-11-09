"""CW1, frozen P2 predictions and past-only weather weighting; one instance."""
from __future__ import annotations
import argparse
import datetime as dt
import heapq
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import pilot_conformal_c0 as c0
from scripts import pilot_conformal_c0b as cb
save, sha, finite = cb.save, cb.sha, cb.finite
BASE = ('B0', 'B1', 'B2-A', 'B3-A')
VARIANTS = ('W1', 'W2', 'W3', 'W4')


def weather_fit(source, target, source_times, target_times, year):
    """Only past unique-hour features; caller supplies no evaluation features."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.pipeline import make_pipeline
    issue = np.array([f'{year}-01-01'], dtype='datetime64[ns]')
    cb.causal_check(source_times, issue, year); cb.causal_check(target_times, issue, year)
    return make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(),
                         KMeans(n_clusters=4, random_state=2024, n_init=10)).fit(np.r_[source, target])


def hgb_features(pack, h):
    hour = pack['issue'].astype('datetime64[h]').astype(np.int64) % 24
    return np.c_[pack['features'], np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24), pack['prediction'][:, h]]


def hgb_weights(source, target, test, year):
    from sklearn.ensemble import HistGradientBoostingClassifier
    cb.causal_check(source['times'], test['issue'], year)
    cb.causal_check(target['times'], test['issue'], year)
    result = []
    for h in range(4):
        xs, xt = hgb_features(source, h), hgb_features(target, h)
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_leaf_nodes=15,
                   min_samples_leaf=100, l2_regularization=1., random_state=2024, early_stopping=False)
        model.fit(np.r_[xs, xt], np.r_[np.zeros(len(xs)), np.ones(len(xt))])
        prior = len(xs)/len(xt)
        result.append((c0.density(model, prior, xs), c0.density(model, prior, hgb_features(test, h))))
    return result


def intervals(source, target_month, test, weights, year, methods, groups=None, power_edges=None, strong=None):
    """State per method/turbine/h/issue-regime; feedback returns to original key."""
    cb.causal_check(source['times'], test['issue'], year)
    cb.causal_check(target_month['times'], test['issue'], year)
    for pack in (source, target_month, test):
        finite(pack['truth'], 'truth'); finite(pack['prediction'], 'prediction')
    se = np.abs(source['truth']-source['prediction']); te = np.abs(target_month['truth']-target_month['prediction'])
    error = np.abs(test['truth']-test['prediction']); ws, wt = weights
    finite(ws, 'source weights'); finite(wt, 'target weights')
    if np.any(ws <= 0) or np.any(wt <= 0): raise ValueError('Nonpositive weights')
    radius = {m: np.empty(error.shape) for m in methods}
    caps = {m: np.zeros(error.shape, bool) for m in methods}
    order = np.lexsort((test['turbines'], test['issue']))
    for h in range(4):
        edges = np.unique(np.quantile(source['prediction'][:, h], [.2,.4,.6,.8]))
        bins = np.searchsorted(edges, source['prediction'][:, h], side='right')
        pools = {}
        for b in np.unique(bins):
            ix = np.flatnonzero(bins == b); ix = ix[np.argsort(se[ix, h], kind='stable')]
            pools[int(b)] = ix
        es = np.sort(se[:, h]); cs = np.arange(1, len(es)+1)
        q0 = cb.finite_sorted_quantile(es, cs, 1., .1)
        q1 = cb.finite_quantile(te[:, h], np.ones(len(te)), 1., .1)
        sorted_power = np.argsort(source['prediction'][:, h], kind='stable')
        power = source['prediction'][sorted_power, h]
        states, queue = {}, {}
        for i in order:
            turbine, now = int(test['turbines'][i]), test['issue'][i]
            events = queue.setdefault(turbine, [])
            while events and events[0][0] <= now:
                _, _, feedback = heapq.heappop(events)
                for key, miss in feedback:
                    states[key] = float(np.clip(states[key]+.005*(.1-miss), .005, .5))
            p = test['prediction'][i, h]
            b = int(np.searchsorted(edges, p, side='right')); b = min(pools, key=lambda g:(abs(g-b),g))
            ix = pools[b]
            j = np.searchsorted(power, p); pool = sorted_power[max(0,j-500):min(len(power),j+500)]
            nearest = pool[np.argsort(np.abs(source['prediction'][pool,h]-p), kind='stable')[:500]]
            nearest = nearest[np.argsort(se[nearest,h],kind='stable')]
            feedback = []
            for m in methods:
                if m in ('B0','B1'):
                    q, cap = q0 if m == 'B0' else q1
                else:
                    group = int(groups[i]) if m == 'W4' else (int(np.searchsorted(power_edges[h],p,side='right')) if m == 'B3-G' else 0)
                    key = (m, turbine, group)  # h has its own outer-loop states
                    a = states.setdefault(key, .1)
                    if m == 'B2-A': e, c, own = es, cs, 1.
                    elif m in ('B3-A','B3-G'): e, c, own = se[nearest,h], np.arange(1,len(nearest)+1), 1.
                    elif m == 'W1': e, c, own = se[nearest,h], np.cumsum(ws[nearest]), wt[i]
                    elif m == 'W2': e, c, own = se[ix,h], np.cumsum(strong[h][0][ix]), strong[h][1][i]
                    else: e, c, own = se[ix,h], np.cumsum(ws[ix]), wt[i]  # M-A and W4
                    q, cap = cb.finite_sorted_quantile(e,c,own,a)
                    feedback.append((key,float(error[i,h]>q)))
                finite(q,'emitted radius'); radius[m][i,h], caps[m][i,h] = q,cap
            arrival = test['times'][i,h]
            if arrival != now + np.timedelta64(h+1,'h'): raise ValueError('Incorrect feedback delay')
            heapq.heappush(events,(arrival,int(i),feedback))
    return radius,caps


def reports(radii, caps, error):
    return {m:{'pooled':cb.checked_metrics(r,error,caps[m]),
               'horizons':[cb.checked_metrics(r[:,h],error[:,h],caps[m][:,h]) for h in range(4)]}
            for m,r in radii.items()}


def combination(radii, caps, choices):
    return (np.column_stack([radii[m][:,h] for h,m in enumerate(choices)]),
            np.column_stack([caps[m][:,h] for h,m in enumerate(choices)]))


def choose(year_reports):
    choices = []
    for h in range(4):
        scores = {m:np.mean([r['horizons'][m][h]['winkler_kw'] for r in year_reports]) for m in ('M-A','B3-A')}
        choices.append('M-A' if scores['M-A'] < scores['B3-A'] else 'B3-A')
    return choices


def select_variant(year_reports):
    eligible = [m for m in VARIANTS if all(.88 <= r['mean_90'][m]['coverage'] <= .92 for r in year_reports)]
    gains = {m:float(np.mean([100*(1-r['mean_90'][m]['winkler_kw']/r['mean_90']['B3-A']['winkler_kw']) for r in year_reports])) for m in VARIANTS}
    selected = min(eligible,key=lambda m:(-gains[m],VARIANTS.index(m))) if eligible else None
    return selected,gains


def paired_ci(loss_difference, issue, out=None, replicates=5000):
    """Seed-mean loss [windows,h]; preserve all turbine/h counts per issue."""
    finite(loss_difference,'bootstrap losses')
    times, inverse = np.unique(issue,return_inverse=True)
    sums = np.bincount(inverse,weights=loss_difference.sum(axis=1))
    counts = np.bincount(inverse)*loss_difference.shape[1]
    n=len(times); block=min(8,n); needed=(n+block-1)//block
    rng=np.random.default_rng(2024)
    starts=rng.integers(0,n-block+1,size=(replicates,needed),dtype=np.int32)
    estimates=np.empty(replicates)
    for k,row in enumerate(starts):
        index=(row[:,None]+np.arange(block)).ravel()[:n]
        estimates[k]=sums[index].sum()/counts[index].sum()
    low,high=np.quantile(estimates,[.025,.975])
    if out is not None: np.savez_compressed(out,issue=times,block_starts=starts,block_length=block,truncate_to=n,cluster_sums=sums,cluster_counts=counts,estimates=estimates)
    return {'difference':'comparator-minus-candidate','mean_gain_kw':float(sums.sum()/counts.sum()),
            'ci95_low_kw':float(low),'ci95_high_kw':float(high),'block_length':block,'replicates':replicates,'seed':2024}


def verdict(means, selected, cis):
    eligible=[m for m in ('B0','B1','B2-A') if means[m]['coverage']>=.88] or ['B0','B1','B2-A']
    comparators=['B3-A','B3-G'] if selected=='W4' else ['B3-A']
    matched=min(comparators,key=lambda m:(means[m]['winkler_kw'],m))
    candidate=means[selected]
    flags={'C1':.88<=candidate['coverage']<=.92,
           'C2':cb.finite_leq(candidate['winkler_kw'],.95*min(means[m]['winkler_kw'] for m in eligible)),
           'C3':cb.finite_leq(candidate['winkler_kw'],.97*means[matched]['winkler_kw']) and cis[matched]['ci95_low_kw']>0}
    return {**flags,'pass_all':bool(all(flags.values())),'matched_comparator':matched,'C2_comparators':eligible,'cis':cis}


class Data:
    def __init__(self,root,year,reverse):
        import pandas as pd
        from scripts import train_penmanshiel_p2_v35 as p2
        from scripts import train_n1ph as n1
        self.root,self.year,self.reverse=root,year,reverse
        self.p2,self.u,self.n1=p2,p2.u,n1
        u=p2.u; self.recipe=u.read_json(p2.PROTOCOL); self.old=u.read_json(ROOT/'configs/n1ph_protocol.json')
        self.identities={}; self.direction_cache={}
        prepared=root/self.old['p3_run']/'prepared'
        expected=u.read_json(ROOT/'output/penmanshiel_hourly/p3_v35/r2/prepared/prepared_complete.json')
        for name,digest in expected['artifacts'].items(): n1.checked(prepared/name,digest,self.identities)
        py=year-1 if reverse else year; ky=year if reverse else year-1
        pf=pd.concat([c[c.label_year<=py] for c in pd.read_csv(prepared/'penmanshiel.csv',chunksize=50000)],ignore_index=True)
        pup=pd.read_csv(prepared/'upstream.csv',index_col='timestamp_hour_end_utc'); pup.index=pd.to_datetime(pup.index)
        pup=pup[pup.index<pd.Timestamp(f'{py+1}-01-01T01:00')]
        pm=u.read_json(prepared/'preparation.json'); self.registry=u.read_json(prepared/'input_hashes.json')
        if ky<=2023:
            ref=self.recipe['data']['v31_registered_preparation']; csv=root/ref['csv']
            n1.checked(csv,ref['csv_sha256'],self.identities)
            mpath=csv.parent.parent/'metadata/preparation.json'; n1.checked(mpath,ref['preparation_json_sha256'],self.identities)
            km=u.read_json(mpath)
        else:
            csv=root/self.recipe['data']['v32_preparation']['csv']; mpath=csv.parent.parent/'metadata/preparation.json'
            km=u.read_json(mpath); u.verify_preparation(km,self.recipe,sha(csv))
            self.identities[str(csv)]=sha(csv); self.identities[str(mpath)]=sha(mpath)
        kf=pd.concat([c[c.label_year<=ky] for c in pd.read_csv(csv,chunksize=50000)],ignore_index=True)
        spec=self.recipe['data']['upstream']; manifest=root/spec['download_manifest']
        if u.text_sha256_lf(manifest)!=spec['download_manifest_sha256_lf']: raise ValueError('MIDAS manifest changed')
        self.identities[str(manifest)]=sha(manifest); self.hashes=u.manifest_hashes(u.read_json(manifest))
        stations=spec['stations']; ids=[int(s['src_id']) for s in stations]; files=[]
        for item in spec['development_roots']:
            years=[y for y in item['years'] if y<=ky]
            if years: files+=u.station_files(root/item['root'],ids,years,self.hashes,True)[0]
        if ky>=2024: files+=u.station_files(root/spec['test_root'],ids,[2024],self.hashes,True)[0]
        for e in files: n1.checked(e['path'],e['sha256'],self.identities)
        obs=u.load_station_observations(files,ids)
        kup=u.hourly_upstream(obs,stations,pd.date_range('2016-05-01',f'{ky+1}-01-01',freq='h'),1)
        self.frames={'K':(kf,kup,km,'2016-05-01',ky),'P':(pf,pup,pm,pm['start'],py)}
        self.arm='P-upstream' if reverse else 'K-upstream'; self.source='P' if reverse else 'K'; self.target='K' if reverse else 'P'
        self.records=[u.read_json(ROOT/f'output/penmanshiel_hourly/p2_v35/fits/{self.arm}/s{s}/trained.json') for s in p2.SEEDS]
        for s,r in zip(p2.SEEDS,self.records):
            folder=root/self.old['p2_run']/f'fits/{self.arm}/s{s}'
            p2.trainer.verify_trained(folder,r['contract'])
            if u.canonical_sha256(u.read_json(folder/'trained.json'))!=u.canonical_sha256(r): raise ValueError('P2 identity changed')
            for name,digest in r['artifacts'].items(): n1.checked(folder/name,digest,self.identities)
        state=self.records[0]['contract']['statistics']
        if any(r['contract']['statistics']!=state for r in self.records): raise ValueError('Seed statistics differ')
        self.series={}
        for farm,(frame,up,meta,start,maxyear) in self.frames.items():
            splits={'train':[2016,2017,2018,2019],'validation':[2020],'calibration':[year-1]}
            if farm==self.target: splits['test']=[year]
            series=u.KelmarshUpstreamSeries(frame,up,splits=splits,start=start,end=f'{maxyear+1}-01-01T01:00:00Z',preparation=meta)
            self.series[farm]=p2.adapt_statistics(series,up,state=state)
        for path in [p2.PROTOCOL,ROOT/'configs/n1ph_protocol.json',ROOT/'docs/STAGE_PLAN.md',ROOT/p2.farm.D2_SUBDIR/'part_A.json']:
            self.identities[str(path)]=sha(path)

    def direction(self,farm,labels):
        import pandas as pd
        if farm not in self.direction_cache:
            u,p2=self.u,self.p2
            stations=self.recipe['data']['upstream']['stations'] if farm=='K' else [s for s in u.read_json(ROOT/p2.farm.D2_SUBDIR/'part_A.json')['midas']['stations'] if s['src_id'] in p2.farm.STATIONS]
            sid=int(min(stations,key=lambda s:(s['distance_km'],s['src_id']))['src_id']); files=[]
            maxyear=self.frames[farm][-1]
            for y in range(self.year-2,maxyear+1):
                if farm=='P':
                    matches=[(p,e) for p,e in self.registry.items() if Path(p).parent.name==str(sid) and Path(p).name.endswith(f'_qcv-1_{y}.csv')]
                    if len(matches)!=1: raise ValueError('Ambiguous station file')
                    name,e=matches[0]; path=self.root/'datasets'/name.split('/datasets/',1)[1]
                    self.n1.checked(path,e['sha256'],self.identities); files.append({'path':path,'year':y})
                else:
                    spec=self.recipe['data']['upstream']
                    roots=[x['root'] for x in spec['development_roots'] if y in x['years']]
                    if y==2024: roots.append(spec['test_root'])
                    for root in roots: files+=u.station_files(self.root/root,[sid],[y],self.hashes,True)[0]
            for e in files: self.identities[str(e['path'])]=sha(e['path'])
            self.direction_cache[farm]=u.load_station_observations(files,[sid])[sid]
        return self.direction_cache[farm].reindex(pd.DatetimeIndex(labels)-pd.Timedelta(hours=1)).wind_direction.to_numpy()

    def hourly(self,farm):
        import pandas as pd
        times=pd.date_range(f'{self.year-1}-01-01',f'{self.year-1}-12-31T23:00',freq='h')
        raw=self.frames[farm][1].reindex(times).to_numpy(float)
        return c0.direction_features(raw,self.direction(farm,times)),times.to_numpy()

    def windows(self,farm,month=False,test=False):
        series=self.series[farm]
        if test: return series.windows('test')
        lo=np.datetime64(f'{self.year-1}-12-01' if month else f'{self.year-1}-01-01')
        hi=np.datetime64(f'{self.year-1}-12-31T23:00')
        s=series.starts; times=series.dates[s[:,None]+24+np.arange(4)]
        return self.p2.MultiYearWindows(series,s[(times[:,0]>=lo)&(times[:,-1]<=hi)],'CW1','all')

    def pack(self,w,farm):
        import pandas as pd
        issue=w.target_times()[:,0]-np.timedelta64(1,'h')
        raw=self.frames[farm][1].reindex(pd.DatetimeIndex(issue)).to_numpy(float)
        return dict(truth=w.truth_and_anchor()[0],times=w.target_times(),issue=issue,turbines=w.turbines[w.window_starts],
                    features=c0.direction_features(raw,self.direction(farm,issue)),starts=w.window_starts)


def summarize(out):
    rows=[json.loads((out/f'seed_{s}.json').read_text()) for s in (2024,2025,2026)]
    keys=('coverage','width_kw','winkler_kw','cap_fraction'); methods=rows[0]['methods']
    result={'mean_90':{m:{k:float(np.mean([r['methods'][m]['pooled'][k] for r in rows])) for k in keys} for m in methods},
            'horizons':{m:[{k:float(np.mean([r['methods'][m]['horizons'][h][k] for r in rows])) for k in keys} for h in range(4)] for m in methods},
            'artifacts':{p.name:sha(p) for p in out.glob('*.npz')}}
    save(out/'result.json',result); return result


def phase(root,out,year,reverse,selected=None,choices=None):
    import torch
    out.mkdir(parents=True,exist_ok=False)
    data=Data(root,year,reverse); p2,u=data.p2,data.u
    methods=list(BASE)
    if selected is None: methods+=['M-A','W1','W2','W4','B3-G']
    elif selected=='W3': methods+=['M-A']
    else: methods+=[selected]+(['B3-G'] if selected=='W4' else [])
    windows=[data.windows(data.source),data.windows(data.target,month=True),data.windows(data.target),data.windows(data.target,test=True)]
    farms=[data.source,data.target,data.target,data.target]
    packs=[data.pack(w,f) for w,f in zip(windows,farms)]
    for p in packs[:3]: cb.causal_check(p['times'],packs[-1]['issue'],year)
    hs,ts=data.hourly(data.source); ht,tt=data.hourly(data.target)
    ratio,prior=c0.ratio_fit(hs,ht)
    weights=(c0.density(ratio,prior,packs[0]['features']),c0.density(ratio,prior,packs[-1]['features']))
    weather=weather_fit(hs[:,:11],ht[:,:11],ts,tt,year) if 'W4' in methods else None
    groups=weather.predict(packs[-1]['features'][:,:11]) if weather is not None else None
    audit={'year':year,'direction':'P-to-K' if reverse else 'K-to-P','point_arm':data.arm,'point_training':False,
           'calibration_cutoff':f'{year-1}-12-31T23:00','windows':dict(zip(('source','target_month','target_year','evaluation'),map(len,windows))),
           'input_hashes':data.identities,'source_code':p2.source_code(),'methods':methods,
           'kmeans':None if weather is None else {'fill':weather[0].statistics_.tolist(),'mean':weather[1].mean_.tolist(),'scale':weather[1].scale_.tolist(),'centers':weather[2].cluster_centers_.tolist()},
           'W2':{'max_iter':200,'learning_rate':.05,'max_leaf_nodes':15,'min_samples_leaf':100,'l2_regularization':1.,'random_state':2024,'early_stopping':False}}
    save(out/'calibration_freeze.json',audit)
    save(out/'evaluation_started.json',{'utc':p2.now(),'year':year})
    for seed in p2.SEEDS:
        model,_=p2.trainer.build_model(p2.configuration(data.arm,data.recipe))
        state=torch.load(root/data.old['p2_run']/f'fits/{data.arm}/s{seed}/best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'],strict=True)
        for name,w,p in zip(('source','target_month','target_year','evaluation'),windows,packs):
            p['prediction']=p2.trainer.v26run.predict(model,p2.view(w,data.arm),seed,torch.device('cpu'),512)
            finite(p['prediction'],'prediction'); np.savez_compressed(out/f'{name}_s{seed}.npz',**p)
        strong=hgb_weights(packs[0],packs[2],packs[3],year) if 'W2' in methods else None
        power_edges=np.quantile(np.r_[packs[0]['prediction'],packs[2]['prediction']],[.25,.5,.75],axis=0).T
        radii,caps=intervals(packs[0],packs[1],packs[3],weights,year,methods,groups,power_edges,strong)
        if selected=='W3':
            radii['W3'],caps['W3']=combination(radii,caps,choices)
            del radii['M-A']; del caps['M-A']
        np.savez_compressed(out/f'intervals_s{seed}.npz',**radii)
        np.savez_compressed(out/f'caps_s{seed}.npz',**caps)
        save(out/f'seed_{seed}.json',{'seed':seed,'methods':reports(radii,caps,np.abs(packs[3]['truth']-packs[3]['prediction'])),'power_edges':power_edges.tolist()})
        print(f'[CW1] {out.name} seed {seed} complete',flush=True)
    return summarize(out),audit


def add_combination(out,choices):
    for s in (2024,2025,2026):
        with np.load(out/f'intervals_s{s}.npz') as z: radii=dict(z)
        with np.load(out/f'caps_s{s}.npz') as z: caps=dict(z)
        with np.load(out/f'evaluation_s{s}.npz') as z: error=np.abs(z['truth']-z['prediction'])
        radii['W3'],caps['W3']=combination(radii,caps,choices)
        np.savez_compressed(out/f'intervals_s{s}.npz',**radii); np.savez_compressed(out/f'caps_s{s}.npz',**caps)
        report=json.loads((out/f'seed_{s}.json').read_text()); report['methods']=reports(radii,caps,error)
        save(out/f'seed_{s}.json',report)
    return summarize(out)


def judge(out,result,selected):
    comparisons=['B3-A','B3-G'] if selected=='W4' else ['B3-A']
    differences={m:[] for m in comparisons}; issue=None
    for s in (2024,2025,2026):
        with np.load(out/f'evaluation_s{s}.npz') as p, np.load(out/f'intervals_s{s}.npz') as z:
            if issue is None: issue=p['issue']
            else: np.testing.assert_array_equal(issue,p['issue'])
            error=np.abs(p['truth']-p['prediction'])
            losses={m:2*z[m]+20*np.maximum(error-z[m],0) for m in [selected]+comparisons}
            for m in comparisons: differences[m].append(losses[m]-losses[selected])
    cis={m:paired_ci(np.mean(differences[m],axis=0),issue,out/f'bootstrap_{m}.npz') for m in comparisons}
    result.update(verdict(result['mean_90'],selected,cis))
    result['artifacts']={p.name:sha(p) for p in out.glob('*.npz')}
    save(out/'result.json',result); return result


def run(root,out):
    initial=[]; audits=[]
    for y in (2022,2023):
        r,a=phase(root,out/f'selection_{y}',y,False); initial.append(r); audits.append(a)
    choices=choose(initial)
    results=[add_combination(out/f'selection_{y}',choices) for y in (2022,2023)]
    selected,gains=select_variant(results)
    freeze={'selected':selected,'W3_choices':choices,'gains_pct':gains,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'selection_results':{str(y):sha(out/f'selection_{y}/result.json') for y in (2022,2023)},
            'source_code':audits[-1]['source_code'],'input_hashes':{k:v for a in audits for k,v in a['input_hashes'].items()}}
    save(out/'selection_freeze.json',freeze)
    if selected is None:
        result={'status':'COMPLETED','selected':None,'decision':'STOP_AT_SELECTION','gains_pct':gains}
    else:
        for path,digest in freeze['input_hashes'].items():
            if sha(path)!=digest: raise ValueError('Input changed after selection')
        for path,digest in freeze['source_code'].items():
            if sha(ROOT/path)!=digest: raise ValueError('Code changed after selection')
        with (out/'judgment_started.json').open('x') as f: json.dump({'selection_sha256':sha(out/'selection_freeze.json'),'utc':dt.datetime.now(dt.timezone.utc).isoformat()},f)
        judged={}
        for name,reverse in [('J1',False),('J2',True)]:
            r,_=phase(root,out/name,2024,reverse,selected,choices)
            judged[name]=judge(out/name,r,selected)
        result={'status':'COMPLETED','selected':selected,'W3_choices':choices,
                'decision':'CONTINUE' if all(r['pass_all'] for r in judged.values()) else 'STOP_CW1',
                'judgments':judged,'selection_sha256':sha(out/'selection_freeze.json')}
    save(out/'result.json',result); return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data-root',type=Path); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true'); args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    save(args.out/'launch.json',{'argv':sys.argv,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'script_sha256':sha(__file__),
                               'plan_sha256':sha(ROOT/'docs/STAGE_PLAN.md'),'frozen_dependencies':{n:sha(ROOT/'scripts'/n) for n in ['pilot_conformal_c0.py','pilot_conformal_c0b.py']}})
    try:
        if args.smoke:
            from tests.test_pilot_conformal_cw1 import smoke
            result=smoke(args.out)
        else:
            import torch
            torch.set_num_threads(2); result=run(args.data_root,args.out)
        save(args.out/'completed.json',{'status':result['status'],'seconds':time.monotonic()-start,'utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    except Exception as exc:
        save(args.out/'failed.json',{'type':type(exc).__name__,'error':str(exc),'seconds':time.monotonic()-start}); raise


if __name__=='__main__': main()
