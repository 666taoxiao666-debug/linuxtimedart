"""L0 H0: preregistered descriptive residual probe; no deep-model training."""
from pathlib import Path
import sys,json,argparse,warnings
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from sklearn.ensemble import HistGradientBoostingRegressor
from scripts import train_e1_baselines as e
from scripts import screen_kelmarsh_upstream_observations as screen
u,save=e.u,e.save

def verdict(directions,t):
    farms={f:[d for d in directions if d['farm']==f] for f in ('Kelmarsh','Penmanshiel')}
    if any(len(v)!=2 for v in farms.values()):raise ValueError('Need two directions per farm')
    means={f:{k:float(np.mean([d[k] for d in ds])) for k in ('H_MAE_pct','H_RMSE_pct')} for f,ds in farms.items()}
    if any(v['H_MAE_pct']<t['stop_mae'] for v in means.values()):decision='STOP'
    elif all(v['H_MAE_pct']>=t['go_mae'] and v['H_RMSE_pct']>=t['go_rmse'] for v in means.values()) and all(d['H_MAE_pct']>0 for d in directions):decision='GO'
    else:decision='MARGINAL'
    return {'decision':decision,'farm_means':means,'interpretation':'fixed probe only; not an information-theoretic upper bound'}

def year_windows(series,year):
    starts=series.starts;dates=series.dates
    mask=(dates[starts]>=np.datetime64(f'{year}-01-01'))&(dates[starts+27]<np.datetime64(f'{year+1}-01-01'))
    return e.p2.MultiYearWindows(series,starts[mask],'probe','all')

def station_features(issue,obs,stations):
    issue=pd.DatetimeIndex(issue);t0=issue-pd.Timedelta(hours=1)
    def matrix(delay,name):return screen.station_matrix(obs,stations,t0-pd.Timedelta(hours=delay),name)
    ws,pres,direction=matrix(0,'wind_speed'),matrix(0,'msl_pressure'),matrix(0,'wind_direction')
    dist=np.array([s['distance_km'] for s in stations]);bearing=np.array([s['bearing_deg'] for s in stations]);nearest=int(np.argmin(dist))
    radians=np.radians(direction)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        regional=np.degrees(np.arctan2(np.nanmean(np.sin(radians),1),np.nanmean(np.cos(radians),1)))%360
    wind=np.where(np.isfinite(direction[:,nearest]),direction[:,nearest],regional)
    angle=(bearing[None,:]-wind[:,None]+180)%360-180
    parallel=dist*np.cos(np.radians(angle));cross=dist*np.sin(np.radians(angle))
    values=[ws,np.sin(radians),np.cos(radians),pres];names=['ws','direction_sin','direction_cos','pressure']
    for lag in (1,2,3,5):values.extend([matrix(lag,'wind_speed'),matrix(lag,'msl_pressure')]);names.extend([f'ws_lag{lag}',f'pressure_lag{lag}'])
    values.extend([parallel,cross,parallel/np.maximum(ws[:,nearest,None],2)]);names.extend(['parallel_km','cross_km','distance_over_speed'])
    b1=np.stack(values,-1).reshape(len(issue),-1)
    n1=[f's{s["src_id"]}_{n}' for s in stations for n in names]
    delta_ws=ws-matrix(3,'wind_speed');delta_p=pres-matrix(3,'msl_pressure');b2=[];n2=[]
    sectors=[(angle>=-45)&(angle<45),(angle>=-135)&(angle<-45),(angle>=45)&(angle<135),(angle>=135)|(angle<-135)]
    for lo,hi in [(20,50),(50,100),(100,150)]:
        ring=(dist>=lo)&((dist<=hi) if hi==150 else (dist<hi))
        for sn,sector in zip(['upwind','left','right','downwind'],sectors):
            mask=sector&ring[None,:]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                b2.extend([np.nanmean(np.where(mask,v,np.nan),1) for v in (ws,delta_ws,delta_p)])
            b2.append(np.sum(mask&np.isfinite(ws)&np.isfinite(pres),1))
            n2.extend([f'r{lo}_{hi}_{sn}_{n}' for n in ['ws','dws3','dp3','valid_count']])
    return b1,np.column_stack(b2),n1,n2

def pack(w,pred,obs,stations):
    truth=w.truth_and_anchor()[0];issue=w.target_times()[:,0]-np.timedelta64(1,'h');unique,inverse=np.unique(issue,return_inverse=True)
    b1,b2,n1,n2=station_features(unique,obs,stations)
    hand,names=screen.upstream_features(unique,obs,stations,1);past,_=screen.upstream_features(pd.DatetimeIndex(unique)-pd.Timedelta(hours=3),obs,stations,1)
    clock=pd.DatetimeIndex(issue);angle=2*np.pi*clock.hour.to_numpy()/24;season=2*np.pi*(clock.dayofyear.to_numpy()-1)/365.25
    last=w.window_starts+23;raw=w.series.raw
    a=np.column_stack([w.turbines[last],raw[last,-1],w.series.power_last10[last],np.sin(angle),np.cos(angle),np.sin(season),np.cos(season),hand[inverse],past[inverse],~np.isfinite(hand[inverse]).all(1)])
    an=['turbine','hour_power','last10_power','hour_sin','hour_cos','day_sin','day_cos']+['now_'+n for n in names]+['past3_'+n for n in names]+['upstream_missing']
    common=np.column_stack([pred.reshape(-1),np.tile(np.arange(1,5),len(w)),np.repeat(a,4,axis=0)])
    return {'truth':truth,'prediction':pred,'error':(truth-pred).reshape(-1),'a':common.astype(np.float32),'b1':np.column_stack([common,np.repeat(b1[inverse],4,axis=0)]).astype(np.float32),'b2':np.column_stack([common,np.repeat(b2[inverse],4,axis=0)]).astype(np.float32),'names':{'a':['prediction','h']+an,'b1':['prediction','h']+an+n1,'b2':['prediction','h']+an+n2},'issue':issue,'missing':~np.isfinite(hand[inverse]).all(1),'starts':w.window_starts,'times':w.target_times()}

def station_inputs(root,p,farm,recipe,identity):
    if farm=='Kelmarsh':
        spec=recipe['data']['upstream'];stations=spec['stations'];ids=[int(s['src_id']) for s in stations]
        hashes=u.manifest_hashes(u.read_json(root/spec['download_manifest']));files=[]
        for r in spec['development_roots']:files+=u.station_files(root/r['root'],ids,r['years'],hashes,True)[0]
        files+=u.station_files(root/spec['test_root'],ids,spec['test_years'],hashes,False)[0]
    else:
        part=u.read_json(ROOT/e.p2.farm.D2_SUBDIR/'part_A.json');stations=[s for s in part['midas']['stations'] if s['src_id'] in e.p2.farm.STATIONS];ids=[int(s['src_id']) for s in stations]
        registry=u.read_json(root/p['p3_run']/'prepared/input_hashes.json');files=[]
        for name,record in registry.items():
            path=Path(name)
            if path.parent.name in {str(i) for i in ids} and path.suffix=='.csv' and '_qcv-1_' in path.name:
                year=int(path.stem.split('_')[-1])
                if year in (2021,2022,2023):
                    path=root/'datasets'/name.split('/datasets/',1)[1];e.n1.checked(path,record['sha256'],identity);files.append({'path':path,'year':year})
    for f in files:identity[str(f['path'])]=u.file_sha256(f['path'])
    return stations,u.load_station_observations(files,ids)

def load_probe(root,p,identity):
    ep=u.read_json(ROOT/'configs/e1_baselines_protocol.json');sets,recipe=e.load_data(root,ep,identity)
    vdir=root/p['v31_run']
    for n,h in p['v31_hashes'].items():e.n1.checked(vdir/n,h,identity)
    audit=u.read_json(vdir/'data_manifest.json')['series_audit'];old=u.read_json(vdir/'protocol.json')
    spec=recipe['data']['v32_preparation'];csv=root/spec['csv'];meta=u.read_json(csv.parent.parent/'metadata/preparation.json')
    sk,ok=station_inputs(root,p,'Kelmarsh',recipe,identity)
    labels=pd.date_range('2016-04-30','2025-01-01',freq='h');kup=u.hourly_upstream(ok,sk,labels,1)
    kel=u.KelmarshUpstreamSeries(csv,kup,splits={'train':list(range(2016,2022)),'validation':[2022],'test':[2023,2024]},start=recipe['data']['start_utc'])
    state={'channels':audit['channels'],'mean':audit['scaler_mean'],'scale':audit['scaler_scale'],'fill':audit['upstream']['fill_values_training_mean']}
    kel=e.p2.adapt_statistics(kel,kup,state=state)
    data,_=e.p3.load_inputs(root,root/p['p3_run']/'prepared');pf,pup,pm,_,_,_=data
    pen=u.KelmarshUpstreamSeries(pf,pup,splits={'train':list(range(2016,2021)),'validation':[2021],'test':[2022,2023]},start=pm['start'],end='2024-01-01T01:00:00Z',preparation=pm)
    rec=p['p3_records']['2024'];state=rec['contract']['statistics'];pen=e.p2.adapt_statistics(pen,pup,state=state)
    sp,op=station_inputs(root,p,'Penmanshiel',recipe,identity)
    return {'Kelmarsh':(kel,sk,ok,vdir,old),'Penmanshiel':(pen,sp,op,root/p['p3_run']/'T22',recipe)}

def predictions(series,year,farm,folder,recipe,p,device,identity):
    w=year_windows(series,year);all_preds=[];first=p['years'][farm][0]
    for seed in p['seeds']:
        if farm=='Kelmarsh':
            fit=folder/f'runs/TSMixer_upstream_all/s{seed}';record=u.read_json(fit/'trained.json');cfg=record['contract']['config'];name=f'TSMixer_upstream_all_s{seed}.npz';cohortfile=folder/'evaluation/test_cohort.npz';key='window_starts'
        else:
            fit=folder/f'fits/P-upstream/s{seed}';record=p['p3_records'][str(seed)];cfg=e.p2.configuration('P-upstream',recipe);name=f'P-upstream_s{seed}.npz';cohortfile=folder/'evaluation/cohort.npz';key='starts'
            if u.read_json(fit/'trained.json')!=record:raise ValueError('P3 fit identity changed')
            for n,h in record['artifacts'].items():e.n1.checked(fit/n,h,identity)
        e.trainer.verify_trained(fit,record['contract'])
        if year==first:
            if farm=='Penmanshiel':
                registry=u.read_json(ROOT/'output/penmanshiel_hourly/p3_v35/r2/T22/evaluation_hashes.json')
                e.n1.checked(cohortfile,registry['cohort.npz'],identity);e.n1.checked(folder/'evaluation'/name,registry[name],identity)
            with np.load(cohortfile) as z:
                oldtimes=z['times'];oldtruth=z['truth']
                # Match by turbine and all target timestamps; flattened starts change when later years extend the series.
                if 'turbines' in z.files: turbines=z['turbines']
                elif 'turbine' in z.files: turbines=z['turbine']
                else: turbines=None
                oldstarts=z[key]
            # Original cohort is ordered by turbine/time. Filter its dates to noncrossing year windows.
            keep=(oldtimes[:,0]-np.timedelta64(24,'h')>=np.datetime64(f'{year}-01-01'))&(oldtimes[:,-1]<np.datetime64(f'{year+1}-01-01'))
            np.testing.assert_array_equal(w.target_times(),oldtimes[keep]);np.testing.assert_array_equal(w.truth_and_anchor()[0],oldtruth[keep])
            with np.load(folder/'evaluation'/name) as z: pred=z['prediction'][keep]
        else:
            model,_=e.trainer.build_model(cfg);ck=torch.load(fit/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True);model.to(device)
            pred=e.trainer.v26run.predict(model,w,seed,device,512)
        all_preds.append(pred)
    return w,np.mean(all_preds,axis=0)

def run(root,out,p):
    identity={};data=load_probe(root,p,identity);packs={}
    for farm,(series,stations,obs,folder,recipe) in data.items():
        packs[farm]={}
        for year in p['years'][farm]:
            w,pred=predictions(series,year,farm,folder,recipe,p,torch.device('cpu'),identity)
            packs[farm][year]=pack(w,pred,obs,stations)
        years=p['years'][farm];a,b=[packs[farm][y] for y in years]
        if a['times'].max()>=b['issue'].min()-np.timedelta64(23,'h'):raise ValueError('Probe windows overlap')
    save(out/'feature_freeze.json',{'protocol':p,'input_hashes':identity,'features':{f:next(iter(ds.values()))['names'] for f,ds in packs.items()},'windows':{f:{str(y):len(d['truth']) for y,d in ds.items()} for f,ds in packs.items()}})
    reports=[]
    for farm,ds in packs.items():
        for train_year,test_year in (p['years'][farm],p['years'][farm][::-1]):
            train,test=ds[train_year],ds[test_year];errs={};lead={};missing={}
            for loss,metric in [('absolute_error','MAE'),('squared_error','RMSE')]:
                base=np.mean(np.abs(test['error'])) if metric=='MAE' else np.sqrt(np.mean(test['error']**2))
                for feature in ('a','b1','b2'):
                    model=HistGradientBoostingRegressor(loss=loss,**p['gbm']);model.fit(train[feature],train['error'])
                    correction=model.predict(test[feature]);error=(test['error']-correction).reshape(-1,4)
                    score=lambda a:float(np.mean(np.abs(a)) if metric=='MAE' else np.sqrt(np.mean(a*a)))
                    errs[feature+'_'+metric]=score(error);lead[feature+'_'+metric]=[score(error[:,h]) for h in range(4)]
                    missing[feature+'_'+metric]=score(error[test['missing']]) if test['missing'].any() else None
                    np.savez_compressed(out/f'{farm}_{train_year}_{test_year}_{feature}_{metric}.npz',correction=correction,truth=test['truth'],baseline=test['prediction'],times=test['times'])
                errs['baseline_'+metric]=float(base)
            row={'farm':farm,'train_year':train_year,'test_year':test_year,'errors':errs,'lead_errors':lead,'missing_errors':missing}
            for metric in ('MAE','RMSE'):row['H_'+metric+'_pct']=(errs['a_'+metric]-min(errs['b1_'+metric],errs['b2_'+metric]))/errs['baseline_'+metric]*100
            reports.append(row);save(out/f'{farm}_{train_year}_{test_year}.json',row)
    result={'directions':reports,**verdict(reports,p['thresholds']),'fits':24,'artifacts':{x.name:u.file_sha256(x) for x in out.glob('*.npz')}};save(out/'result.json',result)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();p=u.read_json(ROOT/'configs/l0_headroom_protocol.json')
    if a.out.resolve()!=(a.data_root/p['output_run']).resolve():raise ValueError('Only registered output allowed')
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    save(a.out/'launch.json',{'utc':e.p2.now(),'argv':sys.argv,'protocol_sha256':u.file_sha256(ROOT/'configs/l0_headroom_protocol.json'),'source_sha256':u.file_sha256(__file__)})
    try:run(a.data_root,a.out,p);save(a.out/'completed.json',{'utc':e.p2.now()})
    except Exception as err:save(a.out/'failed.json',{'utc':e.p2.now(),'type':type(err).__name__,'error':str(err)});raise
if __name__=='__main__':main()
