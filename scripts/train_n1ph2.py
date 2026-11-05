"""One bounded PH optimization; source-only selection precedes target loading."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts import train_n1ph as n
import numpy as np
import pandas as pd
import torch

p2, p3, u, trainer, save, n0 = n.p2, n.p3, n.u, n.trainer, n.save, n.n0
PROTOCOL = ROOT/'configs/n1ph2_protocol.json'


def choose(records):
    if set(records)!= {'V0','V1','V2','V3'}: raise ValueError('Undeclared variants')
    scores = {}
    for variant in ('V0','V1','V2','V3'):
        rows = records[variant]
        if len(rows)!=3 or sorted(r['seed'] for r in rows)!=[2024,2025,2026]:
            raise ValueError('Selection seed cohort mismatch')
        for r in rows:
            if set(r)!= {'seed','farm','year','split','mae'} or (r['farm'],r['year'],r['split'])!=('Kelmarsh',2020,'validation'):
                raise ValueError('Selection accepts only source validation metrics, no target labels')
            if not np.isfinite(r['mae']) or r['mae']<0: raise ValueError('Invalid MAE')
        scores[variant] = float(np.mean([r['mae'] for r in rows]))
    minimum = min(scores.values())
    return {'selected':next(v for v in scores if scores[v]-minimum < .5), 'mean_mae':scores,'records':records}


def require_selection(out, digest):
    path = out/'variant_selection.json'
    if not path.is_file() or u.file_sha256(path)!=digest:
        raise ValueError('Target access requires unchanged source selection freeze')
    selected = u.read_json(path)
    expected = choose(selected['records'])
    if any(selected[k]!=expected[k] for k in expected): raise ValueError('Selection inconsistent')
    return selected['selected']


def features(raw, model, stats, channels):
    ph=n.ph_features(raw,model,stats)
    if channels==14: return ph
    if channels!=25: raise ValueError('Undeclared channel set')
    return pd.concat([raw.reindex(ph.index),ph],axis=1)


def series(template, extra, state=None):
    if len(extra.columns)==5:
        result=n.ph_series(template,extra,state)
        result.audit['fill']=result.audit['mean']
        return result
    result=copy.copy(template)
    raw=np.c_[template.base.raw[:,:-1],extra.reindex(pd.DatetimeIndex(template.dates)).to_numpy(float),template.base.raw[:,-1]]
    channels=list(template.base.channels[:-1])+list(extra.columns)+['power']
    if len(channels) not in (14,25) or channels.count('power')!=1 or len(set(channels))!=len(channels):
        raise ValueError('Channel schema mismatch')
    if state is None:
        fit=np.isin(template.label_year,template.splits['train'])
        fill=np.nanmean(raw[fit],axis=0)
    else:
        if state['channels']!=channels: raise ValueError('Source schema mismatch')
        fill=np.asarray(state['fill'])
    # Only raw H may be missing; never impute a label or any SCADA/PH channel.
    missing=~np.isfinite(raw)
    allowed=np.zeros(raw.shape[1],bool)
    if len(channels)==25: allowed[8:19]=True
    # Existing invalid SCADA/label rows remain missing; original window masks exclude them.
    raw=np.where(missing & allowed[None,:],fill,raw)
    if state is None:
        mean,scale=np.nanmean(raw[fit],axis=0),np.nanstd(raw[fit],axis=0)
        scale=np.where(scale==0,1.,scale)
    else: mean,scale=np.asarray(state['mean']),np.asarray(state['scale'])
    if not np.isfinite([mean,scale]).all() or not (scale>0).all():
        raise ValueError('Invalid fit normalization')
    result.raw,result.channels=raw,channels
    result.scaler_mean,result.scaler_scale=mean,scale
    result.values=((raw-mean)/scale).astype(np.float32)
    result.audit={'channels':channels,'fill':fill.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),
                  'origin':'source_only' if state is not None else 'farm_fit_only'}
    np.testing.assert_array_equal(raw[:,-1],template.base.raw[:,-1])
    return result


def config(arm, recipe, channels):
    return {'arm':arm,'family':'TSMixerBasicSmall','channels':channels,
            'args':copy.deepcopy(recipe['families']['TSMixerBasicSmall']['args']),
            'expected_parameters':{14:5872,25:9414}[channels]}


def train_mix(arrays, ids, device, destination):
    """N0 PH loop, with only the training loss changed; validation stays MSE."""
    torch.manual_seed(2024); rng=np.random.default_rng(2024)
    model=torch.nn.Sequential(torch.nn.Linear(11,32),torch.nn.ReLU(),torch.nn.Linear(32,32),torch.nn.ReLU(),torch.nn.Linear(32,4)).to(device)
    stats=n0.scaler(np.concatenate([arrays[s]['H'][arrays[s]['year']<=2019] for s in ids]))
    prepared={s:n0.transform(arrays[s]['H'],stats) for s in ids}
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    begin=time.monotonic(); history=[]; best=float('inf'); stale=0
    for epoch in range(1,21):
        model.train()
        for sid in rng.permutation(ids):
            a=arrays[int(sid)]; indices=rng.permutation(np.flatnonzero(a['year']<=2019))
            for offset in range(0,len(indices),n0.BATCH):
                idx=indices[offset:offset+n0.BATCH]
                pred=model(torch.from_numpy(prepared[int(sid)][idx]).to(device))
                error=pred-torch.from_numpy(a['y'][idx]).to(device)
                loss=.8*error.abs().mean()+.2*error.square().mean()
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                opt.zero_grad(); loss.backward(); opt.step()
                if time.monotonic()-begin>=900: break
            if time.monotonic()-begin>=900: break
        model.eval(); error=0.; count=0
        with torch.no_grad():
            for sid in ids:
                a=arrays[sid]; indices=np.flatnonzero(a['year']==2020)
                for offset in range(0,len(indices),n0.BATCH):
                    idx=indices[offset:offset+n0.BATCH]
                    pred=model(torch.from_numpy(prepared[sid][idx]).to(device))
                    error+=float(((pred-torch.from_numpy(a['y'][idx]).to(device))**2).sum()); count+=len(idx)*4
        value=error/count; history.append({'epoch':epoch,'validation_mse':value})
        if value<best: best=value; chosen=copy.deepcopy(model.state_dict()); best_epoch=epoch; stale=0
        else: stale+=1
        if stale>=4 or time.monotonic()-begin>=900: break
    model.load_state_dict(chosen); model.eval()
    destination.mkdir(parents=True,exist_ok=True)
    torch.save({'state':chosen,'stats':stats},destination/'selected.pt')
    save(destination/'trained.json',{'kind':'PH-mix','train_stations':ids,'parameters':1572,
         'best_epoch':best_epoch,'history':history,'elapsed':time.monotonic()-begin,
         'checkpoint_sha256':u.file_sha256(destination/'selected.pt'),
         'fit_years':[2016,2017,2018,2019],'selection_year':2020,'seed':2024})


def station_data(data_root, identities, training_only):
    # Fixed N0 registration: no new station discovery or split selection.
    inv=u.read_json(ROOT/'output/station_operator_n0/inventory.json')
    cohorts=u.read_json(ROOT/'output/station_operator_n0/cohorts.json')
    train=inv['split']['train']
    if len(train)!=28: raise ValueError('N0 split identity mismatch')
    ids=train if training_only else [s['src_id'] for s in inv['stations']]
    entries=[]
    for e in inv['files']:
        if e['src_id'] not in ids: continue
        path=data_root/'datasets'/e['path'].split('/datasets/',1)[1]
        n0.permitted_file(path,e['year']); n.checked(path,e['sha256'],identities)
        entries.append({**e,'path':path})
    obs=n.up.load_station_observations(entries,ids)
    arrays={}
    selected=train if training_only else inv['split']['kel_holdout']+inv['split']['pen_holdout']
    for q in inv['stations']:
        if q['src_id'] not in selected: continue
        a=n0.station_arrays(q,obs,inv['stations'],set(train) if training_only else set(ids),len(ids))
        registered=cohorts[str(q['src_id'])]
        counts={str(y):int((a['year']==y).sum()) for y in n0.YEARS}
        if counts!=registered['windows_by_year'] or a['candidates']!=registered['candidates']:
            raise ValueError('N0 station cohort or neighbor identity changed')
        # PH does not consume token tensors. Discard them immediately, unchanged H/L/y.
        del a['z'],a['mask'],a['undefined']
        arrays[q['src_id']]=a
    return inv, arrays


def source_data(data_root, recipe, identities):
    ref=recipe['data']['v31_registered_preparation']
    csv=n.checked(data_root/ref['csv'],ref['csv_sha256'],identities)
    meta=n.checked(csv.parent.parent/'metadata/preparation.json',ref['preparation_json_sha256'],identities)
    frame=pd.concat([c[c.label_year.between(2016,2020)] for c in pd.read_csv(csv,chunksize=50000)],ignore_index=True)
    spec=recipe['data']['upstream']; manifest=data_root/spec['download_manifest']
    if u.text_sha256_lf(manifest)!=spec['download_manifest_sha256_lf']: raise ValueError('Manifest identity')
    ids=[int(s['src_id']) for s in spec['stations']]
    entries=u.station_files(data_root/spec['development_roots'][0]['root'],ids,range(2016,2021),u.manifest_hashes(u.read_json(manifest)),required=True)[0]
    for e in entries: n.checked(e['path'],e['sha256'],identities)
    raw=u.hourly_upstream(u.load_station_observations(entries,ids),spec['stations'],pd.date_range('2016-05-01','2021-01-01',freq='h'),1)
    template=u.KelmarshUpstreamSeries(frame,raw,splits={'train':[2016,2017,2018,2019],'validation':[2020]},start='2016-05-01',end='2021-01-01T01:00:00Z',preparation=u.read_json(meta))
    return template,raw


def previous_fit(data_root, protocol, seed, identities):
    relative=Path('T21/fits/K-PH')/f's{seed}'
    expected=u.read_json(ROOT/'output/n1ph/r1'/relative/'trained.json')
    folder=data_root/protocol['previous_run']/relative
    if u.canonical_sha256(u.read_json(folder/'trained.json'))!=u.canonical_sha256(expected): raise ValueError('V0 registration changed')
    trainer.verify_trained(folder,expected['contract'])
    for name,digest in expected['artifacts'].items(): n.checked(folder/name,digest,identities)
    return folder,expected


def station_description(data_root,out,mix,identities):
    inv,arrays=station_data(data_root,identities,False)
    old=u.read_json(ROOT/'output/station_operator_n0/result.json')
    matrices={s:np.c_[a['L'],n0.outputs(*mix,a,'PH','cpu')] for s,a in arrays.items()}
    models={s:n0.ridge_fit(matrices[s],a['y'],a['year']) for s,a in arrays.items()}
    result={'role':'descriptive_only','per_station':{},'pairs':[]}
    for s,a in arrays.items():
        test=a['year']==2021; m=models[s]
        pred=n0.ridge_predict(m,matrices[s][test],m['stats'])
        dest=out/'station_description'/f'{s}.npz'; dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,prediction=pred,truth=a['y'][test],times=a['time'][test],coef=m['coef'])
        result['per_station'][str(s)]={'mix_mae':float(np.abs(pred-a['y'][test]).mean()),'mse_mae':old['per_station'][str(s)]['PH']['mae'],'array_sha256':u.file_sha256(dest)}
    for r in old['pairs']:
        s,q=r['source'],r['target']; a=arrays[q]; test=a['year']==2021
        pred=n0.ridge_predict(models[s],matrices[q][test],models[q]['stats'])
        result['pairs'].append({'source':s,'target':q,'mix_mae':float(np.abs(pred-a['y'][test]).mean()),'mse_mae':r['mae']['PH']})
    save(out/'station_description/result.json',result)


def run(data_root,out,protocol,device):
    began=time.monotonic()
    def timeout(*_): raise TimeoutError('N1-PH2 5400s budget exceeded; no restart')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(protocol['budget_seconds'])
    parent=ROOT/protocol['parent']
    if u.text_sha256_lf(parent)!=protocol['parent_sha256_lf']: raise ValueError('Parent protocol changed')
    base=u.read_json(parent); recipe=u.read_json(ROOT/base['recipe'])
    if u.text_sha256_lf(ROOT/base['recipe'])!=base['recipe_sha256_lf']: raise ValueError('Recipe changed')
    before=p2.source_code(); identities={}; records=[]; fit_paths={}; target_views={}
    identity={'protocol_sha256_lf':u.text_sha256_lf(PROTOCOL),'source_code':before}
    for path in (PROTOCOL,parent,ROOT/base['recipe'],ROOT/'tests/test_train_n1ph2.py',
                 ROOT/'output/station_operator_n0/inventory.json',ROOT/'output/station_operator_n0/cohorts.json',
                 ROOT/'output/station_operator_n0/result.json'):
        n.checked(path,u.file_sha256(path),identities)
    mse=n.load_ph(data_root/base['ph_run']/'models/PH/selected.pt',base['ph_sha256'],identities)
    inv,arrays=station_data(data_root,identities,True)
    train_mix(arrays,inv['split']['train'],device,out/'PH-mix'); del arrays
    mix_sha=u.file_sha256(out/'PH-mix/selected.pt')
    mix=n.load_ph(out/'PH-mix/selected.pt',mix_sha,identities)
    save(out/'ph_freeze.json',{'sha256':mix_sha,'trained_sha256':u.file_sha256(out/'PH-mix/trained.json'),'utc':p2.now()})
    operators={'mse':mse,'mix':mix}
    kt,ku=source_data(data_root,recipe,identities)
    kseries={v:series(kt,features(ku,*operators[s['loss']],s['channels'])) for v,s in protocol['variants'].items()}
    scores={v:[] for v in protocol['variants']}
    def fit(fid,v,arm,ds,fit,val,seed,warm=None):
        folder=out/fid/v/'fits'/arm/f's{seed}'; channels=len(ds.channels)
        contract={'stage':'N1-PH2','fold':fid,'variant':v,'arm':arm,'seed':seed,'identity':identity,
                  'statistics':ds.audit,'input_sha256':u.canonical_sha256(identities),
                  'warm_state_sha256':None if warm is None else trainer.v26run.state_digest(warm),
                  'fit_starts_sha256':hashlib.sha256(fit.window_starts.tobytes()).hexdigest(),
                  'validation_starts_sha256':hashlib.sha256(val.window_starts.tobytes()).hexdigest()}
        print(f'[FIT] {fid}/{v}/{arm}/s{seed} fit={len(fit)} val={len(val)}',flush=True)
        record=p2.fit_one(config(arm,recipe,channels),recipe,trainer.ChannelView(fit,list(range(channels))),trainer.ChannelView(val,list(range(channels))),seed,device,folder,contract,warm)
        records.append({'directory':str(folder.relative_to(out)),'record':record,'trained_sha256':u.file_sha256(folder/'trained.json')})
        fit_paths[(fid,v,arm,seed)]=folder
        return record
    for v,ds in kseries.items():
        fitw,val=ds.windows('train'),ds.windows('validation')
        for seed in base['seeds']:
            if v=='V0':
                folder,record=previous_fit(data_root,protocol,seed,identities)
                # Source validation cohorts and normalization must match the reused fit.
                if record['contract']['validation_starts_sha256']!=hashlib.sha256(val.window_starts.tobytes()).hexdigest(): raise ValueError('V0 source cohort mismatch')
                if record['contract']['fit_starts_sha256']!=hashlib.sha256(fitw.window_starts.tobytes()).hexdigest(): raise ValueError('V0 fit cohort mismatch')
                for key in ('mean','scale','channels'):
                    np.testing.assert_array_equal(ds.audit[key],record['contract']['statistics'][key])
                fit_paths[('T21',v,'K-PH',seed)]=folder
            else: record=fit('T21',v,'K-PH',ds,fitw,val,seed)
            scores[v].append({'seed':seed,'farm':'Kelmarsh','year':2020,'split':'validation','mae':record['validation_best']['mae']})
    selection=choose(scores); selection.update(utc=p2.now(),protocol_sha256_lf=identity['protocol_sha256_lf'])
    save(out/'variant_selection.json',selection); selection_sha=u.file_sha256(out/'variant_selection.json')
    selected=require_selection(out,selection_sha)
    print('[SELECTED]',selected,json.dumps(selection['mean_mae']),flush=True)
    # No Penmanshiel farm labels or reference arrays are loaded before this gate.
    pen,pup,_,p2ids=p2.load_data(data_root,recipe); identities.update(p2ids)
    folds=base['folds'] if selected!='V0' else base['folds'][:1]
    refs={f['id']:n.reference(data_root,f,base,identities) for f in folds}
    old_hashes=u.read_json(ROOT/'output/n1ph/r1/T21/evaluation_hashes.json')
    for seed in base['seeds']:
        name=f'K-PH_s{seed}.npz'
        n.checked(data_root/protocol['previous_run']/'T21/evaluation'/name,old_hashes[name],identities)
    later=n.other_inputs(data_root,base,recipe,identities) if selected!='V0' else None
    for fold in folds:
        fid=fold['id']; split='development_eval' if fid=='T21' else 'test'
        if fid=='T21': pt,pu=pen,pup
        else:
            pf,pu,meta,kf,ku2,kmeta=later
            splits={'train':fold['fit_years'],'validation':[fold['validation_year']],'test':[fold['test_year']]}
            pt=u.KelmarshUpstreamSeries(pf,pu,splits=splits,start=meta['start'],end=f"{fold['test_year']+1}-01-01T01:00:00Z",preparation=meta)
            kt2=u.KelmarshUpstreamSeries(kf,ku2,splits={k:v for k,v in splits.items() if k!='test'},start='2016-05-01',end=f"{fold['test_year']}-01-01T01:00:00Z",preparation=kmeta)
        for v in (list(protocol['variants']) if fid=='T21' else [selected]):
            spec=protocol['variants'][v]; op=operators[spec['loss']]
            ks=kseries[v] if fid=='T21' else series(kt2,features(ku2,*op,spec['channels']))
            extra=features(pu,*op,spec['channels']); transfer=series(pt,extra,ks.audit)
            n.assert_cohort(transfer.windows(split),refs[fid][1],pt.normal_window[pt.split_of==split])
            views={'K-PH':transfer,'FT-1-PH':transfer,'FT-3-PH':transfer}
            if v==selected and selected!='V0': views['P-PH']=series(pt,extra)
            target_views[(fid,v)]=views
            save(out/fid/v/'data_audit.json',{'K':ks.audit,'transfer':transfer.audit,'P':views['P-PH'].audit if 'P-PH' in views else None})
            if v!=selected or selected=='V0': continue
            for arm in base['arms']:
                if fid=='T21' and arm=='K-PH': continue
                ds=ks if arm=='K-PH' else views[arm]
                if arm.startswith('FT'): fw,vw=p3.fewshot_windows(ds,fold['fewshot_year'],int(arm.split('-')[1]))
                else: fw,vw=ds.windows('train'),ds.windows('validation')
                for seed in base['seeds']:
                    warm=torch.load(fit_paths[(fid,v,'K-PH',seed)]/'best.pt',map_location='cpu',weights_only=False)['model'] if arm.startswith('FT') else None
                    fit(fid,v,arm,ds,fw,vw,seed,warm)
    require_selection(out,selection_sha)
    if p2.source_code()!=before or u.text_sha256_lf(PROTOCOL)!=identity['protocol_sha256_lf']: raise ValueError('Source/protocol changed')
    for path,digest in identities.items(): n.checked(path,digest['sha256'] if isinstance(digest,dict) else digest,{})
    expected=9 if selected=='V0' else 42
    freeze={'expected_fits':expected,'fits':records,'identity':identity,'variant_selection_sha256':selection_sha,'ph_sha256':mix_sha,'utc':p2.now()}
    save(out/'input_identity.json',identities); save(out/'selection_freeze.json',freeze)
    p3.begin_evaluation(out,freeze,expected)
    results={}; descriptive={}
    for fold in folds:
        fid=fold['id']; cohort=refs[fid][1]; split='development_eval' if fid=='T21' else 'test'
        predictions={}
        for arm in p2.ARMS:
            predictions[arm]=[]
            for seed in base['seeds']:
                with np.load(refs[fid][0]/f'evaluation/{arm}_s{seed}.npz',allow_pickle=False) as z: predictions[arm].append(z['prediction'])
        predictions.update(PersistenceHour=[cohort['persistence']],PersistenceLast10=[cohort['last10']])
        for v in (list(protocol['variants']) if fid=='T21' else [selected]):
            arms=base['arms'] if v==selected and selected!='V0' else ['K-PH']
            vp={}
            for arm in arms:
                vp[arm]=[]; windows=target_views[(fid,v)][arm].windows(split)
                for seed in base['seeds']:
                    if v=='V0':
                        path=data_root/protocol['previous_run']/fid/f'evaluation/{arm}_s{seed}.npz'
                        digest=u.read_json(ROOT/'output/n1ph/r1'/fid/'evaluation_hashes.json')[path.name]
                        n.checked(path,digest,identities)
                        with np.load(path,allow_pickle=False) as z:
                            np.testing.assert_array_equal(z['starts'],windows.window_starts); pred=z['prediction']
                    else:
                        folder=fit_paths[(fid,v,arm,seed)]; channels=protocol['variants'][v]['channels']
                        model,_=trainer.build_model(config(arm,recipe,channels))
                        model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['model'])
                        pred=trainer.v26run.predict(model.to(device),trainer.ChannelView(windows,list(range(channels))),seed,device,recipe['training']['eval_batch_size'])
                        path=out/fid/v/'evaluation'/f'{arm}_s{seed}.npz'; path.parent.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(path,prediction=pred,starts=windows.window_starts)
                    # Statistics are computed from persisted arrays, never a new forward pass.
                    with np.load(path,allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved['starts'],windows.window_starts)
                        np.testing.assert_array_equal(saved['prediction'],pred)
                        pred=saved['prediction']
                    if pred.shape!=cohort['truth'].shape or not np.isfinite(pred).all(): raise ValueError('Invalid prediction array')
                    vp[arm].append(pred)
            if fid=='T21': descriptive[v]=u.seed_mean([u.metrics(p,cohort['truth'],2050) for p in vp['K-PH']])
            if v==selected and selected!='V0': predictions.update(vp)
        if selected=='V0': results[fid]={**u.read_json(ROOT/'output/n1ph/r1/T21/result.json'),'reused_result':'output/n1ph/r1/T21/result.json'}
        else: results[fid]=n.report(predictions,cohort,base,fold['role'])
        save(out/fid/'result.json',results[fid])
    if selected=='V0':
        for fold in base['folds'][1:]:
            fid=fold['id']; path=ROOT/'output/n1ph/r1'/fid/'result.json'
            results[fid]={**u.read_json(path),'reused_result':str(path.relative_to(ROOT))}
            save(out/fid/'result.json',results[fid])
    station_description(data_root,out,mix,identities)
    require_selection(out,selection_sha)
    if p2.source_code()!=before: raise ValueError('Source changed')
    for path,digest in identities.items(): n.checked(path,digest['sha256'] if isinstance(digest,dict) else digest,{})
    p3.validate_freeze(out,freeze,expected)
    save(out/'input_identity.json',identities)
    save(out/'evaluation_hashes.json',{str(p.relative_to(out)):u.file_sha256(p) for p in out.rglob('*.npz') if 'evaluation' in p.parts or 'station_description' in p.parts})
    save(out/'result.json',{'selection':selection,'folds':results,'primary_zero_shot_descriptive':descriptive})
    save(out/'audit/completed.json',{'utc':p2.now(),'wind_fits':len(records),'ph_mix_fits':1,'elapsed':time.monotonic()-began,'source_unchanged':True})
    signal.alarm(0)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True); parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args(); protocol=u.read_json(PROTOCOL)
    args.out.mkdir(parents=True,exist_ok=True)
    lockpath=args.data_root/'outputs/logs/Penmanshiel/n1ph2.lock'
    with lockpath.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with lockpath.with_suffix('.launched.json').open('x') as f: json.dump({'out':str(args.out),'utc':p2.now()},f)
        torch.set_num_threads(2); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        save(args.out/'audit/launch.json',{'utc':p2.now(),'pid':os.getpid(),'command':sys.argv,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'protocol_sha256_lf':u.text_sha256_lf(PROTOCOL),'source_code':p2.source_code()})
        try: run(args.data_root,args.out,protocol,args.device)
        except BaseException as error:
            save(args.out/'audit/failed.json',{'utc':p2.now(),'type':type(error).__name__,'message':str(error),'test_started':(args.out/'test_evaluation_started.json').exists()})
            raise


if __name__=='__main__': main()
