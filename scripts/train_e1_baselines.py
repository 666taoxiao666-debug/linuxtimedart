"""E1-A: immutable recipe; tune only on Kelmarsh validation; freeze all then test."""
from __future__ import annotations
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
import types
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scripts import train_penmanshiel_p2_v35 as p2
from scripts import train_penmanshiel_p3_v35 as p3
from scripts import train_n1ph as n1
u,trainer,save=p2.u,p2.trainer,p2.save


class PatchAdapter(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        from models.PatchTST import Model
        self.model=Model(config)
    def forward(self,x): return self.model(x,None,None,None)


def build_model(cfg):
    args=types.SimpleNamespace(**cfg['args'],enc_in=cfg['channels'])
    if cfg['family']=='PatchTST': model=PatchAdapter(args)
    else: model=importlib.import_module('models.'+cfg['family']).Model(args)
    count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if 'expected_parameters' in cfg and count!=cfg['expected_parameters']: raise ValueError('Parameter count changed')
    return model,count


def configuration(protocol,family,channel,d):
    args=dict(protocol['fixed_args'],d_model=d,d_ff=4*d)
    cfg={'family':family,'arm':family+'_'+channel,'channels':{'power':1,'last10':9,'upstream':21}[channel],
         'channel':channel,'args':args}
    _,cfg['expected_parameters']=build_model(cfg)
    return cfg


def view(w,channel):
    return trainer.ChannelView(w,{'power':[20],'last10':list(range(8))+[20],'upstream':list(range(21))}[channel])


def fit(cfg,recipe,fit,selection,seed,device,out,contract):
    f=types.FunctionType(trainer.fit_config.__code__,{**trainer.fit_config.__globals__,'build_model':build_model},
                         trainer.fit_config.__name__,trainer.fit_config.__defaults__)
    f.__kwdefaults__=trainer.fit_config.__kwdefaults__.copy()
    return f(cfg,recipe,fit=view(fit,cfg['channel']),selection=view(selection,cfg['channel']),
             seed=seed,device=device,out=out,contract=contract)


def choose(records):
    return min(records,key=lambda x:(x['record']['validation_best']['mae'],x['config']['args']['d_model']))['config']


def load_data(root,protocol,identity):
    old=u.read_json(ROOT/'configs/n1ph_protocol.json'); recipe=u.read_json(p2.PROTOCOL)
    prepared=root/old['p3_run']/'prepared'
    expected=u.read_json(ROOT/'output/penmanshiel_hourly/p3_v35/r2/prepared/prepared_complete.json')
    for name,digest in expected['artifacts'].items(): n1.checked(prepared/name,digest,identity)
    data,ids=p3.load_inputs(root,prepared); identity.update(ids)
    sets={}
    for fold in protocol['folds'][1:]:
        sets[fold['id']]=p3.build_fold(data,fold)[0]
    import pandas as pd
    d=recipe['data']; csv=root/d['v32_preparation']['csv']; meta=u.read_json(csv.parent.parent/'metadata/preparation.json')
    u.verify_preparation(meta,recipe,u.file_sha256(csv))
    identity[str(csv)]=u.file_sha256(csv)
    spec=d['upstream']; manifest=root/spec['download_manifest']
    if u.text_sha256_lf(manifest)!=spec['download_manifest_sha256_lf']: raise ValueError('Manifest mismatch')
    hashes=u.manifest_hashes(u.read_json(manifest)); ids=[s['src_id'] for s in spec['stations']]; files=[]
    for r in spec['development_roots']: files+=u.station_files(root/r['root'],ids,r['years'],hashes,True)[0]
    files+=u.station_files(root/spec['test_root'],ids,spec['test_years'],hashes,False)[0]
    for e in files: identity[str(e['path'])]=e['sha256']
    obs=u.load_station_observations(files,ids)
    labels=pd.date_range('2016-04-30','2025-01-01',freq='h')
    upstream=u.hourly_upstream(obs,spec['stations'],labels,1)
    sets['Kelmarsh']=u.KelmarshUpstreamSeries(csv,upstream,splits=d['splits'],start=d['start_utc'])
    return sets,recipe


def reference(root,fid,windows):
    if fid=='Kelmarsh':
        local=ROOT/'output/kelmarsh_hourly/upstream_v32_review/results'
        registry=u.read_json(local/'post_test_manifest.json')
        # Locate by already registered cohort hash, not newest run name.
        matches=[]
        for p in (root/'outputs/logs/Kelmarsh').rglob('post_test_manifest.json'):
            if u.read_json(p).get('test_cohort_sha256')==registry['test_cohort_sha256']: matches.append(p.parent)
        if len(matches)!=1: raise ValueError('Cannot uniquely identify frozen v32 reference')
        folder=matches[0]/'evaluation'
        cohortfile=folder/'cohort.npz'
        if not cohortfile.exists(): cohortfile=folder/'test_cohort.npz'
        n1.checked(cohortfile,registry['test_cohort_sha256'],{})
        with np.load(cohortfile,allow_pickle=False) as z: cohort={k:z[k] for k in z.files}
        starts=cohort['window_starts']; preds=[]
        for seed in p2.SEEDS:
            name=f'TSMixer_upstream_all_s{seed}.npz'
            record=u.read_json(local/'evaluation'/name.replace('.npz','.json'))
            n1.checked(folder/name,record['prediction_sha256'],{})
            with np.load(folder/name,allow_pickle=False) as z: preds.append(z['prediction'])
    else:
        old=u.read_json(ROOT/'configs/n1ph_protocol.json'); folder=root/old['p3_run']/fid/'evaluation'
        hashes=u.read_json(ROOT/f'output/penmanshiel_hourly/p3_v35/r2/{fid}/evaluation_hashes.json')
        for name,digest in hashes.items(): n1.checked(folder/name,digest,{})
        with np.load(folder/'cohort.npz',allow_pickle=False) as z: cohort={k:z[k] for k in z.files}
        starts=cohort['starts']; preds=[]
        for seed in p2.SEEDS:
            with np.load(folder/f'P-upstream_s{seed}.npz',allow_pickle=False) as z: preds.append(z['prediction'])
    for actual,expected in ((windows.window_starts,starts),(windows.target_times(),cohort['times']),
                             (windows.truth_and_anchor()[0],cohort['truth'])):
        np.testing.assert_array_equal(actual,expected)
    return preds


def model_smoke(out,protocol):
    series,_,_=p2.synthetic_data()
    recipe=copy.deepcopy(u.read_json(p2.PROTOCOL));recipe['training']['max_epochs']=1
    fitset=series.windows('train'); val=series.windows('validation')
    fitset.window_starts=fitset.window_starts[:16]; val.window_starts=val.window_starts[:16]
    results={}
    for family,channels in protocol['arms'].items():
        for channel in channels:
            cfg=configuration(protocol,family,channel,64)
            result=fit(cfg,recipe,fitset,val,2024,torch.device('cpu'),out/cfg['arm'],{'smoke':True,'config':cfg})
            results[cfg['arm']]={'parameters':result['parameters'],'epochs':result['epochs_completed']}
    save(out/'smoke.json',{'passed':True,'fits':results})


def execute(root,out,protocol,device):
    identity={}
    sets,recipe=load_data(root,protocol,identity)
    save(out/'input_identity.json',identity)
    selected={}; records=[]
    def one(fid,cfg,seed,kind):
        series=sets[fid]; folder=out/kind/fid/cfg['arm']/f'd{cfg["args"]["d_model"]}_s{seed}'
        contract={'stage':'E1-A','fold':fid,'config':cfg,'seed':seed,'kind':kind,
                  'protocol_sha256':u.file_sha256(ROOT/'configs/e1_baselines_protocol.json'),
                  'input_identity_sha256':u.file_sha256(out/'input_identity.json')}
        record=fit(cfg,recipe,series.windows('train'),series.windows('validation'),seed,device,folder,contract)
        entry={'folder':str(folder.relative_to(out)),'config':cfg,'seed':seed,'fold':fid,'record':record}
        records.append(entry); return entry
    for family,channels in protocol['arms'].items():
        for channel in channels:
            if family=='DLinear': cfg=configuration(protocol,family,channel,64)
            else:
                options=[one('Kelmarsh',configuration(protocol,family,channel,d),2024,'search') for d in protocol['d_model_candidates']]
                cfg=choose(options)
            selected[cfg['arm']]=cfg
    save(out/'hyperparameter_selection.json',selected)
    formal=[]
    for fold in protocol['folds']:
        for cfg in selected.values():
            for seed in protocol['seeds']: formal.append(one(fold['id'],cfg,seed,'fits'))
    if len(records)!=64 or len(formal)!=54: raise ValueError('Fit count mismatch')
    for entry in records: trainer.verify_trained(out/entry['folder'],entry['record']['contract'])
    save(out/'selection_freeze.json',{'fits':records,'count':64,'formal_count':54})
    with (out/'evaluation_started.json').open('x') as f: json.dump({'utc':p2.now()},f)
    reports={}
    for fold in protocol['folds']:
        fid=fold['id']; windows=sets[fid].windows('test'); truth=windows.truth_and_anchor()[0]
        predictions={'TSMixer-upstream':reference(root,fid,windows)}
        dest=out/'evaluation'/fid; dest.mkdir(parents=True)
        np.savez_compressed(dest/'cohort.npz',truth=truth,times=windows.target_times(),starts=windows.window_starts)
        for entry in [r for r in formal if r['fold']==fid]:
            model,_=build_model(entry['config']); ck=torch.load(out/entry['folder']/'best.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(ck['model'],strict=True); model.to(device)
            prediction=trainer.v26run.predict(model,view(windows,entry['config']['channel']),entry['seed'],device,512)
            arm=entry['config']['arm']; predictions.setdefault(arm,[]).append(prediction)
            np.savez_compressed(dest/f'{arm}_s{entry["seed"]}.npz',prediction=prediction)
        comparisons={}
        for family in ('iTransformer','TimeXer'):
            comparisons[family+'_upstream_gain']={'candidate':family+'_upstream','reference':family+'_last10','question':'descriptive'}
        for arm in selected: comparisons[arm+'_vs_TSMixer']={'candidate':arm,'reference':'TSMixer-upstream','question':'descriptive'}
        from utils.kelmarsh_multiyear_benchmark import paired
        issue=windows.target_times()[:,0]-np.timedelta64(1,'h')
        comp={name:{**spec,**paired(predictions[spec['candidate']],predictions[spec['reference']],truth,issue,protocol['statistics'])}
              for name,spec in comparisons.items()}
        report={'metrics':{arm:u.seed_mean([u.metrics(p,truth,2050) for p in ps]) for arm,ps in predictions.items()},
                'comparisons':comp,'windows':len(windows),'scope':'descriptive only; no Holm; previously opened tests'}
        save(dest/'report.json',report); reports[fid]=report
    save(out/'result.json',{'status':'COMPLETED','fits':64,'reports':reports,'chronos':'SKIPPED: verification not available at launch',
                           'artifacts':{str(p.relative_to(out)):u.file_sha256(p) for p in (out/'evaluation').rglob('*.npz')}})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.out.mkdir(parents=True,exist_ok=False)
    protocol=u.read_json(ROOT/'configs/e1_baselines_protocol.json')
    if u.text_sha256_lf(p2.PROTOCOL)!=protocol['recipe_sha256_lf']: raise ValueError('Frozen recipe changed')
    sources=p2.source_code()
    for name in ('scripts/train_e1_baselines.py','tests/test_train_e1_baselines.py','models/iTransformer.py','models/TimeXer.py','models/PatchTST.py','models/DLinear.py'):
        sources[name]=u.file_sha256(ROOT/name)
    save(args.out/'launch.json',{'utc':p2.now(),'argv':sys.argv,'source_code':sources,'protocol':protocol,
                              'implementation_sha256':u.file_sha256(__file__)})
    try:
        if args.smoke: model_smoke(args.out,protocol)
        else: execute(args.data_root,args.out,protocol,torch.device(args.device))
        save(args.out/'completed.json',{'utc':p2.now()})
    except Exception as e:
        save(args.out/'failed.json',{'utc':p2.now(),'error':str(e),'type':type(e).__name__});raise

if __name__=='__main__':main()
