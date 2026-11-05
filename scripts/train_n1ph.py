"""N1-PH: frozen N0 feature extractor, unchanged P2 trainer, one development run."""
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from scripts import train_penmanshiel_p2_v35 as p2
from scripts import train_penmanshiel_p3_v35 as p3
from scripts import pilot_station_operator_n0 as n0
from data_provider.kelmarsh_multiyear import MultiYearWindows
from utils import kelmarsh_upstream_v31 as up

u, trainer, save = p2.u, p2.trainer, p2.save
PROTOCOL = ROOT / 'configs/n1ph_protocol.json'


def checked(path, digest, identities):
    path = Path(path)
    if u.file_sha256(path) != digest:
        raise ValueError(f'Artifact identity mismatch: {path}')
    identities[str(path)] = digest
    return path


def load_ph(path, digest, identities):
    checked(path, digest, identities)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    mean, scale = [np.asarray(x) for x in checkpoint['stats']]
    if mean.shape != (11,) or scale.shape != (11,) or not np.isfinite([mean, scale]).all() or not (scale > 0).all():
        raise ValueError('Invalid N0 statistics')
    model = torch.nn.Sequential(torch.nn.Linear(11,32), torch.nn.ReLU(), torch.nn.Linear(32,32),
                                torch.nn.ReLU(), torch.nn.Linear(32,4))
    model.load_state_dict(checkpoint['state'], strict=True)
    model.eval().requires_grad_(False)
    return model, (mean, scale)


def ph_features(frame, model, stats):
    if list(frame.columns) != up.UPSTREAM_FEATURES or frame.index.has_duplicates:
        raise ValueError('PH input schema mismatch')
    # Sparse timestamps are missing observations, not missing forecast rows.
    # Fill their H11 through the same frozen N0 transform before PH inference.
    frame = frame.reindex(pd.date_range(frame.index.min(),frame.index.max(),freq='h'))
    raw = frame.to_numpy(dtype=float)
    x = n0.transform(raw, stats)
    with torch.inference_mode():
        values = np.concatenate([model(torch.from_numpy(x[i:i+8192])).numpy() for i in range(0,len(x),8192)])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite PH prediction')
    return pd.DataFrame(np.c_[values, (~np.isfinite(raw).all(1)).astype(float)], index=frame.index,
                        columns=['PH_h1','PH_h2','PH_h3','PH_h4','upstream_missing'])


def ph_series(template, features, state=None):
    """The source-state branch never accesses template.scaler/audit or computes target reductions."""
    series = copy.copy(template)
    extra = features.reindex(pd.DatetimeIndex(template.dates)).to_numpy(dtype=float)
    if not np.isfinite(extra).all():
        raise ValueError('PH features must cover every hourly input row')
    raw = np.c_[template.base.raw[:,:-1], extra, template.base.raw[:,-1]]
    channels = list(template.base.channels[:-1]) + list(features.columns) + ['power']
    if len(channels) != 14 or channels.count('power') != 1:
        raise ValueError('Expected 14 channels and one final power')
    if state is None:
        rows = np.isin(template.label_year, template.splits['train'])
        mean, scale = np.nanmean(raw[rows],axis=0), np.nanstd(raw[rows],axis=0)
        scale = np.where(scale == 0, 1., scale)
    else:
        if state['channels'] != channels:
            raise ValueError('Source channel mismatch')
        mean, scale = np.array(state['mean']), np.array(state['scale'])
    if not np.isfinite([mean,scale]).all() or not (scale>0).all():
        raise ValueError('Invalid outer fit statistics')
    series.raw, series.channels = raw, channels
    series.scaler_mean, series.scaler_scale = mean, scale
    series.values = ((raw-mean)/scale).astype(np.float32)
    series.audit = {'channels':channels,'mean':mean.tolist(),'scale':scale.tolist(),
                    'origin':'source_only' if state is not None else 'farm_fit_only'}
    np.testing.assert_array_equal(series.raw[:,-1], template.base.raw[:,-1])
    return series


def assert_cohort(windows, reference, normal):
    for name, value in [('starts', windows.window_starts), ('truth',windows.truth_and_anchor()[0]),
                        ('times',windows.target_times()), ('normal_mask',normal)]:
        if not np.array_equal(value, reference[name]):
            raise ValueError(f'Reference cohort mismatch: {name}')


def reference(data_root, fold, protocol, identities):
    if fold['id']=='T21':
        folder = data_root/protocol['p2_run']
        metadata = ROOT/'output/penmanshiel_hourly/p2_v35'
        hashes = {k:v['sha256'] for k,v in u.read_json(metadata/'audit/final_verification.json')['evaluation_files'].items()}
    else:
        folder = data_root/protocol['p3_run']/fold['id']
        metadata = ROOT/'output/penmanshiel_hourly/p3_v35/r2'/fold['id']
        hashes = {'evaluation/'+k:v for k,v in u.read_json(metadata/'evaluation_hashes.json').items()}
    for name, digest in hashes.items():
        checked(folder/name, digest, identities)
    # Validate historical selected state identities against committed trained records, not fresh discovery.
    for arm in ('K-last10','K-upstream','FT-1','FT-3','P-upstream','P-last10','Scratch-1','Scratch-3'):
        for seed in protocol['seeds']:
            relative = Path('fits')/arm/f's{seed}'
            old = u.read_json(metadata/relative/'trained.json')
            actual = u.read_json(folder/relative/'trained.json')
            if u.canonical_sha256(old) != u.canonical_sha256(actual):
                raise ValueError('Historical training registration mismatch')
            trainer.verify_trained(folder/relative, old['contract'])
    with np.load(folder/'evaluation/cohort.npz',allow_pickle=False) as archive:
        cohort = {k:archive[k] for k in archive.files}
    return folder, cohort


def other_inputs(data_root, protocol, recipe, identities):
    prepared = data_root/protocol['p3_run']/'prepared'
    expected = u.read_json(ROOT/'output/penmanshiel_hourly/p3_v35/r2/prepared/prepared_complete.json')
    for name,digest in expected['artifacts'].items():
        checked(prepared/name,digest,identities)
    pf = pd.concat([c[c.label_year<=2023] for c in pd.read_csv(prepared/'penmanshiel.csv',chunksize=50000)],ignore_index=True)
    pup = pd.read_csv(prepared/'upstream.csv',index_col='timestamp_hour_end_utc')
    pup.index = pd.to_datetime(pup.index)
    pup = pup[pup.index < pd.Timestamp('2024-01-01T01:00:00')]
    meta = u.read_json(prepared/'preparation.json')
    ref = recipe['data']['v31_registered_preparation']
    kcsv = checked(data_root/ref['csv'],ref['csv_sha256'],identities)
    km = checked(kcsv.parent.parent/'metadata/preparation.json',ref['preparation_json_sha256'],identities)
    kf = pd.concat([c[c.label_year<=2022] for c in pd.read_csv(kcsv,chunksize=50000)],ignore_index=True)
    spec = recipe['data']['upstream']
    manifest = data_root/spec['download_manifest']
    if u.text_sha256_lf(manifest) != spec['download_manifest_sha256_lf']:
        raise ValueError('MIDAS manifest mismatch')
    hashes = u.manifest_hashes(u.read_json(manifest))
    ids = [int(s['src_id']) for s in spec['stations']]
    entries=[]
    for item in spec['development_roots']:
        years=sorted(set(item['years']) & set(range(2016,2023)))
        if years:
            entries.extend(u.station_files(data_root/item['root'],ids,years,hashes,required=True)[0])
    for e in entries:
        checked(e['path'],e['sha256'],identities)
    obs=u.load_station_observations(entries,ids)
    kup=u.hourly_upstream(obs,spec['stations'],pd.date_range('2016-05-01','2023-01-01',freq='h'),1)
    return pf,pup,meta,kf,kup,u.read_json(km)


def configuration(arm, recipe, protocol):
    return {'arm':arm,'family':'TSMixerBasicSmall','channels':14,
            'args':copy.deepcopy(recipe['families']['TSMixerBasicSmall']['args']),
            'expected_parameters':protocol['parameters']}


def verdict(comparisons, margin=2):
    supported={k:c['absolute_error']['ci95_low']>0 and c['holm_p']<=.05
               and c['rmse_difference']['ci95_low_kw']>0 for k,c in comparisons.items() if k!='IP'}
    ni=comparisons['IP']['absolute_error']['ci95_low'] > -margin
    result='continue' if supported['ZP'] and ni else ('partial' if not supported['ZP'] and
             (supported['F1P'] or supported['F3P']) else 'stop')
    return {'outcome':result,'supported':supported,'IP_noninferior':ni}


def report(predictions, cohort, protocol, role):
    truth=cohort['truth']; issue=cohort['times'][:,0]-np.timedelta64(1,'h')
    comparisons={}
    for names in (protocol['family'],['IP']):
        specs={k:{'candidate':protocol['comparisons'][k][0],'reference':protocol['comparisons'][k][1],'question':k} for k in names}
        comparisons.update(u.compare_all(predictions,truth,issue,specs,None,protocol['statistics']))
    comparisons['IP'].pop('holm_p',None)
    decision=verdict(comparisons,protocol['ip_margin_kw']) if role=='primary_development' else None
    if role=='descriptive':
        for c in comparisons.values(): c.pop('holm_p',None)
    metrics={}
    for name,mask in [('all',np.ones(len(truth),bool)),('normal',cohort['normal_mask'])]:
        metrics[name]={a:u.seed_mean([u.metrics(p[mask],truth[mask],2050) for p in ps]) for a,ps in predictions.items()}
    return {'role':role,'metrics':metrics,'comparisons':comparisons,'decision':decision,
            'per_seed':{a:[u.metrics(p,truth,2050) for p in ps] for a,ps in predictions.items()},
            'windows':len(truth)}


def run(data_root, out, protocol, device, preflight=False):
    began=time.monotonic()
    def timeout(*_): raise TimeoutError('N1-PH total budget exceeded; no restart')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(protocol['budget_seconds'])
    identities={}
    recipe=u.read_json(ROOT/protocol['recipe'])
    if u.text_sha256_lf(ROOT/protocol['recipe']) != protocol['recipe_sha256_lf']:
        raise ValueError('Recipe identity mismatch')
    model,stats=load_ph(data_root/protocol['ph_run']/'models/PH/selected.pt',protocol['ph_sha256'],identities)
    refs={f['id']:reference(data_root,f,protocol,identities) for f in protocol['folds']}
    pen,pup,kel,p2ids=p2.load_data(data_root,recipe)
    identities.update(p2ids)
    # Obtain original H11, not the Pen/Kel filled/standardized representation.
    # Rebuild raw H11 on unique timestamps: filled features are forbidden as PH input.
    spec=recipe['data']['upstream']; mh=u.manifest_hashes(u.read_json(data_root/spec['download_manifest']))
    ids=[int(s['src_id']) for s in spec['stations']]
    entries=u.station_files(data_root/spec['development_roots'][0]['root'],ids,range(2016,2021),mh,required=True)[0]
    kup=u.hourly_upstream(u.load_station_observations(entries,ids),spec['stations'],pd.date_range('2016-05-01','2021-01-01',freq='h'),1)
    data=other_inputs(data_root,protocol,recipe,identities)
    cohorts={}; variants={}; all_records=[]; before=p2.source_code()
    identity={'protocol_sha256_lf':u.text_sha256_lf(PROTOCOL),'input_sha256':u.canonical_sha256(identities),'source_code':before}
    save(out/'input_identity.json',identities)
    for fold in protocol['folds']:
        fid=fold['id']
        if fid=='T21':
            pt,kt,pu,ku=pen,kel,pup,kup
            split='development_eval'
        else:
            pf,pu,meta,kf,ku,kmeta=data
            splits={'train':fold['fit_years'],'validation':[fold['validation_year']],'test':[fold['test_year']]}
            pt=u.KelmarshUpstreamSeries(pf,pu,splits=splits,start=meta['start'],end=f"{fold['test_year']+1}-01-01T01:00:00Z",preparation=meta)
            kt=u.KelmarshUpstreamSeries(kf,ku,splits={k:v for k,v in splits.items() if k!='test'},start='2016-05-01',end=f"{fold['test_year']}-01-01T01:00:00Z",preparation=kmeta)
            split='test'
        kph=ph_series(kt,ph_features(ku,model,stats))
        target_features=ph_features(pu,model,stats)
        transfer=ph_series(pt,target_features,state=kph.audit)
        pph=ph_series(pt,target_features)
        normal=pt.normal_window[pt.split_of==split]
        assert_cohort(pph.windows(split),refs[fid][1],normal)
        assert_cohort(transfer.windows(split),refs[fid][1],normal)
        cohorts[fid]=refs[fid][1]
        variants[fid]={'K-PH':transfer,'FT-1-PH':transfer,'FT-3-PH':transfer,'P-PH':pph}
        save(out/fid/'data_audit.json',{'K':kph.audit,'P':pph.audit,'target_transfer':transfer.audit,'channels':pph.channels})
        if preflight:
            for months in (1,3):
                fit,val=p3.fewshot_windows(transfer,fold['fewshot_year'],months)
                save(out/fid/f'fewshot_{months}.json',{'fit':len(fit),'validation':len(val)})
            continue
        source={}
        for arm in protocol['arms']:
            series=kph if arm=='K-PH' else variants[fid][arm]
            if arm.startswith('FT'):
                fit,val=p3.fewshot_windows(series,fold['fewshot_year'],int(arm.split('-')[1]))
            else: fit,val=series.windows('train'),series.windows('validation')
            for seed in protocol['seeds']:
                folder=out/fid/'fits'/arm/f's{seed}'
                warm=source[seed] if arm.startswith('FT') else None
                contract={'stage':'N1-PH','fold':fold,'arm':arm,'seed':seed,'identity':identity,'statistics':series.audit,
                          'warm_state_sha256':None if warm is None else trainer.v26run.state_digest(warm),
                          'fit_starts_sha256':hashlib.sha256(fit.window_starts.tobytes()).hexdigest(),
                          'validation_starts_sha256':hashlib.sha256(val.window_starts.tobytes()).hexdigest()}
                print(f'[FIT] {fid}/{arm}/s{seed} fit={len(fit)} val={len(val)}',flush=True)
                record=p2.fit_one(configuration(arm,recipe,protocol),recipe,trainer.ChannelView(fit,list(range(14))),
                                  trainer.ChannelView(val,list(range(14))),seed,device,folder,contract,warm)
                record.pop('reused_from_registry',None)
                if arm=='K-PH': source[seed]=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['model']
                all_records.append({'directory':str(folder.relative_to(out)),'record':record,'trained_sha256':u.file_sha256(folder/'trained.json')})
    if preflight:
        save(out/'preflight.json',{'passed':True,'fits':0,'identity':identity,'folds':list(cohorts),
                                 'cohort_equality':['starts','truth','times','normal_mask']})
        signal.alarm(0)
        return
    if p2.source_code()!=before or u.text_sha256_lf(PROTOCOL)!=identity['protocol_sha256_lf']:
        raise ValueError('Source or protocol changed')
    # Recheck registered references/inputs before opening the evaluation gate.
    for path,value in identities.items():
        checked(path,value['sha256'] if isinstance(value,dict) else value,{})
    freeze={'expected_fits':36,'fits':all_records,'identity':identity,'utc':p2.now()}
    save(out/'selection_freeze.json',freeze)
    p3.begin_evaluation(out,freeze,36)
    results={}
    for fold in protocol['folds']:
        fid=fold['id']; cohort=cohorts[fid]; dest=out/fid/'evaluation'; dest.mkdir(parents=True)
        predictions={}
        for old in p2.ARMS:
            predictions[old]=[]
            for seed in protocol['seeds']:
                with np.load(refs[fid][0]/f'evaluation/{old}_s{seed}.npz',allow_pickle=False) as z:
                    predictions[old].append(z['prediction'])
        predictions.update(PersistenceHour=[cohort['persistence']],PersistenceLast10=[cohort['last10']])
        for arm in protocol['arms']:
            predictions[arm]=[]
            windows=variants[fid][arm].windows('development_eval' if fid=='T21' else 'test')
            for seed in protocol['seeds']:
                folder=out/fid/'fits'/arm/f's{seed}'
                m,_=trainer.build_model(configuration(arm,recipe,protocol))
                m.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['model'])
                pred=trainer.v26run.predict(m.to(device),trainer.ChannelView(windows,list(range(14))),seed,device,recipe['training']['eval_batch_size'])
                np.savez_compressed(dest/f'{arm}_s{seed}.npz',prediction=pred,starts=windows.window_starts)
                predictions[arm].append(pred)
        results[fid]=report(predictions,cohort,protocol,fold['role'])
        save(out/fid/'result.json',results[fid])
        save(out/fid/'evaluation_hashes.json',{p.name:u.file_sha256(p) for p in dest.glob('*.npz')})
        reloaded=dict(predictions)
        for arm in protocol['arms']:
            reloaded[arm]=[]
            for seed in protocol['seeds']:
                with np.load(dest/f'{arm}_s{seed}.npz',allow_pickle=False) as z: reloaded[arm].append(z['prediction'])
        if u.canonical_sha256(report(reloaded,cohort,protocol,fold['role'])) != u.canonical_sha256(results[fid]):
            raise ValueError('Report replay mismatch')
    if p2.source_code()!=before: raise ValueError('Source changed')
    save(out/'result.json',results)
    save(out/'audit/completed.json',{'utc':p2.now(),'fits':36,'elapsed':time.monotonic()-began,'source_unchanged':True,'report_recomputed':True})
    signal.alarm(0)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True); parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args(); protocol=u.read_json(PROTOCOL)
    args.out.mkdir(parents=True,exist_ok=True)
    if args.preflight_only:
        torch.set_num_threads(2)
        run(args.data_root,args.out,protocol,'cpu',preflight=True)
        return
    lockpath=args.data_root/'outputs/logs/Penmanshiel/n1ph.lock'
    with lockpath.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        registry=lockpath.with_suffix('.launched.json')
        with registry.open('x') as f: json.dump({'out':str(args.out),'utc':p2.now()},f)
        torch.set_num_threads(2); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        save(args.out/'audit/launch.json',{'utc':p2.now(),'pid':os.getpid(),'command':sys.argv,
             'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'protocol_sha256_lf':u.text_sha256_lf(PROTOCOL),'source_code':p2.source_code()})
        try: run(args.data_root,args.out,protocol,args.device)
        except BaseException as error:
            save(args.out/'audit/failed.json',{'utc':p2.now(),'type':type(error).__name__,'message':str(error),'test_started':(args.out/'test_evaluation_started.json').exists()})
            raise


if __name__=='__main__': main()
