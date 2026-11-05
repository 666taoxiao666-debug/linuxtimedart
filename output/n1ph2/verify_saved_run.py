"""Run via stdin from the frozen server checkout; no training or model prediction."""
from pathlib import Path
import json
import tarfile
import numpy as np
from scripts import train_n1ph2 as n

n.torch.set_num_threads(2)
data=Path('/home/XT/nuist/pythoncode/TimeDARTFirst')
out=data/'outputs/logs/Penmanshiel/n1ph2_324e708_r1'
p=n.u.read_json(n.PROTOCOL)
base=n.u.read_json(n.ROOT/p['parent'])
result=n.u.read_json(out/'result.json')
complete=n.u.read_json(out/'audit/completed.json')
assert complete['wind_fits']==42 and complete['ph_mix_fits']==1
assert not (out/'audit/failed.json').exists()
launch=n.u.read_json(out/'audit/launch.json')
assert n.u.text_sha256_lf(n.PROTOCOL)==launch['protocol_sha256_lf']
for path,digest in launch['source_code'].items(): n.n.checked(n.ROOT/path,digest,{})
identities=n.u.read_json(out/'input_identity.json')
for path,digest in identities.items(): n.n.checked(path,digest['sha256'] if isinstance(digest,dict) else digest,{})
freeze=n.u.read_json(out/'selection_freeze.json')
n.p3.validate_freeze(out,freeze,42)
selection=n.u.read_json(out/'variant_selection.json')
assert n.require_selection(out,freeze['variant_selection_sha256'])=='V2'
ph=n.u.read_json(out/'ph_freeze.json')
n.n.checked(out/'PH-mix/selected.pt',ph['sha256'],{})
n.n.checked(out/'PH-mix/trained.json',ph['trained_sha256'],{})
marker=n.u.read_json(out/'test_evaluation_started.json')
assert marker['selection_sha256']==n.u.file_sha256(out/'selection_freeze.json')
assert selection['utc']<freeze['utc']<=marker['utc']<=complete['utc']
arrays=n.u.read_json(out/'evaluation_hashes.json')
for path,digest in arrays.items(): n.n.checked(out/path,digest,{})
replay={}; cohorts={}
for fold in base['folds']:
    fid=fold['id']
    ref=data/(base['p2_run'] if fid=='T21' else base['p3_run']+'/'+fid)
    with np.load(ref/'evaluation/cohort.npz',allow_pickle=False) as z: cohort={k:z[k] for k in z.files}
    cohorts[fid]=cohort; preds={}
    for arm in list(n.p2.ARMS)+base['arms']:
        preds[arm]=[]
        for seed in base['seeds']:
            historical=arm in n.p2.ARMS
            path=ref/f'evaluation/{arm}_s{seed}.npz' if historical else out/fid/'V2'/f'evaluation/{arm}_s{seed}.npz'
            with np.load(path,allow_pickle=False) as z:
                # P2 saves prediction-only files; their identity and cohort hash were checked above.
                if not (historical and fid=='T21'):
                    np.testing.assert_array_equal(z['starts'],cohort['starts'])
                assert z['prediction'].shape==cohort['truth'].shape
                preds[arm].append(z['prediction'])
    preds.update(PersistenceHour=[cohort['persistence']],PersistenceLast10=[cohort['last10']])
    recomputed=n.n.report(preds,cohort,base,fold['role'])
    assert n.u.canonical_sha256(recomputed)==n.u.canonical_sha256(result['folds'][fid]),fid
    replay[fid]=True
for v in p['variants']:
    preds=[]
    for seed in base['seeds']:
        folder=data/p['previous_run']/'T21' if v=='V0' else out/'T21'/v
        with np.load(folder/f'evaluation/K-PH_s{seed}.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['starts'],cohorts['T21']['starts'])
            preds.append(z['prediction'])
    metrics=n.u.seed_mean([n.u.metrics(x,cohorts['T21']['truth'],2050) for x in preds])
    assert n.u.canonical_sha256(metrics)==n.u.canonical_sha256(result['primary_zero_shot_descriptive'][v])
station=n.u.read_json(out/'station_description/result.json')
for sid,record in station['per_station'].items():
    with np.load(out/'station_description'/f'{sid}.npz',allow_pickle=False) as z:
        assert float(np.abs(z['prediction']-z['truth']).mean())==record['mix_mae']
verification={'utc':n.p2.now(),'passed':True,'wind_fits_verified':42,'ph_mix_verified':True,
    'source_files_verified':len(launch['source_code']),'inputs_verified':len(identities),
    'saved_arrays_verified':len(arrays),'fold_report_replay':replay,
    'all_variant_primary_metrics_replay':True,'station_own_metrics_replay':len(station['per_station']),
    'no_training_or_new_predictions':True,
    'transfer_station_pairs_note':'Pair metrics recorded by original run; individual pair predictions not persisted, no new forward passes for replay'}
n.save(out/'audit/final_verification.json',verification)
export={str(x.relative_to(out)):n.u.file_sha256(x) for x in out.rglob('*')
        if x.is_file() and x.suffix in ('.json','.log') and x.name!='metadata_sha256.json'}
n.save(out/'metadata_sha256.json',export)
archive=data/'outputs/audits/n1ph2_324e708_metadata.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for rel in list(export)+['metadata_sha256.json']: t.add(out/rel,arcname=rel)
compact={}
for fid,r in result['folds'].items():
    compact[fid]={'decision':r['decision'],
        'metrics':{a:{k:m[k] for k in ('mae','rmse','r2')} for a,m in r['metrics']['all'].items()},
        'comparisons':{k:{'gain':c['absolute_error']['gain'],
            'ci':[c['absolute_error']['ci95_low'],c['absolute_error']['ci95_high']],
            'holm_p':c.get('holm_p'),'rmse_gain':c['rmse_difference']['rmse_gain_kw'],
            'rmse_ci':[c['rmse_difference']['ci95_low_kw'],c['rmse_difference']['ci95_high_kw']]}
            for k,c in r['comparisons'].items()}}
print(json.dumps({'verification':verification,'results':compact,
    'archive_sha256':n.u.file_sha256(archive),'metadata_files':len(export)+1}))
