"""Read-only T1 acceptance: hashes and saved arrays only; emits metadata bundle."""
import base64
import collections
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

RUN = Path('/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/Tuning/t1_a99037f_r1')
CODE = Path('/home/XT/nuist/pythoncode/TimeDARTFirst_T1_a99037f')

def read(p):
    return json.loads(p.read_text())

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

counts = collections.Counter()
def check(p, expected, kind):
    assert sha(p) == expected, str(p)
    counts[kind] += 1

def metrics(p, y):
    assert p.shape == y.shape and np.isfinite(p).all() and np.isfinite(y).all()
    d = p-y
    return dict(mae=float(np.abs(d).mean()), rmse=float(np.sqrt(np.square(d).mean())),
                r2=float(1-np.square(d).sum()/np.square(y-y.mean()).sum()))

def close(actual, expected):
    for k in ('mae', 'rmse', 'r2'):
        np.testing.assert_allclose(actual[k], expected[k], rtol=1e-12, atol=1e-10)

completed = read(RUN/'completed.json')
assert completed['status'] == 'COMPLETED' and not (RUN/'failed.json').exists()
launch = read(RUN/'launch.json')
for p,h in launch['source_code'].items(): check(CODE/p,h,'source_files')
for p,h in read(RUN/'input_identity.json').items(): check(Path(p),h,'input_files')
tuning = read(RUN/'tuning_freeze.json')
training = read(RUN/'training_freeze.json')
evaluation = read(RUN/'evaluation_started.json')
check(RUN/'tuning_freeze.json',training['tuning_sha256'],'freeze_links')
check(RUN/'training_freeze.json',evaluation['training_freeze_sha256'],'freeze_links')
assert tuning['utc'] <= training['utc'] <= evaluation['utc'] <= completed['utc']

def fit(entry):
    folder = RUN/entry['folder']
    assert read(folder/'trained.json') == entry['record'], str(folder)
    for name,h in entry['record']['artifacts'].items(): check(folder/name,h,'fit_artifacts')
    c = entry['record']['contract']
    for name,key in [('launch.json','launch_sha256'),('input_identity.json','input_identity_sha256')]:
        check(RUN/name,c[key],'contract_links')
    check(CODE/'configs/t1_tuning_protocol.json',c['protocol_sha256'],'contract_links')

trial_status = {}
for family in launch['protocol']['models']:
    path = RUN/'search'/family/'trials.json'
    check(path,tuning['trial_metadata_sha256'][family],'trial_metadata')
    trials = read(path)
    assert len(trials)==60 and sorted(t['number'] for t in trials)==list(range(60))
    valid = [t for t in trials if t['status']=='COMPLETE']
    assert min(valid,key=lambda t:(t['value'],t['number'])) == tuning['selected'][family]
    for t in valid: fit(t['fit'])
    trial_status[family] = dict(collections.Counter(t['status'] for t in trials))
assert tuning['attempted_trials']==180 and training['count']==len(training['records'])==60
expected = {(f,m,k,s) for f in launch['protocol']['folds'] for m in launch['protocol']['models']
            for k in ('existing','tuned') for s in launch['protocol']['seeds']}
actual = {(e['farm'],e['config']['family'],e['kind'],e['seed']) for e in training['records']}
assert actual==expected
for entry in training['records']: fit(entry)
result = read(RUN/'result.json')
assert result['status']=='COMPLETED' and result['formal_fits']==60 and result['attempted_search_trials']==180
assert set(result['artifacts'])=={str(p.relative_to(RUN)) for p in (RUN/'evaluation').rglob('*.npz')}
for p,h in result['artifacts'].items(): check(RUN/p,h,'prediction_artifacts')
gains = {}
for farm,report in result['reports'].items():
    folder=RUN/'evaluation'/farm
    assert read(folder/'report.json')==report
    with np.load(folder/'cohort.npz') as z: truth=z['truth']
    for arm,row in report.items():
        preds=[]
        for i,seed in enumerate(launch['protocol']['seeds']):
            with np.load(folder/f'{arm}_s{seed}.npz') as z: p=z['prediction']
            close(metrics(p,truth),row['per_seed'][i]); preds.append(p)
        ensemble=np.mean(preds,axis=0)
        with np.load(folder/f'{arm}_ensemble.npz') as z: np.testing.assert_array_equal(ensemble,z['prediction'])
        close(metrics(ensemble,truth),row['ensemble'])
    a,b=(report['TSMixerBasicSmall_'+kind]['ensemble'] for kind in ('existing','tuned'))
    gains[farm]={'mae_reduction_pct':100*(1-b['mae']/a['mae']), 'rmse_change_kw':b['rmse']-a['rmse'],
                 'passes':bool(b['mae']<=.98*a['mae'] and b['rmse']<=a['rmse'])}
decision='ADOPT_TUNED_DEVELOPMENT_ONLY' if all(g['passes'] for g in gains.values()) else 'RETAIN_EXISTING'
assert result['decision']==decision
verification={'status':'PASSED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'run':str(RUN),'code':str(CODE),'counts':dict(counts),'trial_status':trial_status,
              'formal_fits':60,'gates':gains,'decision':decision,'no_training':True,'no_inference':True,
              'result_sha256':sha(RUN/'result.json')}
files={}
for p in sorted(RUN.rglob('*.json')):
    files[str(p.relative_to(RUN))]=base64.b64encode(p.read_bytes()).decode()
files['run.log']=base64.b64encode(Path(str(RUN)+'.log').read_bytes()).decode()
print(json.dumps({'verification':verification,'files':files,'file_sha256':{n:hashlib.sha256(base64.b64decode(b)).hexdigest() for n,b in files.items()}}))
