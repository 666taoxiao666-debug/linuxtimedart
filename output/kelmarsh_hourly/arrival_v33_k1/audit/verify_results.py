import os, sys, json, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
workspace = Path('/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/audits/arrival_v33_k1/20261031T060631Z/workspace')
audit = workspace.parent
out = Path('/home/XT/nuist/pythoncode/TimeDARTFirst/outputs/logs/Kelmarsh/v33_k1_20261031T060631Z')
os.chdir(workspace)
sys.path.insert(0, str(workspace))
from scripts import train_kelmarsh_arrival_v33 as ctl
assert (out/'k1_decision.json').exists(), 'run not complete'
proc = Path('/proc/4636/cmdline')
assert not proc.exists() or b'train_kelmarsh_arrival_v33.py' not in proc.read_bytes(), 'real run still alive'
cfg = json.loads(ctl.CONFIG.read_text())
decision = json.loads((out/'k1_decision.json').read_text())
manifest = json.loads((out/'data_manifest.json').read_text())
launch = json.loads((audit/'launch.json').read_text())
preflight = json.loads((audit/'preflight.json').read_text())
assert manifest['code'] == launch['source_code'] == ctl.code_digest()
assert not decision['smoke'] and not manifest['smoke']
ctl.v26run.configure_determinism()
p,series,observations,stations,identity = ctl.load_real(cfg)
assert identity == manifest['identity'] == preflight['identity']
views = {name:ctl.v28.ChannelView(series.windows(split,'all'),list(range(len(series.channels))))
         for name,split in [('validation','validation'),('development_eval','test')]}
truth = {name:v.truth_and_anchor()[0] for name,v in views.items()}
def independent(pred,y):
    assert pred.shape == y.shape and pred.size and np.isfinite(pred).all() and np.isfinite(y).all()
    err = np.asarray(pred,dtype=np.float64)-np.asarray(y,dtype=np.float64)
    return {'mae':float(np.mean(np.abs(err))), 'rmse':float(np.sqrt(np.mean(err**2)))}
def match(s,t,tol=1e-8):
    assert all(abs(s[k]-t[k])<=tol for k in ['mae','rmse']), (s,t)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
b0 = json.loads((out/'B0/trained.json').read_text())
b0hist = json.loads((out/'B0/training_history.json').read_text())
assert b0['best_epoch'] == min(b0hist,key=lambda r:(r['selection']['mae'],r['epoch']))['epoch']
for name,digest in b0['artifacts'].items():
    assert sha(out/'B0'/name)==digest, name
base, _ = ctl.v28.build_model(b0['contract']['config'])
base_state = torch.load(out/'B0/best.pt',map_location='cpu',weights_only=False)['model']
assert ctl.v26run.state_digest(base_state)==b0['best_state_sha256']
base.load_state_dict(base_state)
device = torch.device('cuda')
base.to(device).eval()
recomputed = {}
b0preds = {}
for name,view in views.items():
    pred = ctl.v26run.predict(base,view,cfg['seed'],device,512)
    b0preds[name] = pred
    s = independent(pred,truth[name])
    match(s,decision['metrics']['B0'][name],1e-6)
    recomputed.setdefault('B0',{})[name]=s
    anchor = view.truth_and_anchor()[1]
    match(independent(anchor,truth[name]),decision['metrics']['Persistence'][name])
with np.load(out/'B0/validation_predictions.npz') as z:
    assert np.array_equal(z['window_starts'],views['validation'].window_starts)
    assert np.array_equal(z['truth'],truth['validation'])
    match(independent(z['prediction'],truth['validation']),decision['metrics']['B0']['validation'])
arms = cfg['budget']['planned'][1:]
assert sorted(q.parent.name for q in (out/'arms').glob('*/trained.json')) == sorted(arms)
selected = {}
param_counts = {}
prediction_hashes = {}
base_frozen = {}
for arm in arms:
    folder=out/'arms'/arm
    trained=json.loads((folder/'trained.json').read_text())
    hist=json.loads((folder/'history.json').read_text())
    assert trained['contract']['code']==manifest['code']
    assert trained['contract']['seed']==cfg['seed']
    assert trained['contract']['base_state_sha256']==b0['best_state_sha256']
    assert trained['best_epoch']==min(hist,key=lambda r:(r['J'],r['epoch']))['epoch']
    assert trained['validation_J']==min(r['J'] for r in hist)
    assert len(hist)-1<=cfg['training']['max_epochs']
    ckpt=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    assert ckpt['epoch']==trained['best_epoch'] and ckpt['contract']==trained['contract']
    assert all(torch.equal(ckpt['model']['base.'+k],v) for k,v in base_state.items()), arm
    base_frozen[arm]=True
    with np.load(folder/'predictions.npz') as z:
        assert np.array_equal(z['validation_starts'],views['validation'].window_starts)
        assert np.array_equal(z['eval_starts'],views['development_eval'].window_starts)
        for name in views:
            s=independent(z[name],truth[name])
            match(s,trained[name])
            match(s,decision['metrics'][arm][name])
            recomputed.setdefault(arm,{})[name]=s
    j=recomputed[arm]['validation']['mae']/recomputed['B0']['validation']['mae'] + recomputed[arm]['validation']['rmse']/recomputed['B0']['validation']['rmse']
    assert abs(j-trained['validation_J'])<1e-8
    param_counts[arm]=trained['trainable_parameters']
    selected[arm]={'best_epoch':trained['best_epoch'],'epochs_completed':len(hist)-1,'validation_J':trained['validation_J'],'kappa':trained['kappa']}
    prediction_hashes[arm]=sha(folder/'predictions.npz')
    assert trained['support_rate_eval_by_h']==decision['support_rate_eval_by_h'][arm]
assert param_counts['A']-param_counts['B2']==1 and max(param_counts.values())<=cfg['branch']['max_parameters']
grid={float(k):v for k,v in decision['kappa_grid_validation_J'].items()}
assert set(grid)==set(cfg['kappa_star']['grid'])
for k,v in grid.items():
    assert v==selected['K'+format(k,'g')]['validation_J']
minimum=min(grid.values())
kstar=sorted([k for k,v in grid.items() if v-minimum<cfg['kappa_star']['tie_tolerance']],key=lambda k:(abs(k-1),k))[0]
assert kstar==decision['kappa_star']
assert selected['A']['kappa']==decision['learned_kappa']
m={k:v['development_eval'] for k,v in recomputed.items()}
dr=lambda x,y:100*(m[y]['rmse']-m[x]['rmse'])/m[y]['rmse']
dm=lambda x,y:m[y]['mae']-m[x]['mae']
d1,d2=cfg['decision']['D1_mechanism'],cfg['decision']['D2_fusion']
branch='D1' if (dr('A','B2')>=d1['rmse_gain_vs_B2_pct_min'] and dm('A','B2')>=d1['mae_gain_vs_B2_kw_min'] and dr('A','B0')>=d1['rmse_gain_vs_B0_pct_min'] and dm('A','B0')>=d1['mae_gain_vs_B0_kw_min']) else ('D2' if dr('B2','B0')>=d2['rmse_gain_B2_vs_B0_pct_min'] and dm('B2','B0')>=d2['mae_gain_B2_vs_B0_kw_min'] else 'D3')
assert branch==decision['decision']['branch']
records=['k1_decision.json','k1_summary.md','data_manifest.json','B0/trained.json','B0/training_history.json']
records += [f'arms/{a}/{n}' for a in arms for n in ['trained.json','history.json']]
result={'passed':True,'utc':datetime.now(timezone.utc).isoformat(),'training_restarted':False,'real_fits':1+len(arms),
        'method':'NumPy absolute/squared error recomputation on saved branch predictions, B0 selected-checkpoint inference replay; no training',
        'scope':'validation 2021 and development evaluation 2022, ALL h1-4',
        'identity_matches_preflight':True,'complete_finite_cohort':True,'windows':manifest['windows'],
        'selected_epochs':{'B0':{'best_epoch':b0['best_epoch'],'epochs_completed':b0['epochs_completed']},**selected},
        'all_branch_base_states_unchanged':base_frozen,'parameters':param_counts,'metrics':recomputed,
        'kappa_star':kstar,'learned_kappa':decision['learned_kappa'],'decision':branch,
        'result_file_sha256':{f:sha(out/f) for f in records},'branch_prediction_sha256':prediction_hashes,
        'base_checkpoint_sha256':sha(out/'B0/best.pt'),'source_code':manifest['code']}
print(json.dumps(result,indent=2))
