"""T1: fixed-budget retrospective tuning, common 72h queue, single evaluation."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import types
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts import train_e1_baselines as e1
from scripts import pilot_conformal_c0b as cb
u, trainer, p2, n1 = e1.u, e1.trainer, e1.p2, e1.n1
save, sha = cb.save, cb.sha
PROTOCOL = ROOT/'configs/t1_tuning_protocol.json'


def common_view(windows, length):
    if windows.seq_len != 72 or length not in (24, 48, 72):
        raise ValueError('A shared 72h queue and approved input length are required')
    view = copy.copy(windows)
    view.seq_len = length
    view.window_starts = windows.window_starts + 72 - length
    np.testing.assert_array_equal(view.target_times(), windows.target_times())
    np.testing.assert_array_equal(view.truth_and_anchor()[0], windows.truth_and_anchor()[0])
    return view


def build_model(config):
    args = config['args']
    if config['family'] == 'TSMixerBasicSmall':
        from models.TSMixerBasicSmall import Model
        model = Model(**args, channels=21, target_slot=20, input_mode='full')
    else:
        model = importlib.import_module('models.'+config['family']).Model(types.SimpleNamespace(**args, enc_in=21))
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if 'expected_parameters' in config and count != config['expected_parameters']:
        raise ValueError('Parameter count changed')
    return model, count


def config_for(family, params):
    if family == 'TSMixerBasicSmall':
        args = dict(u.read_json(p2.PROTOCOL)['families'][family]['args'], seq_len=24, pred_len=4)
    else:
        args = copy.deepcopy(u.read_json(ROOT/'output/e1_baselines/a2/r1/hyperparameter_selection.json')[family+'_upstream']['args'])
    for key in ('blocks', 'ff_dim', 'd_model', 'e_layers', 'dropout', 'seq_len'):
        if key in params: args[key] = params[key]
    if family != 'TSMixerBasicSmall':
        args['d_ff'] = 4*args['d_model']
        if args['use_norm'] is not False: raise ValueError('A2 normalization must remain disabled')
    cfg = {'family': family, 'arm': family+'_upstream', 'channels': 21, 'args': args}
    _, cfg['expected_parameters'] = build_model(cfg)
    return cfg


def recipe_for(params, short=False):
    recipe = copy.deepcopy(u.read_json(p2.PROTOCOL))
    for key in ('learning_rate', 'weight_decay', 'batch_size', 'loss_mae_weight'):
        if key in params: recipe['training'][key] = params[key]
    recipe['training']['loss_mse_weight'] = 1-recipe['training']['loss_mae_weight']
    if short: recipe['training'].update(max_epochs=10, patience=3)
    return recipe


def suggest(trial, family, protocol):
    space = protocol['search_space']
    keys = ('blocks', 'ff_dim') if family == 'TSMixerBasicSmall' else ('d_model', 'e_layers')
    params = {k: trial.suggest_categorical(k, space[k]) for k in keys}
    params['dropout'] = trial.suggest_float('dropout', *space['dropout'])
    for k in ('learning_rate', 'weight_decay'):
        params[k] = trial.suggest_float(k, *space[k+'_log'], log=True)
    for k in ('batch_size', 'seq_len', 'loss_mae_weight'):
        params[k] = trial.suggest_categorical(k, space[k])
    return params


def fit(config, recipe, windows, seed, device, out, contract):
    if out.exists(): raise FileExistsError('No fit reuse or resume: '+str(out))
    fn = types.FunctionType(trainer.fit_config.__code__, {**trainer.fit_config.__globals__, 'build_model': build_model},
                            trainer.fit_config.__name__, trainer.fit_config.__defaults__)
    fn.__kwdefaults__ = trainer.fit_config.__kwdefaults__.copy()
    length = config['args']['seq_len']
    views = {s: e1.view(common_view(windows[s], length), 'upstream') for s in ('train', 'validation')}
    return fn(config, recipe, fit=views['train'], selection=views['validation'], seed=seed, device=device, out=out, contract=contract)


def signature(w):
    # Canonical target identity, independent of the 24/48/72h start offset.
    h = hashlib.sha256()
    for a in (w.target_times(), w.turbines[w.window_starts], w.truth_and_anchor()[0]):
        h.update(np.ascontiguousarray(a).tobytes())
    return {'windows': len(w), 'targets_sha256': h.hexdigest()}


class Data:
    def __init__(self, root, protocol):
        self.protocol, self.identity = protocol, {}
        old, recipe = u.read_json(ROOT/'configs/n1ph_protocol.json'), u.read_json(p2.PROTOCOL)
        pf, pu, pm, kf, ku, km = n1.other_inputs(root, old, recipe, self.identity)
        self.frames = {'Kelmarsh': (kf, ku, km, '2016-05-01'), 'Penmanshiel': (pf, pu, pm, pm['start'])}

    def build(self, farm, evaluation=False):
        splits = self.protocol['folds'][farm]
        selected = splits if evaluation else {k:v for k,v in splits.items() if k != 'test'}
        frame, upstream, meta, start = self.frames[farm]
        last = max(y for ys in selected.values() for y in ys)
        series = u.KelmarshUpstreamSeries(frame[frame.label_year <= last], upstream, splits=selected, input_len=72,
                                         start=start, end=f'{last+1}-01-01T01:00:00Z', preparation=meta)
        windows = {k:series.windows(k) for k in selected}
        audit = {k:signature(v) for k,v in windows.items()}
        audit['scaler_mean'], audit['scaler_scale'] = series.scaler_mean.tolist(), series.scaler_scale.tolist()
        # Full train/validation features must remain identical when test rows are introduced later.
        rows = np.isin(series.label_year, selected['train']+selected['validation'])
        audit['development_values_sha256'] = hashlib.sha256(np.ascontiguousarray(series.values[rows]).tobytes()).hexdigest()
        return windows, audit


class NumericalTrialFailure(Exception): pass


def numerical(error):
    return isinstance(error, torch.OutOfMemoryError) or str(error).startswith(('Nonfinite training loss', 'Nonfinite or misaligned selection predictions'))


def checked_hashes(mapping, base=None):
    for name, digest in mapping.items():
        path = Path(name) if base is None else base/name
        if sha(path) != digest: raise ValueError('Frozen identity changed: '+str(path))


def select_trial(trials):
    rows = [r for r in trials if r['status']=='COMPLETE' and np.isfinite(r['value'])]
    if not rows: raise ValueError('No successful trial')
    return min(rows, key=lambda r:(r['value'], r['number']))


def metrics(prediction, truth):
    if prediction.shape != truth.shape or not np.isfinite(prediction).all(): raise ValueError('Invalid prediction')
    d = prediction-truth
    return {'mae':float(np.abs(d).mean()), 'rmse':float(np.sqrt(np.square(d).mean())),
            'r2':float(1-np.square(d).sum()/np.square(truth-truth.mean()).sum())}


def accept(reports):
    return all(r['TSMixerBasicSmall_tuned']['ensemble']['mae'] <= .98*r['TSMixerBasicSmall_existing']['ensemble']['mae']
               and r['TSMixerBasicSmall_tuned']['ensemble']['rmse'] <= r['TSMixerBasicSmall_existing']['ensemble']['rmse']
               for r in reports.values())


def source_hashes():
    paths = set(p2.source_code())
    for folder in ('scripts','models','layers','utils','data_provider'):
        paths.update(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/folder).glob('*.py'))
    paths.update(('configs/t1_tuning_protocol.json','configs/kelmarsh_upstream_protocol_v32.json',
                  'configs/n1ph_protocol.json','output/e1_baselines/a2/r1/hyperparameter_selection.json'))
    return {p:sha(ROOT/p) for p in sorted(paths)}


def run(root, out, protocol, device):
    import optuna
    if optuna.__version__ != protocol['optuna_version']: raise ValueError('Optuna version differs from installation receipt')
    data = Data(root, protocol)
    save(out/'input_identity.json', data.identity)
    sets, audits = {}, {}
    for farm in protocol['folds']: sets[farm], audits[farm] = data.build(farm)
    save(out/'cohorts_before_evaluation.json', audits)
    selected, alltrials = {}, {}
    def one(farm, cfg, params, seed, path, short=False):
        recipe = recipe_for(params, short)
        contract = {'stage':'M2-T1','farm':farm,'config':cfg,'training':recipe['training'], 'seed':seed,
                    'protocol_sha256':sha(PROTOCOL),'input_identity_sha256':sha(out/'input_identity.json'),
                    'launch_sha256':sha(out/'launch.json'),'cohort':audits[farm]}
        record = fit(cfg, recipe, sets[farm], seed, device, path, contract)
        return {'folder':str(path.relative_to(out)), 'config':cfg, 'record':record, 'farm':farm, 'seed':seed}
    for family in protocol['models']:
        trials = []
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=2024), pruner=optuna.pruners.NopPruner())
        def objective(trial):
            params = suggest(trial, family, protocol)
            cfg = config_for(family, params)
            entry = {'number':trial.number, 'params':params, 'config':cfg}
            try:
                record = one('Kelmarsh',cfg,params,2024,out/'search'/family/f'trial_{trial.number:02d}',True)
                entry.update(status='COMPLETE', value=record['record']['validation_best']['mae'], fit=record)
                if not np.isfinite(entry['value']): raise NumericalTrialFailure('Nonfinite objective')
            except Exception as exc:
                entry.update(status='FAIL',value=None,error=str(exc),error_type=type(exc).__name__)
                if numerical(exc) or isinstance(exc,NumericalTrialFailure): raise NumericalTrialFailure(str(exc)) from exc
                raise
            finally:
                trials.append(entry); save(out/'search'/family/'trials.json',trials)
                gc.collect()
                if device.type=='cuda': torch.cuda.empty_cache()
            return entry['value']
        study.optimize(objective,n_trials=protocol['trials_per_model'],catch=(NumericalTrialFailure,),gc_after_trial=True)
        if len(trials)!=60: raise ValueError('Trial budget mismatch')
        selected[family]=select_trial(trials); alltrials[family]=trials
        try: importance=optuna.importance.get_param_importances(study)
        except (ValueError, RuntimeError) as exc: importance={'unavailable':str(exc)}
        save(out/'search'/family/'importance.json',{'descriptive_only':True,'importance':importance})
    save(out/'tuning_freeze.json',{'utc':p2.now(),'selected':selected,'attempted_trials':180,
         'trial_metadata_sha256':{f:sha(out/'search'/f/'trials.json') for f in protocol['models']}})
    records=[]
    for farm in protocol['folds']:
        for family in protocol['models']:
            for kind in ('existing','tuned'):
                params = {} if kind=='existing' else selected[family]['params']
                cfg = config_for(family,params)
                for seed in protocol['seeds']:
                    entry=one(farm,cfg,params,seed,out/'fits'/farm/(family+'_'+kind)/f's{seed}')
                    entry['kind']=kind; records.append(entry)
    if len(records)!=protocol['formal_fits']: raise ValueError('Formal fit budget mismatch')
    for entry in records: trainer.verify_trained(out/entry['folder'],entry['record']['contract'])
    for rows in alltrials.values():
        for row in rows:
            if row['status']=='COMPLETE': trainer.verify_trained(out/row['fit']['folder'],row['fit']['record']['contract'])
    checked_hashes(data.identity)
    checked_hashes(u.read_json(out/'launch.json')['source_code'],ROOT)
    save(out/'training_freeze.json',{'records':records,'count':60,'tuning_sha256':sha(out/'tuning_freeze.json'),'utc':p2.now()})
    with (out/'evaluation_started.json').open('x') as f:
        json.dump({'utc':p2.now(),'training_freeze_sha256':sha(out/'training_freeze.json')},f)
    reports={}
    for farm in protocol['folds']:
        testsets, audit = data.build(farm,True)
        if {k:v for k,v in audit.items() if k!='test'} != audits[farm]: raise ValueError('Development cohort/statistics changed')
        windows=testsets['test']; truth=windows.truth_and_anchor()[0]
        dest=out/'evaluation'/farm; dest.mkdir(parents=True)
        np.savez_compressed(dest/'cohort.npz',truth=truth,times=windows.target_times(),turbines=windows.turbines[windows.window_starts])
        groups={}
        for entry in [r for r in records if r['farm']==farm]:
            cfg=entry['config']; model,_=build_model(cfg)
            ck=torch.load(out/entry['folder']/'best.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(ck['model'],strict=True); model.to(device)
            pred=trainer.v26run.predict(model,e1.view(common_view(windows,cfg['args']['seq_len']),'upstream'),entry['seed'],device,512)
            key=cfg['family']+'_'+entry['kind']; metrics(pred,truth)
            np.savez_compressed(dest/f'{key}_s{entry["seed"]}.npz',prediction=pred)
            groups.setdefault(key,[]).append(pred)
            del model,ck; gc.collect()
            if device.type=='cuda': torch.cuda.empty_cache()
        report={}
        for key,preds in groups.items():
            if len(preds)!=5: raise ValueError('Incomplete ensemble')
            ensemble=np.mean(preds,axis=0)
            np.savez_compressed(dest/f'{key}_ensemble.npz',prediction=ensemble)
            report[key]={'ensemble':metrics(ensemble,truth),'per_seed':[metrics(p,truth) for p in preds]}
        save(dest/'report.json',report); reports[farm]=report
    result={'status':'COMPLETED','decision':'ADOPT_TUNED_DEVELOPMENT_ONLY' if accept(reports) else 'RETAIN_EXISTING',
            'reports':reports,'attempted_search_trials':180,'formal_fits':60,'scope':protocol['scope'],
            'artifacts':{str(p.relative_to(out)):sha(p) for p in (out/'evaluation').rglob('*.npz')}}
    save(out/'result.json',result)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True); parser.add_argument('--device',default='cuda:0'); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    protocol=u.read_json(PROTOCOL); start=time.monotonic()
    save(args.out/'launch.json',{'utc':p2.now(),'argv':sys.argv,'source_code':source_hashes(),'protocol':protocol})
    try:
        if u.text_sha256_lf(p2.PROTOCOL)!=protocol['recipe_sha256_lf']: raise ValueError('Recipe identity changed')
        if args.smoke:
            from tests.test_tune_t1 import smoke
            smoke(args.out,torch.device(args.device))
        else: run(args.data_root,args.out,protocol,torch.device(args.device))
        save(args.out/'completed.json',{'utc':p2.now(),'seconds':time.monotonic()-start,'status':'COMPLETED'})
    except Exception as exc:
        save(args.out/'failed.json',{'utc':p2.now(),'seconds':time.monotonic()-start,'type':type(exc).__name__,'error':str(exc)}); raise


if __name__=='__main__': main()
