"""L0 A2: one normalization ablation; immutable E1 inputs, records and recipe."""
from pathlib import Path
import sys, types, copy, argparse, json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from scripts import train_e1_baselines as e
from models.iTransformer import Model as Inverted
u,save=e.u,e.save

class WithoutWindowNorm(Inverted):
    def forecast(self,x_enc,x_mark_enc,x_dec,x_mark_dec):
        n=x_enc.shape[-1]
        encoded=self.enc_embedding(x_enc,x_mark_enc)
        encoded,_=self.encoder(encoded,attn_mask=None)
        return self.projection(encoded).permute(0,2,1)[:,:,:n]

def build_model(cfg):
    if cfg['family']=='iTransformer' and not cfg['args']['use_norm']:
        model=WithoutWindowNorm(types.SimpleNamespace(**cfg['args'],enc_in=cfg['channels']))
        count=sum(p.numel() for p in model.parameters() if p.requires_grad)
        if count!=cfg['expected_parameters']: raise ValueError('Parameter identity changed')
        return model,count
    return e.build_model(cfg)

def choose(records):
    return min(records,key=lambda r:(r['record']['validation_best']['mae'],not r['config']['args']['use_norm'],r['config']['args']['d_model']))['config']

def scoped(fn,**changes):
    f=types.FunctionType(fn.__code__,{**fn.__globals__,**changes},fn.__name__,fn.__defaults__)
    f.__kwdefaults__=copy.copy(fn.__kwdefaults__);return f

def execute(root,out,p,device):
    recovery=u.read_json(ROOT/'configs/e1_baselines_protocol_eval_v1.json')
    source,_,_=e.checked_recovery_records(root,recovery)
    old=u.read_json(source/'selection_freeze.json')['fits']
    identity={};sets,recipe=e.load_data(root,p,identity)
    if identity!=u.read_json(source/'input_identity.json'):raise ValueError('E1 input identity changed')
    save(out/'input_identity.json',identity)
    records=[];selected={}
    def train(fid,cfg,seed,kind):
        folder=out/kind/fid/cfg['arm']/f'd{cfg["args"]["d_model"]}_s{seed}'
        contract={'stage':'L0-A2','config':cfg,'seed':seed,'fold':fid,'kind':kind,'protocol_sha256':u.file_sha256(ROOT/'configs/e1_baselines_protocol_a2.json'),'input_identity_sha256':u.file_sha256(out/'input_identity.json')}
        fit=scoped(e.fit,build_model=build_model)
        record=fit(cfg,recipe,sets[fid].windows('train'),sets[fid].windows('validation'),seed,device,folder,contract)
        r={'folder':str(folder),'config':cfg,'seed':seed,'fold':fid,'record':record,'origin':'new'};records.append(r);return r
    for family,channels in p['arms'].items():
        for channel in channels:
            arm=family+'_'+channel
            candidates=[r for r in old if r['folder'].startswith('search/') and r['config']['arm']==arm]
            if len(candidates)!=2:raise ValueError('Missing true search fits')
            for d in p['d_model_candidates']:
                cfg=e.configuration(p,family,channel,d);cfg['args']['use_norm']=False
                candidates.append(train('Kelmarsh',cfg,2024,'search'))
            selected[arm]=choose(candidates)
    save(out/'hyperparameter_selection.json',selected)
    formal=[]
    for fold in p['folds']:
        fid=fold['id']
        for arm,cfg in selected.items():
            for seed in p['seeds']:
                if cfg['args']['use_norm']:
                    matches=[r for r in old if r['folder'].startswith('fits/') and r['fold']==fid and r['seed']==seed and r['config']==cfg]
                    if len(matches)!=1:raise ValueError('True selected fit not found')
                    r=copy.deepcopy(matches[0]);r['folder']=str(source/r['folder']);r['origin']='E1';formal.append(r)
                else:formal.append(train(fid,cfg,seed,'fits'))
    if len(formal)!=36 or len(records)>44:raise ValueError('Budget violation')
    for r in records+formal:e.trainer.verify_trained(Path(r['folder']),r['record']['contract'])
    save(out/'selection_freeze.json',{'new_fits':records,'formal':formal,'selected':selected,'new_fit_count':len(records),'reused_formal':sum(r['origin']=='E1' for r in formal)})
    # Frozen E1 evaluator; only factory substituted, no shared module mutation.
    scoped(e.evaluate,build_model=build_model)(root,out,p,device,sets,selected,formal,out,recovery)
    result=u.read_json(out/'result.json');result['fits']=len(records);result['formal_count']=36;result['reused_formal']=sum(r['origin']=='E1' for r in formal)
    for fold in p['folds']:
        fid=fold['id'];w=sets[fid].windows('test');truth,hour=w.truth_and_anchor();last=w.last10_anchor()
        if not np.isfinite(last).all():raise ValueError('Last10 baseline has missing values on common cohort')
        result['reports'][fid]['persistence']={a:u.metrics(v,truth,2050) for a,v in [('PersistenceHour',hour),('PersistenceLast10',last)]}
    save(out/'result.json',result)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    p=u.read_json(ROOT/'configs/e1_baselines_protocol_a2.json')
    if a.out.resolve()!=(a.data_root/p['output_run']).resolve():raise ValueError('Output must be unique registered run')
    e.n1.checked(ROOT/'scripts/train_e1_baselines.py',p['e1_source_sha256'],{})
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    save(a.out/'launch.json',{'utc':e.p2.now(),'argv':sys.argv,'protocol':p,'source_sha256':u.file_sha256(__file__),'tests_sha256':u.file_sha256(ROOT/'tests/test_train_e1_baselines_a2.py')})
    try:execute(a.data_root,a.out,p,torch.device(a.device));save(a.out/'completed.json',{'utc':e.p2.now()})
    except Exception as err:save(a.out/'failed.json',{'utc':e.p2.now(),'type':type(err).__name__,'error':str(err)});raise
if __name__=='__main__':main()
