"""Read-only numerical audit of saved N0 artifacts; no model inference or fitting."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def audit(run,code):
    run,code=Path(run),Path(code)
    done=read(run/'completed.json')
    assert done['fits']==5 and done['source_unchanged']
    assert not (run/'failed.json').exists()
    for name,h in read(run/'launch.json')['source_sha256'].items():assert digest(code/name)==h,name
    inv=read(run/'inventory.json')
    assert len(inv['files'])==282 and len(inv['stations'])==47 and not inv['missing']
    for f in inv['files']:
        assert 2016<=f['year']<=2021
        assert digest(f['path'])==f['sha256'],f['path']
    artifacts=read(run/'artifact_hashes.json')
    for name,h in artifacts.items():assert digest(run/name)==h,name
    fits=list((run/'models').glob('*/trained.json'))
    assert len(fits)==5
    for p in fits:
        r=read(p);assert digest(p.parent/'selected.pt')==r['checkpoint_sha256']
        assert r['selection_year']==2020 and r['seed']==2024 and r['parameters']<=20000
        assert set(r['train_stations'])<=set(inv['split']['train'])
    count=0;gates={}
    for result_name,array_dir in [('result.json','full')]+[(f'scale_{s}.json',f'scale_{s}') for s in (7,14,28)]:
        r=read(run/result_name);own={}
        for sid,arms in r['per_station'].items():
            own[sid]={};reference=None
            for arm,entry in arms.items():
                with np.load(run/'arrays'/array_dir/f'{sid}_{arm}.npz',allow_pickle=False) as a:
                    y,p,t=a['truth'],a['prediction'],a['times']
                    assert np.isfinite(y).all() and np.isfinite(p).all()
                    assert (t.astype('datetime64[Y]')==np.datetime64('2021','Y')).all()
                    if reference is None:reference=(y.copy(),t.copy())
                    else:np.testing.assert_array_equal(y,reference[0]);np.testing.assert_array_equal(t,reference[1])
                    err=np.abs(p-y);own[sid][arm]=float(err.mean())
                    np.testing.assert_allclose(err.mean(),entry['mae'],rtol=1e-12)
                    np.testing.assert_allclose(err.mean(0),entry['lead_mae'],rtol=1e-12);count+=1
        kel=inv['split']['kel_holdout'];tr={}
        assert len(r['pairs'])==56
        for q in kel:
            rows=[p for p in r['pairs'] if p['target']==q]
            assert {p['source'] for p in rows}==set(kel)-{q}
            tr[q]={}
            for arm in ('H','O','PH'):
                gains=[p['mae']['L']-p['mae'][arm] for p in rows]
                np.testing.assert_allclose(gains,[p['gain'][arm] for p in rows],rtol=1e-12)
                tr[q][arm]=float(np.median(gains))
                np.testing.assert_allclose(tr[q][arm],r['transfer'][str(q)][arm]['gain'],rtol=1e-12)
        od=np.array([own[str(q)]['H']-own[str(q)]['O'] for q in kel])
        d=np.array([tr[q]['O']-tr[q]['H'] for q in kel]);go=np.array([tr[q]['O'] for q in kel])
        own_ok=np.median(od)>0 and np.mean(od>0)>=2/3
        tr_ok=np.median(d)>0 and np.mean(d>0)>=2/3 and np.median(go)>0
        assert bool(own_ok and tr_ok)==r['gate']['continue']
        assert bool(not (own_ok and tr_ok))==r['gate']['stop']
        gates[result_name]=r['gate']
    return {'status':'VERIFIED','inputs':len(inv['files']),'saved_prediction_arrays':count,
            'artifact_hashes':len(artifacts),'fits':len(fits),'new_fits':0,'new_predictions':0,
            'elapsed_seconds':done['elapsed'],'gates':gates}


if __name__=='__main__':
    print(json.dumps(audit(sys.argv[1],sys.argv[2]),indent=2))
