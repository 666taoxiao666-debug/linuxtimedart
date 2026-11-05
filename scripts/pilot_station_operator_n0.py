"""N0 diagnostic: existing 2016--2021 station observations only; no farm labels.

One exclusive run, fixed seed, no resume. Metadata is exportable; arrays/checkpoints
remain server-side. Run --inventory first; --run is the single formal instance.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import warnings

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch import nn
from scripts import screen_kelmarsh_upstream_observations as screen
from utils import kelmarsh_upstream_v31 as upstream

YEARS = tuple(range(2016, 2022))
PEN_IDS = (150, 161, 177, 235, 982, 987, 1023, 1074, 16589, 24125, 30523)
ARMS = ('L', 'H', 'PH', 'O')
COLS = ('wind_speed', 'wind_direction', 'msl_pressure')
TOKEN_DIM = 22
BATCH = 4096


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def permitted_file(path, year):
    match = re.search(r'_qcv-1_(\d{4})\.csv$', Path(path).name)
    if year not in YEARS or match is None or int(match[1]) != year:
        raise ValueError('N0 forbids opening undeclared years: ' + str(path))


def split_stations(stations):
    kel = sorted(s['src_id'] for s in stations if s['region'] == 'Kelmarsh')
    pen = sorted(s['src_id'] for s in stations if s['region'] == 'Penmanshiel')
    held = kel[::5]
    return {'train': [s for s in kel if s not in held], 'kel_holdout': held, 'pen_holdout': pen}


def inventory(data_root):
    data_root = Path(data_root)
    kel = read(ROOT / 'configs/kelmarsh_upstream_protocol_v32.json')['data']['upstream']['stations']
    hashes = upstream.manifest_hashes(read(data_root / 'outputs/audits/upstream_screen/midas_v202607/download_manifest.json'))
    d2 = ROOT / 'output/kelmarsh_hourly/d2_feasibility_v34_r2'
    part, pen_hash = read(d2 / 'part_A.json'), read(d2 / 'input_hashes.json')
    meta_path = data_root / 'datasets/upstream_observations/midas_open_v202607/development/midas-open_uk-hourly-weather-obs_dv-202607_station-metadata.csv'
    meta = screen.read_badc_csv(meta_path)
    idcol = screen.column(meta, 'src_id')
    latcol = screen.column(meta, 'station_latitude', 'latitude', 'lat')
    loncol = screen.column(meta, 'station_longitude', 'longitude', 'lon')
    stations, files, missing = [], [], []
    for region, ids in [('Kelmarsh', sorted(s['src_id'] for s in kel)), ('Penmanshiel', PEN_IDS)]:
        for sid in ids:
            if region == 'Kelmarsh':
                station = dict(next(s for s in kel if s['src_id'] == sid))
            else:
                rows = meta[pd.to_numeric(meta[idcol], errors='coerce') == sid]
                coords = rows[[latcol, loncol]].drop_duplicates()
                if len(coords) != 1:
                    raise ValueError(f'Ambiguous coordinates for {sid}')
                station = {'src_id': sid, 'lat': float(coords.iloc[0, 0]), 'lon': float(coords.iloc[0, 1])}
            station['region'] = region
            entries = []
            for year in YEARS:
                if region == 'Kelmarsh':
                    folder = data_root / 'datasets/upstream_observations/midas_open_v202607/development' / str(sid)
                    paths = list(folder.glob(f'*_qcv-1_{year}.csv'))
                    expected = hashes.get(paths[0].name) if len(paths) == 1 else None
                else:
                    rec = next(s for s in part['midas']['stations'] if s['src_id'] == sid)
                    names = rec['years'][str(year)]['files']
                    paths = [data_root / 'datasets' / name.split('/datasets/', 1)[1] for name in names]
                    expected = pen_hash[names[0]]['sha256'] if len(names) == 1 else None
                if len(paths) != 1 or not paths[0].is_file():
                    missing.append({'src_id': sid, 'year': year, 'reason': 'missing_or_ambiguous_file'})
                    continue
                path = paths[0]
                permitted_file(path, year)  # before hashing or parsing
                digest = sha(path)
                if digest != expected:
                    raise ValueError('Registered SHA256 mismatch: ' + str(path))
                entries.append({'src_id': sid, 'year': year, 'path': str(path), 'sha256': digest})
            if len(entries) == len(YEARS):
                stations.append(station)
                files.extend(entries)
    split = split_stations(stations)
    return {'stations': stations, 'files': files, 'missing': missing, 'split': split,
            'metadata_sha256': sha(meta_path), 'years': list(YEARS)}


def geometry(q, s):
    a, b, c, d = np.radians([q['lat'], q['lon'], s['lat'], s['lon']])
    v = np.sin((c-a)/2)**2 + np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    dist = 12742 * np.arcsin(np.sqrt(np.clip(v, 0, 1)))
    bearing = np.degrees(np.arctan2(np.sin(d-b)*np.cos(c), np.cos(a)*np.sin(c)-np.sin(a)*np.cos(c)*np.cos(d-b))) % 360
    return float(dist), float(bearing)


def candidates(q, stations, allowed):
    result = []
    for s in stations:
        distance, bearing = geometry(q, s)
        if s['src_id'] != q['src_id'] and s['src_id'] in allowed and 20 <= distance <= 150:
            result.append({**s, 'distance_km': distance, 'bearing_deg': bearing})
    return sorted(result, key=lambda s: s['src_id'])


def tokens(times, observations, stations, padded):
    """Exact past timestamps, no forward fill. Direction represented as unit u/v."""
    n = len(times)
    out = np.zeros((n, padded, TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n, padded), dtype=bool)
    if not stations:
        return out, mask, np.ones(n, bool)
    a = np.stack([np.stack([observations[s['src_id']].reindex(times-pd.Timedelta(hours=k))[list(COLS)].to_numpy(float)
                           for s in stations], axis=1) for k in (1, 2, 3, 4)], axis=2)
    valid = np.isfinite(a).all(axis=(2, 3))
    # Calm wind direction is physically undefined but speed/pressure remain usable.
    calm = a[:, :, :, 0] == 0
    a[:, :, :, 1] = np.where(calm, 0, a[:, :, :, 1])
    valid = np.isfinite(a).all(axis=(2, 3))
    rad = np.radians(a[:, :, 0, 1])
    east = np.where(valid, -a[:, :, 0, 0]*np.sin(rad), 0).sum(1)
    north = np.where(valid, -a[:, :, 0, 0]*np.cos(rad), 0).sum(1)
    count = valid.sum(1)
    east /= np.maximum(count, 1)
    north /= np.maximum(count, 1)
    speed = np.hypot(east, north)
    undef = (speed <= 1e-8) | (count == 0)
    # Projection toward the upwind source (meteorological FROM direction).
    from_angle = np.where(undef, 0, np.arctan2(-east, -north))
    distance = np.array([s['distance_km'] for s in stations])[None, :]
    bearing = np.radians([s['bearing_deg'] for s in stations])[None, :]
    parallel = distance*np.cos(bearing-from_angle[:, None])
    transverse = distance*np.sin(bearing-from_angle[:, None])
    arrival = np.clip(parallel/np.maximum(speed[:, None], .5)/3.6, -12, 12)
    features = []
    for k in range(3):
        direction = np.radians(a[:, :, k, 1])
        features.extend([a[:, :, k, 0], np.sin(direction), np.cos(direction), a[:, :, k, 2]])
    features.extend([a[:, :, 0, 0]-a[:, :, 3, 0],
                     np.sin(np.radians(a[:, :, 0, 1]))-np.sin(np.radians(a[:, :, 3, 1])),
                     np.cos(np.radians(a[:, :, 0, 1]))-np.cos(np.radians(a[:, :, 3, 1])),
                     a[:, :, 0, 2]-a[:, :, 3, 2], parallel, transverse, arrival,
                     np.ones_like(parallel), np.broadcast_to(undef[:, None], parallel.shape),
                     (a[:, :, 0, 0] == 0)])
    z = np.stack(features, axis=-1)
    out[:, :len(stations)] = np.where(valid[:, :, None], z, 0)
    mask[:, :len(stations)] = valid
    return out, mask, undef


def station_arrays(q, observations, stations, allowed, padded):
    obs = observations[q['src_id']]
    times = pd.date_range('2016-01-01', '2022-01-01', freq='h', inclusive='left')
    y = np.column_stack([obs.wind_speed.reindex(times+pd.Timedelta(hours=h)).to_numpy() for h in range(1, 5)])
    # Full four-hour label belongs to same split year, including December edges.
    keep = np.isfinite(y).all(1) & ((times+pd.Timedelta(hours=4)).year == times.year)
    times, y = times[keep], y[keep].astype(np.float32)
    local = np.column_stack([obs.wind_speed.reindex(times-pd.Timedelta(hours=k)).to_numpy() for k in range(1, 25)])
    selected = candidates(q, stations, allowed)
    if selected:
        hand, _ = screen.upstream_features(times, observations, selected, lag=1)
    else:
        hand = np.full((len(times), 11), np.nan)
    z, mask, undef = tokens(times, observations, selected, padded)
    return {'L': local.astype(np.float32), 'H': hand.astype(np.float32), 'z': z, 'mask': mask,
            'y': y, 'year': times.year.to_numpy(), 'time': times.to_numpy(), 'undefined': undef,
            'candidates': [s['src_id'] for s in selected]}


class Operator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(TOKEN_DIM, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.weights = nn.Linear(32, 4)
        self.null_weight = nn.Parameter(torch.zeros(4))
        self.null_value = nn.Parameter(torch.zeros(4, 32))
        self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, mask):
        value = self.embed(x)
        weight = torch.nn.functional.softplus(self.weights(value))*mask[:, :, None]
        null = torch.nn.functional.softplus(self.null_weight)+1e-6
        total = (torch.einsum('bsh,bsd->bhd', weight, value)+null[None, :, None]*self.null_value[None])
        total = total/(weight.sum(1)+null)[:, :, None]
        return self.head(total).squeeze(-1)


def scaler(x):
    finite = np.isfinite(x)
    count = finite.sum(0)
    mean = np.where(finite, x, 0).sum(0)/np.maximum(count, 1)
    var = np.where(finite, (x-mean)**2, 0).sum(0)/np.maximum(count, 1)
    std = np.sqrt(var)
    return mean, np.where((count == 0) | (std < 1e-6), 1., std)


def transform(x, stats):
    mean, std = stats
    return ((np.where(np.isfinite(x), x, mean)-mean)/std).astype(np.float32)


def train_model(arrays, ids, kind, device, seconds, destination):
    torch.manual_seed(2024)
    rng = np.random.default_rng(2024)
    model = Operator() if kind == 'O' else nn.Sequential(nn.Linear(11, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 4))
    assert sum(p.numel() for p in model.parameters()) <= 20000
    model.to(device)
    # All normalization is computed exclusively on training target stations/years.
    raw = []
    for sid in ids:
        a = arrays[sid]; fit = a['year'] <= 2019
        raw.append(a['z'][fit][a['mask'][fit]] if kind == 'O' else a['H'][fit])
    stats = scaler(np.concatenate(raw)); del raw
    prepared = {}
    for sid in ids:
        a = arrays[sid]
        prepared[sid] = transform(a['z'] if kind == 'O' else a['H'], stats)
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    begin = time.monotonic(); history=[]; best=float('inf'); stale=0; chosen=None
    for epoch in range(1, 21):
        model.train()
        for sid in rng.permutation(ids):
            a=arrays[int(sid)]; indices=rng.permutation(np.flatnonzero(a['year'] <= 2019))
            for offset in range(0, len(indices), BATCH):
                idx=indices[offset:offset+BATCH]
                x=torch.from_numpy(prepared[int(sid)][idx]).to(device)
                pred=model(x,torch.from_numpy(a['mask'][idx]).to(device)) if kind=='O' else model(x)
                loss=((pred-torch.from_numpy(a['y'][idx]).to(device))**2).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                opt.zero_grad();loss.backward();opt.step()
                if time.monotonic()-begin >= seconds:break
            if time.monotonic()-begin >= seconds:break
        model.eval();error=0.;count=0
        with torch.no_grad():
            for sid in ids:
                a=arrays[sid];indices=np.flatnonzero(a['year']==2020)
                for offset in range(0,len(indices),BATCH):
                    idx=indices[offset:offset+BATCH];x=torch.from_numpy(prepared[sid][idx]).to(device)
                    pred=model(x,torch.from_numpy(a['mask'][idx]).to(device)) if kind=='O' else model(x)
                    error+=float(((pred-torch.from_numpy(a['y'][idx]).to(device))**2).sum());count+=len(idx)*4
        value=error/count;history.append({'epoch':epoch,'validation_mse':value})
        if value < best:
            best=value;chosen=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        if stale>=4 or time.monotonic()-begin >= seconds:break
    model.load_state_dict(chosen);model.eval()
    destination.mkdir(parents=True,exist_ok=True)
    torch.save({'state':chosen,'stats':stats},destination/'selected.pt')
    save(destination/'trained.json',{'kind':kind,'train_stations':ids,'parameters':sum(p.numel() for p in model.parameters()),
         'best_epoch':best_epoch,'history':history,'elapsed':time.monotonic()-begin,'checkpoint_sha256':sha(destination/'selected.pt'),
         'fit_years':[2016,2017,2018,2019],'selection_year':2020,'seed':2024})
    return model,stats


def outputs(model, stats, a, kind, device):
    x=transform(a['z'] if kind=='O' else a['H'],stats);result=[]
    with torch.no_grad():
        for start in range(0,len(x),BATCH):
            tensor=torch.from_numpy(x[start:start+BATCH]).to(device)
            pred=model(tensor,torch.from_numpy(a['mask'][start:start+BATCH]).to(device)) if kind=='O' else model(tensor)
            result.append(pred.cpu().numpy())
    return np.concatenate(result)


def ridge_fit(x,y,year):
    fit, val = year<=2019, year==2020
    stats=scaler(x[fit]);z=transform(x,stats).astype(float)
    z=np.column_stack([np.ones(len(z)),z]);xf=z[fit];yf=y[fit].astype(float)
    gram=xf.T@xf;rhs=xf.T@yf;penalty=np.eye(z.shape[1]);penalty[0,0]=0
    best=np.full(4,np.inf);coef=np.zeros((z.shape[1],4));lambdas=np.zeros(4)
    for alpha in (.1,1,10,100):
        c=np.linalg.solve(gram+alpha*penalty,rhs);loss=np.abs(z[val]@c-y[val]).mean(0)
        take=loss<best;best[take]=loss[take];coef[:,take]=c[:,take];lambdas[take]=alpha
    return {'coef':coef,'stats':stats,'lambda':lambdas}


def ridge_predict(model,x,target_stats):
    z=transform(x,target_stats).astype(float)
    return np.column_stack([np.ones(len(z)),z])@model['coef']


def retention(own,transferred):
    return float(transferred/own) if own >= .02 else None


def decision(own_delta, d, go, own_ph_gap, transfer_ph_gap):
    med=lambda a:float(np.median(a))
    positive=lambda a:float(np.mean(np.asarray(a)>0))
    own_ok=med(own_delta)>0 and positive(own_delta)>=2/3
    transfer_ok=med(d)>0 and positive(d)>=2/3 and med(go)>0
    proceed=own_ok and transfer_ok
    simple=proceed and med(own_ph_gap)<med(own_delta)/2 and med(transfer_ph_gap)<med(d)/2
    return {'own_pass':own_ok,'transfer_pass':transfer_ok,'continue':proceed,'stop':not proceed,'choose_simple':simple,
            'own_median':med(own_delta),'own_positive_fraction':positive(own_delta),'D_median':med(d),'D_positive_fraction':positive(d),
            'G_tr_O_median':med(go),'own_PH_minus_O_median':med(own_ph_gap),'transfer_O_minus_PH_median':med(transfer_ph_gap)}


def evaluate(arrays, holdouts, kel_ids, op, ph, device, folder):
    matrices={};models={};own={};by_station={}
    for sid in holdouts:
        a=arrays[sid];o=outputs(*op,a,'O',device);h=outputs(*ph,a,'PH',device)
        matrices[sid]={'L':a['L'],'H':np.c_[a['L'],a['H']],'O':np.c_[a['L'],o],'PH':np.c_[a['L'],h]}
        models[sid]={};own[sid]={};by_station[sid]={}
        test=a['year']==2021
        for arm in ARMS:
            x=matrices[sid][arm];m=ridge_fit(x,a['y'],a['year']);models[sid][arm]=m
            pred=ridge_predict(m,x[test],m['stats']);err=np.abs(pred-a['y'][test])
            own[sid][arm]=float(err.mean());by_station[sid][arm]={'mae':float(err.mean()),'lead_mae':err.mean(0).tolist(),'lambda':m['lambda'].tolist()}
            folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/f'{sid}_{arm}.npz',prediction=pred,truth=a['y'][test],times=a['time'][test],coef=m['coef'])
    transfer={q:{} for q in kel_ids};pairs=[]
    for q in kel_ids:
        test=arrays[q]['year']==2021;truth=arrays[q]['y'][test]
        for s in kel_ids:
            if s==q:continue
            mae={arm:float(np.abs(ridge_predict(models[s][arm],matrices[q][arm][test],models[q][arm]['stats'])-truth).mean()) for arm in ARMS}
            pairs.append({'source':s,'target':q,'mae':mae,'gain':{a:mae['L']-mae[a] for a in ARMS[1:]}})
        for arm in ARMS[1:]:
            tr=float(np.median([r['gain'][arm] for r in pairs if r['target']==q]));ow=own[q]['L']-own[q][arm]
            transfer[q][arm]={'gain':tr,'own_gain':ow,'loss':ow-tr,'retention':retention(ow,tr)}
    od=[own[q]['H']-own[q]['O'] for q in kel_ids]
    d=[transfer[q]['O']['gain']-transfer[q]['H']['gain'] for q in kel_ids]
    go=[transfer[q]['O']['gain'] for q in kel_ids]
    result=decision(od,d,go,[own[q]['PH']-own[q]['O'] for q in kel_ids],[transfer[q]['O']['gain']-transfer[q]['PH']['gain'] for q in kel_ids])
    result['G_tr_H_median']=float(np.median([transfer[q]['H']['gain'] for q in kel_ids]))
    result['retention']={a:{'median':float(np.median(v)) if v else None,'undefined':len(kel_ids)-len(v)} for a in ('H','O') for v in [[transfer[q][a]['retention'] for q in kel_ids if transfer[q][a]['retention'] is not None]]}
    return {'gate':result,'per_station':by_station,'transfer':transfer,'pairs':pairs,
            'penmanshiel_own_delta':{str(q):own[q]['H']-own[q]['O'] for q in holdouts if q not in kel_ids}}


def run(data_root,out,device):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'RUN_STARTED').open('x') as stream:stream.write(str(os.getpid()))
    import signal
    def timeout(*_):raise TimeoutError('N0 1.5h total budget exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(5400)
    started=time.monotonic();torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    source={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'tests/test_pilot_station_operator_n0.py',Path(screen.__file__),Path(upstream.__file__),ROOT/'docs/STAGE_PLAN.md']}
    save(out/'launch.json',{'utc':pd.Timestamp.now(tz='UTC').isoformat(),'pid':os.getpid(),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':source,'device':device})
    try:
        inv=inventory(data_root);save(out/'inventory.json',inv)
        split=inv['split'];train=split['train'];held=split['kel_holdout']+split['pen_holdout']
        if len(train)<14 or len(split['kel_holdout'])<2:raise ValueError('Insufficient stations for fixed N0 design')
        observations=upstream.load_station_observations(inv['files'],[s['src_id'] for s in inv['stations']])
        coverage={str(sid):{str(y):{'rows':int((f.index.year==y).sum()),
                  'finite_counts':{c:int(f.loc[f.index.year==y,c].notna().sum()) for c in COLS}}
                  for y in YEARS} for sid,f in observations.items()}
        save(out/'coverage.json',coverage)
        arrays={};padded=len(inv['stations'])
        for q in inv['stations']:
            allowed=set(train) if q['src_id'] in train else set(observations)
            a=station_arrays(q,observations,inv['stations'],allowed,padded)
            if any(not np.any(a['year']==y) for y in YEARS):raise ValueError(f"No eligible labels in a year at {q['src_id']}")
            arrays[q['src_id']]=a
            print('PREPARED',q['src_id'],len(a['y']),flush=True)
        save(out/'cohorts.json',{str(s):{'windows_by_year':{str(y):int((a['year']==y).sum()) for y in YEARS},'candidates':a['candidates'],'empty_set_rows':int((~a['mask'].any(1)).sum())} for s,a in arrays.items()})
        op=train_model(arrays,train,'O',device,900,out/'models/O');print('FIT O',flush=True)
        ph=train_model(arrays,train,'PH',device,900,out/'models/PH');print('FIT PH',flush=True)
        result=evaluate(arrays,held,split['kel_holdout'],op,ph,device,out/'arrays/full')
        save(out/'result.json',result);print('PRIMARY',json.dumps(result['gate']),flush=True)
        curves=[]
        for size in (7,14,len(train)):
            selected=train[:size];small=train_model(arrays,selected,'O',device,600,out/f'models/scale_{size}')
            r=evaluate(arrays,held,split['kel_holdout'],small,ph,device,out/f'arrays/scale_{size}')
            save(out/f'scale_{size}.json',r);curves.append({'train_stations':size,'gate_descriptive':r['gate']})
            print('SCALE',size,flush=True)
        save(out/'scale_curve.json',curves)
        if any(sha(ROOT/p)!=h for p,h in source.items()):raise ValueError('Source changed during run')
        save(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*')
             if p.is_file() and p.suffix in ('.npz','.pt')})
        save(out/'completed.json',{'elapsed':time.monotonic()-started,'source_unchanged':True,'fits':5,'status':'COMPLETED'})
        print('COMPLETE',flush=True)
    except BaseException as exc:
        save(out/'failed.json',{'type':type(exc).__name__,'message':str(exc),'elapsed':time.monotonic()-started});raise
    finally:
        signal.alarm(0)
        save(out/'export_sha256.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix=='.json' and p.name!='export_sha256.json'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--inventory',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.inventory==args.run:parser.error('Choose exactly one: --inventory or --run')
    if args.inventory:
        inv=inventory(args.data_root);save(args.out/'inventory_preflight.json',inv)
        print(json.dumps({'stations':len(inv['stations']),'files':len(inv['files']),'missing':inv['missing'],'split':inv['split']}))
    else:run(args.data_root,args.out,args.device)


if __name__=='__main__':main()
