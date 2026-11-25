"""Claude's independent recompute (own hashing, own block bootstrap; no repo statistics code)."""
import hashlib, json
from pathlib import Path
import numpy as np
R = Path('/sessions/funny-kind-mayer/mnt/TimeDARTFirst_G1')
W = R / 'output/wjb/wjb_1fe7ab6'
def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def rd(p): return json.loads(Path(p).read_text(encoding='utf-8'))
# --- identity
reg = {}
for m in ('krr', 'lgbh', 'cabin', 'tsg', 'report'):
    done = rd(W / m / 'completed.json')
    assert sha(W / m / 'registry.json') == done['registry_sha256'], m
    reg[m] = rd(W / m / 'registry.json')
    if 'v1_freeze_sha256' in done: assert sha(W / m / 'v1_freeze.json') == done['v1_freeze_sha256'], m
    assert rd(W / m / 'launch.json')['commit'].startswith('1fe7ab6') and not rd(W / m / 'launch.json')['smoke']
assert sha(W / 'report/results.json') == rd(W / 'report/completed.json')['results_sha256']
nchk = 0
for m in ('krr', 'lgbh', 'cabin', 'tsg'):
    for rel, h in reg[m].items():
        p = W / m / rel
        if p.suffix in ('.npy', '.npz') and p.exists():
            assert sha(p) == h, rel; nchk += 1
print('local arrays/keys hash-checked:', nchk)
x5d = R / 'output/x5/x5_e01cc6b/run'
assert sha(x5d / 'completed.json') == 'a4dc5269c70d5f69772028d82ee2ce77bd8fd3807d8b6ef3d7994199711e3e72'
x5reg = rd(x5d / 'X5/registry.json'); assert sha(x5d / 'X5/registry.json') == rd(x5d / 'completed.json')['registry_sha256']
x2d = R / 'output/x2/x2_0f24875/a/A'
assert sha(x2d / 'completed.json') == '1e8ed09c41f9d664817a26b6eb52784144f75e63c134e4ecde62c73bddef28b6'
x2reg = rd(x2d / 'registry.json'); assert sha(x2d / 'registry.json') == rd(x2d / 'completed.json')['registry_sha256']
def x5a(rel):
    assert sha(x5d / rel) == x5reg[rel], rel; return np.load(x5d / rel).astype(np.float64)
def x2a(rel):
    assert sha(x2d.parent / rel) == x2reg[rel], rel; return np.load(x2d.parent / rel).astype(np.float64)
def own(m, rel): return np.load(W / m / rel).astype(np.float64)
# --- keys
coh = {f: rd(x2d / f'{f}_cohorts.json')['keys']['V2'] for f in ('Kelmarsh', 'Penmanshiel', 'HoT')}
keys = {}
for f in coh:
    ks = [dict(np.load(W / m / f'keys/{f}_V2.npz')) for m in ('krr', 'lgbh', 'cabin', 'tsg')]
    for k in ks[1:]:
        for c in ('truth', 'issue', 'turbines'): assert np.array_equal(k[c], ks[0][c])
    kb = {c: hashlib.sha256(np.ascontiguousarray(ks[0][c]).tobytes()).hexdigest() for c in ('issue', 'turbines', 'truth')}
    assert kb == coh[f], f
    x2k = dict(np.load(x2d / f'keys/{f}_V2.npz'))
    assert np.array_equal(x2k['truth'], ks[0]['truth']) and np.array_equal(x2k['issue'], ks[0]['issue'])
    keys[f] = ks[0]
print('keys: 4 lines identical and equal to X2 part A V2 cohorts')
# --- own block bootstrap: clusters by issue time; non-circular moving blocks of 8 clusters; ratio of sums
def boot(d, issue, reps=2000, seed=7, L=8):
    t = np.asarray(issue).astype('datetime64[ns]').astype(np.int64)
    u, inv = np.unique(t, return_inverse=True)
    s = np.bincount(inv, weights=d); c = np.bincount(inv).astype(float)
    n = len(u); nb = int(np.ceil(n / L)); starts = np.arange(n - L + 1)
    rng = np.random.default_rng(seed)
    cs, cc = np.concatenate([[0], np.cumsum(s)]), np.concatenate([[0], np.cumsum(c)])
    st = rng.choice(starts, size=(reps, nb))
    S = (cs[st + L] - cs[st]).sum(1); C = (cc[st + L] - cc[st]).sum(1)
    est = S / C
    return float(d.mean()), float(np.quantile(est, .025)), float(np.quantile(est, .975))
def gain(cand, ref, y, issue, **kw):          # MAE(ref) - MAE(cand), window-uniform
    d = np.abs(ref - y).mean(1) - np.abs(cand - y).mean(1)
    return boot(d, issue, **kw)
def rise(lg6, lug6, lg18, lug18, y, issue, **kw):
    e = [np.abs(p - y).mean(1) for p in (lg6, lug6, lg18, lug18)]
    d = (e[2] - e[3]) - (e[0] - e[1])
    return boot(d, issue, **kw)
out = {}
for f, k in keys.items():
    y, iss = k['truth'].astype(np.float64), k['issue']
    o = out[f] = {}
    for d in (6, 18):
        for i in ('LG', 'LUG'):
            c, r = own('krr', f'{f}/{i}_d{d}/V2.npy'), x5a(f'D{d}/{f}/{i}_d{d}/2024/V2.npy')
            o[f'krr {i} d{d}'] = gain(c, r, y, iss)
            c, r = own('tsg', f'{f}/{i}/V2@{d}.npy'), x5a(f'X5/{f}/{i}_mix@{d}/V2.npy')
            o[f'tsg {i} d{d}'] = gain(c, r, y, iss)
    for i in ('L0', 'LU'):
        o[f'cabin {i}'] = gain(own('cabin', f'{f}/{i}/V2.npy'), own('lgbh', f'{f}/{i}/V2.npy'), y, iss)
    K = {(i, d): own('krr', f'{f}/{i}_d{d}/V2.npy') for i in ('LG', 'LUG') for d in (6, 18)}
    T = {(i, d): own('tsg', f'{f}/{i}/V2@{d}.npy') for i in ('LG', 'LUG') for d in (6, 18)}
    for name, P in (('krr', K), ('tsg', T)):
        for d in (6, 18):
            o[f'U {name} d{d}'] = gain(P['LUG', d], P['LG', d], y, iss)
        o[f'rise {name}'] = rise(P['LG', 6], P['LUG', 6], P['LG', 18], P['LUG', 18], y, iss)
# compare with results.json
res = rd(W / 'report/results.json')['farms']
def theirs(f, key):
    r = res[f]; p = key.split()
    if p[0] in ('krr', 'tsg') and p[1] in ('LG', 'LUG'):
        return r[p[0]][p[2]][p[1]][f'{p[0]}_vs_x5_lightgbm' if p[0] == 'krr' else 'tsg_vs_x5_mix_lightgbm']
    if p[0] == 'cabin': return r['cabin'][p[1]]['cabin_vs_lightgbm_hist']
    if p[0] == 'U': return r[p[1]][p[2]][f'U_{p[1]}']
    if p[0] == 'rise': return r[p[1]][f'rise_{p[1]}']
mx_pt, mx_ci = 0, 0
for f, o in out.items():
    for key, (g, lo, hi) in o.items():
        t = theirs(f, key)
        mx_pt = max(mx_pt, abs(g - t['gain'])); mx_ci = max(mx_ci, abs(lo - t['ci95_low']), abs(hi - t['ci95_high']))
        flag = ''
        if (lo > 0) != (t['ci95_low'] > 0) or (hi < 0) != (t['ci95_high'] < 0): flag = '  <-- CI class differs'
        print(f'{f:12s} {key:16s} mine {g:8.2f} [{lo:7.2f}, {hi:7.2f}]  theirs {t["gain"]:8.2f} [{t["ci95_low"]:7.2f}, {t["ci95_high"]:7.2f}]{flag}')
print('max |point diff|', mx_pt, ' max |CI endpoint diff| (different RNG/impl)', mx_ci)
