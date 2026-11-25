exec(open('/tmp/rc/recompute.py').read().split('# compare with results.json')[0].replace("print('local", "_=('local").replace("print('keys", "_=('keys"))
import numpy as np
# ---------- 1) Kelmarsh Cabin LU: seed / block-length sensitivity, per horizon
f = 'Kelmarsh'; k = keys[f]; y = k['truth'].astype(float); iss = k['issue']
c, h = own('cabin', f'{f}/LU/V2.npy'), own('lgbh', f'{f}/LU/V2.npy')
d = np.abs(h - y).mean(1) - np.abs(c - y).mean(1)
hi = [boot(d, iss, seed=s)[2] for s in range(50)]
print('K cabin LU gain %.3f; upper bound over 50 seeds: min %.2f median %.2f max %.2f; share upper<0: %.2f'
      % (d.mean(), min(hi), np.median(hi), max(hi), np.mean(np.array(hi) < 0)))
for L in (4, 16, 24, 56):
    print('  block', L, ['%.2f' % v for v in boot(d, iss, seed=2024, L=L)])
print('  per horizon gain:', np.round(np.abs(h - y).mean(0) - np.abs(c - y).mean(0), 2))
for i in ('L0', 'LU'):
    print('  %s bias cabin %.1f lgbh %.1f' % (i, (own('cabin', f'{f}/{i}/V2.npy') - y).mean(), (own('lgbh', f'{f}/{i}/V2.npy') - y).mean()))
# Cabin U - LGBM-hist U (descriptive, not registered)
for f2 in keys:
    y2, i2 = keys[f2]['truth'].astype(float), keys[f2]['issue']
    e = {(m, i): np.abs(own(m, f'{f2}/{i}/V2.npy') - y2).mean(1) for m in ('cabin', 'lgbh') for i in ('L0', 'LU')}
    dd = (e['cabin', 'L0'] - e['cabin', 'LU']) - (e['lgbh', 'L0'] - e['lgbh', 'LU'])
    print('  U_cabin - U_lgbh', f2, ['%.2f' % v for v in boot(dd, i2, seed=11)])
# ---------- 2) HoT TSG: where does U(6) < 0 come from?
f = 'HoT'; k = keys[f]; y = k['truth'].astype(float); iss = k['issue']; tur = k['turbines']
T = {(i, dl): own('tsg', f'{f}/{i}/V2@{dl}.npy') for i in ('LG', 'LUG') for dl in (6, 18)}
M = {(i, dl): x5a(f'X5/{f}/{i}_mix@{dl}/V2.npy') for i in ('LG', 'LUG') for dl in (6, 18)}
for name, P in (('TSG', T), ('LGBM mix', M)):
    mae = {kk: np.abs(v - y).mean() for kk, v in P.items()}
    print(name, 'MAE', {f'{a}@{b}': round(v, 2) for (a, b), v in mae.items()},
          'degradation 6->18: LG %.2f LUG %.2f' % (mae['LG', 18] - mae['LG', 6], mae['LUG', 18] - mae['LUG', 6]))
    for dl in (6, 18):
        print('   d%d U per horizon' % dl, np.round(np.abs(P['LG', dl] - y).mean(0) - np.abs(P['LUG', dl] - y).mean(0), 2),
              ' bias LG %.1f LUG %.1f' % ((P['LG', dl] - y).mean(), (P['LUG', dl] - y).mean()))
u6 = np.abs(T['LG', 6] - y).mean(1) - np.abs(T['LUG', 6] - y).mean(1)
per_t = [u6[tur == t].mean() for t in np.unique(tur)]
print('TSG HoT U(6) per turbine: negative %d/%d, range %.1f..%.1f' % (np.sum(np.array(per_t) < 0), len(per_t), min(per_t), max(per_t)))
# offset between TSG and LGBM-mix per arm, both delays
for i in ('LG', 'LUG'):
    print('  TSG - LGBM MAE', i, ['%.2f' % (np.abs(T[i, dl] - y).mean() - np.abs(M[i, dl] - y).mean()) for dl in (6, 18)])
# ---------- 3) LR candidates at K/P: what do the failed predictions look like?
import json
fr = json.load(open(W / 'tsg/v1_freeze.json'))
for f in ('Kelmarsh', 'Penmanshiel', 'HoT'):
    y1 = dict(np.load(W / f'tsg/keys/{f}_V1.npz'))['truth'].astype(float)
    rated = 2300.0 if f == 'HoT' else 2050.0
    for i in ('LG', 'LUG'):
        for ci in (0, 1):
            p = own('tsg', f'{f}/{i}/cand{ci}/V1.npy')
            print('  %s %s cand%d V1 MAE %.1f; share pred==0: %.2f, ==rated: %.2f, mean pred %.0f, mean truth %.0f'
                  % (f, i, ci, np.abs(p - y1).mean(), np.mean(p == 0), np.mean(np.isclose(p, rated)), p.mean(), y1.mean()))
