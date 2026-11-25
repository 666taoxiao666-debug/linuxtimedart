exec(open('/tmp/rc/recompute.py').read().split('# compare with results.json')[0].replace("print('local", "_=('local").replace("print('keys", "_=('keys"))
import numpy as np
for f in ('Kelmarsh', 'Penmanshiel', 'HoT'):
    k = keys[f]; y = k['truth'].astype(float)
    h = np.asarray(k['issue']).astype('datetime64[ns]').astype('datetime64[h]').astype(np.int64)
    age = h - ((h - 6) // 6 * 6)
    root = age >= 8
    T = {i: own('tsg', f'{f}/{i}/V2@6.npy') for i in ('LG', 'LUG')}
    M = {i: x5a(f'X5/{f}/{i}_mix@6/V2.npy') for i in ('LG', 'LUG')}
    for name, P in (('TSG', T), ('LGBM', M)):
        e = {i: np.abs(P[i] - y).mean(1) for i in P}
        b = {i: (P[i] - y).mean(1) for i in P}
        print(f, name, 'U(6) age8-11 (root/nominal) %.2f  age6-7 %.2f | bias LUG-LG age8-11 %.1f age6-7 %.1f'
              % ((e['LG'] - e['LUG'])[root].mean(), (e['LG'] - e['LUG'])[~root].mean(),
                 (b['LUG'] - b['LG'])[root].mean(), (b['LUG'] - b['LG'])[~root].mean()), 'share root %.2f' % root.mean())
