"""X5 blocks 1-2: information-source value of upstream observations (U) and GFS (G), zero training.

Author: Claude (2026-11-21). Reads the frozen X2 part-A LightGBM predictions (output/x2/x2_0f24875/a/A, commit
0f24875, arms L0 / LU / LG / LUG, seed 2024 for all four so the 2x2 is matched) and the per-split keys
(truth, issue, times). Nothing is retrained and no new prediction is produced.

Definitions (registered in docs/STAGE_PLAN.md X5 before this script was first run):
  Block 1  exact two-player Shapley decomposition of the MAE reduction L0 -> LUG:
           phi_U = 0.5 * [(L0 - LU) + (LG - LUG)],  phi_G = 0.5 * [(L0 - LG) + (LU - LUG)],
           interaction I = (L0 - LU) - (LG - LUG)  (= upstream value without GFS minus with GFS; > 0 means GFS
           substitutes for upstream).  MAE in kW over all windows x horizons, also in % of MAE(L0).
  Block 2  residual upstream value per point u = |e_LG| - |e_LUG| (> 0: upstream helps on top of GFS).
           C1 disagreement D = |yhat_LG - yhat_LU| per (window, h), observable at issue time.  Per farm and h the
              80th percentile of D is fitted on V1 only; "high" = D > q80.  Reported on V2 (primary, dev) and on
              the test years (exploratory, those years were opened before).  Contrast = mean u(high) - mean u(rest)
              with an issue-time moving-block bootstrap CI; share of total u that falls in the high group.
           C2 GFS age = t0 - C(t0) = 6 + (issue hour mod 6) under the 6 h release rule; mean u per age.
  Statistics: block length 8 issue times, 2000 replicates, seed 2024 (descriptive throughout).
Output: output/x5/local_blocks12/results.json and SUMMARY.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'output/x2/x2_0f24875/a/A'
FARMS = {'Kelmarsh': ['Kelmarsh_2022'], 'Penmanshiel': ['Penmanshiel_2021'], 'HoT': ['HoT_2024', 'HoT_2025']}
ARMS = ('L0', 'LU', 'LG', 'LUG')
SEED = '2024'
BLOCK, REPS, BOOT_SEED, Q = 8, 2000, 2024, 0.8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(src, farm, split):
    keys = np.load(src / 'keys' / f'{farm}_{split}.npz')
    truth = keys['truth'].astype(np.float64)
    issue = keys['issue'].astype('datetime64[ns]')
    pred = {}
    for arm in ARMS:
        p = np.load(src / farm / arm / SEED / f'{split}.npy').astype(np.float64)
        if p.shape != truth.shape or not np.isfinite(p).all():
            raise ValueError(f'{farm}/{arm}/{split}: bad prediction array')
        pred[arm] = p
    return truth, issue, pred


def cluster_index(issue):
    """Sorted unique issue times and the cluster id of every window."""
    uniq, inv = np.unique(issue, return_inverse=True)
    return uniq, inv


def block_bootstrap(cluster_stats, stat_fn, rng):
    """Moving-block bootstrap over issue-time clusters. cluster_stats: [n_clusters, k] sums; stat_fn maps a
    column-sum vector to a scalar."""
    n = len(cluster_stats)
    block = min(BLOCK, n)
    starts = np.arange(n - block + 1)
    need = int(np.ceil(n / block))
    out = np.empty(REPS)
    for r in range(REPS):
        s = rng.choice(starts, need)
        idx = (s[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out[r] = stat_fn(cluster_stats[idx].sum(0))
    return np.percentile(out, [2.5, 97.5]), float(np.mean(out > 0))


def mean_gain_ci(gain_pts, inv, n_clusters, rng):
    """gain_pts [n, h] -> mean over points with CI (windows clustered by issue time)."""
    s = np.zeros((n_clusters, 2))
    np.add.at(s[:, 0], inv, gain_pts.sum(1))
    np.add.at(s[:, 1], inv, gain_pts.shape[1])
    ci, ppos = block_bootstrap(s, lambda v: v[0] / v[1], rng)
    return {'gain_kw': float(gain_pts.mean()), 'ci95': [float(ci[0]), float(ci[1])], 'p_positive': ppos}


def contrast_ci(u, high, inv, n_clusters, rng):
    """Mean u in high minus mean u in rest, clustered by issue time."""
    s = np.zeros((n_clusters, 4))
    np.add.at(s[:, 0], inv, (u * high).sum(1))
    np.add.at(s[:, 1], inv, high.sum(1))
    np.add.at(s[:, 2], inv, (u * ~high).sum(1))
    np.add.at(s[:, 3], inv, (~high).sum(1))
    f = lambda v: v[0] / max(v[1], 1) - v[2] / max(v[3], 1)  # noqa: E731
    ci, ppos = block_bootstrap(s, f, rng)
    return float(f(s.sum(0))), [float(ci[0]), float(ci[1])], ppos


def shapley(truth, pred):
    mae = {a: float(np.abs(truth - pred[a]).mean()) for a in ARMS}
    m = mae
    phi_u = 0.5 * ((m['L0'] - m['LU']) + (m['LG'] - m['LUG']))
    phi_g = 0.5 * ((m['L0'] - m['LG']) + (m['LU'] - m['LUG']))
    inter = (m['L0'] - m['LU']) - (m['LG'] - m['LUG'])
    pct = lambda x: 100 * x / m['L0']  # noqa: E731
    return {'mae': mae, 'phi_U_kw': phi_u, 'phi_G_kw': phi_g, 'interaction_kw': inter,
            'phi_U_pct': pct(phi_u), 'phi_G_pct': pct(phi_g), 'interaction_pct': pct(inter),
            'U_without_G_kw': m['L0'] - m['LU'], 'U_with_G_kw': m['LG'] - m['LUG'],
            'G_without_U_kw': m['L0'] - m['LG'], 'G_with_U_kw': m['LU'] - m['LUG'],
            'check_total_kw': [phi_u + phi_g, m['L0'] - m['LUG']]}


def analyse(src):
    rng = np.random.default_rng(BOOT_SEED)
    res = {'source': str(src.relative_to(ROOT)), 'seed': SEED, 'block_length': BLOCK, 'replicates': REPS,
           'quantile': Q, 'farms': {}}
    for farm, tests in FARMS.items():
        truth1, issue1, pred1 = load(src, farm, 'V1')
        d1 = np.abs(pred1['LG'] - pred1['LU'])
        q80 = np.quantile(d1, Q, axis=0)                       # per horizon, V1 only
        fr = {'q80_kw_by_h': q80.tolist(), 'splits': {}}
        for split in ['V1', 'V2'] + tests:
            truth, issue, pred = load(src, farm, split)
            uniq, inv = cluster_index(issue)
            n_cl = len(uniq)
            e = {a: np.abs(truth - pred[a]) for a in ARMS}
            u = e['LG'] - e['LUG']
            d = np.abs(pred['LG'] - pred['LU'])
            high = d > q80[None, :]
            row = {'role': 'threshold_fit' if split == 'V1' else ('dev_primary' if split == 'V2' else 'exploratory_test'),
                   'windows': int(len(truth)), 'issue_clusters': int(n_cl), 'shapley': shapley(truth, pred)}
            row['U_with_G'] = mean_gain_ci(u, inv, n_cl, rng)
            row['U_with_G_by_h_kw'] = u.mean(0).tolist()
            row['U_without_G_by_h_kw'] = (e['L0'] - e['LU']).mean(0).tolist()
            con, ci, ppos = contrast_ci(u, high, inv, n_cl, rng)
            tot = u.sum()
            row['C1_disagreement'] = {
                'high_share_points': float(high.mean()),
                'u_high_kw': float(u[high].mean()), 'u_rest_kw': float(u[~high].mean()),
                'contrast_kw': con, 'contrast_ci95': ci, 'contrast_p_positive': ppos,
                'share_of_total_u_in_high': float(u[high].sum() / tot) if tot != 0 else None,
                'u_high_rel_pct_of_LG_err': float(100 * u[high].sum() / e['LG'][high].sum()),
                'u_rest_rel_pct_of_LG_err': float(100 * u[~high].sum() / e['LG'][~high].sum()),
                'u_high_by_h_kw': [float(u[:, h][high[:, h]].mean()) for h in range(4)],
                'u_rest_by_h_kw': [float(u[:, h][~high[:, h]].mean()) for h in range(4)]}
            age = 6 + (issue.astype('datetime64[h]').astype(np.int64) % 24) % 6
            row['C2_gfs_age'] = {str(int(a)): {'windows': int((age == a).sum()),
                                              'u_with_G_kw': float(u[age == a].mean()),
                                              'G_with_U_kw': float((e['LU'] - e['LUG'])[age == a].mean())}
                                 for a in np.unique(age)}
            fr['splits'][split] = row
        res['farms'][farm] = fr
    return res


def summary(res):
    lines = ['# X5 blocks 1-2 (local, zero training; LightGBM X2 arms, seed 2024)', '',
             'Positive = MAE reduction in kW. V1 fits the D threshold; V2 is the primary development read; test years '
             'are exploratory (opened before). Descriptive statistics only.', '',
             '| farm | split | MAE L0/LU/LG/LUG | phi_U | phi_G | U without G | U with G [CI] | interaction |',
             '|---|---|---|---|---|---|---|---|']
    for farm, fr in res['farms'].items():
        for split, r in fr['splits'].items():
            s = r['shapley']
            m = s['mae']
            ci = r['U_with_G']['ci95']
            lines.append(f"| {farm} | {split} | {m['L0']:.1f}/{m['LU']:.1f}/{m['LG']:.1f}/{m['LUG']:.1f} | "
                         f"{s['phi_U_kw']:.2f} ({s['phi_U_pct']:.1f}%) | {s['phi_G_kw']:.2f} ({s['phi_G_pct']:.1f}%) | "
                         f"{s['U_without_G_kw']:.2f} | {s['U_with_G_kw']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | "
                         f"{s['interaction_kw']:.2f} |")
    lines += ['', '## C1: GFS-only vs upstream-only forecast disagreement (top 20% by V1 threshold)', '',
              '| farm | split | u high | u rest | contrast [CI] | share of u in high | u/err high | u/err rest |',
              '|---|---|---|---|---|---|---|---|']
    for farm, fr in res['farms'].items():
        for split, r in fr['splits'].items():
            c = r['C1_disagreement']
            lines.append(f"| {farm} | {split} | {c['u_high_kw']:.2f} | {c['u_rest_kw']:.2f} | {c['contrast_kw']:.2f} "
                         f"[{c['contrast_ci95'][0]:.2f}, {c['contrast_ci95'][1]:.2f}] | {c['share_of_total_u_in_high']:.2f} | "
                         f"{c['u_high_rel_pct_of_LG_err']:.1f}% | {c['u_rest_rel_pct_of_LG_err']:.1f}% |")
    lines += ['', '## C2: residual upstream value by GFS age (hours since the used cycle)', '']
    for farm, fr in res['farms'].items():
        for split, r in fr['splits'].items():
            cells = ', '.join(f"age {a}: u {v['u_with_G_kw']:.2f} (n {v['windows']})" for a, v in r['C2_gfs_age'].items())
            lines.append(f'- {farm} {split}: {cells}')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, default=SRC)
    ap.add_argument('--out', type=Path, default=ROOT / 'output/x5/local_blocks12')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = analyse(a.src)
    res['script_sha256'] = sha(__file__)
    (a.out / 'results.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (a.out / 'SUMMARY.md').write_text(summary(res), encoding='utf-8')
    print(summary(res))


if __name__ == '__main__':
    main()
