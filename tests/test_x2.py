"""Static tests for scripts/run_x2.py (numpy only; LightGBM replaced by a ridge stand-in, torch parts on the server)."""
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_x1 as x1  # noqa: E402
from scripts import run_x2 as x2  # noqa: E402


def make_pack(n_issue=120, units=3, start='2021-01-01T03', hot=False, seed=0, rated=2050.):
    rng = np.random.default_rng(seed)
    issue = np.datetime64(start, 'h') + (np.arange(n_issue) * 3).astype('timedelta64[h]')
    issue = np.repeat(issue, units)
    turb = np.tile(np.arange(units), n_issue)
    n = len(issue)
    gF = rng.uniform(2, 16, size=(n, 13)).astype(np.float32)
    mask = np.ones((n, 13), np.float32)
    mask[rng.random((n, 13)) < .03] = 0
    curve = lambda w: rated / (1 + np.exp(-(w - 9)))
    truth = (curve(gF[:, 6:10]) + rng.normal(0, 60, (n, 4))).clip(0, rated).astype(np.float32)
    anchor = (curve(gF[:, 5]) + rng.normal(0, 60, n)).clip(0, rated).astype(np.float32)
    x50 = rng.normal(size=(n, 24, 50)).astype(np.float32)
    pack = dict(x50=x50, truth=truth, anchor=anchor, gF=gF, gF_mask=mask, turbines=turb,
                issue=issue.astype('datetime64[ns]').astype(np.int64),
                times=(issue[:, None] + np.arange(1, 5).astype('timedelta64[h]')).astype('datetime64[ns]').astype(np.int64),
                power_mean=np.array(700.), power_scale=np.array(600.))
    if hot:
        pack['retrofit'] = np.ones(n, np.float32)
    return pack


CURVE = lambda w, rated=2050.: rated / (1 + np.exp(-(w - 9)))


def make_world(n_hours=1500, units=3, start='2021-01-01T01', seed=0, rated=2050., fit=('2021-01-01', '2021-02-15')):
    """Hourly sources (load_hourly layout): GFS history wind, per-turbine power, hour-end labels."""
    rng = np.random.default_rng(seed)
    hours = (np.datetime64(start, 'h') + np.arange(n_hours).astype('timedelta64[h]')).astype('datetime64[ns]')
    hist = rng.uniform(2, 16, n_hours)
    hist[rng.random(n_hours) < .03] = np.nan
    power, wind = {}, {}
    for u in range(units):
        w = (np.nan_to_num(hist, nan=9.) + rng.normal(0, .3, n_hours)).clip(0, None)   # nacelle wind
        p = (CURVE(w, rated) + rng.normal(0, 40, n_hours)).clip(0, rated)
        p[rng.random(n_hours) < .02] = np.nan
        power[u], wind[u] = p, w
    return dict(hours=hours, hist=hist, power=power, wind=wind, exclude=None, fit_bounds=list(fit), nwp_mean=8.,
                nwp_scale=3., wind_mean=8.5, wind_scale=3.5, identities={'synthetic': 'sha'})


def windows_from_world(src, first, last, stride=3, hot=False, seed=0):
    """Frozen-pack-like windows whose GFS history channel and labels come from the hourly sources."""
    rng = np.random.default_rng(seed)
    hist, hours, m, s = src['hist'], src['hours'], src['nwp_mean'], src['nwp_scale']
    rows = []
    for k in range(first, last, stride):
        for u, p in src['power'].items():
            if np.isfinite(p[k:k + 5]).all():
                rows.append((k, u))
    k = np.array([r[0] for r in rows])
    u = np.array([r[1] for r in rows])
    n = len(rows)
    x50 = rng.normal(size=(n, 24, 50)).astype(np.float32)
    h = hist[k[:, None] + np.arange(-23, 1)]
    x50[:, :, 20] = np.where(np.isfinite(h), (h - m) / s, 0)
    wind = np.stack([src['wind'][int(a)] for a in u])
    x50[:, :, 0] = (wind[np.arange(n)[:, None], k[:, None] + np.arange(-23, 1)] - src['wind_mean']) / src['wind_scale']
    g = hist[k[:, None] + np.arange(-5, 8)]
    gF = np.where(np.isfinite(g), g, 0).astype(np.float32)
    power = np.stack([src['power'][int(a)] for a in u])
    truth = power[np.arange(n)[:, None], k[:, None] + np.arange(1, 5)].astype(np.float32)
    pack = dict(x50=x50, truth=truth, anchor=power[np.arange(n), k].astype(np.float32), gF=gF,
                gF_mask=np.isfinite(g).astype(np.float32), turbines=u,
                issue=hours[k].astype(np.int64), times=hours[k[:, None] + np.arange(1, 5)].astype(np.int64),
                power_mean=np.array(700.), power_scale=np.array(600.))
    if hot:
        pack['retrofit'] = np.ones(n, np.float32)
    return pack


def fake_lgb():
    class Booster:
        def __init__(self, model_file=None, coef=None):
            self.coef = coef if coef is not None else json.loads(Path(model_file).read_text())

        def save_model(self, path, num_iteration=None):
            Path(path).write_text(json.dumps(self.coef))

        def predict(self, X):
            return np.nan_to_num(X) @ np.asarray(self.coef['w']) + self.coef['b']

    class LGBMRegressor:
        def __init__(self, **params):
            self.params = params

        def fit(self, X, y, eval_set=None, eval_metric=None, categorical_feature=None, callbacks=None):
            A = np.nan_to_num(X).astype(float)
            mu = A.mean(0)
            w = np.linalg.solve((A - mu).T @ (A - mu) + 10 * np.eye(A.shape[1]), (A - mu).T @ (y - y.mean()))
            self.booster_ = Booster(coef=dict(w=w.tolist(), b=float(y.mean() - mu @ w)))
            self.best_iteration_ = 7
            return self

        def predict(self, X):
            return self.booster_.predict(X)

    return types.SimpleNamespace(LGBMRegressor=LGBMRegressor, Booster=Booster, __version__='4.5.0',
                                 early_stopping=lambda rounds, verbose=False: None)


class Helpers(unittest.TestCase):
    def test_report_rejects_mixed_commit_protocol_and_smoke(self):
        expected = dict(commit='reviewed', script_sha256='script', protocol_sha256='protocol', mode='A')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x2.save(root / 'launch.json', expected)
            x2.check_part_identity(root, 'A', expected)
            for key in ('commit', 'script_sha256', 'protocol_sha256', 'mode'):
                with self.subTest(key=key):
                    x2.save(root / 'launch.json', dict(expected, **{key: 'SMOKE' if key == 'mode' else 'other'}))
                    with self.assertRaises(ValueError):
                        x2.check_part_identity(root, 'A', expected)

    def test_split_masks_disjoint_and_crossing_dropped(self):
        p = make_pack(n_issue=200, start='2021-06-29T00')
        m1, m2 = x2.split_masks(p, ['2021-01-01', '2021-07-01'], ['2021-07-01', '2022-01-01'])
        self.assertFalse((m1 & m2).any())
        t = p['times'].astype('datetime64[ns]') - np.timedelta64(1, 'h')
        cut = np.datetime64('2021-07-01')
        crossing = (t.min(1) < cut) & (t.max(1) >= cut)
        self.assertTrue(crossing.any())
        self.assertFalse((m1 | m2)[crossing].any())
        self.assertTrue((t[m1] < cut).all() and (t[m2] >= cut).all())

    def test_lug_equals_a1_features_and_info_sets(self):
        for hot in (False, True):
            p = make_pack(hot=hot)
            np.testing.assert_array_equal(x2.tree_matrix(p, 'LUG', hot), x1.tree_features(p, hot))
        self.assertEqual(len(x2.INFO['L0']), 9)
        self.assertFalse(set(x2.INFO['LU']) & set(x2.GFS))
        self.assertFalse(set(x2.INFO['LG']) & set(x2.UPSTREAM))
        self.assertEqual(sorted(x2.LOCAL + x2.UPSTREAM + x2.GFS), list(range(50)))

    def test_hourly_samples_fit_boundary_and_exclusion(self):
        src = make_world(n_hours=100, start='2021-01-01T01', fit=('2021-01-01', '2021-01-02'))
        src['hist'][:] = 5.
        ex = {0: np.zeros(100, bool)}
        ex[0][3] = True
        smp = x2.hourly_samples(src['hours'], src['hist'], src['power'], src['fit_bounds'], ex)
        fin = {u: np.isfinite(p[:24]) for u, p in src['power'].items()}     # labels 01:00 .. 24:00 are inside
        self.assertEqual(len(smp[1][0]), int(fin[1].sum()))                 # hour-end 2021-01-02T00 counts, T01 not
        self.assertEqual(len(smp[0][0]), int(fin[0].sum()) - int(fin[0][3]))  # works hour removed
        src['hist'][5] = np.nan
        smp2 = x2.hourly_samples(src['hours'], src['hist'], src['power'], src['fit_bounds'])
        self.assertEqual(len(smp2[1][0]), int(fin[1].sum()) - int(fin[1][5]))

    def test_curves_from_hourly_unaffected_by_window_duplication(self):
        src = make_world()
        train = windows_from_world(src, 30, 1000)
        dup = x2.subset(train, np.repeat(np.arange(len(train['truth'])), 16))   # every window 16 times
        cfg = dict(width=.5, minimum=30, min_bins=2)
        a, meta = x2.farm_curves(src, train, cfg)
        b, _ = x2.farm_curves(src, dup, cfg)
        for u in a:
            np.testing.assert_array_equal(a[u]['points'], b[u]['points'])
            self.assertEqual(a[u]['samples'], b[u]['samples'])
        self.assertEqual(meta['identities'], {'synthetic': 'sha'})
        self.assertTrue(meta['bound_windows'] > 0)

    def test_bind_hourly_rejects_mismatch(self):
        src = make_world()
        train = windows_from_world(src, 30, 600)
        x2.bind_hourly(train, src)
        bad = dict(train, x50=train['x50'].copy())
        bad['x50'][:, 5, 20] += .5
        with self.assertRaises(ValueError):
            x2.bind_hourly(bad, src)
        bad = dict(train, truth=train['truth'] + 5)
        with self.assertRaises(ValueError):
            x2.bind_hourly(bad, src)

    def test_curves_and_ani_features(self):
        src = make_world()
        curves, _ = x2.farm_curves(src, windows_from_world(src, 30, 700), dict(width=.5, minimum=5, min_bins=2))
        self.assertEqual({c['source'] for c in curves.values()}, {'turbine'})
        ev = windows_from_world(src, 800, 1400, seed=2)
        pcl = x2.ani_features(ev, curves, 2050., 'PCL')
        ani = x2.ani_features(ev, curves, 2050., 'ANI')
        self.assertEqual(pcl.shape[1], 5)
        self.assertEqual(ani.shape[1], 8)
        bad = ev['gF_mask'][:, 5] == 0
        self.assertTrue(bad.any() and np.isnan(pcl[bad, 0]).all() and np.isnan(ani[bad, :]).all())
        ok = ~bad & (ev['gF_mask'][:, 6] > 0)
        np.testing.assert_allclose(ani[ok, 0], pcl[ok, 1] - pcl[ok, 0], rtol=1e-5, atol=1e-2)
        fin = np.isfinite(ani[:, 4:])
        self.assertTrue((ani[:, 4:][fin] >= 0).all() and (ani[:, 4:][fin] <= 2050.).all())
        good = np.isfinite(pcl[:, 1])
        self.assertLess(np.abs(pcl[good, 1] - CURVE(ev['gF'][good, 6])).mean(), 80)   # curve recovered

    def test_curve_fallbacks(self):
        src = make_world()
        smp = x2.hourly_samples(src['hours'], src['hist'], src['power'], src['fit_bounds'])
        smp[9] = (smp[0][0][:3], smp[0][1][:3])                               # turbine 9: 3 hours only
        curves = x2.fit_power_curves(smp, minimum=5)
        self.assertEqual(curves[9]['source'], 'pooled')
        empty = x2.fit_power_curves(smp, minimum=10 ** 6)
        self.assertTrue(all(c['points'] is None for c in empty.values()))
        ev = windows_from_world(src, 800, 900)
        self.assertTrue(np.isnan(x2.ani_features(ev, empty, 2050., 'ANI')).all())

    def test_pooled_effect_and_gates(self):
        def s(gain, lo, hi, ref=100.):
            return dict(gain=gain, ci95_low=lo, ci95_high=hi, reference_mean_loss=ref)
        p = x2.pooled_effect({'K': s(2., 1., 3.), 'P': s(2., 1., 3.)})
        self.assertAlmostEqual(p['pooled_relative_gain'], .02)
        self.assertAlmostEqual(p['pooled_se'], (1 / x2.Z95) / 100 / np.sqrt(2))
        self.assertEqual(x2.pooled_effect({'K': s(1., 1., 1.)})['status'], 'TECHNICAL_FAILURE')
        good = {f: s(2., 1., 3.) for f in 'KPH'}
        self.assertEqual(x2.gate_decisions(good, good)['outcome'], 'GATES_1_2_PASS_RUN_GATE3')
        small = dict(good, H=s(.5, .1, .9))
        d = x2.gate_decisions(small, good)
        self.assertFalse(d['gate1'])
        self.assertEqual(d['outcome'], 'NEAR_PASS_REGISTER_BEFORE_ANY_ROUND')
        neg = dict(good, H=s(-1., -2., 0.))
        self.assertEqual(x2.gate_decisions(neg, good)['outcome'], 'NEGATIVE')
        self.assertEqual(x2.gate_decisions(good, {f: s(-1., -2., 0.) for f in 'KPH'})['outcome'],
                         'GATE1_ONLY_FEATURE_ENGINEERING')

    def test_select_base_margin(self):
        mae = {'LUG': {'K': 100., 'P': 100.}, 'CNN': {'K': 99.5, 'P': 99.5}, 'N_v1': {'K': 103., 'P': 103.}}
        self.assertEqual(x2.select_base(mae)['selected'], 'LUG')
        mae['CNN'] = {'K': 98., 'P': 98.}
        self.assertEqual(x2.select_base(mae)['selected'], 'CNN')


class VariantsA2(unittest.TestCase):
    def test_degenerate_statistics_cannot_be_reported_as_negative(self):
        stats = {'farm': dict(gain=0., reference_mean_loss=100., ci95_low=0., ci95_high=0.)}
        gates = x2.gate_decisions(stats, stats, z=x2.bonferroni_z(3))
        with self.assertRaisesRegex(ValueError, 'technical statistical failure'):
            x2.require_valid_variant_gates({'LUG_VN': dict(gates=gates)})
        x2.require_valid_variant_gates({'LUG_VN': dict(gates={'outcome': 'NEGATIVE'})})

    def ctx(self, src, train):
        cfg = dict(width=.5, minimum=5, min_bins=2)
        g, _ = x2.farm_curves(src, train, cfg)
        o = x2.fit_power_curves(x2.hourly_samples(src['hours'], src['wind'], src['power'], src['fit_bounds']), **cfg)
        return dict(g=g, o=o, **{k: src[k] for k in ('wind_mean', 'wind_scale', 'nwp_mean', 'nwp_scale')})

    def test_bind_checks_nacelle_wind(self):
        src = make_world()
        train = windows_from_world(src, 30, 600)
        x2.bind_hourly(train, src)
        bad = dict(train, x50=train['x50'].copy())
        bad['x50'][:, 3, 0] += .2
        with self.assertRaises(ValueError):
            x2.bind_hourly(bad, src)

    def test_observed_wind_and_variants(self):
        src = make_world()
        train, ev = windows_from_world(src, 30, 700), windows_from_world(src, 800, 1400, seed=3)
        ctx = self.ctx(src, train)
        obs, hist = x2.observed_wind(ev, ctx)
        k = np.searchsorted(src['hours'], ev['issue'].astype('datetime64[ns]'))
        np.testing.assert_allclose(obs[:, -1], [src['wind'][int(u)][i] for u, i in zip(ev['turbines'], k)], atol=1e-4)
        self.assertTrue(np.isnan(hist[ev['x50'][:, :, 20] == 0]).all())
        vn, vr, vb = (x2.variant_features(ev, ctx, 2050., kind) for kind in ('VN', 'VR', 'VB'))
        self.assertEqual((vn.shape[1], vr.shape[1], vb.shape[1]), (8, 4, 5))
        same = ev.copy()                                   # no forecast change -> P* = PC_obs(w_obs), dP* = 0
        same['gF'] = np.repeat(ev['gF'][:, 5:6], 13, axis=1)
        same['gF_mask'] = np.repeat(ev['gF_mask'][:, 5:6], 13, axis=1)
        v0 = x2.variant_features(same, ctx, 2050., 'VN')
        fin = np.isfinite(v0[:, 4])
        self.assertTrue(fin.any())
        np.testing.assert_allclose(v0[fin, 4:], 0, atol=1e-3)
        r0 = x2.variant_features(same, ctx, 2050., 'VR')
        fin = np.isfinite(r0[:, 0]) & (x2.curve_values(same, ctx['g'], [5])[:, 0] > 20.5)
        np.testing.assert_allclose(r0[fin, 0], same['anchor'][fin], rtol=1e-4)
        self.assertTrue(((vr[np.isfinite(vr)] >= 0) & (vr[np.isfinite(vr)] <= 2050)).all())
        b = vb[:, -1]
        diff = obs[:, -6:] - hist[:, -6:]
        enough = np.isfinite(diff).sum(1) >= 3
        np.testing.assert_allclose(b[enough], np.nanmean(diff[enough], 1), atol=1e-4)
        self.assertTrue(np.isnan(b[~enough]).all())

    def test_a2_round_and_report2(self):
        protocol = x2.read(x2.PROTOCOL)
        x1p = dict(x2.read(ROOT / 'configs/x1_protocol.json'),
                   datasets={'Syn_2022': dict(farm='Syn', rated_kw=2050, count=150, family='G1', year=2022)})
        bud = x2.budget(protocol, smoke=True)
        src = make_world(n_hours=3000, fit=('2021-01-01', '2021-03-01'))
        train = windows_from_world(src, 30, 1400)
        v1, v2 = windows_from_world(src, 1450, 1900, seed=6), windows_from_world(src, 1950, 2400, seed=7)
        tests = {'Syn_2022': windows_from_world(src, 2450, 2990, seed=8)}
        ref = protocol['part_a2']['reference']
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2, tempfile.TemporaryDirectory() as t3:
            r1, r2, rep = Path(t1), Path(t2), Path(t3)
            x2.save(r1 / 'launch.json', dict(mode='A', commit=ref['commit'], protocol_sha256=ref['protocol_sha256']))
            st1 = x2.Store(r1)
            x2.save_keys(st1, 'A', 'Syn', dict(V1=v1, V2=v2, **tests))
            ctx = self.ctx(src, train)
            x2.run_a_farm(fake_lgb(), 'Syn', 2050., train, v1, v2, tests, protocol, x1p, st1, bud, 2, r1, ctx['g'], {})
            x2.save(r1 / 'A' / 'registry.json', st1.registry)
            x2.save(r1 / 'A' / 'completed.json', dict(status='COMPLETED', registry_sha256=x2.digest(r1 / 'A' / 'registry.json')))
            ident = dict(commit='round2', script_sha256='s', protocol_sha256='p')
            x2.save(r2 / 'launch.json', dict(ident, mode='A2'))
            st2 = x2.Store(r2)
            x2.save_keys(st2, 'A2', 'Syn', dict(V1=v1, V2=v2, **tests))
            x2.run_a_farm(fake_lgb(), 'Syn', 2050., train, v1, v2, tests, protocol, x1p, st2, bud, 2, r2, ctx['g'], {},
                          part='A2', arms=protocol['part_a2']['arms'], ctx=ctx)
            x2.save(r2 / 'A2' / 'registry.json', st2.registry)
            x2.save(r2 / 'A2' / 'completed.json', dict(status='COMPLETED', registry_sha256=x2.digest(r2 / 'A2' / 'registry.json')))
            freeze = x2.read(r2 / 'A2' / 'Syn' / 'dev_freeze.json')
            self.assertEqual(set(freeze['arms']), {'LUG_VN', 'LUG_VR', 'LUG_VB'})
            self.assertFalse(any('Syn_2022' in k for k in freeze['predictions']))
            x2.save(rep / 'launch.json', dict(ident, mode='REPORT2'))
            args = types.SimpleNamespace(a_results=r1, a2_results=r2, skip_frozen_n=True, logs=rep)
            with patch.dict(sys.modules, {'fcntl': types.ModuleType('fcntl')}) if sys.platform == 'win32' else patch.dict({}):
                x2.run_report2(args, rep, protocol, x1p)
            dec = x2.read(rep / 'dev_decisions.json')
            self.assertEqual(dec['family'], 3)
            self.assertAlmostEqual(dec['z'], 2.3939797998185104, places=6)
            self.assertIn(dec['outcome'], ('VARIANT_PASS_RUN_GATE3', 'NEAR_PASS_ROUND3_NEEDS_REGISTRATION',
                                           'NEGATIVE_STOP_A_LINE'))
            res = x2.read(rep / 'results.json')
            self.assertEqual(res['dev_decisions_sha256'], x2.digest(rep / 'dev_decisions.json'))
            self.assertIn('LUG_VN', res['tests']['Syn_2022']['vs_LUG'])
            x2.save(r1 / 'launch.json', dict(mode='A', commit='other', protocol_sha256=ref['protocol_sha256']))
            with self.assertRaises(ValueError):                    # round-1 reference must be the pinned run
                x2.reference_round1(r1, protocol)


class PartAEndToEnd(unittest.TestCase):
    """run_a_farm -> report with a ridge stand-in for LightGBM: bookkeeping, freeze-before-test, SHA checks."""

    def test_part_a_and_report(self):
        protocol = x2.read(x2.PROTOCOL)
        x1p = x2.read(ROOT / 'configs/x1_protocol.json')
        x1p = dict(x1p, datasets={'Syn_2022': dict(farm='Syn', rated_kw=2050, count=150, family='G1', year=2022)})
        bud = x2.budget(protocol, smoke=True)
        src = make_world(n_hours=3000, fit=('2021-01-01', '2021-03-01'))
        train = windows_from_world(src, 30, 1400)
        v1, v2 = windows_from_world(src, 1450, 1900, seed=6), windows_from_world(src, 1950, 2400, seed=7)
        tests = {'Syn_2022': windows_from_world(src, 2450, 2990, seed=8)}
        curves, curve_meta = x2.farm_curves(src, train, protocol['part_a']['curve'])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            x2.save(out / 'launch.json', dict(mode='SMOKE', commit='synthetic',
                    script_sha256=x2.digest(x2.__file__), protocol_sha256=x2.digest(x2.PROTOCOL)))
            store = x2.Store(out)
            x2.save_keys(store, 'A', 'Syn', dict(V1=v1, V2=v2, **tests))
            x2.run_a_farm(fake_lgb(), 'Syn', 2050., train, v1, v2, tests, protocol, x1p, store, bud, 2, out,
                          curves, curve_meta)
            x2.save(out / 'A' / 'registry.json', store.registry)
            x2.save(out / 'A' / 'completed.json', dict(status='COMPLETED',
                                                      registry_sha256=x2.digest(out / 'A' / 'registry.json')))
            saved_curves = x2.read(out / 'A' / 'Syn' / 'curves.json')
            self.assertEqual(saved_curves['meta']['identities'], {'synthetic': 'sha'})
            self.assertEqual(saved_curves['meta']['fit_bounds'], src['fit_bounds'])
            freeze = x2.read(out / 'A' / 'Syn' / 'dev_freeze.json')
            self.assertEqual(set(freeze['arms']), set(protocol['part_a']['arms']))
            self.assertFalse(any('Syn_2022' in k for k in freeze['predictions']))   # no test prediction in freeze
            tp = x2.read(out / 'A' / 'Syn' / 'test_predictions.json')
            self.assertEqual(len(tp), len(protocol['part_a']['arms']))              # 1 seed each in smoke
            args = types.SimpleNamespace(mode='smoke', a_results=out, b_results=None, skip_frozen_n=True,
                                         logs=out)
            # Only the unused Linux lock dependency is shimmed on Windows.
            # No flock API is provided: accidental lock use still fails, and
            # the real paired statistics (including bootstrap) run unchanged.
            if sys.platform == 'win32':
                with patch.dict(sys.modules, {'fcntl': types.ModuleType('fcntl')}):
                    x2.run_report(args, out, protocol, x1p)
            else:
                x2.run_report(args, out, protocol, x1p)
            dec = x2.read(out / 'dev_decisions.json')
            self.assertIn(dec['part_a']['gates']['outcome'], ('GATES_1_2_PASS_RUN_GATE3', 'GATE1_ONLY_FEATURE_ENGINEERING',
                                                               'NEAR_PASS_REGISTER_BEFORE_ANY_ROUND', 'NEGATIVE',
                                                               'TECHNICAL_FAILURE'))
            res = x2.read(out / 'results.json')
            self.assertEqual(res['dev_decisions_sha256'], x2.digest(out / 'dev_decisions.json'))
            self.assertTrue(res['exploratory_only'])
            self.assertIn('LUG_ANI', res['tests']['Syn_2022']['vs_LUG'])
            # a tampered prediction must be refused
            target = out / 'A' / 'Syn' / 'LUG' / '2024' / 'V2.npy'
            np.save(target, np.zeros((len(v2['truth']), 4)))
            with self.assertRaises(ValueError):
                x2.Saved(out, 'A').preds('Syn', 'LUG', 'V2')


if __name__ == '__main__':
    unittest.main()
