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

    def test_curves_and_ani_features(self):
        train, ev = make_pack(seed=1), make_pack(seed=2)
        curves = x2.fit_power_curves(train, minimum=5)
        self.assertEqual({c['source'] for c in curves.values()}, {'turbine'})
        pcl = x2.ani_features(ev, curves, 2050., 'PCL')
        ani = x2.ani_features(ev, curves, 2050., 'ANI')
        self.assertEqual(pcl.shape[1], 5)
        self.assertEqual(ani.shape[1], 8)
        bad = ev['gF_mask'][:, 5] == 0
        self.assertTrue(np.isnan(pcl[bad, 0]).all() and np.isnan(ani[bad, :]).all())
        ok = ~bad & (ev['gF_mask'][:, 6] > 0)
        np.testing.assert_allclose(ani[ok, 0], pcl[ok, 1] - pcl[ok, 0], rtol=1e-5, atol=1e-2)
        fin = np.isfinite(ani[:, 4:])
        self.assertTrue((ani[:, 4:][fin] >= 0).all() and (ani[:, 4:][fin] <= 2050.).all())

    def test_curve_fallbacks(self):
        train = make_pack(seed=3)
        train['turbines'] = np.where(np.arange(len(train['truth'])) < 3, 9, train['turbines'])  # turbine 9: 3 windows
        curves = x2.fit_power_curves(train, minimum=5)
        self.assertEqual(curves[9]['source'], 'pooled')
        empty = x2.fit_power_curves(train, minimum=10 ** 6)
        self.assertTrue(all(c['points'] is None for c in empty.values()))
        self.assertTrue(np.isnan(x2.ani_features(make_pack(seed=4), empty, 2050., 'ANI')).all())

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


class PartAEndToEnd(unittest.TestCase):
    """run_a_farm -> report with a ridge stand-in for LightGBM: bookkeeping, freeze-before-test, SHA checks."""

    def test_part_a_and_report(self):
        protocol = x2.read(x2.PROTOCOL)
        x1p = x2.read(ROOT / 'configs/x1_protocol.json')
        x1p = dict(x1p, datasets={'Syn_2022': dict(farm='Syn', rated_kw=2050, count=150, family='G1', year=2022)})
        bud = x2.budget(protocol, smoke=True)
        train, v1, v2 = make_pack(300, seed=5), make_pack(80, seed=6, start='2021-02-01T00'), \
            make_pack(80, seed=7, start='2021-08-01T00')
        tests = {'Syn_2022': make_pack(50, seed=8, start='2022-03-01T00')}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            x2.save(out / 'launch.json', dict(mode='SMOKE', commit='synthetic',
                    script_sha256=x2.digest(x2.__file__), protocol_sha256=x2.digest(x2.PROTOCOL)))
            store = x2.Store(out)
            x2.save_keys(store, 'A', 'Syn', dict(V1=v1, V2=v2, **tests))
            x2.run_a_farm(fake_lgb(), 'Syn', 2050., train, v1, v2, tests, protocol, x1p, store, bud, 2, out)
            x2.save(out / 'A' / 'registry.json', store.registry)
            x2.save(out / 'A' / 'completed.json', dict(status='COMPLETED',
                                                      registry_sha256=x2.digest(out / 'A' / 'registry.json')))
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
