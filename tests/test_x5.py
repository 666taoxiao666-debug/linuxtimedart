"""Static tests for scripts/run_x5.py (numpy only; the LightGBM fits run on the server)."""
import unittest
import sys
from contextlib import nullcontext
from types import ModuleType
from unittest.mock import patch
from pathlib import Path

import numpy as np

from scripts import run_x1 as x1
from scripts import run_x2 as x2
from scripts import run_x5 as x5
from scripts import prepare_g1_batches as prep

MEAN, SCALE = 4.65, 2.2


def synthetic_cache(start='2021-03-01T00', days=6, nsites=4, seed=0, drop=()):
    rng = np.random.default_rng(seed)
    cycles = np.arange(np.datetime64(start, 'h'), np.datetime64(start, 'h') + np.timedelta64(24 * days, 'h'),
                       np.timedelta64(6, 'h'))
    cache = {}
    for c in cycles:
        key = x5.cycle_key(c)
        if key in drop:
            continue
        cache[key] = rng.uniform(1, 15, (6, nsites))
    return cache


def issues(start='2021-03-02T00', hours=72, stride=1):
    t = np.arange(np.datetime64(start, 'h'), np.datetime64(start, 'h') + np.timedelta64(hours, 'h'),
                  np.timedelta64(stride, 'h'))
    return np.repeat(t, 3).astype('datetime64[ns]').astype(np.int64)          # 3 turbines share an issue


class DelayedGFS(unittest.TestCase):
    def test_nominal_cycle_rule_equals_preparation(self):
        t = np.arange(np.datetime64('2021-01-01T00', 'h'), np.datetime64('2021-01-03T00', 'h'))
        np.testing.assert_array_equal(x5.delayed_cycle(t, 6), prep.available_cycle(t))

    def test_age_range_and_leads(self):
        cache = synthetic_cache()
        t = np.unique(issues().astype('datetime64[ns]').astype('datetime64[h]'))
        for d in (6, 9, 12, 15, 18):
            hourly, info = x5.delayed_hourly(t, cache, d, MEAN, SCALE, '2024-01-01', 4)
            self.assertEqual((info['age_min'], info['age_max']), (d, d + 5))
            lead = hourly['age'][:, None] + np.arange(-5, 8)[None, :]
            m = hourly['masks'][:, :, 0] > 0
            self.assertTrue(((lead[m] >= 3) & (lead[m] <= 18)).all())        # never outside [3, 18] h
            if d >= 18:
                self.assertFalse(m[:, 6:10].any())                          # t0+1..t0+4 unavailable at d = 18

    def test_nominal_block_reproduces_cached_hourly_and_nwp_input(self):
        """d = 6 must give exactly what run_x1.cached_hourly + prepare_g1_batches.nwp_input give (frozen path)."""
        cache = synthetic_cache(drop={'2021030312'})                       # one missing cycle
        iss = issues()
        block, _ = x5.gfs_block(iss, cache, 6, MEAN, SCALE, '2024-01-01', 4)
        hours = np.unique(iss.astype('datetime64[ns]').astype('datetime64[h]'))
        hourly = x1.cached_hourly(hours, cache, dict(nwp_mean=MEAN, nwp_scale=SCALE))
        idx = np.searchsorted(hours, iss.astype('datetime64[ns]').astype('datetime64[h]'))
        x21 = np.zeros((len(iss), 1, 2), np.float32)
        ref = prep.nwp_input(x21, hourly, idx[:, None])[:, 0, 1:30]
        np.testing.assert_array_equal(block, ref)
        self.assertEqual(block.shape, (len(iss), 29))

    def test_cutoff_cycles_never_read(self):
        cache = synthetic_cache(start='2023-12-30T00', days=3)
        t = np.array(['2023-12-31T20'], dtype='datetime64[h]')
        hourly, _ = x5.delayed_hourly(t, cache, 6, MEAN, SCALE, '2024-01-01', 4)
        self.assertEqual(hourly['cycle_ok'][0], 0.0)                         # cycle 12Z + 18 h reaches 2024
        hourly, _ = x5.delayed_hourly(t, cache, 18, MEAN, SCALE, '2024-01-01', 4)
        self.assertEqual(hourly['cycle_ok'][0], 1.0)                         # an older cycle is development

    def test_set_gfs_only_touches_last_step_gfs(self):
        rng = np.random.default_rng(1)
        pack = dict(x50=rng.normal(size=(5, 24, 50)).astype(np.float32), turbines=np.arange(5), truth=np.zeros((5, 4)))
        before = pack['x50'].copy()
        block = rng.normal(size=(5, 29)).astype(np.float32)
        x5.set_gfs({'s': pack}, {'s': block})
        np.testing.assert_array_equal(pack['x50'][:, -1, 20:49], block)
        mask = np.ones_like(before, bool)
        mask[:, -1, 20:49] = False
        np.testing.assert_array_equal(pack['x50'][mask], before[mask])
        with self.assertRaises(ValueError):
            x5.set_gfs({'s': pack}, {'s': block[:4]})

    def test_tree_columns_change_only_through_gfs(self):
        rng = np.random.default_rng(2)
        pack = dict(x50=rng.normal(size=(6, 24, 50)).astype(np.float32), turbines=np.arange(6))
        lu = x2.tree_matrix(pack, 'LU', False)
        x5.set_gfs({'s': pack}, {'s': rng.normal(size=(6, 29)).astype(np.float32)})
        np.testing.assert_array_equal(x2.tree_matrix(pack, 'LU', False), lu)   # LU never sees GFS

    def test_mix_assignment_and_block(self):
        delays = [6, 9, 12, 15, 18]
        a = x5.mix_assignment(10000, delays, np.random.default_rng(2024))
        self.assertEqual(set(a.tolist()), set(delays))
        np.testing.assert_array_equal(a, x5.mix_assignment(10000, delays, np.random.default_rng(2024)))
        blocks = {d: np.full((10000, 29), float(d), np.float32) for d in delays}
        np.testing.assert_array_equal(x5.mixed_block(blocks, a)[:, 0], a.astype(np.float32))

    def test_rise_losses(self):
        rng = np.random.default_rng(3)
        lg6, lug6, lg18, lug18 = (rng.uniform(0, 100, (50, 4)) for _ in range(4))
        cand, ref = x5.rise_losses(lg6, lug6, lg18, lug18)
        self.assertAlmostEqual((ref - cand).mean(), (lg18 - lug18).mean() - (lg6 - lug6).mean(), places=9)


def statistics_imports():
    # The pure statistics functions share a module with Linux file locking.
    return patch.dict(sys.modules, {'fcntl': ModuleType('fcntl')}) if sys.platform == 'win32' else nullcontext()


class Report(unittest.TestCase):
    def test_trend_uses_average_ranks(self):
        ds = [6, 9, 12, 15, 18]
        self.assertIsNone(x5.spearman(ds, [2] * 5))
        expected = np.corrcoef(range(5), [0.5, 0.5, 2, 3, 4])[0, 1]
        self.assertAlmostEqual(x5.spearman(ds, [1, 1, 2, 3, 4]), expected)

    def test_report_reads_every_registered_path(self):
        import tempfile
        from pathlib import Path
        protocol = x1.read(x5.PROTOCOL)
        delays = [6, 12]
        rng = np.random.default_rng(4)
        n = 400
        truth = rng.uniform(0, 2000, (n, 4))
        issue = np.repeat(np.arange(np.datetime64('2021-07-01T00', 'h'), np.datetime64('2021-07-01T00', 'h')
                                    + np.timedelta64(n // 2 * 3, 'h'), np.timedelta64(3, 'h')), 2)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            put = lambda rel, a: (out / rel).parent.mkdir(parents=True, exist_ok=True) or np.save(out / rel, a)  # noqa: E731
            noisy = lambda s: truth + rng.normal(0, s, truth.shape)  # noqa: E731
            put('D6/F/LU/2024/V2.npy', noisy(200))
            for d in delays:
                put(f'D{d}/F/LG_d{d}/2024/V2.npy', noisy(150 + d))
                put(f'D{d}/F/LUG_d{d}/2024/V2.npy', noisy(140))
                for name in ('LG_d6', 'LUG_d6', 'LG_mix', 'LUG_mix'):
                    put(f'X5/F/{name}@{d}/V2.npy', noisy(160))
            keys = {'F': {'V2': dict(truth=truth, issue=issue.astype('datetime64[ns]').astype(np.int64))}}
            with statistics_imports():
                res = x5.report(out, keys, delays, protocol, ['F'])
            row = res['farms']['F']['V2']
            self.assertEqual(sorted(row['delays']), ['12', '6'])
            u6, u12 = (row['delays'][d]['U_matched']['gain'] for d in ('6', '12'))
            self.assertAlmostEqual(row['U_rise']['gain'], u12 - u6, places=6)
            self.assertIn(res['H_latency']['outcome'], ('SUPPORTED', 'NOT_SUPPORTED'))
            self.assertIn('| F | V2 | 6 |', x5.summary_md(res))


class _FakeBoosterModel:
    def __init__(self, value):
        self.value = value

    def save_model(self, path, num_iteration=None):
        Path(path).write_text(repr(self.value))


class _FakeRegressor:
    def __init__(self, **params):
        self.params = params

    def fit(self, X, y, eval_set=None, eval_metric=None, categorical_feature=None, callbacks=None):
        self.ncol = X.shape[1]
        self.coef = float(np.mean(y)) + 1e-3 * float(np.mean(X[:, 0]))
        self.best_iteration_ = 3
        self.booster_ = _FakeBoosterModel([self.coef, self.ncol])
        return self

    def predict(self, X):
        if X.shape[1] != self.ncol:
            raise ValueError('column count differs')
        return np.full(len(X), self.coef) + 1e-3 * X[:, 1]


class _FakeBooster:
    def __init__(self, model_file):
        self.coef, self.ncol = eval(Path(model_file).read_text())

    def predict(self, X):
        if X.shape[1] != self.ncol:
            raise ValueError('column count differs')
        return np.full(len(X), self.coef) + 1e-3 * X[:, 1]


class _FakeLGB:
    __version__ = '4.5.0'
    LGBMRegressor, Booster = _FakeRegressor, _FakeBooster

    @staticmethod
    def early_stopping(rounds, verbose=False):
        return None


class EndToEnd(unittest.TestCase):
    """run_farm with mocked data loading and a fake LightGBM: exercises every path name and the binding gate."""

    def test_last_farm_preflight_failure_precedes_all_fits(self):
        import tempfile
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=tmp)
            with patch.dict(sys.modules, {'lightgbm': _FakeLGB}), patch.object(x5, 'start_run', return_value=Path(tmp)), \
                    patch.object(x2, 'farms', return_value={'K': {}, 'P': {}, 'H': {}}), \
                    patch.object(x5, 'prepare_farm', side_effect=[None, None, ValueError('last farm binding')]) as prep_all, \
                    patch.object(x5, 'run_farm') as fit:
                with self.assertRaisesRegex(ValueError, 'last farm binding'):
                    x5.run(args, 'RUN')
                self.assertEqual([call.args[2] for call in prep_all.call_args_list], ['K', 'P', 'H'])
                fit.assert_not_called()

    def _packs(self, cache, n, start):
        rng = np.random.default_rng(len(start))
        iss = np.arange(np.datetime64(start, 'h'), np.datetime64(start, 'h') + np.timedelta64(3 * n, 'h'),
                        np.timedelta64(3, 'h'))[:n].astype('datetime64[ns]').astype(np.int64)
        x50 = rng.normal(size=(n, 24, 50)).astype(np.float32)
        x50[:, -1, 20:49], _ = x5.gfs_block(iss, cache, 6, MEAN, SCALE, '2024-01-01', 4)
        times = iss[:, None] + (np.arange(1, 5) * 3600 * 10**9)[None, :]
        return dict(x50=x50, truth=rng.uniform(0, 2000, (n, 4)).astype(np.float32), issue=iss,
                    turbines=np.ones(n, np.int16), times=times, anchor=np.zeros(n, np.float32))

    def test_run_farm_end_to_end(self):
        import tempfile
        from pathlib import Path
        from unittest import mock
        cache = synthetic_cache(start='2021-02-25T00', days=40)
        train, v1, v2 = self._packs(cache, 120, '2021-03-01T00'), self._packs(cache, 60, '2021-03-20T00'), \
            self._packs(cache, 60, '2021-03-28T00')
        test = self._packs(cache, 60, '2021-03-10T00')
        protocol = x1.read(x5.PROTOCOL)
        bud = dict(protocol['budget']['smoke'], smoke=True)
        x1p = {'lightgbm': dict(num_leaves=[31], min_child_samples=[50], learning_rate=.05, n_estimators=10,
                                early_stopping_rounds=2, feature_fraction=.8, bagging_fraction=.8, seed=2024)}
        args = mock.Mock(logs='/nonexistent', threads=1)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(x2, 'load_farm', return_value=(train, v1, v2, {'F_2021': test}, {})), \
                mock.patch.object(x5, 'farm_caches', return_value=(cache, cache, MEAN, SCALE, '2024-01-01',
                                                                    '2024-01-01', {})):
            out = Path(tmp)
            (out / 'X5').mkdir()
            store = x2.Store(out)
            keys = x5.run_farm(_FakeLGB, args, out, store, 'F', {'rated': 2050.0}, x1p, {}, protocol, bud, [6, 12])
            with statistics_imports():
                res = x5.report(out, {'F': keys}, [6, 12], protocol, ['F'])
            self.assertEqual(sorted(res['farms']['F']), ['F_2021', 'V2'])
            audit = x1.read(out / 'X5' / 'F_audit.json')['audit']
            self.assertEqual(audit['train@binding']['exact_fraction'], 1.0)
            # a corrupted frozen x50 must stop the run before any fit
            bad = self._packs(cache, 120, '2021-03-01T00')
            bad['x50'][0, -1, 25] += 1.0
            with mock.patch.object(x2, 'load_farm', return_value=(bad, v1, v2, {}, {})):
                with self.assertRaisesRegex(ValueError, 'nominal GFS columns differ'):
                    x5.run_farm(_FakeLGB, args, out / 'b', x2.Store(out / 'b'), 'F', {'rated': 2050.0}, x1p, {},
                                protocol, bud, [6, 12])


if __name__ == '__main__':
    unittest.main()
