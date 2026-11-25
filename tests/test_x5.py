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


class MatchedCoverage(unittest.TestCase):
    """X5b (2026-11-23): fresh (6 h) vs old (15 h) cycle at matched structural coverage."""

    def _blocks(self, drop=()):
        cache = synthetic_cache(start='2021-02-27T00', days=8, drop=drop)
        iss = issues(start='2021-03-02T00', hours=72, stride=1)
        fresh, _ = x5.gfs_block(iss, cache, 6, MEAN, SCALE, '2024-01-01', 4)
        old, _ = x5.gfs_block(iss, cache, 15, MEAN, SCALE, '2024-01-01', 4)
        return iss, fresh, old

    def test_structural_masks_bound_actual_masks(self):
        _, fresh, old = self._blocks()
        for b in (fresh, old):
            fc, hist = x5.structural_masks(b)
            valid = b[:, x5.B_MASK] > 0
            np.testing.assert_array_equal(valid, fc)                       # complete cache: equal
            self.assertTrue((b[:, x5.B_HIST][~hist] == 0).all())
        _, fresh, old = self._blocks(drop={'2021030300'})
        fc, _ = x5.structural_masks(old)
        self.assertTrue(((old[:, x5.B_MASK] > 0) <= fc).all())             # missing file: strictly inside

    def test_gpt_example_and_common_structure(self):
        """GPT 11-23: at fresh age 6, t0-5 needs f001 (absent) while the old cycle (age >= 15) has it; Cc must drop it,
        D must not gain it; D also loses the targets the old cycle cannot reach."""
        _, fresh, old = self._blocks()
        st = x5.matched_states(fresh, old)
        a6 = fresh[:, x5.B_AGE] == 6
        self.assertTrue(a6.any())
        self.assertTrue((old[a6, x5.B_MASK][:, 0] > 0).all())              # old has t0-5
        self.assertTrue((st['Cc'][a6, x5.B_MASK][:, 0] == 0).all())
        self.assertTrue((st['Cc'][a6, x5.B_FC][:, 0] == 0).all())          # value zeroed with the bit
        self.assertTrue((st['D'][a6, x5.B_MASK][:, 0] == 0).all())
        np.testing.assert_array_equal(st['D'][:, x5.B_MASK] > 0, st['Cc'][:, x5.B_MASK] > 0)   # one coverage
        np.testing.assert_array_equal(st['D'][:, x5.B_AGE], fresh[:, x5.B_AGE])                # own ages
        np.testing.assert_array_equal(st['Cc'][:, x5.B_AGE], old[:, x5.B_AGE])
        targets = st['D'][:, x5.B_MASK][:, x5.TARGETS] > 0
        lead_old = old[:, x5.B_AGE][:, None] + np.arange(1, 5)[None, :]
        np.testing.assert_array_equal(targets, lead_old <= 18)
        hist_old_ok = old[:, x5.B_AGE] <= 18
        self.assertTrue((st['D'][~hist_old_ok, x5.B_HIST] == 0).all())
        np.testing.assert_array_equal(st['D'][hist_old_ok, x5.B_HIST], fresh[hist_old_ok, x5.B_HIST])
        audit = x5.coverage_audit(st)
        self.assertTrue(audit['masks_within_own_structure'] and audit['matched_within_common_structure'])
        self.assertGreater(audit['A']['target_valid'], audit['D']['target_valid'])
        self.assertAlmostEqual(audit['D']['target_valid'], audit['Cc']['target_valid'])

    def test_natural_gaps_are_not_copied(self):
        """A missing OLD file must not mask the fresh cycle (only structure is matched)."""
        _, fresh, old = self._blocks(drop={'2021030218'})
        st = x5.matched_states(fresh, old)
        gap = old[:, x5.B_MISS] == 1
        self.assertTrue(gap.any())
        fc_old, _ = x5.structural_masks(old)
        np.testing.assert_array_equal(st['D'][gap, x5.B_MASK] > 0, (fresh[gap, x5.B_MASK] > 0) & fc_old[gap])
        self.assertTrue((st['D'][gap, x5.B_MASK] > 0).any())
        self.assertTrue((st['Cc'][gap, x5.B_MASK] == 0).all())
        np.testing.assert_array_equal(st['Cc'][:, x5.B_MISS], old[:, x5.B_MISS])

    def test_classification(self):
        g = lambda gain, lo, hi: dict(gain=gain, ci95_low=lo, ci95_high=hi)  # noqa: E731
        c = x5.classify_match(g(8, 5, 11), g(.5, -1, 2), .5)
        self.assertEqual((c['truncation_ci_above_0'], c['cycle'], c['bound_kw']), (True, 'CI_WITHIN_BOUND', 4.0))
        # X5b R1-B2 counterexample: half-width 3.95 <= 4 but the CI reaches 7.8 kW -> must NOT be "within bound"
        self.assertEqual(x5.classify_match(g(8, 5, 11), g(3.8, -.1, 7.8), .5)['cycle'], 'CI_WIDE')
        self.assertEqual(x5.classify_match(g(2, 1, 3), g(.5, -2, 3), .5)['cycle'], 'CI_WIDE')
        self.assertEqual(x5.classify_match(g(2, 1, 3), g(3, 1, 5), .5)['cycle'], 'CI_ABOVE_0')
        self.assertEqual(x5.classify_match(g(2, 1, 3), g(-3, -5, -1), .5)['cycle'], 'CI_BELOW_0')
        self.assertEqual(x5.classify_match(g(2, 1, 3), None, .5)['cycle'], 'NOT_ESTIMABLE')
        self.assertEqual(x5.classify_match(g(2, 1, 3), dict(status='NOT_ESTIMABLE', gain=2.25, ci95_low=None,
                                                            ci95_high=None), .5)['cycle'], 'NOT_ESTIMABLE')
        self.assertFalse(x5.classify_match(g(2, -1, 3), g(0, -.5, .5), .5)['truncation_ci_above_0'])

    def test_bounded_contrast_minimum_clusters(self):
        """X5b R2-B1: below 40 issue clusters only n and the point estimate; GPT's 8-cluster case (gains -1.25..+5.75,
        block 8 -> one candidate block -> zero-width [+2.25, +2.25]) must not yield an interval or a label."""
        n = 60
        start = (np.datetime64('2021-07-01T00', 'h') + np.arange(n) * np.timedelta64(3, 'h')).astype('datetime64[ns]')
        rng = np.random.default_rng(7)
        e = {s: {a: rng.uniform(0, 300, (n, 4)) for a in ('LG', 'LUG')} for s in ('D', 'Cc')}
        kw = dict(block_length=8, replicates=200, seed=2024)
        protocol = x1.read(x5.PROTOCOL_B)
        self.assertEqual(protocol['classification']['min_issue_clusters'], 40)
        for k in (0, 1, 2, 8, 39, 40):
            rows = np.arange(n) < k
            with statistics_imports():
                out = x5.bounded_contrast(e, 'D', 'Cc', start, rows, kw, 40)
            if k < 40:
                self.assertEqual((out['status'], out['ci95_low'], out['ci95_high']), ('NOT_ESTIMABLE', None, None))
                self.assertEqual((out['windows'], out['issue_time_clusters']), (k, k))
                if k == 0:
                    self.assertIsNone(out['gain'])
                else:
                    cand, ref = x5.rise_losses(*(e[s][a][rows] for s, a in (('D', 'LG'), ('D', 'LUG'), ('Cc', 'LG'),
                                                                              ('Cc', 'LUG'))))
                    self.assertAlmostEqual(out['gain'], (ref.mean(1) - cand.mean(1)).mean(), places=9)
            else:
                self.assertNotIn('status', out)
                self.assertLess(out['ci95_low'], out['ci95_high'])
        # GPT's counterexample: 8 clusters, gains -1.25 .. +5.75 (mean +2.25)
        gains = np.arange(8) - 1.25
        e8 = {'D': {'LG': np.zeros((8, 4)), 'LUG': np.zeros((8, 4))},
              'Cc': {'LG': np.repeat(gains[:, None], 4, 1), 'LUG': np.zeros((8, 4))}}
        with statistics_imports():
            out = x5.bounded_contrast(e8, 'D', 'Cc', start[:8], np.ones(8, bool), kw, 40)
        self.assertEqual(out['status'], 'NOT_ESTIMABLE')
        self.assertAlmostEqual(out['gain'], 2.25)

    def test_natural_gap_audit_counts_window_disagreement(self):
        """X5b R1-B1: GPT's synthetic case (one dropped cycle) -> 36 of 216 windows disagree, 18 in the targets,
        while both structural checks still pass; the agreement mask flags exactly those windows."""
        _, fresh, old = self._blocks(drop={'2021030218'})
        st = x5.matched_states(fresh, old)
        audit = x5.coverage_audit(st)
        gap = audit['natural_gap']
        self.assertEqual((gap['windows'], gap['windows_disagree'], gap['windows_target_mask_disagree']), (216, 36, 18))
        self.assertTrue(audit['masks_within_own_structure'] and audit['matched_within_common_structure'])
        agree = x5.natural_gap_agreement(st)
        self.assertEqual(int((~agree).sum()), 36)
        np.testing.assert_array_equal(st['D'][agree][:, x5.B_MASK] > 0, st['Cc'][agree][:, x5.B_MASK] > 0)
        _, fresh, old = self._blocks()
        self.assertTrue(x5.natural_gap_agreement(x5.matched_states(fresh, old)).all())   # complete cache: no gaps

    def test_structure_violation_is_an_input_error(self):
        """X5b R1-S1: a validity bit outside the registered structure stops the run."""
        _, fresh, old = self._blocks()
        st = x5.matched_states(fresh, old)
        st['D'] = st['D'].copy()
        st['D'][st['D'][:, x5.B_AGE] == 6, x5.B_MASK.start] = 1.0         # t0-5 at age 6 (needs f001)
        with self.assertRaisesRegex(ValueError, 'outside the registered structure'):
            x5.check_structure(x5.coverage_audit(st), 'F', 'train')

    def test_match_end_to_end_and_decomposition(self):
        res = self._match_end_to_end()
        self.assertFalse(res['farms']['F']['V2']['natural_gap']['cycle_common_includes_natural_gaps'])
        self.assertEqual(res['farms']['F']['V2']['cycle_common_agreeing_windows'],
                         res['farms']['F']['V2']['deltas']['cycle_common'])

    def test_match_end_to_end_with_natural_gap(self):
        """A missing old-side cycle inside V2: the all-window cycle contrast is flagged, and the label uses the
        contrast on agreeing windows only (X5b R1-B1)."""
        res = self._match_end_to_end(gap_rows=slice(10, 22))
        row = res['farms']['F']['V2']
        self.assertTrue(row['natural_gap']['cycle_common_includes_natural_gaps'])
        self.assertLess(row['natural_gap']['share_agree'], 1.0)
        self.assertTrue(row['reading']['natural_gap_difference'])
        self.assertEqual(res['outcome']['farms_with_natural_gap_difference_on_V2'], ['F'])
        self.assertIsNotNone(row['cycle_common_agreeing_windows'])
        self.assertNotEqual(row['cycle_common_agreeing_windows']['windows'], row['deltas']['cycle_common']['windows'])

    def test_match_end_to_end_two_agreeing_clusters(self):
        """GPT's R2 repro: only 2 agreeing windows in V2 used to crash the report; now NOT_ESTIMABLE, no crash."""
        res = self._match_end_to_end(gap_rows=slice(2, None))
        row = res['farms']['F']['V2']
        self.assertEqual(row['cycle_common_agreeing_windows']['status'], 'NOT_ESTIMABLE')
        self.assertEqual(row['cycle_common_agreeing_windows']['issue_time_clusters'], 2)
        self.assertEqual(row['reading']['cycle'], 'NOT_ESTIMABLE')
        self.assertIn('not estimable', x5.summary_match_md(res))

    def _match_end_to_end(self, gap_rows=None):
        import tempfile
        from unittest import mock
        cache = synthetic_cache(start='2021-02-25T00', days=40)
        nominal = cache
        real_prepare = x5.prepare_farm

        def prepare_with_old_gap(*a, **k):
            """Simulate a missing OLD-cycle file for some V2 windows (natural gap on one side only)."""
            packs_, tests_, blocks_ = real_prepare(*a, **k)
            if gap_rows is not None:
                b = blocks_['V2'][15]
                b[gap_rows, x5.B_FC] = 0
                b[gap_rows, x5.B_MASK] = 0
                b[gap_rows, x5.B_HIST] = 0
                b[gap_rows, x5.B_MISS] = 1
            return packs_, tests_, blocks_
        packs = EndToEnd()._packs
        train, v1, v2 = packs(nominal, 120, '2021-03-01T00'), packs(nominal, 60, '2021-03-20T00'), \
            packs(nominal, 60, '2021-03-28T00')
        test = packs(nominal, 60, '2021-03-10T00')
        protocol = x1.read(x5.PROTOCOL_B)
        bud = dict(protocol['budget']['smoke'], smoke=True)
        x1p = {'lightgbm': dict(num_leaves=[31], min_child_samples=[50], learning_rate=.05, n_estimators=10,
                                early_stopping_rounds=2, feature_fraction=.8, bagging_fraction=.8, seed=2024)}
        args = mock.Mock(logs='/nonexistent', threads=1)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(x2, 'load_farm', return_value=(train, v1, v2, {'F_2021': test}, {})), \
                mock.patch.object(x5, 'farm_caches', return_value=(cache, cache, MEAN, SCALE, '2024-01-01',
                                                                    '2024-01-01', {})), \
                mock.patch.object(x5, 'prepare_farm', side_effect=prepare_with_old_gap):
            out = Path(tmp)
            (out / 'X5').mkdir()
            keys = x5.run_farm_match(_FakeLGB, args, out, x2.Store(out), 'F', {'rated': 2050.0}, x1p, {}, protocol, bud)
            for n in x5.STATES:
                self.assertTrue((out / f'S_{n}/F/LUG_{n}/2024/V2.npy').exists())
            with statistics_imports():
                res = x5.report_match(out, {'F': keys}, protocol, ['F'])
            row = res['farms']['F']['V2']
            d = row['deltas']
            self.assertAlmostEqual(d['total']['gain'], d['truncation_fresh']['gain'] + d['cycle_common']['gain']
                                   + d['old_side_restriction']['gain'], places=6)
            self.assertAlmostEqual(d['truncation_fresh']['gain'], row['U']['D']['gain'] - row['U']['A']['gain'], places=6)
            self.assertIn(res['outcome']['outcome'], ('TRUNCATION_CI_ABOVE_0_AND_CYCLE_CI_WITHIN_BOUND', 'CYCLE_CI_ABOVE_0',
                                                      'NO_COMMON_PATTERN'))
            self.assertIn('| F | V2 |', x5.summary_match_md(res))
            cov = x1.read(out / 'X5' / 'F_coverage.json')
            self.assertTrue(cov['train']['matched_within_common_structure'])
            self.assertIn('windows_disagree', cov['V2']['natural_gap'])
        return res

    def test_formal_match_needs_match_smoke(self):
        import tempfile
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            smoke = Path(tmp) / 'smoke'
            smoke.mkdir()
            x1.save(smoke / 'smoke_passed.json', dict(status='PASS', commit='abc'))            # X5 smoke, no kind
            args = SimpleNamespace(protocol_sha256=x1.digest(x5.PROTOCOL_B), allow_dirty=False, smoke_result=smoke,
                                   output=Path(tmp) / 'run')
            with patch.object(x5, 'git_identity', return_value=('abc', '')):
                with self.assertRaisesRegex(ValueError, 'same kind'):
                    x5.start_run(args, 'MRUN', x5.PROTOCOL_B)
                x1.save(smoke / 'smoke_passed.json', dict(status='PASS', commit='abc', kind='match'))
                self.assertTrue(x5.start_run(args, 'MRUN', x5.PROTOCOL_B).exists())


if __name__ == '__main__':
    unittest.main()
