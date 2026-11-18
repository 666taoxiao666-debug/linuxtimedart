"""Static tests for scripts/run_x3.py (numpy only; Chronos and LightGBM run on the server in the smoke chain)."""
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_x1 as x1  # noqa: E402
from scripts import run_x3 as x3  # noqa: E402

H = 3_600_000_000_000


def make_ctx(units=(1, 2, 3), n_issue=40, start='2021-01-03T00', seed=0):
    """Frozen-window contexts (stride 3 h, 24-h inputs) plus the turbine-hour grid built from them."""
    rng = np.random.default_rng(seed)
    t0 = np.datetime64(start, 'h').astype('datetime64[ns]').astype(np.int64)
    hours = t0 - 200 * H + np.arange(n_issue * 3 + 400) * H
    series = {u: rng.normal(size=(len(hours), 21)).astype(np.float32) for u in units}
    issue, turb, x21 = [], [], []
    for k in range(n_issue):
        t = t0 + 3 * k * H
        j = int((t - hours[0]) // H)
        for u in units:
            issue.append(t), turb.append(u), x21.append(series[u][j - 23:j + 1])
    ctx = dict(issue=np.array(issue, np.int64), turbines=np.array(turb), x21=np.stack(x21),
               gF=rng.uniform(2, 15, (len(issue), 13)), gF_mask=np.ones((len(issue), 13)),
               mean=np.linspace(0, 1, 21), scale=np.linspace(1, 2, 21))
    ctx['historical'] = np.full((len(issue), 24), 5.0)
    grid = x1.history_grid([ctx])
    grid.update(hours=hours.astype('datetime64[ns]'), historical=np.full((len(hours), 2), 5.0))
    return ctx, grid, series, hours


class Context(unittest.TestCase):
    def test_long_context_matches_series_and_ends_at_issue(self):
        ctx, grid, series, hours = make_ctx()
        table = x3.HourTable(grid)
        idx = np.arange(len(ctx['issue']))[-6:]
        x = x3.context_window(ctx, table, idx, 60)
        for row, i in enumerate(idx):
            j = int((ctx['issue'][i] - hours[0]) // H)
            truth = series[int(ctx['turbines'][i])][j - 59:j + 1]
            ok = np.isfinite(x[row, :, 0])
            np.testing.assert_allclose(x[row][ok], truth[ok])
        np.testing.assert_array_equal(x[:, -24:], ctx['x21'][idx])

    def test_future_hours_cannot_enter_the_context(self):
        ctx, grid, _, _ = make_ctx()
        idx = np.arange(12)
        a = x3.context_window(ctx, x3.HourTable(grid), idx, 48)
        later = dict(grid)
        later['grid_value'] = grid['grid_value'].copy()
        latest_issue = ctx['issue'][idx].max()
        later['grid_value'][np.asarray(grid['grid_hour']) > latest_issue] = 999.0   # change only later hours
        b = x3.context_window(ctx, x3.HourTable(later), idx, 48)
        np.testing.assert_array_equal(a, b)

    def test_missing_older_hours_are_nan_and_coverage_reported(self):
        ctx, grid, _, _ = make_ctx(n_issue=10)
        x = x3.context_window(ctx, x3.HourTable(grid), np.arange(3), 168)
        self.assertTrue(np.isnan(x[:, :100]).all())                        # before the first window: absent
        self.assertTrue((x3.coverage(x) < 1).all() and (x3.coverage(x) > 0).all())

    def test_table_disagreement_is_rejected(self):
        ctx, grid, _, _ = make_ctx()
        bad = dict(ctx, x21=ctx['x21'].copy())
        bad['x21'][-1, -2, 3] += 1
        with self.assertRaises(ValueError):
            x3.context_window(bad, x3.HourTable(grid), np.arange(len(ctx['issue'])), 48)

    def test_gfs_history_from_hour_axis(self):
        ctx, grid, _, hours = make_ctx()
        grid['historical'] = np.column_stack([np.arange(len(hours), dtype=float), np.zeros(len(hours))])
        h = x3.gfs_history(ctx, grid, np.arange(3), 30)
        j = int((ctx['issue'][0] - hours[0]) // H)
        np.testing.assert_array_equal(h[0], np.arange(j - 29, j + 1))
        short = dict(grid, hours=grid['hours'][j - 10:], historical=grid['historical'][j - 10:])
        self.assertTrue(np.isnan(x3.gfs_history(ctx, short, np.arange(1), 30)[0, :19]).all())


class Groups(unittest.TestCase):
    def test_sdk_group_batch_counts_covariates_and_keeps_all_turbines(self):
        class Tensor:
            def __init__(self, value): self.value = value
            def __getitem__(self, key): return Tensor(self.value[key])
            def detach(self): return self
            def float(self): return self
            def cpu(self): return self
            def numpy(self): return self.value

        class Pipe:
            def predict(self, rows, **kwargs):
                # Simulate the SDK counting target + covariates in each sequential batch.
                sizes = [1 + len(r['past_covariates']) for r in rows]
                self.assert_batch = kwargs['batch_size']
                groups, current, count = [], [], 0
                for row, size in zip(rows, sizes):
                    if count >= kwargs['batch_size']:
                        groups.append(current)
                        current, count = [], 0
                    current.append(row)
                    count += size
                groups.append(current)
                return [Tensor(np.full((1, 3, 4), sum(r['target'][0] for r in group)))
                        for group in groups for _ in group]

        pipe = Pipe()
        rows = [dict(target=np.full(24, i + 1.), past_covariates={str(k): np.zeros(24) for k in range(14)})
                for i in range(6)]
        predict = x3.chronos_predict_fn(pipe, 24, True, [0, 1, 2])
        result = predict(rows, len(rows))
        self.assertEqual(pipe.assert_batch, 90)
        np.testing.assert_array_equal(np.stack(result), np.full((6, 3, 4), 21.))
        np.testing.assert_array_equal(np.stack(predict(rows[::-1], 6))[::-1], result)

    def test_one_issue_group_per_call(self):
        ctx, grid, _, _ = make_ctx(units=(1, 2, 3, 4))
        order = np.random.default_rng(1).permutation(len(ctx['issue']))
        issue = ctx['issue'][order]
        groups = x3.issue_groups(issue)
        calls = []

        def fake(inputs, batch):
            calls.append([int(v['issue']) for v in inputs])
            self.assertEqual(batch, len(inputs))
            return [v['issue'] for v in inputs]
        out = x3.run_groups(fake, [dict(issue=t) for t in issue], groups)
        self.assertTrue(all(len(set(c)) == 1 for c in calls))
        self.assertEqual(len(calls), len(np.unique(issue)))
        np.testing.assert_array_equal(out, issue)

    def test_smoke_shrink_keeps_whole_groups(self):
        ctx, _, _, _ = make_ctx()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'c.npz'
            np.savez(path, **ctx)
            rec = x3.shrink_context(dict(name='c', path=str(path), sha256=x1.digest(path)), Path(tmp), 5)
            small = x1.load_pack(Path(rec['path']))
            self.assertEqual(len(np.unique(small['issue'])), 5)
            self.assertEqual(len(small['issue']), 5 * 3)

    def test_smoke_validation_keeps_both_v1_and_v2(self):
        ctx, _, _, _ = make_ctx(n_issue=60)
        split = {'V1': ['2021-01-03', '2021-01-07'], 'V2': ['2021-01-07', '2021-01-11']}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'validation.npz'
            np.savez(path, **ctx)
            rec = x3.shrink_context(dict(name='validation', path=str(path), sha256=x1.digest(path)),
                                    Path(tmp), 5, split)
            small = x1.load_pack(Path(rec['path']))
            issue = small['issue'].astype('datetime64[ns]')
            times = issue[:, None] + np.arange(1, 5) * x3.HOUR
            for mask in (x3.x2.period(times, split[s]).all(1) for s in ('V1', 'V2')):
                self.assertEqual(len(np.unique(issue[mask])), 5)
                self.assertEqual(mask.sum(), 15)

    def test_key_positions(self):
        ctx, _, _, _ = make_ctx()
        keys = dict(issue=ctx['issue'][[7, 2]], turbines=ctx['turbines'][[7, 2]])
        np.testing.assert_array_equal(x3.key_positions(ctx['issue'], ctx['turbines'], keys), [7, 2])
        with self.assertRaises(ValueError):
            x3.key_positions(ctx['issue'], ctx['turbines'], dict(issue=np.array([1]), turbines=np.array([1])))


class Decisions(unittest.TestCase):
    def s(self, gain, lo, hi, ref=100.):
        return dict(gain=gain, ci95_low=lo, ci95_high=hi, reference_mean_loss=ref)

    def test_select_config_ties(self):
        self.assertEqual(x3.select_config({(512, 'group'): 1.0, (168, 'single'): 1.0, (168, 'group'): 1.0}), (168, 'single'))
        self.assertEqual(x3.select_config({(512, 'single'): 0.9, (168, 'single'): 1.0}), (512, 'single'))

    def test_ensemble_search(self):
        y = np.zeros((10, 4))
        w, mae = x3.ensemble_search(np.ones((10, 4)), -np.ones((10, 4)), y)
        self.assertEqual((w, mae), (0.5, 0.0))
        w, _ = x3.ensemble_search(np.zeros((10, 4)), np.zeros((10, 4)), y)
        self.assertEqual(w, 1.0)                                             # tie -> no ensemble

    def test_q2_sequence_and_farm_guard(self):
        z = 2.394
        ni = {f: self.s(0.0, -0.5, 0.5) for f in 'KPH'}
        self.assertEqual(x3.q2_decision(ni, z, .01)['outcome'], 'NON_INFERIOR')
        sup = {f: self.s(2.0, 1.5, 2.5) for f in 'KPH'}
        self.assertEqual(x3.q2_decision(sup, z, .01)['outcome'], 'SUPERIOR')
        guard = dict(ni, H=self.s(-2.0, -3.0, -1.5))                        # one farm entirely below -1%
        d = x3.q2_decision(guard, z, .01)
        self.assertEqual(d['outcome'], 'NEGATIVE')
        self.assertEqual(d['farms_significantly_worse_than_margin'], ['H'])
        self.assertEqual(x3.q2_decision({'K': self.s(0, 0, 0)}, z, .01)['outcome'], 'TECHNICAL_FAILURE')

    def test_all_positive_pooled(self):
        good = {f: self.s(2.0, 1.5, 2.5) for f in 'KPH'}
        self.assertTrue(x3.all_positive_pooled(good, 2.394, .01)['passed'])
        self.assertFalse(x3.all_positive_pooled(dict(good, H=self.s(.5, .2, .8)), 2.394, .01)['passed'])
        self.assertTrue(x3.all_positive_pooled(dict(good, H=self.s(.5, .2, .8)), 2.394)['passed'])

    def test_choose_q2_mode(self):
        ok, bad = dict(ok=True), dict(ok=False)
        self.assertEqual(x3.choose_q2_mode({'K': dict(modes=dict(full=ok, lora=ok)), 'H': dict(modes=dict(full=bad, lora=ok))}), 'lora')
        self.assertIsNone(x3.choose_q2_mode({'K': dict(modes=dict(full=bad, lora=bad))}))

    def test_pinball(self):
        y = np.zeros((4, 4))
        q = np.stack([np.full((4, 4), -1.), np.zeros((4, 4)), np.full((4, 4), 1.)], 1)
        r = x3.pinball(q, y)
        self.assertEqual(r['coverage80'], 1.0)
        self.assertEqual(r['width80_kw'], 2.0)
        self.assertAlmostEqual(r['pinball']['0.1'][0], .1)
        self.assertEqual(r['crossing_rate'], 0.0)


class ColdStart(unittest.TestCase):
    def pack(self, start, n, units=(0, 1)):
        issue = np.datetime64(start, 'h') + (np.arange(n) * 3).astype('timedelta64[h]')
        issue = np.repeat(issue, len(units))
        times = issue[:, None] + np.arange(1, 5).astype('timedelta64[h]')
        return dict(issue=issue.astype('datetime64[ns]').astype(np.int64),
                    times=times.astype('datetime64[ns]').astype(np.int64), turbines=np.tile(units, n))

    def test_budget_and_months(self):
        self.assertEqual(x3.months_before('2021-01-01', 1), np.datetime64('2020-12-01', 'ns'))
        p = self.pack('2020-12-30T12', 30)
        m = x3.budget_mask(p, '2020-12-01', '2021-01-01')
        labels = p['times'].astype('datetime64[ns]')
        self.assertTrue((labels[m] < np.datetime64('2021-01-01')).all())
        self.assertTrue(((labels >= np.datetime64('2021-01-01')).any(1) == ~m).all())

    def test_es_split_has_no_fit_label_inside_es(self):
        p = self.pack('2020-12-01T00', 240, units=(0, 1, 2))
        fit, es, cut = x3.es_split(p['issue'], p['times'], .2)
        labels = p['times'].astype('datetime64[ns]')
        self.assertTrue((labels[fit] < cut).all())
        self.assertTrue((p['issue'][es].astype('datetime64[ns]') >= cut).all())
        self.assertFalse((fit & es).any())
        for t in np.unique(p['issue'][es]):                                  # same issue hour never split
            self.assertTrue(es[p['issue'] == t].all())

    def test_label_exactly_at_deployment_is_not_available_before_deployment(self):
        p = self.pack('2020-12-31T20', 1)
        self.assertFalse(x3.budget_mask(p, '2020-12-01', '2021-01-01')[0])

    def test_physical_features(self):
        rng = np.random.default_rng(0)
        n = 5
        mean, scale = rng.normal(size=21), rng.uniform(.5, 2, 21)
        raw = rng.uniform(0, 10, (n, 24, 21))
        raw[:, -1, 19] = [0, 1, 0, 1, 0]
        x50 = np.zeros((n, 24, 50))
        x50[:, :, :20] = (raw[:, :, :20] - mean[:20]) / scale[:20]
        x50[:, :, 49] = (raw[:, :, 20] - mean[20]) / scale[20]
        x50[:, -1, 47], x50[:, -1, 48] = 8, 0
        gF, mask = rng.uniform(2, 12, (n, 13)), np.ones((n, 13))
        mask[0, 3] = 0
        pack = dict(x50=x50, gF=gF, gF_mask=mask, turbines=np.arange(n), retrofit=np.ones(n))
        X = x3.physical_features(pack, 'HoT', mean, scale, 2300.)
        self.assertEqual(X.shape, (n, len(x3.PHYS_NAMES)))
        col = {k: i for i, k in enumerate(x3.PHYS_NAMES)}
        np.testing.assert_allclose(X[:, col['wspd']], raw[:, -1, 0], rtol=1e-5)
        np.testing.assert_allclose(X[:, col['power_last10_pu']], raw[:, -1, 6] / 2300, rtol=1e-4)
        np.testing.assert_allclose(X[:, col['p_hist_0']], raw[:, 0, 20] / 2300, rtol=1e-4)
        self.assertTrue(np.isnan(X[[1, 3], col['up_0']]).all() and np.isfinite(X[[0, 2, 4], col['up_0']]).all())
        self.assertTrue(np.isnan(X[0, col['gfs_3']]))
        self.assertEqual(X[0, col['farm']], 2)
        self.assertEqual(X[4, col['farm_turbine']], 2004)
        K = x3.physical_features(dict(pack, retrofit=np.ones(n)), 'Kelmarsh', mean, scale, 2050.)
        self.assertTrue((K[:, col['retrofit']] == 0).all() and (K[:, col['retrofit_applicable']] == 0).all())

    def test_training_arrays_layout(self):
        ctx, grid, _, _ = make_ctx()
        arr = x3.training_arrays(ctx, x3.HourTable(grid), grid, np.arange(6), 30, np.ones((6, 4)))
        self.assertEqual(arr['target'].shape, (6, 30))
        self.assertEqual(arr['covariates'].shape, (6, 14, 30))
        rows = x3.training_series(arr)
        self.assertEqual(len(rows[0]['target']), 34)
        self.assertTrue(np.isnan(rows[0]['past_covariates']['wind'][-4:]).all())
        np.testing.assert_allclose(rows[0]['past_covariates']['gfs'][-4:], ctx['gF'][0, 6:10])


class Identity(unittest.TestCase):
    def test_formal_gate_rejects_tampered_smoke_results_before_creating_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / 'smoke_report'
            root.mkdir()
            args = SimpleNamespace(base=base, mode='q1', smoke=False, allow_dirty=False,
                                   protocol_sha256=x1.digest(x3.PROTOCOL))
            x1.save(root / 'launch.json', dict(commit='reviewed', script_sha256=x1.digest(x3.__file__),
                     protocol_sha256=args.protocol_sha256, mode='report', smoke=True))
            for name in ('results.json', 'dev_decisions.json'):
                x1.save(root / name, dict(value=1))
            x3.finish(root, x3.x2.Store(root), dict(results_sha256=x1.digest(root / 'results.json'),
                       dev_decisions_sha256=x1.digest(root / 'dev_decisions.json')))
            x1.save(root / 'results.json', dict(value=2))
            with patch.object(x3, 'git_identity', return_value=('reviewed', False)):
                with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
                    x3.start_run(args, 'q1')
            self.assertFalse((base / 'q1').exists())

    def setup_runs(self, tmp):
        base = Path(tmp)
        args = SimpleNamespace(base=base, mode='q3', smoke=False)
        identity = dict(commit='reviewed', script_sha256='script', protocol_sha256='protocol', smoke=False)
        for mode in ('q1', 'q2', 'q3'):
            (base / mode).mkdir()
            x1.save(base / mode / 'launch.json', dict(identity, mode=mode))
        return args

    def test_selection_tamper_rejected_without_changing_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.setup_runs(tmp)
            root = Path(tmp) / 'q1'
            x1.save(root / 'selection.json', dict(winner='single'))
            x3.finish(root, x3.x2.Store(root), dict(selection_sha256=x1.digest(root / 'selection.json')))
            self.assertIsNotNone(x3.upstream(args, 'q1'))
            x1.save(root / 'selection.json', dict(winner='group'))
            with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
                x3.upstream(args, 'q1')

    def test_failed_q2_from_other_identity_cannot_be_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.setup_runs(tmp)
            root = Path(tmp) / 'q2'
            x1.save(root / 'failed.json', dict(type='RuntimeError'))
            self.assertIsNone(x3.q2_state(args))
            launch = x1.read(root / 'launch.json')
            launch['commit'] = 'other'
            x1.save(root / 'launch.json', launch)
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                x3.q2_state(args)

    def test_completed_and_failed_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.setup_runs(tmp)
            root = Path(tmp) / 'q2'
            x3.finish(root, x3.x2.Store(root))
            x1.save(root / 'failed.json', dict(type='RuntimeError'))
            with self.assertRaisesRegex(ValueError, 'failed or not completed'):
                x3.q2_state(args)


if __name__ == '__main__':
    unittest.main()
