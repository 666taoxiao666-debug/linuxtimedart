"""Static tests for scripts/run_x4.py (numpy only; Chronos and LightGBM run on the server in the smoke chain)."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_x1 as x1  # noqa: E402
from scripts import run_x3 as x3  # noqa: E402
from scripts import run_x4 as x4  # noqa: E402
from tests.test_x3 import make_ctx  # noqa: E402

H = np.timedelta64(1, 'h')


def hourly_issues(start, n, units=(0, 1)):
    issue = np.datetime64(start, 'h') + (np.arange(n) * 3).astype('timedelta64[h]')
    return np.repeat(issue, len(units)).astype('datetime64[ns]')


class WindowPlan(unittest.TestCase):
    def test_pool_and_split_reproduce_the_x3_q4_rule(self):
        issues = {'T_train': hourly_issues('2020-10-20T00', 400), 'T_validation': hourly_issues('2021-01-01T00', 50)}
        idx = x4.context_index(issues)
        D = '2021-01-01'
        sp = x4.target_split(idx, D, 1, .2)
        rows = [dict(issue=v.astype(np.int64), times=x4.label_times_of(v).astype(np.int64)) for v in issues.values()]
        tgt = {k: np.concatenate([r[k][x3.budget_mask(r, x3.months_before(D, 1), D)] for r in rows])
               for k in ('issue', 'times')}
        fit, es, cut = x3.es_split(tgt['issue'], tgt['times'], .2)
        self.assertEqual((len(sp['fit']), len(sp['es']), sp['cut']), (int(fit.sum()), int(es.sum()), cut))
        self.assertTrue((idx['times'][sp['fit']] < sp['cut']).all())          # no fit label at/after the cut
        self.assertTrue((idx['issue'][sp['es']] >= sp['cut']).all())
        pool = x4.source_pool(idx, sp['cut'])
        ref = sum(int(x3.budget_mask(r, '1900-01-01', sp['cut']).sum()) for r in rows)
        self.assertEqual(len(pool), ref)
        self.assertTrue((idx['times'][pool] < sp['cut']).all())
        by = x4.rows_by_context(idx, np.r_[sp['fit'], sp['es']])
        self.assertEqual(list(by), ['T_train'])
        np.testing.assert_array_equal(np.sort(np.r_[sp['fit'], sp['es']]), by['T_train'])

    def test_draw_is_fixed_and_without_replacement(self):
        a, b = x4.draw(1000, 50, [2024, 0, 1]), x4.draw(1000, 50, [2024, 0, 1])
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(np.unique(a)), 50)
        self.assertFalse(np.array_equal(a, x4.draw(1000, 50, [2024, 0, 2])))
        self.assertEqual(len(x4.draw(10, 50, [1])), 10)

    def test_key_subset_keeps_whole_issue_groups(self):
        keys = dict(issue=hourly_issues('2021-01-01T00', 10, units=(0, 1, 2)).astype(np.int64),
                    turbines=np.tile([0, 1, 2], 10), truth=np.zeros((30, 4)))
        sub = x4.key_subset(keys, 4)
        self.assertEqual(len(sub['issue']), 12)
        self.assertIs(x4.key_subset(keys, None), keys)


class Arrays(unittest.TestCase):
    def setUp(self):
        ctx, grid, _, hours = make_ctx(units=(1, 2), n_issue=30)
        mean, scale = ctx['mean'], ctx['scale']
        x21 = ctx['x21'].copy()
        flagged = ctx['turbines'] == 2
        x21[:, :, 19] = np.where(flagged[:, None], (1 - mean[19]) / scale[19], (0 - mean[19]) / scale[19])
        self.ctx = dict(ctx, x21=x21)
        self.grid = x1.history_grid([self.ctx])
        self.grid.update(hours=hours.astype('datetime64[ns]'),
                         historical=np.arange(len(hours), dtype=float)[:, None] + 1.0)
        self.table, self.flagged = x3.HourTable(self.grid), flagged

    def test_budget_mask_and_upstream_mask(self):
        idx, L = np.arange(len(self.ctx['issue']))[-6:], 48
        lo = np.datetime64(int(self.ctx['issue'][idx].min()), 'ns') - 20 * H
        arr = x4.window_arrays(self.ctx, self.table, self.grid, idx, L, np.ones((6, 4)), lo=lo)
        plain = x3.training_arrays(self.ctx, self.table, self.grid, idx, L, np.ones((6, 4)))
        before = x4.budget_hours(self.ctx['issue'][idx], L, lo)
        self.assertTrue(before.any() and (~before).any())
        self.assertTrue(np.isnan(arr['target'][before]).all())
        self.assertTrue(np.isnan(arr['covariates'][:, 2][before]).all())          # GFS history masked too
        keep = ~before
        np.testing.assert_allclose(arr['target'][keep], plain['target'][keep], equal_nan=True)
        for i, j in enumerate(idx):
            up = arr['covariates'][i, 3:][:, keep[i]]
            if self.flagged[j]:
                self.assertTrue(np.isnan(up).all())
            else:
                np.testing.assert_allclose(up, plain['covariates'][i, 3:][:, keep[i]], equal_nan=True)
        np.testing.assert_array_equal(arr['issue'], self.ctx['issue'][idx])

    def test_arrays_inputs_match_build_inputs(self):
        idx, L = np.arange(len(self.ctx['issue']))[-5:], 40
        arr = x4.window_arrays(self.ctx, self.table, self.grid, idx, L, np.ones((5, 4)))
        ref, _ = x3.build_inputs(self.ctx, self.table, self.grid, idx, L, masked=True)
        got = x4.arrays_inputs(arr)
        for g, r in zip(got, ref):
            np.testing.assert_allclose(g['target'], r['target'], rtol=1e-6, equal_nan=True)
            self.assertEqual(set(g['past_covariates']), set(r['past_covariates']))
            for k in r['past_covariates']:
                np.testing.assert_allclose(g['past_covariates'][k], r['past_covariates'][k], rtol=1e-5, equal_nan=True)
            np.testing.assert_allclose(g['future_covariates']['gfs'], r['future_covariates']['gfs'], equal_nan=True)

    def test_window_arrays_loads_and_checks_saved_history_receipt(self):
        idx = np.arange(len(self.ctx['issue']))[-3:]
        truth = np.ones((3, 4))
        with tempfile.TemporaryDirectory() as tmp:
            src = dict(self.grid, source='a3_grid')
            rec = x3.save_source(Path(tmp), 'Kelmarsh', src, {})
            got = x4.window_arrays(self.ctx, self.table, rec, idx, 40, truth)
            expected = x4.window_arrays(self.ctx, self.table, src, idx, 40, truth)
            for key in ('target', 'covariates', 'gfs_future', 'truth'):
                np.testing.assert_allclose(got[key], expected[key], equal_nan=True)
            with Path(rec['path']).open('ab') as fh:
                fh.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
                x4.window_arrays(self.ctx, self.table, rec, idx, 40, truth)

    def test_subset_rows_never_slices_constants(self):
        n = 21                                                                # n == len(mean) must not slice mean
        ctx = dict(issue=np.arange(n), x21=np.zeros((n, 24, 21)), mean=np.arange(21.), scale=np.ones(21))
        sub = x4.subset_rows(ctx, [3, 1])
        np.testing.assert_array_equal(sub['issue'], [3, 1])
        self.assertEqual(sub['mean'].shape, (21,))


class Selection(unittest.TestCase):
    def test_pick_lr_ties(self):
        r = [dict(learning_rate=1e-5, mae=1.0), dict(learning_rate=1e-6, mae=1.0)]
        self.assertEqual(x4.pick_lr(r, [1e-6, 1e-5])['learning_rate'], 1e-6)
        r[0]['mae'] = .9
        self.assertEqual(x4.pick_lr(r, [1e-6, 1e-5])['learning_rate'], 1e-5)

    def test_pick_b_prefers_x3_on_ties(self):
        self.assertEqual(x4.pick_b([dict(v1_mae=2.0), dict(v1_mae=2.0), dict(v1_mae=3.0)]), 0)
        self.assertEqual(x4.pick_b([dict(v1_mae=2.0), dict(v1_mae=1.0), dict(v1_mae=1.0)]), 1)

    def test_eval_curve(self):
        issue = np.array(['2020-12-27T00', '2020-12-27T06', '2020-12-28T00'], 'datetime64[ns]')
        truth = np.zeros((3, 4))
        preds = np.stack([np.ones((3, 4)), np.full((3, 4), 3.)])
        c = x4.eval_curve(issue, preds, truth, [0, 20])
        self.assertEqual(c['mae'], [1.0, 3.0])
        self.assertEqual(c['effective_issue_hours'], 3)
        self.assertEqual(c['best_minus_second'], 2.0)
        self.assertEqual(sorted(c['daily']), ['2020-12-27', '2020-12-28'])


class D0(unittest.TestCase):
    def test_month_blocks(self):
        b = x4.month_blocks('2021-01-01')
        self.assertEqual([str(lo)[:10] for lo, _ in b], ['2020-08-01', '2020-09-01', '2020-10-01', '2020-11-01'])
        self.assertEqual(str(b[-1][1])[:10], '2020-12-01')                    # stops at D - 1 month

    def test_event_features_columns(self):
        col = {n: i for i, n in enumerate(x3.PHYS_NAMES)}
        X = np.zeros((2, len(x3.PHYS_NAMES)))
        for i in range(13):
            X[:, col[f'gfs_{i}']] = i
        X[:, col['p_hist_23']], X[:, col['up_missing']] = [.3, .4], [0, 1]
        X[1, col['gfs_7']] = np.nan
        Z = x4.event_features(X)
        np.testing.assert_allclose(Z[0], [7.5, 4.0, .3, 0])
        np.testing.assert_allclose(Z[1], [(6 + 8 + 9) / 3, 4.0, .4, 1])

    def test_kmeans_deterministic_and_frozen_assignment(self):
        rng = np.random.default_rng(0)
        Z = np.r_[rng.normal(0, .1, (50, 2)), rng.normal(5, .1, (50, 2))]
        c1, c2 = x4.kmeans(Z, 2, 2024, 3, 50), x4.kmeans(Z, 2, 2024, 3, 50)
        np.testing.assert_array_equal(c1, c2)
        mean, std = np.zeros(2), np.ones(2)
        lab = x4.assign(Z, mean, std, c1)
        self.assertEqual(len(np.unique(lab[:50])), 1)
        self.assertNotEqual(lab[0], lab[-1])
        later = np.array([[0.05, 0.0], [5.0, 5.1]])                          # a later pool keeps the identities
        np.testing.assert_array_equal(x4.assign(later, mean, std, c1), [lab[0], lab[-1]])

    def test_spearman_and_decisions(self):
        self.assertAlmostEqual(x4.spearman([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(x4.spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertIsNone(x4.spearman([1, 1, 1], [1, 2, 3]))
        cfg = dict(spearman_min=.5, wins_min=2, farms_min=2)
        U = np.array([[1, 2, 3, 4, 5, 6]] * 4, float)
        f = x4.d0_farm(U, [True, True, False], cfg)
        self.assertTrue(f['cond1'] and f['cond2'])
        bad = x4.d0_farm(np.r_[U[:3], np.zeros((1, 6))], [True, True, True], cfg)
        self.assertEqual(bad['status'], 'UNDETERMINED')
        neg = x4.d0_farm(U * np.array([[1], [-1], [1], [-1]]), [False, False, True], cfg)
        self.assertFalse(neg['cond1'] or neg['cond2'])
        self.assertEqual(x4.d0_outcome(dict(a=f, b=f, c=neg), cfg), 'GO_P1')
        self.assertEqual(x4.d0_outcome(dict(a=f, b=neg, c=neg), cfg), 'STOP_P1')
        self.assertEqual(x4.d0_outcome(dict(a=f, b=f, c=bad), cfg), 'INCOMPLETE')
        self.assertEqual(x4.top_groups([[1, 3, 3, 0], [1, 3, 3, 0]]), [1, 2])

    def make_tables(self):
        """Hourly-issue windows per farm for 2020-07..2021-01 with event-like columns and a pu label."""
        rng = np.random.default_rng(1)
        col = {n: i for i, n in enumerate(x3.PHYS_NAMES)}
        tables = {}
        for f in x4.FARMS:
            issue = np.arange(np.datetime64('2020-06-01T00', 'ns'), np.datetime64('2021-01-01T00', 'ns'), 2 * H)
            n = len(issue)
            X = rng.normal(size=(n, len(x3.PHYS_NAMES))).astype(np.float32)
            X[:, col['up_missing']] = rng.integers(0, 2, n)
            tables[f] = [dict(X=X, y=rng.uniform(0, 1, (n, 4)), issue=issue.astype(np.int64),
                              times=x4.label_times_of(issue).astype(np.int64), source='train')]
        return tables

    class FakeLGB:
        class LGBMRegressor:
            def __init__(self, **kw):
                self.kw = kw

            def fit(self, X, y, categorical_feature=None):
                self.m = float(np.mean(y)) + 1e-3 * len(y)                   # depends on the training rows
                return self

            def predict(self, X):
                return np.full(len(X), self.m)

    def test_d0_target_freezes_first_block_and_isolates_labels(self):
        tables = self.make_tables()
        cfg = dict(x1.read(x4.PROTOCOL)['d0'], deployment={f: '2021-01-01' for f in x4.FARMS})
        calls = []
        real = x4.kmeans

        def counting(*a, **k):
            calls.append(len(a[0]))
            return real(*a, **k)
        seen = []
        real_proxy = x4.proxy_mae

        def spy(lgb, c, threads, trees, tr, ev):
            seen.append(len(tr['y']))
            return real_proxy(lgb, c, threads, trees, tr, ev)
        with patch.object(x4, 'kmeans', counting), patch.object(x4, 'proxy_mae', spy):
            res = x4.d0_target(self.FakeLGB, dict(cfg, kmeans_n_init=1, kmeans_iters=20), 1, 'Kelmarsh', tables,
                               float('inf'), smoke=True)
        self.assertEqual(len(calls), 1)                                      # frozen after the first block
        self.assertEqual(res['status'], 'DETERMINED')
        self.assertEqual(len(res['blocks']), 4)
        self.assertEqual(np.asarray(res['U']).shape, (4, 6))
        self.assertEqual(len(res['wins']), 3)
        for blk in res['blocks']:
            cut = np.datetime64(blk['cut'])
            self.assertGreaterEqual(blk['target_dropped_cross_cut'], 1)       # windows crossing c dropped (R2-B2)
            self.assertLess(cut, np.datetime64(blk['block'][1]))
        self.assertEqual(len(seen), 4 * (1 + 6 + 1) + 3)                     # base, 6 groups, random; top-2 from block 2

    def test_d0_timeout_is_incomplete_not_negative(self):
        with self.assertRaises(TimeoutError):
            x4.d0_target(self.FakeLGB, dict(x1.read(x4.PROTOCOL)['d0'], kmeans_n_init=1, kmeans_iters=5), 1,
                         'Kelmarsh', self.make_tables(), deadline=0.0, smoke=True)

    def test_d0_empty_target_and_empty_source_are_undetermined(self):
        cfg = dict(x1.read(x4.PROTOCOL)['d0'], deployment={f: '2021-01-01' for f in x4.FARMS})
        for kind in ('target', 'source', 'nonfinite_source'):
            with self.subTest(kind=kind):
                tables = self.make_tables()
                names = ['Kelmarsh'] if kind == 'target' else ['Penmanshiel', 'HoT']
                for farm in names:
                    row = tables[farm][0]
                    if kind == 'nonfinite_source':
                        row['X'][:, x3.PHYS_NAMES.index('p_hist_23')] = np.nan
                    else:
                        tables[farm] = [{k: (v[:0] if isinstance(v, np.ndarray) else v) for k, v in row.items()}]
                with patch.object(x4, 'kmeans') as km, patch.object(x4, 'proxy_mae') as proxy:
                    res = x4.d0_target(self.FakeLGB, cfg, 1, 'Kelmarsh', tables, float('inf'), True)
                self.assertEqual(res['status'], 'UNDETERMINED')
                self.assertEqual(x4.d0_outcome({'Kelmarsh': res}, cfg), 'INCOMPLETE')
                km.assert_not_called()
                proxy.assert_not_called()

    def test_d0_last_fit_cannot_return_a_verdict_after_deadline(self):
        cfg = dict(x1.read(x4.PROTOCOL)['d0'], deployment={f: '2021-01-01' for f in x4.FARMS})
        now, calls = [0.0], [0]

        def proxy(*args):
            calls[0] += 1
            if calls[0] == 35:  # the last top-2 fit, after 4*(base+6 groups+random)+3 top-2
                now[0] = 11.0
            return float(calls[0])

        with patch.object(x4.time, 'monotonic', side_effect=lambda: now[0]), \
                patch.object(x4, 'proxy_mae', side_effect=proxy):
            with self.assertRaises(TimeoutError):
                x4.d0_target(self.FakeLGB, dict(cfg, kmeans_n_init=1, kmeans_iters=20), 1,
                             'Kelmarsh', self.make_tables(), deadline=10.0, smoke=True)
        self.assertEqual(calls[0], 35)


class Identity(unittest.TestCase):
    def test_saved_model_handoff_rejects_changed_or_missing_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root / 'selected'
            selected.mkdir()
            config = selected / 'config.json'
            weights = selected / 'model.safetensors'
            config.write_text('{"model_type":"chronos2"}', encoding='utf-8')
            weights.write_bytes(b'original checkpoint')
            init = dict(root=str(root), model_sha256={
                p.relative_to(root).as_posix(): x1.digest(p) for p in (config, weights)})
            self.assertEqual(x4.checked_model(init), selected)
            weights.write_bytes(b'different checkpoint')
            with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
                x4.checked_model(init)
            weights.unlink()
            with self.assertRaises(FileNotFoundError):
                x4.checked_model(init)

    def test_formal_gate_needs_smoke_report_of_this_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            args = SimpleNamespace(base=base, mode='a', smoke=False, allow_dirty=False,
                                   protocol_sha256=x1.digest(x4.PROTOCOL))
            with patch.object(x4.x2, 'git_identity', return_value=('reviewed', False)):
                with self.assertRaisesRegex(ValueError, 'smoke chain'):
                    x4.start_run(args, 'a')
                root = base / 'smoke_report'
                root.mkdir()
                x1.save(root / 'launch.json', dict(commit='reviewed', script_sha256=x1.digest(x3.__file__),
                                                   protocol_sha256=args.protocol_sha256, mode='report', smoke=True))
                for name in ('results.json', 'dev_decisions.json'):
                    x1.save(root / name, dict(v=1))
                x3.finish(root, x3.x2.Store(root), dict(results_sha256=x1.digest(root / 'results.json'),
                                                        dev_decisions_sha256=x1.digest(root / 'dev_decisions.json')))
                with self.assertRaisesRegex(ValueError, 'smoke chain'):          # X3's script is not X4's
                    x4.start_run(args, 'a')
            self.assertFalse((base / 'a').exists())

    def test_upstream_checks_meta_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            args = SimpleNamespace(base=base, mode='report', smoke=False)
            ident = dict(commit='c', script_sha256='s', protocol_sha256='p', smoke=False)
            for mode in ('a', 'report'):
                (base / mode).mkdir()
                x1.save(base / mode / 'launch.json', dict(ident, mode=mode))
            x1.save(base / 'a' / 'a_meta.json', dict(v=1))
            x3.finish(base / 'a', x3.x2.Store(base / 'a'), dict(meta_sha256=x1.digest(base / 'a' / 'a_meta.json')))
            self.assertIsNotNone(x4.upstream(args, 'a'))
            x1.save(base / 'a' / 'a_meta.json', dict(v=2))
            with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
                x4.upstream(args, 'a')

    def test_x3_reference_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            root = logs / 'X3/x3'
            shas = {}
            for mode in ('q1', 'q2', 'q4'):
                d = root / mode
                d.mkdir(parents=True)
                x1.save(d / 'launch.json', dict(commit='abc', smoke=False, mode=mode))
                extra = {}
                if mode == 'q1':
                    x1.save(d / 'selection.json', dict(selection={}))
                    extra['selection_sha256'] = x1.digest(d / 'selection.json')
                if mode == 'q4':
                    x1.save(d / 'q4_meta.json', dict(K={}))
                    extra['meta_sha256'] = x1.digest(d / 'q4_meta.json')
                x3.finish(d, x3.x2.Store(d), extra)
                shas[mode] = x1.digest(d / 'completed.json')
            proto = dict(references=dict(x3=dict(dir='X3/x3', commit='abc', completed_sha256=shas, q2_spec_sha256={})))
            ref = x4.X3Run(logs, proto)
            self.assertEqual(ref.q4_meta, dict(K={}))
            x1.save(root / 'q2' / 'launch.json', dict(commit='other', smoke=False, mode='q2'))
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                x4.X3Run(logs, proto)


if __name__ == '__main__':
    unittest.main()
