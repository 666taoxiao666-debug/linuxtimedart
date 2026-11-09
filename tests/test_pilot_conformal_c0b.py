import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scripts import pilot_conformal_c0b as c


def pack(n, date, turbines=None):
    issue = np.datetime64(date, 'ns') + np.arange(n)*np.timedelta64(1, 'h')
    return dict(prediction=np.zeros((n, 4)), truth=np.tile(np.arange(1, n+1)[:, None], (1, 4)).astype(float),
                issue=issue, times=issue[:, None]+np.arange(1, 5)*np.timedelta64(1, 'h'),
                turbines=np.ones(n, int) if turbines is None else turbines, missing=np.zeros(n, bool))


class C0bTests(unittest.TestCase):
    def test_r1_counterexample_F_and_finite_preservation(self):
        e = np.arange(1, 5001)
        q, capped = c.finite_quantile(e, np.full(5000, .05), 20, .005)
        self.assertEqual((q, capped), (5000, True))
        for weights, own, a in [(np.ones(5000), 1, .1), (np.full(5000, .05), 20, .2)]:
            q, capped = c.finite_quantile(e, weights, own, a)
            self.assertEqual(q, c.c0.weighted_quantile(e, weights, own, a))
            self.assertFalse(capped)
        self.assertEqual(c.finite_quantile([1, 2], [1, 1], 1, .005), (2., True))

    def test_both_year_cutoffs_and_issue_availability(self):
        for year in (2022, 2023):
            issue = np.array([f'{year}-01-01'], dtype='datetime64[ns]')
            past = np.array([f'{year-1}-12-31T23:00'], dtype='datetime64[ns]')
            c.causal_check(past, issue, year)
            with self.assertRaises(ValueError): c.causal_check(issue, issue, year)
            with self.assertRaises(ValueError): c.causal_check(past, past-np.timedelta64(1, 'h'), year)

    def test_nonfinite_metrics_and_scale_fail(self):
        with self.assertRaises(ValueError): c.checked_metrics(np.array([np.inf]), np.ones(1), np.zeros(1))
        with self.assertRaises(ValueError): c.finite_quantile([np.nan], [1], 1, .1)
        with self.assertRaises(ValueError): c.scales(np.zeros((3, 4)), np.ones((3, 4)))
        with self.assertRaises(ValueError): c.scales(np.ones((3, 4)), np.full((3, 4), np.nan))
        self.assertFalse(c.finite_leq(np.inf, np.inf))

    def test_M_B3_same_F_scale_and_actual_scaled_feedback(self):
        s, t, e = pack(100, '2021-01-01'), pack(100, '2021-12-01'), pack(40, '2022-01-01')
        # One power bin; <=500 sources, equal weights => identical M and B3.
        t['truth'] *= 2
        e['truth'][:] = 150  # covered after scaling, missed before scaling
        report, radii, caps, kappa = c.intervals(s, t, e, (np.ones(100), np.ones(40)), 2022)
        np.testing.assert_array_equal(kappa, [2]*4)
        for v in c.VARIANTS:
            np.testing.assert_array_equal(radii['M-'+v], radii['B3-'+v])
            np.testing.assert_array_equal(caps['M-'+v], caps['B3-'+v])
        # Emitted S+A covers150, so alpha rises and radius shrinks, not grows.
        self.assertLess(radii['M-S+A'][-1, 0], radii['M-S+A'][0, 0])
        self.assertGreater(radii['M-A'][-1, 0], radii['M-A'][0, 0])
        self.assertEqual(report['M-S']['pooled']['cap_fraction'], 0.)

    def test_delayed_feedback_and_turbine_isolation(self):
        s, t, e = pack(500, '2021-01-01'), pack(100, '2021-12-01'), pack(12, '2022-01-01')
        _, before, _, _ = c.intervals(s, t, e, (np.ones(500), np.ones(12)), 2022)
        e['truth'][0, 3] = 1e6
        _, after, _, _ = c.intervals(s, t, e, (np.ones(500), np.ones(12)), 2022)
        for m in ('M-A', 'B3-A', 'M-S+A', 'B3-S+A', 'B2-A'):
            np.testing.assert_array_equal(before[m][:4], after[m][:4])
        e['turbines'][0] = 2
        _, isolated, _, _ = c.intervals(s, t, e, (np.ones(500), np.ones(12)), 2022)
        e['truth'][0, 3] = 1
        _, control, _, _ = c.intervals(s, t, e, (np.ones(500), np.ones(12)), 2022)
        for m in isolated:
            np.testing.assert_array_equal(isolated[m][1:], control[m][1:])
        e['times'][0, 3] += np.timedelta64(1, 'h')
        with self.assertRaises(ValueError): c.intervals(s, t, e, (np.ones(500), np.ones(12)), 2022)

    def test_selection_and_verdict(self):
        means = {f'M-{v}': dict(coverage=.9, winkler_kw=100.) for v in c.VARIANTS}
        self.assertEqual(c.select_variant(means), 'S')
        means['M-A']['winkler_kw'] = 90
        self.assertEqual(c.select_variant(means), 'A')
        means['M-A']['coverage'] = .95
        self.assertEqual(c.select_variant(means), 'S')
        means = {m: dict(coverage=.89, winkler_kw=100.) for m in ('B0', 'B1', 'B2-A', 'B3-S')}
        means['M-S'] = dict(coverage=.9, winkler_kw=94.)
        self.assertEqual(c.verdict(means, 'S')['decision'], 'continue')
        means['B3-S']['winkler_kw'] = 94.
        self.assertFalse(c.verdict(means, 'S')['C3'])
        means['M-S']['winkler_kw'] = np.inf
        self.assertEqual(c.verdict(means, 'S')['decision'], 'stop')

    def test_selection_freeze_precedes_all_2023_computation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaises(ValueError): c.require_selection(out)
            seen = []
            def fake_phase(data, dest, year, variants, device):
                if year == 2023:
                    self.assertEqual(c.require_selection(out)['selected'], 'S')
                    self.assertEqual(variants, ('S',))
                seen.append(year)
                means = {m: dict(coverage=.9, winkler_kw=100.) for m in
                         ('M-S', 'M-A', 'M-S+A', 'B0', 'B1', 'B2-A', 'B3-S')}
                result = {'mean_90': means}
                c.save(dest/'result.json', result)
                return result, {'source_code': {}, 'input_hashes': {}}
            with patch.object(c, 'phase', side_effect=fake_phase): c.run(Path(tmp), out, None)
            self.assertEqual(seen, [2022, 2023])
            (out/'selection_2022/result.json').write_text('{}')
            with self.assertRaises(ValueError): c.require_selection(out)


if __name__ == '__main__':
    unittest.main()
