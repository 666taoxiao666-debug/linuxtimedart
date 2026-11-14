import unittest
import json
from pathlib import Path
import tempfile
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import prepare_hot_dev as hot
from scripts import train_g2b as train


class RetrofitDiagnosticTests(unittest.TestCase):
    def test_registered_thresholds_and_undetermined(self):
        def select(values):
            return hot.choose_retrofit_case({str(i): {'S2': v} for i, v in enumerate(values)})['case']
        self.assertEqual(select([-.05]*11+[0.]*10), 'A')
        self.assertEqual(select([.05]*11+[0.]*10), 'A')
        self.assertEqual(select([.049]*14+[.08]*7), 'B')
        self.assertEqual(select([-.08]*10+[.08]*10+[None]), 'STOP')

    def test_calendar_physical_period_boundaries(self):
        first, end = hot.works_bounds('2022-07-23', '2022-08-01')
        times = pd.DatetimeIndex(['2021-04-30', '2021-05-01', '2022-07-23',
                                  '2022-08-01', '2022-08-02', '2023-05-01', '2024-01-01'], tz='UTC')
        masks = hot.retrofit_masks(times, first, end)
        np.testing.assert_array_equal(masks['P0'], [False, True, False, False, False, False, False])
        np.testing.assert_array_equal(masks['P1'], [False, False, False, False, True, False, False])
        np.testing.assert_array_equal(masks['P2'], [False, False, False, False, False, True, False])

    def test_equal_weight_bins_and_incomplete_not_imputed(self):
        first, end = hot.works_bounds('2022-07-23', '2022-08-01')
        before = pd.date_range('2021-05-01T01:00', periods=12, freq='h', tz='UTC')
        after = pd.date_range('2023-05-01T01:00', periods=12, freq='h', tz='UTC')
        u = pd.DataFrame({'wind_speed_mps': np.tile(np.arange(5.25, 11, .5), 2),
                          'power_kw': [100.]*12+[90.]*12}, index=before.append(after))
        result = hot.retrofit_scores(u, first, end)
        self.assertAlmostEqual(result['S2'], -.1)
        self.assertIsNone(result['S1'])
        u.iloc[-1, u.columns.get_loc('power_kw')] = np.nan
        self.assertIsNone(hot.retrofit_scores(u, first, end)['S2'])


class G2bExecutionTests(unittest.TestCase):
    def test_recipe_and_seed_budget(self):
        protocol = json.loads((train.ROOT/'configs/g2b_protocol.json').read_text())
        train.validate_protocol(protocol)
        self.assertEqual(sum(map(len, protocol['seeds'].values())), 14)
        protocol['seeds']['N'].append(2029)
        with self.assertRaises(ValueError):
            train.validate_protocol(protocol)

    def test_arm_channel_identity_and_power_last(self):
        x = np.broadcast_to(np.arange(50), (2, 24, 50)).copy()
        pack = dict(x21=x[..., :21], x50=x, truth=np.zeros((2, 4)), retrofit=np.array([0, 1], np.float32))
        for arm, count in [('B0', 9), ('B', 21), ('N0', 38), ('N', 50)]:
            key = 'x21' if arm in ('B0', 'B') else 'x50'
            before = train.arm_pack(pack, arm, False)[key]
            after = train.arm_pack(pack, arm, True)[key]
            self.assertEqual(before.shape[-1], count)
            self.assertEqual(after.shape[-1], count+1)
            np.testing.assert_array_equal(before[..., -1], after[..., -1])
            np.testing.assert_array_equal(after[:, 0, -2], [0, 1])
            np.testing.assert_array_equal(before[..., :-1], after[..., :-2])

    def test_scoped_preparation_does_not_weaken_development_guard(self):
        scoped = train.scoped_functions(hot, YEARS=(2023, 2024, 2025), CUTOFF=pd.Timestamp('2026-01-01', tz='UTC'))
        self.assertEqual(scoped.development_year(2025), 2025)
        with self.assertRaises(ValueError):
            scoped.development_year(2026)
        with self.assertRaises(ValueError):
            hot.development_year(2024)

    def test_sealed_requires_authorization_before_any_file_read(self):
        with patch.object(train, 'frozen_gate', side_effect=AssertionError('access before permission')):
            with self.assertRaisesRegex(ValueError, 'authorization'):
                train.prepare_sealed(Path('/x'), Path('/x'), 'bad', Path('/x'), Path('/x'), 'bad', None)

    def test_sealed_one_shot_marker_precedes_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'test_evaluation_started.json').write_text('{}')
            with patch.object(train, 'frozen_gate', return_value=({}, {})), patch.object(train, 'scoped_functions', side_effect=AssertionError('labels opened')):
                with self.assertRaises(FileExistsError):
                    train.prepare_sealed(root, root, 'x', root/'new', root/'gfs.json', 'x', 'G2b-2024-2025-once')

    @unittest.skipUnless(sys.platform == 'linux', 'shared statistics module requires Linux fcntl')
    def test_joint_holm_uses_worst_year_and_both_ci(self):
        reports = {str(y): {'primary': {q: dict(gain=1., ci95_low=.1, newey_west={'p_value_two_sided': .01})
                                        for q in ('Q1h', 'Qe')}} for y in (2024, 2025)}
        self.assertTrue(train.joint_test_decisions(reports)['Q1h']['passed'])
        reports['2025']['primary']['Q1h']['newey_west']['p_value_two_sided'] = .2
        self.assertFalse(train.joint_test_decisions(reports)['Q1h']['passed'])
        reports['2024']['primary']['Qe']['ci95_low'] = -.01
        self.assertFalse(train.joint_test_decisions(reports)['Qe']['passed'])

    def test_frozen_gfs_scaler_and_protected_cycle(self):
        hours = np.array(['2024-01-01T06', '2025-12-31T23'], dtype='datetime64[h]')
        hourly = train.frozen_hourly(hours, {'2024010100': np.full((6, 13), 10.)}, {'nwp_mean': 7., 'nwp_scale': 2.})
        self.assertEqual(hourly['nwp_mean'], 7.)
        self.assertEqual(hourly['nwp_scale'], 2.)
        self.assertEqual(hourly['cycle_ok'][1], 0.)
        x = np.zeros((1, 24, 21), np.float32)
        # A single repeated history position checks the exact normalized forecast slot.
        out = train.prep.nwp_input(x, hourly, np.zeros((1, 24), int))
        self.assertAlmostEqual(out[0, 0, 20], 1.5)

    def test_common_cohorts_exclude_construction_and_split_crossing(self):
        hours = pd.date_range('2023-06-26T01', periods=192, freq='h').to_numpy()
        raw = np.ones((len(hours), 9))
        raw[:, -1] = np.arange(len(hours))
        stats = dict(upstream_fill=[0.]*11, mean=[0.]*21, scale=[1.]*21)
        hourly = train.frozen_hourly(hours, {}, dict(nwp_mean=0., nwp_scale=1.))
        with tempfile.TemporaryDirectory() as tmp:
            works = Path(tmp)/'works.csv'
            works.write_text('Turbine,start,end\nT01,2023-06-29,2023-06-29\n')
            packs = train.common_packs(raw, hours, ['T01'], np.ones((len(hours), 11)), hourly, stats,
                                       works, train.case_bounds('A'), '2024-01-01')
        self.assertEqual(set(packs), {'train', 'val'})
        for split, pack in packs.items():
            self.assertTrue(train.prep.period_mask(pack['times'].astype('datetime64[ns]'), train.case_bounds('A')[split], hour_end=True).all())
            issue = pd.DatetimeIndex(pack['issue'].astype('datetime64[ns]')).tz_localize('UTC')
            excluded, _ = hot.window_works(issue, *hot.works_bounds('2023-06-29', '2023-06-29'))
            self.assertFalse(excluded.any())
            for arm in ('B', 'N', 'N0', 'B0'):
                np.testing.assert_array_equal(train.arm_pack(pack, arm, True)['truth'], pack['truth'])
        # A changed validation power label must remain exact, never clipped or imputed.
        self.assertGreater(packs['val']['truth'].max(), 100.)

    @unittest.skipUnless(sys.platform == 'linux', 'formal lock requires Linux fcntl')
    def test_exact_budget_and_immutable_freeze_without_real_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            diagnostic = root/'diagnostic'
            diagnostic.mkdir()
            (diagnostic/'completed.json').write_text(json.dumps(dict(artifacts={}, decision=dict(case='A'))))
            prepared = root/'prepared'
            prepared.mkdir()
            (prepared/'completed.json').write_text('{}')
            pack = dict(x21=np.zeros((2, 24, 21), np.float32), x50=np.zeros((2, 24, 50), np.float32),
                truth=np.zeros((2, 4)), retrofit=np.array([0, 1]), times=np.ones((2, 4), np.int64))
            calls = []
            def fake_fit(arm, seed, tr, va, cfg, directory, device):
                calls.append((arm, seed))
                directory.mkdir()
                (directory/'selected.pt').write_bytes(f'{arm}-{seed}'.encode())
                return dict(arm=arm, seed=seed, validation_mae=1., sha256=train.prep.digest(directory/'selected.pt'))
            with patch.object(train, 'input_state', return_value=({'train': pack, 'val': pack}, {})), patch.object(train.runner, 'fit_one', side_effect=fake_fit):
                protocol_sha = train.prep.digest(train.ROOT/'configs/g2b_protocol.json')
                train.fit_development(root, prepared, diagnostic, root/'run', protocol_sha, 'cpu')
                with self.assertRaises(ValueError):
                    train.fit_development(root, prepared, diagnostic, root/'run', protocol_sha, 'cpu')
            self.assertEqual(len(calls), 14)
            self.assertEqual(len(set(calls)), 14)
            self.assertEqual(sum(a == 'N' for a, _ in calls), 5)
            self.assertFalse((root/'run/test_evaluation_started.json').exists())
            frozen = json.loads((root/'run/selection_freeze.json').read_text())
            train.runner.check_freeze(frozen, 14, root/'run')
            (root/'run/B_2024/selected.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                train.runner.check_freeze(frozen, 14, root/'run')

    def test_all_arm_torch_forward_backward(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch available on server only')
        protocol = json.loads((train.ROOT/'configs/g2b_protocol.json').read_text())
        pack = dict(x21=np.zeros((2, 24, 21), np.float32), x50=np.zeros((2, 24, 50), np.float32),
                    truth=np.ones((2, 4), np.float32), retrofit=np.array([0, 1], np.float32))
        for retrofit in (False, True):
            for arm in ('B0', 'B', 'N0', 'N'):
                p = train.arm_pack(pack, arm, retrofit)
                model = train.make_model(arm, p, protocol['fit'], 'cpu')
                result = model(torch.from_numpy(p['x21' if arm in ('B0', 'B') else 'x50']))
                self.assertEqual(tuple(result.shape[:2]), (2, 4))
                result.square().mean().backward()
                self.assertTrue(all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None))


if __name__ == '__main__':
    unittest.main()
