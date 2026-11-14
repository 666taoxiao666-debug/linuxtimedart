import unittest

import numpy as np
import pandas as pd

from scripts import prepare_hot_dev as hot


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


if __name__ == '__main__':
    unittest.main()
