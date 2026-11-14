import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

from scripts import prepare_hot_dev as hot


class HotDevelopmentTests(unittest.TestCase):
    def test_protected_year_rejected_before_zip_open(self):
        z = Mock()
        for y in (2015, 2024, 2025, 2026):
            with self.assertRaises(ValueError):
                hot.read_month(z, y, 1, [1])
        z.open.assert_not_called()

    def test_construction_exact_boundaries(self):
        a, b = hot.works_bounds('2022-07-23', '2022-08-01')
        self.assertEqual(b, pd.Timestamp('2022-08-02', tz='UTC'))
        issue = [a-pd.Timedelta(hours=4), a-pd.Timedelta(hours=3), b,
                 b+pd.Timedelta(hours=23), b+pd.Timedelta(hours=24)]
        excluded, flag = hot.window_works(issue, a, b)
        np.testing.assert_array_equal(excluded, [False, True, True, True, False])
        np.testing.assert_array_equal(flag, [0, 0, 1, 1, 1])

    def ticks(self):
        t = pd.DataFrame({'timestamp': pd.date_range('2022-01-01', periods=12, freq='10min', tz='UTC')})
        for c in hot.enrich.REQUIRED_TICK_COLUMNS:
            t[c] = np.arange(12, dtype=float)
        return t

    def test_hourly_labels_last10_and_incomplete_power(self):
        t = self.ticks()
        t.loc[7, 'power_kw'] = np.nan
        h = hot.aggregate(t)
        self.assertEqual(h.timestamp_hour_end_utc.iloc[0], pd.Timestamp('2022-01-01T01:00Z'))
        self.assertEqual(h.power_kw.iloc[0], 2.5)
        self.assertEqual(h.power_last10_kw.iloc[0], 5)
        self.assertTrue(np.isnan(h.power_kw.iloc[1]))
        self.assertEqual(h.power_last10_kw.iloc[1], 11)
        self.assertEqual(t.power_kw.isna().sum(), 1)

    def test_duplicate_and_off_grid_ticks_refused(self):
        t = self.ticks()
        with self.assertRaises(ValueError):
            hot.aggregate(pd.concat([t, t.iloc[[0]]]))
        t.loc[0, 'timestamp'] += pd.Timedelta(minutes=1)
        with self.assertRaises(ValueError):
            hot.aggregate(t)

    def test_2024_hour_label_not_retained(self):
        t = self.ticks()
        t['timestamp'] = pd.date_range('2023-12-31T22:00', periods=12, freq='10min', tz='UTC')
        h = hot.aggregate(t)
        self.assertEqual(len(h), 1)
        self.assertTrue((h.timestamp_hour_end_utc < hot.CUTOFF).all())

    def test_scada_channel_order_and_no_pitch_imputation(self):
        h = hot.aggregate(self.ticks())
        x = hot.scada_values(h)
        self.assertEqual(x.shape, (2, 9))
        np.testing.assert_array_equal(x[:, -1], h.power_kw)
        np.testing.assert_array_equal(x[:, 6], h.power_last10_kw)
        h.loc[0, 'pitch_a_deg'] = np.nan
        self.assertTrue(np.isnan(hot.scada_values(h)[0, 4]))

    def test_month_join_rejects_outside_year_and_duplicates(self):
        class Archive:
            def __init__(self, date, duplicate=False):
                self.date, self.duplicate = date, duplicate
            def open(self, name):
                columns = hot.MAPPING[name.split('_2016')[0]]
                header = ','.join(['TimeStamp', 'StationId']+list(columns))
                row = ','.join([self.date, '1']+['2']*len(columns))
                return io.BytesIO((header+'\n'+row+'\n'+(row+'\n' if self.duplicate else '')).encode())
        for z in (Archive('2024-01-01'), Archive('2016-01-01', True)):
            with self.assertRaises(ValueError):
                hot.read_month(z, 2016, 1, [1])
        self.assertEqual(len(hot.read_month(Archive('2016-01-01'), 2016, 1, [1])), 1)

    def test_fresh_output_and_data_root_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for out in [root, root/'datasets'/'new']:
                with self.assertRaises(ValueError):
                    hot.run(root, out)

    def test_common_queue_missing_labels_and_construction(self):
        hours = pd.date_range('2022-01-01T01:00', periods=120, freq='h')
        f = pd.DataFrame({'timestamp_hour_end_utc': hours.tz_localize('UTC'), 'turbine_id': 'T01'})
        for c in hot.enrich.REQUIRED_TICK_COLUMNS:
            f[c] = 5.0
        f['power_last10_kw'], f['wind_last10_mps'] = 5., 5.
        f.loc[30, 'power_kw'] = np.nan
        bounds = {'T01': hot.works_bounds('2022-01-03', '2022-01-03')}
        with tempfile.TemporaryDirectory() as d:
            summary = hot.prepare_windows(f, hours, bounds, Path(d))
            with np.load(Path(d)/'train_provisional_windows.npz') as p:
                self.assertTrue(len(p['starts']) > 0)
                for s in p['starts']:
                    self.assertFalse(s <= 30 < s+28)
                issue = pd.to_datetime(p['issue'], utc=True)
                self.assertFalse(hot.window_works(issue, *bounds['T01'])[0].any())
            with np.load(Path(d)/'scada_rows.npz') as p:
                self.assertTrue(np.isnan(p['raw'][30, -1]))
            self.assertGreater(summary['excluded_works_total'], 0)


if __name__ == '__main__':
    unittest.main()
