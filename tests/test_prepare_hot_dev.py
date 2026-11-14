import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from scripts import prepare_hot_dev as hot


class HotDevelopmentTests(unittest.TestCase):
    def test_registered_boundary_resolution_four_classes_and_year(self):
        stamp = pd.Timestamp('2021-01-01', tz='UTC')
        a = pd.DataFrame({'timestamp': [stamp], 'StationId': [1], 'power_kw': [np.nan],
                          'same': [3.], 'empty': [np.nan], 'rounded': [1.2], 'conflict': [5.]})
        b = a.copy()
        b.loc[0, ['power_kw', 'rounded', 'conflict']] = [10., 1.24, 7.]
        a.attrs['raw_boundary_cells'] = {(stamp, 1, 'rounded'): '1.2'}
        b.attrs['raw_boundary_cells'] = {(stamp, 1, 'rounded'): '1.24'}
        audit = dict(cells=hot.Counter(), masked_hours=set(), duplicate_keys=0)
        result = hot.merge_boundary(a, b, audit)
        self.assertEqual(len(result), 1)
        self.assertEqual(result['rounded'].iloc[0], 1.24)
        self.assertEqual(result['same'].iloc[0], 3.)
        self.assertTrue(np.isnan(result.power_kw.iloc[0]))
        self.assertTrue(np.isnan(result['conflict'].iloc[0]))
        self.assertEqual(len(audit['masked_hours']), 1)
        self.assertEqual(audit['cells']['one_missing'], 1)
        self.assertEqual(audit['cells']['both_missing'], 1)
        b['conflict'] = 5.
        clean = hot.merge_boundary(a, b, dict(cells=hot.Counter(), masked_hours=set(), duplicate_keys=0))
        self.assertEqual(clean.power_kw.iloc[0], 10.)
        self.assertTrue(np.isnan(a.power_kw.iloc[0]))

    def test_conflict_masks_whole_hour_even_if_nonpower_column(self):
        a = self.ticks().iloc[:6].copy()
        a['timestamp'] = pd.date_range('2020-12-31T23:10', periods=6, freq='10min', tz='UTC')
        a['StationId'] = 1
        b = a.iloc[[-1]].copy()
        b['wind_speed_mps'] = 99.
        audit = dict(cells=hot.Counter(), masked_hours=set(), duplicate_keys=0)
        merged = hot.merge_boundary(a, b, audit)
        h = hot.aggregate(merged.drop(columns='StationId'))
        self.assertTrue(np.isnan(h.power_kw.iloc[0]))
        self.assertEqual(audit['masked_hours'], {('2021-01-01 00:00:00+00:00', 1)})

    def test_boundary_raw_classification_and_rounding(self):
        self.assertEqual(hot.boundary_class('1.00', '1'), ('same', '0.00', False))
        self.assertEqual(hot.boundary_class('', 'NaN'), ('both_missing', None, False))
        self.assertEqual(hot.boundary_class('', '42'), ('one_missing', None, False))
        self.assertEqual(hot.boundary_class('1.2', '1.24'), ('different', '0.04', False))
        self.assertEqual(hot.boundary_class('1.2', '1.26'), ('different', '0.06', True))
        with self.assertRaises(ValueError):
            hot.boundary_class('inf', '1')

    def test_missing_value_is_conflict_in_original_merge(self):
        a = pd.DataFrame({'timestamp': [pd.Timestamp('2021-01-01', tz='UTC')],
                          'StationId': [1], 'power_kw': [np.nan]})
        b = a.copy()
        b['power_kw'] = 1.
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            hot.merge_boundary(a, b)

    def test_diagnostic_cross_year_stop_and_protected_sentinel(self):
        class Archive:
            def __init__(self, path):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def open(self, name):
                table, y, m = name.removesuffix('.csv').rsplit('_', 2)
                lo = pd.Timestamp(year=int(y), month=int(m), day=1)
                hi = lo + pd.offsets.MonthBegin(1)
                columns = list(hot.MAPPING[table])
                rows = [','.join(['TimeStamp', 'StationId']+columns)]
                for stamp in (lo, hi):
                    if table == 'tblSCTurTemp' and int(y) == 2021 and int(m) == 1 and stamp == lo:
                        continue  # Absent table row becomes NaN after outer join.
                    value = 'protected-not-numeric' if stamp.year >= 2024 else '1'
                    if int(y) == 2021 and int(m) == 1 and stamp == lo:
                        value = '100'
                    rows.append(','.join([str(stamp), '1']+[value]*len(columns)))
                return io.BytesIO(('\n'.join(rows)+'\n').encode())
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'diagnostic'
            with patch.object(hot, 'raw_inventory', return_value=Path(d)), \
                 patch.object(hot.zipfile, 'ZipFile', Archive):
                hot.diagnose_boundaries(Path(d), out)
            report = hot.json.loads((out/'diagnostic.json').read_text())
            self.assertEqual(report['duplicate_keys'], 95)
            self.assertEqual(report['beyond_rounding_keys'], 1)
            self.assertIn('entire_annual_boundary_batch_differs', report['stop_reasons'])
            self.assertIn('beyond_rounding_keys_exceed_one_percent', report['stop_reasons'])
            self.assertNotIn('protected-not-numeric', (out/'diagnostic.json').read_text())
            temp = next(g for g in report['groups'] if g['table'] == 'tblSCTurTemp'
                        and g['boundary'].startswith('2021-01-01'))
            self.assertEqual(temp['columns']['wtc_AmbieTmp_mean']['one_missing'], 1)

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
        t = pd.DataFrame({'timestamp': pd.date_range('2022-01-01T00:10', periods=12, freq='10min', tz='UTC')})
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
        t = self.ticks().iloc[:11].copy()
        t['timestamp'] = pd.date_range('2023-12-31T22:10', periods=11, freq='10min', tz='UTC')
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

    def test_month_and_year_boundary_merge_preserves_physical_hour(self):
        for end in ('2016-02-01', '2017-01-01'):
            t = self.ticks().iloc[:6].copy()
            t['timestamp'] = pd.date_range(pd.Timestamp(end, tz='UTC')-pd.Timedelta(minutes=50),
                                          periods=6, freq='10min')
            t['StationId'] = 1
            merged = hot.merge_boundary(t, t.iloc[[-1]].copy())
            self.assertEqual(len(merged), 6)
            h = hot.aggregate(merged.drop(columns='StationId'))
            self.assertEqual(len(h), 1)
            self.assertEqual(h.timestamp_hour_end_utc.iloc[0], pd.Timestamp(end, tz='UTC'))
            self.assertEqual(h.power_kw.iloc[0], 2.5)
            bad = t.iloc[[-1]].copy()
            bad['power_kw'] += 1
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                hot.merge_boundary(t, bad)

    def test_protected_boundary_values_never_converted(self):
        class Archive:
            def open(self, name):
                columns = hot.MAPPING[name.split('_2023')[0]]
                header = ','.join(['TimeStamp', 'StationId']+list(columns))
                good = ','.join(['2023-12-31 23:50:00', '1']+['2']*len(columns))
                forbidden = ','.join(['2024-01-01 00:00:00', 'NOT_NUMERIC']+['DO_NOT_CONVERT']*len(columns))
                return io.BytesIO((header+'\n'+good+'\n'+forbidden+'\n').encode())
        result = hot.read_month(Archive(), 2023, 12, [1])
        self.assertEqual(len(result), 1)
        self.assertTrue((result.timestamp < hot.CUTOFF).all())

    def test_entire_missing_hour_is_retained_as_missing_label(self):
        first = self.ticks().iloc[:6].copy()
        last = first.copy()
        last['timestamp'] += pd.Timedelta(hours=2)
        h = hot.aggregate(pd.concat([first, last]))
        self.assertEqual(len(h), 3)
        self.assertTrue(np.isnan(h.power_kw.iloc[1]))
        self.assertTrue(np.isnan(h.power_last10_kw.iloc[1]))
        np.testing.assert_array_equal(h.power_kw.iloc[[0, 2]], [2.5, 2.5])

    def test_abbreviated_midnight_at_month_boundary(self):
        class Archive:
            def open(self, name):
                columns = hot.MAPPING[name.split('_2017')[0]]
                header = ','.join(['TimeStamp', 'StationId']+list(columns))
                rows = [','.join([date, '1']+['2']*len(columns))
                        for date in ('2017-06-30 23:50:00', '2017-07-01')]
                return io.BytesIO((header+'\n'+'\n'.join(rows)+'\n').encode())
        result = hot.read_month(Archive(), 2017, 6, [1])
        self.assertEqual(len(result), 2)
        self.assertEqual(result.timestamp.iloc[-1], pd.Timestamp('2017-07-01T00:00Z'))


if __name__ == '__main__':
    unittest.main()
