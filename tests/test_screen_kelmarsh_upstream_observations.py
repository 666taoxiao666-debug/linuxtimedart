"""Torch-free regression tests for the retrospective MIDAS screen and raw sealed acquisition."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from scripts import acquire_kelmarsh_sealed_years as sealed
from scripts import screen_kelmarsh_upstream_observations as screen


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def badc(self, rows, year=2021, name='station', units='knots', version=True, end=True):
        header = 'src_id,ob_time,wind_speed,wind_direction,msl_pressure' + (',version_num' if version else '')
        path = self.root/f'{name}_{year}.csv'
        path.write_text(f'unit,wind_speed,{units}\nunit,wind_direction,degrees\nunit,msl_pressure,hPa\ndata,\n'
                        + header+'\n'+'\n'.join(rows)+ ('\nend data,' if end else ''))
        return path

    def test_badc_metadata_and_padded_markers(self):
        path = self.badc(['1,2021-01-01 00:00,10,90,1000,1'])
        parsed = screen.read_badc_csv(path)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed.attrs['units']['wind_speed'], 'knots')

    def test_truncated_badc_rejected(self):
        path = self.badc(['1,2021-01-01 00:00,10,90,1000,1'], end=False)
        with self.assertRaisesRegex(ValueError, 'truncated'):
            screen.read_badc_csv(path)

    def test_knots_to_ms_and_version_one(self):
        self.badc(['1,2021-01-01 00:00,99,90,1000,0', '1,2021-01-01 00:00,10,90,1000,1'])
        obs, files, _ = screen.load_observations(self.root, {1})
        self.assertAlmostEqual(obs[1].wind_speed.iloc[0], 10*1852/3600)
        self.assertEqual(files[0]['version_num'], 1)

    def test_no_version_fails_closed(self):
        self.badc(['1,2021-01-01 00:00,10,90,1000'], version=False)
        with self.assertRaisesRegex(ValueError, 'version_num'):
            screen.load_observations(self.root, {1})

    def test_unknown_units_fail_closed(self):
        self.badc(['1,2021-01-01 00:00,10,90,1000,1'], units='unknown')
        with self.assertRaisesRegex(ValueError, 'unit'):
            screen.load_observations(self.root, {1})

    def test_conflicting_duplicates_not_field_mixed(self):
        self.badc(['1,2021-01-01 00:00,10,,1000,1', '1,2021-01-01 00:00,,90,1000,1'])
        with self.assertRaisesRegex(ValueError, 'conflicting duplicate'):
            screen.load_observations(self.root, {1})

    def test_identical_duplicates_collapse(self):
        row = '1,2021-01-01 00:00,10,90,1000,1'
        self.badc([row, row])
        self.assertEqual(len(screen.load_observations(self.root, {1})[0][1]), 1)

    def test_filename_row_year_mismatch_rejected(self):
        self.badc(['1,2023-01-01 00:00,10,90,1000,1'])
        with self.assertRaisesRegex(ValueError, 'row year'):
            screen.load_observations(self.root, {1})

    def test_sealed_years_and_qc_zero_never_opened(self):
        for name in ('bad_2023.csv', 'bad_2024.csv', 'sealed/bad_2021.csv', 'qc-version-0/bad_2021.csv'):
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('UNREADABLE SEALED CONTENT')
        with mock.patch.object(screen, 'read_badc_csv', side_effect=AssertionError('opened')):
            obs, files, skipped = screen.load_observations(self.root, {1})
        self.assertEqual((obs, files, len(skipped)), ({}, [], 4))

    def test_invalid_observations_are_not_interpolated(self):
        self.badc(['1,2021-01-01 00:00,-999,999,-99,1'])
        obs, _, _ = screen.load_observations(self.root, {1})
        self.assertTrue(obs[1].iloc[0].isna().all())

    def test_station_location_conflicts_rejected(self):
        path = self.root/'metadata.csv'
        path.write_text('src_id,src_name,latitude,longitude\n1,a,52.4,-.94\n1,a,52.5,-.94\n')
        with self.assertRaisesRegex(ValueError, 'conflicting locations'):
            screen.load_metadata(path)

    def test_completeness_requires_every_development_year(self):
        times = pd.date_range('2016-01-01', '2023-01-01', freq='h', inclusive='left')
        obs = pd.DataFrame({'wind_speed': 1.0}, index=times)
        obs.loc['2019', 'wind_speed'] = np.nan
        rows = screen.qualified_stations(pd.DataFrame([{'src_id': 1}]), {1: obs})
        self.assertFalse(rows[0]['qualified'])
        self.assertEqual(rows[0]['completeness'][2016], 1)

    def test_lag_one_ignores_issue_and_future_values(self):
        times = pd.date_range('2021-01-01', periods=10, freq='h')
        stations = [{'src_id': s, 'distance_km': 30+s*5, 'bearing_deg': s*50} for s in range(1,7)]
        obs = {s: pd.DataFrame({'wind_speed': np.arange(10.)+s, 'wind_direction': 90., 'msl_pressure': 1000.+s}, index=times) for s in range(1,7)}
        before, _ = screen.upstream_features(times[5:6], obs, stations, 1)
        for frame in obs.values():
            frame.loc[times[5]:, :] = 99999.
        after, _ = screen.upstream_features(times[5:6], obs, stations, 1)
        np.testing.assert_equal(before, after)

    def test_power_regime_alone_is_not_weather_gain(self):
        rng = np.random.default_rng(7)
        n = 300
        halves = np.repeat([0, 1], n//2)
        anchor = rng.choice([0., 1000.], n)
        mid = anchor > 0
        error = np.repeat((100*mid + rng.normal(size=n))[:, None], 4, axis=1)
        result = screen.score_group(np.zeros((n, 1)), error, anchor, halves)
        for row in result['leads']:
            self.assertAlmostEqual(row['rmse_gain_pct'], 0., places=8)
            self.assertAlmostEqual(row['mae_gain_pct'], 0., places=8)
            self.assertGreater(row['rmse_gain_vs_bias_only_pct'], 90.)

    def test_weather_signal_detected_without_target_leak(self):
        rng = np.random.default_rng(8)
        features = rng.normal(size=(300, 2))
        error = np.repeat((10*features[:, 0] + rng.normal(size=300))[:, None], 4, axis=1)
        result = screen.score_group(features, error, np.ones(300)*1000, np.repeat([0,1],150))
        self.assertGreater(result['leads'][3]['mae_gain_pct'], 50)
        self.assertEqual(set(result['leads'][0]['directions']), {'early_fit_late_score_forward', 'late_fit_early_score_reverse'})

    def test_cross_boundary_targets_are_purged(self):
        early = pd.date_range('2021-06-28', periods=70, freq='h')
        late = pd.date_range('2021-07-01', periods=50, freq='h')
        issue = early.append(late)
        rng = np.random.default_rng(9)
        out = screen.score_group(rng.normal(size=(len(issue),1)), rng.normal(size=(len(issue),4)),
                                 np.ones(len(issue))*1000, np.asarray(issue.month>=7, int), issue)
        self.assertGreater(out['boundary_purged_windows'], 0)
        self.assertLess(out['coverage'], 1)

    def test_absent_half_not_fabricated_as_failure(self):
        with self.assertRaisesRegex(ValueError, 'Both chronological'):
            screen.score_group(np.ones((100,1)), np.ones((100,4)), np.ones(100), np.zeros(100))

    def test_too_small_fit_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Too few'):
            screen.score_group(np.ones((12,3)), np.ones((12,4)), np.ones(12), np.repeat([0,1],6))

    def cohort(self):
        folder = self.root/'run'/'evaluation'
        folder.mkdir(parents=True)
        issue = np.array(['2021-01-02', '2021-08-02'], dtype='datetime64[ns]')
        starts = np.array([1,2])
        data = dict(truth=np.ones((2,4)), times=issue[:,None]+np.arange(1,5)*np.timedelta64(1,'h'),
                    turbines=np.array([1,2]), PersistenceLast10=np.ones((2,4)), PersistenceHour=np.ones((2,4)), window_starts=starts)
        np.savez(folder/'test_cohort.npz', **data)
        for seed in screen.SEEDS:
            np.savez(folder/f'TSMixer_last10_s{seed}.npz', prediction=np.ones((2,4)), window_starts=starts)
        self.register(folder)
        return folder

    def register(self, folder):
        manifest = {f'evaluation/{p.name}': hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.npz')}
        (folder.parent/'completed.json').write_text(json.dumps(manifest))

    def test_registered_seed_mean_cohort_loads(self):
        cohort, prediction, issue = screen.load_year(self.cohort(), 'TSMixer_last10', 2021)
        self.assertEqual(prediction.shape, (2,4))

    def test_modified_predictions_rejected(self):
        folder = self.cohort()
        with (folder/'TSMixer_last10_s2024.npz').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'modified prediction'):
            screen.load_year(folder, 'TSMixer_last10', 2021)

    def test_registered_but_wrong_window_order_rejected(self):
        folder = self.cohort()
        np.savez(folder/'TSMixer_last10_s2024.npz', prediction=np.ones((2,4)), window_starts=np.array([2,1]))
        self.register(folder)
        with self.assertRaisesRegex(ValueError, 'window identity'):
            screen.load_year(folder, 'TSMixer_last10', 2021)

    def test_requested_year_enforced(self):
        with self.assertRaisesRegex(ValueError, 'year differs'):
            screen.load_year(self.cohort(), 'TSMixer_last10', 2022)

    def test_sealed_evaluation_refused_before_open(self):
        with mock.patch.object(Path, 'read_text', side_effect=AssertionError('opened')):
            with self.assertRaisesRegex(ValueError, 'development evaluations'):
                screen.load_year(self.root/'sealed', 'TSMixer_last10', 2021)
            with self.assertRaisesRegex(ValueError, 'development evaluations'):
                screen.load_year(self.root, 'TSMixer_last10', 2023)

    def test_invalid_cli_overwrites_stale_success_with_null_not_false(self):
        output = self.root/'screen.json'
        output.write_text('{"passed":true}')
        argv = ['screen', '--midas-root', str(self.root), '--metadata', str(self.root/'absent.csv'),
                '--eval-2021', str(self.root), '--eval-2022', str(self.root), '--output', str(output)]
        with mock.patch('sys.argv', argv):
            self.assertEqual(screen.main(), 2)
        result = json.loads(output.read_text())
        self.assertEqual(result['status'], 'INVALID_INPUT')
        self.assertIsNone(result['passed'])

    def test_target_hour_alignment_enforced(self):
        folder = self.cohort()
        with np.load(folder/'test_cohort.npz') as saved:
            data = dict(saved)
        data['times'][0,3] += np.timedelta64(1,'h')
        np.savez(folder/'test_cohort.npz', **data)
        self.register(folder)
        with self.assertRaisesRegex(ValueError, 'misaligned'):
            screen.load_year(folder, 'TSMixer_last10', 2021)


class SealedDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)/'sealed'/'kelmarsh'
        self.payloads = {year: f'raw sealed bytes {year}'.encode() for year in sealed.FILES}
        self.official = {year: (len(value), hashlib.md5(value).hexdigest()) for year,value in self.payloads.items()}
        self.record = {'id':16807551,'doi':'10.5281/zenodo.16807551', 'files':[
            {'key': key, 'size':self.official[year][0], 'checksum':'md5:'+self.official[year][1],
             'links':{'self': f'https://zenodo.org/api/records/16807551/files/{key}/content'}} for year,key in sealed.FILES.items()]}

    def download(self, url, target, size, md5, workers):
        year = next(year for year,key in sealed.FILES.items() if key in url)
        target.write_bytes(self.payloads[year])

    def test_only_raw_verified_bytes_written(self):
        with mock.patch.dict(sealed.OFFICIAL, self.official):
            result = sealed.acquire(self.record, self.directory, self.download)
        self.assertEqual(result['status'], 'SEALED_VERIFIED')
        self.assertEqual([f['year'] for f in result['files']], [2023,2024])
        self.assertEqual(len(list(self.directory.iterdir())), 3)
        self.assertTrue(all(len(f['sha256'])==64 for f in result['files']))

    def test_integrity_failure_preserves_interrupted_state(self):
        def wrong(url,target,*args,**kwargs):
            target.write_bytes(b'corrupt')
        with mock.patch.dict(sealed.OFFICIAL, self.official), self.assertRaisesRegex(ValueError, 'byte integrity'):
            sealed.acquire(self.record, self.directory, wrong)
        manifest = json.loads((self.directory/'download_manifest.json').read_text())
        self.assertEqual(manifest['status'], 'INTERRUPTED')
        self.assertEqual(manifest['files'], [])

    def test_provider_mismatch_refuses_before_download(self):
        self.record['files'][1]['size'] += 1
        downloader = mock.Mock()
        with mock.patch.dict(sealed.OFFICIAL, self.official), self.assertRaisesRegex(ValueError, 'Provider identity'):
            sealed.acquire(self.record, self.directory, downloader)
        downloader.assert_not_called()

    def test_nonsealed_directory_rejected(self):
        with self.assertRaisesRegex(ValueError, 'sealed directory'):
            sealed.acquire(self.record, Path(self.tmp.name)/'development', mock.Mock())


if __name__ == '__main__':
    unittest.main()
