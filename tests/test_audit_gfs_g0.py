"""Synthetic NetCDF regression tests; no real or sealed meteorological data."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

SPEC = importlib.util.spec_from_file_location('g0audit', Path(__file__).parents[1] / 'scripts/audit_gfs_g0.py')
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def fixture():
    steps = np.array(a.LEADS, dtype='timedelta64[h]')
    time = np.datetime64('2016-01-01T00:00')
    ds = xr.Dataset(coords=dict(time=time, step=steps, valid_time=('step', time + steps),
        latitude=np.arange(49.5, 60.01, .25), longitude=np.arange(-8, 2.51, .25),
        heightAboveGround=10., meanSea=0.))
    for name in a.REQUIRED + a.OPTIONAL:
        pressure = name == 'prmsl'
        ds[name] = (('step', 'latitude', 'longitude'), np.full((6, 43, 43), 100000. if pressure else 5.))
        ds[name].attrs = dict(units='Pa' if pressure else 'm s**-1',
            GRIB_paramId=260074 if pressure else (131 if name.startswith('u') else 132),
            GRIB_typeOfLevel='meanSea' if pressure else 'heightAboveGround',
            GRIB_level=0 if pressure else (100 if name in a.OPTIONAL else 10))
    return ds


class AuditTests(unittest.TestCase):
    def check(self, ds):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'sample.nc'
            ds.to_netcdf(p)
            entry = dict(nc_sha256=a.sha256(p), dev_bytes=p.stat().st_size,
                         complete_18=all(n in ds for n in a.REQUIRED) and ds.sizes['step'] == 6)
            return a.audit_file(p, '2016010100', entry)

    def test_valid_netcdf(self):
        r = self.check(fixture())
        self.assertTrue(r['complete'])
        self.assertTrue(r['optional_level_verified'])

    def test_missing_step(self):
        r = self.check(fixture().isel(step=slice(0, 5)))
        self.assertFalse(r['complete'])
        self.assertEqual(len(r['missing_messages']), 3)
        self.assertFalse(r['errors'])

    def test_wrong_grid(self):
        self.assertIn('grid_shape_or_duplicate:latitude', self.check(fixture().isel(latitude=slice(0, 42)))['errors'])

    def test_duplicate_zero_longitude(self):
        ds = fixture()
        lon = ds.longitude.values.copy()
        lon[0] = 360  # also already contains 0
        ds = ds.assign_coords(longitude=lon)
        self.assertIn('grid_shape_or_duplicate:longitude', self.check(ds)['errors'])

    def test_f000(self):
        ds = fixture()
        step = ds.step.values.copy()
        step[0] = np.timedelta64(0, 'h')
        ds = ds.assign_coords(step=step, valid_time=('step', ds.time.values + step))
        self.assertIn('f000_refused', self.check(ds)['errors'])

    def test_sealed_refused_before_value_scan(self):
        ds = fixture().assign_coords(time=np.datetime64('2023-12-31T18'))
        ds = ds.assign_coords(valid_time=('step', ds.time.values + ds.step.values))
        r = self.check(ds)
        self.assertIn('sealed_valid_time_refused', r['errors'])
        self.assertEqual(r['ranges'], {})

    def test_physical_bounds(self):
        for name, value in [('u10', 80), ('v10', -81), ('prmsl', 89000), ('prmsl', 110001)]:
            ds = fixture()
            ds[name].values[0, 0, 0] = value
            self.assertIn(f'physical_range:{name}:f003', self.check(ds)['errors'])

    def test_valid_time_and_identity(self):
        ds = fixture()
        ds = ds.assign_coords(valid_time=('step', ds.valid_time.values + np.timedelta64(1, 'h')))
        self.assertIn('valid_time_mismatch', self.check(ds)['errors'])
        ds = fixture()
        ds.u10.attrs['GRIB_level'] = 100
        self.assertIn('level_identity:u10', self.check(ds)['errors'])

    def test_optional_absent_or_unproven(self):
        r = self.check(fixture().drop_vars(list(a.OPTIONAL)))
        self.assertTrue(r['complete'])
        self.assertFalse(r['optional_complete'])
        ds = fixture()
        for name in a.OPTIONAL:
            del ds[name].attrs['GRIB_level']
        r = self.check(ds)
        self.assertTrue(r['complete'])
        self.assertFalse(r['optional_level_verified'])

    def test_hash_mismatch_does_not_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'bad.nc'
            p.write_bytes(b'not netcdf')
            with patch.object(a.xr, 'open_dataset', side_effect=AssertionError('must not decode')):
                r = a.audit_file(p, '2016010100', dict(nc_sha256='0'*64, dev_bytes=10))
            self.assertIn('manifest_hash_or_size_mismatch', r['errors'])

    def test_calendar_keeps_missing_cycle(self):
        self.assertEqual(len(a.cycles(2016)), 1464)
        self.assertEqual(len(a.cycles(2018)), 1460)
        self.assertIn('2018052618', a.cycles(2018))
        r = a.audit_file(Path('/nonexistent_gfs_test.nc'), '2018052618', None)
        self.assertTrue(r['missing_file'])
        self.assertFalse(r['complete'])

    def test_partial_nonfinite(self):
        ds = fixture()
        ds.u10.values[0, 0, 0] = np.nan
        self.assertIn('nonfinite:u10:f003', self.check(ds)['errors'])


if __name__ == '__main__':
    unittest.main()
