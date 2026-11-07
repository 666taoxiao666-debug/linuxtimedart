import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import audit_hill_of_towie_e1 as b


class AcquisitionTests(unittest.TestCase):
    def record(self):
        return {"id": 22662930, "version": "2.1.0", "files": [
            {"key": f"{y}.zip", "size": 1, "checksum": "md5:" + "a"*32} for y in b.YEARS]}

    def plan(self):
        return {"base": b.BASE, "radius_km": 150, "sealed_years": list(b.SEALED),
                "stations": [{"src_id": 1, "county": "moray", "station_file_name": "example"}],
                "files": [{"name": f"obs_{y}.csv", "path": f"{b.BASE}/moray/00001_example/qc-version-1/obs_{y}.csv",
                           "year": y, "src_id": 1, "sealed": y in b.SEALED, "md5": "a"*32, "size": 12}
                          for y in b.YEARS]}

    def test_registration_protects_two_full_and_partial_year(self):
        r = b.registration(self.record())
        self.assertEqual(r["test_years"], [2024, 2025])
        self.assertEqual(r["protected_partial_years"], [2026])
        self.assertFalse(set(r["development_years"]) & set(b.SEALED))

    def test_changed_record_and_duplicate_rejected(self):
        for mutate in (lambda r: r.update(version="latest"),
                       lambda r: r["files"].append(copy.deepcopy(r["files"][0]))):
            r = self.record()
            mutate(r)
            with self.assertRaises(ValueError):
                b.registration(r)

    def test_sealed_routing_cannot_be_changed(self):
        b.validate_plan(self.plan())
        for year in b.SEALED:
            p = self.plan()
            next(f for f in p["files"] if f["year"] == year)["sealed"] = False
            with self.assertRaises(ValueError):
                b.validate_plan(p)

    def test_invalid_identity_rejected(self):
        for key, value in (("path", "/tmp/obs_2016.csv"), ("name", "../obs_2016.csv"),
                           ("md5", "not-a-hash"), ("size", -1), ("year", 2027)):
            p = self.plan()
            p["files"][0][key] = value
            with self.assertRaises(ValueError):
                b.validate_plan(p)
        p = self.plan()
        p["files"].append(p["files"][0].copy())
        with self.assertRaises(ValueError):
            b.validate_plan(p)

    def test_fixed_registration_preserved_on_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "registration.json"
            b.frozen(path, {"years": [2024, 2025]})
            original = path.read_bytes()
            b.frozen(path, {"years": [2024, 2025]})
            with self.assertRaises(ValueError):
                b.frozen(path, {"years": [2023, 2024]})
            self.assertEqual(path.read_bytes(), original)

    def test_geography_uses_supplied_farm_not_kelmarsh(self):
        import pandas as pd
        center = [57.5, -3.0]
        rows = [{"src_id": sid, "station_latitude": lat, "station_longitude": -3.0,
                 "first_year": 2000, "last_year": 2025, "historic_county": "moray",
                 "station_file_name": f"station{sid}"} for sid, lat in [(1, 57.5), (2, 52.0), (3, 58.0)]]
        selected = b.nearby(pd.DataFrame(rows), center)
        self.assertEqual([r["src_id"] for r in selected], [1, 3])
        self.assertAlmostEqual(selected[0]["distance_km"], 0.)
        self.assertAlmostEqual(selected[1]["bearing_deg"], 0.)


class ReadonlyTests(unittest.TestCase):
    def test_protected_header_refused_before_open(self):
        for year in (2024, 2025, 2026):
            with self.assertRaises(ValueError):
                b.development_header(Path('does-not-exist') / f'{year}.zip', year)

    def test_header_only_and_no_extraction(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / '2016.zip'
            with zipfile.ZipFile(p, 'w') as z:
                z.writestr('tblSCTurbine_2016_01.csv', b'TimeStamp,wind\n\xff\xfe\x00UNPARSED')
            self.assertEqual(b.development_header(p, 2016), {'tblSCTurbine_2016_01.csv': ['TimeStamp', 'wind']})
            self.assertEqual(list(Path(temp).iterdir()), [p])

    def test_fixed_gate_boundaries_and_qualified_only(self):
        rows = [{'qualified': True, 'has_pressure': i < 5, 'bearing_deg': (i%3)*90.0} for i in range(8)]
        self.assertTrue(b.coverage_gate(rows)['passed'])
        for mutation in ('station', 'pressure', 'quadrant'):
            x = copy.deepcopy(rows)
            if mutation == 'station': x[7]['qualified'] = False
            if mutation == 'pressure': x[4]['has_pressure'] = False
            if mutation == 'quadrant':
                for v in x: v['bearing_deg'] %= 180
            self.assertFalse(b.coverage_gate(x)['passed'])


if __name__ == "__main__":
    unittest.main()
