"""Synthetic regression tests for the approved D2-0 r2 changes."""
import hashlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from scripts.audit_d2_feasibility_v34_r2 import (
    B1_THRESHOLD, curve_diagnostics, gate, interval_metrics, mask_coverage,
    scada_header, static_metadata, validated_record,
)


class D2R2Tests(unittest.TestCase):
    def metadata(self, text, ids):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "static.csv"
            path.write_text(text, encoding="utf-8")
            return static_metadata(path, ids)

    def test_wholly_empty_static_rows_only(self):
        result = self.metadata("Title,Model,Rated power (kW),Latitude,Longitude\n"
                               "Penmanshiel 01,MM82,2050,55,-2\n,,,,\n,,,,\n", [1])
        self.assertTrue(result["consistent"])
        self.assertEqual(result["rows"], 1)
        self.assertEqual(result["original_rows"], 3)
        self.assertEqual(result["entirely_empty_rows_removed"], 2)

    def test_nonempty_invalid_device_rows_rejected(self):
        for row in (",MM82,2050,55,-2", "garbage 01,MM82,2050,55,-2", "Penmanshiel XX,MM82,2050,55,-2"):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "device ID"):
                self.metadata("Title,Model,Rated power (kW),Latitude,Longitude\n" + row + "\n", [1])

    def test_duplicate_or_mismatching_devices_not_accepted(self):
        result = self.metadata("Title,Model,Rated power (kW)\nPenmanshiel 01,MM82,2050\nPenmanshiel 01,MM82,2050\n", [1, 2])
        self.assertFalse(result["consistent"])

    def test_prior_registration_hash_must_match(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "registration.json"
            body = b'{"test_year": 2022}\n'
            p.write_bytes(body)
            sha = hashlib.sha256(body).hexdigest()
            self.assertEqual(validated_record(p, sha), {"test_year": 2022})
            p.write_bytes(body + b" ")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                validated_record(p, sha)

    def test_exact_known_wind_alignment_and_persistence(self):
        raw = np.column_stack([np.arange(10), np.zeros(10)])
        curves = {"1": {"centers_mps": [0, 10], "curve_kw": [0, 100]}}
        truth = np.array([[20., 30.], [50., 60.]])
        oracle, physical = curve_diagnostics(raw, np.array([0, 3]), np.ones(10, dtype=int), curves, truth, history=2, horizon=2)
        self.assertEqual(oracle["pooled"]["rmse_lower"], 0)
        self.assertEqual(oracle["pooled"]["rmse_upper"], 0)
        self.assertEqual(physical["pooled"]["mae"], 15)
        self.assertEqual(oracle["missing_cells"], 0)

    def test_missing_intervals_enclose_every_possible_prediction(self):
        raw = np.zeros((4, 2)); raw[1:, 0] = np.nan
        curves = {"1": {"centers_mps": [1, 2], "curve_kw": [50, 100]}}
        truth = np.array([[-10., 50., 110.]])
        oracle, physical = curve_diagnostics(raw, np.array([0]), np.ones(4, dtype=int), curves, truth, history=1, horizon=3)
        self.assertEqual(oracle["curve_output_ranges"]["1"], [0., 100.])  # left=0 is essential
        self.assertEqual(oracle["missing_cells"], 3)
        self.assertEqual(physical["pooled"]["mae"], 170/3)
        bounds = oracle["pooled"]
        self.assertAlmostEqual(bounds["mae_lower"], 20/3)
        self.assertAlmostEqual(bounds["mae_upper"], 270/3)
        for x in np.linspace(0, 100, 11):
            for y in np.linspace(0, 100, 11):
                for z in np.linspace(0, 100, 11):
                    err = np.array([x, y, z]) - truth[0]
                    self.assertLessEqual(bounds["mae_lower"], np.abs(err).mean()+1e-12)
                    self.assertGreaterEqual(bounds["mae_upper"], np.abs(err).mean()-1e-12)
                    self.assertLessEqual(bounds["rmse_lower"], np.sqrt(np.square(err).mean())+1e-12)
                    self.assertGreaterEqual(bounds["rmse_upper"], np.sqrt(np.square(err).mean())-1e-12)

    def test_pooled_rmse_uses_all_cells_before_sqrt(self):
        result = interval_metrics(np.array([[0., 3.]]), np.array([[4., 3.]]))
        self.assertAlmostEqual(result["rmse_lower"], np.sqrt(9/2))
        self.assertAlmostEqual(result["rmse_upper"], np.sqrt(25/2))

    def test_gate_three_states_and_equalities(self):
        self.assertEqual(gate({"rmse_lower": 0., "rmse_upper": B1_THRESHOLD}), "PASS")
        self.assertEqual(gate({"rmse_lower": B1_THRESHOLD, "rmse_upper": B1_THRESHOLD+1}), "UNASSESSABLE")
        self.assertEqual(gate({"rmse_lower": B1_THRESHOLD+.001, "rmse_upper": B1_THRESHOLD+1}), "FAIL")

    def test_test_year_body_refused(self):
        class Poison:
            def read(self, *args):
                raise AssertionError("protected body read")
        with self.assertRaisesRegex(ValueError, "Protected"):
            mask_coverage(Poison(), [], 2022, 2022, time.monotonic()+5)
        stream = io.BytesIO(b"# Date and time,Power (kW)\nDO_NOT_PARSE\n")
        scada_header(stream)
        self.assertEqual(stream.readline(), b"DO_NOT_PARSE\n")

    def test_nonfinite_labels_and_issue_wind_refused(self):
        raw = np.zeros((3, 2)); curves = {"1": {"centers_mps": [0, 1], "curve_kw": [0, 100]}}
        with self.assertRaises(ValueError):
            curve_diagnostics(raw, np.array([0]), np.ones(3, dtype=int), curves, np.array([[np.nan]]), history=1, horizon=1)
        raw[0, 0] = np.nan
        with self.assertRaises(ValueError):
            curve_diagnostics(raw, np.array([0]), np.ones(3, dtype=int), curves, np.array([[1.]]), history=1, horizon=1)


if __name__ == "__main__":
    unittest.main()
