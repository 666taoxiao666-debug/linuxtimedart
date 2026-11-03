"""Synthetic tests: no real SCADA or station files are opened."""
import io
import time
import unittest

import numpy as np

from scripts.audit_d2_feasibility_v34 import (
    curve_predictions, full_year, mask_coverage, metrics, registration, scada_header,
)


class D2AuditTests(unittest.TestCase):
    def test_registration_all_turbines_not_latest_partial_year(self):
        entries = [{"turbine": t, "begin": "2022-01-01", "end": "2023-01-01"} for t in (1, 2)]
        entries += [{"turbine": 1, "begin": "2023-02-01", "end": "2024-01-01"},
                    {"turbine": 2, "begin": "2023-01-01", "end": "2024-01-01"},
                    {"turbine": 2, "begin": "2024-01-01", "end": "2025-01-01"}]
        reg = registration(entries)
        self.assertEqual(reg["test_year"], 2022)
        self.assertEqual(reg["protected_years"], [2022, 2023, 2024])

    def test_contiguous_months_and_gap(self):
        self.assertTrue(full_year([("2020-01-01", "2020-06-01"), ("2020-06-01", "2021-01-01")], 2020))
        self.assertFalse(full_year([("2020-01-01", "2020-06-01"), ("2020-06-02", "2021-01-01")], 2020))
        self.assertFalse(full_year([], 2020))

    def test_no_complete_year_fails_closed(self):
        self.assertIsNone(registration([])["test_year"])

    def test_header_does_not_consume_body(self):
        stream = io.BytesIO(b"# description\n# Date and time,Power (kW)\nSECRET_BODY\n")
        self.assertEqual(scada_header(stream), ["Date and time", "Power (kW)"])
        self.assertEqual(stream.readline(), b"SECRET_BODY\n")

    def test_protected_body_refused_before_read(self):
        class Poison:
            def read(self, *args):
                raise AssertionError("body accessed")
        with self.assertRaisesRegex(ValueError, "Protected"):
            mask_coverage(Poison(), [], 2022, 2022, time.monotonic()+5)

    def test_missingness_not_numeric_conversion(self):
        stream = io.BytesIO(b"2020-01-01 00:00:00,NOT_A_NUMBER\n2020-01-01 00:10:00,\n2020-01-01 00:10:00,also_text\n")
        result = mask_coverage(stream, ["Date and time", "Power (kW)"], 2020, 2022, time.monotonic()+5)
        self.assertEqual(result["duplicate_ticks"], 1)
        self.assertEqual(result["expected_ticks"], 366*24*6)
        self.assertAlmostEqual(result["timestamp_coverage"], 2/(366*24*6))
        self.assertAlmostEqual(result["missing_fraction"]["Power (kW)"], 1/3)

    def test_curve_issue_and_future_alignment(self):
        raw = np.column_stack([np.arange(10), np.zeros(10)])
        curves = {"1": {"centers_mps": [0, 10], "curve_kw": [0, 100]}}
        oracle, pers = curve_predictions(raw, np.array([0, 3]), np.ones(10, dtype=int), curves, history=2, horizon=2)
        np.testing.assert_array_equal(oracle, [[20, 30], [50, 60]])
        np.testing.assert_array_equal(pers, [[10, 10], [40, 40]])

    def test_missing_future_wind_does_not_filter_queue(self):
        raw = np.zeros((8, 2)); raw[2, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing future wind"):
            curve_predictions(raw, np.array([0]), np.ones(8, dtype=int), {}, history=2, horizon=2)

    def test_metrics_fail_closed_and_pooled_units(self):
        self.assertEqual(metrics(np.array([0., 2.]), np.array([1., 1.])), {"mae": 1., "rmse": 1.})
        with self.assertRaises(ValueError):
            metrics(np.array([np.nan]), np.array([1.]))
        with self.assertRaises(ValueError):
            metrics(np.array([1., 2.]), np.array([1.]))


if __name__ == "__main__":
    unittest.main()
