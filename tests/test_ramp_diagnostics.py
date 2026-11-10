"""Tests for scripts/ramp_diagnostics.py (pure numpy; runs without torch)."""
import importlib.util
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ramp_diagnostics_under_test", ROOT / "scripts" / "ramp_diagnostics.py")
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)
KEYS = {"truth": "truth", "anchor": "PersistenceHour", "missing": "issue_upstream_missing", "time": "times",
        "pred": "prediction", "window": "window_starts"}
K24 = ROOT / "output/kelmarsh_hourly/upstream_v32_review/results/evaluation"


def synthetic(tmp, n=600, seed=0, missing_bad=False):
    rng = np.random.default_rng(seed)
    anchor = rng.uniform(0, 2050, n)
    truth = np.clip(anchor[:, None] + rng.normal(0, 400, (n, 4)) * np.arange(1, 5) / 2, 0, 2050)
    t0 = np.datetime64("2022-01-01T01:00") + np.arange(n) // 3 * np.timedelta64(3, "h")
    times = t0[:, None] + np.arange(4) * np.timedelta64(1, "h")
    miss = np.arange(0, n, 10)  # spread over time, as a block bootstrap needs
    missing = np.zeros(n)
    missing[miss] = 1
    ws = np.arange(n) * 3
    np.savez(tmp / "cohort.npz", truth=truth, PersistenceHour=np.repeat(anchor[:, None], 4, 1), times=times,
             issue_upstream_missing=missing, window_starts=ws)
    for s in (1, 2):
        ref = np.repeat(anchor[:, None], 4, 1) + rng.normal(0, 30, (n, 4))  # persistence-like, no change signal
        cand = truth + rng.normal(0, 30, (n, 4))  # near-perfect candidate
        if missing_bad:
            cand[miss] = truth[miss] + 500.0  # candidate is bad exactly where upstream is missing
            ref[miss] = truth[miss]
        np.savez(tmp / f"C_s{s}.npz", prediction=cand, window_starts=ws)
        np.savez(tmp / f"R_s{s}.npz", prediction=ref, window_starts=ws)
    return tmp


class Auc(unittest.TestCase):
    def test_prepared_missing_uses_issue_hour_and_checks_hash(self):
        import pandas as pd
        columns = ["upwind_ws", "upwind_dws3", "upwind_minus_local", "local_ws", "local_dws3", "regional_ws",
                   "regional_dws3", "local_dp3", "grad_east", "grad_north", "grad_magnitude_change3"]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            frame = pd.DataFrame(np.ones((6, 11)), columns=columns,
                                 index=pd.date_range("2022-01-01", periods=6, freq="h"))
            frame.index.name = "timestamp_hour_end_utc"
            frame.iloc[0, 0] = np.nan
            frame.to_csv(p / "upstream.csv")
            (p / "prepared_complete.json").write_text(json.dumps({"artifacts": {"upstream.csv": rd.sha256(p / "upstream.csv")}}))
            times = np.datetime64("2022-01-01T01") + np.arange(4)[None, :] * np.timedelta64(1, "h")
            mask, _ = rd.prepared_missing(p, times)
            self.assertTrue(mask[0])
            mask, _ = rd.prepared_missing(p, times + np.timedelta64(1, "h"))
            self.assertFalse(mask[0])
            with (p / "upstream.csv").open("a") as f:
                f.write("\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                rd.prepared_missing(p, times)

    def test_empty_event_clusters_keep_time_axis(self):
        ci = rd.ratio_ci(np.array([1., 3.]), np.array([0, 99]), 8, 200, 2024,
                         all_issue=np.arange(100))
        self.assertEqual(ci["total_time_clusters"], 100)
        self.assertEqual(ci["zero_event_clusters"], 98)
        self.assertEqual(ci["mean_gain_kw"], 2.)

    def test_auc_values_and_ties(self):
        lab = np.array([0, 0, 1, 1], bool)
        self.assertEqual(rd.auc(np.array([1., 2., 3., 4.]), lab), 1.0)
        self.assertEqual(rd.auc(np.array([4., 3., 2., 1.]), lab), 0.0)
        self.assertEqual(rd.auc(np.zeros(4), lab), 0.5)
        self.assertEqual(rd.auc(np.array([1., 2., 2., 3.]), np.array([0, 1, 0, 1], bool)), 0.875)
        self.assertTrue(np.isnan(rd.auc(np.arange(4.), np.zeros(4, bool))))


class Fold(unittest.TestCase):
    def test_good_candidate_passes_and_fallback_rules(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = synthetic(Path(d))
            r = rd.analyse_fold("S", tmp / "cohort.npz", str(tmp / "C_s*.npz"), str(tmp / "R_s*.npz"), 2050.0, KEYS,
                                reps=200)
            self.assertTrue(r["M1"]["pass"])
            self.assertGreater(r["M2"]["gain"], 0.05)
            self.assertFalse(r["M3"]["adopt"])  # candidate is good everywhere, fallback only hurts
            self.assertEqual(r["seeds"], [1, 2])
        with tempfile.TemporaryDirectory() as d:
            tmp = synthetic(Path(d), missing_bad=True)
            r = rd.analyse_fold("S", tmp / "cohort.npz", str(tmp / "C_s*.npz"), str(tmp / "R_s*.npz"), 2050.0, KEYS,
                                reps=200)
            self.assertTrue(r["M3"]["adopt"])
            self.assertEqual(r["M3"]["missing_windows"], 60)

    def test_misaligned_windows_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = synthetic(Path(d))
            with np.load(tmp / "R_s1.npz") as z:
                prediction, windows = z["prediction"], z["window_starts"] + 1
            np.savez(tmp / "R_s1.npz", prediction=prediction, window_starts=windows)
            with self.assertRaises(ValueError):
                rd.analyse_fold("S", tmp / "cohort.npz", str(tmp / "C_s*.npz"), str(tmp / "R_s*.npz"), 2050.0, KEYS,
                                reps=200)

    def test_decision_truth_table(self):
        def f(m1, m2, m3):
            return {"M1": {"pass": m1}, "M2": {"pass": m2}, "M3": {"adopt": m3}}
        self.assertEqual(rd.decide([f(1, 1, 1), f(1, 1, 1)]), {"ramp": "RAMP_REPLICATED", "fallback": "ADOPT_FALLBACK"})
        self.assertEqual(rd.decide([f(1, 1, 0), f(1, 0, 1)]), {"ramp": "RAMP_NOT_REPLICATED", "fallback": "NO_FALLBACK"})
        self.assertEqual(rd.decide([f(0, 1, 1), f(1, 1, 1)])["ramp"], "RAMP_NOT_REPLICATED")

    def test_clean_removes_nan(self):
        self.assertEqual(rd._clean({"a": [float("nan"), 1.0]}), {"a": [None, 1.0]})

    def test_missing_metadata_is_not_no_missing(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = synthetic(Path(d))
            keys = dict(KEYS, missing="not_stored")
            result = rd.analyse_fold("S", tmp / "cohort.npz", str(tmp / "C_s*.npz"),
                                     str(tmp / "R_s*.npz"), 2050.0, keys, reps=100)
            self.assertTrue(result["M1"]["pass"])
            self.assertEqual(result["M3"]["status"], "UNAVAILABLE")
            self.assertEqual(rd.decide([result])["fallback"], "UNDETERMINED")


@unittest.skipUnless((K24 / "test_cohort.npz").exists(), "Kelmarsh v32 evaluation arrays not present")
class KelmarshRegression(unittest.TestCase):
    def test_reproduces_claude_2026_11_10_numbers(self):
        r = rd.analyse_fold("K24", K24 / "test_cohort.npz", str(K24 / "TSMixer_upstream_all_s*.npz"),
                            str(K24 / "TSMixer_last10_all_s*.npz"), 2050.0, KEYS, reps=200)
        self.assertAlmostEqual(r["M2"]["candidate"]["auc_up"], 0.8218, places=3)
        self.assertAlmostEqual(r["M2"]["reference"]["auc_up"], 0.7300, places=3)
        self.assertAlmostEqual(r["M2"]["candidate"]["auc_down"], 0.8828, places=3)
        self.assertAlmostEqual(r["M1"]["relative_gain"], 0.1789, places=3)
        self.assertEqual(r["M3"]["missing_windows"], 102)


if __name__ == "__main__":
    unittest.main()
