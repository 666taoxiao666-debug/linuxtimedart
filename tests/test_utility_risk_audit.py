import unittest

import numpy as np

from utils.utility_risk_audit import calibration_rows, summarize_risk_head


class UtilityRiskAuditTests(unittest.TestCase):
    def test_veto_value_distinguishes_helpful_and_harmful_rejections(self):
        probability = np.array([0.1, 0.8, 0.9, 0.2])
        gain = np.array([2.0, -4.0, 3.0, -1.0])
        result = summarize_risk_head(
            probability, gain, np.ones(4, dtype=bool), threshold=0.5,
        )
        self.assertEqual(result["samples"], 4)
        self.assertAlmostEqual(result["harm_prevalence_pct"], 50.0)
        self.assertAlmostEqual(result["harm_recall_pct"], 50.0)
        self.assertAlmostEqual(result["accepted_gain_kw"], 0.5)
        self.assertAlmostEqual(result["vetoed_gain_kw"], -0.5)
        self.assertAlmostEqual(result["net_gain_before_veto_kw"], 0.0)
        self.assertAlmostEqual(result["net_gain_after_veto_kw"], 0.25)
        self.assertAlmostEqual(result["veto_value_kw_per_candidate"], 0.25)

    def test_empty_mask_is_explicit(self):
        result = summarize_risk_head(
            np.array([0.2]), np.array([1.0]), np.array([False])
        )
        self.assertEqual(result["samples"], 0)
        self.assertIsNone(result["brier"])

    def test_calibration_bins_preserve_every_selected_sample(self):
        probability = np.array([0.0, 0.1, 0.49, 0.5, 0.9, 1.0])
        gain = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
        rows = calibration_rows(
            probability, gain, np.ones(6, dtype=bool), bins=5, factor="x"
        )
        self.assertEqual(sum(row["samples"] for row in rows), 6)
        self.assertTrue(all(row["factor"] == "x" for row in rows))


if __name__ == "__main__":
    unittest.main()
