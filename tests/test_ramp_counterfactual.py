import unittest

import numpy as np

from utils.ramp_counterfactual import summarize_ramp_counterfactual


class RampCounterfactualTests(unittest.TestCase):
    def test_reports_exact_selected_gain_and_abstention(self):
        truth = np.array([[10.0, 10.0], [0.0, 0.0]])
        without = np.array([[12.0, 8.0], [1.0, 1.0]])
        with_ramp = np.array([[11.0, 9.0], [1.0, 1.0]])
        report = summarize_ramp_counterfactual(
            with_ramp, without, truth, [True, False], [True, False],
        )
        self.assertEqual(report["eligible_windows"], 1)
        self.assertEqual(report["effective_window_pct"], 50.0)
        self.assertAlmostEqual(report["gain_from_ramp_kw"], 0.5)
        self.assertAlmostEqual(report["effective_gain_kw"], 1.0)
        self.assertEqual(report["effective_harm_point_pct"], 0.0)
        self.assertEqual(report["gain_by_horizon_kw"], [0.5, 0.5])

    def test_reports_harm_without_dividing_by_all_windows(self):
        report = summarize_ramp_counterfactual(
            [[2.0, 0.0]], [[1.0, 1.0]], [[0.0, 0.0]], [True], [True],
        )
        self.assertEqual(report["gain_from_ramp_kw"], 0.0)
        self.assertEqual(report["effective_harm_point_pct"], 50.0)

    def test_rejects_impossible_effective_mask(self):
        with self.assertRaises(ValueError):
            summarize_ramp_counterfactual([[1.0]], [[0.0]], [[0.0]], [False], [True])

    def test_empty_effective_set_is_explicit(self):
        report = summarize_ramp_counterfactual([[1.0]], [[1.0]], [[0.0]], [False], [False])
        self.assertEqual(report["effective_windows"], 0)
        self.assertIsNone(report["effective_gain_kw"])


if __name__ == "__main__":
    unittest.main()
