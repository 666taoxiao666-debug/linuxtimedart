import unittest
import numpy as np
import pandas as pd
from utils.paired_forecast_statistics import paired_forecast_statistics, moving_block_bootstrap_ratio


class PairedEstimandTests(unittest.TestCase):
    def test_unequal_turbine_counts_can_flip_sign_but_primary_ci_matches_main_mae(self):
        # One positive time has 20 turbines; negative times have only one each.
        dates = np.repeat(pd.date_range("2023-01-01", periods=4), [20, 1, 1, 1])
        gain = np.r_[np.ones(20), [-2., -2., -2.]]
        truth = np.zeros((23, 1)); reference = np.full_like(truth, 10.)
        candidate = reference-gain[:, None]
        result = paired_forecast_statistics(candidate, reference, truth,
            pd.DataFrame(dict(forecast_start=dates)), block_length=1, replicates=300,
            estimand="window_uniform")
        self.assertGreater(result["paired_gain_kw"], 0)
        self.assertLess(result["time_uniform_gain_kw"], 0)
        self.assertAlmostEqual(result["primary_gain_kw"], result["paired_gain_kw"])
        self.assertAlmostEqual(result["time_block_bootstrap"]["mean_gain_kw"], result["paired_gain_kw"])
        legacy = paired_forecast_statistics(candidate, reference, truth,
            pd.DataFrame(dict(forecast_start=dates)), block_length=1, replicates=300)
        self.assertEqual(legacy["primary_estimand"], "time_uniform")
        self.assertLess(legacy["primary_gain_kw"], 0)

    def test_ratio_bootstrap_preserves_identical_forecasts_and_reproducibility(self):
        counts = np.array([1, 3, 1, 2, 4])
        first = moving_block_bootstrap_ratio(np.zeros(5), counts, replicates=300)
        self.assertEqual(first["ci95_low_kw"], 0.)
        self.assertEqual(first["ci95_high_kw"], 0.)
        self.assertEqual(first, moving_block_bootstrap_ratio(np.zeros(5), counts, replicates=300))
        with self.assertRaises(ValueError):
            moving_block_bootstrap_ratio(np.zeros(5), [1, 0, 1, 2, 4])


if __name__ == "__main__":
    unittest.main()
