import inspect
import unittest

import numpy as np

from utils.conditional_bias_calibration import fit_conditional_bias, apply_conditional_bias


class ConditionalBiasCalibrationTests(unittest.TestCase):
    def test_only_stable_supported_physical_cell_is_enabled(self):
        prediction = np.full((8, 2), 40.0)
        group = np.array([0] * 4 + [1] * 4)
        fit_truth = np.column_stack((np.full(8, 50.0), np.full(8, 50.0)))
        calibration_truth = fit_truth.copy()
        calibration_truth[group == 1] = 30.0
        model = fit_conditional_bias(
            [(prediction, fit_truth, group), (prediction, calibration_truth, group)],
            n_groups=2, rated_power=100.0, max_capacity_ratio=0.2, min_windows=4,
        )
        self.assertTrue(np.asarray(model["accepted"])[:, 0].all())
        self.assertFalse(np.asarray(model["accepted"])[:, 1].any())
        actual = apply_conditional_bias(model, prediction, group)
        np.testing.assert_array_equal(actual[group == 0], np.full((4, 2), 50.0))
        np.testing.assert_array_equal(actual[group == 1], np.full((4, 2), 40.0))
        self.assertNotIn("truth", inspect.signature(apply_conditional_bias).parameters)

    def test_cap_support_and_group_identity_are_enforced(self):
        prediction = np.full((4, 1), 95.0)
        truth = np.full((4, 1), 200.0)
        group = np.zeros(4, int)
        model = fit_conditional_bias(
            [(prediction, truth, group), (prediction, truth, group)], n_groups=1,
            rated_power=100.0, max_capacity_ratio=0.05, min_windows=4,
        )
        self.assertEqual(model["correction_kw"], [[5.0]])
        np.testing.assert_array_equal(apply_conditional_bias(model, prediction, group), np.full((4, 1), 100.0))
        scarce = fit_conditional_bias(
            [(prediction, truth, group), (prediction, truth, group)], n_groups=1,
            rated_power=100.0, max_capacity_ratio=0.05, min_windows=5,
        )
        self.assertEqual(scarce["correction_kw"], [[0.0]])
        with self.assertRaises(ValueError):
            apply_conditional_bias(model, prediction, np.ones(4, int))


if __name__ == "__main__":
    unittest.main()
