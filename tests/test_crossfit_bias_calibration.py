import inspect
import unittest

import numpy as np

from utils.crossfit_bias_calibration import fit_bias_guard, apply_bias


class CrossfitBiasCalibrationTests(unittest.TestCase):
    def test_magnitude_fit_and_second_block_gate_are_separate(self):
        fit_prediction = np.full((8, 3), 40.0)
        fit_truth = np.full((8, 3), 50.0)
        calibration_truth = np.column_stack((
            np.full(8, 50.0), np.full(8, 30.0), np.full(8, 40.0)
        ))
        calibration_prediction = np.full((8, 3), 40.0)
        model = fit_bias_guard(
            [(fit_prediction, fit_truth), (calibration_prediction, calibration_truth)],
            rated_power=100.0, max_capacity_ratio=0.2, min_windows=8,
        )
        self.assertEqual(model["correction_kw"], [10.0, 0.0, 0.0])
        self.assertEqual(model["accepted"], [True, False, False])
        actual = apply_bias(model, np.full((2, 3), 40.0))
        np.testing.assert_array_equal(actual, [[50.0, 40.0, 40.0]] * 2)
        self.assertNotIn("truth", inspect.signature(apply_bias).parameters)

    def test_physical_cap_clipping_support_and_shape_checks(self):
        prediction = np.array([[95.0], [95.0]])
        truth = np.array([[200.0], [200.0]])
        model = fit_bias_guard(
            [(prediction, truth), (prediction, truth)], rated_power=100.0,
            max_capacity_ratio=0.05, min_windows=2,
        )
        self.assertEqual(model["correction_kw"], [5.0])
        np.testing.assert_array_equal(apply_bias(model, prediction), [[100.0], [100.0]])
        scarce = fit_bias_guard(
            [(prediction, truth), (prediction, truth)], rated_power=100.0,
            max_capacity_ratio=0.05, min_windows=3,
        )
        self.assertEqual(scarce["correction_kw"], [0.0])
        with self.assertRaises(ValueError):
            fit_bias_guard([(prediction, truth)] * 3)
        with self.assertRaises(ValueError):
            apply_bias(model, np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()
