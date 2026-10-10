import copy
import unittest
import numpy as np

from utils.residual_scale_calibration import chronological_masks, fit_scale, apply_scale


class ResidualScaleTests(unittest.TestCase):
    def test_midpoint_purges_overlapping_targets(self):
        start = np.datetime64("2023-06-12T00:00", "ns")
        times = start + np.array([0, 710, 720])[:, None] * np.timedelta64(1, "m") + np.arange(12) * np.timedelta64(10, "m")
        fit, check, midpoint = chronological_masks(times, start, start + np.timedelta64(1, "D"))
        np.testing.assert_array_equal(fit, [True, False, False])
        np.testing.assert_array_equal(check, [False, False, True])
        self.assertEqual(midpoint, start + np.timedelta64(12, "h"))
        with self.assertRaises(ValueError):
            chronological_masks(times, start, times[-1, -1])

    def test_damping_reduces_overcorrection_and_is_bounded(self):
        anchor = np.full((100, 12), 300.)
        base = anchor - 80
        truth = anchor - 20
        model = fit_scale(base, anchor, truth, np.zeros(100, int))
        np.testing.assert_array_equal(model["states"][1]["weights"], [.25] * 12)
        predicted = apply_scale(model, base, anchor, np.zeros(100, int))
        np.testing.assert_allclose(predicted, truth)
        self.assertTrue(((predicted >= base) & (predicted <= anchor)).all())

    def test_perfect_model_and_insufficient_states_are_preserved(self):
        base = np.ones((100, 12)) * 300
        anchor = base + 50
        model = fit_scale(base, anchor, base, np.zeros(100, int))
        for state in model["states"]:
            self.assertEqual(state["weights"], [1.] * 12)
        np.testing.assert_array_equal(apply_scale(model, base, anchor, np.zeros(100, int)), base)

    def test_application_never_receives_check_labels(self):
        base = np.ones((100, 12)) * 300
        anchor = base + 50
        model = fit_scale(base, anchor, anchor, np.zeros(100, int))
        self.assertEqual(model["states"][1]["weights"], [0.] * 12)
        other_targets = np.ones_like(base) * 900  # absent from apply_scale's API
        np.testing.assert_array_equal(apply_scale(model, base, anchor, np.zeros(100, int)), anchor)
        self.assertFalse(np.array_equal(anchor, other_targets))
        bad = copy.deepcopy(model)
        bad["states"][1]["weights"][0] = 2
        with self.assertRaises(ValueError):
            apply_scale(bad, base, anchor, np.zeros(100, int))


if __name__ == "__main__":
    unittest.main()
