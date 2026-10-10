import inspect
import unittest
import numpy as np
from utils.physical_scale_guard import fit_guard, apply_guard, three_blocks


class PhysicalScaleGuardTests(unittest.TestCase):
    def test_two_block_support_and_joint_gain_required(self):
        model = {"states": [{"state": s, "weights": [1.] * 12} for s in (-1, 0, 1)]}
        model["states"][1]["weights"][0] = .75
        ref, anchor = np.full((2, 12), 100.), np.zeros((2, 12))
        good = dict(prediction=ref, persistence=anchor, truth=anchor, trend=np.zeros(2, int))
        bad = dict(good, truth=ref)
        bins = np.zeros(2, int)
        guard = fit_guard(model, [(good, bins), (good, bins)], min_windows=2)
        self.assertTrue(guard["trust"][1][0][0])
        pred = apply_guard(guard, ref, anchor, good["trend"], bins)
        np.testing.assert_array_equal(pred[:, 0], [75., 75.])
        np.testing.assert_array_equal(pred[:, 1:], ref[:, 1:])
        rejected = fit_guard(model, [(good, bins), (bad, bins)], min_windows=2)
        np.testing.assert_array_equal(apply_guard(rejected, ref, anchor, good["trend"], bins), ref)
        scarce = fit_guard(model, [(good, bins), (good, bins)], min_windows=3)
        self.assertFalse(np.asarray(scarce["trust"]).any())
        self.assertNotIn("truth", inspect.signature(apply_guard).parameters)
        with self.assertRaises(ValueError):
            fit_guard(model, [(good, bins)] * 3)

    def test_fixed_three_block_boundaries_purge_crossing_forecasts(self):
        start = np.datetime64("2023-06-01", "ns")
        end = start + np.timedelta64(4, "D")
        starts = start + np.array([0, 2800, 3000, 4200, 4500]) * np.timedelta64(1, "m")
        times = starts[:, None] + np.arange(12) * np.timedelta64(10, "m")
        a, b, c, middle, check_start = three_blocks(times, start, end)
        np.testing.assert_array_equal(a, [1, 0, 0, 0, 0])
        np.testing.assert_array_equal(b, [0, 0, 1, 1, 0])
        np.testing.assert_array_equal(c, [0, 0, 0, 0, 1])
        self.assertEqual(middle, start + np.timedelta64(2, "D"))
        self.assertEqual(check_start, start + np.timedelta64(3, "D"))


if __name__ == "__main__":
    unittest.main()
