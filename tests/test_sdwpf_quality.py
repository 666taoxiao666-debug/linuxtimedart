import unittest

import pandas as pd

from data_provider.sdwpf_quality import aggregate_blade_pitch


class BladePitchQualityTests(unittest.TestCase):
    def test_robust_mode_only_replaces_disagreeing_same_row_sensor(self):
        frame = pd.DataFrame({
            "Pab1": [1.0, 40.0, 2.0],
            "Pab2": [1.1, 1.0, 2.0],
            "Pab3": [0.9, 1.0, 2.0],
            "power": [300.0, 400.0, 500.0],
        })
        columns = ["Pab1", "Pab2", "Pab3"]
        original, _ = aggregate_blade_pitch(frame, columns)
        repaired, disagreement = aggregate_blade_pitch(frame, columns, robust=True)
        self.assertEqual(disagreement.tolist(), [False, True, False])
        self.assertEqual(original.tolist(), [1.0, 14.0, 2.0])
        self.assertEqual(repaired.tolist(), [1.0, 1.0, 2.0])
        self.assertEqual(frame["power"].tolist(), [300.0, 400.0, 500.0])


if __name__ == "__main__":
    unittest.main()
