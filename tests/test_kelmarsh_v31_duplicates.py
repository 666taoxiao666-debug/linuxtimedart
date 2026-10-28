import unittest
import numpy as np
import pandas as pd
from scripts.prepare_kelmarsh_multiyear_v31 import canonical_ticks


class DuplicateTests(unittest.TestCase):
    def frame(self, values):
        return pd.DataFrame(values, columns=["timestamp", "power_kw", "wind_speed_mps"])

    def test_existing_whole_row_and_missing_timestamp_preserved(self):
        frame = self.frame([(1, 2., 3.), (1, np.nan, np.nan), (1, 2., np.nan),
                            (1, 2., 3.), (2, np.nan, np.nan), (2, np.nan, np.nan)])
        result, audit = canonical_ticks([frame])
        pd.testing.assert_frame_equal(result, frame.iloc[[0, 4]].reset_index(drop=True))
        self.assertEqual(audit["cells_filled"], 0)
        self.assertEqual(audit["output_rows"], 2)

    def test_conflicting_power_refused(self):
        with self.assertRaisesRegex(ValueError, "Conflicting nonmissing"):
            canonical_ticks([self.frame([(1, 2., 3.), (1, 4., 3.)])])

    def test_conflicting_input_refused(self):
        with self.assertRaisesRegex(ValueError, "Conflicting nonmissing"):
            canonical_ticks([self.frame([(1, 2., 3.), (1, 2., 4.)])])

    def test_complementary_rows_not_coalesced(self):
        with self.assertRaisesRegex(ValueError, "forbidden cell coalescing"):
            canonical_ticks([self.frame([(1, 2., np.nan), (1, np.nan, 3.)])])

    def test_unique_values_negative_and_above_rated_unchanged(self):
        frame = self.frame([(1, -20., 3.), (2, 2100., 4.), (3, np.nan, 5.)])
        result, _ = canonical_ticks([frame])
        pd.testing.assert_frame_equal(result, frame)


if __name__ == "__main__":
    unittest.main()
