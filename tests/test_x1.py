import tempfile
from pathlib import Path
import unittest
import numpy as np
from scripts.run_x1 import persistence, checked, digest


class X1Tests(unittest.TestCase):
    def test_persistence_uses_anchor_not_labels(self):
        y = np.arange(12, dtype=float).reshape(3, 4)
        anchor = np.array([20., 50., 70.])
        result = persistence(y, anchor)
        np.testing.assert_array_equal(result, np.repeat(anchor[:, None], 4, axis=1))
        np.testing.assert_array_equal(result, persistence(y+1000, anchor))

    def test_invalid_cohort_not_repaired(self):
        with self.assertRaises(ValueError):
            persistence(np.zeros((2, 4)), np.array([1., np.nan]))
        with self.assertRaises(ValueError):
            persistence(np.zeros((2, 4)), np.zeros(3))

    def test_hash_identity_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'array'
            path.write_bytes(b'original')
            identity = digest(path)
            self.assertEqual(checked(path, identity), path)
            path.write_bytes(b'modified')
            with self.assertRaises(ValueError):
                checked(path, identity)


if __name__ == '__main__':
    unittest.main()
