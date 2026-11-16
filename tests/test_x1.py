import tempfile
from pathlib import Path
import unittest
import numpy as np
from scripts.run_x1 import persistence, checked, digest, curve_points, fit_curves, curve_predict


class X1Tests(unittest.TestCase):
    def test_curve_validity_and_fallback(self):
        pack = dict(gF=np.full((40, 13), 5.25), gF_mask=np.ones((40, 13)),
                    truth=np.full((40, 4), 100.), turbines=np.zeros(40),
                    anchor=np.full(40, 70.), retrofit=np.ones(40))
        curves = fit_curves([pack], True)
        pack['gF_mask'][0, 6] = 0
        pack['gF'][0, 6] = 0
        pred, audit = curve_predict(pack, curves)
        self.assertEqual(pred[0, 0], 70.)
        self.assertEqual(pred[0, 1], 100.)
        self.assertEqual(audit['missing_gfs_cells'], 1)
        self.assertEqual(pred.shape, pack['truth'].shape)

    def test_curve_excludes_missing_and_sparse_bins(self):
        w = np.r_[np.ones(30), np.full(29, 2.), np.zeros(40)]
        y = np.r_[np.full(30, 10.), np.full(29, 50.), np.full(40, 999.)]
        pts = curve_points(w, y, w > 0)
        np.testing.assert_allclose(pts, [[1.25, 10.]])

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
