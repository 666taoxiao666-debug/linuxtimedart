import tempfile
from pathlib import Path
import unittest
import numpy as np
from scripts.run_x1 import persistence, checked, digest, curve_points, fit_curves, curve_predict
from scripts.run_x1 import normal_cells, farm_metrics, tree_features


class X1Tests(unittest.TestCase):
    def test_tree_features_ignore_targets_and_keep_history(self):
        x = np.arange(2*24*50).reshape(2, 24, 50)
        pack = dict(x50=x, turbines=np.array([0, 2]), retrofit=np.array([0, 1]), truth=np.zeros((2, 4)))
        result = tree_features(pack, True)
        self.assertEqual(result.shape, (2, 100))
        np.testing.assert_array_equal(result[:, :50], x[:, -1])
        np.testing.assert_array_equal(result[:, 50:74], x[:, :, -1])
        np.testing.assert_array_equal(result[:, 74:98], x[:, :, 0])
        pack['truth'][:] = 999
        np.testing.assert_array_equal(result, tree_features(pack, True))

    def test_normal_mask_only_descriptive(self):
        wind = np.array([[5., 8., 8., np.nan]])
        truth = np.array([[1., 20., 80., 70.]])
        before = truth.copy()
        mask, audit = normal_cells(wind, truth, {'0': [[6., 100.], [10., 100.]]}, np.array([0]), 100.)
        np.testing.assert_array_equal(mask, [[False, False, True, False]])
        np.testing.assert_array_equal(truth, before)
        self.assertEqual(audit['missing_wind_cells'], 1)

    def test_farm_coverage_and_capacity(self):
        issue = np.array([1, 1, 1, 1, 2])
        units = np.array([0, 1, 2, 3, 0])
        truth = np.full((5, 4), 50.)
        m = farm_metrics(truth+10, truth, issue, units, 100., 5)
        self.assertEqual(m['issue_count'], 1)
        self.assertAlmostEqual(m['nmae'], .1)
        self.assertAlmostEqual(m['one_minus_nrmse'], .9)

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
