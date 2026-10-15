import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from data_provider.kelmarsh_hourly import KelmarshWindows
from test_numeric_foundation import fixture
from utils import numeric_foundation as foundation, numeric_rolling as rolling, numeric_shrink as shrink


class NumericShrinkTests(unittest.TestCase):
    def test_nested_calibration_is_disjoint_causal_and_original_scaler_available(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            origins, _ = rolling.rolling_partitions(s)
            for data in origins.values():
                fit, calibration, audit = shrink.nested_partition(s, data["cut"])
                first_issue = calibration.target_times()[:, 0].min() - np.timedelta64(1, "h")
                self.assertLessEqual(fit.target_times().max(), first_issue)
                self.assertLessEqual(s.bounds["base_fit_end"], first_issue)
                self.assertLessEqual(calibration.target_times().max(), data["selection"].target_times()[:, 0].min() - np.timedelta64(1, "h"))
                self.assertFalse(set(fit.window_starts).intersection(calibration.window_starts))
                self.assertTrue(audit["original_scaler_causal"])
            with self.assertRaisesRegex(ValueError, "scaler uses outcomes"):
                shrink.nested_partition(s, s.bounds["base_fit_end"])

    def test_joint_coefficients_quadratic_solution_and_mae_feasible_boundary(self):
        d = np.ones((4, 2))
        y = np.column_stack((np.full(4, .25), [-1., .1, .1, 10.]))
        alpha = shrink.joint_coefficients(d, y)
        np.testing.assert_allclose(alpha, [.25, .2], atol=1e-10)
        self.assertTrue((np.mean(abs(y - d * alpha), axis=0) <= np.mean(abs(y), axis=0) + 1e-10).all())
        self.assertTrue((np.mean((y - d * alpha)**2, axis=0) <= np.mean(y**2, axis=0) + 1e-10).all())

    def test_wrong_direction_zero_residual_and_invalid_calibration(self):
        d = np.ones((4, 2))
        y = np.column_stack(([-1., -1., -1., 10.], np.full(4, -1.)))
        np.testing.assert_array_equal(shrink.joint_coefficients(d, y), [0., 0.])
        np.testing.assert_array_equal(shrink.joint_coefficients(d * 0, y), [0., 0.])
        with self.assertRaisesRegex(ValueError, "Finite paired"):
            shrink.joint_coefficients(d, y[:, :1])
        with self.assertRaisesRegex(ValueError, "Finite paired"):
            shrink.joint_coefficients(d * np.nan, y)

    def test_certificate_binds_actual_expert_scaler_and_supports_unlabeled_inference(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            origins, _ = rolling.rolling_partitions(s)
            fit, calibration, _ = shrink.nested_partition(s, origins["o90"]["cut"])
            model = foundation.fit_ridge(fit, 100.)
            cert = shrink.make_certificate(model, calibration, "verified_file_hash")
            query = origins["o90"]["selection"]
            query.truth_and_anchor = lambda: (_ for _ in ()).throw(AssertionError("Query outcomes used"))
            self.assertTrue(np.isfinite(shrink.predict_bound(model, query, cert, "verified_file_hash")).all())
            changed = copy.deepcopy(model)
            changed["coefficients"][0, 0] += 1e-3
            with self.assertRaisesRegex(ValueError, "mismatch"):
                shrink.predict_bound(changed, query, cert, "verified_file_hash")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                shrink.predict_bound(model, query, cert, "another_file")
            cert["scaler_sha256"] = "tampered_scaler"
            with self.assertRaisesRegex(ValueError, "mismatch"):
                shrink.predict_bound(model, query, cert, "verified_file_hash")

    def test_zero_coefficients_preserve_negative_raw_anchor_exactly(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            origins, _ = rolling.rolling_partitions(s)
            fit, calibration, _ = shrink.nested_partition(s, origins["o90"]["cut"])
            model = foundation.fit_ridge(fit, 100.)
            cert = shrink.make_certificate(model, calibration, "hash")
            query = origins["o90"]["selection"]
            s.raw[query.window_starts + query.seq_len - 1, -1] = -10.5117
            cert["coefficients"] = [0., 0.]
            np.testing.assert_array_equal(shrink.predict_bound(model, query, cert, "hash"), foundation.observed_anchor(query))

    def test_calibration_resume_reuses_and_rejects_modified_certificate(self):
        from scripts import train_kelmarsh_numeric_shrink as runner
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            fit, calibration, _ = shrink.nested_partition(s, s.bounds["train_end"])
            model = foundation.fit_ridge(fit, 100.)
            directory = Path(root) / "calibration"
            cert = runner.calibrate(model, calibration, "hash", directory, {"fixture": True})
            with patch.object(shrink, "make_certificate", side_effect=AssertionError("Completed calibration repeated")):
                self.assertEqual(runner.calibrate(model, calibration, "hash", directory, {"fixture": True}), cert)
            with (directory / "certificate.json").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(ValueError, "calibration changed"):
                runner.calibrate(model, calibration, "hash", directory, {"fixture": True})

    def test_final_expert_reuse_is_exact_and_never_retrained(self):
        from scripts import train_kelmarsh_numeric_shrink as runner
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            s = fixture(root)
            fit, calibration, _ = shrink.nested_partition(s, s.bounds["train_end"])
            prior, v12, v11 = root / "v13", root / "v12", root / "v11"
            runner.f.old.write(prior / "protocol.json", dict(previous_directory="v12"))
            runner.f.old.write(v12 / "protocol.json", dict(previous_directory="v11"))
            directory = v11 / "inner/DeltaRidge_a100"
            model, done = runner.f.ridge_fit("DeltaRidge_a100", fit, calibration, 100., {"rated_power_kw": 2050.}, directory, {"fixture": True})
            with patch.object(runner, "ROOT", root), patch.object(runner.f, "ridge_fit", side_effect=AssertionError("Expert refit")):
                actual, sha = runner.final_expert(prior, fit, calibration, root / "new")
            self.assertEqual(sha, done["model_sha256"])
            self.assertEqual(shrink.arrays_digest(actual), shrink.arrays_digest(model))
            with (directory / "model.npz").open("ab") as stream:
                stream.write(b"tamper")
            with patch.object(runner, "ROOT", root), self.assertRaisesRegex(ValueError, "artifact changed"):
                runner.final_expert(prior, fit, calibration, root / "new")

    def test_fallback_does_not_fit_or_calibrate_and_outer_has_no_coefficient_selection(self):
        from scripts import train_kelmarsh_numeric_shrink as runner
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            with patch.object(runner, "final_expert", side_effect=AssertionError("Fallback loaded/fitted expert")), \
                 patch.object(runner, "calibrate", side_effect=AssertionError("Fallback calibrated")):
                prediction = runner.final_prediction(dict(model="Persistence"), s, Path("unused"), Path(root), {})
            np.testing.assert_array_equal(prediction, foundation.observed_anchor(s.subset("validation")))
            fit, calibration, _ = shrink.nested_partition(s, s.bounds["train_end"])
            model = foundation.fit_ridge(fit, 100.)
            cert = shrink.make_certificate(model, calibration, "hash")
            with patch.object(runner, "final_expert", return_value=(model, "hash")), patch.object(runner, "calibrate", return_value=cert) as calibrate:
                runner.final_prediction(dict(model="RidgeJointShrink"), s, Path("unused"), Path(root), {})
                self.assertLess(calibrate.call_args.args[1].target_times().max(), s.bounds["train_end"])

    def test_protocol_preserves_one_finite_algorithm_without_wiki_or_extra_seeds(self):
        from scripts import train_kelmarsh_numeric_shrink as runner
        root = Path(__file__).resolve().parents[1]
        spec = json.loads((root / "configs/kelmarsh_numeric_shrink_protocol_v14.json").read_text())
        self.assertEqual(spec["candidates"], list(runner.NAMES))
        self.assertEqual(spec["ridge_alpha"], 100.)
        self.assertEqual(spec["expert_fit_fraction"], .9)
        self.assertEqual((spec["fold"], spec["seed"]), (1, 2024))
        self.assertFalse(spec["wiki_enabled"])


if __name__ == "__main__":
    unittest.main()
