import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_numeric_foundation import fixture
from test_numeric_subhour import tables_for
from utils import numeric_subhour as endpoint, numeric_subhour_filter as filt


class NumericFilterTests(unittest.TestCase):
    def test_exact_convex_solver_recovers_known_interior_and_boundary_optima(self):
        rng = np.random.default_rng(2024)
        x = rng.normal(size=(300, 6))
        x -= x.mean(1, keepdims=True)
        for actual in (np.array([.05, .1, .15, .2, .2, .3]), np.array([0., 0., 0., .1, .3, .6])):
            y = x @ actual
            weights = filt.simplex_least_squares(x, y, np.ones(6) / 6)
            np.testing.assert_allclose(weights, actual, atol=1e-10)
            self.assertAlmostEqual(float(weights.sum()), 1., places=12)
            self.assertTrue((weights >= 0).all())

    def test_flat_signal_ties_keep_baseline_and_fit_never_worsens_prior(self):
        prior = filt.endpoint_weights([.4])[0]
        np.testing.assert_array_equal(filt.simplex_least_squares(np.zeros((10, 6)), np.ones(10), prior), prior)
        rng = np.random.default_rng(2024)
        x, y = rng.normal(size=(40, 6)), rng.normal(size=40)
        actual = filt.simplex_least_squares(x, y, prior)
        self.assertLessEqual(np.mean((x @ actual - y)**2), np.mean((x @ prior - y)**2) + 1e-10)
        with self.assertRaisesRegex(ValueError, "simplex prior"):
            filt.simplex_least_squares(x, y * np.nan, prior)

    def test_prior_filter_reproduces_frozen_v15_and_future_targets_unused(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            ds = s.subset("validation")
            ds.window_starts = ds.window_starts[:1]
            tables = tables_for(s)
            beta = [.53, .35]
            expected = endpoint.predict(beta, ds, tables)
            np.testing.assert_allclose(filt.predict(filt.endpoint_weights(beta), ds, tables), expected, atol=1e-10, rtol=0)
            issue = ds.dates[ds.window_starts[0] + ds.seq_len - 1]
            for series in tables.values():
                series.loc[series.index >= issue] = 1e12
            ds.truth_and_anchor = lambda: (_ for _ in ()).throw(AssertionError("Query labels accessed"))
            np.testing.assert_allclose(filt.predict(filt.endpoint_weights(beta), ds, tables), expected, atol=1e-10, rtol=0)
            with self.assertRaisesRegex(ValueError, "nonnegative sum1"):
                filt.predict(np.ones((2, 6)), ds, tables)

    def test_missing_ticks_and_unit_swap_fail_without_changing_cohort(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            ds = s.subset("selection")
            tables = tables_for(s)
            tables[1], tables[2] = tables[2], tables[1]
            with self.assertRaisesRegex(ValueError, "reconstruct"):
                filt.centered_ticks(ds, tables)
            tables = tables_for(s)
            tables[1].iloc[:] = np.nan
            with self.assertRaisesRegex(ValueError, "Missing source tick"):
                filt.centered_ticks(ds, tables)

    def test_three_block_gate_uses_both_v15_and_persistence(self):
        refs = {key: dict(v15=dict(mae=10., rmse=20.), persistence=dict(mae=11., rmse=21.)) for key in ("o70", "o80", "o90")}
        records = {key: dict(mae=9., rmse=19.) for key in refs}
        self.assertEqual(filt.select_filter(records, refs)["model"], "SimplexSubhourFilter")
        records["o70"]["mae"] = 10.5  # Beats Persistence, but NOT the current v15.
        self.assertTrue(filt.select_filter(records, refs)["reused_v15"])
        with self.assertRaisesRegex(ValueError, "three declared"):
            filt.select_filter({"o70": records["o70"]}, refs)

    def test_cached_baseline_and_new_stage_reuse_actual_checkpoint_no_retraining(self):
        from scripts import train_kelmarsh_numeric_filter as runner
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            fit = s.subset("fit")
            tables = tables_for(s)
            baseline_dir = Path(root) / "v15"
            beta, baseline = runner.previous.fit_stage(fit, tables, baseline_dir, {"fixture": True})
            actual, done = runner.cached_baseline(baseline_dir, fit, tables)
            np.testing.assert_array_equal(actual, beta)
            newdir = Path(root) / "filter"
            weights, _ = runner.fit_stage(fit, tables, beta, done["model_sha256"], newdir, {"fixture": True})
            with patch.object(filt, "fit_weights", side_effect=AssertionError("Completed filter repeated")):
                again, _ = runner.fit_stage(fit, tables, beta, done["model_sha256"], newdir, {"fixture": True})
            np.testing.assert_array_equal(weights, again)
            with (newdir / "model.npz").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "fit changed"):
                runner.fit_stage(fit, tables, beta, done["model_sha256"], newdir, {"fixture": True})
            with (baseline_dir / "model.npz").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "executor/window identity"):
                runner.cached_baseline(baseline_dir, fit, tables)

    def test_rejected_candidate_never_queries_outer_or_refits_anything(self):
        from scripts import train_kelmarsh_numeric_filter as runner
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            runner.f.old.write(root / "previous/reports/SelectedNumeric/result.json", dict(metrics={"mae": 97.592426}))
            with patch.object(runner, "cached_baseline", side_effect=AssertionError("Fallback loaded final executor")), \
                 patch.object(runner, "fit_stage", side_effect=AssertionError("Fallback trained")), \
                 patch.object(runner.f, "outer_report", side_effect=AssertionError("Rejected filter used outer")):
                runner.finish_frozen_selection({"reused_v15": True}, root / "previous", None, None, None,
                    root / "new", {}, {}, {}, None)
            result = runner.f.old.read(root / "new/result.json")
            self.assertFalse(result["new_outer_evaluation"])
            self.assertFalse(result["primary_joint_success"])

    def test_protocol_is_one_algorithm_and_stops_after_training_rejection(self):
        from scripts import train_kelmarsh_numeric_filter as runner
        root = Path(__file__).resolve().parents[1]
        p = json.loads((root / "configs/kelmarsh_numeric_filter_protocol_v16.json").read_text())
        self.assertEqual(p["candidates"], [runner.NAME])
        self.assertEqual(p["protocol_id"], runner.PROTOCOL_ID)
        self.assertFalse(p["wiki_enabled"])
        self.assertEqual((p["fold"], p["seed"]), (1, 2024))
        self.assertIn("no new outer evaluation", p["selection"])


if __name__ == "__main__":
    unittest.main()
