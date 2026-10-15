import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from data_provider.kelmarsh_hourly import KelmarshWindows
from test_numeric_foundation import fixture
from utils import numeric_spatial as spatial, numeric_rolling as rolling
from utils.numeric_foundation import observed_anchor


class NumericSpatialTests(unittest.TestCase):
    def test_non_self_exact_clock_schema_and_causal_future_invariance(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            ds = KelmarshWindows(s, s.subset("validation").window_starts[:1], "query")
            unit = int(s.turbines[ds.window_starts[0]])
            for u in range(1, 7):
                s.values[s.turbines == u, 0] = u
                s.values[s.turbines == u, 6] = u * 10
            context = spatial.FarmContext(s)
            result = context.features(ds)
            other = np.array([u for u in range(1, 7) if u != unit])
            expected = [other.mean() - unit, 10 * (other.mean() - unit), other.std(), 10 * other.std(), 1., 1.]
            np.testing.assert_allclose(result[0], np.tile(expected, 7))
            self.assertEqual(spatial.features(ds, "FarmDeltaHGB", context).shape, (1, 119))
            self.assertEqual(spatial.features(ds, "OwnDeltaHGB", context).shape, (1, 77))
            issue = s.dates[ds.window_starts[0] + ds.seq_len - 1]
            s.values[s.dates > issue] = 1e20
            s.raw[s.dates > issue] = -1e20
            ds.truth_and_anchor = lambda: (_ for _ in ()).throw(AssertionError("Query targets read"))
            np.testing.assert_array_equal(context.features(ds), result)

    def test_missing_neighbors_are_flagged_without_sample_dropping(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            ds = KelmarshWindows(s, s.subset("validation").window_starts[:1], "query")
            own = s.turbines[ds.window_starts[0]]
            s.values[s.turbines != own] = np.nan
            result = spatial.FarmContext(s).features(ds)
            self.assertEqual(result.shape, (1, 42))
            np.testing.assert_array_equal(result, np.zeros((1, 42)))

    def test_duplicate_unit_clock_and_wrong_series_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            context = spatial.FarmContext(s)
            with self.assertRaisesRegex(ValueError, "different series"):
                context.features(SimpleNamespace(series=object()))
            s.dates[1], s.turbines[1] = s.dates[0], s.turbines[0]
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                spatial.FarmContext(s)

    def test_ridge_scaler_is_fit_only_and_query_target_not_accessed(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            data, _ = rolling.rolling_partitions(s)
            fit, query = data["o80"]["fit"], data["o80"]["selection"]
            context = spatial.FarmContext(s)
            model = spatial.fit(fit, "FarmDeltaRidge_a100", context)
            np.testing.assert_allclose(model["scale"].mean_, spatial.features(fit, model["model"], context).mean(0))
            before = spatial.predict(model, query, context)
            query.truth_and_anchor = lambda: (_ for _ in ()).throw(AssertionError("Query targets read"))
            np.testing.assert_array_equal(spatial.predict(model, query, context), before)
            self.assertTrue(np.isfinite(before).all())

    def test_zero_delta_retains_exact_raw_anchor_no_output_clipping(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            query = s.subset("validation")
            s.raw[query.window_starts + query.seq_len - 1, -1] = -10.5117
            zero = SimpleNamespace(predict=lambda x: np.zeros(len(x)))
            model = dict(model="FarmDeltaHGB", scale=None, predictor=[zero, zero])
            np.testing.assert_array_equal(spatial.predict(model, query, spatial.FarmContext(s)), observed_anchor(query))

    def test_real_fixed_hgb_completes_exact_iteration_budget_and_unlabeled_inference(self):
        from threadpoolctl import threadpool_limits
        with tempfile.TemporaryDirectory() as root, threadpool_limits(limits=2):
            s = fixture(root)
            data, _ = rolling.rolling_partitions(s)
            fit, query = data["o90"]["fit"], data["o90"]["selection"]
            context = spatial.FarmContext(s)
            model = spatial.fit(fit, "FarmDeltaHGB", context)
            self.assertEqual([head.n_iter_ for head in model["predictor"]], [200, 200])
            query.truth_and_anchor = lambda: (_ for _ in ()).throw(AssertionError("Query targets read"))
            prediction = spatial.predict(model, query, context)
            self.assertEqual(prediction.shape, (len(query), 2))
            self.assertTrue(np.isfinite(prediction).all())

    def test_fixed_stage_reuses_and_rejects_changed_checkpoint_and_contract(self):
        from scripts import train_kelmarsh_numeric_spatial as runner
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            data, _ = rolling.rolling_partitions(s)
            fit, query = data["o70"]["fit"], data["o70"]["selection"]
            context = spatial.FarmContext(s)
            path = Path(root) / "model"
            args = ("FarmDeltaRidge_a100", fit, query, context, {"rated_power_kw": 2050.}, path, {"fixture": True})
            _, first = runner.fixed_fit(*args)
            with patch.object(spatial, "fit", side_effect=AssertionError("Completed stage retrained")):
                _, again = runner.fixed_fit(*args)
            self.assertEqual(first, again)
            with self.assertRaisesRegex(ValueError, "stage changed"):
                runner.fixed_fit(*args[:-1], {"fixture": False})
            with (path / "model.joblib").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "stage changed"):
                runner.fixed_fit(*args)

    def test_fixed_refit_passes_no_selection_and_fallback_never_fits(self):
        from scripts import train_kelmarsh_numeric_spatial as runner
        args = (dict(model="FarmDeltaHGB"), object(), SimpleNamespace(subset=lambda _: "query"), object(), {}, Path("unused"), {})
        with patch.object(runner, "fixed_fit", return_value=("model", {})) as fit, patch.object(spatial, "predict", return_value="forecast"):
            self.assertEqual(runner.refit_prediction(*args), "forecast")
            self.assertIsNone(fit.call_args.args[2])
        with patch.object(runner, "fixed_fit", side_effect=AssertionError("Fallback fitted")), \
             patch.object(runner.f.foundation, "observed_anchor", return_value="exact"):
            self.assertEqual(runner.refit_prediction(dict(model="Persistence"), *args[1:]), "exact")

    def test_frozen_candidate_budget_and_fixed_hgb_not_epoch_search(self):
        root = Path(__file__).resolve().parents[1]
        spec = json.loads((root / "configs/kelmarsh_numeric_spatial_protocol_v13.json").read_text())
        self.assertEqual(spec["candidates"], list(spatial.NAMES))
        self.assertEqual(spec["hgb"], spatial.HGB)
        self.assertFalse(spec["wiki_enabled"])
        self.assertFalse(spec["hgb"]["early_stopping"])
        base = {o: dict(mae=100., rmse=150.) for o in ("o70", "o80", "o90")}
        row = dict(model="FarmDeltaHGB", best_epoch=None, parameters=spatial.capacity("FarmDeltaHGB"),
                   per_origin={o: dict(mae=99., rmse=149.) for o in base})
        self.assertEqual(rolling.select_rolling([row], base)["model"], "FarmDeltaHGB")
        row["per_origin"]["o80"]["rmse"] = 151.
        self.assertEqual(rolling.select_rolling([row], base)["model"], "Persistence")


if __name__ == "__main__":
    unittest.main()
