import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from data_provider.kelmarsh_hourly import FEATURES, valid_window_starts, time_bounds
from scripts.train_kelmarsh_hourly import read, model_args, model_metrics, extract, numeric_residual_control, safe, report
from utils import kelmarsh_hourly_wiki as wiki
from utils.forecast_report import build_forecast_report


ROOT = Path(__file__).resolve().parents[1]


class KelmarshHourlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_future_weather_missing_is_not_a_label_selection_rule(self):
        x = np.ones((8, 3))
        x[3, 0] = np.nan
        dates = np.arange(8).astype("timedelta64[h]") + np.datetime64("2016-05-01")
        starts, _ = valid_window_starts(x, dates, np.ones(8), 3, 2, 1)
        self.assertIn(0, starts)  # missing future weather must not remove this forecast
        self.assertNotIn(1, starts)  # missing past weather does remove this forecast
        x[4, -1] = np.nan
        starts, _ = valid_window_starts(x, dates, np.ones(8), 3, 2, 1)
        self.assertNotIn(0, starts)

    def test_windows_do_not_cross_turbines_or_gaps(self):
        x = np.ones((12, 3))
        dates = np.tile(np.arange(6).astype("timedelta64[h]") + np.datetime64("2016-05-01"), 2)
        turbines = np.repeat([1, 2], 6)
        starts, _ = valid_window_starts(x, dates, turbines, 3, 2, 1)
        self.assertEqual(starts.tolist(), [0, 1, 6, 7])
        dates[4] += np.timedelta64(1, "h")
        starts, _ = valid_window_starts(x, dates, turbines, 3, 2, 1)
        self.assertNotIn(0, starts)

    def test_chronological_blocks_and_sealed_suffix(self):
        times = pd.date_range("2016-05-01", periods=1000, freq="1h").to_numpy()
        for fold in (0, 1, 2):
            bounds = time_bounds(times, fold)
            stages = [bounds[k] for k in ("base_fit_end", "base_selection_end", "memory_fit_end",
                       "calibration_early_end", "train_end", "validation_end")]
            self.assertTrue(all(a < b for a, b in zip(stages, stages[1:])))
            self.assertLessEqual(bounds["validation_end"], bounds["sealed_test_start"])

    def test_old_capacity_clipping_default_and_new_raw_reporting(self):
        class Data:
            scale = False
        true = np.array([[-10., 2100.], [0., 1000.]])
        pred = np.array([[-8., 2080.], [50., 950.]])
        # Only plots are mocked; actual metrics, tables and prediction arrays run.
        with tempfile.TemporaryDirectory() as tmp, patch("utils.forecast_report._plot_horizon"), \
                patch("utils.forecast_report._plot_horizon_segments"), patch("utils.forecast_report._plot_examples"), \
                patch("utils.forecast_report._plot_continuous_forecast"), patch("utils.forecast_report._plot_scatter"), \
                patch("utils.forecast_report._plot_accuracy_overview"), patch("utils.forecast_report._plot_residuals"), \
                patch("utils.forecast_report._plot_per_turbine"), patch("utils.forecast_report._plot_power_bins"):
            raw = build_forecast_report(pred, true, Data(), str(Path(tmp) / "raw"), rated_power=2050,
                                        step_minutes=60, clip_to_capacity=False)
            np.testing.assert_allclose(raw["mae"], np.abs(pred - true).mean())
            with np.load(Path(tmp) / "raw/predictions.npz") as saved:
                np.testing.assert_array_equal(saved["truth_original"], true)
            old = build_forecast_report(pred, true, Data(), str(Path(tmp) / "old"), rated_power=2050)
            expected = np.abs(np.clip(pred, 0, 2050) - np.clip(true, 0, 2050)).mean()
            self.assertAlmostEqual(old["mae"], expected)

    def test_float32_baseline_report_has_no_extra_normalization_quantization(self):
        from types import SimpleNamespace
        class Data:
            scale = True
            target = "power"
            feature_columns = ["Wspd", "power"]
            scaler = SimpleNamespace(mean_=np.array([5., 428.22492242214264]),
                                     scale_=np.array([2., 459.78659052225674]))
            def truth_and_anchor(self):
                return np.array([[0., 2100.], [350., 600.]]), np.array([[200., 200.], [500., 500.]])
            def target_times(self):
                return np.array([["2016-10-01T01", "2016-10-01T02"],
                                 ["2016-10-01T04", "2016-10-01T05"]], dtype="datetime64[h]")
            turbines = np.array([1, 2])
            window_starts = np.array([0, 1])
            def __len__(self):
                return len(self.window_starts)
        prediction = np.array([[-.0042, 2071.9233], [372.3372, 642.0123]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp, patch("utils.forecast_report._plot_horizon"), \
                patch("utils.forecast_report._plot_horizon_segments"), patch("utils.forecast_report._plot_examples"), \
                patch("utils.forecast_report._plot_continuous_forecast"), patch("utils.forecast_report._plot_scatter"), \
                patch("utils.forecast_report._plot_accuracy_overview"), patch("utils.forecast_report._plot_residuals"), \
                patch("utils.forecast_report._plot_per_turbine"), patch("utils.forecast_report._plot_power_bins"):
            result = report("Ridge", prediction, Data(), Path(temp), 2050.)
            self.assertAlmostEqual(result["metrics"]["mae"], np.abs(prediction.astype(float) - Data().truth_and_anchor()[0]).mean(), places=10)

    def test_actual_model_implementations_forward_and_pretrain(self):
        from models import DLinear, PatchTST, PromptTimeDART
        protocol = read(ROOT / "configs/kelmarsh_hourly_protocol_v1.json")
        x = torch.randn(2, protocol["input_len"], len(FEATURES))
        for name, module in (("DLinear", DLinear), ("PatchTST", PatchTST), ("PromptTimeDART", PromptTimeDART)):
            args = model_args(name, None, protocol, torch.device("cpu"))
            model = module.Model(args).eval()
            if name == "PromptTimeDART":
                model.regime_down_thresh.fill_(-.1)
                model.regime_up_thresh.fill_(.1)
            output = model(x)
            self.assertEqual(output.shape[:2], (2, protocol["pred_len"]))
            self.assertTrue(torch.isfinite(output).all())
        args = model_args("PromptTimeDART", None, protocol, torch.device("cpu"), task="pretrain")
        model = PromptTimeDART.Model(args).eval()
        model.regime_down_thresh.fill_(-.1)
        model.regime_up_thresh.fill_(.1)
        self.assertEqual(extract(model(x)).shape, x.shape)

    def test_expert_and_policy_identity_and_held_out_labels(self):
        n, h = 200, 2
        rng = np.random.default_rng(2024)
        context = {"power_ratio": rng.uniform(.1, .5, n), "wind_mean_mps": rng.uniform(3, 8, n),
                   "wind_std_mps": rng.uniform(1, 2, n), "power_change_kw": rng.normal(0, 10, n),
                   "wind_change_mps": rng.normal(0, 1, n), "power_bin": np.ones(n, int), "joint_bin": np.ones(n, int)}
        for name in wiki.CONTROLS:
            context[name] = rng.normal(size=n)
        anchor = np.full((n, h), 500.)
        block = {"context": context, "anchor": anchor, "reference": anchor.copy(), "truth": anchor + 20,
                 "turbines": np.tile([1, 2, 3, 4, 5], 40), "events": np.tile([1, 0, 0, 0], (n, 1)).astype(bool)}
        contract = {"source_turbines": [1, 2, 3, 4, 5], "pred_len": h, "rated_power_kw": 2050.,
                    "base_checkpoint_sha256": "a" * 64}
        settings = {"use_controls": True, "neighbors": 8, "min_neighbor_turbines": 2,
                    "min_fit_windows": 20, "support_quantile": .95, "max_scale": .2, "composition_max_scale": .05}
        expert = wiki.fit_experts(block, contract, settings)
        a, b = copy.deepcopy(block), copy.deepcopy(block)
        a["times"] = np.tile(np.array(["2016-08-01", "2016-08-01T01"], dtype="datetime64[h]"), (n, 1))
        b["times"] = a["times"] + np.timedelta64(24, "h")
        policy = wiki.fit_policy(expert, [a, b], contract, {"min_windows": 1, "min_turbines": 1, "positive_fraction": 2/3})
        output = wiki.apply(expert, policy, a, contract)
        self.assertEqual(output["prediction"].shape, (n, h))
        inference = {key: value for key, value in a.items() if key not in ("truth", "times", "turbines")}
        np.testing.assert_array_equal(wiki.apply(expert, policy, inference, contract)["prediction"], output["prediction"])
        quiet = dict(inference, events=np.zeros_like(a["events"]))
        np.testing.assert_array_equal(wiki.apply(expert, policy, quiet, contract)["prediction"], a["reference"])
        bad = dict(contract, base_checkpoint_sha256="b" * 64)
        wiki.validate(expert, bad)  # raw transition memory can survive a base change
        with self.assertRaises(ValueError):
            wiki.apply(expert, policy, a, bad)
        changed = copy.deepcopy(expert)
        changed["settings"]["max_scale"] = .1
        with self.assertRaises(ValueError):
            wiki.apply(changed, policy, a, contract)
        block["turbines"][0] = 6
        with self.assertRaises(ValueError):
            wiki.fit_experts(block, contract, settings)
        with self.assertRaises(ValueError):
            wiki.fit_policy(expert, [a, a], contract, {"min_windows": 1, "min_turbines": 1, "positive_fraction": 2/3})
        p = read(ROOT / "configs/kelmarsh_hourly_protocol_v1.json")
        prediction, artifact = numeric_residual_control({"memory": a, "calibration_early": a,
            "calibration_late": b}, a, contract, p)
        self.assertEqual(prediction.shape, (n, h))
        import json
        json.dumps(safe(artifact), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
