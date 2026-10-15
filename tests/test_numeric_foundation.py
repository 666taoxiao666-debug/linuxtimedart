import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from data_provider.kelmarsh_hourly import KelmarshSeries, KelmarshWindows
from models.CausalResidualBase import CausalResidualBase
from utils import numeric_foundation as numeric


def fixture(directory):
    dates = pd.date_range("2016-05-01", periods=240, freq="h", tz="UTC")
    n = np.arange(240)
    frame = pd.concat([pd.DataFrame(dict(timestamp_hour_end_utc=dates.astype(str), turbine_id=unit,
        wind_speed_mps=5 + np.sin(n / 12), ambient_temp_c=10 + np.cos(n / 24), nacelle_position_deg=n % 360,
        pitch_a_deg=2., pitch_b_deg=2., pitch_c_deg=2., rotor_speed_rpm=10 + np.sin(n / 12),
        power_kw=100 + 2 * n + unit * 3 + 20 * np.sin(n / 12))) for unit in range(1, 7)], ignore_index=True)
    path = Path(directory) / "fixture.csv"
    frame.to_csv(path, index=False)
    return KelmarshSeries(path, fold=1, input_len=72, horizon=2, stride=3,
                          start="2016-05-01T00:00:00Z", end="2016-05-11T00:00:00Z")


class NumericFoundationTests(unittest.TestCase):
    def test_partitions_recover_recent_labels_without_changing_outer_or_scaler(self):
        with tempfile.TemporaryDirectory() as directory:
            series = fixture(directory)
            before = series.values.copy()
            mean, scale = series.scaler.mean_.copy(), series.scaler.scale_.copy()
            outer = series.subset("validation").window_starts.copy()
            data, audit = numeric.partitions(series)
            self.assertGreater(len(data["fit"]), len(series.subset("fit")))
            self.assertTrue((data["fit"].target_times().max() <= data["selection"].target_times()[:, 0].min() - np.timedelta64(1, "h")))
            self.assertTrue((data["outer_refit"].target_times() < series.bounds["train_end"]).all())
            self.assertEqual(set(data["outer_refit"].turbines[data["outer_refit"].window_starts]), set(range(1, 7)))
            np.testing.assert_array_equal(outer, series.subset("validation").window_starts)
            np.testing.assert_array_equal(before, series.values)
            np.testing.assert_array_equal(mean, series.scaler.mean_)
            np.testing.assert_array_equal(scale, series.scaler.scale_)
            self.assertFalse(audit["wiki_memory_or_labels_reserved"])

    def test_lag_features_have_declared_schema_and_no_long_prefix_dependence(self):
        x = np.arange(2 * 72 * 7, dtype=float).reshape(2, 72, 7)
        z = numeric.lag_features(x)
        self.assertEqual(z.shape, (2, 77))
        np.testing.assert_array_equal(z[:, :7], x[:, -1])
        np.testing.assert_array_equal(z[:, 7:14], x[:, -1] - x[:, -2])
        x[:, :-25] = -1e10
        np.testing.assert_array_equal(numeric.lag_features(x), z)
        with self.assertRaises(ValueError):
            numeric.lag_features(np.zeros((2, 24, 7)))

    def test_ridge_inference_does_not_read_query_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            series = fixture(directory)
            data, _ = numeric.partitions(series)
            model = numeric.fit_ridge(data["fit"], 10.)
            ds = KelmarshWindows(series, data["selection"].window_starts[:1], "unlabeled_query")
            before = numeric.ridge_predict(model, ds)
            rows = ds.window_starts[0] + ds.seq_len + np.arange(ds.pred_len)
            series.raw[rows, -1] = 1e12
            ds.truth_and_anchor = lambda: (_ for _ in ()).throw(AssertionError("Query outcomes accessed"))
            np.testing.assert_array_equal(before, numeric.ridge_predict(model, ds))

    def test_zero_initialization_is_exact_last_power_and_trainable(self):
        x = torch.randn(4, 72, 7)
        model = CausalResidualBase(dropout=0.)
        np.testing.assert_array_equal(model.residual(x).detach().numpy(), 0.)
        torch.testing.assert_close(model(x)[..., 0], x[:, -1:, -1].expand(-1, 2), rtol=0, atol=0)
        (model.residual(x) - .1).square().mean().backward()
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0.)

    def test_absolute_wind_is_not_window_normalized_away(self):
        model = CausalResidualBase(dropout=0.).eval()
        torch.nn.init.normal_(model.head.weight, std=.01)
        x = torch.randn(4, 72, 7)
        a = model.residual(x)
        x[:, :, 0] += 1.
        self.assertFalse(torch.allclose(a, model.residual(x)))
        self.assertEqual(tuple(model(x).shape), (4, 2, 1))

    def test_joint_selection_rejects_single_metric_gain_and_epoch_zero(self):
        base = dict(mae=100., rmse=150.)
        rows = [dict(model="one", best_epoch=3, parameters=1, metrics=dict(mae=99., rmse=151.)),
                dict(model="zero", best_epoch=0, parameters=1, metrics=dict(mae=98., rmse=149.))]
        self.assertEqual(numeric.select_candidate(rows, base)["model"], "Persistence")
        rows.append(dict(model="both", best_epoch=None, parameters=20, metrics=dict(mae=98., rmse=148.)))
        self.assertEqual(numeric.select_candidate(rows, base)["model"], "both")
        self.assertEqual(numeric.select_candidate([dict(model="nan", best_epoch=1, parameters=1,
            metrics=dict(mae=float("nan"), rmse=140.))], base)["model"], "Persistence")

    def test_neural_completed_reuse_and_fixed_refit_never_assesses_outer(self):
        from scripts import train_kelmarsh_numeric_foundation as runner
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            series = fixture(directory)
            data, _ = numeric.partitions(series)
            spec = dict(seed=2024, training=dict(width=8, dropout=0., bound_capacity=1., epochs=2,
                batch_size=32, residual_learning_rate=.001, weight_decay=.0001, grad_clip=1.))
            p = dict(rated_power_kw=2050.)
            resources = SimpleNamespace(check=lambda: None, after_step=lambda: None)
            path = Path(directory) / "inner"
            _, done = runner.neural_fit("ResidualTCN", data["fit"], data["selection"], p, spec, path,
                {"fixture": True}, None, torch.device("cpu"), resources)
            digest = runner.old.sha(path / "checkpoint.pt")
            _, reused = runner.neural_fit("ResidualTCN", data["fit"], data["selection"], p, spec, path,
                {"fixture": True}, None, torch.device("cpu"), resources)
            self.assertEqual(done, reused)
            self.assertEqual(digest, runner.old.sha(path / "checkpoint.pt"))
            with patch.object(runner, "predict", side_effect=AssertionError("Fixed refit must not assess outer")):
                _, fixed = runner.neural_fit("ResidualTCN", data["outer_refit"], None, p, spec, Path(directory) / "fixed",
                    {"fixture": True}, None, torch.device("cpu"), resources, fixed_epochs=1)
            self.assertEqual(fixed["epochs_completed"], 1)
            self.assertIsNone(fixed["metrics"])
            self.assertIsNone(fixed["contract"]["selection"])

    def test_protocol_disables_wiki_and_freezes_finite_budget(self):
        path = Path(__file__).resolve().parents[1] / "configs/kelmarsh_numeric_foundation_protocol_v11.json"
        spec = json.loads(path.read_text())
        self.assertFalse(spec["wiki_enabled"])
        self.assertEqual(spec["ridge_alphas"], [1., 10., 100.])
        self.assertEqual(spec["training"]["epochs"], 20)
        self.assertEqual(len(spec["candidates"]), 5)

    def test_prompt_transfer_recalibrates_fit_thresholds_after_loading(self):
        from scripts import train_kelmarsh_numeric_foundation as runner
        torch.set_num_threads(2)
        p = json.loads((Path(__file__).resolve().parents[1] / "configs/kelmarsh_hourly_protocol_v1.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            series = fixture(directory)
            data, _ = numeric.partitions(series)
            args = runner.old.model_args("PromptTimeDART", data["fit"], p, torch.device("cpu"), "pretrain")
            initial_model = runner.old.PromptTimeDART.Model(args)
            initial_model.regime_down_thresh.fill_(-123.)
            initial_model.regime_up_thresh.fill_(123.)
            path = Path(directory) / "pretrain.pt"
            torch.save({"model": initial_model.state_dict()}, path)
            model = runner.make_model("PromptExpanded", data["fit"], p, {"seed": 2024}, torch.device("cpu"), path)
            expected = runner.old.calibrate_regime_thresholds_from_dataset(data["fit"])
            self.assertAlmostEqual(float(model.regime_down_thresh), expected["down_thresh"], places=6)
            self.assertAlmostEqual(float(model.regime_up_thresh), expected["up_thresh"], places=6)
            self.assertFalse(model.utility_wiki)
            self.assertFalse(model.ramp_residual)
            x, _, _, _ = next(iter(runner.old.loader(data["fit"], 2024, batch=2)))
            output = runner.old.extract(model(x.float()))
            self.assertEqual(tuple(output.shape), (2, 2, 1))
            self.assertTrue(torch.isfinite(output).all())

    def test_ridge_completed_reuse_and_tamper_detection(self):
        from scripts import train_kelmarsh_numeric_foundation as runner
        with tempfile.TemporaryDirectory() as directory:
            data, _ = numeric.partitions(fixture(directory))
            path = Path(directory) / "ridge"
            p = dict(rated_power_kw=2050.)
            _, done = runner.ridge_fit("DeltaRidge_a10", data["fit"], data["selection"], 10., p, path, {"fixture": True})
            with patch.object(numeric, "fit_ridge", side_effect=AssertionError("Completed Ridge must not refit")):
                _, reused = runner.ridge_fit("DeltaRidge_a10", data["fit"], data["selection"], 10., p, path, {"fixture": True})
            self.assertEqual(done, reused)
            with (path / "model.npz").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "Completed Ridge stage changed"):
                runner.ridge_fit("DeltaRidge_a10", data["fit"], data["selection"], 10., p, path, {"fixture": True})


if __name__ == "__main__":
    unittest.main()
