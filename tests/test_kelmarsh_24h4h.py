import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from data_provider.kelmarsh_hourly import FEATURES, KelmarshSeries
from scripts import train_kelmarsh_24h4h as task


class Kelmarsh24h4hTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def protocol(self):
        return task.shared.read(task.PROTOCOL)

    def test_frozen_task_does_not_reuse_two_lead_experts_or_certificates(self):
        p = self.protocol()
        task.validate_protocol(p)
        self.assertEqual(p["models"], ["DLinear", "PatchTST", "PromptTimeDART"])
        for key, value in (("input_len", 72), ("pred_len", 2), ("wiki_enabled", True),
                           ("old_checkpoint_reuse", True), ("seeds", [2024, 2025])):
            bad = copy.deepcopy(p)
            bad[key] = value
            with self.assertRaises(ValueError):
                task.validate_protocol(bad)

    def test_real_neural_models_receive_24_hours_and_emit_four_leads(self):
        from models import DLinear, PatchTST, PromptTimeDART
        p, x = self.protocol(), torch.randn(2, 24, len(FEATURES))
        for name, module in (("DLinear", DLinear), ("PatchTST", PatchTST), ("PromptTimeDART", PromptTimeDART)):
            with self.subTest(model=name):
                args = task.shared.model_args(name, None, p, torch.device("cpu"))
                self.assertEqual((args.input_len, args.pred_len), (24, 4))
                self.assertFalse(args.utility_wiki)
                model = module.Model(args).eval()
                if name == "PromptTimeDART":
                    model.regime_down_thresh.fill_(-.1)
                    model.regime_up_thresh.fill_(.1)
                output = task.shared.extract(model(x))
                self.assertEqual(output.shape[:2], (2, 4))
                self.assertTrue(torch.isfinite(output).all())
        args = task.shared.model_args("PromptTimeDART", None, p, torch.device("cpu"), "pretrain")
        model = PromptTimeDART.Model(args).eval()
        model.regime_down_thresh.fill_(-.1)
        model.regime_up_thresh.fill_(.1)
        self.assertEqual(task.shared.extract(model(x)).shape, x.shape)

    def test_daily_persistence_and_target_boundaries_are_causal(self):
        dates = pd.date_range("2016-05-01", periods=1200, freq="h", tz="UTC")
        parts = []
        for turbine in range(1, 7):
            power = np.arange(len(dates), dtype=float) + turbine
            parts.append(pd.DataFrame({"timestamp_hour_end_utc": dates, "turbine_id": turbine,
                "wind_speed_mps": 6., "ambient_temp_c": 12., "nacelle_position_deg": 90.,
                "pitch_a_deg": 1., "pitch_b_deg": 1., "pitch_c_deg": 1.,
                "rotor_speed_rpm": 10., "power_kw": power}))
        with patch("data_provider.kelmarsh_hourly.pd.read_csv", return_value=pd.concat(parts)):
            series = KelmarshSeries("unused", fold=1, input_len=24, horizon=4, stride=3,
                start="2016-05-01T00:00:00Z", end="2016-06-20T00:00:00Z")
        data = {stage: series.subset(stage) for stage in ("fit", "selection", "validation")}
        audit = task.audit_dataset(series, data, self.protocol())
        self.assertEqual(audit["actual_forecast_hours"], 4)
        for stage, dataset in data.items():
            daily = task.daily_prediction(dataset)
            truth, _ = dataset.truth_and_anchor()
            np.testing.assert_array_equal(truth - daily, np.full_like(truth, 24.))
            first = dataset.target_times()[:, 0]
            if stage == "selection":
                self.assertTrue((first >= series.bounds["base_fit_end"]).all())
            if stage == "validation":
                self.assertTrue((first >= series.bounds["train_end"]).all())
        self.assertTrue((series.dates < series.bounds["sealed_test_start"]).all())

    def test_reports_have_four_independently_computed_lead_metrics(self):
        class Data:
            def truth_and_anchor(self):
                return np.arange(32.).reshape(8, 4), np.ones((8, 4))
            def target_times(self):
                return np.datetime64("2016-08-01") + np.arange(8)[:, None] * np.timedelta64(3, "h") + np.arange(4) * np.timedelta64(1, "h")
        dataset = Data()
        truth, anchor = dataset.truth_and_anchor()
        prediction = truth + np.arange(1, 5)[None, :]
        p = self.protocol()
        p["statistics"]["replicates"] = 100
        with tempfile.TemporaryDirectory() as tmp, patch.object(task.shared, "report", return_value={"model": "Test"}):
            result = task.write_report("Test", prediction, dataset, Path(tmp), p)
            self.assertEqual([x["lead_hours"] for x in result["per_hour"]], [1, 2, 3, 4])
            self.assertEqual([x["mae"] for x in result["per_hour"]], [1., 2., 3., 4.])
            for h, metrics in enumerate(result["per_hour"]):
                expected = task.shared.model_metrics(prediction[:, h:h+1], truth[:, h:h+1], anchor[:, h:h+1], 2050.)
                self.assertEqual(metrics["r2"], expected["r2"])

    def test_completed_suite_verifies_nested_prediction_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs/f1_s2024/Ridge"
            run.mkdir(parents=True)
            artifact = run / "prediction.txt"
            artifact.write_text("original")
            marker = run / "completed.json"
            task.shared.write(marker, {artifact.name: task.shared.sha(artifact)})
            task.shared.write(root / "completed.json", {str(marker.relative_to(root)): task.shared.sha(marker)})
            task.verify_completed_suite(root)
            artifact.write_text("changed")
            with self.assertRaises(ValueError):
                task.verify_completed_suite(root)


if __name__ == "__main__":
    unittest.main()
