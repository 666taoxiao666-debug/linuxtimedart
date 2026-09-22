import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from utils.utility_calibration import physical_history_features, split_utility_training, utility_action_probabilities
from utils.utility_wiki import utility_supervision_loss
from scripts.summarize_hierarchical_wiki import attach_training_progress
from scripts.summarize_sdwpf_cv import parse_summary


class CalibratedUtilityTests(unittest.TestCase):
    def test_absolute_gain_not_per_sample_relative_gain(self):
        aux = {
            "adapter_mode": "calibrated_evidence",
            "base_prediction": torch.tensor([[[1.]], [[100.]]]),
            "event_prediction": torch.tensor([[[0.]], [[99.]]]),
            "composition_prediction": torch.tensor([[[2.]], [[101.]]]),
            "utilities": torch.zeros(2, 1, 2, requires_grad=True),
            "availability": torch.ones(2, 2, dtype=torch.bool),
        }
        loss, targets = utility_supervision_loss(aux, torch.zeros(2, 1, 1))
        torch.testing.assert_close(targets, torch.tensor([[[1., -1.]], [[1., -1.]]]))
        loss.backward()
        self.assertTrue(torch.all(aux["utilities"].grad[..., 0] < 0))
        self.assertTrue(torch.all(aux["utilities"].grad[..., 1] > 0))

    def test_soft_forecast_gradient_and_hard_physical_abstention(self):
        scores = torch.zeros(2, 3, 2, requires_grad=True)
        available = torch.tensor([[True, False], [False, False]])
        probabilities = utility_action_probabilities(scores, available, .001, .05)
        prediction = probabilities[..., 1:2] * .2
        (prediction - 1).abs().mean().backward()
        self.assertGreater(scores.grad[0, :, 0].abs().sum().item(), 0)
        self.assertTrue(torch.all(scores.grad[1] == 0))
        self.assertTrue(torch.all(probabilities[1, :, 0] == 1))
        self.assertTrue(torch.all(probabilities[..., 2] == 0))

    def test_physical_features_use_observed_last_3_6_12_samples(self):
        x = torch.zeros(2, 24, 3)
        x[..., 0] = torch.arange(24.)
        x[..., 1] = 750.
        x[..., 2] = 45.
        result = physical_history_features(x, ["Wspd", "power", "Pab_mean"], 1500)
        self.assertEqual(result.shape, (2, 45))
        self.assertAlmostEqual(result[0, 0].item(), 23 / 25, places=6)
        self.assertAlmostEqual(result[0, 3].item(), 1 / 25, places=6)
        self.assertAlmostEqual(result[0, 15].item(), .5, places=6)
        changed = x.clone()
        changed[:, :12] = 99999
        torch.testing.assert_close(result, physical_history_features(changed, ["Wspd", "power", "Pab_mean"], 1500))

    def test_chronological_split_purges_crossing_targets_across_turbines(self):
        dates = np.arange(100).astype("timedelta64[m]") + np.datetime64("2020-01-01")
        dataset = SimpleNamespace(flag="train", dates=np.tile(dates, 2), seq_len=12, pred_len=5,
                                  window_starts=np.concatenate([np.arange(84), np.arange(84) + 100]))
        fit, gate, audit = split_utility_training(dataset, .2)
        fit_end = dataset.dates[fit.window_starts + 16]
        gate_start = dataset.dates[gate.window_starts + 12]
        self.assertLess(fit_end.max(), gate_start.min())
        self.assertGreater(audit["purged_windows"], 0)
        self.assertEqual(len(fit.window_starts) + len(gate.window_starts) + audit["purged_windows"], 168)
        self.assertEqual(len(dataset.window_starts), 168)
        self.assertFalse(audit["full_model_cross_fitted"])
        dataset.flag = "val"
        with self.assertRaises(ValueError):
            split_utility_training(dataset)

    def test_reports_keep_trained_degradation_and_exclude_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs/f0_s2024/finetune.log"
            path.parent.mkdir(parents=True)
            def epoch(n, mae, phase):
                return f"Epoch: {n}, Steps: 5 Phase: {phase} Val MAE(kW): {mae} Persist MAE(kW): 140 MAE Skill: 1% RMSE Skill: 2% Select(mae): {mae}\n"
            text = epoch(0, 133, "initial") + epoch(1, 100, "adapter_warmup") + epoch(2, 135, "utility_gate") + epoch(3, 136, "utility_gate")
            path.write_text(text, encoding="utf-8")
            rows = attach_training_progress([{"fold": 0, "seed": 2024}], directory)
            self.assertEqual(rows[0]["trained_best_mae_kw"], 135)
            self.assertEqual(rows[0]["last_mae_kw"], 136)
            summary = Path(directory) / "cv.log"
            summary.write_text("===== CV fold=0 seed=2024 =====\n" + text, encoding="utf-8")
            self.assertEqual(parse_summary(summary)[0].best_epoch, 0)


if __name__ == "__main__":
    unittest.main()
