from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from layers.TimeDART_EncDec import ChannelMixer
from models.PowerAnchoredTimeDART import Model, PowerAnchoredMixer
from models import PromptTimeDART
from scripts import train_kelmarsh_power_anchor as task


def fixture():
    dates = pd.date_range("2016-05-01", periods=200, freq="h").to_numpy()
    return SimpleNamespace(dates=dates, starts=np.arange(0, 173, 3), raw=np.ones((200, 7)),
        values=np.ones((200, 7), dtype=np.float32), turbines=np.ones(200, dtype=np.int16),
        input_len=24, horizon=4, scaler=SimpleNamespace(mean_=np.zeros(7), scale_=np.ones(7)),
        bounds={"base_fit_end": dates[100], "base_selection_end": dates[120],
                "calibration_early_end": dates[150], "train_end": dates[170]},
        subset=lambda _: (_ for _ in ()).throw(AssertionError("Outer subset accessed")))


class PowerAnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_mixer_is_exact_power_identity_at_initialization(self):
        original = ChannelMixer(7, 8, .1, context_dim=4)
        count = sum(p.numel() for p in original.parameters())
        rng = torch.get_rng_state().clone()
        mixer = PowerAnchoredMixer(original)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual(count, sum(p.numel() for p in mixer.parameters()))
        x, context = torch.randn(3, 7, 4, 8), torch.randn(3, 4)
        for train in (True, False):
            mixer.train(train)
            torch.testing.assert_close(mixer(x, context), x[:, -1:], rtol=0, atol=0)

    def test_zero_correction_can_learn(self):
        mixer = PowerAnchoredMixer(ChannelMixer(7, 8, .0, context_dim=4))
        x, context = torch.randn(3, 7, 4, 8), torch.randn(3, 4)
        mixer(x, context).square().mean().backward()
        self.assertGreater(mixer.correction.proj[-1].weight.grad.abs().sum(), 0)
        self.assertGreater(mixer.correction.context_proj.weight.grad.abs().sum(), 0)

    def test_real_models_have_same_parameter_count_rng_and_initial_forecast(self):
        _, p = task.protocol()
        args = task.shared.model_args("PromptTimeDART", None, p, torch.device("cpu"))
        torch.manual_seed(2024)
        old = PromptTimeDART.Model(args).eval()
        old_rng = torch.get_rng_state().clone()
        torch.manual_seed(2024)
        new = Model(args).eval()
        torch.testing.assert_close(torch.get_rng_state(), old_rng, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in old.parameters()), sum(p.numel() for p in new.parameters()))
        for model in (old, new):
            model.regime_down_thresh.fill_(-.1)
            model.regime_up_thresh.fill_(.1)
        x = torch.randn(3, 24, 7)
        expected = x[:, -1:, -1:].repeat(1, 4, 1)
        torch.testing.assert_close(new(x), expected, rtol=0, atol=0)
        torch.testing.assert_close(old(x), expected, rtol=0, atol=0)

    def test_inner_dataset_builder_never_requests_outer_subset(self):
        s = fixture()
        data = task.train_datasets_only(s)
        self.assertEqual(set(data), {"fit", "selection"})
        self.assertLess(data["fit"].target_times().max(), data["selection"].target_times().min())
        self.assertLess(data["selection"].target_times().max(), s.bounds["train_end"])

    def test_gate_rejects_one_lucky_fold_and_rmse_harm(self):
        recipe, _ = task.protocol()
        def rows(candidates):
            return [{"candidate": {"mae": a, "rmse": b}, "reference": {"mae": 10., "rmse": 20.},
                     "persistence": {"mae": 12., "rmse": 22.}} for a, b in candidates]
        settings = recipe["outer_gate"]
        self.assertFalse(task.decision(rows([(0, 0), (11, 21), (11, 21)]), settings)["passed"])
        self.assertFalse(task.decision(rows([(9, 21)] * 3), settings)["passed"])
        accepted = task.decision(rows([(9, 19), (9, 19), (10.5, 20.5)]), settings)
        self.assertTrue(accepted["passed"])
        self.assertEqual(len(accepted["rows"]), 3)  # never drop an internally harmed fold

    def test_stage_factory_is_bound_to_checkpoint_contract(self):
        _, p = task.protocol()
        p["training"] = dict(p["training"], epochs=1, patience=1)
        data = task.train_datasets_only(fixture())
        with tempfile.TemporaryDirectory() as tmp, patch.object(task.shared, "gpu_available_to_this_process"), \
                patch.object(task.shared, "calibrate_regime_thresholds_from_dataset", return_value={"down_thresh": -.1, "up_thresh": .1}), \
                patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.get_rng_state_all", return_value=[]):
            root = Path(tmp)
            task.shared.write(root / "protocol.json", p)
            output = root / "runs/f0_s2024/Candidate"
            _, state = task.shared.train_stage("PromptTimeDART", data["fit"], data["selection"], p,
                output, 2024, torch.device("cpu"), model_factory=Model)
            self.assertEqual(state["contract"]["model_factory"], "models.PowerAnchoredTimeDART.Model")
            self.assertEqual(task.shared.read(output / "manifest.json")["contract"], state["contract"])
            legacy = root / "runs/f0_s2024/Legacy"
            _, old_state = task.shared.train_stage("PromptTimeDART", data["fit"], data["selection"], p,
                legacy, 2024, torch.device("cpu"))
            self.assertNotIn("model_factory", old_state["contract"])
            self.assertEqual(task.shared.read(legacy / "manifest.json")["parameters"],
                             task.shared.read(output / "manifest.json")["parameters"])


if __name__ == "__main__":
    unittest.main()
