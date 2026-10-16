from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from models.ChannelTimingTimeDART import EarlyModel, LateModel, PatchChannelAttention
from models.PowerAnchoredTimeDART import Model as AnchoredModel
from scripts import train_kelmarsh_channel_timing as task
from tests.test_kelmarsh_power_anchor import fixture
from utils import runtime_snapshot


class ChannelTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identity_initialization_and_trainable_correction(self):
        block = PatchChannelAttention(7, 8, 2, .1)
        x = torch.randn(14, 4, 8)
        for mode in (True, False):
            block.train(mode)
            torch.testing.assert_close(block(x), x, rtol=0, atol=0)
        block(x).square().mean().backward()
        self.assertGreater(float(block.attention.out_proj.weight.grad.abs().sum()), 0)

    def test_cross_channel_mixing_never_crosses_windows_or_patches(self):
        torch.manual_seed(12)
        block = PatchChannelAttention(7, 8, 2, 0).eval()
        with torch.no_grad():
            block.attention.out_proj.weight.copy_(torch.eye(8))
        x = torch.randn(14, 4, 8)
        changed = x.clone()
        changed[0, 1, 0] += 3
        before, after = block(x), block(changed)
        self.assertGreater(float((after[1:7, 1] - before[1:7, 1]).abs().sum()), 0)
        torch.testing.assert_close(after[7:], before[7:], rtol=0, atol=0)
        torch.testing.assert_close(after[:7, [0, 2, 3]], before[:7, [0, 2, 3]], rtol=0, atol=0)

    def test_matched_models_have_identical_initial_state_rng_and_real_timing(self):
        _, p, _ = task.protocol()
        args = task.shared.model_args("PromptTimeDART", None, p, torch.device("cpu"))
        models, rng = [], []
        for factory in (EarlyModel, LateModel):
            torch.manual_seed(2024)
            model = factory(args).eval()
            model.regime_down_thresh.fill_(-.1)
            model.regime_up_thresh.fill_(.1)
            models.append(model)
            rng.append(torch.get_rng_state().clone())
        self.assertTrue(torch.equal(*rng))
        self.assertEqual(sum(p.numel() for p in models[0].parameters()), sum(p.numel() for p in models[1].parameters()))
        for key, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[key], rtol=0, atol=0)
        x = torch.randn(2, 24, 7)
        for model, expected_order in zip(models, (["channel", "temporal"], ["temporal", "channel"])):
            calls = []
            model.channel_mixer.cross_attention.register_forward_hook(lambda *args: calls.append("channel"))
            model.encoder.layers[0].register_forward_hook(lambda *args: calls.append("temporal"))
            forecast = model(x)
            torch.testing.assert_close(forecast, x[:, -1:, -1:].repeat(1, 4, 1), rtol=0, atol=0)
            self.assertEqual(calls, expected_order)

    def test_two_real_training_paths_bind_factories_and_use_new_module_lr(self):
        _, p, _ = task.protocol()
        p["training"] = dict(p["training"], epochs=2, patience=2)
        s = fixture()
        s.values = np.random.default_rng(2024).normal(size=(200, 7)).astype(np.float32)
        s.raw = s.values.astype(np.float64)
        data = task.anchor.train_datasets_only(s)
        with tempfile.TemporaryDirectory() as tmp, patch.object(task.shared, "gpu_available_to_this_process"), \
                patch.object(task.shared, "calibrate_regime_thresholds_from_dataset", return_value={"down_thresh": -.1, "up_thresh": .1}), \
                patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.get_rng_state_all", return_value=[]):
            root = Path(tmp)
            task.shared.write(root / "protocol.json", p)
            args = task.shared.model_args("PromptTimeDART", None, p, torch.device("cpu"), "pretrain")
            initial = root / "pretrain.pt"
            torch.save({"model": AnchoredModel(args).state_dict()}, initial)
            for name, factory in task.FACTORIES.items():
                output = root / "runs/f0_s2024" / name
                _, state = task.shared.train_stage("PromptTimeDART", data["fit"], data["selection"], p,
                    output, 2024, torch.device("cpu"), initial=initial, model_factory=factory)
                self.assertEqual(state["contract"]["model_factory"], f"{factory.__module__}.{factory.__qualname__}")
                last = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
                self.assertGreater(float(last["model"]["channel_mixer.cross_attention.attention.out_proj.weight"].abs().sum()), 0)
                self.assertEqual(last["optimizer"]["param_groups"][1]["initial_lr"], p["training"]["new_module_lr"])

    def test_gate_rejects_capacity_only_gain_and_single_lucky_fold(self):
        recipe, _, _ = task.protocol()
        def rows(values, late=(10., 20.), dlinear=(12., 22.)):
            return [dict(early=dict(mae=a, rmse=b), late=dict(mae=late[0], rmse=late[1]),
                reference=dict(mae=11., rmse=21.), dlinear=dict(mae=dlinear[0], rmse=dlinear[1]),
                persistence=dict(mae=13., rmse=23.)) for a,b in values]
        self.assertFalse(task.decision(rows([(9.,19.)]*3, late=(8.,18.)), recipe["outer_gate"])["passed"])
        self.assertFalse(task.decision(rows([(9.,19.)]*3, dlinear=(8.,18.)), recipe["outer_gate"])["passed"])
        self.assertFalse(task.decision(rows([(0.,0.),(11.,21.),(11.,21.)]), recipe["outer_gate"])["passed"])
        result = task.decision(rows([(9.,19.),(9.,19.),(10.5,20.5)]), recipe["outer_gate"])
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["rows"]), 3)

    def test_history_partition_does_not_delete_or_route_using_future_labels(self):
        class Data:
            truth = np.arange(12.).reshape(3,4)
            def truth_and_anchor(self):
                return self.truth, np.repeat([[0.],[800.],[2000.]],4,axis=1)
        data = Data()
        first = task.history_power_partition(data, np.ones((3,4)), 2050.)
        data.truth = data.truth + 10000
        second = task.history_power_partition(data, np.ones((3,4)), 2050.)
        self.assertEqual({k:v["windows"] for k,v in first.items()}, {k:v["windows"] for k,v in second.items()})
        self.assertEqual(sum(v["windows"] for v in first.values()), 3)

    def test_launch_snapshot_is_cpu_only_immutable_and_excludes_secret_environment(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runtime_snapshot, "_command", return_value={"unavailable":"unit-test"}), \
                patch.object(runtime_snapshot.importlib.metadata, "distributions", return_value=[]), \
                patch.dict("os.environ", {"TEST_SECRET_TOKEN":"never-record-this"}), \
                patch("torch.cuda.current_device", side_effect=AssertionError("CUDA context forbidden")):
            first = runtime_snapshot.capture_at_launch(tmp, Path(tmp), command=["trial.py"])
            second = runtime_snapshot.capture_at_launch(tmp, Path(tmp), command=["changed.py"])
            self.assertEqual(first, second)
            self.assertNotIn("never-record-this", str(first))
            self.assertEqual(first["capture_phase"], "actual_process_start")

    def test_dependency_urls_exclude_credentials_and_query_tokens(self):
        self.assertEqual(runtime_snapshot._safe_url("https://user:password@example.org/a.whl?token=secret#frag"), "https://example.org/a.whl")
        self.assertEqual(runtime_snapshot._safe_url("https://conda.example.org/t/private-token/channel"), "https://conda.example.org/t/REDACTED/channel")

    def test_resume_rejects_changed_package_versions(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runtime_snapshot, "_command", return_value={"unavailable":"unit-test"}), \
                patch.object(runtime_snapshot.importlib.metadata, "distributions", return_value=[]):
            runtime_snapshot.capture_at_launch(tmp, tmp)
            changed = SimpleNamespace(metadata={"Name":"changed-package"}, version="2.0", read_text=lambda _:None)
            with patch.object(runtime_snapshot.importlib.metadata, "distributions", return_value=[changed]):
                with self.assertRaisesRegex(ValueError, "environment changed"):
                    runtime_snapshot.capture_at_launch(tmp, tmp)


if __name__ == "__main__":
    unittest.main()
