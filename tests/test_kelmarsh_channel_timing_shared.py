from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from scripts import run_kelmarsh_channel_timing_shared as task
from models.PowerAnchoredTimeDART import Model as AnchoredModel
from tests.test_kelmarsh_power_anchor import fixture


class SharedChannelTimingTests(unittest.TestCase):
    def test_adapter_preserves_batch_identity_order_and_restores_original_functions(self):
        batches = [object(), object()]
        resources = SimpleNamespace(mode="shared_limited", check=Mock(), after_step=Mock(), configure=Mock())
        original_loader = Mock(return_value=iter(batches))
        original_guard, original_wait = Mock(), Mock()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(task.trial.shared, "loader", original_loader), \
                patch.object(task.trial.shared, "gpu_available_to_this_process", original_guard), \
                patch.object(task.trial.prior, "wait_for_gpu", original_wait):
            with task.sharing_adapter(Path(tmp), resources):
                task.trial.prior.wait_for_gpu(Path(tmp), 0, 2024, "EarlyChannelTimeDART")
                task.trial.shared.gpu_available_to_this_process()
                got = list(task.trial.shared.loader("dataset", 2024, shuffle=True, batch=64))
                self.assertTrue(all(a is b for a,b in zip(got,batches)))
                original_loader.assert_called_once_with("dataset", 2024, shuffle=True, batch=64)
                self.assertEqual(resources.after_step.call_count, 2)
                self.assertEqual(task.trial.shared.read(Path(tmp) / "progress.json")["stage"], "EarlyChannelTimeDART")
                original_wait.assert_not_called()
                original_guard.assert_not_called()
            self.assertIs(task.trial.shared.loader, original_loader)
            self.assertIs(task.trial.prior.wait_for_gpu, original_wait)
            self.assertIs(task.trial.shared.gpu_available_to_this_process, original_guard)

    def test_first_stage_sets_resource_bounds_before_model_execution(self):
        resources = SimpleNamespace(mode=None, check=Mock(), after_step=Mock(), configure=Mock())
        with tempfile.TemporaryDirectory() as tmp, task.sharing_adapter(Path(tmp), resources):
            task.trial.prior.wait_for_gpu(Path(tmp), 1, 2024, "LateChannelTimeDART")
        resources.configure.assert_called_once_with()

    def test_adapter_restores_guards_on_allocation_failure_without_signaling_jobs(self):
        resources = SimpleNamespace(mode="shared_limited", check=Mock(side_effect=RuntimeError("headroom")), after_step=Mock())
        loader, guard, wait = task.trial.shared.loader, task.trial.shared.gpu_available_to_this_process, task.trial.prior.wait_for_gpu
        with tempfile.TemporaryDirectory() as tmp, patch("os.kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "headroom"), task.sharing_adapter(Path(tmp), resources):
                task.trial.prior.wait_for_gpu(Path(tmp), 0, 2024, "EarlyChannelTimeDART")
            kill.assert_not_called()
        self.assertIs(task.trial.shared.loader, loader)
        self.assertIs(task.trial.shared.gpu_available_to_this_process, guard)
        self.assertIs(task.trial.prior.wait_for_gpu, wait)

    def test_execution_record_preserves_original_snapshot_and_rejects_changed_policy_source(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(task.trial.shared, "sha", return_value="fixed-source"), \
                patch("subprocess.check_output", return_value="unit-commit\n"):
            root = Path(tmp)
            original = '{"original":"queue"}'
            (root / "runtime_snapshot.json").write_text(original)
            task.bind_execution(root)
            self.assertEqual((root / "runtime_snapshot.json").read_text(), original)
            contract = task.trial.shared.read(root / "shared_execution_contract.json")
            self.assertEqual(contract["shared_allocator_gib"], 2.)
            self.assertTrue(contract["model_data_loss_lr_seeds_unchanged"])
            contract["shared_allocator_gib"] = 8.
            task.trial.shared.write(root / "shared_execution_contract.json", contract)
            with self.assertRaisesRegex(ValueError, "execution identity changed"):
                task.bind_execution(root)

    def test_real_trainer_keeps_frozen_hyperparameters_and_calls_batch_controls(self):
        torch.set_num_threads(2)
        _, p, _ = task.trial.protocol()
        p["training"] = dict(p["training"], epochs=1, patience=1)
        s = fixture()
        s.values = np.random.default_rng(2024).normal(size=(200,7)).astype(np.float32)
        s.raw = s.values.astype(np.float64)
        data = task.trial.anchor.train_datasets_only(s)
        resources = SimpleNamespace(mode="shared_limited", check=Mock(), after_step=Mock())
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(task.trial.shared, "calibrate_regime_thresholds_from_dataset", return_value={"down_thresh":-.1,"up_thresh":.1}), \
                patch("torch.cuda.is_available", return_value=False), \
                patch("torch.cuda.get_rng_state_all", return_value=[]):
            root = Path(tmp)
            task.trial.shared.write(root / "protocol.json", p)
            args = task.trial.shared.model_args("PromptTimeDART", None, p, torch.device("cpu"), "pretrain")
            initial = root / "initial.pt"
            torch.save({"model":AnchoredModel(args).state_dict()}, initial)
            with task.sharing_adapter(root, resources):
                _, state = task.trial.shared.train_stage("PromptTimeDART", data["fit"], data["selection"], p,
                    root / "runs/f0_s2024/EarlyChannelTimeDART", 2024, torch.device("cpu"),
                    initial=initial, model_factory=task.trial.EarlyModel)
            self.assertEqual(state["epochs_completed"], 1)
            last = torch.load(root / "runs/f0_s2024/EarlyChannelTimeDART/last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(last["optimizer"]["param_groups"][1]["initial_lr"], p["training"]["new_module_lr"])
            self.assertGreater(resources.after_step.call_count, 0)
            self.assertGreater(resources.check.call_count, resources.after_step.call_count)


if __name__ == "__main__":
    unittest.main()
