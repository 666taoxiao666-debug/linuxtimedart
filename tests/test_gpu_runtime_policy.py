import unittest
from unittest.mock import patch
from types import SimpleNamespace

from utils.gpu_runtime_policy import GPURuntimePolicy


class GPUResourceTests(unittest.TestCase):
    def test_default_remains_exclusive_and_does_not_set_cap_or_priority(self):
        policy = GPURuntimePolicy()
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.set_per_process_memory_fraction") as cap:
            self.assertEqual(policy.configure()["mode"], "idle_only")
            cap.assert_not_called()
        with patch("scripts.train_kelmarsh_hourly.gpu_available_to_this_process") as check:
            policy.check()
            check.assert_called_once()

    def test_shared_mode_caps_only_our_allocator_and_yields(self):
        policy = GPURuntimePolicy(shared=True)
        with patch.object(policy, "memory", return_value=(15., 24.)), patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.set_per_process_memory_fraction") as cap, patch("utils.gpu_runtime_policy.os.nice", create=True), \
             patch("torch.cuda.synchronize") as sync, patch("time.sleep") as sleep:
            result = policy.configure()
            self.assertEqual(result["mode"], "shared_explicit_user_authorization")
            cap.assert_called_once_with(2/24, 0)
            policy.after_step()
            sync.assert_called_once_with(0)
            sleep.assert_called_once_with(.02)

    def test_low_initial_headroom_fails_before_allocating(self):
        policy = GPURuntimePolicy(shared=True)
        with patch.object(policy, "memory", return_value=(5., 24.)), patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.set_per_process_memory_fraction") as cap:
            with self.assertRaisesRegex(RuntimeError, "headroom"):
                policy.configure()
            cap.assert_not_called()

    def test_runtime_pressure_stops_our_run_without_any_process_signal(self):
        policy = GPURuntimePolicy(shared=True)
        with patch.object(policy, "memory", return_value=(3., 24.)), patch("os.kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "only this experiment"):
                policy.check()
            kill.assert_not_called()

    def test_gpu_query_is_read_only_and_converts_mib_to_gib(self):
        with patch("subprocess.run", return_value=SimpleNamespace(stdout="15360, 24576\n")) as query:
            self.assertEqual(GPURuntimePolicy.memory(), (15., 24.))
            self.assertEqual(query.call_args.args[0][0], "nvidia-smi")
            self.assertFalse(any(flag in query.call_args.args[0] for flag in ("-pl", "-lgc", "-c", "-r")))


if __name__ == "__main__":
    unittest.main()
