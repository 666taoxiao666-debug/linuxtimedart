from contextlib import contextmanager, ExitStack
import os
import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace

from utils.gpu_runtime_policy import GPURuntimePolicy


class GPUResourceTests(unittest.TestCase):
    @contextmanager
    def hardware(self, others=None, free=15.):
        with ExitStack() as stack:
            mocks = {}
            for name, target, value in (
                ("available", "torch.cuda.is_available", True),
                ("cap", "torch.cuda.set_per_process_memory_fraction", None),
                ("sync", "torch.cuda.synchronize", None),
                ("empty", "torch.cuda.empty_cache", None),
                ("reserved", "torch.cuda.memory_reserved", 0),
                ("sleep", "time.sleep", None),
                ("kill", "os.kill", None),
            ):
                mocks[name] = stack.enter_context(patch(target, return_value=value))
            mocks["memory"] = stack.enter_context(patch.object(GPURuntimePolicy, "memory", return_value=(free, 24.)))
            mocks["pids"] = stack.enter_context(patch.object(GPURuntimePolicy, "competitors", return_value=others or []))
            mocks["cpu"] = stack.enter_context(patch.object(GPURuntimePolicy, "_cpu_threads", return_value=True))
            stack.enter_context(patch.object(GPURuntimePolicy, "available_cpu_threads", return_value=8))
            yield mocks

    def test_alone_has_no_shared_allocator_limit_or_delay(self):
        with self.hardware() as mocks:
            policy = GPURuntimePolicy(shared=True)
            state = policy.configure()
            self.assertEqual(state["mode"], "solo_unrestricted")
            self.assertIsNone(state["pytorch_allocator_limit_gib"])
            self.assertEqual(state["cpu_threads"], 8)
            mocks["cap"].assert_called_once_with(1., 0)
            mocks["cpu"].assert_called_once_with(8)
            policy.after_step()
            mocks["sleep"].assert_not_called()
            mocks["kill"].assert_not_called()

    def test_shared_mode_caps_only_our_allocator_and_yields(self):
        with self.hardware(others=[99999999]) as mocks:
            policy = GPURuntimePolicy(shared=True)
            state = policy.configure()
            self.assertEqual(state["mode"], "shared_limited")
            self.assertTrue(state["cpu_nice_unchanged"])
            self.assertEqual(state["cpu_threads"], 2)
            mocks["cap"].assert_called_once_with(2/24, 0)
            policy.after_step()
            mocks["sync"].assert_called_once_with(0)
            mocks["sleep"].assert_called_once_with(.02)
            mocks["kill"].assert_not_called()

    def test_other_job_finishing_restores_full_allocator_and_removes_delay(self):
        changed = Mock()
        with self.hardware(others=[99999999]) as mocks:
            policy = GPURuntimePolicy(shared=True, on_change=changed)
            policy.configure()
            mocks["pids"].return_value = []
            policy.check(force=True)
            self.assertEqual(policy.mode, "solo_unrestricted")
            mocks["cap"].assert_called_with(1., 0)
            mocks["cpu"].assert_called_with(8)
            policy.after_step()
            mocks["sleep"].assert_not_called()
            self.assertEqual([call.args[0]["mode"] for call in changed.call_args_list],
                             ["shared_limited", "solo_unrestricted"])

    def test_new_competitor_enables_bounds_and_releases_only_our_cache(self):
        with self.hardware() as mocks:
            policy = GPURuntimePolicy(shared=True)
            policy.configure()
            mocks["pids"].return_value = [99999999]
            policy.check(force=True)
            self.assertEqual(policy.mode, "shared_limited")
            mocks["empty"].assert_called_once_with()
            mocks["cap"].assert_called_with(2/24, 0)
            mocks["cpu"].assert_called_with(2)
            policy.after_step()
            mocks["sleep"].assert_called_once_with(.02)
            mocks["kill"].assert_not_called()

    def test_default_still_refuses_unapproved_concurrent_execution(self):
        with self.hardware() as mocks:
            policy = GPURuntimePolicy()
            self.assertEqual(policy.configure()["policy"], "idle_only")
            mocks["pids"].return_value = [99999999]
            with self.assertRaisesRegex(RuntimeError, "idle-only"):
                policy.check(force=True)

    def test_low_initial_headroom_fails_before_allocating(self):
        with self.hardware(others=[99999999], free=5.) as mocks:
            with self.assertRaisesRegex(RuntimeError, "headroom"):
                GPURuntimePolicy(shared=True).configure()
            mocks["cap"].assert_not_called()

    def test_cannot_shrink_live_tensors_to_claim_safe_sharing(self):
        with self.hardware() as mocks:
            policy = GPURuntimePolicy(shared=True)
            policy.configure()
            mocks["pids"].return_value = [99999999]
            mocks["reserved"].return_value = 3 * 1024**3
            with self.assertRaisesRegex(RuntimeError, "only this experiment"):
                policy.check(force=True)
            self.assertEqual(policy.mode, "solo_unrestricted")
            mocks["cap"].assert_called_once_with(1., 0)
            mocks["kill"].assert_not_called()

    def test_runtime_pressure_stops_only_our_run(self):
        with self.hardware(others=[99999999]) as mocks:
            policy = GPURuntimePolicy(shared=True)
            policy.configure()
            mocks["memory"].return_value = (3., 24.)
            with self.assertRaisesRegex(RuntimeError, "only this experiment"):
                policy.check(force=True)
            mocks["kill"].assert_not_called()

    def test_gpu_query_is_read_only_and_converts_mib_to_gib(self):
        with patch("subprocess.run", return_value=SimpleNamespace(stdout="15360, 24576\n")) as query:
            self.assertEqual(GPURuntimePolicy.memory(), (15., 24.))
            self.assertEqual(query.call_args.args[0][0], "nvidia-smi")
            self.assertFalse(any(flag in query.call_args.args[0] for flag in ("-pl", "-lgc", "-c", "-r")))

    def test_competitor_detection_excludes_self_not_other_jobs(self):
        with patch("subprocess.run", return_value=SimpleNamespace(stdout=f"{os.getpid()}\n99999999\n")):
            self.assertEqual(GPURuntimePolicy.competitors(), [99999999])

    def test_unknown_process_query_never_means_free_gpu(self):
        with patch("subprocess.run", return_value=SimpleNamespace(stdout="N/A\n")):
            with self.assertRaisesRegex(RuntimeError, "do not assume idle"):
                GPURuntimePolicy.competitors()


if __name__ == "__main__":
    unittest.main()
