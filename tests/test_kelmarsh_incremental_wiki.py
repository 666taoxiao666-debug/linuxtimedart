import copy
import hashlib
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from models.IncrementalWindWiki import IncrementalWindWiki, CONTEXT_SCALES, inputs
from utils.wiki_training_replay import verify_replay, replay_diagnostic


def fixture(n=20):
    numeric = np.full((n, 2), 100., dtype=float)
    issue = np.datetime64("2016-09-01T00", "h") + np.arange(n) * np.timedelta64(3, "h")
    return dict(numeric=numeric, reference=numeric - 10, anchor=numeric - 20,
                truth=numeric + 4, times=issue[:, None] + np.array([1, 2], dtype="timedelta64[h]")[None],
                turbines=np.tile(np.arange(1, 6), (n + 4) // 5)[:n],
                memory=np.ones((n, 2, 20), np.float32), overlap=np.ones((n, 2, 4), bool),
                context={key: np.ones(n) for key, _ in CONTEXT_SCALES})


def call(model, pack, memory=True):
    data = inputs(pack, 2050.)
    return model(*(torch.from_numpy(data[key]) for key in ("context", "numeric_state", "memory", "supported")), use_memory=memory)


class IncrementalWikiTests(unittest.TestCase):
    def test_zero_initialization_is_exact_frozen_numeric_forecast(self):
        pack = fixture()
        model = IncrementalWindWiki().eval()
        np.testing.assert_array_equal(call(model, pack).detach().numpy(), 0.)
        prediction = pack["numeric"] + 2050 * call(model, pack).detach().numpy()
        np.testing.assert_array_equal(prediction, pack["numeric"])

    def test_unsupported_leads_are_exact_zero_after_nonzero_weights(self):
        pack = fixture()
        pack["overlap"][:, 1] = False
        model = IncrementalWindWiki().eval()
        torch.nn.init.constant_(model.head.bias, .1)
        delta = call(model, pack).detach().numpy()
        self.assertTrue(np.all(delta[:, 0] > 0))
        np.testing.assert_array_equal(delta[:, 1], 0.)
        self.assertLessEqual(np.abs(delta).max(), .2)

    def test_outcome_mutation_does_not_change_inputs_or_predictions(self):
        pack = fixture()
        model = IncrementalWindWiki().eval()
        torch.nn.init.normal_(model.head.weight)
        before = copy.deepcopy(pack)
        original = call(model, pack).detach().numpy()
        pack["truth"][:] = -1e10
        np.testing.assert_array_equal(original, call(model, pack).detach().numpy())
        for key in ("numeric", "reference", "anchor", "memory", "overlap"):
            np.testing.assert_array_equal(pack[key], before[key])

    def test_memory_rebased_to_actual_numeric_and_masked_per_event(self):
        pack = fixture()
        pack["overlap"][:, :, 2] = False
        result = inputs(pack, 2050.)["memory"].reshape(20, 2, 5, 4)
        np.testing.assert_allclose(result[:, :, 0, 0], 1 - 10 / 2050, rtol=1e-6)
        np.testing.assert_array_equal(result[:, :, :, 2], 0.)

    def test_zero_memory_ablation_has_matched_initialization_and_capacity(self):
        torch.manual_seed(2024)
        first = IncrementalWindWiki()
        torch.manual_seed(2024)
        second = IncrementalWindWiki()
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in first.parameters()), sum(p.numel() for p in second.parameters()))

    def test_training_updates_increment_only_and_completed_arm_is_reused(self):
        from scripts.train_kelmarsh_incremental_wiki import train_arm
        from scripts.train_kelmarsh_hourly import sha
        pack = fixture()
        numeric = pack["numeric"].copy()
        config = dict(seed=2024, training=dict(width=8, dropout=0., correction_bound_capacity=.2,
            epochs=2, learning_rate=.001, weight_decay=.0001, mix_mse_weight=.2, batch_size=10, grad_clip=1.))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            model, done = train_arm("WikiIncrement", pack, pack, 2050., config, path, {"fixture": True}, torch.device("cpu"))
            self.assertEqual(done["epochs_completed"], 2)
            checkpoint = sha(path / "checkpoint.pt")
            _, replay = train_arm("WikiIncrement", pack, pack, 2050., config, path, {"fixture": True}, torch.device("cpu"))
            self.assertEqual(replay, done)
            self.assertEqual(sha(path / "checkpoint.pt"), checkpoint)
        np.testing.assert_array_equal(pack["numeric"], numeric)


class FrozenTrainingReplayTests(unittest.TestCase):
    def test_busy_gpu_is_queued_without_starting_training(self):
        from scripts.train_kelmarsh_incremental_wiki import wait_for_gpu
        from scripts.train_kelmarsh_hourly import read
        with tempfile.TemporaryDirectory() as directory:
            with patch("torch.cuda.is_available", return_value=True), patch(
                    "scripts.train_kelmarsh_incremental_wiki.subprocess.run",
                    return_value=SimpleNamespace(stdout="99999999\n")):
                with self.assertRaises(TimeoutError):
                    wait_for_gpu(Path(directory), 0.)
            state = read(Path(directory) / "status.json")
            self.assertEqual(state["status"], "WAITING_FOR_GPU")
            self.assertEqual(state["other_gpu_pids"], [99999999])

    def test_strict_hash_mismatch_stops_without_replacing_evidence(self):
        pack = fixture()
        numeric, candidate = pack["numeric"], pack["numeric"] + 1.
        digest = hashlib.sha256(np.ascontiguousarray(np.stack((numeric, candidate), -1)).tobytes()).hexdigest()
        certificate = dict(calibration_sha256="fixture", actual_outputs_sha256=digest)
        self.assertEqual(verify_replay(pack, numeric, candidate, certificate, lambda _: "fixture"), digest)
        with self.assertRaisesRegex(ValueError, "bit-identical"):
            verify_replay(pack, numeric, candidate + 1e-8, certificate, lambda _: "fixture")
        with self.assertRaisesRegex(ValueError, "input hash"):
            verify_replay(pack, numeric, candidate, certificate, lambda _: "other")
        self.assertEqual(certificate["actual_outputs_sha256"], digest)

    def test_block_gain_and_oracle_are_training_only_retrospective(self):
        pack = fixture()
        cert = dict(entries={"event:0:0:0": dict(supported=True, accepted=True)})
        report = replay_diagnostic(pack, pack["numeric"], pack["numeric"] + 1., cert, train_end="2016-10-01")
        self.assertAlmostEqual(report["oracle_gain_all_points_kw"], 1.)
        self.assertGreater(len(report["cells"]["event:0:0:0"]["blocks"]), 1)
        with self.assertRaisesRegex(ValueError, "outer validation"):
            replay_diagnostic(pack, pack["numeric"], pack["numeric"], cert, train_end="2016-09-01")


if __name__ == "__main__":
    unittest.main()
