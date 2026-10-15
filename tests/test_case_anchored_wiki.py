import copy
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from models.CaseAnchoredWindWiki import CaseAnchoredWindWiki, case_proposal
from models.IncrementalWindWiki import IncrementalWindWiki, inputs
from tests.test_kelmarsh_incremental_wiki import fixture


def tensors(pack):
    data = inputs(pack, 2050.)
    return [torch.from_numpy(data[key]) for key in ("context", "numeric_state", "memory", "supported")]


class CaseAnchoredWikiTests(unittest.TestCase):
    def test_exact_zero_start_has_nonzero_learning_gradient(self):
        model = CaseAnchoredWindWiki(dropout=0.)
        args = tensors(fixture())
        delta = model(*args)
        torch.testing.assert_close(delta, torch.zeros_like(delta), rtol=0, atol=0)
        (delta - .01).square().mean().backward()
        self.assertGreater(float(model.head.bias.grad.abs().sum()), 0.)

    def test_same_parameters_and_initialization_as_free_control(self):
        torch.manual_seed(2024)
        first = CaseAnchoredWindWiki()
        torch.manual_seed(2024)
        second = IncrementalWindWiki()
        self.assertEqual(sum(p.numel() for p in first.parameters()), sum(p.numel() for p in second.parameters()))
        for key, tensor in first.state_dict().items():
            torch.testing.assert_close(tensor, second.state_dict()[key], rtol=0, atol=0)

    def test_correction_cannot_reverse_or_exceed_case_proposal(self):
        pack = fixture()
        pack["reference"][:] = pack["numeric"]
        tokens = pack["memory"].reshape(20, 2, 5, 4)
        tokens[:, 0, 0, :] = .03
        tokens[:, 1, 0, :] = -.04
        model = CaseAnchoredWindWiki(dropout=0.).eval()
        torch.nn.init.constant_(model.head.bias, 2.)
        out = model.components(*tensors(pack))
        self.assertTrue(torch.all((out["confidence"] >= 0) & (out["confidence"] <= 1)))
        self.assertTrue(torch.all(out["delta"] * out["proposal"] >= 0))
        self.assertTrue(torch.all(out["delta"].abs() <= out["proposal"].abs()))
        self.assertTrue(torch.all(out["delta"][:, 0] > 0))
        self.assertTrue(torch.all(out["delta"][:, 1] < 0))

    def test_zero_case_and_no_support_stay_exact_zero_after_training_weights(self):
        pack = fixture()
        pack["reference"][:] = pack["numeric"]
        pack["memory"].reshape(20, 2, 5, 4)[:, 0, 0, :] = 0.
        pack["overlap"][:, 1] = False
        model = CaseAnchoredWindWiki().eval()
        torch.nn.init.constant_(model.head.bias, 10.)
        np.testing.assert_array_equal(model(*tensors(pack)).detach().numpy(), 0.)

    def test_query_target_mutation_cannot_change_prediction(self):
        pack = fixture()
        model = CaseAnchoredWindWiki().eval()
        torch.nn.init.constant_(model.head.bias, 1.)
        before = model(*tensors(pack)).detach().numpy()
        pack["truth"][:] = 1e9
        np.testing.assert_array_equal(before, model(*tensors(pack)).detach().numpy())

    def test_inconsistent_token_support_and_fake_zero_memory_ablation_rejected(self):
        pack = fixture()
        model = CaseAnchoredWindWiki()
        args = tensors(pack)
        args[2] = args[2].clone()
        args[2].reshape(20, 2, 5, 4)[:, :, 4, :] = 0.
        with self.assertRaisesRegex(ValueError, "support disagree"):
            model(*args)
        with self.assertRaisesRegex(ValueError, "explicit free zero-memory"):
            model(*tensors(pack), use_memory=False)

    def test_direct_control_is_the_same_bounded_proposal_and_target_free(self):
        from scripts.train_kelmarsh_case_anchored import direct_prediction
        pack = fixture()
        original = copy.deepcopy(pack)
        args = tensors(pack)
        expected = pack["numeric"] + 2050 * case_proposal(args[2], args[3], .2).numpy().astype(float)
        np.testing.assert_array_equal(expected, direct_prediction(pack, 2050., .2))
        pack["truth"][:] = -1e8
        np.testing.assert_array_equal(expected, direct_prediction(pack, 2050., .2))
        np.testing.assert_array_equal(original["memory"], pack["memory"])

    def test_actual_trace_matches_standard_execution_without_label_input(self):
        from scripts.train_kelmarsh_case_anchored import anchored_output
        from scripts.train_kelmarsh_incremental_wiki import infer
        pack = fixture()
        model = CaseAnchoredWindWiki().eval()
        torch.nn.init.constant_(model.head.bias, .5)
        out = anchored_output(model, pack, 2050., torch.device("cpu"))
        np.testing.assert_array_equal(out["prediction"], infer(model, pack, 2050., torch.device("cpu"), True))
        np.testing.assert_allclose(out["delta"], out["confidence"] * out["proposal"], atol=1e-8)
        pack["truth"][:] = -1e8
        np.testing.assert_array_equal(out["prediction"], anchored_output(model, pack, 2050., torch.device("cpu"))["prediction"])

    def test_actual_factory_training_and_completed_reuse(self):
        from scripts.train_kelmarsh_incremental_wiki import train_arm
        from scripts.train_kelmarsh_hourly import sha, read
        pack = fixture()
        spec = dict(seed=2024, training=dict(width=8, dropout=0., correction_bound_capacity=.2, epochs=2,
            batch_size=10, learning_rate=.001, weight_decay=.0001, mix_mse_weight=.2, grad_clip=1.))
        identity = dict(executor="case-anchor-fixture", base="frozen")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model, done = train_arm("WikiIncrement", pack, pack, 2050., spec, directory, identity,
                                    torch.device("cpu"), model_factory=CaseAnchoredWindWiki)
            self.assertIsInstance(model, CaseAnchoredWindWiki)
            self.assertEqual(done["epochs_completed"], 2)
            self.assertIn("CaseAnchoredWindWiki", read(directory / "manifest.json")["model_type"])
            hashed = sha(directory / "checkpoint.pt")
            _, reused = train_arm("WikiIncrement", pack, pack, 2050., spec, directory, identity,
                                    torch.device("cpu"), model_factory=CaseAnchoredWindWiki)
            self.assertEqual(done, reused)
            self.assertEqual(hashed, sha(directory / "checkpoint.pt"))


if __name__ == "__main__":
    unittest.main()
