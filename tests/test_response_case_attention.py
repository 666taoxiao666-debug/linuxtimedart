import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from models.ResponseCaseAttention import ResponseCaseAttention, tensor_inputs
from tests import test_kelmarsh_short_dynamics as source
from utils import response_case_view as views


def fixture():
    data, contract, _, bank, query, qi, _ = source.fixture()
    pack = dict(query, numeric=query["reference"].copy(), truth=data["truth"].copy(), turbines=data["turbines"].copy(),
                memory=np.zeros((len(data["truth"]), 2, 20), np.float32), change=np.zeros((len(data["truth"]), 2)),
                anchor_ratio=np.zeros((len(data["truth"]), 1)))
    return pack, views.build(pack, qi, bank, contract), qi, bank, contract


class ResponseCaseAttentionTests(unittest.TestCase):
    def test_cross_unit_exclusion_at_fit_and_inference(self):
        pack, view, _, bank, _ = fixture()
        present = view["case_indices"] >= 0
        self.assertTrue(present.any())
        self.assertFalse(np.any((view["case_units"] == pack["turbines"][:, None, None]) & present))
        for event in range(4):
            group = next(g for g in bank["groups"] if g["event"] == event and g["state"] == 5)
            if not group["active"]:
                continue
            idx = view["case_indices"][:, event]
            expected = np.asarray(group["values"])[idx].transpose(0, 2, 1)
            np.testing.assert_allclose(view["values"][:, :, event], expected, atol=1e-8)

    def test_query_truth_does_not_change_any_case_view_field(self):
        pack, before, info, bank, contract = fixture()
        pack["truth"][:] = 1e12
        after = views.build(pack, info, bank, contract)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])

    def test_zero_fallback_does_not_kill_proposal_learning(self):
        _, view, _, _, _ = fixture()
        view["values"][:] = np.linspace(-.04, .06, view["values"].shape[-1])
        model = ResponseCaseAttention(dropout=0.)
        out = model.components(*tensor_inputs(view))
        np.testing.assert_array_equal(out["delta"].detach().numpy(), 0.)
        (out["proposal"] - .02).square().mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None), 0.)

    def test_weights_only_read_actual_values_and_remain_bounded(self):
        _, view, _, _, _ = fixture()
        model = ResponseCaseAttention(dropout=0.).eval()
        model.set_alpha([.7, 1.])
        out = model.components(*tensor_inputs(view))
        np.testing.assert_allclose(out["weights"].detach().sum(-1).numpy(), 1., atol=1e-6)
        self.assertTrue(torch.all(out["weights"] >= 0))
        expected = (out["weights"] * torch.from_numpy(view["values"])).sum(-1)
        group = (expected - torch.from_numpy(view["base_change"])[:, :, None]).clamp(-.2, .2)
        torch.testing.assert_close(group, out["group_proposal"])
        self.assertTrue(torch.all(out["delta"].abs() <= .2))

    def test_no_value_head_means_zero_cases_cannot_invent_change(self):
        _, view, _, _, _ = fixture()
        view["values"][:] = 0.
        view["base_change"][:] = 0.
        model = ResponseCaseAttention().eval()
        model.set_alpha([1., 1.])
        np.testing.assert_array_equal(model(*tensor_inputs(view)).detach().numpy(), 0.)

    def test_unsupported_and_unavailable_memory_stay_zero(self):
        pack, _, info, bank, contract = fixture()
        v = views.build(pack, info, bank, contract, ready=False)
        self.assertFalse(v["supported"].any())
        self.assertTrue((v["case_indices"] == -1).all())
        model = ResponseCaseAttention().eval()
        model.set_alpha([1., 1.])
        np.testing.assert_array_equal(model(*tensor_inputs(v)).detach().numpy(), 0.)
        pack["times"] -= np.timedelta64(30, "D")
        info["last_observed_ns"] -= 30 * 24 * 3600 * 10**9
        with self.assertRaisesRegex(ValueError, "not yet available"):
            views.build(pack, info, bank, contract)

    def test_future_metadata_or_changed_source_expert_rejected(self):
        pack, _, info, bank, contract = fixture()
        info["last_observed_ns"] += 2 * 3600 * 10**9
        with self.assertRaisesRegex(ValueError, "Future query"):
            views.build(pack, info, bank, contract)
        info["last_observed_ns"] -= 2 * 3600 * 10**9
        bank["source_expert_id"] = "tampered"
        with self.assertRaises(ValueError):
            views.build(pack, info, bank, contract)

    def test_new_support_view_preserves_numerical_and_target_arrays(self):
        pack, view, _, _, _ = fixture()
        before = copy.deepcopy(pack)
        cpack = views.certificate_pack(pack, view)
        for key in ("truth", "history", "numeric", "reference", "anchor", "times", "turbines"):
            np.testing.assert_array_equal(cpack[key], before[key])
            np.testing.assert_array_equal(pack[key], before[key])
        np.testing.assert_array_equal(cpack["overlap"], view["supported"])

    def test_fixed_kernel_has_no_trainable_neural_parameters(self):
        _, view, _, _, _ = fixture()
        model = ResponseCaseAttention(fixed=True)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        model.set_alpha([.5, .5])
        out = model.components(*tensor_inputs(view))
        self.assertTrue(torch.isfinite(out["delta"]).all())
        for invalid in ([1.1, 0.], [0.], [float("nan"), 0.]):
            with self.assertRaises(ValueError):
                model.set_alpha(invalid)

    def test_actual_fitting_analytic_confidence_and_completed_reuse(self):
        from scripts import train_kelmarsh_response_attention as runner
        pack, view, _, _, _ = fixture()
        spec = dict(seed=2024, training=dict(width=8, dropout=0., correction_bound_capacity=.2, epochs=2,
            batch_size=32, learning_rate=.001, weight_decay=.0001, mix_mse_weight=.2, grad_clip=1.),
            attention=dict(features=23, embedding=4))
        resources = SimpleNamespace(check=lambda: None, after_step=lambda: None)
        cutoff = str(pack["times"].max() + np.timedelta64(1, "D"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            model, done = runner.train(spec, path, pack, view, pack, view, {"fixture": True}, 2050.,
                torch.device("cpu"), resources, cutoff)
            self.assertEqual(done["epochs_completed"], 2)
            original = runner.old.sha(path / "checkpoint.pt")
            _, reused = runner.train(spec, path, pack, view, pack, view, {"fixture": True}, 2050.,
                torch.device("cpu"), resources, cutoff)
            self.assertEqual(done, reused)
            self.assertEqual(original, runner.old.sha(path / "checkpoint.pt"))
            self.assertTrue(torch.all((model.alpha >= 0) & (model.alpha <= 1)))
        with tempfile.TemporaryDirectory() as directory:
            _, done = runner.train(spec, Path(directory), pack, view, pack, view, {"fixed": True}, 2050.,
                torch.device("cpu"), resources, cutoff, fixed=True)
            self.assertEqual(done["epochs_completed"], 0)


if __name__ == "__main__":
    unittest.main()
