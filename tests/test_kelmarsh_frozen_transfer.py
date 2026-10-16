from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.PowerAnchoredTimeDART import Model as UnfrozenModel
from models.FrozenTransferTimeDART import Model
from scripts import train_kelmarsh_frozen_transfer as task
from tests.test_kelmarsh_power_anchor import fixture


class FrozenTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_recipe_keeps_numerics_and_all_training_hyperparameters(self):
        recipe, effective, _ = task.protocol()
        _, reference = task.anchor.protocol()
        for key in reference:
            if key not in ("protocol_id", "executed_models"):
                self.assertEqual(reference[key], effective[key], key)
        self.assertEqual(recipe["folds"], [0, 1, 2])
        self.assertEqual(recipe["seeds"], [2024])
        self.assertFalse(recipe["labels_modified"])

    def test_only_gradient_mask_changes_not_rng_forward_or_parameter_budget(self):
        _, p, _ = task.protocol()
        args = task.shared.model_args("PromptTimeDART", None, p, torch.device("cpu"))
        torch.manual_seed(2024)
        reference = UnfrozenModel(args).eval()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(2024)
        model = Model(args).eval()
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(sum(x.numel() for x in model.parameters()), sum(x.numel() for x in reference.parameters()))
        for name, param in model.named_parameters():
            self.assertEqual(param.requires_grad, name.startswith(Model.TRAINABLE_PREFIXES), name)
        for m in (model, reference):
            m.regime_down_thresh.fill_(-.1)
            m.regime_up_thresh.fill_(.1)
        x = torch.randn(3, 24, 7)
        torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)
        audit = model.trainability_audit()
        self.assertGreater(audit["frozen_parameters"], 0)
        self.assertGreater(audit["trainable_parameters"], 0)
        self.assertEqual(audit["frozen_parameters"] + audit["trainable_parameters"], audit["total_parameters"])

    def test_real_training_preserves_loaded_representation_and_updates_head(self):
        _, p, _ = task.protocol()
        p["training"] = dict(p["training"], epochs=1, patience=1)
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
            torch.save({"model": UnfrozenModel(args).state_dict()}, initial)
            output = root / "runs/f0_s2024/Candidate"
            model, state = task.shared.train_stage("PromptTimeDART", data["fit"], data["selection"], p,
                output, 2024, torch.device("cpu"), initial=initial, model_factory=Model)
            audit = task.audit_frozen_state(model, initial)
            self.assertTrue(audit["frozen_tensors_bitwise_unchanged"])
            self.assertEqual(state["contract"]["model_factory"], "models.FrozenTransferTimeDART.Model")
            last = torch.load(output / "last.pt", map_location="cpu", weights_only=False)["model"]
            self.assertGreater(float(last["head.forecast_head.weight"].abs().sum()), 0)
            # A hash-matching source must still contain the actual executed representation.
            with torch.no_grad():
                model.enc_embedding.patch_embedding.weight.add_(.01)
            with self.assertRaisesRegex(ValueError, "Frozen representation changed"):
                task.audit_frozen_state(model, initial)

    def test_missing_frozen_pretrain_tensor_is_rejected(self):
        _, p, _ = task.protocol()
        model = Model(task.shared.model_args("PromptTimeDART", None, p, torch.device("cpu")))
        with tempfile.TemporaryDirectory() as tmp:
            initial = Path(tmp) / "bad.pt"
            torch.save({"model": {}}, initial)
            with self.assertRaisesRegex(ValueError, "missing from matched pretrain"):
                task.audit_frozen_state(model, initial)

    def test_gate_requires_joint_improvement_and_the_strong_internal_baseline(self):
        recipe, _, _ = task.protocol()
        def rows(values, dlinear=(11., 21.)):
            return [dict(candidate=dict(mae=a, rmse=b), reference=dict(mae=10., rmse=20.),
                         dlinear=dict(mae=dlinear[0], rmse=dlinear[1]), persistence=dict(mae=12., rmse=22.)) for a, b in values]
        self.assertFalse(task.decision(rows([(9., 19.)] * 3, dlinear=(8., 18.)), recipe["outer_gate"])["passed"])
        self.assertFalse(task.decision(rows([(0., 0.), (11., 21.), (11., 21.)]), recipe["outer_gate"])["passed"])
        self.assertFalse(task.decision(rows([(9., 21.)] * 3), recipe["outer_gate"])["passed"])
        result = task.decision(rows([(9., 19.), (9., 19.), (10.5, 20.5)]), recipe["outer_gate"])
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["rows"]), 3)


if __name__ == "__main__":
    unittest.main()
