import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from utils.utility_wiki import (
    EvidenceResidualAdapter, configure_frozen_utility_mode,
    evidence_training_weights, utility_candidate_specialization_loss,
    utility_decision_loss,
)


class HierarchicalUtilityTests(unittest.TestCase):
    def test_trend_experts_zero_init_bounds_and_gradients(self):
        adapter = EvidenceResidualAdapter(4, 3, 2, max_scale=0.5)
        hidden = torch.randn(3, 1, 5, 4)
        trend = torch.eye(3)
        activations = torch.ones(3, 2)
        out = adapter(hidden, hidden, trend, activations, activations)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        with torch.no_grad():
            adapter.forecast_head.bias.copy_(torch.tensor([-100.] * 3 + [0.] * 3 + [100.] * 3))
        out = adapter(hidden, hidden, trend, activations, activations)
        self.assertTrue(torch.all(out[0] == -0.5))
        self.assertTrue(torch.all(out[1] == 0))
        self.assertTrue(torch.all(out[2] == 0.5))
        with torch.no_grad():
            adapter.forecast_head.bias.zero_()
        adapter(hidden, hidden, trend, activations, activations).sum().backward()
        self.assertGreater(adapter.forecast_head.weight.grad.abs().sum().item(), 0)

    def test_frozen_base_has_no_dropout_noise_but_adapter_gradients_flow(self):
        model = nn.Module()
        model.backbone = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.9))
        model.utility_event_adapter = nn.Linear(4, 1)
        model.utility_composition_adapter = nn.Linear(4, 1)
        model.utility_gate = nn.Sequential(nn.Linear(4, 2), nn.Dropout(0.9))
        model.backbone.requires_grad_(False)
        model.utility_gate.requires_grad_(False)
        model.train()
        configure_frozen_utility_mode(model)
        x = torch.ones(8, 4)
        self.assertTrue(torch.equal(model.backbone(x), model.backbone(x)))
        self.assertFalse(model.utility_gate.training)
        self.assertTrue(model.utility_event_adapter.training)
        model.utility_event_adapter(model.backbone(x)).sum().backward()
        self.assertIsNotNone(model.utility_event_adapter.weight.grad)
        self.assertIsNone(model.backbone[0].weight.grad)

    def test_history_balancing_is_tempered_and_does_not_use_labels(self):
        aux = {
            "trend_probs": torch.tensor([[1., 0., 0.]] * 9 + [[0., 0., 1.]]),
            "activations": torch.tensor([[1., 0.]] * 9 + [[1., 1.]]),
        }
        weights = evidence_training_weights(aux)
        self.assertAlmostEqual(weights.mean().item(), 1., places=6)
        self.assertAlmostEqual((weights[-1] / weights[0]).item(), 3., places=5)
        aux["target"] = torch.randn(10, 12, 1) * 999
        self.assertTrue(torch.equal(weights, evidence_training_weights(aux)))

    def test_composition_loss_does_not_update_inherited_event(self):
        event = torch.tensor([[[0.2]]], requires_grad=True)
        delta = torch.tensor([[[0.1]]], requires_grad=True)
        base = torch.zeros(1, 1, 1, requires_grad=True)
        aux = {
            "adapter_mode": "hierarchical_evidence",
            "base_prediction": base,
            "event_prediction": base.detach() + event,
            "composition_prediction": base.detach() + event.detach() + delta,
            "availability": torch.tensor([[False, True]]),
            "trend_probs": torch.tensor([[1., 0., 0.]]),
            "activations": torch.ones(1, 2),
        }
        fit, rank, _ = utility_candidate_specialization_loss(aux, torch.ones(1, 1, 1))
        (fit + rank).backward()
        self.assertLess(delta.grad.item(), 0.)
        self.assertTrue(event.grad is None or event.grad.item() == 0.)
        self.assertIsNone(base.grad)

    def test_cost_sensitive_gate_penalizes_harmful_composition(self):
        scores = torch.zeros(4, 2, 2, requires_grad=True)
        aux = {
            "adapter_mode": "hierarchical_evidence", "utilities": scores,
            "base_prediction": torch.zeros(4, 2, 1),
            "event_prediction": torch.ones(4, 2, 1),
            "composition_prediction": torch.full((4, 2, 1), 10.),
            "availability": torch.ones(4, 2, dtype=torch.bool),
        }
        loss, _ = utility_decision_loss(aux, torch.ones(4, 2, 1))
        loss.backward()
        self.assertTrue(torch.all(scores.grad[..., 0] < 0))
        self.assertTrue(torch.all(scores.grad[..., 1] > 0))

    def test_full_model_hierarchical_forward_backward_and_reload(self):
        from run import build_parser, configure_args
        from models.TimeDART import PromptGuidedModel
        from utils.wind_regime_wiki import load_wind_regime_wiki_spec

        config = Path(__file__).resolve().parents[1] / "configs/wind_event_factor_wiki.json"
        spec = load_wind_regime_wiki_spec(config)
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "wiki.npz"
            np.savez_compressed(bundle,
                embeddings=np.random.default_rng(4).standard_normal((4, 16)).astype(np.float32),
                scene_ids=np.asarray(spec["scene_ids"]), encoder_name=np.asarray("test"),
                config_sha256=np.asarray(spec["sha256"]),
            )
            args = configure_args(build_parser().parse_args([
                "--task_name", "finetune", "--model_id", "SDWPF",
                "--model", "PromptTimeDART", "--data", "SDWPF",
                "--prompt_router", "compositional_wiki", "--utility_wiki",
                "--utility_adapter_mode", "hierarchical_evidence",
                "--utility_intervention_floor", "1",
                "--scene_wiki_config", str(config), "--scene_wiki_embeddings", str(bundle),
                "--input_len", "24", "--pred_len", "3", "--patch_len", "6", "--stride", "6",
                "--d_model", "16", "--d_ff", "32", "--n_heads", "4",
                "--e_layers", "1", "--d_layers", "1", "--no-use_gpu",
            ]))
            args.device = torch.device("cpu")
            model = PromptGuidedModel(args)
            model.set_scene_scaler(np.zeros(8), np.ones(8))
            for name, p in model.named_parameters():
                p.requires_grad_(name.startswith("utility_"))
            model.utility_gate.requires_grad_(False)
            configure_frozen_utility_mode(model)
            history = torch.zeros(3, 24, 8)
            history[..., 0] = torch.tensor([5., 12.] * 12)
            history[..., -1] = torch.linspace(700, 1100, 24)
            first = model(history)
            self.assertEqual(first.shape, (3, 3, 1))
            self.assertTrue(torch.equal(first, model._last_utility_aux["base_prediction"]))
            self.assertTrue(torch.equal(first, model(history)))
            base = model._last_utility_aux["base_prediction"].detach().clone()
            # Force physical support for this gradient test; production support
            # remains computed from the input-history router.
            aux = model._last_utility_aux
            aux["availability"] = torch.ones(3, 2, dtype=torch.bool)
            fit, ranking, _ = utility_candidate_specialization_loss(aux, base + 25)
            optimizer = torch.optim.Adam(model.utility_event_adapter.parameters(), lr=0.001)
            (fit + ranking).backward()
            optimizer.step()
            out = model(history)
            aux = model._last_utility_aux
            self.assertTrue(torch.equal(base, aux["base_prediction"]))
            self.assertTrue(torch.equal(aux["event_prediction"], aux["composition_prediction"]))
            self.assertFalse(torch.equal(aux["event_prediction"], base))
            model.eval()
            saved = Path(directory) / "checkpoint.pth"
            torch.save(model.state_dict(), saved)
            restored = PromptGuidedModel(args).eval()
            restored.load_state_dict(torch.load(saved, weights_only=True))
            self.assertTrue(torch.equal(model(history), restored(history)))


if __name__ == "__main__":
    unittest.main()
