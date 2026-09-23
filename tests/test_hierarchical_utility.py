import tempfile
import copy
import os
from types import SimpleNamespace
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from utils.utility_wiki import (
    EvidenceResidualAdapter, configure_frozen_utility_mode,
    evidence_training_weights, utility_candidate_specialization_loss,
    utility_decision_loss,
)


class _CalibrationWindowDataset(Dataset):
    def __init__(self, flag):
        self.flag, self.seq_len, self.pred_len = flag, 24, 3
        self.window_starts = np.arange(0, 40, 2)
        offset = 0 if flag == "train" else 1000
        self.dates = np.datetime64("2020-01-01") + (np.arange(70) + offset).astype("timedelta64[m]")
        self.data_x = np.zeros((70, 8), dtype=np.float32)
        self.data_x[:, 0] = np.tile([5., 12.], 35)
        self.data_x[:, -1] = 700 + 200 * np.sin(np.arange(70) / 5)

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, index):
        start = self.window_starts[index]
        return (self.data_x[start:start + 24], self.data_x[start + 24:start + 27],
                np.zeros((24, 1), dtype=np.float32), np.zeros((3, 1), dtype=np.float32))


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
        self._exercise_full_model("hierarchical_evidence")

    def test_full_model_calibrated_forward_backward_and_reload(self):
        self._exercise_full_model("calibrated_evidence")

    def test_full_model_factorized_forward_backward_and_reload(self):
        self._exercise_full_model("calibrated_evidence", factorized=True)

    def _exercise_full_model(self, adapter_mode, factorized=False):
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
                "--utility_adapter_mode", adapter_mode,
                "--is_training", "0", "--freq", "10min",
                "--utility_intervention_floor", "1",
                # The synthetic semantic keys are random, not a pretrained
                # retriever. Enable their soft support while retaining the
                # physical-rule mask, making this gradient fixture reliable.
                "--scene_wiki_activation_threshold", "0",
                "--scene_wiki_config", str(config), "--scene_wiki_embeddings", str(bundle),
                "--input_len", "24", "--pred_len", "3", "--patch_len", "6", "--stride", "6",
                "--d_model", "16", "--d_ff", "32", "--n_heads", "4",
                "--e_layers", "1", "--d_layers", "1", "--no-use_gpu",
                *(["--utility_factorized"] if factorized else []),
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
            if factorized:
                # Verify the real overlay path from a plain trend checkpoint,
                # not merely consistency within the new model itself.
                from utils.tools import overlay_forecast_weights
                trend_args = copy.copy(args)
                trend_args.utility_wiki = trend_args.utility_factorized = False
                trend_args.prompt_router = 'trend'
                trend_model = PromptGuidedModel(trend_args).eval()
                trend_model.load_state_dict({k: v for k, v in model.state_dict().items()
                                             if not k.startswith(('utility_', 'scene_wiki_router.'))}, strict=False)
                trend_checkpoint = Path(directory) / 'trend.pth'
                torch.save(trend_model.state_dict(), trend_checkpoint)
                overlay_forecast_weights(trend_checkpoint, model)
                with torch.no_grad():
                    torch.testing.assert_close(model(history), trend_model(history))
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
            if not factorized:
                self.assertTrue(torch.equal(aux["event_prediction"], aux["composition_prediction"]))
            else:
                self.assertEqual(aux['factor_predictions'].shape, (3, 3, 1, 5))
                self.assertFalse(model.scene_wiki_router.semantic_keys.requires_grad)
                optimizer.zero_grad()
                semantic_fit, _, _ = utility_candidate_specialization_loss(aux, base + 25)
                semantic_fit.backward()
                self.assertGreater(model.utility_event_adapter.semantic_projection.weight.grad.abs().sum().item(), 0)
                self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
            self.assertFalse(torch.equal(aux["event_prediction"], base))
            if adapter_mode == "calibrated_evidence":
                model.utility_event_adapter.requires_grad_(False)
                model.utility_composition_adapter.requires_grad_(False)
                model.utility_gate.requires_grad_(True)
                model.zero_grad(set_to_none=True)
                configure_frozen_utility_mode(model)
                prediction = model(history)
                self.assertTrue(torch.equal(prediction, model._last_utility_aux["soft_prediction"]))
                (prediction - (base + 25)).abs().mean().backward()
                gate_grad = sum(p.grad.abs().sum().item() for p in model.utility_gate.parameters() if p.grad is not None)
                self.assertGreater(gate_grad, 0.)
                self.assertTrue(all(p.grad is None for p in model.utility_event_adapter.parameters()))
            model.eval()
            saved = Path(directory) / "checkpoint.pth"
            torch.save(model.state_dict(), saved)
            restored = PromptGuidedModel(args).eval()
            restored.load_state_dict(torch.load(saved, weights_only=True))
            self.assertTrue(torch.equal(model(history), restored(history)))
            if adapter_mode == "calibrated_evidence":
                # Exercise the real epoch-zero, two-stage train/valid loop,
                # unequal loader lengths, scheduler, and all checkpoint saves.
                from exp.exp_timedart import Exp_TimeDART
                args.checkpoints = str(Path(directory) / "checkpoints")
                args.freeze_non_utility = True
                args.utility_adapter_warmup_epochs = 1
                args.train_epochs, args.patience, args.batch_size = 3, 3, 8
                args.lradj, args.pct_start = "step", .2
                args.validate_before_training = True
                experiment = Exp_TimeDART.__new__(Exp_TimeDART)
                experiment.args, experiment.model, experiment.device = args, restored, args.device
                experiment.writer = Mock()
                experiment.amp_enabled = False
                experiment.grad_scaler = torch.amp.GradScaler("cuda", enabled=False)
                datasets = {flag: _CalibrationWindowDataset(flag) for flag in ("train", "val")}
                experiment._get_data = lambda flag: (datasets[flag], DataLoader(datasets[flag], batch_size=8))
                # Scaler already installed above. Scene-support fitting itself
                # has separate tests; this tiny fixture cannot contain all scenes.
                experiment._calibrate_regime_labels = Mock()
                one_cycle = torch.optim.lr_scheduler.OneCycleLR
                with patch("exp.exp_timedart.lr_scheduler.OneCycleLR", wraps=one_cycle) as scheduler:
                    experiment.train("tiny_calibrated")
                self.assertEqual(scheduler.call_count, 2)
                self.assertEqual([call.kwargs["total_steps"] for call in scheduler.call_args_list], [2, 2])
                folder = Path(args.checkpoints) / "tiny_calibrated"
                for name in ("checkpoint.pth", "checkpoint_best_trained.pth", "checkpoint_last.pth"):
                    self.assertTrue((folder / name).is_file(), name)
                manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["extra"]["epochs_completed"], 3)
                self.assertLess(args.utility_calibration_audit["adapter_target_end"], args.utility_calibration_audit["gate_target_start"])
                if factorized:
                    self.assertTrue(manifest['args']['utility_factorized'])
                    from scripts.calibrate_factorized_wiki import calibrate, restore_experiment
                    for dataset in datasets.values():
                        dataset.scaler = SimpleNamespace(mean_=np.zeros(8), scale_=np.ones(8))
                    requested = []
                    def tracked_data(flag):
                        requested.append(flag)
                        return datasets[flag], DataLoader(datasets[flag], batch_size=8)
                    policy_dir = Path(directory) / 'gain_policy'
                    policy_dir.mkdir()
                    with patch.dict(os.environ, {'SDWPF_LOG_DIR': str(policy_dir)}):
                        policy_exp = restore_experiment(manifest['args'], policy_dir)
                    policy_exp._get_data = tracked_data
                    try:
                        result = calibrate(policy_exp, manifest, folder / 'checkpoint_last.pth',
                                           policy_dir, blocks=2, min_windows=1)
                    finally:
                        policy_exp.writer.close()
                    self.assertEqual(requested, ['train', 'val'])
                    self.assertEqual(set(result), {'calibrated', 'trend', 'previous_neural_last'})
                    self.assertTrue((policy_dir / 'summary.txt').is_file())
                    policy = PromptGuidedModel(args).eval()
                    policy.load_state_dict(torch.load(policy_dir / 'checkpoint.pth', weights_only=True))
                    self.assertTrue(policy.utility_gate.policy_enabled)
                    with torch.no_grad():
                        torch.testing.assert_close(policy(history), policy_exp.model(history))


if __name__ == "__main__":
    unittest.main()
