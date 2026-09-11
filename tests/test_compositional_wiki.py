import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from layers.TimeDART_EncDec import CompositionalEventWikiRouter
from utils.wind_regime_wiki import (
    EVENT_FACTOR_IDS,
    compute_event_factor_rule_logits,
    load_wind_regime_wiki_spec,
    summarize_event_factor_confusion,
    update_event_factor_confusion,
)
from utils.tools import transfer_weights


ROOT = Path(__file__).resolve().parents[1]
EVENT_WIKI_CONFIG = ROOT / "configs" / "wind_event_factor_wiki.json"


def _write_bundle(path, spec, hidden_size=16):
    rng = np.random.default_rng(2024)
    embeddings = rng.standard_normal((len(spec["scene_ids"]), hidden_size)).astype(
        np.float32
    )
    np.savez_compressed(
        path,
        embeddings=embeddings,
        scene_ids=np.asarray(spec["scene_ids"]),
        encoder_name=np.asarray("unit-test-encoder"),
        config_sha256=np.asarray(spec["sha256"]),
    )


class CompositionalWikiTests(unittest.TestCase):
    def test_event_factor_order_is_checkpoint_contract(self):
        spec = load_wind_regime_wiki_spec(EVENT_WIKI_CONFIG)
        self.assertEqual(tuple(spec["scene_ids"]), EVENT_FACTOR_IDS)
        self.assertEqual(spec["entry_type"], "event_factor")
        self.assertEqual(len(spec["sha256"]), 64)

    def test_rules_are_multi_label_and_have_explicit_null_state(self):
        history = torch.zeros(4, 12, 2)
        # Gust/turbulence and rated saturation deliberately overlap.
        history[0, :, 0] = torch.tensor([9.0, 12.0] * 6)
        history[0, :, 1] = 1450.0
        # Ordinary operation: no event factor should be active.
        history[1, :, 0] = 7.0
        history[1, :, 1] = 700.0
        history[2, :, 0] = 8.0
        history[2, :, 1] = 20.0
        history[3, :, 0] = 2.0
        history[3, :, 1] = 0.0

        evidence, targets = compute_event_factor_rule_logits(
            history,
            ["Wspd", "power"],
            rated_power=1500.0,
        )

        self.assertEqual(tuple(evidence.shape), (4, 4))
        self.assertEqual(tuple(targets.shape), (4, 4))
        self.assertEqual(targets[0].tolist(), [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(targets[1].tolist(), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(targets[2].tolist(), [0.0, 1.0, 0.0, 0.0])
        self.assertEqual(targets[3].tolist(), [0.0, 0.0, 0.0, 1.0])
        self.assertGreater(float(evidence[0, 0]), 0.0)
        self.assertGreater(float(evidence[0, 2]), 0.0)
        self.assertTrue(torch.all(evidence[1] <= 0.0))

    def test_router_abstains_exactly_without_positive_physical_support(self):
        router = CompositionalEventWikiRouter(
            np.random.default_rng(2024).standard_normal((4, 16)).astype(np.float32),
            d_model=8,
            top_k=2,
            rule_weight=100.0,
            activation_threshold=0.55,
            dropout=0.0,
        )
        prompt, _, _, activations = router(
            torch.randn(3, 10, 8),
            torch.full((3, 4), -0.25),
        )
        self.assertTrue(torch.equal(activations, torch.zeros_like(activations)))
        self.assertTrue(torch.equal(prompt, torch.zeros_like(prompt)))

    def test_router_composes_independent_factors_with_top_k_sparsity(self):
        router = CompositionalEventWikiRouter(
            np.random.default_rng(2025).standard_normal((4, 16)).astype(np.float32),
            d_model=8,
            top_k=2,
            rule_weight=100.0,
            activation_threshold=0.0,
            dropout=0.0,
        )
        rule = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
        prompt, retrieval_logits, probabilities, activations = router(
            torch.randn(1, 10, 8), rule
        )
        self.assertEqual(tuple(prompt.shape), (1, 8))
        self.assertEqual(tuple(retrieval_logits.shape), (1, 4))
        self.assertEqual(tuple(probabilities.shape), (1, 4))
        self.assertEqual(int((activations > 0).sum()), 2)
        self.assertGreater(float(prompt.abs().sum()), 0.0)
        # Independent sigmoid probabilities are intentionally not normalized.
        self.assertFalse(torch.allclose(probabilities.sum(dim=-1), torch.ones(1)))

    def test_multi_label_metrics_report_null_false_intervention(self):
        confusion = np.zeros((4, 4), dtype=np.int64)
        logits = torch.tensor(
            [
                [10.0, -10.0, 10.0, -10.0],
                [10.0, -10.0, -10.0, -10.0],
            ]
        )
        targets = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        sample_stats = update_event_factor_confusion(confusion, logits, targets)
        metrics = summarize_event_factor_confusion(confusion, sample_stats)
        self.assertEqual(metrics["event_positive_counts"], [1, 0, 1, 0])
        self.assertAlmostEqual(metrics["event_exact_match"], 0.5)
        self.assertAlmostEqual(metrics["event_null_recall"], 0.0)
        self.assertAlmostEqual(metrics["event_false_intervention_on_null"], 1.0)

    def test_legacy_router_checkpoint_is_rejected_explicitly(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.prompt_router = "compositional_wiki"
                self.weight = torch.nn.Parameter(torch.ones(1))

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "legacy.pth"
            torch.save(
                {
                    "prompt_router": "hybrid_wiki",
                    "model_state_dict": TinyModel().state_dict(),
                },
                checkpoint_path,
            )
            with self.assertRaisesRegex(RuntimeError, "prompt router is incompatible"):
                transfer_weights(
                    checkpoint_path,
                    TinyModel(),
                    exclude_head=False,
                    strict=False,
                )

    @unittest.skipUnless(
        importlib.util.find_spec("reformer_pytorch"),
        "full model dependencies are not installed",
    )
    def test_model_keeps_trend_logits_and_returns_event_targets(self):
        from exp.exp_timedart import Exp_TimeDART
        from models.TimeDART import PromptGuidedModel
        from run import build_parser, configure_args

        spec = load_wind_regime_wiki_spec(EVENT_WIKI_CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "event_wiki.npz"
            _write_bundle(bundle, spec)
            args = configure_args(
                build_parser().parse_args(
                    [
                        "--task_name", "pretrain",
                        "--model_id", "SDWPF",
                        "--model", "PromptTimeDART",
                        "--data", "SDWPF",
                        "--prompt_router", "compositional_wiki",
                        "--scene_wiki_config", str(EVENT_WIKI_CONFIG),
                        "--scene_wiki_embeddings", str(bundle),
                        "--input_len", "24",
                        "--pred_len", "6",
                        "--patch_len", "6",
                        "--stride", "6",
                        "--d_model", "16",
                        "--d_ff", "32",
                        "--n_heads", "4",
                        "--e_layers", "1",
                        "--d_layers", "1",
                        "--no-use_gpu",
                    ]
                )
            )
            args.device = torch.device("cpu")
            args.time_steps = 8
            args.dropout = 0.0
            args.head_dropout = 0.0
            model = PromptGuidedModel(args)
            model.set_scene_scaler(np.zeros(8), np.ones(8))
            with torch.no_grad():
                model.regime_down_thresh.fill_(-0.1)
                model.regime_up_thresh.fill_(0.1)
            output = model(torch.zeros(2, 24, 8))
            experiment = Exp_TimeDART.__new__(Exp_TimeDART)
            experiment.args = args
            experiment.model = model
            Exp_TimeDART._save_pretrain_checkpoint(
                experiment, temporary, "event_pretrain.pth", epoch=0
            )
            checkpoint = torch.load(
                Path(temporary) / "event_pretrain.pth", map_location="cpu"
            )
            transferred = transfer_weights(
                Path(temporary) / "event_pretrain.pth",
                PromptGuidedModel(args),
                strict=True,
            )

        self.assertIsInstance(output, dict)
        self.assertEqual(tuple(output["regime_logits"].shape), (2, 3))
        self.assertEqual(tuple(output["event_logits"].shape), (2, 4))
        self.assertEqual(tuple(output["event_targets"].shape), (2, 4))
        self.assertEqual(output["event_targets"].dtype, torch.float32)
        self.assertEqual(args.regime_label_method, "trend_quantile")
        self.assertIsNone(args.scene_wiki_ramp_delta_min_ratio)
        self.assertEqual(checkpoint["prompt_router"], "compositional_wiki")
        self.assertEqual(checkpoint["scene_wiki_scene_ids"], list(EVENT_FACTOR_IDS))
        self.assertEqual(checkpoint["scene_wiki_activation_threshold"], 0.55)
        self.assertEqual(
            transferred.pretrain_transfer_audit[
                "required_backbone_coverage_pct"
            ],
            100.0,
        )


if __name__ == "__main__":
    unittest.main()
