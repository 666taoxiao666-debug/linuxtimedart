import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np
import torch

from layers.TimeDART_EncDec import SceneWikiPromptRouter
from utils.wind_regime_wiki import (
    REQUIRED_SCENE_IDS,
    compute_scene_wiki_rule_logits,
    load_wind_regime_wiki_spec,
)


ROOT = Path(__file__).resolve().parents[1]
WIKI_CONFIG = ROOT / "configs" / "wind_regime_wiki.json"


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
    return embeddings


class SceneWikiTests(unittest.TestCase):
    def test_versioned_scene_order_is_checkpoint_contract(self):
        spec = load_wind_regime_wiki_spec(WIKI_CONFIG)
        self.assertEqual(tuple(spec["scene_ids"]), REQUIRED_SCENE_IDS)
        self.assertEqual(len(spec["sha256"]), 64)

    def test_causal_rules_cover_all_seven_observable_scenes(self):
        history = torch.zeros(7, 12, 2)
        history[0, :, 0] = 7.0
        history[0, :, 1] = 700.0
        history[1, :, 0] = 7.0
        history[1, :, 1] = torch.linspace(200.0, 700.0, 12)
        history[2, :, 0] = 7.0
        history[2, :, 1] = torch.linspace(700.0, 200.0, 12)
        history[3, :, 0] = torch.tensor([5.0, 8.0] * 6)
        history[3, :, 1] = 600.0
        history[4, :, 0] = 8.0
        history[4, :, 1] = 20.0
        history[5, :, 0] = 11.0
        history[5, :, 1] = 1450.0
        history[6, :, 0] = 2.0
        history[6, :, 1] = 0.0
        logits, labels = compute_scene_wiki_rule_logits(
            history,
            ["Wspd", "power"],
            rated_power=1500.0,
        )
        self.assertEqual(labels.tolist(), list(range(7)))
        self.assertEqual(tuple(logits.shape), (7, 7))

    def test_top_k_router_probabilities_are_normalized_and_sparse(self):
        router = SceneWikiPromptRouter(
            np.random.default_rng(2024).standard_normal((7, 16)).astype(np.float32),
            d_model=8,
            top_k=2,
            temperature=0.2,
            rule_weight=2.0,
            dropout=0.0,
        )
        x_out = torch.randn(4, 10, 8)
        rule = torch.nn.functional.one_hot(
            torch.tensor([0, 1, 2, 3]), num_classes=7
        ).float()
        prompt, logits, probabilities = router(x_out, rule)
        self.assertEqual(tuple(prompt.shape), (4, 8))
        self.assertEqual(tuple(logits.shape), (4, 7))
        self.assertTrue(torch.allclose(probabilities.sum(dim=-1), torch.ones(4)))
        self.assertTrue(torch.all((probabilities > 0).sum(dim=-1) <= 2))

    @unittest.skipUnless(
        importlib.util.find_spec("reformer_pytorch"),
        "full run.py dependencies are not installed",
    )
    def test_configure_scene_wiki_records_bundle_identity(self):
        from run import build_parser, configure_args

        spec = load_wind_regime_wiki_spec(WIKI_CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "wiki.npz"
            _write_bundle(bundle, spec)
            args = configure_args(
                build_parser().parse_args(
                    [
                        "--task_name",
                        "pretrain",
                        "--model_id",
                        "SDWPF",
                        "--model",
                        "PromptTimeDART",
                        "--data",
                        "SDWPF",
                        "--prompt_router",
                        "scene_wiki",
                        "--scene_wiki_config",
                        str(WIKI_CONFIG),
                        "--scene_wiki_embeddings",
                        str(bundle),
                        "--no-use_gpu",
                    ]
                )
            )
        self.assertEqual(args.num_modes, 7)
        self.assertEqual(args.regime_label_method, "scene_wiki")
        self.assertEqual(args.scene_wiki_scene_ids, list(REQUIRED_SCENE_IDS))
        self.assertEqual(len(args.scene_wiki_bundle_sha256), 64)

    @unittest.skipUnless(
        importlib.util.find_spec("reformer_pytorch"),
        "full model dependencies are not installed",
    )
    def test_prompt_model_routes_history_and_returns_scene_supervision(self):
        from exp.exp_timedart import Exp_TimeDART
        from models.TimeDART import PromptGuidedModel
        from run import build_parser, configure_args

        spec = load_wind_regime_wiki_spec(WIKI_CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "wiki.npz"
            _write_bundle(bundle, spec)
            args = configure_args(
                build_parser().parse_args(
                    [
                        "--task_name", "pretrain",
                        "--model_id", "SDWPF",
                        "--model", "PromptTimeDART",
                        "--data", "SDWPF",
                        "--prompt_router", "scene_wiki",
                        "--scene_wiki_config", str(WIKI_CONFIG),
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
            prediction, logits, labels = model(torch.zeros(2, 24, 8))
            experiment = Exp_TimeDART.__new__(Exp_TimeDART)
            experiment.args = args
            experiment.model = model
            Exp_TimeDART._save_pretrain_checkpoint(
                experiment, temporary, "wiki_pretrain.pth", epoch=0
            )
            saved = torch.load(
                Path(temporary) / "wiki_pretrain.pth", map_location="cpu"
            )["model_state_dict"]
        self.assertEqual(tuple(prediction.shape), (2, 24, 8))
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertEqual(labels.tolist(), [6, 6])
        self.assertIn("scene_wiki_router.semantic_keys", saved)
        self.assertIn("scene_wiki_router.prompt_gate_logit", saved)
        self.assertIn("scene_scaler_mean", saved)
        self.assertIn("scene_scaler_scale", saved)


if __name__ == "__main__":
    unittest.main()
