import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from utils.wiki_diagnostic import paired_forward, group_metrics, without_event_forward


class Router(torch.nn.Module):
    def forward(self, x, rule_logits=None):
        a = rule_logits.clamp_min(0)
        return a, a, a.sigmoid(), a


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = Router()
        self.trend = torch.nn.Parameter(torch.tensor(3.0))

    def forward(self, x):
        prompt, *_ = self.router(x, rule_logits=x)
        return self.trend + prompt


class WikiDiagnosticTests(unittest.TestCase):
    def test_real_forecast_model_pair(self):
        from models.TimeDART import PromptGuidedModel
        from run import build_parser, configure_args
        from utils.wind_regime_wiki import load_wind_regime_wiki_spec
        config = Path("configs/wind_event_factor_wiki.json")
        spec = load_wind_regime_wiki_spec(config)
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "wiki.npz"
            np.savez(bundle, embeddings=np.random.default_rng(1).normal(size=(4, 16)).astype(np.float32),
                     scene_ids=np.asarray(spec["scene_ids"]), encoder_name=np.asarray("unit-test"),
                     config_sha256=np.asarray(spec["sha256"]))
            args = configure_args(build_parser().parse_args([
                "--task_name", "finetune", "--model_id", "SDWPF", "--model", "PromptTimeDART",
                "--data", "SDWPF", "--features", "MS", "--prompt_router", "compositional_wiki",
                "--scene_wiki_config", str(config), "--scene_wiki_embeddings", str(bundle),
                "--input_len", "24", "--pred_len", "6", "--patch_len", "6", "--stride", "6",
                "--d_model", "16", "--d_ff", "32", "--n_heads", "4", "--e_layers", "1",
                "--d_layers", "1", "--no-use_gpu"]))
            args.device = torch.device("cpu")
            model = PromptGuidedModel(args).eval()
            model.set_scene_scaler(np.zeros(8), np.ones(8))
            with torch.no_grad():
                model.regime_down_thresh.fill_(-0.1)
                model.regime_up_thresh.fill_(0.1)
                x = torch.zeros(2, 24, 8)
                on, off, captured = paired_forward(model, model.scene_wiki_router, x)
                torch.testing.assert_close(model(x), on)
                for index in range(4):
                    removed = without_event_forward(model, model.scene_wiki_router, x, index)
                    self.assertTrue(torch.isfinite(removed).all())
                torch.testing.assert_close(model(x), on)
            self.assertEqual(on.shape, off.shape)
            self.assertEqual(on.shape[1], 6)
            self.assertEqual(captured[0].shape, (2, 4))
            self.assertTrue(torch.isfinite(on).all())
            self.assertEqual(len(model.scene_wiki_router._forward_hooks), 0)

    def test_suppression_preserves_trend_and_removes_hooks(self):
        model = Toy().eval()
        x = torch.tensor([[2.0], [-1.0]])
        before = {k: v.clone() for k, v in model.state_dict().items()}
        on, off, _ = paired_forward(model, model.router, x)
        torch.testing.assert_close(on, torch.tensor([[5.0], [3.0]]))
        torch.testing.assert_close(off, torch.full_like(x, 3.0))
        torch.testing.assert_close(model(x), on)
        self.assertEqual(len(model.router._forward_hooks), 0)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k])

    def test_gain_sign_and_empty_group(self):
        on = np.array([[1., 1.], [3., 3.]])
        off = np.full((2, 2), 2.)
        truth = np.zeros((2, 2))
        result = group_metrics(on, off, truth, np.array([True, False]), "first")
        self.assertEqual(result["mae_gain_kw"], 1.)
        empty = group_metrics(on, off, truth, np.array([False, False]), "empty")
        self.assertIsNone(empty["mae_gain_kw"])

    def test_delete_contribution_preserves_other_event_scale(self):
        class AdditiveRouter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_factors = 3
                self.register_buffer("semantic_keys", torch.tensor([[2., 0.], [0., 4.], [9., 9.]]))
                self.semantic_projection = torch.nn.Identity()
                self.prompt_norm = torch.nn.Identity()
                self.prompt_delta = torch.nn.Parameter(torch.zeros(3, 2))
                self.prompt_gate_logit = torch.nn.Parameter(torch.tensor(0.))
            def forward(self, x):
                a = x
                values = self.semantic_keys + self.prompt_delta
                p = (a @ values) / (a > 0).sum(-1, keepdim=True).clamp_min(1).sqrt() * .5
                return p, a, a, a
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.router = AdditiveRouter()
            def forward(self, x):
                return self.router(x)[0] + 7
        model = Model().eval()
        x = torch.tensor([[1., 1., 0.], [0., 0., 0.]])
        original = model(x)
        removed = without_event_forward(model, model.router, x, 0)
        torch.testing.assert_close(removed[0], torch.tensor([7., 7.+2/(2**.5)]))
        torch.testing.assert_close(removed[1], original[1])
        torch.testing.assert_close(without_event_forward(model, model.router, x, 2), original)
        self.assertEqual(len(model.router._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
