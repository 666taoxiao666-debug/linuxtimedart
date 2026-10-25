"""Model-level tests for v26 DLinearAnchoredResidual: exact epoch-0 identity, frozen anchor, gradients, masking, RNG."""
import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:  # pragma: no cover - the training host always has torch
    raise unittest.SkipTest("torch unavailable")

from models.DLinear import Model as DLinear
from models.DLinearAnchoredResidual import Model
from models.TSMixerBasicSmall import Model as TSMixer

ANCHOR_ARGS = {"task_name": "finetune", "downstream_task": "forecast", "seq_len": 24, "pred_len": 4, "enc_in": 7,
               "individual": 0, "moving_avg": 25}


def anchor_state(seed=3):
    torch.manual_seed(seed)
    return {k: v.detach().clone() for k, v in DLinear(SimpleNamespace(**ANCHOR_ARGS)).state_dict().items()}


def build(mode="full", seed=0, state=None):
    torch.manual_seed(seed)
    model = Model(seq_len=24, channels=7, pred_len=4, blocks=2, ff_dim=32, dropout=.1, norm_eps=1e-3, target_slot=6,
                  input_mode=mode, anchor_args=dict(ANCHOR_ARGS))
    model.load_anchor_state(state if state is not None else anchor_state())
    return model


def reference_dlinear(state):
    model = DLinear(SimpleNamespace(**ANCHOR_ARGS))
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model.eval()


class AnchoredResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def batch(self, n=17, seed=1):
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(n, 24, 7, generator=generator)

    def test_parameter_budget_trainable_vs_frozen(self):
        model = build()
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 3618)
        self.assertEqual(sum(p.numel() for p in model.parameters() if not p.requires_grad), 200)
        self.assertTrue(all(not p.requires_grad for p in model.anchor.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.residual.parameters()))
        self.assertTrue(torch.equal(model.residual.head.weight, torch.zeros(4, 24)))
        self.assertTrue(torch.equal(model.residual.head.bias, torch.zeros(4)))

    def test_initial_forecast_is_exactly_the_frozen_dlinear(self):
        state = anchor_state()
        x = self.batch()
        expected = reference_dlinear(state)(x)[..., -1:]
        for mode in ("full", "power_only"):
            model = build(mode, state=state).eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(model.correction(x), torch.zeros(17, 4, 1)), mode)
                self.assertTrue(torch.equal(model(x), expected), mode)
                self.assertTrue(torch.equal(model.anchor_forecast(x), expected), mode)
            model.train()  # dropout inside the residual cannot move a zero head
            with torch.no_grad():
                self.assertTrue(torch.equal(model(x), expected), mode)

    def test_anchor_is_channel_independent_and_power_only_ignores_auxiliaries(self):
        state = anchor_state()
        x = self.batch()
        y = x.clone()
        y[..., :6] = torch.randn(17, 24, 6) * 50
        z = x.clone()
        z[..., :6] = float("nan")
        model = build("power_only", state=state).eval()
        with torch.no_grad():
            # perturb the residual so the invariance is tested on a non-trivial branch
            torch.nn.init.normal_(model.residual.head.weight)
            a, b, c = model(x), model(y), model(z)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a, c))
        full = build("full", state=state).eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(full.anchor_forecast(x), full.anchor_forecast(y)))

    def test_residual_blocks_start_from_the_v25_tsmixer_tensors_and_anchor_consumes_no_rng(self):
        torch.manual_seed(2024)
        tsmixer = TSMixer(seq_len=24, channels=7, pred_len=4, blocks=2, ff_dim=32, dropout=.1, norm_eps=1e-3, target_slot=6)
        after_tsmixer = torch.get_rng_state()
        torch.manual_seed(2024)
        model = Model(anchor_args=dict(ANCHOR_ARGS))
        after_model = torch.get_rng_state()
        self.assertTrue(torch.equal(after_tsmixer, after_model), "anchor construction must not consume global draws")
        reference, residual = tsmixer.state_dict(), model.residual.state_dict()
        self.assertEqual(set(reference), set(residual))
        for key in reference:
            if key.startswith("head."):
                continue
            self.assertTrue(torch.equal(reference[key], residual[key]), key)

    def test_paired_arms_have_identical_state(self):
        state = anchor_state()
        a, b = build("full", 7, state).state_dict(), build("power_only", 7, state).state_dict()
        self.assertEqual(set(a), set(b))
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))

    def test_gradients_head_first_then_blocks_anchor_never(self):
        torch.manual_seed(0)
        model = build()
        model.train()
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
        self.assertEqual({id(p) for g in optimizer.param_groups for p in g["params"]}, {id(p) for p in trainable})
        before = {k: v.clone() for k, v in model.anchor.state_dict().items()}
        x, y = self.batch(32, 4), torch.randn(32, 4)

        def step():
            optimizer.zero_grad(set_to_none=True)
            difference = model(x)[..., -1] - y
            (0.8 * difference.abs().mean() + 0.2 * difference.square().mean()).backward()
            grads = {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in model.named_parameters()}
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            return grads

        first = step()
        self.assertTrue(torch.isfinite(first["residual.head.weight"]).all())
        self.assertGreater(first["residual.head.weight"].abs().sum().item(), 0)
        self.assertGreater(first["residual.head.bias"].abs().sum().item(), 0)
        # zero head => no upstream gradient at the very first step (expected, not a bug)
        self.assertEqual(first["residual.blocks.0.feature_out.weight"].abs().sum().item(), 0.0)
        second = step()
        self.assertGreater(second["residual.blocks.0.feature_out.weight"].abs().sum().item(), 0)
        self.assertGreater(second["residual.blocks.1.temporal.weight"].abs().sum().item(), 0)
        for name, grad in second.items():
            if name.startswith("anchor."):
                self.assertIsNone(grad, name)
        after = model.anchor.state_dict()
        self.assertTrue(all(torch.equal(before[k], after[k]) for k in before))

    def test_anchor_stays_in_eval_and_strict_loading(self):
        model = build()
        model.train()
        self.assertTrue(model.training and model.residual.training)
        self.assertFalse(model.anchor.training)
        model.eval()
        self.assertFalse(model.anchor.training)
        with self.assertRaises(RuntimeError):
            model.load_anchor_state({"seasonal.weight": torch.zeros(4, 24)})
        self.assertTrue(all(not p.requires_grad for p in model.anchor.parameters()))

    def test_input_validation(self):
        model = build()
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 24, 6))
        with self.assertRaises(ValueError):
            Model(input_mode="aux_only")
        with self.assertRaises(ValueError):
            Model(anchor_args={**ANCHOR_ARGS, "individual": 1})
        with self.assertRaises(ValueError):
            Model(anchor_args={**ANCHOR_ARGS, "pred_len": 8})


if __name__ == "__main__":
    unittest.main()
