"""Model-level tests for the v25 TSMixerBasicSmall port (shape, parameters, axis semantics, masking)."""
import inspect
import math
import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - the training host always has torch
    raise unittest.SkipTest("torch unavailable")

from models.TSMixerBasicSmall import Model, power_only_input


def build(mode="full", seed=0):
    torch.manual_seed(seed)
    return Model(seq_len=24, channels=7, pred_len=4, blocks=2, ff_dim=32, dropout=.1, norm_eps=1e-3,
                 target_slot=6, input_mode=mode)


def numpy_forward(model, x):
    """Independent re-implementation of the upstream res_block axes (eval mode, no dropout)."""
    s = {k: v.detach().double().numpy() for k, v in model.state_dict().items()}
    h = x.astype(np.float64)

    def layer_norm(v, w, b):
        mean = v.mean(axis=(1, 2), keepdims=True)
        var = v.var(axis=(1, 2), keepdims=True)
        return (v - mean) / np.sqrt(var + 1e-3) * w + b

    for i in range(2):
        p = f"blocks.{i}."
        z = layer_norm(h, s[p + "temporal_norm.weight"], s[p + "temporal_norm.bias"])
        z = np.transpose(z, (0, 2, 1)) @ s[p + "temporal.weight"].T + s[p + "temporal.bias"]  # mixes TIME
        res = np.transpose(np.maximum(z, 0), (0, 2, 1)) + h
        z = layer_norm(res, s[p + "feature_norm.weight"], s[p + "feature_norm.bias"])
        z = np.maximum(z @ s[p + "feature_in.weight"].T + s[p + "feature_in.bias"], 0)  # mixes CHANNELS
        h = z @ s[p + "feature_out.weight"].T + s[p + "feature_out.bias"] + res
    return (h[:, :, 6] @ s["head.weight"].T + s["head.bias"])[..., None]


class TSMixerBasicSmallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_parameter_budget_and_breakdown(self):
        model = build()
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 3618)
        block = model.blocks[0]
        self.assertEqual(sum(p.numel() for p in block.temporal_norm.parameters()) +
                         sum(p.numel() for p in block.feature_norm.parameters()), 672)
        self.assertEqual(sum(p.numel() for p in block.temporal.parameters()), 600)
        self.assertEqual(sum(p.numel() for p in block.feature_in.parameters()) +
                         sum(p.numel() for p in block.feature_out.parameters()), 487)
        self.assertEqual(sum(p.numel() for p in model.head.parameters()), 100)
        self.assertEqual(len(model.blocks), 2)
        self.assertEqual(tuple(block.temporal_norm.normalized_shape), (24, 7))
        self.assertEqual(block.temporal_norm.eps, 1e-3)
        self.assertEqual(block.dropout.p, .1)

    def test_shapes_and_forward_signature(self):
        model = build().eval()
        for batch in (1, 3, 64):
            self.assertEqual(tuple(model(torch.randn(batch, 24, 7)).shape), (batch, 4, 1))
        with self.assertRaises(ValueError):
            model(torch.randn(2, 25, 7))
        with self.assertRaises(ValueError):
            model(torch.randn(2, 24, 6))
        self.assertEqual(list(inspect.signature(model.forward).parameters), ["history"])

    def test_manual_numpy_reference_matches_time_and_channel_axes(self):
        model = build(seed=3).eval()
        x = np.random.default_rng(1).normal(size=(5, 24, 7)).astype(np.float32)
        with torch.no_grad():
            torch_out = model(torch.from_numpy(x)).double().numpy()
        np.testing.assert_allclose(torch_out, numpy_forward(model, x), rtol=1e-5, atol=1e-5)
        # swapping the temporal weight for its transpose must change the output (axis really used)
        with torch.no_grad():
            model.blocks[0].temporal.weight.copy_(model.blocks[0].temporal.weight.T.clone())
            self.assertFalse(np.allclose(model(torch.from_numpy(x)).double().numpy(), torch_out, atol=1e-6))

    def test_layernorm_never_mixes_batch_samples_and_non_contiguous_input(self):
        model = build(seed=4).eval()
        x = torch.randn(6, 24, 7)
        with torch.no_grad():
            whole = model(x)
            single = torch.cat([model(x[i:i + 1]) for i in range(6)])
            other = model(torch.cat([x[:2], 100 * torch.randn(9, 24, 7)]))[:2]
            non_contiguous = x.transpose(1, 2).contiguous().transpose(1, 2)
            self.assertFalse(non_contiguous.is_contiguous())
            torch.testing.assert_close(model(non_contiguous), whole)
        torch.testing.assert_close(whole, single, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(other, whole[:2], rtol=1e-5, atol=1e-5)

    def test_eval_is_deterministic_and_dropout_only_in_train(self):
        model = build(seed=5)
        x = torch.randn(8, 24, 7)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)
        model.train()
        torch.manual_seed(1)
        first = model(x)
        torch.manual_seed(2)
        self.assertFalse(torch.allclose(first, model(x)))

    def test_forward_backward_finite_and_learnable(self):
        torch.manual_seed(0)
        model = build(seed=6)
        x = torch.randn(256, 24, 7)
        y = (x[:, -4:, 6] * .8 + .2 * x[:, -1:, 0]).unsqueeze(-1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(150):
            optimizer.zero_grad()
            loss = torch.mean((model(x) - y) ** 2)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], .35 * losses[0])

    def test_power_only_ignores_arbitrary_auxiliary_values_but_not_power(self):
        model = build("power_only", seed=7).eval()
        full = build("full", seed=7).eval()
        x = torch.randn(4, 24, 7)
        perturbed = x.clone()
        perturbed[..., :6] = torch.randn(4, 24, 6) * 1e3
        perturbed[0, 3, 2] = float("nan")
        perturbed[1, 5, 0] = float("inf")
        with torch.no_grad():
            torch.testing.assert_close(model(perturbed), model(x), rtol=0, atol=0)
            power = x.clone()
            power[..., 6] += 1.
            self.assertFalse(torch.allclose(model(power), model(x)))
            self.assertFalse(torch.allclose(full(perturbed.nan_to_num(0., 0., 0.)), full(x)))
        masked = power_only_input(perturbed, 6)
        self.assertTrue(torch.equal(masked[..., :6], torch.zeros_like(masked[..., :6])))
        self.assertTrue(torch.equal(masked[..., 6], perturbed[..., 6]))

    def test_paired_arms_share_byte_identical_initial_state(self):
        full, power_only = build("full", seed=2024).state_dict(), build("power_only", seed=2024).state_dict()
        self.assertEqual(list(full), list(power_only))
        for key in full:
            self.assertTrue(torch.equal(full[key], power_only[key]), key)
            self.assertEqual(full[key].dtype, torch.float32)

    def test_initialization_xavier_uniform_zero_bias_layernorm_identity(self):
        model = build(seed=8)
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                fan_out, fan_in = module.weight.shape
                bound = math.sqrt(6. / (fan_in + fan_out))
                self.assertLessEqual(module.weight.abs().max().item(), bound + 1e-7)
                self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))
            if isinstance(module, torch.nn.LayerNorm):
                self.assertTrue(torch.equal(module.weight, torch.ones_like(module.weight)))
                self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            Model(input_mode="wiki")
        with self.assertRaises(ValueError):
            Model(target_slot=7)


if __name__ == "__main__":
    unittest.main()
