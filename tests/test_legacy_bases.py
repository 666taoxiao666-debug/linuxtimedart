"""Static tests for models/LegacyBases.py (numpy parts run anywhere; torch parts skip without torch)."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Load the file directly: models/__init__.py imports torch-only models.
_spec = importlib.util.spec_from_file_location("LegacyBases", ROOT / "models" / "LegacyBases.py")
lb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lb)

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _issue(n_times, per=3, start="2021-01-01T00"):
    t = np.datetime64(start, "h") + np.arange(n_times).astype("timedelta64[h]") * 3
    return np.repeat(t, per)


class NumpyHelpers(unittest.TestCase):
    def test_lambda_schedule_warmup_ramp_and_end(self):
        vals = [lb.lambda_schedule(e, 40, 0.05) for e in range(1, 41)]
        self.assertTrue(all(v == 0.0 for v in vals[:16]))           # epochs 1..16 = first 40%
        self.assertGreater(vals[16], 0.0)
        self.assertAlmostEqual(vals[-1], 0.05)                       # reaches lam_end once, not lam_end**2
        self.assertTrue(all(b >= a for a, b in zip(vals, vals[1:])))

    def test_time_domains_three_contiguous_segments(self):
        issue = _issue(300)
        d = lb.time_domains(issue, 3)
        self.assertEqual(set(d.tolist()), {0, 1, 2})
        self.assertTrue(np.all(np.diff(d) >= 0))                     # monotone in time
        t = issue.astype("datetime64[s]").astype(np.int64)
        for k in range(2):                                           # every domain-k time precedes domain k+1
            self.assertLess(t[d == k].max(), t[d == k + 1].min())

    def test_forward_blocks_keep_issue_time_together_and_ordered(self):
        issue = np.random.default_rng(0).permutation(_issue(90, per=4))
        b = lb.forward_blocks(issue, 3)
        for t in np.unique(issue):
            self.assertEqual(len(set(b[issue == t].tolist())), 1)
        tt = issue.astype("datetime64[s]").astype(np.int64)
        for k in range(2):
            self.assertLess(tt[b == k].max(), tt[b == k + 1].min())

    def test_stacking_recovers_known_weights(self):
        rng = np.random.default_rng(1)
        n = 3000
        issue = _issue(n // 3)
        p = rng.uniform(0, 2000, size=(n, 3, 4))
        y = 0.2 * p[:, 0] + 0.5 * p[:, 1] + 0.3 * p[:, 2] + 5.0
        model = lb.fit_stacking(p, y, issue, scale=2050.0)
        for h in model["horizons"]:
            np.testing.assert_allclose(h["weights"], [0.2, 0.5, 0.3], atol=2e-3)
        np.testing.assert_allclose(lb.apply_stacking(p, model), y, atol=0.5)

    def test_stacking_uses_only_given_calibration_rows(self):
        rng = np.random.default_rng(2)
        issue = _issue(400)
        p = rng.uniform(0, 2000, size=(len(issue), 2, 4))
        y = p.mean(1)
        m1 = lb.fit_stacking(p, y, issue, 2050.0)
        later = rng.uniform(0, 2000, size=(500, 2, 4))              # evaluation rows never enter the fit
        out1 = lb.apply_stacking(later, m1)
        m2 = lb.fit_stacking(p, y, issue, 2050.0)
        np.testing.assert_array_equal(out1, lb.apply_stacking(later, m2))

    def test_stacking_rejects_bad_layout(self):
        issue = _issue(50)
        with self.assertRaises(ValueError):
            lb.fit_stacking(np.zeros((len(issue), 2)), np.zeros((len(issue), 4)), issue, 2050.0)
        model = lb.fit_stacking(np.random.rand(len(issue), 2, 4), np.random.rand(len(issue), 4), issue, 1.0)
        with self.assertRaises(ValueError):
            lb.apply_stacking(np.random.rand(10, 3, 4), model)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TorchModels(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.set_num_threads(2)
        torch.manual_seed(0)

    def test_shapes_interface_all_arms(self):
        x = self.torch.randn(8, 24, 51)
        for arm in lb.ARMS:
            m = lb.build_legacy(arm, 51).eval()
            self.assertEqual(m.arm, "N")
            self.assertIsNone(m.fit_station_normalization(iter([])))
            self.assertEqual(tuple(m(x, None).shape), (8, 4))

    def test_grl_applies_lambda_once(self):
        x = self.torch.randn(5, 3, requires_grad=True)
        lb.grad_reverse(x, 0.3).sum().backward()
        self.assertTrue(self.torch.allclose(x.grad, self.torch.full_like(x, -0.3)))

    def test_da_loss_domain_weight_one_and_feature_gradient_reversed(self):
        torch = self.torch
        m = lb.DATCNGRU(10).train()
        for mod in m.modules():                                       # deterministic comparison
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0
        x, y = torch.randn(6, 24, 10), torch.randn(6, 4)
        d = torch.tensor([0, 1, 2, 0, 1, 2])
        total, task, dom = lb.da_loss(m, x, y, d, lam=0.05)
        m.zero_grad(); total.backward()
        g_disc = [p.grad.clone() for p in m.discriminator.parameters()]
        f = m.features(x).detach()
        m.zero_grad()
        torch.nn.functional.cross_entropy(m.discriminator(f), d).backward()
        for a, b in zip(g_disc, [p.grad for p in m.discriminator.parameters()]):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))          # discriminator trained at weight 1

    def test_tcn_is_causal_within_window(self):
        m = lb.TCNGRU(7).eval()
        x = self.torch.randn(2, 24, 7)
        z1 = m.tcn(x.transpose(1, 2))
        x2 = x.clone(); x2[:, -1] += 5.0
        z2 = m.tcn(x2.transpose(1, 2))
        self.assertTrue(self.torch.allclose(z1[:, :, :-1], z2[:, :, :-1], atol=1e-6))

    def test_cpu_smoke_fit_predict_and_da_step(self):
        import tempfile
        import types
        from scripts import train_g1 as g1

        torch = self.torch
        n, c = 16, 50
        rng = np.random.default_rng(7)
        pack = {
            "x50": rng.standard_normal((n, 24, c)).astype(np.float32),
            "truth": rng.uniform(100.0, 1500.0, size=(n, 4)).astype(np.float32),
            "issue": _issue(n // 2, per=2),
            "power_mean": np.float32(600.0),
            "power_scale": np.float32(450.0),
            "ws_scale": np.float32(4.0),
            "geo": np.zeros((3, 5), dtype=np.float32),
        }
        for k in g1.BATCH_KEYS:
            shape = (n,) if k in ("age", "cycle_ok") else (n, 13)
            pack[k] = np.ones(shape, dtype=np.float32)
        cfg = dict(max_epochs=1, patience=1, batch_size=8, learning_rate=1e-3,
                   eta_min=1e-5, weight_decay=1e-4, mae_weight=0.8, mse_weight=0.2, grad_clip=1.0)
        for arm in lb.ARMS:
            factory = lambda _a, p, _cfg, dev, a=arm: lb.build_legacy(a, p["x50"].shape[-1]).to(dev)
            fit = types.FunctionType(g1.fit_one.__code__, {**vars(g1), "make_model": factory}, "fit_one")
            with tempfile.TemporaryDirectory() as tmp:
                res = fit("N", 2024, pack, pack, cfg, Path(tmp) / arm, "cpu")
                self.assertEqual(res["epochs"], 1)
                self.assertTrue(np.isfinite(res["validation_mae"]))
                m = factory("N", pack, cfg, "cpu")
                m.load_state_dict(torch.load(Path(tmp) / arm / "selected.pt", map_location="cpu", weights_only=True))
                pred = g1.predict(m, pack, "cpu")
                self.assertEqual(pred.shape, (n, 4))
                self.assertTrue(np.isfinite(pred).all())
        da_model = lb.DATCNGRU(c).train()
        opt = torch.optim.AdamW(da_model.parameters(), lr=1e-3, weight_decay=1e-4)
        dom = torch.as_tensor(lb.time_domains(pack["issue"], 3), dtype=torch.long)
        x = torch.as_tensor(pack["x50"], dtype=torch.float32)
        y = torch.as_tensor((pack["truth"] - 600.0) / 450.0, dtype=torch.float32)
        lam = lb.lambda_schedule(20, 40, 0.05)
        opt.zero_grad()
        total, task, dloss = lb.da_loss(da_model, x, y, dom, lam)
        total.backward()
        torch.nn.utils.clip_grad_norm_(da_model.parameters(), 1.0)
        opt.step()
        self.assertTrue(bool(torch.isfinite(total) and torch.isfinite(task) and torch.isfinite(dloss)))


if __name__ == "__main__":
    unittest.main()
