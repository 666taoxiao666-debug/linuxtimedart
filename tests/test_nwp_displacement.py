"""Tests for models/NWPDisplacement.py (G1 station blocks: I_pool, generic phi control I, displacement method M).

The numpy tests run without torch; the torch tests (numpy-vs-torch parity, gradients, wrapper) run on the server.
The module is loaded by file path so that models/__init__.py is not imported.
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("nwp_displacement_under_test", ROOT / "models" / "NWPDisplacement.py")
nd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nd)
HAS_TORCH = nd._torch is not None
SHARP = {"kp": np.asarray(nd.KERNEL_INIT), "t_raw": -12.0, "l_raw": 0.0}  # temperature ~ TEMP_FLOOR


def make_batch(B=2, S=3, age=11, seed=0):
    rng = np.random.default_rng(seed)
    return {"oS": rng.uniform(2, 12, (B, S, 3)), "oS_mask": np.ones((B, S, 3)),
            "gS": rng.uniform(2, 12, (B, S, 8)), "gS_mask": np.ones((B, S, 8)),
            "dS": np.tile([[0.0, 1.0]], (B, S, 1)), "dS_mask": np.ones((B, S)),
            "gF": rng.uniform(2, 12, (B, 13)), "gF_mask": np.ones((B, 13)),
            "age": np.full(B, float(age)), "cycle_ok": np.ones(B)}


def make_geo(S=3):
    # all stations due north (bearing sin 0, cos 1) of the farm, i.e. upwind of a northerly wind
    return np.tile([[0.5, 0.0, 1.0]], (S, 1))


def shifted_station_batch(delta, age=11, S=2):
    """Station forecasts are a ramp f(u); observations equal f(tau + delta), so the true shift is delta."""
    b = make_batch(B=1, S=S, age=age)
    u = np.asarray(nd.GS_OFF, dtype=float)
    for s in range(S):
        f = lambda t, s=s: 5.0 + 0.8 * t + 0.1 * s
        b["gS"][0, s] = f(u)
        b["oS"][0, s] = f(np.asarray(nd.W_OFF, dtype=float) + delta)
    b["gF"][0] = 6.0 + 0.5 * np.asarray(nd.GF_OFF, dtype=float)
    return b


class Constants(unittest.TestCase):
    def test_widths_and_param_counts(self):
        self.assertEqual(nd.phi_param_count(), 599)
        self.assertEqual([nd.block_width(a) for a in nd.ARMS], [0, 0, 28, 35, 35])
        self.assertEqual(nd.structural_param_counts()["M"], 6)
        self.assertEqual(nd.structural_param_counts()["I"], 603)

    def test_admissible_shifts_whole_window_rule(self):
        o = nd._NumpyOps()
        ages = np.asarray([5.0, 6.0, 7.0, 8.0, 11.0, 12.0])
        adm = nd.admissible_shifts(o, ages, np.ones(6))
        got = {a: [d for d, m in zip(nd.SHIFTS, row) if m] for a, row in zip(ages, adm)}
        self.assertEqual(got[6.0], [0, 1, 2, 3])            # delta >= 6 - age
        self.assertEqual(got[7.0], [-1, 0, 1, 2, 3])
        self.assertEqual(got[8.0], [-2, -1, 0, 1, 2, 3])
        self.assertEqual(got[11.0], [-2, -1, 0, 1, 2, 3])
        self.assertEqual(got[5.0], [1, 2, 3])
        self.assertEqual(got[12.0], [-2, -1, 0, 1, 2])      # farm read h=4, delta=3 would need lead 19
        self.assertFalse(nd.admissible_shifts(o, np.asarray([8.0]), np.zeros(1)).any())

    def test_lead_masks(self):
        o = nd._NumpyOps()
        ms, mf = nd.lead_masks(o, np.asarray([6.0]), np.ones(1))
        self.assertEqual(ms[0].tolist(), [0, 0, 1, 1, 1, 1, 1, 1])  # t0-5, t0-4 have lead 1, 2
        self.assertEqual(mf[0].sum(), 13 - 2)


class Blocks(unittest.TestCase):
    def test_validate_batch(self):
        b = make_batch()
        nd.validate_batch(b, 3)
        bad = dict(b, gS=b["gS"].copy())
        bad["gS"][0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            nd.validate_batch(bad, 3)
        with self.assertRaises(ValueError):
            nd.validate_batch(dict(b, oS_mask=b["oS_mask"] * 0.5), 3)

    def test_pooled_block_matches_hand_computation(self):
        b = make_batch(B=1, S=2, age=11)
        b["oS_mask"][0, 1, 0] = 0.0
        geo = make_geo(2)
        out = nd.station_blocks_numpy("I_pool", b, geo)
        self.assertEqual(out.shape, (1, 28))
        # equal geometry -> equal weights -> plain masked means
        self.assertAlmostEqual(out[0, 0], b["oS"][0, 0, 0], places=5)                 # only station 0 valid
        self.assertAlmostEqual(out[0, 1], b["oS"][0, :, 1].mean(), places=5)
        self.assertAlmostEqual(out[0, 3], b["gS"][0, :, 0].mean(), places=5)
        r = b["oS"][0, :, 2] - b["gS"][0, :, 4]
        self.assertAlmostEqual(out[0, 13], r.mean(), places=5)
        self.assertAlmostEqual(out[0, 14], 0.5, places=5)                             # coverage of o(t0-3)
        self.assertAlmostEqual(out[0, 15], 1.0, places=5)

    def test_upwind_station_gets_more_weight(self):
        o = nd._NumpyOps()
        geo = np.asarray([[0.5, 0.0, 1.0], [0.5, 0.0, -1.0]])   # north (upwind of a northerly) vs south
        w, align = nd.kernel_weights(o, np.asarray(nd.KERNEL_INIT), geo, np.tile([[0.0, 1.0]], (1, 2, 1)),
                                     np.ones((1, 2)))
        self.assertGreater(w[0, 0], 5 * w[0, 1])
        self.assertAlmostEqual(align[0, 0], 1.0, places=5)

    def test_displacement_recovers_known_shift(self):
        for delta in (-2, 0, 2, 3):
            out, diag = nd.displacement_diagnostics_numpy(shifted_station_batch(delta), make_geo(2), SHARP)
            self.assertAlmostEqual(out[0, 4], delta, places=2, msg=f"delta_F for {delta}")
            self.assertAlmostEqual(out[0, 5], 0.0, places=2)                           # no residual bias
            self.assertGreater(out[0, 6], 0.99)
            expect = 6.0 + 0.5 * (np.asarray(nd.HORIZONS) + delta)
            np.testing.assert_allclose(out[0, :4], expect, atol=1e-2)

    def test_fractional_shift_interpolates_linearly(self):
        o = nd._NumpyOps()
        b = shifted_station_batch(0)
        ms, mf = nd.lead_masks(o, b["age"], b["cycle_ok"])
        adm = nd.admissible_shifts(o, b["age"], b["cycle_ok"])
        w = np.ones((1, 2))
        # force a 50/50 split between delta 0 and 1 by making both costs equal
        b["oS"][0] = (b["gS"][0, :, 2:5] + b["gS"][0, :, 3:6]) / 2
        out = nd.displacement_block(o, b, w, ms, mf, adm, temp=0.05, lam=0.0)
        self.assertAlmostEqual(out[0, 4], 0.5, places=3)
        np.testing.assert_allclose(out[0, :4], 6.0 + 0.5 * (np.asarray(nd.HORIZONS) + 0.5), atol=1e-3)

    def test_r2_b1_counterexample(self):
        """Swapping which station carries which pattern leaves every station average equal (I_pool cannot tell),
        while M (per-station matching) and I (per-station phi) must tell the two cases apart."""
        geo = make_geo(2)
        s = np.zeros(8)
        s[:3], s[3:6] = 1.0, -1.0                       # offsets -5..-3 -> +1, -2..0 -> -1, 1..2 -> 0
        cases = {}
        for name, o1 in (("X", 1.0), ("Y", -1.0)):
            b = make_batch(B=1, S=2, age=11)
            b["gS"][0, 0], b["gS"][0, 1] = 5 + s, 5 - s
            b["oS"][0, 0], b["oS"][0, 1] = 5 + o1, 5 - o1
            b["gF"][0] = 6.0 + 0.5 * np.asarray(nd.GF_OFF, dtype=float)
            cases[name] = b
        pool = {k: nd.station_blocks_numpy("I_pool", v, geo) for k, v in cases.items()}
        np.testing.assert_allclose(pool["X"], pool["Y"], atol=1e-9)
        m = {k: nd.station_blocks_numpy("M", v, geo, SHARP) for k, v in cases.items()}
        self.assertAlmostEqual(m["X"][0, 28 + 4], -2.0, places=2)
        self.assertAlmostEqual(m["Y"][0, 28 + 4], 1.0, places=2)
        i = {k: nd.station_blocks_numpy("I", v, geo, seed=3) for k, v in cases.items()}
        self.assertGreater(np.abs(i["X"][0, 28:] - i["Y"][0, 28:]).max(), 1e-4)

    def test_masked_and_inadmissible_inputs_are_not_read(self):
        geo = make_geo(3)
        base = make_batch(B=1, S=3, age=6, seed=4)
        base["oS_mask"][0, 2, 1] = 0.0
        ref = nd.station_blocks_numpy("M", base, geo, SHARP)
        b = {k: v.copy() for k, v in base.items()}
        b["gS"][0, :, :2] += 50.0                       # t0-5, t0-4: lead 1, 2 at age 6 (masked)
        b["oS"][0, 2, 1] += 50.0                        # masked observation
        b["gF"][0, :4] += 50.0                          # farm t0-5..t0-2: never read by the correction
        out = nd.station_blocks_numpy("M", b, geo, SHARP)
        np.testing.assert_allclose(out[0, 28:], ref[0, 28:], atol=1e-9)

    def test_missing_stations_and_missing_cycle(self):
        geo = make_geo(3)
        b = make_batch(B=1, S=3, age=9)
        b["oS_mask"][:] = 0.0
        out = nd.station_blocks_numpy("M", b, geo, SHARP)
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(out[0, 32:].tolist(), [0.0, 0.0, 0.0])            # delta_F, b_F, confidence
        np.testing.assert_allclose(out[0, 28:32], b["gF"][0, 6:10], atol=1e-6)  # uncorrected g_F(t0+h)
        b2 = make_batch(B=1, S=3, age=9)
        b2["cycle_ok"][:] = 0.0
        for arm in ("I_pool", "I", "M"):
            out2 = nd.station_blocks_numpy(arm, b2, geo, SHARP if arm == "M" else None)
            self.assertTrue(np.isfinite(out2).all())
        self.assertEqual(np.abs(nd.station_blocks_numpy("M", b2, geo, SHARP)[0, 28:]).max(), 0.0)

    def test_station_permutation_and_batch_independence(self):
        geo = np.asarray([[0.3, 0.2, 0.98], [0.9, -0.6, 0.8], [0.5, 1.0, 0.0]])
        b = make_batch(B=3, S=3, age=8, seed=7)
        b["dS"] = np.tile([[0.3, 0.95]], (3, 3, 1))
        perm = [2, 0, 1]
        bp = {k: (v[:, perm] if v.ndim >= 2 and v.shape[1] == 3 and k.startswith(("oS", "gS", "dS")) else v)
              for k, v in b.items()}
        for arm in ("I_pool", "I", "M"):
            a = nd.station_blocks_numpy(arm, b, geo, seed=1)
            c = nd.station_blocks_numpy(arm, bp, geo[perm], seed=1)
            np.testing.assert_allclose(a, c, atol=1e-9, err_msg=arm)
            single = nd.station_blocks_numpy(arm, {k: v[1:2] for k, v in b.items()}, geo, seed=1)
            np.testing.assert_allclose(single[0], a[1], atol=1e-9, err_msg=arm)


@unittest.skipUnless(HAS_TORCH, "torch not installed (parity and gradient checks run on the server)")
class Torch(unittest.TestCase):
    def _torch_params_from_numpy(self, blk, params):
        import torch
        with torch.no_grad():
            blk.kp.copy_(torch.as_tensor(params["kp"]))
            if blk.arm == "M":
                blk.t_raw.fill_(float(params["t_raw"]))
                blk.l_raw.fill_(float(params["l_raw"]))
            if blk.arm == "I":
                blk.phi[0].weight.copy_(torch.as_tensor(params["W1"]).T)
                blk.phi[0].bias.copy_(torch.as_tensor(params["b1"]))
                blk.phi[2].weight.copy_(torch.as_tensor(params["W2"]).T)
                blk.phi[2].bias.copy_(torch.as_tensor(params["b2"]))

    def test_numpy_torch_parity_and_gradients(self):
        import torch
        geo = np.asarray([[0.3, 0.2, 0.98], [0.9, -0.6, 0.8], [0.5, 1.0, 0.0]])
        b = make_batch(B=4, S=3, age=7, seed=11)
        b["age"] = np.asarray([6.0, 7.0, 9.0, 11.0])
        b["oS_mask"][1, 2, :] = 0.0
        for arm in ("I_pool", "I", "M"):
            params = nd.init_params_numpy(arm, seed=5)
            ref = nd.station_blocks_numpy(arm, b, geo, params)
            blk = nd.StationBlocks(arm, geo).double()
            # Match float64 reference inputs, avoiding constructor float32 rounding.
            blk.geo.copy_(torch.as_tensor(geo, dtype=torch.float64))
            self._torch_params_from_numpy(blk, params)
            out = blk({k: torch.as_tensor(v) for k, v in b.items()})
            np.testing.assert_allclose(out.detach().numpy(), ref, atol=1e-8, rtol=0, err_msg=arm)
            out.square().sum().backward()
            for name, p in blk.named_parameters():
                self.assertTrue(torch.isfinite(p.grad).all(), f"{arm}.{name}")
                self.assertGreater(float(p.grad.abs().sum()), 0.0, f"{arm}.{name}")
            self.assertGreater(float(blk.kp.grad.abs().sum()), 0.0)
            if arm == "M":
                self.assertGreater(abs(float(blk.t_raw.grad)) + abs(float(blk.l_raw.grad)), 0.0)

    def test_wrapper_layout_and_normalization(self):
        import torch
        geo = make_geo(3)
        b = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in make_batch(B=5, S=3).items()}
        x = torch.randn(5, 24, 50)
        for arm in nd.ARMS:
            m = nd.G1TSMixerModel(arm, 50, geo)
            self.assertEqual(m.backbone.channels, 50 + nd.block_width(arm))
            if nd.block_width(arm):
                with self.assertRaises(RuntimeError):
                    m(x, b)
                m.fit_station_normalization([b])
            captured = []
            hook = m.backbone.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
            y = m(x, b)
            hook.remove()
            torch.testing.assert_close(captured[0][..., -1], x[..., -1])
            torch.testing.assert_close(captured[0][..., :49], x[..., :49])
            self.assertEqual(m.backbone.target_slot, captured[0].shape[-1] - 1)
            self.assertEqual(tuple(y.shape), (5, 4, 1))

    def test_empty_normalization_rejected(self):
        for arm in ("I_pool", "I", "M"):
            m = nd.G1TSMixerModel(arm, 50, make_geo(3))
            for batches in ([], [make_batch(B=0)]):
                with self.assertRaises(ValueError):
                    m.fit_station_normalization(batches)
                self.assertFalse(bool(m.normalized))


if __name__ == "__main__":
    unittest.main()
