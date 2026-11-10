"""Tests for models/ProgramEncoder.py (L1 program sandbox, expert P, grammar R, TSMixer wrapper).

Static tests run without torch; the dynamic tests need torch and run on the server.
The module is loaded by file path so that models/__init__.py (TimeDART etc.) is not imported.
"""
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("program_encoder_under_test", ROOT / "models" / "ProgramEncoder.py")
pe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pe)
HAS_TORCH = pe._torch is not None
GRAMMAR = json.loads((ROOT / "configs" / "l1_llm_encoder_protocol.json").read_text(encoding="utf-8"))["grammar_R"] \
    if (ROOT / "configs" / "l1_llm_encoder_protocol.json").exists() else {
        "cone_kernels": list(pe.CONE_KERNELS), "distance_kernels": list(pe.DISTANCE_KERNELS),
        "advection_kernels": list(pe.ADVECTION_KERNELS), "lag_steps": [1, 2, 3, 4, 6],
        "channel_counts": [6, 8, 9, 10, 11]}

SIMPLE = """INIT_P = [0.5, -1.0]

def encode(obs, geo, P):
    ws = obs[..., 0]
    vw = obs[..., 5]
    near = torch.exp(-geo[:, 0] / F.softplus(P[0]))
    a = wmean(ws, near * vw, dim=2)
    b = ws.sum(2) / (vw.sum(2) + 1e-6) + P[1]
    return torch.stack([a, b, a - lag(a, 3)], dim=-1)
"""


def body(lines, init="INIT_P = [1.0]"):
    return init + "\n\ndef encode(obs, geo, P):\n" + "".join("    " + x + "\n" for x in lines)


class StaticValidation(unittest.TestCase):
    def test_simple_and_expert_accepted(self):
        info = pe.validate_source(SIMPLE)
        self.assertEqual(info["n_params"], 2)
        expert = pe.validate_source(pe.EXPERT_PROGRAM)
        self.assertEqual(expert["n_params"], len(pe.INIT_P_TEMPLATE))

    def test_rejections(self):
        bad = {
            "import": "import os\n" + SIMPLE,
            "dunder": body(["return obs.__class__"]),
            "for_loop": body(["x = obs[..., 0]", "for i in [1]:", "    x = x", "return x"]),
            "lambda": body(["f = lambda z: z", "return obs[..., 0:1, 0]"]),
            "full_reduction": body(["return obs.sum()"]),
            "batch_dim": body(["return obs.mean(dim=0)"]),
            "cat_batch": body(["return torch.cat([obs, obs], dim=0)"]),
            "torch_load": body(["return torch.load(1)"]),
            "open": body(["return open(1)"]),
            "builtin": body(["return len(obs)"]),
            "huge_pow": body(["return obs[..., 0:3, 0] ** 100"]),
            "const_pow": body(["a = 2 ** 3", "return obs[..., 0:3, 0]"]),
            "too_many_params": body(["return obs[..., 0:3, 0]"], init="INIT_P = [" + ", ".join(["1.0"] * 17) + "]"),
            "no_init_p": "def encode(obs, geo, P):\n    return obs[..., 0:3, 0]\n",
            "extra_def": SIMPLE + "\ndef helper(x):\n    return x\n",
            "undefined_name": body(["return zz"]),
            "name_index": body(["i = 1", "return obs[i]"]),
            "inplace_aug": body(["a = obs[..., 0]", "a += 1.0", "return a"]),
            "item_store": body(["a = obs[..., 0]", "a[0] = 1.0", "return a"]),
            "inplace_method": body(["a = obs[..., 0]", "return a.clamp_(min=0.0)"]),
            "out_keyword": body(["return torch.exp(obs, out=obs)"]),
            "if_stmt": body(["if True:", "    a = 1", "return obs[..., 0:3, 0]"]),
            "lag_k_name": body(["k = 2", "return lag(obs[..., 0], k)"]),
            "lag_k_range": body(["return lag(obs[..., 0], 30)"]),
            "squeeze_no_dim": body(["return obs.squeeze()"]),
            "reserved_assign": body(["torch = 1", "return obs[..., 0:3, 0]"]),
            "bad_signature": "INIT_P = [1.0]\n\ndef encode(obs, geo, P, extra=1):\n    return obs\n",
            "torch_submodule": body(["return torch.nn.Linear(1, 1)"]),
            "syntax": "INIT_P = [1.0\n",
        }
        for name, src in bad.items():
            with self.subTest(name):
                with self.assertRaises(pe.ProgramRejected):
                    pe.validate_source(src)

    def test_rules_text_states_structure_rules(self):
        # R2-B1: the prompt must state the single-name assignment rule that validate_source enforces
        self.assertIn(pe.PROGRAM_STRUCTURE_TEXT, pe.PROGRAM_RULES_TEXT)
        with self.assertRaises(pe.ProgramRejected):
            pe.validate_source(body(["a, b = obs[..., 0], obs[..., 1]", "return torch.stack([a, b], dim=-1)"]))

    def test_torch_nn_functional_chain_allowed(self):
        pe.validate_source(body(["return torch.nn.functional.softplus(obs[..., 0:3, 0])"]))

    def test_canonical_hash_ignores_local_names_and_docstrings(self):
        a = body(["foo = obs[..., 0]", "return torch.stack([foo, foo], dim=-1)"])
        b = body(['"""doc"""', "bar = obs[..., 0]", "return torch.stack([bar, bar], dim=-1)"])
        c = body(["bar = obs[..., 1]", "return torch.stack([bar, bar], dim=-1)"])
        self.assertEqual(pe.canonical_hash(a), pe.canonical_hash(b))
        self.assertNotEqual(pe.canonical_hash(a), pe.canonical_hash(c))

    def test_extract_code(self):
        self.assertEqual(pe.extract_code("text\n```python\nINIT_P = [1.0]\n```\nmore"), "INIT_P = [1.0]\n")
        self.assertEqual(pe.extract_code("INIT_P = [1.0]"), "INIT_P = [1.0]\n")
        # unclosed fence (reply truncated at max_new_tokens): keep the text after the opening fence
        self.assertEqual(pe.extract_code("```python\r\nINIT_P = [1.0]\n\ndef encode("), "INIT_P = [1.0]\n\ndef encode(\n")
        self.assertEqual(pe.extract_code("```python\n" + SIMPLE), SIMPLE)
        pe.validate_source(pe.extract_code("```python\n" + SIMPLE))
        with self.assertRaises(pe.ProgramRejected):  # truncated program still fails on its own text
            pe.validate_source(pe.extract_code("```python\n" + SIMPLE[:-40]))


class GrammarR(unittest.TestCase):
    def test_sample_is_deterministic_unique_valid_and_excludes_expert(self):
        a = pe.sample_grammar_programs(GRAMMAR, seed=2024, budget=30)
        b = pe.sample_grammar_programs(GRAMMAR, seed=2024, budget=30)
        self.assertEqual([x["spec"] for x in a], [x["spec"] for x in b])
        self.assertEqual(len(a), 30)
        hashes = {pe.validate_source(x["source"])["hash"] for x in a}
        self.assertEqual(len(hashes), 30)
        self.assertNotIn(pe.canonical_hash(pe.EXPERT_PROGRAM), hashes)
        for x in a:
            self.assertIn(x["spec"]["k"], GRAMMAR["channel_counts"])

    def test_every_kernel_renders_and_validates(self):
        for cone in pe.CONE_KERNELS:
            for dist in pe.DISTANCE_KERNELS:
                for adv in pe.ADVECTION_KERNELS:
                    pe.validate_source(pe.render_program(cone, dist, adv, 2, 6))


@unittest.skipUnless(HAS_TORCH, "torch not installed (dynamic checks run on the server)")
class Dynamic(unittest.TestCase):
    def test_expert_and_grammar_pass_dynamic_checks(self):
        r = pe.check_program(pe.EXPERT_PROGRAM)
        self.assertEqual(r["status"], "OK", r)
        self.assertEqual(r["K"], 11)
        self.assertGreater(r["grad_nonzero"], 0)
        for x in pe.sample_grammar_programs(GRAMMAR, seed=2024, budget=30):
            with self.subTest(x["spec"]):
                self.assertEqual(pe.check_program(x["source"])["status"], "OK")
        self.assertEqual(pe.check_program(SIMPLE)["status"], "OK")

    def test_dynamic_catches_cross_sample_and_station_order(self):
        leaks = {
            "batch_size_dependence": body(["a = obs[..., 0].mean(dim=2) * obs.shape[0]", "return a.unsqueeze(-1)"]),
            "negative_dim_hits_batch": body(["a = obs[..., 0].mean(dim=-3, keepdim=True)",
                                             "return (obs[..., 0] - a).mean(dim=2).unsqueeze(-1)"]),
            "station_order": body(["return obs[:, :, 0, 0].unsqueeze(-1)"]),
            "fixed_station_count": body(["return obs[..., 0].sum(dim=2).unsqueeze(-1) * geo[8, 0]"]),
            "station_axis_as_channels": body(["return obs[..., 0]"]),
        }
        for name, src in leaks.items():
            with self.subTest(name):
                self.assertNotEqual(pe.check_program(src)["status"], "OK")

    def test_subprocess_ok_reject_and_timeout(self):
        self.assertEqual(pe.verify_in_subprocess(pe.EXPERT_PROGRAM)["status"], "OK")
        self.assertEqual(pe.verify_in_subprocess(body(["return obs.sum()"]))["status"], "REJECTED")
        self.assertEqual(pe.verify_in_subprocess(pe.EXPERT_PROGRAM, timeout_sec=0.01)["status"], "TIMEOUT")

    def test_wrapper_matches_h_layout_and_trains_p(self):
        import torch
        g = torch.Generator().manual_seed(0)
        obs, geo = pe._synthetic(16, 36, g, torch.float32)
        m = pe.ProgramTSMixerModel(pe.EXPERT_PROGRAM, geo, blocks=2, ff_dim=32, dropout=0.1, norm_eps=1e-3)
        self.assertEqual(m.k, 11)
        self.assertEqual(sum(p.numel() for p in m.backbone.parameters()), 8126)  # same as v32 21-channel TSMixer
        x = torch.randn(16, 24, 10, generator=g)
        with self.assertRaises(RuntimeError):
            m(x, obs)
        m.fit_encoder_normalization(obs)
        y = m(x, obs)
        self.assertEqual(tuple(y.shape), (16, 4, 1))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(m.P.grad).all())
        self.assertGreater(int((m.P.grad.abs() > 0).sum()), 0)


if __name__ == "__main__":
    unittest.main()
