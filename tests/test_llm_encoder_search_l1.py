"""Unit tests for L1 LLM upstream encoder search and decision rules."""
from __future__ import annotations
import importlib.util
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_pe_spec = importlib.util.spec_from_file_location("program_encoder_test", ROOT / "models" / "ProgramEncoder.py")
pe = importlib.util.module_from_spec(_pe_spec)
_pe_spec.loader.exec_module(pe)
HAS_TORCH = pe._torch is not None

from scripts import llm_encoder_search_l1 as l1


class TestLlmEncoderSearchL1(unittest.TestCase):
    def test_prompt_examples_static(self):
        for source in (pe.EXPERT_PROGRAM, l1.BLIND_EXAMPLE):
            pe.validate_source(source)
        protocol = l1.read_json(l1.PROTOCOL)
        for blind in (False, True):
            self.assertIn(pe.PROGRAM_STRUCTURE_TEXT, l1.build_llm_prompt(protocol, blind, 1, 1, [])[1])

    @unittest.skipUnless(HAS_TORCH, "torch required")
    def test_prompt_examples_dynamic(self):
        for source in (pe.EXPERT_PROGRAM, l1.BLIND_EXAMPLE):
            self.assertEqual(pe.check_program(source)["status"], "OK")

    def test_preflight_checks_transport_not_legality(self):
        protocol = l1.read_json(l1.PROTOCOL)
        b = pe.feedback_breakdown([[1., 2.]], [[1., 3.]], [[1., 1.]])
        refs = {"target_mae": 150., "expert": {"program_id": "P", "val_mae": 151., "breakdown": b}}
        for tokens, expected in ((100, 0), (1536, 1)):
            rows = [{"slot": i, "raw_output": "```python\ninvalid source\n```", "n_new_tokens": tokens}
                    for i in (1,)]
            with tempfile.TemporaryDirectory() as tmp, patch.object(l1, "verify_llm_weights", return_value={"weights_dir": tmp}), patch.object(l1, "run_llm_worker_batch", return_value=rows), patch.object(l1, "load_real_farm"), patch.object(l1, "prepare_feedback", return_value=refs):
                self.assertEqual(l1.preflight(protocol, Path(tmp), ROOT, "unused"), expected)
                record = l1.read_json(Path(tmp) / "preflight.json")
                self.assertEqual([r["seed"] for r in record["requests"]], [900005])
                self.assertIn("Target to beat", record["requests"][0]["user"])
                self.assertFalse(record["outputs"][0]["source_valid"])

    def test_all_failed_fallback_is_json_serializable(self):
        best = l1.select_best_record([{"status": "FAILED"}], pe.EXPERT_PROGRAM)
        self.assertIsNone(best["val_mae"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fallback.json"
            l1.save(path, best)
            self.assertTrue(l1.read_json(path)["all_failed_fallback"])

    def test_token_limit_gate_boundary(self):
        for counts, expected in [({"L": 0}, False), ({"L": 5}, False), ({"L": 6}, True)]:
            with self.subTest(counts=counts):
                self.assertEqual(l1.token_limit_failed(counts), expected)

    def test_feedback_uses_shown_programs_and_previous_generation_only(self):
        b = pe.feedback_breakdown([[1., 2.]], [[1., 3.]], [[1., 1.]])
        refs = {"target_mae": 150., "expert": {"program_id": "P", "val_mae": 151., "breakdown": b}}
        history = [{"program_id": "L_g1_s1", "slot_index": 1, "generation": 1,
                    "status": "COMPLETE", "val_mae": 149., "breakdown": b, "source": pe.EXPERT_PROGRAM},
                   {"program_id": "bad_old", "generation": 1, "status": "FAILED", "failure_reason": "old_error"},
                   {"program_id": "bad_prev", "generation": 2, "status": "FAILED", "failure_reason": "recent_error"}]
        _, prompt = l1.build_llm_prompt(l1.read_json(l1.PROTOCOL), False, 3, 1, history, refs)
        self.assertIn("L_g1_s1: validation MAE", prompt)
        self.assertIn("recent_error", prompt)
        self.assertNotIn("old_error", prompt)
        self.assertLess(prompt.index("Target to beat"), prompt.index(pe.PROGRAM_RULES_TEXT))

    def test_build_hourly_obs_and_geo_lag1h_and_fit_only_stats(self):
        hours_mixed = pd.DatetimeIndex([
            "2019-12-31 20:00:00+00:00",
            "2019-12-31 21:00:00+00:00",
            "2019-12-31 22:00:00+00:00",
            "2019-12-31 23:00:00+00:00",
            "2020-01-01 00:00:00+00:00",
            "2020-01-01 01:00:00+00:00",
        ])
        s1_df = pd.DataFrame({
            "wind_speed": [2.0, 4.0, 6.0, 8.0, 100.0, 200.0],
            "wind_direction": [0.0, 90.0, 180.0, 270.0, 0.0, 90.0],
            "msl_pressure": [1000.0, 1002.0, 1004.0, 1006.0, 1050.0, 1080.0],
        }, index=hours_mixed - pd.Timedelta(hours=1))
        stations = [{"src_id": 101, "distance_km": 50.0, "bearing_deg": 90.0}]
        obs_arr, geo_arr, stats = l1.build_hourly_obs_and_geo({101: s1_df}, stations, hours_mixed, fit_years=(2019,))
        self.assertEqual(obs_arr.shape, (6, 1, 7))
        self.assertEqual(geo_arr.shape, (1, 3))
        self.assertAlmostEqual(float(geo_arr[0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(stats["ws_mean"]), 5.0, places=5)
        self.assertAlmostEqual(float(np.mean(obs_arr[:4, 0, 0])), 0.0, places=5)
        self.assertGreater(float(obs_arr[4, 0, 0]), 10.0)

    def test_prompts_and_blind_terminology(self):
        protocol = l1.read_json(l1.PROTOCOL)
        sys_l, usr_l = l1.build_llm_prompt(protocol, blind=False, gen=1, slot=1, history=[])
        sys_b, usr_b = l1.build_llm_prompt(protocol, blind=True, gen=1, slot=1, history=[])
        self.assertIn("wind", usr_l.lower())
        self.assertIn("len(INIT_P)", sys_b)
        for forbidden in ("wind", "pressure", "advection", "upwind", "kelmarsh", "penmanshiel", "station"):
            self.assertNotIn(forbidden, sys_b.lower())
            self.assertNotIn(forbidden, usr_b.lower())

    def test_select_best_record_tie_break(self):
        recs = [
            {"program_id": "R_02", "slot_index": 2, "status": "COMPLETE", "val_mae": 150.0, "ast_sha256": "h2", "source": "s2", "verify": {"K": 4, "n_params": 2}},
            {"program_id": "R_01", "slot_index": 1, "status": "COMPLETE", "val_mae": 150.0, "ast_sha256": "h1", "source": "s1", "verify": {"K": 4, "n_params": 2}},
        ]
        best = l1.select_best_record(recs, pe.EXPERT_PROGRAM)
        self.assertEqual(best["program_id"], "R_01")

    def test_decide_l1_truth_table(self):
        def make_summary(k_l, p_l, k_r=99.0, p_r=99.0):
            return {
                "Kelmarsh": {"arms": {
                    "H": {"mean_mae": 100.0},
                    "P": {"mean_mae": 99.0},
                    "R_star": {"mean_mae": k_r},
                    "L_star": {"mean_mae": k_l},
                    "L_blind_star": {"mean_mae": 99.5},
                }},
                "Penmanshiel": {"arms": {
                    "H": {"mean_mae": 100.0},
                    "P": {"mean_mae": 99.0},
                    "R_star": {"mean_mae": p_r},
                    "L_star": {"mean_mae": p_l},
                    "L_blind_star": {"mean_mae": 99.5},
                }},
            }
        self.assertEqual(l1.decide_l1(make_summary(98.0, 98.0)), "CONTINUE")
        self.assertEqual(l1.decide_l1(make_summary(98.4, 98.0, k_r=98.5, p_r=98.5)), "SEARCH_ONLY_STOP_L1")
        self.assertEqual(l1.decide_l1(make_summary(99.0, 98.0)), "STOP_L1")

    @unittest.skipUnless(HAS_TORCH, "torch required")
    def test_smoke_and_duplicate_slot_handling(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            original = l1.predict_checkpoint
            original_fit = l1.fit_arm
            def fit_guard(*args, **kwargs):
                self.assertEqual(kwargs.get("eval_splits", ("validation",)), ("validation",))
                self.assertFalse((out / "test_evaluation_started.json").exists())
                return original_fit(*args, **kwargs)
            def guarded(*args, **kwargs):
                freeze = l1.read_json(out / "formal_selection_freeze.json")
                self.assertEqual(freeze["total_fits"], 8)
                self.assertEqual(len(freeze["checkpoints"]), 8)
                self.assertTrue((out / "test_evaluation_started.json").is_file())
                return original(*args, **kwargs)
            with patch.object(l1, "predict_checkpoint", side_effect=guarded), patch.object(l1, "fit_arm", side_effect=fit_guard):
                rc = l1.smoke(out)
            self.assertEqual(rc, 0)
            self.assertTrue((out / "smoke_summary.json").is_file())
            l_cands = l1.read_json(out / "search_L_candidates.json")
            self.assertEqual(len(l_cands), 2)
            self.assertEqual(l_cands[0]["status"], "COMPLETE")
            self.assertEqual(l_cands[1]["status"], "FAILED")
            self.assertEqual(l_cands[1]["failure_reason"], "duplicate_ast_sha256")
            self.assertEqual(len(list((out / "predictions").rglob("*.npz"))), 10)
            summary = l1.read_json(out / "smoke_summary.json")
            for rel, digest in summary["formal"]["artifact_sha256"].items():
                self.assertEqual(l1.sha(out / rel), digest)
            with np.load(out / "predictions/Kelmarsh/cohort.npz", allow_pickle=False) as cohort:
                self.assertIn("window_starts", cohort.files)
            for arm in summary["formal"]["farms"]["Kelmarsh"]["comparisons"].values():
                self.assertEqual(arm["mae"]["newey_west"]["lag"], 7)


if __name__ == "__main__":
    unittest.main()
