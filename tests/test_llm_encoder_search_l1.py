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
    def test_o3_pool_identities_budget_and_blind_exclusion(self):
        protocol = l1.read_json(l1.PROTOCOL)
        pool, identities = l1.o3_candidate_pool(protocol)
        self.assertEqual(len(pool), 21)
        self.assertEqual(len(identities), 3)
        for arm in ("L", "R"):
            rows = [r for r in pool if r["arm"] == arm]
            self.assertEqual(len(rows), 10)
            self.assertEqual(len({r["ast_sha256"] for r in rows}), 10)
            self.assertTrue(all("blind" not in r["origin"] for r in rows))
        broken = l1.copy.deepcopy(protocol)
        broken["o3"]["candidate_files"][0]["sha256_lf"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            l1.o3_candidate_pool(broken)

    def test_o3_rerank_grid_uses_validation_only_and_all_multiplier_one_results(self):
        protocol = l1.read_json(l1.PROTOCOL)
        pool, _ = l1.o3_candidate_pool(protocol)
        calls = []
        def fake_fit(farm, source, seed, cfg, device, eval_splits):
            self.assertEqual(eval_splits, ("validation",))
            self.assertEqual((seed, cfg["max_epochs"], cfg["patience"]), (2024, 40, 5))
            calls.append((source, cfg["p_lr_multiplier"]))
            score = 100.0 + (0 if cfg["p_lr_multiplier"] == 1 else 10)
            return {"validation_mae": score, "best_epoch": 39, "epochs_ran": 40,
                    "hit_epoch_limit": True, "best_at_epoch_limit": False, "history": []}
        with tempfile.TemporaryDirectory() as tmp, patch.object(l1, "fit_arm", side_effect=fake_fit), \
                patch.object(l1, "run_llm_worker_batch", side_effect=AssertionError("No LLM calls")):
            result = l1.run_o3_search(None, protocol, Path(tmp), "cpu", pool)
            self.assertEqual((result["rerank_fits"], result["grid_fits"]), (21, 10))
            self.assertEqual(len(calls), 31)
            for arm in ("L_star", "R_star", "P"):
                self.assertEqual(result[arm]["p_lr_multiplier"], 1)
            self.assertTrue((Path(tmp) / "search_selection_freeze.json").exists())
            self.assertFalse((Path(tmp) / "test_evaluation_started.json").exists())

    @unittest.skipUnless(HAS_TORCH, "torch required")
    def test_o3_optimizer_separates_only_p_and_records_epoch_cap(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.P = torch.nn.Parameter(torch.ones(2))
                self.layer = torch.nn.Linear(2, 2)
        model = Model()
        groups = l1.optimizer_groups(model, .001, 10)
        self.assertEqual([g["lr"] for g in groups], [.001, .01])
        self.assertEqual([id(p) for p in groups[1]["params"]], [id(model.P)])
        ids = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        cfg = {**l1.read_json(l1.PROTOCOL)["formal_phase"], "max_epochs": 1, "p_lr_multiplier": .1}
        result = l1.fit_arm(l1.make_synthetic_farm(), pe.EXPERT_PROGRAM, 2024, cfg, "cpu")
        self.assertTrue(result["hit_epoch_limit"])
        self.assertEqual(result["epochs_ran"], 1)
        self.assertEqual(result["history"][0]["next_learning_rates"], [.001, .0001])
        self.assertEqual(set(result["outputs"]), {"validation"})

    def test_prompt_examples_static(self):
        for source in (pe.EXPERT_PROGRAM, l1.BLIND_EXAMPLE):
            pe.validate_source(source)
        protocol = l1.read_json(l1.PROTOCOL)
        for blind in (False, True):
            self.assertIn(pe.PROGRAM_STRUCTURE_TEXT, l1.build_llm_prompt(protocol, blind, 1, 1, [])[1])

    @unittest.skipUnless(HAS_TORCH, "torch required")
    def test_o3_synthetic_pipeline_propagates_multiplier_and_freezes_before_eval(self):
        protocol = l1.read_json(l1.PROTOCOL)
        protocol["o3"].update(max_epochs=1, patience=1, rerank_top_for_grid=1)
        protocol["formal_phase"].update(max_epochs=1, patience=1, seeds=[2024])
        pool = [{"candidate_id": arm, "arm": arm, "source": pe.EXPERT_PROGRAM} for arm in ("L", "R", "P")]
        farms = {"Kelmarsh": l1.make_synthetic_farm(1, 6), "Penmanshiel": l1.make_synthetic_farm(2, 4)}
        original = l1.fit_arm
        original_predict = l1.predict_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            selected = {"P": .1, "R_star": 10., "L_star": .1}
            def guard(*args, **kwargs):
                self.assertFalse((out / "test_evaluation_started.json").exists())
                self.assertEqual(kwargs.get("eval_splits"), ("validation",))
                return original(*args, **kwargs)
            def predict_guard(*args, **kwargs):
                freeze = l1.read_json(out / "formal_selection_freeze.json")
                self.assertEqual(freeze["total_fits"], 8)
                self.assertEqual(freeze["p_lr_multipliers"], {"H": 1., **selected})
                self.assertTrue((out / "test_evaluation_started.json").exists())
                return original_predict(*args, **kwargs)
            with patch.object(l1, "fit_arm", side_effect=guard), \
                    patch.object(l1, "run_llm_worker_batch", side_effect=AssertionError("No LLM")):
                search = l1.run_o3_search(farms["Kelmarsh"], protocol, out, "cpu", pool)
                self.assertEqual((search["rerank_fits"], search["grid_fits"]), (3, 6))
                for arm, mult in selected.items():
                    search[arm]["p_lr_multiplier"] = mult
                with patch.object(l1, "predict_checkpoint", side_effect=predict_guard):
                    formal = l1.run_formal_phase(farms, protocol, out, search, "cpu")
            for farm, result in formal["farms"].items():
                for arm, metrics in result["arms"].items():
                    self.assertEqual(metrics["seeds"][0]["p_lr_multiplier"], selected.get(arm, 1.))
            self.assertEqual(len(formal["artifact_sha256"]), 18)

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
