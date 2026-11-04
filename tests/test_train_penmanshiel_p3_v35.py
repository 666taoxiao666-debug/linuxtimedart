"""P3 isolation, identity, rolling-year and one-shot-evaluation tests (synthetic)."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from scripts import train_penmanshiel_p3_v35 as p3


class P3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.p = p3.prep.protocol()
        cls.data = p3.synthetic_data()

    def test_protocol_hash_lock_rejects_modified_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text('{}', encoding="utf-8")
            with patch.object(p3.prep, "PROTOCOL", path), self.assertRaisesRegex(ValueError, "SHA256"):
                p3.prep.protocol()

    def test_three_folds_and_exact_budget(self):
        self.assertEqual([f["test_year"] for f in self.p["folds"]], [2022, 2023, 2024])
        self.assertEqual(len(self.p["folds"]) * len(self.p["trained_arms"]) * len(self.p["seeds"]), 72)
        for f in self.p["folds"]:
            self.assertLess(max(f["fit_years"]), f["validation_year"])
            self.assertEqual(f["validation_year"] + 1, f["test_year"])

    def test_development_rows_exact_including_missing_and_labels(self):
        old = pd.DataFrame({"turbine_id": [1, 1], "timestamp_hour_end_utc": ["2021-01-01", "2021-02-01"],
                            "label_year": [2021, 2021], "power_kw": [np.nan, -3.2]})
        extra = old.iloc[:1].copy()
        extra.label_year = 2022
        current = pd.concat([old, extra], ignore_index=True)
        self.assertTrue(p3.prep.verify_old_rows(current, old)["passed"])
        for value in (0, -3.199999999):
            bad = current.copy()
            bad.loc[1, "power_kw"] = value
            with self.assertRaises(AssertionError):
                p3.prep.verify_old_rows(bad, old)
        with self.assertRaises(AssertionError):
            p3.prep.verify_old_rows(current.iloc[1:], old)

    def test_window_gate_and_t24_always_descriptive(self):
        for f in self.p["folds"]:
            self.assertEqual(p3.fold_role(f, 4999, self.p), "descriptive")
            self.assertEqual(p3.fold_role(f, 5000, self.p), "confirmation" if f["id"] != "T24" else "descriptive")
        self.assertEqual(p3.fold_role(self.p["folds"][2], 100000, self.p), "descriptive")

    def count_frames(self):
        old = pd.DataFrame({"turbine_id": [1, 1], "timestamp_hour_end_utc": ["2021-01-01", "2021-02-01"],
            "label_year": [2021, 2021], "power_kw": [-3.2, 10.0],
            "n_state_ticks": [6, 0], "n_capacity_ticks": [6, 4]})
        current = old.copy()
        current[["n_state_ticks", "n_capacity_ticks"]] = current[["n_state_ticks", "n_capacity_ticks"]].astype(float)
        return current, old

    def test_count_dtype_integer_equivalence_without_mutation(self):
        current, old = self.count_frames()
        before, reference = current.copy(), old.copy()
        audit = p3.prep.verify_old_rows(current, old)
        self.assertEqual(audit["comparison_only_nullable_count_columns"], ["n_state_ticks", "n_capacity_ticks"])
        pd.testing.assert_frame_equal(current, before, check_exact=True)
        pd.testing.assert_frame_equal(old, reference, check_exact=True)
        # Equal missing positions are allowed; they are not filled.
        current.loc[0, "n_state_ticks"] = np.nan
        old["n_state_ticks"] = old.n_state_ticks.astype("Int64")
        old.loc[0, "n_state_ticks"] = pd.NA
        self.assertTrue(p3.prep.verify_old_rows(current, old)["passed"])

    def test_count_fractional_and_infinite_values_rejected(self):
        for column in ("n_state_ticks", "n_capacity_ticks"):
            for value in (5.5, np.inf, -np.inf):
                current, old = self.count_frames()
                current.loc[0, column] = value
                with self.assertRaises((ValueError, TypeError)):
                    p3.prep.verify_old_rows(current, old)

    def test_count_missing_position_change_rejected(self):
        current, old = self.count_frames()
        current.loc[0, "n_capacity_ticks"] = np.nan
        with self.assertRaises(AssertionError):
            p3.prep.verify_old_rows(current, old)

    def test_count_normalization_does_not_hide_label_or_other_dtype_change(self):
        for column, value in (("power_kw", -3.199999999), ("n_state_ticks", 5.0)):
            current, old = self.count_frames()
            current.loc[0, column] = value
            with self.assertRaises(AssertionError):
                p3.prep.verify_old_rows(current, old)
        current, old = self.count_frames()
        current["label_year"] = current.label_year.astype(float)
        with self.assertRaises(AssertionError):
            p3.prep.verify_old_rows(current, old)

    def failed_preparation_fixture(self, folder):
        root = Path(folder)
        prior, replacement, registry = root / "r1", root / "r2", root / "registry.json"
        p3.save(prior / "audit/failed.json", {"type": "AssertionError", "test_started": False,
            "message": 'column name="n_state_ticks" Attribute "dtype" differs'})
        p3.save(prior / "audit/launch.json", {"pid": 999999999,
            "identity": {"protocol_sha256": p3.PROTOCOL_SHA256}})
        p3.save(registry, {"out": str(prior), "attempts": [{"pid": 999999999, "utc": "prior"}]})
        return prior, replacement, registry

    def test_new_directory_recovery_preserves_history_and_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            prior, replacement, registry = self.failed_preparation_fixture(folder)
            original = registry.read_bytes()
            before = (prior / "audit/failed.json").read_bytes()
            plan = p3.plan_attempt(registry, replacement, restart_from=prior)
            self.assertEqual(len(plan["attempts"]), 1)
            self.assertEqual(plan["attempts"][0]["out"], str(prior))
            self.assertEqual(plan["recovery"]["failed_sha256"], p3.u.file_sha256(prior / "audit/failed.json"))
            self.assertEqual(registry.read_bytes(), original)
            self.assertEqual((prior / "audit/failed.json").read_bytes(), before)
            plan["attempts"].append({"pid": 999999998, "out": str(replacement)})
            p3.save(registry, plan)
            with self.assertRaisesRegex(ValueError, "budget exhausted"):
                p3.plan_attempt(registry, replacement, resume=True)

    def test_new_directory_recovery_rejects_unapproved_routes(self):
        with tempfile.TemporaryDirectory() as folder:
            prior, replacement, registry = self.failed_preparation_fixture(folder)
            for kwargs in ({}, {"restart_from": prior, "resume": True}, {"restart_from": replacement}):
                with self.assertRaises(ValueError):
                    p3.plan_attempt(registry, replacement, **kwargs)
            (prior / "test_evaluation_started.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "restart forbidden"):
                p3.plan_attempt(registry, replacement, restart_from=prior)

    def test_new_directory_recovery_rejects_fits_and_unrelated_failure(self):
        for marker in ("selection_freeze.json", "prepared/prepared_complete.json", "T22/fits/P-upstream/s2024/trained.json"):
            with tempfile.TemporaryDirectory() as folder:
                prior, replacement, registry = self.failed_preparation_fixture(folder)
                p3.save(prior / marker, {})
                with self.assertRaisesRegex(ValueError, "pre-training"):
                    p3.plan_attempt(registry, replacement, restart_from=prior)
        with tempfile.TemporaryDirectory() as folder:
            prior, replacement, registry = self.failed_preparation_fixture(folder)
            p3.save(prior / "audit/failed.json", {"type": "ValueError", "test_started": False, "message": "wrong hash"})
            with self.assertRaisesRegex(ValueError, "Not the reviewed"):
                p3.plan_attempt(registry, replacement, restart_from=prior)

    def test_every_fold_fewshot_time_isolation(self):
        for f in self.p["folds"]:
            pen, kel, variants, _ = p3.build_fold(self.data, f)
            for n in (1, 3):
                lo, cut, hi = p3.fewshot_bounds(f["validation_year"], n)
                self.assertEqual(str(lo)[:4], str(f["validation_year"]))
                fit, val = p3.fewshot_windows(variants[f"Scratch-{n}"], f["validation_year"], n)
                self.assertTrue((fit.dates[fit.window_starts] >= lo).all())
                self.assertTrue((fit.target_times() < cut).all())
                self.assertTrue((val.target_times() >= cut).all())
                self.assertTrue((val.target_times() < hi).all())
                self.assertFalse(np.intersect1d(fit.target_times(), val.target_times()).size)
            self.assertTrue(np.isin(kel.label_year, f["fit_years"] + [f["validation_year"]]).all())
            self.assertTrue((pen.windows("test").target_times() > np.datetime64(f"{f['test_year']}-01-01")).all())

    def test_test_perturbation_cannot_change_training_statistics_or_windows(self):
        fold = self.p["folds"][0]
        pen, kel, variants, _ = p3.build_fold(self.data, fold)
        changed = list(self.data)
        changed[0] = self.data[0].copy()
        mask = changed[0].label_year >= fold["test_year"]
        changed[0].loc[mask, "power_kw"] += 9000
        changed[1] = self.data[1].copy()
        changed[1].loc[changed[1].index >= "2022-01-01"] += 9000
        alt, _, altvariants, _ = p3.build_fold(changed, fold)
        for arm in p3.p2.ARMS:
            np.testing.assert_array_equal(variants[arm].scaler_mean, altvariants[arm].scaler_mean)
            np.testing.assert_array_equal(variants[arm].scaler_scale, altvariants[arm].scaler_scale)
        np.testing.assert_array_equal(pen.windows("train").window_starts, alt.windows("train").window_starts)
        np.testing.assert_array_equal(pen.windows("train").truth_and_anchor()[0], alt.windows("train").truth_and_anchor()[0])

    def test_global_freeze_is_required_before_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with self.assertRaisesRegex(ValueError, "All folds"):
                p3.begin_evaluation(out, {"expected_fits": 72, "fits": []}, 72)
            self.assertFalse((out / "test_evaluation_started.json").exists())

    def test_eval_marker_forbids_retry_even_without_any_result(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            p3.assert_can_resume(out)
            (out / "test_evaluation_started.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "restart forbidden"):
                p3.assert_can_resume(out)

    def test_exclusive_marker_precedes_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            p3.save(out / "selection_freeze.json", {"expected_fits": 72})
            with patch.object(p3, "validate_freeze"):
                p3.begin_evaluation(out, {}, 72)
                with self.assertRaises(FileExistsError):
                    p3.begin_evaluation(out, {}, 72)

    def test_confirmatory_holm_is_three_and_descriptive_never_confirms(self):
        recipe = p3.u.read_json(p3.p2.PROTOCOL)
        recipe["statistics"]["replicates"] = 100
        truth = np.arange(400, dtype=float).reshape(100, 4)
        rng = np.random.default_rng(3)
        predictions = {arm: [truth + rng.normal(0, 2+i, truth.shape) for _ in range(3)]
                      for i, arm in enumerate(p3.p2.ARMS)}
        predictions.update(PersistenceHour=[truth+10], PersistenceLast10=[truth+8])
        issue = pd.date_range("2022-01-01", periods=100, freq="3h").to_numpy()
        result = p3.evaluate_arrays(predictions, truth, issue, np.ones(100, bool), recipe, "confirmation")
        comps = result["comparisons"]
        names = sorted(self.p["confirmation_family"])
        adjusted = p3.holm_adjust([comps[n]["absolute_error"]["newey_west"]["p_value_two_sided"] for n in names])
        np.testing.assert_array_equal([comps[n]["holm_p"] for n in names], adjusted)
        self.assertNotIn("holm_p", comps["Z1"])
        self.assertFalse(result["decisions"]["Z1"]["confirmed"])
        descriptive = p3.evaluate_arrays(predictions, truth, issue, np.zeros(100, bool), recipe, "descriptive")
        self.assertFalse(any(d["confirmed"] for d in descriptive["decisions"].values()))

    def test_completed_fit_reused_and_tampering_rejected(self):
        recipe = p3.u.read_json(p3.p2.PROTOCOL)
        recipe["training"].update(max_epochs=0)
        pen, _, variants, _ = p3.build_fold(self.data, self.p["folds"][0])
        fit, val = pen.windows("train"), pen.windows("validation")
        fit.window_starts, val.window_starts = fit.window_starts[:8], val.window_starts[:8]
        config = p3.p2.configuration("P-last10", recipe)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            args = (config, recipe, p3.p2.view(fit, "P-last10"), p3.p2.view(val, "P-last10"), 2024, "cpu", out, {"synthetic": True})
            original = p3.p2.fit_one(*args)
            digest = p3.u.file_sha256(out / "best.pt")
            reused = p3.p2.fit_one(*args)
            self.assertTrue(reused["reused_from_registry"])
            self.assertEqual(digest, p3.u.file_sha256(out / "best.pt"))
            self.assertEqual(original["best_state_sha256"], reused["best_state_sha256"])
            with (out / "best.pt").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaises(ValueError):
                p3.p2.fit_one(*args)


if __name__ == "__main__":
    unittest.main()
