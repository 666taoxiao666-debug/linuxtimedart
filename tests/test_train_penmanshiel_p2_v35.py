"""Synthetic-only isolation and frozen-trainer integration tests for P2."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import train_penmanshiel_p2_v35 as p2


class P2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series, cls.up, _ = p2.synthetic_data()
        cls.protocol = p2.u.read_json(p2.PROTOCOL)
        torch.set_num_threads(2)

    def test_calendar_boundaries_are_fixed(self):
        lo, cut, hi = p2.fewshot_bounds(1)
        self.assertEqual(str(cut), "2019-12-25T19:00:00.000000000")
        lo, cut, hi = p2.fewshot_bounds(3)
        self.assertEqual(str(cut), "2019-12-13T14:00:00.000000000")
        with self.assertRaises(ValueError):
            p2.fewshot_bounds(2)

    def test_validation_perturbation_does_not_change_scratch_training(self):
        for n in (1, 3):
            lo, cut, hi = p2.fewshot_bounds(n)
            mask = (self.series.dates >= lo) & (self.series.dates < cut)
            original = p2.adapt_statistics(self.series, self.up, fit_rows=mask)
            changed = copy.copy(self.series)
            changed.base = copy.copy(self.series.base)
            changed.base.raw = self.series.base.raw.copy()
            after_cut = (self.series.dates >= cut) & (self.series.dates < hi)
            changed.base.raw[after_cut] += 1000
            features = self.up.copy()
            features.loc[(features.index >= cut) & (features.index < hi)] += 999
            altered = p2.adapt_statistics(changed, features, fit_rows=mask)
            np.testing.assert_array_equal(original.scaler_mean, altered.scaler_mean)
            np.testing.assert_array_equal(original.scaler_scale, altered.scaler_scale)
            self.assertEqual(original.audit, altered.audit)
            fit, _ = p2.fewshot_windows(original, n)
            fit2, _ = p2.fewshot_windows(altered, n)
            np.testing.assert_array_equal(fit.window_starts, fit2.window_starts)
            np.testing.assert_array_equal(fit.truth_and_anchor()[0], fit2.truth_and_anchor()[0])
            np.testing.assert_array_equal(original.values[mask], altered.values[mask])

    def test_windows_do_not_cross_target_boundary(self):
        for n in (1, 3):
            lo, cut, hi = p2.fewshot_bounds(n)
            fit, val = p2.fewshot_windows(self.series, n)
            self.assertTrue((fit.dates[fit.window_starts] >= lo).all())
            self.assertTrue((fit.target_times() < cut).all())
            self.assertTrue((val.target_times() >= cut).all())
            self.assertTrue((val.target_times() < hi).all())
            self.assertFalse(np.intersect1d(fit.target_times(), val.target_times()).size)
            starts = self.series.starts
            crossing = starts[(self.series.dates[starts + 24] < cut) & (self.series.dates[starts + 27] >= cut)]
            self.assertGreater(len(crossing), 0)
            self.assertFalse(np.intersect1d(crossing, np.r_[fit.window_starts, val.window_starts]).size)

    def test_zero_shot_uses_source_only_no_target_reductions(self):
        state = p2.preprocessing_state(self.series)
        # No statistic-fitting operation is allowed on the external-state path.
        with patch.object(p2.np, "nanmean", side_effect=AssertionError("target mean used")), \
             patch.object(p2.np, "nanstd", side_effect=AssertionError("target std used")), \
             patch.object(p2.np, "mean", side_effect=AssertionError("target fill used")):
            target = p2.adapt_statistics(self.series, self.up, state=state)
        np.testing.assert_array_equal(target.scaler_mean, state["mean"])
        np.testing.assert_array_equal(target.scaler_scale, state["scale"])
        np.testing.assert_array_equal(target.raw[:, -1], self.series.raw[:, -1])
        np.testing.assert_array_equal(target.window_starts if hasattr(target, "window_starts") else target.starts, self.series.starts)
        self.assertEqual(target.audit["statistics_origin"], "source_only")

    def test_recomputed_local_statistics_match_frozen_provider(self):
        rows = np.isin(self.series.label_year, self.series.splits["train"])
        target = p2.adapt_statistics(self.series, self.up, fit_rows=rows)
        np.testing.assert_allclose(target.scaler_mean, self.series.scaler_mean, rtol=0, atol=1e-12)
        np.testing.assert_allclose(target.scaler_scale, self.series.scaler_scale, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(target.values, self.series.values)

    def test_state_channel_mismatch_is_rejected(self):
        state = p2.preprocessing_state(self.series)
        state["channels"] = state["channels"][::-1]
        with self.assertRaises(ValueError):
            p2.adapt_statistics(self.series, self.up, state=state)

    def test_model_parameter_counts_and_power_last(self):
        for arm, expected in (("P-last10", 4262), ("FT-1", 8126)):
            model, count = p2.trainer.build_model(p2.configuration(arm, self.protocol))
            self.assertEqual(count, expected)
            windows = self.series.windows("train")
            v = p2.view(windows, arm)
            np.testing.assert_array_equal(v[0][1][:, -1], windows[0][1][:, -1])
            self.assertEqual(v[0][0].shape[-1], 9 if arm.endswith("last10") else 21)

    def test_decision_requires_all_three_gates(self):
        good = {"absolute_error": {"ci95_low": 1}, "holm_p": .04, "rmse_difference": {"ci95_low_kw": 1}}
        comparisons = {name: copy.deepcopy(good) for name in p2.COMPARISONS}
        self.assertTrue(all(x["supported"] for x in p2.decisions(comparisons).values()))
        comparisons["R"]["absolute_error"]["ci95_low"] = 0
        comparisons["Z1"]["holm_p"] = .051
        comparisons["F1"]["rmse_difference"]["ci95_low_kw"] = 0
        answer = p2.decisions(comparisons)
        self.assertFalse(answer["R"]["supported"])
        self.assertFalse(answer["Z1"]["supported"])
        self.assertFalse(answer["F1"]["supported"])
        self.assertTrue(answer["F3"]["supported"])
        with self.assertRaises(ValueError):
            p2.decisions({"R": good})

    def test_statistics_holm_covers_five_comparisons(self):
        from utils.kelmarsh_backbone_information import holm_adjust
        windows = self.series.windows("development_eval")
        truth, anchor = windows.truth_and_anchor()
        rng = np.random.default_rng(1)
        predictions = {a: [truth + rng.normal(0, 10 + i, truth.shape) for _ in p2.SEEDS] for i, a in enumerate(p2.ARMS)}
        predictions.update(PersistenceHour=[anchor], PersistenceLast10=[windows.last10_anchor()])
        protocol = copy.deepcopy(self.protocol)
        protocol["statistics"]["replicates"] = 100
        result = p2.evaluate(predictions, windows, np.ones(len(windows), bool), protocol)
        comps = result["comparisons"]
        self.assertEqual(set(comps), set(p2.COMPARISONS))
        names = sorted(comps)
        adjusted = holm_adjust([comps[n]["absolute_error"]["newey_west"]["p_value_two_sided"] for n in names])
        np.testing.assert_allclose([comps[n]["holm_p"] for n in names], adjusted)

    def test_warm_start_is_loaded_before_epoch_zero_without_monkeypatch(self):
        config = p2.configuration("FT-1", self.protocol)
        model, _ = p2.trainer.build_model(config)
        warm = p2.trainer.v26run.cpu_state(model)
        fit, val = p2.fewshot_windows(self.series, 1)
        fit.window_starts = fit.window_starts[:8]
        val.window_starts = val.window_starts[:8]
        recipe = copy.deepcopy(self.protocol)
        recipe["training"]["max_epochs"] = 0
        factory = p2.trainer.build_model
        with tempfile.TemporaryDirectory() as folder:
            record = p2.fit_one(config, recipe, p2.view(fit, "FT-1"), p2.view(val, "FT-1"),
                               2024, "cpu", Path(folder), {"synthetic": True}, warm)
            self.assertEqual(record["init_state_sha256"], p2.trainer.v26run.state_digest(warm))
            self.assertEqual(record["best_epoch"], 0)
            self.assertEqual(p2.u.read_json(Path(folder) / "training_history.json")[0]["lr"], 1e-4)
        self.assertIs(p2.trainer.build_model, factory)


if __name__ == "__main__":
    unittest.main()
