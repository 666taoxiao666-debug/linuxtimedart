import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from test_numeric_foundation import fixture
from utils import numeric_rolling as rolling


class NumericRollingTests(unittest.TestCase):
    def test_three_expanding_origins_are_causal_disjoint_and_keep_original_outer(self):
        with tempfile.TemporaryDirectory() as directory:
            series = fixture(directory)
            before = series.subset("validation").window_starts.copy()
            values = series.values.copy()
            origins, full = rolling.rolling_partitions(series)
            self.assertEqual(list(origins), ["o70", "o80", "o90"])
            prior = set()
            for data in origins.values():
                fit, query = data["fit"], data["selection"]
                self.assertLessEqual(fit.target_times().max(), query.target_times()[:, 0].min() - np.timedelta64(1, "h"))
                self.assertTrue((query.target_times() < series.bounds["train_end"]).all())
                clocks = set(query.target_times()[:, 0])
                self.assertFalse(clocks.intersection(prior))
                prior.update(clocks)
                self.assertEqual(fit.flag, "train")
            self.assertTrue((full.target_times() < series.bounds["train_end"]).all())
            np.testing.assert_array_equal(before, series.subset("validation").window_starts)
            np.testing.assert_array_equal(values, series.values)

    def test_pretrain_selection_outcomes_also_must_be_available(self):
        with tempfile.TemporaryDirectory() as directory:
            series = fixture(directory)
            origins, _ = rolling.rolling_partitions(series)
            contract = {key: hashlib.sha256(series.subset(part).window_starts.tobytes()).hexdigest()
                        for key, part in (("fit_window_hash", "fit"), ("selection_window_hash", "selection"))}
            self.assertTrue(rolling.initializer_audit(series, origins["o70"]["selection"], contract)["pretrain_fit_and_selection_both_causal"])
            with self.assertRaisesRegex(ValueError, "unavailable"):
                rolling.initializer_audit(series, series.subset("fit"), contract)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                rolling.initializer_audit(series, origins["o70"]["selection"], dict(contract, selection_window_hash="tamper"))

    def test_global_selector_rejects_late_only_gain_and_does_not_stitch_epochs(self):
        base = {key: dict(mae=100., rmse=150.) for key in ("o70", "o80", "o90")}
        records = {origin: {"TCN": dict(parameters=10, epochs={"1": dict(mae=99., rmse=149.), "2": dict(mae=99., rmse=149.)})}
                   for origin in base}
        records["o70"]["TCN"]["epochs"]["2"] = dict(mae=102., rmse=151.)
        records["o80"]["TCN"]["epochs"]["1"] = dict(mae=102., rmse=151.)
        grid = rolling.candidate_grid(records, 2)
        self.assertEqual(rolling.select_rolling(grid, base)["model"], "Persistence")
        records["o80"]["TCN"]["epochs"]["1"] = dict(mae=98., rmse=148.)
        selected = rolling.select_rolling(rolling.candidate_grid(records, 2), base)
        self.assertEqual(selected["best_epoch"], 1)
        self.assertEqual(selected["model"], "TCN")
        grid[0]["per_origin"]["o70"]["mae"] = float("nan")
        self.assertEqual(rolling.select_rolling(grid[:1], base)["model"], "Persistence")

    def test_candidate_and_origin_mismatch_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Exactly three"):
            rolling.candidate_grid({}, 20)
        base = {key: dict(mae=1., rmse=1.) for key in ("o70", "o80", "o90")}
        with self.assertRaisesRegex(ValueError, "missing an internal origin"):
            rolling.select_rolling([dict(model="X", best_epoch=1, parameters=2, per_origin={})], base)

    def test_fixed_refit_has_no_selection_and_preserves_global_epoch(self):
        from scripts import train_kelmarsh_numeric_rolling as runner
        args = dict(selected=dict(model="ResidualTCN", best_epoch=7), all_train=object(), validation=object(),
                    p={}, spec={"seed": 2024}, output=Path("unused"), identity={}, initial=None, device=None, resources=None)
        with patch.object(runner.f, "neural_fit", return_value=(object(), {})) as fit, patch.object(runner.f, "predict", return_value="prediction"):
            self.assertEqual(runner.refit_prediction(**args), "prediction")
            self.assertIsNone(fit.call_args.args[2])
            self.assertEqual(fit.call_args.kwargs["fixed_epochs"], 7)
        with patch.object(runner.f, "neural_fit", side_effect=AssertionError("Fallback must not train")), \
             patch.object(runner.f.foundation, "observed_anchor", return_value="exact_persistence"):
            self.assertEqual(runner.refit_prediction(**dict(args, selected={"model": "Persistence"})), "exact_persistence")

    def test_existing_stage_reuse_requires_identical_windows_and_hashes(self):
        from scripts import train_kelmarsh_numeric_rolling as runner
        with tempfile.TemporaryDirectory() as directory:
            series = fixture(directory)
            origins, _ = rolling.rolling_partitions(series)
            fit, query = origins["o90"]["fit"], origins["o90"]["selection"]
            path = Path(directory) / "stage"
            _, done = runner.f.ridge_fit("DeltaRidge_a10", fit, query, 10., {"rated_power_kw": 2050.}, path, {"fixture": True})
            self.assertTrue(runner.reusable_stage(path, fit, query, None, "DeltaRidge_a10"))
            self.assertFalse(runner.reusable_stage(path, origins["o70"]["fit"], origins["o70"]["selection"], None, "DeltaRidge_a10"))
            self.assertEqual(runner.stage_record("DeltaRidge_a10", path)["metrics"]["mae"], done["metrics"]["mae"])
            with (path / "model.npz").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                runner.stage_record("DeltaRidge_a10", path)

    def test_protocol_keeps_v11_candidates_and_hyperparameters(self):
        root = Path(__file__).resolve().parents[1]
        old = json.loads((root / "configs/kelmarsh_numeric_foundation_protocol_v11.json").read_text())
        new = json.loads((root / "configs/kelmarsh_numeric_rolling_protocol_v12.json").read_text())
        for key in ("training", "candidates", "ridge_alphas", "seed", "fold"):
            self.assertEqual(old[key], new[key])
        self.assertFalse(new["wiki_enabled"])

    def test_identical_full_refit_reuses_prediction_without_training(self):
        from scripts import train_kelmarsh_numeric_rolling as runner
        with tempfile.TemporaryDirectory() as directory:
            series = fixture(directory)
            _, full = rolling.rolling_partitions(series)
            query = series.subset("validation")
            previous = Path(directory) / "previous"
            p = {"rated_power_kw": 2050.}
            selected = dict(model="DeltaRidge_a10", best_epoch=None, alpha=10.)
            model, _ = runner.f.ridge_fit(selected["model"], full, None, 10., p, previous / "outer_refit", {"fixture": True})
            runner.f.old.write(previous / "selected.json", selected)
            prediction = runner.f.foundation.ridge_predict(model, query)
            truth, anchor = query.truth_and_anchor()
            folder = previous / "reports/SelectedNumeric"
            folder.mkdir(parents=True)
            cached = folder / "raw_predictions.npz"
            np.savez_compressed(cached, prediction=prediction, truth=truth, persistence=anchor, times=query.target_times(),
                                turbines=query.turbines[query.window_starts], window_starts=query.window_starts)
            runner.f.old.write(previous / "completed.json", dict(artifacts={"reports/SelectedNumeric/raw_predictions.npz": runner.f.old.sha(cached)}))
            output = Path(directory) / "new"
            output.mkdir()
            with patch.object(runner.f, "ridge_fit", side_effect=AssertionError("Identical full refit must not repeat")):
                actual = runner.refit_prediction(selected, full, query, p, {"seed": 2024}, output, {}, None, None, None, previous)
            np.testing.assert_array_equal(actual, prediction)
            self.assertTrue((output / "refit_reuse.json").exists())


if __name__ == "__main__":
    unittest.main()
