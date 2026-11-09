import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
from scripts import tune_t1 as t


def synthetic_72h():
    frame = t.trainer.synthetic_prepared_frame(years=(2019, 2020, 2021), days=10)
    labels = pd.DatetimeIndex(
        pd.to_datetime(frame.timestamp_hour_end_utc, utc=True).dt.tz_localize(None).unique()
    ).sort_values()
    rng = np.random.default_rng(2024)
    up = pd.DataFrame(rng.normal(size=(len(labels), 11)), index=labels, columns=t.u.UPSTREAM_FEATURES)
    up.iloc[::17, 0] = np.nan
    prep = {"schema_version": 2, "curve_years": [2019], "prepared": {"sha256": "synthetic"}}
    splits = {"train": [2019], "validation": [2020], "test": [2021]}
    series_dev = t.u.KelmarshUpstreamSeries(
        frame[frame.label_year <= 2020],
        up,
        splits={"train": [2019], "validation": [2020]},
        input_len=72,
        preparation=prep,
    )
    series_all = t.u.KelmarshUpstreamSeries(frame, up, splits=splits, input_len=72, preparation=prep)
    return series_dev, series_all


def smoke(out, device):
    protocol = t.u.read_json(t.PROTOCOL)
    _, series = synthetic_72h()
    windows = {k: series.windows(k) for k in ("train", "validation", "test")}
    summaries = {}
    for family in protocol["models"]:
        for length in (24, 48, 72):
            params = {
                "seq_len": length,
                "batch_size": 32,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "loss_mae_weight": 0.8,
                "dropout": 0.1,
            }
            if family == "TSMixerBasicSmall":
                params.update(blocks=2, ff_dim=32)
            else:
                params.update(d_model=64, e_layers=1)
            cfg = t.config_for(family, params)
            recipe = t.recipe_for(params, short=True)
            recipe["training"].update(max_epochs=1, patience=1)
            folder = out / "smoke_fits" / f"{family}_L{length}"
            contract = {"stage": "M2-T1-smoke", "family": family, "seq_len": length}
            rec = t.fit(cfg, recipe, windows, 2024, device, folder, contract)
            t.trainer.verify_trained(folder, rec["contract"])
            summaries[f"{family}_L{length}"] = rec["validation_best"]["mae"]
    result = {"status": "PASSED", "smoke": True, "validation_mae": summaries}
    t.save(out / "result.json", result)
    return result


class T1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.protocol = t.u.read_json(t.PROTOCOL)
        cls.series_dev, cls.series_all = synthetic_72h()

    def test_common_72h_view_alignment_and_suffix_invariance(self):
        w72 = self.series_all.windows("validation")
        sig72 = t.signature(w72)
        for length in (24, 48, 72):
            v = t.common_view(w72, length)
            self.assertEqual(len(v), len(w72))
            self.assertEqual(t.signature(v), sig72)
            x72, y72 = t.e1.view(w72, "upstream")[0]
            xv, yv = t.e1.view(v, "upstream")[0]
            self.assertEqual(xv.shape, (length, 21))
            np.testing.assert_array_equal(xv, x72[-length:])
            np.testing.assert_array_equal(yv, y72)
        with self.assertRaises(ValueError):
            t.common_view(w72, 36)
        np.testing.assert_array_equal(self.series_dev.scaler_mean, self.series_all.scaler_mean)
        np.testing.assert_array_equal(self.series_dev.scaler_scale, self.series_all.scaler_scale)
        dev_rows = np.isin(self.series_all.label_year, [2019, 2020])
        np.testing.assert_array_equal(self.series_dev.values, self.series_all.values[dev_rows])

    def test_model_configs_and_norm_disabled(self):
        for family in self.protocol["models"]:
            for length in (24, 48, 72):
                cfg = t.config_for(family, {"seq_len": length})
                model, count = t.build_model(cfg)
                self.assertEqual(count, cfg["expected_parameters"])
                x = torch.randn(2, length, 21)
                y = model(x)
                self.assertEqual(y.shape[:2], (2, 4))
                if family != "TSMixerBasicSmall":
                    self.assertFalse(cfg["args"]["use_norm"])
                    self.assertEqual(cfg["args"]["d_ff"], 4 * cfg["args"]["d_model"])

    def test_trial_selection_ties_and_failures(self):
        trials = [
            {"number": 0, "status": "FAIL", "value": None},
            {"number": 1, "status": "COMPLETE", "value": 10.0},
            {"number": 2, "status": "COMPLETE", "value": 9.5},
            {"number": 3, "status": "COMPLETE", "value": 9.5},
        ]
        self.assertEqual(t.select_trial(trials)["number"], 2)
        with self.assertRaises(ValueError):
            t.select_trial([{"number": 0, "status": "FAIL", "value": None}])
        self.assertTrue(t.numerical(RuntimeError("Nonfinite training loss at epoch 1")))
        self.assertFalse(t.numerical(RuntimeError("Unexpected disk error")))

    def test_acceptance_rule_requires_both_folds_and_rmse_guard(self):
        reports = {
            "Kelmarsh": {
                "TSMixerBasicSmall_existing": {"ensemble": {"mae": 100.0, "rmse": 150.0}},
                "TSMixerBasicSmall_tuned": {"ensemble": {"mae": 97.5, "rmse": 150.0}},
            },
            "Penmanshiel": {
                "TSMixerBasicSmall_existing": {"ensemble": {"mae": 200.0, "rmse": 300.0}},
                "TSMixerBasicSmall_tuned": {"ensemble": {"mae": 195.0, "rmse": 299.0}},
            },
        }
        self.assertTrue(t.accept(reports))
        reports["Penmanshiel"]["TSMixerBasicSmall_tuned"]["ensemble"]["rmse"] = 300.1
        self.assertFalse(t.accept(reports))
        reports["Penmanshiel"]["TSMixerBasicSmall_tuned"]["ensemble"]["rmse"] = 299.0
        reports["Kelmarsh"]["TSMixerBasicSmall_tuned"]["ensemble"]["mae"] = 98.5
        self.assertFalse(t.accept(reports))


if __name__ == "__main__":
    unittest.main()
