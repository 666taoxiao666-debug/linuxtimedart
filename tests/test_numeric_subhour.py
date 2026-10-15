import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from test_numeric_foundation import fixture
from utils import numeric_foundation as foundation, numeric_rolling as rolling, numeric_subhour as subhour


def tables_for(series):
    tables = {}
    for unit in range(1, 7):
        rows = np.flatnonzero(series.turbines == unit)
        times = series.dates[rows, None] - np.arange(60, 0, -10)[None] * np.timedelta64(1, "m")
        # Zero-mean within-hour pattern; last measurement differs from hourly mean.
        values = series.raw[rows, -1, None] + np.array([-5., -3., -1., 1., 3., 5.])[None]
        tables[unit] = pd.Series(values.ravel(), index=pd.DatetimeIndex(times.ravel()))
    return tables


class NumericSubhourTests(unittest.TestCase):
    def test_last_tick_causal_alignment_and_exact_hourly_reconstruction(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            origins, _ = rolling.rolling_partitions(s)
            ds = origins["o90"]["selection"]
            x, audit = subhour.end_state_delta(ds, tables_for(s))
            np.testing.assert_allclose(x, 5., atol=1e-12)
            self.assertLess(audit["reconstruction_max_error_kw"], 1e-10)
            self.assertEqual(np.datetime64(audit["max_source_tick"]) + np.timedelta64(10, "m"),
                             ds.target_times()[:, 0].max() - np.timedelta64(1, "h"))

    def test_missing_or_wrong_turbine_raw_tick_stops_without_dropping_samples(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            ds = s.subset("validation")
            tables = tables_for(s)
            issue = ds.dates[ds.window_starts[0] + ds.seq_len - 1]
            unit = int(ds.turbines[ds.window_starts[0]])
            tables[unit].loc[pd.Timestamp(issue) - pd.Timedelta(minutes=10)] = np.nan
            with self.assertRaisesRegex(ValueError, "Missing raw tick"):
                subhour.end_state_delta(ds, tables)
            tables = tables_for(s)
            tables[1], tables[2] = tables[2], tables[1]
            with self.assertRaisesRegex(ValueError, "reconstruction differs"):
                subhour.end_state_delta(ds, tables)

    def test_inference_does_not_read_targets_or_any_future_raw_tick(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            ds = s.subset("validation")
            # Isolate one issue to poison every future row without altering another issue's inputs.
            ds.window_starts = ds.window_starts[:1]
            tables = tables_for(s)
            before = subhour.predict([.5, .3], ds, tables)
            issue = ds.dates[ds.window_starts[0] + ds.seq_len - 1]
            for values in tables.values():
                values.loc[values.index >= pd.Timestamp(issue)] = 1e12
            s.raw[ds.window_starts[0] + ds.seq_len:ds.window_starts[0] + ds.seq_len + ds.pred_len, -1] = 1e12
            ds.truth_and_anchor = lambda: (_ for _ in ()).throw(AssertionError("Query labels accessed"))
            np.testing.assert_array_equal(subhour.predict([.5, .3], ds, tables), before)

    def test_bounded_closed_form_fit_and_zero_increment_negative_anchor(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            ds = s.subset("fit")
            tables = tables_for(s)
            x, _ = subhour.end_state_delta(ds, tables)
            truth, anchor = ds.truth_and_anchor()
            expected = np.clip(x @ (truth - anchor) / (x @ x), 0, 1)
            coefficients, _ = subhour.fit_coefficients(ds, tables)
            np.testing.assert_array_equal(coefficients, expected)
            rows = ds.window_starts + ds.seq_len - 1
            s.raw[rows, -1] = -10.5117
            tables = tables_for(s)
            np.testing.assert_array_equal(subhour.predict([0., 0.], ds, tables), foundation.observed_anchor(ds))
            with self.assertRaisesRegex(ValueError, "coefficient"):
                subhour.predict([1.1, 0.], ds, tables)

    def test_loader_reads_only_prefix_and_rejects_duplicate_unit(self):
        with tempfile.TemporaryDirectory() as root:
            archive = Path(root) / "raw.zip"
            header = "# Time zone: UTC\n# Date and time,Power (kW)\n"
            with zipfile.ZipFile(archive, "w") as z:
                for unit in range(1, 7):
                    # Poisoned power at/after stop must not be numerically parsed.
                    z.writestr(f"Turbine_Data_Kelmarsh_{unit}_test.csv", header +
                        "2016-05-01 00:00:00,10\n2016-05-01 00:10:00,-2\n2016-05-01 00:20:00,DO_NOT_PARSE\n")
            tables = subhour.load_power_ticks(archive, start="2016-05-01T00:00:00Z", stop="2016-05-01T00:20:00Z")
            self.assertEqual(set(tables), set(range(1, 7)))
            np.testing.assert_array_equal(tables[1].to_numpy(), [10., -2.])
            with zipfile.ZipFile(archive, "a") as z:
                z.writestr("Turbine_Data_Kelmarsh_1_duplicate.csv", header)
            with self.assertRaisesRegex(ValueError, "six source"):
                subhour.load_power_ticks(archive, start="2016-05-01", stop="2016-05-02")

    def test_loader_preserves_missing_values_and_integrates_with_hourly_alignment(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            tables = tables_for(s)
            archive = Path(root) / "raw.zip"
            with zipfile.ZipFile(archive, "w") as z:
                for unit, values in tables.items():
                    frame = pd.DataFrame({"Date and time": values.index.astype(str), "Power (kW)": values.to_numpy()})
                    text = "# Date and time,Power (kW)\n" + frame.to_csv(index=False, header=False)
                    z.writestr(f"Turbine_Data_Kelmarsh_{unit}_test.csv", text)
            parsed = subhour.load_power_ticks(archive, start="2016-04-30", stop=str(s.bounds["sealed_test_start"]))
            x, _ = subhour.end_state_delta(s.subset("validation"), parsed)
            np.testing.assert_allclose(x, 5., atol=1e-10)
            missing = Path(root) / "missing.zip"
            with zipfile.ZipFile(missing, "w") as z:
                for unit in range(1, 7):
                    z.writestr(f"Turbine_Data_Kelmarsh_{unit}_test.csv",
                        "# Date and time,Power (kW)\n2016-05-01 00:00:00,NaN\n2016-05-01 00:10:00,-2\n")
            parsed = subhour.load_power_ticks(missing, start="2016-05-01", stop="2016-05-02")
            self.assertTrue(np.isnan(parsed[1].iloc[0]))
            self.assertEqual(parsed[1].iloc[1], -2.)

    def test_resume_reuses_actual_coefficients_and_rejects_input_or_file_changes(self):
        from scripts import train_kelmarsh_numeric_subhour as runner
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            fit = s.subset("fit")
            tables = tables_for(s)
            directory = Path(root) / "fit"
            coefficients, done = runner.fit_stage(fit, tables, directory, {"fixture": True})
            with patch.object(subhour, "fit_coefficients", side_effect=AssertionError("Completed stage refit")):
                actual, reused = runner.fit_stage(fit, tables, directory, {"fixture": True})
            np.testing.assert_array_equal(actual, coefficients)
            self.assertEqual(reused, done)
            with self.assertRaisesRegex(ValueError, "fit changed"):
                runner.fit_stage(fit, tables, directory, {"fixture": False})
            with (directory / "model.npz").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "fit changed"):
                runner.fit_stage(fit, tables, directory, {"fixture": True})

    def test_final_fit_has_no_outer_labels_and_fallback_never_fits(self):
        from scripts import train_kelmarsh_numeric_subhour as runner
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            _, all_train = rolling.rolling_partitions(s)
            tables = tables_for(s)
            with patch.object(runner, "fit_stage", side_effect=AssertionError("Fallback fitted")):
                prediction, done = runner.final_prediction({"model": "Persistence"}, s, all_train, tables, Path(root), {}, "unused", "unused")
            np.testing.assert_array_equal(prediction, foundation.observed_anchor(s.subset("validation")))
            self.assertIsNone(done)
            validation = s.subset("validation")
            validation.truth_and_anchor = lambda: (_ for _ in ()).throw(AssertionError("Outer fit label access"))
            original_subset = s.subset
            s.subset = lambda name: validation if name == "validation" else original_subset(name)
            with patch.object(subhour, "load_power_ticks", return_value=tables) as load:
                prediction, done = runner.final_prediction({"model": runner.NAME}, s, all_train, tables, Path(root) / "final", {}, "raw", "2016-05-01")
            self.assertTrue(np.isfinite(prediction).all())
            self.assertLess(np.datetime64(done["contract"]["fit"]["last_target"]), s.bounds["train_end"])
            self.assertLess(np.datetime64(load.call_args.kwargs["stop"]), s.bounds["sealed_test_start"])

    def test_protocol_declares_information_change_and_single_finite_candidate(self):
        from scripts import train_kelmarsh_numeric_subhour as runner
        root = Path(__file__).resolve().parents[1]
        p = json.loads((root / "configs/kelmarsh_numeric_subhour_protocol_v15.json").read_text())
        self.assertEqual(p["protocol_id"], runner.PROTOCOL_ID)
        self.assertEqual(p["candidates"], [runner.NAME])
        self.assertEqual((p["fold"], p["seed"], p["wiki_enabled"]), (1, 2024, False))
        self.assertIn("ADD one causal", p["data_contract"])
        self.assertIn("NOT architecture/Wiki", p["comparisons"])


if __name__ == "__main__":
    unittest.main()
