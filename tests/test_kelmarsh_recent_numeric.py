import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_provider.kelmarsh_hourly import KelmarshSeries
from scripts import train_kelmarsh_recent_numeric as task


class RecentNumericTests(unittest.TestCase):
    def series(self, fold=1):
        dates = pd.date_range("2016-05-01", periods=1200, freq="h", tz="UTC")
        parts = []
        for turbine in range(1, 7):
            power = np.arange(len(dates), dtype=float) + turbine - 10.
            parts.append(pd.DataFrame({"timestamp_hour_end_utc": dates, "turbine_id": turbine,
                "wind_speed_mps": 5. + np.arange(len(dates)) / 1000., "ambient_temp_c": 12.,
                "nacelle_position_deg": 90., "pitch_a_deg": 1., "pitch_b_deg": 1., "pitch_c_deg": 1.,
                "rotor_speed_rpm": 10., "power_kw": power}))
        with patch("data_provider.kelmarsh_hourly.pd.read_csv", return_value=pd.concat(parts)):
            return KelmarshSeries("unused", fold=fold, input_len=24, horizon=4, stride=3,
                start="2016-05-01T00:00:00Z", end="2016-06-20T00:00:00Z")

    def test_single_change_protocol_keeps_models_loss_and_information_fixed(self):
        p = task.shared.read(task.PROTOCOL)
        task.validate_protocol(p)
        bad = copy.deepcopy(p)
        bad["training"]["backbone_lr"] *= 2
        with self.assertRaises(ValueError):
            task.validate_protocol(bad)
        self.assertEqual(p["within_outer_train"], {"base_fit": [0., .9], "base_selection": [.9, 1.]})

    def test_recent_fit_selection_and_outer_labels_are_disjoint(self):
        for fold in (0, 1, 2):
            s = self.series(fold)
            raw, normalized = s.raw.copy(), s.values.copy()
            mean, scale = s.scaler.mean_.copy(), s.scaler.scale_.copy()
            old_validation = s.subset("validation")
            data = task.recent_datasets(s)
            fit, select, val = (data[k] for k in ("fit", "selection", "validation"))
            self.assertGreater(len(fit), len(s.subset("fit")))
            self.assertLess(fit.target_times().max(), select.target_times().min())
            self.assertLess(select.target_times().max(), val.target_times().min())
            self.assertLess(fit.target_times().max(), s.bounds["calibration_early_end"])
            self.assertGreaterEqual(select.target_times().min(), s.bounds["calibration_early_end"])
            self.assertLess(select.target_times().max(), s.bounds["train_end"])
            self.assertEqual(fit.train_cutoff, s.bounds["calibration_early_end"])
            np.testing.assert_array_equal(val.window_starts, old_validation.window_starts)
            for x, y in zip((s.raw, s.values, s.scaler.mean_, s.scaler.scale_), (raw, normalized, mean, scale)):
                np.testing.assert_array_equal(x, y)

    def test_reference_cohort_rejects_changed_targets(self):
        s = self.series()
        data = task.recent_datasets(s)
        val = data["validation"]
        truth, anchor = val.truth_and_anchor()
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp)
            task.shared.write(reference / "data_audit_f1.json", s.audit)
            folder = reference / "runs/f1_s2024/Persistence"
            folder.mkdir(parents=True)
            def save(y):
                np.savez_compressed(folder / "raw_predictions.npz", truth=y, persistence=anchor,
                    times=val.target_times(), window_starts=val.window_starts,
                    turbines=val.turbines[val.window_starts])
            save(truth)
            task.assert_reference_cohort(s, data, reference)
            changed = truth.copy()
            changed[0, 0] += 1.
            save(changed)
            with self.assertRaises(AssertionError):
                task.assert_reference_cohort(s, data, reference)

    def test_training_quality_audit_never_summarizes_outer_targets(self):
        s = self.series()
        data = task.recent_datasets(s)
        audit = task.train_audit(s, data)
        # Changing only future raw rows cannot change the train-quality profile.
        mask = s.dates >= s.bounds["train_end"]
        s.raw[mask, -1] = 1e10
        again = task.train_audit(s, data)
        self.assertEqual(audit["train_only_quality"], again["train_only_quality"])
        self.assertFalse(audit["target_values_changed"])
        self.assertFalse(audit["negative_power_clipped"])
        self.assertLess(audit["new_fit_staleness_days"], audit["old_fit_staleness_days"])
        self.assertLess(audit["new_selection_staleness_days"], audit["old_selection_staleness_days"])


if __name__ == "__main__":
    unittest.main()
