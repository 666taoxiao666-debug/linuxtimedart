import inspect
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from utils.residual_utility_drift import history_context, decompose_gain, drift_report


class ResidualUtilityDriftTests(unittest.TestCase):
    def test_context_uses_history_only_and_fixed_edge_bins(self):
        wind = np.tile(np.array([2., 3., 5., 10.])[:, None], (1, 12))
        power = np.tile((np.array([.05, .08, .7, .9]) * 1500)[:, None], (1, 12))
        context = history_context(wind, power)
        np.testing.assert_array_equal(context["power_bin"], [0, 1, 3, 4])
        np.testing.assert_array_equal(context["wind_bin"], [0, 1, 2, 3])
        np.testing.assert_array_equal(context["joint_bin"], [0, 5, 14, 19])
        self.assertNotIn("truth", inspect.signature(history_context).parameters)
        with self.assertRaises(ValueError):
            history_context(wind[:, :11], power[:, :11])

    def test_composition_shift_without_conditional_shift(self):
        # Same benefit per stratum but different mix in the later block.
        groups = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        early = np.arange(8) < 4
        result = decompose_gain(np.where(groups == 0, 4., -2.), groups, early, ~early)
        self.assertEqual(result["late_minus_early"], -3.)
        self.assertEqual(result["composition_shift"], -3.)
        self.assertEqual(result["conditional_utility_shift"], 0.)

    def test_conditional_shift_and_missing_support_are_conserved(self):
        groups = np.array([0, 1, 0, 2])
        early = np.array([1, 1, 0, 0], bool)
        result = decompose_gain(np.array([4., 2., -2., 8.]), groups, early, ~early)
        self.assertEqual(result["late_minus_early"], 0.)
        self.assertEqual(result["conditional_utility_shift"], -3.)
        self.assertEqual(result["unshared_support_shift"], 3.)
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            decompose_gain(np.zeros(4), groups, early, early)

    def test_report_only_existing_changed_cells_and_negative_gain(self):
        model = {"states": [{"state": s, "weights": [1.] * 12} for s in (-1, 0, 1)]}
        model["states"][1]["weights"][3] = .75
        ref = np.full((4, 12), 100.)
        anchor = np.zeros_like(ref)
        target = ref.copy()
        arrays = dict(prediction=ref, persistence=anchor, truth=target, trend=np.zeros(4, int))
        context = history_context(np.full((4, 12), 6.), np.full((4, 12), 300.))
        early = np.arange(4) < 2
        report = drift_report(model, arrays, context, early, ~early)
        self.assertEqual(len(report["changed_cells"]), 1)
        self.assertEqual(report["changed_cells"][0]["absolute_error"]["late_gain"], -25.)
        self.assertFalse(report["fitting_performed"])
        self.assertFalse(report["outer_validation_used"])
        json.dumps(report, allow_nan=False)

    def test_reconstruction_matches_saved_windows_without_future_context(self):
        from scripts.audit_sdwpf_residual_drift import reconstruct_context
        from utils.train_error_audit import observable_groups
        thresholds = dict(recent_steps=12, gust_std_min_mps=1., gust_step_min_mps=2.,
            low_power_max_ratio=.08, active_wind_min_mps=5.,
            rated_power_min_ratio=.9, low_wind_max_mps=3.)
        dates = np.datetime64("2023-06-01", "ns") + np.arange(360) * np.timedelta64(10, "m")
        x = np.column_stack((np.full(360, 6.), np.full(360, 300.)))
        # Values after both histories are changed arbitrarily: only the second
        # window's legitimately observed history may use the first forecast.
        x[348:, 1] = 1499.
        starts = np.array([0, 12])
        data = SimpleNamespace(feature_columns=["Wspd", "power"], target="power",
            scaler=SimpleNamespace(mean_=np.zeros(2), scale_=np.ones(2)),
            train_cutoff=dates[100], val_cutoff=dates[200], segments=[], dates=dates,
            turbines=np.ones(360, int), data_x=x)
        events, codes, state = observable_groups(np.full((2, 12), 6.), np.full((2, 12), 300.),
                                                rated_power=1500, thresholds=thresholds)
        arrays = dict(window_starts=starts, target_timestamps=dates[starts[:, None] + 336 + np.arange(12)],
            turbines=np.ones(2, int), events=events, codes=codes, trend=state, persistence=np.full((2, 12), 300.))
        manifest = dict(args=dict(feature_columns=["Wspd", "power"], root_path=".", data_path="unused.csv"),
            datasets=dict(train=dict(scaler=dict(mean=[0., 0.], scale=[1., 1.]))))
        audit = dict(source_plan=dict(outer_data_sha256="data"), fit_cutoff=str(dates[100]),
            selection_end=str(dates[200]), oof_end_exclusive=str(dates[-1]), event_thresholds=thresholds)
        with patch("scripts.audit_sdwpf_residual_drift.train_dataset", return_value=data) as provider, \
                patch("scripts.audit_sdwpf_residual_drift.sha256", return_value="data"), \
                patch("scripts.audit_sdwpf_residual_drift.forward_oof_starts", return_value=starts):
            result = reconstruct_context(manifest, audit, arrays)
            self.assertEqual(provider.call_args.args[0], manifest["args"])
            np.testing.assert_allclose(result["power_ratio"], [.2, .2])
            self.assertTrue((result["history_last_time"] < arrays["target_timestamps"][:, 0]).all())
            arrays["window_starts"] = starts + 1
            with self.assertRaises(AssertionError):
                reconstruct_context(manifest, audit, arrays)


if __name__ == "__main__":
    unittest.main()
