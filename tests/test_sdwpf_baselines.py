import unittest
from types import SimpleNamespace

import numpy as np

from run import authorize_forecast_report_split
from utils.sdwpf_baselines import (
    _history_stats,
    _window_truth,
    authorize_eval_split,
    power_curve_forecast,
    seasonal_persistence_forecast,
    tree_forecast,
)


class _IdentityScaler:
    mean_ = np.asarray([0.0, 0.0])
    scale_ = np.asarray([1.0, 1.0])


class _BaselineDataset:
    def __init__(self):
        self.feature_columns = ["Wspd", "power"]
        self.target = "power"
        self.scaler = _IdentityScaler()
        self.seq_len = 4
        self.pred_len = 2
        self.window_starts = np.asarray([0], dtype=np.int64)
        self.data_x = np.asarray(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
                [6.0, 60.0],
            ],
            dtype=np.float64,
        )
        self.wspd = self.data_x[:, 0].copy()
        self.data_stamp = np.zeros((len(self.data_x), 5), dtype=np.float64)

    def last_history_wspd(self):
        return self.wspd[self.window_starts + self.seq_len - 1]


class _Curve:
    def predict(self, wind_speed):
        return np.asarray(wind_speed) * 10.0


class _Tree:
    def predict(self, features):
        return np.asarray(features).sum(axis=1)


class BaselineLeakageTests(unittest.TestCase):
    def test_validation_is_the_default_authorized_split(self):
        self.assertEqual(authorize_eval_split("val", {}), "val")

    def test_test_split_requires_explicit_final_confirmation(self):
        with self.assertRaises(PermissionError):
            authorize_eval_split("test", {})
        self.assertEqual(
            authorize_eval_split("test", {"CONFIRM_FINAL_EVAL": "1"}), "test"
        )

    def test_forecast_validation_plot_does_not_open_test(self):
        args = SimpleNamespace(
            data="SDWPF",
            report_split="val",
            downstream_task="forecast",
            model="PromptTimeDART",
        )
        self.assertEqual(authorize_forecast_report_split(args, {}), "val")

    def test_forecast_test_report_requires_explicit_confirmation(self):
        args = SimpleNamespace(data="SDWPF", report_split="test")
        with self.assertRaises(PermissionError):
            authorize_forecast_report_split(args, {})
        self.assertEqual(
            authorize_forecast_report_split(
                args, {"CONFIRM_FINAL_EVAL": "1"}
            ),
            "test",
        )

    def test_truth_starts_immediately_after_history(self):
        dataset = _BaselineDataset()
        np.testing.assert_array_equal(_window_truth(dataset), [[50.0, 60.0]])

    def test_predictions_do_not_change_when_future_scada_changes(self):
        dataset = _BaselineDataset()
        history_features = _history_stats(dataset).copy()
        curve_prediction = power_curve_forecast(dataset, _Curve()).copy()
        tree_prediction = tree_forecast(dataset, _Tree()).copy()
        seasonal_prediction = seasonal_persistence_forecast(dataset, seasonal_lag=4).copy()

        dataset.data_x[dataset.seq_len :, :] = 999999.0
        dataset.wspd[dataset.seq_len :] = 999999.0

        np.testing.assert_array_equal(_history_stats(dataset), history_features)
        np.testing.assert_array_equal(
            power_curve_forecast(dataset, _Curve()), curve_prediction
        )
        np.testing.assert_array_equal(tree_forecast(dataset, _Tree()), tree_prediction)
        np.testing.assert_array_equal(
            seasonal_persistence_forecast(dataset, seasonal_lag=4),
            seasonal_prediction,
        )


if __name__ == "__main__":
    unittest.main()
