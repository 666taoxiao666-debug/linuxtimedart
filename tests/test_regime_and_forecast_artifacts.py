import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run import build_parser, configure_args
from utils.forecast_report import _plot_continuous_forecast
from utils.regime_labels import (
    calibrate_regime_thresholds_from_dataset,
    compute_regime_pseudo_labels_from_series,
    summarize_regime_confusion,
    update_regime_confusion,
)


class _TrendDataset:
    def __init__(self):
        self.seq_len = 12
        histories = []
        starts = []
        cursor = 0
        for slope in np.linspace(-2.0, 2.0, 90):
            starts.append(cursor)
            history = slope * np.linspace(-0.5, 0.5, self.seq_len)
            histories.extend(np.column_stack([np.zeros(self.seq_len), history]))
            cursor += self.seq_len
        self.data_x = np.asarray(histories, dtype=np.float32)
        self.window_starts = np.asarray(starts, dtype=np.int64)

    def __len__(self):
        return len(self.window_starts)


class RegimeCalibrationTests(unittest.TestCase):
    def test_sdwpf_defaults_to_train_calibrated_regimes(self):
        args = configure_args(
            build_parser().parse_args(
                [
                    "--task_name",
                    "pretrain",
                    "--model_id",
                    "SDWPF",
                    "--model",
                    "PromptTimeDART",
                    "--data",
                    "SDWPF",
                    "--no-use_gpu",
                ]
            )
        )
        self.assertEqual(args.regime_label_method, "trend_quantile")

    def test_train_quantiles_produce_supported_classes(self):
        dataset = _TrendDataset()
        audit = calibrate_regime_thresholds_from_dataset(
            dataset, target_index=-1, max_samples=len(dataset)
        )
        self.assertEqual(audit["source_split"], "train")
        self.assertLess(audit["down_thresh"], audit["up_thresh"])
        self.assertEqual(sum(audit["class_counts"]), len(dataset))
        self.assertGreaterEqual(min(audit["class_fractions"]), 0.30)

        rows = dataset.window_starts[:, None] + np.arange(dataset.seq_len)[None, :]
        histories = torch.tensor(dataset.data_x[rows, -1])
        labels = compute_regime_pseudo_labels_from_series(
            histories,
            method="trend_quantile",
            down_thresh=audit["down_thresh"],
            up_thresh=audit["up_thresh"],
        )
        self.assertEqual(torch.bincount(labels, minlength=3).sum().item(), len(dataset))

    def test_confusion_summary_penalizes_a_missing_class(self):
        confusion = np.zeros((3, 3), dtype=np.int64)
        update_regime_confusion(
            confusion,
            predictions=torch.tensor([1, 1, 2, 2]),
            labels=torch.tensor([1, 1, 2, 2]),
        )
        result = summarize_regime_confusion(confusion)
        self.assertEqual(result["regime_accuracy"], 1.0)
        self.assertAlmostEqual(result["regime_macro_f1"], 2.0 / 3.0)
        self.assertEqual(result["regime_counts"], "0;2;2")

    def test_flat_plateau_ties_remain_one_stable_class(self):
        dataset = _TrendDataset()
        for index in range(20, 70):
            start = dataset.window_starts[index]
            dataset.data_x[start : start + dataset.seq_len, -1] = 0.0
        audit = calibrate_regime_thresholds_from_dataset(
            dataset, target_index=-1, max_samples=len(dataset), min_class_fraction=0.05
        )
        self.assertLess(audit["down_thresh"], 0.0)
        self.assertGreater(audit["up_thresh"], 0.0)
        self.assertGreater(audit["class_counts"][0], audit["class_counts"][1])


class ForecastTraceTests(unittest.TestCase):
    def test_trace_png_pdf_csv_and_selection_are_saved(self):
        windows = 13
        horizon = 12
        starts = pd.date_range("2023-08-01", periods=windows, freq="120min")
        metadata = pd.DataFrame(
            {
                "window_index": np.arange(windows),
                "forecast_start": starts,
                "TurbID": np.full(windows, 7),
            }
        )
        truth = np.arange(windows * horizon, dtype=float).reshape(windows, horizon)
        pred = truth + 2.0
        persistence = truth - 3.0
        with tempfile.TemporaryDirectory() as temporary:
            _plot_continuous_forecast(
                pred,
                truth,
                metadata,
                temporary,
                persistence=persistence,
                max_points=150,
            )
            output = Path(temporary)
            self.assertTrue((output / "forecast_trace.png").is_file())
            self.assertTrue((output / "forecast_trace.pdf").is_file())
            trace = pd.read_csv(output / "forecast_trace.csv")
            selection = json.loads(
                (output / "forecast_trace_selection.json").read_text(encoding="utf-8")
            )
        self.assertEqual(len(trace), 150)
        self.assertEqual(selection["TurbID"], 7)
        self.assertIn("selection_rule", selection)


if __name__ == "__main__":
    unittest.main()
