import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from models.DLinear import Model as DLinear
from models.PatchTST import Model as PatchTST
from utils.paired_forecast_statistics import paired_forecast_statistics
from utils.wiki_diagnostic import summarize_intervention_policy


def _args():
    return SimpleNamespace(
        task_name="finetune",
        downstream_task="forecast",
        seq_len=32,
        pred_len=4,
        enc_in=3,
        d_model=8,
        d_ff=16,
        n_heads=2,
        e_layers=1,
        dropout=0.0,
        factor=1,
        activation="gelu",
        moving_avg=5,
        individual=0,
    )


class StrongBaselineContractTests(unittest.TestCase):
    def test_patchtst_accepts_repository_finetune_forecast_contract(self):
        model = PatchTST(_args(), patch_len=8, stride=4)
        # Exp_TimeDART intentionally calls forecast models with batch_x only.
        output = model(torch.randn(2, 32, 3))
        self.assertEqual(tuple(output.shape), (2, 4, 3))

    def test_dlinear_accepts_repository_finetune_forecast_contract(self):
        model = DLinear(_args())
        output = model(torch.randn(2, 32, 3))
        self.assertEqual(tuple(output.shape), (2, 4, 3))


class PairedInferenceTests(unittest.TestCase):
    def test_time_block_inference_reports_positive_known_gain(self):
        truth = np.arange(96, dtype=float).reshape(24, 4)
        reference = truth + 2.0
        candidate = truth + 1.0
        metadata = pd.DataFrame(
            {"forecast_start": pd.date_range("2024-01-01", periods=24, freq="2h")}
        )
        report = paired_forecast_statistics(
            candidate,
            reference,
            truth,
            metadata,
            block_length=4,
            replicates=200,
            seed=7,
        )
        self.assertAlmostEqual(report["paired_gain_kw"], 1.0)
        self.assertGreater(report["time_block_bootstrap"]["ci95_low_kw"], 0.0)
        self.assertLess(report["dm_newey_west"]["p_value_two_sided_normal"], 0.05)

    def test_intervention_summary_reports_coverage_abstention_and_harm(self):
        truth = np.zeros((4, 2))
        off = np.ones((4, 2))
        on = np.asarray([[0.0, 0.0], [2.0, 2.0], [1.0, 1.0], [1.0, 1.0]])
        supported = np.asarray([[1, 0], [1, 0], [0, 0], [0, 0]], dtype=bool)
        active = np.asarray([[1, 0], [1, 0], [0, 0], [0, 0]], dtype=bool)
        report = summarize_intervention_policy(on, off, truth, supported, active)
        self.assertEqual(report["intervention_coverage_pct"], 50.0)
        self.assertEqual(report["abstention_pct"], 50.0)
        self.assertEqual(report["selected_harm_window_pct"], 50.0)
        self.assertAlmostEqual(report["selected_gain_kw"], 0.0)


if __name__ == "__main__":
    unittest.main()
