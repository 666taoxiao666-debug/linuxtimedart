import copy
import unittest
from pathlib import Path
import json

import numpy as np

from utils.train_error_audit import observable_groups, error_report, verify_audit_sources


class TrainErrorAuditTests(unittest.TestCase):
    def setUp(self):
        self.thresholds = json.loads((Path(__file__).resolve().parents[1] /
            "configs/wind_event_factor_wiki.json").read_text(encoding="utf-8"))["rule_defaults"]

    def test_observable_regimes_overlap_but_codes_partition(self):
        wind = np.array([[1] * 12, [6, 9] * 6, [6] * 12, [5] * 12], float)
        power = np.array([[0] * 12, [0] * 12, [1450] * 12, [500] * 12], float)
        masks, codes, trend = observable_groups(wind, power, rated_power=1500, thresholds=self.thresholds)
        np.testing.assert_array_equal(codes, [8, 3, 4, 0])
        self.assertTrue(masks[1, 0] and masks[1, 1])
        np.testing.assert_array_equal(trend, [0, 0, 0, 0])

    def test_history_trend_does_not_use_labels_and_thresholds_are_strict(self):
        wind = np.full((3, 12), 3.)
        power = np.array([[0] * 6 + [100] * 6, [100] * 6 + [0] * 6, [0] * 12])
        masks, _, trend = observable_groups(wind, power, rated_power=1500, thresholds=self.thresholds)
        self.assertFalse(masks[:, 3].any())  # equality is not "below"
        np.testing.assert_array_equal(trend, [1, -1, 0])

    def test_partition_shares_sum_and_metrics_are_paired(self):
        y = np.array([[0, 100], [300, 400], [600, 800]], float)
        ref, candidate, persistence = y + 10, y + 20, y + 40
        events = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]], bool)
        report = error_report({"reference": ref, "physics_norm_v6": candidate}, y, persistence,
            [1, 1, 2], events, [3, 0, 4], [-1, 0, 1], rated_power=1500)
        for axis in ("horizon", "event_combinations", "turbines", "historical_trend"):
            for share in ("absolute_error_share_pct", "squared_error_share_pct"):
                self.assertAlmostEqual(sum(r["models"]["reference"][share] for r in report[axis]), 100.)
        overall = report["overall"]["models"]
        self.assertEqual(overall["reference"]["mae_skill_pct"], 75.)
        self.assertEqual(overall["physics_norm_v6"]["gain_vs_reference_kw"], -10.)
        self.assertEqual(overall["physics_norm_v6"]["harm_vs_reference_point_pct"], 100.)
        self.assertEqual(sum(r["point_count"] for r in report["event_combinations"]), y.size)

    def test_zero_variance_and_perfect_persistence_are_not_fake_accuracy(self):
        y = np.ones((2, 2))
        report = error_report({"reference": y, "physics_norm_v6": y}, y, y, [1, 2],
            np.zeros((2, 4), bool), [0, 0], [0, 0], rated_power=1500)
        model = report["overall"]["models"]["reference"]
        self.assertIsNone(model["r2"])
        self.assertIsNone(model["mae_skill_pct"])
        self.assertEqual(model["squared_error_share_pct"], 0.)
        json.dumps(report, allow_nan=False)

    def test_nonfinite_or_misaligned_inputs_rejected(self):
        with self.assertRaises(ValueError):
            observable_groups([[np.nan] * 12], [[0] * 12], rated_power=1500, thresholds=self.thresholds)
        y = np.ones((2, 2))
        with self.assertRaises(ValueError):
            error_report({"reference": y, "physics_norm_v6": y}, y, y, [1, 2],
                np.zeros((2, 4), bool), [1, 0], [0, 0], rated_power=1500)

    def test_checkpoint_provenance_and_oof_boundaries(self):
        plan = dict(sdwpf_fold=1, seed=2024, pred_len=12, inner_train_ratio=.58,
                    inner_val_ratio=.07, outer_data_sha256="data", outer_train_cutoff="2023-06-30")
        manifest = {"stage": "finetune", "extra": {"status": "complete", "best_epoch": 8},
            "data_file": {"sha256": "data"}, "args": dict(data="SDWPF", model="PromptTimeDART",
                prompt_router="trend", sdwpf_split="time_ratio", sdwpf_fold=1, seed=2024,
                sdwpf_train_ratio=.58, sdwpf_val_ratio=.07, seq_len=336, pred_len=12, features="MS", rated_power=1500.),
            "datasets": {"train": {"train_cutoff": "2023-05-25", "val_cutoff": "2023-06-12",
                "scaler": {"mean": [0, 0], "scale": [1, 1]}}}}
        v6 = copy.deepcopy(manifest)
        v6["args"]["consistent_physics_norm"] = True
        self.assertEqual(verify_audit_sources([manifest, v6], plan)[1], "2023-06-12")
        leaked = copy.deepcopy(v6)
        leaked["datasets"]["train"]["val_cutoff"] = "2023-07-01"
        with self.assertRaises(ValueError):
            verify_audit_sources([manifest, leaked], plan)
        mismatched = copy.deepcopy(v6)
        mismatched["datasets"]["train"]["scaler"]["scale"][0] = 2
        with self.assertRaises(AssertionError):
            verify_audit_sources([manifest, mismatched], plan)


if __name__ == "__main__":
    unittest.main()
