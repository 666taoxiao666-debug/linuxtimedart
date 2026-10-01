import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from utils.wind_wiki_oof import (forward_oof_starts, inner_ratios,
                                 summarize_event_evidence)
from scripts.generate_wind_wiki_oof_evidence import make_plan


class WindWikiOOFTests(unittest.TestCase):
    def test_plan_uses_matched_trend_manifest_and_outer_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"checkpoint")
            (root / "run_manifest.json").write_text(json.dumps({
                "args": {
                    "data": "SDWPF", "sdwpf_split": "rolling_holdout",
                    "prompt_router": "trend", "sdwpf_fold": 0, "seed": 2024,
                    "sdwpf_train_ratio": .7, "sdwpf_val_ratio": .1,
                    "sdwpf_n_folds": 3, "pred_len": 12,
                },
                "data_file": {"sha256": "abc"},
                "datasets": {"train": {"train_cutoff": "2023-06-21T23:40:00"}},
            }), encoding="utf-8")
            run = root / "runs/f0_s2024"
            run.mkdir(parents=True)
            (run / "pipeline.env").write_text(
                f"FINETUNE_CHECKPOINT={checkpoint}\n", encoding="utf-8")
            plan = make_plan(root, 0, 2024)
            self.assertEqual(plan["outer_data_sha256"], "abc")
            self.assertAlmostEqual(plan["inner_train_ratio"], .56)
            self.assertAlmostEqual(plan["inner_val_ratio"], .07)
            wrong_run = root / "runs/f1_s2024"
            wrong_run.mkdir(parents=True)
            (wrong_run / "pipeline.env").write_text(
                f"FINETUNE_CHECKPOINT={checkpoint}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "matching"):
                make_plan(root, 1, 2024)

    def test_inner_split_stays_inside_each_outer_fold(self):
        for fold in range(3):
            train, val = inner_ratios(.7, .1, fold, 3)
            outer = .7 + fold * .1 / 3
            self.assertLess(train + val, outer)
            self.assertGreater(outer - train - val, 0)

    def test_oof_windows_do_not_cross_turbines_or_outer_boundary(self):
        first = np.arange("2023-01-01T00:00", "2023-01-01T01:40",
                          dtype="datetime64[10m]").astype("datetime64[ns]")
        dates = np.concatenate([first, first])
        starts = forward_oof_starts(
            [(0, 10), (10, 20)], dates, seq_len=2, pred_len=2, stride=1,
            evidence_start=first[5], outer_train_cutoff=first[9])
        np.testing.assert_array_equal(starts, [3, 4, 5, 13, 14, 15])
        target_end = dates[starts + 3]
        self.assertTrue(np.all(target_end < first[9]))

    def test_candidate_gain_is_paired_and_horizon_specific(self):
        scenes = [{"id": "gust", "prompt": "gust evidence", "title": "Gust"}]
        base = np.zeros((6, 2), dtype=np.float64)
        target = np.ones((6, 2), dtype=np.float64)
        candidate = np.zeros((6, 2, 1), dtype=np.float64)
        candidate[:, 0, 0] = .5  # helps first step
        candidate[:, 1, 0] = -.5  # harms second step
        available = np.ones_like(candidate, dtype=bool)
        turbines = np.array([1, 1, 2, 2, 3, 3])
        spec = summarize_event_evidence(
            base, candidate, target, available, turbines, scenes,
            fold=0, seed=2024, as_of_step=100)
        self.assertEqual(spec["source_split"], "train_oof")
        self.assertEqual(spec["source_turbines"], [1, 2, 3])
        self.assertEqual(spec["pred_len"], 2)
        for row in spec["candidates"][0]["turbine_evidence"]:
            self.assertAlmostEqual(row["mean_utility"], 0)
            self.assertEqual(row["horizon_mean_utility"], [.5, -.5])

    def test_rejects_shape_mismatch_and_no_supported_windows(self):
        scenes = [{"id": "gust", "prompt": "gust"}]
        base = np.zeros((2, 2))
        target = np.ones((2, 2))
        candidate = np.zeros((2, 2, 1))
        with self.assertRaisesRegex(ValueError, "shapes"):
            summarize_event_evidence(base, candidate, target,
                                     np.ones((2, 1, 1), bool), [1, 1], scenes,
                                     fold=0, seed=1, as_of_step=1)
        with self.assertRaisesRegex(ValueError, "No event"):
            summarize_event_evidence(base, candidate, target,
                                     np.zeros_like(candidate, bool), [1, 1], scenes,
                                     fold=0, seed=1, as_of_step=1)


if __name__ == "__main__":
    unittest.main()
