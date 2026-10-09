import copy
import unittest

from utils.accuracy_tuning import choose_candidate, verify_inner_manifest


def history(mae, rmse):
    return [{"epoch": i + 1, "val_mae_kw": m, "val_rmse_kw": r,
             "val_r2_original": 0.5, "val_persistence_mae_kw": 10,
             "val_persistence_rmse_kw": 20} for i, (m, r) in enumerate(zip(mae, rmse))]


class AccuracyTuningTests(unittest.TestCase):
    def test_lower_rmse_cannot_buy_a_mae_regression(self):
        result = choose_candidate({"reference": history([8, 7], [15, 14]),
                                   "tail": history([9, 8], [11, 10])}, ["reference", "tail"], 2)
        self.assertEqual(result["selected"]["candidate_id"], "reference")
        self.assertEqual(result["selected"]["epoch"], 2)
        self.assertFalse(result["inner_improves_both"])

    def test_selects_joint_improvement_and_its_epoch(self):
        result = choose_candidate({"reference": history([8, 7], [15, 14]),
                                   "balanced": history([7.5, 6.5], [14, 13])}, ["reference", "balanced"], 2)
        self.assertEqual(result["selected"]["candidate_id"], "balanced")
        self.assertEqual(result["selected"]["epoch"], 2)
        self.assertTrue(result["inner_improves_both"])
        self.assertFalse(result["outer_validation_used_for_selection"])

    def test_rejects_mismatched_targets_and_incomplete_grid(self):
        rows = {"reference": history([8, 7], [15, 14]), "tail": history([7, 6], [14, 13])}
        rows["tail"][0]["val_persistence_mae_kw"] = 11
        with self.assertRaises(ValueError):
            choose_candidate(rows, ["reference", "tail"], 2)
        with self.assertRaises(ValueError):
            choose_candidate({"reference": history([8], [15])}, ["reference"], 2)

    def test_inner_boundary_and_data_hash_are_required(self):
        plan = {"sdwpf_fold": 1, "seed": 2024, "outer_data_sha256": "data",
                "inner_train_ratio": .58, "inner_val_ratio": .07,
                "outer_train_cutoff": "2023-06-30T00:00:00.000000000"}
        manifest = {"stage": "finetune", "extra": {"status": "complete"},
                    "args": {"data": "SDWPF", "model": "PromptTimeDART", "prompt_router": "trend",
                             "sdwpf_split": "time_ratio", "sdwpf_fold": 1, "seed": 2024,
                             "sdwpf_train_ratio": .58, "sdwpf_val_ratio": .07},
                    "data_file": {"sha256": "data"},
                    "datasets": {"train": {"train_cutoff": "2023-05-10T00:00:00.000000000",
                                            "val_cutoff": "2023-06-01T00:00:00.000000000"}}}
        verify_inner_manifest(manifest, plan, "finetune")
        bad = copy.deepcopy(manifest)
        bad["datasets"]["train"]["val_cutoff"] = plan["outer_train_cutoff"]
        with self.assertRaises(ValueError):
            verify_inner_manifest(bad, plan, "finetune")
        bad = copy.deepcopy(manifest)
        bad["data_file"]["sha256"] = "different"
        with self.assertRaises(ValueError):
            verify_inner_manifest(bad, plan, "finetune")


if __name__ == "__main__":
    unittest.main()
