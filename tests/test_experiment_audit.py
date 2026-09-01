import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from exp.exp_timedart import Exp_TimeDART
from run import build_parser, configure_args, pretrain_signature
from utils.experiment_audit import (
    dataset_summary,
    split_boundary_checks,
    write_run_manifest,
)
from utils.run_tags import forecast_result_tag
from utils.tools import transfer_weights


class _WindowDataset:
    def __init__(self, dates, starts, seq_len=2, pred_len=2):
        self.dates = np.asarray(dates, dtype="datetime64[m]")
        self.window_starts = np.asarray(starts, dtype=np.int64)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, index):
        return (
            np.zeros((self.seq_len, 2), dtype=np.float32),
            np.zeros((self.pred_len, 2), dtype=np.float32),
        )


class _PersistenceModel(nn.Module):
    def forward(self, batch_x):
        return batch_x[:, -1:, -1:].expand(-1, 2, -1)


class _TransferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sos_token = nn.Parameter(torch.zeros(1, 1, 2))
        self.enc_embedding = nn.Linear(2, 2)
        self.encoder = nn.Linear(2, 2)
        self.head = nn.Linear(2, 1)


class ExperimentAuditTests(unittest.TestCase):
    def test_split_boundary_check_rejects_overlap(self):
        dates = np.arange(
            np.datetime64("2023-01-01T00:00"),
            np.datetime64("2023-01-01T02:00"),
            np.timedelta64(10, "m"),
        )
        train = _WindowDataset(dates, [0, 1])
        val = _WindowDataset(dates, [2, 3])
        with self.assertRaises(AssertionError):
            split_boundary_checks({"train": train, "val": val})

    def test_dataset_summary_rejects_cross_turbine_window(self):
        dates = np.arange(
            np.datetime64("2023-01-01T00:00"),
            np.datetime64("2023-01-01T01:00"),
            np.timedelta64(10, "m"),
        )
        dataset = _WindowDataset(dates, [0])
        dataset.turbines = np.asarray([1, 1, 1, 2, 2, 2])
        with self.assertRaises(AssertionError):
            dataset_summary(dataset, "train")

    def test_pretrain_run_id_is_part_of_checkpoint_signature(self):
        parser = build_parser()
        args = configure_args(
            parser.parse_args(
                [
                    "--task_name",
                    "pretrain",
                    "--model_id",
                    "SDWPF",
                    "--model",
                    "PromptTimeDART",
                    "--data",
                    "SDWPF",
                    "--pretrain_run_id",
                    "audit-a",
                    "--no-use_gpu",
                ]
            )
        )
        self.assertIn("ridaudit-a", pretrain_signature(args))

    def test_manifest_records_hash_shapes_and_split_boundaries(self):
        dates = np.arange(
            np.datetime64("2023-01-01T00:00"),
            np.datetime64("2023-01-01T02:00"),
            np.timedelta64(10, "m"),
        )
        train = _WindowDataset(dates, [0])
        val = _WindowDataset(dates, [4])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data.csv").write_text("date,power\n", encoding="utf-8")
            output = root / "audit"
            path = write_run_manifest(
                output,
                Namespace(root_path=str(root), data_path="data.csv", audit_hash_data=True),
                "test",
                datasets={"train": train, "val": val},
            )
            manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertTrue(manifest["data_file"]["sha256"])
        self.assertEqual(manifest["datasets"]["train"]["sample_shapes"], [[2, 2], [2, 2]])
        self.assertTrue(manifest["split_boundary_checks"]["train_before_val"]["passed"])
        self.assertTrue(manifest["environment"]["command_line"])

    def test_result_tag_uses_checkpoint_training_hyperparameters(self):
        args = Namespace(
            features="MS",
            input_len=336,
            pred_len=24,
            d_model=128,
            e_layers=2,
            patch_len=12,
            stride=12,
            loss="MSE",
            learning_rate=1e-4,
            residual_forecast=True,
            residual_gate_init=-4.0,
            horizon_weight_end=1.0,
            power_weight_alpha=0.0,
            mix_mse_weight=0.8,
            mix_channels=True,
            sdwpf_physics_features=True,
            revin_keep_wind=True,
            sdwpf_split="rolling",
            sdwpf_fold=0,
            seed=999,
            run_id="eval",
            checkpoint_training_args={"loss": "MIXED", "learning_rate": 3e-5, "seed": 2024},
            loaded_finetune_checkpoint_info={"sha256": "abcdef1234567890"},
        )
        tag = forecast_result_tag(args)
        self.assertIn("lossMIXED", tag)
        self.assertIn("lr3e-05", tag)
        self.assertIn("seed2024", tag)
        self.assertIn("ckptabcdef123456", tag)

    def test_validation_persistence_skill_is_zero_for_persistence_model(self):
        experiment = Exp_TimeDART.__new__(Exp_TimeDART)
        experiment.model = _PersistenceModel()
        experiment.device = torch.device("cpu")
        experiment.amp_enabled = False
        experiment.args = Namespace(
            features="MS",
            pred_len=2,
            feature_columns=["Wspd", "power"],
        )
        batch_x = torch.tensor(
            [
                [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]],
                [[0.0, 2.0], [0.0, 3.0], [0.0, 4.0], [0.0, 5.0]],
            ]
        )
        batch_y = torch.tensor(
            [
                [[0.0, 5.0], [0.0, 6.0]],
                [[0.0, 6.0], [0.0, 7.0]],
            ]
        )
        marks = torch.zeros(2, 1, 1)
        loader = [(batch_x, batch_y, marks, marks)]
        result = experiment.valid(loader, nn.MSELoss())
        self.assertAlmostEqual(result["mae"], result["persistence_mae"])
        self.assertAlmostEqual(result["mae_skill_vs_persistence_pct"], 0.0)

    def test_transfer_audit_reports_complete_required_backbone(self):
        source = _TransferModel()
        target = _TransferModel()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "pretrain.pth"
            torch.save({"model_state_dict": source.state_dict()}, checkpoint)
            transferred = transfer_weights(checkpoint, target, strict=True)
        audit = transferred.pretrain_transfer_audit
        self.assertEqual(audit["required_backbone_coverage_pct"], 100.0)
        self.assertGreater(audit["matched_parameter_elements"], 0)
        self.assertLess(audit["target_parameter_coverage_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
