import tempfile
import unittest
from pathlib import Path

from scripts.summarize_sdwpf_cv import parse_summary, validate_matrix, write_reports


def _epoch(epoch, mae, persistence, mae_skill, rmse_skill):
    return (
        f"Epoch: {epoch}, Steps: 10, Time: 1.00s | Train Loss: 0.1 "
        f"Vali Loss: 0.2 Vali MSE: 0.3 Vali MAE: 0.4 "
        f"Val MAE(kW): {mae} Persist MAE(kW): {persistence} "
        f"MAE Skill: {mae_skill:+.2f}% RMSE Skill: {rmse_skill:+.2f}% "
        f"Gate: 0.1 GradNorm: 0.2 LR(backbone/new): 1e-6/1e-5 "
        f"Select(original_mae): {mae}"
    )


class SDWPFCVSummaryTests(unittest.TestCase):
    def test_parser_ignores_pretrain_and_can_select_epoch_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "summary.txt"
            source.write_text(
                "\n".join(
                    [
                        "===== CV fold=0 seed=2024 =====",
                        "Epoch: 1/20, Time: 1.0, Train Total/Diff/CE: 1/1/1",
                        _epoch(0, 100.0, 100.0, 0.0, 0.0),
                        _epoch(1, 101.0, 100.0, -1.0, -1.0),
                        "===== CV fold=0 seed=2025 =====",
                        _epoch(0, 100.0, 100.0, 0.0, 0.0),
                        _epoch(1, 98.0, 100.0, 2.0, 3.0),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            runs = parse_summary(source)
            validate_matrix(runs, [0], [2024, 2025])
            self.assertEqual([run.best_epoch for run in runs], [0, 1])
            self.assertEqual([run.mae_kw for run in runs], [100.0, 98.0])

            csv_path, text_path = write_reports(runs, root)
            self.assertTrue(csv_path.exists())
            report = text_path.read_text(encoding="utf-8")
            self.assertIn("RUNS=2", report)
            self.assertIn("MACRO_ABSOLUTE_GAIN_KW=1.000000", report)
            self.assertIn("ALL_BEST_CHECKPOINTS_BEAT_PERSISTENCE=0", report)
            self.assertIn("EPOCH_ZERO_SELECTED_RUNS=1", report)
            self.assertIn("BEST_EPOCH_COUNTS=0:1,1:1", report)

    def test_incomplete_matrix_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "summary.txt"
            source.write_text(
                "===== CV fold=0 seed=2024 =====\n"
                + _epoch(1, 98.0, 100.0, 2.0, 3.0)
                + "\n",
                encoding="utf-8",
            )
            runs = parse_summary(source)
            with self.assertRaisesRegex(ValueError, "incomplete CV matrix"):
                validate_matrix(runs, [0, 1], [2024])


if __name__ == "__main__":
    unittest.main()
