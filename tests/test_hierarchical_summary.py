import csv
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_hierarchical_wiki import compare, fold_summary


class PairedSummaryTests(unittest.TestCase):
    def write(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "fold", "seed", "best_epoch", "val_mae_kw", "persistence_mae_kw",
            ])
            writer.writeheader()
            writer.writerows(rows)

    def test_pairs_by_fold_seed_not_row_order_or_overall_macro(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, trend = Path(directory) / "wiki.csv", Path(directory) / "trend.csv"
            self.write(wiki, [dict(fold=1, seed=2024, best_epoch=4, val_mae_kw=98, persistence_mae_kw=110)])
            self.write(trend, [
                dict(fold=0, seed=2024, best_epoch=2, val_mae_kw=200, persistence_mae_kw=210),
                dict(fold=1, seed=2024, best_epoch=1, val_mae_kw=100, persistence_mae_kw=110),
            ])
            rows = compare(wiki, trend)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["gain_kw"], 2.)
            self.assertEqual(rows[0]["gain_pct"], 2.)
            self.write(trend, [dict(fold=1, seed=2024, best_epoch=1, val_mae_kw=100, persistence_mae_kw=120)])
            with self.assertRaisesRegex(ValueError, "Persistence mismatch"):
                compare(wiki, trend)

    def test_fallback_is_zero_gain_and_missing_pair_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            wiki, trend = Path(directory) / "wiki.csv", Path(directory) / "trend.csv"
            row = dict(fold=0, seed=2024, best_epoch=0, val_mae_kw=133.636, persistence_mae_kw=135.662)
            self.write(wiki, [row])
            self.write(trend, [row])
            self.assertEqual(compare(wiki, trend)[0]["gain_kw"], 0.)
            self.write(trend, [{**row, "seed": 2025}])
            with self.assertRaisesRegex(ValueError, "No matching"):
                compare(wiki, trend)

    def test_fold_summary_exposes_epoch_zero_and_trained_regression(self):
        rows = [
            dict(fold=0, seed=2025, best_epoch=0, trend_mae_kw=134., wiki_mae_kw=134.,
                 gain_kw=0., trained_epochs_reported=7, trained_best_mae_kw=135.),
            dict(fold=0, seed=2026, best_epoch=0, trend_mae_kw=133., wiki_mae_kw=133.,
                 gain_kw=0., trained_epochs_reported=7, trained_best_mae_kw=134.),
            dict(fold=1, seed=2026, best_epoch=5, trend_mae_kw=128., wiki_mae_kw=127.6,
                 gain_kw=.4, trained_epochs_reported=7, trained_best_mae_kw=127.6),
        ]
        lines = fold_summary(rows)
        self.assertIn('FOLD_0_WIKI_GAIN_VS_TREND_KW=0.000000', lines)
        self.assertIn('FOLD_0_EPOCH_ZERO_SELECTED=2', lines)
        self.assertIn('FOLD_0_TRAINED_BEST_GAIN_VS_TREND_KW=-1.000000', lines)
        self.assertIn('FOLD_1_RUNS_BEATING_TREND=1', lines)


if __name__ == "__main__":
    unittest.main()
