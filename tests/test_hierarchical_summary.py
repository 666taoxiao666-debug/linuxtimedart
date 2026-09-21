import csv
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_hierarchical_wiki import compare


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


if __name__ == "__main__":
    unittest.main()
