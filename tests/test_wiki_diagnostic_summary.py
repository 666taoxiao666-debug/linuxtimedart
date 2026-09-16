import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_wiki_diagnostic import summarize


class SummaryTests(unittest.TestCase):
    def test_existing_reports_need_no_weights_or_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "runs/f2_s2024/artifacts"
            artifact.mkdir(parents=True)
            payload = {"evaluation_split": "val", "fold": 2, "seed": 2024,
                       "windows": 100, "actual_false_intervention_on_null": 0,
                       "null_max_prediction_delta_kw": 0,
                       "all": {"mae_on_kw": 75.8, "mae_off_kw": 75.82, "mae_gain_kw": .02}}
            (artifact / "diagnostic_summary.json").write_text(json.dumps(payload))
            rows, text = summarize(directory)
            self.assertEqual(len(rows), 1)
            self.assertIn("gain=+0.020000", text)
            self.assertIn("MISSING:", text)
            payload["evaluation_split"] = "test"
            (artifact / "diagnostic_summary.json").write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                summarize(directory)

    def test_empty_directory_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                summarize(directory)


if __name__ == "__main__":
    unittest.main()
