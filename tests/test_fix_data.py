import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.fix_data import convert_sdwpf


class SDWPFConversionTests(unittest.TestCase):
    def test_conversion_preserves_turbine_identity_and_writes_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.csv"
            output = root / "fixed.csv"
            pd.DataFrame(
                {
                    "TurbID": [2, 1, 1, 2],
                    "Day": [1, 1, 1, 1],
                    "Tmstamp": ["00:00", "00:10", "00:00", "00:10"],
                    "Wspd": [7.0, 6.0, 5.0, 8.0],
                    "Patv": [300.0, 200.0, 100.0, 400.0],
                }
            ).to_csv(source, index=False)

            converted = convert_sdwpf(source, output)
            manifest_path = Path(f"{output}.manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(converted["TurbID"].tolist(), [1, 1, 2, 2])
            self.assertEqual(manifest["input"]["rows"], 4)
            self.assertEqual(manifest["output"]["rows"], 4)
            self.assertEqual(manifest["output"]["turbine_count"], 2)
            self.assertEqual(len(manifest["input"]["sha256"]), 64)
            self.assertEqual(len(manifest["output"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
