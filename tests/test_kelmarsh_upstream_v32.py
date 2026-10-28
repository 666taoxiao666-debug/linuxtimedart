"""v32 tests: protocol freeze/design (2024 confirmation of the frozen v31 design), preparation identity with explicit
turbine ids, the protocol-driven normalized preparer, year-aware report; with torch: the subclassed controller end to
end on synthetic data (test access only after S1). Everything else is covered by the frozen v31 tests."""
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tests.test_kelmarsh_upstream_v31 import fake_run
from utils import kelmarsh_upstream_v32 as u

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/kelmarsh_upstream_protocol_v32.json"
V31_PROTOCOL = ROOT / "configs/kelmarsh_upstream_protocol_v31.json"
SCRIPT = ROOT / "scripts/train_kelmarsh_upstream_v32.py"
try:
    import torch
    import sklearn  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def protocol():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


class ProtocolTests(unittest.TestCase):
    def test_frozen_hash_and_unchanged_v31_design(self):
        constant = re.search(r'PROTOCOL_SHA256 = "([0-9a-f]{64})"', SCRIPT.read_text(encoding="utf-8")).group(1)
        self.assertEqual(u.text_sha256_lf(PROTOCOL), constant)
        p, v31 = protocol(), json.loads(V31_PROTOCOL.read_text(encoding="utf-8"))
        self.assertTrue(u.validate_protocol(p))
        self.assertEqual(p["data"]["splits"], {"train": list(range(2016, 2023)), "validation": [2023], "test": [2024]})
        self.assertEqual(p["data"]["sealed_years"], [])
        for key in ("arms", "families", "seeds", "training", "budget", "comparisons", "statistics", "diagnostics", "cohorts"):
            if key == "training":
                self.assertEqual({k: v for k, v in p[key].items() if k != "selection_metric"},
                                 {k: v for k, v in v31[key].items() if k != "selection_metric"})
            else:
                self.assertEqual(p[key], v31[key], key)
        for key in ("non_inferiority_margin_kw", "upstream_information_supported", "balanced_loss_supported", "beats_naive"):
            self.assertEqual(p["rules"][key], v31["rules"][key])
        up, up31 = p["data"]["upstream"], v31["data"]["upstream"]
        for key in ("stations", "features", "lag_hours", "download_manifest", "download_manifest_sha256_lf", "missing_policy"):
            self.assertEqual(up[key], up31[key], key)
        self.assertEqual([r["years"] for r in up["development_roots"]], [list(range(2016, 2023)), [2023]])
        self.assertEqual(up["test_years"], [2024])
        self.assertEqual(p["data"]["v31_registered_preparation"]["csv_sha256"],
                         "dc2365fbda0377f973b02300655067ae2ddbdc1e956742992eee3c0631a80564")

    def test_departures_are_refused(self):
        def mutated(path, value):
            p = protocol()
            target = p
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            return p
        for path, value in ((("data", "splits", "test"), [2023]), (("data", "sealed_years"), [2025]),
                            (("data", "v32_preparation", "curve_years"), [2016, 2017]),
                            (("data", "v32_preparation", "duplicate_normalization"), "coalesce"),
                            (("data", "upstream", "development_roots"), [{"root": "x", "years": list(range(2016, 2023))}]),
                            (("data", "upstream", "lag_hours"), 2), (("rules", "non_inferiority_margin_kw"), 3.0),
                            (("arms", "TSMixer_upstream_all_balanced", "training_overrides"), {"loss_mae_weight": .3, "loss_mse_weight": .7}),
                            (("budget", "fits"), 12), (("data", "v31_registered_preparation", "years"), list(range(2016, 2025)))):
            with self.assertRaises(ValueError, msg=str(path)):
                u.validate_protocol(mutated(path, value))


def good_meta(p, sha="abc"):
    units = [{"turbine_id": i, "cells_filled": 0, "conflicting_nonmissing_values": 0, "timestamps_preserved": True,
              "nonmissing_values_preserved": True, "selected_existing_whole_rows": True} for i in range(1, 7)]
    return {"schema_version": 2, "years": list(range(2016, 2025)), "curve_years": [2016, 2017, 2018, 2019],
            "prepared": {"sha256": sha}, "labels_modified": False, "clipping": False, "interpolation": False,
            "source_sha256": {s: u.file_sha256(ROOT / s) for s in u.PREPARER_SOURCES},
            "duplicate_normalization": {"version": "existing_dominating_row_v1", "turbine_id_source": "preparer unit loop",
                                        "units": units}}


class PreparationIdentityTests(unittest.TestCase):
    def test_explicit_ids_and_preservation_evidence(self):
        p = protocol()
        self.assertTrue(u.verify_preparation(good_meta(p), p, "abc"))
        bad = []
        meta = good_meta(p); meta["duplicate_normalization"]["units"][0]["turbine_id"] = 2; bad.append(meta)
        meta = good_meta(p); meta["duplicate_normalization"]["turbine_id_source"] = "call order"; bad.append(meta)
        meta = good_meta(p); meta["duplicate_normalization"]["units"][3]["cells_filled"] = 1; bad.append(meta)
        meta = good_meta(p); meta["years"] = list(range(2016, 2024)); bad.append(meta)
        meta = good_meta(p); meta["source_sha256"]["scripts/prepare_kelmarsh_multiyear_v31.py"] = "0"; bad.append(meta)
        for meta in bad:
            with self.assertRaises(ValueError):
                u.verify_preparation(meta, p, "abc")
        with self.assertRaises(ValueError):
            u.verify_preparation(good_meta(p), p, "other")


class NormalizedPreparerTests(unittest.TestCase):
    def frames(self):
        return [pd.DataFrame({"timestamp": [1, 1, 2], "power_kw": [5., np.nan, 6.], "wind_speed_mps": [3., np.nan, 4.]})]

    def test_records_take_the_unit_from_the_preparer_loop(self):
        from scripts import prepare_kelmarsh_multiyear_normalized as prep
        saved = {}

        def fake_prepare(root, years, curve_years, download):
            for unit in ("2", "5"):                      # non-contiguous ids: order-based numbering would be wrong
                ticks = prep.original.merge_ticks(self.frames())
                prep.original.prepare_turbine(unit, ticks, 2050., None, None)
            return {"source_sha256": {}, "rows": 2, "years": list(years), "prepared": {"sha256": "x"}}
        with patch.object(prep.original, "prepare", fake_prepare), \
                patch.object(prep.original, "prepare_turbine", lambda *a: (None, None)), \
                patch.object(prep.original, "save_json", lambda path, value: saved.update(value=value)):
            metadata = prep.prepare("root", [2016], [2016], "existing_dominating_row_v1")
        units = metadata["duplicate_normalization"]["units"]
        self.assertEqual([r["turbine_id"] for r in units], [2, 5])
        self.assertTrue(all(r["cells_filled"] == 0 and r["output_rows"] == 2 for r in units))
        self.assertEqual(set(metadata["source_sha256"]), set(u.PREPARER_SOURCES))
        self.assertIs(saved["value"], metadata)

    def test_unpaired_records_are_refused(self):
        from scripts import prepare_kelmarsh_multiyear_normalized as prep

        def fake_prepare(root, years, curve_years, download):
            prep.original.merge_ticks(self.frames())
            prep.original.merge_ticks(self.frames())
            prep.original.prepare_turbine("1", None, 2050., None, None)
        with patch.object(prep.original, "prepare", fake_prepare), \
                patch.object(prep.original, "prepare_turbine", lambda *a: (None, None)):
            with self.assertRaisesRegex(ValueError, "not paired"):
                prep.prepare("root", [2016], [2016], "existing_dominating_row_v1")

    def test_protocol_section_guards(self):
        from scripts import prepare_kelmarsh_multiyear_normalized as prep
        self.assertEqual(prep.section(PROTOCOL, "v32_preparation")["years"], list(range(2016, 2025)))
        with tempfile.TemporaryDirectory() as tmp:
            p = protocol()
            p["data"]["v32_preparation"]["duplicate_normalization"] = "coalesce"
            path = Path(tmp) / "p.json"
            path.write_text(json.dumps(p), encoding="utf-8")
            with self.assertRaises(ValueError):
                prep.section(path, "v32_preparation")


class ReportTests(unittest.TestCase):
    def test_year_labels_come_from_the_protocol(self):
        p = protocol()
        p["statistics"]["replicates"] = 200
        with tempfile.TemporaryDirectory() as tmp:
            run = fake_run(tmp, p)
            result = u.build_report(run)
            text = (run / "report.md").read_text(encoding="utf-8")
            self.assertIn("Test 2024 windows", text)
            self.assertIn("Validation (2023, ALL cohort)", text)
            self.assertNotIn("2022", text.split("## Test metrics")[0])
            self.assertIn("single 2024 test evaluation", result["scope"])
            self.assertEqual(result["protocol_id"], "kelmarsh_upstream_v32")


@unittest.skipUnless(HAS_TORCH, "torch/sklearn unavailable on this host")
class ControllerTests(unittest.TestCase):
    def make(self, tmp):
        from scripts import train_kelmarsh_upstream_v32 as task
        self.task = task
        p = copy.deepcopy(protocol())
        p["data"]["splits"] = {"train": [2016], "validation": [2017], "test": [2018]}
        p["seeds"] = [2024]
        p["training"].update(max_epochs=2, scheduler_t_max=2)
        p["budget"].update(fits=3, new_fits_max=3)
        p["statistics"]["replicates"] = 200
        return task.V32Run(Path(tmp) / "run", protocol=p, protocol_sha="test", series_factory=task.v31.synthetic_series,
                           require_preflight=False, write_pointer=False, lock_path=Path(tmp) / "lock", verify_data=False)

    def test_end_to_end_freeze_before_test_and_reuse(self):
        torch.set_num_threads(2)
        with patch("scripts.train_kelmarsh_anchored_residual.gpu_snapshot", return_value=None):
            with tempfile.TemporaryDirectory() as tmp:
                run = self.make(tmp)
                self.assertEqual(run.run(), "COMPLETED")
                self.assertEqual(len(run.registered_fits()), 3)
                events = [json.loads(line) for line in (run.dir / "events.jsonl").read_text().splitlines()]
                frozen = next(i for i, e in enumerate(events) if e["event"] == "S1_frozen")
                first_test = next(i for i, e in enumerate(events) if e.get("split") == "test")
                self.assertLess(frozen, first_test)
                self.assertIn("Test 2018 windows", (run.dir / "report.md").read_text(encoding="utf-8"))
                again = self.make(tmp)
                with patch.object(self.task.v28, "fit_config", side_effect=AssertionError("retrained")):
                    self.assertEqual(again.run(), "COMPLETED")

    def test_test_access_refused_before_freeze_and_preflight_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp)
            run.dir.mkdir(parents=True)
            run.record_sources()
            run.stage_d0()
            with self.assertRaises(PermissionError):
                run.windows("test", "all")
            with self.assertRaises(PermissionError):
                run.upstream_frame(include_test=True)
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp)
            run.require_preflight = True
            with patch.object(self.task, "PREFLIGHT_DIR", Path(tmp) / "pf"), \
                    patch("scripts.train_kelmarsh_anchored_residual.gpu_snapshot", return_value=None):
                with self.assertRaisesRegex(ValueError, "preflight"):
                    run.run()
            self.assertFalse(list(run.dir.rglob("trained.json")))


if __name__ == "__main__":
    unittest.main()
