import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_numeric_foundation import fixture
from test_numeric_subhour import tables_for
from utils import numeric_horizon2 as head, numeric_subhour as endpoint


class NumericHorizon2Tests(unittest.TestCase):
    def test_feature_schema_consumes_earliest_hour_and_all_seven_channels(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            ds = s.subset("selection")
            ds.window_starts = ds.window_starts[:1]
            start = ds.window_starts[0]
            x = s.values[start:start + 72].astype(float)
            z = head.history_features(ds)
            self.assertEqual(z.shape, (1, 105))
            np.testing.assert_array_equal(z[0, :7], x[-1])
            np.testing.assert_array_equal(z[0, 7:14], x[-1] - x[-2])
            np.testing.assert_array_equal(z[0, 56:63], x[-1] - x[0])
            np.testing.assert_allclose(z[0, -7:], x.mean(0) - x[-1], atol=1e-12)
            s.values[start] += 10
            self.assertFalse(np.array_equal(head.history_features(ds), z))
            ds.seq_len = 24
            with self.assertRaisesRegex(ValueError, "72 observed"):
                head.history_features(ds)

    def test_first_hour_immutable_and_inference_ignores_all_future_values(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            tables = tables_for(s)
            model = head.fit(s.subset("fit"), tables, [.53, .35])
            ds = s.subset("validation")
            ds.window_starts = ds.window_starts[:1]
            expected = head.predict(model, ds, tables, [.53, .35])
            np.testing.assert_array_equal(expected[:, 0], endpoint.predict([.53, .35], ds, tables)[:, 0])
            future = ds.window_starts[0] + ds.seq_len + np.arange(2)
            issue = s.dates[future[0]] - np.timedelta64(1, "h")
            s.values[future], s.raw[future] = 1e12, 1e12
            for raw in tables.values():
                raw.loc[raw.index >= issue] = 1e12
            ds.truth_and_anchor = lambda: (_ for _ in ()).throw(AssertionError("Future labels read"))
            np.testing.assert_array_equal(head.predict(model, ds, tables, [.53, .35]), expected)
            model["coefficients"] = np.zeros(104)
            with self.assertRaisesRegex(ValueError, "105-feature"):
                head.predict(model, ds, tables, [.53, .35])

    def test_fitting_never_uses_first_hour_outcomes(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            fit = s.subset("fit")
            tables = tables_for(s)
            expected = head.fit(fit, tables, [.53, .35])
            truth, anchor = fit.truth_and_anchor()
            truth[:, 0] = 1e12
            fit.truth_and_anchor = lambda: (truth, anchor)
            actual = head.fit(fit, tables, [.53, .35])
            for key in expected:
                np.testing.assert_array_equal(expected[key], actual[key])

    def test_gate_rejects_h2_damage_hidden_by_overall_and_any_missing_reference(self):
        refs = {key: {name: {scope: dict(mae=10., rmse=20.) for scope in ("overall", "horizon2")}
                      for name in ("v15", "persistence")} for key in ("o70", "o80", "o90")}
        records = {key: {scope: dict(mae=9., rmse=19.) for scope in ("overall", "horizon2")} for key in refs}
        self.assertFalse(head.choose(records, refs)["reused_v15"])
        records["o70"]["horizon2"]["rmse"] = 21.
        self.assertTrue(head.choose(records, refs)["reused_v15"])
        records["o70"]["horizon2"]["rmse"] = 19.
        refs["o80"]["v15"]["horizon2"]["mae"] = float("nan")
        self.assertTrue(head.choose(records, refs)["reused_v15"])
        del refs["o80"]["persistence"]
        with self.assertRaisesRegex(ValueError, "Both frozen"):
            head.choose(records, refs)

    def test_completed_fit_reused_and_actual_inputs_or_model_tamper_rejected(self):
        from scripts import train_kelmarsh_numeric_horizon2 as runner
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            fit, tables = s.subset("fit"), tables_for(s)
            directory = Path(root) / "head"
            actual, done = runner.fit_stage(fit, tables, [.53, .35], "baseline", directory, {"fixture": True})
            with patch.object(head, "fit", side_effect=AssertionError("Completed head retrained")):
                again, _ = runner.fit_stage(fit, tables, [.53, .35], "baseline", directory, {"fixture": True})
            np.testing.assert_array_equal(actual["coefficients"], again["coefficients"])
            original_value = s.values[fit.window_starts[0], 0].copy()
            s.values[fit.window_starts[0], 0] += 2
            with self.assertRaisesRegex(ValueError, "stage changed"):
                runner.fit_stage(fit, tables, [.53, .35], "baseline", directory, {"fixture": True})
            s.values[fit.window_starts[0], 0] = original_value
            with (directory / "model.npz").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "stage changed"):
                runner.fit_stage(fit, tables, [.53, .35], "baseline", directory, {"fixture": True})

    def test_rejected_head_never_refits_or_queries_outer(self):
        from scripts import train_kelmarsh_numeric_horizon2 as runner
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            runner.f.old.write(root / "prior/reports/SelectedNumeric/result.json", dict(metrics={"mae": 97.592426}))
            with patch.object(runner.base, "cached_baseline", side_effect=AssertionError("Final executor read")), \
                 patch.object(runner, "fit_stage", side_effect=AssertionError("Final head fit")), \
                 patch.object(runner.f, "outer_report", side_effect=AssertionError("Outer evaluated")):
                runner.finish({"reused_v15": True}, root / "prior", None, None, None, root / "new", {}, {}, {}, None)
            result = runner.f.old.read(root / "new/result.json")
            self.assertFalse(result["new_outer_evaluation"])
            self.assertFalse(result["primary_joint_success"])

    def test_protocol_has_one_h2_only_candidate_and_no_outer_search(self):
        from scripts import train_kelmarsh_numeric_horizon2 as runner
        root = Path(__file__).resolve().parents[1]
        p = json.loads((root / "configs/kelmarsh_numeric_horizon2_protocol_v17.json").read_text())
        self.assertEqual(p["protocol_id"], runner.PROTOCOL_ID)
        self.assertEqual(p["candidates"], [runner.NAME])
        self.assertFalse(p["wiki_enabled"])
        self.assertIn("First-hour", p["mechanism"])
        self.assertIn("no new outer evaluation", p["selection"])


if __name__ == "__main__":
    unittest.main()
