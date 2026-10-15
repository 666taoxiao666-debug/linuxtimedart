import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from test_numeric_foundation import fixture
from test_numeric_subhour import tables_for
from utils import numeric_state as state, numeric_subhour as endpoint


class NumericStateTests(unittest.TestCase):
    def test_closed_form_shrinkage_and_sparse_cells_exact_prior(self):
        x = np.linspace(-20, 20, 120)
        anchor = np.zeros((120, 2))
        groups = np.repeat([0, 1, 5], [50, 69, 1])
        issue = np.datetime64("2016-05-01T00") + np.arange(120) * np.timedelta64(4, "h")
        beta = np.array([.5, .3])
        desired = np.array([[.8, .9], [.2, -.5], [10., 10.]])
        y = x[:, None] * desired[np.searchsorted([0, 1, 5], groups)]
        ds = SimpleNamespace(truth_and_anchor=lambda: (y, anchor))
        with patch.object(state, "design", return_value=(x, anchor, groups, issue)):
            model = state.fit(ds, None, beta)
        penalty = x @ x / 6
        for g in (0, 1):
            m = groups == g
            np.testing.assert_allclose(model["coefficients"][g], np.clip((x[m] @ y[m] + penalty * beta) / (x[m] @ x[m] + penalty), 0, 1))
        for g in (2, 3, 4, 5):
            np.testing.assert_array_equal(model["coefficients"][g], beta)
        self.assertEqual(model["support"][5].tolist(), [1, 1, 1])
        self.assertTrue((model["coefficients"] >= 0).all() and (model["coefficients"] <= 1).all())

    def test_day_support_is_not_replaced_by_correlated_window_count(self):
        x, anchor = np.ones(100), np.zeros((100, 2))
        issue = np.datetime64("2016-05-01T00") + np.arange(100) % 25 * np.timedelta64(1, "h")
        ds = SimpleNamespace(truth_and_anchor=lambda: (np.full((100, 2), 100.), anchor))
        with patch.object(state, "design", return_value=(x, anchor, np.zeros(100, int), issue)):
            model = state.fit(ds, None, [.5, .3])
        np.testing.assert_array_equal(model["coefficients"][0], [.5, .3])
        self.assertEqual(model["support"][0].tolist(), [100, 25, 2])

    def test_causal_inference_disabled_lead_and_convex_output(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            ds = s.subset("validation")
            ds.window_starts = ds.window_starts[:1]
            tables = tables_for(s)
            model = dict(coefficients=np.tile([.9, .8], (6, 1)), prior=np.array([.53, .35]))
            before = state.predict(model, ds, tables, enabled=(True, False))
            baseline = endpoint.predict([.53, .35], ds, tables)
            np.testing.assert_array_equal(before[:, 1], baseline[:, 1])
            delta, anchor, _, issue = state.design(ds, tables)
            self.assertTrue((before >= np.minimum(anchor, anchor + delta[:, None])).all())
            self.assertTrue((before <= np.maximum(anchor, anchor + delta[:, None])).all())
            future = ds.window_starts[0] + 72 + np.arange(2)
            s.raw[future], s.values[future] = 1e12, 1e12
            for ticks in tables.values():
                ticks.loc[ticks.index >= issue[0]] = 1e12
            ds.truth_and_anchor = lambda: (_ for _ in ()).throw(AssertionError("Future labels read"))
            np.testing.assert_array_equal(state.predict(model, ds, tables, (True, False)), before)
            with self.assertRaisesRegex(ValueError, "Boolean"):
                state.predict(model, ds, tables, [1, 0])

    def test_no_output_clipping_and_zero_delta_remains_negative_anchor(self):
        with patch.object(state, "design", return_value=(np.zeros(1), np.full((1, 2), -10.5117), np.zeros(1, int), np.array([0]))):
            result = state.predict(dict(coefficients=np.ones((6, 2)), prior=np.zeros(2)), None, None)
        np.testing.assert_array_equal(result, -10.5117)

    def test_train_gate_separately_protects_both_leads_and_rejects_nan(self):
        refs = {key: {name: {scope: dict(mae=10., rmse=20.) for scope in ("overall", "horizon1", "horizon2")}
            for name in ("v15", "persistence")} for key in ("o70", "o80", "o90")}
        records = {key: {scope: dict(mae=9., rmse=19.) for scope in ("overall", "horizon1", "horizon2")} for key in refs}
        self.assertEqual(state.choose(records, refs)["enabled_leads"], [True, True])
        records["o80"]["horizon1"]["rmse"] = 21.
        self.assertEqual(state.choose(records, refs)["enabled_leads"], [False, True])
        refs["o70"]["v15"]["horizon2"]["mae"] = float("nan")
        self.assertTrue(state.choose(records, refs)["reused_v15"])

    def test_completed_executor_reuse_actual_inputs_and_file_tamper(self):
        from scripts import train_kelmarsh_numeric_state as runner
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            ds, tables = s.subset("fit"), tables_for(s)
            directory = Path(root) / "state"
            model, done = runner.fit_stage(ds, tables, [.53, .35], "baseline", directory, {"fixture": True})
            with patch.object(state, "fit", side_effect=AssertionError("Completed fit repeated")):
                again, _ = runner.fit_stage(ds, tables, [.53, .35], "baseline", directory, {"fixture": True})
            np.testing.assert_array_equal(model["coefficients"], again["coefficients"])
            with self.assertRaisesRegex(ValueError, "executor changed"):
                runner.fit_stage(ds, tables, [.5, .3], "baseline", directory, {"fixture": True})
            with (directory / "model.npz").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "fit changed"):
                runner.fit_stage(ds, tables, [.53, .35], "baseline", directory, {"fixture": True})

    def test_no_outer_or_final_fit_when_all_leads_rejected(self):
        from scripts import train_kelmarsh_numeric_state as runner
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            runner.f.old.write(root / "prior/reports/SelectedNumeric/result.json", dict(metrics={"mae": 97.592426}))
            with patch.object(runner.base, "cached_baseline", side_effect=AssertionError("Final cache read")), \
                 patch.object(runner, "fit_stage", side_effect=AssertionError("Final refit")), \
                 patch.object(runner.f, "outer_report", side_effect=AssertionError("Outer evaluated")):
                runner.finish({"reused_v15": True}, root / "prior", None, None, None, root / "new", {}, {}, {}, None)
            self.assertFalse(runner.f.old.read(root / "new/result.json")["new_outer_evaluation"])

    def test_protocol_has_one_fixed_candidate_not_state_or_threshold_search(self):
        from scripts import train_kelmarsh_numeric_state as runner
        p = json.loads((Path(__file__).resolve().parents[1] / "configs/kelmarsh_numeric_state_protocol_v18.json").read_text())
        self.assertEqual(p["protocol_id"], runner.PROTOCOL_ID)
        self.assertEqual(p["candidates"], [state.NAME])
        self.assertFalse(p["wiki_enabled"])
        self.assertIn("no new outer evaluation", p["selection"])


if __name__ == "__main__":
    unittest.main()
