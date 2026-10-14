import copy
import unittest
import numpy as np

from utils import kelmarsh_pareto_wiki as new
from utils import kelmarsh_hourly_wiki as old


class ParetoWikiTests(unittest.TestCase):
    def test_read_only_summary_uses_each_saved_fold_once(self):
        import json
        from pathlib import Path
        import tempfile
        from scripts.eval_kelmarsh_pareto_wiki import summarize_outputs
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for fold, mae in ((0, 1.), (1, 3.)):
                path = root / "runs" / f"f{fold}_s2024" / "Test" / "result.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(dict(model="Test", best_epoch=0, metrics=dict(mae=mae, rmse=mae + 1, r2=.5))))
            frame = summarize_outputs(root)
            self.assertEqual(len(frame), 2)
            import pandas as pd
            macro = pd.read_csv(root / "metrics_macro.csv")
            self.assertEqual(macro.iloc[0]["mae_mean"], 2.)
            self.assertEqual(macro.iloc[0]["mae_count"], 2)

    def test_long_history_summaries_alias_different_current_wind_states(self):
        from types import SimpleNamespace
        from scripts.audit_kelmarsh_phase_coordinates import phase_coordinates
        wind_a = np.array([4.] * 10 + [2., 8.])
        wind_b = np.array([4.] * 10 + [8., 2.])
        def make(wind):
            raw = np.full((14, 7), 500.)
            raw[:12, 0] = wind
            context = dict(power_ratio=np.array([500/2050]), wind_mean_mps=np.array([wind.mean()]),
                wind_std_mps=np.array([wind.std()]), power_change_kw=np.array([0.]),
                wind_change_mps=np.array([wind[6:].mean() - wind[:6].mean()]))
            for key in old.CONTROLS:
                context[key] = np.zeros(1)
            dataset = SimpleNamespace(seq_len=12, window_starts=np.array([0]), series=SimpleNamespace(raw=raw))
            anchor = np.full((1, 2), 500.)
            return old.keys(context, anchor, 2050., True), phase_coordinates(dataset, 2050.)
        old_a, phase_a = make(wind_a)
        old_b, phase_b = make(wind_b)
        np.testing.assert_array_equal(old_a, old_b)
        self.assertNotEqual(phase_a[0, 0], phase_b[0, 0])

    def test_closed_form_shrinkage_improves_both_empirical_losses(self):
        values = np.tile([0., 0., 0., .6, .9], (1, 3))
        ids = np.repeat([[1, 2, 3]], 5, axis=1)
        output = new.local_readout(values, ids, np.array([.4]), .5, 2/3)
        np.testing.assert_allclose(output["alpha"], [.25])
        np.testing.assert_allclose(output["prediction"], [.3])
        self.assertTrue(output["eligible"].all())
        self.assertGreater(output["mae_gain"][0], 0)
        self.assertGreater(output["mse_gain"][0], 0)
        whole = new.local_readout(values, ids, np.array([.4]), .5, 2/3, "median_local")
        self.assertFalse(whole["eligible"].any())
        self.assertLess(whole["mse_gain"][0], 0)

    def test_conflicting_mean_and_median_cannot_force_intervention(self):
        values = np.tile([0., 0., 0., .6, .9], (1, 3))
        output = new.local_readout(values, np.repeat([[1, 2, 3]], 5, axis=1), np.array([.2]), .5, 2/3)
        np.testing.assert_array_equal(output["alpha"], [0.])
        self.assertFalse(output["eligible"].any())

    def test_donor_balance_and_bounds_and_property(self):
        rng = np.random.default_rng(2024)
        values = rng.normal(size=(100, 15))
        ids = np.tile(np.repeat([1, 2, 3], 5), (100, 1))
        base = rng.normal(size=100)
        out = new.local_readout(values, ids, base, .2, 2/3)
        self.assertTrue((np.abs(out["prediction"] - base) <= .2 + 1e-12).all())
        self.assertTrue((out["mae_gain"] >= -1e-12).all())
        self.assertTrue((out["mse_gain"] >= -1e-12).all())
        a = new.local_readout(np.array([[0., .5, 1.]]), np.array([[1, 2, 3]]), np.array([.8]), 1., 0.)
        b = new.local_readout(np.array([[0., 0., 0., .5, 1.]]), np.array([[1, 1, 1, 2, 3]]), np.array([.8]), 1., 0.)
        np.testing.assert_allclose(a["prediction"], b["prediction"])

    def test_identity_inference_without_labels_and_exact_fallback(self):
        n, horizon = 150, 2
        rng = np.random.default_rng(2024)
        context = {"power_ratio": rng.uniform(.2, .5, n), "wind_mean_mps": rng.uniform(3, 8, n),
                   "wind_std_mps": rng.uniform(1, 2, n), "power_change_kw": rng.normal(0, 10, n),
                   "wind_change_mps": rng.normal(size=n), "power_bin": np.ones(n, int), "joint_bin": np.ones(n, int)}
        for key in old.CONTROLS:
            context[key] = rng.normal(size=n)
        anchor = np.full((n, horizon), 500.)
        data = dict(context=context, anchor=anchor, reference=anchor.copy(), truth=anchor + 20,
                    turbines=np.tile([1, 2, 3, 4, 5], 30), events=np.tile([1, 0, 0, 0], (n, 1)).astype(bool))
        contract = dict(source_turbines=[1, 2, 3, 4, 5], pred_len=2, rated_power_kw=2050., base_checkpoint_sha256="a" * 64)
        settings = dict(use_controls=True, neighbors=8, min_neighbor_turbines=2, min_fit_windows=20,
                        support_quantile=.95, max_scale=.2, composition_max_scale=.05)
        memory = old.fit_experts(data, contract, settings)
        model = new.make_model(memory, contract, mode="pareto", protocol_sha256="p" * 64, positive_fraction=2/3)
        early, late = copy.deepcopy(data), copy.deepcopy(data)
        early["times"] = np.tile(np.array(["2016-08-01", "2016-08-01T01"], dtype="datetime64[h]"), (n, 1))
        late["times"] = early["times"] + np.timedelta64(24, "h")
        policy = new.fit_policy(model, [early, late], contract, dict(min_windows=1, min_turbines=1, positive_fraction=2/3))
        inference = {k: v for k, v in data.items() if k not in ("truth", "turbines")}
        np.testing.assert_array_equal(new.apply(model, policy, data, contract)["prediction"], new.apply(model, policy, inference, contract)["prediction"])
        quiet = dict(inference, events=np.zeros_like(inference["events"]))
        np.testing.assert_array_equal(new.apply(model, policy, quiet, contract)["prediction"], anchor)
        self.assertFalse(new.candidates(model, inference, contract)["physical"][..., -1].any())
        with self.assertRaises(ValueError):
            new.apply(model, policy, inference, dict(contract, base_checkpoint_sha256="b" * 64))
        with self.assertRaises(ValueError):
            new.fit_policy(model, [early, early], contract, policy["settings"])
        bad = copy.deepcopy(late)
        bad["turbines"][0] = 6
        with self.assertRaises(ValueError):
            new.fit_policy(model, [early, bad], contract, policy["settings"])
        changed = copy.deepcopy(model)
        changed["mode"] = "median_local"
        with self.assertRaises(ValueError):
            new.apply(changed, policy, inference, contract)


if __name__ == "__main__":
    unittest.main()
