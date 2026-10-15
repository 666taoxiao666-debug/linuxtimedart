import copy
from pathlib import Path
import unittest
import numpy as np
from tests import test_kelmarsh_wiki_quality as quality_tests
from utils import kelmarsh_short_dynamics_wiki as short


def fixture():
    data, contract, source, query, qi = quality_tests.WikiQualityTests().fixture()
    history = np.random.default_rng(2024).normal(size=(len(data["truth"]), 7, 72)).astype(np.float32)
    data["history"] = history
    query["history"] = history.copy()
    ti = copy.deepcopy(qi)
    ti["last_observed_ns"] -= 30 * 24 * 3600 * 10**9
    bank = short.fit_memory(source, data, ti, contract)
    return data, contract, source, bank, query, qi, ti


class ShortDynamicsWikiTests(unittest.TestCase):
    def test_fixed_lag_coordinates_and_prefix_independence(self):
        x = np.zeros((1, 7, 72))
        x[0, 0, -4:] = [1, 2, 4, 8]
        x[0, 5, -4:] = [3, 5, 7, 11]
        x[0, 6, -4:] = [9, 10, 12, 15]
        np.testing.assert_array_equal(short.short_keys(x), [[8, 11, 3, 6, 4, 7, 4, 8]])
        x[:, :, :-4] = 1e6
        np.testing.assert_array_equal(short.short_keys(x), [[8, 11, 3, 6, 4, 7, 4, 8]])
        for invalid in (np.zeros((1, 6, 72)), np.zeros((1, 7, 3)), np.full((1, 7, 72), np.nan)):
            with self.assertRaises(ValueError):
                short.short_keys(invalid)

    def test_same_source_cases_labels_units_availability_and_settings(self):
        data, contract, source, bank, query, qi, ti = fixture()
        self.assertEqual(bank["source_expert_id"], source["expert_id"])
        self.assertEqual(bank["settings"], source["settings"])
        for before, after in zip(source["groups"], bank["groups"]):
            for key in ("event", "state", "active", "windows", "turbines"):
                self.assertEqual(before[key], after[key])
            if before["active"]:
                for key in ("values", "units", "weather_blocks", "available_ns"):
                    self.assertEqual(before[key], after[key])
                self.assertEqual(len(after["keys"][0]), len(before["keys"][0]) + len(short.SHORT_NAMES))
        data["truth"] += 1
        with self.assertRaisesRegex(ValueError, "Source case rows"):
            short.fit_memory(source, data, ti, contract)

    def test_query_outcomes_and_turbines_do_not_enter_retrieval(self):
        _, contract, _, bank, query, qi, _ = fixture()
        before = short.candidates(bank, query, qi, contract)
        query["truth"] = np.full_like(query["reference"], 1e9)
        query["turbines"] = np.full(len(query["reference"]), 999)
        after = short.candidates(bank, query, qi, contract)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])

    def test_transform_only_changes_wiki_view_not_numerical_inputs_or_labels(self):
        data, contract, _, bank, query, qi, _ = fixture()
        pack = dict(query, truth=data["truth"].copy(), numeric=query["reference"].copy(),
                    memory=np.zeros((len(query["reference"]), 2, 20), np.float32),
                    overlap=np.zeros((len(query["reference"]), 2, 4), bool))
        saved = copy.deepcopy(pack)
        result = short.transform(pack, qi, bank, contract)
        for key in ("history", "truth", "numeric", "anchor", "reference", "times"):
            np.testing.assert_array_equal(saved[key], pack[key])
            np.testing.assert_array_equal(saved[key], result[key])
        self.assertEqual(result["memory"].shape, (len(pack["truth"]), 2, 20))
        self.assertTrue(result["overlap"].any())

    def test_same_state_restriction_and_exact_unsupported_fallback(self):
        _, contract, _, bank, query, qi, _ = fixture()
        qi["state"][:] = 4
        out = short.candidates(bank, query, qi, contract)
        self.assertFalse(out["overlap"].any())
        np.testing.assert_array_equal(out["prediction"], np.repeat(query["reference"][..., None], 4, -1))

    def test_implementation_base_and_future_maturity_checks(self):
        _, contract, _, bank, query, qi, _ = fixture()
        changed = copy.deepcopy(bank)
        changed["groups"][5]["values"][0][0] += .1
        with self.assertRaisesRegex(ValueError, "identity changed"):
            short.candidates(changed, query, qi, contract)
        wrong = dict(contract, scaler_sha256="another-scaler")
        with self.assertRaises(ValueError):
            short.candidates(bank, query, qi, wrong)
        qi["last_observed_ns"] += 2 * 3600 * 10**9
        with self.assertRaisesRegex(ValueError, "Future operating"):
            short.candidates(bank, query, qi, contract)
        qi["last_observed_ns"] -= (30 * 24 + 2) * 3600 * 10**9
        query["times"] -= np.timedelta64(30, "D")
        with self.assertRaisesRegex(ValueError, "not yet mature"):
            short.candidates(bank, query, qi, contract)

    def test_training_labels_from_held_out_turbine_are_rejected(self):
        data, contract, source, _, _, _, ti = fixture()
        data["turbines"][0] = 6
        with self.assertRaises(ValueError):
            short.fit_memory(source, data, ti, contract)

    def test_controls_follow_their_real_saved_provenance_not_assumed_copies(self):
        from scripts.train_kelmarsh_short_dynamics import control_reports
        refs = control_reports(Path("v7"), Path("v8"))
        self.assertEqual(refs["FreeWikiSelective"], Path("v7/reports/WikiIncrementSelective"))
        self.assertEqual(refs["CaseAnchoredWikiSelective"], Path("v8/reports/CaseAnchoredWikiSelective"))


if __name__ == "__main__":
    unittest.main()
