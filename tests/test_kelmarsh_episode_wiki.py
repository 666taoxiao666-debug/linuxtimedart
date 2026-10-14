import copy
import unittest
import numpy as np

from utils import kelmarsh_hourly_wiki as old
from utils import kelmarsh_pareto_wiki as pareto
from utils import kelmarsh_episode_wiki as new


class EpisodeWikiTests(unittest.TestCase):
    def fixture(self):
        n, h = 150, 2
        rng = np.random.default_rng(2024)
        context = dict(power_ratio=rng.uniform(.2, .5, n), wind_mean_mps=rng.uniform(3, 8, n),
            wind_std_mps=rng.uniform(1, 2, n), power_change_kw=rng.normal(0, 10, n),
            wind_change_mps=rng.normal(size=n), power_bin=np.ones(n, int), joint_bin=np.ones(n, int))
        for key in old.CONTROLS:
            context[key] = rng.normal(size=n)
        anchor = np.full((n, h), 500.)
        data = dict(context=context, anchor=anchor, reference=anchor.copy(), truth=anchor + 20,
            turbines=np.tile([1, 2, 3, 4, 5], 30), events=np.tile([1, 0, 0, 0], (n, 1)).astype(bool),
            times=np.datetime64("2016-07-01T00", "ns") + np.arange(n)[:, None] * np.timedelta64(3, "h") +
                  np.arange(h)[None] * np.timedelta64(1, "h"))
        contract = dict(source_turbines=[1, 2, 3, 4, 5], pred_len=h, rated_power_kw=2050., base_checkpoint_sha256="a" * 64)
        settings = dict(use_controls=True, neighbors=8, min_neighbor_turbines=2, min_fit_windows=20,
            support_quantile=.95, max_scale=.2, composition_max_scale=.05)
        memory = old.fit_experts(data, contract, settings)
        return data, contract, memory

    def test_duplicate_weather_case_cannot_inflate_block_mass(self):
        units, groups = [[1, 2, 1]], [[10, 10, 20]]
        w = new.balanced_weights(units, groups)
        np.testing.assert_allclose(w, [[.25, .25, .5]])
        repeated = new.balanced_weights([[1, 1, 1, 2, 1]], [[10, 10, 10, 10, 20]])
        self.assertAlmostEqual(repeated[0, :3].sum(), .25)
        self.assertAlmostEqual(repeated[0, -1], .5)

    def test_control_reproduces_previous_readout_and_episode_changes_only_weights(self):
        values, ids, groups = np.array([[0., .1, .8, .9]]), np.array([[1, 2, 1, 1]]), np.array([[1, 1, 2, 2]])
        a = pareto.local_readout(values, ids, np.array([.6]), 1., 0.)
        b = new.readout(values, ids, groups, [.6], 1., 0., "turbine_control")
        np.testing.assert_allclose(a["prediction"], b["prediction"])
        c = new.readout(values, ids, groups, [.6], 1., 0.)
        self.assertNotEqual(b["prediction"][0], c["prediction"][0])

    def test_joint_empirical_losses_and_bounds(self):
        rng = np.random.default_rng(4)
        out = new.readout(rng.normal(size=(80, 12)), np.tile([1, 2, 3] * 4, (80, 1)),
            np.tile(np.repeat([1, 2, 3, 4], 3), (80, 1)), rng.normal(size=80), .2, 2/3)
        self.assertTrue((out["mae_gain"] >= -1e-12).all())
        self.assertTrue((out["mse_gain"] >= -1e-12).all())
        np.testing.assert_allclose(out["weights"].sum(1), 1.)

    def test_exact_clock_alignment_and_no_future_outcome_retrieval(self):
        data, contract, memory = self.fixture()
        model = new.make_model(memory, data, contract)
        with self.assertRaises(ValueError):
            new.candidates(model, data, contract)
        query = copy.deepcopy(data)
        query["times"] += np.timedelta64(30, "D")
        a = new.candidates(model, query, contract)
        query.pop("truth")
        query.pop("turbines")
        b = new.candidates(model, query, contract)
        np.testing.assert_array_equal(a["prediction"], b["prediction"])
        wrong = copy.deepcopy(data)
        wrong["times"][0, 0] = np.datetime64("NaT")
        with self.assertRaises(ValueError):
            new.make_model(memory, wrong, contract)
        wrong = copy.deepcopy(data)
        wrong["truth"][0, 0] += 1
        with self.assertRaises(ValueError):
            new.make_model(memory, wrong, contract)

    def test_new_readout_requires_new_utility_evidence_and_exact_fallback(self):
        data, contract, memory = self.fixture()
        model = new.make_model(memory, data, contract)
        blocks = []
        for days in (30, 60):
            b = copy.deepcopy(data)
            b["times"] += np.timedelta64(days, "D")
            blocks.append(b)
        policy = new.fit_policy(model, blocks, contract, dict(min_windows=1, min_turbines=1, positive_fraction=2/3))
        query = copy.deepcopy(blocks[-1])
        query["events"][:] = False
        query.pop("truth")
        out = new.candidates(model, query, contract)
        np.testing.assert_array_equal(new.apply_with_policy(model, policy, out, query, contract)["prediction"], query["reference"])
        changed = new.make_model(memory, data, contract, mode="turbine_control")
        with self.assertRaises(ValueError):
            new.apply_with_policy(changed, policy, new.candidates(changed, query, contract), query, contract)
        with self.assertRaises(ValueError):
            new.validate(model, dict(contract, base_checkpoint_sha256="b" * 64))
        with self.assertRaises(ValueError):
            new.fit_policy(model, [blocks[0], blocks[0]], contract, policy["settings"])


if __name__ == "__main__":
    unittest.main()
