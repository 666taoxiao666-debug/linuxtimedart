import copy
import unittest
import numpy as np

from utils import local_transport_event_wiki as local


class LocalTransportTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n = 128
        self.context = dict(power_ratio=rng.uniform(.1, .3, n), wind_mean_mps=rng.uniform(4, 8, n),
            wind_std_mps=rng.uniform(.1, 1, n), power_change_kw=rng.uniform(-20, 20, n),
            wind_change_mps=rng.uniform(-1, 1, n), power_bin=np.ones(n, int), joint_bin=np.full(n, 5, int))
        self.ref = np.full((n, 2), 300.)
        self.truth = self.ref+30.
        self.events = np.ones((n, 1), bool)
        self.turbines = np.tile(np.arange(1, 9), 16)
        self.contract = dict(base_checkpoint_sha256="a"*64, rated_power_kw=1500., pred_len=2,
            event_ids=["gust"], source_turbines=list(range(1, 9)))
        self.model = local.fit_experts(self.context, self.ref, self.ref, self.events, self.truth,
            self.turbines, self.contract, neighbors=8, min_neighbor_turbines=4, min_fit_windows=16)

    def test_local_values_cannot_extrapolate_beyond_neighbor_residuals_and_identity_is_bound(self):
        prediction, physical, support = local.candidates(self.model, self.context, self.ref,
            self.ref, self.events, self.contract)
        np.testing.assert_allclose(prediction[..., 0], 330.)
        self.assertTrue(physical[..., 0].all())
        self.assertTrue(support[..., 0].any())
        changed = copy.deepcopy(self.model)
        changed["experts"][0]["residuals"][0][0] = .9
        with self.assertRaisesRegex(ValueError, "cases"):
            local.validate_identity(changed, self.contract)
        with self.assertRaisesRegex(ValueError, "Base checkpoint"):
            local.validate_identity(self.model, dict(self.contract, base_checkpoint_sha256="b"*64))

    def test_turbine_balance_and_leave_one_turbine_support(self):
        value = local._balanced_median(np.array([[0., 0., 0., 100., 100.]]), np.array([[1, 1, 1, 2, 3]]))
        self.assertEqual(value[0], 100.)
        keys = np.arange(12.)[:, None]
        ids = np.repeat([1, 2, 3], 4)
        idx, distances, diversity = local._neighbors(keys, keys, ids, 4, excluded_turbines=ids)
        self.assertFalse((ids[idx] == ids[:, None]).any())
        self.assertTrue(np.isfinite(distances).all())

    def test_real_policy_uses_exact_candidates_and_abstains_outside_support(self):
        blocks = [dict(context=self.context, reference=self.ref, anchor=self.ref, events=self.events,
            truth=self.truth, turbines=self.turbines, times=np.full(self.ref.shape, np.datetime64(day, "ns")))
            for day in ("2023-06-01", "2023-06-03")]
        policy = local.fit_transfer_policy(self.model, blocks, self.contract, min_windows=16, min_turbines=4)
        result = local.apply_transfer(self.model, policy, self.context, self.ref, self.ref, self.events, self.contract)
        self.assertTrue((result["action"]>0).any())
        self.assertLess(np.abs(result["prediction"]-self.truth).mean(), 30.)
        far = dict(self.context, wind_mean_mps=np.full(128, 100.))
        result = local.apply_transfer(self.model, policy, far, self.ref, self.ref, self.events, self.contract)
        np.testing.assert_array_equal(result["prediction"], self.ref)
        for h in range(2):
            policy["entries"][f"joint:5:{h}:0"].update(supported=True, accepted=False, score_kw=0.)
        policy["policy_id"] = local.digest({k:v for k,v in policy.items() if k != "policy_id"})
        result = local.apply_transfer(self.model, policy, self.context, self.ref, self.ref, self.events, self.contract)
        np.testing.assert_array_equal(result["prediction"], self.ref)

    def test_held_out_labels_cannot_enter_memory(self):
        ids = self.turbines.copy(); ids[0] = 99
        with self.assertRaisesRegex(ValueError, "Held-out turbine"):
            local.fit_experts(self.context, self.ref, self.ref, self.events, self.truth,
                ids, self.contract, neighbors=8, min_neighbor_turbines=4, min_fit_windows=16)


if __name__ == "__main__":
    unittest.main()
