import copy
import inspect
import unittest
import numpy as np

from utils import control_transition_wiki as wiki


class ControlTransitionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2024)
        self.n = 192
        self.context = dict(power_ratio=rng.uniform(.15, .25, self.n),
            wind_mean_mps=rng.uniform(4, 7, self.n), wind_std_mps=rng.uniform(.1, .5, self.n),
            power_change_kw=rng.uniform(-5, 5, self.n), wind_change_mps=rng.uniform(-.1, .1, self.n),
            power_bin=np.ones(self.n, int), joint_bin=np.full(self.n, 5, int))
        self.context.update(wiki.control_context(np.full((self.n, 12), 2.), np.ones((self.n, 12))))
        self.anchor = np.full((self.n, 2), 300.)
        self.reference = self.anchor+10.
        self.truth = self.anchor+50.
        self.events = np.ones((self.n, 1), bool)
        self.ids = np.tile(np.arange(1, 13), 16)
        self.contract = dict(base_checkpoint_sha256="a"*64, base_manifest_sha256="b"*64,
            rated_power_kw=1500., pred_len=2, event_ids=["gust"], source_turbines=list(range(1, 13)),
            physical_coordinate_version="control_history_v1", event_config_sha256="c"*64)

    def fit(self, **kwargs):
        return wiki.fit_experts(self.context, self.reference, self.anchor, self.events,
            self.truth, self.ids, self.contract, neighbors=16, min_neighbor_turbines=2,
            min_fit_windows=32, **kwargs)

    def block(self, day, reference=None):
        return dict(context=self.context, reference=self.reference if reference is None else reference,
            anchor=self.anchor, events=self.events, truth=self.truth, turbines=self.ids,
            times=np.full(self.reference.shape, np.datetime64(day, "ns")))

    def policy(self, model, contract=None, reference=None):
        return wiki.fit_transfer_policy(model, [self.block("2023-06-01", reference),
            self.block("2023-06-03", reference)], contract or self.contract,
            min_windows=8, min_turbines=2)

    def test_transition_memory_is_base_independent_but_evidence_is_not(self):
        model = self.fit()
        new_contract = dict(self.contract, base_checkpoint_sha256="d"*64, base_manifest_sha256="e"*64)
        another = wiki.fit_experts(self.context, self.reference+20., self.anchor, self.events,
            self.truth, self.ids, new_contract, neighbors=16, min_neighbor_turbines=2, min_fit_windows=32)
        self.assertEqual(model["expert_bundle_id"], another["expert_bundle_id"])
        policy = self.policy(model)
        with self.assertRaisesRegex(ValueError, "execution pair"):
            wiki.validate_identity(model, new_contract, policy)
        # No memory refit is needed; new base-specific evidence IS needed.
        new_policy = self.policy(model, new_contract, self.reference+20.)
        wiki.validate_identity(model, new_contract, new_policy)
        self.assertNotEqual(policy["policy_id"], new_policy["policy_id"])
        pred, _, _ = wiki.candidates(model, self.context, self.reference+20., self.anchor,
                                     self.events, new_contract)
        np.testing.assert_allclose(pred[..., 0], 350.)

    def test_residual_ablation_cannot_reuse_another_base(self):
        model = self.fit(readout="residual")
        with self.assertRaisesRegex(ValueError, "memory contract"):
            wiki.validate_identity(model, dict(self.contract, base_checkpoint_sha256="d"*64))

    def test_memory_and_policy_mutations_are_detected(self):
        model = self.fit(); policy = self.policy(model)
        changed = copy.deepcopy(model); changed["experts"][0]["values"][0][0] += .01
        with self.assertRaisesRegex(ValueError, "memory"):
            wiki.validate_identity(changed, self.contract, policy)
        changed_policy = copy.deepcopy(policy); changed_policy["entries"]["event:0:0:0"]["score_kw"] += 1
        with self.assertRaisesRegex(ValueError, "policy"):
            wiki.validate_identity(model, self.contract, changed_policy)

    def test_control_conditioning_can_distinguish_opposite_physical_responses(self):
        context = copy.deepcopy(self.context)
        # Identical power/wind keys in both regimes, different observed pitch.
        for name in ("power_ratio", "wind_mean_mps", "wind_std_mps", "power_change_kw", "wind_change_mps"):
            context[name] = np.full(self.n, self.context[name].mean())
        pitch = np.repeat(np.where(np.arange(self.n) < 96, 2., 80.)[:, None], 12, 1)
        context.update(wiki.control_context(pitch, np.ones_like(pitch)))
        truth = self.anchor + np.where(np.arange(self.n) < 96, 50., -50.)[:, None]
        model = wiki.fit_experts(context, self.reference, self.anchor, self.events, truth, self.ids,
            self.contract, neighbors=16, min_neighbor_turbines=2, min_fit_windows=32)
        prediction, _, _ = wiki.candidates(model, context, self.reference, self.anchor, self.events, self.contract)
        np.testing.assert_allclose(prediction[:96, :, 0], 350.)
        np.testing.assert_allclose(prediction[96:, :, 0], 250.)
        # Synthetic mechanism test only, not evidence of real wind-data gains.

    def test_exact_abstention_and_known_harmful_fine_cell(self):
        model = self.fit(); policy = self.policy(model)
        result = wiki.apply_transfer(model, policy, self.context, self.reference,
            self.anchor, self.events, self.contract)
        self.assertTrue((result["action"] > 0).any())
        for h in range(2):
            policy["entries"][f"joint:5:{h}:0"].update(supported=True, accepted=False, score_kw=0.)
        policy["policy_id"] = wiki.digest({k: v for k, v in policy.items() if k != "policy_id"})
        result = wiki.apply_transfer(model, policy, self.context, self.reference,
            self.anchor, self.events, self.contract)
        np.testing.assert_array_equal(result["prediction"], self.reference)
        self.assertFalse(result["action"].any())
        for events in (np.zeros_like(self.events), self.events):
            far = dict(self.context, wind_mean_mps=np.full(self.n, 100.))
            result = wiki.apply_transfer(model, policy, far, self.reference, self.anchor, events, self.contract)
            np.testing.assert_array_equal(result["prediction"], self.reference)

    def test_no_labels_or_turbine_ids_at_inference_and_fit_split_rejection(self):
        self.assertNotIn("truth", inspect.signature(wiki.apply_transfer).parameters)
        self.assertNotIn("turbines", inspect.signature(wiki.apply_transfer).parameters)
        ids = self.ids.copy(); ids[0] = 99
        with self.assertRaisesRegex(ValueError, "Held-out turbine"):
            wiki.fit_experts(self.context, self.reference, self.anchor, self.events,
                self.truth, ids, self.contract, neighbors=16, min_neighbor_turbines=2, min_fit_windows=32)
        model = self.fit()
        with self.assertRaisesRegex(ValueError, "overlap"):
            wiki.fit_transfer_policy(model, [self.block("2023-06-01")]*2, self.contract)

    def test_history_and_anchor_shape_validation(self):
        self.assertNotIn("truth", inspect.signature(wiki.control_context).parameters)
        with self.assertRaises(ValueError):
            wiki.control_context(np.ones((4, 11)), np.ones((4, 11)))
        with self.assertRaises(ValueError):
            wiki.control_context(np.ones((4, 12)), np.full((4, 12), 1.2))
        anchor = self.anchor.copy(); anchor[0, 1] += 1
        with self.assertRaisesRegex(ValueError, "anchor"):
            wiki.fit_experts(self.context, self.reference, anchor, self.events,
                            self.truth, self.ids, self.contract)


if __name__ == "__main__":
    unittest.main()
