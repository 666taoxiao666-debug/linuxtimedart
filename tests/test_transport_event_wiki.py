import copy
import inspect
import unittest

import numpy as np

from utils.transport_event_wiki import (fit_experts, candidates, fit_transfer_policy,
    apply_transfer, digest, validate_identity, calibration_blocks)


class TransportEventWikiTests(unittest.TestCase):
    def setUp(self):
        self.contract = dict(base_checkpoint_sha256="a"*64, base_manifest_sha256="b"*64,
            event_config_sha256="c"*64, rated_power_kw=1500., pred_len=3,
            event_ids=["gust", "idle"], source_turbines=list(range(1, 9)),
            physical_coordinate_version="observed_last12_v1")
        rng = np.random.default_rng(2024)
        n = 160
        self.context = dict(power_ratio=np.full(n, .2), wind_mean_mps=rng.uniform(5, 8, n),
            wind_std_mps=rng.uniform(1, 2, n), power_change_kw=rng.uniform(-50, 50, n),
            wind_change_mps=rng.uniform(-1, 1, n), power_bin=np.ones(n, int),
            joint_bin=np.full(n, 5, int))
        self.reference = np.full((n, 3), 300.)
        self.anchor = self.reference.copy()
        self.truth = self.reference+40.+10.*self.context["wind_change_mps"][:, None]
        self.events = np.column_stack((np.ones(n, bool), np.zeros(n, bool)))
        self.turbines = np.tile(np.arange(1, 9), 20)
        self.model = fit_experts(self.context, self.reference, self.anchor, self.events,
            self.truth, self.turbines, self.contract, min_fit_windows=32)

    def block(self, day):
        return dict(context=self.context, reference=self.reference, anchor=self.anchor,
            events=self.events, truth=self.truth, turbines=self.turbines,
            times=np.full((160, 3), np.datetime64(day, "ns")))

    def policy(self):
        return fit_transfer_policy(self.model, [self.block("2023-06-01"), self.block("2023-06-03")],
                                   self.contract, min_windows=32, min_turbines=8)

    def test_actual_expert_base_and_policy_identity_are_bound(self):
        policy = self.policy()
        result = apply_transfer(self.model, policy, self.context, self.reference,
                                self.anchor, self.events, self.contract)
        self.assertGreater((result["action"] > 0).sum(), 0)
        self.assertLess(np.abs(result["prediction"]-self.truth).mean(), 5.)
        changed = copy.deepcopy(self.model)
        changed["experts"][0]["coefficients"][0][0] += .001
        with self.assertRaisesRegex(ValueError, "Frozen expert"):
            validate_identity(changed, self.contract, policy)
        new_base = dict(self.contract, base_checkpoint_sha256="d"*64)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            apply_transfer(self.model, policy, self.context, self.reference, self.anchor, self.events, new_base)
        self.assertNotIn("truth", inspect.signature(apply_transfer).parameters)
        self.assertNotIn("turbines", inspect.signature(apply_transfer).parameters)

    def test_overlap_physical_evidence_and_exact_abstention(self):
        policy = self.policy()
        no_events = np.zeros_like(self.events)
        result = apply_transfer(self.model, policy, self.context, self.reference, self.anchor,
                                no_events, self.contract)
        np.testing.assert_array_equal(result["prediction"], self.reference)
        far = dict(self.context, wind_mean_mps=np.full(160, 100.))
        result = apply_transfer(self.model, policy, far, self.reference, self.anchor,
                                self.events, self.contract)
        self.assertFalse(result["history_overlap"].any())
        np.testing.assert_array_equal(result["prediction"], self.reference)

    def test_harmful_supported_fine_cell_cannot_backoff_to_good_coarse_cell(self):
        policy = self.policy()
        for h in range(3):
            policy["entries"][f"joint:5:{h}:0"].update(supported=True, accepted=False, score_kw=0.)
        policy["policy_id"] = digest({k:v for k,v in policy.items() if k != "policy_id"})
        result = apply_transfer(self.model, policy, self.context, self.reference,
                                self.anchor, self.events, self.contract)
        np.testing.assert_array_equal(result["prediction"], self.reference)
        coarse = apply_transfer(self.model, policy, self.context, self.reference,
                                self.anchor, self.events, self.contract, event_only=True)
        self.assertTrue((coarse["action"] > 0).any())

    def test_held_out_labels_and_overlapping_time_evidence_are_rejected(self):
        held_out = self.turbines.copy(); held_out[0] = 99
        with self.assertRaisesRegex(ValueError, "Held-out turbine"):
            fit_experts(self.context, self.reference, self.anchor, self.events,
                        self.truth, held_out, self.contract)
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_transfer_policy(self.model, [self.block("2023-06-01")]*2, self.contract)
        first, second, cutoff = calibration_blocks(np.array([
            ["2023-06-01", "2023-06-01"], ["2023-06-02", "2023-06-04"],
            ["2023-06-05", "2023-06-05"]], dtype="datetime64[ns]"), np.ones(3, bool))
        self.assertFalse((first & second).any())
        self.assertFalse(first[1] or second[1])


if __name__ == "__main__":
    unittest.main()
