import copy
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from tests.test_kelmarsh_incremental_wiki import fixture
from utils import certified_numeric_residual as guard


def pack_for_clocks(clocks=36, start="2016-09-01T00"):
    pack = fixture(clocks * 5)
    issue = np.repeat(np.datetime64(start, "h") + np.arange(clocks) * np.timedelta64(3, "h"), 5)
    pack["times"] = issue[:, None] + np.array([1, 2], dtype="timedelta64[h]")[None]
    pack["reference"][:] = 100.
    pack["numeric"][:] = 120.
    pack["truth"][:] = 110.
    return pack


SETTINGS = dict(temporal_blocks=3, min_windows=24, min_turbines=3, min_issue_times=4, positive_fraction=2/3)
CUTOFF = "2016-10-01"


class CertifiedNumericTests(unittest.TestCase):
    def test_train_shrinkage_finds_convex_solution_without_grid_search(self):
        pack = pack_for_clocks()
        fit = guard.fit_shrinkage(pack, train_end=CUTOFF)
        np.testing.assert_allclose(fit["alpha"], [.5, .5], atol=1e-12)
        prediction = guard.predict_residual(pack["reference"], pack["numeric"], fit["alpha"])
        np.testing.assert_allclose(prediction, pack["truth"], atol=1e-11)

    def test_harmful_residual_is_not_forced_by_loss_or_positive_gate(self):
        pack = pack_for_clocks()
        pack["truth"][:] = 90.
        fit = guard.fit_shrinkage(pack, train_end=CUTOFF)
        np.testing.assert_array_equal(fit["alpha"], [0., 0.])
        np.testing.assert_array_equal(guard.predict_residual(pack["reference"], pack["numeric"], fit["alpha"]), pack["reference"])

    def test_zero_reference_error_and_zero_delta_fall_back_exactly(self):
        pack = pack_for_clocks()
        pack["truth"][:] = pack["reference"]
        self.assertEqual(guard.fit_shrinkage(pack, train_end=CUTOFF)["alpha"], [0., 0.])
        pack["numeric"][:] = pack["reference"]
        self.assertEqual(guard.fit_shrinkage(pack, train_end=CUTOFF)["alpha"], [0., 0.])

    def test_partition_is_by_clock_not_turbine_order_and_purges_crossing_targets(self):
        pack = pack_for_clocks()
        issue = np.repeat(np.datetime64("2016-09-01T00", "h") + np.arange(36) * np.timedelta64(1, "h"), 5)
        pack["times"] = issue[:, None] + np.array([1, 2], dtype="timedelta64[h]")[None]
        parts, report = guard.chronological_parts(pack, 3, train_end=CUTOFF)
        self.assertGreater(report["purged_windows"], 0)
        self.assertEqual(len(parts), 3)
        for first, second in zip(parts, parts[1:]):
            self.assertLessEqual(first["times"].max(), (second["times"][:, 0] - np.timedelta64(1, "h")).min())

    def test_all_temporal_blocks_required_not_only_aggregate_gain(self):
        pack = pack_for_clocks()
        # Raw residual helps early 2/3 but is harmful in the last block.
        pack["truth"][:] = 120.
        pack["truth"][-60:, 0] = 90.
        policy = guard.certify(pack, dict(alpha=[1., 1.]), {"checkpoint": "actual"}, SETTINGS, train_end=CUTOFF)
        self.assertEqual(policy["accepted"], [False, True])
        self.assertEqual(policy["weights"], [0., 1.])
        future = pack_for_clocks(6, "2016-09-20T00")
        predicted = guard.apply(policy, future["reference"], future["numeric"], future["times"], {"checkpoint": "actual"})
        np.testing.assert_array_equal(predicted[:, 0], future["reference"][:, 0])
        np.testing.assert_array_equal(predicted[:, 1], future["numeric"][:, 1])

    def test_checkpoint_policy_and_outcome_availability_fail_closed(self):
        pack = pack_for_clocks()
        policy = guard.certify(pack, dict(alpha=[.5, .5]), {"checkpoint": "a"}, SETTINGS, train_end=CUTOFF)
        future = pack_for_clocks(6, "2016-09-20T00")
        with self.assertRaisesRegex(ValueError, "identity"):
            guard.apply(policy, future["reference"], future["numeric"], future["times"], {"checkpoint": "b"})
        forged = copy.deepcopy(policy)
        forged["weights"][0] = 0.
        with self.assertRaisesRegex(ValueError, "identity"):
            guard.apply(forged, future["reference"], future["numeric"], future["times"], {"checkpoint": "a"})
        with self.assertRaisesRegex(ValueError, "unavailable"):
            guard.apply(policy, pack["reference"], pack["numeric"], pack["times"], {"checkpoint": "a"})

    def test_turbine6_outer_times_and_sparse_evidence_rejected(self):
        pack = pack_for_clocks()
        pack["turbines"][0] = 6
        with self.assertRaisesRegex(ValueError, "donor"):
            guard.fit_shrinkage(pack, train_end=CUTOFF)
        pack["turbines"][0] = 1
        with self.assertRaisesRegex(ValueError, "outer"):
            guard.certify(pack, dict(alpha=[.5, .5]), {}, SETTINGS, train_end="2016-09-01")
        small = pack_for_clocks(9)
        policy = guard.certify(small, dict(alpha=[.5, .5]), {}, SETTINGS, train_end=CUTOFF)
        self.assertEqual(policy["accepted"], [False, False])

    def test_inference_does_not_read_outcomes_or_mutate_source_arrays(self):
        pack = pack_for_clocks()
        before = guard.array_digest(pack)
        policy = guard.certify(pack, dict(alpha=[.5, .5]), {}, SETTINGS, train_end=CUTOFF)
        self.assertEqual(before, guard.array_digest(pack))
        future = pack_for_clocks(6, "2016-09-20T00")
        first = guard.apply(policy, future["reference"], future["numeric"], future["times"], {})
        future["truth"][:] = -1e9
        np.testing.assert_array_equal(first, guard.apply(policy, future["reference"], future["numeric"], future["times"], {}))

    def test_all_pack_fields_are_identity_bound(self):
        pack = pack_for_clocks()
        original = guard.array_digest(pack)
        pack["context"]["power_ratio"][0] += 1
        self.assertNotEqual(original, guard.array_digest(pack))
        original = guard.array_digest(pack)
        pack["numeric"][0, 0] += 1
        self.assertNotEqual(original, guard.array_digest(pack))

    def test_both_matched_arms_train_on_new_common_base_and_resume(self):
        from scripts.train_kelmarsh_incremental_wiki import train_arm, infer
        pack = pack_for_clocks()
        pack["numeric"] = guard.predict_residual(pack["reference"], pack["numeric"], [0., .25])
        pack["overlap"][:, 1] = False
        source_hash = guard.array_digest(pack)
        spec = dict(seed=2024, training=dict(width=8, dropout=0., correction_bound_capacity=.2, epochs=2,
            batch_size=128, learning_rate=.001, weight_decay=.0001, mix_mse_weight=.2, grad_clip=1.))
        with tempfile.TemporaryDirectory() as directory:
            counts = []
            for name in ("WikiIncrement", "ZeroMemoryIncrement"):
                model, done = train_arm(name, pack, pack, 2050., spec, Path(directory) / name,
                    dict(numerical_gate_id="new-common-base", arm=name), torch.device("cpu"))
                counts.append(sum(p.numel() for p in model.parameters()))
                result = infer(model, pack, 2050., torch.device("cpu"), name == "WikiIncrement")
                np.testing.assert_array_equal(result[:, 1], pack["numeric"][:, 1])
                self.assertEqual(done["epochs_completed"], 2)
            self.assertEqual(counts[0], counts[1])
        self.assertEqual(source_hash, guard.array_digest(pack))


if __name__ == "__main__":
    unittest.main()
