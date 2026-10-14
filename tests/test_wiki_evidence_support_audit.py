import copy
import unittest

import numpy as np
from utils.wiki_evidence_support_audit import clock_support, certificate_support_audit


class EvidenceSupportAuditTests(unittest.TestCase):
    def fixture(self):
        issue = np.repeat(np.array(["2016-09-01T03", "2016-09-01T06"], dtype="datetime64[h]"), 5)
        times = issue[:, None] + np.array([1, 2], dtype="timedelta64[h]")[None]
        pack = dict(times=times, turbines=np.tile(np.arange(1, 6), 2),
                    overlap=np.ones((10, 2, 1), bool), context=dict(power_bin=np.zeros(10, int)))
        row = dict(windows=10, turbines=5, supported=True, accepted=True, gain_kw=1., mse_gain=2.)
        return pack, dict(entries={"event:0:0:0": row})

    def test_correlated_turbines_do_not_count_as_distinct_issue_times(self):
        pack, cert = self.fixture()
        result = certificate_support_audit(pack, cert, train_end="2016-09-02")
        row = result["supported_cells"]["event:0:0:0"]
        self.assertEqual((row["windows"], row["turbines"], row["unique_issue_times"], row["utc_day_blocks"]), (10, 5, 2, 1))
        self.assertEqual(row["largest_day_window_share"], 1.)
        self.assertIsNone(result["gain_by_time_block"])

    def test_mismatched_membership_cannot_be_reported_as_reproduced(self):
        pack, cert = self.fixture()
        cert["entries"]["event:0:0:0"]["windows"] += 1
        with self.assertRaisesRegex(ValueError, "membership"):
            certificate_support_audit(pack, cert, train_end="2016-09-02")

    def test_outer_clocks_and_heldout_turbine_are_rejected(self):
        pack, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "outer validation"):
            clock_support(pack["times"], pack["turbines"], train_end="2016-09-01T08")
        pack["turbines"][0] = 6
        with self.assertRaisesRegex(ValueError, "donor"):
            clock_support(pack["times"], pack["turbines"], train_end="2016-09-02")

    def test_empty_cell_is_finite_and_does_not_mutate_inputs(self):
        pack, cert = self.fixture()
        before = copy.deepcopy(pack)
        cert["entries"]["power:1:0:0"] = dict(windows=0, turbines=0, supported=False, accepted=False)
        result = certificate_support_audit(pack, cert, train_end="2016-09-02")
        self.assertEqual(len(result["supported_cells"]), 1)
        for key in ("times", "turbines", "overlap"):
            np.testing.assert_array_equal(pack[key], before[key])

    def test_metadata_only_mask_matches_actual_quality_readout(self):
        from tests.test_kelmarsh_wiki_quality import WikiQualityTests
        from scripts.audit_kelmarsh_wiki_evidence import metadata_overlap
        from utils import kelmarsh_wiki_quality as quality
        _, contract, expert, query, info = WikiQualityTests().fixture()
        expected = quality.candidates(expert, query, info, contract)["overlap"]
        np.testing.assert_array_equal(metadata_overlap(expert, query, info, contract), expected)
        query["reference"][:] = 1e6
        np.testing.assert_array_equal(metadata_overlap(expert, query, info, contract), expected)


if __name__ == "__main__":
    unittest.main()
