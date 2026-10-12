import unittest

from scripts.summarize_sdwpf_knowledge_transport import calibration_summary


class KnowledgeTransportSummaryTests(unittest.TestCase):
    def test_missing_support_is_not_reported_as_zero_harm(self):
        model = dict(contract=dict(event_ids=["idle"], pred_len=1),
                     experts=[dict(active=True, fit_windows=150)])
        missing = dict(windows=0, turbines=0, turbine_balanced_gain_kw=None,
                       turbine_balanced_mse_gain=None, positive_turbine_fraction=None)
        row = dict(accepted=False, blocks=[missing, missing])
        policy = dict(min_windows=64, min_turbines=8, min_positive_fraction=2/3,
                      entries={"event:0:0:0":row, "event:0:0:1":row})
        result = calibration_summary(model, policy)
        step = result[0]["event_level_horizons"][0]
        self.assertEqual(step["reasons"], ["block1:insufficient_support", "block2:insufficient_support"])
        self.assertIsNone(step["blocks"][0]["turbine_balanced_gain_kw"])

    def test_supported_negative_gain_is_distinguished_from_missing_data(self):
        model = dict(contract=dict(event_ids=["gust"], pred_len=1),
                     experts=[dict(active=True, fit_windows=150)])
        bad = dict(windows=100, turbines=10, turbine_balanced_gain_kw=-5.,
                   turbine_balanced_mse_gain=-10., positive_turbine_fraction=.4)
        row = dict(accepted=False, blocks=[bad, bad])
        policy = dict(min_windows=64, min_turbines=8, min_positive_fraction=2/3,
                      entries={"event:0:0:0":row, "event:0:0:1":row})
        reasons = calibration_summary(model, policy)[0]["event_level_horizons"][0]["reasons"]
        self.assertIn("block2:nonpositive_MAE_gain", reasons)
        self.assertNotIn("block2:insufficient_support", reasons)


if __name__ == "__main__":
    unittest.main()
