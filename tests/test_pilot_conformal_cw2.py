import tempfile
from pathlib import Path
import unittest
import numpy as np
from scripts import pilot_conformal_cw2 as c2


def pack(n, date, seed=0):
    rng = np.random.default_rng(seed)
    issue = np.datetime64(date, "ns") + np.arange(n) * np.timedelta64(1, "h")
    pred = rng.uniform(200.0, 1500.0, size=(n, 4))
    truth = pred + rng.normal(0.0, 100.0, size=(n, 4))
    feat = np.zeros((n, 20))
    feat[:, :11] = rng.normal(0.0, 1.0, size=(n, 11))
    sec = rng.integers(0, 9, size=n)
    feat[:, 11:20] = np.eye(9)[sec]
    return {
        "prediction": pred,
        "truth": truth,
        "anchor": pred[:, 0] - 10.0,
        "issue": issue,
        "times": issue[:, None] + np.arange(1, 5) * np.timedelta64(1, "h"),
        "turbines": np.ones(n, int),
        "features": feat,
    }


def smoke(out):
    s = pack(500, "2021-01-01", seed=1)
    tm = pack(100, "2021-12-01", seed=2)
    e = pack(30, "2022-01-01", seed=3)
    methods = c2.BASE + c2.SCALE_ARMS
    radii, caps, bundle = c2.intervals(s, tm, e, 2022, methods)
    error = np.abs(e["truth"] - e["prediction"])
    rep = c2.cw1.reports(radii, caps, error)
    strata = c2.descriptive_strata(radii, caps, e["truth"], e["prediction"], e["anchor"], e["features"])
    ci = c2.cw1.paired_ci(radii["N-P"] - radii["N-R"], e["issue"], out / "bootstrap.npz", 100)
    result = {
        "status": "PASSED",
        "smoke": True,
        "methods": list(rep),
        "floors": {arm: bundle[arm]["floors"] for arm in c2.SCALE_ARMS},
        "strata_keys": list(strata["N-R"]),
        "ci": ci,
    }
    c2.save(out / "result.json", result)
    return result


class CW2Tests(unittest.TestCase):
    def test_feature_dimensions_and_missing_flag(self):
        p = pack(10, "2022-01-01")
        p["features"][3, 2] = np.nan
        f_np = c2.scale_features(p, 0, "N-P")
        f_nrs = c2.scale_features(p, 0, "N-Rs")
        f_nr = c2.scale_features(p, 0, "N-R")
        self.assertEqual(f_np.shape, (10, 1))
        self.assertEqual(f_nrs.shape, (10, 6))
        self.assertEqual(f_nr.shape, (10, 18))
        self.assertTrue((f_nrs[:, 1:] >= 0).all())
        self.assertEqual(f_nr[3, 8], 1.0)
        self.assertEqual(f_nr[0, 8], 0.0)

    def test_source_only_fit_and_sigma_floor(self):
        s = pack(450, "2021-01-01", seed=10)
        tm = pack(80, "2021-12-01", seed=11)
        e1 = pack(20, "2022-01-01", seed=12)
        e2 = {k: v.copy() for k, v in e1.items()}
        e2["features"][5:] += 500.0
        e2["truth"][5:] += 5000.0
        r1, _, b1 = c2.intervals(s, tm, e1, 2022, ("N-P", "N-R", "N-Rs"))
        r2, _, b2 = c2.intervals(s, tm, e2, 2022, ("N-P", "N-R", "N-Rs"))
        for arm in c2.SCALE_ARMS:
            np.testing.assert_array_equal(r1[arm][:5], r2[arm][:5])
            for h in range(4):
                med = np.median(np.abs(s["truth"][:, h] - s["prediction"][:, h]))
                self.assertAlmostEqual(b1[arm]["floors"][h], 0.1 * med)
                self.assertTrue((b1[arm]["sigma_test"][h] >= 0.1 * med).all())

    def test_unit_scale_matches_B2_and_rule_F_cap(self):
        s = pack(50, "2021-01-01", seed=20)
        tm = pack(60, "2021-12-01", seed=21)
        e = pack(40, "2022-01-01", seed=22)
        e["truth"][:] = 1e6  # Force ACI alpha_t toward 0.005 so quantile hits +inf mass
        se = np.abs(s["truth"] - s["prediction"])
        unit_bundle = {
            arm: {
                "scores": [se[:, h] / 2.0 for h in range(4)],
                "sigma_test": [np.full(len(e["issue"]), 2.0) for _ in range(4)],
                "floors": [1.0] * 4,
                "importance": [{}] * 4,
            }
            for arm in c2.SCALE_ARMS
        }
        radii, caps, _ = c2.intervals(s, tm, e, 2022, ("B2-A", "N-P"), scale_bundle=unit_bundle)
        np.testing.assert_allclose(radii["N-P"], radii["B2-A"], rtol=1e-12, atol=1e-9)
        np.testing.assert_array_equal(caps["N-P"], caps["B2-A"])
        self.assertTrue(caps["N-P"].any())
        self.assertTrue(np.isfinite(radii["N-P"]).all())

    def test_selection_requires_all_four_cohorts_and_tie_prefers_N_Rs(self):
        def cohort(nr_wink, nrs_wink, cov=0.90):
            return {
                "mean_90": {
                    "N-P": {"coverage": 0.90, "winkler_kw": 100.0},
                    "B3-A": {"coverage": 0.90, "winkler_kw": 98.0},
                    "N-R": {"coverage": cov, "winkler_kw": nr_wink},
                    "N-Rs": {"coverage": cov, "winkler_kw": nrs_wink},
                }
            }

        # Exact tie on all 4 cohorts -> N-Rs wins
        cohorts = [cohort(95.0, 95.0) for _ in range(4)]
        sel, gains = c2.select_candidate(cohorts)
        self.assertEqual(sel, "N-Rs")
        # If one cohort fails Winkler < min(N-P, B3-A)=98.0 for N-Rs, N-R wins
        cohorts[2] = cohort(96.0, 98.0)
        sel2, _ = c2.select_candidate(cohorts)
        self.assertEqual(sel2, "N-R")
        # If both fail at least one cohort -> None
        cohorts[0] = cohort(99.0, 99.0)
        sel3, _ = c2.select_candidate(cohorts)
        self.assertIsNone(sel3)

    def test_verdict_uses_min_of_NP_and_B3A(self):
        means = {m: {"coverage": 0.90, "winkler_kw": 100.0} for m in c2.BASE + ("N-P", "N-R")}
        means["B3-A"]["winkler_kw"] = 95.0
        means["N-P"]["winkler_kw"] = 92.0
        means["N-R"]["winkler_kw"] = 89.0  # <= 0.97 * 92 = 89.24
        cis = {"N-P": {"ci95_low_kw": 0.5}, "B3-A": {"ci95_low_kw": 2.0}}
        v = c2.verdict(means, "N-R", cis)
        self.assertEqual(v["matched_comparator"], "N-P")
        self.assertTrue(v["pass_all"])
        cis["N-P"]["ci95_low_kw"] = -0.1
        self.assertFalse(c2.verdict(means, "N-R", cis)["C3"])


if __name__ == "__main__":
    unittest.main()
