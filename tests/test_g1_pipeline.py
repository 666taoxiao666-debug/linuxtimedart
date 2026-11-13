"""Causal boundaries, controls, freeze and statistic regression tests."""
import unittest
import tempfile
import os
from pathlib import Path

import numpy as np

from scripts import prepare_g1_batches as p
from scripts import train_g1 as t


class TemporalTests(unittest.TestCase):
    def test_release_boundary(self):
        hours = np.arange(np.datetime64("2020-01-02T00"), np.datetime64("2020-01-03T00"), np.timedelta64(1, "h"))
        age = (hours - p.available_cycle(hours)) / np.timedelta64(1, "h")
        self.assertTrue(np.array_equal(age, np.tile(np.arange(6, 12), 4)))

    def test_lead_mask_no_extrapolation(self):
        y, mask, audit = p.interpolate_leads(np.arange(6.), [2, 3, 4.5, 18, 19])
        np.testing.assert_array_equal(mask, [0, 1, 1, 1, 0])
        np.testing.assert_allclose(y, [0, 0, .5, 5, 0])
        self.assertEqual((audit["min_read_lead"], audit["max_read_lead"]), (3, 18))

    def test_missing_endpoint(self):
        v = np.array([1, np.nan, 3, 4, 5, 6])
        _, m, _ = p.interpolate_leads(v, [3, 4, 6, 9])
        np.testing.assert_array_equal(m, [1, 0, 0, 1])

    def test_bilinear_descending_latitude(self):
        lat, lon = np.array([2., 1., 0.]), np.array([359., 0., 1.])
        v = lat[:, None] + ((lon+180) % 360-180)[None, :]
        np.testing.assert_allclose(p.bilinear(v, lat, lon, [[.5, -.5], [2, 1]]), [0, 3])
        with self.assertRaises(ValueError):
            p.bilinear(v, lat, lon, [[3, 0]])

    def test_half_year_hour_end(self):
        m = p.period_mask(["2022-07-01T00", "2022-07-01T01"], ["2021-01-01", "2022-07-01"], hour_end=True)
        np.testing.assert_array_equal(m, [True, False])


class SelectionTests(unittest.TestCase):
    def test_validation_selection_only(self):
        self.assertEqual(t.select_information_control({"I": [2]*3, "I_w": [1]*3, "R_sp": [1]*3})[0], "I_w")
        with self.assertRaises(ValueError):
            t.select_information_control({"I": [2], "I_w": [1]*3, "R_sp": [1]*3})

    @unittest.skipIf(os.name == "nt", "registered statistics module requires Linux fcntl")
    def test_max_p_holm_and_direction(self):
        results = {f: {q: dict(newey_west={"p_value_two_sided": .001}, gain=2., ci95_low=1., reference_mean_loss=100.)
                      for q in ("Q1", "Q3", "Q4", "Q5")} for f in ("Kelmarsh", "Penmanshiel")}
        self.assertTrue(t.joint_decisions(results)["enter_G2"])
        results["Penmanshiel"]["Q5"]["newey_west"]["p_value_two_sided"] = .2
        decision = t.joint_decisions(results)
        self.assertEqual(decision["Q5"]["p_max"], .2)
        self.assertFalse(decision["enter_G2"])
        results["Penmanshiel"]["Q1"]["ci95_low"] = -1
        self.assertTrue(t.joint_decisions(results)["stop_G"])

    def test_freeze_tamper_and_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d)/"checkpoint"
            f.write_bytes(b"one")
            freeze = {"fits": [{"key": "a", "checkpoint": "checkpoint", "sha256": p.digest(f)}]}
            t.check_freeze(freeze, 1, d)
            f.write_bytes(b"two")
            with self.assertRaises(ValueError):
                t.check_freeze(freeze, 1, d)
            with self.assertRaises(ValueError):
                t.check_freeze({"fits": freeze["fits"]*2}, 2, d)

    def test_unfrozen_formal_refused(self):
        with self.assertRaises(ValueError):
            t.formal_gate("absent", "0"*64)


def synthetic_pack(n=18):
    from tests.test_nwp_displacement import make_batch
    rng = np.random.default_rng(33)
    batch = make_batch(B=n, S=3)
    batch.update(oF=np.ones((n, 3))*5, oF_mask=np.ones((n, 3)))
    x21 = rng.normal(size=(n, 24, 21)).astype(np.float32)
    x50 = np.concatenate([x21[..., :-1], np.zeros((n, 24, 29)), x21[..., -1:]], -1).astype(np.float32)
    batch.update(x21=x21, x50=x50, truth=rng.normal(size=(n, 4)).astype(np.float32),
                 geo=np.array([[.2, 0, 1], [.4, 1, 0], [.7, -1, 0]], dtype=np.float32),
                 ws_scale=np.array(2.), power_mean=np.array(0.), power_scale=np.array(1.))
    return batch


class PreparationTests(unittest.TestCase):
    def test_calibration_fit_only_and_missing_cycle(self):
        hours = np.arange(np.datetime64("2020-01-02T00"), np.datetime64("2020-02-12T00"), np.timedelta64(1, "h"))
        class Forecast:
            def get(self, cycle):
                if cycle == np.datetime64("2020-02-01T00"):
                    return np.full((6, 3), np.nan)
                a = int(cycle.astype("datetime64[h]").astype(int))
                return np.tile((5+np.sin(a/50)+p.LEADS/10)[:, None], (1, 3))
        wind = 7+np.sin(np.arange(len(hours))/50)
        obs = np.tile(wind[:, None], (1, 2))
        bounds = ["2020-01-02", "2020-02-02"]
        first = p.prepare_hourly(hours, Forecast(), obs, wind, np.zeros_like(obs), bounds)
        changed = obs.copy()
        changed[~p.period_mask(hours, bounds, hour_end=True)] = 999
        second = p.prepare_hourly(hours, Forecast(), changed, wind, np.zeros_like(obs), bounds)
        self.assertEqual(first["calibration"], second["calibration"])
        ix = np.array([40, 200, 720])
        batch = p.station_batch(first, ix)
        p.nd.validate_batch(batch, 2, "M_self")
        np.testing.assert_allclose(batch["oS"], first["calibrated"][ix[:, None]+np.arange(-3, 0), 1:].transpose(0, 2, 1))
        self.assertTrue((batch["cycle_ok"][:2] == 1).all())
        x = np.ones((len(ix), 24, 21), dtype=np.float32)
        indices = ix[:, None]+np.arange(-23, 1)
        augmented = p.nwp_input(x, first, indices)
        self.assertEqual(augmented.shape, (3, 24, 50))
        np.testing.assert_array_equal(augmented[..., -1], x[..., -1])

    def test_csv_filters_before_numeric_parse(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d)/"input.csv"
            f.write_text("label_year,power_kw\n2020,2\n2024,SEALED_SENTINEL\n", encoding="utf-8")
            frame = p.development_csv(f, 2020)
            self.assertEqual(frame.power_kw.tolist(), [2])

    def test_sealed_forecast_refused_before_open(self):
        store = object.__new__(p.PointForecasts)
        with self.assertRaises(ValueError):
            store.get("2023-12-31T06")


@unittest.skipIf(p.nd._torch is None, "server torch required")
class TorchPipelineTests(unittest.TestCase):
    def test_all_ten_arms_fit_predict(self):
        import json
        config = json.loads((p.ROOT/"configs/g1_protocol.json").read_text())["fit"]
        config = {**config, "max_epochs": 1, "batch_size": 8, "patience": 1}
        train, val = synthetic_pack(), synthetic_pack(8)
        with tempfile.TemporaryDirectory() as d:
            for arm in p.nd.ARMS:
                with self.subTest(arm=arm):
                    result = t.fit_one(arm, 2024, train, val, config, Path(d)/arm, "cpu")
                    self.assertTrue(np.isfinite(result["validation_mae"]))
                    self.assertEqual(p.digest(Path(d)/arm/"selected.pt"), result["sha256"])

    def test_mechanism_diagnostic_contract(self):
        import json
        pack = synthetic_pack(30)
        pack.update(truth_wind=pack["gF"][:, 6:10]+.5, truth_mask=np.ones((30, 4)),
                    issue=(np.datetime64("2020-06-01")+np.arange(30)*np.timedelta64(24, "h")).astype("datetime64[ns]").astype(np.int64))
        config = json.loads((p.ROOT/"configs/g1_protocol.json").read_text())["fit"]
        model = t.make_model("M", pack, config, "cpu")
        result = t.mechanism_diagnostics(pack, model, pack["geo"], 2.)
        self.assertLessEqual(result["best_both_mae"], result["no_correction_mae"])
        self.assertEqual(result["paired_windows"], 30)


if __name__ == "__main__":
    unittest.main()
