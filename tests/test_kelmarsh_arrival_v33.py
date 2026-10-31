"""v33 tests: masks, geometry, prior reference, decisions (torch-free); with torch: module == NumPy prior,
epoch-0 identity with the base, permutation invariance, parameter parity, and an end-to-end smoke run."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from utils import kelmarsh_station_history_v33 as hist

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/kelmarsh_arrival_v33_k1.json").read_text(encoding="utf-8"))
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def bank_from(stations, frames, hours=None):
    hours = hours if hours is not None else pd.date_range("2020-01-01", periods=60, freq="h")
    obs = {s["src_id"]: frames[i].set_index(hours) for i, s in enumerate(stations)}
    return hist.build_bank(obs, stations, hours, np.ones(len(hours), bool)), hours


def frame(ws, d, p=1010.0, n=60):
    return pd.DataFrame({"wind_speed": np.full(n, ws, float), "wind_direction": np.full(n, d, float),
                         "msl_pressure": np.full(n, p, float)})


class GeometryTests(unittest.TestCase):
    def test_upwind_and_signed_cross_distance_mirror(self):
        off = hist.station_offsets([{"distance_km": 50, "bearing_deg": 60}, {"distance_km": 50, "bearing_deg": 120}])
        d_par, d_perp = hist.project(off, np.array([90.0, 90.0]))
        self.assertTrue((d_par > 0).all())                       # both in the eastern (upwind) half-plane
        self.assertAlmostEqual(d_par[0], d_par[1], places=9)
        self.assertAlmostEqual(d_perp[0], -d_perp[1], places=9)   # mirror stations remain distinguishable
        self.assertNotAlmostEqual(d_perp[0], 0.0)

    def test_content_invariant_under_global_rotation(self):
        st = [{"src_id": i, "distance_km": 30 + 20 * i, "bearing_deg": 40 * i} for i in range(4)]
        rot = [{**s, "bearing_deg": (s["bearing_deg"] + 73) % 360} for s in st]
        dirs = [200, 210, 190, 205]
        b1, hours = bank_from(st, [frame(6 + i, dirs[i]) for i in range(4)])
        b2, _ = bank_from(rot, [frame(6 + i, (dirs[i] + 73) % 360) for i in range(4)])
        r1, ref1, ages = hist.token_rows(b1, hours[30:32], 1)
        r2, ref2, _ = hist.token_rows(b2, hours[30:32], 1)
        c1, p1, v1 = hist.gather(b1, r1, ref1, ages)
        c2, p2, v2 = hist.gather(b2, r2, ref2, ages)
        np.testing.assert_allclose(c1, c2, atol=1e-5)
        np.testing.assert_allclose(p1, p2, atol=1e-5)
        np.testing.assert_array_equal(v1, v2)


class MaskAndPriorTests(unittest.TestCase):
    def setUp(self):
        self.st = [{"src_id": 1, "distance_km": 10, "bearing_deg": 270}, {"src_id": 2, "distance_km": 60, "bearing_deg": 90}]

    def test_filled_speed_never_opens_the_prior(self):
        # Codex counterexample: raw speed missing (content filled with the fit mean), d_par = 10 km upwind
        f1 = frame(np.nan, 270)
        f1["wind_direction"] = 270.0
        bank, hours = bank_from(self.st, [f1, frame(8, 270)])
        rows, ref, ages = hist.token_rows(bank, hours[40:41], 1)
        content, phys, valid = hist.gather(bank, rows, ref, ages)
        q, support, qt = hist.prior_numpy(phys, valid, 1.0)
        station = np.tile(np.arange(2), 24)
        self.assertTrue((q[0, station == 0] == 0).all())
        self.assertTrue((phys[0, station == 0, 5] == 0).all())         # base_ok false
        self.assertTrue(valid[0, station == 0].any())                  # pressure still valid content

    def test_regional_direction_fallback_uses_raw_valid_observations_only(self):
        f1 = frame(6, np.nan)
        bank, _ = bank_from(self.st, [f1, frame(8, 90)])
        self.assertTrue(np.allclose(bank.prior_dir[:, 0], 90.0))
        both = frame(6, np.nan), frame(8, np.nan)
        bank2, _ = bank_from(self.st, list(both))
        self.assertTrue(np.isnan(bank2.prior_dir).all())
        self.assertTrue((bank2.ref_flag == 2).all())

    def test_no_support_gives_b2_weights(self):
        bank, hours = bank_from(self.st, [frame(1.0, 270), frame(1.0, 90)])          # calm: below 2 m/s
        rows, ref, ages = hist.token_rows(bank, hours[40:41], 1)
        _, phys, valid = hist.gather(bank, rows, ref, ages)
        q, support, qt = hist.prior_numpy(phys, valid, 1.0)
        self.assertFalse(support.any())
        self.assertTrue((qt == 1).all())

    def test_arrival_support_exists_for_reachable_station(self):
        bank, hours = bank_from(self.st, [frame(5, 90), frame(10, 90)])              # wind from east; station 2 upwind
        rows, ref, ages = hist.token_rows(bank, hours[40:41], 1)
        _, phys, valid = hist.gather(bank, rows, ref, ages)
        q, support, qt = hist.prior_numpy(phys, valid, 1.0)
        self.assertTrue(support.any())
        self.assertTrue((q[0, np.tile(np.arange(2), 24) == 0] == 0).all())         # station 1 is downwind
        q_geo, _, _ = hist.prior_numpy(phys, valid, 1.0, variant="geo180")
        self.assertTrue((q_geo[0, np.tile(np.arange(2), 24) == 1] == 0).all())

    def test_token_rows_are_strictly_past(self):
        bank, hours = bank_from(self.st, [frame(5, 90), frame(10, 90)])
        rows, ref, ages = hist.token_rows(bank, hours[30:31], 1)
        self.assertEqual(ages[0], 1)
        self.assertTrue((rows[0] <= 29).all() and rows[0].max() == 29 and rows[0].min() == 6)
        self.assertEqual(ref[0], 29)
        rows_early, _, _ = hist.token_rows(bank, hours[5:6], 1)
        self.assertTrue((rows_early[0][ages > 5] == -1).all())


class DecisionTests(unittest.TestCase):
    def test_codex_counterexample_and_branches(self):
        cfg = CONFIG["decision"]
        m = {"B0": {"rmse": 100, "mae": 60}, "B2": {"rmse": 100, "mae": 60}, "A": {"rmse": 98.5, "mae": 62}}
        self.assertEqual(hist.a1_decision(m, cfg)["branch"], "D3")
        m1 = {"B0": {"rmse": 100, "mae": 60}, "B2": {"rmse": 100, "mae": 60}, "A": {"rmse": 98.5, "mae": 59.5}}
        self.assertEqual(hist.a1_decision(m1, cfg)["branch"], "D1")
        m2 = {"B0": {"rmse": 100, "mae": 60}, "B2": {"rmse": 98.5, "mae": 59.9}, "A": {"rmse": 98.4, "mae": 61}}
        self.assertEqual(hist.a1_decision(m2, cfg)["branch"], "D2")
        self.assertEqual(hist.a1_decision({"B0": m["B0"], "A": m["A"]}, cfg)["branch"], "D0")
        self.assertEqual(hist.a1_decision({**m, "A": {"rmse": np.nan, "mae": 1}}, cfg)["branch"], "D0")

    def test_kappa_star_ties(self):
        self.assertEqual(hist.kappa_star({0.5: 1.9, 1.0: 1.95, 2.0: 1.90005}), 0.5)
        self.assertEqual(hist.kappa_star({0.5: 1.9, 1.0: 1.90005, 2.0: 1.9}), 1.0)
        self.assertEqual(hist.kappa_star({0.5: 1.9, 2.0: 1.9}), 0.5)
        with self.assertRaises(ValueError):
            hist.kappa_star({1.0: np.nan})


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TorchTests(unittest.TestCase):
    def setUp(self):
        from models.TSMixerBasicSmall import Model as TSMixer
        torch.manual_seed(0)
        self.base = TSMixer(seq_len=24, channels=21, pred_len=4, blocks=2, ff_dim=32, dropout=.1, norm_eps=1e-3, target_slot=20)
        st = [{"src_id": i, "distance_km": 20 + 15 * i, "bearing_deg": 30 * i} for i in range(6)]
        rng = np.random.default_rng(0)
        hours = pd.date_range("2020-01-01", periods=80, freq="h")
        frames = [pd.DataFrame({"wind_speed": rng.uniform(0, 12, 80), "wind_direction": rng.uniform(0, 360, 80),
                                "msl_pressure": 1010 + rng.normal(0, 3, 80)}) for _ in st]
        frames[0].loc[::5, "wind_speed"] = np.nan
        self.bank, self.hours = bank_from(st, frames, hours)
        rows, ref, ages = hist.token_rows(self.bank, hours[40:48], 1)
        self.content, self.phys, self.valid = hist.gather(self.bank, rows, ref, ages)
        self.x = torch.randn(8, 24, 21)

    def module(self, **kw):
        from models.UpstreamArrivalResidual import ArrivalResidual
        return ArrivalResidual(self.base, **kw)

    def test_prior_matches_numpy(self):
        for variant in ("full", "age0", "hflat", "geo180"):
            m = self.module(prior="arrival", kappa=1.7, variant=variant)
            q, s, qt = m.arrival_q(torch.tensor(self.phys), torch.tensor(self.valid))
            qn, sn, qtn = hist.prior_numpy(self.phys, self.valid, 1.7, variant=variant)
            np.testing.assert_allclose(q.numpy(), qn, atol=1e-5)
            np.testing.assert_array_equal(s.numpy(), sn)
            np.testing.assert_allclose(qt.numpy(), qtn, rtol=1e-4, atol=1e-4)

    def test_epoch0_identity_permutation_and_parity(self):
        from models.UpstreamArrivalResidual import base_forward, trainable_parameters
        self.base.eval()
        a = self.module(prior="arrival", kappa="learned")
        b2 = self.module(prior="none")
        args = (torch.tensor(self.content), torch.tensor(self.phys), torch.tensor(self.valid))
        with torch.no_grad():
            ref, _ = base_forward(self.base, self.x)
            np.testing.assert_allclose(a(self.x, *args)[..., -1].numpy(), ref.numpy(), atol=1e-6)
        self.assertEqual(trainable_parameters(a), trainable_parameters(b2) + 1)
        torch.nn.init.normal_(a.head[-1].weight)
        perm = torch.randperm(args[0].shape[1])
        with torch.no_grad():
            y1 = a(self.x, *args)
            y2 = a(self.x, args[0][:, perm], args[1][:, perm], args[2][:, perm])
        np.testing.assert_allclose(y1.numpy(), y2.numpy(), atol=1e-5)
        self.assertTrue(all(not p.requires_grad for p in self.base.parameters()))

    def test_smoke_end_to_end(self):
        from scripts import train_kelmarsh_arrival_v33 as ctl
        with tempfile.TemporaryDirectory() as tmp:
            report = ctl.run(Path(tmp), smoke=True, device="cpu")
            self.assertIn(report["decision"]["branch"], ("D0", "D1", "D2", "D3"))
            self.assertTrue((Path(tmp) / "k1_summary.md").exists())
            manifest = json.loads((Path(tmp) / "data_manifest.json").read_text())
            self.assertIn("determinism", manifest)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)

    def test_real_loader_supplies_parser_year_without_opening_data(self):
        from scripts import train_kelmarsh_arrival_v33 as ctl
        seen = []

        def station_files(root, ids, years, hashes, required):
            self.assertTrue(required)
            self.assertEqual(len(years), 1)
            year = years[0]
            return [{"src_id": ids[0], "path": Path(f"station_{year}.csv"), "sha256": "synthetic"}], []

        def observations(files, ids):
            seen.extend(entry["year"] for entry in files)
            for entry in files:
                self.assertTrue(entry["path"].name.endswith(f"_{entry['year']}.csv"))
            return {}

        with patch.object(ctl.u31, "file_sha256", return_value=CONFIG["prepared_csv_sha256"]), \
                patch.object(ctl.u31, "text_sha256_lf", return_value=json.loads(
                    (ROOT / CONFIG["data_source_protocol"]).read_text())["data"]["upstream"]["download_manifest_sha256_lf"]), \
                patch.object(ctl.u31, "read_json", return_value={"files": []}), \
                patch.object(ctl.u31, "station_files", side_effect=station_files), \
                patch.object(ctl.u31, "load_station_observations", side_effect=observations), \
                patch.object(ctl.u31, "hourly_upstream", return_value=None), \
                patch.object(ctl.u31, "KelmarshUpstreamSeries", return_value=None):
            ctl.load_real(CONFIG)
        self.assertEqual(seen, CONFIG["station_years"])
        self.assertLessEqual(max(seen), 2022)

    def test_scores_reject_dropped_or_misaligned_all_cohort(self):
        from scripts import train_kelmarsh_arrival_v33 as ctl
        truth = np.array([[0., 10., 20., 30.], [10., 20., 30., 40.]])
        self.assertEqual(ctl.scores(truth, truth, 2050.)["mae"], 0.)
        missing = truth.copy()
        missing[0, 0] = np.nan
        for pred, target in ((missing, truth), (truth, missing), (truth[:1], truth)):
            with self.assertRaisesRegex(ValueError, "complete ALL cohort"):
                ctl.scores(pred, target, 2050.)


if __name__ == "__main__":
    unittest.main()
