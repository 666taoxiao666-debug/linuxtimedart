"""v27 tests: protocol freeze, forward blocks and isolation, per-step scaler, trainer views, decision rules, report, and
(with torch) the controller end-to-end on synthetic data. Torch-free classes run anywhere."""
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils import kelmarsh_forward_block_residual as v27

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/kelmarsh_forward_block_residual_protocol_v27.json"
SCRIPT = ROOT / "scripts/train_kelmarsh_forward_block_residual.py"
V26_PROTOCOL = ROOT / "configs/kelmarsh_anchored_residual_protocol_v26.json"
try:
    import torch
    import sklearn  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

H = np.timedelta64(1, "h")


def protocol():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


class ProtocolTests(unittest.TestCase):
    def test_frozen_hash_and_design(self):
        constant = re.search(r'PROTOCOL_SHA256 = "([0-9a-f]{64})"', SCRIPT.read_text(encoding="utf-8")).group(1)
        self.assertEqual(v27.text_sha256_lf(PROTOCOL), constant)
        p = protocol()
        self.assertTrue(v27.validate_protocol(p))
        self.assertEqual(v27.planned_fits(p), {"steps": 4, "per_step": 11, "total": 44})
        self.assertFalse(p["internal_selection_access"] or p["outer_validation_access"] or p["sealed_test_access"])

    def test_recipe_and_architecture_are_exactly_v26(self):
        old, new = json.loads(V26_PROTOCOL.read_text(encoding="utf-8")), protocol()
        for key in v27.RECIPE_KEYS + ("epoch0_selectable", "first_selectable_epoch"):
            self.assertEqual(old["training"][key], new["candidate"]["training"][key], key)
        for key in ("blocks", "feature_hidden_dim", "norm_eps", "dropout", "trainable_parameters"):
            self.assertEqual(old["candidate"]["residual"][key], new["candidate"]["residual"][key], key)
        self.assertEqual(old["candidate"]["anchor"]["model_args"], new["anchor"]["model_args"])
        self.assertEqual(old["candidate"]["epoch0_identity_tolerance_kw"], new["candidate"]["epoch0_identity_tolerance_kw"])
        self.assertEqual(old["reference"]["dlinear_checkpoints"][0]["fit_window_hash"], new["timeline"]["fold0_fit_window_hash"])
        for key in ("data_path", "data_sha256", "input_len", "pred_len", "train_stride", "feature_columns", "start_utc", "end_utc_exclusive"):
            self.assertEqual(old[key], new[key], key)

    def test_departures_are_refused(self):
        for path, value in ((("residual_seeds",), [2024, 2025, 2026]), (("timeline", "blocks"), 8),
                            (("candidate", "training", "epoch0_selectable"), True), (("budget", "new_fits_max"), 60),
                            (("outer_validation_access",), True), (("anchor", "training", "epoch0_included"), False),
                            (("timeline", "fold0_fit_window_hash"), "0" * 64)):
            p = protocol()
            target = p
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(ValueError, msg=str(path)):
                v27.validate_protocol(p)
        p = protocol()
        p["anchor"]["training"]["learning_rate"] = 3e-4
        with self.assertRaises(ValueError):
            v27.validate_protocol(p)

    def test_views_carry_every_key_the_reused_trainers_read(self):
        p = protocol()
        anchor, residual = v27.anchor_view(p), v27.residual_view(p)
        for key in ("loss_mse_weight", "loss_mae_weight", "learning_rate", "weight_decay", "scheduler_t_max", "scheduler_eta_min",
                    "max_epochs", "patience", "batch_size", "eval_batch_size", "grad_clip_norm"):
            self.assertIn(key, anchor["training"])
            self.assertIn(key, residual["training"])
        self.assertTrue(residual["training"]["first_selectable_epoch"] == 1 and not residual["training"]["epoch0_selectable"])
        self.assertEqual(anchor["arms"]["DLinear"]["expected_parameters"], 200)
        for key in ("blocks", "feature_hidden_dim", "dropout", "norm_eps"):
            self.assertIn(key, residual["candidate"]["residual"])
        for key in ("input_len", "pred_len", "feature_columns", "target_slot", "rated_power_kw"):
            self.assertIn(key, residual)
        self.assertEqual(set(residual["arms"]), {"AnchoredFull", "AnchoredPowerOnly", "DLinear"})


class BlockTests(unittest.TestCase):
    def timeline(self):
        times = pd.date_range("2016-05-01", "2017-01-01", freq="1h", inclusive="left").to_numpy()
        return times[24], times[int(int(len(times) * .6) * .9)]

    def test_registered_edges_and_steps(self):
        t_lo, t_hi = self.timeline()
        edges = v27.block_edges(t_lo, t_hi, 6)
        self.assertEqual([str(e) for e in edges], ["2016-05-02T00", "2016-05-23T21", "2016-06-14T18", "2016-07-06T15",
                                                   "2016-07-28T12", "2016-08-19T09", "2016-09-10T07"])
        expected = {3: ("2016-06-10T09", "2016-06-14T18"), 4: ("2016-06-30T01", "2016-07-06T15"),
                    5: ("2016-07-19T18", "2016-07-28T12"), 6: ("2016-08-08T10", "2016-08-19T09")}
        for k, (cut, end) in expected.items():
            bounds = v27.step_bounds(edges, k)
            self.assertEqual((str(bounds["fit_cut"]), str(bounds["train_end"]), str(bounds["eval_end"])), (cut, end, str(edges[k])))

    def test_masks_are_disjoint_chronological_and_complete_windows_only(self):
        t_lo, t_hi = self.timeline()
        edges = v27.block_edges(t_lo, t_hi, 6)
        first = np.arange(t_lo, t_hi - 3 * H, 3 * H).astype("datetime64[ns]")
        last = first + 3 * H
        for k in (3, 4, 5, 6):
            bounds = v27.step_bounds(edges, k)
            masks = v27.step_masks(first, last, bounds)
            self.assertLess(last[masks["fit"]].max(), bounds["fit_cut"])
            self.assertGreaterEqual(first[masks["selection"]].min(), bounds["fit_cut"])
            self.assertLess(last[masks["selection"]].max(), bounds["train_end"])
            self.assertGreaterEqual(first[masks["evaluation"]].min(), bounds["train_end"])
            self.assertLess(last[masks["evaluation"]].max(), bounds["eval_end"])
            straddle = (first < bounds["train_end"]) & (last >= bounds["train_end"])
            self.assertFalse((masks["evaluation"] & straddle).any() or (masks["selection"] & straddle).any())
        with self.assertRaises(ValueError):
            v27.step_masks(first[:3], last[:3], v27.step_bounds(edges, 6))

    def test_scaler_uses_only_rows_before_the_cut(self):
        dates = (np.datetime64("2016-05-01T00", "h") + np.arange(200) * H).astype("datetime64[ns]")
        raw = np.random.default_rng(0).normal(size=(200, 3))
        raw[:, 2] = 5.
        raw[150:] = 1e6
        raw[3, 0] = np.nan
        mean, scale, rows, constant = v27.scaler_stats(raw, dates, dates[150])
        self.assertEqual(rows, 150)
        self.assertAlmostEqual(mean[0], np.nanmean(raw[:150, 0]))
        self.assertAlmostEqual(scale[1], raw[:150, 1].std())
        self.assertEqual((constant, scale[2]), ([2], 1.0))
        with self.assertRaises(ValueError):
            v27.scaler_stats(raw, dates, dates[0])


def fake_step(k, rng, *, gain=.3, seed_noise=25., harm_lead1=False, power_gain=0., n_issue=120):
    times = (np.datetime64("2016-06-01T00", "h") + (k * 1000 + np.arange(n_issue) * 3) * H)
    times = np.repeat(times, 6)[:, None] + np.arange(1, 5)[None] * H
    n = len(times)
    truth = rng.normal(500, 300, (n, 4))
    anchor = truth + rng.normal(0, 120, (n, 4))
    shift = (truth - anchor)

    def residuals(g):
        out = []
        for _ in range(5):
            p = anchor + g * shift + rng.normal(0, seed_noise, (n, 4))
            if harm_lead1:
                p[:, 0] = anchor[:, 0] + rng.normal(0, 80, n)
            out.append(p)
        return np.stack(out)
    full = residuals(gain)
    return {"truth": truth, "persistence": truth + rng.normal(0, 140, (n, 4)), "anchor": anchor,
            "times": times.astype("datetime64[ns]"), "turbines": np.tile(np.arange(1, 7, dtype=np.int16), n_issue),
            "window_starts": np.arange(n, dtype=np.int64), "seeds": np.array([2024, 2025, 2026, 2027, 2028]),
            "full_selected_epochs": np.array([1, 2, 3, 1, 2]), "power_selected_epochs": np.array([2, 2, 1, 4, 1]),
            "full_best": full, "full_last": full + rng.normal(0, 10, full.shape),
            "power_best": residuals(power_gain), "power_last": residuals(power_gain)}


def small_settings():
    return {"block_length": 8, "replicates": 300, "seed": 2024}


class DecisionTests(unittest.TestCase):
    def run_case(self, **kwargs):
        rng = np.random.default_rng(1)
        arrays = [fake_step(k, rng, **kwargs) for k in (3, 4, 5, 6)]
        records = [v27.step_record(a) for a in arrays]
        pooled = v27.pooled_statistics(arrays, small_settings())
        return v27.decide(records, pooled, protocol()), records, pooled

    def test_consistent_gain_with_seed_noise_is_eligible_for_bagged_confirmation(self):
        decision, records, _ = self.run_case(gain=.3, seed_noise=60.)
        self.assertEqual(decision["outcome"], "BAGGED_RESIDUAL_MAY_ENTER_CONFIRMATION", decision)
        self.assertTrue(decision["flags"]["R3_auxiliary_attribution"])
        self.assertTrue(all(r["full_seed_mean"]["mae"] < r["full_single_mean"]["mae"] for r in records))

    def test_no_seed_noise_means_no_bagging_evidence(self):
        decision, _, _ = self.run_case(gain=.3, seed_noise=0.)
        self.assertEqual(decision["outcome"], "RESIDUAL_MAY_ENTER_CONFIRMATION_WITHOUT_BAGGING", decision)

    def test_ensemble_evidence_cannot_release_a_harmful_fixed_seed(self):
        _, records, pooled = self.run_case(gain=.3, seed_noise=0.)
        # Force R2 false, keep all ensemble evidence unchanged, but the actual fixed
        # seed fails. The original implementation incorrectly released this route.
        for r in records:
            r["full_single"][0]["rmse"] = r["anchor"]["rmse"] + 1
        pooled["full_fixed_seed_vs_anchor"]["absolute_error"]["ci95_low"] = -1
        decision = v27.decide(records, pooled, protocol())
        self.assertTrue(decision["flags"]["R1_residual_transfer"])
        self.assertFalse(decision["flags"]["R2_averaging_benefit"])
        self.assertFalse(decision["flags"]["R5_fixed_seed_transfer"])
        self.assertEqual(decision["outcome"], "STOP_RESIDUAL_LINE_ON_THIS_DATA")

    def test_metrics_refuse_broadcast_or_misaligned_seeds(self):
        with self.assertRaisesRegex(ValueError, "broadcasting"):
            v27.basic_metrics(np.ones((5, 1)), np.ones((5, 4)))
        a = fake_step(3, np.random.default_rng(0))
        a["power_best"] = a["power_best"][:2]
        with self.assertRaisesRegex(ValueError, "Aligned"):
            v27.step_record(a)

    def test_harmful_or_near_term_damaging_residual_stops_the_line(self):
        self.assertEqual(self.run_case(gain=-.2)[0]["outcome"], "STOP_RESIDUAL_LINE_ON_THIS_DATA")
        decision, _, pooled = self.run_case(gain=.3, harm_lead1=True)
        self.assertLess(pooled["full_mean_vs_anchor"]["lead1_mae_gain_kw"], 0)
        self.assertFalse(decision["flags"]["R4_near_term"])
        self.assertEqual(decision["outcome"], "STOP_RESIDUAL_LINE_ON_THIS_DATA")

    def test_aux_claim_requires_full_to_beat_power_only(self):
        decision, _, _ = self.run_case(gain=.3, power_gain=.3, seed_noise=60.)
        self.assertFalse(decision["flags"]["R3_auxiliary_attribution"])
        self.assertFalse(decision["auxiliary_claim_allowed"])

    def test_diagnostics_shapes(self):
        rng = np.random.default_rng(2)
        a = fake_step(3, rng)
        c = v27.correction_diagnostics(a)
        self.assertEqual(len(c["full"]["corr_seed_mean_correction_vs_anchor_error_by_lead"]), 4)
        self.assertGreater(c["full"]["corr_seed_mean_correction_vs_anchor_error_by_lead"][0], 0.5)
        r = v27.regime_diagnostics(a, 2050.)
        self.assertEqual(set(r["power_regime"]), set(v27.POWER_LABELS))
        self.assertEqual(sum(v["windows"] for v in r["turbine"].values()), len(a["truth"]))


def build_fake_run(root):
    p = protocol()
    p["evaluation"]["pooled_statistics"]["replicates"] = 300
    run = root / "run"
    rng = np.random.default_rng(3)
    arrays = [fake_step(k, rng, seed_noise=60.) for k in (3, 4, 5, 6)]
    hashes = {}
    for k, a in zip((3, 4, 5, 6), arrays):
        path = run / f"steps/k{k}/evaluation_predictions.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **a)
        hashes[f"steps/k{k}/evaluation_predictions.npz"] = v27.file_sha256(path)
    records = [v27.step_record(a) for a in arrays]
    pooled = v27.pooled_statistics(arrays, p["evaluation"]["pooled_statistics"])
    v27.write_json(run / "protocol.json", {"protocol": p, "protocol_id": p["protocol_id"]})
    v27.write_json(run / "blocks_manifest.json", {"edges_utc": ["e0", "e1"]})
    v27.write_json(run / "decision.json", {"decision": v27.decide(records, pooled, p), "pooled": pooled})
    v27.write_json(run / "training_completed.json", {"hashes": hashes})
    return run


class ReportTests(unittest.TestCase):
    def test_report_recomputes_and_writes_outputs(self):
        from scripts import report_kelmarsh_forward_block_residual as report
        with tempfile.TemporaryDirectory() as tmp:
            run = build_fake_run(Path(tmp))
            result = report.build_report(run)
            self.assertEqual(result["decision"]["outcome"], "BAGGED_RESIDUAL_MAY_ENTER_CONFIRMATION")
            frame = pd.read_csv(run / "metrics_by_step.csv")
            self.assertEqual(sorted(frame.step.unique()), [3, 4, 5, 6])
            self.assertIn("AnchoredFull last-epoch 5-seed mean (diagnostic)", set(frame.forecast))
            for name in ("step_gains", "lead_gains"):
                self.assertGreater((run / "figures" / f"{name}.png").stat().st_size, 1000)
            self.assertIn("training-side diagnostic", (run / "report.md").read_text(encoding="utf-8"))
            self.assertTrue((run / "report_manifest.json").exists())

    def test_tampered_arrays_or_decision_are_refused(self):
        from scripts import report_kelmarsh_forward_block_residual as report
        with tempfile.TemporaryDirectory() as tmp:
            run = build_fake_run(Path(tmp))
            (run / "steps/k4/evaluation_predictions.npz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "changed or missing"):
                report.build_report(run)
        with tempfile.TemporaryDirectory() as tmp:
            run = build_fake_run(Path(tmp))
            record = v27.read_json(run / "decision.json")
            record["decision"]["outcome"] = "STOP_RESIDUAL_LINE_ON_THIS_DATA"
            v27.write_json(run / "decision.json", record)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                report.build_report(run)


# =========================================================================== torch: data view and controller
def synthetic_frame(hours=1200):
    dates = pd.date_range("2016-05-01", periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    parts = []
    for turbine in range(1, 7):
        t = np.arange(hours, dtype=float)
        wind = 6 + 2 * np.sin(t / 7 + turbine) + .1 * rng.normal(size=hours)
        parts.append(pd.DataFrame({"timestamp_hour_end_utc": dates, "turbine_id": turbine, "wind_speed_mps": wind,
                                   "ambient_temp_c": 12 + np.sin(t / 24), "nacelle_position_deg": (t * 3) % 360,
                                   "pitch_a_deg": 1., "pitch_b_deg": 1., "pitch_c_deg": 1., "rotor_speed_rpm": 10 + wind / 3,
                                   "power_kw": 300 + 200 * np.sin(t / 7 + turbine + .5) + 10 * rng.normal(size=hours)}))
    return pd.concat(parts, ignore_index=True)


def synthetic_series(fold, frame):
    from data_provider.kelmarsh_hourly import KelmarshSeries
    with patch("data_provider.kelmarsh_hourly.pd.read_csv", return_value=frame.copy()):
        return KelmarshSeries("unused", fold=fold, input_len=24, horizon=4, stride=3,
                              start="2016-05-01T00:00:00Z", end="2016-06-20T00:00:00Z")


@unittest.skipUnless(HAS_TORCH, "torch/sklearn unavailable on this host")
class DataViewTests(unittest.TestCase):
    def test_prefix_loader_preserves_queue_without_loading_later_numeric_rows(self):
        from scripts import train_kelmarsh_forward_block_residual as task
        frame = synthetic_frame()
        old = synthetic_series(0, frame)
        old_queue = task.candidate_windows(old, protocol(), verify_hash=False)
        cutoff = old.bounds["calibration_early_end"]
        later = pd.to_datetime(frame.timestamp_hour_end_utc, utc=True).dt.tz_localize(None) >= cutoff
        # A later value that cannot be converted to float must never reach feature engineering.
        frame["power_kw"] = frame["power_kw"].astype(object)
        frame.loc[later, "power_kw"] = "FORBIDDEN_SUFFIX"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.csv"
            frame.to_csv(path, index=False)
            new = task.FitPrefixSeries(path, fold=0, input_len=24, horizon=4, stride=3,
                                      start="2016-05-01T00:00:00Z", end="2016-06-20T00:00:00Z")
        queue = task.candidate_windows(new, protocol(), verify_hash=False)
        np.testing.assert_array_equal(queue["starts"], old_queue["starts"])
        np.testing.assert_allclose(new.raw[new.dates < cutoff], old.raw[old.dates < cutoff], rtol=0, atol=1e-12)
        self.assertTrue(np.isnan(new.raw[new.dates >= cutoff]).all())
        self.assertFalse(hasattr(new, "scaler"))
        self.assertEqual(new.audit["suffix_numeric_rows_read"], 0)

    def test_controller_and_dependencies_are_part_of_fit_identity(self):
        from scripts import train_kelmarsh_forward_block_residual as task
        original = task.training_code_sha256()
        real = task.v27.text_sha256_lf
        for filename in ("scripts/train_kelmarsh_forward_block_residual.py", "layers/Autoformer_EncDec.py"):
            with patch.object(task.v27, "text_sha256_lf", side_effect=lambda p: "changed" if Path(p) == ROOT / filename else real(p)):
                self.assertNotEqual(task.training_code_sha256(), original)

    def test_forward_windows_match_kelmarsh_windows_with_the_same_scaler(self):
        from scripts import train_kelmarsh_forward_block_residual as task
        from utils.kelmarsh_backbone_information import inner_datasets
        series = synthetic_series(0, synthetic_frame())
        fit = inner_datasets(series)["fit"]
        values = task.standardized(series.raw, series.scaler.mean_, series.scaler.scale_)
        view = task.ForwardWindows(series, fit.window_starts, values, series.scaler.mean_, series.scaler.scale_, "fit")
        for i in (0, 7, len(fit) - 1):
            for a, b in zip(fit[i][:2], view[i][:2]):
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
        for a, b in zip(fit.truth_and_anchor(), view.truth_and_anchor()):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(fit.target_times(), view.target_times())
        windows = task.candidate_windows(series, protocol(), verify_hash=False)
        np.testing.assert_array_equal(windows["starts"], fit.window_starts)
        with self.assertRaisesRegex(ValueError, "differs from the registered"):
            task.candidate_windows(series, protocol(), verify_hash=True)


@unittest.skipUnless(HAS_TORCH, "torch/sklearn unavailable on this host")
class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import train_kelmarsh_forward_block_residual as task
        cls.task = task
        torch.set_num_threads(2)
        cls.series = synthetic_series(0, synthetic_frame())
        cls.gpu_patch = patch.object(task.v26run, "gpu_snapshot", return_value=None)
        cls.gpu_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.gpu_patch.stop()

    def make(self, tmp):
        p = protocol()
        p["residual_seeds"] = [2024, 2025]
        for block in (p["anchor"]["training"], p["candidate"]["training"]):
            block.update(max_epochs=2, scheduler_t_max=2)
        p["budget"]["new_fits_max"] = 20
        p["evaluation"]["pooled_statistics"]["replicates"] = 200
        series = copy.copy(self.series)
        self.requested = []

        def spy(name):
            self.requested.append(name)
            raise AssertionError("v27 must never instantiate a named subset")
        series.subset = spy
        return self.task.V27Run(Path(tmp) / "run", protocol=p, protocol_sha="test-protocol", series_factory=lambda fold: series,
                                require_preflight=False, write_pointer=False, lock_path=Path(tmp) / "lock",
                                verify_fold0_hash=False)

    def test_end_to_end_isolated_paired_and_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp)
            self.assertEqual(run.run(), "COMPLETED")
            self.assertEqual(self.requested, [])
            self.assertEqual(len(run.registered_fits()), 4 * (1 + 2 * 2))
            self.assertEqual(run.fit_calls[:3], [(3, "DLinear", 2024), (3, "AnchoredFull", 2024), (3, "AnchoredPowerOnly", 2024)])
            blocks = v27.read_json(run.dir / "blocks_manifest.json")
            self.assertFalse(blocks["internal_selection_or_outer_instantiated"])
            for k in (3, 4, 5, 6):
                step = blocks["steps"][str(k)]
                self.assertTrue(all(step["windows"][part] > 0 for part in ("fit", "selection", "evaluation")))
                self.assertLess(step["bounds_utc"]["fit_cut"], step["bounds_utc"]["train_end"])
                anchor = v27.read_json(run.dir / f"steps/k{k}/anchor/DLinear/trained.json")
                for seed in (2024, 2025):
                    full = v27.read_json(run.dir / f"steps/k{k}/s{seed}/AnchoredFull/trained.json")
                    power = v27.read_json(run.dir / f"steps/k{k}/s{seed}/AnchoredPowerOnly/trained.json")
                    self.assertEqual(full["init_state_sha256"], power["init_state_sha256"])
                    self.assertEqual(full["anchor_state_sha256"], anchor["best_state_sha256"])
                    self.assertGreaterEqual(full["best_epoch"], 1)
                    self.assertEqual(full["epoch0_max_abs_diff_vs_anchor_kw"], 0.0)
                with np.load(run.dir / f"steps/k{k}/evaluation_predictions.npz") as saved:
                    self.assertEqual(saved["full_best"].shape[0], 2)
                    self.assertGreaterEqual(str(saved["times"][:, 0].min()), step["bounds_utc"]["train_end"])
            decision = v27.read_json(run.dir / "decision.json")["decision"]
            self.assertIn(decision["outcome"], {"BAGGED_RESIDUAL_MAY_ENTER_CONFIRMATION",
                                                "RESIDUAL_MAY_ENTER_CONFIRMATION_WITHOUT_BAGGING",
                                                "STOP_RESIDUAL_LINE_ON_THIS_DATA"})
            self.assertEqual(v27.read_json(run.dir / "status.json")["report"], "REPORT_COMPLETED")
            again = self.make(tmp)
            with patch.object(self.task.v25run, "fit_arm", side_effect=AssertionError("retrained")), \
                    patch.object(self.task.v26run, "fit_arm", side_effect=AssertionError("retrained")):
                self.assertEqual(again.run(), "COMPLETED")
            (run.dir / "result.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Registered artifact"):
                again.run()

    def test_preflight_is_required_before_any_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp)
            run.require_preflight = True
            with patch.object(self.task, "PREFLIGHT_DIR", Path(tmp) / "pf"):
                with self.assertRaisesRegex(ValueError, "preflight"):
                    run.run()
            self.assertFalse(list(run.dir.rglob("trained.json")))


if __name__ == "__main__":
    unittest.main()
