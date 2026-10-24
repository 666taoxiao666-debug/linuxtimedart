"""v25 contract tests: protocol freeze, data boundaries, gates, paired statistics, access audit, devices."""
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils import kelmarsh_backbone_information as v25
from utils.paired_forecast_statistics import paired_forecast_statistics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/kelmarsh_backbone_information_protocol_v25.json"
SCRIPT = ROOT / "scripts/train_kelmarsh_backbone_information.py"
try:
    import torch  # noqa: F401
    import sklearn  # noqa: F401
    HAS_TORCH = True
except ImportError:  # local design sandboxes may lack torch; the training host never does
    HAS_TORCH = False


def protocol():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def rec(fold, seed, arm, mae, rmse):
    return {"fold": fold, "seed": seed, "arm": arm, "mae": mae, "rmse": rmse}


PERSISTENCE = {f: {"mae": 200., "rmse": 300.} for f in (0, 1, 2)}
G1 = {"seed": 2024, "min_folds_jointly_better_than_DLinear": 2}
G2 = {"seeds": [2024, 2025, 2026], "min_seed_averaged_folds_jointly_better_than_DLinear": 2,
      "min_fold_averaged_seeds_jointly_better_than_DLinear": 2}


def synthetic_frame(hours=1200, seed=0):
    dates = pd.date_range("2016-05-01", periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
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


class ProtocolTests(unittest.TestCase):
    def test_frozen_hash_matches_runner_constant_and_design(self):
        constant = re.search(r'PROTOCOL_SHA256 = "([0-9a-f]{64})"', SCRIPT.read_text(encoding="utf-8")).group(1)
        self.assertEqual(v25.text_sha256_lf(PROTOCOL), constant)
        p = protocol()
        self.assertTrue(v25.validate_protocol(p))
        self.assertEqual(v25.planned_new_fits(p), {"A1": 6, "A2_max": 18, "max_new_fits": 24})
        self.assertEqual(p["reference"]["protocol_sha256"], v25.text_sha256_lf(ROOT / p["reference"]["protocol_path"]))
        self.assertFalse(p["wiki_enabled"] or p["sealed_test_access"] or p["prediction_clipping"])
        self.assertEqual([c["fold"] for c in p["reference"]["dlinear_checkpoints"]], [0, 1, 2])

    def test_any_departure_from_design_is_refused(self):
        for path, value in ((("candidate", "blocks"), 3), (("candidate", "feature_hidden_dim"), 64),
                            (("training", "learning_rate"), 3e-4), (("seeds", "A2"), [2025, 2026, 2027]),
                            (("budget", "new_fits_max"), 30), (("candidate", "expected_trainable_parameters"), 5000),
                            (("training", "max_epochs"), 40), (("statistics", "block_length"), 12)):
            p = protocol()
            target = p
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(ValueError, msg=str(path)):
                v25.validate_protocol(p)
        p = protocol()
        p["gates"]["primary_candidate"] = "TSMixerPowerOnly"
        with self.assertRaises(ValueError):
            v25.validate_protocol(p)

    def test_reference_values_match_local_v20_mirror_when_available(self):
        mirror = ROOT / "output/kelmarsh_hourly/recent_numeric_v20_review"
        if not mirror.exists():
            self.skipTest("v20 review mirror not present on this host")
        p = protocol()
        self.assertEqual(v25.file_sha256(mirror / "completed.json"), p["reference"]["completed_sha256"])
        for spec in p["reference"]["dlinear_checkpoints"]:
            run = mirror / "runs" / f"f{spec['fold']}_s2024" / "DLinear"
            trained = v25.read_json(run / "trained.json")
            entry = next(h["selection"] for h in v25.read_json(run / "training_history.json") if h["epoch"] == spec["best_epoch"])
            self.assertEqual(trained["artifacts"]["checkpoint.pt"], spec["checkpoint_sha256"])
            self.assertEqual(trained["best_epoch"], spec["best_epoch"])
            self.assertEqual(trained["contract"]["fit_window_hash"], spec["fit_window_hash"])
            self.assertEqual(trained["contract"]["selection_window_hash"], spec["selection_window_hash"])
            self.assertEqual((entry["mae"], entry["rmse"]), (spec["internal_mae_kw"], spec["internal_rmse_kw"]))


class GateTests(unittest.TestCase):
    def g1(self, full, dlinear=((160., 240.),) * 3, persistence=PERSISTENCE):
        records = [rec(f, 2024, "TSMixerFull", *full[f]) for f in range(3)]
        records += [rec(f, 2024, "DLinear", *dlinear[f]) for f in range(3)]
        return v25.gate_g1(records, persistence, G1, [0, 1, 2])

    def test_g1_requires_macro_mae_and_rmse_and_two_joint_folds(self):
        self.assertTrue(self.g1([(150., 230.)] * 3)["passed"])
        self.assertFalse(self.g1([(150., 245.)] * 3)["passed"], "MAE better but RMSE worse")
        one_lucky = self.g1([(100., 150.), (161., 241.), (161., 241.)])
        self.assertTrue(one_lucky["checks"]["macro_mae_rmse_better_than_dlinear"])
        self.assertFalse(one_lucky["passed"], "macro win carried by one fold")
        self.assertFalse(self.g1([(150., 230.)] * 3, persistence={f: {"mae": 140., "rmse": 400.} for f in range(3)})["passed"])

    def test_numeric_tie_is_not_an_improvement(self):
        result = self.g1([(160. - 5e-7, 240. - 5e-7)] * 3)
        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"]["jointly_better_folds_vs_dlinear"], 0)
        self.assertFalse(v25.improves(10., 10. - 1e-6))
        self.assertTrue(v25.improves(10., 10. - 2e-6))

    def test_missing_or_nonfinite_records_never_pass(self):
        records = [rec(f, 2024, "TSMixerFull", 1., 1.) for f in (0, 1)] + [rec(f, 2024, "DLinear", 9., 9.) for f in range(3)]
        result = v25.gate_g1(records, PERSISTENCE, G1, [0, 1, 2])
        self.assertFalse(result["passed"])
        self.assertFalse(result["complete"])
        with self.assertRaises(ValueError):
            v25.improves(float("nan"), 1.)
        with self.assertRaises(ValueError):
            v25.gate_g1(records + [rec(0, 2024, "DLinear", 1., 1.)], PERSISTENCE, G1, [0, 1, 2])

    def g2(self, full_by_seed, dlinear=(160., 240.)):
        records = []
        for seed, values in full_by_seed.items():
            for f in range(3):
                records += [rec(f, seed, "TSMixerFull", *values[f]), rec(f, seed, "DLinear", *dlinear)]
        return v25.gate_g2(records, PERSISTENCE, G2, [0, 1, 2])

    def test_g2_seed_average_cannot_hide_widespread_damage(self):
        good = [(150., 230.)] * 3
        self.assertTrue(self.g2({2024: good, 2025: good, 2026: good})["passed"])
        masked = self.g2({2024: [(60., 100.)] * 3, 2025: [(170., 250.)] * 3, 2026: [(170., 250.)] * 3})
        self.assertTrue(masked["checks"]["macro_mae_rmse_better_than_dlinear"])
        self.assertEqual(masked["checks"]["fold_averaged_seeds_jointly_better"], 1)
        self.assertFalse(masked["passed"])
        fold_masked = self.g2({s: [(60., 100.), (170., 250.), (170., 250.)] for s in (2024, 2025, 2026)})
        self.assertFalse(fold_masked["passed"])
        missing = self.g2({2024: good, 2025: good})
        self.assertFalse(missing["passed"])
        self.assertFalse(missing["complete"])

    def test_information_evidence_is_independent_of_the_gate(self):
        records = [rec(f, 2024, "TSMixerFull", 150., 230.) for f in range(3)]
        records += [rec(f, 2024, "DLinear", 160., 240.) for f in range(3)]
        records += [rec(f, 2024, "TSMixerPowerOnly", 149., 229.) for f in range(3)]
        self.assertTrue(v25.gate_g1(records, PERSISTENCE, G1, [0, 1, 2])["passed"])
        info = v25.info_i1(records, {"seed": 2024}, [0, 1, 2])
        self.assertFalse(info["supported"])
        self.assertIn("never changes the primary candidate", info["interpretation"])
        all_seeds = [dict(r, seed=s) for r in records for s in (2024, 2025, 2026)]
        self.assertFalse(v25.info_i2(all_seeds, {"seeds": [2024, 2025, 2026]}, [0, 1, 2])["supported"])


class StatisticsTests(unittest.TestCase):
    def cohort(self, issues=60, turbines=6, seed=0):
        rng = np.random.default_rng(seed)
        starts = np.repeat(pd.date_range("2016-09-01", periods=issues, freq="3h").to_numpy(), turbines)
        common = np.repeat(np.cumsum(rng.normal(size=issues)), turbines)  # autocorrelated, shared across turbines
        truth = 500 + 50 * common[:, None] + 20 * rng.normal(size=(issues * turbines, 4))
        return starts, truth, rng

    def test_seed_loss_statistics_reproduce_library_for_one_seed(self):
        starts, truth, rng = self.cohort()
        a, b = truth + rng.normal(0, 30, truth.shape), truth + rng.normal(0, 35, truth.shape)
        library = paired_forecast_statistics(a, b, truth, pd.DataFrame({"forecast_start": starts}), block_length=8,
                                             replicates=500, seed=2024, estimand="window_uniform")
        mine = v25.paired_loss_statistics(np.abs(a - truth), np.abs(b - truth), starts, block_length=8, replicates=500, seed=2024)
        self.assertAlmostEqual(mine["gain"], library["primary_gain_kw"], places=12)
        self.assertAlmostEqual(mine["ci95_low"], library["time_block_bootstrap"]["ci95_low_kw"], places=12)
        self.assertAlmostEqual(mine["ci95_high"], library["time_block_bootstrap"]["ci95_high_kw"], places=12)
        self.assertAlmostEqual(mine["newey_west"]["p_value_two_sided"], library["dm_newey_west"]["p_value_two_sided_normal"], places=12)
        self.assertEqual(mine["issue_time_clusters"], 60)
        self.assertEqual(mine["windows"], 360)
        self.assertEqual(mine["newey_west"]["lag"], 7)

    def test_identical_forecasts_have_zero_gain_and_interval(self):
        starts, truth, rng = self.cohort()
        a = truth + rng.normal(0, 30, truth.shape)
        mine = v25.paired_loss_statistics(np.abs(a - truth), np.abs(a - truth), starts, block_length=8, replicates=200, seed=1)
        self.assertEqual((mine["gain"], mine["ci95_low"], mine["ci95_high"]), (0., 0., 0.))
        rmse = v25.rmse_difference_bootstrap((a - truth) ** 2, (a - truth) ** 2, starts, block_length=8, replicates=200, seed=1)
        self.assertEqual((rmse["rmse_gain_kw"], rmse["ci95_low_kw"], rmse["ci95_high_kw"]), (0., 0., 0.))

    def test_rmse_interval_is_not_the_mae_interval(self):
        starts, truth, rng = self.cohort(seed=3)
        n = len(truth)
        # same absolute error everywhere, but the reference concentrates it in rare large errors
        cand_err = np.full(truth.shape, 10.)
        ref_err = np.zeros(truth.shape)
        ref_err.flat[::10] = 100.  # exactly 10% of points: identical MAE, very different RMSE
        mae = v25.paired_loss_statistics(np.abs(cand_err), np.abs(ref_err), starts, block_length=8, replicates=300, seed=2)
        rmse = v25.rmse_difference_bootstrap(cand_err ** 2, ref_err ** 2, starts, block_length=8, replicates=300, seed=2)
        self.assertAlmostEqual(rmse["rmse_candidate_kw"], 10.)
        self.assertAlmostEqual(rmse["rmse_reference_kw"], np.sqrt((ref_err ** 2).mean()))
        self.assertGreater(rmse["rmse_gain_kw"], 15.)
        self.assertLess(abs(mae["gain"]), 2.)
        self.assertNotAlmostEqual(rmse["ci95_low_kw"], mae["ci95_low"])
        self.assertIn("NOT a reuse", rmse["method"])
        self.assertEqual(n, 360)

    def test_misaligned_rows_are_refused(self):
        starts, truth, _ = self.cohort()
        meta = {"times": starts.reshape(-1, 1), "turbines": np.tile(np.arange(6), 60), "window_starts": np.arange(360),
                "truth": truth, "persistence": truth + 1}
        self.assertTrue(v25.assert_aligned(meta, copy.deepcopy(meta)))
        order = np.random.default_rng(0).permutation(360)
        shuffled = {k: v[order] for k, v in meta.items()}
        with self.assertRaises(ValueError):
            v25.assert_aligned(meta, shuffled)
        changed = copy.deepcopy(meta)
        changed["truth"][0, 0] += 1e-9
        with self.assertRaises(ValueError):
            v25.assert_aligned(meta, changed)

    def test_holm_adjustment(self):
        self.assertEqual(v25.holm_adjust([.01, .04, .03]), [.03, .06, .06])
        self.assertEqual(v25.holm_adjust([.5, .9]), [1., 1.])
        self.assertEqual(v25.holm_adjust([.2, .01, .04]), [.2, .03, .08])
        with self.assertRaises(ValueError):
            v25.holm_adjust([.1, float("nan")])

    def test_point_metrics_undefined_r2_and_skill(self):
        truth = np.full((10, 4), 100.)
        metrics = v25.point_metrics(truth + 1, truth, truth + 2, 2050.)
        self.assertIsNone(metrics["r2"])
        self.assertFalse(metrics["r2_defined"])
        self.assertAlmostEqual(metrics["mae_skill_pct"], 50.)
        self.assertEqual(metrics["within_5pct_capacity_pct"], 100.)

    def test_issue_coverage_reports_gaps(self):
        times = pd.to_datetime(["2016-01-01 00:00", "2016-01-01 03:00", "2016-01-01 12:00", "2016-01-02 00:00"]).to_numpy()
        coverage = v25.issue_time_coverage(np.repeat(times, 6), 3)
        self.assertEqual(coverage["issue_times"], 4)
        self.assertEqual(coverage["gaps_above_stride"], 2)
        self.assertEqual(coverage["covered_utc_days"], 2)


class OperationsTests(unittest.TestCase):
    ops = protocol()["operations"]

    def test_pair_device_policy(self):
        self.assertEqual(v25.choose_pair_device(None, self.ops)["device"], "cpu")
        busy = v25.choose_pair_device({"total_mib": 24576, "free_mib": 4000, "other_compute_pids": [1]}, self.ops)
        self.assertEqual(busy["device"], "cpu")
        shared = v25.choose_pair_device({"total_mib": 24576, "free_mib": 12000, "other_compute_pids": [1]}, self.ops)
        self.assertEqual(shared["device"], "cuda")
        self.assertAlmostEqual(shared["allocator_cap_gib"], 4.)
        self.assertFalse(shared["compute_share_guaranteed"])
        idle = v25.choose_pair_device({"total_mib": 24576, "free_mib": 24000, "other_compute_pids": []}, self.ops)
        self.assertAlmostEqual(idle["memory_fraction"], .9)

    def test_data_access_audit_blocks_outer_before_g2_and_sealed_test_always(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = v25.DataAccessLog(Path(tmp) / "events.jsonl")
            for stage in ("A0", "A1", "A2"):
                log.request(stage, "fit", 0)
                with self.assertRaises(PermissionError):
                    log.request(stage, "validation", 0, g2_passed=True)
            with self.assertRaises(PermissionError):
                log.request("A3", "validation", 0, g2_passed=False)
            with self.assertRaises(PermissionError):
                log.request("A3", "sealed_test", 0, g2_passed=True)
            log.request("A3", "validation", 0, g2_passed=True)
            events = [json.loads(line) for line in (Path(tmp) / "events.jsonl").read_text().splitlines()]
            self.assertEqual([e["subset"] for e in events if e["subset"] == "validation"], ["validation"])

    def test_controller_lock_admits_exactly_one_holder(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = v25.acquire_exclusive_lock(Path(tmp) / "x.lock")
            with self.assertRaises(BlockingIOError):
                v25.acquire_exclusive_lock(Path(tmp) / "x.lock")
            first.close()
            v25.acquire_exclusive_lock(Path(tmp) / "x.lock").close()

    def test_trace_rule_is_fixed_and_not_error_driven(self):
        times = pd.date_range("2016-10-01", periods=40, freq="3h").to_numpy()
        times = np.delete(times, [5])  # a gap: first run has only 5 issues
        turbines = np.ones(len(times), int)
        rule = v25.select_trace_segment(np.r_[times, times], np.r_[turbines, turbines + 1])
        self.assertEqual(rule["turbine"], 1)
        self.assertTrue(rule["complete"])
        self.assertEqual(len(rule["rows"]), 24)
        self.assertEqual(rule["first_issue"], str(times[5]))
        short = v25.select_trace_segment(times[:10], turbines[:10])
        self.assertFalse(short["complete"])
        self.assertEqual(len(short["rows"]), 5)  # longest run (5 vs 5) -> earliest

    def test_contract_mismatch_names_keys(self):
        self.assertEqual(v25.contract_mismatches({"a": 1, "b": [1, 2]}, {"a": 1, "b": [1, 3], "c": None}), ["b"])
        self.assertEqual(v25.contract_mismatches({"a": 1}, {"a": 1, "c": 2}), ["c"])


@unittest.skipUnless(HAS_TORCH, "torch/sklearn unavailable on this host")
class DataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = synthetic_frame()

    def test_inner_boundaries_scaler_prefix_and_no_outer_instantiation(self):
        frame = self.frame.copy()
        for fold in (0, 1, 2):
            series = synthetic_series(fold, frame)
            with patch.object(type(series), "subset", side_effect=AssertionError("outer/legacy subset requested")):
                data = v25.inner_datasets(series)
            fit, selection = data["fit"], data["selection"]
            b = series.bounds
            self.assertLess(fit.target_times().max(), b["calibration_early_end"])
            self.assertGreaterEqual(selection.target_times().min(), b["calibration_early_end"])
            self.assertLess(selection.target_times().max(), b["train_end"])
            issue = series.dates[fit.window_starts + 23]
            np.testing.assert_array_equal(fit.target_times() - issue[:, None],
                                          np.broadcast_to(np.arange(1, 5).astype("timedelta64[h]"), fit.target_times().shape))
            x, y, _, _ = fit[0]
            self.assertEqual((x.shape, y.shape), ((24, 7), (4, 7)))

    def test_scaler_ignores_values_after_the_sixty_percent_prefix(self):
        frame = self.frame.copy()
        dates = pd.to_datetime(frame.timestamp_hour_end_utc).dt.tz_localize(None)
        clean = synthetic_series(1, frame)
        spiked = frame.copy()
        spiked.loc[dates >= clean.bounds["base_fit_end"], "power_kw"] = 1e9
        series = synthetic_series(1, spiked)
        np.testing.assert_array_equal(series.scaler.mean_, clean.scaler.mean_)
        prefix = clean.raw[clean.dates < clean.bounds["base_fit_end"]]
        np.testing.assert_allclose(series.scaler.mean_, np.nanmean(prefix, 0))

    def test_gaps_turbine_boundaries_and_aux_missingness_are_shared_by_both_arms(self):
        frame = self.frame.copy()
        frame.loc[(frame.turbine_id == 2) & (frame.index % 1200 == 300), "wind_speed_mps"] = np.nan
        frame = frame[~((frame.turbine_id == 3) & (frame.index % 1200 == 400))]
        series = synthetic_series(0, frame)
        data = v25.inner_datasets(series)
        for name, dataset in data.items():
            starts = dataset.window_starts
            self.assertTrue(np.all(series.turbines[starts] == series.turbines[starts + 27]), name)
            span = series.dates[starts + 27] - series.dates[starts]
            self.assertTrue(np.all(span == np.timedelta64(27, "h")), name)
            self.assertTrue(np.isfinite(series.raw[starts[:, None] + np.arange(24)]).all(), name)
        # PowerOnly uses the same queue object, so the masking can never relax the missing-input filter
        self.assertIs(v25.inner_datasets(series)["fit"].series, series)

    def test_prediction_inverts_only_the_power_channel(self):
        from scripts import train_kelmarsh_backbone_information as task
        series = synthetic_series(0, self.frame.copy())
        selection = v25.inner_datasets(series)["selection"]

        class Constant(torch.nn.Module):
            def forward(self, x):
                return torch.ones(x.shape[0], 4, 1)
        prediction = task.predict(Constant(), selection, 0, torch.device("cpu"))
        np.testing.assert_allclose(prediction, series.scaler.mean_[-1] + series.scaler.scale_[-1])
        self.assertEqual(prediction.shape, (len(selection), 4))

    def test_dlinear_args_and_parameters_match_the_v20_recipe(self):
        from scripts import train_kelmarsh_backbone_information as task
        from scripts import train_kelmarsh_hourly as shared
        p = protocol()
        v20 = shared.read(ROOT / p["reference"]["protocol_path"])
        args = shared.model_args("DLinear", None, v20, torch.device("cpu"))
        for key, value in p["arms"]["DLinear"]["model_args"].items():
            self.assertEqual(getattr(args, key), value, key)
        self.assertEqual(v20["training"]["DLinear_lr"], p["training"]["learning_rate"])
        self.assertEqual(v20["training"]["weight_decay"], p["training"]["weight_decay"])
        self.assertEqual(v20["training"]["mix_mse_weight"], p["training"]["loss_mse_weight"])
        for key in ("batch_size", "epochs", "patience", "grad_clip"):
            ours = {"epochs": "max_epochs", "grad_clip": "grad_clip_norm"}.get(key, key)
            self.assertEqual(v20["training"][key], p["training"][ours], key)
        model, factory = task.build_model("DLinear", p)
        self.assertEqual(factory["trainable_parameters"], 200)
        for arm in ("TSMixerFull", "TSMixerPowerOnly"):
            self.assertEqual(task.build_model(arm, p)[1]["trainable_parameters"], 3618)

    def test_fit_loop_reproduces_v20_train_stage_for_dlinear(self):
        from scripts import train_kelmarsh_backbone_information as task
        from scripts import train_kelmarsh_hourly as shared
        series = synthetic_series(0, self.frame.copy())
        data = v25.inner_datasets(series)
        p = protocol()
        p["training"].update(max_epochs=3, scheduler_t_max=3)
        v20 = shared.read(ROOT / p["reference"]["protocol_path"])
        v20["training"]["epochs"] = 3
        cpu = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            stage = tmp / "suite/runs/f0_s2024/DLinear"
            stage.mkdir(parents=True)
            (tmp / "suite/protocol.json").write_text("{}")
            with patch.object(shared, "gpu_available_to_this_process"), patch("torch.cuda.is_available", return_value=False), \
                    patch("torch.cuda.get_rng_state_all", return_value=[]):
                _, old_state = shared.train_stage("DLinear", data["fit"], data["selection"], v20, stage, 2024, cpu)
                new_state = task.fit_arm("DLinear", p, fit=data["fit"], selection=data["selection"], contract={"t": 1},
                                         seed=2024, device=cpu, out=tmp / "new")
            old = torch.load(stage / "checkpoint.pt", map_location="cpu", weights_only=False)["model"]
            new = torch.load(tmp / "new/best.pt", map_location="cpu", weights_only=False)["model"]
            for key in old:
                self.assertTrue(torch.equal(old[key], new[key]), key)
            old_history, new_history = shared.read(stage / "training_history.json"), v25.read_json(tmp / "new/training_history.json")
            self.assertEqual([h["selection"]["mae"] for h in old_history], [h["selection"]["mae"] for h in new_history])
            self.assertEqual([h.get("train_loss") for h in old_history], [h.get("train_loss") for h in new_history])
            self.assertEqual(old_state["best_epoch"], new_state["best_epoch"])


if __name__ == "__main__":
    unittest.main()
