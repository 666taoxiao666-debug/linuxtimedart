"""v31 tests: protocol freeze/design, MIDAS parsing and timing, the composed upstream series (queue unchanged,
input-only fill), joint validation choice, evaluation (C1-C3, both-metric rules, farm metrics), report recomputation and
tamper refusal; with torch: models and the controller end-to-end on synthetic data (test access only after S1)."""
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import pandas as pd

from data_provider.kelmarsh_multiyear import KelmarshMultiYearSeries
from utils import kelmarsh_upstream_v31 as u

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/kelmarsh_upstream_protocol_v31.json"
SCRIPT = ROOT / "scripts/train_kelmarsh_upstream_v31.py"
SYNTHETIC_PREPARATION = {"schema_version": 2, "curve_years": [2016], "prepared": {"sha256": "synthetic"}}
try:
    import torch
    import sklearn  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def protocol():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def synthetic_frame(years=(2016, 2017, 2018), days=14, seed=0):
    """Same columns as the v28 synthetic frame (torch-free copy for these tests)."""
    rng = np.random.default_rng(seed)
    parts = []
    for year in years:
        dates = pd.date_range(f"{year}-12-{31 - days + 1:02d} 01:00", periods=days * 24, freq="h", tz="UTC")
        t = np.arange(len(dates), dtype=float)
        for unit in range(1, 7):
            wind = 7 + 3 * np.sin(t / 9 + unit) + rng.normal(0, .3, len(t))
            power = np.clip(2050 / (1 + np.exp(-(wind - 8) / 1.4)) + rng.normal(0, 30, len(t)), -10, 2060)
            parts.append(pd.DataFrame({
                "timestamp_hour_end_utc": dates.astype(str), "turbine_id": unit, "wind_speed_mps": wind,
                "ambient_temp_c": 5 + np.sin(t / 24), "nacelle_position_deg": (t * 7) % 360, "pitch_a_deg": 1.,
                "pitch_b_deg": 1.1, "pitch_c_deg": .9, "rotor_speed_rpm": 8 + wind / 2, "power_kw": power,
                "power_last10_kw": power + rng.normal(0, 40, len(t)), "wind_last10_mps": wind + rng.normal(0, .3, len(t)),
                "normal_operation": rng.random(len(t)) > .04}))
    return pd.concat(parts, ignore_index=True)


def synthetic_upstream(frame, seed=0, missing=0.05):
    rng = np.random.default_rng(seed + 30)
    dates = pd.to_datetime(frame["timestamp_hour_end_utc"], utc=True).dt.tz_localize(None)
    labels = pd.date_range(dates.min() - pd.Timedelta(hours=48), dates.max(), freq="h")
    values = rng.normal(5, 2, (len(labels), len(u.UPSTREAM_FEATURES)))
    values[rng.random(values.shape) < missing] = np.nan
    return pd.DataFrame(values, index=labels, columns=u.UPSTREAM_FEATURES)


class ProtocolTests(unittest.TestCase):
    def test_frozen_hash_design_and_parameters(self):
        constant = re.search(r'PROTOCOL_SHA256 = "([0-9a-f]{64})"', SCRIPT.read_text(encoding="utf-8")).group(1)
        self.assertEqual(u.text_sha256_lf(PROTOCOL), constant)
        p = protocol()
        self.assertTrue(u.validate_protocol(p))
        self.assertEqual(tuple(p["arms"]), u.ARMS)
        self.assertEqual({a: u.arm_config(p, a)["expected_parameters"] for a in u.TRAINED_ARMS},
                         {"TSMixer_last10_all": 4262, "TSMixer_upstream_all": 8126, "TSMixer_upstream_all_balanced": 8126})
        self.assertEqual(p["data"]["splits"], {"train": [2016, 2017, 2018, 2019, 2020, 2021], "validation": [2022], "test": [2023]})
        self.assertEqual(p["data"]["sealed_years"], [2024])
        self.assertEqual((p["budget"]["fits"], p["budget"]["new_epochs_max"]), (9, 180))
        self.assertEqual(len(p["data"]["upstream"]["stations"]), 36)
        self.assertEqual(p["data"]["upstream"]["lag_hours"], 1)
        self.assertEqual(u.arm_protocol(p, "TSMixer_upstream_all_balanced")["training"]["loss_mse_weight"], 0.5)
        self.assertIs(u.arm_protocol(p, "TSMixer_upstream_all"), p)

    def test_departures_are_refused(self):
        def mutated(path, value):
            p = protocol()
            target = p
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            return p
        for path, value in ((("data", "splits", "test"), [2022]), (("data", "v31_preparation", "curve_years"), [2016, 2022]),
                            (("data", "sealed_years"), [2023]), (("budget", "fits"), 12), (("seeds",), [2024, 2025]),
                            (("data", "channel_sets", "upstream"), list(range(20))),
                            (("families", "TSMixerBasicSmall", "parameters", "upstream"), 8000),
                            (("arms", "TSMixer_upstream_all", "training_overrides"), {"loss_mae_weight": .5, "loss_mse_weight": .5}),
                            (("arms", "TSMixer_upstream_all_balanced", "training_overrides"), {"learning_rate": .01}),
                            (("arms", "TSMixer_last10_all", "train_cohort"), "normal"), (("data", "upstream", "lag_hours"), 0),
                            (("data", "upstream", "test_years"), [2022]), (("training", "selection_cohort"), "normal"),
                            (("rules", "non_inferiority_margin_kw"), 0), (("budget", "hyperparameter_search"), True)):
            with self.assertRaises(ValueError, msg=str(path)):
                u.validate_protocol(mutated(path, value))
        p = protocol()
        p["data"]["upstream"]["stations"] = p["data"]["upstream"]["stations"][:-1]
        with self.assertRaises(ValueError):
            u.validate_protocol(p)


def badc_file(path, rows, unit="knots"):
    header = ["Conventions,G,BADC-CSV,1", f"unit,wind_speed,{unit}", "unit,wind_direction,degrees", "unit,msl_pressure,hPa", "data"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header) + "\n" + rows.to_csv(index=False) + "end data\n", encoding="utf-8")


def station_rows(sid, year, speed):
    times = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left")
    return pd.DataFrame({"ob_time": times.strftime("%Y-%m-%d %H:%M:%S"), "src_id": sid, "version_num": 1,
                         "wind_speed": speed(times), "wind_direction": 225, "msl_pressure": 1013.0})


class MidasTests(unittest.TestCase):
    def test_parsing_units_versions_and_years(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1" / "midas-open_uk-hourly-weather-obs_dv-202607_x_00001_a_qcv-1_2021.csv"
            rows = station_rows(1, 2021, lambda t: np.full(len(t), 10.0))
            rows = pd.concat([rows, rows.iloc[:2].assign(version_num=0, wind_speed=99.0)])
            badc_file(path, rows)
            frame = u.parse_station_file(path, 2021, {1})
            self.assertEqual(len(frame), len(rows) - 2)                     # version 0 rows dropped
            self.assertAlmostEqual(frame.wind_speed.iloc[0], 10 * 1852 / 3600)
            with self.assertRaisesRegex(ValueError, "row year"):
                u.parse_station_file(path, 2022, {1})
            badc_file(path, station_rows(1, 2021, lambda t: np.full(len(t), 10.0)), unit="furlongs")
            with self.assertRaisesRegex(ValueError, "wind-speed unit"):
                u.parse_station_file(path, 2021, {1})

    def test_manifest_hashes_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "1" / "midas-open_uk-hourly-weather-obs_dv-202607_x_00001_a_qcv-1_2021.csv"
            badc_file(path, station_rows(1, 2021, lambda t: np.full(len(t), 10.0)))
            hashes = u.manifest_hashes({"files": [{"path": f"/srv/{path.name}", "sha256": u.file_sha256(path)}]})
            found, missing = u.station_files(root, [1, 2], [2021], hashes, required=False)
            self.assertEqual((len(found), missing), (1, [{"src_id": 2, "year": 2021}]))
            with self.assertRaisesRegex(ValueError, "Missing development"):
                u.station_files(root, [1, 2], [2021], hashes, required=True)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                u.station_files(root, [1], [2021], {path.name: "0" * 64})

    def test_feature_rows_use_observations_one_hour_earlier(self):
        p = protocol()
        stations = p["data"]["upstream"]["stations"][:6]
        times = pd.date_range("2021-01-01", "2021-01-03", freq="h")
        observations = {}
        for k, s in enumerate(stations):
            speed = pd.Series(np.arange(len(times), dtype=float) + 100 * k, index=times)
            observations[int(s["src_id"])] = pd.DataFrame({"wind_speed": speed, "wind_direction": 225.0,
                                                           "msl_pressure": 1000.0 + speed / 10})
        labels = times[10:20]
        frame = u.hourly_upstream(observations, stations, labels, 1)
        nearest = min(stations, key=lambda s: s["distance_km"])
        expected = observations[int(nearest["src_id"])]["wind_speed"].reindex(labels - pd.Timedelta(hours=1)).to_numpy()
        np.testing.assert_array_equal(frame["local_ws"].to_numpy(), expected)
        np.testing.assert_allclose(frame["local_dws3"].to_numpy(), 3.0)


class SeriesTests(unittest.TestCase):
    def make(self, missing=0.05):
        frame = synthetic_frame()
        kwargs = {"splits": {"train": (2016,), "validation": (2017,)}, "extras": ("power_last10", "wind_last10")}
        base = KelmarshMultiYearSeries(frame, preparation=SYNTHETIC_PREPARATION, **kwargs)
        series = u.KelmarshUpstreamSeries(frame, synthetic_upstream(frame, missing=missing), preparation=SYNTHETIC_PREPARATION, **kwargs)
        return base, series

    def test_queue_scaler_and_input_only_fill(self):
        base, series = self.make()
        self.assertEqual(series.channels, u.CHANNELS)
        np.testing.assert_array_equal(series.starts, base.starts)
        np.testing.assert_array_equal(np.nan_to_num(series.raw[:, -1], nan=-1), np.nan_to_num(base.raw[:, -1], nan=-1))
        scada = list(range(8)) + [20]
        np.testing.assert_allclose(series.scaler_mean[scada], base.scaler_mean, atol=1e-9)
        flag = series.raw[:, u.CHANNELS.index(u.MISSING_CHANNEL)]
        self.assertTrue(set(np.unique(flag)) <= {0.0, 1.0} and flag.any())
        upstream = series.values[:, 8:19]
        self.assertTrue(np.isfinite(upstream).all())
        train = np.isin(series.label_year, [2016])
        np.testing.assert_allclose(series.raw[train][:, 8:19].mean(0), series.audit["upstream"]["fill_values_training_mean"], atol=1e-9)
        windows = series.windows("train", "all")
        x, y, _, _ = windows[0]
        self.assertEqual(x.shape, (24, 21))
        truth, _ = windows.truth_and_anchor()
        np.testing.assert_array_equal(truth, base.windows("train", "all").truth_and_anchor()[0])
        self.assertGreater(len(series.windows("train", "all")), len(series.windows("train", "normal")))

    def test_upstream_columns_are_checked(self):
        frame = synthetic_frame()
        with self.assertRaises(ValueError):
            u.KelmarshUpstreamSeries(frame, synthetic_upstream(frame).iloc[:, :-1], splits={"train": (2016,)},
                                     preparation=SYNTHETIC_PREPARATION)

    def test_heldout_weather_cannot_change_training_fill_or_inputs(self):
        frame = synthetic_frame()
        upstream = synthetic_upstream(frame)
        changed = upstream.copy()
        # Hour-end 2017-01-01 00:00 still belongs to the 2016 label year.
        heldout = (changed.index - pd.Timedelta(hours=1)).year > 2016
        changed.loc[heldout] = 1e9
        kwargs = {"splits": {"train": (2016,), "validation": (2017,), "test": (2018,)},
                  "extras": ("power_last10", "wind_last10"), "preparation": SYNTHETIC_PREPARATION}
        before = u.KelmarshUpstreamSeries(frame, upstream, **kwargs)
        after = u.KelmarshUpstreamSeries(frame, changed, **kwargs)
        np.testing.assert_array_equal(before.scaler_mean, after.scaler_mean)
        np.testing.assert_array_equal(before.scaler_scale, after.scaler_scale)
        np.testing.assert_array_equal(before.audit["upstream"]["fill_values_training_mean"],
                                      after.audit["upstream"]["fill_values_training_mean"])
        train = np.isin(before.label_year, [2016])
        np.testing.assert_array_equal(before.values[train], after.values[train])
        for split in kwargs["splits"]:
            np.testing.assert_array_equal(before.windows(split).truth_and_anchor()[0],
                                          after.windows(split).truth_and_anchor()[0])


NOISE = {"TSMixer_last10_all": 100, "TSMixer_upstream_all": 80, "TSMixer_upstream_all_balanced": 78}


def fake_evaluation(root, p, rng):
    folder = Path(root) / "evaluation"
    folder.mkdir(parents=True)
    issues = 160
    times = np.repeat(np.datetime64("2023-03-01T00", "h") + np.arange(issues) * 3 * np.timedelta64(1, "h"), 6)
    times = (times[:, None] + np.arange(1, 5)[None] * np.timedelta64(1, "h")).astype("datetime64[ns]")
    n = len(times)
    level = rng.normal(600, 400, n)
    truth = level[:, None] + rng.normal(0, 80, (n, 4)) * np.sqrt(np.arange(1, 5))[None]
    anchor = level + rng.normal(0, 40, n)
    cohort = {"truth": truth, "times": times, "turbines": np.tile(np.arange(1, 7, dtype=np.int16), issues),
              "window_starts": np.arange(n, dtype=np.int64), "normal_mask": rng.random(n) > .08,
              "PersistenceHour": np.repeat((anchor + rng.normal(0, 40, n))[:, None], 4, axis=1),
              "PersistenceLast10": np.repeat(anchor[:, None], 4, axis=1),
              "issue_upstream_missing": (rng.random(n) < .03).astype(float)}
    np.savez_compressed(folder / "test_cohort.npz", **cohort)
    common = rng.normal(0, 1, truth.shape)
    for arm in u.TRAINED_ARMS:
        for seed in p["seeds"]:
            prediction = truth + NOISE[arm] * (.8 * common + .6 * rng.normal(0, 1, truth.shape))
            np.savez_compressed(folder / f"{arm}_s{seed}.npz", prediction=prediction, window_starts=cohort["window_starts"])
    return cohort


class EvaluationTests(unittest.TestCase):
    def test_rules_cohorts_farm_and_regimes(self):
        p = protocol()
        p["statistics"]["replicates"] = 300
        with tempfile.TemporaryDirectory() as tmp:
            cohort = fake_evaluation(tmp, p, np.random.default_rng(0))
            out = u.evaluate_registered(tmp, p, "TSMixer_upstream_all")
        self.assertEqual(set(out["comparisons"]["all"]), set(u.COMPARISONS))
        self.assertEqual(out["comparisons"]["all"]["C3"]["candidate"], "TSMixer_upstream_all")
        self.assertTrue(out["rules"]["upstream_information_supported"])
        self.assertTrue(out["rules"]["beats_naive"])
        farm = out["farm"]
        self.assertEqual(farm["issue_times"], 160)
        self.assertEqual(farm["capacity_kw"], 6 * 2050.0)
        f = farm["arms"]["TSMixer_upstream_all"]
        self.assertAlmostEqual(f["accuracy_pct"], 100 - f["nrmse_pct"])
        self.assertLess(f["nrmse_pct"], farm["arms"]["TSMixer_last10_all"]["nrmse_pct"])
        # independent farm computation (seed mean of per-seed six-turbine normalized RMSE)
        with tempfile.TemporaryDirectory() as tmp:
            fake_evaluation(tmp, p, np.random.default_rng(0))
            seeds = [np.load(Path(tmp) / f"evaluation/TSMixer_last10_all_s{s}.npz")["prediction"] for s in p["seeds"]]
        independent = np.mean([100 * np.sqrt((((q - cohort["truth"]).reshape(160, 6, 4).sum(1)) ** 2).mean()) / (6 * 2050)
                               for q in seeds])
        self.assertAlmostEqual(independent, farm["arms"]["TSMixer_last10_all"]["nrmse_pct"], places=9)
        regimes = {r["regime"] for r in out["regimes"]["all"]}
        self.assertIn("upstream_missing_at_issue", regimes)
        self.assertLessEqual(out["ensemble"]["all"]["mae"], out["metrics"]["all"]["TSMixer_upstream_all"]["mae"] + 1e-9)

    def test_both_metric_rules(self):
        good = {"absolute_error": {"ci95_low": .5, "gain": 2.}, "rmse_difference": {"rmse_gain_kw": 1., "ci95_low_kw": .2}, "holm_p": .01}
        comparisons = {c: {n: copy.deepcopy(good) for n in u.COMPARISONS} for c in u.COHORTS}
        self.assertTrue(all(u.rules(comparisons, 2.0).values()))
        comparisons["all"]["C1"]["rmse_difference"]["ci95_low_kw"] = -.1          # MAE alone is not enough
        self.assertFalse(u.rules(comparisons, 2.0)["upstream_information_supported"])
        comparisons["all"]["C2"]["absolute_error"]["ci95_low"] = -3.               # balanced loss: MAE non-inferiority
        self.assertFalse(u.rules(comparisons, 2.0)["balanced_loss_supported"])
        comparisons["normal"]["C3"]["holm_p"] = .2
        self.assertFalse(u.rules(comparisons, 2.0)["beats_naive"])

    def test_joint_validation_choice(self):
        p = protocol()
        def records(values):
            return [{"arm": a, "seed": s, "validation": {"mae": values[a][0], "rmse": values[a][1]}} for a in u.TRAINED_ARMS for s in p["seeds"]]
        values = {"TSMixer_last10_all": (100., 200.), "TSMixer_upstream_all": (98., 199.), "TSMixer_upstream_all_balanced": (99., 194.)}
        choice = u.validation_choice(records(values), p)
        self.assertEqual(choice["arm"], "TSMixer_upstream_all_balanced")   # 0.99 + 0.97 < 0.98 + 0.995
        self.assertAlmostEqual(choice["ranking"][-1]["joint_score"], 2.0)
        tie = {a: (100., 200.) for a in u.TRAINED_ARMS}
        self.assertEqual(u.validation_choice(records(tie), p)["arm"], "TSMixer_last10_all")   # fewer parameters
        with self.assertRaises(ValueError):
            u.validation_choice(records(values)[:-1], p)


def fake_run(root, p):
    run = Path(root) / "run"
    cohort = fake_evaluation(run, p, np.random.default_rng(1))
    records = [{"arm": a, "seed": s, "best_epoch": 3, "validation": {"mae": 150. - (10 if "upstream" in a else 0), "rmse": 220., "mse": 48400.}}
               for a in u.TRAINED_ARMS for s in p["seeds"]]
    pre = {"train": {"all": 900, "normal": 850}, "validation": {"all": 300, "normal": 280}}
    u.write_json(run / "data_manifest.json", {"identity": {}, "series_audit": {"windows": pre}, "test_windows_instantiated": False})
    freeze = {"records": records, "validation_choice": u.validation_choice(records, p)}
    u.write_json(run / "selection_freeze.json", freeze)
    u.write_json(run / "protocol.json", {"protocol": p})
    u.write_json(run / "evaluation/upstream_test_files.json", {"files": {}, "missing_station_years": []})
    test = {"all": len(cohort["truth"]), "normal": int(cohort["normal_mask"].sum())}
    u.write_json(run / "post_test_manifest.json", u.post_test_manifest(run, pre | {"test": test}))
    u.write_json(run / "decision.json", u.evaluate_registered(run, p, freeze["validation_choice"]["arm"]))
    names = ["selection_freeze.json", "decision.json", "post_test_manifest.json", "evaluation/test_cohort.npz",
             "evaluation/upstream_test_files.json"]
    names += [f"evaluation/{arm}_s{seed}.npz" for arm in u.TRAINED_ARMS for seed in p["seeds"]]
    u.write_json(run / "training_completed.json", {"hashes": {name: u.file_sha256(run / name) for name in names}})
    return run


class ReportTests(unittest.TestCase):
    def test_report_recomputes_and_writes_tables(self):
        p = protocol()
        p["statistics"]["replicates"] = 300
        with tempfile.TemporaryDirectory() as tmp:
            run = fake_run(tmp, p)
            result = u.build_report(run)
            self.assertEqual(result["primary_cohort"], "all")
            frame = pd.read_csv(run / "test_metrics.csv")
            self.assertEqual(len(frame), 2 * (len(u.ARMS) + 1))
            self.assertEqual(len(pd.read_csv(run / "test_comparisons.csv")), 2 * len(u.COMPARISONS))
            farm = pd.read_csv(run / "farm_metrics.csv")
            self.assertEqual(set(farm.arm), set(u.ARMS))
            self.assertTrue(np.allclose(farm.accuracy_pct, 100 - farm.nrmse_pct))
            for name in ("test_metrics", "lead_mae", "forecast_trace"):
                self.assertGreater((run / "figures" / f"{name}.png").stat().st_size, 1000)
            text = (run / "report.md").read_text(encoding="utf-8")
            self.assertIn("not a compliance claim", text)

    def test_tampering_is_refused(self):
        p = protocol()
        p["statistics"]["replicates"] = 100
        for name in ("evaluation/TSMixer_upstream_all_s2024.npz", "decision.json", "post_test_manifest.json",
                     "evaluation/upstream_test_files.json"):
            with tempfile.TemporaryDirectory() as tmp:
                run = fake_run(tmp, p)
                path = run / name
                if name.endswith(".npz"):
                    with np.load(path) as saved:
                        arrays = {k: saved[k] for k in saved.files}
                    arrays["prediction"] = arrays["prediction"] * 0.9
                    np.savez_compressed(path, **arrays)
                else:
                    record = u.read_json(path)
                    record["tampered"] = True
                    u.write_json(path, record)
                with self.assertRaisesRegex(ValueError, "changed or missing"):
                    u.build_report(run)


@unittest.skipUnless(HAS_TORCH, "torch/sklearn unavailable on this host")
class ModelTests(unittest.TestCase):
    def test_models_and_channel_views(self):
        from scripts import train_kelmarsh_upstream_v31 as task
        p = protocol()
        for arm in u.TRAINED_ARMS:
            config = u.arm_config(p, arm)
            model, count = task.v28.build_model(config)
            out = model(torch.randn(5, 24, config["channels"]))
            self.assertEqual(count, config["expected_parameters"])
            self.assertEqual(out[..., -1].shape, (5, 4))
        series = task.synthetic_series({"splits": {"train": (2016,), "validation": (2017,), "test": (2018,)},
                                        "extras": tuple(p["data"]["extras_in_common_series"])})
        windows = series.windows("train", "all")
        view = task.v28.ChannelView(windows, p["data"]["channel_sets"]["last10"])
        np.testing.assert_array_equal(view[0][0], windows[0][0][:, [0, 1, 2, 3, 4, 5, 6, 7, 20]])


@unittest.skipUnless(HAS_TORCH, "torch/sklearn unavailable on this host")
class ControllerTests(unittest.TestCase):
    def make(self, tmp):
        from scripts import train_kelmarsh_upstream_v31 as task
        self.task = task
        p = copy.deepcopy(protocol())
        p["data"]["splits"] = {"train": [2016], "validation": [2017], "test": [2018]}
        p["seeds"] = [2024]
        p["training"].update(max_epochs=2, scheduler_t_max=2)
        p["budget"].update(fits=3, new_fits_max=3)
        p["statistics"]["replicates"] = 200
        return task.V31Run(Path(tmp) / "run", protocol=p, protocol_sha="test", series_factory=task.synthetic_series,
                           require_preflight=False, write_pointer=False, lock_path=Path(tmp) / "lock", verify_data=False)

    def test_end_to_end_freeze_before_test_and_reuse(self):
        from unittest.mock import patch
        torch.set_num_threads(2)
        with patch("scripts.train_kelmarsh_anchored_residual.gpu_snapshot", return_value=None):
            with tempfile.TemporaryDirectory() as tmp:
                run = self.make(tmp)
                self.assertEqual(run.run(), "COMPLETED")
                self.assertEqual(len(run.registered_fits()), 3)
                events = [json.loads(line) for line in (run.dir / "events.jsonl").read_text().splitlines()]
                frozen = next(i for i, e in enumerate(events) if e["event"] == "S1_frozen")
                first_test = next(i for i, e in enumerate(events) if e.get("split") == "test")
                self.assertLess(frozen, first_test)
                balanced = u.read_json(run.dir / "runs/TSMixer_upstream_all_balanced/s2024/trained.json")
                self.assertEqual(balanced["contract"]["training"]["loss_mse_weight"], 0.5)
                reference = u.read_json(run.dir / "runs/TSMixer_last10_all/s2024/trained.json")
                self.assertEqual(reference["contract"]["training"]["loss_mse_weight"], 0.2)
                decision = u.read_json(run.dir / "decision.json")
                self.assertEqual(set(decision["metrics"]), {"normal", "all"})
                self.assertIn("farm", decision)
                self.assertTrue((run.dir / "evaluation/upstream_test_files.json").exists())
                self.assertEqual(u.read_json(run.dir / "status.json")["report"], "REPORT_COMPLETED")
                again = self.make(tmp)
                with patch.object(self.task.v28, "fit_config", side_effect=AssertionError("retrained")):
                    self.assertEqual(again.run(), "COMPLETED")

    def test_test_access_refused_before_freeze_and_preflight_required(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp)
            run.dir.mkdir(parents=True)
            run.record_sources()
            run.stage_d0()
            self.assertNotIn("test", run.series.splits)
            with self.assertRaises(PermissionError):
                run.windows("test", "all")
            with self.assertRaises(PermissionError):
                run.load_series(include_test=True)
            with self.assertRaises(PermissionError):
                run.upstream_frame(include_test=True)
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp)
            run.require_preflight = True
            with patch.object(self.task, "PREFLIGHT_DIR", Path(tmp) / "pf"), \
                    patch("scripts.train_kelmarsh_anchored_residual.gpu_snapshot", return_value=None):
                with self.assertRaisesRegex(ValueError, "preflight"):
                    run.run()
            self.assertFalse(list(run.dir.rglob("trained.json")))


if __name__ == "__main__":
    unittest.main()
