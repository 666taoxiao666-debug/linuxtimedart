"""v28 tests: protocol freeze, data identity, arm configs, metrics/rules, report recomputation and tamper refusal; with
torch: models, channel views and the controller end-to-end on synthetic data (test access only after the S1 freeze)."""
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import pandas as pd

from utils import kelmarsh_multiyear_benchmark as b

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/kelmarsh_multiyear_base_benchmark_protocol_v28.json"
SCRIPT = ROOT / "scripts/train_kelmarsh_multiyear_benchmark.py"
try:
    import torch
    import sklearn  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def protocol():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


class ProtocolTests(unittest.TestCase):
    def test_frozen_hash_design_and_parameters(self):
        constant = re.search(r'PROTOCOL_SHA256 = "([0-9a-f]{64})"', SCRIPT.read_text(encoding="utf-8")).group(1)
        self.assertEqual(b.text_sha256_lf(PROTOCOL), constant)
        p = protocol()
        self.assertTrue(b.validate_protocol(p))
        self.assertEqual({a: b.arm_config(p, a)["expected_parameters"] for a in b.TRAINED_ARMS},
                         {"DLinear_normal": 200, "DLinear_alltrain": 200, "LinearFlat_base": 676, "LinearFlat_last10": 868,
                          "TSMixer_base": 3618, "TSMixer_last10": 4262})
        self.assertEqual(b.arm_config(p, "DLinear_alltrain")["train_cohort"], "all")
        self.assertTrue(all(b.arm_config(p, a)["selection_cohort"] == "normal" for a in b.TRAINED_ARMS))
        self.assertEqual(p["data"]["splits"], {"train": [2016, 2017, 2018, 2019], "validation": [2020], "test": [2021]})
        self.assertEqual(p["budget"]["fits"], 18)

    def test_departures_are_refused(self):
        def mutated(path, value):
            p = protocol()
            target = p
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            return p
        for path, value in ((("data", "splits", "test"), [2020]), (("data", "v28_preparation", "curve_years"), [2016, 2020]),
                            (("data", "yaw_error_channels_enabled"), True), (("budget", "fits"), 24),
                            (("data", "channel_sets", "base"), [0, 1, 2, 3, 4, 5]), (("seeds",), [2024, 2025, 2026, 2027]),
                            (("families", "LinearFlat", "parameters", "last10"), 900), (("data", "sealed_years"), [2021])):
            with self.assertRaises(ValueError, msg=str(path)):
                b.validate_protocol(mutated(path, value))
        p = protocol()
        p["arms"].pop("TSMixer_last10")
        with self.assertRaises(ValueError):
            b.validate_protocol(p)


class DataIdentityTests(unittest.TestCase):
    def meta(self):
        return {"schema_version": 2, "years": [2016, 2017, 2018, 2019, 2020, 2021], "curve_years": [2016, 2017, 2018, 2019],
                "prepared": {"sha256": "abc"}, "labels_modified": False, "clipping": False, "interpolation": False}

    def test_preparation_identity(self):
        p = protocol()
        self.assertTrue(b.verify_preparation(self.meta(), p, "abc"))
        for change in ({"years": [2016, 2017, 2018, 2019, 2020, 2021, 2022]}, {"curve_years": [2016, 2017]},
                       {"schema_version": 1}, {"labels_modified": True}):
            with self.assertRaises(ValueError):
                b.verify_preparation(self.meta() | change, p, "abc")
        with self.assertRaises(ValueError):
            b.verify_preparation(self.meta(), p, "other")

    def frames(self):
        stamps = pd.date_range("2019-12-31 20:00", periods=8, freq="h", tz="UTC").astype(str)
        rows = [{"timestamp_hour_end_utc": s, "turbine_id": t, "power_kw": float(i * 10 + t), "flag_stopped": i == 3,
                 "power_last10_kw": np.nan if i == 2 else float(i)} for i, s in enumerate(stamps) for t in (1, 2)]
        v28 = pd.DataFrame(rows)
        dev = v28[v28.timestamp_hour_end_utc <= "2020-01-01 00:00:00+00:00"].copy()
        return v28, dev

    def test_development_rows_must_be_identical(self):
        v28, dev = self.frames()
        self.assertEqual(b.verify_development_rows(v28.sample(frac=1, random_state=0), dev)["rows_compared"], len(dev))
        changed = v28.copy()
        changed.loc[0, "power_kw"] += 1e-9
        with self.assertRaisesRegex(ValueError, "power_kw"):
            b.verify_development_rows(changed, dev)
        with self.assertRaisesRegex(ValueError, "development rows"):
            b.verify_development_rows(v28.iloc[1:], dev)
        with self.assertRaises(ValueError):
            b.verify_power_curves({"power_curves": {"1": [1]}}, {"power_curves": {"1": [2]}})


def fake_evaluation(root, p, rng):
    folder = Path(root) / "evaluation"
    folder.mkdir(parents=True)
    issues = 160
    times = np.repeat(np.datetime64("2021-03-01T00", "h") + np.arange(issues) * 3 * np.timedelta64(1, "h"), 6)
    times = (times[:, None] + np.arange(1, 5)[None] * np.timedelta64(1, "h")).astype("datetime64[ns]")
    n = len(times)
    truth = rng.normal(600, 400, (n, 4))
    cohort = {"truth": truth, "times": times, "turbines": np.tile(np.arange(1, 7, dtype=np.int16), issues),
              "window_starts": np.arange(n, dtype=np.int64), "normal_mask": rng.random(n) > .08,
              "PersistenceHour": truth + rng.normal(0, 150, truth.shape), "PersistenceLast10": truth + rng.normal(0, 130, truth.shape)}
    np.savez_compressed(folder / "test_cohort.npz", **cohort)
    noise = {"DLinear_normal": 110, "DLinear_alltrain": 112, "LinearFlat_base": 110, "LinearFlat_last10": 85,
             "TSMixer_base": 110, "TSMixer_last10": 86}
    common = rng.normal(0, 1, truth.shape)
    for arm in b.TRAINED_ARMS:
        for seed in p["seeds"]:
            prediction = truth + noise[arm] * (.8 * common + .6 * rng.normal(0, 1, truth.shape))
            np.savez_compressed(folder / f"{arm}_s{seed}.npz", prediction=prediction, window_starts=cohort["window_starts"])
    return cohort


class EvaluationTests(unittest.TestCase):
    def test_rules_and_both_cohorts(self):
        p = protocol()
        p["statistics"]["replicates"] = 300
        with tempfile.TemporaryDirectory() as tmp:
            fake_evaluation(tmp, p, np.random.default_rng(0))
            out = b.evaluate_registered(tmp, p, "LinearFlat_last10")
        self.assertEqual(set(out["metrics"]), {"normal", "all"})
        self.assertGreater(out["windows"]["all"], out["windows"]["normal"])
        self.assertEqual(set(out["comparisons"]["normal"]), {"C1", "C2", "C3", "C4", "C5"})
        self.assertEqual(out["comparisons"]["normal"]["C5"]["candidate"], "LinearFlat_last10")
        self.assertTrue(out["rules"]["information_supported"])
        self.assertEqual(out["rules"]["information_supported_by"], ["C1", "C2"])
        self.assertFalse(out["rules"]["capacity_supported"])
        self.assertTrue(out["rules"]["beats_naive"])
        self.assertTrue(all(0 <= c["holm_p"] <= 1 for c in out["comparisons"]["all"].values()))
        self.assertEqual(len(out["metrics"]["all"]["TSMixer_base"]["lead_mae"]), 4)

    def test_validation_choice_and_tie_break(self):
        p = protocol()
        records = [{"arm": a, "seed": s, "validation": {"mae": {"LinearFlat_base": 150., "LinearFlat_last10": 140.,
                                                                  "TSMixer_base": 150., "TSMixer_last10": 140.}[a]}}
                   for a in b.CHOICE_CANDIDATES for s in p["seeds"]]
        self.assertEqual(b.validation_choice(records, p)["arm"], "LinearFlat_last10")  # tie -> fewer parameters
        with self.assertRaises(ValueError):
            b.validation_choice(records[:-1], p)

    def test_metrics_definitions(self):
        truth = np.array([[0., 10.], [20., 30.]])
        m = b.metrics(truth + np.array([[1., -1.], [2., -2.]]), truth, 2050.)
        self.assertAlmostEqual(m["mae"], 1.5)
        self.assertAlmostEqual(m["mbe"], 0.)
        self.assertAlmostEqual(m["rmse"], np.sqrt(2.5))

    def test_holm_rejection_overrides_positive_raw_intervals(self):
        positive = {"absolute_error": {"ci95_low": .1, "gain": 1.}, "rmse_difference": {"rmse_gain_kw": 1.}, "holm_p": .2}
        comparisons = {cohort: {c: copy.deepcopy(positive) for c in ("C1", "C2", "C3", "C4", "C5")} for cohort in b.COHORTS}
        flags = b.rules(comparisons)
        self.assertFalse(flags["information_supported"])
        self.assertFalse(flags["capacity_supported"])
        self.assertFalse(flags["training_cleaning_supported"])
        self.assertFalse(flags["beats_naive"])


def fake_run(root, p):
    run = Path(root) / "run"
    rng = np.random.default_rng(1)
    fake_evaluation(run, p, rng)
    records = [{"arm": a, "seed": s, "best_epoch": 3, "train_cohort": b.arm_config(p, a)["train_cohort"],
                "validation": {"mae": 150. - (5 if a.endswith("last10") else 0), "rmse": 220., "mse": 48400.}}
               for a in b.TRAINED_ARMS for s in p["seeds"]]
    freeze = {"records": records, "validation_choice": b.validation_choice([r for r in records if r["train_cohort"] == "normal"], p)}
    b.write_json(run / "protocol.json", {"protocol": p})
    b.write_json(run / "selection_freeze.json", freeze)
    b.write_json(run / "decision.json", b.evaluate_registered(run, p, freeze["validation_choice"]["arm"]))
    names = ["selection_freeze.json", "decision.json", "evaluation/test_cohort.npz"]
    names += [f"evaluation/{arm}_s{seed}.npz" for arm in b.TRAINED_ARMS for seed in p["seeds"]]
    hashes = {name: b.file_sha256(run / name) for name in names}
    b.write_json(run / "training_completed.json", {"hashes": hashes})
    return run


class ReportTests(unittest.TestCase):
    def test_report_recomputes_and_writes_both_cohorts(self):
        from scripts import report_kelmarsh_multiyear_benchmark as report
        p = protocol()
        p["statistics"]["replicates"] = 300
        with tempfile.TemporaryDirectory() as tmp:
            run = fake_run(tmp, p)
            result = report.build_report(run)
            self.assertEqual(result["validation_choice"]["arm"], "LinearFlat_last10")
            frame = pd.read_csv(run / "test_metrics.csv")
            self.assertEqual(set(frame.cohort), {"normal", "all"})
            self.assertEqual(len(frame), 2 * 8)
            self.assertEqual(len(pd.read_csv(run / "test_comparisons.csv")), 10)
            for name in ("test_metrics", "lead_mae"):
                self.assertGreater((run / "figures" / f"{name}.png").stat().st_size, 1000)
            self.assertIn("conditional accuracy only", (run / "report.md").read_text(encoding="utf-8"))

    def test_tampering_is_refused(self):
        from scripts import report_kelmarsh_multiyear_benchmark as report
        p = protocol()
        p["statistics"]["replicates"] = 300
        with tempfile.TemporaryDirectory() as tmp:
            run = fake_run(tmp, p)
            with np.load(run / "evaluation/LinearFlat_last10_s2024.npz") as saved:
                arrays = {k: saved[k] for k in saved.files}
            arrays["prediction"] = arrays["prediction"] * 0.9
            np.savez_compressed(run / "evaluation/LinearFlat_last10_s2024.npz", **arrays)
            with self.assertRaises(ValueError):
                report.build_report(run)
        with tempfile.TemporaryDirectory() as tmp:
            run = fake_run(tmp, p)
            record = b.read_json(run / "decision.json")
            record["rules"]["beats_naive"] = not record["rules"]["beats_naive"]
            b.write_json(run / "decision.json", record)
            with self.assertRaisesRegex(ValueError, "changed or missing"):
                report.build_report(run)

    def test_every_statistic_is_recomputed(self):
        from scripts import report_kelmarsh_multiyear_benchmark as report
        p = protocol()
        p["statistics"]["replicates"] = 100
        with tempfile.TemporaryDirectory() as tmp:
            fake_evaluation(tmp, p, np.random.default_rng(0))
            fresh = b.safe(b.evaluate_registered(tmp, p, "LinearFlat_last10"))
            for path in (("comparisons", "all", "C1", "holm_p"),
                         ("comparisons", "normal", "C2", "rmse_difference", "ci95_high_kw"),
                         ("metrics", "all", "TSMixer_base", "mse")):
                changed = copy.deepcopy(fresh)
                node = changed
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] += .01
                with self.assertRaisesRegex(ValueError, "disagree"):
                    report.check_against_registered(fresh, changed)


@unittest.skipUnless(HAS_TORCH, "torch/sklearn unavailable on this host")
class ModelTests(unittest.TestCase):
    def test_models_and_channel_views(self):
        from scripts import train_kelmarsh_multiyear_benchmark as task
        from data_provider.kelmarsh_multiyear import KelmarshMultiYearSeries
        p = protocol()
        for arm in b.TRAINED_ARMS:
            config = b.arm_config(p, arm)
            model, count = task.build_model(config)
            out = model(torch.randn(5, 24, config["channels"]))
            self.assertEqual(count, config["expected_parameters"])
            self.assertEqual(out[..., -1].shape, (5, 4))
        series = KelmarshMultiYearSeries(task.synthetic_prepared_frame(), splits={"train": (2016,), "validation": (2017,), "test": (2018,)},
                                         extras=("power_last10", "wind_last10"), preparation=task.SYNTHETIC_PREPARATION)
        windows = series.windows("train", "normal")
        base = task.ChannelView(windows, p["data"]["channel_sets"]["base"])
        x, y, _, _ = base[0]
        full_x = windows[0][0]
        np.testing.assert_array_equal(x, full_x[:, [0, 1, 2, 3, 4, 5, 8]])
        self.assertEqual(base.scaler.mean_[-1], windows.scaler.mean_[-1])
        with self.assertRaises(ValueError):
            task.ChannelView(windows, [0, 1, 2])


@unittest.skipUnless(HAS_TORCH, "torch/sklearn unavailable on this host")
class ControllerTests(unittest.TestCase):
    def make(self, tmp):
        from scripts import train_kelmarsh_multiyear_benchmark as task
        from data_provider.kelmarsh_multiyear import KelmarshMultiYearSeries
        self.task = task
        p = copy.deepcopy(protocol())
        p["data"]["splits"] = {"train": [2016], "validation": [2017], "test": [2018]}
        p["seeds"] = [2024]
        p["training"].update(max_epochs=2, scheduler_t_max=2)
        p["budget"].update(fits=6, new_fits_max=6)
        p["statistics"]["replicates"] = 200

        def factory(kwargs):
            return KelmarshMultiYearSeries(task.synthetic_prepared_frame(), preparation=task.SYNTHETIC_PREPARATION, **kwargs)
        return task.V28Run(Path(tmp) / "run", protocol=p, protocol_sha="test", series_factory=factory, require_preflight=False,
                           write_pointer=False, lock_path=Path(tmp) / "lock", verify_data=False)

    def test_end_to_end_freeze_before_test_and_reuse(self):
        from unittest.mock import patch
        torch.set_num_threads(2)
        with patch("scripts.train_kelmarsh_anchored_residual.gpu_snapshot", return_value=None):
            with tempfile.TemporaryDirectory() as tmp:
                run = self.make(tmp)
                self.assertEqual(run.run(), "COMPLETED")
                self.assertEqual(len(run.registered_fits()), 6)
                events = [json.loads(line) for line in (run.dir / "events.jsonl").read_text().splitlines()]
                frozen = next(i for i, e in enumerate(events) if e["event"] == "S1_frozen")
                first_test = next(i for i, e in enumerate(events) if e.get("split") == "test")
                self.assertLess(frozen, first_test)
                self.assertFalse(any(e.get("split") == "test" for e in events[:frozen]))
                decision = b.read_json(run.dir / "decision.json")
                self.assertEqual(set(decision["metrics"]), {"normal", "all"})
                self.assertEqual(b.read_json(run.dir / "status.json")["report"], "REPORT_COMPLETED")
                for arm in b.TRAINED_ARMS:
                    trained = b.read_json(run.dir / f"runs/{arm}/s2024/trained.json")
                    self.assertEqual(trained["parameters"], b.arm_config(run.p, arm)["expected_parameters"])
                again = self.make(tmp)
                with patch.object(self.task, "fit_config", side_effect=AssertionError("retrained")):
                    self.assertEqual(again.run(), "COMPLETED")

    def test_test_windows_refused_before_freeze_and_preflight_required(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp)
            run.dir.mkdir(parents=True)
            run.record_sources()
            run.stage_d0()
            self.assertNotIn("test", run.series.splits)
            self.assertNotIn("test", run.series.audit["windows"])
            self.assertLessEqual(int(run.series.label_year.max()), 2017)
            with self.assertRaises(PermissionError):
                run.windows("test", "all")
            with self.assertRaises(PermissionError):
                run.load_series(include_test=True)
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp)
            run.require_preflight = True
            with patch.object(self.task, "PREFLIGHT_DIR", Path(tmp) / "pf"), \
                    patch("scripts.train_kelmarsh_anchored_residual.gpu_snapshot", return_value=None):
                with self.assertRaisesRegex(ValueError, "preflight"):
                    run.run()
            self.assertFalse(list(run.dir.rglob("trained.json")))

    def test_interrupted_training_matches_uninterrupted(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp)
            run.dir.mkdir(parents=True)
            run.stage_d0()
            task = self.task
            task.v26run.configure_determinism()
            config = b.arm_config(run.p, "TSMixer_base")
            fit = task.ChannelView(run.windows("train", "normal"), config["channel_indices"])
            val = task.ChannelView(run.windows("validation", "normal"), config["channel_indices"])
            arguments = dict(fit=fit, selection=val, contract={"test": "resume"}, seed=2024, device=torch.device("cpu"))
            full = task.fit_config(config, run.p, out=Path(tmp) / "full", **arguments)
            with self.assertRaises(task.SimulatedInterrupt):
                task.fit_config(config, run.p, out=Path(tmp) / "resumed", stop_after_epoch=1, **arguments)
            resumed = task.fit_config(config, run.p, out=Path(tmp) / "resumed", **arguments)
            self.assertEqual(full["best_state_sha256"], resumed["best_state_sha256"])
            self.assertEqual(full["validation_best"], resumed["validation_best"])
            full_last = torch.load(Path(tmp) / "full/last.pt", map_location="cpu", weights_only=False)
            resumed_last = torch.load(Path(tmp) / "resumed/last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(task.v26run.state_digest(full_last["model"]), task.v26run.state_digest(resumed_last["model"]))


if __name__ == "__main__":
    unittest.main()
