"""v29 tests: protocol freeze/design, D0 identity, the fixed Guard rule, validation choice, evaluation (C1-C6, rules,
ensemble, ex-ante regimes), the post-test manifest, report recomputation and tamper refusal; with torch: models, channel
views and the controller end-to-end on synthetic data (test access only after the S1 freeze)."""
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import pandas as pd

from utils import kelmarsh_operating_state as u

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/kelmarsh_operating_state_protocol_v29.json"
SCRIPT = ROOT / "scripts/train_kelmarsh_operating_state.py"
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
        self.assertEqual(u.text_sha256_lf(PROTOCOL), constant)
        p = protocol()
        self.assertTrue(u.validate_protocol(p))
        self.assertEqual(tuple(p["arms"]), u.ARMS)
        self.assertEqual({a: u.arm_config(p, a)["expected_parameters"] for a in u.TRAINED_ARMS},
                         {"TSMixer_last10_normal": 4262, "TSMixer_last10_all": 4262,
                          "TSMixer_state_normal": 4906, "TSMixer_state_all": 4906})
        self.assertEqual(u.arm_parameters(p, u.GUARD_ARM), 4262)
        self.assertEqual({a: u.arm_config(p, a)["train_cohort"] for a in u.TRAINED_ARMS},
                         {"TSMixer_last10_normal": "normal", "TSMixer_last10_all": "all",
                          "TSMixer_state_normal": "normal", "TSMixer_state_all": "all"})
        self.assertTrue(all(u.arm_config(p, a)["selection_cohort"] == "all" for a in u.TRAINED_ARMS))
        self.assertEqual(p["data"]["splits"], {"train": [2016, 2017, 2018, 2019, 2020], "validation": [2021], "test": [2022]})
        self.assertEqual(p["data"]["sealed_years"], [2023, 2024])
        self.assertEqual(p["data"]["v29_preparation"]["curve_years"], [2016, 2017, 2018, 2019])
        self.assertEqual((p["budget"]["fits"], p["budget"]["new_epochs_max"]), (12, 240))
        self.assertEqual(p["arms"][u.GUARD_ARM]["threshold_frac_rated"], 0.02)

    def test_departures_are_refused(self):
        def mutated(path, value):
            p = protocol()
            target = p
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            return p
        for path, value in ((("data", "splits", "test"), [2021]), (("data", "v29_preparation", "curve_years"), [2016, 2021]),
                            (("data", "yaw_error_channels_enabled"), True), (("budget", "fits"), 15),
                            (("data", "channel_sets", "state"), list(range(10))), (("seeds",), [2024, 2025, 2026, 2027]),
                            (("families", "TSMixerBasicSmall", "parameters", "state"), 5000), (("data", "sealed_years"), [2022]),
                            (("arms", u.GUARD_ARM, "threshold_frac_rated"), 0.2),
                            (("arms", u.GUARD_ARM, "base_arm"), "TSMixer_state_all"),
                            (("arms", "TSMixer_state_all", "train_cohort"), "normal"),
                            (("training", "selection_cohort"), "normal"), (("cohorts", "primary"), "normal"),
                            (("rules", "non_inferiority_margin_kw"), 0),
                            (("data", "v28_registered_preparation", "years"), [2016, 2017, 2018, 2019, 2020, 2021, 2022]),
                            (("budget", "hyperparameter_search"), True), (("diagnostics", "anchor_bins_frac_rated"), [0.3, 0.1])):
            with self.assertRaises(ValueError, msg=str(path)):
                u.validate_protocol(mutated(path, value))
        p = protocol()
        p["arms"].pop("TSMixer_state_normal")
        with self.assertRaises(ValueError):
            u.validate_protocol(p)


class DataIdentityTests(unittest.TestCase):
    def meta(self):
        return {"schema_version": 2, "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022], "curve_years": [2016, 2017, 2018, 2019],
                "prepared": {"sha256": "abc"}, "labels_modified": False, "clipping": False, "interpolation": False}

    def test_preparation_identity(self):
        p = protocol()
        self.assertTrue(u.verify_preparation(self.meta(), p, "abc"))
        for change in ({"years": self.meta()["years"] + [2023]}, {"curve_years": [2016, 2017]}, {"schema_version": 1},
                       {"labels_modified": True}, {"years": [2016, 2017, 2018, 2019, 2020, 2021]}):
            with self.assertRaises(ValueError):
                u.verify_preparation(self.meta() | change, p, "abc")
        with self.assertRaises(ValueError):
            u.verify_preparation(self.meta(), p, "other")

    def test_registered_v28_rows_must_reappear_unchanged(self):
        stamps = pd.date_range("2021-12-31 20:00", periods=8, freq="h", tz="UTC").astype(str)
        v29 = pd.DataFrame([{"timestamp_hour_end_utc": s, "turbine_id": t, "power_kw": float(i * 10 + t), "flag_stopped": i == 3}
                            for i, s in enumerate(stamps) for t in (1, 2)])
        v28 = v29[v29.timestamp_hour_end_utc <= "2022-01-01 00:00:00+00:00"].copy()
        self.assertEqual(u.verify_development_rows(v29.sample(frac=1, random_state=0), v28)["rows_compared"], len(v28))
        changed = v29.copy()
        changed.loc[0, "power_kw"] += 1e-9
        with self.assertRaisesRegex(ValueError, "power_kw"):
            u.verify_development_rows(changed, v28)


class GuardTests(unittest.TestCase):
    def test_rule_uses_issue_anchor_and_strict_threshold(self):
        base = np.full((3, 4), 500.)
        anchor = np.repeat(np.array([[10.], [41.], [100.]]), 4, axis=1)
        guarded = u.guard_prediction(base, anchor, 2050., .02)       # threshold 41 kW, strict
        np.testing.assert_array_equal(guarded[0], anchor[0])
        np.testing.assert_array_equal(guarded[1:], base[1:])
        with self.assertRaises(ValueError):
            u.guard_prediction(base[:2], anchor, 2050., .02)
        bad = anchor.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            u.guard_prediction(base, bad, 2050., .02)
        bad = anchor.copy()
        bad[0, 3] = np.nan
        with self.assertRaises(ValueError):
            u.guard_prediction(base, bad, 2050., .02)
        bad[0, 3] = 99.
        with self.assertRaisesRegex(ValueError, "same issue-time"):
            u.guard_prediction(base, bad, 2050., .02)


NOISE = {"TSMixer_last10_normal": 100, "TSMixer_last10_all": 99, "TSMixer_state_normal": 97, "TSMixer_state_all": 75}


def fake_evaluation(root, p, rng):
    folder = Path(root) / "evaluation"
    folder.mkdir(parents=True)
    issues = 160
    times = np.repeat(np.datetime64("2022-03-01T00", "h") + np.arange(issues) * 3 * np.timedelta64(1, "h"), 6)
    times = (times[:, None] + np.arange(1, 5)[None] * np.timedelta64(1, "h")).astype("datetime64[ns]")
    n = len(times)
    low = rng.random(n) < .15
    level = rng.normal(600, 400, n)
    truth = level[:, None] + rng.normal(0, 80, (n, 4)) * np.sqrt(np.arange(1, 5))[None]
    truth[low] = rng.uniform(0, 30, (int(low.sum()), 4))
    anchor = np.where(low, rng.uniform(0, 20, n), level + rng.normal(0, 40, n))
    stopped = low & (rng.random(n) < .5)
    cohort = {"truth": truth, "times": times, "turbines": np.tile(np.arange(1, 7, dtype=np.int16), issues),
              "window_starts": np.arange(n, dtype=np.int64), "normal_mask": ~stopped & (rng.random(n) > .03),
              "PersistenceHour": np.repeat((anchor + rng.normal(0, 40, n))[:, None], 4, axis=1),
              "PersistenceLast10": np.repeat(anchor[:, None], 4, axis=1),
              "issue_stopped_frac": np.where(stopped, 1., 0.), "issue_capacity_frac": np.where(rng.random(n) < .05, .5, 1.)}
    np.savez_compressed(folder / "test_cohort.npz", **cohort)
    common = rng.normal(0, 1, truth.shape)
    predictions = {}
    for arm in u.TRAINED_ARMS:
        for seed in p["seeds"]:
            prediction = truth + NOISE[arm] * (.8 * common + .6 * rng.normal(0, 1, truth.shape))
            if arm != "TSMixer_state_all":
                prediction[low] += 150.                                  # restart expected where the turbine is stopped
            predictions[(arm, seed)] = prediction
            np.savez_compressed(folder / f"{arm}_s{seed}.npz", prediction=prediction, window_starts=cohort["window_starts"])
    guard = p["arms"][u.GUARD_ARM]
    for seed in p["seeds"]:
        prediction = u.guard_prediction(predictions[(guard["base_arm"], seed)], cohort["PersistenceLast10"],
                                        p["data"]["rated_power_kw"], guard["threshold_frac_rated"])
        np.savez_compressed(folder / f"{u.GUARD_ARM}_s{seed}.npz", prediction=prediction, window_starts=cohort["window_starts"])
    return cohort


class EvaluationTests(unittest.TestCase):
    def test_rules_cohorts_ensemble_and_regimes(self):
        p = protocol()
        p["statistics"]["replicates"] = 300
        with tempfile.TemporaryDirectory() as tmp:
            cohort = fake_evaluation(tmp, p, np.random.default_rng(0))
            out = u.evaluate_registered(tmp, p, "TSMixer_state_all")
        self.assertEqual(set(out["metrics"]), {"normal", "all"})
        self.assertEqual(out["primary_cohort"], "all")
        self.assertGreater(out["windows"]["all"], out["windows"]["normal"])
        self.assertEqual(set(out["comparisons"]["all"]), set(u.COMPARISONS))
        self.assertEqual(out["comparisons"]["all"]["C5"]["candidate"], "TSMixer_state_all")
        self.assertEqual(out["comparisons"]["all"]["C6"]["reference"], u.GUARD_ARM)
        self.assertEqual(out["rules"], {"state_information_supported": True, "all_window_training_supported": True,
                                        "learned_state_beats_rule": True, "beats_naive": True})
        for cohort_name in u.COHORTS:
            ensemble = out["ensemble"][cohort_name]["mae"]
            self.assertLessEqual(ensemble, out["metrics"][cohort_name]["TSMixer_state_all"]["mae"] + 1e-9)
        regimes = {row["regime"]: row for row in out["regimes"]["all"]}
        self.assertEqual(set(regimes), {"anchor<0.02_rated", "0.02<=anchor<0.1_rated", "0.1<=anchor<0.3_rated",
                                        "0.3<=anchor<0.6_rated", "0.6<=anchor<0.9_rated", "anchor>=0.9_rated",
                                        "stopped_at_issue", "derated_at_issue", "all_windows"})
        self.assertEqual(regimes["all_windows"]["windows"], len(cohort["truth"]))
        self.assertAlmostEqual(regimes["all_windows"]["TSMixer_last10_all_mae"], out["metrics"]["all"]["TSMixer_last10_all"]["mae"])
        self.assertLess(regimes["anchor<0.02_rated"][f"{u.GUARD_ARM}_mae"], regimes["anchor<0.02_rated"]["TSMixer_last10_normal_mae"])
        self.assertTrue(all(0 <= c["holm_p"] <= 1 for c in out["comparisons"]["normal"].values()))

    def test_guard_arrays_must_follow_the_rule(self):
        p = protocol()
        p["statistics"]["replicates"] = 100
        with tempfile.TemporaryDirectory() as tmp:
            fake_evaluation(tmp, p, np.random.default_rng(0))
            path = Path(tmp) / f"evaluation/{u.GUARD_ARM}_s2025.npz"
            with np.load(path) as saved:
                arrays = {k: saved[k] for k in saved.files}
            arrays["prediction"][0, 0] += 1.
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "fixed rule"):
                u.evaluate_registered(tmp, p, "TSMixer_state_all")

    def test_validation_choice_and_tie_breaks(self):
        p = protocol()
        maes = {"TSMixer_last10_normal": [150.] * 3, "TSMixer_last10_all": [140.] * 3, "TSMixer_state_normal": [150.] * 3,
                "TSMixer_state_all": [140.] * 3, u.GUARD_ARM: [160.] * 3}
        self.assertEqual(u.validation_choice(maes, p)["arm"], "TSMixer_last10_all")      # tie -> fewer parameters
        self.assertEqual(u.validation_choice(maes | {u.GUARD_ARM: [150.] * 3, "TSMixer_last10_all": [150.] * 3,
                                                     "TSMixer_state_all": [150.] * 3}, p)["arm"], "TSMixer_last10_normal")
        self.assertEqual(u.validation_choice(maes | {u.GUARD_ARM: [139.] * 3}, p)["arm"], u.GUARD_ARM)
        with self.assertRaises(ValueError):
            u.validation_choice(maes | {u.GUARD_ARM: [150.] * 2}, p)

    def test_non_inferiority_and_holm(self):
        positive = {"absolute_error": {"ci95_low": .5, "gain": 2.}, "rmse_difference": {"rmse_gain_kw": 1.}, "holm_p": .01}
        comparisons = {c: {n: copy.deepcopy(positive) for n in u.COMPARISONS} for c in u.COHORTS}
        self.assertTrue(all(u.rules(comparisons, 2.0).values()))
        comparisons["normal"]["C1"]["absolute_error"]["ci95_low"] = -1.
        self.assertTrue(u.rules(comparisons, 2.0)["state_information_supported"])
        comparisons["normal"]["C1"]["absolute_error"]["ci95_low"] = -3.
        self.assertFalse(u.rules(comparisons, 2.0)["state_information_supported"])
        comparisons["all"]["C6"]["holm_p"] = .2
        self.assertFalse(u.rules(comparisons, 2.0)["learned_state_beats_rule"])
        comparisons["normal"]["C5"]["absolute_error"]["ci95_low"] = -.1
        self.assertFalse(u.rules(comparisons, 2.0)["beats_naive"])


def write_pre_test_records(run, windows):
    u.write_json(run / "data_manifest.json", {"identity": {}, "series_audit": {"windows": windows}, "test_windows_instantiated": False})
    u.write_json(run / "selection_freeze.json", {"records": [], "validation_choice": {"arm": "TSMixer_state_all"}})


class PostTestManifestTests(unittest.TestCase):
    def test_counts_must_agree_with_registered_cohort(self):
        p = protocol()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            cohort = fake_evaluation(run, p, np.random.default_rng(3))
            pre = {"train": {"all": 900, "normal": 850}, "validation": {"all": 300, "normal": 280}}
            write_pre_test_records(run, pre)
            test = {"all": len(cohort["truth"]), "normal": int(cohort["normal_mask"].sum())}
            manifest = u.post_test_manifest(run, pre | {"test": test})
            self.assertEqual(manifest["windows"]["test"], test)
            self.assertEqual(len(manifest["test_predictions_sha256"]), len(u.SEEDED_ARMS) * len(p["seeds"]))
            with self.assertRaisesRegex(ValueError, "registered test cohort"):
                u.post_test_manifest(run, pre | {"test": {"all": test["all"], "normal": test["normal"] - 1}})
            with self.assertRaisesRegex(ValueError, "changed after opening"):
                u.post_test_manifest(run, pre | {"train": {"all": 901, "normal": 850}, "test": test})
            write_pre_test_records(run, pre | {"test": test})
            with self.assertRaisesRegex(ValueError, "must not contain test"):
                u.post_test_manifest(run, pre | {"test": test})


def fake_run(root, p):
    run = Path(root) / "run"
    cohort = fake_evaluation(run, p, np.random.default_rng(1))
    records, maes = [], {}
    for arm in u.SEEDED_ARMS:
        for seed in p["seeds"]:
            mae = 150. - (20 if arm == "TSMixer_state_all" else 0)
            records.append({"arm": arm, "seed": seed, "best_epoch": 3, "train_cohort": "all" if arm.endswith("_all") else "normal",
                            "validation": {"mae": mae, "rmse": 220., "mse": 48400.}})
            maes.setdefault(arm, []).append(mae)
    pre = {"train": {"all": 900, "normal": 850}, "validation": {"all": 300, "normal": 280}}
    write_pre_test_records(run, pre)
    freeze = {"records": records, "validation_choice": u.validation_choice(maes, p)}
    u.write_json(run / "selection_freeze.json", freeze)
    u.write_json(run / "protocol.json", {"protocol": p})
    test = {"all": len(cohort["truth"]), "normal": int(cohort["normal_mask"].sum())}
    u.write_json(run / "post_test_manifest.json", u.post_test_manifest(run, pre | {"test": test}))
    u.write_json(run / "decision.json", u.evaluate_registered(run, p, freeze["validation_choice"]["arm"]))
    names = ["selection_freeze.json", "decision.json", "post_test_manifest.json", "evaluation/test_cohort.npz"]
    names += [f"evaluation/{arm}_s{seed}.npz" for arm in u.SEEDED_ARMS for seed in p["seeds"]]
    u.write_json(run / "training_completed.json", {"hashes": {name: u.file_sha256(run / name) for name in names}})
    return run


class ReportTests(unittest.TestCase):
    def test_report_recomputes_and_writes_both_cohorts(self):
        p = protocol()
        p["statistics"]["replicates"] = 300
        with tempfile.TemporaryDirectory() as tmp:
            run = fake_run(tmp, p)
            result = u.build_report(run)
            self.assertEqual(result["validation_choice"]["arm"], "TSMixer_state_all")
            self.assertEqual(result["primary_cohort"], "all")
            frame = pd.read_csv(run / "test_metrics.csv")
            self.assertEqual(set(frame.cohort), {"normal", "all"})
            self.assertEqual(len(frame), 2 * (len(u.ARMS) + 1))
            self.assertIn("ENSEMBLE(TSMixer_state_all)", set(frame.arm))
            self.assertIn("lead4_mse", frame.columns)
            self.assertEqual(len(pd.read_csv(run / "test_comparisons.csv")), 2 * len(u.COMPARISONS))
            self.assertEqual(len(pd.read_csv(run / "regime_metrics.csv")), 2 * 9)
            for name in ("test_metrics", "lead_mae", "forecast_trace"):
                self.assertGreater((run / "figures" / f"{name}.png").stat().st_size, 1000)
            text = (run / "report.md").read_text(encoding="utf-8")
            self.assertIn("conditional accuracy only", text)
            self.assertIn("ALL cohort is primary", text)

    def test_tampering_is_refused(self):
        p = protocol()
        p["statistics"]["replicates"] = 100
        for name in ("evaluation/TSMixer_state_all_s2024.npz", "decision.json", "post_test_manifest.json"):
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

    def test_post_test_hashes_are_independently_rechecked(self):
        p = protocol()
        p["statistics"]["replicates"] = 100
        with tempfile.TemporaryDirectory() as tmp:
            run = fake_run(tmp, p)
            path = run / "post_test_manifest.json"
            manifest = u.read_json(path)
            manifest["selection_freeze_sha256"] = "0" * 64
            u.write_json(path, manifest)
            training = u.read_json(run / "training_completed.json")
            training["hashes"]["post_test_manifest.json"] = u.file_sha256(path)
            u.write_json(run / "training_completed.json", training)
            with self.assertRaisesRegex(ValueError, "manifest hashes"):
                u.build_report(run)

    def test_every_statistic_is_recomputed(self):
        from scripts import report_kelmarsh_multiyear_benchmark as r28
        p = protocol()
        p["statistics"]["replicates"] = 100
        with tempfile.TemporaryDirectory() as tmp:
            fake_evaluation(tmp, p, np.random.default_rng(0))
            fresh = u.safe(u.evaluate_registered(tmp, p, "TSMixer_state_all"))
        for path in (("comparisons", "all", "C6", "holm_p"), ("ensemble", "normal", "mae"),
                     ("metrics", "all", u.GUARD_ARM, "mse")):
            changed = copy.deepcopy(fresh)
            node = changed
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] += .01
            with self.assertRaisesRegex(ValueError, "disagree"):
                r28.check_against_registered(fresh, changed)
        changed = copy.deepcopy(fresh)
        changed["regimes"]["all"][0]["TSMixer_state_all_mae"] += .01
        with self.assertRaisesRegex(ValueError, "disagree"):
            r28.check_against_registered(fresh, changed)


@unittest.skipUnless(HAS_TORCH, "torch/sklearn unavailable on this host")
class ModelTests(unittest.TestCase):
    def test_models_and_channel_views(self):
        from scripts import train_kelmarsh_operating_state as task
        from data_provider.kelmarsh_multiyear import KelmarshMultiYearSeries
        p = protocol()
        for arm in u.TRAINED_ARMS:
            config = u.arm_config(p, arm)
            model, count = task.v28.build_model(config)
            out = model(torch.randn(5, 24, config["channels"]))
            self.assertEqual(count, config["expected_parameters"])
            self.assertEqual(out[..., -1].shape, (5, 4))
        series = KelmarshMultiYearSeries(task.synthetic_prepared_frame(), splits={"train": (2016,), "validation": (2017,), "test": (2018,)},
                                         extras=tuple(p["data"]["extras_in_common_series"]), preparation=task.SYNTHETIC_PREPARATION)
        self.assertEqual(series.channels, p["data"]["channels"])
        windows = series.windows("train", "all")
        last10 = task.v28.ChannelView(windows, p["data"]["channel_sets"]["last10"])
        x, _, _, _ = last10[0]
        np.testing.assert_array_equal(x, windows[0][0][:, [0, 1, 2, 3, 4, 5, 6, 7, 10]])
        self.assertGreater(len(series.windows("train", "all")), len(series.windows("train", "normal")))


@unittest.skipUnless(HAS_TORCH, "torch/sklearn unavailable on this host")
class ControllerTests(unittest.TestCase):
    def make(self, tmp):
        from scripts import train_kelmarsh_operating_state as task
        from data_provider.kelmarsh_multiyear import KelmarshMultiYearSeries
        self.task = task
        p = copy.deepcopy(protocol())
        p["data"]["splits"] = {"train": [2016], "validation": [2017], "test": [2018]}
        p["seeds"] = [2024]
        p["training"].update(max_epochs=2, scheduler_t_max=2)
        p["budget"].update(fits=4, new_fits_max=4)
        p["statistics"]["replicates"] = 200

        def factory(kwargs):
            return KelmarshMultiYearSeries(task.synthetic_prepared_frame(), preparation=task.SYNTHETIC_PREPARATION, **kwargs)
        return task.V29Run(Path(tmp) / "run", protocol=p, protocol_sha="test", series_factory=factory, require_preflight=False,
                           write_pointer=False, lock_path=Path(tmp) / "lock", verify_data=False)

    def test_resume_after_prediction_write_reuses_fits(self):
        from unittest.mock import patch
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp, \
                patch("scripts.train_kelmarsh_anchored_residual.gpu_snapshot", return_value=None):
            run = self.make(tmp)
            original = run._register_npz

            def interrupted(path, arrays):
                original(path, arrays)
                if path.name == "TSMixer_last10_normal_s2024.npz":
                    raise RuntimeError("synthetic evaluation interruption")

            with patch.object(run, "_register_npz", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "evaluation interruption"):
                    run.run()
            fitted = {str(p.relative_to(run.dir)): u.file_sha256(p) for p in run.dir.glob("runs/*/*/trained.json")}
            self.assertEqual(len(fitted), 4)
            again = self.make(tmp)
            with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("retrained")):
                self.assertEqual(again.run(), "COMPLETED")
            u.verify_registered_files(again.dir, fitted)
            u.verify_registered_files(again.dir, u.read_json(again.dir / "completed.json"))

    def test_end_to_end_freeze_before_test_guard_manifest_and_reuse(self):
        from unittest.mock import patch
        torch.set_num_threads(2)
        with patch("scripts.train_kelmarsh_anchored_residual.gpu_snapshot", return_value=None):
            with tempfile.TemporaryDirectory() as tmp:
                run = self.make(tmp)
                self.assertEqual(run.run(), "COMPLETED")
                self.assertEqual(len(run.registered_fits()), 4)
                events = [json.loads(line) for line in (run.dir / "events.jsonl").read_text().splitlines()]
                frozen = next(i for i, e in enumerate(events) if e["event"] == "S1_frozen")
                first_test = next(i for i, e in enumerate(events) if e.get("split") == "test")
                self.assertLess(frozen, first_test)
                self.assertFalse(any(e.get("split") == "test" for e in events[:frozen]))
                freeze = u.read_json(run.dir / "selection_freeze.json")
                self.assertEqual({r["arm"] for r in freeze["records"]}, set(u.SEEDED_ARMS))
                decision = u.read_json(run.dir / "decision.json")
                self.assertEqual(set(decision["metrics"]), {"normal", "all"})
                manifest = u.read_json(run.dir / "post_test_manifest.json")
                self.assertEqual(manifest["windows"]["test"], decision["windows"])
                self.assertNotIn("test", u.read_json(run.dir / "data_manifest.json")["series_audit"]["windows"])
                self.assertTrue((run.dir / f"evaluation/{u.GUARD_ARM}_s2024.npz").exists())
                self.assertEqual(u.read_json(run.dir / "status.json")["report"], "REPORT_COMPLETED")
                for arm in u.TRAINED_ARMS:
                    trained = u.read_json(run.dir / f"runs/{arm}/s2024/trained.json")
                    self.assertEqual(trained["parameters"], u.arm_config(run.p, arm)["expected_parameters"])
                again = self.make(tmp)
                with patch.object(self.task.v28, "fit_config", side_effect=AssertionError("retrained")):
                    self.assertEqual(again.run(), "COMPLETED")

    def test_test_windows_refused_before_freeze_and_preflight_required(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp)
            run.dir.mkdir(parents=True)
            run.record_sources()
            run.stage_d0()
            self.assertNotIn("test", run.series.splits)
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


if __name__ == "__main__":
    unittest.main()
