"""v25 recovery tests: report-only reruns, epoch-boundary resume, registered-fit reuse, controller stages."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils import kelmarsh_backbone_information as v25

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/kelmarsh_backbone_information_protocol_v25.json"
try:
    import torch
    import sklearn  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def protocol(**training):
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    p["statistics"]["replicates"] = 200
    p["training"].update(training)
    return p


def cohort(fold, issues=40):
    rng = np.random.default_rng(100 + fold)
    times = np.repeat(pd.date_range("2016-10-01", periods=issues, freq="3h").to_numpy(), 6)[:, None] + \
        (np.arange(1, 5) * np.timedelta64(1, "h"))[None]
    truth = 500 + 300 * np.sin(np.arange(issues * 6) / 9.)[:, None] + rng.normal(0, 40, (issues * 6, 4))
    return {"times": times.astype("datetime64[ns]"), "turbines": np.tile(np.arange(1, 7, dtype=np.int16), issues),
            "window_starts": np.arange(issues * 6, dtype=np.int64), "truth": truth,
            "persistence": truth + rng.normal(0, 80, truth.shape)}


def build_fake_run(root, outer):
    """Directory with the exact registered-file layout the v25 controller writes (no torch needed)."""
    p = protocol()
    run, ref = root / "run", root / "ref"
    v25.write_json(run / "protocol.json", {"protocol": p, "protocol_id": p["protocol_id"]})
    v25.write_json(run / "a0_result.json", {"stage": "A0"})
    seeds = [2024, 2025, 2026] if outer else [2024]
    scale = {"TSMixerFull": 30., "TSMixerPowerOnly": 35., "DLinear": 40.}
    index, rng = {}, np.random.default_rng(0)
    for fold in (0, 1, 2):
        inner, outer_cohort = cohort(fold), cohort(fold + 10)
        refrun = ref / p["reference"]["dir"] / "runs" / f"f{fold}_s2024" / "DLinear"
        v25.write_json(refrun / "trained.json", {"best_epoch": 1})
        v25.write_json(refrun / "training_history.json", [{"epoch": 0, "selection": {"mae": 170., "rmse": 250.}},
                                                         {"epoch": 1, "train_loss": .4, "selection": {"mae": 160., "rmse": 240.}}])
        if outer:
            prediction = outer_cohort["truth"] + rng.normal(0, 40, outer_cohort["truth"].shape)
            np.savez_compressed(refrun / "raw_predictions.npz", prediction=prediction, **outer_cohort)
            v25.write_json(refrun / "result.json", {"metrics": v25.point_metrics(prediction, outer_cohort["truth"],
                                                                                 outer_cohort["persistence"], 2050.)})
            index[f"f{fold}/s2024/DLinear"] = {"fold": fold, "seed": 2024, "arm": "DLinear", "path": str(refrun / "raw_predictions.npz"),
                                               "sha256": v25.file_sha256(refrun / "raw_predictions.npz"), "source": "v20"}
            (run / "runs").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(run / "runs" / f"f{fold}_outer_cohort.npz", **outer_cohort)
            index[f"f{fold}/Persistence"] = {"fold": fold, "seed": None, "arm": "Persistence",
                                             "path": str(run / "runs" / f"f{fold}_outer_cohort.npz"),
                                             "sha256": v25.file_sha256(run / "runs" / f"f{fold}_outer_cohort.npz")}
        for seed in seeds:
            for arm in ("TSMixerFull", "TSMixerPowerOnly") + (("DLinear",) if seed != 2024 else ()):
                folder = run / "runs" / f"f{fold}_s{seed}" / arm
                folder.mkdir(parents=True)
                prediction = inner["truth"] + rng.normal(0, scale[arm], inner["truth"].shape)
                np.savez_compressed(folder / "selection_predictions.npz", prediction=prediction, **inner)
                metrics = v25.point_metrics(prediction, inner["truth"], inner["persistence"], 2050.)
                v25.write_json(folder / "training_history.json", [{"epoch": 0, "selection": {"mae": metrics["mae"] + 9}},
                               {"epoch": 1, "train_loss": .3, "selection": metrics}])
                v25.write_json(folder / "trained.json", {
                    "arm": arm, "seed": seed, "best_epoch": 1, "epochs_completed": 1, "stopped_early": True,
                    "selection_best": metrics, "trainable_parameters": 200 if arm == "DLinear" else 3618, "device": "cpu",
                    "timing": {"train_seconds": 1., "assess_seconds": .1}, "peak_memory": {"kind": "rss", "value": 1},
                    "artifacts": {"selection_predictions.npz": v25.file_sha256(folder / "selection_predictions.npz")}})
                if outer:
                    prediction = outer_cohort["truth"] + rng.normal(0, scale[arm], outer_cohort["truth"].shape)
                    np.savez_compressed(folder / "outer_predictions.npz", prediction=prediction, **outer_cohort)
                    v25.write_json(folder / "result.json", {"inference_seconds": .01, "metrics": v25.point_metrics(
                        prediction, outer_cohort["truth"], outer_cohort["persistence"], 2050.)})
                    index[f"f{fold}/s{seed}/{arm}"] = {"fold": fold, "seed": seed, "arm": arm,
                                                       "path": str(folder / "outer_predictions.npz"),
                                                       "sha256": v25.file_sha256(folder / "outer_predictions.npz"), "source": "v25"}
    g1 = {"G1": {"passed": outer, "checks": {"macro_mae_rmse_better_than_dlinear": outer}}, "I1": {"supported": False}}
    v25.write_json(run / "inner_gate_seed2024.json", g1)
    if outer:
        v25.write_json(run / "inner_gate_all_seeds.json", {"G2": {"passed": True, "checks": {}}, "I2": {"supported": False}})
        v25.write_json(run / "outer_index.json", {"entries": index})
    v25.write_json(run / "training_completed.json", {"outer_evaluated": outer, "hashes": {}})
    return run, ref


class ReportTests(unittest.TestCase):
    def test_inner_only_report_after_failed_gate_has_no_outer_numbers(self):
        from scripts import report_kelmarsh_backbone_information as report
        with tempfile.TemporaryDirectory() as tmp:
            run, ref = build_fake_run(Path(tmp), outer=False)
            result = report.build_report(run, reference_root=ref)
            self.assertFalse(result["outer_evaluated"])
            self.assertIn("G1 FAILED", result["conclusion"])
            self.assertIn("no new outer accuracy", result["conclusion"])
            self.assertEqual(set(result["macro"]), {"inner_selection"})
            self.assertEqual(result["wiki"], "not_applicable_wiki_disabled")
            for name in ("accuracy_overview", "lead_metrics", "training_history", "forecast_trace", "paired_gain"):
                self.assertTrue((run / "figures" / f"{name}.png").stat().st_size > 1000, name)
            frame = pd.read_csv(run / "metrics_by_lead.csv")
            self.assertEqual(sorted(frame.lead_hours.unique()), [1, 2, 3, 4])
            self.assertEqual(result["independent_recomputation_max_abs_diff"], 0.)
            markdown = (run / "report.md").read_text(encoding="utf-8")
            self.assertIn("G1 passed: False", markdown)
            self.assertIn("not a percentage accuracy", markdown)
            self.assertTrue((run / "report_manifest.json").exists())

    def test_outer_report_recomputes_pairs_and_applies_holm(self):
        from scripts import report_kelmarsh_backbone_information as report
        with tempfile.TemporaryDirectory() as tmp:
            run, ref = build_fake_run(Path(tmp), outer=True)
            result = report.build_report(run, reference_root=ref)
            stats = v25.read_json(run / "paired_statistics.json")["outer_validation"]
            self.assertEqual(set(stats["primary_family"]["raw_p"]), {"0", "1", "2"})
            block = stats["folds"]["0"]
            self.assertEqual(block["primary_full_vs_dlinear"]["seeds_used"], [2024, 2025, 2026])
            self.assertEqual(set(block["full_vs_dlinear_per_seed_library"]), {"2024", "2025", "2026"})
            self.assertIn("NOT a reuse", block["primary_full_vs_dlinear"]["rmse_difference"]["method"])
            self.assertEqual(block["primary_full_vs_dlinear"]["squared_error"]["units"], "kW^2")
            macro = result["macro"]["outer_validation"]
            self.assertEqual(macro["TSMixerFull"]["units"], 9)
            self.assertEqual(macro["Persistence"]["units"], 3)
            self.assertIn("G1 and G2 passed", result["conclusion"])

    def test_tampered_or_unregistered_predictions_are_refused(self):
        from scripts import report_kelmarsh_backbone_information as report
        with tempfile.TemporaryDirectory() as tmp:
            run, ref = build_fake_run(Path(tmp), outer=True)
            target = run / "runs/f1_s2025/TSMixerFull/outer_predictions.npz"
            with np.load(target) as saved:
                arrays = {k: saved[k] for k in saved.files}
            arrays["prediction"] = arrays["prediction"] + 1.
            np.savez_compressed(target, **arrays)
            with self.assertRaises(ValueError):
                report.build_report(run, reference_root=ref)
        with tempfile.TemporaryDirectory() as tmp:
            run, ref = build_fake_run(Path(tmp), outer=True)
            v25.write_json(run / "training_completed.json", {"outer_evaluated": False, "hashes": {}})
            with self.assertRaises(ValueError):
                report.build_report(run, reference_root=ref)


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
class FitResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import train_kelmarsh_backbone_information as task
        cls.task = task
        torch.set_num_threads(2)
        cls.p = protocol(max_epochs=4, scheduler_t_max=4)
        cls.fit, cls.selection = task.SyntheticWindows(160, 1), task.SyntheticWindows(48, 2)

    def run_fit(self, out, **kwargs):
        return self.task.fit_arm("TSMixerFull", self.p, fit=self.fit, selection=self.selection, contract={"c": 1}, seed=11,
                                 device=torch.device("cpu"), out=out, **kwargs)

    def assert_same_final_state(self, a, b):
        x = torch.load(a / "last.pt", map_location="cpu", weights_only=False)
        y = torch.load(b / "last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(self.task.state_digest(x["model"]), self.task.state_digest(y["model"]))
        for gx, gy in zip(x["optimizer"]["state"].values(), y["optimizer"]["state"].values()):
            for key in gx:
                self.assertTrue(torch.equal(torch.as_tensor(gx[key]), torch.as_tensor(gy[key])), key)
        self.assertEqual(x["scheduler"], y["scheduler"])
        self.assertTrue(torch.equal(x["rng"]["torch_cpu"], y["rng"]["torch_cpu"]))
        self.assertEqual(x["rng"]["python"], y["rng"]["python"])
        np.testing.assert_array_equal(x["rng"]["numpy"][1], y["rng"]["numpy"][1])
        strip = lambda h: [{k: v for k, v in e.items() if "seconds" not in k} for e in h]
        self.assertEqual(strip(x["history"]), strip(y["history"]))
        self.assertEqual((x["epoch"], x["best_epoch"], x["best"]), (y["epoch"], y["best_epoch"], y["best"]))
        bx = torch.load(a / "best.pt", map_location="cpu", weights_only=False)["model"]
        by = torch.load(b / "best.pt", map_location="cpu", weights_only=False)["model"]
        self.assertEqual(self.task.state_digest(bx), self.task.state_digest(by))

    def test_epoch_boundary_and_mid_epoch_interruptions_resume_identically(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            reference = self.run_fit(tmp / "straight")
            with self.assertRaises(self.task.SimulatedInterrupt):
                self.run_fit(tmp / "boundary", stop_after_epoch=2)
            self.assertFalse((tmp / "boundary/trained.json").exists())
            self.assertEqual(torch.load(tmp / "boundary/last.pt", weights_only=False)["epoch"], 2)
            resumed = self.run_fit(tmp / "boundary")
            self.assert_same_final_state(tmp / "straight", tmp / "boundary")
            with self.assertRaises(self.task.SimulatedInterrupt):
                self.run_fit(tmp / "inside", stop_inside_epoch=(3, 1))
            self.assertEqual(v25.read_json(tmp / "inside/fit_progress.json")["epoch_in_progress"], 3)
            self.run_fit(tmp / "inside")
            self.assert_same_final_state(tmp / "straight", tmp / "inside")
            self.assertEqual(resumed["best_epoch"], reference["best_epoch"])
            self.assertEqual(reference["artifacts"].keys(), {"init.pt", "best.pt", "last.pt", "training_history.json",
                                                             "selection_predictions.npz"})

    def test_registered_fit_is_reused_without_training_and_tampering_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "fit"
            first = self.run_fit(out)
            with patch.object(self.task, "make_loader", side_effect=AssertionError("retrained")):
                again = self.run_fit(out)
            self.assertTrue(again["reused_from_registry"])
            self.assertEqual(again["best_state_sha256"], first["best_state_sha256"])
            with self.assertRaises(ValueError):
                self.task.fit_arm("TSMixerFull", self.p, fit=self.fit, selection=self.selection, contract={"c": 2},
                                  seed=11, device=torch.device("cpu"), out=out)
            with (out / "best.pt").open("ab") as handle:
                handle.write(b"x")
            with self.assertRaises(ValueError):
                self.run_fit(out)

    def test_interrupted_fit_refuses_a_changed_contract_and_pairs_share_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with self.assertRaises(self.task.SimulatedInterrupt):
                self.run_fit(tmp / "a", stop_after_epoch=1)
            with self.assertRaises(ValueError):
                self.task.fit_arm("TSMixerFull", self.p, fit=self.fit, selection=self.selection, contract={"c": 3},
                                  seed=11, device=torch.device("cpu"), out=tmp / "a")
            full = self.run_fit(tmp / "full")
            power = self.task.fit_arm("TSMixerPowerOnly", self.p, fit=self.fit, selection=self.selection, contract={"c": 1},
                                      seed=11, device=torch.device("cpu"), out=tmp / "power", init_reference=tmp / "full/init.pt")
            self.assertEqual(full["init_state_sha256"], power["init_state_sha256"])
            with self.assertRaises(ValueError):
                self.task.fit_arm("TSMixerPowerOnly", self.p, fit=self.fit, selection=self.selection, contract={"c": 1},
                                  seed=12, device=torch.device("cpu"), out=tmp / "other", init_reference=tmp / "full/init.pt")


@unittest.skipUnless(HAS_TORCH, "torch/sklearn unavailable on this host")
class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import train_kelmarsh_backbone_information as task
        cls.task = task
        torch.set_num_threads(2)
        cls.frame = synthetic_frame()
        cls.series = {fold: synthetic_series(fold, cls.frame) for fold in (0, 1, 2)}
        # Controller tests never touch a shared GPU: the pair-device probe reports "no CUDA".
        cls.gpu_patch = patch.object(task, "gpu_snapshot", return_value=None)
        cls.gpu_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.gpu_patch.stop()

    def make(self, tmp, dlinear_inner, **kwargs):
        p = protocol(max_epochs=2, scheduler_t_max=2)
        p["dlinear_cache"]["search_glob"] = str(Path(tmp) / "nocache/{fold}/{seed}/trained.json")
        pin = {"commit": "a" * 40, "file_sha256": "b" * 64, "license": "Apache-2.0"}
        self.requested = []

        def factory(fold):
            series = copy.copy(self.series[fold])
            original = series.subset
            def spy(name):
                self.requested.append(name)
                return original(name)
            series.subset = spy
            return series
        run = self.task.V25Run(Path(tmp) / "run", protocol=p, protocol_sha="test-protocol", pin=pin, series_factory=factory,
                               require_preflight=False, test_reference={f: {"mae": dlinear_inner, "rmse": dlinear_inner} for f in (0, 1, 2)},
                               write_pointer=False, lock_path=Path(tmp) / "lock", reference_root=Path(tmp) / "ref", **kwargs)
        return run

    def test_failed_g1_stops_without_any_outer_request_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp, dlinear_inner=0.)  # DLinear "perfect" -> G1 must fail
            with patch("scripts.train_kelmarsh_backbone_information.load_pin"):
                self.assertEqual(run.run(), "COMPLETED")
            self.assertEqual(self.requested, [])
            self.assertEqual(run.fit_calls, [(f, 2024, a) for f in (0, 1, 2) for a in ("TSMixerFull", "TSMixerPowerOnly")])
            self.assertFalse(v25.read_json(run.dir / "inner_gate_seed2024.json")["G1"]["passed"])
            self.assertFalse((run.dir / "outer_index.json").exists())
            self.assertFalse(list(run.dir.rglob("outer_predictions.npz")))
            self.assertFalse(v25.read_json(run.dir / "training_completed.json")["outer_evaluated"])
            self.assertEqual(v25.read_json(run.dir / "status.json")["report"], "REPORT_COMPLETED")
            for fold in (0, 1, 2):
                full = v25.read_json(run.dir / f"runs/f{fold}_s2024/TSMixerFull/trained.json")
                power = v25.read_json(run.dir / f"runs/f{fold}_s2024/TSMixerPowerOnly/trained.json")
                self.assertEqual(full["init_state_sha256"], power["init_state_sha256"])
                self.assertEqual(full["contract"]["fit_window_hash"], power["contract"]["fit_window_hash"])
                self.assertEqual(full["contract"]["device_type"], power["contract"]["device_type"])
            subsets = {json.loads(l).get("subset") for l in (run.dir / "events.jsonl").read_text().splitlines()}
            self.assertNotIn("validation", subsets)
            again = self.make(tmp, dlinear_inner=0.)
            with patch.object(self.task, "fit_arm", side_effect=AssertionError("retrained")):
                self.assertEqual(again.run(), "COMPLETED")

    def test_report_failure_never_reruns_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp, dlinear_inner=0.)
            with patch("scripts.train_kelmarsh_backbone_information.load_pin"), \
                    patch("scripts.report_kelmarsh_backbone_information.build_report", side_effect=RuntimeError("figure bug")):
                with self.assertRaises(RuntimeError):
                    run.run()
            status = v25.read_json(run.dir / "status.json")
            self.assertEqual((status["train"], status["report"]), ("TRAIN_COMPLETED", "REPORT_FAILED"))
            self.assertFalse((run.dir / "completed.json").exists())
            again = self.make(tmp, dlinear_inner=0.)
            with patch.object(self.task, "fit_arm", side_effect=AssertionError("retrained")):
                self.assertEqual(again.run(report_only=True), "COMPLETED")
            self.assertTrue((run.dir / "completed.json").exists())

    def test_preflight_and_lock_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp, dlinear_inner=0.)
            run.require_preflight = True
            with patch("scripts.train_kelmarsh_backbone_information.load_pin"), patch.object(self.task, "PREFLIGHT_DIR", Path(tmp) / "pf"):
                with self.assertRaises(ValueError):
                    run.run()
            self.assertFalse(list(run.dir.rglob("trained.json")))
            holder = v25.acquire_exclusive_lock(Path(tmp) / "lock")
            with self.assertRaises(BlockingIOError):
                self.make(tmp, dlinear_inner=0.).run()
            holder.close()

    def test_passed_gates_run_a2_then_one_outer_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run = self.make(tmp, dlinear_inner=1e9)
            p = run.p
            for fold in (0, 1, 2):
                validation = self.series[fold].subset("validation")
                truth, anchor = validation.truth_and_anchor()
                arrays = {"times": validation.target_times(), "turbines": validation.turbines[validation.window_starts],
                          "window_starts": validation.window_starts, "truth": truth, "persistence": anchor}
                base = tmp / "ref" / p["reference"]["dir"] / "runs" / f"f{fold}_s2024"
                (base / "Persistence").mkdir(parents=True)
                (base / "DLinear").mkdir(parents=True)
                np.savez_compressed(base / "Persistence/raw_predictions.npz", prediction=anchor, **arrays)
                np.savez_compressed(base / "DLinear/raw_predictions.npz", prediction=anchor + 5., **arrays)
                v25.write_json(base / "DLinear/result.json", {"metrics": v25.point_metrics(anchor + 5., truth, anchor, 2050.)})
                v25.write_json(base / "DLinear/trained.json", {"best_epoch": 0})
                v25.write_json(base / "DLinear/training_history.json", [{"epoch": 0, "selection": {"mae": 1., "rmse": 1.}}])
            real_g1, real_g2 = v25.gate_g1, v25.gate_g2
            force = lambda real: (lambda *a, **k: {**real(*a, **k), "passed": True})
            with patch("scripts.train_kelmarsh_backbone_information.load_pin"), \
                    patch.object(self.task.v25, "gate_g1", side_effect=force(real_g1)), \
                    patch.object(self.task.v25, "gate_g2", side_effect=force(real_g2)):
                self.assertEqual(run.run(), "COMPLETED")
            self.assertEqual(len(run.registered_new_fits()), 24)
            order = run.fit_calls[6:]
            self.assertEqual(order[:3], [(0, 2025, "TSMixerFull"), (0, 2025, "TSMixerPowerOnly"), (0, 2025, "DLinear")])
            self.assertEqual(self.requested, ["validation"] * 3)
            freeze = v25.read_json(run.dir / "outer_freeze.json")
            self.assertEqual(len(freeze["freeze"]["fits"]), 24)
            entries = v25.read_json(run.dir / "outer_index.json")["entries"]
            self.assertEqual(len(entries), 30)
            self.assertIn("v20", entries["f0/s2024/DLinear"]["source"])
            result = v25.read_json(run.dir / "result.json")
            self.assertTrue(result["outer_evaluated"])
            self.assertLess(result["independent_recomputation_max_abs_diff"], 1e-6)
            events = [json.loads(l) for l in (run.dir / "events.jsonl").read_text().splitlines()]
            g2_index = next(i for i, e in enumerate(events) if e["event"] == "G2")
            first_outer = next(i for i, e in enumerate(events) if e.get("subset") == "validation")
            self.assertLess(g2_index, first_outer)


if __name__ == "__main__":
    unittest.main()
