from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from models.PowerAnchoredTimeDART import Model
from scripts import train_kelmarsh_matched_recipe as task


def fixture():
    dates = pd.date_range("2016-05-01", periods=200, freq="h").to_numpy()
    return SimpleNamespace(dates=dates, starts=np.arange(0, 173, 3), raw=np.ones((200, 7)),
        values=np.ones((200, 7), dtype=np.float32), turbines=np.ones(200, dtype=np.int16),
        input_len=24, horizon=4, scaler=SimpleNamespace(mean_=np.zeros(7), scale_=np.ones(7)),
        bounds={"base_fit_end": dates[100], "base_selection_end": dates[120],
                "calibration_early_end": dates[150], "train_end": dates[170]},
        subset=lambda _: (_ for _ in ()).throw(AssertionError("Outer subset accessed")))


def stage_patches():
    return (patch.object(task.shared, "gpu_available_to_this_process"),
            patch.object(task.shared, "calibrate_regime_thresholds_from_dataset",
                         return_value={"down_thresh": -.1, "up_thresh": .1}),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.cuda.get_rng_state_all", return_value=[]))


class MatchedRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_frozen_recipe_declares_one_schedule_change_and_one_pretraining_control(self):
        recipe, v21, effective = task.protocol()
        arms = recipe["arms"]
        self.assertEqual(arms["PretrainedOriginal"]["epochs"], effective["training"]["epochs"])
        self.assertEqual({a["init"] for a in arms.values()}, {"pretrained", "scratch"})
        differing = lambda a, b: [k for k in ("init", "epochs") if arms[a][k] != arms[b][k]]
        self.assertEqual(differing("PretrainedMatched", "PretrainedOriginal"), ["epochs"])
        self.assertEqual(differing("PretrainedMatched", "ScratchMatched"), ["init"])
        self.assertEqual(recipe["folds"], v21["folds"])
        self.assertFalse(recipe["sealed_test_evaluated"])
        self.assertFalse(recipe["wiki_enabled"])
        self.assertEqual(effective["protocol_id"], recipe["protocol_id"])

    def test_arm_protocol_changes_only_epochs_and_never_mutates_the_base(self):
        _, _, effective = task.protocol()
        before = effective["training"]["epochs"]
        arm = task.arm_protocol(effective, {"epochs": 8})
        self.assertEqual(effective["training"]["epochs"], before)
        self.assertEqual(arm["training"]["epochs"], 8)
        arm["training"]["epochs"] = before
        self.assertEqual(arm, effective)

    def test_source_lock_is_unique_and_covers_the_runner(self):
        self.assertEqual(len(task.SOURCE_FILES), len(set(task.SOURCE_FILES)))
        self.assertIn("scripts/train_kelmarsh_matched_recipe.py", task.SOURCE_FILES)
        self.assertIn("scripts/train_kelmarsh_hourly.py", task.SOURCE_FILES)
        self.assertIn("models/PowerAnchoredTimeDART.py", task.SOURCE_FILES)

    def test_residual_slope_measures_overdispersion(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=(50, 4))
        residual = rng.normal(size=(50, 4))
        calibrated = task.residual_slope(base + residual, base + residual, base)
        doubled = task.residual_slope(base + 2 * residual, base + residual, base)
        self.assertAlmostEqual(calibrated["pooled"], 1.0)
        self.assertAlmostEqual(doubled["pooled"], .5)
        self.assertEqual(len(doubled["per_hour"]), 4)
        self.assertIsNone(task.residual_slope(base, base + residual, base)["pooled"])

    def test_gate_rejects_one_lucky_fold_and_rmse_harm_but_does_not_gate_on_dlinear(self):
        recipe, _, _ = task.protocol()
        def rows(candidates, dlinear=(1., 1.)):
            return [{"candidate": {"mae": a, "rmse": b}, "reference": {"mae": 10., "rmse": 20.},
                     "persistence": {"mae": 12., "rmse": 22.},
                     "dlinear": {"mae": dlinear[0], "rmse": dlinear[1]}} for a, b in candidates]
        settings = recipe["outer_gate"]
        self.assertFalse(task.decision(rows([(0, 0), (11, 21), (11, 21)]), settings)["passed"])
        self.assertFalse(task.decision(rows([(9, 21)] * 3), settings)["passed"])
        accepted = task.decision(rows([(9, 19), (9, 19), (10.5, 20.5)]), settings)
        self.assertTrue(accepted["passed"])  # still passes although far worse than the DLinear row
        self.assertEqual(len(accepted["rows"]), 3)
        self.assertIn("dlinear", accepted["macro_inner"])

    def test_paired_effect_sign_convention(self):
        def record(fold, seed, arm, mae):
            return {"fold": fold, "seed": seed, "arm": arm, "metrics": {"mae": mae, "rmse": mae + 1}}
        records = [record(f, s, "A", 9.) for f in (0, 1) for s in (1, 2)] + \
                  [record(f, s, "B", 10.) for f in (0, 1) for s in (1, 2)]
        effect = task.paired_effect(records, "A", "B")
        self.assertEqual(effect["units"], 4)
        self.assertAlmostEqual(effect["mae"]["mean_difference"], -1.)
        self.assertEqual(effect["mae"]["units_where_first_is_better"], 4)
        self.assertEqual(task.paired_effect(records, "B", "A")["mae"]["units_where_first_is_better"], 0)

    def test_seed_mean_ensemble_uses_each_saved_seed(self):
        rng = np.random.default_rng(1)
        truth, base = rng.normal(size=(30, 4)), rng.normal(size=(30, 4))
        first, second = base + rng.normal(size=(30, 4)), base + rng.normal(size=(30, 4))
        recipe = {"seeds": [1, 2]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed, prediction in ((1, first), (2, second)):
                folder = root / "runs" / f"f0_s{seed}" / "Arm"
                folder.mkdir(parents=True)
                np.savez(folder / "inner_predictions.npz", prediction=prediction, truth=truth, anchor=base)
            result = task.ensemble_inner(root, recipe, 0, "Arm", 2050.)
        expected = task.shared.model_metrics((first + second) / 2, truth, base, 2050.)
        self.assertAlmostEqual(result["mae"], expected["mae"])
        self.assertAlmostEqual(result["rmse"], expected["rmse"])
        self.assertIn("slope", result)

    def test_cosine_horizon_follows_the_declared_epochs_and_scratch_differs_only_by_transfer(self):
        _, _, effective = task.protocol()
        data = task.anchor.train_datasets_only(fixture())
        # A real pretrain checkpoint holds no head or channel mixer; keep that shape here.
        torch.manual_seed(7)
        donor = Model(task.shared.model_args("PromptTimeDART", None, effective, torch.device("cpu")))
        state = {k: v for k, v in donor.state_dict().items() if not k.startswith(("head.", "channel_mixer."))}
        patches = stage_patches()
        with tempfile.TemporaryDirectory() as tmp, patches[0], patches[1], patches[2], patches[3]:
            root = Path(tmp)
            initial = root / "pretrain.pt"
            torch.save({"model": state}, initial)
            arm = task.arm_protocol(effective, {"epochs": 2})
            arm["training"]["patience"] = 5
            task.shared.write(root / "protocol.json", {"recipe": {}, "effective": effective})
            outputs = {}
            for name, source in (("Pretrained", initial), ("Scratch", None)):
                output = root / "runs/f0_s2024" / name
                _, outputs[name] = task.shared.train_stage("PromptTimeDART", data["fit"], data["selection"],
                    arm, output, 2024, torch.device("cpu"), initial=source, model_factory=Model)
            pretrained, scratch = root / "runs/f0_s2024/Pretrained", root / "runs/f0_s2024/Scratch"
            for folder in (pretrained, scratch):
                last = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
                self.assertEqual(last["epoch"], 2)
                for rate in last["scheduler"]["_last_lr"]:
                    self.assertAlmostEqual(rate, 0., places=12)  # cosine reaches zero exactly at the declared end
            self.assertIsNone(outputs["Scratch"]["contract"]["initial_checkpoint_sha256"])
            self.assertEqual(outputs["Pretrained"]["contract"]["initial_checkpoint_sha256"], task.shared.sha(initial))
            manifests = {n: task.shared.read(r / "manifest.json") for n, r in (("Pretrained", pretrained), ("Scratch", scratch))}
            self.assertEqual(manifests["Scratch"]["transferred_state_keys"], [])
            self.assertGreater(len(manifests["Pretrained"]["transferred_state_keys"]), 0)
            self.assertEqual(manifests["Scratch"]["parameters"], manifests["Pretrained"]["parameters"])
            self.assertEqual(manifests["Pretrained"]["model_args"]["train_epochs"], 2)
            a = torch.load(pretrained / "checkpoint.pt", map_location="cpu", weights_only=False)["model"]
            b = torch.load(scratch / "checkpoint.pt", map_location="cpu", weights_only=False)["model"]
            for key in a:
                if key.startswith(("head.", "channel_mixer.", "residual_gate")):
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)
            moved = [k for k in manifests["Pretrained"]["transferred_state_keys"]
                     if a[k].dtype.is_floating_point and not torch.equal(a[k], b[k])]
            self.assertGreater(len(moved), 0)  # the control really lacks the transferred weights


def synthetic_inner_run(root, recipe, rng):
    """Write the per-run files that run() leaves behind, with arbitrary but valid numbers."""
    rated, records, persistence, dlinear = 2050., [], {}, {}
    for fold in recipe["folds"]:
        truth, base = rng.uniform(0, rated, (40, 4)), rng.uniform(0, rated, (40, 4))
        persistence[fold] = task.shared.model_metrics(base, truth, base, rated)
        dlinear[fold] = {"mae": 150. + fold, "rmse": 230. + fold}
        previous = root / "previous" / "runs" / f"f{fold}_s{task.PRETRAIN_SEED}" / "PowerAnchoredTimeDART"
        for seed in recipe["seeds"]:
            for arm in recipe["arms"]:
                prediction = base + .5 * (truth - base) + rng.normal(0, 30, base.shape)
                folder = root / "runs" / f"f{fold}_s{seed}" / arm
                folder.mkdir(parents=True)
                np.savez(folder / "inner_predictions.npz", prediction=prediction, truth=truth, anchor=base)
                metrics = task.shared.model_metrics(prediction, truth, base, rated)
                records.append({"fold": fold, "seed": seed, "arm": arm, "best_epoch": int(rng.integers(0, 6)),
                                "metrics": metrics, "slope": task.residual_slope(prediction, truth, base)})
                if arm == recipe["reference_arm"] and seed == task.PRETRAIN_SEED:
                    previous.mkdir(parents=True, exist_ok=True)
                    task.shared.write(previous / "inner_metrics.json", {"candidate": metrics})
    return records, persistence, dlinear, rated


class AggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_inner_summary_runs_end_to_end_and_serialises(self):
        recipe, _, _ = task.protocol()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, persistence, dlinear, rated = synthetic_inner_run(root, recipe, np.random.default_rng(3))
            gate, factorial = task.summarize_inner(root, recipe, records, persistence, dlinear, root / "previous", rated)
            task.shared.write(root / "gate.json", gate)
            task.shared.write(root / "factorial.json", factorial)  # must be strict-JSON serialisable
            self.assertEqual(set(factorial["arms"]), set(recipe["arms"]))
            for summary in factorial["arms"].values():
                self.assertEqual(set(summary["macro_by_seed"]["mae"]), {str(s) for s in recipe["seeds"]})
                self.assertEqual(len(summary["seed_std_of_macro"]), 2)
                self.assertIn("macro_seed_mean_ensemble", summary)
            self.assertEqual(set(factorial["paired_effects"]), {name for name, _, _ in task.PAIRS})
            for effect in factorial["paired_effects"].values():
                self.assertEqual(effect["units"], len(recipe["folds"]) * len(recipe["seeds"]))
            # The synthetic seed-2024 reference arm equals the stored v21 metrics by construction.
            self.assertEqual(len(factorial["reproduces_v21_seed_2024"]), len(recipe["folds"]))
            for row in factorial["reproduces_v21_seed_2024"]:
                self.assertEqual(row["mae_abs_difference"], 0.)
            self.assertEqual(len(gate["rows"]), len(recipe["folds"]))
            self.assertIn("dlinear", gate["macro_inner"])

    def test_outer_evaluation_path_runs_on_fixture_data(self):
        from data_provider.kelmarsh_hourly import KelmarshWindows
        _, _, effective = task.protocol()
        rng = np.random.default_rng(5)
        series = fixture()
        series.raw = rng.uniform(0, 2050., (200, 7))
        series.values = rng.normal(size=(200, 7)).astype(np.float32)
        series.subset = lambda name: KelmarshWindows(series, np.arange(120, 173, 3), name)
        fit = KelmarshWindows(series, np.arange(0, 100, 3), "fit")
        recipe = {"primary_candidate": "PretrainedMatched", "seeds": [2024, 2025], "v21_reference_dir": "previous",
                  "arms": {"PretrainedMatched": {"init": "pretrained", "epochs": 2}}}
        v21 = {"reference_dir": "reference"}
        validation = series.subset("validation")
        truth, base = validation.truth_and_anchor()
        arm_p = task.arm_protocol(effective, recipe["arms"]["PretrainedMatched"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "result"
            for name, sub, key in (("previous", "PowerAnchoredTimeDART", "prediction"),
                                   ("reference", "DLinear", "prediction")):
                folder = root / name / "runs/f0_s2024" / sub
                folder.mkdir(parents=True)
                np.savez(folder / "raw_predictions.npz", **{key: base + 10.}, truth=truth)
            for seed in recipe["seeds"]:
                output = directory / "runs" / f"f0_s{seed}" / "PretrainedMatched"
                output.mkdir(parents=True)
                torch.manual_seed(seed)
                model = Model(task.shared.model_args("PromptTimeDART", None, arm_p, torch.device("cpu")))
                model.regime_down_thresh.fill_(-.1)
                model.regime_up_thresh.fill_(.1)
                torch.save({"model": model.state_dict()}, output / "checkpoint.pt")
                task.shared.write(output / "trained.json", {"best_epoch": 1})
            stub_report = lambda name, prediction, data, folder, p, state=None: {"model": name}
            with patch.object(task, "ROOT", root), patch.object(task.recent, "assert_reference_cohort"), \
                    patch.object(task.prior, "wait_for_gpu"), patch.object(task.prior, "update_summary"), \
                    patch.object(task.prior, "write_report", side_effect=stub_report), \
                    patch.object(task.prior, "paired_forecast_statistics", return_value={"stub": True}):
                records = task.outer_evaluation(directory, recipe, v21, effective, {0: (series, {"fit": fit})},
                                                torch.device("cpu"))
            names = [r["name"] for r in records]
            self.assertEqual(names, ["PretrainedMatched_s2024", "PretrainedMatched_s2025", "PretrainedMatchedSeedMean",
                                     "v21_single_seed", "DLinear_v20", "Persistence"])
            persistence = records[-1]
            self.assertAlmostEqual(persistence["mae_skill_pct"], 0.)
            self.assertTrue((directory / "outer_records.json").exists())
            self.assertTrue((directory / "runs/f0_s2024/PretrainedMatched/completed.json").exists())
            ensemble = task.shared.read(directory / "runs/ensemble_f0/PretrainedMatchedSeedMean/result.json")
            self.assertEqual(set(ensemble) - {"model"}, {"paired_vs_v21", "paired_vs_dlinear"})
            self.assertIn("paired_vs_v21", task.shared.read(directory / "runs/f0_s2025/PretrainedMatched/result.json"))


if __name__ == "__main__":
    unittest.main()
