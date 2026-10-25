"""v26 contract tests: protocol freeze, epoch rule, gates (lead-1, attribution), access audit, report, and (with torch)
fit/resume/controller behaviour on synthetic data with synthetic anchors. Torch-free classes run anywhere."""
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils import kelmarsh_anchored_residual as v26
from utils import kelmarsh_backbone_information as v25

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/kelmarsh_anchored_residual_protocol_v26.json"
SCRIPT = ROOT / "scripts/train_kelmarsh_anchored_residual.py"
V25_PROTOCOL = ROOT / "configs/kelmarsh_backbone_information_protocol_v25.json"
try:
    import torch
    import sklearn  # noqa: F401
    HAS_TORCH = True
except ImportError:  # local design sandboxes may lack torch; the training host never does
    HAS_TORCH = False

FOLDS = (0, 1, 2)
PERSISTENCE = {f: {"mae": 200., "rmse": 300.} for f in FOLDS}
G1 = {"seed": 2024, "min_folds_jointly_better_than_DLinear": 2}
G2 = {"seeds": [2024, 2025, 2026], "min_seed_averaged_folds_jointly_better_than_DLinear": 2,
      "min_fold_averaged_seeds_jointly_better_than_DLinear": 2}


def protocol():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def rec(fold, seed, arm, mae, rmse, lead1_mae=None, lead1_rmse=None, best_epoch=5):
    return {"fold": fold, "seed": seed, "arm": arm, "mae": mae, "rmse": rmse,
            "lead1_mae": mae - 40 if lead1_mae is None else lead1_mae,
            "lead1_rmse": rmse - 60 if lead1_rmse is None else lead1_rmse,
            "best_epoch": None if arm == "DLinear" else best_epoch}


def dlinear(fold=None):
    return [rec(f, 2024, "DLinear", 160., 240., 113., 170.) for f in (FOLDS if fold is None else [fold])]


# =========================================================================== protocol
class ProtocolTests(unittest.TestCase):
    def test_frozen_hash_matches_runner_constant_and_design(self):
        constant = re.search(r'PROTOCOL_SHA256 = "([0-9a-f]{64})"', SCRIPT.read_text(encoding="utf-8")).group(1)
        self.assertEqual(v26.text_sha256_lf(PROTOCOL), constant)
        p = protocol()
        self.assertTrue(v26.validate_protocol(p))
        self.assertEqual(v26.planned_new_fits(p), {"A1": 6, "A2_max": 12, "max_new_fits": 18})
        self.assertEqual(p["reference"]["protocol_sha256"], v26.text_sha256_lf(ROOT / p["reference"]["protocol_path"]))
        self.assertEqual(p["v25_historical"]["protocol_sha256"], v26.text_sha256_lf(V25_PROTOCOL))
        self.assertFalse(p["wiki_enabled"] or p["sealed_test_access"] or p["prediction_clipping"])
        self.assertFalse(p["training"]["epoch0_selectable"])

    def test_v25_is_untouched_and_the_anchor_is_the_v25_reference(self):
        v25_constant = re.search(r'PROTOCOL_SHA256 = "([0-9a-f]{64})"',
                                 (ROOT / "scripts/train_kelmarsh_backbone_information.py").read_text(encoding="utf-8")).group(1)
        self.assertEqual(v25.text_sha256_lf(V25_PROTOCOL), v25_constant)
        old, new = json.loads(V25_PROTOCOL.read_text(encoding="utf-8")), protocol()
        self.assertEqual(old["reference"]["dlinear_checkpoints"], new["reference"]["dlinear_checkpoints"])
        for key in ("dir", "completed_sha256", "protocol_sha256", "seed"):
            self.assertEqual(old["reference"][key], new["reference"][key])
        for key in ("data_path", "data_sha256", "input_len", "pred_len", "train_stride", "folds", "feature_columns",
                    "start_utc", "end_utc_exclusive", "target_slot"):
            self.assertEqual(old[key], new[key], key)
        recipe = ("learning_rate", "weight_decay", "batch_size", "eval_batch_size", "max_epochs", "patience",
                  "grad_clip_norm", "scheduler_t_max", "loss_mae_weight", "loss_mse_weight")
        self.assertEqual({k: old["training"][k] for k in recipe}, {k: new["training"][k] for k in recipe})
        residual = new["candidate"]["residual"]
        self.assertEqual((residual["blocks"], residual["feature_hidden_dim"], residual["dropout"], residual["norm_eps"]),
                         (old["candidate"]["blocks"], old["candidate"]["feature_hidden_dim"], old["candidate"]["dropout"],
                          old["candidate"]["norm_eps"]))
        self.assertEqual(new["candidate"]["anchor"]["model_args"], old["arms"]["DLinear"]["model_args"])

    def test_any_departure_from_design_is_refused(self):
        for path, value in ((("candidate", "residual", "blocks"), 3), (("training", "learning_rate"), 3e-4),
                            (("training", "epoch0_selectable"), True), (("seeds", "A2"), [2025, 2026, 2027]),
                            (("budget", "new_fits_max"), 24), (("candidate", "gate"), True),
                            (("candidate", "expected_frozen_parameters"), 0), (("training", "max_epochs"), 40),
                            (("gates", "G1", "lead1_macro_mae_rmse_not_worse_than"), None),
                            (("statistics", "block_length"), 12), (("protocol_id",), "kelmarsh_backbone_information_v25")):
            p = protocol()
            target = p
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(ValueError, msg=str(path)):
                v26.validate_protocol(p)
        p = protocol()
        p["arms"]["AnchoredPowerOnly"]["input_mode"] = "full"
        with self.assertRaises(ValueError):
            v26.validate_protocol(p)
        p = protocol()
        p["arms"]["DLinear"]["model_args"]["moving_avg"] = 13
        with self.assertRaises(ValueError):
            v26.validate_protocol(p)


# =========================================================================== epoch rule and summaries
class EpochRuleTests(unittest.TestCase):
    def history(self, maes):
        return [{"epoch": i, "selection": {"mae": m}} for i, m in enumerate(maes)]

    def test_epoch0_is_never_selected_and_ties_keep_earlier(self):
        self.assertEqual(v26.select_epoch(self.history([1., 5., 4., 4., 6.])), (2, 4.))
        self.assertEqual(v26.select_epoch(self.history([1., 9.])), (1, 9.))

    def test_bad_histories_fail_closed(self):
        with self.assertRaises(ValueError):
            v26.select_epoch(self.history([1.]))
        with self.assertRaises(ValueError):
            v26.select_epoch(self.history([1., float("nan")]))

    def test_lead_summary_and_records(self):
        rng = np.random.default_rng(0)
        truth = rng.normal(500, 100, (50, 4))
        pred, pers = truth + rng.normal(0, 30, truth.shape), truth + rng.normal(0, 40, truth.shape)
        leads = v26.lead_summary(pred, truth, pers, 2050.)
        self.assertEqual([r["lead_hours"] for r in leads], [1, 2, 3, 4])
        self.assertAlmostEqual(leads[0]["mae"], float(np.abs(pred[:, 0] - truth[:, 0]).mean()), places=12)
        self.assertAlmostEqual(np.mean([r["mae"] for r in leads]), v26.point_metrics(pred, truth, pers, 2050.)["mae"], places=10)
        record = v26.record_from_metrics(0, 2024, "AnchoredFull", {"mae": 1., "rmse": 2.}, leads, best_epoch=3)
        self.assertEqual((record["lead1_mae"], record["best_epoch"]), (leads[0]["mae"], 3))
        with self.assertRaises(ValueError):
            v26.lead_summary(pred[:, :3], truth, pers, 2050.)

    def test_correction_summary(self):
        truth = np.array([[10., 10.], [10., 10.]])
        anchor = np.array([[12., 10.], [8., 10.]])
        pred = np.array([[11., 10.], [6., 10.]])
        out = v26.correction_summary(pred, anchor, truth)
        self.assertAlmostEqual(out["mean_abs_correction_kw"], 0.75)
        self.assertAlmostEqual(out["share_points_changed_pct"], 50.)
        self.assertAlmostEqual(out["share_changed_points_helped_pct"], 50.)
        self.assertAlmostEqual(out["mean_abs_error_gain_kw"], -0.25)  # (+1 helped, -2 harmed) / 4 points
        self.assertIsNone(v26.correction_summary(anchor, anchor, truth)["share_changed_points_helped_pct"])


# =========================================================================== gates
class GateTests(unittest.TestCase):
    def full(self, values, seed=2024, arm="AnchoredFull", **kw):
        return [rec(f, seed, arm, *values[f], **kw) for f in FOLDS]

    def test_g1_passes_only_with_macro_folds_and_lead1(self):
        good = self.full({0: (158., 238.), 1: (157., 237.), 2: (161., 241.)}, lead1_mae=112., lead1_rmse=169.)
        g = v26.gate_g1(good + dlinear(), PERSISTENCE, G1, FOLDS)
        self.assertTrue(g["passed"], g["checks"])
        self.assertEqual(g["checks"]["jointly_better_folds_vs_dlinear"], 2)
        worse_lead1 = self.full({0: (158., 238.), 1: (157., 237.), 2: (161., 241.)}, lead1_mae=114., lead1_rmse=169.)
        g = v26.gate_g1(worse_lead1 + dlinear(), PERSISTENCE, G1, FOLDS)
        self.assertFalse(g["passed"])
        self.assertFalse(g["checks"]["lead1_macro_mae_not_worse_than_dlinear"])
        tie_lead1 = self.full({0: (158., 238.), 1: (157., 237.), 2: (161., 241.)}, lead1_mae=113. + 5e-7, lead1_rmse=170.)
        self.assertTrue(v26.gate_g1(tie_lead1 + dlinear(), PERSISTENCE, G1, FOLDS)["passed"])

    def test_g1_requires_two_jointly_better_folds_and_rmse(self):
        one = self.full({0: (150., 230.), 1: (161., 241.), 2: (161., 241.)}, lead1_mae=100., lead1_rmse=150.)
        g = v26.gate_g1(one + dlinear(), PERSISTENCE, G1, FOLDS)
        self.assertFalse(g["passed"])
        self.assertEqual(g["checks"]["jointly_better_folds_vs_dlinear"], 1)
        mae_only = self.full({0: (150., 245.), 1: (150., 245.), 2: (150., 245.)}, lead1_mae=100., lead1_rmse=150.)
        self.assertFalse(v26.gate_g1(mae_only + dlinear(), PERSISTENCE, G1, FOLDS)["passed"])

    def test_g1_exact_anchor_tie_is_not_a_pass_even_for_a_trained_epoch(self):
        tied = self.full({f: (160., 240.) for f in FOLDS}, lead1_mae=113., lead1_rmse=170.)
        result = v26.gate_g1(tied + dlinear(), PERSISTENCE, G1, FOLDS)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["macro_mae_rmse_better_than_dlinear"])

    def test_g1_persistence_incomplete_and_epoch0_records(self):
        good = self.full({0: (158., 238.), 1: (157., 237.), 2: (156., 236.)}, lead1_mae=100., lead1_rmse=150.)
        bad_persistence = {f: {"mae": 150., "rmse": 300.} for f in FOLDS}
        self.assertFalse(v26.gate_g1(good + dlinear(), bad_persistence, G1, FOLDS)["passed"])
        g = v26.gate_g1(good[:2] + dlinear(), PERSISTENCE, G1, FOLDS)
        self.assertFalse(g["passed"] or g["complete"])
        self.assertIn([2, 2024, "AnchoredFull"], g["missing"])
        fallback = self.full({0: (158., 238.), 1: (157., 237.), 2: (156., 236.)}, lead1_mae=100., lead1_rmse=150., best_epoch=0)
        g = v26.gate_g1(fallback + dlinear(), PERSISTENCE, G1, FOLDS)
        self.assertFalse(g["passed"])
        self.assertEqual(g["checks"]["folds_selecting_epoch0"], [0, 1, 2])
        with self.assertRaises(ValueError):
            v26.gate_g1(good + dlinear() + dlinear(0), PERSISTENCE, G1, FOLDS)

    def test_i1_attribution(self):
        full = self.full({0: (158., 238.), 1: (157., 237.), 2: (161., 241.)})
        power = self.full({0: (159., 239.), 1: (158., 238.), 2: (160., 240.)}, arm="AnchoredPowerOnly")
        i1 = v26.info_i1(full + power, {"seed": 2024, "min_folds_jointly_better": 2}, FOLDS)
        self.assertTrue(i1["supported"])
        same = self.full({0: (158., 238.), 1: (157., 237.), 2: (161., 241.)}, arm="AnchoredPowerOnly")
        self.assertFalse(v26.info_i1(full + same, {"seed": 2024}, FOLDS)["supported"])
        self.assertFalse(v26.info_i1(full, {"seed": 2024}, FOLDS)["complete"])

    def seeds(self, table, arm="AnchoredFull", **kw):
        return [rec(f, s, arm, *table[(f, s)], **kw) for f in FOLDS for s in (2024, 2025, 2026)]

    def test_g2_single_anchor_reference_and_widespread_damage(self):
        good = self.seeds({(f, s): (158., 238.) for f in FOLDS for s in (2024, 2025, 2026)}, lead1_mae=110., lead1_rmse=165.)
        g = v26.gate_g2(good + dlinear(), PERSISTENCE, G2, FOLDS, 2024)
        self.assertTrue(g["passed"], g["checks"])
        self.assertEqual(g["dlinear"]["macro"]["mae"], 160.)
        damaged = {(f, s): (158., 238.) for f in FOLDS for s in (2024, 2025, 2026)}
        damaged.update({(f, 2025): (170., 250.) for f in FOLDS})
        damaged.update({(f, 2026): (162., 242.) for f in FOLDS})
        g = v26.gate_g2(self.seeds(damaged, lead1_mae=110., lead1_rmse=165.) + dlinear(), PERSISTENCE, G2, FOLDS, 2024)
        self.assertFalse(g["passed"])
        self.assertEqual(g["checks"]["fold_averaged_seeds_jointly_better"], 1)
        lead1 = v26.gate_g2(self.seeds({(f, s): (158., 238.) for f in FOLDS for s in (2024, 2025, 2026)},
                                       lead1_mae=115., lead1_rmse=165.) + dlinear(), PERSISTENCE, G2, FOLDS, 2024)
        self.assertFalse(lead1["passed"])
        missing = v26.gate_g2(good, PERSISTENCE, G2, FOLDS, 2024)
        self.assertFalse(missing["passed"] or missing["complete"])

    def test_i2(self):
        full = self.seeds({(f, s): (158., 238.) for f in FOLDS for s in (2024, 2025, 2026)})
        power = self.seeds({(f, s): (159., 239.) for f in FOLDS for s in (2024, 2025, 2026)}, arm="AnchoredPowerOnly")
        self.assertTrue(v26.info_i2(full + power, {"seeds": [2024, 2025, 2026]}, FOLDS)["supported"])
        self.assertFalse(v26.info_i2(full + full[:0], {"seeds": [2024, 2025, 2026]}, FOLDS)["complete"])


class AccessTests(unittest.TestCase):
    def test_validation_requires_g2_and_sealed_test_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = v26.DataAccessLog(Path(tmp) / "events.jsonl")
            log.request("A1", "fit", 0)
            log.request("A1", "selection", 0)
            for stage, subset, kwargs in (("A1", "validation", {}), ("A3", "validation", {}), ("A2", "test", {}),
                                          ("A3", "sealed_test", {"g2_passed": True}), ("A9", "fit", {})):
                with self.assertRaises(PermissionError, msg=(stage, subset)):
                    log.request(stage, subset, 0, **kwargs)
            log.request("A3", "validation", 0, g2_passed=True)
            lines = [json.loads(l) for l in (Path(tmp) / "events.jsonl").read_text().splitlines()]
            self.assertEqual([l["subset"] for l in lines], ["fit", "selection", "validation"])


class FitSideDiagnosticTests(unittest.TestCase):
    def test_regimes_are_frozen_from_history_and_fit_only(self):
        from utils import kelmarsh_fit_side_diagnostics as diag
        power = np.arange(100, dtype=float) * 20.
        last, ramp = diag.last_power_and_ramp(power, np.array([0, 10]), input_len=24)
        np.testing.assert_array_equal(last, [460., 660.])
        np.testing.assert_array_equal(ramp, [60., 60.])
        edges = diag.ramp_edges_from_fit(np.array([0., 10., 20., 30., 40., 50.]))
        labels = diag.regime_labels(np.array([100., 700., 2000.]), np.array([0., 25., 100.]), 2050., edges)
        self.assertEqual(labels["power"].tolist(), ["<10%", "30-60%", ">=90%"])
        self.assertEqual(labels["ramp"].tolist(), ["ramp_low", "ramp_mid", "ramp_high"])
        with self.assertRaises(ValueError):
            diag.ramp_edges_from_fit(np.array([1., np.nan, 2.]))

    def test_group_summaries_keep_support_and_empty_groups(self):
        from utils import kelmarsh_fit_side_diagnostics as diag
        truth = np.zeros((4, 2))
        pred = np.array([[1., -1.], [2., 2.], [0., 0.], [-3., 3.]])
        out = diag.by_group(pred, truth, np.array(["a", "a", "b", "b"]), ("a", "b", "c"))
        self.assertEqual(out["a"]["windows"], 2)
        self.assertAlmostEqual(out["a"]["mae"], 1.5)
        self.assertAlmostEqual(out["b"]["per_lead"][0]["mbe"], -1.5)
        self.assertEqual(out["c"], {"windows": 0, "note": "no support"})


# =========================================================================== report (torch-free)
def cohort(fold, issues=40, seed=100):
    rng = np.random.default_rng(seed + fold)
    times = np.repeat(pd.date_range("2016-10-01", periods=issues, freq="3h").to_numpy(), 6)[:, None] + \
        (np.arange(1, 5) * np.timedelta64(1, "h"))[None]
    truth = 500 + 300 * np.sin(np.arange(issues * 6) / 9.)[:, None] + rng.normal(0, 40, (issues * 6, 4))
    return {"times": times.astype("datetime64[ns]"), "turbines": np.tile(np.arange(1, 7, dtype=np.int16), issues),
            "window_starts": np.arange(issues * 6, dtype=np.int64), "truth": truth,
            "persistence": truth + rng.normal(0, 80, truth.shape)}


def build_fake_run(root, outer, with_v25=False):
    """Directory with the exact registered-file layout the v26 controller writes (no torch needed)."""
    p = protocol()
    p["statistics"]["replicates"] = 200
    run, ref = root / "run", root / "ref"
    rng = np.random.default_rng(0)
    capacity = 2050.
    manifest, a0 = {}, {"dlinear_anchor_inner": {}}
    seeds = [2024, 2025, 2026] if outer else [2024]
    scale = {"AnchoredFull": 30., "AnchoredPowerOnly": 35.}
    index, v25_hashes = {}, {}
    for fold in FOLDS:
        inner, outer_cohort = cohort(fold), cohort(fold + 10)
        anchor = inner["truth"] + rng.normal(0, 40, inner["truth"].shape)
        replay = run / "anchor_replay" / f"f{fold}_s2024_DLinear_selection_predictions.npz"
        replay.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(replay, prediction=anchor, **inner)
        metrics = v26.point_metrics(anchor, inner["truth"], inner["persistence"], capacity)
        leads = v26.lead_summary(anchor, inner["truth"], inner["persistence"], capacity)
        manifest[str(fold)] = {"selection_predictions": f"anchor_replay/{replay.name}",
                               "selection_predictions_sha256": v26.file_sha256(replay),
                               "replay_selection": {"mae": metrics["mae"], "rmse": metrics["rmse"]},
                               "replay_fit": {"overall": {"mae": metrics["mae"] - 3, "rmse": metrics["rmse"] - 4, "mbe": -1.}}}
        a0["dlinear_anchor_inner"][str(fold)] = {"mae": metrics["mae"], "rmse": metrics["rmse"],
                                                 "lead1_mae": leads[0]["mae"], "lead1_rmse": leads[0]["rmse"], "source": "test"}
        if with_v25:
            for arm in ("TSMixerFull", "TSMixerPowerOnly"):
                name = f"runs/f{fold}_s2024/{arm}/selection_predictions.npz"
                target = ref / p["v25_historical"]["dir"] / name
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target, prediction=inner["truth"] + rng.normal(0, 45, inner["truth"].shape), **inner)
                v25_hashes[name] = v26.file_sha256(target)
        if outer:
            refrun = ref / p["reference"]["dir"] / "runs" / f"f{fold}_s2024" / "DLinear"
            refrun.mkdir(parents=True, exist_ok=True)
            prediction = outer_cohort["truth"] + rng.normal(0, 40, outer_cohort["truth"].shape)
            np.savez_compressed(refrun / "raw_predictions.npz", prediction=prediction, **outer_cohort)
            v26.write_json(refrun / "result.json", {"metrics": v26.point_metrics(prediction, outer_cohort["truth"],
                                                                                 outer_cohort["persistence"], capacity)})
            index[f"f{fold}/s2024/DLinear"] = {"fold": fold, "seed": 2024, "arm": "DLinear", "path": str(refrun / "raw_predictions.npz"),
                                               "sha256": v26.file_sha256(refrun / "raw_predictions.npz"), "source": "v20"}
            (run / "runs").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(run / "runs" / f"f{fold}_outer_cohort.npz", **outer_cohort)
            index[f"f{fold}/Persistence"] = {"fold": fold, "seed": None, "arm": "Persistence",
                                             "path": str(run / "runs" / f"f{fold}_outer_cohort.npz"),
                                             "sha256": v26.file_sha256(run / "runs" / f"f{fold}_outer_cohort.npz")}
        for seed in seeds:
            for arm in ("AnchoredFull", "AnchoredPowerOnly"):
                folder = run / "runs" / f"f{fold}_s{seed}" / arm
                folder.mkdir(parents=True)
                prediction = anchor + rng.normal(0, scale[arm] / 3, anchor.shape)
                np.savez_compressed(folder / "selection_predictions.npz", prediction=prediction, anchor=anchor, **inner)
                best = v26.point_metrics(prediction, inner["truth"], inner["persistence"], capacity)
                history = [{"epoch": 0, "selection": metrics}, {"epoch": 1, "train_loss": .3, "selection": best},
                           {"epoch": 2, "train_loss": .2, "selection": {**best, "mae": best["mae"] + 2}}]
                v26.write_json(folder / "training_history.json", history)
                v26.write_json(folder / "trained.json", {
                    "arm": arm, "seed": seed, "best_epoch": 1, "epochs_completed": 2, "stopped_early": True,
                    "selection_best": best, "selection_best_by_lead": v26.lead_summary(prediction, inner["truth"], inner["persistence"], capacity),
                    "selection_epoch0": metrics, "selection_last": history[-1]["selection"],
                    "epoch0_max_abs_diff_vs_anchor_kw": 0.0,
                    "fit_eval_best": {"overall": {"mae": best["mae"] - 5, "rmse": best["rmse"] - 5, "mbe": 0.5}},
                    "trainable_parameters": 3618, "frozen_parameters": 200, "device": "cpu",
                    "timing": {"train_seconds": 1., "assess_seconds": .1}, "peak_memory": {"kind": "rss", "value": 1},
                    "artifacts": {"selection_predictions.npz": v26.file_sha256(folder / "selection_predictions.npz")}})
                if outer:
                    with np.load(ref / p["reference"]["dir"] / "runs" / f"f{fold}_s2024" / "DLinear" / "raw_predictions.npz") as s:
                        outer_anchor = s["prediction"]
                    prediction = outer_anchor + rng.normal(0, scale[arm] / 3, outer_anchor.shape)
                    np.savez_compressed(folder / "outer_predictions.npz", prediction=prediction, **outer_cohort)
                    v26.write_json(folder / "result.json", {"inference_seconds": .01, "metrics": v26.point_metrics(
                        prediction, outer_cohort["truth"], outer_cohort["persistence"], capacity)})
                    index[f"f{fold}/s{seed}/{arm}"] = {"fold": fold, "seed": seed, "arm": arm,
                                                       "path": str(folder / "outer_predictions.npz"),
                                                       "sha256": v26.file_sha256(folder / "outer_predictions.npz"), "source": "v26"}
    if with_v25:
        v26.write_json(ref / p["v25_historical"]["dir"] / "completed.json", v25_hashes)
        p["v25_historical"]["completed_sha256"] = v26.file_sha256(ref / p["v25_historical"]["dir"] / "completed.json")
    v26.write_json(run / "protocol.json", {"protocol": p, "protocol_id": p["protocol_id"]})
    v26.write_json(run / "reference_manifest.json", {"dlinear_anchor": manifest})
    v26.write_json(run / "a0_result.json", a0)
    g1 = {"G1": {"passed": outer, "checks": {"macro_mae_rmse_better_than_dlinear": outer}}, "I1": {"supported": False}}
    v26.write_json(run / "inner_gate_seed2024.json", g1)
    if outer:
        v26.write_json(run / "inner_gate_all_seeds.json", {"G2": {"passed": True, "checks": {}}, "I2": {"supported": False}})
        v26.write_json(run / "outer_index.json", {"entries": index})
    v26.write_json(run / "training_completed.json", {"outer_evaluated": outer, "hashes": {}})
    return run, ref


class ReportTests(unittest.TestCase):
    def test_inner_only_report_after_failed_gate(self):
        from scripts import report_kelmarsh_anchored_residual as report
        with tempfile.TemporaryDirectory() as tmp:
            run, ref = build_fake_run(Path(tmp), outer=False, with_v25=True)
            result = report.build_report(run, reference_root=ref)
            self.assertFalse(result["outer_evaluated"])
            self.assertIn("G1 FAILED", result["conclusion"])
            self.assertIn("must not be attributed", result["conclusion"])
            self.assertEqual(set(result["macro"]), {"inner_selection"})
            self.assertTrue(result["historical_v25"]["available"], result["historical_v25"])
            self.assertIn("v25_TSMixerFull", result["macro"]["inner_selection"])
            self.assertEqual(result["independent_recomputation_max_abs_diff"], 0.)
            for name in ("accuracy_overview", "lead_metrics", "training_history", "forecast_trace", "paired_gain"):
                self.assertTrue((run / "figures" / f"{name}.png").stat().st_size > 1000, name)
            epochs = pd.read_csv(run / "epoch_disclosure.csv")
            self.assertEqual(len(epochs), 6)
            self.assertTrue((epochs.last_mae > epochs.best_mae).all())
            corrections = pd.read_csv(run / "correction_disclosure.csv")
            self.assertTrue((corrections.mean_abs_correction_kw > 0).all())
            fit = pd.read_csv(run / "fit_side_disclosure.csv")
            self.assertEqual(len(fit), 9)
            stats = v26.read_json(run / "paired_statistics.json")["inner_selection"]
            self.assertEqual(set(stats["primary_family"]["raw_p"]), {"0", "1", "2"})
            self.assertIn("full_vs_power_only", stats["folds"]["0"])
            markdown = (run / "report.md").read_text(encoding="utf-8")
            for text in ("G1 passed: False", "not a percentage accuracy", "epoch 0 = the anchor", "Fit-side"):
                self.assertIn(text, markdown)
            self.assertTrue((run / "report_manifest.json").exists())

    def test_outer_report_uses_single_anchor_and_all_residual_seeds(self):
        from scripts import report_kelmarsh_anchored_residual as report
        with tempfile.TemporaryDirectory() as tmp:
            run, ref = build_fake_run(Path(tmp), outer=True)
            result = report.build_report(run, reference_root=ref)
            stats = v26.read_json(run / "paired_statistics.json")["outer_validation"]
            block = stats["folds"]["0"]
            self.assertEqual(block["primary_full_vs_dlinear"]["candidate_seeds"], 3)
            self.assertEqual(block["primary_full_vs_dlinear"]["reference_seeds"], 1)
            self.assertEqual(set(block["full_vs_dlinear_per_seed_library"]), {"2024", "2025", "2026"})
            self.assertIn("NOT a reuse", block["primary_full_vs_dlinear"]["rmse_difference"]["method"])
            macro = result["macro"]["outer_validation"]
            self.assertEqual(macro["AnchoredFull"]["units"], 9)
            self.assertEqual(macro["DLinear"]["units"], 3)
            self.assertIn("G1 and G2 passed", result["conclusion"])
            self.assertFalse(result["historical_v25"]["available"])

    def test_tampering_and_anchor_mismatch_are_refused(self):
        from scripts import report_kelmarsh_anchored_residual as report
        with tempfile.TemporaryDirectory() as tmp:
            run, ref = build_fake_run(Path(tmp), outer=False)
            target = run / "runs/f1_s2024/AnchoredFull/selection_predictions.npz"
            with np.load(target) as saved:
                arrays = {k: saved[k] for k in saved.files}
            arrays["anchor"] = arrays["anchor"] + 1.
            np.savez_compressed(target, **arrays)
            trained = v26.read_json(target.parent / "trained.json")
            trained["artifacts"]["selection_predictions.npz"] = v26.file_sha256(target)
            v26.write_json(target.parent / "trained.json", trained)
            with self.assertRaisesRegex(ValueError, "stored anchor differs"):
                report.build_report(run, reference_root=ref)
        with tempfile.TemporaryDirectory() as tmp:
            run, ref = build_fake_run(Path(tmp), outer=False)
            (run / "anchor_replay/f0_s2024_DLinear_selection_predictions.npz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "prediction file changed"):
                report.build_report(run, reference_root=ref)
        with tempfile.TemporaryDirectory() as tmp:
            run, ref = build_fake_run(Path(tmp), outer=True)
            v26.write_json(run / "training_completed.json", {"outer_evaluated": False, "hashes": {}})
            with self.assertRaises(ValueError):
                report.build_report(run, reference_root=ref)
        with tempfile.TemporaryDirectory() as tmp:
            run, ref = build_fake_run(Path(tmp), outer=False)
            name = "inner_gate_seed2024.json"
            v26.write_json(run / "training_completed.json", {"outer_evaluated": False, "hashes": {name: v26.file_sha256(run / name)}})
            record = v26.read_json(run / name)
            record["G1"]["passed"] = True
            v26.write_json(run / name, record)
            with self.assertRaisesRegex(ValueError, "Registered artifact"):
                report.build_report(run, reference_root=ref)


# =========================================================================== torch: fits and controller
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
class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import train_kelmarsh_anchored_residual as task
        cls.task = task
        torch.set_num_threads(2)
        cls.p = protocol()
        cls.p["training"].update(max_epochs=4, scheduler_t_max=4)
        cls.fit, cls.selection = task.SyntheticWindows(160, 1), task.SyntheticWindows(48, 2)
        cls.anchor = task.synthetic_anchor_state(cls.p)
        cls.reference = task.anchor_reference_for(cls.p, cls.anchor, cls.selection)

    def run_fit(self, out, arm="AnchoredFull", contract=None, **kwargs):
        kwargs.setdefault("anchor_state", self.anchor)
        kwargs.setdefault("anchor_reference_kw", self.reference)
        return self.task.fit_arm(arm, self.p, fit=self.fit, selection=self.selection, contract=contract or {"c": 1},
                                 seed=11, device=torch.device("cpu"), out=out, **kwargs)

    def test_reference_replay_uses_the_frozen_executor(self):
        model, factory = self.task.build_model("DLinear", self.p, self.anchor)
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertEqual((factory["trainable_parameters"], factory["frozen_parameters"]), (0, 200))

    def test_preflight_respects_shared_gpu_cap_and_cpu_fallback(self):
        snapshot = {"total_mib": 24576., "free_mib": 8192., "other_compute_pids": [123]}
        with patch.object(self.task, "gpu_snapshot", return_value=snapshot), \
                patch.object(torch.cuda, "set_per_process_memory_fraction") as cap:
            device, decision = self.task.preflight_device(self.p, 0)
            self.assertEqual(str(device), "cuda:0")
            cap.assert_called_once_with(4. / 24., device)
            self.assertTrue(decision["shared_with_other_processes"])
        with patch.object(self.task, "gpu_snapshot", return_value={**snapshot, "free_mib": 1024.}), \
                patch.object(torch.cuda, "set_per_process_memory_fraction") as cap:
            device, _ = self.task.preflight_device(self.p, 0)
            self.assertEqual(str(device), "cpu")
            cap.assert_not_called()

    def test_namespace_uses_full_id_and_rejects_duplicate_or_unresolved_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, logs = root / "configs", root / "logs"
            config.mkdir()
            logs.mkdir()
            own = config / "kelmarsh_anchored_residual_protocol_v26.json"
            pointer = logs / "anchored_residual_latest.txt"
            v26.write_json(own, self.p)
            other = config / "kelmarsh_capacity_protocol_v26.json"
            old_id = "kelmarsh_24h4h_forecast_only_capacity_v26"
            v26.write_json(other, {"protocol_id": old_id})
            old = root / "capacity_v26_run"
            old.mkdir()
            v26.write_json(old / "protocol.json", {"recipe": {"protocol_id": old_id}})
            other_pointer = logs / "capacity_v26_latest.txt"
            other_pointer.write_text(str(old), encoding="utf-8")
            check = lambda: self.task.protocol_namespace_check(self.p, config, logs, own, pointer)
            self.assertTrue(check()["passed"])
            # Even a different shorthand version must not reuse this full identity.
            duplicate = config / "another_protocol_v99.json"
            v26.write_json(duplicate, {"protocol_id": self.p["protocol_id"]})
            self.assertFalse(check()["passed"])
            duplicate.unlink()
            v26.write_json(old / "protocol.json", {"protocol": {"protocol_id": self.p["protocol_id"]}})
            self.assertFalse(check()["passed"])
            (old / "protocol.json").unlink()
            self.assertFalse(check()["passed"])

    def assert_same_final_state(self, a, b):
        x = torch.load(a / "last.pt", map_location="cpu", weights_only=False)
        y = torch.load(b / "last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(self.task.state_digest(x["model"]), self.task.state_digest(y["model"]))
        for gx, gy in zip(x["optimizer"]["state"].values(), y["optimizer"]["state"].values()):
            for key in gx:
                self.assertTrue(torch.equal(torch.as_tensor(gx[key]), torch.as_tensor(gy[key])), key)
        self.assertEqual(x["scheduler"], y["scheduler"])
        strip = lambda h: [{k: v for k, v in e.items() if "seconds" not in k} for e in h]
        self.assertEqual(strip(x["history"]), strip(y["history"]))
        self.assertEqual((x["epoch"], x["best_epoch"], x["best"]), (y["epoch"], y["best_epoch"], y["best"]))

    def test_fit_starts_at_anchor_never_selects_epoch0_and_keeps_anchor_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            trained = self.run_fit(Path(tmp) / "fit")
            self.assertGreaterEqual(trained["best_epoch"], 1)
            self.assertEqual(trained["epoch0_max_abs_diff_vs_anchor_kw"], 0.0)
            self.assertEqual(trained["anchor_state_sha256"], self.task.state_digest(self.anchor))
            self.assertEqual((trained["trainable_parameters"], trained["frozen_parameters"]), (3618, 200))
            self.assertEqual([r["lead_hours"] for r in trained["selection_best_by_lead"]], [1, 2, 3, 4])
            self.assertIn("overall", trained["fit_eval_best"])
            history = v26.read_json(Path(tmp) / "fit/training_history.json")
            self.assertFalse(history[0]["selectable"])
            self.assertEqual(v26.select_epoch(history)[0], trained["best_epoch"])
            with np.load(Path(tmp) / "fit/selection_predictions.npz") as saved:
                np.testing.assert_array_equal(saved["anchor"], self.reference)
            for name in ("best.pt", "last.pt"):
                state = torch.load(Path(tmp) / "fit" / name, map_location="cpu", weights_only=False)["model"]
                self.assertEqual(self.task.anchor_digest(state), trained["anchor_state_sha256"])

    def test_wrong_anchor_reference_is_refused_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Epoch-0 forecast differs"):
                self.run_fit(Path(tmp) / "fit", anchor_reference_kw=self.reference + 1.0)
            self.assertFalse((Path(tmp) / "fit/best.pt").exists())
            self.assertFalse((Path(tmp) / "fit/init.pt").exists())

    def test_interruptions_resume_identically_and_registry_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            reference = self.run_fit(tmp / "straight")
            with self.assertRaises(self.task.SimulatedInterrupt):
                self.run_fit(tmp / "boundary", stop_after_epoch=2)
            self.run_fit(tmp / "boundary")
            self.assert_same_final_state(tmp / "straight", tmp / "boundary")
            with self.assertRaises(self.task.SimulatedInterrupt):
                self.run_fit(tmp / "inside", stop_inside_epoch=(1, 1))
            self.assertFalse((tmp / "inside/best.pt").exists())
            self.run_fit(tmp / "inside")
            self.assert_same_final_state(tmp / "straight", tmp / "inside")
            with patch.object(self.task, "make_loader", side_effect=AssertionError("retrained")):
                again = self.run_fit(tmp / "straight")
            self.assertTrue(again["reused_from_registry"])
            self.assertEqual(again["best_state_sha256"], reference["best_state_sha256"])
            with self.assertRaises(ValueError):
                self.run_fit(tmp / "straight", contract={"c": 2})
            with (tmp / "straight/best.pt").open("ab") as handle:
                handle.write(b"x")
            with self.assertRaises(ValueError):
                self.run_fit(tmp / "straight")

    def test_pairs_share_init_and_changed_anchor_is_refused_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            full = self.run_fit(tmp / "full")
            power = self.run_fit(tmp / "power", arm="AnchoredPowerOnly", init_reference=tmp / "full/init.pt")
            self.assertEqual(full["init_state_sha256"], power["init_state_sha256"])
            with self.assertRaises(ValueError):
                self.task.fit_arm("AnchoredPowerOnly", self.p, fit=self.fit, selection=self.selection, contract={"c": 1},
                                  seed=12, device=torch.device("cpu"), out=tmp / "other", anchor_state=self.anchor,
                                  anchor_reference_kw=self.reference, init_reference=tmp / "full/init.pt")
            with self.assertRaises(self.task.SimulatedInterrupt):
                self.run_fit(tmp / "a", stop_after_epoch=1)
            other = {k: v + 0.01 for k, v in self.anchor.items()}
            other_reference = self.task.anchor_reference_for(self.p, other, self.selection)
            with self.assertRaisesRegex(ValueError, "Saved initial state differs"):  # init.pt digest includes the anchor
                self.run_fit(tmp / "a", anchor_state=other, anchor_reference_kw=other_reference)
            (tmp / "a/init.pt").unlink()  # reach the last.pt anchor check as well
            with self.assertRaisesRegex(ValueError, "changed anchor"):
                self.run_fit(tmp / "a", anchor_state=other, anchor_reference_kw=other_reference)

    def test_refused_start_leaves_no_init_and_stale_best_is_rewritten_from_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with self.assertRaises(ValueError):
                self.run_fit(tmp / "r", anchor_reference_kw=self.reference + 1.0)
            self.assertFalse((tmp / "r/init.pt").exists())
            self.run_fit(tmp / "r")  # a corrected retry is not blocked
            straight = self.run_fit(tmp / "s")
            with self.assertRaises(self.task.SimulatedInterrupt):
                self.run_fit(tmp / "c", stop_after_epoch=2)
            # emulate a crash after best.pt was written for an epoch that last.pt never registered
            stale = torch.load(tmp / "c/last.pt", map_location="cpu", weights_only=False)
            self.task.atomic_save({"model": stale["model"], "epoch": 99, "contract": {"c": 1}, "selection_mae_kw": 0.}, tmp / "c/best.pt")
            resumed = self.run_fit(tmp / "c")
            self.assertEqual(resumed["best_state_sha256"], straight["best_state_sha256"])


@unittest.skipUnless(HAS_TORCH, "torch/sklearn unavailable on this host")
class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import train_kelmarsh_anchored_residual as task
        cls.task = task
        torch.set_num_threads(2)
        cls.frame = synthetic_frame()
        cls.series = {fold: synthetic_series(fold, cls.frame) for fold in FOLDS}
        cls.gpu_patch = patch.object(task, "gpu_snapshot", return_value=None)
        cls.gpu_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.gpu_patch.stop()

    def make(self, tmp, dlinear_inner, **kwargs):
        p = protocol()
        p["training"].update(max_epochs=2, scheduler_t_max=2)
        p["statistics"]["replicates"] = 200
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
        anchors = {f: {"state": self.task.synthetic_anchor_state(p, seed=20 + f), "mae": dlinear_inner, "rmse": dlinear_inner}
                   for f in FOLDS}
        return self.task.V26Run(Path(tmp) / "run", protocol=p, protocol_sha="test-protocol", pin=pin, series_factory=factory,
                                require_preflight=False, test_anchors=anchors, write_pointer=False,
                                lock_path=Path(tmp) / "lock", reference_root=Path(tmp) / "ref", **kwargs)

    def test_failed_g1_stops_without_outer_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp, dlinear_inner=0.)  # anchor "perfect" -> G1 must fail
            with patch("scripts.train_kelmarsh_anchored_residual.load_pin"):
                self.assertEqual(run.run(), "COMPLETED")
            self.assertEqual(self.requested, [])
            self.assertEqual(run.fit_calls, [(f, 2024, a) for f in FOLDS for a in ("AnchoredFull", "AnchoredPowerOnly")])
            a0 = v26.read_json(run.dir / "a0_result.json")
            for fold in ("0", "1", "2"):
                for arm in ("AnchoredFull", "AnchoredPowerOnly"):
                    self.assertEqual(a0["epoch0_identity"][fold]["arms"][arm]["max_abs_diff_kw"], 0.0)
            self.assertFalse(v26.read_json(run.dir / "inner_gate_seed2024.json")["G1"]["passed"])
            self.assertFalse((run.dir / "outer_index.json").exists())
            self.assertFalse(list(run.dir.rglob("outer_predictions.npz")))
            self.assertEqual(v26.read_json(run.dir / "status.json")["report"], "REPORT_COMPLETED")
            for fold in FOLDS:
                full = v26.read_json(run.dir / f"runs/f{fold}_s2024/AnchoredFull/trained.json")
                power = v26.read_json(run.dir / f"runs/f{fold}_s2024/AnchoredPowerOnly/trained.json")
                self.assertEqual(full["init_state_sha256"], power["init_state_sha256"])
                self.assertEqual(full["anchor_state_sha256"], power["anchor_state_sha256"])
                self.assertGreaterEqual(full["best_epoch"], 1)
            again = self.make(tmp, dlinear_inner=0.)
            with patch.object(self.task, "fit_arm", side_effect=AssertionError("retrained")):
                self.assertEqual(again.run(), "COMPLETED")

    def test_report_failure_never_reruns_training_and_preflight_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp, dlinear_inner=0.)
            with patch("scripts.train_kelmarsh_anchored_residual.load_pin"), \
                    patch("scripts.report_kelmarsh_anchored_residual.build_report", side_effect=RuntimeError("figure bug")):
                with self.assertRaises(RuntimeError):
                    run.run()
            status = v26.read_json(run.dir / "status.json")
            self.assertEqual((status["train"], status["report"]), ("TRAIN_COMPLETED", "REPORT_FAILED"))
            again = self.make(tmp, dlinear_inner=0.)
            with patch.object(self.task, "fit_arm", side_effect=AssertionError("retrained")):
                self.assertEqual(again.run(report_only=True), "COMPLETED")
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make(tmp, dlinear_inner=0.)
            run.require_preflight = True
            with patch("scripts.train_kelmarsh_anchored_residual.load_pin"), patch.object(self.task, "PREFLIGHT_DIR", Path(tmp) / "pf"):
                with self.assertRaises(ValueError):
                    run.run()
            self.assertFalse(list(run.dir.rglob("trained.json")))
            holder = v26.acquire_exclusive_lock(Path(tmp) / "lock")
            with self.assertRaises(BlockingIOError):
                self.make(tmp, dlinear_inner=0.).run()
            holder.close()

    def test_passed_gates_run_a2_then_one_outer_evaluation_with_anchor_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run = self.make(tmp, dlinear_inner=1e9)
            p = run.p
            for fold in FOLDS:
                validation = self.series[fold].subset("validation")
                truth, persistence = validation.truth_and_anchor()
                arrays = {"times": validation.target_times(), "turbines": validation.turbines[validation.window_starts],
                          "window_starts": validation.window_starts, "truth": truth, "persistence": persistence}
                model, _ = self.task.build_model("DLinear", p, anchor_state=run.test_anchors[fold]["state"])
                anchor_outer = self.task.predict(model, validation, 0, torch.device("cpu"), 128)
                base = tmp / "ref" / p["reference"]["dir"] / "runs" / f"f{fold}_s2024"
                (base / "Persistence").mkdir(parents=True)
                (base / "DLinear").mkdir(parents=True)
                np.savez_compressed(base / "Persistence/raw_predictions.npz", prediction=persistence, **arrays)
                np.savez_compressed(base / "DLinear/raw_predictions.npz", prediction=anchor_outer, **arrays)
                v26.write_json(base / "DLinear/result.json", {"metrics": v26.point_metrics(anchor_outer, truth, persistence, 2050.)})
            force = lambda real: (lambda *a, **k: {**real(*a, **k), "passed": True})
            with patch("scripts.train_kelmarsh_anchored_residual.load_pin"), \
                    patch.object(self.task.v26, "gate_g1", side_effect=force(v26.gate_g1)), \
                    patch.object(self.task.v26, "gate_g2", side_effect=force(v26.gate_g2)):
                self.assertEqual(run.run(), "COMPLETED")
            self.assertEqual(len(run.registered_new_fits()), 18)
            self.assertEqual(run.fit_calls[6:8], [(0, 2025, "AnchoredFull"), (0, 2025, "AnchoredPowerOnly")])
            self.assertNotIn("DLinear", {arm for _, _, arm in run.fit_calls})
            self.assertEqual(self.requested, ["validation"] * 3)
            outer = v26.read_json(run.dir / "outer_index.json")
            self.assertEqual(len(outer["entries"]), 3 * (1 + 1 + 3 * 2))
            self.assertTrue(all(c["max_abs_diff_kw"] <= 0.01 for c in outer["anchor_executor_checks"].values()))
            result = v26.read_json(run.dir / "result.json")
            self.assertTrue(result["outer_evaluated"])
            self.assertLess(result["independent_recomputation_max_abs_diff"], 1e-6)
            events = [json.loads(l) for l in (run.dir / "events.jsonl").read_text().splitlines()]
            g2_index = next(i for i, e in enumerate(events) if e["event"] == "G2")
            first_outer = next(i for i, e in enumerate(events) if e.get("subset") == "validation")
            self.assertLess(g2_index, first_outer)

    def test_outer_anchor_executor_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run = self.make(tmp, dlinear_inner=1e9)
            validation = self.series[0].subset("validation")
            truth, persistence = validation.truth_and_anchor()
            cohort_arrays = {"times": validation.target_times(), "turbines": validation.turbines[validation.window_starts],
                             "window_starts": validation.window_starts, "truth": truth, "persistence": persistence}
            path = tmp / "wrong.npz"
            np.savez_compressed(path, prediction=persistence, **cohort_arrays)
            with self.assertRaisesRegex(ValueError, "Anchor executor on outer differs"):
                run.outer_anchor_check(0, validation, cohort_arrays, path)


if __name__ == "__main__":
    unittest.main()
