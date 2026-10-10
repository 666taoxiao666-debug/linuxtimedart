import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.confirm_sdwpf_residual_scale import followup_dataset, summarize, frozen_sources
from scripts.tune_sdwpf_accuracy import write_json, sha256
from utils.residual_scale_calibration import fit_scale


class ResidualScaleFollowupTests(unittest.TestCase):
    def test_checkpoint_hash_and_outer_boundary_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit, scale, inner, outer = [root / name for name in ("audit", "scale", "inner", "outer")]
            for path in (audit, scale, inner, outer):
                path.mkdir()
            write_json(inner / "checkpoint", {"weight": 1})
            source_hash = sha256(inner / "checkpoint")
            args = dict(prompt_router="trend", utility_wiki=False, sdwpf_fold=1, seed=2024,
                        seq_len=336, pred_len=12, sdwpf_split="time_ratio")
            manifest = dict(args=args, extra=dict(best_epoch=8), data_file=dict(sha256="data"))
            write_json(inner / "run_manifest.json", manifest)
            outer_manifest = dict(args=dict(args, sdwpf_split="rolling_holdout"),
                data_file=dict(sha256="data"), datasets=dict(val=dict(train_cutoff="2023-06-30",
                    val_cutoff="2023-07-08", test_start_cutoff="2023-07-16", windows=4,
                    target_date_min="2023-06-30", target_date_max="2023-07-07")))
            write_json(outer / "run_manifest.json", outer_manifest)
            write_json(audit / "protocol.json", dict(source_plan=dict(outer_trend_checkpoint=str(outer / "checkpoint"),
                outer_data_sha256="data"), oof_end_exclusive="2023-06-30", event_thresholds={}))
            write_json(audit / "reference_predictions.npz", {"cache": "unread in source checks"})
            model = fit_scale(np.ones((1, 12)), np.zeros((1, 12)), np.zeros((1, 12)), np.array([0]))
            model["source_checkpoint_sha256"] = source_hash
            write_json(scale / "calibration.json", model)
            write_json(scale / "protocol.json", dict(protocol_id="sdwpf_train_tail_residual_scale_pilot_v7",
                source_audit=str(audit), source_audit_protocol_sha256=sha256(audit / "protocol.json"),
                source_predictions_sha256=sha256(audit / "reference_predictions.npz"),
                source_checkpoint=dict(checkpoint=str(inner / "checkpoint"), checkpoint_sha256=source_hash,
                                       manifest_sha256=sha256(inner / "run_manifest.json"))))
            write_json(scale / "result.json", dict(check_joint_improvement=True, outer_validation_used=False,
                sealed_test_evaluated=False, utility_wiki=False, source_checkpoint_sha256=source_hash))
            _, _, protocol = frozen_sources(scale)
            self.assertEqual(protocol["source_checkpoint_sha256"], source_hash)
            outer_manifest["datasets"]["val"]["val_cutoff"] = "2023-07-20"
            write_json(outer / "run_manifest.json", outer_manifest)
            with self.assertRaises(ValueError):
                frozen_sources(scale)
            write_json(inner / "checkpoint", {"weight": 2})
            with self.assertRaises(ValueError):
                frozen_sources(scale)

    def test_exact_inner_scaler_is_preserved_and_sealed_targets_excluded(self):
        dates = np.datetime64("2023-01-01", "ns") + np.arange(400) * np.timedelta64(10, "m")
        dataset = SimpleNamespace(segments=np.array([[0, 400]]), dates=dates, feature_columns=["power"],
            scaler=SimpleNamespace(mean_=np.array([5.]), scale_=np.array([2.])),
            train_cutoff=dates[100], val_cutoff=dates[200], turbines=np.ones(400), available_mask=np.ones(400, bool))
        manifest = dict(args=dict(feature_columns=["power"]), datasets=dict(train=dict(
            train_cutoff=str(dates[100]), val_cutoff=str(dates[200]), scaler=dict(mean=[5.], scale=[2.]))))
        protocol = dict(val_start=str(dates[336]), val_end_exclusive=str(dates[384]),
            sealed_test_start=str(dates[384]), expected_windows=4,
            target_date_min=str(dates[336]), target_date_max=str(dates[383]))
        new, times, _, _ = followup_dataset(dataset, manifest, protocol)
        self.assertIs(new.scaler, dataset.scaler)
        self.assertTrue((times < dates[384]).all())
        self.assertFalse(hasattr(dataset, "window_starts"))
        bad = copy.deepcopy(manifest)
        bad["datasets"]["train"]["scaler"]["mean"] = [6.]
        with self.assertRaises(AssertionError):
            followup_dataset(dataset, bad, protocol)
        bad_protocol = dict(protocol, sealed_test_start=str(dates[380]))
        with self.assertRaises(ValueError):
            followup_dataset(dataset, manifest, bad_protocol)

    def test_frozen_application_never_refits_and_reports_harm_honestly(self):
        base, anchor, target = np.full((64, 12), 20.), np.zeros((64, 12)), np.full((64, 12), 10.)
        model = fit_scale(base, anchor, target, np.zeros(64))
        arrays = dict(prediction=np.full((3, 12), 20.), persistence=np.zeros((3, 12)),
            truth=np.array([[10.] * 12, [20.] * 12, [0.] * 12]), trend=np.zeros(3),
            target_timestamps=np.datetime64("2023-07-01") + np.arange(36).reshape(3, 12) * np.timedelta64(10, "m"))
        protocol = dict(protocol_id="frozen", source_checkpoint_sha256="same", bootstrap_block_time_points=144,
                        bootstrap_replicates=5000, bootstrap_seed=2024)
        with patch("utils.residual_scale_calibration.fit_scale", side_effect=AssertionError("no fit allowed")), \
                patch("scripts.confirm_sdwpf_residual_scale.paired_forecast_statistics", return_value={}) as stats:
            prediction, report = summarize(model, arrays, protocol)
        np.testing.assert_array_equal(prediction, np.full((3, 12), 10.))
        self.assertEqual(stats.call_count, 2)
        self.assertEqual(report["correction"]["changed_window_pct"], 100.)
        self.assertAlmostEqual(report["correction"]["harm_changed_window_pct"], 100 / 3)
        self.assertFalse(report["fitting_performed"])
        self.assertFalse(report["utility_wiki"])
        self.assertEqual(len(report["horizon"]), 12)

    def test_duplicate_starts_or_wrong_window_count_cannot_be_called_matched(self):
        dates = np.datetime64("2023-01-01") + np.arange(360) * np.timedelta64(10, "m")
        dataset = SimpleNamespace(segments=np.array([[0, 360]]), dates=dates, feature_columns=["power"],
            scaler=SimpleNamespace(mean_=np.array([0.]), scale_=np.array([1.])),
            train_cutoff=dates[100], val_cutoff=dates[200], turbines=np.ones(360), available_mask=np.ones(360, bool))
        manifest = dict(args=dict(feature_columns=["power"]), datasets=dict(train=dict(
            train_cutoff=str(dates[100]), val_cutoff=str(dates[200]), scaler=dict(mean=[0.], scale=[1.]))))
        protocol = dict(val_start=str(dates[336]), val_end_exclusive=str(dates[359]),
            sealed_test_start=str(dates[359]), expected_windows=99,
            target_date_min=str(dates[336]), target_date_max=str(dates[347]))
        with self.assertRaises(ValueError):
            followup_dataset(dataset, manifest, protocol)


if __name__ == "__main__":
    unittest.main()
