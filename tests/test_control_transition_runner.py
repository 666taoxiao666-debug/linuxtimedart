import unittest
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from scripts.train_sdwpf_control_transition import reconstruct_controls, control_audit, run
from scripts.tune_sdwpf_accuracy import sha256


class ControlTransitionRunnerTests(unittest.TestCase):
    def fixture(self):
        dates = np.arange(np.datetime64("2023-01-01", "m"), np.datetime64("2023-01-03", "m"),
                          np.timedelta64(10, "m")).astype("datetime64[ns]")
        columns = ["Wspd", "Pab_mean", "Wdir_cos", "yaw_cos", "power"]
        x = np.zeros((len(dates), len(columns))); x[:, 1] = 2.; x[:, 2] = .9; x[:, 3] = -.4
        data = SimpleNamespace(feature_columns=columns, data_x=x, dates=dates,
            turbines=np.ones(len(dates), int), train_cutoff=dates[100], val_cutoff=dates[200],
            scaler=SimpleNamespace(mean_=np.zeros(5), scale_=np.ones(5)))
        starts = np.array([0, 12])
        # Extend the fixture without constructing data_y or accessing targets.
        dates = np.arange(np.datetime64("2023-01-01", "m"), np.datetime64("2023-01-06", "m"),
                          np.timedelta64(10, "m")).astype("datetime64[ns]")
        data.dates = dates; data.turbines = np.ones(len(dates), int)
        data.data_x = np.tile(x[0], (len(dates), 1))
        arrays = dict(window_starts=starts, turbines=np.ones(2, int),
            target_timestamps=dates[starts[:, None]+336+np.arange(12)])
        manifest = dict(args=dict(feature_columns=columns, sdwpf_causal_fill=True),
            datasets=dict(train=dict(scaler=dict(mean=np.zeros(5), scale=np.ones(5)))))
        audit = dict(fit_cutoff=str(data.train_cutoff), selection_end=str(data.val_cutoff))
        return data, arrays, manifest, audit

    def test_legacy_context_uses_raw_relative_wdir_not_wrong_yaw_difference(self):
        data, arrays, manifest, audit = self.fixture()
        with patch("scripts.train_sdwpf_control_transition.train_dataset", return_value=data):
            context = reconstruct_controls(manifest, audit, arrays)
            np.testing.assert_allclose(context["relative_yaw_cos_mean"], .9)
            np.testing.assert_allclose(context["pitch_mean_div90"], 2/90)
            arrays["target_timestamps"][0, 0] = data.dates[0]
            with self.assertRaises(AssertionError):
                reconstruct_controls(manifest, audit, arrays)

    def test_noncausal_source_is_rejected_before_loading(self):
        _, arrays, manifest, audit = self.fixture()
        manifest["args"]["sdwpf_causal_fill"] = False
        with patch("scripts.train_sdwpf_control_transition.train_dataset") as loader:
            with self.assertRaisesRegex(ValueError, "Noncausal"):
                reconstruct_controls(manifest, audit, arrays)
            loader.assert_not_called()

    def test_fixed_control_diagnostic_does_not_aggregate_excluded_check_labels(self):
        controls = dict(pitch_mean_div90=np.array([2., 80.])/90, relative_yaw_cos_mean=np.ones(2))
        arrays = dict(events=np.ones((2, 4), bool), truth=np.array([[330.], [1e9]]),
            prediction=np.full((2, 1), 310.), persistence=np.full((2, 1), 300.))
        result = control_audit(controls, arrays, dict(fit=np.array([True, False])))
        self.assertFalse(result["uses_check_labels"])
        for row in result["rows"]:
            self.assertEqual(row["windows"], 1)
            self.assertEqual(row["power_transition_kw"], 30.)

    def test_factorial_run_and_resume_reuse_completed_memory_and_predictions(self):
        from utils import control_transition_wiki as wiki
        n = 384
        rng = np.random.default_rng(2024)
        issue = np.datetime64("2023-06-01", "ns") + np.arange(n)*np.timedelta64(120, "m")
        times = issue[:, None] + np.arange(12)*np.timedelta64(10, "m")
        ids = np.tile(np.arange(1, 13), n//12)
        ref = np.full((n, 12), 310.); anchor = np.full_like(ref, 300.); truth = np.full_like(ref, 350.)
        controls = wiki.control_context(np.full((n, 12), 2.), np.ones((n, 12)))
        context = dict(power_ratio=rng.uniform(.15, .25, n), wind_mean_mps=rng.uniform(4, 7, n),
            wind_std_mps=rng.uniform(.1, .5, n), power_change_kw=rng.uniform(-5, 5, n),
            wind_change_mps=rng.uniform(-.1, .1, n), power_bin=np.ones(n, int), joint_bin=np.full(n, 5, int),
            window_starts=np.arange(n), history_last_time=issue-np.timedelta64(10, "m"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root/"source"; source.mkdir()
            directory = root/"run"; directory.mkdir()
            pred = source/"pred.npz"; manifest = source/"manifest.json"
            manifest.write_text(json.dumps({"args": {}}), encoding="utf-8")
            np.savez_compressed(pred, prediction=ref, persistence=anchor, truth=truth, turbines=ids,
                events=np.ones((n, 4), bool), window_starts=np.arange(n), target_timestamps=times)
            np.savez_compressed(source/"history_context.npz", **context)
            protocol = json.loads((Path(__file__).parents[1]/"configs/sdwpf_control_transition_protocol_v15.json").read_text())
            protocol["source_predictions_sha256"] = sha256(pred)
            protocol["expert"].update(neighbors=8, min_neighbor_turbines=2, min_fit_windows=16)
            protocol["transport"].update(min_windows=8, min_turbines=2)
            protocol["statistics"]["replicates"] = 100
            path = root/"protocol.json"; path.write_text(json.dumps(protocol), encoding="utf-8")
            drift = dict(source_checkpoint_sha256=protocol["source_base_checkpoint_sha256"], data_sha256="a"*64,
                selection_end=str(times.min()), end_exclusive=str(times.max()+np.timedelta64(1, "ns")))
            audit = dict(event_config_sha256="b"*64)
            def plot(prediction, truth, plot_dir, **kwargs):
                (Path(plot_dir)/"forecast_accuracy_overview.png").write_bytes(b"test fixture, not an experimental figure")
            # Windows lacks fcntl. Linux server tests exercise its real flock;
            # local tests mock only this platform boundary, not fit/evaluation.
            platform_modules = {} if os.name == "posix" else {"fcntl": SimpleNamespace(
                LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)}
            with patch("scripts.train_sdwpf_control_transition.verify_source", return_value=(drift, audit, pred, manifest)), \
                    patch("scripts.train_sdwpf_control_transition.reconstruct_controls", return_value=controls), \
                    patch("utils.forecast_report._plot_accuracy_overview", side_effect=plot), \
                    patch("scripts.train_sdwpf_control_transition.finish_log_directory"), \
                    patch("torch.set_num_interop_threads"), patch.dict("sys.modules", platform_modules):
                run(directory, source, path)
                self.assertTrue((directory/"completed.json").exists())
                result = json.loads((directory/"result.json").read_text())
                self.assertEqual(len(result["arms"]), 4)
                self.assertTrue(result["no_winner_selection_on_check"])
                receipt = sha256(directory/"completed.json")
                with patch("utils.control_transition_wiki.fit_experts", side_effect=AssertionError("must not refit")):
                    run(directory, source, path)
                self.assertEqual(receipt, sha256(directory/"completed.json"))


if __name__ == "__main__":
    unittest.main()
