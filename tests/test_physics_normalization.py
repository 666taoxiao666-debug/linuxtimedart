import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from models.TimeDART import Model
from run import build_parser, configure_args, pretrain_signature
from utils.physics_normalization import normalize_history, check_normalization_contract
from utils.run_tags import experiment_setting
from utils.tools import transfer_weights
from utils.accuracy_tuning import compare_fixed_epoch


def args_for(task, aligned):
    args = configure_args(build_parser().parse_args([
        "--task_name", task, "--model_id", "SDWPF", "--model", "TimeDART",
        "--data", "SDWPF", "--features", "MS", "--allow_random_init", "--no-use_gpu",
        "--input_len", "24", "--pred_len", "12", "--d_model", "16", "--n_heads", "4",
        "--e_layers", "1", "--d_ff", "32", "--patch_len", "6", "--stride", "6",
        "--revin_keep_wind",
        "--consistent_physics_norm" if aligned else "--no-consistent_physics_norm"]))
    args.device = torch.device("cpu")
    args.use_soft_prompt = False
    return args


class PhysicsNormalizationTests(unittest.TestCase):
    def test_legacy_normalization_is_bitwise_unchanged(self):
        x = torch.randn(2, 24, 8)
        mean = x.mean(1, keepdim=True).detach()
        centered = x - mean
        scale = torch.sqrt(centered.var(1, keepdim=True, unbiased=False) + 1e-5).detach()
        actual, actual_mean, actual_scale = normalize_history(x)
        self.assertTrue(torch.equal(actual, centered / scale))
        self.assertTrue(torch.equal(mean, actual_mean))
        self.assertTrue(torch.equal(scale, actual_scale))

    def test_kept_wind_is_invertible_and_power_unchanged(self):
        x = torch.randn(2, 24, 8)
        x[:, :, 0] = 2.5
        new, mean, scale = normalize_history(x, [0])
        old, _, _ = normalize_history(x)
        self.assertTrue(torch.equal(new[:, :, 0], x[:, :, 0]))
        self.assertTrue(torch.equal(new[:, :, 1:], old[:, :, 1:]))
        self.assertTrue(torch.allclose(new * scale + mean, x, atol=1e-6))
        self.assertTrue(torch.equal(mean[:, :, 0], torch.zeros(2, 1)))
        self.assertTrue(torch.equal(scale[:, :, 0], torch.ones(2, 1)))

    def test_actual_pretrain_and_forecast_patch_inputs_match(self):
        x = torch.randn(2, 24, 8)
        x[:, :, 0] += 2.0
        captures = []
        class Captured(Exception):
            pass
        def capture(module, inputs):
            captures.append(inputs[0].clone())
            raise Captured()
        for task in ("pretrain", "finetune"):
            model = Model(args_for(task, True)).eval()
            hook = model.patch.register_forward_pre_hook(capture)
            with self.assertRaises(Captured):
                model(x)
            hook.remove()
        self.assertTrue(torch.equal(captures[0], captures[1]))

    def test_legacy_forecast_power_output_is_bitwise_unchanged(self):
        legacy = Model(args_for("finetune", False)).eval()
        matched = Model(args_for("finetune", True)).eval()
        matched.load_state_dict(legacy.state_dict(), strict=True)
        legacy.zero_init_residual_head = False
        with torch.no_grad():
            legacy.head.forecast_head.weight.normal_(0, .02)
            matched.load_state_dict(legacy.state_dict(), strict=True)
            x = torch.randn(2, 24, 8)
            self.assertTrue(torch.equal(legacy(x), matched(x)))

    def test_matched_pretrain_reconstruction_has_finite_backbone_gradients(self):
        args = args_for("pretrain", True)
        args.time_steps = 8
        args.dropout = 0.0
        model = Model(args)
        x = torch.randn(2, 24, 8)
        x[:, :, 0] += 1.5
        prediction = model(x)
        self.assertEqual(tuple(prediction.shape), tuple(x.shape))
        loss = (prediction - x).square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradient = model.enc_embedding.patch_embedding.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_checkpoint_contract_refuses_cross_conventions(self):
        target = SimpleNamespace(consistent_physics_norm=True, feature_columns=["Wspd", "power"])
        with self.assertRaisesRegex(RuntimeError, "checkpoint mismatch"):
            check_normalization_contract({}, target)
        source = {"consistent_physics_norm": True, "use_norm": True, "revin_keep_wind": True,
                  "feature_columns": ["Wspd", "power"]}
        check_normalization_contract(source, target)
        with self.assertRaisesRegex(RuntimeError, "feature order"):
            check_normalization_contract(dict(source, feature_columns=["power", "Wspd"]), target)
        with self.assertRaisesRegex(RuntimeError, "wind/scale"):
            check_normalization_contract(dict(source, revin_keep_wind=False), target)

    def test_transfer_cannot_silently_use_old_pretrain(self):
        model = Model(args_for("finetune", True))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save({"model_state_dict": model.state_dict()}, path)
            with self.assertRaisesRegex(RuntimeError, "checkpoint mismatch"):
                transfer_weights(str(path), model)
            torch.save({"model_state_dict": model.state_dict(), "consistent_physics_norm": True,
                "use_norm": True, "revin_keep_wind": True,
                "feature_columns": model.feature_columns}, path)
            transfer_weights(str(path), model)

    def test_new_normalization_separates_run_ids_and_pretrain_discovery(self):
        old, new = args_for("finetune", False), args_for("finetune", True)
        self.assertNotEqual(pretrain_signature(old), pretrain_signature(new))
        self.assertNotEqual(experiment_setting(old, 0), experiment_setting(new, 0))

    def test_invalid_cli_convention_rejected(self):
        with self.assertRaisesRegex(ValueError, "consistent_physics_norm requires"):
            configure_args(build_parser().parse_args([
                "--task_name", "finetune", "--model_id", "SDWPF", "--data", "SDWPF", "--model", "TimeDART",
                "--consistent_physics_norm", "--no-revin_keep_wind"]))

    def test_fixed_epoch_comparison_ignores_a_better_earlier_epoch(self):
        reference = [{"epoch": 8, "val_mae_kw": 10, "val_rmse_kw": 20, "val_r2_original": .5,
            "val_mae_skill_original_pct": 5, "val_rmse_skill_original_pct": 5,
            "val_persistence_mae_kw": 11, "val_persistence_rmse_kw": 21}]
        candidate = [dict(reference[0], epoch=1, val_mae_kw=8, val_rmse_kw=18),
                     dict(reference[0], val_mae_kw=10.2, val_rmse_kw=19)]
        result = compare_fixed_epoch(reference, candidate, 8)
        self.assertFalse(result["joint_improvement"])
        self.assertAlmostEqual(result["gain_mae_kw"], -.2)
        bad = copy.deepcopy(candidate)
        bad[1]["val_persistence_mae_kw"] = 12
        with self.assertRaisesRegex(ValueError, "targets/scaler"):
            compare_fixed_epoch(reference, bad, 8)


if __name__ == "__main__":
    unittest.main()
