import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from data_provider.data_loader import (
    Dataset_SDWPF,
    _ffill_short_missing_runs,
    _sdwpf_cutoffs,
    _sdwpf_test_start_cutoff,
)
from data_provider.sdwpf_features import (
    channel_prior_vector,
    sdwpf_feature_columns,
)
from layers.TimeDART_EncDec import ChannelMixer
from exp.exp_timedart import Exp_TimeDART
from models.TimeDART import Model
from run import build_parser, configure_args, resolve_pretrained_checkpoint


def _tiny_sdwpf_csv(path, n_times=40, n_turbines=2):
    rows = []
    start = pd.Timestamp("2023-01-01 00:00:00")
    for turb in range(1, n_turbines + 1):
        for step in range(n_times):
            stamp = start + pd.Timedelta(minutes=10 * step)
            wind = 8.0 + 0.05 * step + 0.2 * turb
            rows.append(
                {
                    "date": stamp,
                    "TurbID": turb,
                    "Day": 1 + step // 144,
                    "Wspd": wind,
                    "Wdir": 30.0,
                    "Etmp": 20.0,
                    "Itmp": 25.0,
                    "Ndir": 40.0,
                    "Pab1": 1.0,
                    "Pab2": 1.0,
                    "Pab3": 1.0,
                    "Prtv": 0.0,
                    "power": max(0.0, 50.0 * wind),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


class ChannelMixerTests(unittest.TestCase):
    def test_wind_channel_changes_power_token(self):
        mixer = ChannelMixer(num_features=3, d_model=8, dropout=0.0)
        mixer.eval()
        base = torch.zeros(2, 3, 4, 8)
        base[:, 2] = 0.1
        wind = base.clone()
        wind[:, 0] = 1.7
        with torch.no_grad():
            out_base = mixer(base)
            out_wind = mixer(wind)
        self.assertEqual(tuple(out_wind.shape), (2, 1, 4, 8))
        self.assertGreater((out_wind - out_base).abs().max().item(), 1e-6)

    def test_rejects_feature_mismatch(self):
        mixer = ChannelMixer(num_features=2, d_model=4, dropout=0.0)
        with self.assertRaises(ValueError):
            mixer(torch.zeros(1, 3, 2, 4))

    def test_context_gate_starts_at_physics_prior_then_adapts(self):
        mixer = ChannelMixer(
            num_features=2,
            d_model=4,
            dropout=0.0,
            channel_prior=[2.5, 0.5],
            context_dim=3,
        )
        context = torch.zeros(2, 3)
        initial = mixer.effective_channel_scale(context)
        expected = torch.tensor([[2.5, 0.5], [2.5, 0.5]])
        self.assertTrue(torch.allclose(initial, expected))
        with torch.no_grad():
            mixer.context_gate.bias.copy_(torch.tensor([2.0, -2.0]))
        adapted = mixer.effective_channel_scale(context)
        self.assertGreater(adapted[0, 0].item(), initial[0, 0].item())
        self.assertLess(adapted[0, 1].item(), initial[0, 1].item())

    def test_full_model_wind_perturbation_changes_power(self):
        parser = build_parser()
        args = configure_args(
            parser.parse_args(
                [
                    "--task_name",
                    "finetune",
                    "--model_id",
                    "SDWPF",
                    "--model",
                    "TimeDART",
                    "--data",
                    "SDWPF",
                    "--allow_random_init",
                    "--no-use_gpu",
                    "--d_model",
                    "32",
                    "--n_heads",
                    "4",
                    "--e_layers",
                    "1",
                    "--d_ff",
                    "64",
                ]
            )
        )
        args.device = torch.device("cpu")
        args.time_steps = 8
        args.scheduler = "cosine"
        args.dropout = 0.0
        args.head_dropout = 0.0
        args.residual_forecast = True
        args.zero_init_residual_head = False
        args.use_soft_prompt = False
        args.use_prompt_adaln = False
        model = Model(args).eval()
        self.assertLess(torch.sigmoid(model.residual_gate_logit).item(), 0.05)
        x = torch.randn(2, args.input_len, args.enc_in)
        x2 = x.clone()
        x2[:, :80, 0] += 2.0
        with torch.no_grad():
            delta = (model(x2) - model(x)).abs().max().item()
        self.assertGreater(delta, 1e-6)

    def test_zero_initialized_mixed_residual_starts_at_persistence(self):
        parser = build_parser()
        args = configure_args(
            parser.parse_args(
                [
                    "--task_name",
                    "finetune",
                    "--model_id",
                    "SDWPF",
                    "--model",
                    "TimeDART",
                    "--data",
                    "SDWPF",
                    "--allow_random_init",
                    "--no-use_gpu",
                    "--d_model",
                    "32",
                    "--n_heads",
                    "4",
                    "--e_layers",
                    "1",
                    "--d_ff",
                    "64",
                    "--residual_forecast",
                    "--zero_init_residual_head",
                ]
            )
        )
        args.device = torch.device("cpu")
        args.time_steps = 8
        args.dropout = 0.0
        args.head_dropout = 0.0
        model = Model(args).eval()
        x = torch.randn(2, args.input_len, args.enc_in)
        with torch.no_grad():
            prediction = model(x)
        persistence = x[:, -1:, -1:].expand(-1, args.pred_len, -1)
        self.assertTrue(torch.equal(prediction, persistence))

    def test_constant_wind_level_changes_power(self):
        """Absolute Wspd must survive instance norm, not only within-window shape."""
        parser = build_parser()
        args = configure_args(
            parser.parse_args(
                [
                    "--task_name",
                    "finetune",
                    "--model_id",
                    "SDWPF",
                    "--model",
                    "TimeDART",
                    "--data",
                    "SDWPF",
                    "--allow_random_init",
                    "--no-use_gpu",
                    "--d_model",
                    "32",
                    "--n_heads",
                    "4",
                    "--e_layers",
                    "1",
                    "--d_ff",
                    "64",
                ]
            )
        )
        args.device = torch.device("cpu")
        args.time_steps = 8
        args.scheduler = "cosine"
        args.dropout = 0.0
        args.head_dropout = 0.0
        args.residual_forecast = False
        args.use_soft_prompt = False
        args.use_prompt_adaln = False
        model = Model(args).eval()
        self.assertEqual(args.feature_columns[0], "Wspd")
        self.assertIn(0, model.revin_keep_indices)
        x = torch.randn(2, args.input_len, args.enc_in)
        x2 = x.clone()
        x2[:, :, 0] += 1.5
        with torch.no_grad():
            delta = (model(x2) - model(x)).abs().max().item()
        self.assertGreater(delta, 1e-4)

    def test_mixer_prior_prefers_wind_over_reactive(self):
        names = sdwpf_feature_columns(
            physics_features=False,
            drop_weak_features=False,
        )
        prior = channel_prior_vector(names, physics_init=True)
        self.assertGreater(prior[names.index("Wspd")], prior[names.index("Prtv")])
        self.assertGreater(prior[names.index("power")], prior[names.index("Itmp")])


class CausalFillTests(unittest.TestCase):
    def test_ffill_does_not_use_future(self):
        series = pd.Series([1.0, np.nan, np.nan, 9.0])
        filled = _ffill_short_missing_runs(series, max_gap=6)
        self.assertEqual(filled.tolist(), [1.0, 1.0, 1.0, 9.0])


class OptimizerGroupTests(unittest.TestCase):
    def test_new_forecast_modules_receive_the_configured_higher_lr(self):
        class TinyForecastModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(2, 2)
                self.head = torch.nn.Linear(2, 1)
                self.channel_mixer = torch.nn.Linear(2, 2)
                self.residual_gate_logit = torch.nn.Parameter(torch.tensor(-2.2))

        experiment = object.__new__(Exp_TimeDART)
        experiment.model = TinyForecastModel()
        experiment.args = SimpleNamespace(
            task_name="finetune",
            downstream_task="forecast",
            learning_rate=1e-6,
            new_module_learning_rate=1e-4,
            weight_decay=1e-4,
        )
        optimizer = Exp_TimeDART._select_optimizer(experiment)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["group_name"], "transferred_backbone")
        self.assertEqual(optimizer.param_groups[1]["group_name"], "new_forecast_modules")
        self.assertAlmostEqual(optimizer.param_groups[0]["target_lr"], 1e-6)
        self.assertAlmostEqual(optimizer.param_groups[1]["target_lr"], 1e-4)
        grouped = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        expected = [id(parameter) for parameter in experiment.model.parameters()]
        self.assertCountEqual(grouped, expected)
        self.assertEqual(len(grouped), len(set(grouped)))


class CutoffTests(unittest.TestCase):
    def test_rolling_train_never_includes_later_fold_test(self):
        dates = pd.date_range("2023-01-01", periods=100, freq="10min").to_numpy()
        train0, val0, test0 = _sdwpf_cutoffs(dates, "rolling", 0, 3, 0.7, 0.1)
        train1, val1, test1 = _sdwpf_cutoffs(dates, "rolling", 1, 3, 0.7, 0.1)
        self.assertLess(train0, val0)
        self.assertLess(train1, val1)
        self.assertLessEqual(train0, train1)
        self.assertLessEqual(val0, val1)
        self.assertEqual(test0, val1)
        self.assertIsNotNone(test1)

    def test_rolling_test_blocks_are_pairwise_disjoint(self):
        dates = pd.date_range("2023-01-01", periods=100, freq="10min").to_numpy()
        blocks = []
        for fold in range(3):
            _, start, end = _sdwpf_cutoffs(dates, "rolling", fold, 3, 0.7, 0.1)
            mask = dates >= start
            if end is not None:
                mask &= dates < end
            blocks.append(set(dates[mask].tolist()))
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                self.assertTrue(blocks[i].isdisjoint(blocks[j]))

    def test_rolling_holdout_validation_blocks_are_disjoint_and_before_test(self):
        dates = pd.date_range("2023-01-01", periods=1000, freq="10min").to_numpy()
        validation_blocks = []
        holdout_start = None
        for fold in range(3):
            train_end, val_end, test_end = _sdwpf_cutoffs(
                dates, "rolling_holdout", fold, 3, 0.7, 0.1
            )
            self.assertIsNone(test_end)
            test_start = _sdwpf_test_start_cutoff(
                dates, "rolling_holdout", 0.7, 0.1, val_end
            )
            holdout_start = test_start if holdout_start is None else holdout_start
            self.assertEqual(test_start, holdout_start)
            self.assertLess(train_end, val_end)
            self.assertLessEqual(val_end, test_start)
            validation_blocks.append(set(dates[(dates >= train_end) & (dates < val_end)]))
        for left in range(len(validation_blocks)):
            for right in range(left + 1, len(validation_blocks)):
                self.assertTrue(validation_blocks[left].isdisjoint(validation_blocks[right]))


class PretrainIsolationTests(unittest.TestCase):
    def test_epoch_zero_validation_is_explicitly_opt_in(self):
        parser = build_parser()
        self.assertFalse(parser.get_default("validate_before_training"))
        self.assertTrue(
            parser.parse_args(
                [
                    "--task_name",
                    "finetune",
                    "--model_id",
                    "SDWPF",
                    "--model",
                    "PromptTimeDART",
                    "--data",
                    "SDWPF",
                    "--validate_before_training",
                ]
            ).validate_before_training
        )

    def test_none_policy_cannot_auto_load_an_existing_pretrain(self):
        parser = build_parser()
        args = configure_args(
            parser.parse_args(
                [
                    "--task_name",
                    "finetune",
                    "--model_id",
                    "SDWPF",
                    "--model",
                    "TimeDART",
                    "--data",
                    "SDWPF",
                    "--pretrain_init",
                    "none",
                    "--allow_random_init",
                    "--no-use_gpu",
                ]
            )
        )
        self.assertIsNone(resolve_pretrained_checkpoint(args))


class DatasetAlignmentTests(unittest.TestCase):
    def setUp(self):
        Dataset_SDWPF._cache.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.csv = os.path.join(self.tmp.name, "tiny.csv")
        _tiny_sdwpf_csv(self.csv)

    def tearDown(self):
        Dataset_SDWPF._cache.clear()
        self.tmp.cleanup()

    def _make(self, flag, **kwargs):
        params = dict(
            root_path=self.tmp.name,
            data_path="tiny.csv",
            flag=flag,
            size=[8, 0, 4],
            features="MS",
            target="power",
            scale=True,
            timeenc=0,
            freq="10min",
            train_ratio=0.6,
            val_ratio=0.2,
            expected_freq="10min",
            window_stride=4,
            rated_power=1500.0,
            split="time_ratio",
        )
        params.update(kwargs)
        return Dataset_SDWPF(**params)

    def test_history_ends_before_target(self):
        data = self._make("train")
        x, y, _, _ = data[0]
        self.assertEqual(x.shape[0], 8)
        self.assertEqual(y.shape[0], 4)
        start = int(data.window_starts[0])
        hist_end = data.dates[start + data.seq_len - 1]
        target_start = data.dates[start + data.seq_len]
        self.assertLess(hist_end, target_start)

    def test_rolling_holdout_windows_keep_cv_and_test_targets_disjoint(self):
        _tiny_sdwpf_csv(self.csv, n_times=240, n_turbines=1)
        Dataset_SDWPF._cache.clear()

        def target_dates(dataset):
            rows = (
                dataset.window_starts[:, None]
                + dataset.seq_len
                + np.arange(dataset.pred_len)[None, :]
            )
            return set(dataset.dates[rows].reshape(-1).tolist())

        validation_sets = []
        test_sets = []
        for fold in range(2):
            common = dict(
                split="rolling_holdout",
                fold=fold,
                n_folds=2,
                train_ratio=0.6,
                val_ratio=0.2,
            )
            train = self._make("train", **common)
            val = self._make("val", **common)
            test = self._make("test", **common)
            train_targets = target_dates(train)
            val_targets = target_dates(val)
            test_targets = target_dates(test)
            self.assertTrue(train_targets.isdisjoint(val_targets))
            self.assertTrue(train_targets.isdisjoint(test_targets))
            self.assertTrue(val_targets.isdisjoint(test_targets))
            validation_sets.append(val_targets)
            test_sets.append(test_targets)
        self.assertTrue(validation_sets[0].isdisjoint(validation_sets[1]))
        self.assertEqual(test_sets[0], test_sets[1])

    def test_scaler_fits_train_only(self):
        train = self._make("train")
        val = self._make("val")
        self.assertEqual(tuple(train.scaler.mean_), tuple(val.scaler.mean_))
        train_power = train.data_x[train.dates < train.train_cutoff, -1]
        self.assertGreater(train_power.size, 0)

    def test_power_is_clipped_non_negative(self):
        data = self._make("train", clip_power=True)
        index = data.feature_columns.index("power")
        mean = data.scaler.mean_[index]
        scale = data.scaler.scale_[index]
        power = data.data_x[:, index] * scale + mean
        self.assertGreaterEqual(power.min(), -1e-4)

    def test_long_invalid_run_creates_a_hard_window_boundary(self):
        _tiny_sdwpf_csv(self.csv, n_times=80, n_turbines=1)
        frame = pd.read_csv(self.csv)
        invalid_dates = set(pd.to_datetime(frame.loc[20:27, "date"]).to_numpy().tolist())
        frame.loc[20:27, "Wspd"] = 99.0
        frame.to_csv(self.csv, index=False)
        Dataset_SDWPF._cache.clear()
        data = self._make("train")
        for start in data.window_starts:
            end = int(start) + data.seq_len + data.pred_len
            used_dates = set(data.dates[int(start):end].tolist())
            self.assertTrue(used_dates.isdisjoint(invalid_dates))

    def test_missing_power_label_is_removed_not_imputed(self):
        _tiny_sdwpf_csv(self.csv, n_times=80, n_turbines=1)
        frame = pd.read_csv(self.csv)
        missing_date = pd.to_datetime(frame.loc[20, "date"]).to_datetime64()
        frame.loc[20, "power"] = np.nan
        frame.to_csv(self.csv, index=False)
        Dataset_SDWPF._cache.clear()
        data = self._make("train")
        self.assertNotIn(missing_date, data.dates)

    def test_feature_count_with_circular_and_pitch(self):
        data = self._make(
            "train",
            circular_wind=True,
            collapse_pitch=True,
            physics_features=False,
            drop_weak_features=False,
        )
        self.assertEqual(len(data.feature_columns), 10)
        self.assertIn("Wdir_sin", data.feature_columns)
        self.assertIn("Pab_mean", data.feature_columns)
        self.assertEqual(data.feature_columns[-1], "power")

    def test_physics_features_use_yaw_and_drop_weak(self):
        data = self._make(
            "train",
            circular_wind=True,
            collapse_pitch=True,
            physics_features=True,
            drop_weak_features=True,
        )
        self.assertEqual(
            data.feature_columns,
            [
                "Wspd",
                "Etmp",
                "Wdir_sin",
                "Wdir_cos",
                "yaw_sin",
                "yaw_cos",
                "Pab_mean",
                "power",
            ],
        )
        self.assertNotIn("Prtv", data.feature_columns)
        self.assertNotIn("Itmp", data.feature_columns)
        self.assertNotIn("Ndir_sin", data.feature_columns)


if __name__ == "__main__":
    unittest.main()
