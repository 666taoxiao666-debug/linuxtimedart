from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
    transfer_weights,
    show_series,
    show_matrix,
)
from utils.augmentations import masked_data
from utils.forecast_report import build_forecast_report, save_training_history
from utils.metrics import forecast_metrics
from utils.forecast_losses import ForecastLoss
from utils.run_tags import forecast_result_tag
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter
import random
from contextlib import nullcontext
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")


class Exp_TimeDART(Exp_Basic):
    def __init__(self, args):
        super(Exp_TimeDART, self).__init__(args)
        log_dir = os.path.join("./outputs/logs", args.model, args.data)
        self.writer = SummaryWriter(log_dir)
        self.amp_enabled = bool(args.use_amp and self.device.type == "cuda")
        self.grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.amp_enabled,
        )

    def _autocast(self):
        if self.amp_enabled:
            return torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            )
        return nullcontext()

    def _build_model(self):
        if self.args.downstream_task == "forecast":
            model = self.model_dict[
                self.args.model
            ].Model(self.args).float()

        elif (
            self.args.downstream_task
            == "classification"
        ):
            model = self.model_dict[
                self.args.model
            ].ClsModel(self.args).float()

        if self.args.load_checkpoints:
            print(
                "Loading ckpt: {}".format(
                    self.args.load_checkpoints
                )
            )

            model = transfer_weights(
                self.args.load_checkpoints,
                model,
                device=self.device,
                strict=True,
            )

        if self.args.use_multi_gpu:
            print(
                "Let's use",
                torch.cuda.device_count(),
                "GPUs!",
                self.args.device_ids,
            )

            model = nn.DataParallel(
                model,
                device_ids=self.args.device_ids,
            )

        print(
            "number of model params",
            sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            ),
        )

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(
            self.args,
            flag,
        )
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        return model_optim

    def _select_criterion(self):
        if (
            self.args.task_name == "finetune"
            and self.args.downstream_task
            == "classification"
        ):
            criterion = nn.CrossEntropyLoss()
            print("Using CrossEntropyLoss")

        else:
            criterion = ForecastLoss(
                loss_name=self.args.loss,
                huber_delta=self.args.huber_delta,
                mix_mse_weight=(
                    self.args.mix_mse_weight
                ),
                horizon_weight_end=(
                    self.args.horizon_weight_end
                ),
                power_weight_alpha=(
                    self.args.power_weight_alpha
                ),
            )

            print(
                "Using ForecastLoss("
                f"base={self.args.loss}, "
                f"mix_mse="
                f"{self.args.mix_mse_weight}, "
                f"horizon_end="
                f"{self.args.horizon_weight_end}, "
                f"power_alpha="
                f"{self.args.power_weight_alpha})"
            )

        return criterion

    def pretrain(self):
        train_data, train_loader = (
            self._get_data(flag="train")
        )
        vali_data, vali_loader = (
            self._get_data(flag="val")
        )

        path = self.args.pretrain_run_dir
        os.makedirs(
            path,
            exist_ok=True,
        )

        model_optim = (
            self._select_optimizer()
        )

        model_scheduler = (
            torch.optim.lr_scheduler.ExponentialLR(
                optimizer=model_optim,
                gamma=self.args.lr_decay,
            )
        )

        min_vali_loss = None
        history = []

        for epoch in range(
            self.args.train_epochs
        ):
            start_time = time.time()

            current_lr = (
                model_scheduler
                .get_last_lr()[0]
            )

            print(
                "Current learning rate: "
                "{:.7f}".format(current_lr)
            )

            train_loss = (
                self.pretrain_one_epoch(
                    train_loader,
                    model_optim,
                    model_scheduler,
                )
            )

            vali_loss = (
                self.valid_one_epoch(
                    vali_loader
                )
            )

            end_time = time.time()

            print(
                "Epoch: {}/{}, Time: {:.2f}, "
                "Train Loss: {:.4f}, "
                "Vali Loss: {:.4f}".format(
                    epoch + 1,
                    self.args.train_epochs,
                    end_time - start_time,
                    train_loss,
                    vali_loss,
                )
            )

            loss_scalar_dict = {
                "train_loss": train_loss,
                "vali_loss": vali_loss,
            }

            self.writer.add_scalars(
                "/pretrain_loss",
                loss_scalar_dict,
                epoch,
            )

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(
                        train_loss
                    ),
                    "val_loss": float(
                        vali_loss
                    ),
                    "learning_rate": float(
                        current_lr
                    ),
                }
            )

            save_training_history(
                history,
                path,
                title="Pretraining history",
            )

            if (
                not min_vali_loss
                or vali_loss <= min_vali_loss
            ):
                if epoch == 0:
                    min_vali_loss = vali_loss

                print(
                    "Validation loss decreased "
                    "({:.6f} --> {:.6f}).  "
                    "Saving model epoch{}..."
                    .format(
                        min_vali_loss,
                        vali_loss,
                        epoch,
                    )
                )

                min_vali_loss = vali_loss

                self._save_pretrain_checkpoint(
                    path,
                    "ckpt_best.pth",
                    epoch,
                )

            if (epoch + 1) % 10 == 0:
                print(
                    "Saving model at epoch "
                    "{}...".format(epoch + 1)
                )

                self._save_pretrain_checkpoint(
                    path,
                    f"ckpt{epoch + 1}.pth",
                    epoch,
                )

        self.writer.flush()

    def _save_pretrain_checkpoint(
        self,
        path,
        filename,
        epoch,
    ):
        transfer_prefixes = (
            "sos_token",
            "encoder.",
            "enc_embedding.",
            "positional_encoding.",
            "regime_predictor.",
            "soft_prompt_generator.",
        )

        state = OrderedDict()

        for (
            name,
            value,
        ) in self.model.state_dict().items():
            clean_name = (
                name[7:]
                if name.startswith("module.")
                else name
            )

            if clean_name.startswith(
                transfer_prefixes
            ):
                state[clean_name] = (
                    value.detach().cpu()
                )

        checkpoint = {
            "epoch": epoch,
            "model": self.args.model,
            "data": self.args.data,
            "model_state_dict": state,
        }

        torch.save(
            checkpoint,
            os.path.join(
                path,
                filename,
            ),
        )

    def _unpack_pretrain_output(
        self,
        output,
    ):
        if isinstance(output, tuple):
            if len(output) == 3:
                return (
                    output[0],
                    output[1],
                    output[2],
                )

            if len(output) == 2:
                return (
                    output[0],
                    output[1],
                    None,
                )

        if isinstance(output, dict):
            return (
                output["pred"],
                output.get(
                    "regime_logits"
                ),
                output.get(
                    "pseudo_labels"
                ),
            )

        return output, None, None

    def pretrain_one_epoch(
        self,
        train_loader,
        model_optim,
        model_scheduler,
    ):
        train_loss = []

        model_criterion = (
            self._select_criterion()
        )
        ce_criterion = (
            nn.CrossEntropyLoss()
        )

        self.model.train()

        accumulation_steps = max(
            1,
            self.args.accumulation_steps,
        )

        model_optim.zero_grad(
            set_to_none=True
        )

        for i, (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
        ) in enumerate(train_loader):
            batch_x = (
                batch_x.float()
                .to(self.device)
            )

            with self._autocast():
                output = self.model(
                    batch_x
                )

                (
                    pred_x,
                    logits,
                    pseudo_labels,
                ) = (
                    self._unpack_pretrain_output(
                        output
                    )
                )

                diff_loss = model_criterion(
                    pred_x,
                    batch_x,
                )

                if (
                    logits is not None
                    and pseudo_labels
                    is not None
                ):
                    ce_loss = ce_criterion(
                        logits,
                        pseudo_labels,
                    )

                    total_loss = (
                        diff_loss
                        + self.args.lambda_ce
                        * ce_loss
                    )

                else:
                    total_loss = diff_loss

            self.grad_scaler.scale(
                total_loss
                / accumulation_steps
            ).backward()

            if (
                (i + 1)
                % accumulation_steps
                == 0
                or (i + 1)
                == len(train_loader)
            ):
                self.grad_scaler.step(
                    model_optim
                )
                self.grad_scaler.update()

                model_optim.zero_grad(
                    set_to_none=True
                )

            train_loss.append(
                total_loss.item()
            )

        model_scheduler.step()
        train_loss = np.mean(
            train_loss
        )

        return train_loss

    def valid_one_epoch(
        self,
        vali_loader,
    ):
        vali_loss = []

        model_criterion = (
            self._select_criterion()
        )
        ce_criterion = (
            nn.CrossEntropyLoss()
        )

        self.model.eval()

        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
            ) in enumerate(vali_loader):
                batch_x = (
                    batch_x.float()
                    .to(self.device)
                )

                with self._autocast():
                    output = self.model(
                        batch_x
                    )

                    (
                        pred_x,
                        logits,
                        pseudo_labels,
                    ) = (
                        self._unpack_pretrain_output(
                            output
                        )
                    )

                    diff_loss = (
                        model_criterion(
                            pred_x,
                            batch_x,
                        )
                    )

                    if (
                        logits is not None
                        and pseudo_labels
                        is not None
                    ):
                        ce_loss = (
                            ce_criterion(
                                logits,
                                pseudo_labels,
                            )
                        )

                        total_loss = (
                            diff_loss
                            + self.args.lambda_ce
                            * ce_loss
                        )

                    else:
                        total_loss = (
                            diff_loss
                        )

                vali_loss.append(
                    total_loss.item()
                )

        vali_loss = np.mean(
            vali_loss
        )

        return vali_loss

    def train(self, setting):
        train_data, train_loader = (
            self._get_data(flag="train")
        )
        vali_data, vali_loader = (
            self._get_data(flag="val")
        )

        path = os.path.join(
            self.args.checkpoints,
            setting,
        )

        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
        )

        model_optim = (
            self._select_optimizer()
        )
        model_criteria = (
            self._select_criterion()
        )

        model_scheduler = (
            lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=len(
                    train_loader
                ),
                pct_start=self.args.pct_start,
                epochs=self.args.train_epochs,
                max_lr=(
                    self.args.learning_rate
                ),
            )
        )

        history = []

        for epoch in range(
            self.args.train_epochs
        ):
            iter_count = 0
            train_loss = []

            progress = tqdm(
                train_loader,
                desc="Training",
            )

            print(
                "Current learning rate: "
                "{:.7f}".format(
                    model_optim
                    .param_groups[0]["lr"]
                )
            )

            self.model.train()
            start_time = time.time()

            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
            ) in enumerate(progress):
                iter_count += 1

                model_optim.zero_grad(
                    set_to_none=True
                )

                batch_x = (
                    batch_x.float()
                    .to(self.device)
                )
                batch_y = (
                    batch_y.float()
                    .to(self.device)
                )

                with self._autocast():
                    pred_x = self.model(
                        batch_x
                    )

                    f_dim = (
                        -1
                        if self.args.features
                        == "MS"
                        else 0
                    )

                    pred_x = pred_x[
                        :,
                        -self.args.pred_len:,
                        f_dim:,
                    ]

                    batch_y = batch_y[
                        :,
                        -self.args.pred_len:,
                        f_dim:,
                    ]

                    loss = model_criteria(
                        pred_x,
                        batch_y,
                    )

                self.grad_scaler.scale(
                    loss
                ).backward()

                self.grad_scaler.unscale_(
                    model_optim
                )

                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.grad_clip,
                    )

                self.grad_scaler.step(
                    model_optim
                )
                self.grad_scaler.update()

                if self.args.lradj == "step":
                    adjust_learning_rate(
                        model_optim,
                        model_scheduler,
                        epoch + 1,
                        self.args,
                        printout=False,
                    )
                    model_scheduler.step()

                train_loss.append(
                    loss.item()
                )

            train_loss = np.mean(
                train_loss
            )

            validation = self.valid(
                vali_loader,
                model_criteria,
            )

            vali_loss = validation[
                "loss"
            ]

            selection_value = validation[
                self.args.early_stop_metric
            ]

            current_lr = max(
                group["lr"]
                for group
                in model_optim.param_groups
            )

            end_time = time.time()

            print(
                "Epoch: {0}, Steps: {1}, "
                "Time: {2:.2f}s | "
                "Train Loss: {3:.7f} "
                "Vali Loss: {4:.7f} "
                "Vali MSE: {5:.7f} "
                "Vali MAE: {6:.7f} "
                "Select({7}): {8:.7f}"
                .format(
                    epoch + 1,
                    len(train_loader),
                    end_time - start_time,
                    train_loss,
                    vali_loss,
                    validation["mse"],
                    validation["mae"],
                    self.args
                    .early_stop_metric,
                    selection_value,
                )
            )

            log_path = (
                path + "/" + "log.txt"
            )

            with open(
                log_path,
                "a",
            ) as log_file:
                log_file.write(
                    "Epoch: {0}, Steps: {1}, "
                    "Time: {2:.2f}s | "
                    "Train Loss: {3:.7f} "
                    "Vali Loss: {4:.7f} "
                    "Vali MSE: {5:.7f} "
                    "Vali MAE: {6:.7f} "
                    "Select({7}): {8:.7f}\n"
                    .format(
                        epoch + 1,
                        len(train_loader),
                        end_time
                        - start_time,
                        train_loss,
                        vali_loss,
                        validation["mse"],
                        validation["mae"],
                        self.args
                        .early_stop_metric,
                        selection_value,
                    )
                )

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(
                        train_loss
                    ),
                    "val_loss": float(
                        vali_loss
                    ),
                    "val_mse": float(
                        validation["mse"]
                    ),
                    "val_mae": float(
                        validation["mae"]
                    ),
                    "selection_value": float(
                        selection_value
                    ),
                    "learning_rate": float(
                        current_lr
                    ),
                }
            )

            save_training_history(
                history,
                path,
            )

            self.writer.add_scalars(
                "/finetune_loss",
                {
                    "train_loss": (
                        train_loss
                    ),
                    "vali_loss": (
                        vali_loss
                    ),
                },
                epoch + 1,
            )

            self.writer.add_scalar(
                "/finetune_lr",
                current_lr,
                epoch + 1,
            )

            early_stopping(
                selection_value,
                self.model,
                path=path,
            )

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != "step":
                adjust_learning_rate(
                    model_optim,
                    model_scheduler,
                    epoch + 1,
                    self.args,
                )

        best_model_path = (
            path
            + "/"
            + "checkpoint.pth"
        )

        self.model.load_state_dict(
            torch.load(
                best_model_path,
                map_location=self.device,
            )
        )

        self.lr = (
            model_scheduler
            .get_last_lr()[0]
        )

        self.writer.flush()

        return self.model

    def valid(
        self,
        vali_loader,
        model_criteria,
    ):
        vali_loss = []
        squared_error_sum = 0.0
        absolute_error_sum = 0.0
        point_count = 0

        self.model.eval()

        vali_loader = tqdm(
            vali_loader,
            desc="Validation",
        )

        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
            ) in enumerate(vali_loader):
                batch_x = (
                    batch_x.float()
                    .to(self.device)
                )
                batch_y = (
                    batch_y.float()
                    .to(self.device)
                )

                with self._autocast():
                    pred_x = self.model(
                        batch_x
                    )

                    f_dim = (
                        -1
                        if self.args.features
                        == "MS"
                        else 0
                    )

                    pred_x = pred_x[
                        :,
                        -self.args.pred_len:,
                        f_dim:,
                    ]

                    batch_y = batch_y[
                        :,
                        -self.args.pred_len:,
                        f_dim:,
                    ]

                pred = (
                    pred_x.detach().cpu()
                )
                true = (
                    batch_y.detach().cpu()
                )

                loss = model_criteria(
                    pred_x,
                    batch_y,
                )

                vali_loss.append(
                    loss.item()
                )

                error = (
                    pred_x.float()
                    - batch_y.float()
                )

                squared_error_sum += (
                    error.square()
                    .sum()
                    .item()
                )

                absolute_error_sum += (
                    error.abs()
                    .sum()
                    .item()
                )

                point_count += (
                    error.numel()
                )

        vali_loss = float(
            np.mean(vali_loss)
        )

        self.model.train()

        if point_count == 0:
            raise RuntimeError(
                "Validation loader produced "
                "no forecast points"
            )

        return {
            "loss": vali_loss,
            "mse": (
                squared_error_sum
                / point_count
            ),
            "mae": (
                absolute_error_sum
                / point_count
            ),
        }

    def test(self):
        test_data, test_loader = (
            self._get_data(flag="test")
        )

        preds = []
        trues = []
        last_observations = []

        result_tag = forecast_result_tag(self.args)

        folder_path = os.path.join(
            "./outputs/test_results",
            self.args.model,
            self.args.data,
            result_tag,
        )

        os.makedirs(
            folder_path,
            exist_ok=True,
        )

        self.model.eval()

        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
            ) in enumerate(test_loader):
                batch_x = (
                    batch_x.float()
                    .to(self.device)
                )
                batch_y = (
                    batch_y.float()
                    .to(self.device)
                )

                with self._autocast():
                    pred_x = self.model(
                        batch_x
                    )

                    f_dim = (
                        -1
                        if self.args.features
                        == "MS"
                        else 0
                    )

                    pred_x = pred_x[
                        :,
                        -self.args.pred_len:,
                        f_dim:,
                    ]

                    batch_y = batch_y[
                        :,
                        -self.args.pred_len:,
                        f_dim:,
                    ]

                pred = (
                    pred_x.detach().cpu()
                )
                true = (
                    batch_y.detach().cpu()
                )

                preds.append(pred)
                trues.append(true)

                if (
                    self.args.features
                    == "MS"
                ):
                    last_observations.append(
                        batch_x[
                            :,
                            -1:,
                            -1:,
                        ]
                        .detach()
                        .cpu()
                    )

                elif batch_x.shape[-1] == 1:
                    last_observations.append(
                        batch_x[
                            :,
                            -1:,
                            :,
                        ]
                        .detach()
                        .cpu()
                    )

        preds = (
            torch.cat(
                preds,
                dim=0,
            )
            .numpy()
        )

        trues = (
            torch.cat(
                trues,
                dim=0,
            )
            .numpy()
        )

        last_values = (
            torch.cat(
                last_observations,
                dim=0,
            ).numpy()
            if last_observations
            else None
        )

        if preds.shape[-1] == 1:
            rated_power = (
                self.args.rated_power
                if self.args.rated_power > 0
                else None
            )

            values = build_forecast_report(
                preds_scaled=preds,
                trues_scaled=trues,
                dataset=test_data,
                output_dir=folder_path,
                last_observation_scaled=(
                    last_values
                ),
                step_minutes=(
                    10
                    if self.args.data
                    == "SDWPF"
                    else 1
                ),
                rated_power=rated_power,
            )

            print(
                f"{self.args.input_len}"
                f"->{self.args.pred_len} | "
                f"original-scale "
                f"RMSE={values['rmse']:.6f}, "
                f"MAE={values['mae']:.6f}, "
                f"R2={values['r2']:.6f}, "
                f"sMAPE="
                f"{values['smape_pct']:.3f}%"
            )

        else:
            values = forecast_metrics(
                preds,
                trues,
            )

            with open(
                os.path.join(
                    folder_path,
                    "score.txt",
                ),
                "w",
            ) as handle:
                for (
                    name,
                    value,
                ) in values.items():
                    handle.write(
                        f"{name}: {value}\n"
                    )

            print(
                f"{self.args.input_len}"
                f"->{self.args.pred_len} | "
                f"normalized "
                f"RMSE={values['rmse']:.6f}, "
                f"MAE={values['mae']:.6f}"
            )

        print(
            "Detailed forecast report: "
            f"{folder_path}"
        )

    def cls_train(self, setting):
        train_data, train_loader = (
            self._get_data(flag="train")
        )
        vali_data, vali_loader = (
            self._get_data(flag="val")
        )

        path = os.path.join(
            self.args.checkpoints,
            setting,
        )

        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = (
            self._select_optimizer()
        )

        selection_metric = (
            self.args
            .classification_early_stop_metric
        )

        selection_label = (
            "Validation Macro F1"
            if selection_metric
            == "macro_f1"
            else "Validation Accuracy"
        )

        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
            mode="max",
            metric_name=selection_label,
        )

        model_criteria = (
            self._select_criterion()
        )

        model_scheduler = (
            lr_scheduler.OneCycleLR(
                optimizer=model_optim,
                steps_per_epoch=len(
                    train_loader
                ),
                pct_start=self.args.pct_start,
                epochs=self.args.train_epochs,
                max_lr=(
                    self.args.learning_rate
                ),
            )
        )

        for epoch in range(
            self.args.train_epochs
        ):
            train_loss_sum = 0.0
            train_sample_count = 0
            train_preds = []
            train_trues = []

            print(
                "Current learning rate: "
                "{:.7f}".format(
                    model_optim
                    .param_groups[0]["lr"]
                )
            )

            self.model.train()

            progress = tqdm(
                train_loader,
                desc=(
                    "Classification training"
                ),
            )

            start_time = time.time()

            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
            ) in enumerate(progress):
                model_optim.zero_grad(
                    set_to_none=True
                )

                batch_x = (
                    batch_x.float()
                    .to(self.device)
                )
                batch_y = (
                    batch_y.long()
                    .to(self.device)
                )

                with self._autocast():
                    outputs = self.model(
                        batch_x
                    )
                    loss = model_criteria(
                        outputs,
                        batch_y,
                    )

                self.grad_scaler.scale(
                    loss
                ).backward()

                self.grad_scaler.step(
                    model_optim
                )
                self.grad_scaler.update()

                if self.args.lradj == "step":
                    adjust_learning_rate(
                        model_optim,
                        model_scheduler,
                        epoch + 1,
                        self.args,
                        printout=False,
                    )
                    model_scheduler.step()

                preds = (
                    torch.argmax(
                        outputs,
                        dim=1,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                trues = (
                    batch_y.detach()
                    .cpu()
                    .numpy()
                )

                batch_size = batch_y.size(0)

                train_loss_sum += (
                    loss.item()
                    * batch_size
                )
                train_sample_count += (
                    batch_size
                )

                train_preds.extend(
                    preds.tolist()
                )
                train_trues.extend(
                    trues.tolist()
                )

            if train_sample_count == 0:
                raise RuntimeError(
                    "The classification "
                    "training loader is empty"
                )

            train_loss = (
                train_loss_sum
                / train_sample_count
            )

            train_acc = accuracy_score(
                train_trues,
                train_preds,
            )

            train_f1 = f1_score(
                train_trues,
                train_preds,
                average="macro",
                zero_division=0,
            )

            (
                vali_loss,
                vali_acc,
                vali_f1,
            ) = self.cls_valid(
                vali_loader,
                model_criteria,
            )

            end_time = time.time()

            print(
                "Epoch: {0}, Steps: {1}, "
                "Time: {2:.2f}s | ".format(
                    epoch + 1,
                    len(train_loader),
                    end_time - start_time,
                )
                + (
                    "Train Loss: {:.7f}, "
                    "Acc: {:.4f}, "
                    "F1: {:.4f} | "
                ).format(
                    train_loss,
                    train_acc,
                    train_f1,
                )
                + (
                    "Vali Loss: {:.7f}, "
                    "Acc: {:.4f}, "
                    "F1: {:.4f}"
                ).format(
                    vali_loss,
                    vali_acc,
                    vali_f1,
                )
            )

            log_path = (
                path + "/" + "log.txt"
            )

            with open(
                log_path,
                "a",
            ) as log_file:
                log_file.write(
                    "Epoch: {0}, Steps: {1}, "
                    "Time: {2:.2f}s | "
                    "Train Loss: {3:.7f}, "
                    "Acc: {4:.4f}, "
                    "F1: {5:.4f} | "
                    "Vali Loss: {6:.7f}, "
                    "Acc: {7:.4f}, "
                    "F1: {8:.4f}\n".format(
                        epoch + 1,
                        len(train_loader),
                        end_time
                        - start_time,
                        train_loss,
                        train_acc,
                        train_f1,
                        vali_loss,
                        vali_acc,
                        vali_f1,
                    )
                )

            selection_value = (
                vali_f1
                if selection_metric
                == "macro_f1"
                else vali_acc
            )

            early_stopping(
                selection_value,
                self.model,
                path=path,
            )

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != "step":
                adjust_learning_rate(
                    model_optim,
                    model_scheduler,
                    epoch + 1,
                    self.args,
                )

        best_model_path = (
            path
            + "/"
            + "checkpoint.pth"
        )

        self.model.load_state_dict(
            torch.load(
                best_model_path,
                map_location=self.device,
            )
        )

        return self.model

    def cls_valid(
        self,
        vali_loader,
        model_criteria,
    ):
        vali_loss_sum = 0.0
        vali_sample_count = 0
        vali_preds = []
        vali_trues = []

        self.model.eval()

        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
            ) in enumerate(vali_loader):
                batch_x = (
                    batch_x.float()
                    .to(self.device)
                )
                batch_y = (
                    batch_y.long()
                    .to(self.device)
                )

                with self._autocast():
                    outputs = self.model(
                        batch_x
                    )
                    loss = model_criteria(
                        outputs,
                        batch_y,
                    )

                preds = (
                    torch.argmax(
                        outputs,
                        dim=1,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                trues = (
                    batch_y.detach()
                    .cpu()
                    .numpy()
                )

                batch_size = batch_y.size(0)

                vali_loss_sum += (
                    loss.item()
                    * batch_size
                )
                vali_sample_count += (
                    batch_size
                )

                vali_preds.extend(
                    preds.tolist()
                )
                vali_trues.extend(
                    trues.tolist()
                )

        if vali_sample_count == 0:
            raise RuntimeError(
                "The classification "
                "validation loader is empty"
            )

        vali_loss = (
            vali_loss_sum
            / vali_sample_count
        )

        vali_acc = accuracy_score(
            vali_trues,
            vali_preds,
        )

        vali_f1 = f1_score(
            vali_trues,
            vali_preds,
            average="macro",
            zero_division=0,
        )

        return (
            vali_loss,
            vali_acc,
            vali_f1,
        )

    def cls_test(
        self,
        write_log=True,
    ):
        test_data, test_loader = (
            self._get_data(flag="test")
        )

        model_criteria = (
            self._select_criterion()
        )

        preds_all = []
        trues_all = []
        test_loss = []

        folder_path = os.path.join(
            "./outputs/test_results",
            self.args.model,
            self.args.data,
        )

        if not os.path.exists(
            folder_path
        ):
            os.makedirs(folder_path)

        self.model.eval()

        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
            ) in enumerate(test_loader):
                batch_x = (
                    batch_x.float()
                    .to(self.device)
                )
                batch_y = (
                    batch_y.long()
                    .to(self.device)
                )

                with self._autocast():
                    outputs = self.model(
                        batch_x
                    )
                    loss = model_criteria(
                        outputs,
                        batch_y,
                    )

                preds = (
                    torch.argmax(
                        outputs,
                        dim=1,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                trues = (
                    batch_y.detach()
                    .cpu()
                    .numpy()
                )

                test_loss.append(
                    loss.item()
                )
                preds_all.extend(preds)
                trues_all.extend(trues)

        test_loss = np.mean(test_loss)

        test_acc = accuracy_score(
            trues_all,
            preds_all,
        )

        test_f1 = f1_score(
            trues_all,
            preds_all,
            average="macro",
            zero_division=0,
        )

        print(
            "Test Loss: {:.7f}, "
            "Acc: {:.4f}, "
            "F1: {:.4f}".format(
                test_loss,
                test_acc,
                test_f1,
            )
        )

        if write_log:
            f = open(
                folder_path + "/score.txt",
                "a",
            )

            f.write(
                "Test Loss: {:.7f}, "
                "Acc: {:.4f}, "
                "F1: {:.4f}\n".format(
                    test_loss,
                    test_acc,
                    test_f1,
                )
            )

            f.close()