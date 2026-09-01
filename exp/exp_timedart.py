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
from utils.experiment_audit import checkpoint_info, model_runtime_summary, write_run_manifest
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torch import optim
import os
import sys
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

        write_run_manifest(
            path,
            self.args,
            "pretrain",
            model=self.model,
            datasets={"train": train_data, "val": vali_data},
            checkpoints={"pretrained_source": checkpoint_info(self.args.load_checkpoints)},
            extra={"status": "started"},
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

        min_vali_loss = float("inf")
        best_epoch = None
        no_improvement = 0
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

            train_metrics = (
                self.pretrain_one_epoch(
                    train_loader,
                    model_optim,
                    model_scheduler,
                )
            )

            validation = (
                self.valid_one_epoch(
                    vali_loader
                )
            )

            end_time = time.time()

            print(
                "Epoch: {}/{}, Time: {:.2f}, "
                "Train Total/Diff/CE: {:.4f}/{:.4f}/{:.4f}, "
                "Val Total/Diff/CE: {:.4f}/{:.4f}/{:.4f}, "
                "Val Regime Acc: {:.3f}, GradNorm: {:.3f}".format(
                    epoch + 1,
                    self.args.train_epochs,
                    end_time - start_time,
                    train_metrics["total_loss"],
                    train_metrics["diff_loss"],
                    train_metrics["ce_loss"],
                    validation["total_loss"],
                    validation["diff_loss"],
                    validation["ce_loss"],
                    validation["regime_accuracy"],
                    train_metrics["grad_norm"],
                )
            )

            loss_scalar_dict = {
                "train_total": train_metrics["total_loss"],
                "train_diff": train_metrics["diff_loss"],
                "train_ce": train_metrics["ce_loss"],
                "vali_total": validation["total_loss"],
                "vali_diff": validation["diff_loss"],
                "vali_ce": validation["ce_loss"],
            }

            self.writer.add_scalars(
                "/pretrain_loss",
                loss_scalar_dict,
                epoch,
            )

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(train_metrics["total_loss"]),
                    "train_diff_loss": float(train_metrics["diff_loss"]),
                    "train_ce_loss": float(train_metrics["ce_loss"]),
                    "train_regime_accuracy": float(train_metrics["regime_accuracy"]),
                    "train_grad_norm": float(train_metrics["grad_norm"]),
                    "train_regime_counts": train_metrics["regime_counts"],
                    "val_loss": float(validation["total_loss"]),
                    "val_diff_loss": float(validation["diff_loss"]),
                    "val_ce_loss": float(validation["ce_loss"]),
                    "val_regime_accuracy": float(validation["regime_accuracy"]),
                    "val_regime_counts": validation["regime_counts"],
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

            vali_loss = validation["total_loss"]
            if vali_loss < min_vali_loss:
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
                best_epoch = epoch + 1
                no_improvement = 0

                self._save_pretrain_checkpoint(
                    path,
                    "ckpt_best.pth",
                    epoch,
                )
            else:
                no_improvement += 1
                print(
                    f"Pretrain early-stopping counter: {no_improvement} "
                    f"out of {self.args.patience}"
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

            if no_improvement >= self.args.patience:
                print("Pretrain early stopping")
                break

        self.writer.flush()
        best_path = os.path.join(path, "ckpt_best.pth")
        write_run_manifest(
            path,
            self.args,
            "pretrain",
            model=self.model,
            datasets={"train": train_data, "val": vali_data},
            checkpoints={"best": checkpoint_info(best_path)},
            extra={
                "status": "complete",
                "best_epoch": best_epoch,
                "best_validation_loss": min_vali_loss,
                "epochs_completed": len(history),
            },
        )

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
            "split": getattr(self.args, "sdwpf_split", None),
            "fold": getattr(self.args, "sdwpf_fold", None),
            "n_folds": getattr(self.args, "sdwpf_n_folds", None),
            "seed": self.args.seed,
            "feature_columns": list(getattr(self.args, "feature_columns", []) or []),
            "input_len": self.args.input_len,
            "pred_len": self.args.pred_len,
            "runtime_model": model_runtime_summary(self.model, self.args),
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
        total_losses = []
        diff_losses = []
        ce_losses = []
        grad_norms = []
        regime_correct = 0
        regime_total = 0
        regime_counts = np.zeros(int(self.args.num_modes), dtype=np.int64)

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
                    ce_loss = torch.zeros((), device=diff_loss.device)
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
                self.grad_scaler.unscale_(model_optim)
                if self.args.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.grad_clip,
                    )
                    grad_norms.append(float(grad_norm.detach().cpu().item()))
                self.grad_scaler.step(
                    model_optim
                )
                self.grad_scaler.update()

                model_optim.zero_grad(
                    set_to_none=True
                )

            total_losses.append(total_loss.item())
            diff_losses.append(diff_loss.item())
            ce_losses.append(ce_loss.item())
            if logits is not None and pseudo_labels is not None:
                predictions = logits.detach().argmax(dim=-1)
                labels = pseudo_labels.detach().reshape(-1)
                regime_correct += int((predictions.reshape(-1) == labels).sum().item())
                regime_total += int(labels.numel())
                counts = torch.bincount(labels, minlength=len(regime_counts)).cpu().numpy()
                regime_counts += counts[: len(regime_counts)]

        model_scheduler.step()
        return {
            "total_loss": float(np.mean(total_losses)),
            "diff_loss": float(np.mean(diff_losses)),
            "ce_loss": float(np.mean(ce_losses)),
            "regime_accuracy": float(regime_correct / regime_total) if regime_total else 0.0,
            "regime_counts": ";".join(str(int(value)) for value in regime_counts),
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        }

    def valid_one_epoch(
        self,
        vali_loader,
    ):
        total_losses = []
        diff_losses = []
        ce_losses = []
        regime_correct = 0
        regime_total = 0
        regime_counts = np.zeros(int(self.args.num_modes), dtype=np.int64)

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
                        ce_loss = torch.zeros((), device=diff_loss.device)
                        total_loss = diff_loss

                total_losses.append(total_loss.item())
                diff_losses.append(diff_loss.item())
                ce_losses.append(ce_loss.item())
                if logits is not None and pseudo_labels is not None:
                    predictions = logits.detach().argmax(dim=-1)
                    labels = pseudo_labels.detach().reshape(-1)
                    regime_correct += int((predictions.reshape(-1) == labels).sum().item())
                    regime_total += int(labels.numel())
                    counts = torch.bincount(
                        labels, minlength=len(regime_counts)
                    ).cpu().numpy()
                    regime_counts += counts[: len(regime_counts)]

        return {
            "total_loss": float(np.mean(total_losses)),
            "diff_loss": float(np.mean(diff_losses)),
            "ce_loss": float(np.mean(ce_losses)),
            "regime_accuracy": float(regime_correct / regime_total) if regime_total else 0.0,
            "regime_counts": ";".join(str(int(value)) for value in regime_counts),
        }

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

        write_run_manifest(
            path,
            self.args,
            "finetune",
            model=self.model,
            datasets={"train": train_data, "val": vali_data},
            checkpoints={"pretrained_source": checkpoint_info(self.args.load_checkpoints)},
            extra={"status": "started", "setting": setting},
        )

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
            train_squared_error = 0.0
            train_absolute_error = 0.0
            train_point_count = 0
            grad_norms = []

            progress = tqdm(
                train_loader,
                desc="Training",
                disable=not sys.stderr.isatty(),
                mininterval=5.0,
                dynamic_ncols=True,
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

                train_error = pred_x.detach().float() - batch_y.detach().float()
                train_squared_error += train_error.square().sum().item()
                train_absolute_error += train_error.abs().sum().item()
                train_point_count += train_error.numel()

                self.grad_scaler.scale(
                    loss
                ).backward()

                self.grad_scaler.unscale_(
                    model_optim
                )

                if self.args.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.grad_clip,
                    )
                    grad_norms.append(float(grad_norm.detach().cpu().item()))

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
            train_mse = train_squared_error / max(1, train_point_count)
            train_mae = train_absolute_error / max(1, train_point_count)
            train_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0

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
                "Persist MAE: {7:.7f} "
                "MAE Skill: {8:+.2f}% "
                "Gate: {9:.5f} GradNorm: {10:.3f} "
                "Select({11}): {12:.7f}"
                .format(
                    epoch + 1,
                    len(train_loader),
                    end_time - start_time,
                    train_loss,
                    vali_loss,
                    validation["mse"],
                    validation["mae"],
                    validation["persistence_mae"],
                    validation["mae_skill_vs_persistence_pct"],
                    validation["diagnostics"].get("residual_gate", 0.0),
                    train_grad_norm,
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
                    "Persist MAE: {7:.7f} "
                    "MAE Skill: {8:+.2f}% "
                    "Gate: {9:.5f} GradNorm: {10:.3f} "
                    "Select({11}): {12:.7f}\n"
                    .format(
                        epoch + 1,
                        len(train_loader),
                        end_time
                        - start_time,
                        train_loss,
                        vali_loss,
                        validation["mse"],
                        validation["mae"],
                        validation["persistence_mae"],
                        validation["mae_skill_vs_persistence_pct"],
                        validation["diagnostics"].get("residual_gate", 0.0),
                        train_grad_norm,
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
                    "train_mse": float(train_mse),
                    "train_mae": float(train_mae),
                    "train_grad_norm": train_grad_norm,
                    "val_loss": float(
                        vali_loss
                    ),
                    "val_mse": float(
                        validation["mse"]
                    ),
                    "val_mae": float(
                        validation["mae"]
                    ),
                    "val_persistence_mse": float(validation["persistence_mse"]),
                    "val_persistence_mae": float(validation["persistence_mae"]),
                    "val_mae_skill_vs_persistence_pct": float(
                        validation["mae_skill_vs_persistence_pct"]
                    ),
                    "val_rmse_skill_vs_persistence_pct": float(
                        validation["rmse_skill_vs_persistence_pct"]
                    ),
                    "selection_value": float(
                        selection_value
                    ),
                    "learning_rate": float(
                        current_lr
                    ),
                    **validation["diagnostics"],
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

        best_record = min(history, key=lambda record: record["selection_value"])
        self.args.selected_finetune_checkpoint = os.path.abspath(best_model_path)
        write_run_manifest(
            path,
            self.args,
            "finetune",
            model=self.model,
            datasets={"train": train_data, "val": vali_data},
            checkpoints={
                "pretrained_source": checkpoint_info(self.args.load_checkpoints),
                "best_finetuned": checkpoint_info(best_model_path),
            },
            extra={
                "status": "complete",
                "setting": setting,
                "best_epoch": int(best_record["epoch"]),
                "best_selection_value": float(best_record["selection_value"]),
                "epochs_completed": len(history),
            },
        )
        print(f"[AUDIT] FINETUNE_CHECKPOINT={os.path.abspath(best_model_path)}")

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
        persistence_squared_error_sum = 0.0
        persistence_absolute_error_sum = 0.0
        point_count = 0
        dynamic_scale_samples = []
        core_model = (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )

        self.model.eval()

        vali_loader = tqdm(
            vali_loader,
            desc="Validation",
            disable=not sys.stderr.isatty(),
            mininterval=5.0,
            dynamic_ncols=True,
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

                    persistence = batch_x[:, -1:, f_dim:].expand(
                        -1, self.args.pred_len, -1
                    )

                    mixer = getattr(core_model, "channel_mixer", None)
                    if mixer is not None and hasattr(core_model, "_operating_context"):
                        context = core_model._operating_context(batch_x)
                        if context is not None:
                            dynamic_scale_samples.append(
                                mixer.effective_channel_scale(context)
                                .detach()
                                .float()
                                .cpu()
                                .numpy()
                            )

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

                persistence_error = persistence.float() - batch_y.float()
                persistence_squared_error_sum += persistence_error.square().sum().item()
                persistence_absolute_error_sum += persistence_error.abs().sum().item()

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

        model_mse = squared_error_sum / point_count
        model_mae = absolute_error_sum / point_count
        persistence_mse = persistence_squared_error_sum / point_count
        persistence_mae = persistence_absolute_error_sum / point_count
        eps = np.finfo(float).eps

        diagnostics = {}
        runtime = model_runtime_summary(self.model, self.args)
        if "residual_gate" in runtime:
            diagnostics["residual_gate"] = float(runtime["residual_gate"])
        for name, value in runtime.get("channel_scale", {}).items():
            diagnostics[f"channel_static_{name}"] = float(value)
        if dynamic_scale_samples:
            dynamic = np.concatenate(dynamic_scale_samples, axis=0)
            names = list(getattr(self.args, "feature_columns", []) or [])
            for index in range(dynamic.shape[1]):
                name = names[index] if index < len(names) else f"feature_{index}"
                values = dynamic[:, index]
                diagnostics[f"channel_dynamic_{name}_mean"] = float(np.mean(values))
                diagnostics[f"channel_dynamic_{name}_std"] = float(np.std(values))
                diagnostics[f"channel_dynamic_{name}_p05"] = float(np.quantile(values, 0.05))
                diagnostics[f"channel_dynamic_{name}_p50"] = float(np.quantile(values, 0.50))
                diagnostics[f"channel_dynamic_{name}_p95"] = float(np.quantile(values, 0.95))

        return {
            "loss": vali_loss,
            "mse": model_mse,
            "mae": model_mae,
            "persistence_mse": persistence_mse,
            "persistence_mae": persistence_mae,
            "mae_skill_vs_persistence_pct": 100.0
            * (1.0 - model_mae / max(persistence_mae, eps)),
            "rmse_skill_vs_persistence_pct": 100.0
            * (1.0 - np.sqrt(model_mse) / max(np.sqrt(persistence_mse), eps)),
            "diagnostics": diagnostics,
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
            if "0_4h_mae" in values:
                print(
                    "  0-4 h  MAE="
                    f"{values['0_4h_mae']:.3f}  RMSE="
                    f"{values['0_4h_rmse']:.3f}"
                    + (
                        f"  MAE_skill={values['0_4h_mae_skill_vs_persistence_pct']:+.2f}%"
                        if "0_4h_mae_skill_vs_persistence_pct" in values
                        else ""
                    )
                )
            if "4_16h_mae" in values:
                print(
                    "  4-16 h MAE="
                    f"{values['4_16h_mae']:.3f}  RMSE="
                    f"{values['4_16h_rmse']:.3f}"
                    "  (no NWP; not a SOTA claim)"
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

        checkpoint_path = getattr(
            self.args,
            "loaded_finetune_checkpoint",
            getattr(self.args, "selected_finetune_checkpoint", None),
        )
        write_run_manifest(
            folder_path,
            self.args,
            "test",
            model=self.model,
            datasets={"test": test_data},
            checkpoints={"fine_tuned": checkpoint_info(checkpoint_path)},
            extra={
                "status": "complete",
                "prediction_shape": list(preds.shape),
                "target_shape": list(trues.shape),
                "metrics": values,
            },
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
                disable=not sys.stderr.isatty(),
                mininterval=5.0,
                dynamic_ncols=True,
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
