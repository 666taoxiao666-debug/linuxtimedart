from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
    overlay_forecast_weights,
    transfer_weights,
    show_series,
    show_matrix,
)
from utils.augmentations import masked_data
from utils.forecast_report import build_forecast_report, save_training_history
from utils.metrics import forecast_metrics
from utils.forecast_losses import ForecastLoss
from utils.run_tags import forecast_result_tag
from utils.sdwpf_logging import tensorboard_log_directory
from utils.experiment_audit import checkpoint_info, model_runtime_summary, write_run_manifest
from utils.utility_wiki import (
    configure_frozen_utility_mode,
    selected_intervention_metrics,
    utility_candidate_specialization_loss,
    utility_decision_loss,
    utility_supervision_loss,
)
from utils.regime_labels import (
    calibrate_regime_thresholds_from_dataset,
    summarize_regime_confusion,
    update_regime_confusion,
)
from utils.wind_regime_wiki import (
    audit_event_factor_labels_from_dataset,
    audit_scene_wiki_labels_from_dataset,
    compute_event_factor_rule_logits,
    compute_scene_wiki_rule_logits,
    summarize_event_factor_confusion,
    update_event_factor_confusion,
)
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torch import optim
import os
import json
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


class _LifecycleWeightedBCE(nn.Module):
    """Factor-wise BCE masked by frozen train-only lifecycle reliability."""

    def __init__(self, pos_weight, factor_weight):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.register_buffer("factor_weight", factor_weight)

    def forward(self, logits, targets):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        denominator = self.factor_weight.sum() * max(1, int(logits.shape[0]))
        if float(denominator.detach().cpu()) <= 0.0:
            return logits.sum() * 0.0
        return (loss * self.factor_weight.view(1, -1)).sum() / denominator


def _dynamic_channel_scale_sample(mixer, context):
    """Return per-sample channel scales, excluding static-only mixers."""

    if getattr(mixer, "context_gate", None) is None:
        return None
    scale = mixer.effective_channel_scale(context).detach().float().cpu().numpy()
    if scale.ndim != 2 or scale.shape[1] != mixer.num_features:
        raise ValueError(
            "Dynamic channel scale must have shape [batch, num_features], "
            f"got {scale.shape}"
        )
    return scale


class Exp_TimeDART(Exp_Basic):
    def __init__(self, args):
        super(Exp_TimeDART, self).__init__(args)
        log_dir = tensorboard_log_directory(
            model=args.model,
            data=args.data,
            task=args.task_name,
            run_id=getattr(args, "run_id", ""),
        )
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

    def _calibrate_regime_labels(self, train_data):
        """Fit regime thresholds from training histories and update model buffers."""

        core_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if not getattr(core_model, "use_soft_prompt", False):
            return None
        prompt_router = getattr(core_model, "prompt_router", "trend")
        scene_audit = None
        if prompt_router in (
            "scene_wiki",
            "hybrid_wiki",
            "compositional_wiki",
        ):
            if not hasattr(train_data, "scaler"):
                raise TypeError("Scene Wiki requires the SDWPF train-fitted scaler")
            core_model.set_scene_scaler(
                train_data.scaler.mean_,
                train_data.scaler.scale_,
            )
            if prompt_router == "compositional_wiki":
                scene_audit = audit_event_factor_labels_from_dataset(
                    train_data,
                    self.args.feature_columns,
                    rated_power=self.args.rated_power,
                    max_samples=self.args.regime_calibration_samples,
                    rule_kwargs=core_model.scene_wiki_rule_kwargs,
                    factor_ids=core_model.scene_wiki_scene_ids,
                )
            else:
                scene_audit = audit_scene_wiki_labels_from_dataset(
                    train_data,
                    self.args.feature_columns,
                    rated_power=self.args.rated_power,
                    max_samples=self.args.regime_calibration_samples,
                    rule_kwargs=core_model.scene_wiki_rule_kwargs,
                    scene_ids=core_model.scene_wiki_scene_ids,
                )
            scene_audit.update(
                {
                    "wiki_config_sha256": self.args.scene_wiki_config_sha256,
                    "wiki_bundle_sha256": self.args.scene_wiki_bundle_sha256,
                    "wiki_encoder": self.args.scene_wiki_encoder_name,
                }
            )
            self.args.regime_calibration_source = "train"
            if prompt_router == "compositional_wiki":
                positive_counts = scene_audit["positive_counts"]
                reliability = list(
                    getattr(self.args, "scene_wiki_factor_reliability", []) or []
                )
                if len(reliability) != len(positive_counts):
                    raise ValueError("Event Wiki lifecycle reliability/count mismatch")
                missing = [
                    factor_id
                    for factor_id, count, weight in zip(
                        scene_audit["factor_ids"], positive_counts, reliability
                    )
                    if weight > 0.0 and count <= 0
                ]
                if missing:
                    raise ValueError(
                        "Training split has no positive support for event factors: "
                        + ", ".join(missing)
                    )
                self.args.event_factor_positive_counts = positive_counts
                self.args.event_factor_negative_counts = scene_audit["negative_counts"]
                print(
                    "[AUDIT] EVENT_FACTOR_CALIBRATION="
                    f"source=train samples={scene_audit['sample_count']} "
                    f"factors={';'.join(scene_audit['factor_ids'])} "
                    f"positive_counts={';'.join(str(value) for value in positive_counts)} "
                    f"lifecycle_weights={';'.join(f'{value:.6f}' for value in reliability)} "
                    f"no_intervention={scene_audit['no_intervention_count']} "
                    f"mean_active={scene_audit['mean_active_factors']:.6f} "
                    f"config_sha256={scene_audit['wiki_config_sha256'][:12]} "
                    f"bundle_sha256={scene_audit['wiki_bundle_sha256'][:12]}"
                )
            else:
                self.args.scene_wiki_calibration_counts = scene_audit["class_counts"]
                print(
                    "[AUDIT] SCENE_WIKI_CALIBRATION="
                    f"source=train samples={scene_audit['sample_count']} "
                    f"scenes={';'.join(scene_audit['scene_ids'])} "
                    f"counts={';'.join(str(value) for value in scene_audit['class_counts'])} "
                    f"config_sha256={scene_audit['wiki_config_sha256'][:12]} "
                    f"bundle_sha256={scene_audit['wiki_bundle_sha256'][:12]}"
                )
            if prompt_router == "scene_wiki":
                self.args.regime_calibration_counts = scene_audit["class_counts"]
                core_model.regime_calibration = scene_audit
                return scene_audit
        if getattr(self.args, "regime_label_method", "legacy_volatility") != "trend_quantile":
            print("[WARNING] Using legacy fixed regime thresholds for reproduction only.")
            return None
        audit = calibrate_regime_thresholds_from_dataset(
            train_data,
            target_index=self.args.regime_target_index,
            quantile=self.args.regime_calibration_quantile,
            max_samples=self.args.regime_calibration_samples,
            min_class_fraction=self.args.regime_min_class_fraction,
        )
        with torch.no_grad():
            core_model.regime_down_thresh.fill_(audit["down_thresh"])
            core_model.regime_up_thresh.fill_(audit["up_thresh"])
        self.args.regime_down_thresh = audit["down_thresh"]
        self.args.regime_up_thresh = audit["up_thresh"]
        self.args.regime_calibration_source = audit["source_split"]
        self.args.regime_calibration_counts = audit["class_counts"]
        if prompt_router == "hybrid_wiki":
            combined_audit = {
                "method": "causal_trend_wiki_residual",
                "source_split": "train",
                "trend": audit,
                "exception_wiki": scene_audit,
            }
        elif prompt_router == "compositional_wiki":
            combined_audit = {
                "method": "causal_compositional_event_wiki_residual",
                "source_split": "train",
                "trend": audit,
                "event_factor_wiki": scene_audit,
            }
        else:
            combined_audit = audit
        core_model.regime_calibration = combined_audit
        print(
            "[AUDIT] REGIME_CALIBRATION="
            f"method={audit['method']} source={audit['source_split']} "
            f"samples={audit['sample_count']} quantile={audit['quantile']:.6f} "
            f"down={audit['down_thresh']:.8f} up={audit['up_thresh']:.8f} "
            f"counts={';'.join(str(value) for value in audit['class_counts'])}"
        )
        return combined_audit

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

        if getattr(self.args, "overlay_checkpoint", None):
            print(f"Overlaying validated forecast ckpt: {self.args.overlay_checkpoint}")
            model = overlay_forecast_weights(
                self.args.overlay_checkpoint,
                model,
                device=self.device,
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
        base_lr = float(self.args.learning_rate)
        new_module_lr = float(getattr(self.args, "new_module_learning_rate", 0.0))
        utility_lr = float(getattr(self.args, "utility_learning_rate", new_module_lr))
        use_utility_group = bool(getattr(self.args, "utility_wiki", False))
        utility_tokens = (
            "utility_gate.",
            "utility_event_adapter.",
            "utility_composition_adapter.",
        )
        if bool(getattr(self.args, "freeze_non_utility", False)):
            if not use_utility_group:
                raise RuntimeError("freeze_non_utility requires utility_wiki")
            utility_parameters = []
            frozen_count = 0
            for name, parameter in self.model.named_parameters():
                clean_name = name.removeprefix("module.")
                if any(token in clean_name for token in utility_tokens):
                    parameter.requires_grad_(True)
                    utility_parameters.append(parameter)
                else:
                    parameter.requires_grad_(False)
                    frozen_count += parameter.numel()
            if not utility_parameters:
                raise RuntimeError("No Utility-Wiki parameters were found to train")
            model_optim = optim.AdamW(
                [
                    {
                        "params": utility_parameters,
                        "lr": utility_lr,
                        "target_lr": utility_lr,
                        "group_name": "utility_modules",
                    }
                ],
                lr=utility_lr,
                weight_decay=self.args.weight_decay,
            )
            print(
                "Optimizer groups: "
                f"frozen_base={frozen_count:,} params; "
                f"utility={sum(p.numel() for p in utility_parameters):,} params "
                f"@ {utility_lr:g}"
            )
            return model_optim
        use_groups = (
            self.args.task_name == "finetune"
            and self.args.downstream_task == "forecast"
            and (
                (new_module_lr > 0 and not np.isclose(new_module_lr, base_lr))
                or (use_utility_group and not np.isclose(utility_lr, base_lr))
            )
        )
        if not use_groups:
            model_optim = optim.AdamW(
                [{
                    "params": self.model.parameters(),
                    "lr": base_lr,
                    "target_lr": base_lr,
                    "group_name": "all",
                }],
                lr=base_lr,
                weight_decay=self.args.weight_decay,
            )
            return model_optim

        new_tokens = (
            "head.",
            "channel_mixer.",
            "residual_gate_logit",
        )
        transferred = []
        newly_initialized = []
        utility_parameters = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            clean_name = name.removeprefix("module.")
            if use_utility_group and any(
                token in clean_name for token in utility_tokens
            ):
                destination = utility_parameters
            elif any(token in clean_name for token in new_tokens):
                destination = newly_initialized
            else:
                destination = transferred
            destination.append(parameter)
        if not newly_initialized or not transferred:
            raise RuntimeError(
                "Differential fine-tuning LR requested, but optimizer parameter "
                "groups could not be separated"
            )
        parameter_groups = [
                {
                    "params": transferred,
                    "lr": base_lr,
                    "target_lr": base_lr,
                    "group_name": "transferred_backbone",
                },
                {
                    "params": newly_initialized,
                    "lr": new_module_lr,
                    "target_lr": new_module_lr,
                    "group_name": "new_forecast_modules",
                },
            ]
        if use_utility_group:
            if not utility_parameters:
                raise RuntimeError(
                    "utility_wiki is enabled, but no utility_gate parameters were found"
                )
            parameter_groups.append(
                {
                    "params": utility_parameters,
                    "lr": utility_lr,
                    "target_lr": utility_lr,
                    "group_name": "utility_modules",
                }
            )
        model_optim = optim.AdamW(
            parameter_groups,
            weight_decay=self.args.weight_decay,
        )
        group_summary = [
            f"backbone={sum(p.numel() for p in transferred):,} params @ {base_lr:g}",
            f"forecast={sum(p.numel() for p in newly_initialized):,} params @ {new_module_lr:g}",
        ]
        if utility_parameters:
            group_summary.append(
                f"utility={sum(p.numel() for p in utility_parameters):,} params @ {utility_lr:g}"
            )
        print("Optimizer groups: " + "; ".join(group_summary))
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

    def _configure_utility_training_phase(self, epoch):
        """Use fixed candidate outcomes when learning the intervention gate."""

        warmup_epochs = int(
            getattr(self.args, "utility_adapter_warmup_epochs", 0)
        )
        staged = (
            bool(getattr(self.args, "utility_wiki", False))
            and bool(getattr(self.args, "freeze_non_utility", False))
            and warmup_epochs > 0
        )
        if not staged:
            return "joint"

        adapter_warmup = int(epoch) < warmup_epochs
        gate_count = 0
        adapter_count = 0
        for name, parameter in self.model.named_parameters():
            clean_name = name.removeprefix("module.")
            if "utility_gate." in clean_name:
                parameter.requires_grad_(not adapter_warmup)
                gate_count += parameter.numel()
            elif (
                "utility_event_adapter." in clean_name
                or "utility_composition_adapter." in clean_name
            ):
                parameter.requires_grad_(adapter_warmup)
                adapter_count += parameter.numel()
        if gate_count == 0 or adapter_count == 0:
            raise RuntimeError(
                "Staged Utility-Wiki training requires both gate and adapter parameters"
            )
        phase = "adapter_warmup" if adapter_warmup else "utility_gate"
        print(
            "[UTILITY] Training phase: "
            f"{phase} (gate={gate_count:,}, adapters={adapter_count:,} params) "
            f"base_mode=eval adapter_mode={getattr(self.args, 'utility_adapter_mode', 'legacy')}"
        )
        if getattr(self.args, "utility_adapter_mode", "legacy") == "hierarchical_evidence":
            print(
                "[UTILITY] HierarchicalEvidence: trend_experts=3 "
                "composition=event_plus_increment balance=train_history_sqrt_cap3 "
                f"event_cap={self.args.utility_event_max_scale:g} "
                f"increment_cap={self.args.utility_composition_max_scale:g} "
                "cap_units=input_window_target_std gate=cost_sensitive_candidate_conditioned"
            )
        return phase

    def pretrain(self):
        train_data, train_loader = (
            self._get_data(flag="train")
        )
        vali_data, vali_loader = (
            self._get_data(flag="val")
        )
        regime_calibration = self._calibrate_regime_labels(train_data)

        path = self.args.pretrain_run_dir
        os.makedirs(
            path,
            exist_ok=True,
        )
        if regime_calibration is not None:
            with open(
                os.path.join(path, "regime_calibration.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(regime_calibration, handle, ensure_ascii=False, indent=2)

        write_run_manifest(
            path,
            self.args,
            "pretrain",
            model=self.model,
            datasets={"train": train_data, "val": vali_data},
            checkpoints={
                "pretrained_source": checkpoint_info(self.args.load_checkpoints),
                "overlay_source": checkpoint_info(
                    getattr(self.args, "overlay_checkpoint", None)
                ),
            },
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
                "Val Regime Acc/Macro-F1: {:.3f}/{:.3f}, "
                "Val Regime Counts: {}, Recall: {}, Confusion: {}, "
                "GradNorm: {:.3f}".format(
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
                    validation["regime_macro_f1"],
                    validation["regime_counts"],
                    validation["regime_recall"],
                    validation["regime_confusion"],
                    train_metrics["grad_norm"],
                )
            )
            prompt_router = getattr(self.args, "prompt_router", "trend")
            if prompt_router == "hybrid_wiki":
                print(
                    "Legacy Wiki Train/Val CE: {:.4f}/{:.4f}, "
                    "Val Acc/Macro-F1: {:.3f}/{:.3f}".format(
                        train_metrics["scene_ce_loss"],
                        validation["scene_ce_loss"],
                        validation.get("scene_accuracy", 0.0),
                        validation.get("scene_macro_f1", 0.0),
                    )
                )
            elif prompt_router == "compositional_wiki":
                print(
                    "Event Wiki Train/Val BCE: {:.4f}/{:.4f}, "
                    "Val Hamming/Macro-F1/Micro-F1/Exact: "
                    "{:.3f}/{:.3f}/{:.3f}/{:.3f}, "
                    "Null Recall/False Intervention: {:.3f}/{:.3f}".format(
                        train_metrics["event_bce_loss"],
                        validation["event_bce_loss"],
                        validation.get("event_hamming_accuracy", 0.0),
                        validation.get("event_macro_f1", 0.0),
                        validation.get("event_micro_f1", 0.0),
                        validation.get("event_exact_match", 0.0),
                        validation.get("event_null_recall", 0.0),
                        validation.get("event_false_intervention_on_null", 0.0),
                    )
                )

            loss_scalar_dict = {
                "train_total": train_metrics["total_loss"],
                "train_diff": train_metrics["diff_loss"],
                "train_ce": train_metrics["ce_loss"],
                "train_scene_ce": train_metrics["scene_ce_loss"],
                "train_event_bce": train_metrics["event_bce_loss"],
                "vali_total": validation["total_loss"],
                "vali_diff": validation["diff_loss"],
                "vali_ce": validation["ce_loss"],
                "vali_scene_ce": validation["scene_ce_loss"],
                "vali_event_bce": validation["event_bce_loss"],
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
                    "train_regime_macro_f1": float(train_metrics["regime_macro_f1"]),
                    "train_grad_norm": float(train_metrics["grad_norm"]),
                    "train_regime_counts": train_metrics["regime_counts"],
                    "train_regime_pred_counts": train_metrics["regime_pred_counts"],
                    "train_regime_recall": train_metrics["regime_recall"],
                    "train_regime_f1": train_metrics["regime_f1"],
                    "train_regime_confusion": train_metrics["regime_confusion"],
                    "train_scene_accuracy": train_metrics.get("scene_accuracy"),
                    "train_scene_macro_f1": train_metrics.get("scene_macro_f1"),
                    "train_scene_counts": train_metrics.get("scene_counts"),
                    "train_scene_confusion": train_metrics.get("scene_confusion"),
                    "train_event_bce_loss": float(train_metrics["event_bce_loss"]),
                    "train_event_hamming_accuracy": train_metrics.get(
                        "event_hamming_accuracy"
                    ),
                    "train_event_macro_f1": train_metrics.get("event_macro_f1"),
                    "train_event_micro_f1": train_metrics.get("event_micro_f1"),
                    "train_event_exact_match": train_metrics.get("event_exact_match"),
                    "train_event_null_recall": train_metrics.get("event_null_recall"),
                    "train_event_false_intervention_on_null": train_metrics.get(
                        "event_false_intervention_on_null"
                    ),
                    "train_event_positive_counts": train_metrics.get(
                        "event_positive_counts"
                    ),
                    "train_event_pred_positive_counts": train_metrics.get(
                        "event_pred_positive_counts"
                    ),
                    "train_event_precision": train_metrics.get("event_precision"),
                    "train_event_recall": train_metrics.get("event_recall"),
                    "train_event_f1": train_metrics.get("event_f1"),
                    "train_event_confusion": train_metrics.get(
                        "event_confusion_tn_fp_fn_tp"
                    ),
                    "val_loss": float(validation["total_loss"]),
                    "val_diff_loss": float(validation["diff_loss"]),
                    "val_ce_loss": float(validation["ce_loss"]),
                    "val_regime_accuracy": float(validation["regime_accuracy"]),
                    "val_regime_macro_f1": float(validation["regime_macro_f1"]),
                    "val_regime_counts": validation["regime_counts"],
                    "val_regime_pred_counts": validation["regime_pred_counts"],
                    "val_regime_recall": validation["regime_recall"],
                    "val_regime_f1": validation["regime_f1"],
                    "val_regime_confusion": validation["regime_confusion"],
                    "val_scene_accuracy": validation.get("scene_accuracy"),
                    "val_scene_macro_f1": validation.get("scene_macro_f1"),
                    "val_scene_counts": validation.get("scene_counts"),
                    "val_scene_confusion": validation.get("scene_confusion"),
                    "val_event_bce_loss": float(validation["event_bce_loss"]),
                    "val_event_hamming_accuracy": validation.get(
                        "event_hamming_accuracy"
                    ),
                    "val_event_macro_f1": validation.get("event_macro_f1"),
                    "val_event_micro_f1": validation.get("event_micro_f1"),
                    "val_event_exact_match": validation.get("event_exact_match"),
                    "val_event_null_recall": validation.get("event_null_recall"),
                    "val_event_false_intervention_on_null": validation.get(
                        "event_false_intervention_on_null"
                    ),
                    "val_event_positive_counts": validation.get(
                        "event_positive_counts"
                    ),
                    "val_event_pred_positive_counts": validation.get(
                        "event_pred_positive_counts"
                    ),
                    "val_event_precision": validation.get("event_precision"),
                    "val_event_recall": validation.get("event_recall"),
                    "val_event_f1": validation.get("event_f1"),
                    "val_event_confusion": validation.get(
                        "event_confusion_tn_fp_fn_tp"
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
            "scene_wiki_router.",
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

            if clean_name.startswith(transfer_prefixes) or clean_name in {
                "regime_down_thresh",
                "regime_up_thresh",
                "scene_scaler_mean",
                "scene_scaler_scale",
            }:
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
            "prompt_router": getattr(self.args, "prompt_router", "trend"),
            "scene_wiki_config_sha256": getattr(
                self.args, "scene_wiki_config_sha256", None
            ),
            "scene_wiki_bundle_sha256": getattr(
                self.args, "scene_wiki_bundle_sha256", None
            ),
            "scene_wiki_scene_ids": list(
                getattr(self.args, "scene_wiki_scene_ids", []) or []
            ),
            "scene_wiki_activation_threshold": getattr(
                self.args, "scene_wiki_activation_threshold", None
            ),
            "scene_wiki_confidence_power": getattr(
                self.args, "scene_wiki_confidence_power", None
            ),
            "scene_wiki_top_k": getattr(self.args, "scene_wiki_top_k", None),
            "scene_wiki_temperature": getattr(
                self.args, "scene_wiki_temperature", None
            ),
            "scene_wiki_rule_weight": getattr(
                self.args, "scene_wiki_rule_weight", None
            ),
            "scene_wiki_rule_kwargs": dict(
                getattr(
                    self.model.module if isinstance(self.model, nn.DataParallel) else self.model,
                    "scene_wiki_rule_kwargs",
                    {},
                )
            ),
            "scene_wiki_factor_reliability": list(
                getattr(self.args, "scene_wiki_factor_reliability", []) or []
            ),
            "lambda_event_bce": getattr(self.args, "lambda_event_bce", None),
            "event_factor_positive_counts": getattr(
                self.args, "event_factor_positive_counts", None
            ),
            "event_factor_negative_counts": getattr(
                self.args, "event_factor_negative_counts", None
            ),
            "event_factor_pos_weight": getattr(
                self.args, "event_factor_pos_weight", None
            ),
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
                    None,
                    None,
                )

            if len(output) == 2:
                return (
                    output[0],
                    output[1],
                    None,
                    None,
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
                output.get("event_logits", output.get("scene_logits")),
                output.get("event_targets", output.get("scene_labels")),
            )

        return output, None, None, None, None

    def _wiki_auxiliary_criterion(self):
        """Build train-only balancing for legacy scenes or event factors."""

        if getattr(self.args, "prompt_router", "trend") == "compositional_wiki":
            positives = getattr(self.args, "event_factor_positive_counts", None)
            negatives = getattr(self.args, "event_factor_negative_counts", None)
            if not positives or not negatives:
                raise RuntimeError(
                    "Event-factor BCE requires train-only positive/negative counts"
                )
            positive_values = torch.as_tensor(
                positives, dtype=torch.float32, device=self.device
            )
            negative_values = torch.as_tensor(
                negatives, dtype=torch.float32, device=self.device
            )
            factor_weight = torch.as_tensor(
                getattr(self.args, "scene_wiki_factor_reliability", None),
                dtype=torch.float32,
                device=self.device,
            )
            if (
                positive_values.shape != factor_weight.shape
                or torch.any((positive_values <= 0) & (factor_weight > 0))
                or torch.any(negative_values < 0)
                or not torch.isfinite(positive_values).all()
                or not torch.isfinite(negative_values).all()
                or not torch.isfinite(factor_weight).all()
                or torch.any(factor_weight < 0)
            ):
                raise ValueError("Event-factor calibration counts are invalid")
            pos_weight = torch.sqrt(
                negative_values / positive_values.clamp_min(1.0)
            ).clamp(0.25, 4.0)
            self.args.event_factor_pos_weight = [
                float(value) for value in pos_weight.detach().cpu().tolist()
            ]
            return _LifecycleWeightedBCE(pos_weight, factor_weight)

        counts = getattr(self.args, "scene_wiki_calibration_counts", None)
        if not counts:
            return nn.CrossEntropyLoss()
        values = torch.as_tensor(counts, dtype=torch.float32, device=self.device)
        if torch.any(values <= 0) or not torch.isfinite(values).all():
            return nn.CrossEntropyLoss()
        weights = torch.sqrt(values.sum() / values).clamp(0.25, 4.0)
        weights = weights / weights.mean()
        return nn.CrossEntropyLoss(weight=weights)

    def pretrain_one_epoch(
        self,
        train_loader,
        model_optim,
        model_scheduler,
    ):
        total_losses = []
        diff_losses = []
        ce_losses = []
        scene_ce_losses = []
        event_bce_losses = []
        grad_norms = []
        regime_confusion = np.zeros(
            (int(self.args.num_modes), int(self.args.num_modes)), dtype=np.int64
        )
        scene_confusion = None
        event_confusion = None
        event_sample_stats = None
        if getattr(self.args, "prompt_router", "trend") == "hybrid_wiki":
            scene_modes = int(self.args.scene_wiki_num_modes)
            scene_confusion = np.zeros((scene_modes, scene_modes), dtype=np.int64)
        elif getattr(self.args, "prompt_router", "trend") == "compositional_wiki":
            event_modes = int(self.args.scene_wiki_num_modes)
            event_confusion = np.zeros((event_modes, 4), dtype=np.int64)
            event_sample_stats = np.zeros(6, dtype=np.int64)

        model_criterion = (
            self._select_criterion()
        )
        ce_criterion = (
            nn.CrossEntropyLoss()
        )
        wiki_auxiliary_criterion = self._wiki_auxiliary_criterion()

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
                    scene_logits,
                    scene_labels,
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

                if scene_logits is not None and scene_labels is not None:
                    wiki_auxiliary_loss = wiki_auxiliary_criterion(
                        scene_logits, scene_labels
                    )
                    auxiliary_weight = (
                        self.args.lambda_event_bce
                        if event_confusion is not None
                        else self.args.lambda_scene_ce
                    )
                    total_loss = total_loss + auxiliary_weight * wiki_auxiliary_loss
                else:
                    wiki_auxiliary_loss = torch.zeros((), device=diff_loss.device)
                scene_ce_loss = (
                    wiki_auxiliary_loss
                    if scene_confusion is not None
                    else torch.zeros((), device=diff_loss.device)
                )
                event_bce_loss = (
                    wiki_auxiliary_loss
                    if event_confusion is not None
                    else torch.zeros((), device=diff_loss.device)
                )

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
            scene_ce_losses.append(scene_ce_loss.item())
            event_bce_losses.append(event_bce_loss.item())
            if logits is not None and pseudo_labels is not None:
                predictions = logits.detach().argmax(dim=-1)
                labels = pseudo_labels.detach().reshape(-1)
                update_regime_confusion(regime_confusion, predictions, labels)
            if scene_confusion is not None and scene_logits is not None and scene_labels is not None:
                scene_predictions = scene_logits.detach().argmax(dim=-1)
                update_regime_confusion(
                    scene_confusion,
                    scene_predictions,
                    scene_labels.detach().reshape(-1),
                )
            if event_confusion is not None and scene_logits is not None and scene_labels is not None:
                event_sample_stats += update_event_factor_confusion(
                    event_confusion,
                    scene_logits,
                    scene_labels,
                )

        model_scheduler.step()
        regime_metrics = summarize_regime_confusion(regime_confusion)
        scene_metrics = (
            {
                f"scene_{key.removeprefix('regime_')}": value
                for key, value in summarize_regime_confusion(scene_confusion).items()
            }
            if scene_confusion is not None
            else {}
        )
        event_metrics = (
            summarize_event_factor_confusion(event_confusion, event_sample_stats)
            if event_confusion is not None
            else {}
        )
        return {
            "total_loss": float(np.mean(total_losses)),
            "diff_loss": float(np.mean(diff_losses)),
            "ce_loss": float(np.mean(ce_losses)),
            "scene_ce_loss": float(np.mean(scene_ce_losses)),
            "event_bce_loss": float(np.mean(event_bce_losses)),
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            **regime_metrics,
            **scene_metrics,
            **event_metrics,
        }

    def valid_one_epoch(
        self,
        vali_loader,
    ):
        total_losses = []
        diff_losses = []
        ce_losses = []
        scene_ce_losses = []
        event_bce_losses = []
        regime_confusion = np.zeros(
            (int(self.args.num_modes), int(self.args.num_modes)), dtype=np.int64
        )
        scene_confusion = None
        event_confusion = None
        event_sample_stats = None
        if getattr(self.args, "prompt_router", "trend") == "hybrid_wiki":
            scene_modes = int(self.args.scene_wiki_num_modes)
            scene_confusion = np.zeros((scene_modes, scene_modes), dtype=np.int64)
        elif getattr(self.args, "prompt_router", "trend") == "compositional_wiki":
            event_modes = int(self.args.scene_wiki_num_modes)
            event_confusion = np.zeros((event_modes, 4), dtype=np.int64)
            event_sample_stats = np.zeros(6, dtype=np.int64)

        model_criterion = (
            self._select_criterion()
        )
        ce_criterion = (
            nn.CrossEntropyLoss()
        )
        wiki_auxiliary_criterion = self._wiki_auxiliary_criterion()

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
                        scene_logits,
                        scene_labels,
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

                    if scene_logits is not None and scene_labels is not None:
                        wiki_auxiliary_loss = wiki_auxiliary_criterion(
                            scene_logits, scene_labels
                        )
                        auxiliary_weight = (
                            self.args.lambda_event_bce
                            if event_confusion is not None
                            else self.args.lambda_scene_ce
                        )
                        total_loss = total_loss + auxiliary_weight * wiki_auxiliary_loss
                    else:
                        wiki_auxiliary_loss = torch.zeros((), device=diff_loss.device)
                    scene_ce_loss = (
                        wiki_auxiliary_loss
                        if scene_confusion is not None
                        else torch.zeros((), device=diff_loss.device)
                    )
                    event_bce_loss = (
                        wiki_auxiliary_loss
                        if event_confusion is not None
                        else torch.zeros((), device=diff_loss.device)
                    )

                total_losses.append(total_loss.item())
                diff_losses.append(diff_loss.item())
                ce_losses.append(ce_loss.item())
                scene_ce_losses.append(scene_ce_loss.item())
                event_bce_losses.append(event_bce_loss.item())
                if logits is not None and pseudo_labels is not None:
                    predictions = logits.detach().argmax(dim=-1)
                    labels = pseudo_labels.detach().reshape(-1)
                    update_regime_confusion(regime_confusion, predictions, labels)
                if scene_confusion is not None and scene_logits is not None and scene_labels is not None:
                    scene_predictions = scene_logits.detach().argmax(dim=-1)
                    update_regime_confusion(
                        scene_confusion,
                        scene_predictions,
                        scene_labels.detach().reshape(-1),
                    )
                if event_confusion is not None and scene_logits is not None and scene_labels is not None:
                    event_sample_stats += update_event_factor_confusion(
                        event_confusion,
                        scene_logits,
                        scene_labels,
                    )

        regime_metrics = summarize_regime_confusion(regime_confusion)
        scene_metrics = (
            {
                f"scene_{key.removeprefix('regime_')}": value
                for key, value in summarize_regime_confusion(scene_confusion).items()
            }
            if scene_confusion is not None
            else {}
        )
        event_metrics = (
            summarize_event_factor_confusion(event_confusion, event_sample_stats)
            if event_confusion is not None
            else {}
        )
        return {
            "total_loss": float(np.mean(total_losses)),
            "diff_loss": float(np.mean(diff_losses)),
            "ce_loss": float(np.mean(ce_losses)),
            "scene_ce_loss": float(np.mean(scene_ce_losses)),
            "event_bce_loss": float(np.mean(event_bce_losses)),
            **regime_metrics,
            **scene_metrics,
            **event_metrics,
        }

    def train(self, setting):
        train_data, train_loader = (
            self._get_data(flag="train")
        )
        vali_data, vali_loader = (
            self._get_data(flag="val")
        )
        core_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if getattr(core_model, "prompt_router", "trend") in (
            "scene_wiki",
            "hybrid_wiki",
            "compositional_wiki",
        ):
            # A random-init ablation has no pretraining checkpoint carrying the
            # scaler buffers. Reinstall the scaler from this run's training split
            # before the epoch-0 validation; never inspect validation/test data.
            self._calibrate_regime_labels(train_data)

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
            checkpoints={
                "pretrained_source": checkpoint_info(self.args.load_checkpoints),
                "overlay_source": checkpoint_info(
                    getattr(self.args, "overlay_checkpoint", None)
                ),
            },
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
                max_lr=[
                    float(group.get("target_lr", self.args.learning_rate))
                    for group in model_optim.param_groups
                ],
            )
        )

        history = []

        if getattr(self.args, "validate_before_training", False):
            initial_validation = self.valid(
                vali_loader,
                model_criteria,
            )
            initial_selection_value = initial_validation[
                self.args.early_stop_metric
            ]
            validation_unit = "kW" if self.args.data == "SDWPF" else "original"
            metric_suffix = "kw" if self.args.data == "SDWPF" else "original"
            initial_summary = (
                "Epoch: 0, Steps: 0, Time: 0.00s | "
                "Train Loss: n/a "
                f"Vali Loss: {initial_validation['loss']:.7f} "
                f"Vali MSE: {initial_validation['mse']:.7f} "
                f"Vali MAE: {initial_validation['mae']:.7f} "
                f"Val MAE({validation_unit}): "
                f"{initial_validation['original_mae']:.3f} "
                f"Persist MAE({validation_unit}): "
                f"{initial_validation['original_persistence_mae']:.3f} "
                f"MAE Skill: "
                f"{initial_validation['original_mae_skill_vs_persistence_pct']:+.2f}% "
                f"RMSE Skill: "
                f"{initial_validation['original_rmse_skill_vs_persistence_pct']:+.2f}% "
                f"Gate: {initial_validation['diagnostics'].get('residual_gate', 0.0):.5f} "
                "GradNorm: 0.000 "
                "LR(backbone/new): 0/0 "
                f"Select({self.args.early_stop_metric}): "
                f"{initial_selection_value:.7f}"
            )
            initial_diagnostics = initial_validation["diagnostics"]
            if "event_wiki_zero_intervention_fraction" in initial_diagnostics:
                initial_summary += (
                    " EventWiki(zero/active): "
                    f"{initial_diagnostics['event_wiki_zero_intervention_fraction']:.3f}/"
                    f"{initial_diagnostics['event_wiki_mean_active_factors']:.3f}"
                )
            if "val_available_mae_kw" in initial_diagnostics:
                initial_summary += (
                    " Available MAE(kW): "
                    f"{initial_diagnostics['val_available_mae_kw']:.3f} "
                    "Available MAE Skill: "
                    f"{initial_diagnostics['val_available_mae_skill_pct']:+.2f}%"
                )
            print(initial_summary)
            log_path = path + "/log.txt"
            with open(log_path, "a") as log_file:
                log_file.write(initial_summary + "\n")

            history.append(
                {
                    "epoch": 0,
                    "utility_phase": "validated_trend_baseline",
                    "selection_eligible": True,
                    "train_loss": np.nan,
                    "train_mse": np.nan,
                    "train_mae": np.nan,
                    "train_grad_norm": 0.0,
                    "train_utility_loss": np.nan,
                    "train_utility_decision_loss": np.nan,
                    "train_utility_candidate_loss": np.nan,
                    "train_utility_ranking_loss": np.nan,
                    "val_loss": float(initial_validation["loss"]),
                    "val_mse": float(initial_validation["mse"]),
                    "val_mae": float(initial_validation["mae"]),
                    "val_persistence_mse": float(
                        initial_validation["persistence_mse"]
                    ),
                    "val_persistence_mae": float(
                        initial_validation["persistence_mae"]
                    ),
                    "val_mae_skill_vs_persistence_pct": float(
                        initial_validation["mae_skill_vs_persistence_pct"]
                    ),
                    "val_rmse_skill_vs_persistence_pct": float(
                        initial_validation["rmse_skill_vs_persistence_pct"]
                    ),
                    "val_mae_kw": float(initial_validation["original_mae"]),
                    "val_rmse_kw": float(initial_validation["original_rmse"]),
                    "val_persistence_mae_kw": float(
                        initial_validation["original_persistence_mae"]
                    ),
                    "val_persistence_rmse_kw": float(
                        initial_validation["original_persistence_rmse"]
                    ),
                    "val_mae_skill_original_pct": float(
                        initial_validation[
                            "original_mae_skill_vs_persistence_pct"
                        ]
                    ),
                    "val_rmse_skill_original_pct": float(
                        initial_validation[
                            "original_rmse_skill_vs_persistence_pct"
                        ]
                    ),
                    "selection_value": float(initial_selection_value),
                    "learning_rate": 0.0,
                    "new_module_learning_rate": 0.0,
                    "utility_learning_rate": 0.0,
                    **initial_validation["diagnostics"],
                }
            )
            save_training_history(
                history,
                path,
            )
            self.writer.add_scalar(
                "/finetune_loss/vali_loss",
                initial_validation["loss"],
                0,
            )
            early_stopping(
                initial_selection_value,
                self.model,
                path=path,
            )

        for epoch in range(
            self.args.train_epochs
        ):
            utility_phase = self._configure_utility_training_phase(epoch)
            iter_count = 0
            train_loss = []
            train_squared_error = 0.0
            train_absolute_error = 0.0
            train_point_count = 0
            grad_norms = []
            train_utility_losses = []
            train_utility_decision_losses = []
            train_utility_candidate_losses = []
            train_utility_ranking_losses = []

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
            if getattr(self.args, "freeze_non_utility", False):
                configure_frozen_utility_mode(self.model)
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

                    forecast_loss = model_criteria(
                        pred_x,
                        batch_y,
                    )
                    utility_loss = forecast_loss.new_zeros(())
                    utility_decision = forecast_loss.new_zeros(())
                    utility_candidate_loss = forecast_loss.new_zeros(())
                    utility_ranking_loss = forecast_loss.new_zeros(())
                    if getattr(core_model, "utility_wiki", False):
                        utility_loss, _ = utility_supervision_loss(
                            core_model._last_utility_aux,
                            batch_y,
                            eps=self.args.utility_target_eps,
                        )
                        utility_decision, _ = utility_decision_loss(
                            core_model._last_utility_aux,
                            batch_y,
                            eps=self.args.utility_target_eps,
                            min_gain=self.args.utility_min_gain,
                            temperature=self.args.utility_gate_temperature,
                        )
                        (
                            utility_candidate_loss,
                            utility_ranking_loss,
                            _,
                        ) = utility_candidate_specialization_loss(
                            core_model._last_utility_aux,
                            batch_y,
                            margin=self.args.utility_ranking_margin,
                        )
                    if utility_phase == "adapter_warmup":
                        # First make both physically available candidates useful.
                        # The trend base and utility gate are frozen, so no moving
                        # decision target or baseline degradation is possible.
                        loss = (
                            self.args.utility_candidate_loss_weight
                            * utility_candidate_loss
                            + self.args.utility_ranking_loss_weight
                            * utility_ranking_loss
                        )
                    elif utility_phase == "utility_gate":
                        # Candidate outcomes are now fixed; learn only whether,
                        # where and at which granularity they should intervene.
                        loss = (
                            forecast_loss
                            + self.args.utility_loss_weight * utility_loss
                            + self.args.utility_decision_loss_weight
                            * utility_decision
                        )
                    else:
                        loss = (
                            forecast_loss
                            + self.args.utility_loss_weight * utility_loss
                            + self.args.utility_decision_loss_weight
                            * utility_decision
                            + self.args.utility_candidate_loss_weight
                            * utility_candidate_loss
                            + self.args.utility_ranking_loss_weight
                            * utility_ranking_loss
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
                train_utility_losses.append(float(utility_loss.detach().cpu().item()))
                train_utility_decision_losses.append(
                    float(utility_decision.detach().cpu().item())
                )
                train_utility_candidate_losses.append(
                    float(utility_candidate_loss.detach().cpu().item())
                )
                train_utility_ranking_losses.append(
                    float(utility_ranking_loss.detach().cpu().item())
                )

            train_loss = np.mean(
                train_loss
            )
            train_mse = train_squared_error / max(1, train_point_count)
            train_mae = train_absolute_error / max(1, train_point_count)
            train_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
            train_utility_loss = float(np.mean(train_utility_losses))
            train_utility_decision_loss = float(
                np.mean(train_utility_decision_losses)
            )
            train_utility_candidate_loss = float(
                np.mean(train_utility_candidate_losses)
            )
            train_utility_ranking_loss = float(
                np.mean(train_utility_ranking_losses)
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

            current_lr = max(group["lr"] for group in model_optim.param_groups)
            lr_summary = "/".join(
                f"{group.get('group_name', 'group')}={group['lr']:.3g}"
                for group in model_optim.param_groups
            )

            end_time = time.time()

            validation_unit = "kW" if self.args.data == "SDWPF" else "original"
            epoch_summary = (
                f"Epoch: {epoch + 1}, Steps: {len(train_loader)}, "
                f"Time: {end_time - start_time:.2f}s | "
                f"Phase: {utility_phase} "
                f"Train Loss: {train_loss:.7f} "
                f"Utility Loss: {train_utility_loss:.7f} "
                f"Decision Loss: {train_utility_decision_loss:.7f} "
                f"Candidate Loss: {train_utility_candidate_loss:.7f} "
                f"Ranking Loss: {train_utility_ranking_loss:.7f} "
                f"Vali Loss: {vali_loss:.7f} "
                f"Vali MSE: {validation['mse']:.7f} "
                f"Vali MAE: {validation['mae']:.7f} "
                f"Val MAE({validation_unit}): {validation['original_mae']:.3f} "
                f"Persist MAE({validation_unit}): "
                f"{validation['original_persistence_mae']:.3f} "
                f"MAE Skill: {validation['original_mae_skill_vs_persistence_pct']:+.2f}% "
                f"RMSE Skill: {validation['original_rmse_skill_vs_persistence_pct']:+.2f}% "
                f"Gate: {validation['diagnostics'].get('residual_gate', 0.0):.5f} "
                f"GradNorm: {train_grad_norm:.3f} "
                f"LR({lr_summary}) "
                f"Select({self.args.early_stop_metric}): {selection_value:.7f}"
            )
            scale_parts = []
            for feature_name in ("Wspd", "power"):
                static_key = f"channel_static_{feature_name}"
                dynamic_key = f"channel_dynamic_{feature_name}_mean"
                if static_key in validation["diagnostics"]:
                    static_value = validation["diagnostics"][static_key]
                    dynamic_value = validation["diagnostics"].get(dynamic_key)
                    value = f"{static_value:.3f}"
                    if dynamic_value is not None:
                        value += f"/{dynamic_value:.3f}"
                    scale_parts.append(f"{feature_name}={value}")
            if scale_parts:
                epoch_summary += " ChannelScale(static/dynamic): " + ",".join(scale_parts)
            diagnostics = validation["diagnostics"]
            if "scene_wiki_prompt_gate" in diagnostics:
                epoch_summary += (
                    " WikiPromptGate: "
                    f"{diagnostics['scene_wiki_prompt_gate']:.5f}"
                )
            if "event_wiki_zero_intervention_fraction" in diagnostics:
                epoch_summary += (
                    " EventWiki(zero/active): "
                    f"{diagnostics['event_wiki_zero_intervention_fraction']:.3f}/"
                    f"{diagnostics['event_wiki_mean_active_factors']:.3f}"
                )
            if "utility_abstention_fraction" in diagnostics:
                epoch_summary += (
                    " UtilityWiki(abstain/event/combo/gate): "
                    f"{diagnostics['utility_abstention_fraction']:.3f}/"
                    f"{diagnostics['utility_single_event_fraction']:.3f}/"
                    f"{diagnostics['utility_composition_fraction']:.3f}/"
                    f"{diagnostics['utility_mean_strength']:.3f}"
                )
            trend_utility_parts = []
            for trend_name in ("down", "stable", "up"):
                gain_key = f"utility_trend_{trend_name}_gain_vs_base_pct"
                intervention_key = (
                    f"utility_trend_{trend_name}_intervention_fraction"
                )
                if gain_key in diagnostics:
                    trend_utility_parts.append(
                        f"{trend_name}="
                        f"{diagnostics[gain_key]:+.2f}%/"
                        f"int{100.0 * diagnostics[intervention_key]:.1f}%"
                    )
            if trend_utility_parts:
                epoch_summary += (
                    " TrendConditionedWiki(gain/intervention): "
                    + ",".join(trend_utility_parts)
                )
            candidate_gain_parts = []
            for candidate_name in ("event", "composition"):
                gain_key = (
                    f"utility_{candidate_name}_candidate_gain_vs_base_pct"
                )
                correction_key = (
                    f"utility_{candidate_name}_correction_abs_mean_{metric_suffix}"
                )
                if gain_key in diagnostics:
                    candidate_gain_parts.append(
                        f"{candidate_name}="
                        f"{diagnostics[gain_key]:+.2f}%/"
                        f"{diagnostics[correction_key]:.2f}{validation_unit}"
                    )
            if candidate_gain_parts:
                epoch_summary += (
                    " WikiCandidate(gain/correction): "
                    + ",".join(candidate_gain_parts)
                )
            selected_gain_parts = []
            for candidate_name in ("event", "composition"):
                count = diagnostics.get(f"utility_{candidate_name}_selected_count", 0)
                gain = diagnostics.get(
                    f"utility_{candidate_name}_selected_gain_vs_base_pct"
                )
                harmful = diagnostics.get(
                    f"utility_{candidate_name}_selected_harmful_fraction"
                )
                if count and gain is not None and harmful is not None:
                    selected_gain_parts.append(
                        f"{candidate_name}={gain:+.2f}%/"
                        f"harm{100.0 * harmful:.1f}%/n{int(count)}"
                    )
            if selected_gain_parts:
                epoch_summary += (
                    " WikiSelected(gain/harm/count): "
                    + ",".join(selected_gain_parts)
                )
            if "val_available_mae_kw" in diagnostics:
                epoch_summary += (
                    " Available MAE(kW): "
                    f"{diagnostics['val_available_mae_kw']:.3f} "
                    "Available MAE Skill: "
                    f"{diagnostics['val_available_mae_skill_pct']:+.2f}%"
                )
            first_horizon = diagnostics.get(f"val_h01_mae_{metric_suffix}")
            last_horizon = diagnostics.get(
                f"val_h{self.args.pred_len:02d}_mae_{metric_suffix}"
            )
            if first_horizon is not None and last_horizon is not None:
                epoch_summary += (
                    f" Horizon MAE(first/last): "
                    f"{first_horizon:.3f}/{last_horizon:.3f}"
                )
            print(epoch_summary)

            log_path = (
                path + "/" + "log.txt"
            )

            with open(
                log_path,
                "a",
            ) as log_file:
                log_file.write(epoch_summary + "\n")

            history.append(
                {
                    "epoch": epoch + 1,
                    "utility_phase": utility_phase,
                    "selection_eligible": utility_phase != "adapter_warmup",
                    "train_loss": float(
                        train_loss
                    ),
                    "train_mse": float(train_mse),
                    "train_mae": float(train_mae),
                    "train_grad_norm": train_grad_norm,
                    "train_utility_loss": train_utility_loss,
                    "train_utility_decision_loss": train_utility_decision_loss,
                    "train_utility_candidate_loss": train_utility_candidate_loss,
                    "train_utility_ranking_loss": train_utility_ranking_loss,
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
                    "val_mae_kw": float(validation["original_mae"]),
                    "val_rmse_kw": float(validation["original_rmse"]),
                    "val_persistence_mae_kw": float(
                        validation["original_persistence_mae"]
                    ),
                    "val_persistence_rmse_kw": float(
                        validation["original_persistence_rmse"]
                    ),
                    "val_mae_skill_original_pct": float(
                        validation["original_mae_skill_vs_persistence_pct"]
                    ),
                    "val_rmse_skill_original_pct": float(
                        validation["original_rmse_skill_vs_persistence_pct"]
                    ),
                    "selection_value": float(
                        selection_value
                    ),
                    "learning_rate": float(model_optim.param_groups[0]["lr"]),
                    "new_module_learning_rate": float(
                        next(
                            (
                                group["lr"]
                                for group in model_optim.param_groups
                                if group.get("group_name") == "new_forecast_modules"
                            ),
                            model_optim.param_groups[0]["lr"],
                        )
                    ),
                    "utility_learning_rate": float(
                        next(
                            (
                                group["lr"]
                                for group in model_optim.param_groups
                                if group.get("group_name") == "utility_modules"
                            ),
                            0.0,
                        )
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

            if utility_phase == "adapter_warmup":
                print(
                    "[UTILITY] Adapter warm-up validation is diagnostic only; "
                    "checkpoint selection and patience are deferred."
                )
            else:
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

        eligible_history = [
            record
            for record in history
            if bool(record.get("selection_eligible", True))
        ]
        best_record = min(
            eligible_history,
            key=lambda record: record["selection_value"],
        )
        self.args.selected_finetune_checkpoint = os.path.abspath(best_model_path)
        write_run_manifest(
            path,
            self.args,
            "finetune",
            model=self.model,
            datasets={"train": train_data, "val": vali_data},
            checkpoints={
                "pretrained_source": checkpoint_info(self.args.load_checkpoints),
                "overlay_source": checkpoint_info(
                    getattr(self.args, "overlay_checkpoint", None)
                ),
                "best_finetuned": checkpoint_info(best_model_path),
            },
            extra={
                "status": "complete",
                "setting": setting,
                "best_epoch": int(best_record["epoch"]),
                "best_selection_value": float(best_record["selection_value"]),
                "epochs_completed": sum(
                    int(record["epoch"] > 0) for record in history
                ),
                "validation_records": len(history),
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
        prediction_batches = []
        truth_batches = []
        persistence_batches = []
        scene_label_batches = []
        wiki_activation_batches = []
        utility_granularity_batches = []
        utility_strength_batches = []
        utility_score_batches = []
        utility_trend_probability_batches = []
        utility_availability_batches = []
        utility_base_prediction_batches = []
        utility_event_prediction_batches = []
        utility_composition_prediction_batches = []
        vali_data = getattr(vali_loader, "dataset", None)
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

                    if getattr(core_model, "utility_wiki", False):
                        utility_aux = core_model._last_utility_aux
                        utility_granularity_batches.append(
                            utility_aux["granularity"].detach().cpu().numpy()
                        )
                        utility_strength_batches.append(
                            utility_aux["strength"].detach().float().cpu().numpy()
                        )
                        utility_score_batches.append(
                            utility_aux["utilities"].detach().float().cpu().numpy()
                        )
                        utility_trend_probability_batches.append(
                            utility_aux["trend_probs"]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                        )
                        utility_availability_batches.append(
                            utility_aux["availability"].detach().cpu().numpy()
                        )
                        utility_base_prediction_batches.append(
                            utility_aux["base_prediction"]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                        )
                        utility_event_prediction_batches.append(
                            utility_aux["event_prediction"]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                        )
                        utility_composition_prediction_batches.append(
                            utility_aux["composition_prediction"]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
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

                    if getattr(core_model, "prompt_router", "trend") in (
                        "scene_wiki",
                        "hybrid_wiki",
                        "compositional_wiki",
                    ):
                        raw_history = (
                            batch_x
                            * core_model.scene_scaler_scale.view(1, 1, -1)
                            + core_model.scene_scaler_mean.view(1, 1, -1)
                        )
                        if core_model.prompt_router == "compositional_wiki":
                            _, scene_labels = compute_event_factor_rule_logits(
                                raw_history,
                                core_model.feature_columns,
                                rated_power=core_model.rated_power,
                                factor_ids=core_model.scene_wiki_scene_ids,
                                **core_model.scene_wiki_rule_kwargs,
                            )
                        else:
                            _, scene_labels = compute_scene_wiki_rule_logits(
                                raw_history,
                                core_model.feature_columns,
                                rated_power=core_model.rated_power,
                                scene_ids=core_model.scene_wiki_scene_ids,
                                **core_model.scene_wiki_rule_kwargs,
                            )
                        scene_label_batches.append(scene_labels.detach().cpu().numpy())
                        factor_activations = getattr(
                            core_model, "_last_wiki_factor_activations", None
                        )
                        if factor_activations is not None:
                            wiki_activation_batches.append(
                                factor_activations.detach().float().cpu().numpy()
                            )

                    mixer = getattr(core_model, "channel_mixer", None)
                    if mixer is not None and hasattr(core_model, "_operating_context"):
                        context = core_model._operating_context(batch_x)
                        if context is not None:
                            dynamic_scale = _dynamic_channel_scale_sample(
                                mixer, context
                            )
                            if dynamic_scale is not None:
                                dynamic_scale_samples.append(dynamic_scale)

                pred = (
                    pred_x.detach().cpu()
                )
                true = (
                    batch_y.detach().cpu()
                )
                prediction_batches.append(pred.numpy())
                truth_batches.append(true.numpy())
                persistence_batches.append(persistence.detach().float().cpu().numpy())

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
        pred_scaled = np.concatenate(prediction_batches, axis=0)
        true_scaled = np.concatenate(truth_batches, axis=0)
        persistence_scaled = np.concatenate(persistence_batches, axis=0)
        rated_power = float(getattr(self.args, "rated_power", 0.0))
        can_inverse_target = (
            hasattr(vali_data, "feature_columns")
            and hasattr(vali_data, "target")
            and hasattr(vali_data, "scaler")
            and vali_data.target in list(vali_data.feature_columns)
        )
        if can_inverse_target:
            target_index = list(vali_data.feature_columns).index(vali_data.target)
            target_mean = float(vali_data.scaler.mean_[target_index])
            target_scale = float(vali_data.scaler.scale_[target_index])
            pred_original = pred_scaled * target_scale + target_mean
            true_original = true_scaled * target_scale + target_mean
            persistence_original = persistence_scaled * target_scale + target_mean
        else:
            # Generic datasets retain their existing normalized-space behavior.
            pred_original = pred_scaled
            true_original = true_scaled
            persistence_original = persistence_scaled
            rated_power = 0.0
        if rated_power > 0:
            pred_original = np.clip(pred_original, 0.0, rated_power)
            true_original = np.clip(true_original, 0.0, rated_power)
            persistence_original = np.clip(persistence_original, 0.0, rated_power)
        original_metrics = forecast_metrics(
            pred_original, true_original, rated_power=rated_power or None
        )
        original_persistence = forecast_metrics(
            persistence_original, true_original, rated_power=rated_power or None
        )
        original_mae_skill = 100.0 * (
            1.0
            - original_metrics["mae"]
            / max(original_persistence["mae"], np.finfo(float).eps)
        )
        original_rmse_skill = 100.0 * (
            1.0
            - original_metrics["rmse"]
            / max(original_persistence["rmse"], np.finfo(float).eps)
        )
        per_horizon_mae = np.mean(
            np.abs(pred_original - true_original), axis=(0, 2)
        )
        metric_suffix = (
            "kw" if getattr(self.args, "data", None) == "SDWPF" else "original"
        )
        diagnostics[f"val_mae_{metric_suffix}"] = float(original_metrics["mae"])
        diagnostics[f"val_rmse_{metric_suffix}"] = float(original_metrics["rmse"])
        diagnostics[f"val_persistence_mae_{metric_suffix}"] = float(
            original_persistence["mae"]
        )
        diagnostics[f"val_persistence_rmse_{metric_suffix}"] = float(
            original_persistence["rmse"]
        )
        diagnostics["val_mae_skill_original_pct"] = float(original_mae_skill)
        diagnostics["val_rmse_skill_original_pct"] = float(original_rmse_skill)
        diagnostics[f"val_h01_mae_{metric_suffix}"] = float(per_horizon_mae[0])
        diagnostics[f"val_h{len(per_horizon_mae):02d}_mae_{metric_suffix}"] = float(
            per_horizon_mae[-1]
        )
        if scene_label_batches:
            scene_labels = np.concatenate(scene_label_batches, axis=0)
            if core_model.prompt_router == "compositional_wiki":
                selections = [
                    (factor_id, scene_labels[:, factor_index] >= 0.5)
                    for factor_index, factor_id in enumerate(
                        core_model.scene_wiki_scene_ids
                    )
                ]
                selections.append(("no_intervention", ~scene_labels.astype(bool).any(axis=1)))
            else:
                selections = [
                    (scene_id, scene_labels == scene_index)
                    for scene_index, scene_id in enumerate(
                        core_model.scene_wiki_scene_ids
                    )
                ]
            for scene_id, selected in selections:
                diagnostics[f"val_scene_{scene_id}_windows"] = int(selected.sum())
                if not selected.any():
                    continue
                scene_metrics = forecast_metrics(
                    pred_original[selected],
                    true_original[selected],
                    rated_power=rated_power or None,
                )
                scene_persistence = forecast_metrics(
                    persistence_original[selected],
                    true_original[selected],
                    rated_power=rated_power or None,
                )
                diagnostics[f"val_scene_{scene_id}_mae_kw"] = float(
                    scene_metrics["mae"]
                )
                diagnostics[f"val_scene_{scene_id}_rmse_kw"] = float(
                    scene_metrics["rmse"]
                )
                diagnostics[f"val_scene_{scene_id}_mae_skill_pct"] = float(
                    100.0
                    * (
                        1.0
                        - scene_metrics["mae"]
                        / max(scene_persistence["mae"], np.finfo(float).eps)
                    )
                )
        if wiki_activation_batches:
            activations = np.concatenate(wiki_activation_batches, axis=0)
            active = activations > 0.0
            diagnostics["event_wiki_zero_intervention_fraction"] = float(
                (~active.any(axis=1)).mean()
            )
            diagnostics["event_wiki_mean_active_factors"] = float(
                active.sum(axis=1).mean()
            )
            for factor_index, factor_id in enumerate(core_model.scene_wiki_scene_ids):
                diagnostics[f"event_wiki_{factor_id}_activation_fraction"] = float(
                    active[:, factor_index].mean()
                )
                diagnostics[f"event_wiki_{factor_id}_mean_strength"] = float(
                    activations[:, factor_index].mean()
                )
        if utility_granularity_batches:
            granularities = np.concatenate(utility_granularity_batches, axis=0)
            strengths = np.concatenate(utility_strength_batches, axis=0)
            utility_scores = np.concatenate(utility_score_batches, axis=0)
            trend_probabilities = np.concatenate(
                utility_trend_probability_batches, axis=0
            )
            availability = np.concatenate(utility_availability_batches, axis=0)
            diagnostics["utility_abstention_fraction"] = float(
                (granularities == 0).mean()
            )
            diagnostics["utility_single_event_fraction"] = float(
                (granularities == 1).mean()
            )
            diagnostics["utility_composition_fraction"] = float(
                (granularities == 2).mean()
            )
            diagnostics["utility_mean_strength"] = float(strengths.mean())
            diagnostics["utility_single_event_available_fraction"] = float(
                availability[:, 0].mean()
            )
            diagnostics["utility_composition_available_fraction"] = float(
                availability[:, 1].mean()
            )
            diagnostics["utility_single_event_score_mean"] = float(
                utility_scores[..., 0][
                    np.broadcast_to(
                        availability[:, None, 0], utility_scores[..., 0].shape
                    )
                ].mean()
                if availability[:, 0].any()
                else 0.0
            )
            diagnostics["utility_composition_score_mean"] = float(
                utility_scores[..., 1][
                    np.broadcast_to(
                        availability[:, None, 1], utility_scores[..., 1].shape
                    )
                ].mean()
                if availability[:, 1].any()
                else 0.0
            )
            branch_predictions = (
                ("event", np.concatenate(utility_event_prediction_batches, axis=0)),
                (
                    "composition",
                    np.concatenate(utility_composition_prediction_batches, axis=0),
                ),
            )
            utility_base_scaled = np.concatenate(
                utility_base_prediction_batches, axis=0
            )
            if can_inverse_target:
                utility_base_original = utility_base_scaled * target_scale + target_mean
            else:
                utility_base_original = utility_base_scaled
            if rated_power > 0:
                utility_base_original = np.clip(
                    utility_base_original, 0.0, rated_power
                )
            trend_indices = trend_probabilities.argmax(axis=-1)
            trend_names = ("down", "stable", "up")
            for trend_index, trend_name in enumerate(trend_names):
                sample_mask = trend_indices == trend_index
                diagnostics[f"utility_trend_{trend_name}_samples"] = int(
                    sample_mask.sum()
                )
                if not sample_mask.any():
                    continue
                trend_base_mae = np.abs(
                    utility_base_original[sample_mask] - true_original[sample_mask]
                ).mean()
                trend_selected_mae = np.abs(
                    pred_original[sample_mask] - true_original[sample_mask]
                ).mean()
                diagnostics[
                    f"utility_trend_{trend_name}_intervention_fraction"
                ] = float((granularities[sample_mask] != 0).mean())
                diagnostics[f"utility_trend_{trend_name}_gain_vs_base_pct"] = float(
                    100.0
                    * (
                        1.0
                        - trend_selected_mae
                        / max(trend_base_mae, np.finfo(float).eps)
                    )
                )
            for candidate_index, (candidate_name, candidate_scaled) in enumerate(
                branch_predictions
            ):
                candidate_original = (
                    candidate_scaled * target_scale + target_mean
                    if can_inverse_target
                    else candidate_scaled
                )
                if rated_power > 0:
                    candidate_original = np.clip(
                        candidate_original, 0.0, rated_power
                    )
                candidate_available = availability[:, candidate_index].astype(bool)
                if not candidate_available.any():
                    continue
                candidate_metrics = forecast_metrics(
                    candidate_original[candidate_available],
                    true_original[candidate_available],
                    rated_power=rated_power or None,
                )
                base_metrics = forecast_metrics(
                    utility_base_original[candidate_available],
                    true_original[candidate_available],
                    rated_power=rated_power or None,
                )
                diagnostics[
                    f"utility_{candidate_name}_candidate_mae_{metric_suffix}"
                ] = float(candidate_metrics["mae"])
                diagnostics[
                    f"utility_{candidate_name}_candidate_gain_vs_base_pct"
                ] = float(
                    100.0
                    * (
                        1.0
                        - candidate_metrics["mae"]
                        / max(base_metrics["mae"], np.finfo(float).eps)
                    )
                )
                diagnostics[
                    f"utility_{candidate_name}_correction_abs_mean_{metric_suffix}"
                ] = float(
                    np.abs(
                        candidate_original[candidate_available]
                        - utility_base_original[candidate_available]
                    ).mean()
                )
                selected_metrics = selected_intervention_metrics(
                    granularities,
                    strengths,
                    utility_base_original,
                    candidate_original,
                    true_original,
                    selected_index=candidate_index + 1,
                )
                for selected_name, selected_value in selected_metrics.items():
                    diagnostics[
                        f"utility_{candidate_name}_selected_{selected_name}"
                    ] = selected_value
        if vali_data is not None and hasattr(vali_data, "target_available_mask"):
            available = np.asarray(vali_data.target_available_mask(), dtype=bool)
            if available.shape == pred_original.shape[:2] and available.any():
                pred_available = pred_original[..., 0][available]
                true_available = true_original[..., 0][available]
                persistence_available = persistence_original[..., 0][available]
                available_metrics = forecast_metrics(
                    pred_available, true_available, rated_power=rated_power or None
                )
                available_persistence = forecast_metrics(
                    persistence_available,
                    true_available,
                    rated_power=rated_power or None,
                )
                diagnostics["val_available_point_pct"] = float(100.0 * available.mean())
                diagnostics["val_available_mae_kw"] = float(available_metrics["mae"])
                diagnostics["val_available_rmse_kw"] = float(available_metrics["rmse"])
                diagnostics["val_available_mae_skill_pct"] = float(
                    100.0
                    * (
                        1.0
                        - available_metrics["mae"]
                        / max(available_persistence["mae"], np.finfo(float).eps)
                    )
                )
        runtime = model_runtime_summary(self.model, self.args)
        if "residual_gate" in runtime:
            diagnostics["residual_gate"] = float(runtime["residual_gate"])
        wiki_runtime = runtime.get("scene_wiki", {})
        if wiki_runtime:
            diagnostics["scene_wiki_prompt_gate"] = float(
                wiki_runtime["prompt_gate"]
            )
            for key in ("last_intervention_mean", "last_intervention_max"):
                if key in wiki_runtime:
                    diagnostics[f"scene_wiki_{key}"] = float(wiki_runtime[key])
            for name, value in (
                wiki_runtime.get("retrieval_channel_weights") or {}
            ).items():
                diagnostics[f"scene_wiki_channel_weight_{name}"] = float(value)
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
            "original_mae": float(original_metrics["mae"]),
            "original_rmse": float(original_metrics["rmse"]),
            "original_persistence_mae": float(original_persistence["mae"]),
            "original_persistence_rmse": float(original_persistence["rmse"]),
            "original_mae_skill_vs_persistence_pct": float(original_mae_skill),
            "original_rmse_skill_vs_persistence_pct": float(original_rmse_skill),
            "diagnostics": diagnostics,
        }

    def test(self):
        report_split = str(getattr(self.args, "report_split", "test"))
        if report_split not in {"val", "test"}:
            raise ValueError(f"Unsupported forecast report split: {report_split!r}")
        test_data, test_loader = (
            self._get_data(flag=report_split)
        )

        preds = []
        trues = []
        last_observations = []

        result_tag = forecast_result_tag(self.args)

        requested_report_dir = str(
            getattr(self.args, "report_output_dir", "") or ""
        ).strip()
        folder_path = (
            os.path.abspath(os.path.expanduser(requested_report_dir))
            if requested_report_dir
            else os.path.join(
                "./outputs/test_results",
                self.args.model,
                self.args.data,
                result_tag,
            )
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
                trace_points=self.args.forecast_plot_points,
                trace_turbine_id=self.args.forecast_plot_turbine_id,
                trace_start=self.args.forecast_plot_start,
                model_name=self.args.model,
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
            "test" if report_split == "test" else "validation_report",
            model=self.model,
            datasets={report_split: test_data},
            checkpoints={"fine_tuned": checkpoint_info(checkpoint_path)},
            extra={
                "status": "complete",
                "evaluation_split": report_split,
                "prediction_shape": list(preds.shape),
                "target_shape": list(trues.shape),
                "metrics": values,
            },
        )

        report_label = (
            "Detailed forecast report"
            if report_split == "test"
            else "Detailed val forecast report"
        )
        print(f"{report_label}: {folder_path}")
        trace_path = os.path.join(folder_path, "forecast_trace.png")
        if os.path.isfile(trace_path):
            print(f"[AUDIT] Forecast trace: {os.path.abspath(trace_path)}")

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
