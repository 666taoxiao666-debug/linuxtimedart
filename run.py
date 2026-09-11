import argparse
import json
import os
import random

import numpy as np
import torch

from data_provider.sdwpf_features import sdwpf_feature_columns
from exp.exp_simmtm import Exp_SimMTM
from exp.exp_timedart import Exp_TimeDART
from exp.exp_timedart_v2 import Exp_TimeDART_v2
from utils.run_tags import bounded_component, experiment_setting, forecast_result_tag
from utils.experiment_audit import checkpoint_info
from utils.wind_regime_wiki import (
    EVENT_FACTOR_IDS,
    load_wind_regime_wiki_bundle,
    load_wind_regime_wiki_spec,
)


def build_parser():
    parser = argparse.ArgumentParser(description="TimeDART")

    # Basic configuration
    parser.add_argument("--task_name", required=True, choices=["pretrain", "finetune"])
    parser.add_argument(
        "--downstream_task",
        default="forecast",
        choices=["forecast", "classification"],
    )
    parser.add_argument(
        "--is_training",
        type=int,
        choices=[0, 1],
        default=1,
        help="1: train/validate; 0: load a fine-tuned checkpoint and test only",
    )
    parser.add_argument(
        "--evaluate_test_after_train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "also evaluate the test split after training; disabled by default "
            "to prevent repeated test-set peeking"
        ),
    )
    parser.add_argument("--model_id", required=True)
    parser.add_argument(
        "--model",
        required=True,
        choices=["TimeDART", "PromptTimeDART", "TimeDART_v2", "SimMTM"],
    )
    parser.add_argument("--llm_path", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--backbone", default="Qwen2.5-0.5B")

    # Data
    parser.add_argument("--data", required=True)
    parser.add_argument("--root_path", default="./datasets")
    parser.add_argument("--data_path", default="ETTh1.csv")
    parser.add_argument("--features", choices=["M", "S", "MS"], default="M")
    parser.add_argument("--target", default="OT")
    parser.add_argument("--freq", default="h")
    parser.add_argument("--select_channels", type=float, default=1.0)
    parser.add_argument("--use_norm", type=int, choices=[0, 1], default=1)

    # SDWPF-specific, leakage-safe window construction
    parser.add_argument("--sdwpf_train_ratio", type=float, default=0.7)
    parser.add_argument("--sdwpf_val_ratio", type=float, default=0.1)
    parser.add_argument("--sdwpf_expected_freq", default="10min")
    parser.add_argument(
        "--sdwpf_train_stride",
        type=int,
        default=6,
        help="training window step; 6 means one new window per hour for 10-minute data",
    )
    parser.add_argument("--sdwpf_eval_stride", type=int, default=6)
    parser.add_argument(
        "--sdwpf_filter_abnormal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply the standard SDWPF invalid-operation filters",
    )
    parser.add_argument(
        "--sdwpf_clip_power",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="clip power labels to [0, rated_power]",
    )
    parser.add_argument(
        "--sdwpf_circular_wind",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="encode Wdir/Ndir as sin/cos instead of raw degrees",
    )
    parser.add_argument(
        "--sdwpf_collapse_pitch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="replace Pab1/Pab2/Pab3 with their mean",
    )
    parser.add_argument(
        "--sdwpf_keep_curtailment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep high-wind zero-power rows as actual=0 with an available mask",
    )
    parser.add_argument(
        "--sdwpf_causal_fill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="forward-fill short sensor gaps; never use future points",
    )
    parser.add_argument(
        "--sdwpf_split",
        choices=["time_ratio", "rolling", "rolling_holdout", "seasonal"],
        default="time_ratio",
        help=(
            "time_ratio=70/10/20 final protocol; rolling=legacy walk-forward; "
            "rolling_holdout=disjoint expanding-origin validation folds before "
            "one sealed final holdout; seasonal=month blocks"
        ),
    )
    parser.add_argument("--sdwpf_fold", type=int, default=0)
    parser.add_argument("--sdwpf_n_folds", type=int, default=3)
    parser.add_argument(
        "--mix_channels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="mix all encoder channels into the power head (MS fine-tune). "
        "Default: on for SDWPF MS, off otherwise",
    )
    parser.add_argument(
        "--sdwpf_physics_features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="yaw error instead of nacelle heading; drop weak SCADA (Prtv/Itmp)",
    )
    parser.add_argument(
        "--sdwpf_drop_weak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop Prtv and Itmp when physics features are on",
    )
    parser.add_argument(
        "--channel_prior",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="initial mixer scales: Wspd/power large, Prtv/Itmp small. "
        "Default: on for SDWPF MS mixing",
    )
    parser.add_argument(
        "--op_context",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="inject last/mean wind and last power/yaw/pitch into the mixer "
        "before instance norm. Default: on for SDWPF MS mixing",
    )
    parser.add_argument(
        "--revin_keep_wind",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="do not instance-normalise Wspd, so the power-curve operating "
        "point survives. Default: on for SDWPF MS mixing",
    )

    # Checkpoints
    parser.add_argument("--checkpoints", default="./outputs/checkpoints/")
    parser.add_argument("--pretrain_checkpoints", default="./outputs/pretrain_checkpoints/")
    parser.add_argument("--transfer_checkpoints", default="ckpt_best.pth")
    parser.add_argument(
        "--pretrain_run_id",
        default="",
        help="optional immutable pretraining run id included in checkpoint discovery",
    )
    parser.add_argument(
        "--load_checkpoints",
        default=None,
        help="explicit pre-trained checkpoint; takes priority over automatic discovery",
    )
    parser.add_argument(
        "--pretrain_init",
        choices=["auto", "none"],
        default="auto",
        help=(
            "auto discovers a fold-matched pre-trained checkpoint; none guarantees "
            "that fine-tuning starts from random initialisation"
        ),
    )
    parser.add_argument(
        "--finetune_checkpoint",
        default=None,
        help="fine-tuned checkpoint used with --is_training 0",
    )
    parser.add_argument(
        "--allow_random_init",
        action="store_true",
        help="allow finetuning without a pre-trained checkpoint",
    )

    # Forecasting
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--input_len", type=int, default=336)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--test_pred_len", type=int, default=96)
    parser.add_argument("--seasonal_patterns", default="Monthly")
    parser.add_argument(
        "--residual_forecast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="predict a correction to the last observed value",
    )
    parser.add_argument(
        "--zero_init_residual_head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="start a residual model from the exact persistence forecast",
    )
    parser.add_argument(
        "--residual_gate_init",
        type=float,
        default=-4.0,
        help="initial logit of the residual correction gate; -4 is near persistence",
    )

    # Model
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=3)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--distil", action="store_false", default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fc_dropout", type=float, default=0.0)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--embed", default="timeF")
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--output_attention", action="store_true")
    parser.add_argument("--individual", type=int, default=0)
    parser.add_argument("--pct_start", type=float, default=0.3)
    parser.add_argument("--patch_len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)

    # Optimisation
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--new_module_learning_rate",
        type=float,
        default=0.0,
        help=(
            "optional fine-tuning LR for newly initialized forecast_head, "
            "channel_mixer, and residual gate; <=0 reuses --learning_rate"
        ),
    )
    parser.add_argument("--des", default="test")
    parser.add_argument(
        "--loss",
        default="MSE",
        choices=["MSE", "MAE", "Huber", "MIXED", "mse", "mae", "huber", "mixed"],
    )
    parser.add_argument(
        "--huber_delta",
        type=float,
        default=1.0,
        help="Huber transition point in standardized target units",
    )
    parser.add_argument(
        "--mix_mse_weight",
        type=float,
        default=0.8,
        help="MSE fraction for MIXED loss; the remainder is MAE",
    )
    parser.add_argument(
        "--horizon_weight_end",
        type=float,
        default=1.0,
        help="linear weight at the last horizon; 1 disables horizon weighting",
    )
    parser.add_argument(
        "--power_weight_alpha",
        type=float,
        default=0.0,
        help="extra bounded weight for high-power targets; 0 disables it",
    )
    parser.add_argument(
        "--early_stop_metric",
        choices=["loss", "mse", "mae", "original_mae", "original_rmse"],
        default="mse",
        help=(
            "validation quantity used for checkpoint selection; original_* "
            "uses clipped, inverse-scaled kW and matches final reporting"
        ),
    )
    parser.add_argument(
        "--validate_before_training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "evaluate and checkpoint epoch 0 before the first optimizer step; "
            "recommended for residual forecasts initialized to persistence"
        ),
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="maximum gradient norm; set <=0 to disable clipping",
    )
    parser.add_argument(
        "--rated_power",
        type=float,
        default=0.0,
        help="optional turbine rated power for capacity-normalized test metrics",
    )
    parser.add_argument("--lradj", default="decay")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="request deterministic PyTorch/CUDA execution for reproducible experiments",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="",
        help="unique run tag appended to checkpoint/test dirs to avoid overwriting",
    )
    parser.add_argument(
        "--audit_hash_data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="record SHA256 of the source dataset in run_manifest.json",
    )
    parser.add_argument(
        "--report_output_dir",
        default="",
        help="optional exact directory for final metrics, arrays, and figures",
    )
    parser.add_argument(
        "--report_split",
        choices=["val", "test"],
        default="test",
        help=(
            "dataset split used by evaluation-only forecast reporting; "
            "SDWPF test access additionally requires CONFIRM_FINAL_EVAL=1"
        ),
    )
    parser.add_argument(
        "--forecast_plot_points",
        type=int,
        default=150,
        help="maximum continuous 10-minute points in forecast_trace.png",
    )
    parser.add_argument(
        "--forecast_plot_turbine_id",
        type=int,
        default=None,
        help="optional predeclared turbine for the continuous prediction figure",
    )
    parser.add_argument(
        "--forecast_plot_start",
        default=None,
        help="optional predeclared ISO timestamp for the prediction figure",
    )

    # Device
    parser.add_argument(
        "--use_gpu", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true")
    parser.add_argument("--devices", default="0")

    # Diffusion and prompt guidance
    parser.add_argument("--time_steps", type=int, default=1000)
    parser.add_argument("--scheduler", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--use_prompt_adaln", action="store_true")
    parser.add_argument("--prompt_dim", type=int, default=None)
    parser.add_argument("--use_soft_prompt", action="store_true")
    parser.add_argument(
        "--disable_regime_prompt",
        action="store_true",
        help="ablation: discard regime predictor/prompt while retaining the backbone",
    )
    parser.add_argument(
        "--prompt_router",
        choices=["trend", "scene_wiki", "hybrid_wiki", "compositional_wiki"],
        default="trend",
        help=(
            "3-trend prompt, Wiki replacement ablation, legacy single-label "
            "hybrid, or sparse compositional event-Wiki residual prompting"
        ),
    )
    parser.add_argument("--num_modes", type=int, default=3)
    parser.add_argument(
        "--scene_wiki_config",
        default=None,
        help="versioned observable-scene definitions",
    )
    parser.add_argument(
        "--scene_wiki_embeddings",
        default=None,
        help="offline LLM embeddings built before training",
    )
    parser.add_argument("--scene_wiki_top_k", type=int, default=2)
    parser.add_argument("--scene_wiki_temperature", type=float, default=0.2)
    parser.add_argument("--scene_wiki_rule_weight", type=float, default=2.0)
    parser.add_argument("--scene_wiki_prompt_gate_init", type=float, default=-2.2)
    parser.add_argument("--scene_wiki_activation_threshold", type=float, default=0.55)
    parser.add_argument("--scene_wiki_confidence_power", type=float, default=1.0)
    parser.add_argument("--scene_wiki_recent_steps", type=int, default=None)
    parser.add_argument("--scene_wiki_low_wind_max_mps", type=float, default=None)
    parser.add_argument("--scene_wiki_active_wind_min_mps", type=float, default=None)
    parser.add_argument("--scene_wiki_low_power_max_ratio", type=float, default=None)
    parser.add_argument("--scene_wiki_rated_power_min_ratio", type=float, default=None)
    parser.add_argument("--scene_wiki_gust_std_min_mps", type=float, default=None)
    parser.add_argument("--scene_wiki_gust_step_min_mps", type=float, default=None)
    parser.add_argument("--scene_wiki_ramp_delta_min_ratio", type=float, default=None)
    parser.add_argument("--lambda_ce", type=float, default=0.1)
    parser.add_argument(
        "--lambda_scene_ce",
        type=float,
        default=0.02,
        help="legacy auxiliary exception-scene CE weight",
    )
    parser.add_argument(
        "--lambda_event_bce",
        type=float,
        default=None,
        help=(
            "multi-label event-factor BCE weight for compositional_wiki; "
            "defaults to lambda_scene_ce for command compatibility"
        ),
    )
    parser.add_argument(
        "--regime_target_index",
        type=int,
        default=-1,
        help="power channel index; SDWPF loaders always place power last",
    )
    parser.add_argument("--regime_stable_thresh", type=float, default=0.15)
    parser.add_argument("--regime_ramp_thresh", type=float, default=0.25)
    parser.add_argument(
        "--regime_label_method",
        choices=["auto", "trend_quantile", "legacy_volatility", "scene_wiki"],
        default="auto",
        help="auto selects train-calibrated trend quantiles for SDWPF",
    )
    parser.add_argument(
        "--regime_calibration_quantile",
        type=float,
        default=1.0 / 3.0,
        help="lower/upper train-only quantiles defining down/stable/up regimes",
    )
    parser.add_argument(
        "--regime_calibration_samples",
        type=int,
        default=50000,
        help="deterministic maximum number of training histories used for calibration",
    )
    parser.add_argument(
        "--regime_min_class_fraction",
        type=float,
        default=0.05,
        help="fail pretraining when a calibrated training regime has less support",
    )
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--mask_ratio", type=float, default=1.0)

    # Classification
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument(
        "--classification_early_stop_metric",
        choices=["macro_f1", "accuracy"],
        default="macro_f1",
        help=(
            "validation metric used for classification checkpoint selection and "
            "early stopping; macro_f1 is robust to class imbalance"
        ),
    )

    # SimMTM
    parser.add_argument("--lm", type=int, default=3)
    parser.add_argument("--positive_nums", type=int, default=3)
    parser.add_argument("--rbtp", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--masked_rule", default="geometric")
    parser.add_argument("--mask_rate", type=float, default=0.5)
    return parser


def pretrain_signature(args):
    parts = [
        args.model,
        args.data,
        args.features,
        f"il{args.input_len}",
        f"dm{args.d_model}",
        f"df{args.d_ff}",
        f"nh{args.n_heads}",
        f"el{args.e_layers}",
        f"dl{args.d_layers}",
        f"p{args.patch_len}",
        f"s{args.stride}",
        args.backbone,
    ]
    if args.model == "PromptTimeDART" or args.use_soft_prompt:
        parts.extend(
            [
                "prompt",
                f"router{args.prompt_router}",
                f"m{args.num_modes}",
                f"reg{args.regime_label_method}",
                f"rq{args.regime_calibration_quantile:g}",
            ]
        )
        if args.prompt_router in (
            "scene_wiki",
            "hybrid_wiki",
            "compositional_wiki",
        ):
            parts.extend(
                [
                    f"wk{args.scene_wiki_bundle_sha256[:12]}",
                    f"top{args.scene_wiki_top_k}",
                    f"wt{args.scene_wiki_rule_weight:g}",
                    f"wg{args.scene_wiki_prompt_gate_init:g}",
                ]
            )
            if args.prompt_router == "compositional_wiki":
                parts.extend(
                    [
                        f"ef{args.scene_wiki_num_modes}",
                        f"at{args.scene_wiki_activation_threshold:g}",
                        f"cp{args.scene_wiki_confidence_power:g}",
                    ]
                )
    if args.data == "SDWPF":
        # Fold-specific pretraining is required: a checkpoint trained on later
        # rolling folds must never be auto-discovered for an earlier fold.
        parts.extend(
            [
                f"split{args.sdwpf_split}",
                f"fold{args.sdwpf_fold}of{args.sdwpf_n_folds}",
                f"seed{args.seed}",
            ]
        )
    if str(getattr(args, "pretrain_run_id", "")).strip():
        parts.append(f"rid{args.pretrain_run_id}")
    return bounded_component("_".join(str(part) for part in parts))


def resolve_pretrained_checkpoint(args):
    explicit = args.load_checkpoints
    if isinstance(explicit, str) and explicit.strip().lower() in {"", "none", "null"}:
        explicit = None
    if args.pretrain_init == "none":
        if explicit:
            raise ValueError("--pretrain_init none conflicts with --load_checkpoints")
        return None
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Explicit pre-trained checkpoint not found: {path}")
        return path

    candidates = [
        os.path.join(args.pretrain_run_dir, args.transfer_checkpoints),
        # Compatibility with checkpoints created by the original repository.
        os.path.join(args.pretrain_checkpoints, args.data, args.transfer_checkpoints),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    if args.allow_random_init:
        return None
    raise FileNotFoundError(
        "No pre-trained checkpoint was found. Expected one of:\n- "
        + "\n- ".join(os.path.abspath(path) for path in candidates)
        + "\nPass --load_checkpoints PATH, or use --allow_random_init intentionally."
    )


def configure_args(args):
    args.seq_len = args.input_len if args.seq_len is None else args.seq_len
    if args.seq_len != args.input_len:
        raise ValueError("seq_len and input_len must be equal in this implementation")
    if args.data == "SDWPF":
        if args.regime_label_method == "auto":
            args.regime_label_method = "trend_quantile"
        if args.target == "OT":
            args.target = "power"
        if args.features == "M":
            print("[INFO] SDWPF defaults to multivariate-to-univariate forecasting (MS).")
            args.features = "MS"
        if args.freq == "h":
            args.freq = "10min"
        if args.rated_power <= 0:
            args.rated_power = 1500.0
            print("[INFO] SDWPF rated_power defaulted to 1500 kW.")
        args.feature_columns = sdwpf_feature_columns(
            features=args.features,
            target=args.target,
            circular_wind=args.sdwpf_circular_wind,
            collapse_pitch=args.sdwpf_collapse_pitch,
            physics_features=args.sdwpf_physics_features,
            drop_weak_features=args.sdwpf_drop_weak,
        )
        n_features = len(args.feature_columns)
        args.enc_in = n_features
        args.dec_in = n_features
        if args.mix_channels is None:
            args.mix_channels = (
                args.features == "MS" and args.task_name == "finetune"
            )
        mixing = bool(args.mix_channels) and args.task_name == "finetune"
        if args.channel_prior is None:
            args.channel_prior = mixing
        if args.op_context is None:
            args.op_context = mixing
        if args.revin_keep_wind is None:
            args.revin_keep_wind = mixing
        args.c_out = 1 if mixing else n_features
        print(f"[INFO] SDWPF features ({n_features}): {args.feature_columns}")
        if args.pred_len >= 96:
            print(
                "[INFO] 4-16 h (steps 25-96) is reported as an NWP-free ceiling, "
                "not as a SOTA claim. The primary SCADA-only task is 0-4 h."
            )
    else:
        if args.regime_label_method == "auto":
            args.regime_label_method = "legacy_volatility"
        args.feature_columns = list(getattr(args, "feature_columns", None) or [])
        if args.mix_channels is None:
            args.mix_channels = False
        if args.channel_prior is None:
            args.channel_prior = False
        if args.op_context is None:
            args.op_context = False
        if args.revin_keep_wind is None:
            args.revin_keep_wind = False
    if args.lambda_event_bce is None:
        args.lambda_event_bce = args.lambda_scene_ce
    if args.prompt_router in (
        "scene_wiki",
        "hybrid_wiki",
        "compositional_wiki",
    ):
        if args.lambda_scene_ce < 0:
            raise ValueError("lambda_scene_ce cannot be negative")
        if args.lambda_event_bce < 0:
            raise ValueError("lambda_event_bce cannot be negative")
        if args.model != "PromptTimeDART" or args.downstream_task != "forecast":
            raise ValueError(
                f"--prompt_router {args.prompt_router} currently requires "
                "PromptTimeDART forecast"
            )
        if args.data != "SDWPF":
            raise ValueError(f"--prompt_router {args.prompt_router} currently requires SDWPF")
        if args.scene_wiki_config is None:
            args.scene_wiki_config = {
                "scene_wiki": "configs/wind_regime_wiki.json",
                "hybrid_wiki": "configs/wind_exception_wiki.json",
                "compositional_wiki": "configs/wind_event_factor_wiki.json",
            }[args.prompt_router]
        if args.scene_wiki_embeddings is None:
            args.scene_wiki_embeddings = {
                "scene_wiki": "outputs/wiki/wind_regime_wiki_qwen.npz",
                "hybrid_wiki": "outputs/wiki/wind_exception_wiki_qwen.npz",
                "compositional_wiki": "outputs/wiki/wind_event_factor_wiki_qwen.npz",
            }[args.prompt_router]
        spec = load_wind_regime_wiki_spec(args.scene_wiki_config)
        bundle = load_wind_regime_wiki_bundle(
            args.scene_wiki_embeddings,
            expected_scene_ids=spec["scene_ids"],
        )
        if bundle["config_sha256"] != spec["sha256"]:
            raise ValueError(
                "Wiki embeddings were built from a different config. Rebuild the bundle."
            )
        defaults = spec.get("rule_defaults", {})
        rule_names = [
            "recent_steps",
            "low_wind_max_mps",
            "active_wind_min_mps",
            "low_power_max_ratio",
            "rated_power_min_ratio",
            "gust_std_min_mps",
            "gust_step_min_mps",
        ]
        if args.prompt_router != "compositional_wiki":
            rule_names.append("ramp_delta_min_ratio")
        for name in rule_names:
            attr = f"scene_wiki_{name}"
            if getattr(args, attr) is None:
                if name not in defaults:
                    raise ValueError(f"Wiki config has no rule default for {name}")
                setattr(args, attr, defaults[name])
        args.scene_wiki_scene_ids = list(spec["scene_ids"])
        args.scene_wiki_config_sha256 = spec["sha256"]
        args.scene_wiki_bundle_sha256 = bundle["sha256"]
        args.scene_wiki_encoder_name = bundle["encoder_name"]
        args.scene_wiki_num_modes = len(spec["scene_ids"])
        if args.prompt_router == "scene_wiki":
            args.num_modes = args.scene_wiki_num_modes
            args.regime_label_method = "scene_wiki"
        elif args.prompt_router == "hybrid_wiki":
            if tuple(spec["scene_ids"][:1]) != ("no_exception",):
                raise ValueError(
                    "hybrid_wiki requires configs/wind_exception_wiki.json or an "
                    "equivalent Wiki beginning with no_exception"
                )
            if args.num_modes != 3:
                raise ValueError("hybrid_wiki retains the original three trend modes")
            if args.regime_label_method in ("auto", "scene_wiki"):
                args.regime_label_method = "trend_quantile"
        else:
            if tuple(spec["scene_ids"]) != EVENT_FACTOR_IDS:
                raise ValueError(
                    "compositional_wiki requires configs/wind_event_factor_wiki.json "
                    f"with factor order {EVENT_FACTOR_IDS}"
                )
            if args.num_modes != 3:
                raise ValueError(
                    "compositional_wiki retains the original three trend modes"
                )
            if args.regime_label_method in ("auto", "scene_wiki"):
                args.regime_label_method = "trend_quantile"
        if not 1 <= args.scene_wiki_top_k <= args.scene_wiki_num_modes:
            raise ValueError("scene_wiki_top_k must be between 1 and the scene count")
        if args.scene_wiki_temperature <= 0:
            raise ValueError("scene_wiki_temperature must be positive")
        if args.scene_wiki_rule_weight < 0:
            raise ValueError("scene_wiki_rule_weight cannot be negative")
        if not 0.0 <= args.scene_wiki_activation_threshold < 1.0:
            raise ValueError("scene_wiki_activation_threshold must be in [0, 1)")
        if args.scene_wiki_confidence_power <= 0:
            raise ValueError("scene_wiki_confidence_power must be positive")
        if int(args.scene_wiki_recent_steps) < 3:
            raise ValueError("scene_wiki_recent_steps must be at least 3")
        ratio_rule_names = [
            "low_power_max_ratio",
            "rated_power_min_ratio",
        ]
        if args.prompt_router != "compositional_wiki":
            ratio_rule_names.append("ramp_delta_min_ratio")
        for name in ratio_rule_names:
            value = float(getattr(args, f"scene_wiki_{name}"))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"scene_wiki_{name} must be in (0, 1]")
        for name in (
            "low_wind_max_mps",
            "active_wind_min_mps",
            "gust_std_min_mps",
            "gust_step_min_mps",
        ):
            if float(getattr(args, f"scene_wiki_{name}")) < 0.0:
                raise ValueError(f"scene_wiki_{name} cannot be negative")
        print(
            "[INFO] Semantic Wiki prompt router: "
            f"mode={args.prompt_router} entries={args.scene_wiki_num_modes} "
            f"top_k={args.scene_wiki_top_k} "
            f"activation_threshold={args.scene_wiki_activation_threshold:g} "
            f"encoder={args.scene_wiki_encoder_name} "
            f"bundle_sha256={args.scene_wiki_bundle_sha256[:12]}"
        )
    elif args.num_modes != 3:
        raise ValueError("The trend prompt router requires --num_modes 3")
    args.use_gpu = bool(args.use_gpu and torch.cuda.is_available())
    if args.use_multi_gpu and not args.use_gpu:
        raise ValueError("--use_multi_gpu requires CUDA")
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(device_id) for device_id in args.devices.split(",")]
        if not args.device_ids:
            raise ValueError("No GPU ids were supplied in --devices")
        args.gpu = args.device_ids[0]
    else:
        args.device_ids = [args.gpu] if args.use_gpu else []
    args.pretrain_run_dir = os.path.join(
        args.pretrain_checkpoints, args.data, pretrain_signature(args)
    )
    if not getattr(args, "run_id", None):
        args.run_id = os.environ.get("RUN_ID", "")
    return args


def authorize_forecast_report_split(args, environ=None):
    """Keep validation figures separate from the sealed SDWPF test split."""
    split = str(getattr(args, "report_split", "test"))
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported forecast report split: {split!r}")
    if split == "val" and getattr(args, "downstream_task", "forecast") != "forecast":
        raise ValueError("--report_split val is supported for forecast reporting only")
    if split == "val" and getattr(args, "model", None) not in {
        "TimeDART",
        "PromptTimeDART",
    }:
        raise ValueError(
            "--report_split val is implemented only for TimeDART/PromptTimeDART"
        )
    environment = os.environ if environ is None else environ
    if (
        getattr(args, "data", None) == "SDWPF"
        and split == "test"
        and environment.get("CONFIRM_FINAL_EVAL") != "1"
    ):
        raise PermissionError(
            "SDWPF test evaluation is sealed. Use --report_split val for an "
            "interim figure, or set CONFIRM_FINAL_EVAL=1 only for the frozen "
            "one-time final evaluation."
        )
    return split


def load_finetuned_model(exp, checkpoint_path):
    path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {path}")
    state = torch.load(path, map_location=exp.device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    exp.model.load_state_dict(state, strict=True)
    exp.args.loaded_finetune_checkpoint = path
    exp.args.loaded_finetune_checkpoint_info = checkpoint_info(path)
    manifest_path = os.path.join(os.path.dirname(path), "run_manifest.json")
    exp.args.checkpoint_training_args = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        training_args = manifest.get("args", {})
        exp.args.checkpoint_training_args = training_args
        critical_keys = (
            "model",
            "data",
            "features",
            "input_len",
            "pred_len",
            "d_model",
            "e_layers",
            "patch_len",
            "stride",
            "sdwpf_split",
            "sdwpf_fold",
            "sdwpf_n_folds",
            "mix_channels",
            "residual_forecast",
            "disable_regime_prompt",
            "regime_label_method",
            "regime_calibration_quantile",
            "prompt_router",
            "scene_wiki_config_sha256",
            "scene_wiki_bundle_sha256",
            "scene_wiki_scene_ids",
            "scene_wiki_top_k",
            "scene_wiki_temperature",
            "scene_wiki_rule_weight",
            "scene_wiki_prompt_gate_init",
            "scene_wiki_activation_threshold",
            "scene_wiki_confidence_power",
            "lambda_scene_ce",
            "lambda_event_bce",
            "scene_wiki_num_modes",
            "scene_wiki_recent_steps",
            "scene_wiki_low_wind_max_mps",
            "scene_wiki_active_wind_min_mps",
            "scene_wiki_low_power_max_ratio",
            "scene_wiki_rated_power_min_ratio",
            "scene_wiki_gust_std_min_mps",
            "scene_wiki_gust_step_min_mps",
            "scene_wiki_ramp_delta_min_ratio",
            "feature_columns",
        )
        mismatches = []
        for key in critical_keys:
            if key not in training_args or not hasattr(exp.args, key):
                continue
            current = getattr(exp.args, key)
            if training_args[key] != current:
                mismatches.append(f"{key}: checkpoint={training_args[key]!r}, eval={current!r}")
        if mismatches:
            raise ValueError(
                "Fine-tuned checkpoint manifest does not match evaluation arguments:\n- "
                + "\n- ".join(mismatches)
            )
        exp.args.loaded_finetune_manifest = os.path.abspath(manifest_path)
        print(f"Loaded training manifest: {os.path.abspath(manifest_path)}")
    else:
        print(
            "[WARNING] Fine-tuned checkpoint has no run_manifest.json; "
            "training loss/LR/data provenance cannot be verified."
        )
    print(f"Loaded fine-tuned checkpoint: {path}")


def main():
    args = configure_args(build_parser().parse_args())
    seed = args.seed
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # Scientific runs should fail loudly if an operation has no
        # deterministic implementation; a warning would silently weaken the
        # reproducibility claim.
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    exp_map = {
        "TimeDART": Exp_TimeDART,
        "PromptTimeDART": Exp_TimeDART,
        "TimeDART_v2": Exp_TimeDART_v2,
        "SimMTM": Exp_SimMTM,
    }
    print("Args in experiment:")
    print(args)

    if args.task_name == "pretrain":
        os.makedirs(args.pretrain_run_dir, exist_ok=True)
        for _ in range(args.itr):
            exp = exp_map[args.model](args)
            print(f">>>>>>> start pre-training: {args.pretrain_run_dir}")
            exp.pretrain()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return

    if args.is_training == 0:
        authorize_forecast_report_split(args)
        args.load_checkpoints = None
        setting = experiment_setting(args, 0)
        checkpoint = args.finetune_checkpoint or os.path.join(
            args.checkpoints, setting, "checkpoint.pth"
        )
        exp = exp_map[args.model](args)
        load_finetuned_model(exp, checkpoint)
        exp.cls_test() if args.downstream_task == "classification" else exp.test()
        return

    args.load_checkpoints = resolve_pretrained_checkpoint(args)
    if args.load_checkpoints:
        print(f"Using pre-trained checkpoint: {args.load_checkpoints}")
    else:
        print("[WARNING] Finetuning from random initialisation was explicitly enabled.")

    for run_index in range(args.itr):
        setting = experiment_setting(args, run_index)
        exp = exp_map[args.model](args)
        print(f">>>>>>> start finetuning: {setting}")
        if args.downstream_task == "classification":
            exp.cls_train(setting)
            if args.evaluate_test_after_train:
                exp.cls_test()
        else:
            exp.train(setting)
            if args.evaluate_test_after_train:
                exp.test()
        if not args.evaluate_test_after_train:
            print(
                "[INFO] Test evaluation was intentionally skipped. Run with "
                "--is_training 0 and an explicit --finetune_checkpoint for the final test."
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
