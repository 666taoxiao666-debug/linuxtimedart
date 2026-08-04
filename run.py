import argparse
import os
import random
import re

import numpy as np
import torch

from exp.exp_simmtm import Exp_SimMTM
from exp.exp_timedart import Exp_TimeDART
from exp.exp_timedart_v2 import Exp_TimeDART_v2


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
        help="1: train then test; 0: load a fine-tuned checkpoint and test only",
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

    # Checkpoints
    parser.add_argument("--checkpoints", default="./outputs/checkpoints/")
    parser.add_argument("--pretrain_checkpoints", default="./outputs/pretrain_checkpoints/")
    parser.add_argument("--transfer_checkpoints", default="ckpt_best.pth")
    parser.add_argument(
        "--load_checkpoints",
        default=None,
        help="explicit pre-trained checkpoint; takes priority over automatic discovery",
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
        choices=["loss", "mse", "mae"],
        default="mse",
        help="validation quantity used for checkpoint selection",
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
    parser.add_argument("--num_modes", type=int, choices=[3], default=3)
    parser.add_argument("--lambda_ce", type=float, default=0.1)
    parser.add_argument(
        "--regime_target_index",
        type=int,
        default=-1,
        help="power channel index; SDWPF loaders always place power last",
    )
    parser.add_argument("--regime_stable_thresh", type=float, default=0.15)
    parser.add_argument("--regime_ramp_thresh", type=float, default=0.25)
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


def _safe_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


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
        parts.extend(["prompt", f"m{args.num_modes}"])
    return "_".join(_safe_component(part) for part in parts)


def experiment_setting(args, run_index):
    return (
        f"{args.task_name}_{args.model}_{args.data}_{args.features}_"
        f"il{args.input_len}_ll{args.label_len}_pl{args.pred_len}_"
        f"dm{args.d_model}_df{args.d_ff}_nh{args.n_heads}_el{args.e_layers}_"
        f"dl{args.d_layers}_fc{args.factor}_dp{args.dropout}_hdp{args.head_dropout}_"
        f"ep{args.train_epochs}_bs{args.batch_size}_lr{args.learning_rate}_"
        f"loss{args.loss}_res{int(args.residual_forecast)}_"
        f"seed{args.seed}_run{run_index}"
    )


def resolve_pretrained_checkpoint(args):
    explicit = args.load_checkpoints
    if isinstance(explicit, str) and explicit.strip().lower() in {"", "none", "null"}:
        explicit = None
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
        if args.target == "OT":
            args.target = "power"
        if args.features == "M":
            print("[INFO] SDWPF defaults to multivariate-to-univariate forecasting (MS).")
            args.features = "MS"
        if args.freq == "h":
            args.freq = "10min"
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
    return args


def load_finetuned_model(exp, checkpoint_path):
    path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {path}")
    state = torch.load(path, map_location=exp.device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    exp.model.load_state_dict(state, strict=True)
    print(f"Loaded fine-tuned checkpoint: {path}")


def main():
    args = configure_args(build_parser().parse_args())
    seed = args.seed
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
            exp.cls_test()
        else:
            exp.train(setting)
            exp.test()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()