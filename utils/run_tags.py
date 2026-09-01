import re


def safe_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def forecast_result_tag(args):
    """根据超参数生成独立的测试报告目录名。"""
    parts = [
        args.features,
        f"il{args.input_len}",
        f"pl{args.pred_len}",
        f"dm{args.d_model}",
        f"el{args.e_layers}",
        f"p{args.patch_len}",
        f"s{args.stride}",
        f"loss{args.loss}",
        f"lr{args.learning_rate}",
        f"res{int(args.residual_forecast)}",
        f"rg{getattr(args, 'residual_gate_init', -4.0)}",
        f"hw{args.horizon_weight_end}",
        f"pw{args.power_weight_alpha}",
        f"mix{args.mix_mse_weight}",
        f"chmix{int(getattr(args, 'mix_channels', False))}",
        f"phy{int(getattr(args, 'sdwpf_physics_features', False))}",
        f"keepw{int(getattr(args, 'revin_keep_wind', False))}",
        f"split{getattr(args, 'sdwpf_split', 'time')}",
        f"fold{getattr(args, 'sdwpf_fold', 0)}",
        f"seed{args.seed}",
    ]
    run_id = getattr(args, "run_id", "") or ""
    if str(run_id).strip():
        parts.append(f"id{safe_component(run_id)}")
    return "_".join(safe_component(part) for part in parts)


def experiment_setting(args, run_index):
    base = (
        f"{args.task_name}_{args.model}_{args.data}_{args.features}_"
        f"il{args.input_len}_ll{args.label_len}_pl{args.pred_len}_"
        f"dm{args.d_model}_df{args.d_ff}_nh{args.n_heads}_el{args.e_layers}_"
        f"dl{args.d_layers}_fc{args.factor}_dp{args.dropout}_hdp{args.head_dropout}_"
        f"ep{args.train_epochs}_bs{args.batch_size}_lr{args.learning_rate}_"
        f"loss{args.loss}_res{int(args.residual_forecast)}_"
        f"rg{getattr(args, 'residual_gate_init', -4.0)}_"
        f"hw{args.horizon_weight_end}_pw{args.power_weight_alpha}_"
        f"mix{args.mix_mse_weight}_chmix{int(getattr(args, 'mix_channels', False))}_"
        f"phy{int(getattr(args, 'sdwpf_physics_features', False))}_"
        f"keepw{int(getattr(args, 'revin_keep_wind', False))}_"
        f"split{getattr(args, 'sdwpf_split', 'time')}_fold{getattr(args, 'sdwpf_fold', 0)}_"
        f"seed{args.seed}"
    )
    run_id = getattr(args, "run_id", "") or ""
    if str(run_id).strip():
        return f"{base}_id{safe_component(run_id)}"
    return f"{base}_run{run_index}"
