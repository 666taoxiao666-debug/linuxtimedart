"""Fit Persistence / power-curve / tree baselines on SDWPF using train data only."""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from exp.exp_timedart import Exp_TimeDART
from run import build_parser, configure_args
from utils.experiment_audit import write_run_manifest
from utils.forecast_report import _horizon_segment_table
from utils.sdwpf_baselines import (
    _window_truth,
    authorize_eval_split,
    evaluate_methods,
    fit_power_curve,
    fit_tree_baseline,
    persistence_forecast,
    power_curve_forecast,
    seasonal_persistence_forecast,
    tree_forecast,
)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def main():
    parser = build_parser()
    parser.description = "SDWPF train-only baselines"
    parser.add_argument("--baseline_output_dir", default=None)
    parser.add_argument(
        "--baseline_max_samples",
        type=int,
        default=500000,
        help="maximum balanced window-horizon rows used by the HGB baseline",
    )
    parser.add_argument(
        "--eval_split",
        choices=["val", "test"],
        default="val",
        help="validation by default; test requires CONFIRM_FINAL_EVAL=1",
    )
    args = configure_args(parser.parse_args())
    if args.data != "SDWPF":
        raise ValueError("run_baselines.py currently supports SDWPF only")
    args.eval_split = authorize_eval_split(args.eval_split)
    args.load_checkpoints = None
    args.allow_random_init = True
    args.task_name = "finetune"

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.baseline_output_dir or os.path.join(
        "./outputs/baseline_results", args.data, f"id{run_id}"
    )
    os.makedirs(output_dir, exist_ok=True)

    exp = Exp_TimeDART(args)
    train_set, _ = exp._get_data(flag="train")
    eval_set, _ = exp._get_data(flag=args.eval_split)
    if len(train_set) == 0 or len(eval_set) == 0:
        raise ValueError(
            f"Empty baseline split: train={len(train_set)}, "
            f"{args.eval_split}={len(eval_set)}"
        )

    write_run_manifest(
        output_dir,
        args,
        "baseline_evaluation",
        datasets={"train": train_set, args.eval_split: eval_set},
        extra={
            "fit_split": "train",
            "evaluation_split": args.eval_split,
            "future_scada_used": False,
            "test_access_confirmed": args.eval_split == "test",
        },
    )

    print(
        "[AUDIT] Baseline fit_split=train "
        f"eval_split={args.eval_split} pred_len={args.pred_len} "
        f"eval_stride={args.sdwpf_eval_stride}"
    )
    print("Fitting power curve and tree on the training split only...")
    curve = fit_power_curve(train_set)
    tree = fit_tree_baseline(
        train_set,
        max_samples=args.baseline_max_samples,
        random_state=args.seed,
    )
    print(
        "Tree baseline samples: "
        f"{tree.sdwpf_training_samples_:,} rows from "
        f"{tree.sdwpf_windows_sampled_:,} windows; every horizon balanced"
    )

    truth = _window_truth(eval_set)
    persistence = persistence_forecast(eval_set)
    methods = {
        "daily_persistence": seasonal_persistence_forecast(eval_set),
        "power_curve": power_curve_forecast(eval_set, curve),
        "tree": tree_forecast(eval_set, tree),
    }
    if args.rated_power > 0:
        cap = float(args.rated_power)
        truth = np.clip(truth, 0.0, cap)
        persistence = np.clip(persistence, 0.0, cap)
        methods = {name: np.clip(pred, 0.0, cap) for name, pred in methods.items()}

    comparison = evaluate_methods(truth, methods, args.rated_power, persistence)
    _write_json(
        os.path.join(output_dir, f"{args.eval_split}_metrics.json"), comparison
    )

    rows = []
    for name, metrics in comparison.items():
        row = {"method": name}
        row.update(metrics)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)

    all_methods = {"persistence": persistence, **methods}
    segment_frames = []
    step_minutes = int(pd.to_timedelta(args.freq).total_seconds() // 60)
    for name, pred in all_methods.items():
        table = _horizon_segment_table(
            pred, truth, persistence, step_minutes, args.rated_power
        )
        table.insert(0, "method", name)
        segment_frames.append(table)
    pd.concat(segment_frames, ignore_index=True).to_csv(
        os.path.join(output_dir, "metrics_by_horizon_segment.csv"), index=False
    )

    lines = [
        "SDWPF train-fit baselines (history-only, no NWP)",
        "=" * 72,
        f"Evaluation split: {args.eval_split}",
        f"Forecast horizon: {args.pred_len} x {step_minutes} min "
        f"= {args.pred_len * step_minutes / 60:.1f} h",
        "",
    ]
    for name, metrics in comparison.items():
        lines.append(
            f"{name:14s} MAE={metrics['mae']:.4f} kW  RMSE={metrics['rmse']:.4f} kW  "
            f"R2={metrics.get('r2', float('nan')):.4f}"
        )
        if "mae_skill_vs_persistence_pct" in metrics:
            lines[-1] += (
                f"  MAE_skill={metrics['mae_skill_vs_persistence_pct']:+.2f}%  "
                f"RMSE_skill={metrics['rmse_skill_vs_persistence_pct']:+.2f}%"
            )
    with open(os.path.join(output_dir, "score.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
