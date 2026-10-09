#!/usr/bin/env python3
"""Read-only diagnostics on training histories/targets, never outer val/test."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_provider.data_loader import Dataset_SDWPF


def window_moments(values, starts, length):
    values = np.asarray(values, dtype=np.float64)
    sums = np.r_[0., np.cumsum(values)]
    squares = np.r_[0., np.cumsum(values ** 2)]
    ends = starts + length
    mean = (sums[ends] - sums[starts]) / length
    variance = np.maximum((squares[ends] - squares[starts]) / length - mean ** 2, 0)
    return mean, np.sqrt(variance + 1e-5)


def audit(root, data_path, train_ratio, val_ratio):
    data = Dataset_SDWPF(str(root), data_path=data_path, flag="train", size=[336, 0, 12],
        features="MS", split="time_ratio", fold=1, window_stride=6,
        train_ratio=train_ratio, val_ratio=val_ratio)
    starts = data.window_starts
    ends = starts + data.seq_len
    if not np.all(data.dates[ends + data.pred_len - 1] < data.train_cutoff):
        raise RuntimeError("Training audit touches a selection/validation target")
    wind_mean, wind_sd = window_moments(data.data_x[:, 0], starts, data.seq_len)
    _, power_sd = window_moments(data.data_x[:, -1], starts, data.seq_len)
    target_scale = float(data.scaler.scale_[-1])
    future_move = np.abs(data.data_y[ends[:, None] + np.arange(12), -1]
                         - data.data_x[ends - 1, -1, None]) * target_scale
    digest = hashlib.sha256()
    with (root / data_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    quantiles = [0., .01, .05, .1, .25, .5, .9, 1.]
    bins = []
    for threshold in [.01, .05, .1, .2, .3]:
        mask = power_sd < threshold
        bins.append({"history_sd_below_global_fraction": threshold, "windows": int(mask.sum()),
            "mean_future_move_kw": float(future_move[mask].mean()) if mask.any() else None,
            "fraction_persistence_abs_error": float(future_move[mask].sum() / future_move.sum())})
    return {"scope": "inner_fit_training_histories_and_targets_only", "data_sha256": digest.hexdigest(),
        "train_cutoff": str(data.train_cutoff), "windows": len(starts),
        "target_global_sd_kw": target_scale, "quantiles": quantiles,
        "history_power_sd_kw_quantiles": (np.quantile(power_sd, quantiles) * target_scale).tolist(),
        "wind_global_standardized_history_mean_quantiles": np.quantile(wind_mean, quantiles).tolist(),
        "wind_global_standardized_history_sd_quantiles": np.quantile(wind_sd, quantiles).tolist(),
        "wind_old_pretrain_history_mean": 0., "wind_old_pretrain_history_sd_approx": 1.,
        "residual_scale_bins": bins, "outer_validation_used": False, "sealed_test_accessed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(ROOT / "datasets", "sdwpf_fixed.csv", .5866666666666667, .07333333333333333)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
