#!/usr/bin/env python3
"""Train-only information probe: no outer prediction, threshold or model selection."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.linear_model import Ridge
from data_provider.kelmarsh_hourly import KelmarshSeries
from scripts.train_kelmarsh_hourly import read, write, sha, model_metrics
from utils import kelmarsh_hourly_wiki as wiki


PHASE_NAMES = ["last_wind_mps", "wind_last_hour_change_mps", "wind_last_three_hour_change_mps",
               "power_last_hour_change_ratio", "last_rotor_rpm", "rotor_last_three_hour_change_rpm"]


def phase_coordinates(dataset, capacity):
    if dataset.seq_len < 4:
        raise ValueError("Four historical observations required")
    row = dataset.window_starts + dataset.seq_len - 1
    raw = dataset.series.raw
    return np.column_stack((raw[row, 0], raw[row, 0] - raw[row - 1, 0], raw[row, 0] - raw[row - 3, 0],
                            (raw[row, -1] - raw[row - 1, -1]) / capacity,
                            raw[row, 5], raw[row, 5] - raw[row - 3, 5]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/kelmarsh_hourly_protocol_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    if sha(ROOT / p["data_path"]) != p["data_sha256"]:
        raise ValueError("Source data identity changed")
    capacity, results = p["rated_power_kw"], []
    for fold in p["folds"]:
        series = KelmarshSeries(ROOT / p["data_path"], fold=fold, input_len=p["input_len"], horizon=p["pred_len"],
                               stride=p["train_stride"], start=p["start_utc"], end=p["end_utc_exclusive"])
        stages = {}
        for name in ("memory", "calibration_early", "calibration_late"):
            data = series.subset(name)
            assert data.target_times().max() < series.bounds["train_end"]
            truth, anchor = data.truth_and_anchor()
            context = data.physical_context()
            x = wiki.keys(context, anchor, capacity, True)
            phases = phase_coordinates(data, capacity)
            ids = data.turbines[data.window_starts]
            mask = ids != 6
            stages[name] = dict(x=x[mask], phases=phases[mask], truth=truth[mask], anchor=anchor[mask], ids=ids[mask])
        for variant in ("legacy_coordinates", "plus_instantaneous_phase"):
            def coordinates(data):
                return data["x"] if variant == "legacy_coordinates" else np.column_stack((data["x"], data["phases"]))
            memory = stages["memory"]
            x = coordinates(memory)
            mean, scale = x.mean(0), x.std(0)
            scale = np.where(scale > 1e-6, scale, 1.)
            units, counts = np.unique(memory["ids"], return_counts=True)
            weights = 1. / counts[np.searchsorted(units, memory["ids"])]
            model = Ridge(alpha=p["numeric_residual_ridge_alpha"]).fit((x - mean) / scale,
                (memory["truth"] - memory["anchor"]) / capacity, sample_weight=weights)
            for name in ("calibration_early", "calibration_late"):
                data = stages[name]
                prediction = data["anchor"] + capacity * model.predict((coordinates(data) - mean) / scale)
                results.append(dict(fold=fold, variant=variant, stage=name, windows=len(prediction),
                    metrics=model_metrics(prediction, data["truth"], data["anchor"], capacity)))
    write(args.output, dict(scope="donor-only outer training; no outer validation/test predictions or tuning",
        model="fixed-alpha global Ridge information probe, not proposed full Wiki", alpha=p["numeric_residual_ridge_alpha"],
        data_sha256=p["data_sha256"], phase_features=PHASE_NAMES, results=results))
    for row in results:
        print(row["fold"], row["variant"], row["stage"], f"MAE={row['metrics']['mae']:.6f}",
              f"RMSE={row['metrics']['rmse']:.6f}", flush=True)


if __name__ == "__main__":
    main()
