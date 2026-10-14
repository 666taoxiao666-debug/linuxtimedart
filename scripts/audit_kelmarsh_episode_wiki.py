#!/usr/bin/env python3
"""Training-only paired probe of timestamped weather-block case weighting.

Reuses frozen checkpoints. Does not access outer-validation/test subsets, train
neural parameters, select thresholds, or revise any old experimental artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from data_provider.kelmarsh_hourly import KelmarshSeries
from models import PromptTimeDART
from scripts.train_kelmarsh_hourly import read, write, sha, model_args, predict, block, model_metrics, gpu_available_to_this_process
from utils import kelmarsh_episode_wiki as wiki


def run(source, output, device):
    if output.exists():
        raise ValueError("Audit output already exists; preserve it rather than rerun")
    p = read(source / "protocol.json")
    if p["protocol_id"] != "kelmarsh_hourly_physical_transition_wiki_v1" or p["pred_len"] != 2:
        raise ValueError("This clock contract is defined for the frozen hourly pilot")
    if sha(ROOT / p["data_path"]) != p["data_sha256"]:
        raise ValueError("Source data changed")
    gpu_available_to_this_process()
    torch.set_num_threads(2)
    rows = []
    for fold in (0, 1, 2):
        directory = source / "runs" / f"f{fold}_s2024"
        series = KelmarshSeries(ROOT / p["data_path"], fold=fold, input_len=p["input_len"], horizon=p["pred_len"],
            stride=p["train_stride"], start=p["start_utc"], end=p["end_utc_exclusive"])
        contract = read(directory / "Wiki_full/utility_policy.json")["contract"]
        if (sha(directory / "PromptTimeDART/checkpoint.pt") != contract["base_checkpoint_sha256"] or
                sha(directory / "PromptTimeDART/manifest.json") != contract["base_manifest_sha256"]):
            raise ValueError("Actual base differs from the utility evidence")
        fit = series.subset("fit")
        state = read(directory / "PromptTimeDART/trained.json")
        if hashlib.sha256(fit.window_starts.tobytes()).hexdigest() != state["contract"]["fit_window_hash"]:
            raise ValueError("Base training windows changed")
        np.testing.assert_array_equal(state["contract"]["scaler_mean"], series.scaler.mean_)
        np.testing.assert_array_equal(state["contract"]["scaler_scale"], series.scaler.scale_)
        base = PromptTimeDART.Model(model_args("PromptTimeDART", fit, p, device)).to(device)
        base.load_state_dict(torch.load(directory / "PromptTimeDART/checkpoint.pt", map_location=device, weights_only=False)["model"], strict=True)
        memory_ds = series.subset("memory")
        _, anchor = memory_ds.truth_and_anchor()
        memory_data = block(memory_ds, anchor, p["rules"], donors=True)
        memory = read(directory / "Wiki_full/expert.json")
        blocks = {}
        for stage in ("calibration_early", "calibration_late"):
            ds = series.subset(stage)
            blocks[stage] = block(ds, predict(base, ds, 2024, device), p["rules"], donors=True)
        for mode in ("turbine_control", "episode"):
            model = wiki.make_model(memory, memory_data, contract, mode=mode, block_hours=24)
            policy = wiki.fit_policy(model, list(blocks.values()), contract, p["utility"])
            write(output / f"f{fold}" / mode / "expert.json", model)
            write(output / f"f{fold}" / mode / "utility_policy.json", policy)
            for stage, data in blocks.items():
                candidate = wiki.candidates(model, data, contract)
                applied = wiki.apply_with_policy(model, policy, candidate, data, contract)
                active = applied["action"] > 0
                row = dict(fold=fold, mode=mode, stage=stage,
                    scope="training utility data reused for fit; descriptive, not independent accuracy evidence",
                    expert_id=model["expert_id"], accepted_cells=sum(r["accepted"] for r in policy["entries"].values()),
                    selected=model_metrics(applied["prediction"], data["truth"], data["anchor"], p["rated_power_kw"]),
                    base=model_metrics(data["reference"], data["truth"], data["anchor"], p["rated_power_kw"]),
                    intervention_pct=float(100 * active.any(1).mean()), events=[])
                for k, event in enumerate(wiki.legacy.EVENTS):
                    mask = candidate["physical"][:, 0, k]
                    if not mask.any():
                        continue
                    row["events"].append(dict(event=event, query_windows=int(mask.sum()),
                        candidate=model_metrics(candidate["prediction"][mask, :, k], data["truth"][mask], data["anchor"][mask], p["rated_power_kw"]),
                        base=model_metrics(data["reference"][mask], data["truth"][mask], data["anchor"][mask], p["rated_power_kw"]),
                        median_weather_blocks=float(np.median(candidate["weather_blocks"][mask, 0, k])),
                        median_kish_weight_count=float(np.median(candidate["kish_weight_count"][mask, 0, k]))))
                rows.append(row)
                print(json.dumps({key: value for key, value in row.items() if key not in ("expert_id", "events", "base", "selected")}) +
                      f" MAE={row['selected']['mae']:.6f} RMSE={row['selected']['rmse']:.6f}", flush=True)
        del base
        torch.cuda.empty_cache()
    write(output / "audit.json", dict(rows=rows, source_protocol_sha256=sha(source / "protocol.json"),
        implementation_sha256=wiki.implementation_digest(), script_sha256=sha(Path(__file__)),
        block_hours=24, seed=2024, outer_validation_evaluated=False, sealed_test_evaluated=False,
        inference_only=True, neural_retraining=False, threshold_search=False,
        interpretation="Paired training-side mechanism probe, not a new claimed forecast result"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve(), torch.device(f"cuda:{args.gpu}"))
