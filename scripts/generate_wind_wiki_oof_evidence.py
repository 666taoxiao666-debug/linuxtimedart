#!/usr/bin/env python3
"""Plan and collect forward-OOF event evidence inside an SDWPF training fold.

The source model is trained/selected on earlier timestamps by the companion
shell launcher.  Collection refuses any model whose fit or selection boundary
touches the evidence interval or whose data differs from the outer trend run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from utils.wind_wiki_oof import (forward_oof_starts, inner_ratios,
                                 summarize_event_evidence,
                                 summarize_temporal_event_evidence)


def read_manifest(checkpoint: Path) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    path = checkpoint.parent / "run_manifest.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_pipeline_value(path: Path, key: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    prefix = key + "="
    matches = [line[len(prefix):].strip() for line in path.read_text(encoding="utf-8").splitlines()
               if line.startswith(prefix)]
    if len(matches) != 1 or not matches[0]:
        raise ValueError(f"Expected exactly one {key} in {path}")
    return matches[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_plan(trend_cv_dir: Path, fold: int, seed: int) -> dict:
    checkpoint = Path(read_pipeline_value(
        trend_cv_dir / f"runs/f{fold}_s{seed}/pipeline.env", "FINETUNE_CHECKPOINT"))
    manifest = read_manifest(checkpoint)
    args = manifest["args"]
    if (args.get("data") != "SDWPF" or args.get("sdwpf_split") != "rolling_holdout"
            or args.get("prompt_router") != "trend"
            or int(args.get("sdwpf_fold", -1)) != fold
            or int(args.get("seed", -1)) != seed):
        raise ValueError("TREND_CV_DIR must contain the matching rolling_holdout trend run")
    data_hash = manifest.get("data_file", {}).get("sha256")
    outer_cutoff = manifest.get("datasets", {}).get("train", {}).get("train_cutoff")
    if not data_hash or not outer_cutoff:
        raise ValueError("Trend manifest lacks the dataset hash or outer training cutoff")
    train_ratio, val_ratio = inner_ratios(
        float(args["sdwpf_train_ratio"]), float(args["sdwpf_val_ratio"]),
        fold, int(args["sdwpf_n_folds"]))
    return {
        "schema_version": 1,
        "sdwpf_fold": fold,
        "seed": seed,
        "pred_len": int(args["pred_len"]),
        "outer_train_cutoff": outer_cutoff,
        "outer_trend_checkpoint": str(checkpoint.resolve()),
        "outer_data_sha256": data_hash,
        "inner_train_ratio": train_ratio,
        "inner_val_ratio": val_ratio,
        "protocol": "earlier_fit_then_earlier_selection_then_unseen_train_oof",
    }


def collect(plan: dict, checkpoint: Path, base_config: Path,
            temporal_report: Path | None = None) -> dict:
    if temporal_report is not None and temporal_report.exists():
        raise FileExistsError(temporal_report)
    import torch
    from torch.utils.data import DataLoader

    from run import load_finetuned_model
    from scripts.calibrate_factorized_wiki import restore_experiment
    from utils.wind_regime_wiki import load_wind_regime_wiki_spec

    manifest = read_manifest(checkpoint)
    args = manifest["args"]
    fold, seed = int(plan["sdwpf_fold"]), int(plan["seed"])
    if (args.get("data") != "SDWPF" or args.get("sdwpf_split") != "time_ratio"
            or int(args.get("sdwpf_fold", -1)) != fold
            or int(args.get("seed", -1)) != seed
            or int(args.get("pred_len", -1)) != int(plan["pred_len"])
            or not args.get("utility_factorized")):
        raise ValueError("OOF checkpoint is not a matching trained factorized inner model")
    for name in ("train", "val"):
        if not np.isclose(float(args[f"sdwpf_{name}_ratio"]),
                          float(plan[f"inner_{name}_ratio"]), rtol=0, atol=1e-9):
            raise ValueError(f"OOF checkpoint has the wrong inner {name} ratio")
    inner_hash = manifest.get("data_file", {}).get("sha256")
    if not inner_hash or inner_hash != plan["outer_data_sha256"]:
        raise ValueError("Inner model and outer trend run did not use the same dataset")
    inner_train = manifest.get("datasets", {}).get("train", {})
    fit_cutoff = np.datetime64(inner_train["train_cutoff"], "ns")
    selection_cutoff = np.datetime64(inner_train["val_cutoff"], "ns")
    outer_cutoff = np.datetime64(plan["outer_train_cutoff"], "ns")
    if not fit_cutoff < selection_cutoff < outer_cutoff:
        raise ValueError("Inner fit/selection must finish before outer training ends")
    if args.get("evaluate_test_after_train"):
        raise ValueError("Inner model training must not evaluate the test partition")

    experiment = restore_experiment(args, checkpoint.parent)
    load_finetuned_model(experiment, str(checkpoint))
    experiment.model.eval().requires_grad_(False)
    train_data, _ = experiment._get_data("train")
    if (train_data.train_cutoff != fit_cutoff
            or train_data.val_cutoff != selection_cutoff):
        raise ValueError("Rebuilt dataset boundaries differ from checkpoint manifest")
    starts = forward_oof_starts(
        train_data.segments, train_data.dates, train_data.seq_len,
        train_data.pred_len, int(experiment.args.sdwpf_eval_stride),
        selection_cutoff, outer_cutoff)
    oof_data = copy.copy(train_data)
    oof_data.flag = "train_oof"
    oof_data.window_starts = starts
    loader = DataLoader(oof_data, batch_size=experiment.args.eval_batch_size,
                        shuffle=False, drop_last=False, num_workers=0)
    arrays = {key: [] for key in ("base", "candidates", "target", "available", "turbines")}
    offset = 0
    with torch.no_grad():
        for batch_number, (x, y, _, _) in enumerate(loader, 1):
            with experiment._autocast():
                experiment.model(x.float().to(experiment.device))
            aux = experiment.model._last_utility_aux
            if aux is None or "factor_predictions" not in aux:
                raise RuntimeError("OOF source did not produce factorized event predictions")
            size = int(x.shape[0])
            values = {
                "base": aux["base_prediction"][:, :, 0],
                "candidates": aux["factor_predictions"][:, :, 0],
                "target": y[:, -experiment.args.pred_len:, -1],
                "available": aux["factor_availability"],
            }
            for key, value in values.items():
                arrays[key].append(value.detach().cpu().numpy())
            arrays["turbines"].append(
                train_data.turbines[starts[offset:offset + size] + train_data.seq_len])
            offset += size
            if batch_number == 1 or batch_number % 100 == 0 or offset == len(starts):
                print(f"[OOF] collected {offset}/{len(starts)} windows", flush=True)
    if offset != len(starts):
        raise RuntimeError("OOF loader/window index mismatch")
    arrays = {key: np.concatenate(parts) for key, parts in arrays.items()}
    spec = load_wind_regime_wiki_spec(base_config)
    scenes = spec["scenes"]
    if list(experiment.model.scene_wiki_scene_ids) != [row["id"] for row in scenes]:
        raise ValueError("Checkpoint event order differs from base Wiki config")
    evidence = summarize_event_evidence(
        arrays["base"], arrays["candidates"][:, :, :len(scenes)],
        arrays["target"], arrays["available"][:, :, :len(scenes)],
        arrays["turbines"], scenes, fold=fold, seed=seed,
        as_of_step=int(np.datetime64(outer_cutoff, "m").astype(np.int64) // 10))
    evidence["provenance"] = {
        "method": "forward_chronological_full_model_oof",
        "inner_train_cutoff": str(fit_cutoff),
        "inner_selection_cutoff": str(selection_cutoff),
        "oof_target_start": str(np.asarray(train_data.dates)[starts + train_data.seq_len].min()),
        "oof_target_end": str(np.asarray(train_data.dates)[
            starts + train_data.seq_len + train_data.pred_len - 1].max()),
        "outer_train_cutoff": str(outer_cutoff),
        "oof_windows": int(len(starts)),
        "model_checkpoint": str(checkpoint.resolve()),
        "model_checkpoint_sha256": file_sha256(checkpoint),
        "data_sha256": inner_hash,
        "outer_trend_checkpoint": plan["outer_trend_checkpoint"],
        "selection_labels_end_before_oof": True,
        "validation_or_test_labels_used": False,
    }
    if temporal_report is not None:
        report = summarize_temporal_event_evidence(
            arrays["base"], arrays["candidates"][:, :, :len(scenes)],
            arrays["target"], arrays["available"][:, :, :len(scenes)],
            arrays["turbines"],
            np.asarray(train_data.dates)[starts + train_data.seq_len],
            np.asarray(train_data.dates)[
                starts + train_data.seq_len + train_data.pred_len - 1],
            scenes, fold=fold, seed=seed,
            as_of_step=evidence["as_of_step"],
            evidence_start=selection_cutoff,
            outer_train_cutoff=outer_cutoff,
        )
        report["provenance"] = evidence["provenance"]
        temporal_report.parent.mkdir(parents=True, exist_ok=True)
        temporal_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[OOF] train-only temporal report={temporal_report.resolve()}")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    planning = sub.add_parser("plan")
    planning.add_argument("--trend-cv-dir", type=Path, required=True)
    planning.add_argument("--fold", type=int, required=True)
    planning.add_argument("--seed", type=int, required=True)
    planning.add_argument("--output", type=Path, required=True)
    collecting = sub.add_parser("collect")
    collecting.add_argument("--plan", type=Path, required=True)
    collecting.add_argument("--checkpoint", type=Path, required=True)
    collecting.add_argument("--base-config", type=Path, default=Path("configs/wind_event_factor_wiki.json"))
    collecting.add_argument("--output", type=Path, required=True)
    collecting.add_argument("--temporal-report", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "plan":
        result = make_plan(args.trend_cv_dir, args.fold, args.seed)
    else:
        if args.temporal_report is not None and args.output.exists():
            raise FileExistsError(args.output)
        result = collect(json.loads(args.plan.read_text(encoding="utf-8")),
                         args.checkpoint, args.base_config,
                         temporal_report=args.temporal_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"[OOF] {args.command} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
