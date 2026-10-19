#!/usr/bin/env python3
"""Inner-gated matched-schedule and pretraining-value factorial on the unchanged v21 architecture."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
import numpy as np
import pandas as pd
import torch

from models.PowerAnchoredTimeDART import Model
from scripts import train_kelmarsh_24h4h as prior
from scripts import train_kelmarsh_power_anchor as anchor
from scripts import train_kelmarsh_recent_numeric as recent

shared = prior.shared
PROTOCOL = ROOT / "configs/kelmarsh_matched_recipe_protocol_v24.json"
PROTOCOL_SHA256 = "0e4a97e5f414289af1ce77ab8eb263110986cf7de04a7a23fbe535be72d570ea"
POINTER = ROOT / "outputs/logs/Kelmarsh/matched_recipe_latest.txt"
PRETRAIN_SEED = 2024  # the only seed for which completed v20 pretraining exists
SOURCE_FILES = tuple(dict.fromkeys(("scripts/train_kelmarsh_matched_recipe.py",) + anchor.SOURCE_FILES))
PAIRS = (("schedule_effect", "PretrainedMatched", "PretrainedOriginal"),
         ("pretraining_effect", "PretrainedMatched", "ScratchMatched"),
         ("combined_effect", "ScratchMatched", "PretrainedOriginal"))


def protocol():
    recipe = shared.read(PROTOCOL)
    if hashlib.sha256(PROTOCOL.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen v24 recipe changed")
    base = ROOT / recipe["base_recipe"]
    if hashlib.sha256(base.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != recipe["base_recipe_sha256"]:
        raise ValueError("v21 recipe changed")
    v21, effective = anchor.protocol()
    effective = copy.deepcopy(effective)
    effective.update(protocol_id=recipe["protocol_id"], executed_models=list(recipe["arms"]))
    for arm in (recipe["primary_candidate"], recipe["reference_arm"], recipe["control_arm"]):
        if arm not in recipe["arms"]:
            raise ValueError("Undeclared arm: " + arm)
    for name, cfg in recipe["arms"].items():
        if cfg["init"] not in ("pretrained", "scratch") or int(cfg["epochs"]) < 1:
            raise ValueError("Invalid arm declaration: " + name)
    if recipe["arms"][recipe["reference_arm"]]["epochs"] != effective["training"]["epochs"]:
        raise ValueError("The reference arm must be the unchanged v21 schedule")
    if recipe["folds"] != v21["folds"] or len(set(recipe["seeds"])) != len(recipe["seeds"]):
        raise ValueError("Folds must match v21 and seeds must be distinct")
    return recipe, v21, effective


def arm_protocol(effective, arm):
    """Only the declared epoch count differs; T_max of CosineAnnealingLR follows it by construction."""
    result = copy.deepcopy(effective)
    result["training"]["epochs"] = int(arm["epochs"])
    return result


def residual_slope(prediction, truth, anchor_prediction):
    """Least-squares slope of realised residual on predicted correction; 1 calibrated, below 1 overdispersed."""
    correction = np.asarray(prediction, dtype=float) - anchor_prediction
    residual = np.asarray(truth, dtype=float) - anchor_prediction

    def one(c, r):
        denominator = float(np.sum(c * c))
        return float(np.sum(c * r) / denominator) if denominator > 0 else None
    return {"pooled": one(correction, residual),
            "per_hour": [one(correction[:, h], residual[:, h]) for h in range(correction.shape[1])]}


def decision(rows, settings):
    jointly = [r["candidate"]["mae"] < r["reference"]["mae"] and
               r["candidate"]["rmse"] < r["reference"]["rmse"] for r in rows]
    names = ("candidate", "reference", "persistence") + (("dlinear",) if all("dlinear" in r for r in rows) else ())
    means = {name: {key: float(np.mean([r[name][key] for r in rows])) for key in ("mae", "rmse")} for name in names}
    passed = (sum(jointly) >= settings["minimum_jointly_improved_folds_vs_reference_arm"] and
              all(means["candidate"][key] < means[name][key]
                  for name in ("reference", "persistence") for key in ("mae", "rmse")))
    return {"passed": bool(passed), "jointly_improved_folds": int(sum(jointly)), "macro_inner": means,
            "rows": rows, "selection_scope": "chronological internal selection only",
            "dlinear_is_reported_not_gated": True}


def paired_effect(records, higher, lower):
    """Per (fold, seed) unit difference higher - lower; negative means the first arm is better."""
    index = {(r["fold"], r["seed"], r["arm"]): r["metrics"] for r in records}
    units = sorted({(f, s) for f, s, a in index if a == higher} & {(f, s) for f, s, a in index if a == lower})
    result = {"units": len(units)}
    for key in ("mae", "rmse"):
        diffs = np.array([index[(f, s, higher)][key] - index[(f, s, lower)][key] for f, s in units])
        result[key] = {"mean_difference": float(diffs.mean()) if len(diffs) else None,
                       "std_difference": float(diffs.std(ddof=1)) if len(diffs) > 1 else None,
                       "units_where_first_is_better": int((diffs < 0).sum()), "units": int(len(diffs))}
    return result


def arm_summary(records, arm, ensembles):
    rows = [r for r in records if r["arm"] == arm]
    folds = sorted({r["fold"] for r in rows})
    seed_mean = {key: float(np.mean([np.mean([r["metrics"][key] for r in rows if r["fold"] == f])
                                     for f in folds])) for key in ("mae", "rmse")}
    seeds = sorted({r["seed"] for r in rows})
    seed_macro = {key: [float(np.mean([r["metrics"][key] for r in rows if r["fold"] == f and r["seed"] == s])
                              for f in folds) for s in seeds] for key in ("mae", "rmse")}
    slopes = [r["slope"]["pooled"] for r in rows if r["slope"]["pooled"] is not None]
    summary = {"macro_single_seed_mean": seed_mean,
               "macro_by_seed": {key: dict(zip(map(str, seeds), values)) for key, values in seed_macro.items()},
               "seed_std_of_macro": {key: float(np.std(values, ddof=1)) if len(values) > 1 else None
                                     for key, values in seed_macro.items()},
               "mean_best_epoch": float(np.mean([r["best_epoch"] for r in rows])),
               "best_epoch_zero_runs": int(sum(r["best_epoch"] == 0 for r in rows)),
               "mean_pooled_residual_slope": float(np.mean(slopes)) if slopes else None}
    if arm in ensembles:
        values = ensembles[arm]
        summary["macro_seed_mean_ensemble"] = {key: float(np.mean([values[f][key] for f in folds]))
                                               for key in ("mae", "rmse")}
        summary["ensemble_pooled_residual_slope"] = float(np.mean(
            [values[f]["slope"] for f in folds if values[f]["slope"] is not None])) if any(
                values[f]["slope"] is not None for f in folds) else None
    return summary


def verify_references(recipe, v21, effective):
    reference = ROOT / v21["reference_dir"]
    if shared.sha(reference / "completed.json") != v21["reference_completed_sha256"]:
        raise ValueError("Completed v20 reference changed")
    prior.verify_completed_suite(reference)
    previous = ROOT / recipe["v21_reference_dir"]
    if shared.sha(previous / "completed.json") != recipe["v21_reference_completed_sha256"]:
        raise ValueError("Completed v21 reference changed")
    prior.verify_completed_suite(previous)
    if shared.sha(ROOT / effective["data_path"]) != effective["data_sha256"]:
        raise ValueError("Prepared source changed")
    return reference, previous


def fold_inputs(v21, effective, fold):
    reference = ROOT / v21["reference_dir"]
    series = shared.KelmarshSeries(ROOT / effective["data_path"], fold=fold, input_len=24, horizon=4,
        stride=3, start=effective["start_utc"], end=effective["end_utc_exclusive"])
    data = anchor.train_datasets_only(series)
    ref_run = reference / "runs" / f"f{fold}_s{PRETRAIN_SEED}"
    pretrain = ref_run / "pretrain/checkpoint.pt"
    if shared.sha(pretrain) != v21["pretrain_checkpoint_sha256"][str(fold)]:
        raise ValueError("Pretrained expert changed")
    contract = shared.read(ref_run / "pretrain/trained.json")["contract"]
    for key, stage in (("fit_window_hash", "fit"), ("selection_window_hash", "selection")):
        if hashlib.sha256(data[stage].window_starts.tobytes()).hexdigest() != contract[key]:
            raise ValueError("Pretrain and candidate training queue differ")
    np.testing.assert_array_equal(series.scaler.mean_, contract["scaler_mean"])
    np.testing.assert_array_equal(series.scaler.scale_, contract["scaler_scale"])
    return series, data, pretrain


def dlinear_inner(reference, fold):
    run = reference / "runs" / f"f{fold}_s{PRETRAIN_SEED}" / "DLinear"
    epoch = shared.read(run / "trained.json")["best_epoch"]
    return next(r["selection"] for r in shared.read(run / "training_history.json") if r["epoch"] == epoch)


def check():
    recipe, v21, effective = protocol()
    verify_references(recipe, v21, effective)
    counts = {}
    for fold in recipe["folds"]:
        _, data, _ = fold_inputs(v21, effective, fold)
        counts[fold] = {name: len(dataset) for name, dataset in data.items()}
    print("[CHECK] v24 protocol, references, pretrain checkpoints and window identities verified", flush=True)
    print(f"[CHECK] windows per fold {counts}", flush=True)
    print(f"[CHECK] runs {len(recipe['folds']) * len(recipe['seeds']) * len(recipe['arms'])}: "
          f"folds={recipe['folds']} seeds={recipe['seeds']} arms={list(recipe['arms'])}", flush=True)


def ensemble_inner(directory, recipe, fold, arm, rated):
    predictions = []
    for seed in recipe["seeds"]:
        with np.load(directory / "runs" / f"f{fold}_s{seed}" / arm / "inner_predictions.npz", allow_pickle=False) as saved:
            predictions.append(saved["prediction"])
            truth, base = saved["truth"], saved["anchor"]
    mean = np.mean(predictions, axis=0)
    metrics = shared.model_metrics(mean, truth, base, rated)
    metrics["slope"] = residual_slope(mean, truth, base)["pooled"]
    return metrics


def outer_evaluation(directory, recipe, v21, effective, prepared, device):
    reference, previous = ROOT / v21["reference_dir"], ROOT / recipe["v21_reference_dir"]
    arm_name = recipe["primary_candidate"]
    protocol_arm = arm_protocol(effective, recipe["arms"][arm_name])
    records = []
    for fold, (series, data) in prepared.items():
        data["validation"] = series.subset("validation")
        recent.assert_reference_cohort(series, data, reference)
        truth, base = data["validation"].truth_and_anchor()
        frame = pd.DataFrame({"forecast_start": data["validation"].target_times()[:, 0]})
        with np.load(previous / "runs" / f"f{fold}_s{PRETRAIN_SEED}/PowerAnchoredTimeDART/raw_predictions.npz",
                     allow_pickle=False) as saved:
            np.testing.assert_array_equal(truth, saved["truth"])
            v21_prediction = saved["prediction"]
        with np.load(reference / "runs" / f"f{fold}_s{PRETRAIN_SEED}/DLinear/raw_predictions.npz",
                     allow_pickle=False) as saved:
            np.testing.assert_array_equal(truth, saved["truth"])
            dlinear_prediction = saved["prediction"]
        predictions = []
        for seed in recipe["seeds"]:
            output = directory / "runs" / f"f{fold}_s{seed}" / arm_name
            prior.wait_for_gpu(directory, fold, seed, "OUTER_EVALUATION")
            model = Model(shared.model_args("PromptTimeDART", data["fit"], protocol_arm, device)).to(device)
            model.load_state_dict(torch.load(output / "checkpoint.pt", map_location=device,
                                             weights_only=False)["model"], strict=True)
            prediction = shared.predict(model, data["validation"], seed, device)
            predictions.append(prediction)
            state = shared.read(output / "trained.json")
            result = prior.write_report(arm_name, prediction, data["validation"], output, effective, state)
            result["paired_vs_v21"] = prior.paired_forecast_statistics(
                prediction, v21_prediction, truth, frame, **effective["statistics"])
            shared.write(output / "result.json", result)
            shared.write(output / "completed.json", {str(f.relative_to(output)): shared.sha(f)
                for f in output.rglob("*") if f.is_file() and f.name != "completed.json"})
            records.append({"fold": fold, "name": f"{arm_name}_s{seed}", **shared.model_metrics(
                prediction, truth, base, effective["rated_power_kw"])})
            del model
            gc.collect()
            torch.cuda.empty_cache()
        ensemble = np.mean(predictions, axis=0)
        output = directory / "runs" / f"ensemble_f{fold}" / (arm_name + "SeedMean")
        result = prior.write_report(arm_name + "SeedMean", ensemble, data["validation"], output, effective)
        for label, comparison in (("paired_vs_v21", v21_prediction), ("paired_vs_dlinear", dlinear_prediction)):
            result[label] = prior.paired_forecast_statistics(
                ensemble, comparison, truth, frame, **effective["statistics"])
        shared.write(output / "result.json", result)
        records.append({"fold": fold, "name": arm_name + "SeedMean", **shared.model_metrics(
            ensemble, truth, base, effective["rated_power_kw"])})
        records.append({"fold": fold, "name": "v21_single_seed", **shared.model_metrics(
            v21_prediction, truth, base, effective["rated_power_kw"])})
        records.append({"fold": fold, "name": "DLinear_v20", **shared.model_metrics(
            dlinear_prediction, truth, base, effective["rated_power_kw"])})
        records.append({"fold": fold, "name": "Persistence", **shared.model_metrics(
            base, truth, base, effective["rated_power_kw"])})
        prior.update_summary(directory)
    shared.write(directory / "outer_records.json", records)
    return records


def run(directory, gpu=0):
    import fcntl
    recipe, v21, effective = protocol()
    directory.mkdir(parents=True, exist_ok=True)
    POINTER.parent.mkdir(parents=True, exist_ok=True)
    with (POINTER.parent / "matched_recipe.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = {"recipe": recipe, "v21_recipe": v21, "effective": effective}
        if (directory / "protocol.json").exists() and shared.read(directory / "protocol.json") != record:
            raise ValueError("Frozen protocol changed on resume")
        shared.write(directory / "protocol.json", record)
        source = {name: shared.sha(ROOT / name) for name in SOURCE_FILES}
        if (directory / "source_hashes.json").exists() and shared.read(directory / "source_hashes.json") != source:
            raise ValueError("Implementation changed; never loosen the source lock")
        shared.write(directory / "source_hashes.json", source)
        reference, previous = verify_references(recipe, v21, effective)
        POINTER.write_text(str(directory), encoding="utf-8")
        if (directory / "completed.json").exists():
            prior.verify_completed_suite(directory)
            print("[REUSE] Completed v24, no retraining", flush=True)
            return
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        (directory / "status.env").write_text("STATUS=RUNNING\n", encoding="utf-8")
        device, prepared, records = torch.device(f"cuda:{gpu}"), {}, []
        rated = effective["rated_power_kw"]
        persistence, dlinear = {}, {}
        for fold in recipe["folds"]:
            series, data, pretrain = fold_inputs(v21, effective, fold)
            truth, base = data["selection"].truth_and_anchor()
            persistence[fold] = shared.model_metrics(base, truth, base, rated)
            dlinear[fold] = dlinear_inner(reference, fold)
            reference_count = shared.read(previous / "runs" / f"f{fold}_s{PRETRAIN_SEED}/PowerAnchoredTimeDART/manifest.json")["parameters"]
            for seed in recipe["seeds"]:
                for arm, cfg in recipe["arms"].items():
                    output = directory / "runs" / f"f{fold}_s{seed}" / arm
                    prior.wait_for_gpu(directory, fold, seed, "INNER_" + arm)
                    model, state = shared.train_stage("PromptTimeDART", data["fit"], data["selection"],
                        arm_protocol(effective, cfg), output, seed, device,
                        initial=pretrain if cfg["init"] == "pretrained" else None, model_factory=Model)
                    prediction = shared.predict(model, data["selection"], seed, device)
                    metrics = shared.model_metrics(prediction, truth, base, rated)
                    count = shared.read(output / "manifest.json")["parameters"]
                    if count != reference_count:
                        raise ValueError("An arm changed the parameter budget")
                    if not (output / "inner_predictions.npz").exists():
                        np.savez_compressed(output / "inner_predictions.npz", prediction=prediction,
                                            truth=truth, anchor=base)
                    row = {"fold": fold, "seed": seed, "arm": arm, "init": cfg["init"], "epochs": cfg["epochs"],
                           "best_epoch": state["best_epoch"], "epochs_completed": state["epochs_completed"],
                           "metrics": metrics, "slope": residual_slope(prediction, truth, base),
                           "parameter_count": count}
                    shared.write(output / "inner_metrics.json", row)
                    records.append(row)
                    shared.write(directory / "inner_records.json", records)
                    print(f"[INNER] f{fold} s{seed} {arm} MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f} "
                          f"best_epoch={state['best_epoch']}", flush=True)
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
            prepared[fold] = (series, data)
        arms = recipe["arms"]
        ensembles = {arm: {fold: ensemble_inner(directory, recipe, fold, arm, rated) for fold in recipe["folds"]}
                     for arm in arms}
        rows = []
        for fold in recipe["folds"]:
            def seed_mean(arm):
                return {key: float(np.mean([r["metrics"][key] for r in records
                                            if r["fold"] == fold and r["arm"] == arm])) for key in ("mae", "rmse")}
            rows.append({"fold": fold, "candidate": seed_mean(recipe["primary_candidate"]),
                         "reference": seed_mean(recipe["reference_arm"]),
                         "persistence": {key: persistence[fold][key] for key in ("mae", "rmse")},
                         "dlinear": {key: dlinear[fold][key] for key in ("mae", "rmse")}})
        gate = decision(rows, recipe["outer_gate"])
        reproduction = []
        for r in records:
            if r["arm"] == recipe["reference_arm"] and r["seed"] == PRETRAIN_SEED:
                old = shared.read(previous / "runs" / f"f{r['fold']}_s{PRETRAIN_SEED}/PowerAnchoredTimeDART/inner_metrics.json")
                reproduction.append({"fold": r["fold"], **{f"{key}_abs_difference": abs(r["metrics"][key] - old["candidate"][key])
                                                           for key in ("mae", "rmse")}})
        factorial = {"arms": {arm: arm_summary(records, arm, ensembles) for arm in arms},
                     "paired_effects": {name: {"first": a, "second": b, **paired_effect(records, a, b)}
                                        for name, a, b in PAIRS},
                     "reproduces_v21_seed_2024": reproduction,
                     "persistence_inner": persistence, "dlinear_inner_seed_2024": dlinear,
                     "scope": "chronological internal selection only; negative differences favour the first arm"}
        shared.write(directory / "factorial.json", factorial)
        shared.write(directory / "gate.json", gate)
        print(f"[TRAIN_ONLY_GATE] passed={gate['passed']} jointly_improved_folds={gate['jointly_improved_folds']}", flush=True)
        outer = None
        if gate["passed"]:
            outer = outer_evaluation(directory, recipe, v21, effective, prepared, device)
        shared.write(directory / "result.json", {"outer_evaluated": gate["passed"], "gate": gate,
            "v21_reference": str(previous), "wiki_enabled": False, "labels_modified": False,
            "note": "A failed inner gate is a negative result, not permission for another epoch count, threshold or seed."})
        shared.write(directory / "progress.json", {"stage": "COMPLETED", "outer_evaluated": gate["passed"]})
        (directory / "status.env").write_text("STATUS=COMPLETED\nEXIT_CODE=0\n", encoding="utf-8")
        shared.write(directory / "completed.json", {str(f.relative_to(directory)): shared.sha(f)
            for f in directory.rglob("*") if f.is_file() and f.name not in
            ("suite.lock", "launch.log", "launcher.pid", "status.env", "progress.json", "last.pt")})
        print(f"[COMPLETED] {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--check", action="store_true", help="CPU-only verification of hashes, references and window identities")
    args = parser.parse_args()
    if args.check:
        check()
        return
    if args.status:
        if not POINTER.exists():
            print("STATUS=NOT_STARTED")
            return
        directory = Path(POINTER.read_text().strip())
        print(f"RESULT_DIR={directory}")
        for name in ("status.env", "progress.json"):
            if (directory / name).exists():
                print((directory / name).read_text())
        return
    if args.output is None:
        raise ValueError("Independent output directory required")
    try:
        run(args.output.resolve(), args.gpu)
    except BlockingIOError:
        print("[BUSY] Existing v24 task left unchanged", flush=True)
    except Exception as error:
        if args.output.exists():
            (args.output / "status.env").write_text(f"STATUS=FAILED\nERROR={type(error).__name__}: {error}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
