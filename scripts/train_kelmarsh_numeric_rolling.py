#!/usr/bin/env python3
"""Frozen numerical-only expanding-origin selection; never outer-guided search."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_kelmarsh_numeric_foundation as f
from utils import numeric_rolling as rolling

PROTOCOL_ID = "kelmarsh_h2_numeric_expanding_origin_v12"
METRICS = ("mae", "mse", "rmse", "r2", "mae_skill_pct", "rmse_skill_pct")


def previous_evidence(spec):
    directory = ROOT / spec["previous_directory"]
    if f.old.sha(directory / "completed.json") != spec["previous_completed_sha256"]:
        raise ValueError("Previous completed evidence changed")
    complete = f.old.read(directory / "completed.json")
    if f.old.sha(directory / "protocol.json") != spec["previous_protocol_sha256"]:
        raise ValueError("Previous protocol changed")
    for name, expected in complete["artifacts"].items():
        if f.old.sha(directory / name) != expected:
            raise ValueError("Previous completed artifact changed: " + name)
    if f.old.sha(ROOT / spec["training_evidence_path"]) != spec["training_evidence_sha256"]:
        raise ValueError("Training-only diagnostic evidence changed")
    old_spec = f.old.read(directory / "protocol.json")
    for key in ("seed", "fold", "training", "candidates", "ridge_alphas", "source_directory", "source_protocol_sha256"):
        if spec[key] != old_spec[key]:
            raise ValueError("Unapproved change to frozen candidate pool/hyperparameters: " + key)
    prepared = f.old.read(directory / "prepared.json")
    for name, expected in prepared["identity"]["source_sha256"].items():
        if f.old.sha(ROOT / name) != expected:
            raise ValueError("Existing numerical implementation changed: " + name)
    return directory


def verified_stage(name, directory):
    done = f.old.read(directory / "trained.json")
    if done["model"] != name:
        raise ValueError("Completed origin model identity changed")
    artifacts = done.get("artifacts", {"model.npz": done.get("model_sha256")})
    if any(f.old.sha(directory / key) != value for key, value in artifacts.items()):
        raise ValueError("Completed origin artifact changed")
    return done


def stage_record(name, directory):
    done = verified_stage(name, directory)
    record = dict(model=name, parameters=done["parameters"])
    if name.startswith("DeltaRidge"):
        record.update(alpha=done["alpha"], metrics={key: done["metrics"][key] for key in METRICS})
    else:
        history = f.old.read(directory / "training_history.json")
        record["epochs"] = {str(row["epoch"]): {key: row["selection"][key] for key in METRICS}
                            for row in history if row["epoch"] > 0}
        if len(record["epochs"]) != done["epochs_completed"]:
            raise ValueError("Incomplete epoch evaluation history")
    return record


def reusable_stage(directory, fit, selection, initial, name):
    marker = directory / "trained.json"
    if not marker.exists():
        return False
    done = f.old.read(marker)
    if done["model"] != name:
        return False
    contract = done["contract"]
    query = f.stage_identity(selection) if selection is not None else None
    if contract["fit"] != f.stage_identity(fit) or contract["selection"] != query:
        return False
    if name == "PromptExpanded" and contract["original_pretrain_sha256"] != f.old.sha(initial):
        raise ValueError("Reusable Prompt initialization changed")
    return True


def refit_prediction(selected, all_train, validation, p, spec, output, identity, initial, device, resources, previous=None):
    name = selected["model"]
    if name == "Persistence":
        return f.foundation.observed_anchor(validation)
    if previous is not None and reusable_stage(previous / "outer_refit", all_train, None, initial, name):
        done = verified_stage(name, previous / "outer_refit")
        same = (done.get("alpha") == selected["alpha"] if name.startswith("DeltaRidge") else
                done["contract"].get("fixed_refit_epochs") == selected["best_epoch"])
        if same:
            old_selected = f.old.read(previous / "selected.json")
            if old_selected["model"] != name or old_selected["best_epoch"] != selected["best_epoch"]:
                raise ValueError("Previous forecast is not the verified refit")
            cached = previous / "reports/SelectedNumeric/raw_predictions.npz"
            registered = f.old.read(previous / "completed.json")["artifacts"]["reports/SelectedNumeric/raw_predictions.npz"]
            if f.old.sha(cached) != registered:
                raise ValueError("Previous frozen prediction changed")
            truth, anchor = validation.truth_and_anchor()
            with f.np.load(cached, allow_pickle=False) as saved:
                for key, expected in dict(truth=truth, persistence=anchor, times=validation.target_times(),
                    turbines=validation.turbines[validation.window_starts], window_starts=validation.window_starts).items():
                    f.np.testing.assert_array_equal(saved[key], expected)
                prediction = saved["prediction"].copy()
            f.old.write(output / "refit_reuse.json", dict(source=str(previous),
                trained_sha256=f.old.sha(previous / "outer_refit/trained.json"), selected_before_prediction_reuse=True))
            print("[REUSE_REFIT] exact frozen recipe already completed; no retraining", flush=True)
            return prediction
    if name.startswith("DeltaRidge"):
        model, _ = f.ridge_fit(name, all_train, None, selected["alpha"], p, output / "outer_refit", identity)
        return f.foundation.ridge_predict(model, validation)
    model, _ = f.neural_fit(name, all_train, None, p, spec, output / "outer_refit", identity, initial, device, resources,
                             fixed_epochs=selected["best_epoch"])
    return f.predict(model, validation, device, spec["seed"], resources)


def run(protocol, output, shared=True):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / "suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = f.old.read(protocol)
        if (spec["protocol_id"] != PROTOCOL_ID or spec["wiki_enabled"] or spec["fold"] != 1 or spec["seed"] != 2024
                or spec["origins"] != list(rolling.ORIGINS) or spec["selection_ends"] != list(rolling.ENDS)):
            raise ValueError("Unknown numerical rolling pilot")
        saved = output / "protocol.json"
        if saved.exists() and f.old.sha(saved) != f.old.sha(protocol):
            raise ValueError("Frozen rolling protocol changed")
        if not saved.exists():
            saved.write_bytes(protocol.read_bytes())
        if (output / "completed.json").exists():
            complete = f.old.read(output / "completed.json")
            if complete["protocol_sha256"] != f.old.sha(saved) or any(f.old.sha(output / k) != v for k, v in complete["artifacts"].items()):
                raise ValueError("Completed rolling suite changed")
            print("[REUSE_SUITE] completed; no new training", flush=True)
            return
        resources = f.GPURuntimePolicy(shared=shared, on_change=lambda value: f.old.write(output / "resource_policy.json", value))
        resources.configure()
        f.old.write(output / "status.json", dict(status="RUNNING", pid=os.getpid()))
        f.old.write(output / "progress.json", dict(stage="causal_preflight"))
        previous = previous_evidence(spec)
        original, p = f.source_configuration(spec)
        series = f.old.KelmarshSeries(ROOT / p["data_path"], fold=1, input_len=72, horizon=2, stride=3,
                                      start=p["start_utc"], end=p["end_utc_exclusive"])
        origins, all_train = rolling.rolling_partitions(series)
        initial = original / "pretrain/checkpoint.pt"
        pretrain = f.old.read(original / "pretrain/manifest.json")
        initializer = rolling.initializer_audit(series, origins["o70"]["selection"], pretrain["contract"])
        original_manifest = f.old.read(original / "PromptTimeDART/manifest.json")
        f.np.testing.assert_array_equal(series.scaler.mean_, original_manifest["contract"]["scaler_mean"])
        f.np.testing.assert_array_equal(series.scaler.scale_, original_manifest["contract"]["scaler_scale"])
        files = ("scripts/train_kelmarsh_numeric_rolling.py", "utils/numeric_rolling.py", "utils/regime_labels.py")
        source_hashes = f.old.read(previous / "prepared.json")["identity"]["source_sha256"]
        identity = dict(protocol_sha256=f.old.sha(saved), source_completed_sha256=f.old.sha(original / "completed.json"),
            previous_completed_sha256=f.old.sha(previous / "completed.json"), data_sha256=p["data_sha256"],
            initial_pretrain_sha256=f.old.sha(initial), source_sha256={**source_hashes, **{key: f.old.sha(ROOT / key) for key in files}},
            partitions={key: {part: f.stage_identity(value[part]) for part in ("fit", "selection")} for key, value in origins.items()},
            outer_refit=f.stage_identity(all_train))
        prepared = output / "prepared.json"
        if prepared.exists() and f.old.read(prepared)["identity"] != identity:
            raise ValueError("Rolling data/implementation changed on resume")
        f.old.write(prepared, dict(identity=identity, initializer_audit=initializer, source_data_audit=series.audit, wiki_enabled=False))
        device = f.torch.device("cuda:0")
        f.torch.backends.cudnn.benchmark, f.torch.backends.cudnn.deterministic = False, True
        records, baselines, reuse = {}, {}, []
        for origin, data in origins.items():
            fit, selection = data["fit"], data["selection"]
            truth, anchor = selection.truth_and_anchor()
            baselines[origin] = f.old.model_metrics(anchor, truth, anchor, p["rated_power_kw"])
            records[origin] = {}
            for name in spec["candidates"]:
                f.old.write(output / "progress.json", dict(stage="inner", origin=origin, model=name))
                directory = output / "origins" / origin / name
                cached = previous / "inner" / name
                if reusable_stage(cached, fit, selection, initial, name):
                    records[origin][name] = stage_record(name, cached)
                    reuse.append(dict(origin=origin, model=name, source=str(cached.relative_to(ROOT)),
                                      trained_sha256=f.old.sha(cached / "trained.json")))
                    print(f"[REUSE_ORIGIN] {origin} {name}; verified same training/labels/code", flush=True)
                else:
                    if name.startswith("DeltaRidge"):
                        alpha = float(name.removeprefix("DeltaRidge_a"))
                        if alpha not in spec["ridge_alphas"]:
                            raise ValueError("Undeclared Ridge penalty")
                        f.ridge_fit(name, fit, selection, alpha, p, directory, identity)
                    else:
                        model, _ = f.neural_fit(name, fit, selection, p, spec, directory, identity, initial, device, resources)
                        del model
                        gc.collect()
                        f.torch.cuda.empty_cache()
                    records[origin][name] = stage_record(name, directory)
                f.old.write(output / "reuse.json", reuse)
            f.old.write(output / "inner_records.json", dict(records=records, persistence=baselines))
        grid = rolling.candidate_grid(records, spec["training"]["epochs"])
        selected = rolling.select_rolling(grid, baselines)
        selected.update(protocol_sha256=f.old.sha(saved), outer_used_for_selection=False,
                        selection_rule="Common config/epoch jointly non-worse in every original training origin")
        if (output / "selected.json").exists() and f.old.read(output / "selected.json") != selected:
            raise ValueError("Frozen rolling selection changed")
        f.old.write(output / "candidate_grid.json", grid)
        f.old.write(output / "selected.json", selected)
        f.old.write(output / "progress.json", dict(stage="fixed_refit_no_outer_selection", selected=selected["model"]))
        prediction = refit_prediction(selected, all_train, series.subset("validation"), p, spec, output, identity, initial, device, resources, previous)
        f.old.write(output / "progress.json", dict(stage="outer_report_no_selection"))
        f.outer_report(spec, output, series, prediction, original, selected)
        excluded = ("suite.lock", "launch.log", "launcher.pid", "source_commits.log", "status.json", "progress.json", "completed.json")
        artifacts = {str(path.relative_to(output)): f.old.sha(path) for path in output.rglob("*") if path.is_file() and path.name not in excluded}
        f.old.write(output / "completed.json", dict(protocol_sha256=f.old.sha(saved), artifacts=artifacts))
        f.old.write(output / "status.json", dict(status="COMPLETED", exit_code=0, pid=os.getpid()))
        f.old.write(output / "progress.json", dict(stage="COMPLETED"))
        print("[SUITE_COMPLETE] rolling numerical-only pilot", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/kelmarsh_numeric_rolling_protocol_v12.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--idle-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args.protocol.resolve(), args.output.resolve(), not args.idle_only)
    except BlockingIOError:
        print("[BUSY] numerical rolling suite already running", flush=True)
    except Exception as error:
        f.old.write(args.output / "status.json", dict(status="FAILED", error=f"{type(error).__name__}: {error}", pid=os.getpid()))
        raise
