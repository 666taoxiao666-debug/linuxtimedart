#!/usr/bin/env python3
"""Frozen nested-training, exact-executor numerical shrinkage pilot."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sklearn
from threadpoolctl import threadpool_limits
from scripts import train_kelmarsh_numeric_spatial as previous
from utils import numeric_shrink as shrink

f = previous.f
PROTOCOL_ID = "kelmarsh_h2_executor_bound_joint_shrink_v14"
NAMES = ("RidgeUnscaled", "RidgeJointShrink")


def evidence(spec):
    directory = ROOT / spec["previous_directory"]
    if f.old.sha(directory / "completed.json") != spec["previous_completed_sha256"]:
        raise ValueError("Previous completion changed")
    complete = f.old.read(directory / "completed.json")
    if f.old.sha(directory / "protocol.json") != spec["previous_protocol_sha256"]:
        raise ValueError("Previous protocol changed")
    for name, value in complete["artifacts"].items():
        if f.old.sha(directory / name) != value:
            raise ValueError("Previous completed artifact changed: " + name)
    source = f.old.read(directory / "prepared.json")["identity"]["source_sha256"]
    for name, value in source.items():
        if f.old.sha(ROOT / name) != value:
            raise ValueError("Frozen implementation changed: " + name)
    previous.previous_evidence(f.old.read(directory / "protocol.json"))
    if f.old.sha(ROOT / spec["training_evidence_path"]) != spec["training_evidence_sha256"]:
        raise ValueError("Training diagnostic evidence changed")
    return directory, source


def calibrate(model, calibration, expert_sha, directory, identity):
    directory.mkdir(parents=True, exist_ok=True)
    contract = dict(identity, expert_checkpoint_sha256=expert_sha, expert_array_sha256=shrink.arrays_digest(model),
                    scaler_sha256=shrink.scaler_digest(calibration), calibration=f.stage_identity(calibration))
    marker, path = directory / "calibrated.json", directory / "certificate.json"
    if marker.exists():
        done = f.old.read(marker)
        if done["contract"] != contract or f.old.sha(path) != done["certificate_sha256"]:
            raise ValueError("Completed calibration changed")
        return f.old.read(path)
    certificate = shrink.make_certificate(model, calibration, expert_sha)
    for key in ("mae", "mse"):
        after = f.np.array(certificate[f"calibration_{key}_after"])
        baseline = f.np.array(certificate[f"calibration_persistence_{key}"])
        if not (after <= baseline + 1e-9 * f.np.maximum(1., baseline)).all():
            raise ValueError("Calibration fails its constrained joint loss")
    f.old.write(path, certificate)
    f.old.write(marker, dict(contract=contract, certificate_sha256=f.old.sha(path)))
    return certificate


def final_expert(previous_directory, fit, calibration, output):
    """The final90% expert already exists in v11: verify/reuse, never refit."""
    v12 = ROOT / f.old.read(previous_directory / "protocol.json")["previous_directory"]
    v11 = ROOT / f.old.read(v12 / "protocol.json")["previous_directory"]
    directory = v11 / "inner/DeltaRidge_a100"
    name = "DeltaRidge_a100"
    if not previous.a.reusable_stage(directory, fit, calibration, None, name):
        raise ValueError("Required completed final90% expert does not match; stop, do not retrain")
    done = previous.a.verified_stage(name, directory)
    if done["alpha"] != 100.:
        raise ValueError("Completed final expert penalty differs")
    with f.np.load(directory / "model.npz", allow_pickle=False) as saved:
        model = {key: saved[key].copy() for key in saved.files}
    f.old.write(output / "final_expert_reuse.json", dict(source=str(directory.relative_to(ROOT)),
        trained_sha256=f.old.sha(directory / "trained.json"), model_sha256=done["model_sha256"],
        fit=f.stage_identity(fit), calibration=f.stage_identity(calibration), refitted_after_calibration=False))
    return model, done["model_sha256"]


def final_prediction(selected, series, previous_directory, output, identity):
    if selected["model"] == "Persistence":
        return f.foundation.observed_anchor(series.subset("validation"))
    fit, calibration, audit = shrink.nested_partition(series, series.bounds["train_end"])
    model, expert_sha = final_expert(previous_directory, fit, calibration, output)
    f.old.write(output / "final_partition.json", audit)
    if selected["model"] == "RidgeJointShrink":
        certificate = calibrate(model, calibration, expert_sha, output / "final_calibration", identity)
        return shrink.predict_bound(model, series.subset("validation"), certificate, expert_sha)
    return f.foundation.ridge_predict(model, series.subset("validation"))


def run(protocol, output):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / "suite.lock").open("a") as lock, threadpool_limits(limits=2):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = f.old.read(protocol)
        if (spec["protocol_id"] != PROTOCOL_ID or spec["wiki_enabled"] or spec["fold"] != 1 or spec["seed"] != 2024
                or spec["candidates"] != list(NAMES) or spec["ridge_alpha"] != 100. or spec["expert_fit_fraction"] != .9):
            raise ValueError("Unknown frozen nested-shrink configuration")
        saved = output / "protocol.json"
        if saved.exists() and f.old.sha(saved) != f.old.sha(protocol):
            raise ValueError("Frozen shrink protocol changed")
        if not saved.exists():
            saved.write_bytes(protocol.read_bytes())
        if (output / "completed.json").exists():
            complete = f.old.read(output / "completed.json")
            if complete["protocol_sha256"] != f.old.sha(saved) or any(f.old.sha(output / k) != v for k, v in complete["artifacts"].items()):
                raise ValueError("Completed shrink result changed")
            print("[REUSE_SUITE] completed; no repeated training", flush=True)
            return
        f.old.write(output / "status.json", dict(status="RUNNING", pid=os.getpid()))
        f.old.write(output / "progress.json", dict(stage="causal_preflight"))
        prev, source_hashes = evidence(spec)
        original, p = f.source_configuration(spec)
        series = f.old.KelmarshSeries(ROOT / p["data_path"], fold=1, input_len=72, horizon=2, stride=3,
                                      start=p["start_utc"], end=p["end_utc_exclusive"])
        origins, _ = previous.a.rolling.rolling_partitions(series)
        manifest = f.old.read(original / "PromptTimeDART/manifest.json")
        f.np.testing.assert_array_equal(series.scaler.mean_, manifest["contract"]["scaler_mean"])
        f.np.testing.assert_array_equal(series.scaler.scale_, manifest["contract"]["scaler_scale"])
        nested = {key: shrink.nested_partition(series, value["cut"]) for key, value in origins.items()}
        for key, (fit, calibration, _) in nested.items():
            first_issue = origins[key]["selection"].target_times()[:, 0].min() - f.np.timedelta64(1, "h")
            if calibration.target_times().max() > first_issue:
                raise ValueError("Calibration labels unavailable at first query issue")
        files = ("scripts/train_kelmarsh_numeric_shrink.py", "utils/numeric_shrink.py")
        identity = dict(protocol_sha256=f.old.sha(saved), data_sha256=p["data_sha256"],
            previous_completed_sha256=f.old.sha(prev / "completed.json"),
            source_sha256={**source_hashes, **{key: f.old.sha(ROOT / key) for key in files}},
            packages=dict(sklearn=sklearn.__version__, numpy=f.np.__version__),
            partitions={key: dict(fit=f.stage_identity(value[0]), calibration=f.stage_identity(value[1]),
                                  query=f.stage_identity(origins[key]["selection"])) for key, value in nested.items()})
        prepared = output / "prepared.json"
        if prepared.exists() and f.old.read(prepared)["identity"] != identity:
            raise ValueError("Nested-shrink implementation/data changed on resume")
        f.old.write(prepared, dict(identity=identity, partitions={key: value[2] for key, value in nested.items()},
                                  wiki_enabled=False, cuda_initialized=False, cpu_thread_limit=2))
        records, baselines, coefficients = {}, {}, {}
        for key, (fit, calibration, _) in nested.items():
            query = origins[key]["selection"]
            f.old.write(output / "progress.json", dict(stage="nested_fit_and_calibration", origin=key))
            model, done = f.ridge_fit("DeltaRidge_a100", fit, calibration, 100., p, output / "origins" / key / "expert", identity)
            expert_sha = done["model_sha256"]
            certificate = calibrate(model, calibration, expert_sha, output / "origins" / key / "calibration", identity)
            coefficients[key] = certificate["coefficients"]
            truth, anchor = query.truth_and_anchor()
            baselines[key] = f.old.model_metrics(anchor, truth, anchor, p["rated_power_kw"])
            forecasts = dict(RidgeUnscaled=f.foundation.ridge_predict(model, query),
                             RidgeJointShrink=shrink.predict_bound(model, query, certificate, expert_sha))
            records[key] = {name: dict(model=name, best_epoch=None, parameters=156 + (2 if name == "RidgeJointShrink" else 0),
                metrics=f.old.model_metrics(prediction, truth, anchor, p["rated_power_kw"])) for name, prediction in forecasts.items()}
            f.old.write(output / "inner_records.json", dict(records=records, persistence=baselines, coefficients=coefficients))
            print(f"[ORIGIN_COMPLETE] {key} alpha={coefficients[key]} " + str({name: {k: row['metrics'][k] for k in ('mae', 'rmse')} for name, row in records[key].items()}), flush=True)
        grid = [dict(model=name, best_epoch=None, parameters=156 + (2 if name == "RidgeJointShrink" else 0),
                     per_origin={key: value[name]["metrics"] for key, value in records.items()}) for name in NAMES]
        selected = previous.a.rolling.select_rolling(grid, baselines)
        selected.update(protocol_sha256=f.old.sha(saved), outer_used_for_selection=False)
        if (output / "selected.json").exists() and f.old.read(output / "selected.json") != selected:
            raise ValueError("Frozen nested-shrink selection changed")
        f.old.write(output / "candidate_grid.json", grid)
        f.old.write(output / "selected.json", selected)
        f.old.write(output / "progress.json", dict(stage="final_frozen_pipeline", selected=selected["model"]))
        prediction = final_prediction(selected, series, prev, output, identity)
        f.old.write(output / "progress.json", dict(stage="outer_report_no_selection"))
        f.outer_report(spec, output, series, prediction, original, selected)
        artifacts = {str(path.relative_to(output)): f.old.sha(path) for path in output.rglob("*") if path.is_file() and path.name not in previous.EXCLUDED}
        f.old.write(output / "completed.json", dict(protocol_sha256=f.old.sha(saved), artifacts=artifacts))
        f.old.write(output / "status.json", dict(status="COMPLETED", exit_code=0, pid=os.getpid()))
        f.old.write(output / "progress.json", dict(stage="COMPLETED"))
        print("[SUITE_COMPLETE] exact-executor numerical shrink; no Wiki", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/kelmarsh_numeric_shrink_protocol_v14.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.protocol.resolve(), args.output.resolve())
    except BlockingIOError:
        print("[BUSY] numerical shrink suite already running", flush=True)
    except Exception as error:
        f.old.write(args.output / "status.json", dict(status="FAILED", error=f"{type(error).__name__}: {error}", pid=os.getpid()))
        raise
