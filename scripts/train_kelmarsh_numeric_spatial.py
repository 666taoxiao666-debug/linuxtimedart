#!/usr/bin/env python3
"""Finite causal farm-context pilot; train-only selection, numerical base only."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import sklearn
from threadpoolctl import threadpool_limits
from scripts import train_kelmarsh_numeric_rolling as a
from utils import numeric_spatial as spatial

f = a.f
PROTOCOL_ID = "kelmarsh_h2_causal_farm_context_v13"
EXCLUDED = ("suite.lock", "launch.log", "launcher.pid", "source_commits.log", "status.json", "progress.json", "completed.json")


def previous_evidence(spec):
    directory = ROOT / spec["previous_directory"]
    if f.old.sha(directory / "completed.json") != spec["previous_completed_sha256"]:
        raise ValueError("Previous completion changed")
    complete = f.old.read(directory / "completed.json")
    if f.old.sha(directory / "protocol.json") != spec["previous_protocol_sha256"]:
        raise ValueError("Previous protocol changed")
    for name, expected in complete["artifacts"].items():
        if f.old.sha(directory / name) != expected:
            raise ValueError("Previous completed artifact changed: " + name)
    for name, expected in f.old.read(directory / "prepared.json")["identity"]["source_sha256"].items():
        if f.old.sha(ROOT / name) != expected:
            raise ValueError("Frozen previous implementation changed: " + name)
    # Also validates v11 artifacts, which contain the reused o90 Ridge.
    a.previous_evidence(f.old.read(directory / "protocol.json"))
    return directory


def cached_own_ridge(previous, origin, fit, query, p, output):
    name = "DeltaRidge_a100"
    directory = previous / "origins" / origin / name
    if origin == "o90":
        directory = ROOT / f.old.read(previous / "protocol.json")["previous_directory"] / "inner" / name
    if not a.reusable_stage(directory, fit, query, None, name):
        raise ValueError("Required own-Ridge cache does not match exact windows")
    done = a.verified_stage(name, directory)
    if done["alpha"] != 100.:
        raise ValueError("Required fixed Ridge penalty differs")
    with f.np.load(directory / "model.npz", allow_pickle=False) as saved:
        model = {key: saved[key].copy() for key in saved.files}
    truth, anchor = query.truth_and_anchor()
    metrics = f.old.model_metrics(f.foundation.ridge_predict(model, query), truth, anchor, p["rated_power_kw"])
    for key in a.METRICS:
        if not f.np.isclose(metrics[key], done["metrics"][key], atol=1e-10, rtol=1e-12):
            raise ValueError("Reused own-Ridge predictions do not reproduce metrics")
    f.old.write(output / "origins" / origin / "OwnDeltaRidge_a100" / "reuse.json",
                dict(source=str(directory.relative_to(ROOT)), trained_sha256=f.old.sha(directory / "trained.json"),
                     model_sha256=done["model_sha256"], fit=f.stage_identity(fit), selection=f.stage_identity(query)))
    print(f"[REUSE] {origin} own Ridge alpha100; no training", flush=True)
    return dict(model="OwnDeltaRidge_a100", best_epoch=None, metrics=metrics,
                parameters=spatial.capacity("OwnDeltaRidge_a100"))


def fixed_fit(name, fit, query, context, p, directory, identity):
    directory.mkdir(parents=True, exist_ok=True)
    x = spatial.features(fit, name, context)
    truth, anchor = fit.truth_and_anchor()
    contract = dict(identity, model=name, fit=f.stage_identity(fit),
                    selection=f.stage_identity(query) if query is not None else None,
                    fit_arrays_sha256=f.identity_tools.array_digest(dict(features=x, truth=truth, anchor=anchor)),
                    query_features_sha256=f.identity_tools.array_digest(dict(features=spatial.features(query, name, context))) if query is not None else None)
    marker, checkpoint = directory / "trained.json", directory / "model.joblib"
    if marker.exists():
        done = f.old.read(marker)
        if done["contract"] != contract or f.old.sha(checkpoint) != done["model_sha256"]:
            raise ValueError("Completed fixed-model stage changed")
        # Only local, hash-verified checkpoints created by this runner are loaded.
        return joblib.load(checkpoint), done
    model = spatial.fit(fit, name, context)
    joblib.dump(model, checkpoint, compress=3)
    metrics = None
    if query is not None:
        truth, anchor = query.truth_and_anchor()
        metrics = f.old.model_metrics(spatial.predict(model, query, context), truth, anchor, p["rated_power_kw"])
    done = dict(contract=contract, model=name, best_epoch=None, metrics=metrics,
                parameters=spatial.capacity(name), complexity_semantics="Fixed declared capacity proxy, not neural parameters",
                model_sha256=f.old.sha(checkpoint))
    f.old.write(marker, done)
    print(f"[FIT_COMPLETE] {name} n={len(fit)} metrics={metrics}", flush=True)
    return model, done


def refit_prediction(selected, all_train, series, context, p, output, identity):
    if selected["model"] == "Persistence":
        return f.foundation.observed_anchor(series.subset("validation"))
    model, _ = fixed_fit(selected["model"], all_train, None, context, p, output / "outer_refit", identity)
    return spatial.predict(model, series.subset("validation"), context)


def run(protocol, output):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / "suite.lock").open("a") as lock, threadpool_limits(limits=2):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = f.old.read(protocol)
        if (spec["protocol_id"] != PROTOCOL_ID or spec["wiki_enabled"] or spec["fold"] != 1 or spec["seed"] != 2024
                or spec["candidates"] != list(spatial.NAMES) or spec["hgb"] != spatial.HGB
                or spec["ridge_alpha"] != 100. or spec["spatial_features"]["lags_hours"] != list(spatial.LAGS)):
            raise ValueError("Unknown frozen spatial configuration")
        saved = output / "protocol.json"
        if saved.exists() and f.old.sha(saved) != f.old.sha(protocol):
            raise ValueError("Frozen protocol changed")
        if not saved.exists():
            saved.write_bytes(protocol.read_bytes())
        if (output / "completed.json").exists():
            complete = f.old.read(output / "completed.json")
            if complete["protocol_sha256"] != f.old.sha(saved) or any(f.old.sha(output / k) != v for k, v in complete["artifacts"].items()):
                raise ValueError("Completed spatial result changed")
            print("[REUSE_SUITE] already completed; no training", flush=True)
            return
        f.old.write(output / "status.json", dict(status="RUNNING", pid=os.getpid()))
        f.old.write(output / "progress.json", dict(stage="causal_preflight"))
        previous = previous_evidence(spec)
        original, p = f.source_configuration(spec)
        series = f.old.KelmarshSeries(ROOT / p["data_path"], fold=1, input_len=72, horizon=2, stride=3,
                                      start=p["start_utc"], end=p["end_utc_exclusive"])
        origins, all_train = a.rolling.rolling_partitions(series)
        context = spatial.FarmContext(series)
        manifest = f.old.read(original / "PromptTimeDART/manifest.json")
        f.np.testing.assert_array_equal(series.scaler.mean_, manifest["contract"]["scaler_mean"])
        f.np.testing.assert_array_equal(series.scaler.scale_, manifest["contract"]["scaler_scale"])
        source_hashes = f.old.read(previous / "prepared.json")["identity"]["source_sha256"]
        code = ("scripts/train_kelmarsh_numeric_spatial.py", "utils/numeric_spatial.py")
        identity = dict(protocol_sha256=f.old.sha(saved), data_sha256=p["data_sha256"],
            previous_completed_sha256=f.old.sha(previous / "completed.json"),
            original_completed_sha256=f.old.sha(original / "completed.json"),
            source_sha256={**source_hashes, **{key: f.old.sha(ROOT / key) for key in code}},
            packages=dict(sklearn=sklearn.__version__, numpy=f.np.__version__, joblib=joblib.__version__),
            partitions={key: {part: f.stage_identity(data[part]) for part in ("fit", "selection")} for key, data in origins.items()},
            full_refit=f.stage_identity(all_train))
        prepared = output / "prepared.json"
        if prepared.exists() and f.old.read(prepared)["identity"] != identity:
            raise ValueError("Data/code/packages changed on resume")
        f.old.write(prepared, dict(identity=identity, wiki_enabled=False, cuda_initialized=False,
            cpu_thread_limit=2, context_units=context.units.tolist(), own_features=77, farm_features=119,
            telemetry_assumption="Other-turbine hourly end-stamped observations available at own issue time"))
        records, baselines = {}, {}
        for origin, data in origins.items():
            fit, query = data["fit"], data["selection"]
            truth, anchor = query.truth_and_anchor()
            baselines[origin] = f.old.model_metrics(anchor, truth, anchor, p["rated_power_kw"])
            records[origin] = {}
            for name in spatial.NAMES:
                f.old.write(output / "progress.json", dict(stage="inner", origin=origin, model=name))
                if name == "OwnDeltaRidge_a100":
                    done = cached_own_ridge(previous, origin, fit, query, p, output)
                else:
                    _, done = fixed_fit(name, fit, query, context, p, output / "origins" / origin / name, identity)
                records[origin][name] = {key: done[key] for key in ("model", "best_epoch", "parameters", "metrics")}
                f.old.write(output / "inner_records.json", dict(records=records, persistence=baselines))
        grid = [dict(model=name, best_epoch=None, parameters=spatial.capacity(name),
                     per_origin={origin: records[origin][name]["metrics"] for origin in origins}) for name in spatial.NAMES]
        selected = a.rolling.select_rolling(grid, baselines)
        selected.update(protocol_sha256=f.old.sha(saved), outer_used_for_selection=False)
        if (output / "selected.json").exists() and f.old.read(output / "selected.json") != selected:
            raise ValueError("Frozen train-only selection changed")
        f.old.write(output / "candidate_grid.json", grid)
        f.old.write(output / "selected.json", selected)
        f.old.write(output / "progress.json", dict(stage="fixed_outer_refit", selected=selected["model"]))
        # Instantiate/query outer predictions only AFTER selected configuration freezes.
        prediction = refit_prediction(selected, all_train, series, context, p, output, identity)
        f.old.write(output / "progress.json", dict(stage="outer_report_no_selection"))
        f.outer_report(spec, output, series, prediction, original, selected)
        artifacts = {str(path.relative_to(output)): f.old.sha(path) for path in output.rglob("*") if path.is_file() and path.name not in EXCLUDED}
        f.old.write(output / "completed.json", dict(protocol_sha256=f.old.sha(saved), artifacts=artifacts))
        f.old.write(output / "status.json", dict(status="COMPLETED", exit_code=0, pid=os.getpid()))
        f.old.write(output / "progress.json", dict(stage="COMPLETED"))
        print("[SUITE_COMPLETE] finite causal numerical pilot; Wiki remains off", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/kelmarsh_numeric_spatial_protocol_v13.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.protocol.resolve(), args.output.resolve())
    except BlockingIOError:
        print("[BUSY] numerical spatial suite already running", flush=True)
    except Exception as error:
        f.old.write(args.output / "status.json", dict(status="FAILED", error=f"{type(error).__name__}: {error}", pid=os.getpid()))
        raise
