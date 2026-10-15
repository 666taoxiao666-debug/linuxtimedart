#!/usr/bin/env python3
"""Finite constrained state-filter pilot, strictly compared with cached v15."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from threadpoolctl import threadpool_limits
from scripts import train_kelmarsh_numeric_subhour as previous
from utils import numeric_subhour_filter as state_filter

f, subhour, rolling = previous.f, previous.subhour, previous.rolling
NAME = "SimplexSubhourFilter"
PROTOCOL_ID = "kelmarsh_h2_convex_subhour_state_filter_v16"


def evidence(spec):
    directory = ROOT / spec["previous_directory"]
    if (f.old.sha(directory / "completed.json") != spec["previous_completed_sha256"]
            or f.old.sha(directory / "protocol.json") != spec["previous_protocol_sha256"]):
        raise ValueError("Frozen v15 completion/protocol changed")
    for key, expected in f.old.read(directory / "completed.json")["artifacts"].items():
        if f.old.sha(directory / key) != expected:
            raise ValueError("Frozen v15 artifact changed: " + key)
    source = f.old.read(directory / "prepared.json")["identity"]["source_sha256"]
    for key, expected in source.items():
        if f.old.sha(ROOT / key) != expected:
            raise ValueError("Frozen implementation changed: " + key)
    if f.old.sha(ROOT / spec["raw_path"]) != spec["raw_sha256"]:
        raise ValueError("Original raw source changed")
    return directory, source


def cached_baseline(directory, fit, tables):
    """Check the actual data seen by the cached executor, not just model names."""
    done = f.old.read(directory / "trained.json")
    if f.old.sha(directory / "model.npz") != done["model_sha256"] or done["contract"]["fit"] != f.stage_identity(fit):
        raise ValueError("Cached v15 executor/window identity changed")
    x, _ = subhour.end_state_delta(fit, tables)
    truth, anchor = fit.truth_and_anchor()
    digest = f.identity_tools.array_digest(dict(delta=x, anchor=anchor, truth=truth,
        times=fit.target_times(), units=fit.turbines[fit.window_starts]))
    if digest != done["contract"]["input_target_sha256"]:
        raise ValueError("Cached v15 actual fitting inputs/labels changed")
    with f.np.load(directory / "model.npz", allow_pickle=False) as saved:
        beta = saved["coefficients"].copy()
    if beta.tolist() != done["coefficients"]:
        raise ValueError("Cached v15 actual coefficients changed")
    return beta, done


def fit_stage(fit, tables, beta, baseline_sha, directory, identity):
    directory.mkdir(parents=True, exist_ok=True)
    x = state_filter.centered_ticks(fit, tables)
    truth, anchor = fit.truth_and_anchor()
    contract = dict(identity, fit=f.stage_identity(fit), baseline_checkpoint_sha256=baseline_sha,
        input_target_sha256=f.identity_tools.array_digest(dict(ticks=x, truth=truth, anchor=anchor,
            times=fit.target_times(), units=fit.turbines[fit.window_starts])))
    marker, path = directory / "trained.json", directory / "model.npz"
    if marker.exists():
        done = f.old.read(marker)
        if done["contract"] != contract or f.old.sha(path) != done["model_sha256"]:
            raise ValueError("Completed filter fit changed")
        with f.np.load(path, allow_pickle=False) as saved:
            weights = saved["weights"].copy()
        if weights.tolist() != done["weights"]:
            raise ValueError("Stored filter executor changed")
        return weights, done
    weights = state_filter.fit_weights(fit, tables, beta)
    f.np.savez_compressed(path, weights=weights)
    prediction = anchor + x @ weights.T
    prior_prediction = subhour.predict(beta, fit, tables)
    if not (f.np.mean((prediction - truth)**2, axis=0) <= f.np.mean((prior_prediction - truth)**2, axis=0) + 1e-7).all():
        raise ValueError("Training convex optimum is worse than feasible v15")
    done = dict(contract=contract, weights=weights.tolist(), parameters=10, serialized_weights=12,
                model_sha256=f.old.sha(path), fitting_mse_per_lead=f.np.mean((prediction - truth)**2, axis=0).tolist())
    f.old.write(marker, done)
    return weights, done


def finish_frozen_selection(selected, previous_dir, series, all_train, tables, output, identity, spec, p, original):
    if selected["reused_v15"]:
        # Do NOT evaluate the rejected filter on outer labels, even for curiosity.
        old_metrics = f.old.read(previous_dir / "reports/SelectedNumeric/result.json")["metrics"]
        f.old.write(output / "result.json", dict(selected=selected, primary_joint_success=False,
            new_outer_evaluation=False, retained_v15_outer_metrics_cached=old_metrics,
            interpretation="Negative train-only filter result; retain exact v15, not a new improvement",
            wiki_enabled=False, sealed_test_evaluated=False))
        return
    beta, prior = cached_baseline(previous_dir / "final_fit", all_train, tables)
    weights, done = fit_stage(all_train, tables, beta, prior["model_sha256"], output / "final_fit", identity)
    # Freeze actual executor before loading any new outer input prefix or outcomes.
    validation = series.subset("validation")
    issue = validation.target_times()[:, 0].max() - f.np.timedelta64(1, "h")
    if issue >= series.bounds["sealed_test_start"]:
        raise ValueError("Raw input prefix would enter sealed suffix")
    outer_tables = subhour.load_power_ticks(ROOT / spec["raw_path"], start=p["start_utc"], stop=str(issue))
    prediction = state_filter.predict(weights, validation, outer_tables)
    f.outer_report(spec, output, series, prediction, original, selected)
    with f.np.load(previous_dir / "reports/SelectedNumeric/raw_predictions.npz", allow_pickle=False) as saved:
        for key, expected in dict(truth=validation.truth_and_anchor()[0], times=validation.target_times(),
                turbines=validation.turbines[validation.window_starts], window_starts=validation.window_starts).items():
            f.np.testing.assert_array_equal(saved[key], expected)
        baseline_prediction = saved["prediction"].copy()
    f.np.testing.assert_allclose(subhour.predict(beta, validation, outer_tables), baseline_prediction, atol=1e-10, rtol=0)
    baseline_result = f.old.read(previous_dir / "reports/SelectedNumeric/result.json")
    current = f.old.read(output / "reports/SelectedNumeric/result.json")
    truth, _ = validation.truth_and_anchor()
    paired = f.old.paired_forecast_statistics(prediction, baseline_prediction, truth,
        f.pd.DataFrame({"forecast_start": validation.target_times()[:, 0]}), **spec["statistics"])
    current["paired"]["FrozenV15"] = paired
    f.old.write(output / "reports/SelectedNumeric/result.json", current)
    result = f.old.read(output / "result.json")
    result.update(primary_joint_success=bool(result["primary_joint_success"] and all(
        current["metrics"][key] < baseline_result["metrics"][key] for key in ("mae", "rmse"))),
        new_outer_evaluation=True, paired_vs_frozen_v15=paired, information_set="same completed-hour raw six ticks; new pooling",
        architecture_or_wiki_contribution_proven=False)
    result["results"].append(dict(model="FrozenV15", **baseline_result["metrics"]))
    f.old.write(output / "result.json", result)
    f.pd.DataFrame(result["results"]).to_csv(output / "metrics_all_runs.csv", index=False)


def run(protocol, output):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / "suite.lock").open("a") as lock, threadpool_limits(limits=2):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = f.old.read(protocol)
        if (spec["protocol_id"] != PROTOCOL_ID or spec["candidates"] != [NAME]
                or spec["fold"] != 1 or spec["seed"] != 2024 or spec["wiki_enabled"]):
            raise ValueError("Unknown frozen convex-filter trial")
        saved = output / "protocol.json"
        if saved.exists() and f.old.sha(saved) != f.old.sha(protocol):
            raise ValueError("Frozen filter protocol changed")
        if not saved.exists():
            saved.write_bytes(protocol.read_bytes())
        if (output / "completed.json").exists():
            complete = f.old.read(output / "completed.json")
            for key, value in f.old.read(output / "prepared.json")["identity"]["source_sha256"].items():
                if f.old.sha(ROOT / key) != value:
                    raise ValueError("Completed implementation changed")
            if complete["protocol_sha256"] != f.old.sha(saved) or any(f.old.sha(output / k) != v for k, v in complete["artifacts"].items()):
                raise ValueError("Completed filter suite changed")
            print("[REUSE_SUITE] no repeated training", flush=True)
            return
        f.old.write(output / "status.json", dict(status="RUNNING", pid=os.getpid()))
        f.old.write(output / "progress.json", dict(stage="causal_source_preflight"))
        previous_dir, sources = evidence(spec)
        original, p = f.source_configuration(spec)
        series = f.old.KelmarshSeries(ROOT / p["data_path"], fold=1, input_len=72, horizon=2, stride=3,
            start=p["start_utc"], end=p["end_utc_exclusive"])
        origins, all_train = rolling.rolling_partitions(series)
        tables = subhour.load_power_ticks(ROOT / spec["raw_path"], start=p["start_utc"], stop=str(series.bounds["train_end"]))
        code = ("scripts/train_kelmarsh_numeric_filter.py", "utils/numeric_subhour_filter.py")
        if f.torch.cuda.is_initialized():
            raise ValueError("CPU-only pilot must not initialize CUDA")
        identity = dict(protocol_sha256=f.old.sha(saved), previous_completed_sha256=spec["previous_completed_sha256"],
            data_sha256=p["data_sha256"], raw_sha256=spec["raw_sha256"],
            source_sha256={**sources, **{key: f.old.sha(ROOT / key) for key in code}},
            packages=dict(numpy=f.np.__version__, pandas=f.pd.__version__),
            scaler_sha256=f.identity_tools.array_digest(dict(mean=series.scaler.mean_, scale=series.scaler.scale_)))
        if (output / "prepared.json").exists() and f.old.read(output / "prepared.json")["identity"] != identity:
            raise ValueError("Filter identity changed on resume")
        f.old.write(output / "prepared.json", dict(identity=identity, cuda_initialized=False, own_cpu_threads=2,
            source_data_audit=series.audit, wiki_enabled=False))
        records, references, rows = {}, {}, []
        old_records = f.old.read(previous_dir / "inner_records.json")
        for key, data in origins.items():
            f.old.write(output / "progress.json", dict(stage="prefix_filter_fit", origin=key))
            fit, query = data["fit"], data["selection"]
            if fit.target_times().max() > query.target_times()[:, 0].min() - f.np.timedelta64(1, "h"):
                raise ValueError("Fitting outcome unavailable at query issue")
            beta, baseline = cached_baseline(previous_dir / "origins" / key, fit, tables)
            truth, anchor = query.truth_and_anchor()
            baseline_metrics = f.old.model_metrics(subhour.predict(beta, query, tables), truth, anchor, 2050.)
            for metric in ("mae", "mse", "rmse", "r2"):
                f.np.testing.assert_allclose(baseline_metrics[metric], old_records["records"][key]["metrics"][metric], atol=1e-9, rtol=1e-11)
            weights, done = fit_stage(fit, tables, beta, baseline["model_sha256"], output / "origins" / key, identity)
            prediction = state_filter.predict(weights, query, tables)
            metrics = f.old.model_metrics(prediction, truth, anchor, 2050.)
            records[key] = metrics
            references[key] = dict(v15=baseline_metrics, persistence=f.old.model_metrics(anchor, truth, anchor, 2050.))
            for name, values in [(NAME, metrics), *references[key].items()]:
                rows.append(dict(split="original_train_query", origin=key, model=name, **values))
            f.old.write(output / "inner_records.json", dict(records=records, references=references))
            f.pd.DataFrame(rows).to_csv(output / "metrics_train_origins.csv", index=False)
            print(f"[ORIGIN_COMPLETE] {key} MAE={metrics['mae']} RMSE={metrics['rmse']} vs_v15_MAE={baseline_metrics['mae']} weights={weights.tolist()}", flush=True)
        selected = state_filter.select_filter(records, references)
        selected.update(protocol_sha256=f.old.sha(saved), outer_used_for_selection=False)
        if (output / "selected.json").exists() and f.old.read(output / "selected.json") != selected:
            raise ValueError("Frozen filter selection changed")
        f.old.write(output / "selected.json", selected)
        f.old.write(output / "progress.json", dict(stage="frozen_selection", selected=selected["model"]))
        finish_frozen_selection(selected, previous_dir, series, all_train, tables, output, identity, spec, p, original)
        artifacts = {str(path.relative_to(output)): f.old.sha(path) for path in output.rglob("*")
            if path.is_file() and path.name not in previous.EXCLUDED}
        f.old.write(output / "completed.json", dict(protocol_sha256=f.old.sha(saved), artifacts=artifacts))
        f.old.write(output / "status.json", dict(status="COMPLETED", pid=os.getpid(), exit_code=0))
        f.old.write(output / "progress.json", dict(stage="COMPLETED"))
        print("[SUITE_COMPLETE] new_outer_evaluation=" + str(not selected["reused_v15"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/kelmarsh_numeric_filter_protocol_v16.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.protocol.resolve(), args.output.resolve())
    except BlockingIOError:
        print("[BUSY] filter suite already running", flush=True)
    except Exception as error:
        f.old.write(args.output / "status.json", dict(status="FAILED", pid=os.getpid(), error=f"{type(error).__name__}: {error}"))
        raise
