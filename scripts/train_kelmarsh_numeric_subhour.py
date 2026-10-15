#!/usr/bin/env python3
"""One finite end-state information pilot; unchanged hourly labels, no Wiki."""
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
from scripts import train_kelmarsh_numeric_foundation as f
from utils import numeric_rolling as rolling, numeric_subhour as subhour

f.torch.set_num_threads(2)

PROTOCOL_ID = "kelmarsh_h2_causal_subhour_end_state_v15"
NAME = "SubhourEndState"
EXCLUDED = {"suite.lock", "launch.log", "launcher.pid", "source_commits.log", "status.json", "progress.json", "completed.json"}


def evidence(spec):
    directory = ROOT / spec["previous_directory"]
    if f.old.sha(directory / "completed.json") != spec["previous_completed_sha256"]:
        raise ValueError("Previous completion changed")
    if f.old.sha(directory / "protocol.json") != spec["previous_protocol_sha256"]:
        raise ValueError("Previous protocol changed")
    for name, expected in f.old.read(directory / "completed.json")["artifacts"].items():
        if f.old.sha(directory / name) != expected:
            raise ValueError("Previous completed artifact changed: " + name)
    sources = f.old.read(directory / "prepared.json")["identity"]["source_sha256"]
    for name, expected in sources.items():
        if f.old.sha(ROOT / name) != expected:
            raise ValueError("Frozen implementation changed: " + name)
    if f.old.sha(ROOT / spec["training_evidence_path"]) != spec["training_evidence_sha256"]:
        raise ValueError("Training evidence changed")
    if f.old.sha(ROOT / spec["raw_path"]) != spec["raw_sha256"]:
        raise ValueError("Raw source archive changed")
    return sources


def fit_stage(fit, tables, directory, identity):
    directory.mkdir(parents=True, exist_ok=True)
    x, audit = subhour.end_state_delta(fit, tables)
    truth, anchor = fit.truth_and_anchor()
    contract = dict(identity, fit=f.stage_identity(fit),
        input_target_sha256=f.identity_tools.array_digest(dict(delta=x, anchor=anchor, truth=truth,
            times=fit.target_times(), units=fit.turbines[fit.window_starts])))
    marker, path = directory / "trained.json", directory / "model.npz"
    if marker.exists():
        done = f.old.read(marker)
        if done["contract"] != contract or f.old.sha(path) != done["model_sha256"]:
            raise ValueError("Completed end-state fit changed")
        with f.np.load(path, allow_pickle=False) as saved:
            coefficients = saved["coefficients"].copy()
        if coefficients.tolist() != done["coefficients"]:
            raise ValueError("Stored executor coefficients changed")
        return coefficients, done
    coefficients, _ = subhour.fit_coefficients(fit, tables)
    f.np.savez_compressed(path, coefficients=coefficients)
    done = dict(contract=contract, model=NAME, parameters=fit.pred_len, best_epoch=None,
        coefficients=coefficients.tolist(), model_sha256=f.old.sha(path), observed_hour_audit=audit)
    f.old.write(marker, done)
    return coefficients, done


def final_prediction(selected, series, all_train, tables, output, identity, raw_path, start):
    """Selection is already frozen; outer labels are not used to fit anything."""
    if selected["model"] == "Persistence":
        return f.foundation.observed_anchor(series.subset("validation")), None
    coefficients, done = fit_stage(all_train, tables, output / "final_fit", identity)
    validation = series.subset("validation")
    last_issue = validation.target_times()[:, 0].max() - f.np.timedelta64(1, "h")
    if last_issue >= series.bounds["sealed_test_start"]:
        raise ValueError("Outer input prefix would enter sealed suffix")
    outer_tables = subhour.load_power_ticks(raw_path, start=start, stop=str(last_issue))
    return subhour.predict(coefficients, validation, outer_tables), done


def report_fixed_ablation(output, series, raw_path, start, spec):
    validation = series.subset("validation")
    last_issue = validation.target_times()[:, 0].max() - f.np.timedelta64(1, "h")
    if last_issue >= series.bounds["sealed_test_start"]:
        raise ValueError("Ablation inputs would enter sealed suffix")
    tables = subhour.load_power_ticks(raw_path, start=start, stop=str(last_issue))
    prediction = subhour.predict(f.np.ones(validation.pred_len), validation, tables)
    result = f.old.report("FixedLast10min", prediction, validation, output / "reports/FixedLast10min", 2050.)
    truth, anchor = validation.truth_and_anchor()
    with f.np.load(output / "reports/SelectedNumeric/raw_predictions.npz", allow_pickle=False) as saved:
        selected_prediction = saved["prediction"].copy()
    meta = f.pd.DataFrame({"forecast_start": validation.target_times()[:, 0]})
    result["paired"] = {"Persistence": f.old.paired_forecast_statistics(prediction, anchor, truth, meta, **spec["statistics"])}
    f.old.write(output / "reports/FixedLast10min/result.json", result)
    summary = f.old.read(output / "result.json")
    summary["results"].append(dict(model="FixedLast10min", **result["metrics"]))
    summary["paired_selected_vs_fixed_last10min"] = f.old.paired_forecast_statistics(
        selected_prediction, prediction, truth, meta, **spec["statistics"])
    summary["information_set_changed"] = True
    summary["new_input"] = "last observed 10min power within already completed hour"
    summary["architecture_or_wiki_contribution_proven"] = False
    f.old.write(output / "result.json", summary)
    f.pd.DataFrame(summary["results"]).to_csv(output / "metrics_all_runs.csv", index=False)


def run(protocol, output):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / "suite.lock").open("a") as lock, threadpool_limits(limits=2):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = f.old.read(protocol)
        if (spec["protocol_id"] != PROTOCOL_ID or spec["wiki_enabled"] or spec["fold"] != 1
                or spec["seed"] != 2024 or spec["candidates"] != [NAME]):
            raise ValueError("Unknown frozen end-state configuration")
        saved = output / "protocol.json"
        if saved.exists() and f.old.sha(saved) != f.old.sha(protocol):
            raise ValueError("Frozen end-state protocol changed")
        if not saved.exists():
            saved.write_bytes(protocol.read_bytes())
        if (output / "completed.json").exists():
            done = f.old.read(output / "completed.json")
            for name, value in f.old.read(output / "prepared.json")["identity"]["source_sha256"].items():
                if f.old.sha(ROOT / name) != value:
                    raise ValueError("Completed implementation changed: " + name)
            if done["protocol_sha256"] != f.old.sha(saved) or any(f.old.sha(output / k) != v for k, v in done["artifacts"].items()):
                raise ValueError("Completed end-state suite changed")
            print("[REUSE_SUITE] completed; no repeated training", flush=True)
            return
        f.old.write(output / "status.json", dict(status="RUNNING", pid=os.getpid()))
        f.old.write(output / "progress.json", dict(stage="causal_raw_preflight"))
        sources = evidence(spec)
        original, p = f.source_configuration(spec)
        series = f.old.KelmarshSeries(ROOT / p["data_path"], fold=1, input_len=72, horizon=2, stride=3,
            start=p["start_utc"], end=p["end_utc_exclusive"])
        origins, all_train = rolling.rolling_partitions(series)
        original_manifest = f.old.read(original / "PromptTimeDART/manifest.json")
        f.np.testing.assert_array_equal(series.scaler.mean_, original_manifest["contract"]["scaler_mean"])
        f.np.testing.assert_array_equal(series.scaler.scale_, original_manifest["contract"]["scaler_scale"])
        raw_path = ROOT / spec["raw_path"]
        tables = subhour.load_power_ticks(raw_path, start=p["start_utc"], stop=str(series.bounds["train_end"]))
        files = ("scripts/train_kelmarsh_numeric_subhour.py", "utils/numeric_subhour.py", "utils/numeric_rolling.py")
        identity = dict(protocol_sha256=f.old.sha(saved), data_sha256=p["data_sha256"],
            raw_sha256=spec["raw_sha256"], source_completed_sha256=f.old.sha(original / "completed.json"),
            source_sha256={**sources, **{key: f.old.sha(ROOT / key) for key in files}},
            packages=dict(numpy=f.np.__version__, pandas=f.pd.__version__, torch=f.torch.__version__),
            original_scaler_sha256=f.identity_tools.array_digest(dict(mean=series.scaler.mean_, scale=series.scaler.scale_)))
        if f.torch.cuda.is_initialized():
            raise ValueError("CPU-only pilot must not initialize CUDA")
        prepared = dict(identity=identity, input_audit=subhour.end_state_delta(all_train, tables)[1],
            data_audit=series.audit, wiki_enabled=False, cuda_initialized=f.torch.cuda.is_initialized(), cpu_thread_limit=2)
        if (output / "prepared.json").exists() and f.old.read(output / "prepared.json") != prepared:
            raise ValueError("Prepared data/implementation changed on resume")
        f.old.write(output / "prepared.json", prepared)
        records, baselines, grid = {}, {}, []
        for key, data in origins.items():
            f.old.write(output / "progress.json", dict(stage="training_prefix_fit", origin=key))
            query, fit = data["selection"], data["fit"]
            issue = query.target_times()[:, 0].min() - f.np.timedelta64(1, "h")
            if fit.target_times().max() > issue:
                raise ValueError("Fitting labels unavailable at first query issue")
            coefficients, done = fit_stage(fit, tables, output / "origins" / key, identity)
            truth, anchor = query.truth_and_anchor()
            baselines[key] = f.old.model_metrics(anchor, truth, anchor, 2050.)
            prediction = subhour.predict(coefficients, query, tables)
            metrics = f.old.model_metrics(prediction, truth, anchor, 2050.)
            fixed = subhour.predict(f.np.ones(query.pred_len), query, tables)
            records[key] = dict(metrics=metrics, coefficients=coefficients.tolist(),
                checkpoint_sha256=done["model_sha256"], query=f.stage_identity(query),
                fixed_last10min=f.old.model_metrics(fixed, truth, anchor, 2050.))
            f.old.write(output / "inner_records.json", dict(records=records, persistence=baselines))
            print(f"[ORIGIN_COMPLETE] {key} beta={coefficients.tolist()} MAE={metrics['mae']} RMSE={metrics['rmse']}", flush=True)
        grid.append(dict(model=NAME, best_epoch=None, parameters=2,
            per_origin={key: value["metrics"] for key, value in records.items()}))
        selected = rolling.select_rolling(grid, baselines)
        selected.update(protocol_sha256=f.old.sha(saved), outer_used_for_selection=False)
        if (output / "selected.json").exists() and f.old.read(output / "selected.json") != selected:
            raise ValueError("Frozen end-state selection changed")
        f.old.write(output / "candidate_grid.json", grid)
        f.old.write(output / "selected.json", selected)
        f.old.write(output / "progress.json", dict(stage="frozen_final_fit", selected=selected["model"]))
        prediction, done = final_prediction(selected, series, all_train, tables, output, identity, raw_path, p["start_utc"])
        f.old.write(output / "progress.json", dict(stage="outer_report_no_selection"))
        f.outer_report(spec, output, series, prediction, original, selected)
        report_fixed_ablation(output, series, raw_path, p["start_utc"], spec)
        artifacts = {str(path.relative_to(output)): f.old.sha(path) for path in output.rglob("*")
            if path.is_file() and path.name not in EXCLUDED}
        f.old.write(output / "completed.json", dict(protocol_sha256=f.old.sha(saved), artifacts=artifacts))
        f.old.write(output / "status.json", dict(status="COMPLETED", exit_code=0, pid=os.getpid()))
        f.old.write(output / "progress.json", dict(stage="COMPLETED"))
        print("[SUITE_COMPLETE] hourly targets with causal end-state input; no Wiki", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/kelmarsh_numeric_subhour_protocol_v15.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.protocol.resolve(), args.output.resolve())
    except BlockingIOError:
        print("[BUSY] end-state suite already running", flush=True)
    except Exception as error:
        f.old.write(args.output / "status.json", dict(status="FAILED", error=f"{type(error).__name__}: {error}", pid=os.getpid()))
        raise
