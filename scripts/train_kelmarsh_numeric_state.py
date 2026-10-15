#!/usr/bin/env python3
"""Finite state-conditional end-point trial, with independent lead protection."""
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
from scripts import train_kelmarsh_numeric_horizon2 as prior_runner
from utils import numeric_state as state

base, f = prior_runner.base, prior_runner.f
scopes = prior_runner.scopes
PROTOCOL_ID = "kelmarsh_h2_state_shrunk_endpoints_v18"


def fit_stage(fit, tables, beta, baseline_sha, directory, identity):
    directory.mkdir(parents=True, exist_ok=True)
    delta, anchor, cells, issues = state.design(fit, tables)
    truth, _ = fit.truth_and_anchor()
    contract = dict(identity, fit=f.stage_identity(fit), baseline_checkpoint_sha256=baseline_sha,
        fit_arrays_sha256=f.identity_tools.array_digest(dict(delta=delta, anchor=anchor, cells=cells,
            truth=truth, issues=issues, units=fit.turbines[fit.window_starts])))
    marker, path = directory / "trained.json", directory / "model.npz"
    if marker.exists():
        done = f.old.read(marker)
        if done["contract"] != contract or f.old.sha(path) != done["model_sha256"]:
            raise ValueError("Completed state fit changed")
        with f.np.load(path, allow_pickle=False) as saved:
            model = {key: saved[key].copy() for key in saved.files}
        if model["coefficients"].tolist() != done["coefficients"] or not f.np.array_equal(model["prior"], beta):
            raise ValueError("Stored state executor changed")
        return model, done
    model = state.fit(fit, tables, beta)
    f.np.savez_compressed(path, **model)
    done = dict(contract=contract, coefficients=model["coefficients"].tolist(), support=model["support"].tolist(),
        penalty=float(model["penalty"]), prior=model["prior"].tolist(), parameters=12, model_sha256=f.old.sha(path))
    f.old.write(marker, done)
    return model, done


def finish(selected, prior, series, all_train, tables, output, identity, spec, p, original):
    if selected["reused_v15"]:
        f.old.write(output / "result.json", dict(selected=selected, primary_joint_success=False,
            new_outer_evaluation=False, retained_v15_outer_metrics_cached=f.old.read(prior / "reports/SelectedNumeric/result.json")["metrics"],
            wiki_enabled=False, sealed_test_evaluated=False,
            interpretation="No stable TRAIN lead passed; exact v15 retained, no new outer improvement"))
        return
    beta, baseline = base.cached_baseline(prior / "final_fit", all_train, tables)
    model, _ = fit_stage(all_train, tables, beta, baseline["model_sha256"], output / "final_fit", identity)
    enabled = f.np.asarray(selected["enabled_leads"], dtype=bool)
    validation = series.subset("validation")
    issue = validation.target_times()[:, 0].max() - f.np.timedelta64(1, "h")
    if issue >= series.bounds["sealed_test_start"]:
        raise ValueError("Raw prefix enters sealed suffix")
    outer_tables = base.subhour.load_power_ticks(ROOT / spec["raw_path"], start=p["start_utc"], stop=str(issue))
    prediction = state.predict(model, validation, outer_tables, enabled)
    truth, anchor = validation.truth_and_anchor()
    reference = base.subhour.predict(beta, validation, outer_tables)
    with f.np.load(prior / "reports/SelectedNumeric/raw_predictions.npz", allow_pickle=False) as saved:
        for key, expected in dict(prediction=reference, truth=truth, times=validation.target_times(),
            turbines=validation.turbines[validation.window_starts], window_starts=validation.window_starts).items():
            f.np.testing.assert_array_equal(saved[key], expected)
    f.np.testing.assert_array_equal(prediction[:, ~enabled], reference[:, ~enabled])
    f.outer_report(spec, output, series, prediction, original, selected)
    new, old, persistence = scopes(prediction, truth, anchor), scopes(reference, truth, anchor), scopes(anchor, truth, anchor)
    meta = f.pd.DataFrame({"forecast_start": validation.target_times()[:, 0]})
    paired = {scope: f.old.paired_forecast_statistics(prediction[:, sl], reference[:, sl], truth[:, sl], meta, **spec["statistics"])
        for scope, sl in (("overall", slice(None)), ("horizon1", slice(0, 1)), ("horizon2", slice(1, 2)))}
    result = f.old.read(output / "result.json")
    success = (all(new["overall"][m] < ref["overall"][m] for ref in (old, persistence) for m in ("mae", "rmse"))
        and all(new[lead][m] <= old[lead][m] + 1e-8 for lead in ("horizon1", "horizon2") for m in ("mae", "rmse")))
    result.update(primary_joint_success=bool(success), new_outer_evaluation=True, per_lead_candidate=new,
        per_lead_frozen_v15=old, paired_vs_v15=paired, disabled_leads_exact_v15=True,
        actual_history_consumed="last completed hour mean + last observed 10min power",
        architecture_or_wiki_contribution_proven=False)
    result["results"].append(dict(model="FrozenV15", **old["overall"]))
    f.old.write(output / "result.json", result)
    f.pd.DataFrame(result["results"]).to_csv(output / "metrics_all_runs.csv", index=False)


def run(protocol, output):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / "suite.lock").open("a") as lock, threadpool_limits(limits=2):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = f.old.read(protocol)
        if (spec["protocol_id"] != PROTOCOL_ID or spec["candidates"] != [state.NAME] or spec["fold"] != 1
                or spec["seed"] != 2024 or spec["wiki_enabled"]):
            raise ValueError("Unknown frozen state trial")
        saved = output / "protocol.json"
        if saved.exists() and f.old.sha(saved) != f.old.sha(protocol):
            raise ValueError("Frozen state protocol changed")
        if not saved.exists():
            saved.write_bytes(protocol.read_bytes())
        if (output / "completed.json").exists():
            complete = f.old.read(output / "completed.json")
            for key, value in f.old.read(output / "prepared.json")["identity"]["source_sha256"].items():
                if f.old.sha(ROOT / key) != value:
                    raise ValueError("Completed state implementation changed")
            if complete["protocol_sha256"] != f.old.sha(saved) or any(f.old.sha(output / k) != v for k, v in complete["artifacts"].items()):
                raise ValueError("Completed state suite changed")
            print("[REUSE_SUITE] no repeated state fitting", flush=True)
            return
        f.old.write(output / "status.json", dict(status="RUNNING", pid=os.getpid()))
        f.old.write(output / "progress.json", dict(stage="causal_preflight"))
        prior, sources = base.evidence(spec)
        if f.old.sha(ROOT / spec["training_audit_path"]) != spec["training_audit_sha256"]:
            raise ValueError("Declared training-only state evidence changed")
        audit = f.old.read(ROOT / spec["training_audit_path"])
        if audit["outer_labels_used"] or audit["sealed_test_evaluated"]:
            raise ValueError("State evidence must exclude outer/test outcomes")
        original, p = f.source_configuration(spec)
        series = f.old.KelmarshSeries(ROOT / p["data_path"], fold=1, input_len=72, horizon=2, stride=3,
            start=p["start_utc"], end=p["end_utc_exclusive"])
        origins, all_train = base.rolling.rolling_partitions(series)
        tables = base.subhour.load_power_ticks(ROOT / spec["raw_path"], start=p["start_utc"], stop=str(series.bounds["train_end"]))
        files = ("scripts/train_kelmarsh_numeric_state.py", "utils/numeric_state.py",
                 "scripts/train_kelmarsh_numeric_horizon2.py", "scripts/train_kelmarsh_numeric_filter.py")
        identity = dict(protocol_sha256=f.old.sha(saved), data_sha256=p["data_sha256"], raw_sha256=spec["raw_sha256"],
            training_audit_sha256=spec["training_audit_sha256"],
            previous_completed_sha256=spec["previous_completed_sha256"],
            source_sha256={**sources, **{key: f.old.sha(ROOT / key) for key in files}},
            packages=dict(numpy=f.np.__version__, pandas=f.pd.__version__),
            scaler_sha256=f.identity_tools.array_digest(dict(mean=series.scaler.mean_, scale=series.scaler.scale_)))
        if f.torch.cuda.is_initialized():
            raise ValueError("CPU-only trial cannot initialize CUDA")
        if (output / "prepared.json").exists() and f.old.read(output / "prepared.json")["identity"] != identity:
            raise ValueError("State data/source identity changed on resume")
        f.old.write(output / "prepared.json", dict(identity=identity, cuda_initialized=False, cpu_threads=2,
            declared_history_hours=72, actual_inputs="same hourly mean and end tick as v15"))
        records, references = {}, {}
        for key, data in origins.items():
            f.old.write(output / "progress.json", dict(stage="fit_state", origin=key))
            fit, query = data["fit"], data["selection"]
            if fit.target_times().max() > query.target_times()[:, 0].min() - f.np.timedelta64(1, "h"):
                raise ValueError("Unavailable fit outcomes")
            beta, cached = base.cached_baseline(prior / "origins" / key, fit, tables)
            model, _ = fit_stage(fit, tables, beta, cached["model_sha256"], output / "origins" / key, identity)
            truth, anchor = query.truth_and_anchor()
            baseline = base.subhour.predict(beta, query, tables)
            prediction = state.predict(model, query, tables)
            records[key] = scopes(prediction, truth, anchor)
            references[key] = dict(v15=scopes(baseline, truth, anchor), persistence=scopes(anchor, truth, anchor))
            f.old.write(output / "inner_records.json", dict(records=records, references=references))
            print(f"[ORIGIN_COMPLETE] {key} MAE={records[key]['overall']['mae']} RMSE={records[key]['overall']['rmse']} "
                f"v15_MAE={references[key]['v15']['overall']['mae']}", flush=True)
        selected = state.choose(records, references)
        selected.update(protocol_sha256=f.old.sha(saved), outer_used_for_selection=False)
        if (output / "selected.json").exists() and f.old.read(output / "selected.json") != selected:
            raise ValueError("Frozen lead flags changed")
        f.old.write(output / "selected.json", selected)
        rows = [dict(origin=key, model=name, scope=scope, **values)
            for key, new in records.items() for name, scopes_ in {state.NAME: new, **references[key]}.items()
            for scope, values in scopes_.items()]
        f.pd.DataFrame(rows).to_csv(output / "metrics_train_origins.csv", index=False)
        f.old.write(output / "progress.json", dict(stage="frozen_selection", selected=selected["model"], enabled_leads=selected["enabled_leads"]))
        finish(selected, prior, series, all_train, tables, output, identity, spec, p, original)
        artifacts = {str(path.relative_to(output)): f.old.sha(path) for path in output.rglob("*")
                     if path.is_file() and path.name not in base.previous.EXCLUDED}
        f.old.write(output / "completed.json", dict(protocol_sha256=f.old.sha(saved), artifacts=artifacts))
        f.old.write(output / "status.json", dict(status="COMPLETED", pid=os.getpid(), exit_code=0))
        f.old.write(output / "progress.json", dict(stage="COMPLETED"))
        print("[SUITE_COMPLETE] enabled=" + str(selected["enabled_leads"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/kelmarsh_numeric_state_protocol_v18.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.protocol.resolve(), args.output.resolve())
    except BlockingIOError:
        print("[BUSY] State trial already running", flush=True)
    except Exception as error:
        f.old.write(args.output / "status.json", dict(status="FAILED", pid=os.getpid(), error=f"{type(error).__name__}: {error}"))
        raise
