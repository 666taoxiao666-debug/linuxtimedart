"""P2: frozen v32 recipe, explicit cross-farm/time-mask adapters; development only.

No downloads, no archive access, no frozen-module writes. Run metadata is exported
separately from server-only weights and predictions. --smoke is synthetic only.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from data_provider.kelmarsh_multiyear import MultiYearWindows
from scripts import train_kelmarsh_multiyear_benchmark as trainer
from utils import kelmarsh_upstream_v32 as u
from utils import penmanshiel_v35 as farm

PROTOCOL = ROOT / "configs/kelmarsh_upstream_protocol_v32.json"
SEEDS = (2024, 2025, 2026)
COMPARISONS = {
    "R": ("P-upstream", "P-last10"),
    "Z1": ("K-upstream", "PersistenceHour"),
    "Z2": ("K-upstream", "K-last10"),
    "F1": ("FT-1", "Scratch-1"),
    "F3": ("FT-3", "Scratch-3"),
}
ARMS = ("K-last10", "K-upstream", "P-last10", "P-upstream",
        "FT-1", "Scratch-1", "FT-3", "Scratch-3")
SPLITS = {"train": (2016, 2017, 2018, 2019), "validation": (2020,), "development_eval": (2021,)}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    u.write_json(path, value)


def source_code():
    paths = {Path(__file__).resolve(), ROOT / "tests/test_train_penmanshiel_p2_v35.py"}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name and Path(name).is_absolute():
            path = Path(name).resolve()
            if path.suffix == ".py" and ROOT in path.parents:
                paths.add(path)
    return {str(p.relative_to(ROOT)): u.file_sha256(p) for p in sorted(paths)}


def fewshot_bounds(months):
    if months not in (1, 3):
        raise ValueError("Only the two preregistered month budgets are allowed")
    lo = np.datetime64("2019-12-01" if months == 1 else "2019-10-01", "ns")
    hi = np.datetime64("2020-01-01", "ns")
    hours = int((hi - lo) / np.timedelta64(1, "h"))
    cut = lo + np.timedelta64(4 * hours // 5, "h")
    return lo, cut, hi


def preprocessing_state(series):
    return {"mean": series.scaler_mean.tolist(), "scale": series.scaler_scale.tolist(),
            "fill": series.audit["upstream"]["fill_values_training_mean"],
            "channels": list(series.channels)}


def adapt_statistics(template, upstream_frame, *, fit_rows=None, state=None):
    """Only raw SCADA/queue are reused. Never reuse target-fitted upstream fill.

    A supplied source state takes a separate branch with no target reductions.
    Few-shot fit_rows is applied BEFORE either filling or fitting the scaler.
    """
    if (fit_rows is None) == (state is None):
        raise ValueError("Supply exactly a fit-row mask or an external frozen state")
    features = upstream_frame.reindex(pd.DatetimeIndex(template.dates)).to_numpy(float)
    if list(upstream_frame.columns) != u.UPSTREAM_FEATURES:
        raise ValueError("Feature order changed")
    missing = ~np.isfinite(features).all(axis=1)
    if state is None:
        fit_rows = np.asarray(fit_rows, dtype=bool)
        if fit_rows.shape != template.dates.shape or not fit_rows.any():
            raise ValueError("Empty or misaligned fit-row mask")
        subset = features[fit_rows]
        fill = np.array([np.mean(c[np.isfinite(c)]) if np.isfinite(c).any() else np.nan for c in subset.T])
    else:
        if state["channels"] != template.channels:
            raise ValueError("Source and target channel orders differ")
        fill = np.asarray(state["fill"], float)
    if not np.isfinite(fill).all():
        raise ValueError("No finite fit observation for an upstream feature")
    features = np.where(np.isfinite(features), features, fill)
    # base.raw contains only the frozen eight SCADA inputs and unchanged power.
    raw = np.column_stack([template.base.raw[:, :-1], features, missing.astype(float), template.base.raw[:, -1]])
    if state is None:
        mean = np.nanmean(raw[fit_rows], axis=0)
        std = np.nanstd(raw[fit_rows], axis=0)
        scale = np.where(std == 0, 1., std)
    else:
        mean, scale = np.asarray(state["mean"], float), np.asarray(state["scale"], float)
    if not (np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all()):
        raise ValueError("Invalid fit-only statistics")
    result = copy.copy(template)
    result.raw, result.scaler_mean, result.scaler_scale = raw, mean.copy(), scale.copy()
    result.values = ((raw - mean) / scale).astype(np.float32)
    result.audit = {"channels": result.channels, "scaler_mean": mean.tolist(), "scaler_scale": scale.tolist(),
                    "statistics_origin": "source_only" if state is not None else "explicit_fit_rows",
                    "fit_rows": None if state is not None else int(fit_rows.sum()),
                    "upstream": {"fill_values_training_mean": fill.tolist()}}
    if not np.array_equal(result.raw[:, -1], template.base.raw[:, -1], equal_nan=True):
        raise AssertionError("Labels changed during input preprocessing")
    return result


def fewshot_windows(series, months):
    lo, cut, hi = fewshot_bounds(months)
    starts = series.starts
    first = series.dates[starts]
    first_target = series.dates[starts + series.input_len]
    last_target = series.dates[starts + series.input_len + series.horizon - 1]
    fit = starts[(first >= lo) & (last_target < cut)]
    selection = starts[(first >= lo) & (first_target >= cut) & (last_target < hi)]
    if not len(fit) or not len(selection):
        raise ValueError("Few-shot fit or validation has no eligible windows")
    return (MultiYearWindows(series, fit, "fewshot_fit", "all"),
            MultiYearWindows(series, selection, "fewshot_validation", "all"))


def configuration(arm, protocol):
    channels = 9 if arm.endswith("last10") else 21
    family = protocol["families"]["TSMixerBasicSmall"]
    return {"arm": arm, "family": "TSMixerBasicSmall", "channels": channels,
            "args": copy.deepcopy(family["args"]),
            "expected_parameters": family["parameters"]["last10" if channels == 9 else "upstream"]}


def view(windows, arm):
    indices = list(range(8)) + [20] if arm.endswith("last10") else list(range(21))
    return trainer.ChannelView(windows, indices)


def fit_one(config, protocol, fit, selection, seed, device, directory, contract, warm=None):
    """Run the exact frozen fit_config bytecode with a scoped model factory.

    No module monkeypatch: the new function has a private globals mapping. The
    only factory difference is loading the registered source checkpoint before
    epoch 0. Optimizer, scheduler, RNG, loss and early stopping remain frozen.
    """
    recipe = copy.deepcopy(protocol)
    if warm is not None:
        recipe["training"]["learning_rate"] = 1e-4

    def factory(cfg):
        model, count = trainer.build_model(cfg)
        if warm is not None:
            model.load_state_dict(warm, strict=True)
        return model, count

    function = types.FunctionType(trainer.fit_config.__code__,
        {**trainer.fit_config.__globals__, "build_model": factory}, trainer.fit_config.__name__, trainer.fit_config.__defaults__)
    function.__kwdefaults__ = trainer.fit_config.__kwdefaults__.copy()
    return function(config, recipe, fit=fit, selection=selection, contract=contract,
                    seed=seed, device=device, out=directory)


def decisions(comparisons):
    if set(comparisons) != set(COMPARISONS):
        raise ValueError("Exactly the five preregistered comparisons are required")
    return {name: {"supported": bool(c["absolute_error"]["ci95_low"] > 0 and c["holm_p"] <= .05
                                     and c["rmse_difference"]["ci95_low_kw"] > 0),
                   "failure_wording": "本次开发实验未获得支持"} for name, c in comparisons.items()}


def evaluate(predictions, windows, normal_mask, protocol):
    truth, _ = windows.truth_and_anchor()
    issue = windows.target_times()[:, 0] - np.timedelta64(1, "h")
    metrics = {}
    for cohort, mask in (("all", np.ones(len(truth), bool)), ("normal", normal_mask)):
        if not mask.any():
            metrics[cohort] = {"status": "NO_WINDOWS"}
            continue
        metrics[cohort] = {arm: u.seed_mean([u.metrics(p[mask], truth[mask], 2050) for p in ps])
                           for arm, ps in predictions.items()}
    specs = {k: {"candidate": a, "reference": b, "question": k} for k, (a, b) in COMPARISONS.items()}
    comparisons = u.compare_all(predictions, truth, issue, specs, None, protocol["statistics"])
    return {"metrics": metrics, "comparisons": comparisons, "decisions": decisions(comparisons),
            "windows": {"all": len(truth), "normal": int(normal_mask.sum())},
            "scope": "Penmanshiel 2021 development only; no independent confirmation; F1/F3 compare complete recipes"}


def checked_file(path, expected, identities):
    digest = u.file_sha256(path)
    if digest != expected:
        raise ValueError(f"Input SHA mismatch: {path}")
    identities[str(path)] = {"sha256": digest, "bytes": Path(path).stat().st_size}
    return path


def load_data(data_root, protocol):
    """Use P1 prepared data and registered normalized Kelmarsh development rows."""
    identities = {}
    penroot = data_root / farm.DATA_SUBDIR
    # Git-tracked P1 evidence supplies expected byte hashes, not a fresh discovery.
    meta = u.read_json(ROOT / "output/penmanshiel_hourly/prep_v35/preparation.json")
    audit = u.read_json(ROOT / "output/penmanshiel_hourly/prep_v35/series_audit.json")
    pcsv = checked_file(penroot / "hourly/penmanshiel_multiyear_hourly_enriched.csv", meta["prepared"]["sha256"], identities)
    upcsv = checked_file(penroot / "hourly/penmanshiel_upstream_raw.csv", audit["upstream_file"]["sha256"], identities)
    pf = pd.read_csv(pcsv)
    if not pf.label_year.isin(range(2016, 2022)).all():
        raise ValueError("Penmanshiel CSV escaped registered development years")
    pup = pd.read_csv(upcsv, index_col="timestamp_hour_end_utc")
    pup.index = pd.to_datetime(pup.index)
    pen = u.KelmarshUpstreamSeries(pf, pup, splits=SPLITS, start=meta["start"], end=farm.END, preparation=meta)
    # v32 requires its 2016-2023 rows to equal this registered v31 CSV exactly.
    # Reuse that verified preparation and retain only <=2020 for K fitting/selection.
    ref = protocol["data"]["v31_registered_preparation"]
    kcsv = checked_file(data_root / ref["csv"], ref["csv_sha256"], identities)
    kmeta_path = checked_file(kcsv.parent.parent / "metadata/preparation.json", ref["preparation_json_sha256"], identities)
    kmeta = u.read_json(kmeta_path)
    kf = pd.concat([chunk[chunk.label_year.between(2016, 2020)] for chunk in pd.read_csv(kcsv, chunksize=50000)], ignore_index=True)
    up = protocol["data"]["upstream"]
    manifest = data_root / up["download_manifest"]
    if u.text_sha256_lf(manifest) != up["download_manifest_sha256_lf"]:
        raise ValueError("MIDAS manifest identity changed")
    hashes = u.manifest_hashes(u.read_json(manifest))
    ids = [int(s["src_id"]) for s in up["stations"]]
    station_paths, _ = u.station_files(data_root / up["development_roots"][0]["root"], ids, range(2016, 2021), hashes, required=True)
    identities.update({str(e["path"]): {"sha256": e["sha256"], "bytes": e["path"].stat().st_size} for e in station_paths})
    obs = u.load_station_observations(station_paths, ids)
    labels = pd.date_range("2016-05-01", "2021-01-01", freq="h")
    kup = u.hourly_upstream(obs, up["stations"], labels, 1)
    kel = u.KelmarshUpstreamSeries(kf, kup, splits={k: v for k, v in SPLITS.items() if k != "development_eval"},
        start="2016-05-01", end="2021-01-01T01:00:00Z", preparation=kmeta)
    if not np.isin(kel.label_year, range(2016, 2021)).all():
        raise ValueError("Kelmarsh includes undeclared training/selection years")
    return pen, pup, kel, identities


def build_variants(pen, pup, kel):
    source = preprocessing_state(kel)
    transfer = adapt_statistics(pen, pup, state=source)
    variants = {"P-last10": pen, "P-upstream": pen, "K-last10": transfer, "K-upstream": transfer}
    audits = {"P": pen.audit, "K": kel.audit, "target_with_source_statistics": transfer.audit}
    for n in (1, 3):
        lo, cut, hi = fewshot_bounds(n)
        scratch = adapt_statistics(pen, pup, fit_rows=(pen.dates >= lo) & (pen.dates < cut))
        variants[f"FT-{n}"] = transfer
        variants[f"Scratch-{n}"] = scratch
        fit, val = fewshot_windows(scratch, n)
        audits[f"fewshot_{n}"] = {"start": str(lo), "cut": str(cut), "end_exclusive": str(hi),
            "fit_windows": len(fit), "validation_windows": len(val), "scratch_statistics": scratch.audit,
            "fit_window_sha256": hashlib.sha256(fit.window_starts.tobytes()).hexdigest(),
            "validation_window_sha256": hashlib.sha256(val.window_starts.tobytes()).hexdigest()}
    return variants, audits


def run_fits(pen, pup, kel, protocol, out, device, identity, *, smoke=False):
    variants, audits = build_variants(pen, pup, kel)
    save(out / "data_audit.json", audits)
    pending = {}
    fit_records = []
    source_states = {}
    for arm in ARMS:
        series = variants[arm]
        if arm.startswith("K-"):
            fit, val = kel.windows("train"), kel.windows("validation")
        elif arm.startswith("P-"):
            fit, val = pen.windows("train"), pen.windows("validation")
        else:
            fit, val = fewshot_windows(series, int(arm[-1]))
        config = configuration(arm, protocol)
        seeds = (2024,) if smoke else SEEDS
        for seed in seeds:
            directory = out / "fits" / arm / f"s{seed}"
            warm = source_states.get(seed) if arm.startswith("FT-") else None
            if arm.startswith("FT-") and warm is None:
                raise ValueError("Source fit for same seed not completed")
            contract = {"stage": "P2", "arm": arm, "seed": seed, "identity": identity,
                        "warm_state_sha256": None if warm is None else trainer.v26run.state_digest(warm),
                        "statistics": preprocessing_state(series if not arm.startswith("K-") else kel),
                        "fit_starts_sha256": hashlib.sha256(fit.window_starts.tobytes()).hexdigest(),
                        "validation_starts_sha256": hashlib.sha256(val.window_starts.tobytes()).hexdigest()}
            print(f"[FIT] {arm} seed={seed} fit={len(fit)} val={len(val)}", flush=True)
            record = fit_one(config, protocol, view(fit, arm), view(val, arm), seed, device, directory, contract, warm)
            checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
            if arm == "K-upstream":
                source_states[seed] = checkpoint["model"]
            fit_records.append(record)
            pending[(arm, seed)] = directory
    expected = 8 if smoke else 24
    if len(fit_records) != expected:
        raise ValueError("Wrong number of fitted models")
    # Seal every selected checkpoint BEFORE any development-evaluation metric.
    freeze = {"utc": now(), "fits": fit_records, "expected_fits": expected,
              "source_code": source_code(), "identity": identity}
    save(out / "selection_freeze.json", freeze)
    evaluation = pen.windows("development_eval")
    truth, persistence = evaluation.truth_and_anchor()
    predictions = {"PersistenceHour": [persistence], "PersistenceLast10": [evaluation.last10_anchor()]}
    for (arm, seed), directory in pending.items():
        target = variants[arm].windows("development_eval")
        if not np.array_equal(target.window_starts, evaluation.window_starts):
            raise ValueError("Evaluation queues differ across arms")
        if not np.array_equal(target.truth_and_anchor()[0], truth):
            raise ValueError("Evaluation labels differ across arms")
        model, _ = trainer.build_model(configuration(arm, protocol))
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        prediction = trainer.v26run.predict(model.to(device), view(target, arm), seed, device, protocol["training"]["eval_batch_size"])
        predictions.setdefault(arm, []).append(prediction)
        directory_eval = out / "evaluation"
        directory_eval.mkdir(exist_ok=True)
        np.savez_compressed(directory_eval / f"{arm}_s{seed}.npz", prediction=prediction)
    normal = pen.normal_window[pen.split_of == "development_eval"]
    np.savez_compressed(out / "evaluation/cohort.npz", truth=truth, times=evaluation.target_times(),
                        starts=evaluation.window_starts, normal_mask=normal, persistence=persistence,
                        last10=predictions["PersistenceLast10"][0])
    result = evaluate(predictions, evaluation, normal, protocol)
    result.update({"status": "COMPLETED", "smoke": smoke, "fits_completed": expected,
                   "per_seed": {arm: [u.metrics(p, truth, 2050) for p in ps] for arm, ps in predictions.items()}})
    save(out / "result.json", result)
    # Independently reload registered arrays for the final statistics computation.
    reloaded = {k: v for k, v in predictions.items() if k.startswith("Persistence")}
    for (arm, seed) in pending:
        with np.load(out / f"evaluation/{arm}_s{seed}.npz", allow_pickle=False) as z:
            reloaded.setdefault(arm, []).append(z["prediction"])
    replay = evaluate(reloaded, evaluation, normal, protocol)
    if u.canonical_sha256(replay) != u.canonical_sha256({k: result[k] for k in replay}):
        raise ValueError("Recomputed report differs from registered result")
    save(out / "report_recomputed.json", {"passed": True, "sha256": u.canonical_sha256(replay)})
    return result


def synthetic_data():
    frame = trainer.synthetic_prepared_frame(years=(2019, 2020, 2021), days=31)
    labels = pd.DatetimeIndex(pd.to_datetime(frame.timestamp_hour_end_utc, utc=True).dt.tz_localize(None).unique()).sort_values()
    rng = np.random.default_rng(2024)
    up = pd.DataFrame(rng.normal(size=(len(labels), 11)), index=labels, columns=u.UPSTREAM_FEATURES)
    up.iloc[::17, 0] = np.nan
    prep = {"schema_version": 2, "curve_years": [2019], "prepared": {"sha256": "synthetic"}}
    series = u.KelmarshUpstreamSeries(frame, up, splits={"train": [2019], "validation": [2020], "development_eval": [2021]}, preparation=prep)
    return series, up, series


def export_metadata(out):
    manifest = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".csv", ".log", ".md") and path.name != "export_sha256.json":
            manifest[str(path.relative_to(out))] = {"bytes": path.stat().st_size, "sha256": u.file_sha256(path)}
    save(out / "audit/export_sha256.json", manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock_path = args.data_root.resolve() / "outputs/logs/Penmanshiel/p2_v35.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "audit/launch.json").exists():
            raise ValueError("Run directory already launched; do not duplicate a real instance")
        started = time.monotonic()
        protocol = u.read_json(PROTOCOL)
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        before = source_code()
        identity = {"protocol_sha256": u.file_sha256(PROTOCOL), "stage_plan_sha256_lf": u.text_sha256_lf(ROOT / "docs/STAGE_PLAN.md"),
                    "source_code": before, "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        save(out / "audit/launch.json", {**identity, "utc": now(), "pid": os.getpid(), "command": sys.argv,
             "smoke": args.smoke, "device": args.device, "python": sys.version, "torch": torch.__version__,
             "determinism": trainer.v26run.determinism_record()})
        try:
            if args.smoke:
                pen, pup, kel = synthetic_data()
                identities = {"synthetic": True}
                protocol["training"].update(max_epochs=1, scheduler_t_max=1)
                protocol["statistics"]["replicates"] = 100
            else:
                pen, pup, kel, identities = load_data(args.data_root.resolve(), protocol)
            save(out / "input_identity.json", identities)
            save(out / "recipe.json", {"training": protocol["training"], "families": protocol["families"],
                "seeds": SEEDS, "statistics": protocol["statistics"], "comparisons": COMPARISONS,
                "ft_learning_rate": 1e-4, "smoke": args.smoke})
            identity["input_identity_sha256"] = u.canonical_sha256(identities)
            result = run_fits(pen, pup, kel, protocol, out, args.device, identity, smoke=args.smoke)
            after = source_code()
            if any(after.get(p) != sha for p, sha in before.items()):
                raise ValueError("Source code changed during execution")
            save(out / "audit/completed.json", {"utc": now(), "elapsed_seconds": time.monotonic() - started,
                 "status": "COMPLETED", "fits": result["fits_completed"], "source_code": after, "source_unchanged": True})
            export_metadata(out)
            print("[COMPLETE] " + json.dumps(result["decisions"], ensure_ascii=False), flush=True)
        except BaseException as error:
            save(out / "audit/failed.json", {"utc": now(), "type": type(error).__name__, "message": str(error),
                 "elapsed_seconds": time.monotonic() - started})
            export_metadata(out)
            raise


if __name__ == "__main__":
    main()
