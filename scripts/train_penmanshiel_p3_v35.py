"""P3: three frozen rolling folds; all fits precede a single held-out evaluation.

--smoke uses synthetic data only. --resume reuses the same real directory and
checks source, input and artifact identities; final evaluation cannot resume.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from scripts import train_penmanshiel_p2_v35 as p2
from scripts import prepare_penmanshiel_v35_p3 as prep
from utils.kelmarsh_backbone_information import holm_adjust

u, farm, trainer = p2.u, p2.farm, p2.trainer
PROTOCOL_SHA256 = prep.PROTOCOL_SHA256
save = p2.save


def source_code():
    paths = {Path(__file__).resolve(), ROOT / "tests/test_train_penmanshiel_p3_v35.py", Path(prep.__file__).resolve()}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name and Path(name).is_absolute():
            path = Path(name).resolve()
            if ROOT in path.parents and path.suffix == ".py" and path.is_file():
                paths.add(path)
    return {str(p.relative_to(ROOT)): u.file_sha256(p) for p in sorted(paths)}


def fewshot_bounds(year, months):
    if months not in (1, 3):
        raise ValueError("Only preregistered month budgets")
    lo = np.datetime64(f"{year}-{'12' if months == 1 else '10'}-01", "ns")
    hi = np.datetime64(f"{year+1}-01-01", "ns")
    hours = int((hi - lo) / np.timedelta64(1, "h"))
    return lo, lo + np.timedelta64(4 * hours // 5, "h"), hi


def fewshot_windows(series, year, months):
    lo, cut, hi = fewshot_bounds(year, months)
    s = series.starts
    first, target, last = series.dates[s], series.dates[s + 24], series.dates[s + 27]
    fit = s[(first >= lo) & (last < cut)]
    val = s[(first >= lo) & (target >= cut) & (last < hi)]
    if not len(fit) or not len(val):
        raise ValueError("Empty few-shot segment")
    return p2.MultiYearWindows(series, fit, "fewshot_fit", "all"), p2.MultiYearWindows(series, val, "fewshot_validation", "all")


def fold_role(fold, count, protocol):
    return "confirmation" if fold["confirmation_candidate"] and count >= protocol["confirmation_window_minimum"] else "descriptive"


def load_inputs(data_root, prepared):
    meta = u.read_json(prepared / "preparation.json")
    pen = pd.read_csv(prepared / "penmanshiel.csv")
    pup = pd.read_csv(prepared / "upstream.csv", index_col="timestamp_hour_end_utc")
    pup.index = pd.to_datetime(pup.index)
    recipe = u.read_json(p2.PROTOCOL)
    identities = u.read_json(prepared / "input_hashes.json")
    ref = recipe["data"]["v31_registered_preparation"]
    kcsv = p2.checked_file(data_root / ref["csv"], ref["csv_sha256"], identities)
    kmeta_path = p2.checked_file(kcsv.parent.parent / "metadata/preparation.json", ref["preparation_json_sha256"], identities)
    kmeta = u.read_json(kmeta_path)
    kframe = pd.concat([chunk[chunk.label_year.between(2016, 2023)] for chunk in pd.read_csv(kcsv, chunksize=50000)], ignore_index=True)
    up = recipe["data"]["upstream"]
    manifest = data_root / up["download_manifest"]
    if u.text_sha256_lf(manifest) != up["download_manifest_sha256_lf"]:
        raise ValueError("Kelmarsh MIDAS registration changed")
    hashes = u.manifest_hashes(u.read_json(manifest))
    ids = [int(s["src_id"]) for s in up["stations"]]
    files = []
    for registered in up["development_roots"]:
        found, _ = u.station_files(data_root / registered["root"], ids, registered["years"], hashes, required=True)
        files.extend(found)
    for e in files:
        identities[str(e["path"])] = {"bytes": e["path"].stat().st_size, "sha256": e["sha256"]}
    obs = u.load_station_observations(files, ids)
    labels = pd.date_range("2016-05-01", "2024-01-01", freq="h")
    kup = u.hourly_upstream(obs, up["stations"], labels, 1)
    for name in ("penmanshiel.csv", "upstream.csv", "preparation.json", "development_equality.json"):
        path = prepared / name
        identities[str(path)] = {"bytes": path.stat().st_size, "sha256": u.file_sha256(path)}
    return (pen, pup, meta, kframe, kup, kmeta), identities


def build_fold(data, fold):
    pf, pup, meta, kf, kup, kmeta = data
    year = fold["test_year"]
    splits = {"train": fold["fit_years"], "validation": [fold["validation_year"]], "test": [year]}
    pen = u.KelmarshUpstreamSeries(pf, pup, splits=splits, start=meta["start"],
                                  end=f"{year+1}-01-01T01:00:00Z", preparation=meta)
    kel = u.KelmarshUpstreamSeries(kf, kup, splits={k: v for k, v in splits.items() if k != "test"},
                                  start=kmeta.get("start", "2016-05-01T00:00:00Z"),
                                  end=f"{year}-01-01T01:00:00Z", preparation=kmeta)
    transfer = p2.adapt_statistics(pen, pup, state=p2.preprocessing_state(kel))
    variants = {"K-last10": transfer, "K-upstream": transfer, "P-last10": pen, "P-upstream": pen}
    audits = {"P": pen.audit, "K": kel.audit, "target_with_source_statistics": transfer.audit}
    for n in (1, 3):
        lo, cut, hi = fewshot_bounds(fold["validation_year"], n)
        scratch = p2.adapt_statistics(pen, pup, fit_rows=(pen.dates >= lo) & (pen.dates < cut))
        variants[f"FT-{n}"] = transfer
        variants[f"Scratch-{n}"] = scratch
        fit, val = fewshot_windows(scratch, fold["validation_year"], n)
        audits[f"fewshot_{n}"] = {"start": str(lo), "cut": str(cut), "end_exclusive": str(hi),
            "fit_windows": len(fit), "validation_windows": len(val), "statistics": scratch.audit,
            "fit_starts_sha256": hashlib.sha256(fit.window_starts.tobytes()).hexdigest(),
            "validation_starts_sha256": hashlib.sha256(val.window_starts.tobytes()).hexdigest()}
    return pen, kel, variants, audits


def fit_fold(data, fold, recipe, out, device, identity, *, smoke=False):
    pen, kel, variants, audits = build_fold(data, fold)
    save(out / fold["id"] / "data_audit.json", audits)
    records, source_states = [], {}
    for arm in p2.ARMS:
        series = variants[arm]
        if arm.startswith("K-"):
            fit, val = kel.windows("train"), kel.windows("validation")
        elif arm.startswith("P-"):
            fit, val = pen.windows("train"), pen.windows("validation")
        else:
            fit, val = fewshot_windows(series, fold["validation_year"], int(arm[-1]))
        if smoke:
            fit.window_starts = fit.window_starts[:16]
            val.window_starts = val.window_starts[:16]
        for seed in ((2024,) if smoke else p2.SEEDS):
            folder = out / fold["id"] / "fits" / arm / f"s{seed}"
            warm = source_states[seed] if arm.startswith("FT-") else None
            contract = {"stage": "P3", "fold": fold, "arm": arm, "seed": seed, "identity": identity,
                "warm_state_sha256": None if warm is None else trainer.v26run.state_digest(warm),
                "statistics": p2.preprocessing_state(kel if arm.startswith("K-") else series),
                "fit_starts_sha256": hashlib.sha256(fit.window_starts.tobytes()).hexdigest(),
                "validation_starts_sha256": hashlib.sha256(val.window_starts.tobytes()).hexdigest()}
            print(f"[FIT] {fold['id']}/{arm}/s{seed} fit={len(fit)} val={len(val)}", flush=True)
            record = p2.fit_one(p2.configuration(arm, recipe), recipe, p2.view(fit, arm), p2.view(val, arm),
                                seed, device, folder, contract, warm)
            if arm == "K-upstream":
                source_states[seed] = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)["model"]
            # Reuse flags are operational, not part of a selected model's identity.
            record.pop("reused_from_registry", None)
            records.append({"directory": str(folder.relative_to(out)), "record": record,
                            "trained_sha256": u.file_sha256(folder / "trained.json")})
    return records


def validate_freeze(out, freeze, expected):
    if freeze["expected_fits"] != expected or len(freeze["fits"]) != expected:
        raise ValueError("All folds must freeze before testing")
    paths = [e["directory"] for e in freeze["fits"]]
    if len(set(paths)) != expected:
        raise ValueError("Duplicate selected model")
    for entry in freeze["fits"]:
        folder = out / entry["directory"]
        if u.file_sha256(folder / "trained.json") != entry["trained_sha256"]:
            raise ValueError("Selected registration changed")
        trainer.verify_trained(folder, entry["record"]["contract"])


def begin_evaluation(out, freeze, expected):
    validate_freeze(out, freeze, expected)
    marker = out / "test_evaluation_started.json"
    # Exclusive create precedes even prediction generation. Crash => no retry.
    with marker.open("x", encoding="utf-8") as stream:
        import json
        json.dump({"utc": p2.now(), "selection_sha256": u.file_sha256(out / "selection_freeze.json")}, stream)
        stream.flush()
        os.fsync(stream.fileno())


def assert_can_resume(out):
    if (out / "test_evaluation_started.json").exists() or (out / "audit/completed.json").exists():
        raise ValueError("Final test started/completed: restart forbidden")


def evaluate_arrays(predictions, truth, issue, normal, recipe, role):
    metrics = {}
    for name, mask in (("all", np.ones(len(truth), bool)), ("normal", normal)):
        metrics[name] = ({a: u.seed_mean([u.metrics(q[mask], truth[mask], 2050) for q in qs])
                          for a, qs in predictions.items()} if mask.any() else {"status": "NO_WINDOWS"})
    comparisons = {}
    for family in (("R", "F1", "F3"), ("Z1", "Z2")):
        specs = {n: {"candidate": p2.COMPARISONS[n][0], "reference": p2.COMPARISONS[n][1], "question": n} for n in family}
        values = u.compare_all(predictions, truth, issue, specs, None, recipe["statistics"])
        for n, value in values.items():
            if n.startswith("Z") or role != "confirmation":
                value.pop("holm_p", None)
                value["role"] = "descriptive"
            else:
                value["role"] = "confirmation"
        comparisons.update(values)
    decisions = {n: {"confirmed": role == "confirmation" and c["absolute_error"]["ci95_low"] > 0
                     and c.get("holm_p", 1) <= .05 and c["rmse_difference"]["ci95_low_kw"] > 0,
                     "role": c["role"]} for n, c in comparisons.items()}
    return {"metrics": metrics, "comparisons": comparisons, "decisions": decisions,
        "per_seed": {a: [u.metrics(q, truth, 2050) for q in qs] for a, qs in predictions.items()},
        "windows": {"all": len(truth), "normal": int(normal.sum())}}


def evaluate_fold(data, fold, recipe, out, device, role, smoke=False):
    pen, _, variants, _ = build_fold(data, fold)
    target = pen.windows("test")
    normal = pen.normal_window[pen.split_of == "test"]
    if smoke:
        target.window_starts = target.window_starts[:32]
        normal = normal[:32]
    truth, persistence = target.truth_and_anchor()
    predictions = {"PersistenceHour": [persistence], "PersistenceLast10": [target.last10_anchor()]}
    directory = out / fold["id"] / "evaluation"
    directory.mkdir(parents=True)
    np.savez_compressed(directory / "cohort.npz", truth=truth, starts=target.window_starts,
        times=target.target_times(), normal_mask=normal, persistence=persistence, last10=target.last10_anchor())
    for arm in p2.ARMS:
        windows = variants[arm].windows("test")
        if smoke:
            windows.window_starts = windows.window_starts[:32]
        np.testing.assert_array_equal(windows.window_starts, target.window_starts)
        np.testing.assert_array_equal(windows.truth_and_anchor()[0], truth)
        for seed in ((2024,) if smoke else p2.SEEDS):
            folder = out / fold["id"] / "fits" / arm / f"s{seed}"
            checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
            trained = u.read_json(folder / "trained.json")
            if trainer.v26run.state_digest(checkpoint["model"]) != trained["best_state_sha256"]:
                raise ValueError("Checkpoint state differs from frozen selection")
            model, _ = trainer.build_model(p2.configuration(arm, recipe))
            model.load_state_dict(checkpoint["model"], strict=True)
            prediction = trainer.v26run.predict(model.to(device), p2.view(windows, arm), seed, device, recipe["training"]["eval_batch_size"])
            if prediction.shape != truth.shape or not np.isfinite(prediction).all():
                raise ValueError("Invalid final prediction")
            np.savez_compressed(directory / f"{arm}_s{seed}.npz", prediction=prediction, starts=target.window_starts)
            predictions.setdefault(arm, []).append(prediction)
    issue = target.target_times()[:, 0] - np.timedelta64(1, "h")
    result = evaluate_arrays(predictions, truth, issue, normal, recipe, role)
    result.update(fold=fold, role=role)
    save(out / fold["id"] / "result.json", result)
    save(out / fold["id"] / "evaluation_hashes.json", {p.name: u.file_sha256(p) for p in sorted(directory.iterdir())})
    return result


def synthetic_data():
    # Fixed synthetic October-December spans make both few-shot budgets nonempty.
    frames = []
    for year in range(2019, 2025):
        f = trainer.synthetic_prepared_frame(years=(year,), days=92)
        dates = pd.to_datetime(f.timestamp_hour_end_utc, utc=True)
        offset = pd.Timestamp(f"{year}-10-01", tz="UTC") - dates.min().floor("D")
        f.timestamp_hour_end_utc = dates + offset
        f.label_year = (pd.to_datetime(f.timestamp_hour_end_utc) - pd.Timedelta(hours=1)).dt.year
        frames.append(f)
    frame = pd.concat(frames, ignore_index=True)
    labels = pd.DatetimeIndex(pd.to_datetime(frame.timestamp_hour_end_utc, utc=True).dt.tz_localize(None).unique()).sort_values()
    rng = np.random.default_rng(2024)
    features = pd.DataFrame(rng.normal(size=(len(labels), 11)), index=labels, columns=u.UPSTREAM_FEATURES)
    features.iloc[::17, 0] = np.nan
    meta = {"schema_version": 2, "curve_years": [2019], "prepared": {"sha256": "synthetic"}, "start": "2019-10-01"}
    return (frame, features, meta, frame, features, meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    p = prep.protocol()
    out, data_root = args.out.resolve(), args.data_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lockpath = (out / "smoke.lock") if args.smoke else data_root / "outputs/logs/Penmanshiel/p3_v35.lock"
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_can_resume(out)
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        identity = {"protocol_sha256": PROTOCOL_SHA256, "source_code": source_code(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "device": args.device, "smoke": args.smoke}
        registry = out / "audit/attempts.json" if args.smoke else data_root / "outputs/logs/Penmanshiel/p3_v35_registry.json"
        attempts = u.read_json(registry) if registry.exists() else {"out": str(out), "attempts": []}
        if attempts["out"] != str(out):
            raise ValueError("P3 already registered in another directory")
        if attempts["attempts"] and not args.resume:
            raise ValueError("Existing P3 launch requires explicit --resume")
        if len(attempts["attempts"]) >= p["resume"]["maximum_process_attempts"]:
            raise ValueError("Two-attempt budget exhausted")
        launch = out / "audit/launch.json"
        if launch.exists():
            previous = u.read_json(launch)
            if previous["identity"] != identity:
                raise ValueError("Source/runtime identity changed; no automatic resume")
        else:
            save(launch, {"identity": identity, "source_code": identity["source_code"], "utc": p2.now(),
                 "pid": os.getpid(), "command": sys.argv, "python": sys.version, "torch": torch.__version__,
                 "determinism": trainer.v26run.determinism_record()})
        attempts["attempts"].append({"utc": p2.now(), "pid": os.getpid(), "command": sys.argv})
        save(registry, attempts)
        started = time.monotonic()
        try:
            recipe = u.read_json(ROOT / p["recipe_path"])
            if args.smoke:
                data, inputs = synthetic_data(), {"synthetic": True}
                recipe["training"].update(max_epochs=1, scheduler_t_max=1)
                recipe["statistics"]["replicates"] = 100
            else:
                prep.prepare(data_root, out / "prepared")
                data, inputs = load_inputs(data_root, out / "prepared")
            input_sha = u.canonical_sha256(inputs)
            if (out / "input_identity.json").exists() and u.canonical_sha256(u.read_json(out / "input_identity.json")) != input_sha:
                raise ValueError("Input identity changed across attempts")
            save(out / "input_identity.json", inputs)
            identity = {**identity, "input_identity_sha256": input_sha}
            save(out / "recipe.json", {"training": recipe["training"], "families": recipe["families"], "statistics": recipe["statistics"], "p3": p})
            # Register all fold window roles before any fit. No performance metrics.
            roles = {}
            for fold in p["folds"]:
                pen, kel, _, audits = build_fold(data, fold)
                count = int((pen.split_of == "test").sum())
                roles[fold["id"]] = {"windows": count, "role": fold_role(fold, count, p),
                                      "upstream_missing_by_split": pen.audit["upstream"]["missing_row_share_by_split"]}
                if count == 0:
                    raise ValueError("No eligible test windows; stop before evaluation")
                save(out / fold["id"] / "data_audit.json", audits)
            save(out / "fold_roles.json", roles)
            fits = []
            for fold in p["folds"]:
                fits.extend(fit_fold(data, fold, recipe, out, args.device, identity, smoke=args.smoke))
            expected = 24 if args.smoke else p["expected_fits"]
            if source_code() != identity["source_code"]:
                raise ValueError("Source changed before evaluation")
            freeze = {"utc": p2.now(), "identity": identity, "expected_fits": expected, "fits": fits, "roles": roles}
            if (out / "selection_freeze.json").exists():
                old = u.read_json(out / "selection_freeze.json")
                if old["fits"] != fits or old["identity"] != identity or old["roles"] != roles:
                    raise ValueError("Previous selection freeze changed")
                freeze = old
            else:
                save(out / "selection_freeze.json", freeze)
            begin_evaluation(out, freeze, expected)
            results = {f["id"]: evaluate_fold(data, f, recipe, out, args.device, roles[f["id"]]["role"], args.smoke) for f in p["folds"]}
            overall = {}
            for comparison in p["confirmation_family"]:
                confirmed = [f["test_year"] for f in p["folds"] if results[f["id"]]["decisions"][comparison]["confirmed"]]
                wording = p["reporting"]["two_years" if len(confirmed) == 2 else "one_year" if confirmed else "zero_years"]
                overall[comparison] = {"confirmed_years": confirmed, "wording": wording}
            save(out / "result.json", {"status": "COMPLETED", "smoke": args.smoke, "fits_completed": expected,
                                       "folds": results, "overall": overall})
            if source_code() != identity["source_code"]:
                raise ValueError("Source changed during evaluation")
            save(out / "audit/completed.json", {"utc": p2.now(), "fits": expected, "source_code": source_code(),
                "elapsed_seconds": time.monotonic() - started, "source_unchanged": True})
            print("[COMPLETE] P3 all folds", flush=True)
        except BaseException as exc:
            save(out / "audit/failed.json", {"utc": p2.now(), "type": type(exc).__name__, "message": str(exc),
                "test_started": (out / "test_evaluation_started.json").exists(), "elapsed_seconds": time.monotonic() - started})
            raise
        finally:
            # Prepared raw CSVs and all predictions/weights remain server-only.
            manifest = {str(path.relative_to(out)): {"bytes": path.stat().st_size, "sha256": u.file_sha256(path)}
                for path in sorted(out.rglob("*")) if path.is_file() and path.suffix in (".json", ".log", ".md")
                and path.name != "export_sha256.json"}
            save(out / "audit/export_sha256.json", manifest)


if __name__ == "__main__":
    main()
