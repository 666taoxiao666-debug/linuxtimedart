#!/usr/bin/env python3
"""P1: offline development-only farm adapter; no training and no held-out content parsing."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scripts import prepare_kelmarsh_multiyear as frozen
from scripts.prepare_kelmarsh_multiyear_v31 import canonical_ticks
from utils import kelmarsh_enrichment as enrich
from utils import kelmarsh_upstream_v31 as upstream
from utils import penmanshiel_v35 as farm


def save(path, value):
    frozen.save_json(path, value)


def source_code():
    paths = {Path(__file__).resolve(), ROOT / "tests/test_prepare_penmanshiel_v35.py"}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        # Runtime-generated modules (e.g. torch's _classes.py) can advertise a
        # bare filename. It is not a repository path relative to our cwd.
        if name and Path(name).is_absolute():
            path = Path(name).resolve()
            if path.suffix == ".py" and ROOT in path.parents:
                paths.add(path)
    return {str(p.relative_to(ROOT)): frozen.digest(p) for p in sorted(paths)}


def inputs(data_root):
    d2 = {}
    for name, sha in farm.D2_SHA.items():
        path = ROOT / farm.D2_SUBDIR / name
        if upstream.text_sha256_lf(path) != sha:
            raise ValueError(f"Frozen D2 metadata changed: {name}")
        d2[name] = json.loads(path.read_text(encoding="utf-8"))
    part, registry, hashes = d2["part_A.json"], d2["test_year_registration.json"], d2["input_hashes.json"]
    if registry["test_year"] != 2022 or registry["development_years"] != list(farm.YEARS):
        raise ValueError("Frozen year registration differs")
    if part["midas"]["qualified"] != list(farm.STATIONS):
        raise ValueError("Frozen qualified station set differs")
    stations = [next(s for s in part["midas"]["stations"] if s["src_id"] == sid) for sid in farm.STATIONS]
    members = [e for e in registry["members"] if int(e["begin"][:4]) in farm.YEARS]
    for entry in members:
        farm.development_member(entry)
    files = farm.station_files(stations)
    selected = set(e["archive"] for e in registry["members"])
    selected.update(e["path"] for e in files)
    # Protect later station files as opaque bytes. Never parse their stored headers either.
    protected_stations = [e["path"] for e in part["midas"]["protected_files"]
                          if int(Path(e["path"]).parent.name) in farm.STATIONS]
    selected.update(protected_stations)
    static = next(p for p in hashes if Path(p).name == "Penmanshiel_WT_static.csv")
    selected.add(static)
    resolved, verified = {}, {}
    for original in sorted(selected):
        suffix = original.split("/datasets/", 1)[1]
        path = (data_root / "datasets" / suffix).resolve()
        if not path.is_relative_to((data_root / "datasets").resolve()):
            raise ValueError("Input escapes data root")
        expected = hashes[original]
        if path.stat().st_size != expected["bytes"] or frozen.digest(path) != expected["sha256"]:
            raise ValueError(f"Frozen source hash/size mismatch: {path}")
        if expected.get("md5") and frozen.digest(path, "md5") != expected["md5"]:
            raise ValueError(f"Frozen source MD5 mismatch: {path}")
        resolved[original] = path
        verified[str(path)] = expected
    return members, stations, files, resolved, verified, static, protected_stations, part["static"]["center"]


def prepare(data_root, out):
    started = time.monotonic()
    destination = data_root / farm.DATA_SUBDIR
    # Refuse even partial prior products; do not overwrite an earlier attempt.
    if destination.exists() or out.exists():
        raise FileExistsError("Preparation/audit directory already exists; preserve prior attempt")
    out.mkdir(parents=True)
    before = source_code()
    save(out / "audit/launch.json", {"utc": datetime.now(timezone.utc).isoformat(),
         "argv": sys.argv, "source_code": before, "d2_metadata_sha256_lf": farm.D2_SHA,
         "training": False, "protected_year_content_access": False, "budget_seconds": 3600})
    members, stations, files, resolved, verified, static, protected, center = inputs(data_root)
    save(out / "input_hashes.json", verified)
    start = farm.first_complete_month(members)
    ratings, static_audit = farm.validate_static(pd.read_csv(resolved[static], encoding="utf-8-sig"))
    save(out / "routing.json", {"start": start, "start_basis": "first full calendar month covered by every turbine's registered file intervals; missing cells retained",
         "center": center, "stations": [{k: s[k] for k in ("src_id", "distance_km", "bearing_deg")} for s in stations],
         "development_members": members, "protected_station_paths_hash_only": protected,
         "protected_scada_paths_hash_only": sorted(p for p in resolved if "_SCADA_" in p and p not in {e["archive"] for e in members}),
         "static": static_audit, "splits": farm.SPLITS})
    print(f"[INPUTS] verified {len(verified)} files; start={start}", flush=True)
    frames, duplicate_records, decisions, optional, label_checks = [], [], {}, {}, []
    for unit in farm.TURBINES:
        key = f"{unit:02d}"
        parts = []
        for entry in members:
            if int(entry["turbine"]) == unit:
                ticks, missing = farm.read_member(entry, resolved[entry["archive"]])
                parts.append(ticks)
                optional[entry["member"]] = missing
        ticks, audit = canonical_ticks(parts)
        duplicate_records.append({"turbine_id": unit, **audit})
        del parts
        hourly, decisions[key] = frozen.prepare_turbine(key, ticks, ratings[key]["rated_power_kw"], start, enrich.RULES)
        # Independent identity check of the label path, using the unchanged strict aggregator.
        selected = ticks[ticks.timestamp >= pd.Timestamp(start)]
        reference = frozen.hourly_scada(selected[["timestamp", "power_kw"]].copy())
        pd.testing.assert_series_equal(hourly.power_kw, reference.power_kw, check_names=False, check_exact=True)
        pd.testing.assert_series_equal(hourly.timestamp_hour_end_utc, reference.timestamp_hour_end_utc, check_exact=True)
        label_checks.append({"turbine_id": unit, "hourly_rows": len(hourly), "label_cells_modified": 0,
                             "label_cells_clipped": 0, "label_cells_imputed": 0})
        frames.append(hourly)
        print(f"[SCADA] turbine={key} ticks={len(ticks)} hourly={len(hourly)}", flush=True)
    frame = pd.concat(frames, ignore_index=True)
    del frames, ticks, hourly, selected, reference
    original_labels = frame.power_kw.copy()
    curves = {}
    frame["flag_underperformance"] = False
    for unit, group in frame.groupby("turbine_id"):
        rated = float(group.rated_power_kw.iloc[0])
        curves[unit] = enrich.fit_power_curve(group.wind_speed_mps, group.power_kw, group.label_year,
            farm.CURVE_YEARS, group.flag_stopped, group.flag_derated, rated, enrich.RULES)
        frame.loc[group.index, "flag_underperformance"] = enrich.underperformance_flag(
            group.wind_speed_mps, group.power_kw, curves[unit], rated, enrich.RULES)
    frame["normal_operation"] = enrich.normal_operation(frame).to_numpy()
    pd.testing.assert_series_equal(frame.power_kw, original_labels, check_exact=True)
    frame = frame.sort_values(["turbine_id", "timestamp_hour_end_utc"])
    if not frame.label_year.isin(farm.YEARS).all():
        raise ValueError("Prepared labels escaped development years")
    prepared = frozen.write_frame(frame, destination / "hourly", "penmanshiel_multiyear_hourly_enriched.csv")
    metadata = {"schema_version": 2, "record": 16807304, "years": farm.YEARS, "start": start,
        "end_exclusive": farm.END, "prepared": prepared, "rules": enrich.RULES,
        "curve_years": farm.CURVE_YEARS, "power_curves": curves, "static": static_audit,
        "source_sha256": before, "available_capacity_decisions": decisions,
        "missing_optional_columns": optional, "labels_modified": False, "clipping": False, "interpolation": False,
        "duplicate_normalization": duplicate_records, "label_identity_checks": label_checks,
        "target_transformation": "strict original 6x10-minute hourly average",
        "normal_cohort_uses_future_target_state": True, "primary_cohort": "all"}
    save(destination / "metadata/preparation.json", metadata)
    save(out / "preparation.json", metadata)
    frozen.quality_by_year(frame).to_csv(out / "quality_by_year.csv", index=False)
    print("[UPSTREAM] loading fixed development station files", flush=True)
    observations = upstream.load_station_observations(
        [{"path": resolved[e["path"]], "year": e["year"]} for e in files], farm.STATIONS)
    labels = pd.date_range(frame.timestamp_hour_end_utc.min(), frame.timestamp_hour_end_utc.max(), freq="h").tz_localize(None)
    features = upstream.hourly_upstream(observations, stations, labels, lag=1)
    features.index.name = "timestamp_hour_end_utc"
    upstream_file = frozen.write_frame(features.reset_index(), destination / "hourly", "penmanshiel_upstream_raw.csv")
    # File-backed series verifies the actual persisted CSV SHA, not just in-memory objects.
    series = farm.assemble_series(prepared["path"], features, metadata, start)
    audit = farm.series_audit(series, features)
    # Exact input-only fill guard against the provider's original label vector.
    if not np.array_equal(series.raw[:, -1], series.base.raw[:, -1], equal_nan=True):
        raise ValueError("Upstream assembly altered labels")
    audit["upstream_file"] = upstream_file
    save(destination / "metadata/series_audit.json", audit)
    save(out / "series_audit.json", audit)
    gates = {
        "Q1": series.channels == upstream.CHANNELS and audit["turbines"] == list(farm.TURBINES),
        "Q2": all(audit["windows"][split]["all"] >= 5000 for split in farm.SPLITS),
        "Q3": all(v["missing_fraction"] is not None and v["missing_fraction"] <= .20 for v in audit["upstream_missing_by_year"].values()),
        "Q4": all(r["cells_filled"] == 0 and r["nonmissing_values_preserved"] for r in duplicate_records),
    }
    after = source_code()
    if any(after.get(p) != sha for p, sha in before.items()):
        raise ValueError("Runtime source changed")
    result = {"status": "PASS" if all(gates.values()) else "GATE_FAILED", "gates": gates,
        "start": start, "windows": audit["windows"], "upstream_missing_by_year": audit["upstream_missing_by_year"],
        "elapsed_seconds": time.monotonic() - started, "labels_modified": 0, "labels_clipped": 0, "labels_imputed": 0,
        "source_unchanged": True, "source_code_after": after, "protected_content_parsed": False, "training": False}
    save(out / "result.json", result)
    save(out / "audit/export_sha256.json", {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": frozen.digest(p)}
         for p in sorted(out.rglob("*")) if p.is_file()})
    print("[COMPLETE] " + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import fcntl  # server-only execution; lock scope is this new preparation, not other tasks
    lockpath = args.data_root / "outputs/audits/penmanshiel_v35.lock"
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def expired(signum, frame):
            raise TimeoutError("P1 preparation exceeded its 60 minute budget")
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(3600)
        try:
            prepare(args.data_root.resolve(), args.out.resolve())
        except Exception as exc:
            # Never replace an earlier result when refusing an existing output directory.
            if not isinstance(exc, FileExistsError):
                save(args.out / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            signal.alarm(0)


if __name__ == "__main__":
    main()
