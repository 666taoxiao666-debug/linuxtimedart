"""D2-0 r2: approved repair and interval audit; no training or downloads.

Penmanshiel holdout is registered from ZIP directory names before any content
parsing. Development SCADA cells are decoded as strings solely for null masks;
only timestamps are converted. No power/wind numeric conversion or distributions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MEMBER = re.compile(r"Turbine_Data_Penmanshiel_(\d+)_(\d{4}-\d{2}-\d{2})_-_(\d{4}-\d{2}-\d{2}).*\.csv$")
PREPARATION_SHA = "6a5a3061a7008f6a4d56f2eba68eb733472aaeb3ed1659594a98463b9d6d7677"
REGISTRATION_SHA = "0622d055799e99c0cda17d3c1ec67755706773856007e0182614466757faf180"
PRIOR_INPUTS_SHA = "f9d3e58b7c4cd4c8a9b8f13e46873a2329e7546f7c1040221ec3f35822184ed8"
B1_THRESHOLD = 180.98952
PREPARED_SHA = "96acfcabb8399ddca415a61011b3cf02ca9a675d7fcc31f2ef70db89a3cb37ef"
REQUIRED = {
    "Wspd / wind_last10": ["Wind speed (m/s)"],
    "Etmp": ["Nacelle ambient temperature (°C)"],
    "Ndir_sin / Ndir_cos": ["Nacelle position (°)"],
    "Pab_mean": [f"Blade angle (pitch position) {b} (°)" for b in "ABC"],
    "rotor_speed": ["Rotor speed (RPM)"],
    "power / power_last10": ["Power (kW)"],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
    return {"bytes": path.stat().st_size, "sha256": sha.hexdigest(), "md5": md5.hexdigest()}


def before(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("Part budget exhausted; incomplete checks fail closed")


def full_year(intervals, year):
    cursor, end = f"{year}-01-01", f"{year+1}-01-01"
    for lo, hi in sorted(intervals):
        if lo > cursor:
            break
        cursor = max(cursor, hi)
    return cursor >= end


def registration(entries):
    turbines = sorted({e["turbine"] for e in entries})
    years = sorted({int(e["begin"][:4]) for e in entries})
    complete = [y for y in years if turbines and all(full_year(
        [(e["begin"], e["end"]) for e in entries if e["turbine"] == t], y) for t in turbines)]
    test = max(complete) if complete else None
    return {"utc": now(), "basis": "archive member filenames only; all observed turbine IDs, continuous calendar coverage",
            "turbines": turbines, "complete_calendar_years": complete, "test_year": test,
            "development_years": [y for y in years if test is not None and y < test],
            "protected_years": [y for y in years if test is None or y >= test],
            "rule": "later incomplete years also protected; never choose holdout from data values"}


def scada_header(stream):
    for _ in range(256):
        line = stream.readline()
        if not line:
            break
        text = line.decode("utf-8-sig").strip()
        if text.startswith("# Date and time,") or text.startswith("Date and time,"):
            return [s.strip() for s in next(csv.reader([text.removeprefix("# ")]))]
    raise ValueError("SCADA header absent in first 256 lines")


def mask_coverage(stream, columns, year, test_year, deadline):
    if year >= test_year:
        raise ValueError("Protected-year body access refused")
    lo, hi = pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year+1}-01-01", tz="UTC")
    size = int((hi - lo).total_seconds() // 600)
    seen = np.zeros(size, dtype=bool)
    missing = np.zeros(len(columns), dtype=np.int64)
    rows = bad_time = duplicates = 0
    # All cells remain strings: no numerical SCADA field is converted or stored.
    reader = pd.read_csv(stream, names=columns, dtype=str, chunksize=4096, keep_default_na=True)
    for chunk in reader:
        before(deadline)
        missing += chunk.isna().sum().to_numpy(dtype=np.int64)
        rows += len(chunk)
        ts = pd.to_datetime(chunk["Date and time"], utc=True, errors="coerce")
        valid = ts.notna() & (ts >= lo) & (ts < hi) & (ts == ts.dt.floor("10min"))
        bad_time += int((~valid).sum())
        slots = ((ts[valid] - lo).dt.total_seconds() // 600).astype(int).to_numpy()
        unique = np.unique(slots)
        duplicates += int(len(slots) - len(unique) + seen[unique].sum())
        seen[unique] = True
    return {"rows": rows, "timestamp_coverage": float(seen.mean()), "expected_ticks": size,
            "invalid_or_outside_year_timestamps": bad_time, "duplicate_ticks": duplicates,
            "missing_fraction": {c: float(n / rows) if rows else None for c, n in zip(columns, missing)},
            "value_access": "strings decoded for null masks only; no numerical SCADA values retained/converted"}


def geometry(lat0, lon0, lat, lon):
    a, b, c, d = np.radians([lat0, lon0, lat, lon])
    distance = 2 * 6371 * np.arcsin(np.sqrt(np.sin((c-a)/2)**2 + np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2))
    bearing = np.degrees(np.arctan2(np.sin(d-b)*np.cos(c), np.cos(a)*np.sin(c)-np.sin(a)*np.cos(c)*np.cos(d-b))) % 360
    return float(distance), float(bearing)


def static_metadata(path, turbines):
    frame = pd.read_csv(path, encoding="utf-8-sig")  # static equipment/location metadata only
    total_rows = len(frame)
    empty = frame.isna().all(axis=1)
    empty_rows = int(empty.sum())
    frame = frame.loc[~empty].copy()  # Only wholly empty rows may be removed.
    names = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in frame.columns}
    def key(*options):
        return next((names[o] for o in options if o in names), None)
    title, rating = key("title", "turbineid"), key("ratedpowerkw")
    model = key("turbinemodel", "model", "turbine", "turbinetype")
    lat, lon = key("latitude", "lat", "latitudewgs84"), key("longitude", "lon", "long", "longitudewgs84")
    result = {"columns": list(frame.columns), "rows": len(frame), "original_rows": total_rows,
              "entirely_empty_rows_removed": empty_rows, "consistent": False, "center": None}
    if title is None or not frame[title].map(lambda v: bool(re.fullmatch(r"Penmanshiel\s+\d{2}", str(v)))).all():
        raise ValueError("Nonempty metadata row without a valid Penmanshiel device ID")
    if title and rating and model:
        ids = [int(re.search(r"(\d+)$", str(v)).group(1)) for v in frame[title]]
        result.update(turbines=ids, models=frame[model].astype(str).tolist(), rated_kw=frame[rating].astype(float).tolist())
        result["consistent"] = (sorted(ids) == turbines and all("MM82" in str(v).replace(" ", "").upper() for v in frame[model])
                                and (frame[rating].astype(float) == 2050).all())
        result["consistent"] = bool(result["consistent"])
    if lat and lon:
        coords = frame[[lat, lon]].astype(float).to_numpy()
        if np.isfinite(coords).all() and (np.abs(coords[:, 0]) <= 90).all() and (np.abs(coords[:, 1]) <= 180).all():
            result["center"] = coords.mean(0).tolist()
    return result


def stations_audit(root, center, reg, deadline, manifest):
    from scripts.screen_kelmarsh_upstream_observations import read_badc_csv, column
    from utils.kelmarsh_upstream_v31 import load_station_observations
    result = {"passed": False, "qualified": [], "stations": [], "protected_files": [], "radius_km": 150,
              "pressure_rule": "at least one valid sea-level pressure hour in every development year",
              "wind_rule": "valid hourly wind >=85% in every development year, existing v31 QC/units"}
    if center is None or not reg["development_years"]:
        return {**result, "reason": "Missing verified farm coordinates or development years"}
    base = root / "datasets/upstream_observations"
    catalogs = sorted(base.rglob("*station-metadata.csv"))
    if not catalogs:
        return {**result, "reason": "MIDAS station catalog absent"}
    # Only the frozen 202607 catalog is eligible; duplicates must be identical.
    catalogs = [p for p in catalogs if "dv-202607" in p.name]
    catalog_hashes = [hashes(p) for p in catalogs]
    if not catalogs or len({h["sha256"] for h in catalog_hashes}) != 1:
        return {**result, "reason": "Missing/conflicting 202607 station catalogs"}
    path = catalogs[0]
    manifest[str(path)] = catalog_hashes[0]
    frame = read_badc_csv(path)
    sid = column(frame, "src_id")
    lat = column(frame, "station_latitude", "latitude", "lat")
    lon = column(frame, "station_longitude", "longitude", "lon")
    if (frame.groupby(sid)[[lat, lon]].nunique() > 1).any().any():
        return {**result, "reason": "Conflicting coordinates for station ID"}
    files = sorted(base.rglob("*_qcv-1_*.csv"))
    for _, row in frame.drop_duplicates(sid).iterrows():
        before(deadline)
        if pd.isna(row[lat]) or pd.isna(row[lon]):
            continue
        distance, bearing = geometry(*center, float(row[lat]), float(row[lon]))
        if distance > 150:
            continue
        unit = int(row[sid])
        selected = [p for p in files if re.search(rf"(?:^|[_/])0*{unit}(?:_|/)", p.as_posix())]
        for p in selected:
            match = re.search(r"_(\d{4})\.csv$", p.name)
            if match and int(match[1]) >= reg["test_year"]:
                before(deadline)
                manifest[str(p)] = hashes(p)
                with p.open(encoding="utf-8-sig") as stream:
                    header = None
                    for _ in range(512):
                        line = stream.readline()
                        if not line:
                            break
                        if line.strip().strip(",").lower() == "data":
                            header = next(csv.reader([stream.readline()]))
                            break
                result["protected_files"].append({"path": str(p), "header": header, "body_parsed": False})
        record = {"src_id": unit, "distance_km": distance, "bearing_deg": bearing, "years": {}}
        for year in reg["development_years"]:
            paths = [p for p in selected if p.stem.endswith(f"_{year}")]
            if not paths:
                record["years"][str(year)] = {"wind_fraction": 0., "pressure_fraction": 0., "files": []}
                continue
            for p in paths:
                manifest[str(p)] = hashes(p)
            obs = load_station_observations([{"path": p, "year": year} for p in paths], [unit])[unit]
            grid = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="h", inclusive="left")
            obs = obs.reindex(grid)
            record["years"][str(year)] = {"wind_fraction": float(obs.wind_speed.notna().mean()),
                                         "pressure_fraction": float(obs.msl_pressure.notna().mean()), "files": [str(p) for p in paths]}
        record["qualified"] = all(v["wind_fraction"] >= .85 for v in record["years"].values())
        record["pressure_capable"] = all(v["pressure_fraction"] > 0 for v in record["years"].values())
        result["stations"].append(record)
    qualified = [s for s in result["stations"] if s["qualified"]]
    result.update(qualified=[s["src_id"] for s in qualified], count=len(qualified),
                  pressure_count=sum(s["pressure_capable"] for s in qualified),
                  quadrants=sorted({int(s["bearing_deg"] // 90) for s in qualified}))
    result["passed"] = result["count"] >= 8 and result["pressure_count"] >= 5 and len(result["quadrants"]) >= 3
    return result


def part_a(root, out, manifest):
    deadline = time.monotonic() + 1800
    base = root / "datasets/external_validation/sealed/penmanshiel_16807304"
    archives = sorted(base.glob("*SCADA*.zip"))
    entries = []
    for path in archives:
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                m = MEMBER.search(Path(info.filename).name)
                if m:
                    entries.append({"archive": str(path), "member": info.filename, "bytes": info.file_size,
                                    "turbine": int(m[1]), "begin": m[2], "end": m[3]})
    previous = ROOT / "output/kelmarsh_hourly/d2_feasibility_v34"
    registered_path = previous / "test_year_registration.json"
    previous_inputs = previous / "input_hashes.json"
    registered = validated_record(registered_path, REGISTRATION_SHA)
    prior_hashes = validated_record(previous_inputs, PRIOR_INPUTS_SHA)
    if registered["test_year"] != 2022 or registered["protected_years"] != [2022, 2023, 2024]:
        raise ValueError("Original holdout identity changed")
    if registered["members"] != entries:
        raise ValueError("Archive directory metadata differs from original registration")
    reg = {k: v for k, v in registered.items() if k != "members"}
    # Copy the exact original registration; never choose a holdout again.
    (out / "test_year_registration.json").write_bytes(registered_path.read_bytes())
    manifest[str(registered_path)] = hashes(registered_path)
    manifest[str(previous_inputs)] = hashes(previous_inputs)
    result = {"A1": False, "A2": False, "A3": False, "registration": reg, "scada": [], "mapping": REQUIRED,
              "limitations": ["Missingness parses strings but never converts SCADA power/wind to numbers.",
                              "Test and later SCADA: header only; no timestamps or null masks.",
                              "Raw 10-minute power/wind support last10 derivation; not derived in this audit."]}
    if reg["test_year"] is None:
        return {**result, "reason": "No all-turbine full calendar year in local archive filenames"}
    inventory_path = root / "outputs/audits/external_validation_download/inventory.json"
    manifest[str(inventory_path)] = hashes(inventory_path)
    inventory = json.loads(inventory_path.read_text())
    official = {f["name"]: f for ds in inventory["datasets"] if "penmanshiel" in ds["name"].lower() for f in ds["files"]}
    # Hash opaque archive bytes; comparing official MD5 does not parse sealed content.
    for path in archives + sorted(base.glob("*WT_static.csv")):
        before(deadline)
        h = hashes(path)
        manifest[str(path)] = h
        if prior_hashes.get(str(path)) != h:
            raise ValueError("Input bytes differ from original audit identity: " + path.name)
        expected = official.get(path.name)
        if not expected or h["bytes"] != expected["bytes"] or h["md5"] != expected["md5"]:
            raise ValueError("Penmanshiel file identity mismatch: " + path.name)
    static_path = base / "Penmanshiel_WT_static.csv"
    static = static_metadata(static_path, reg["turbines"]) if static_path.exists() else {"consistent": False, "center": None}
    result["static"] = static
    save(out / "a_static_metadata.json", static)
    all_headers = True
    for entry in entries:
        before(deadline)
        year = int(entry["begin"][:4])
        with zipfile.ZipFile(entry["archive"]) as z, z.open(entry["member"]) as stream:
            header = scada_header(stream)
            rec = {**entry, "header": header, "protected": year >= reg["test_year"] or entry["end"] > f"{reg['test_year']}-01-01"}
            rec["missing_required"] = [c for cols in REQUIRED.values() for c in cols if c not in header]
            all_headers &= not rec["missing_required"]
            if not rec["protected"]:
                rec.update(mask_coverage(stream, header, year, reg["test_year"], deadline))
            result["scada"].append(rec)
        print("[A SCADA]", Path(entry["archive"]).name, entry["turbine"], "header-only" if rec["protected"] else "timestamp/null masks", flush=True)
        # A crash leaves an explicit partial report without exposing any field values.
    years_ok = []
    for year in reg["development_years"]:
        records = [r for r in result["scada"] if int(r["begin"][:4]) == year]
        # Never sum overlapping chunks: each turbine must have one complete-year member.
        if all(any(r["turbine"] == t and r.get("timestamp_coverage", 0) >= .9 and
                   r["duplicate_ticks"] == 0 for r in records)
               for t in reg["turbines"]):
            years_ok.append(year)
    result.update(A1=bool(years_ok), A2=bool(all_headers and static["consistent"]), qualifying_development_years=years_ok)
    save(out / "a_scada_coverage.json", {k: v for k, v in result.items() if k != "A3"})
    result["midas"] = stations_audit(root, static["center"], reg, deadline, manifest)
    result["A3"] = result["midas"]["passed"]
    return result


def metrics(pred, truth):
    pred, truth = np.asarray(pred), np.asarray(truth)
    if pred.shape != truth.shape or not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("No dropping/imputing non-finite predictions/labels or changing the fixed queue")
    err = pred - truth
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt(np.square(err).mean()))}


def validated_record(path, expected_sha):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha:
        raise ValueError("Prior registration/input manifest SHA256 mismatch")
    return json.loads(content)


def interval_metrics(lower_abs, upper_abs):
    lower_abs, upper_abs = np.asarray(lower_abs), np.asarray(upper_abs)
    if lower_abs.shape != upper_abs.shape or not np.isfinite(lower_abs).all() or not np.isfinite(upper_abs).all():
        raise ValueError("Invalid interval arrays")
    if (lower_abs < 0).any() or (lower_abs > upper_abs).any():
        raise ValueError("Invalid absolute-error interval")
    return {"mae_lower": float(lower_abs.mean()), "mae_upper": float(upper_abs.mean()),
            "rmse_lower": float(np.sqrt(np.square(lower_abs).mean())),
            "rmse_upper": float(np.sqrt(np.square(upper_abs).mean()))}


def gate(bounds):
    if bounds["rmse_upper"] <= B1_THRESHOLD:
        return "PASS"
    if bounds["rmse_lower"] > B1_THRESHOLD:
        return "FAIL"
    return "UNASSESSABLE"


def curve_diagnostics(raw, starts, turbines, curves, truth, history=24, horizon=4):
    from utils.kelmarsh_enrichment import expected_power
    future = starts[:, None] + history + np.arange(horizon)
    issue = starts + history - 1
    if truth.shape != future.shape or not np.isfinite(truth).all() or not np.isfinite(raw[issue, 0]).all():
        raise ValueError("Fixed labels/issue wind invalid; no row deletion or imputation")
    missing = ~np.isfinite(raw[future, 0])
    lower, upper = np.empty(truth.shape), np.empty(truth.shape)
    persistence = np.empty(truth.shape)
    ranges = {}
    for turbine in np.unique(turbines[starts]):
        mask = turbines[starts] == turbine
        curve = curves[str(turbine)]
        centers, values = np.asarray(curve["centers_mps"]), np.asarray(curve["curve_kw"])
        if not np.isfinite(centers).all() or not np.isfinite(values).all() or not (np.diff(centers) > 0).all():
            raise ValueError("Frozen curve has invalid interpolation nodes")
        # np.interp uses left=0 and right=last node: include BOTH in the range.
        cmin, cmax = min(0., float(values.min())), max(0., float(values.max()))
        ranges[str(turbine)] = [cmin, cmax]
        y, absent = truth[mask], missing[mask]
        lo = np.maximum(np.maximum(cmin - y, y - cmax), 0.)
        hi = np.maximum(np.abs(y - cmin), np.abs(y - cmax))
        known_wind = raw[future[mask], 0][~absent]
        err = np.abs(expected_power(curve, known_wind) - y[~absent])
        lo[~absent], hi[~absent] = err, err
        lower[mask], upper[mask] = lo, hi
        persistence[mask] = expected_power(curve, raw[issue[mask], 0])[:, None]
    oracle = {"pooled": interval_metrics(lower, upper),
              "by_horizon": [interval_metrics(lower[:, h], upper[:, h]) for h in range(horizon)],
              "missing_cells": int(missing.sum()), "affected_windows": int(missing.any(1).sum()),
              "missing_by_horizon": missing.sum(0).tolist(), "curve_output_ranges": ranges}
    physical = {"pooled": metrics(persistence, truth),
                "by_horizon": [metrics(persistence[:, h], truth[:, h]) for h in range(horizon)]}
    return oracle, physical


def part_b(root, predictions, manifest):
    from data_provider.kelmarsh_multiyear import KelmarshMultiYearSeries
    deadline = time.monotonic() + 600
    cfg_path = ROOT / "configs/kelmarsh_arrival_v33_k1.json"
    cfg = json.loads(cfg_path.read_text())
    protocol_path = ROOT / cfg["data_source_protocol"]
    d = json.loads(protocol_path.read_text())["data"]
    csv_path = root / d["v32_preparation"]["csv"]
    prep_path = csv_path.parent.parent / "metadata/preparation.json"
    for path in (cfg_path, protocol_path, csv_path, prep_path, predictions):
        before(deadline)
        manifest[str(path)] = hashes(path)
    if manifest[str(prep_path)]["sha256"] != PREPARATION_SHA or manifest[str(csv_path)]["sha256"] != PREPARED_SHA:
        raise ValueError("Frozen v32 input identity mismatch")
    prep = json.loads(prep_path.read_text())
    if prep["curve_years"] != [2016, 2017, 2018, 2019]:
        raise ValueError("Unexpected fitted curve years")
    fold = cfg["fold"]
    series = KelmarshMultiYearSeries(csv_path, splits={"train": fold["train"], "validation": fold["validation"], "test": fold["development_eval"]},
                                    input_len=d["input_len"], horizon=d["pred_len"], stride=d["stride"],
                                    extras=tuple(d["extras_in_common_series"]), start=d["start_utc"], end="2023-01-01T01:00:00Z")
    windows = series.windows("test", "all")
    starts = windows.window_starts
    truth, power_persistence = windows.truth_and_anchor()
    with np.load(predictions, allow_pickle=False) as original:
        if len(starts) != 16949 or not np.array_equal(starts, original["eval_starts"]):
            raise ValueError("Queue differs from frozen K1b eval_starts")
        t21 = metrics(original["development_eval"].reshape(truth.shape), truth)
    pers = metrics(power_persistence, truth)
    for actual, expected in ((t21["mae"], 161.0914799595), (t21["rmse"], 241.3193604677),
                             (pers["mae"], 193.8727237898), (pers["rmse"], 301.8658381265)):
        if abs(actual - expected) > .0001:
            raise ValueError("Reference metric mismatch; cannot claim fixed queue/labels")
    oracle, physical = curve_diagnostics(series.raw, starts, series.turbines, prep["power_curves"], truth)
    if oracle["missing_cells"] != 5 or oracle["missing_by_horizon"] != [0, 2, 2, 1]:
        raise ValueError("Missing-wind pattern differs from approved r2 evidence")
    before(deadline)
    decision = gate(oracle["pooled"])
    return {"B1": decision == "PASS", "B1_status": decision, "threshold_rmse": B1_THRESHOLD,
            "windows": len(starts), "cohort": "2022 development ALL, h1-4 pooled; identical K1b eval_starts",
            "curve_fit_years": prep["curve_years"], "T21": t21, "power_persistence": pers, "O_curve": oracle, "P_curve": physical,
            "eval_starts_sha256": hashlib.sha256(starts.astype("<i8").tobytes()).hexdigest(),
            "truth_sha256": hashlib.sha256(truth.astype("<f8").tobytes()).hexdigest(),
            "O_curve_warning": "noncausal diagnostic using actual future nacelle wind; not a deployable forecast or universal lower bound",
            "P_curve_issue": "last observed hourly nacelle wind, repeated across four future leads"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k1b-predictions", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Fresh output directory required; no audit reruns/overwrites")
    args.out.mkdir(parents=True)
    import fcntl
    # Existing data are never written; this lock is in the result directory's parent.
    with (args.out.parent / ".d2_audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        save(args.out / "audit/launch.json", {"utc": now(), "argv": sys.argv, "source_code": {
            p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in (
                "scripts/audit_d2_feasibility_v34_r2.py", "data_provider/kelmarsh_multiyear.py", "utils/kelmarsh_enrichment.py",
                "utils/kelmarsh_upstream_v31.py", "scripts/screen_kelmarsh_upstream_observations.py")}})
        manifest, result = {}, {"started_utc": now(), "no_training": True, "no_download": True}
        for name, func in (("A", lambda: part_a(args.data_root, args.out, manifest)),
                           ("B", lambda: part_b(args.data_root, args.k1b_predictions, manifest))):
            begin = time.monotonic()
            try:
                result[name] = func()
            except Exception as exc:
                # Exception text may contain raw data; only safe class is recorded.
                frames = traceback.extract_tb(exc.__traceback__)
                result[name] = {"status": "UNASSESSABLE", "error_type": type(exc).__name__,
                                "error_locations": [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name} for f in frames]}
                print(f"[{name}] UNASSESSABLE {type(exc).__name__}", flush=True)
            result[name]["elapsed_seconds"] = time.monotonic() - begin
            save(args.out / f"part_{name}.json", result[name])
        result.update(finished_utc=now(), elapsed_seconds=time.monotonic()-started)
        result["continue_D2"] = all(result["A"].get(k) is True for k in ("A1", "A2", "A3")) and result["B"].get("B1") is True
        save(args.out / "input_hashes.json", manifest)
        save(args.out / "result.json", result)
        print(json.dumps({"continue_D2": result["continue_D2"], "seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
