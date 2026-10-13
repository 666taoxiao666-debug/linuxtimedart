"""Acquire public wind data without touching SDWPF or selecting model parameters.

Only requests/numpy/pandas are required. Raw files are immutable after verification;
partial downloads are never reported as complete. Kelmarsh uses at most four normal
HTTP range requests, with provider Retry-After respected. No training is launched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import time
import zipfile

import numpy as np
import pandas as pd
import requests


COMMIT = "b7e96c24a367700082039123ade8f98ea01b78b6"
GEF_TRAIN = [
    "18a41e3fa460790963a444d2406ac82d0e3c0bc0",
    "8463a1a0d05c1b4e087e55e81fc8095265148d5b",
    "9f49490e627af0e0fd42dcdb780ab547fe81f298",
    "c7eb589e73662073ea6671913bc34c877cda5dbc",
    "6ac8e4ac3531640bc3fef89acf8fea90d0d5f5ed",
    "e1a39f74e549c70742cfe7f4b20d490074b6c5dc",
    "044497ac98574141bb68955e15ba5efe0fe30bd3",
    "e1c36db75ad4e9840627c39ce7204afccb38abea",
    "1b960d057d30dce3050bc78a4551f551795f4f1b",
    "a89b0dce1e02358a914fb37c9dc2cead38e3c693",
]
GEF_FUTURE = [
    "ce6bbc1ebffa03ac609724e224cea974c1e16902",
    "8f98e67e18b2820a87694ded103a789506f633a2",
    "d9ef2da0984bede015b5cccca01763dad87a78c7",
    "eb5ab3ff10fcd3157b4cb1e5ff5aee7310ddd8ff",
    "533be12b3432eec18fda9994cc1842ae7b791dcb",
    "11bcaef971563a0faa116b4e7496442283766740",
    "b448d046facbe4706c2f3f47848741af831568d7",
    "16a02dd8ea50110d258f1d61e50f9c40875eb70c",
    "202e95b68fb4e3405d9e4854a4405b2760358a5a",
    "1d038cb1b2bc8d82af966f50b167a61bd71a8456",
]
GEF_SOLUTION = "8c93ba4fe826797f2e9b2d240e737e1f5411e856"
PATHS = {
    "dswe": "provider_hourly_single_turbine/dswe",
    "gefcom": "native_hourly_farm/gefcom2014",
    "kelmarsh": "resampled_hourly_scada/kelmarsh_2016",
}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def request(url, **kwargs):
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=(30, 120), **kwargs)
            if response.status_code in (429, 503) or (
                response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0"
            ):
                wait = response.headers.get("Retry-After", "30")
                response.close()
                if not wait.isdigit() or int(wait) > 120:
                    raise RuntimeError("Provider requests a long pause; resume later, do not evade rate limit")
                time.sleep(int(wait))
                continue
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("Download deferred by provider")


def checked_download(url, target, size, md5, workers=1):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != size or digest(target, "md5") != md5:
            raise ValueError(f"Existing raw file failed integrity; preserved, not overwritten: {target}")
        return
    prefix = target.with_name(target.name + ".part")
    if not prefix.exists():
        prefix.touch()
    prefix_size = prefix.stat().st_size
    if prefix_size > size:
        raise ValueError("Partial file larger than source; preserved for inspection")
    parts_dir = target.parent / ".download_parts"
    parts_dir.mkdir(exist_ok=True)
    remaining = size - prefix_size
    width = math.ceil(remaining / workers) if remaining else 0
    spans = [(start, min(start + width, size) - 1) for start in range(prefix_size, size, max(width, 1))]

    def fetch_span(span):
        start, end = span
        part = parts_dir / f"{target.name}.{start}-{end}.part"
        expected = end - start + 1
        for attempt in range(4):
            have = part.stat().st_size if part.exists() else 0
            if have > expected:
                raise ValueError("Cached range has invalid length")
            if have == expected:
                return part
            first = start + have
            try:
                with request(url, headers={"Range": f"bytes={first}-{end}"}, stream=True) as response:
                    content_range = f"bytes {first}-{end}/{size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != content_range:
                        raise ValueError(f"Server did not honor exact range: {response.status_code}, {response.headers.get('Content-Range')}")
                    with part.open("ab") as out:
                        for chunk in response.iter_content(256 * 1024):
                            out.write(chunk)
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                print(f"[RESUME] interrupted range {start}-{end}; retaining received bytes", flush=True)
                time.sleep(2 ** (attempt + 1))
        if part.stat().st_size != expected:
            raise ValueError("Incomplete range retained for resume")
        print(f"[DOWNLOAD] range {start}-{end} complete", flush=True)
        return part

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(fetch_span, spans))
    assembled = target.with_name(target.name + ".assembled.part")
    with assembled.open("wb") as out, prefix.open("rb") as first:
        shutil.copyfileobj(first, out)
        for part in parts:
            with part.open("rb") as src:
                shutil.copyfileobj(src, out)
    if assembled.stat().st_size != size or digest(assembled, "md5") != md5:
        raise ValueError("Assembled download failed official MD5; all fragments preserved")
    if target.exists():
        raise FileExistsError("Another downloader published this raw file; no overwrite")
    assembled.rename(target)
    # These exact fragments were created by this downloader, never existing datasets.
    prefix.unlink()
    for part in parts:
        part.unlink()
    print(f"[DOWNLOAD] {target.name}: official MD5 verified", flush=True)


def zenodo_file(record, key, directory, workers=1):
    data = record["files"]
    entry = next(item for item in data if item["key"] == key)
    target = directory / "raw" / key
    checked_download(
        entry["links"]["self"], target, entry["size"], entry["checksum"].removeprefix("md5:"), workers
    )
    return target


def zenodo_record(record_id, directory):
    path = directory / "metadata" / f"zenodo_{record_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = request(f"https://zenodo.org/api/records/{record_id}").json()
    save_json(path, data)
    return data


def parse_gef_timestamp(values):
    """Competition dates may use 24:00; retain raw string elsewhere."""
    parsed = []
    for value in values:
        match = re.fullmatch(r"(\d{8})\s+(\d{1,2}):(\d{2})", str(value))
        if not match:
            raise ValueError(f"Unexpected GEFCom timestamp: {value}")
        day, hour, minute = match.groups()
        hour, minute = int(hour), int(minute)
        if hour > 24 or minute > 59 or (hour == 24 and minute != 0):
            raise ValueError(f"Invalid GEFCom timestamp: {value}")
        parsed.append(pd.to_datetime(day, format="%Y%m%d") + pd.Timedelta(hours=hour, minutes=minute))
    return pd.DatetimeIndex(parsed)


def circular_mean(series):
    finite = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(finite) == 0:
        return np.nan
    vec = np.exp(1j * np.deg2rad(finite)).mean()
    if abs(vec) < 1e-12:
        return np.nan  # opposing bearings have no well-defined mean direction
    return float(np.rad2deg(np.angle(vec)) % 360) % 360


def hourly_scada(frame, angles=()):
    """Strict left-closed 6x10min averages, available at the right hour edge.

    Incomplete bins retain counts and a separate partial-power mean, but primary
    power/covariates are NaN. No imputation, clipping, outlier filtering or labels
    learned from any split are used.
    """
    frame = frame.set_index("timestamp").sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Duplicate turbine timestamps: refuse implicit averaging")
    if not ((frame.index.minute % 10 == 0) & (frame.index.second == 0)).all():
        raise ValueError("Off-grid SCADA timestamps require source inspection")
    groups = frame.resample("1h", closed="left", label="right")
    result = groups.mean()
    counts = groups.count()
    for angle in angles:
        result[angle] = groups[angle].apply(circular_mean)
    result["power_partial_mean_kw"] = result["power_kw"]
    for column in frame.columns:
        result[f"n_{column}"] = counts[column]
        result.loc[counts[column] != 6, column] = np.nan
    result["complete_power_hour"] = counts["power_kw"] == 6
    result.index.name = "timestamp_hour_end_utc"
    return result.reset_index()


def frame_quality(frame, timestamp, entity, power):
    duplicate = int(frame.duplicated([entity, timestamp]).sum())
    if duplicate:
        raise ValueError(f"{duplicate} duplicated entity/time pairs")
    items = []
    for key, group in frame.groupby(entity):
        times = pd.to_datetime(group[timestamp]).sort_values()
        expected = int((times.iloc[-1] - times.iloc[0]) / pd.Timedelta(hours=1)) + 1
        items.append({"id": str(key), "rows": len(group), "start": str(times.iloc[0]),
                      "end": str(times.iloc[-1]), "missing_grid_hours": expected - len(group),
                      "missing_power_rows": int(group[power].isna().sum())})
    values = pd.to_numeric(frame[power], errors="coerce")
    return {"rows": len(frame), "entities": len(items), "duplicate_pairs": duplicate,
            "power_min": None if values.dropna().empty else float(values.min()),
            "power_max": None if values.dropna().empty else float(values.max()),
            "negative_power_rows": int((values < 0).sum()), "per_entity": items}


def write_frame(frame, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    temporary = target.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(target)
    return {"path": str(target), "sha256": digest(target), "rows": len(frame)}


def prepare_dswe(directory):
    raw = directory / "raw" / "Wind_Time_Series_Dataset.zip"
    with zipfile.ZipFile(raw) as archive:
        if archive.testzip():
            raise ValueError("DSWE ZIP CRC failure")
        hourly_name = next(name for name in archive.namelist() if "(hourly)" in name)
        ten_name = next(name for name in archive.namelist() if "(10min)" in name)
        hourly = pd.read_csv(archive.open(hourly_name), parse_dates=["Time"])
        ten = pd.read_csv(archive.open(ten_name), parse_dates=["Time"])
    if hourly.Time.duplicated().any() or ten.Time.duplicated().any():
        raise ValueError("DSWE duplicate timestamps")
    grouped = ten.set_index("Time").resample("1h", closed="right", label="right")
    means = grouped[["WindSpeed", "Power"]].mean()
    comparison = hourly.set_index("Time")[["WindSpeed", "Power"]] - means
    max_difference = float(comparison.abs().max().max())
    if not np.isfinite(max_difference) or max_difference > 1e-8:
        raise ValueError("Provider hourly convention does not match right-closed means; inspect before preparing")
    source = hourly.set_index("Time").sort_index()
    grid = pd.date_range(source.index.min(), source.index.max(), freq="1h")
    frame = source.reindex(grid).rename(columns={"WindSpeed": "wind_speed_source", "Power": "power_source"})
    frame["source_row_present"] = frame.index.isin(source.index)
    frame["n_10min_power"] = grouped.Power.count().reindex(grid, fill_value=0)
    frame["complete_hour"] = (frame.n_10min_power == 6) & frame.source_row_present
    frame["power_complete_hour_source"] = frame.power_source.where(frame.complete_hour)
    frame.index.name = "timestamp_provider_hour_end"
    frame = frame.reset_index()
    frame["turbine_id"] = "1"
    prepared = write_frame(frame, directory / "hourly", "dswe_hourly.csv")
    qc = frame_quality(frame, "timestamp_provider_hour_end", "turbine_id", "power_source")
    qc.update({"provider_rows": len(hourly), "ten_minute_rows": len(ten),
               "absent_provider_hours": int((~frame.source_row_present).sum()),
               "complete_hours": int(frame.complete_hour.sum()), "provider_mean_max_abs_difference": max_difference,
               "timezone": "unspecified", "power_units": "unspecified source scale; not asserted kW or p.u.",
               "wind_speed_units": "not explicitly stated in archive metadata",
               "hour_convention": "provider means verified against (t-1h,t] raw observations",
               "capacity_kw": None, "labels_modified": False, "interpolation": False})
    save_json(directory / "metadata" / "quality.json", qc)
    return prepared, qc


def download_gef(directory):
    meta = directory / "metadata"
    tree = request(f"https://api.github.com/repos/greenlytics/gefcom2014-wind/git/trees/{COMMIT}?recursive=1").json()
    if tree.get("truncated"):
        raise ValueError("Incomplete mirror tree")
    save_json(meta / "mirror_tree.json", tree)
    entries = {item["path"]: item for item in tree["tree"] if item["type"] == "blob"}
    selected = []
    for zone in range(1, 11):
        selected.append((f"data/Task 15/Task15_W_Zone1_10/Task15_W_Zone{zone}.csv", GEF_TRAIN[zone - 1], "historical"))
        selected.append((f"data/Task 15/TaskExpVars15_W_Zone1_10/TaskExpVars15_W_Zone{zone}.csv", GEF_FUTURE[zone - 1], "future_nwp"))
    selected.append(("data/Solution to Task 15/solution15_W.csv", GEF_SOLUTION, "evaluation_labels"))

    def fetch(entry):
        name, sha, category = entry
        if entries[name]["sha"] != sha:
            raise ValueError("Pinned mirror tree/blob identity mismatch")
        target = directory / "raw" / category / Path(name).name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = target.read_bytes() if target.exists() else request(
            f"https://api.github.com/repos/greenlytics/gefcom2014-wind/git/blobs/{sha}",
            headers={"Accept": "application/vnd.github.raw+json"},
        ).content
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != sha or len(data) != entries[name]["size"]:
            raise ValueError(f"Git blob verification failed: {name}")
        if not target.exists():
            temporary = target.with_suffix(".part")
            temporary.write_bytes(data)
            temporary.rename(target)
        print(f"[DOWNLOAD] GEFCom {category}/{target.name} verified", flush=True)
        return {"source_path": name, "path": str(target), "git_blob_sha1": sha,
                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(fetch, selected))
    save_json(meta / "download_manifest.json", {"mirror": "https://github.com/greenlytics/gefcom2014-wind",
              "commit": COMMIT, "license": "not established; IEEE dataset catalog states N/A; no redistribution assumed",
              "files": records})


def prepare_gef(directory):
    frames = []
    for raw in sorted((directory / "raw" / "historical").glob("*.csv")):
        frame = pd.read_csv(raw)
        frame["timestamp"] = parse_gef_timestamp(frame.TIMESTAMP)
        frame = frame.rename(columns={"ZONEID": "farm_id", "TIMESTAMP": "source_timestamp",
                    "TARGETVAR": "power_pu", "U10": "u10_nwp_mps", "V10": "v10_nwp_mps",
                    "U100": "u100_nwp_mps", "V100": "v100_nwp_mps"})
        frame["wind_speed100_nwp_mps"] = np.hypot(frame.u100_nwp_mps, frame.v100_nwp_mps)
        frames.append(frame)
    if len(frames) != 10:
        raise ValueError("Incomplete GEFCom historical files")
    frame = pd.concat(frames, ignore_index=True).sort_values(["farm_id", "timestamp"])
    prepared = write_frame(frame, directory / "hourly", "gefcom2014_historical_hourly.csv")
    qc = frame_quality(frame, "timestamp", "farm_id", "power_pu")
    qc.update({"timezone": "unspecified", "power_units": "farm capacity normalized p.u.",
               "nominal_capacity_kw": None, "wind_role": "ECMWF numerical forecasts, not observed SCADA",
               "above_one_power_rows": int((frame.power_pu > 1).sum()), "labels_modified": False,
               "future_nwp_and_task15_solution_merged_into_history": False, "interpolation": False})
    save_json(directory / "metadata" / "quality.json", qc)
    return prepared, qc


KEL_COLUMNS = {
    "Date and time": "timestamp", "Wind speed (m/s)": "wind_speed_mps",
    "Power (kW)": "power_kw", "Wind direction (°)": "wind_direction_deg",
    "Nacelle position (°)": "nacelle_position_deg", "Nacelle ambient temperature (°C)": "ambient_temp_c",
    "Rotor speed (RPM)": "rotor_speed_rpm", "Blade angle (pitch position) A (°)": "pitch_a_deg",
    "Blade angle (pitch position) B (°)": "pitch_b_deg", "Blade angle (pitch position) C (°)": "pitch_c_deg",
    "Yaw bearing angle (°)": "yaw_bearing_deg", "Turbine Power setpoint (kW)": "power_setpoint_kw",
    "Available Capacity for Production (kW)": "available_capacity_kw", "Data Availability": "data_availability_source",
}


def prepare_kelmarsh(directory):
    raw = directory / "raw" / "Kelmarsh_SCADA_2016_3082.zip"
    static = pd.read_csv(directory / "raw" / "Kelmarsh_WT_static.csv", encoding="utf-8-sig")
    static_map = {str(row["Title"]).split()[-1]: row for _, row in static.iterrows()}
    frames, source_records = [], []
    with zipfile.ZipFile(raw) as archive:
        members = sorted(name for name in archive.namelist() if name.startswith("Turbine_Data_"))
        if len(members) != 6:
            raise ValueError("Expected six Kelmarsh turbine files")
        for name in members:
            with archive.open(name) as stream:
                lines = []
                while True:
                    line = stream.readline().decode("utf-8-sig")
                    if not line:
                        raise ValueError("SCADA header not found")
                    lines.append(line)
                    if line.startswith("# Date and time,"):
                        columns = next(csv.reader([line[2:].strip()]))
                        break
                if not all(column in columns for column in KEL_COLUMNS):
                    raise ValueError("Kelmarsh source schema changed")
                frame = pd.read_csv(stream, names=columns, usecols=list(KEL_COLUMNS), na_values=["NaN"])
            frame = frame.rename(columns=KEL_COLUMNS)
            frame.timestamp = pd.to_datetime(frame.timestamp, utc=True)
            for column in frame.columns.difference(["timestamp"]):
                frame[column] = pd.to_numeric(frame[column], errors="raise")
            turbine = re.search(r"Kelmarsh_(\d+)_", name).group(1)
            hourly = hourly_scada(frame, ["wind_direction_deg", "nacelle_position_deg", "yaw_bearing_deg"])
            hourly["turbine_id"] = turbine
            hourly["rated_power_kw"] = float(static_map[turbine]["Rated power (kW)"])
            frames.append(hourly)
            source_records.append({"turbine_id": turbine, "zip_member": name, "raw_rows": len(frame),
                                   "actual_first_timestamp": str(frame.timestamp.min()),
                                   "actual_last_timestamp": str(frame.timestamp.max())})
            print(f"[PREPARE] Kelmarsh turbine {turbine}: {len(frame)} raw -> {len(hourly)} hourly rows", flush=True)
    frame = pd.concat(frames, ignore_index=True).sort_values(["turbine_id", "timestamp_hour_end_utc"])
    prepared = write_frame(frame, directory / "hourly", "kelmarsh_2016_hourly.csv")
    qc = frame_quality(frame, "timestamp_hour_end_utc", "turbine_id", "power_kw")
    qc.update({"source_records": source_records, "timezone": "UTC", "power_units": "kW",
               "source_resolution_minutes": 10, "prepared_resolution_minutes": 60,
               "complete_power_hours": int(frame.complete_power_hour.sum()),
               "incomplete_power_hours": int((~frame.complete_power_hour).sum()),
               "hour_policy": "[t,t+1h), timestamp and availability at t+1h; six valid on-grid observations required per channel",
               "angles": "circular mean; opposing bearings remain NaN", "labels_modified": False,
               "target_transformation": "explicit 6x10-minute hourly averaging; raw labels unchanged",
               "rated_power_kw_by_turbine": {key: float(row["Rated power (kW)"]) for key, row in static_map.items()},
               "commercial_operations_date_by_turbine": {key: str(row["Commercial Operations Date"]) for key, row in static_map.items()},
               "per_channel_complete_hours": {column: int(frame[column].notna().sum()) for column in KEL_COLUMNS.values() if column != "timestamp"},
               "above_rated_power_rows": int((frame.power_kw > frame.rated_power_kw).sum()),
               "interpolation": False, "negative_labels_clipped": False})
    save_json(directory / "metadata" / "quality.json", qc)
    return prepared, qc


def inventory(directory):
    return [{"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in sorted((directory / "raw").rglob("*")) if path.is_file() and not path.name.endswith(".part")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/wind_public"))
    parser.add_argument("--datasets", nargs="+", choices=list(PATHS), default=list(PATHS))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    repo_data = args.root.parent
    originals = [{"path": str(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                  "classification": "original SDWPF 10-minute; unchanged reference, not moved or re-hashed"}
                 for path in (repo_data / "sdwpf_245days_v1.csv", repo_data / "sdwpf_fixed.csv") if path.exists()]
    save_json(args.root / "original_sdwpf_reference.json", originals)
    catalog_path = args.root / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {"datasets": {}}
    for key in args.datasets:
        directory = args.root / PATHS[key]
        print(f"[START] {key}", flush=True)
        if not args.prepare_only:
            if key == "dswe":
                record = zenodo_record(5516539, directory)
                zenodo_file(record, "Wind_Time_Series_Dataset.zip", directory)
            elif key == "gefcom":
                download_gef(directory)
            else:
                record = zenodo_record(16807551, directory)
                for filename in ("Kelmarsh_WT_static.csv", "Kelmarsh_WT_dataSignalMapping.csv"):
                    zenodo_file(record, filename, directory)
                zenodo_file(record, "Kelmarsh_SCADA_2016_3082.zip", directory, workers=4)
        prepare = {"dswe": prepare_dswe, "gefcom": prepare_gef, "kelmarsh": prepare_kelmarsh}[key]
        prepared, qc = prepare(directory)
        catalog["datasets"][key] = {"status": "COMPLETE_VERIFIED", "directory": str(directory),
            "classification": PATHS[key].split("/")[0], "prepared": prepared, "quality": qc,
            "raw_files": inventory(directory),
            "source": {"dswe": "https://zenodo.org/records/5516539",
                       "gefcom": "https://robjhyndman.com/papers/gefcom2014.pdf",
                       "kelmarsh": "https://zenodo.org/records/16807551"}[key],
            "license": "not established; do not assume redistribution" if key == "gefcom" else "CC-BY-4.0"}
        catalog["updated_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(catalog_path, catalog)
        print(f"[COMPLETE] {key}: {qc['rows']} hourly rows, {qc['entities']} entities", flush=True)
    for entry in originals:
        stat = Path(entry["path"]).stat()
        if stat.st_size != entry["bytes"] or stat.st_mtime_ns != entry["mtime_ns"]:
            raise RuntimeError("Original SDWPF changed during acquisition; investigate external writer")
    save_json(args.root / "original_sdwpf_unchanged_check.json", {"unchanged": True, "files": originals})


if __name__ == "__main__":
    main()
