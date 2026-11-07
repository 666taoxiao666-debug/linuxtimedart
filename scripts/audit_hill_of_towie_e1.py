"""E1 B: register sealed years, inventory bytes, acquire pinned nearby MIDAS.

No SCADA archive extraction, observation parsing, fitting or test evaluation.
Only station/turbine metadata CSVs are parsed. Development screening is separate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RECORD_SHA = "7064b838681db88aa70eb7c9783e3383b074204a41e621bd62681c8abe50b5d0"
BASE = "/badc/ukmo-midas-open/data/uk-hourly-weather-obs/dataset-version-202607"
META = "midas-open_uk-hourly-weather-obs_dv-202607_station-metadata.csv"
YEARS = tuple(range(2016, 2027))
SEALED = (2024, 2025, 2026)


def sha(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def frozen(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Existing registration differs; preserved")
    else:
        save(path, value)


def safe_name(value):
    if not value or value in (".", "..") or any(c in value for c in "/\\?#\x00"):
        raise ValueError("Unsafe provider name")
    return value


def registration(record):
    if int(record["id"]) != 22662930 or record["version"] != "2.1.0":
        raise ValueError("Pinned Zenodo record/version differs")
    files = record["files"]
    names = [safe_name(x["key"]) for x in files]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate record files")
    years = sorted(int(x[:4]) for x in names if re.fullmatch(r"20\d{2}\.zip", x))
    if years != list(YEARS):
        raise ValueError("Year inventory differs from approved record")
    return {"record_id": 22662930, "record_sha256": RECORD_SHA,
            "metadata_source": "https://zenodo.org/records/22662930",
            "development_years": list(range(2016, 2024)), "test_years": [2024, 2025],
            "protected_partial_years": [2026], "partial_year_end": "2026-04",
            "basis": "official record describes Jan 2016 through Apr 2026; filenames only",
            "policy": "test/protected observations: bytes and hashes only; no parsing",
            "files": [{"name": x["key"], "bytes": x["size"], "md5": x["checksum"]} for x in files]}


def distance_bearing(lat, lon, center):
    a, b, c, d = map(math.radians, (center[0], center[1], lat, lon))
    delta = d - b
    h = math.sin((c-a)/2)**2 + math.cos(a)*math.cos(c)*math.sin(delta/2)**2
    distance = 2 * 6371 * math.asin(math.sqrt(min(1., max(0., h))))
    bearing = math.degrees(math.atan2(math.sin(delta)*math.cos(c),
                                     math.cos(a)*math.sin(c)-math.sin(a)*math.cos(c)*math.cos(delta))) % 360
    return distance, bearing


def nearby(frame, center):
    stations = {}
    for _, row in frame.iterrows():
        sid = int(row["src_id"])
        lat, lon = float(row["station_latitude"]), float(row["station_longitude"])
        if not math.isfinite(lat) or not math.isfinite(lon):
            continue
        if abs(lat) > 90 or abs(lon) > 180:
            raise ValueError("Invalid station coordinates")
        dist, bearing = distance_bearing(lat, lon, center)
        if dist > 150 or float(row["last_year"]) < 2016 or float(row["first_year"]) > 2026:
            continue
        item = {"src_id": sid, "latitude": lat, "longitude": lon, "distance_km": dist,
                "bearing_deg": bearing, "county": safe_name(str(row["historic_county"])),
                "station_file_name": safe_name(str(row["station_file_name"]))}
        if sid in stations and stations[sid] != item:
            raise ValueError("Conflicting station identity")
        stations[sid] = item
    return sorted(stations.values(), key=lambda x: x["src_id"])


def validate_plan(plan):
    if plan["base"] != BASE or plan["sealed_years"] != list(SEALED) or plan["radius_km"] != 150:
        raise ValueError("Plan scope differs")
    seen = set()
    stations = {s["src_id"]: s for s in plan["stations"]}
    for item in plan["files"]:
        safe_name(item["name"])
        station = stations[item["src_id"]]
        parent = f'{BASE}/{station["county"]}/{item["src_id"]:05d}_{station["station_file_name"]}/qc-version-1/'
        match = re.search(r"_(\d{4})\.csv$", item["name"])
        if (not match or int(match[1]) != item["year"] or item["year"] not in YEARS
                or item["sealed"] != (item["year"] in SEALED)
                or item["path"] != parent + item["name"] or item["path"] in seen
                or not isinstance(item["size"], int) or item["size"] < 0
                or not re.fullmatch("[0-9a-fA-F]{32}", item["md5"])):
            raise ValueError("Invalid planned identity or sealed routing")
        seen.add(item["path"])


def public_metadata(item, target):
    """Download one public non-observational metadata file with pinned MD5."""
    import requests
    if not target.exists():
        with requests.get(item["links"]["self"], timeout=(30, 180), stream=True) as response:
            response.raise_for_status()
            part = target.with_suffix(".part")
            with part.open("wb") as stream:
                for chunk in response.iter_content(65536):
                    stream.write(chunk)
        if part.stat().st_size != item["size"] or sha(part, "md5") != item["checksum"].removeprefix("md5:"):
            raise ValueError("Public metadata checksum mismatch; preserved")
        part.rename(target)
    if target.stat().st_size != item["size"] or sha(target, "md5") != item["checksum"].removeprefix("md5:"):
        raise ValueError("Existing metadata identity differs")


def acquire(args, audit, record):
    import pandas as pd
    from scripts.acquire_midas_upstream_observations import Archive, checked_roots
    from scripts.screen_kelmarsh_upstream_observations import read_badc_csv
    data = args.data_root.resolve()
    development, sealed = checked_roots(data / "datasets/upstream_observations/hill_of_towie_midas_v202607/development",
                                       data / "datasets/upstream_observations/sealed/hill_of_towie_midas_v202607")
    turbine_meta = audit / "Hill_of_Towie_turbine_metadata.csv"
    item = next(x for x in record["files"] if x["key"] == turbine_meta.name)
    public_metadata(item, turbine_meta)
    turbines = pd.read_csv(turbine_meta, encoding="utf-8-sig")
    if len(turbines) != 21 or not (turbines["Rated power (kW)"] == 2300).all():
        raise ValueError("Unexpected turbine metadata")
    center = [float(turbines.Latitude.mean()), float(turbines.Longitude.mean())]
    archive = Archive(Path.home() / ".ceda_token")
    entries = [x for x in archive.listing(BASE) if x["name"] == META]
    if len(entries) != 1:
        raise ValueError("Missing or ambiguous CEDA metadata")
    metadata = audit / META
    archive.fetch(entries[0], metadata)
    stations = nearby(read_badc_csv(metadata), center)
    identity = {"base": BASE, "radius_km": 150, "center": center,
                "center_rule": "arithmetic mean of 21 official turbine coordinates",
                "sealed_years": list(SEALED), "requested_years": list(YEARS),
                "metadata_sha256": sha(metadata), "turbine_metadata_sha256": sha(turbine_meta),
                "stations": stations}
    frozen(audit / "station_registration.json", identity)
    plan_path = audit / "download_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if any(plan[k] != v for k, v in identity.items()):
            raise ValueError("Existing plan identity changed")
    else:
        files, missing = [], []
        for station in stations:
            sid = station["src_id"]
            directory = f'{BASE}/{station["county"]}/{sid:05d}_{station["station_file_name"]}/qc-version-1'
            found = []
            for entry in archive.listing(directory):
                match = re.search(r"_(\d{4})\.csv$", entry["name"])
                if entry.get("type") != "file" or not match or int(match[1]) not in YEARS:
                    continue
                year = int(match[1])
                files.append({**entry, "src_id": sid, "year": year, "sealed": year in SEALED})
                found.append(year)
            missing.append({"src_id": sid, "missing_years": sorted(set(YEARS)-set(found))})
            print(f"[PLAN] station={sid} files={len(found)}", flush=True)
        plan = {**identity, "files": sorted(files, key=lambda x: (x["src_id"], x["year"])),
                "missing_by_station": missing}
        validate_plan(plan)
        if not files:
            raise ValueError("No available files; not a completed acquisition")
        frozen(plan_path, plan)
    validate_plan(plan)
    manifest_path = audit / "download_manifest.json"
    manifest = {"status": "DOWNLOADING", "plan_sha256": sha(plan_path),
                "planned_files": len(plan["files"]), "planned_bytes": sum(x["size"] for x in plan["files"]),
                "files": [], "policy": "all observations raw bytes only; development screening pending"}
    save(manifest_path, manifest)
    def receive(item):
        target = (sealed if item["sealed"] else development) / str(item["src_id"]) / item["name"]
        archive.fetch(item, target)
        return {"path": str(target), "src_id": item["src_id"], "year": item["year"],
                "sealed": item["sealed"], "bytes": item["size"], "md5": item["md5"], "sha256": sha(target)}
    try:
        pending = iter(plan["files"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(receive, x) for x in (next(pending, None), next(pending, None)) if x}
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                results = [f.result() for f in done]
                manifest["files"].extend(results)
                manifest["files"].sort(key=lambda x: (x["src_id"], x["year"]))
                save(manifest_path, manifest)
                print(f'[VERIFIED] {len(manifest["files"])}/{manifest["planned_files"]}', flush=True)
                for _ in done:
                    item = next(pending, None)
                    if item:
                        futures.add(pool.submit(receive, item))
        manifest["status"] = "AVAILABLE_FILES_VERIFIED"
        save(manifest_path, manifest)
    except Exception as exc:
        manifest.update(status="INTERRUPTED", error_type=type(exc).__name__)
        save(manifest_path, manifest)
        raise



def development_header(archive, year):
    """Read one header per development CSV; never open protected archives."""
    import zipfile
    import csv
    if year not in range(2016, 2024) or Path(archive).name != f"{year}.zip":
        raise ValueError("Protected or mismatched year; header read forbidden")
    headers = {}
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not name.endswith(".csv") or not name.startswith(("tblSCTurbine_", "tblSCTurGrid_", "tblSCTurTemp_")):
                continue
            with z.open(name) as stream:
                line = stream.readline(65537)
            if len(line) > 65536:
                raise ValueError("Oversized header")
            headers[name] = next(csv.reader([line.decode("cp1252").lstrip("\ufeff").strip()]))
    if not headers:
        raise ValueError("No approved SCADA headers")
    return headers


def coverage_gate(stations):
    qualified = [s for s in stations if s["qualified"]]
    pressure = [s for s in qualified if s["has_pressure"]]
    quadrants = sorted({int(s["bearing_deg"] // 90) for s in qualified})
    return {"qualified_stations": len(qualified), "with_pressure": len(pressure),
            "quadrants": quadrants, "passed": len(qualified) >= 8 and len(pressure) >= 5 and len(quadrants) >= 3}


def readonly_inventory(args):
    """No network, fitting, extraction or writes to acquisition/data paths."""
    import csv
    import io
    import types
    import pandas as pd
    from scripts import screen_kelmarsh_upstream_observations as screen
    audit, out, root = args.audit_root, args.readonly_out, args.data_root.resolve()
    if out.resolve() == audit.resolve() or out.exists():
        raise ValueError("Use a fresh separate output directory")
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if b'--download-midas' in argv and any(a.endswith(b'audit_hill_of_towie_e1.py') for a in argv[1:3]):
            raise ValueError("Acquisition still active")
    plan = json.loads((audit / 'download_plan.json').read_text())
    manifest = json.loads((audit / 'download_manifest.json').read_text())
    reg = json.loads((audit / 'station_registration.json').read_text())
    validate_plan(plan)
    if any(plan[k] != v for k, v in reg.items()):
        raise ValueError("Station registration mismatch")
    if manifest['status'] != 'AVAILABLE_FILES_VERIFIED' or not (len(plan['files']) == len(manifest['files']) == manifest['planned_files'] == 512):
        raise ValueError("Acquisition gate failed")
    if manifest['plan_sha256'] != sha(audit / 'download_plan.json'):
        raise ValueError("Plan hash mismatch")
    expected = {(f['src_id'], f['year']): f for f in plan['files']}
    seen = set()
    development = root / 'datasets/upstream_observations/hill_of_towie_midas_v202607/development'
    sealed = root / 'datasets/upstream_observations/sealed/hill_of_towie_midas_v202607'
    for f in manifest['files']:
        key = (f['src_id'], f['year']); item = expected[key]
        target = (sealed if item['sealed'] else development) / str(item['src_id']) / item['name']
        if key in seen or Path(f['path']).resolve() != target.resolve() or f['sealed'] != item['sealed']:
            raise ValueError("Duplicate/path/sealed identity mismatch")
        seen.add(key)
        if not (target.stat().st_size == f['bytes'] == item['size'] and sha(target, 'md5') == f['md5'] == item['md5'] and sha(target) == f['sha256']):
            raise ValueError("MIDAS file hash mismatch")
    if sum(f['bytes'] for f in manifest['files']) != manifest['planned_bytes']:
        raise ValueError("Byte total mismatch")
    print('[VERIFIED] MIDAS 512 files', flush=True)
    # Reuse the reviewed parsing rules, scoped to development 2016-2023 only.
    loader = types.FunctionType(screen.load_observations.__code__, {**screen.load_observations.__globals__, 'YEARS': tuple(range(2016, 2024))})
    loader.__defaults__ = screen.load_observations.__defaults__
    approved = {str(Path(f['path']).resolve()) for f in manifest['files'] if not f['sealed']}
    if any(str(p.resolve()) not in approved for p in development.rglob('*.csv')):
        raise ValueError('Unregistered file in development tree')
    observations, parsed, skipped, unresolved = {}, [], [], {}
    for station in reg['stations']:
        sid = station['src_id']
        try:
            obs, reads, skips = loader(development / str(sid), [sid])
        except ValueError as exc:
            if 'conflicting duplicate observations' not in str(exc):
                raise
            unresolved[str(sid)] = str(exc)
            continue
        observations.update(obs); parsed.extend(reads); skipped.extend(skips)
    approved = {f['path'] for f in manifest['files'] if not f['sealed']}
    if any(f['path'] not in approved for f in parsed):
        raise ValueError("Unregistered development input")
    rows = []
    for station in reg['stations']:
        frame = observations.get(station['src_id'])
        wind, pressure = {}, {}
        for year in range(2016, 2024):
            hours = pd.date_range(f'{year}-01-01', f'{year+1}-01-01', freq='h', inclusive='left')
            wind[str(year)] = 0.0 if frame is None else float(frame.wind_speed.reindex(hours).notna().mean())
            pressure[str(year)] = 0.0 if frame is None else float(frame.msl_pressure.reindex(hours).notna().mean())
        rows.append({**station, 'unresolved': str(station['src_id']) in unresolved, 'wind_completeness': None if str(station['src_id']) in unresolved else wind, 'pressure_completeness': None if str(station['src_id']) in unresolved else pressure,
                     'qualified': None if str(station['src_id']) in unresolved else all(v >= .85 for v in wind.values()), 'has_pressure': any(v > 0 for v in pressure.values())})
    turbine_manifest_path = root / 'output/external_acquisition/hill_of_towie/manifest.json'
    tm = json.loads(turbine_manifest_path.read_text())
    record = json.loads(args.record.read_text())
    registered = {f['key']: f for f in record['files']}
    directory = root / 'datasets/external_validation/sealed/hill_of_towie'
    checked = []
    for f in tm['files']:
        if f['download_status'] == 'SKIPPED_NOT_REQUIRED':
            continue
        item = registered[f['name']]; path = directory / safe_name(f['name'])
        if not (path.stat().st_size == item['size'] == f['actual_bytes'] and sha(path, 'md5') == item['checksum'].removeprefix('md5:') == f['official_md5'] and sha(path) == f['sha256']):
            raise ValueError('Turbine file hash mismatch')
        checked.append(f['name'])
    if any(f'{year}.zip' not in checked for year in YEARS):
        raise ValueError('Missing year archive')
    print('[VERIFIED] turbine bytes', flush=True)
    headers = {str(y): development_header(directory / f'{y}.zip', y) for y in range(2016, 2024)}
    metadata = {}
    for name in ['Hill_of_Towie_turbine_fields_description.csv', 'Hill_of_Towie_AeroUp_install_dates.csv']:
        if name not in checked:
            raise ValueError('Missing metadata')
        metadata[name] = list(csv.DictReader(io.StringIO((directory / name).read_text(encoding='cp1252'))))
    mapping = {'Wspd': ['wtc_AcWindSp_mean'], 'Etmp': ['wtc_AmbieTmp_mean'],
               'Ndir_sin': ['wtc_NacelPos_mean'], 'Ndir_cos': ['wtc_NacelPos_mean'],
               'Pab_mean': ['wtc_PitcPosA_mean', 'wtc_PitcPosB_mean', 'wtc_PitcPosC_mean'],
               'rotor_speed': ['wtc_MainSRpm_mean'], 'power_last10': ['wtc_ActPower_mean'], 'wind_last10': ['wtc_AcWindSp_mean']}
    columns = {c for year in headers.values() for cols in year.values() for c in cols}
    missing_by_year = {year: {k: [c for c in v if c not in {x for cols in hs.values() for x in cols}] for k,v in mapping.items() if not set(v) <= {x for cols in hs.values() for x in cols}} for year,hs in headers.items()}
    missing = {k: [c for c in v if c not in columns] for k, v in mapping.items() if not set(v) <= columns}
    gate = coverage_gate(rows)
    if unresolved:
        gate['passed'] = None
        gate['status'] = 'BLOCKED_UNRESOLVED_DUPLICATES'
    result = {'status': 'COMPLETED_WITH_BLOCKER' if unresolved else 'COMPLETED', 'utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': sha(__file__), 'midas_verified_files': len(seen), 'midas_bytes': manifest['planned_bytes'],
              'development_rule': 'Each full calendar year 2016-2023 >=85%; missing years count zero; original version_num=1 parsing; no imputation',
              'pressure_rule': 'At least one valid positive development pressure observation; per-year coverage reported, no extra pressure threshold imposed',
              'gate': gate, 'unresolved_stations': unresolved, 'stations': rows, 'parsed_development_files': len(parsed),
              'turbine_hash_verified': checked, 'test_years': [2024, 2025], 'protected_partial_years': [2026],
              'headers': headers, 'missing_by_year': missing_by_year, 'scada_mapping': mapping, 'missing_inputs': missing, 'metadata': metadata,
              'limitations': ['Headers only do not establish value availability or temporal alignment.', 'No formal confirmation protocol, training or test-value parsing.']}
    out.mkdir(parents=True)
    for name in ['station_registration.json', 'download_plan.json', 'download_manifest.json']:
        (out / name).write_bytes((audit / name).read_bytes())
    save(out / 'readonly_inventory.json', result)
    print(json.dumps(result['gate']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--record", type=Path, default=ROOT / "output/external_acquisition/hill_of_towie/record_22662930.json")
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--download-midas", action="store_true")
    parser.add_argument("--readonly-out", type=Path)
    args = parser.parse_args()
    if args.readonly_out and args.download_midas:
        parser.error("Read-only and download modes are mutually exclusive")
    if sha(args.record) != RECORD_SHA:
        raise ValueError("Pinned record hash differs")
    record = json.loads(args.record.read_text(encoding="utf-8"))
    if args.readonly_out:
        readonly_inventory(args)
        return 0
    args.audit_root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.audit_root / ".acquire.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen(args.audit_root / "sealed_years.json", registration(record))
        try:
            if args.download_midas:
                acquire(args, args.audit_root, record)
        except Exception as exc:
            # Do not serialize exception text, request objects or auth headers.
            save(args.audit_root / "failed.json", {"error_type": type(exc).__name__,
                 "utc": datetime.now(timezone.utc).isoformat(), "no_automatic_restart": True})
            print("[FAILED] " + type(exc).__name__ + "; see audit; credentials omitted", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
