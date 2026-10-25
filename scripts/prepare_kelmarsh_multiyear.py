#!/usr/bin/env python3
"""Download (optional) and prepare multi-year Kelmarsh SCADA into ONE enriched hourly CSV. No model is trained.

Source: Zenodo record 16807551 (Cubico Sustainable Investments, CC BY 4.0), yearly ``Kelmarsh_SCADA_<year>_<id>.zip``.
Per turbine, all selected years' 10-minute ticks are merged, then:
  * base hourly columns: exactly ``hourly_scada`` of the existing acquisition script (strict 6x10-min means, counts,
    partial power mean); labels are not modified, clipped or imputed;
  * extra causal input features: end-of-hour tick, within-hour std/min/max, yaw misalignment, available capacity;
  * operating-state flags and ``normal_operation`` (utils/kelmarsh_enrichment.RULES, frozen before any model run);
    the power curve of the underperformance flag is fitted per turbine on ``--curve-years`` only.
Outputs (under --root/resampled_hourly_scada/kelmarsh_multiyear): hourly/kelmarsh_multiyear_hourly_enriched.csv,
metadata/preparation.json (files, hashes, rules, curves, decisions) and metadata/quality_by_year.csv.

Run on a machine with internet access for the first download (resumable, official MD5 verified):
  python scripts/prepare_kelmarsh_multiyear.py --download
Defaults prepare only development years 2016-2019. Later years require an explicit authorized --years list.
Existing prepared outputs are never overwritten. Offline preparation reuses verified archives and cached metadata.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.acquire_hourly_wind_datasets import (KEL_COLUMNS, checked_download, digest, hourly_scada,  # noqa: E402
                                                  save_json, write_frame, zenodo_record)
from utils import kelmarsh_enrichment as enrich  # noqa: E402

RECORD_ID = 16807551
SCADA_PATTERN = re.compile(r"^Kelmarsh_SCADA_(\d{4})_(\d+)\.zip$")
SUBDIR = Path("resampled_hourly_scada/kelmarsh_multiyear")
REUSE_SUBDIRS = (Path("resampled_hourly_scada/kelmarsh_2016/raw"),)
DEFAULT_CURVE_YEARS = (2016, 2017, 2018, 2019)
DEFAULT_YEARS = DEFAULT_CURVE_YEARS  # never inspect held-out years implicitly
SCHEMA_VERSION = 2
DEFAULT_START = "2016-05-01T00:00:00Z"   # same start as v19-v27 (commissioning on 2016-04-15 excluded)
ANGLE_COLUMNS = ("wind_direction_deg", "nacelle_position_deg", "yaw_bearing_deg")
OUTPUT_NAME = "kelmarsh_multiyear_hourly_enriched.csv"


def scada_files(record):
    """{year: file entry} for every yearly SCADA archive listed in the Zenodo record."""
    out = {}
    for entry in record["files"]:
        match = SCADA_PATTERN.match(entry["key"])
        if match:
            year = int(match.group(1))
            if year in out:
                raise ValueError(f"Two SCADA archives for {year}; inspect the record")
            out[year] = entry
    return dict(sorted(out.items()))


def locate_or_download(record, key, raw_dir, reuse_dirs, download, workers=4):
    """Verified local copy (size + official MD5) from raw_dir or a reuse dir; else download if allowed."""
    entry = next(item for item in record["files"] if item["key"] == key)
    md5, size = entry["checksum"].removeprefix("md5:"), int(entry["size"])
    for directory in [raw_dir, *reuse_dirs]:
        path = Path(directory) / key
        if path.exists():
            if path.stat().st_size != size or digest(path, "md5") != md5:
                raise ValueError(f"Existing raw file failed integrity, preserved: {path}")
            return path
    if not download:
        raise FileNotFoundError(f"{key} not present locally; rerun with --download on a machine with internet access")
    target = Path(raw_dir) / key
    checked_download(entry["links"]["self"], target, size, md5, workers)
    return target


def read_turbine_ticks(archive, name):
    """Parse one provider member (comment header, then CSV). Optional columns absent in a year become NaN."""
    with archive.open(name) as raw:
        stream = io.TextIOWrapper(raw, encoding="utf-8-sig")
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"SCADA header not found in {name}")
            if line.startswith("# Date and time,"):
                columns = next(csv.reader([line[2:].strip()]))
                break
        present = [c for c in KEL_COLUMNS if c in columns]
        required = [source for source, target in KEL_COLUMNS.items()
                    if target in ("timestamp",) + enrich.REQUIRED_TICK_COLUMNS]
        missing_required = [c for c in required if c not in columns]
        if missing_required:
            raise ValueError(f"{name}: required source columns missing {missing_required}")
        frame = pd.read_csv(stream, names=columns, usecols=present, na_values=["NaN"])
    frame = frame.rename(columns=KEL_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for column in frame.columns.difference(["timestamp"]):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if np.isinf(frame[column].to_numpy(dtype=float)).any():
            raise ValueError(f"{name}: non-finite source values in {column}")
    missing_optional = [c for c in enrich.OPTIONAL_TICK_COLUMNS if c not in frame]
    for column in missing_optional:
        frame[column] = np.nan
    return frame, missing_optional


def merge_ticks(frames):
    """Concatenate yearly tick frames of one turbine; identical boundary duplicates are dropped, conflicts refused."""
    frame = pd.concat(frames, ignore_index=True).sort_values("timestamp", kind="mergesort")
    frame = frame.drop_duplicates()
    if frame["timestamp"].duplicated().any():
        raise ValueError("Conflicting duplicate ticks across archives; inspect the source")
    return frame.reset_index(drop=True)


def static_ratings(path):
    static = pd.read_csv(path, encoding="utf-8-sig")
    out = {}
    for _, row in static.iterrows():
        unit = str(row["Title"]).split()[-1]
        out[unit] = {"rated_power_kw": float(row["Rated power (kW)"]),
                     "commercial_operations_date": str(row.get("Commercial Operations Date"))}
    return out


def available_capacity_mask(ticks, rules):
    """Use each actually observed tick. Annual shares are descriptive, never an input gate.

    A full-year coverage gate would let future missingness change past input features.
    Missing capacity remains unknown rather than a zero/normal capacity observation.
    """
    years = ticks["timestamp"].dt.year
    decisions = {}
    usable = ticks["available_capacity_kw"].notna()
    for year, group in ticks.groupby(years):
        share = enrich.available_capacity_share(group)
        decisions[int(year)] = {"valid_share": share, "derate_rule_used": bool(usable.loc[group.index].any()),
                                "decision_basis": "observed tick only; yearly share is audit-only"}
    return usable, decisions


def prepare_turbine(unit, ticks, rated_kw, start, rules):
    ticks = ticks[ticks["timestamp"] >= pd.Timestamp(start)].copy()
    if ticks.empty:
        raise ValueError(f"Turbine {unit} has no ticks after {start}")
    usable, decisions = available_capacity_mask(ticks, rules)
    state_ticks = ticks.copy()
    state_ticks.loc[~usable, "available_capacity_kw"] = np.nan
    enrich.validate_tick_index(pd.DatetimeIndex(ticks["timestamp"]))
    base = hourly_scada(ticks.copy(), [c for c in ANGLE_COLUMNS if c in ticks])
    extra = enrich.hourly_enrichment(state_ticks.set_index("timestamp"), rated_kw, rules).reset_index()
    hourly = base.merge(extra, on="timestamp_hour_end_utc", how="left", validate="one_to_one")
    hourly["turbine_id"] = unit
    hourly["rated_power_kw"] = rated_kw
    hourly["label_year"] = enrich.label_year(hourly["timestamp_hour_end_utc"]).to_numpy()
    flags = enrich.state_flags(hourly)
    for column in flags:
        hourly[column] = flags[column].to_numpy()
    return hourly, decisions


def quality_by_year(frame):
    rows = []
    for (unit, year), group in frame.groupby(["turbine_id", "label_year"]):
        complete = group["power_kw"].notna()
        rows.append({"turbine_id": unit, "year": int(year), "hours": len(group), "complete_power_hours": int(complete.sum()),
                     "normal_operation_hours": int(group["normal_operation"].sum()),
                     "normal_share_of_complete_pct": float(100 * group["normal_operation"].sum() / max(complete.sum(), 1)),
                     "stopped_hours": int(group["flag_stopped"].sum()), "derated_hours": int(group["flag_derated"].sum()),
                     "underperformance_hours": int(group["flag_underperformance"].sum()),
                     "incomplete_state_hours": int(group["flag_incomplete_state"].sum()),
                     "capacity_fully_observed_hours": int((group["n_capacity_ticks"] == 6).sum()),
                     "negative_power_hours": int((group["power_kw"] < 0).sum()),
                     "end_tick_available_pct": float(100 * group["power_last10_kw"].notna().mean()),
                     "mean_power_kw": float(group["power_kw"].mean()) if complete.any() else None,
                     "capacity_factor_pct": float(100 * group["power_kw"].mean() / group["rated_power_kw"].iloc[0]) if complete.any() else None})
    return pd.DataFrame(rows)


def prepare(root, years=None, curve_years=DEFAULT_CURVE_YEARS, download=False, start=DEFAULT_START, rules=enrich.RULES,
            record=None):
    root = Path(root)
    directory = root / SUBDIR
    for existing in (directory / "hourly" / OUTPUT_NAME, directory / "metadata" / "preparation.json",
                     directory / "metadata" / "quality_by_year.csv"):
        if existing.exists():
            raise FileExistsError(f"Prepared output already exists, preserved: {existing}; use a separate --root")
    raw_dir = directory / "raw"
    reuse = [root / sub for sub in REUSE_SUBDIRS]
    if record is None:
        # Reuse the verified record cached by the original acquisition, including offline runs.
        cached = next((root / sub.parent for sub in REUSE_SUBDIRS
                       if (root / sub.parent / "metadata" / f"zenodo_{RECORD_ID}.json").exists()), directory)
        record = zenodo_record(RECORD_ID, cached)
    archives = scada_files(record)
    chosen = sorted(DEFAULT_YEARS) if years is None else sorted(int(y) for y in years)
    if not chosen or len(chosen) != len(set(chosen)) or not curve_years:
        raise ValueError("Explicit nonempty unique prepared years and curve years are required")
    absent = [y for y in chosen if y not in archives]
    if absent:
        raise ValueError(f"Years not in the Zenodo record: {absent}; available {sorted(archives)}")
    if not set(curve_years) <= set(chosen):
        raise ValueError("Curve years must be prepared years")
    static_path = locate_or_download(record, "Kelmarsh_WT_static.csv", raw_dir, reuse, download)
    ratings = static_ratings(static_path)
    files, per_turbine, missing_columns = [], {}, {}
    for year in chosen:
        key = archives[year]["key"]
        path = locate_or_download(record, key, raw_dir, reuse, download)
        files.append({"year": year, "key": key, "path": str(path), "md5": archives[year]["checksum"],
                      "sha256": digest(path), "bytes": path.stat().st_size})
        with zipfile.ZipFile(path) as archive:
            members = sorted(n for n in archive.namelist() if Path(n).name.startswith("Turbine_Data_"))
            if len(members) != 6:
                raise ValueError(f"{key}: expected six turbine members, found {len(members)}")
            member_units = [re.search(r"Kelmarsh_(\d+)_", Path(n).name) for n in members]
            if any(m is None for m in member_units) or sorted(m.group(1) for m in member_units) != sorted(ratings):
                raise ValueError(f"{key}: turbine member identities do not match static ratings")
            for name in members:
                unit = re.search(r"Kelmarsh_(\d+)_", Path(name).name).group(1)
                frame, missing = read_turbine_ticks(archive, name)
                per_turbine.setdefault(unit, []).append(frame)
                missing_columns[f"{year}/{unit}"] = missing
                print(f"[READ] {key} turbine {unit}: {len(frame)} ticks", flush=True)
    frames, decisions = [], {}
    for unit in sorted(per_turbine, key=int):
        ticks = merge_ticks(per_turbine[unit])
        hourly, decisions[unit] = prepare_turbine(unit, ticks, ratings[unit]["rated_power_kw"], start, rules)
        frames.append(hourly)
    frame = pd.concat(frames, ignore_index=True)
    curves = {}
    frame["flag_underperformance"] = False
    for unit, group in frame.groupby("turbine_id"):
        rated = float(group["rated_power_kw"].iloc[0])
        curve = enrich.fit_power_curve(group["wind_speed_mps"], group["power_kw"], group["label_year"], curve_years,
                                       group["flag_stopped"], group["flag_derated"], rated, rules)
        curves[unit] = curve
        frame.loc[group.index, "flag_underperformance"] = enrich.underperformance_flag(
            group["wind_speed_mps"], group["power_kw"], curve, rated, rules)
    frame["normal_operation"] = enrich.normal_operation(frame).to_numpy()
    frame = frame.sort_values(["turbine_id", "timestamp_hour_end_utc"], key=lambda s: s.astype(int) if s.name == "turbine_id" else s)
    prepared = write_frame(frame, directory / "hourly", OUTPUT_NAME)
    (directory / "metadata").mkdir(parents=True, exist_ok=True)
    quality = quality_by_year(frame)
    quality.to_csv(directory / "metadata" / "quality_by_year.csv", index=False)
    metadata = {"schema_version": SCHEMA_VERSION,
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                                  (Path(__file__), ROOT / "utils/kelmarsh_enrichment.py",
                                   ROOT / "scripts/acquire_hourly_wind_datasets.py")},
                "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "record": RECORD_ID,
                "license": "CC-BY-4.0 (Cubico Sustainable Investments Ltd)", "years": chosen, "start": start,
                "files": files, "static": {"path": str(static_path), "sha256": digest(static_path), "ratings": ratings},
                "prepared": prepared, "rules": rules, "curve_years": sorted(curve_years), "power_curves": curves,
                "available_capacity_decisions": decisions, "missing_optional_columns": missing_columns,
                "labels_modified": False, "interpolation": False, "clipping": False,
                "target_transformation": "strict original 6x10-minute hourly average",
                "normal_operation_definition": "complete power and wind/state hour AND no observed stopped tick AND no observed derated tick AND not "
                                               "underperforming against the curve-year power curve",
                "capacity_unknown_policy": "unknown capacity does not prove absence of derating; report observed coverage",
                "normal_cohort_uses_future_target_state": True,
                "cohort_policy": "user decision 2026-10-25: train and primary evaluation on normal-operation windows; the "
                                 "full original label set must always be reported as a secondary evaluation",
                "rows": int(len(frame)), "normal_operation_hours": int(frame["normal_operation"].sum()),
                "complete_power_hours": int(frame["power_kw"].notna().sum())}
    save_json(directory / "metadata" / "preparation.json", metadata)
    print(f"[COMPLETE] {prepared['rows']} hourly rows, {metadata['normal_operation_hours']} normal-operation hours of "
          f"{metadata['complete_power_hours']} complete -> {prepared['path']}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT / "datasets/wind_public")
    parser.add_argument("--years", type=int, nargs="+", default=None, help="default: development/curve years 2016-2019 only")
    parser.add_argument("--curve-years", type=int, nargs="+", default=list(DEFAULT_CURVE_YEARS))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--download", action="store_true", help="download missing archives (needs internet)")
    args = parser.parse_args()
    prepare(args.root, args.years, tuple(args.curve_years), args.download, args.start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
