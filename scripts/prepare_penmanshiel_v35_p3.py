"""Authorized P3 routing. Frozen P1 numerical functions and labels are reused."""
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
from scripts import prepare_penmanshiel_v35 as p1
from utils import penmanshiel_v35 as farm
from utils import kelmarsh_upstream_v32 as u

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/penmanshiel_p3_protocol_v35.json"
PROTOCOL_SHA256 = "a99b7ca91910b159088991bdbc680ba95ed240410e02c414f3e68ada22fe8926"


def protocol():
    if u.text_sha256_lf(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("P3 frozen protocol SHA256 mismatch")
    p = u.read_json(PROTOCOL)
    if u.text_sha256_lf(ROOT / p["recipe_path"]) != p["recipe_sha256_lf"]:
        raise ValueError("Frozen inherited recipe changed")
    return p


def verify_old_rows(current, original):
    old = current[current.label_year <= 2021].copy()
    order = ["turbine_id", "timestamp_hour_end_utc"]
    pd.testing.assert_frame_equal(old.sort_values(order).reset_index(drop=True),
                                  original.sort_values(order).reset_index(drop=True), check_exact=True)
    return {"passed": True, "rows": len(old), "columns": len(old.columns), "labels_modified": 0}


def read_authorized_member(entry, path):
    # Same validation as P1, with only the explicitly authorized year bound extended.
    match = re.match(r"^Penmanshiel_SCADA_(\d{4})_", Path(entry["archive"]).name)
    member = re.fullmatch(r"Turbine_Data_Penmanshiel_(\d{2})_(\d{4}-\d{2}-\d{2})_-_(\d{4}-\d{2}-\d{2})_\d+\.csv", Path(entry["member"]).name)
    lo, hi = pd.Timestamp(entry["begin"], tz="UTC"), pd.Timestamp(entry["end"], tz="UTC")
    if (match is None or member is None or not 2016 <= lo.year <= 2024
            or int(match[1]) != lo.year or int(member[1]) != int(entry["turbine"])
            or int(member[1]) not in farm.TURBINES or member[2] != entry["begin"]
            or member[3] != entry["end"] or hi <= lo or hi > pd.Timestamp(f"{lo.year+1}-01-01", tz="UTC")
            or Path(path).name != Path(entry["archive"]).name):
        raise ValueError("P3 registered member identity mismatch")
    with zipfile.ZipFile(path) as archive:
        if archive.getinfo(entry["member"]).file_size != entry["bytes"]:
            raise ValueError("Registered member size changed")
        frame, missing = p1.frozen.read_turbine_ticks(archive, entry["member"])
    if not ((frame.timestamp >= lo) & (frame.timestamp < hi)).all():
        raise ValueError("Raw timestamp outside authorized registered interval")
    return frame, missing


def prepare(data_root, out):
    protocol()  # No raw-data access before checking the already committed protocol.
    out = Path(out)
    complete = out / "prepared_complete.json"
    if complete.exists():
        record = u.read_json(complete)
        for name, sha in record["artifacts"].items():
            if u.file_sha256(out / name) != sha:
                raise ValueError("Prepared artifact changed: " + name)
        for name, entry in u.read_json(out / "input_hashes.json").items():
            if Path(name).stat().st_size != entry["bytes"] or u.file_sha256(name) != entry["sha256"]:
                raise ValueError("Registered input changed: " + name)
        return record
    if out.exists():
        raise FileExistsError("Partial preparation preserved; stop for user review")
    out.mkdir(parents=True)
    # P1 inputs checks the D2 metadata locks and all SCADA archives plus all
    # protected files for these eleven stations as bytes BEFORE any parsing.
    dev_members, stations, files, resolved, verified, static, protected, center = p1.inputs(data_root)
    u.write_json(out / "input_hashes.json", verified)
    registry = u.read_json(ROOT / farm.D2_SUBDIR / "test_year_registration.json")
    members = registry["members"]
    if sorted({int(e["begin"][:4]) for e in members}) != list(range(2016, 2025)):
        raise ValueError("Registered years changed")
    for path in protected:
        match = re.search(r"_qcv-1_(\d{4})\.csv$", Path(path).name)
        if match is None or int(match[1]) not in (2022, 2023, 2024):
            raise ValueError("Protected station filename/year changed")
        files.append({"path": path, "year": int(match[1])})
    if len(files) != len({e["path"] for e in files}):
        raise ValueError("Duplicate station registration")
    for sid in farm.STATIONS:
        for year in range(2016, 2025):
            if sum(int(Path(e["path"]).parent.name) == sid and e["year"] == year for e in files) != 1:
                raise ValueError(f"Station/year registration missing or ambiguous: {sid}/{year}")
    start = farm.first_complete_month(dev_members)
    ratings, static_audit = farm.validate_static(pd.read_csv(resolved[static], encoding="utf-8-sig"))
    frames, duplicates, optional, decisions, label_checks = [], [], {}, {}, []
    for unit in farm.TURBINES:
        parts = []
        for entry in members:
            if int(entry["turbine"]) == unit:
                ticks, missing = read_authorized_member(entry, resolved[entry["archive"]])
                parts.append(ticks)
                optional[entry["member"]] = missing
        ticks, audit = p1.canonical_ticks(parts)
        del parts
        duplicates.append({"turbine_id": unit, **audit})
        key = f"{unit:02d}"
        hourly, decisions[key] = p1.frozen.prepare_turbine(key, ticks, ratings[key]["rated_power_kw"], start, p1.enrich.RULES)
        ref = p1.frozen.hourly_scada(ticks.loc[ticks.timestamp >= pd.Timestamp(start), ["timestamp", "power_kw"]].copy())
        pd.testing.assert_series_equal(hourly.power_kw, ref.power_kw, check_names=False, check_exact=True)
        pd.testing.assert_series_equal(hourly.timestamp_hour_end_utc, ref.timestamp_hour_end_utc, check_exact=True)
        label_checks.append({"turbine": unit, "rows": len(hourly), "labels_modified": 0})
        frames.append(hourly)
        print(f"[PREP] turbine={key} hours={len(hourly)}", flush=True)
    frame = pd.concat(frames, ignore_index=True)
    del frames, ticks, hourly, ref
    original_labels = frame.power_kw.copy()
    curves = {}
    frame["flag_underperformance"] = False
    for unit, group in frame.groupby("turbine_id"):
        rated = float(group.rated_power_kw.iloc[0])
        curves[unit] = p1.enrich.fit_power_curve(group.wind_speed_mps, group.power_kw, group.label_year,
            farm.CURVE_YEARS, group.flag_stopped, group.flag_derated, rated, p1.enrich.RULES)
        frame.loc[group.index, "flag_underperformance"] = p1.enrich.underperformance_flag(
            group.wind_speed_mps, group.power_kw, curves[unit], rated, p1.enrich.RULES)
    frame["normal_operation"] = p1.enrich.normal_operation(frame).to_numpy()
    pd.testing.assert_series_equal(frame.power_kw, original_labels, check_exact=True)
    if not frame.label_year.isin(range(2016, 2025)).all():
        raise ValueError("Prepared labels escaped authorized years")
    prepared = p1.frozen.write_frame(frame.sort_values(["turbine_id", "timestamp_hour_end_utc"]), out, "penmanshiel.csv")
    oldmeta = u.read_json(ROOT / "output/penmanshiel_hourly/prep_v35/preparation.json")
    oldaudit = u.read_json(ROOT / "output/penmanshiel_hourly/prep_v35/series_audit.json")
    oldcsv = data_root / farm.DATA_SUBDIR / "hourly/penmanshiel_multiyear_hourly_enriched.csv"
    oldup = data_root / farm.DATA_SUBDIR / "hourly/penmanshiel_upstream_raw.csv"
    if u.file_sha256(oldcsv) != oldmeta["prepared"]["sha256"] or u.file_sha256(oldup) != oldaudit["upstream_file"]["sha256"]:
        raise ValueError("P1 reference hash changed")
    equality = verify_old_rows(pd.read_csv(prepared["path"]), pd.read_csv(oldcsv))
    observations = u.load_station_observations([{"path": resolved[e["path"]], "year": e["year"]} for e in files], farm.STATIONS)
    labels = pd.date_range(frame.timestamp_hour_end_utc.min(), frame.timestamp_hour_end_utc.max(), freq="h").tz_localize(None)
    features = u.hourly_upstream(observations, stations, labels, lag=1)
    features.index.name = "timestamp_hour_end_utc"
    upfile = p1.frozen.write_frame(features.reset_index(), out, "upstream.csv")
    oldfeatures = pd.read_csv(oldup, index_col="timestamp_hour_end_utc")
    newfeatures = pd.read_csv(upfile["path"], index_col="timestamp_hour_end_utc")
    pd.testing.assert_frame_equal(newfeatures.loc[oldfeatures.index], oldfeatures, check_exact=True)
    equality["upstream_rows"] = len(oldfeatures)
    u.write_json(out / "development_equality.json", equality)
    meta = {"schema_version": 2, "start": start, "end_exclusive": "2025-01-01T01:00:00Z",
        "prepared": prepared, "upstream_file": upfile, "curve_years": list(farm.CURVE_YEARS), "power_curves": curves,
        "static": static_audit, "duplicate_normalization": duplicates, "missing_optional_columns": optional,
        "available_capacity_decisions": decisions, "label_identity_checks": label_checks, "labels_modified": False,
        "protocol_sha256": PROTOCOL_SHA256, "center": center}
    u.write_json(out / "preparation.json", meta)
    # Coverage only. No target performance statistics before all models freeze.
    coverage = [{"turbine": str(t), "year": int(y), "rows": len(g), "finite_labels": int(g.power_kw.notna().sum())}
                for (t, y), g in frame.groupby(["turbine_id", "label_year"])]
    u.write_json(out / "coverage.json", coverage)
    record = {"protocol_sha256": PROTOCOL_SHA256, "artifacts": {p.name: u.file_sha256(p) for p in sorted(out.iterdir()) if p.is_file()}}
    u.write_json(complete, record)
    return record
