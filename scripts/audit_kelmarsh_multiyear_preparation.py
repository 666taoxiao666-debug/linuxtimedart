"""Read-only raw-label audit of explicitly authorized prepared years; no model inference or fitting."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


DEV_YEARS = [2016, 2017, 2018, 2019]


def read_power_ticks(zip_path, unit):
    """Raw provider power ticks of one turbine from one yearly archive (no project code)."""
    with zipfile.ZipFile(zip_path) as archive:
        name = next(n for n in archive.namelist() if Path(n).name.startswith(f"Turbine_Data_Kelmarsh_{unit}_"))
        with archive.open(name) as raw:
            stream = io.TextIOWrapper(raw, encoding="utf-8-sig")
            while True:
                line = stream.readline()
                if not line:
                    raise ValueError("missing provider header")
                if line.startswith("# Date and time,"):
                    columns = next(csv.reader([line[2:].strip()]))
                    break
            ticks = pd.read_csv(stream, names=columns, usecols=["Date and time", "Power (kW)"])
    ticks["Date and time"] = pd.to_datetime(ticks["Date and time"], utc=True)
    return ticks


def flags_are_labels_not_deletions(meta, frame):
    """Prove that abnormal operation is only FLAGGED: (1) every turbine keeps a gap-free hourly grid from the start
    label to its last label, so no hour was dropped; (2) every hourly power label of every turbine and year, flagged
    hours included, equals the strict 6-tick mean rebuilt from the raw archives; (3) flagged, negative and above-rated
    hours are still present with their original values."""
    start_label = pd.Timestamp(meta["start"]) + pd.Timedelta(hours=1)
    flags = ["flag_stopped", "flag_derated", "flag_underperformance", "flag_incomplete_state"]
    grid, worst, worst_flagged, per_turbine = {}, 0.0, 0.0, {}
    expected_units = {int(unit) for unit in meta["static"]["ratings"]}
    assert set(frame.turbine_id.unique()) == expected_units, "prepared turbine set differs from raw/static identity"
    for unit in sorted(frame.turbine_id.unique()):
        rows = frame[frame.turbine_id == unit].sort_values("timestamp_hour_end_utc")
        stamps = rows.timestamp_hour_end_utc
        expected = int((stamps.iloc[-1] - stamps.iloc[0]) / pd.Timedelta(hours=1)) + 1
        steps = stamps.diff().dropna()
        assert stamps.iloc[0] == start_label, (unit, "first label", stamps.iloc[0])
        assert (steps == pd.Timedelta(hours=1)).all(), (unit, "hourly grid has gaps: rows were dropped")
        assert len(rows) == expected, (unit, "row count differs from the gap-free grid")
        grid[str(unit)] = {"rows": int(len(rows)), "first_label": str(stamps.iloc[0]), "last_label": str(stamps.iloc[-1])}
        ticks = pd.concat([read_power_ticks(e["path"], unit) for e in meta["files"]], ignore_index=True)
        ticks = ticks.drop_duplicates().sort_values("Date and time")
        assert not ticks["Date and time"].duplicated().any(), (unit, "conflicting raw ticks")
        ticks = ticks[ticks["Date and time"] >= pd.Timestamp(meta["start"])].set_index("Date and time")["Power (kW)"]
        buckets = ticks.resample("h", closed="left", label="right")
        rebuilt = buckets.mean().where(buckets.count() == 6)
        # A grid ending at the last *observed* row cannot detect deleted suffixes.
        # Require the entire raw-derived grid before comparing any labels.
        assert pd.DatetimeIndex(stamps).equals(rebuilt.index), (unit, "prepared grid differs from complete raw grid")
        observed = rows.power_kw.to_numpy(dtype=float)
        np.testing.assert_allclose(observed, rebuilt.to_numpy(dtype=float), rtol=0, atol=1e-9, equal_nan=True)
        difference = np.abs(observed - rebuilt.to_numpy(dtype=float))
        flagged = rows[flags].any(axis=1).to_numpy() & np.isfinite(observed)
        worst = max(worst, float(np.nanmax(difference)) if np.isfinite(difference).any() else 0.0)
        if flagged.any():
            worst_flagged = max(worst_flagged, float(np.nanmax(difference[flagged])))
        per_turbine[str(unit)] = {"hours_rebuilt": int(np.isfinite(observed).sum()), "flagged_hours_with_labels": int(flagged.sum())}
    complete = frame.power_kw.notna()
    return {"passed": True, "rows_dropped": 0, "grid": grid, "raw_rebuild_all_turbines_all_years": per_turbine,
            "raw_label_max_abs_difference_kw": worst, "flagged_hours_raw_label_max_abs_difference_kw": worst_flagged,
            "flagged_rows_present": {f: int(frame[f].sum()) for f in flags},
            "abnormal_complete_hours_present_with_original_labels": int((complete & ~frame.normal_operation.astype(bool)).sum()),
            "negative_power_hours_present": int((frame.power_kw < 0).sum()),
            "above_rated_hours_present": int((frame.power_kw > frame.rated_power_kw).sum()),
            "normal_operation_is_a_column_not_a_filter": True}


def audit(metadata_path, expected_years=DEV_YEARS, expected_curve_years=None):
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    expected_years = sorted(int(y) for y in expected_years)
    expected_curve_years = sorted(int(y) for y in (expected_curve_years or expected_years))
    assert meta["years"] == expected_years, "unexpected/unapproved years"
    assert meta["curve_years"] == expected_curve_years, "unexpected curve years"
    assert set(expected_curve_years) <= set(DEV_YEARS), "curve years must stay within the 2016-2019 training years"
    assert sorted(entry["year"] for entry in meta["files"]) == expected_years, "raw archive years differ from approved years"
    for entry in meta["files"]:
        assert sha(entry["path"]) == entry["sha256"]
        with zipfile.ZipFile(entry["path"]) as archive:
            names = [n for n in archive.namelist() if Path(n).name.startswith("Turbine_Data_")]
            assert len(names) == 6
            for name in names:
                with archive.open(name) as stream:
                    header = []
                    for _ in range(50):
                        line = stream.readline().decode("utf-8-sig").strip()
                        if line.startswith("# Date and time,"):
                            break
                        header.append(line)
                assert "# Time zone: UTC" in header, (entry["year"], name, "unverified timezone")
    assert sha(meta["prepared"]["path"]) == meta["prepared"]["sha256"]
    for source, digest in meta["source_sha256"].items():
        assert sha(source) == digest
    frame = pd.read_csv(meta["prepared"]["path"])
    frame["timestamp_hour_end_utc"] = pd.to_datetime(frame.timestamp_hour_end_utc, utc=True)
    assert not frame.duplicated(["turbine_id", "timestamp_hour_end_utc"]).any()
    assert set(frame.label_year) == set(meta["years"])
    np.testing.assert_array_equal(frame.label_year, (frame.timestamp_hour_end_utc - pd.Timedelta(hours=1)).dt.year)
    flags = ["flag_incomplete_power", "flag_incomplete_state", "flag_stopped", "flag_derated", "flag_underperformance"]
    np.testing.assert_array_equal(frame.normal_operation, ~frame[flags].any(axis=1))
    np.testing.assert_array_equal(frame.flag_incomplete_power, frame.power_kw.isna())
    np.testing.assert_array_equal(frame.flag_incomplete_state, frame.n_state_ticks.fillna(0) != 6)
    quality = pd.read_csv(Path(metadata_path).parent / "quality_by_year.csv")
    for _, row in quality.iterrows():
        group = frame[(frame.turbine_id == row.turbine_id) & (frame.label_year == row.year)]
        assert len(group) == row.hours
        assert group.power_kw.notna().sum() == row.complete_power_hours
        assert group.normal_operation.sum() == row.normal_operation_hours
        for key, flag in [("stopped_hours", "flag_stopped"), ("derated_hours", "flag_derated"),
                          ("underperformance_hours", "flag_underperformance"), ("incomplete_state_hours", "flag_incomplete_state")]:
            assert group[flag].sum() == row[key]
    # Independently reconstruct all 2016 turbine-1 hourly labels from the raw archive.
    entry = next(e for e in meta["files"] if e["year"] == 2016)
    with zipfile.ZipFile(entry["path"]) as archive:
        name = next(n for n in archive.namelist() if Path(n).name.startswith("Turbine_Data_Kelmarsh_1_"))
        with archive.open(name) as raw:
            stream = io.TextIOWrapper(raw, encoding="utf-8-sig")
            while True:
                line = stream.readline()
                if not line:
                    raise ValueError("missing provider header")
                if line.startswith("# Date and time,"):
                    columns = next(csv.reader([line[2:].strip()]))
                    break
            ticks = pd.read_csv(stream, names=columns, usecols=["Date and time", "Power (kW)"])
    ticks["Date and time"] = pd.to_datetime(ticks["Date and time"], utc=True)
    ticks = ticks[ticks["Date and time"] >= pd.Timestamp(meta["start"])]
    ticks = ticks.set_index("Date and time")["Power (kW)"]
    buckets = ticks.resample("h", closed="left", label="right")
    expected = buckets.mean().where(buckets.count() == 6)
    observed = frame[(frame.turbine_id == 1) & (frame.label_year == 2016)].set_index("timestamp_hour_end_utc").power_kw
    expected = expected.reindex(observed.index)
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-9, equal_nan=True)
    annual = frame.groupby("label_year").agg(hours=("power_kw", "size"), complete=("power_kw", "count"),
        normal=("normal_operation", "sum"), stopped=("flag_stopped", "sum"), derated=("flag_derated", "sum"),
        underperformance=("flag_underperformance", "sum"), unknown_state=("flag_incomplete_state", "sum"))
    annual["normal_pct_of_complete"] = 100 * annual.normal / annual.complete
    result = {"passed": True, "new_model_fits": 0, "years": meta["years"], "rows": len(frame),
        "csv_sha256": meta["prepared"]["sha256"], "metadata_sha256": sha(metadata_path),
        "all_input_hashes_match": True, "source_hashes_match": True, "quality_rows_checked": len(quality),
        "all_24_provider_headers_declare_utc": True,
        "independent_raw_label_check_hours": len(observed),
        "raw_label_max_abs_difference_kw": float(np.nanmax(np.abs(observed.to_numpy() - expected.to_numpy()))),
        "annual_quality": annual.reset_index().to_dict(orient="records"),
        "negative_hourly_labels": int((frame.power_kw < 0).sum()),
        "above_rated_hourly_labels": int((frame.power_kw > frame.rated_power_kw).sum()),
        "power_min_kw": float(frame.power_kw.min()), "power_max_kw": float(frame.power_kw.max()),
        "flags_not_deletions": flags_are_labels_not_deletions(meta, frame)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", default=DEV_YEARS,
                        help="approved prepared years (v28 preparation: 2016 2017 2018 2019 2020 2021)")
    parser.add_argument("--curve-years", type=int, nargs="+", default=None, help="default: same as --years")
    args = parser.parse_args()
    result = audit(args.metadata, args.years, args.curve_years)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
