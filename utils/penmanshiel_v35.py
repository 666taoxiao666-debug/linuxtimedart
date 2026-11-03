"""P1 farm-specific routing; numerical preparation stays in frozen Kelmarsh functions."""
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

from scripts import prepare_kelmarsh_multiyear as frozen
from utils import kelmarsh_upstream_v31 as upstream

TURBINES = (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
STATIONS = (150, 161, 177, 235, 982, 987, 1023, 1074, 16589, 24125, 30523)
YEARS = tuple(range(2016, 2022))
CURVE_YEARS = tuple(range(2016, 2020))
SPLITS = {"train": CURVE_YEARS, "validation": (2020,), "development_eval": (2021,)}
END = "2022-01-01T01:00:00Z"  # exclusive hour-end; last permitted label covers 2021-12-31 23h
DATA_SUBDIR = Path("datasets/wind_public_v35/resampled_hourly_scada/penmanshiel_multiyear")
D2_SUBDIR = Path("output/kelmarsh_hourly/d2_feasibility_v34_r2")
D2_SHA = {
    "part_A.json": "f68db4f1812bd2d13f9e497f1ab98c4ded3839d686099a887ab42f010a4e9cf4",
    "input_hashes.json": "331061575f151ff1d47365bdb624ad745c32d53632fe72112a887e9e5684ab07",
    "test_year_registration.json": "0622d055799e99c0cda17d3c1ec67755706773856007e0182614466757faf180",
}


def validate_static(frame):
    """Only entirely empty metadata rows may be discarded; turbine 03 is intentionally absent."""
    clean = frame.dropna(how="all")
    units = []
    for title in clean["Title"]:
        match = re.fullmatch(r"Penmanshiel (\d{2})", str(title))
        if not match:
            raise ValueError(f"Invalid nonempty static row: {title!r}")
        units.append(int(match[1]))
    if sorted(units) != list(TURBINES):
        raise ValueError("Static turbine identities differ from frozen fourteen turbines")
    if not clean["Model"].eq("MM82").all() or not pd.to_numeric(clean["Rated power (kW)"], errors="raise").eq(2050).all():
        raise ValueError("Model/rated power differs from D2 registration")
    return {f"{unit:02d}": {"rated_power_kw": 2050.0} for unit in units}, {
        "original_rows": len(frame), "entirely_empty_rows_removed": len(frame) - len(clean),
        "turbines": sorted(units), "model": "MM82", "rated_kw": 2050.0,
    }


def development_member(entry):
    """Check all routing metadata before opening an archive or parsing any content."""
    match = re.match(r"^Penmanshiel_SCADA_(\d{4})_", Path(entry["archive"]).name)
    member = re.fullmatch(r"Turbine_Data_Penmanshiel_(\d{2})_(\d{4}-\d{2}-\d{2})_-_(\d{4}-\d{2}-\d{2})_\d+\.csv", Path(entry["member"]).name)
    if match is None or member is None:
        raise ValueError("Unrecognized registered SCADA identity")
    year = int(match[1])
    begin, end = pd.Timestamp(entry["begin"]), pd.Timestamp(entry["end"])
    if year not in YEARS or begin.year not in YEARS or end > pd.Timestamp("2022-01-01"):
        raise ValueError("Protected or undeclared year: content access forbidden")
    if (int(member[1]) != int(entry["turbine"]) or int(member[1]) not in TURBINES
            or member[2] != entry["begin"] or member[3] != entry["end"]
            or begin.year != year or end <= begin or end > pd.Timestamp(year + 1, 1, 1)):
        raise ValueError("SCADA registration/member identity mismatch")
    return year


def read_member(entry, path):
    development_member(entry)
    if Path(path).name != Path(entry["archive"]).name:
        raise ValueError("Archive routing mismatch")
    with zipfile.ZipFile(path) as archive:
        if archive.getinfo(entry["member"]).file_size != entry["bytes"]:
            raise ValueError("Registered member size changed")
        frame, missing = frozen.read_turbine_ticks(archive, entry["member"])
    lo, hi = pd.Timestamp(entry["begin"], tz="UTC"), pd.Timestamp(entry["end"], tz="UTC")
    if not ((frame.timestamp >= lo) & (frame.timestamp < hi)).all():
        raise ValueError("Raw SCADA timestamp outside registered development interval")
    return frame, missing


def first_complete_month(members):
    for entry in members:
        development_member(entry)
    if sorted({int(e["turbine"]) for e in members}) != list(TURBINES):
        raise ValueError("Missing turbine in development registration")
    # File interval coverage only: never search months for better labels or missingness.
    for lo in pd.date_range("2016-01-01", "2021-12-01", freq="MS"):
        hi = lo + pd.offsets.MonthBegin(1)
        covered = True
        for unit in TURBINES:
            cursor = lo
            intervals = sorted((pd.Timestamp(e["begin"]), pd.Timestamp(e["end"]))
                               for e in members if int(e["turbine"]) == unit)
            for begin, end in intervals:
                if begin <= cursor < end:
                    cursor = end
            covered &= cursor >= hi
        if covered:
            return lo.isoformat() + "Z"
    raise ValueError("No complete calendar month shared by all fourteen turbines")


def station_files(stations):
    if [int(s["src_id"]) for s in stations] != list(STATIONS):
        raise ValueError("Station identities/order changed")
    files = {}
    for station in stations:
        for year in YEARS:
            paths = station["years"][str(year)]["files"]
            if not paths:
                raise ValueError("Frozen station lacks development files")
            for path in paths:
                match = re.search(r"_qcv-1_(\d{4})\.csv$", Path(path).name)
                if match is None or int(match[1]) != year:
                    raise ValueError("Protected or mismatched MIDAS filename")
                files[path] = {"path": path, "year": year}
    return list(files.values())


def assemble_series(frame_or_path, features, preparation, start):
    return upstream.KelmarshUpstreamSeries(frame_or_path, features, splits=SPLITS,
        input_len=24, horizon=4, stride=3, extras=("power_last10", "wind_last10"),
        start=start, end=END, preparation=preparation)


def series_audit(series, features):
    aligned = features.reindex(pd.DatetimeIndex(series.dates)).to_numpy(dtype=float)
    missing = ~np.isfinite(aligned).all(axis=1)
    yearly = {}
    for year in YEARS:
        mask = series.label_year == year
        yearly[str(year)] = {"rows": int(mask.sum()), "missing_rows": int(missing[mask].sum()),
                             "missing_fraction": float(missing[mask].mean()) if mask.any() else None}
    return {**series.audit, "turbines": sorted(int(x) for x in np.unique(series.turbines)),
            "finite_fraction_after_input_fill": dict(zip(series.channels, np.isfinite(series.raw).mean(0).tolist())),
            "upstream_finite_fraction_before_fill": dict(zip(upstream.UPSTREAM_FEATURES, np.isfinite(aligned).mean(0).tolist())),
            "upstream_missing_by_year": yearly}
