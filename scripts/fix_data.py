"""Convert the original SDWPF table into the format used by Dataset_SDWPF."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd


MODEL_COLUMNS = [
    "Wspd",
    "Wdir",
    "Etmp",
    "Itmp",
    "Ndir",
    "Pab1",
    "Pab2",
    "Pab3",
    "Prtv",
]


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_sdwpf(
    input_file,
    output_file,
    base_date="2023-01-01",
    manifest_file=None,
):
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    try:
        df = pd.read_csv(input_file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding="gbk")
    input_rows = int(len(df))

    required = {"TurbID", "Day", "Tmstamp", "Patv"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    df["Day"] = pd.to_numeric(df["Day"], errors="coerce")
    time_text = df["Tmstamp"].astype(str).str.strip()
    time_part = pd.to_timedelta(time_text, errors="coerce")
    failed = time_part.isna()
    if failed.any():
        time_part.loc[failed] = pd.to_timedelta(
            time_text.loc[failed] + ":00", errors="coerce"
        )
    day_part = pd.to_timedelta(df["Day"] - 1, unit="D")
    df["date"] = pd.Timestamp(base_date) + day_part + time_part
    df = df.rename(columns={"Patv": "power"})

    columns = ["date", "TurbID", "Day"]
    columns.extend(column for column in MODEL_COLUMNS if column in df.columns)
    columns.append("power")
    df = df[columns].dropna(subset=["date", "TurbID", "power"])
    df = (
        df.sort_values(["TurbID", "date"], kind="mergesort")
        .drop_duplicates(["TurbID", "date"], keep="last")
        .reset_index(drop=True)
    )

    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_file, index=False)
    manifest_file = manifest_file or f"{output_file}.manifest.json"
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "converter": os.path.abspath(__file__),
        "input": {
            "path": os.path.abspath(input_file),
            "size_bytes": os.path.getsize(input_file),
            "sha256": _sha256_file(input_file),
            "rows": input_rows,
        },
        "conversion": {
            "base_date": str(base_date),
            "timestamp_formula": "base_date + (Day - 1 days) + Tmstamp",
            "target_rename": {"Patv": "power"},
            "sort_keys": ["TurbID", "date"],
            "duplicate_policy": "keep_last_per_TurbID_date",
        },
        "output": {
            "path": os.path.abspath(output_file),
            "size_bytes": os.path.getsize(output_file),
            "sha256": _sha256_file(output_file),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "turbine_count": int(df["TurbID"].nunique()),
            "date_min": df["date"].min().isoformat() if len(df) else None,
            "date_max": df["date"].max().isoformat() if len(df) else None,
        },
    }
    manifest_dir = os.path.dirname(os.path.abspath(manifest_file))
    os.makedirs(manifest_dir, exist_ok=True)
    with open(manifest_file, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Saved {len(df):,} rows and {len(df.columns)} columns to {output_file}")
    print(f"Conversion manifest: {manifest_file}")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="./datasets/sdwpf_245days_v1.csv",
        help="original SDWPF CSV",
    )
    parser.add_argument(
        "--output",
        default="./datasets/sdwpf_fixed.csv",
        help="converted CSV",
    )
    parser.add_argument("--base_date", default="2023-01-01")
    parser.add_argument(
        "--manifest",
        default=None,
        help="provenance JSON path; defaults to <output>.manifest.json",
    )
    args = parser.parse_args()
    convert_sdwpf(args.input, args.output, args.base_date, args.manifest)


if __name__ == "__main__":
    main()
