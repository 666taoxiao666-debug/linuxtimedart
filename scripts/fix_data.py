"""Convert the original SDWPF table into the format used by Dataset_SDWPF."""

import argparse
import os

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


def convert_sdwpf(input_file, output_file, base_date="2023-01-01"):
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    try:
        df = pd.read_csv(input_file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding="gbk")

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
    print(f"Saved {len(df):,} rows and {len(df.columns)} columns to {output_file}")
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
    args = parser.parse_args()
    convert_sdwpf(args.input, args.output, args.base_date)


if __name__ == "__main__":
    main()
