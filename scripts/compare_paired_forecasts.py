#!/usr/bin/env python3
import argparse
import json

from utils.paired_forecast_statistics import compare_report_directories


def main():
    parser = argparse.ArgumentParser(
        description="Paired time-block bootstrap and approximate DM comparison"
    )
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--block-length", type=int, default=12)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    report = compare_report_directories(
        args.candidate_dir,
        args.reference_dir,
        args.output_dir,
        block_length=args.block_length,
        replicates=args.replicates,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
