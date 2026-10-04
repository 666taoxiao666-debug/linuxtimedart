#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for method_dir in sorted((root / "runs" / "deep").glob("*")):
        for run_dir in sorted(method_dir.glob("f*_s*")):
            metrics_path = run_dir / "artifacts" / "metrics.json"
            env_path = run_dir / "baseline.env"
            if not metrics_path.is_file() or not env_path.is_file():
                continue
            env = {}
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    env[key] = value
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["original"]
            rows.append({
                "family": "deep",
                "method": env["MODEL"],
                "fold": int(env["FOLD"]),
                "seed": int(env["SEED"]),
                "mae_kw": metrics["mae"],
                "rmse_kw": metrics["rmse"],
                "persistence_mae_kw": metrics.get("persistence_mae"),
                "mae_skill_vs_persistence_pct": metrics.get("mae_skill_vs_persistence_pct"),
                "report_dir": str((run_dir / "artifacts").resolve()),
            })
    for csv_path in sorted((root / "runs" / "classical").glob("f*_s*/artifacts/metrics_summary.csv")):
        run_dir = csv_path.parent.parent
        parts = run_dir.name.split("_")
        fold, seed = int(parts[0][1:]), int(parts[1][1:])
        table = pd.read_csv(csv_path)
        for _, row in table.iterrows():
            rows.append({
                "family": "classical",
                "method": row["method"],
                "fold": fold,
                "seed": seed,
                "mae_kw": row["mae"],
                "rmse_kw": row["rmse"],
                "persistence_mae_kw": None,
                "mae_skill_vs_persistence_pct": row.get("mae_skill_vs_persistence_pct"),
                "report_dir": str(run_dir.resolve()),
            })
    if not rows:
        raise SystemExit("No completed frozen baseline results found")
    detail = pd.DataFrame(rows).sort_values(["family", "method", "fold", "seed"])
    detail.to_csv(root / "benchmark_runs.csv", index=False)
    aggregate = detail.groupby(["family", "method"], as_index=False).agg(
        runs=("mae_kw", "count"),
        mae_mean_kw=("mae_kw", "mean"),
        mae_sd_kw=("mae_kw", "std"),
        rmse_mean_kw=("rmse_kw", "mean"),
        skill_mean_pct=("mae_skill_vs_persistence_pct", "mean"),
    )
    aggregate.to_csv(root / "benchmark_summary.csv", index=False)
    (root / "benchmark_summary.txt").write_text(aggregate.to_string(index=False) + "\n", encoding="utf-8")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
