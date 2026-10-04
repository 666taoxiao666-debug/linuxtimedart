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
    paired_root = root / "evidence" / "paired"
    for report_path in sorted(paired_root.glob("*/*/paired_statistics.json")):
        value = json.loads(report_path.read_text(encoding="utf-8"))
        method = report_path.parent.parent.name.replace("_vs_trend", "")
        run = report_path.parent.name
        fold_text, seed_text = run.split("_")
        row = {
            "method": method,
            "fold": int(fold_text[1:]),
            "seed": int(seed_text[1:]),
            "candidate_mae_kw": value["candidate_mae_kw"],
            "trend_mae_kw": value["reference_mae_kw"],
            "gain_kw": value["paired_gain_kw"],
            "gain_pct": value["paired_gain_pct"],
            "window_win_pct": value["candidate_win_window_pct"],
            "bootstrap_low_kw": value["time_block_bootstrap"]["ci95_low_kw"],
            "bootstrap_high_kw": value["time_block_bootstrap"]["ci95_high_kw"],
            "dm_p_value": value["dm_newey_west"]["p_value_two_sided_normal"],
        }
        intervention = root / "evidence" / "reports" / "wiki" / run / "diagnostic" / "intervention_summary.csv"
        if method == "wiki" and intervention.is_file():
            values = pd.read_csv(intervention).iloc[0]
            for name in (
                "candidate_coverage_pct",
                "intervention_coverage_pct",
                "abstention_pct",
                "selected_gain_kw",
                "selected_harm_window_pct",
            ):
                row[name] = values.get(name)
        rows.append(row)
    if not rows:
        raise SystemExit("No paired evidence reports found")
    detail = pd.DataFrame(rows).sort_values(["method", "fold", "seed"])
    detail.to_csv(root / "evidence" / "paired_evidence.csv", index=False)
    aggregate = detail.groupby("method", as_index=False).agg(
        runs=("gain_kw", "count"),
        candidate_mae_mean_kw=("candidate_mae_kw", "mean"),
        trend_mae_mean_kw=("trend_mae_kw", "mean"),
        gain_mean_kw=("gain_kw", "mean"),
        gain_sd_kw=("gain_kw", "std"),
        positive_runs=("gain_kw", lambda value: int((value > 0).sum())),
        intervention_coverage_mean_pct=("intervention_coverage_pct", "mean"),
        abstention_mean_pct=("abstention_pct", "mean"),
        selected_gain_mean_kw=("selected_gain_kw", "mean"),
        selected_harm_mean_pct=("selected_harm_window_pct", "mean"),
    )
    aggregate.to_csv(root / "evidence" / "evidence_summary.csv", index=False)
    (root / "evidence" / "evidence_summary.txt").write_text(
        aggregate.to_string(index=False) + "\n", encoding="utf-8"
    )
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
