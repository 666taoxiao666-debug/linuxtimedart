"""Summarize existing paired Wiki validation artifacts without inference."""
import argparse
import csv
import json
from pathlib import Path


def summarize(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Diagnostic directory not found: {root}")
    paths = sorted(root.glob("runs/f*_s*/artifacts/diagnostic_summary.json"))
    if not paths:
        raise ValueError("No diagnostic_summary.json files found under runs/")
    rows, seen = [], set()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["evaluation_split"] != "val":
            raise ValueError(f"Expected validation diagnostics: {path}")
        identity = (data["fold"], data["seed"])
        if identity in seen:
            raise ValueError(f"Duplicate fold/seed: {identity}")
        seen.add(identity)
        metrics = data["all"]
        rows.append({"fold": identity[0], "seed": identity[1],
                     "windows": data["windows"],
                     "mae_on_kw": metrics["mae_on_kw"],
                     "mae_off_kw": metrics["mae_off_kw"],
                     "mae_gain_kw": metrics["mae_gain_kw"],
                     "null_false_intervention": data["actual_false_intervention_on_null"],
                     "null_max_delta_kw": data["null_max_prediction_delta_kw"]})
    lines = ["Paired Wiki diagnostic (validation; fixed checkpoint)",
             "Positive gain = lower MAE with event prompts enabled.",
             "Units: kW, inverse transformed without clipping.",
             "Not a retrained trend-only ablation; not evidence for gate training.",
             f"Completed artifacts found: {len(rows)} (not a job-status check)"]
    for row in rows:
        lines.append(f"fold={row['fold']} seed={row['seed']} windows={row['windows']} "
                     f"on={row['mae_on_kw']:.6f} off={row['mae_off_kw']:.6f} "
                     f"gain={row['mae_gain_kw']:+.6f} "
                     f"null_intervention={row['null_false_intervention']}")
    lines.append("Macro gain across available runs: "
                 f"{sum(r['mae_gain_kw'] for r in rows)/len(rows):+.6f} kW")
    lines.append("\nPer-run event/activation groups (overlap; do not sum groups):")
    for path in paths:
        group_path = path.parent / "group_metrics.csv"
        if not group_path.is_file():
            lines.append(f"MISSING: {group_path}")
            continue
        lines.append(path.parent.parent.name)
        with group_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                lines.append(f"  {row['group']}: windows={row['windows']} "
                             f"gain_kw={row['mae_gain_kw']}")
        single = path.parent / "single_event_metrics.csv"
        if single.is_file():
            lines.append("  SINGLE EVENT: positive gain favors retaining this event")
            with single.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    lines.append(f"    {row['factor']} {row['group']}: n={row['windows']} "
                                 f"gain_kw={row['mae_gain_kw']} change_kw={row['mean_abs_prediction_change_kw']}")
    return rows, "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", required=True)
    args = parser.parse_args()
    rows, report = summarize(args.diagnostic_dir)
    root = Path(args.diagnostic_dir)
    (root / "diagnostic_report.txt").write_text(report, encoding="utf-8")
    with (root / "diagnostic_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(report)
    print(f"[WIKI-DIAG] Summary saved to: {root}")


if __name__ == "__main__":
    main()
