#!/usr/bin/env python3
"""v28 report: recompute every metric, comparison and rule from the REGISTERED test arrays (torch-free); refuse to
report if they differ from the registered decision. Both cohorts in every table: normal (primary, conditional) and all
(deployment). Never trains, never selects, never reads unregistered data."""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils import kelmarsh_multiyear_benchmark as b  # noqa: E402
from utils.kelmarsh_backbone_information import select_trace_segment  # noqa: E402

REPORT_SOURCES = ("scripts/report_kelmarsh_multiyear_benchmark.py", "utils/kelmarsh_multiyear_benchmark.py",
                  "utils/kelmarsh_backbone_information.py", "utils/paired_forecast_statistics.py")
TOLERANCE = 1e-9
ARMS = b.PERSISTENCE_ARMS + b.TRAINED_ARMS


def table(rows, columns, digits=3):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            cells.append(f"{value:.{digits}f}" if isinstance(value, float) and math.isfinite(value) else
                         ("undefined" if value is None else str(value)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def check_against_registered(fresh, registered):
    def compare(a, c, path):
        if isinstance(a, dict) and isinstance(c, dict) and set(a) == set(c):
            for key in a:
                compare(a[key], c[key], f"{path}/{key}")
            return
        if isinstance(a, list) and isinstance(c, list) and len(a) == len(c):
            for i, (left, right) in enumerate(zip(a, c)):
                compare(left, right, f"{path}/{i}")
            return
        if isinstance(a, (int, float)) and isinstance(c, (int, float)):
            if math.isfinite(a) and math.isfinite(c) and abs(a - c) <= TOLERANCE:
                return
        elif a == c:
            return
        raise ValueError(f"Recomputed {path} disagrees with the registered decision")
    compare(b.safe(fresh), {k: v for k, v in registered.items() if k != "utc"}, "decision")


def figures(fresh, folder):
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    x = np.arange(len(ARMS))
    for ax, key, label in zip(axes, ("mae", "rmse"), ("MAE (kW)", "RMSE (kW)")):
        for offset, cohort in ((-.2, "normal"), (.2, "all")):
            values = [fresh["metrics"][cohort][arm][key] for arm in ARMS]
            ax.bar(x + offset, values, .4, label=f"test {cohort}")
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("_", "\n") for a in ARMS], fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("v28 test 2021 (single evaluation): normal = conditional primary cohort, all = deployment cohort", fontsize=9)
    fig.tight_layout()
    fig.savefig(folder / "test_metrics.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, cohort in zip(axes, b.COHORTS):
        for arm in ARMS:
            ax.plot([1, 2, 3, 4], fresh["metrics"][cohort][arm]["lead_mae"], marker="o", label=arm,
                    ls="--" if arm.startswith("Persistence") else "-")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("lead (hours)")
        ax.set_title(f"test {cohort}: MAE by lead (kW)", fontsize=10)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(folder / "lead_mae.png", dpi=130)
    plt.close(fig)


def forecast_trace(directory, p, choice):
    with np.load(directory / "evaluation/test_cohort.npz", allow_pickle=False) as saved:
        cohort = {k: saved[k] for k in saved.files}
    with np.load(directory / f"evaluation/{choice}_s{p['seeds'][0]}.npz", allow_pickle=False) as saved:
        prediction = saved["prediction"]
    selection = select_trace_segment(cohort["times"][:, 0] - b.HOUR, cohort["turbines"], stride_hours=p["data"]["stride"])
    rows = selection["rows"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharey=True)
    for ax, lead in zip(axes, (0, 3)):
        x = cohort["times"][rows, lead]
        ax.plot(x, cohort["truth"][rows, lead], label="Observed", color="black")
        ax.plot(x, prediction[rows, lead], label=f"{choice}, seed {p['seeds'][0]}")
        ax.plot(x, cohort["PersistenceLast10"][rows, lead], label="PersistenceLast10", ls="--")
        ax.set_title(f"Turbine {selection['turbine']}, lead {lead + 1} h; first eligible 72 h of issue times", fontsize=9)
        ax.set_ylabel("Power (kW)")
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(directory / "figures/forecast_trace.png", dpi=130)
    plt.close(fig)
    b.write_json(directory / "forecast_trace_selection.json", selection)


def build_report(directory):
    directory = Path(directory)
    p = b.read_json(directory / "protocol.json")["protocol"]
    training = b.verify_training_completed(directory)
    freeze = b.read_json(directory / "selection_freeze.json")
    registered = b.read_json(directory / "decision.json")
    required = ["selection_freeze.json", "decision.json", "evaluation/test_cohort.npz"]
    required += [f"evaluation/{arm}_s{seed}.npz" for arm in b.TRAINED_ARMS for seed in p["seeds"]]
    for name in required:
        if training["hashes"].get(name) != b.file_sha256(directory / name):
            raise ValueError(f"Registered artifact changed or missing: {name}")
    fresh = b.evaluate_registered(directory, p, freeze["validation_choice"]["arm"])
    check_against_registered(fresh, registered)
    rows = []
    for cohort in b.COHORTS:
        for arm in ARMS:
            m = fresh["metrics"][cohort][arm]
            rows.append({"cohort": cohort, "arm": arm, "seeds": m["seeds"], "mae": m["mae"], "mse": m["mse"], "rmse": m["rmse"],
                         "r2": m["r2"], "mbe": m["mbe"], "nmae_pct_rated": m["nmae_pct_rated"],
                         "nrmse_pct_rated": m["nrmse_pct_rated"],
                         **{k: v for k, v in m.items() if "skill_pct" in k or k.startswith("hit_")},
                         **{f"lead{h + 1}_mae": v for h, v in enumerate(m["lead_mae"])},
                         **{f"lead{h + 1}_mse": v for h, v in enumerate(m["lead_mse"])},
                         **{f"lead{h + 1}_rmse": v for h, v in enumerate(m["lead_rmse"])},
                         **{f"lead{h + 1}_r2": v for h, v in enumerate(m["lead_r2"])}})
    pd.DataFrame(rows).to_csv(directory / "test_metrics.csv", index=False)
    comparison_rows = []
    for cohort in b.COHORTS:
        for name, c in sorted(fresh["comparisons"][cohort].items()):
            a, r = c["absolute_error"], c["rmse_difference"]
            comparison_rows.append({"cohort": cohort, "comparison": name, "candidate": c["candidate"], "reference": c["reference"],
                                    "mae_gain_kw": a["gain"], "mae_ci95_low": a["ci95_low"], "mae_ci95_high": a["ci95_high"],
                                    "nw_p": a["newey_west"]["p_value_two_sided"], "holm_p": c["holm_p"],
                                    "rmse_gain_kw": r["rmse_gain_kw"], "rmse_ci95_low": r["ci95_low_kw"],
                                    "rmse_ci95_high": r["ci95_high_kw"]})
    pd.DataFrame(comparison_rows).to_csv(directory / "test_comparisons.csv", index=False)
    figures(fresh, directory / "figures")
    forecast_trace(directory, p, freeze["validation_choice"]["arm"])
    result = {"protocol_id": p["protocol_id"], "rules": fresh["rules"], "validation_choice": freeze["validation_choice"],
              "windows": fresh["windows"], "normal_share_pct": fresh["normal_share_pct"],
              "scope": "single 2021 test evaluation; normal = ex-post conditional cohort; all = deployment cohort; "
                       "development years were inspected in earlier protocols; 2022-2024 sealed",
              "independent_recomputation": "all metrics, comparisons and rules recomputed from registered arrays"}
    b.write_json(directory / "result.json", result)
    write_markdown(directory, p, fresh, freeze, rows, comparison_rows)
    outputs = ["test_metrics.csv", "test_comparisons.csv", "result.json", "report.md", "figures/test_metrics.png", "figures/lead_mae.png",
               "figures/forecast_trace.png", "forecast_trace_selection.json"]
    b.write_json(directory / "report_manifest.json", {
        "report_sources": {n: b.file_sha256(ROOT / n) for n in REPORT_SOURCES},
        "inputs": {"training_completed.json": b.file_sha256(directory / "training_completed.json")},
        "outputs": {n: b.file_sha256(directory / n) for n in outputs},
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    return result


def write_markdown(directory, p, fresh, freeze, rows, comparison_rows):
    validation = [{"arm": r["arm"], "seed": r["seed"], "best_epoch": r["best_epoch"], "train_cohort": r["train_cohort"],
                   **{f"val_{k}": v for k, v in r["validation"].items()}} for r in freeze["records"]]
    lines = [f"# v28 multi-year Kelmarsh base benchmark report ({dt.date.today().isoformat()})", "",
             f"**Rules:** {fresh['rules']}", "",
             f"Validation-chosen configuration (2020, normal, frozen before test): **{freeze['validation_choice']['arm']}**", "",
             f"Test 2021 windows: all {fresh['windows']['all']}, normal {fresh['windows']['normal']} "
             f"({fresh['normal_share_pct']:.2f}%). Every arm uses the same windows. R² is not a percentage accuracy; "
             "the normal cohort uses the target period's operating state and is conditional accuracy only.", "",
             p["rules"].get("attribution_limits", ""), "",
             "Skill is reported against both Persistence anchors; capacity hit rates are tolerance statistics, not classification accuracy. "
             "Point RMSE is averaged across seeds; paired RMSE intervals use the root of seed-averaged squared errors. "
             "Eight consecutive issue-time blocks span 24 clock hours only without gaps.", "",
             "## Test metrics (seed-mean over 3 seeds for trained arms)", "",
             table(rows, ["cohort", "arm", "mae", "rmse", "r2", "mbe", "nmae_pct_rated", "lead1_mae", "lead4_mae", "lead1_r2", "lead4_r2"]), "",
             "## Predeclared comparisons (positive gain favours the candidate; Holm over C1-C5 within each cohort)", "",
             table(comparison_rows, ["cohort", "comparison", "candidate", "reference", "mae_gain_kw", "mae_ci95_low", "mae_ci95_high",
                                     "nw_p", "holm_p", "rmse_gain_kw", "rmse_ci95_low", "rmse_ci95_high"]), "",
             "## Validation (2020) selected states", "",
             table(validation, ["arm", "seed", "best_epoch", "train_cohort", "val_mae", "val_rmse"]), "",
             "![test metrics](figures/test_metrics.png)", "", "![lead MAE](figures/lead_mae.png)", "",
             "![first eligible forecast segment, not selected by error](figures/forecast_trace.png)", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(build_report(args.directory)["rules"])


if __name__ == "__main__":
    main()
