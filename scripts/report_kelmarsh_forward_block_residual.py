#!/usr/bin/env python3
"""v27 report: independent recomputation of the forward-block diagnostic from REGISTERED evaluation arrays only.

Torch-free. Recomputes every step record, the pooled paired statistics and the predeclared decision, and refuses to
report if they differ from the registered decision. Adds descriptive diagnostics (corrections, seed agreement, best vs
last epoch, power regimes, turbines). Never trains, never selects anything, never touches a selection/outer split.
"""
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

from utils import kelmarsh_forward_block_residual as v27  # noqa: E402

REPORT_SOURCES = ("scripts/report_kelmarsh_forward_block_residual.py", "utils/kelmarsh_forward_block_residual.py",
                  "utils/kelmarsh_backbone_information.py", "utils/paired_forecast_statistics.py")
TOLERANCE = 1e-9


def load_step(directory, k, training):
    rel = f"steps/k{k}/evaluation_predictions.npz"
    if rel not in training["hashes"] or v27.file_sha256(directory / rel) != training["hashes"][rel]:
        raise ValueError(f"Registered evaluation arrays of step {k} changed or missing")
    with np.load(directory / rel, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


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


def step_rows(k, record, arrays):
    rows = []
    for name, metrics in (("Persistence", record["persistence"]), ("DLinear anchor", record["anchor"]),
                          ("AnchoredFull 5-seed mean", record["full_seed_mean"]),
                          ("AnchoredPowerOnly 5-seed mean", record["power_seed_mean"]),
                          ("AnchoredFull single-seed mean", record["full_single_mean"]),
                          ("AnchoredPowerOnly single-seed mean", record["power_single_mean"])):
        rows.append({"step": k, "forecast": name, "mae": metrics["mae"], "rmse": metrics["rmse"], "mbe": metrics["mbe"],
                     "median_error": metrics.get("median_error"),
                     "lead1_mae": metrics["lead_mae"][0] if "lead_mae" in metrics else None,
                     "lead4_mae": metrics["lead_mae"][-1] if "lead_mae" in metrics else None})
    for tag, label in (("full_last", "AnchoredFull last-epoch 5-seed mean (diagnostic)"),
                       ("power_last", "AnchoredPowerOnly last-epoch 5-seed mean (diagnostic)")):
        m = v27.basic_metrics(np.asarray(arrays[tag]).mean(0), arrays["truth"])
        rows.append({"step": k, "forecast": label, "mae": m["mae"], "rmse": m["rmse"], "mbe": m["mbe"],
                     "median_error": m["median_error"], "lead1_mae": m["lead_mae"][0], "lead4_mae": m["lead_mae"][-1]})
    return rows


def figure_steps(records, steps, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    width = .25
    for i, (key, label) in enumerate((("full_seed_mean", "Full 5-seed mean"), ("full_single_mean", "Full single seed (mean)"),
                                      ("power_seed_mean", "PowerOnly 5-seed mean"))):
        mae = [r["anchor"]["mae"] - r[key]["mae"] for r in records]
        rmse = [r["anchor"]["rmse"] - r[key]["rmse"] for r in records]
        axes[0].bar(np.arange(len(steps)) + (i - 1) * width, mae, width, label=label)
        axes[1].bar(np.arange(len(steps)) + (i - 1) * width, rmse, width, label=label)
    for ax, title in zip(axes, ("MAE gain vs step anchor (kW)", "RMSE gain vs step anchor (kW)")):
        ax.axhline(0, color="black", lw=.8)
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([f"block {k}" for k in steps])
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("v27 forward blocks inside the fold-0 FIT period (training-side diagnostic; positive = better than anchor)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_leads(records, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, label in (("full_seed_mean", "Full 5-seed mean"), ("full_single_mean", None), ("power_seed_mean", "PowerOnly 5-seed mean")):
        if key == "full_single_mean":
            continue
        gains = np.mean([np.asarray(r["anchor"]["lead_mae"]) - np.asarray(r[key]["lead_mae"]) for r in records], axis=0)
        ax.plot([1, 2, 3, 4], gains, marker="o", label=label)
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("lead (hours)")
    ax.set_ylabel("MAE gain vs anchor (kW), mean over steps")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_report(directory):
    directory = Path(directory)
    protocol = v27.read_json(directory / "protocol.json")["protocol"]
    training = v27.verify_training_completed(directory)
    registered = v27.read_json(directory / "decision.json")
    steps = list(protocol["timeline"]["eval_blocks"])
    arrays_list = [load_step(directory, k, training) for k in steps]
    for k, arrays in zip(steps, arrays_list):
        v27.assert_aligned({"seeds": np.asarray(protocol["residual_seeds"])}, {"seeds": arrays["seeds"]}, keys=("seeds",),
                           label=f"step {k} seeds")
    records = [v27.step_record(a) for a in arrays_list]
    pooled = v27.pooled_statistics(arrays_list, protocol["evaluation"]["pooled_statistics"])
    decision = v27.decide(records, pooled, protocol)
    old = registered["decision"]
    if v27.canonical_sha256(decision) != v27.canonical_sha256(old) \
            or v27.canonical_sha256(pooled) != v27.canonical_sha256(registered["pooled"]) \
            or ("records" in registered and v27.canonical_sha256(records) != v27.canonical_sha256(registered["records"])):
        raise ValueError("Independent recomputation disagrees with the registered decision")
    diagnostics = {str(k): {"corrections": v27.correction_diagnostics(a),
                            "regimes": v27.regime_diagnostics(a, protocol["rated_power_kw"]),
                            "selected_epochs": {"full": a["full_selected_epochs"].tolist(),
                                                "power": a["power_selected_epochs"].tolist()}}
                   for k, a in zip(steps, arrays_list)}
    rows = [row for k, r, a in zip(steps, records, arrays_list) for row in step_rows(k, r, a)]
    pd.DataFrame(rows).to_csv(directory / "metrics_by_step.csv", index=False)
    v27.write_json(directory / "diagnostics.json", diagnostics)
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    figure_steps(records, steps, figures / "step_gains.png")
    figure_leads(records, figures / "lead_gains.png")
    result = {"protocol_id": protocol["protocol_id"], "decision": decision, "pooled": pooled,
              "scope": "training-side diagnostic inside the v20-v26 fold-0 FIT period; no original selection/outer/sealed numeric rows; historically developed prefix, not independent confirmation",
              "independent_recomputation": "step records, pooled statistics and decision recomputed from registered arrays"}
    v27.write_json(directory / "result.json", result)
    write_markdown(directory, protocol, decision, pooled, rows, diagnostics, steps)
    outputs = ["metrics_by_step.csv", "diagnostics.json", "result.json", "report.md",
               "figures/step_gains.png", "figures/lead_gains.png"]
    v27.write_json(directory / "report_manifest.json", {
        "report_sources": {name: v27.file_sha256(ROOT / name) for name in REPORT_SOURCES},
        "inputs": {"training_completed.json": v27.file_sha256(directory / "training_completed.json")},
        "outputs": {name: v27.file_sha256(directory / name) for name in outputs},
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    return result


def write_markdown(directory, protocol, decision, pooled, rows, diagnostics, steps):
    fa, fp = pooled["full_mean_vs_anchor"], pooled["full_mean_vs_power_mean"]
    blocks = v27.read_json(directory / "blocks_manifest.json")
    lines = [f"# {protocol['protocol_id']} ({dt.date.today().isoformat()})", "",
             f"**Outcome: {decision['outcome']}** — flags {decision['flags']}, counts {decision['counts']} "
             f"(need {decision['steps_required']} of {decision['steps']} steps). Auxiliary claim allowed: {decision['auxiliary_claim_allowed']}.", "",
             "Scope: only the registered v20-v26 fold-0 FIT period (portions were used by earlier development), cut into 6 blocks; each "
             "step refits scaler, DLinear anchor and residuals on the past and scores the next block. Summer only. This is a "
             "training-side diagnostic: it can authorise at most the design of one later confirmation protocol. R2 is error "
             "cancellation, not proof that seed variance is the primary cause. R4 tests harm to the anchor, not anchor optimality. "
             "Intervals condition on this training recipe and four dependent expanding-history steps; they do not establish seasonal transfer.", "",
             f"Edges (UTC hour-end): {blocks['edges_utc']}", "",
             "## Pooled over all evaluation windows (issue-time clusters)", "",
             f"- Full 5-seed mean vs anchor: MAE gain {fa['absolute_error']['gain']:+.3f} kW "
             f"[{fa['absolute_error']['ci95_low']:+.3f}, {fa['absolute_error']['ci95_high']:+.3f}], NW p="
             f"{fa['absolute_error']['newey_west']['p_value_two_sided']:.4f}; RMSE gain {fa['rmse_difference']['rmse_gain_kw']:+.3f} kW "
             f"[{fa['rmse_difference']['ci95_low_kw']:+.3f}, {fa['rmse_difference']['ci95_high_kw']:+.3f}]; lead-1 MAE gain "
             f"{fa['lead1_mae_gain_kw']:+.3f} kW",
             f"- Full 5-seed mean vs PowerOnly 5-seed mean: MAE gain {fp['absolute_error']['gain']:+.3f} kW "
             f"[{fp['absolute_error']['ci95_low']:+.3f}, {fp['absolute_error']['ci95_high']:+.3f}]",
             f"- windows {pooled['windows']}, issue times {pooled['issue_times']}", "",
             "## Per step", "", table(rows, ["step", "forecast", "mae", "rmse", "mbe", "median_error", "lead1_mae", "lead4_mae"]), "",
             "## Corrections and seed agreement (descriptive)", ""]
    for k in steps:
        c = diagnostics[str(k)]["corrections"]
        lines.append(f"- block {k}: Full cross-seed corr {c['full']['cross_seed_correction_corr_mean']}, corr(mean correction, "
                     f"anchor error) by lead {[None if v is None else round(v, 3) for v in c['full']['corr_seed_mean_correction_vs_anchor_error_by_lead']]}, "
                     f"mean correction {[round(v, 2) for v in c['full']['seed_mean_correction_kw_by_lead']]} kW; PowerOnly corr by lead "
                     f"{[None if v is None else round(v, 3) for v in c['power']['corr_seed_mean_correction_vs_anchor_error_by_lead']]}; "
                     f"selected epochs Full {diagnostics[str(k)]['selected_epochs']['full']}")
    lines += ["", "Regime and turbine tables with support are in diagnostics.json; they are descriptive and must not be used "
                  "to choose regimes, leads, turbines, K, lambda or epochs.", "",
              "![step gains](figures/step_gains.png)", "", "![lead gains](figures/lead_gains.png)", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(build_report(args.directory)["decision"]["outcome"])


if __name__ == "__main__":
    main()
