#!/usr/bin/env python3
"""Explain a completed frozen transport pilot; no fitting or parameter changes."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tune_sdwpf_accuracy import read_json, write_json, sha256
from scripts.train_sdwpf_knowledge_transport import expert_api_for_type


def calibration_summary(model, policy):
    summary = []
    for index, event_id in enumerate(model["contract"]["event_ids"] + ["composition"]):
        rows = []
        for h in range(model["contract"]["pred_len"]):
            row = policy["entries"][f"event:0:{h}:{index}"]
            reasons = []
            for number, block in enumerate(row["blocks"], 1):
                if block["windows"] < policy["min_windows"] or block["turbines"] < policy["min_turbines"]:
                    reasons.append(f"block{number}:insufficient_support")
                if block["turbine_balanced_gain_kw"] is not None:
                    if block["turbine_balanced_gain_kw"] <= 0:
                        reasons.append(f"block{number}:nonpositive_MAE_gain")
                    if block["turbine_balanced_mse_gain"] <= 0:
                        reasons.append(f"block{number}:nonpositive_MSE_gain")
                    if block["positive_turbine_fraction"] < policy["min_positive_fraction"]:
                        reasons.append(f"block{number}:too_few_profitable_turbines")
            rows.append(dict(horizon=h+1, accepted=row["accepted"], reasons=reasons, blocks=row["blocks"]))
        expert = model["experts"][index] if index < len(model["experts"]) else None
        summary.append(dict(event_id=event_id, expert_active=expert["active"] if expert else None,
            expert_fit_windows=expert["fit_windows"] if expert else None, event_level_horizons=rows))
    return summary


def summarize(directory):
    completed = read_json(directory/"completed.json")
    for name, expected in completed.items():
        if sha256(directory/name) != expected:
            raise ValueError("Completed transport artifact changed: "+name)
    model, policy, result = (read_json(directory/name) for name in
        ("expert_model.json", "utility_policy.json", "result.json"))
    expert_api_for_type(model["expert_type"]).validate_identity(model, model["contract"], policy)
    audit = dict(source_completed_sha256=sha256(directory/"completed.json"),
        expert_bundle_id=model["expert_bundle_id"], policy_id=policy["policy_id"],
        supported_cells=sum(row["supported"] for row in policy["entries"].values()),
        accepted_cells=sum(row["accepted"] for row in policy["entries"].values()),
        calibration=calibration_summary(model, policy), no_fitting_performed=True,
        scope="original-train calibration evidence; not outer validation or test")
    write_json(directory/"calibration_audit.json", audit)
    lines = ["EVIDENCE_STATUS=exploratory_original_train_only",
        "EXPERT_BUNDLE_ID="+model["expert_bundle_id"], "POLICY_ID="+policy["policy_id"],
        f"SUPPORTED_CELLS={audit['supported_cells']}", f"ACCEPTED_CELLS={audit['accepted_cells']}"]
    for name, block in result["checks"].items():
        values = block["physical_transport"]
        m = values["metrics"]
        lines.append(f"{name}: windows={block['windows']} turbines={block['turbines']} "
            f"MAE={m['mae']:.6f} MSE={m['mse']:.6f} RMSE={m['rmse']:.6f} R2={m['r2']:.6f} "
            f"MAE_Skill={m['mae_skill_pct']:.6f}% RMSE_Skill={m['rmse_skill_pct']:.6f}% "
            f"gain_vs_same_base={values['gain_mae_kw']:.6f}kW "
            f"intervention={values['coverage_window_pct']:.4f}% abstention={values['abstention_pct']:.4f}% "
            f"harm={values['harm_window_pct']}")
    for event in audit["calibration"]:
        steps = event["event_level_horizons"]
        lines.append(f"{event['event_id']}: active={event['expert_active']} "
            f"fit_windows={event['expert_fit_windows']} "
            f"accepted_event_horizons={sum(s['accepted'] for s in steps)}")
        for number in range(2):
            blocks = [step["blocks"][number] for step in steps]
            gains = [b["turbine_balanced_gain_kw"] for b in blocks if b["turbine_balanced_gain_kw"] is not None]
            gain_range = "n/a" if not gains else f"{min(gains):.6f}..{max(gains):.6f}"
            counts = [b["windows"] for b in blocks]
            lines.append(f"  calibration_block{number+1}: supported_windows_by_horizon="
                f"{min(counts)}..{max(counts)} turbine_balanced_MAE_gain_kw={gain_range}")
    (directory/"summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path)
    args = parser.parse_args()
    directory = args.result_dir or Path((ROOT/"outputs/logs/SDWPF/knowledge_transport_latest.txt").read_text().strip())
    summarize(directory.resolve())


if __name__ == "__main__":
    main()
