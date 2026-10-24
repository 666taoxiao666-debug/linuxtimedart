#!/usr/bin/env python3
"""v25 A4 report: independent recomputation, paired statistics and figures from REGISTERED predictions only.

Torch-free by design: it never trains, never re-selects and never reads unregistered outer data. A report
failure is a report failure only (launcher --report-only re-runs it without retraining).
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils import kelmarsh_backbone_information as v25  # noqa: E402
from utils.paired_forecast_statistics import paired_forecast_statistics  # noqa: E402

REPORT_SOURCES = ("scripts/report_kelmarsh_backbone_information.py", "utils/kelmarsh_backbone_information.py",
                  "utils/paired_forecast_statistics.py")
ARM_ORDER = ("Persistence", "DLinear", "TSMixerFull", "TSMixerPowerOnly")
COLORS = {"Persistence": "#7f7f7f", "DLinear": "#1f77b4", "TSMixerFull": "#d62728", "TSMixerPowerOnly": "#ff7f0e",
          "v21_PowerAnchoredTimeDART_s2024": "#2ca02c", "truth": "#000000"}
RECOMPUTE_TOLERANCE = 1e-6


def resolve(path, root):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def load_npz(path, expected_sha=None):
    if expected_sha is not None and v25.file_sha256(path) != expected_sha:
        raise ValueError(f"Registered prediction file changed: {path}")
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def lead_rows(prediction, truth, persistence, capacity):
    return [{"lead_hours": h + 1, **v25.point_metrics(prediction[:, h:h + 1], truth[:, h:h + 1],
                                                       persistence[:, h:h + 1], capacity)} for h in range(truth.shape[1])]


# --------------------------------------------------------------------------- collection
def collect(directory, protocol, a0, reference_root):
    """Return {split: {(fold, seed, arm): arrays}} for every registered prediction."""
    folds = protocol["folds"]
    sets = {"inner_selection": {}, "outer_validation": {}}
    notes = []
    for trained_path in sorted((directory / "runs").glob("f*_s*/*/trained.json")):
        trained = v25.read_json(trained_path)
        folder = trained_path.parent
        fold, seed = int(folder.parent.name.split("_")[0][1:]), int(folder.parent.name.split("_")[1][1:])
        arrays = load_npz(folder / "selection_predictions.npz", trained["artifacts"]["selection_predictions.npz"])
        sets["inner_selection"][(fold, seed, trained["arm"])] = arrays | {"source": "v25 fit (best.pt)"}
    for fold in folds:
        path = directory / "reference_replay" / f"f{fold}_s{protocol['reference']['seed']}_DLinear_selection_predictions.npz"
        if path.exists():
            sets["inner_selection"][(fold, protocol["reference"]["seed"], "DLinear")] = load_npz(path) | {
                "source": "v20 checkpoint CPU replay (gates used the registered history values)"}
    for cache in sorted((directory / "runs").glob("f*_s*/DLinear/reused_cache.json")):
        record = v25.read_json(cache)
        fold, seed = (int(part[1:]) for part in cache.parent.parent.name.split("_"))
        arrays = load_npz(cache.parent / "selection_predictions.npz", record["selection_predictions_sha256"])
        sets["inner_selection"][(fold, seed, "DLinear")] = arrays | {"source": "exact-contract cache CPU replay"}
        notes.append(f"{cache.parent.parent.name} DLinear reused from cache; selection predictions replayed without training")
    index_path = directory / "outer_index.json"
    if index_path.exists():
        for key, entry in v25.read_json(index_path)["entries"].items():
            arrays = load_npz(resolve(entry["path"], reference_root), entry["sha256"])
            if entry["arm"] == "Persistence":
                continue
            sets["outer_validation"][(entry["fold"], entry["seed"], entry["arm"])] = arrays | {"source": entry["source"]}
    return sets, notes


def cohort_of(entries, fold):
    for (f, _, _), arrays in sorted(entries.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        if f == fold:
            return arrays
    return None


def check_alignment(entries):
    for fold in sorted({k[0] for k in entries}):
        base = cohort_of(entries, fold)
        for key, arrays in entries.items():
            if key[0] == fold:
                v25.assert_aligned(base, arrays, label=f"f{key[0]} s{key[1]} {key[2]}")


# --------------------------------------------------------------------------- metrics tables
def metric_rows(split, entries, capacity):
    rows, leads = [], []
    for fold in sorted({k[0] for k in entries}):
        base = cohort_of(entries, fold)
        unit = {"split": split, "fold": fold, "seed": None, "arm": "Persistence", "source": "last observed power"}
        rows.append(unit | v25.point_metrics(base["persistence"], base["truth"], base["persistence"], capacity))
        leads += [unit | row for row in lead_rows(base["persistence"], base["truth"], base["persistence"], capacity)]
    for (fold, seed, arm), arrays in sorted(entries.items()):
        unit = {"split": split, "fold": fold, "seed": seed, "arm": arm, "source": arrays["source"]}
        rows.append(unit | v25.point_metrics(arrays["prediction"], arrays["truth"], arrays["persistence"], capacity))
        leads += [unit | row for row in lead_rows(arrays["prediction"], arrays["truth"], arrays["persistence"], capacity)]
    return rows, leads


def macro_table(frame, seeds_by_arm):
    """Equal mean over fold-seed units (Persistence: one unit per fold). Skills are per-unit first; the
    macro error-ratio skill is reported separately and named so."""
    keys = ["mae", "mse", "rmse", "r2", "mbe", "mae_skill_pct", "rmse_skill_pct", "within_5pct_capacity_pct",
            "within_10pct_capacity_pct"]
    out = {}
    persistence = frame[frame.arm == "Persistence"]
    for arm, group in frame.groupby("arm", sort=False):
        record = {k: (float(group[k].astype(float).mean()) if group[k].notna().all() else None) for k in keys}
        record["units"] = int(len(group))
        record["seeds"] = seeds_by_arm.get(arm)
        pers_mae = float(persistence.set_index("fold").loc[sorted(group.fold.unique()), "mae"].mean())
        pers_rmse = float(persistence.set_index("fold").loc[sorted(group.fold.unique()), "rmse"].mean())
        record["macro_error_ratio_skill_mae_pct"] = 100 * (1 - record["mae"] / pers_mae)
        record["macro_error_ratio_skill_rmse_pct"] = 100 * (1 - record["rmse"] / pers_rmse)
        record["r2_note"] = "mean of per-unit R2 (same-cohort truth variance); not pooled; not a percentage accuracy"
        out[arm] = record
    return out


# --------------------------------------------------------------------------- paired statistics
def paired_comparison(candidates, references, truth, starts, settings, reference_name):
    """Losses averaged over the seeds available for each arm on the SAME windows (not an ensemble)."""
    cand_abs = np.mean([np.abs(c - truth) for c in candidates], axis=0)
    ref_abs = np.mean([np.abs(r - truth) for r in references], axis=0)
    cand_sq = np.mean([(c - truth) ** 2 for c in candidates], axis=0)
    ref_sq = np.mean([(r - truth) ** 2 for r in references], axis=0)
    kwargs = {"block_length": settings["block_length"], "replicates": settings["replicates"], "seed": settings["seed"]}
    return {"reference": reference_name, "candidate_seeds": len(candidates), "reference_seeds": len(references),
            "absolute_error": v25.paired_loss_statistics(cand_abs, ref_abs, starts, **kwargs),
            "squared_error": v25.paired_loss_statistics(cand_sq, ref_sq, starts, units="kW^2", loss_name="squared_error", **kwargs),
            "rmse_difference": v25.rmse_difference_bootstrap(cand_sq, ref_sq, starts, **kwargs),
            "per_lead_mae_gain_kw": (ref_abs - cand_abs).mean(0).tolist(),
            "per_lead_rmse_gain_kw": (np.sqrt(ref_sq.mean(0)) - np.sqrt(cand_sq.mean(0))).tolist()}


def statistics(entries, protocol, split):
    settings = protocol["statistics"]
    out = {"split": split, "estimand": "window_uniform", "cluster": "forecast_start (issue time) across all turbines",
           "block_length_issue_times": settings["block_length"], "replicates": settings["replicates"],
           "bootstrap_seed": settings["seed"], "newey_west_lag": settings["hac_max_lag"],
           "seed_handling": "losses averaged over seeds on identical windows; seeds share one weather cohort and are "
                            "never treated as independent samples", "folds": {}}
    primary_p = []
    for fold in sorted({k[0] for k in entries}):
        base = cohort_of(entries, fold)
        truth, starts = base["truth"], base["times"][:, 0] - np.timedelta64(1, "h")
        by_arm = {arm: [entries[k]["prediction"] for k in sorted(entries) if k[0] == fold and k[2] == arm]
                  for arm in v25.ALL_ARMS}
        seeds = {arm: [k[1] for k in sorted(entries) if k[0] == fold and k[2] == arm] for arm in v25.ALL_ARMS}
        common = sorted(set(seeds[v25.PRIMARY]) & set(seeds[v25.BASELINE]))
        fold_out = {"seeds": seeds, "coverage": v25.issue_time_coverage(starts, protocol["eval_stride"])}
        if by_arm[v25.PRIMARY]:
            if common:
                pick = lambda arm: [entries[(fold, s, arm)]["prediction"] for s in common]
                fold_out["primary_full_vs_dlinear"] = paired_comparison(pick(v25.PRIMARY), pick(v25.BASELINE), truth, starts,
                                                                        settings, "DLinear") | {"seeds_used": common}
                primary_p.append((fold, fold_out["primary_full_vs_dlinear"]["absolute_error"]["newey_west"]["p_value_two_sided"]))
            fold_out["full_vs_persistence"] = paired_comparison(by_arm[v25.PRIMARY], [base["persistence"]], truth, starts,
                                                                settings, "Persistence")
            both = sorted(set(seeds[v25.PRIMARY]) & set(seeds[v25.INFO_CONTROL]))
            if both:
                fold_out["full_vs_power_only"] = paired_comparison(
                    [entries[(fold, s, v25.PRIMARY)]["prediction"] for s in both],
                    [entries[(fold, s, v25.INFO_CONTROL)]["prediction"] for s in both], truth, starts, settings, "TSMixerPowerOnly")
            per_seed = {}
            for seed in common:
                frame = pd.DataFrame({"forecast_start": starts})
                library = paired_forecast_statistics(entries[(fold, seed, v25.PRIMARY)]["prediction"],
                                                     entries[(fold, seed, v25.BASELINE)]["prediction"], truth, frame,
                                                     block_length=settings["block_length"], replicates=settings["replicates"],
                                                     seed=settings["seed"], estimand="window_uniform")
                mine = v25.paired_loss_statistics(np.abs(entries[(fold, seed, v25.PRIMARY)]["prediction"] - truth),
                                                  np.abs(entries[(fold, seed, v25.BASELINE)]["prediction"] - truth), starts,
                                                  block_length=settings["block_length"], replicates=settings["replicates"],
                                                  seed=settings["seed"])
                consistency = abs(library["primary_gain_kw"] - mine["gain"]) + abs(
                    library["time_block_bootstrap"]["ci95_low_kw"] - mine["ci95_low"])
                if consistency > 1e-9:
                    raise ValueError("Seed-averaging statistics drifted from utils.paired_forecast_statistics")
                per_seed[str(seed)] = {"mae_gain_kw": library["primary_gain_kw"],
                                       "ci95": [library["time_block_bootstrap"]["ci95_low_kw"],
                                                library["time_block_bootstrap"]["ci95_high_kw"]],
                                       "newey_west_p": library["dm_newey_west"]["p_value_two_sided_normal"]}
            fold_out["full_vs_dlinear_per_seed_library"] = per_seed
        out["folds"][str(fold)] = fold_out
    if primary_p:
        adjusted = v25.holm_adjust([p for _, p in primary_p])
        out["primary_family"] = {"definition": protocol["statistics"]["primary_pvalue_family"],
                                 "raw_p": {str(f): p for f, p in primary_p},
                                 "holm_adjusted_p": {str(f): a for (f, _), a in zip(primary_p, adjusted)}}
    return out


# --------------------------------------------------------------------------- figures
def figure_accuracy(macro, split, path):
    arms = [a for a in ARM_ORDER + ("v21_PowerAnchoredTimeDART_s2024",) if a in macro]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, key, label in zip(axes, ("mae", "rmse", "r2", "mae_skill_pct"), ("MAE (kW)", "RMSE (kW)", "R² (per-unit mean)", "MAE Skill vs Persistence (%)")):
        values = [macro[a][key] if macro[a][key] is not None else np.nan for a in arms]
        ax.bar(range(len(arms)), values, color=[COLORS.get(a, "#999") for a in arms])
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a.replace("_", "\n") for a in arms], fontsize=7)
        ax.set_title(label, fontsize=10)
        for i, v in enumerate(values):
            if np.isfinite(v):
                ax.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=7)
    seeds = {a: macro[a].get("seeds") for a in arms}
    fig.suptitle(f"v25 {split}: equal fold-seed macro (development evidence). seeds per arm: {seeds}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_leads(lead_frame, split, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, label in zip(axes, ("mae", "rmse", "mae_skill_pct"), ("MAE (kW)", "RMSE (kW)", "MAE Skill (%)")):
        for arm in ARM_ORDER:
            group = lead_frame[lead_frame.arm == arm]
            if len(group):
                series = group.groupby("lead_hours")[key].mean()
                ax.plot(series.index, series.values, marker="o", label=arm, color=COLORS[arm])
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("lead (hours)")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"v25 {split}: per-lead macro over fold-seed units (development evidence)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_history(directory, protocol, reference_root, path):
    folds = protocol["folds"]
    fig, axes = plt.subplots(len(folds), 2, figsize=(13, 3.3 * len(folds)), squeeze=False)
    seed = protocol["seeds"]["A1"][0]
    for row, fold in enumerate(folds):
        sources = {arm: directory / "runs" / f"f{fold}_s{seed}" / arm for arm in v25.TSMIXER_ARMS}
        sources["DLinear"] = resolve(protocol["reference"]["dir"], reference_root) / "runs" / f"f{fold}_s{seed}" / "DLinear"
        for arm, folder in sources.items():
            history_path = folder / "training_history.json"
            if not history_path.exists():
                continue
            history = v25.read_json(history_path)
            epochs = [h["epoch"] for h in history]
            loss = [h.get("train_loss", np.nan) for h in history]
            mae = [h["selection"]["mae"] for h in history]
            trained = v25.read_json(folder / "trained.json")
            axes[row, 0].plot(epochs[1:], loss[1:], marker=".", label=arm, color=COLORS[arm])
            axes[row, 1].plot(epochs, mae, marker=".", label=arm, color=COLORS[arm])
            axes[row, 1].axvline(trained["best_epoch"], color=COLORS[arm], ls=":", alpha=.7)
        axes[row, 0].set_title(f"f{fold} s{seed}: train loss (standardized 0.8|e|+0.2e²)", fontsize=9)
        axes[row, 1].set_title(f"f{fold} s{seed}: internal selection MAE (kW); dotted = selected epoch", fontsize=9)
        for ax in axes[row]:
            ax.grid(alpha=.3)
            ax.set_xlabel("epoch")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("Training histories (internal selection only; outer curves are never used to choose epochs)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_trace(entries, split, protocol, path):
    fold, seed = protocol["folds"][0], protocol["seeds"]["A1"][0]
    base = cohort_of(entries, fold)
    if base is None:
        return {"available": False}
    issue = base["times"][:, 0] - np.timedelta64(1, "h")
    rule = v25.select_trace_segment(issue, base["turbines"], protocol["eval_stride"])
    rows = np.asarray(rule["rows"])
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    hourly = {}
    for r in rows:
        for h in range(base["truth"].shape[1]):
            hourly[base["times"][r, h]] = base["truth"][r, h]
    times = np.array(sorted(hourly))
    for ax, lead in zip(axes, (0, 3)):
        ax.plot(times, [hourly[t] for t in times], color="black", lw=1.4, label="truth")
        ax.plot(base["times"][rows, lead], base["persistence"][rows, lead], color=COLORS["Persistence"], ls="--", label="Persistence")
        for arm in v25.ALL_ARMS:
            if (fold, seed, arm) in entries:
                ax.plot(base["times"][rows, lead], entries[(fold, seed, arm)]["prediction"][rows, lead], marker=".",
                        color=COLORS[arm], label=f"{arm} s{seed}")
        ax.set_ylabel("power (kW)")
        ax.set_title(f"lead {lead + 1} h", fontsize=9)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8, ncol=3)
    stamp = lambda value: pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
    fig.suptitle(f"{split}, fold {fold}, seed {seed}, turbine {rule['turbine']}: issues {stamp(rule['first_issue'])} .. "
                 f"{stamp(rule['last_issue'])} UTC (complete 72 h run: {rule['complete']})\nrule: {rule['rule']}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"available": True, **{k: v for k, v in rule.items() if k != "rows"}, "windows": len(rows)}


def figure_gain(stats, split, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels, offsets = (("primary_full_vs_dlinear", "Full vs DLinear"), ("full_vs_persistence", "Full vs Persistence"),
                       ("full_vs_power_only", "Full vs PowerOnly")), (-.25, 0., .25)
    folds = sorted(stats["folds"], key=int)
    for (key, label), offset in zip(labels, offsets):
        xs, ys, lo, hi = [], [], [], []
        for i, fold in enumerate(folds):
            comparison = stats["folds"][fold].get(key)
            if comparison:
                a = comparison["absolute_error"]
                xs.append(i + offset)
                ys.append(a["gain"])
                lo.append(a["ci95_low"])
                hi.append(a["ci95_high"])
        if xs:
            # Percentile intervals need not include the estimate: draw their actual endpoints.
            points, = ax.plot(xs, ys, "o", label=label)
            ax.vlines(xs, lo, hi, color=points.get_color())
            ax.plot(xs, lo, "_", color=points.get_color())
            ax.plot(xs, hi, "_", color=points.get_color())
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([f"fold {f}" for f in folds])
    ax.set_ylabel("MAE gain (kW), positive favours TSMixerFull")
    scope = "OUTER validation (development, previously inspected)" if split == "outer_validation" else \
        "INTERNAL selection diagnostic only (no outer evaluation)"
    ax.set_title(f"{scope}: seed-averaged paired MAE gain, issue-time block bootstrap 95% CI", fontsize=9)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- independent recomputation
def recomputation(directory, protocol, sets, reference_root):
    capacity, checks = protocol["rated_power_kw"], []
    for trained_path in sorted((directory / "runs").glob("f*_s*/*/trained.json")):
        trained = v25.read_json(trained_path)
        fold, seed = (int(x[1:]) for x in trained_path.parent.parent.name.split("_"))
        arrays = sets["inner_selection"][(fold, seed, trained["arm"])]
        mine = v25.point_metrics(arrays["prediction"], arrays["truth"], arrays["persistence"], capacity)
        checks.append({"what": f"inner f{fold} s{seed} {trained['arm']}",
                       "max_abs_diff": max(abs(mine[k] - trained["selection_best"][k]) for k in ("mae", "rmse", "mse"))})
        result = trained_path.parent / "result.json"
        if result.exists():
            registered = v25.read_json(result)["metrics"]
            arrays = sets["outer_validation"][(fold, seed, trained["arm"])]
            mine = v25.point_metrics(arrays["prediction"], arrays["truth"], arrays["persistence"], capacity)
            checks.append({"what": f"outer f{fold} s{seed} {trained['arm']}",
                           "max_abs_diff": max(abs(mine[k] - registered[k]) for k in ("mae", "rmse", "mse"))})
    for (fold, seed, arm), arrays in sets["outer_validation"].items():
        if arm == "DLinear" and seed == protocol["reference"]["seed"]:
            registered = v25.read_json(resolve(protocol["reference"]["dir"], reference_root) / "runs" /
                                       f"f{fold}_s{seed}" / "DLinear" / "result.json")["metrics"]
            mine = v25.point_metrics(arrays["prediction"], arrays["truth"], arrays["persistence"], capacity)
            checks.append({"what": f"outer f{fold} s{seed} v20 DLinear",
                           "max_abs_diff": max(abs(mine[k] - registered[k]) for k in ("mae", "rmse", "mse"))})
    worst = max([c["max_abs_diff"] for c in checks] or [0.])
    if worst > RECOMPUTE_TOLERANCE:
        raise ValueError(f"Independent recomputation disagrees with registered metrics by {worst}")
    return {"checks": checks, "max_abs_diff": worst, "tolerance": RECOMPUTE_TOLERANCE}


def v21_historical(protocol, sets, reference_root):
    spec = protocol["v21_historical"]
    folder = resolve(spec["dir"], reference_root)
    rows = []
    try:
        if v25.file_sha256(folder / "completed.json") != spec["completed_sha256"]:
            return {"available": False, "reason": "v21 completed.json identity differs"}
        v25.verify_registered_files(folder, v25.read_json(folder / "completed.json"))
        for fold in protocol["folds"]:
            run = folder / "runs" / f"f{fold}_s{spec['seed']}" / spec["model"]
            hashes = v25.read_json(run / "completed.json")
            arrays = load_npz(run / "raw_predictions.npz", hashes["raw_predictions.npz"])
            base = cohort_of(sets["outer_validation"], fold)
            v25.assert_aligned(base, arrays, keys=("times", "turbines", "window_starts", "truth", "persistence"), label="v21")
            rows.append({"split": "outer_validation", "fold": fold, "seed": spec["seed"], "arm": "v21_PowerAnchoredTimeDART_s2024",
                         "source": "v21 registered outer prediction (historical, single seed)",
                         **v25.point_metrics(arrays["prediction"], arrays["truth"], arrays["persistence"], protocol["rated_power_kw"])})
        manifest = folder / "runs" / f"f{protocol['folds'][0]}_s{spec['seed']}" / spec["model"] / "manifest.json"
        parameters = v25.read_json(manifest).get("parameters") if manifest.exists() else None
        return {"available": True, "rows": rows, "parameters": parameters}
    except (OSError, KeyError, ValueError) as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}


def resources(directory, sets):
    rows = []
    for trained_path in sorted((directory / "runs").glob("f*_s*/*/trained.json")):
        trained = v25.read_json(trained_path)
        result = trained_path.parent / "result.json"
        rows.append({"run": f"{trained_path.parent.parent.name}/{trained['arm']}", "parameters": trained["trainable_parameters"],
                     "best_epoch": trained["best_epoch"], "epochs_completed": trained["epochs_completed"],
                     "stopped_early": trained["stopped_early"], "device": trained["device"],
                     "train_seconds": trained["timing"]["train_seconds"], "selection_eval_seconds": trained["timing"]["assess_seconds"],
                     "outer_inference_seconds": v25.read_json(result)["inference_seconds"] if result.exists() else None,
                     "peak_memory_kind": trained["peak_memory"]["kind"], "peak_memory_value": trained["peak_memory"]["value"]})
    return rows


# --------------------------------------------------------------------------- markdown
def table(rows, columns, digits=4):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            cells.append(f"{value:.{digits}f}" if isinstance(value, float) and math.isfinite(value) else
                         ("undefined" if value is None else str(value)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def conclusion_text(g1, g2, outer, macro_outer):
    if not g1["G1"]["passed"]:
        checks = g1["G1"].get("checks", {})
        return ("G1 FAILED on internal selection (seed 2024): " + ", ".join(f"{k}={v}" for k, v in checks.items())
                + ". The bounded TSMixer recipe stops here; no new outer accuracy exists. DLinear remains the reference.")
    if g2 is None or not g2["G2"]["passed"]:
        checks = (g2 or {}).get("G2", {}).get("checks", {})
        return ("G1 passed but G2 FAILED on internal selection across seeds 2024-2026: "
                + ", ".join(f"{k}={v}" for k, v in checks.items())
                + ". No new outer accuracy exists; DLinear remains the reference.")
    full, dl = macro_outer.get("TSMixerFull"), macro_outer.get("DLinear")
    return (f"G1 and G2 passed; one outer development evaluation: TSMixerFull macro MAE {full['mae']:.3f} / RMSE "
            f"{full['rmse']:.3f} kW vs DLinear {dl['mae']:.3f} / {dl['rmse']:.3f} kW (differences {dl['mae'] - full['mae']:+.3f} / "
            f"{dl['rmse'] - full['rmse']:+.3f} kW, positive favours TSMixerFull). Development evidence only; see paired "
            "statistics for uncertainty. Choose the backbone from these same-protocol numbers, not from the model name.")


def build_report(directory, reference_root=ROOT):
    directory = Path(directory)
    record = v25.read_json(directory / "protocol.json")
    protocol = record["protocol"]
    training = v25.verify_training_completed(directory)
    a0 = v25.read_json(directory / "a0_result.json")
    capacity = protocol["rated_power_kw"]
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    sets, notes = collect(directory, protocol, a0, reference_root)
    for entries in sets.values():
        if entries:
            check_alignment(entries)
    g1 = v25.read_json(directory / "inner_gate_seed2024.json")
    g2 = v25.read_json(directory / "inner_gate_all_seeds.json") if (directory / "inner_gate_all_seeds.json").exists() else None
    outer = bool(training["outer_evaluated"])
    if outer != bool(sets["outer_validation"]):
        raise ValueError("Outer predictions exist without a registered outer evaluation (or vice versa)")
    all_rows, all_leads, macro, stats = [], [], {}, {}
    for split, entries in sets.items():
        if not entries:
            continue
        rows, leads = metric_rows(split, entries, capacity)
        if split == "outer_validation":
            historical = v21_historical(protocol, sets, reference_root)
            rows += historical.get("rows", [])
        all_rows += rows
        all_leads += leads
        frame = pd.DataFrame(rows)
        seeds_by_arm = {arm: sorted({int(s) for s in group.seed.dropna()}) for arm, group in frame.groupby("arm")}
        macro[split] = macro_table(frame, seeds_by_arm)
        stats[split] = statistics(entries, protocol, split)
    pd.DataFrame(all_rows).to_csv(directory / "metrics_all_runs.csv", index=False)
    pd.DataFrame(all_leads).to_csv(directory / "metrics_by_lead.csv", index=False)
    v25.write_json(directory / "paired_statistics.json", stats)
    recompute = recomputation(directory, protocol, sets, reference_root)
    v25.write_json(directory / "independent_recomputation.json", recompute)
    resource_rows = resources(directory, sets)
    pd.DataFrame(resource_rows).to_csv(directory / "resources.csv", index=False)
    split = "outer_validation" if outer else "inner_selection"
    lead_frame = pd.DataFrame([r for r in all_leads if r["split"] == split])
    figure_accuracy(macro[split], split, figures / "accuracy_overview.png")
    figure_leads(lead_frame, split, figures / "lead_metrics.png")
    figure_history(directory, protocol, reference_root, figures / "training_history.png")
    trace = figure_trace(sets[split], split, protocol, figures / "forecast_trace.png")
    figure_gain(stats[split], split, figures / "paired_gain.png")
    conclusion = conclusion_text(g1, g2, outer, macro.get("outer_validation", {}))
    result = {"protocol_id": protocol["protocol_id"], "outer_evaluated": outer, "G1": g1["G1"], "I1": g1["I1"],
              "G2": g2["G2"] if g2 else None, "I2": g2["I2"] if g2 else None, "macro": macro,
              "trace": trace, "notes": notes, "conclusion": conclusion, "wiki": "not_applicable_wiki_disabled",
              "wiki_metrics": {k: "not_applicable" for k in ("coverage", "selected_gain", "harm", "abstention")},
              "evidence_scope": protocol["evidence_scope"], "sealed_test_evaluated": False,
              "independent_recomputation_max_abs_diff": recompute["max_abs_diff"]}
    v25.write_json(directory / "result.json", result)
    write_markdown(directory, protocol, result, macro, stats, resource_rows, a0)
    outputs = ["metrics_all_runs.csv", "metrics_by_lead.csv", "paired_statistics.json", "independent_recomputation.json",
               "resources.csv", "result.json", "report.md"] + [f"figures/{p.name}" for p in sorted(figures.glob("*.png"))]
    v25.write_json(directory / "report_manifest.json", {
        "report_sources": {name: v25.file_sha256(ROOT / name) for name in REPORT_SOURCES},
        "inputs": {"training_completed.json": v25.file_sha256(directory / "training_completed.json")},
        "outputs": {name: v25.file_sha256(directory / name) for name in outputs},
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "note": "report fingerprint is separate from the training contract; editing figures never requires retraining"})
    return result


def write_markdown(directory, protocol, result, macro, stats, resource_rows, a0):
    split = "outer_validation" if result["outer_evaluated"] else "inner_selection"
    lines = [f"# v25 Kelmarsh backbone information report ({dt.date.today().isoformat()})", "",
             f"**Conclusion.** {result['conclusion']}", "",
             "Scope: Kelmarsh 2016, 24 h history -> 4 h direct forecast, six turbines, three rolling folds. Wiki disabled. "
             "All numbers are development evidence (outer and inner splits were inspected in earlier versions); "
             "nothing here is an independent test. Seeds share one weather cohort.", "",
             "## Internal gates (selection split only)", "",
             f"- G1 passed: {result['G1']['passed']} — checks: {result['G1'].get('checks')}",
             f"- I1 (auxiliary package helps, seed 2024): {result['I1']['supported']}",
             f"- G2 passed: {result['G2']['passed'] if result['G2'] else 'not run (G1 failed)'}",
             f"- I2: {result['I2']['supported'] if result['I2'] else 'not run'}", ""]
    for name, block in macro.items():
        rows = [{"arm": arm, **values} for arm, values in block.items()]
        lines += [f"## Macro metrics: {name}", "", table(rows, ["arm", "units", "mae", "mse", "rmse", "r2", "mae_skill_pct",
                                                                "rmse_skill_pct", "mbe", "within_5pct_capacity_pct",
                                                                "within_10pct_capacity_pct", "macro_error_ratio_skill_mae_pct"]), ""]
    lines += ["R² uses the truth variance of the same cohort; it is not a percentage accuracy. Skill = 1 - error/Persistence "
              "error, computed per fold/seed before averaging; macro error-ratio skill is reported separately.", ""]
    lines += [f"## Paired statistics ({split})", ""]
    for fold, block in stats[split]["folds"].items():
        for key in ("primary_full_vs_dlinear", "full_vs_persistence", "full_vs_power_only"):
            comparison = block.get(key)
            if not comparison:
                continue
            a, r, s = comparison["absolute_error"], comparison["rmse_difference"], comparison["squared_error"]
            lines.append(f"- f{fold} {key}: MAE gain {a['gain']:+.3f} kW [{a['ci95_low']:+.3f}, {a['ci95_high']:+.3f}], "
                         f"NW p={a['newey_west']['p_value_two_sided']:.4f}; RMSE gain {r['rmse_gain_kw']:+.3f} kW "
                         f"[{r['ci95_low_kw']:+.3f}, {r['ci95_high_kw']:+.3f}]; MSE (squared loss) gain {s['gain']:+.1f} kW², "
                         f"NW p={s['newey_west']['p_value_two_sided']:.4f}; per-lead MAE gain "
                         f"{[round(v, 3) for v in comparison['per_lead_mae_gain_kw']]}")
        lines.append(f"- f{fold} issue-time coverage: {block['coverage']}")
    if "primary_family" in stats[split]:
        lines += ["", f"Primary family raw p: {stats[split]['primary_family']['raw_p']}; Holm: "
                      f"{stats[split]['primary_family']['holm_adjusted_p']}"]
    lines += ["", "The MAE-difference CI is not an RMSE-difference CI; RMSE CIs resample squared-error sums per issue-time "
                  "cluster and take square roots per arm.", "", "## Resources", "",
              table(resource_rows, ["run", "parameters", "best_epoch", "epochs_completed", "device", "train_seconds",
                                    "selection_eval_seconds", "outer_inference_seconds", "peak_memory_kind", "peak_memory_value"], 2),
              "", "## Figures", ""] + [f"![{p.stem}](figures/{p.name})" for p in sorted((directory / "figures").glob("*.png"))]
    lines += ["", "## Recovery", "", "- status: `bash scripts/train/Kelmarsh_launch_backbone_information.sh --status`",
              "- resume (never retrains registered fits): `bash scripts/train/Kelmarsh_launch_backbone_information.sh --resume --detach`",
              "- report only: `bash scripts/train/Kelmarsh_launch_backbone_information.sh --report-only`", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(build_report(args.directory)["conclusion"])


if __name__ == "__main__":
    main()
