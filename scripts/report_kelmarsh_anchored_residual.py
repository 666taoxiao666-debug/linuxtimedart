#!/usr/bin/env python3
"""v26 A4 report: independent recomputation, paired statistics, disclosures and figures from REGISTERED files only.

Torch-free by design: it never trains, never re-selects and never reads unregistered outer data. A report failure is a
report failure only (launcher --report-only re-runs it without retraining). Besides the v25-style tables it discloses,
for every fit, the selected vs last vs epoch-0 metrics, how far the forecast moved away from the frozen anchor, and the
fit-side (training windows, eval mode) metrics next to the anchor's own fit replay.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils import kelmarsh_anchored_residual as v26  # noqa: E402
from utils.paired_forecast_statistics import paired_forecast_statistics  # noqa: E402

REPORT_SOURCES = ("scripts/report_kelmarsh_anchored_residual.py", "utils/kelmarsh_anchored_residual.py",
                  "utils/kelmarsh_backbone_information.py", "utils/paired_forecast_statistics.py")
ARM_ORDER = ("Persistence", "DLinear", "AnchoredFull", "AnchoredPowerOnly")
HISTORICAL_PREFIX = "v25_"
COLORS = {"Persistence": "#7f7f7f", "DLinear": "#1f77b4", "AnchoredFull": "#d62728", "AnchoredPowerOnly": "#ff7f0e",
          "v25_TSMixerFull": "#9467bd", "v25_TSMixerPowerOnly": "#c5b0d5", "truth": "#000000"}
RECOMPUTE_TOLERANCE = 1e-6
STAT_PAIRS = (("primary_full_vs_dlinear", v26.PRIMARY, v26.BASELINE),
              ("full_vs_persistence", v26.PRIMARY, "Persistence"),
              ("full_vs_power_only", v26.PRIMARY, v26.INFO_CONTROL))


def resolve(path, root):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def load_npz(path, expected_sha=None):
    if expected_sha is not None and v26.file_sha256(path) != expected_sha:
        raise ValueError(f"Registered prediction file changed: {path}")
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def parse_run(folder):
    fold, seed = folder.parent.name.split("_")
    return int(fold[1:]), int(seed[1:])


# --------------------------------------------------------------------------- collection
def collect(directory, protocol, reference_root):
    """{split: {(fold, seed, arm): arrays}} for every registered prediction of this run (historical columns apart)."""
    sets = {"inner_selection": {}, "outer_validation": {}}
    for trained_path in sorted((directory / "runs").glob("f*_s*/*/trained.json")):
        trained = v26.read_json(trained_path)
        fold, seed = parse_run(trained_path.parent)
        arrays = load_npz(trained_path.parent / "selection_predictions.npz", trained["artifacts"]["selection_predictions.npz"])
        sets["inner_selection"][(fold, seed, trained["arm"])] = arrays | {"source": "v26 fit (best.pt)"}
    manifest = v26.read_json(directory / "reference_manifest.json")
    for fold_key, record in manifest["dlinear_anchor"].items():
        path = directory / record["selection_predictions"]
        arrays = load_npz(path, record["selection_predictions_sha256"])
        sets["inner_selection"][(int(fold_key), protocol["seeds"]["anchor_seed"], v26.BASELINE)] = arrays | {
            "source": "frozen v20 anchor, CPU replay (gates used the registered v20 history for overall MAE/RMSE)"}
    index_path = directory / "outer_index.json"
    if index_path.exists():
        for entry in v26.read_json(index_path)["entries"].values():
            base = directory if entry.get("base") == "run" else reference_root
            arrays = load_npz(resolve(entry["path"], base), entry["sha256"])
            if entry["arm"] == "Persistence":
                continue
            sets["outer_validation"][(entry["fold"], entry["seed"], entry["arm"])] = arrays | {"source": entry["source"]}
    return sets


def historical_v25(protocol, inner, reference_root):
    """Optional report-only from-scratch TSMixer columns on the identical inner cohort. Never gated."""
    spec = protocol["v25_historical"]
    folder = resolve(spec["dir"], reference_root)
    out = {}
    try:
        if v26.file_sha256(folder / "completed.json") != spec["completed_sha256"]:
            return {}, {"available": False, "reason": "v25 completed.json identity differs"}
        hashes = v26.read_json(folder / "completed.json")
        for fold in protocol["folds"]:
            for arm in spec["arms"]:
                name = f"runs/f{fold}_s{spec['seed']}/{arm}/selection_predictions.npz"
                arrays = load_npz(folder / name, hashes[name])
                base = cohort_of(inner, fold)
                v26.assert_aligned(base, arrays, label=f"v25 {arm} f{fold}")
                out[(fold, spec["seed"], HISTORICAL_PREFIX + arm)] = arrays | {
                    "source": "v25 registered from-scratch TSMixer (historical, single seed, never gated)"}
        return out, {"available": True, "arms": [HISTORICAL_PREFIX + a for a in spec["arms"]]}
    except (OSError, KeyError, ValueError) as error:
        return {}, {"available": False, "reason": f"{type(error).__name__}: {error}"}


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
                v26.assert_aligned(base, arrays, label=f"f{key[0]} s{key[1]} {key[2]}")


def check_anchor_column(entries):
    """Candidate NPZs carry the anchor forecast they were checked against; it must be the registered anchor replay."""
    for (fold, seed, arm), arrays in entries.items():
        if arm in v26.CANDIDATE_ARMS:
            reference = next(a for (f, _, name), a in entries.items() if f == fold and name == v26.BASELINE)
            if not np.array_equal(arrays["anchor"], reference["prediction"]):
                raise ValueError(f"f{fold} s{seed} {arm}: stored anchor differs from the registered anchor replay")


# --------------------------------------------------------------------------- metrics tables
def metric_rows(split, entries, capacity):
    rows, leads = [], []
    for fold in sorted({k[0] for k in entries}):
        base = cohort_of(entries, fold)
        unit = {"split": split, "fold": fold, "seed": None, "arm": "Persistence", "source": "last observed power"}
        rows.append(unit | v26.point_metrics(base["persistence"], base["truth"], base["persistence"], capacity))
        leads += [unit | row for row in v26.lead_summary(base["persistence"], base["truth"], base["persistence"], capacity)]
    for (fold, seed, arm), arrays in sorted(entries.items()):
        unit = {"split": split, "fold": fold, "seed": seed, "arm": arm, "source": arrays["source"]}
        rows.append(unit | v26.point_metrics(arrays["prediction"], arrays["truth"], arrays["persistence"], capacity))
        leads += [unit | row for row in v26.lead_summary(arrays["prediction"], arrays["truth"], arrays["persistence"], capacity)]
    return rows, leads


def macro_table(frame, seeds_by_arm):
    """Equal mean over fold-seed units (Persistence: one unit per fold). Skills are per-unit first; the macro
    error-ratio skill is reported separately and named so."""
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


def macro_by_lead(lead_frame):
    out = {}
    for (arm, lead), group in lead_frame.groupby(["arm", "lead_hours"], sort=False):
        out.setdefault(arm, {})[str(int(lead))] = {k: float(group[k].mean()) for k in ("mae", "rmse", "mbe", "r2")}
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
            "absolute_error": v26.paired_loss_statistics(cand_abs, ref_abs, starts, **kwargs),
            "squared_error": v26.paired_loss_statistics(cand_sq, ref_sq, starts, units="kW^2", loss_name="squared_error", **kwargs),
            "rmse_difference": v26.rmse_difference_bootstrap(cand_sq, ref_sq, starts, **kwargs),
            "per_lead_mae_gain_kw": (ref_abs - cand_abs).mean(0).tolist(),
            "per_lead_rmse_gain_kw": (np.sqrt(ref_sq.mean(0)) - np.sqrt(cand_sq.mean(0))).tolist()}


def statistics(entries, protocol, split):
    settings = protocol["statistics"]
    out = {"split": split, "estimand": "window_uniform", "cluster": "forecast_start (issue time) across all turbines",
           "block_length_issue_times": settings["block_length"], "replicates": settings["replicates"],
           "bootstrap_seed": settings["seed"], "newey_west_lag": settings["hac_max_lag"],
           "seed_handling": "candidate losses averaged over residual seeds on identical windows; the anchor is one model; "
                            "seeds share one weather cohort and are never treated as independent samples", "folds": {}}
    primary_p = []
    for fold in sorted({k[0] for k in entries}):
        base = cohort_of(entries, fold)
        truth, starts = base["truth"], base["times"][:, 0] - np.timedelta64(1, "h")
        by_arm = {arm: [entries[k]["prediction"] for k in sorted(entries) if k[0] == fold and k[2] == arm]
                  for arm in v26.ALL_ARMS}
        by_arm["Persistence"] = [base["persistence"]]
        seeds = {arm: [k[1] for k in sorted(entries) if k[0] == fold and k[2] == arm] for arm in v26.ALL_ARMS}
        fold_out = {"seeds": seeds, "coverage": v26.issue_time_coverage(starts, protocol["eval_stride"])}
        for key, candidate, reference in STAT_PAIRS:
            if by_arm[candidate] and by_arm[reference]:
                fold_out[key] = paired_comparison(by_arm[candidate], by_arm[reference], truth, starts, settings, reference) \
                    | {"candidate_seeds_used": seeds[candidate]}
        if "primary_full_vs_dlinear" in fold_out:
            primary_p.append((fold, fold_out["primary_full_vs_dlinear"]["absolute_error"]["newey_west"]["p_value_two_sided"]))
            per_seed = {}
            reference = by_arm[v26.BASELINE][0]
            for seed in seeds[v26.PRIMARY]:
                prediction = entries[(fold, seed, v26.PRIMARY)]["prediction"]
                library = paired_forecast_statistics(prediction, reference, truth, pd.DataFrame({"forecast_start": starts}),
                                                     block_length=settings["block_length"], replicates=settings["replicates"],
                                                     seed=settings["seed"], estimand="window_uniform")
                mine = v26.paired_loss_statistics(np.abs(prediction - truth), np.abs(reference - truth), starts,
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
        adjusted = v26.holm_adjust([p for _, p in primary_p])
        out["primary_family"] = {"definition": protocol["statistics"]["primary_pvalue_family"],
                                 "raw_p": {str(f): p for f, p in primary_p},
                                 "holm_adjusted_p": {str(f): a for (f, _), a in zip(primary_p, adjusted)}}
    return out


# --------------------------------------------------------------------------- disclosures
def epoch_disclosure(directory, a0):
    """Selected vs last vs epoch-0 internal metrics of every fit, next to the anchor; harm is never hidden."""
    rows = []
    for trained_path in sorted((directory / "runs").glob("f*_s*/*/trained.json")):
        trained = v26.read_json(trained_path)
        fold, seed = parse_run(trained_path.parent)
        dl = a0["dlinear_anchor_inner"][str(fold)]
        history = v26.read_json(trained_path.parent / "training_history.json")
        trained_epochs = [h for h in history if int(h["epoch"]) >= 1]
        worst = max(trained_epochs, key=lambda h: h["selection"]["mae"]) if trained_epochs else None
        rows.append({"run": f"f{fold}_s{seed}/{trained['arm']}", "fold": fold, "seed": seed, "arm": trained["arm"],
                     "best_epoch": trained["best_epoch"], "epochs_completed": trained["epochs_completed"],
                     "dlinear_mae": dl["mae"], "dlinear_rmse": dl["rmse"],
                     "epoch0_mae": trained["selection_epoch0"]["mae"], "epoch0_rmse": trained["selection_epoch0"]["rmse"],
                     "best_mae": trained["selection_best"]["mae"], "best_rmse": trained["selection_best"]["rmse"],
                     "last_mae": trained["selection_last"]["mae"], "last_rmse": trained["selection_last"]["rmse"],
                     "best_minus_dlinear_mae": trained["selection_best"]["mae"] - dl["mae"],
                     "last_minus_dlinear_mae": trained["selection_last"]["mae"] - dl["mae"],
                     "worst_trained_epoch": worst["epoch"] if worst else None,
                     "worst_trained_minus_dlinear_mae": (worst["selection"]["mae"] - dl["mae"]) if worst else None,
                     "epoch0_max_abs_diff_vs_anchor_kw": trained["epoch0_max_abs_diff_vs_anchor_kw"]})
    return rows


def correction_disclosure(split, entries):
    rows = []
    for (fold, seed, arm), arrays in sorted(entries.items()):
        if arm not in v26.CANDIDATE_ARMS:
            continue
        anchor = next(a["prediction"] for (f, _, name), a in entries.items() if f == fold and name == v26.BASELINE)
        rows.append({"split": split, "fold": fold, "seed": seed, "arm": arm,
                     **v26.correction_summary(arrays["prediction"], anchor, arrays["truth"])})
    return rows


def fit_side_disclosure(directory):
    """Training-window (fit) eval-mode metrics of every selected state next to the anchor's fit replay (same kW scale)."""
    manifest = v26.read_json(directory / "reference_manifest.json")["dlinear_anchor"]
    rows = []
    for fold_key, record in sorted(manifest.items()):
        fit = record["replay_fit"]["overall"]
        rows.append({"run": f"f{fold_key} anchor", "fold": int(fold_key), "seed": None, "arm": v26.BASELINE,
                     "fit_mae": fit["mae"], "fit_rmse": fit["rmse"], "fit_mbe": fit["mbe"],
                     "selection_mae": record["replay_selection"]["mae"], "selection_rmse": record["replay_selection"]["rmse"],
                     "generalization_gap_mae": record["replay_selection"]["mae"] - fit["mae"]})
    for trained_path in sorted((directory / "runs").glob("f*_s*/*/trained.json")):
        trained = v26.read_json(trained_path)
        fold, seed = parse_run(trained_path.parent)
        fit = trained["fit_eval_best"]["overall"]
        rows.append({"run": f"f{fold}_s{seed}/{trained['arm']}", "fold": fold, "seed": seed, "arm": trained["arm"],
                     "fit_mae": fit["mae"], "fit_rmse": fit["rmse"], "fit_mbe": fit["mbe"],
                     "selection_mae": trained["selection_best"]["mae"], "selection_rmse": trained["selection_best"]["rmse"],
                     "generalization_gap_mae": trained["selection_best"]["mae"] - fit["mae"]})
    return rows


# --------------------------------------------------------------------------- figures
def figure_accuracy(macro, split, path):
    arms = [a for a in ARM_ORDER + ("v25_TSMixerFull", "v25_TSMixerPowerOnly") if a in macro]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, key, label in zip(axes, ("mae", "rmse", "r2", "mae_skill_pct"),
                              ("MAE (kW)", "RMSE (kW)", "R² (per-unit mean)", "MAE Skill vs Persistence (%)")):
        values = [macro[a][key] if macro[a][key] is not None else np.nan for a in arms]
        ax.bar(range(len(arms)), values, color=[COLORS.get(a, "#999") for a in arms])
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a.replace("_", "\n") for a in arms], fontsize=7)
        ax.set_title(label, fontsize=10)
        for i, v in enumerate(values):
            if np.isfinite(v):
                ax.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=7)
    seeds = {a: macro[a].get("seeds") for a in arms}
    fig.suptitle(f"v26 {split}: equal fold-seed macro (development evidence). seeds per arm: {seeds}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_leads(lead_frame, split, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, label in zip(axes, ("mae", "rmse", "mbe"), ("MAE (kW)", "RMSE (kW)", "MBE (kW, prediction - truth)")):
        for arm in ARM_ORDER + ("v25_TSMixerFull",):
            group = lead_frame[lead_frame.arm == arm]
            if len(group):
                series = group.groupby("lead_hours")[key].mean()
                ax.plot(series.index, series.values, marker="o", label=arm, color=COLORS.get(arm, "#999"),
                        ls="--" if arm.startswith(HISTORICAL_PREFIX) else "-")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("lead (hours)")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"v26 {split}: per-lead macro over fold-seed units (development evidence; dashed = historical v25)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_history(directory, protocol, a0, path):
    folds = protocol["folds"]
    fig, axes = plt.subplots(len(folds), 2, figsize=(13, 3.3 * len(folds)), squeeze=False)
    seed = protocol["seeds"]["A1"][0]
    for row, fold in enumerate(folds):
        for arm in v26.CANDIDATE_ARMS:
            folder = directory / "runs" / f"f{fold}_s{seed}" / arm
            if not (folder / "training_history.json").exists():
                continue
            history = v26.read_json(folder / "training_history.json")
            epochs = [h["epoch"] for h in history]
            loss = [h.get("train_loss", np.nan) for h in history]
            mae = [h["selection"]["mae"] for h in history]
            trained = v26.read_json(folder / "trained.json")
            axes[row, 0].plot(epochs[1:], loss[1:], marker=".", label=arm, color=COLORS[arm])
            axes[row, 1].plot(epochs, mae, marker=".", label=arm, color=COLORS[arm])
            axes[row, 1].axvline(trained["best_epoch"], color=COLORS[arm], ls=":", alpha=.7)
        dl = a0["dlinear_anchor_inner"][str(fold)]["mae"]
        axes[row, 1].axhline(dl, color=COLORS["DLinear"], ls="--", lw=1, label="DLinear (frozen anchor)")
        axes[row, 0].set_title(f"f{fold} s{seed}: train loss (standardized 0.8|e|+0.2e², dropout on)", fontsize=9)
        axes[row, 1].set_title(f"f{fold} s{seed}: internal selection MAE (kW); epoch 0 = anchor, not selectable", fontsize=9)
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
    rule = v26.select_trace_segment(issue, base["turbines"], protocol["eval_stride"])
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
        for arm in v26.ALL_ARMS:
            key = (fold, seed, arm)
            if key in entries:
                ax.plot(base["times"][rows, lead], entries[key]["prediction"][rows, lead], marker=".",
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
    labels, offsets = (("primary_full_vs_dlinear", "AnchoredFull vs DLinear"), ("full_vs_persistence", "AnchoredFull vs Persistence"),
                       ("full_vs_power_only", "AnchoredFull vs AnchoredPowerOnly")), (-.25, 0., .25)
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
    ax.set_ylabel("MAE gain (kW), positive favours AnchoredFull")
    scope = "OUTER validation (development, previously inspected)" if split == "outer_validation" else \
        "INTERNAL selection diagnostic only (no outer evaluation; epochs were chosen on this split)"
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
        trained = v26.read_json(trained_path)
        fold, seed = parse_run(trained_path.parent)
        arrays = sets["inner_selection"][(fold, seed, trained["arm"])]
        mine = v26.point_metrics(arrays["prediction"], arrays["truth"], arrays["persistence"], capacity)
        checks.append({"what": f"inner f{fold} s{seed} {trained['arm']}",
                       "max_abs_diff": max(abs(mine[k] - trained["selection_best"][k]) for k in ("mae", "rmse", "mse"))})
        leads = v26.lead_summary(arrays["prediction"], arrays["truth"], arrays["persistence"], capacity)
        checks.append({"what": f"inner f{fold} s{seed} {trained['arm']} per-lead",
                       "max_abs_diff": max(abs(a[k] - b[k]) for a, b in zip(leads, trained["selection_best_by_lead"])
                                           for k in ("mae", "rmse", "mbe"))})
        result = trained_path.parent / "result.json"
        if result.exists():
            registered = v26.read_json(result)["metrics"]
            arrays = sets["outer_validation"][(fold, seed, trained["arm"])]
            mine = v26.point_metrics(arrays["prediction"], arrays["truth"], arrays["persistence"], capacity)
            checks.append({"what": f"outer f{fold} s{seed} {trained['arm']}",
                           "max_abs_diff": max(abs(mine[k] - registered[k]) for k in ("mae", "rmse", "mse"))})
    manifest = v26.read_json(directory / "reference_manifest.json")["dlinear_anchor"]
    for fold_key, record in manifest.items():
        arrays = sets["inner_selection"][(int(fold_key), protocol["seeds"]["anchor_seed"], v26.BASELINE)]
        mine = v26.point_metrics(arrays["prediction"], arrays["truth"], arrays["persistence"], capacity)
        checks.append({"what": f"inner f{fold_key} anchor replay",
                       "max_abs_diff": max(abs(mine[k] - record["replay_selection"][k]) for k in ("mae", "rmse"))})
    for (fold, seed, arm), arrays in sets["outer_validation"].items():
        if arm == v26.BASELINE:
            registered = v26.read_json(resolve(protocol["reference"]["dir"], reference_root) / "runs" /
                                       f"f{fold}_s{seed}" / "DLinear" / "result.json")["metrics"]
            mine = v26.point_metrics(arrays["prediction"], arrays["truth"], arrays["persistence"], capacity)
            checks.append({"what": f"outer f{fold} s{seed} v20 DLinear (anchor)",
                           "max_abs_diff": max(abs(mine[k] - registered[k]) for k in ("mae", "rmse", "mse"))})
    worst = max([c["max_abs_diff"] for c in checks] or [0.])
    if worst > RECOMPUTE_TOLERANCE:
        raise ValueError(f"Independent recomputation disagrees with registered metrics by {worst}")
    return {"checks": checks, "max_abs_diff": worst, "tolerance": RECOMPUTE_TOLERANCE}


def resources(directory):
    rows = []
    for trained_path in sorted((directory / "runs").glob("f*_s*/*/trained.json")):
        trained = v26.read_json(trained_path)
        result = trained_path.parent / "result.json"
        rows.append({"run": f"{trained_path.parent.parent.name}/{trained['arm']}",
                     "trainable_parameters": trained["trainable_parameters"], "frozen_parameters": trained["frozen_parameters"],
                     "best_epoch": trained["best_epoch"], "epochs_completed": trained["epochs_completed"],
                     "stopped_early": trained["stopped_early"], "device": trained["device"],
                     "train_seconds": trained["timing"]["train_seconds"], "selection_eval_seconds": trained["timing"]["assess_seconds"],
                     "outer_inference_seconds": v26.read_json(result)["inference_seconds"] if result.exists() else None,
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


def conclusion_text(g1, g2, macro_outer):
    attribution = (" I1 (auxiliary-channel attribution, seed 2024) supported: " + str(g1["I1"]["supported"]) + "."
                   + ("" if g1["I1"]["supported"] else " Any numeric change must not be attributed to the auxiliary channels."))
    if not g1["G1"]["passed"]:
        checks = g1["G1"].get("checks", {})
        return ("G1 FAILED on internal selection (seed 2024): " + ", ".join(f"{k}={v}" for k, v in checks.items())
                + ". The anchored-residual mechanism stops here; no new outer accuracy exists. The frozen DLinear remains "
                  "the reference." + attribution)
    if g2 is None or not g2["G2"]["passed"]:
        checks = (g2 or {}).get("G2", {}).get("checks", {})
        return ("G1 passed but G2 FAILED on internal selection across residual seeds 2024-2026: "
                + ", ".join(f"{k}={v}" for k, v in checks.items())
                + ". No new outer accuracy exists; the frozen DLinear remains the reference." + attribution)
    full, dl = macro_outer.get(v26.PRIMARY), macro_outer.get(v26.BASELINE)
    i2 = g2["I2"]["supported"]
    return (f"G1 and G2 passed; one outer development evaluation: AnchoredFull macro MAE {full['mae']:.3f} / RMSE "
            f"{full['rmse']:.3f} kW vs frozen DLinear {dl['mae']:.3f} / {dl['rmse']:.3f} kW (differences "
            f"{dl['mae'] - full['mae']:+.3f} / {dl['rmse'] - full['rmse']:+.3f} kW, positive favours AnchoredFull). "
            f"I1={g1['I1']['supported']}, I2={i2}: an auxiliary-information increment may be claimed only if both hold. "
            "Development evidence only (outer inspected before); see paired statistics for uncertainty.")


def build_report(directory, reference_root=ROOT):
    directory = Path(directory)
    record = v26.read_json(directory / "protocol.json")
    protocol = record["protocol"]
    training = v26.verify_training_completed(directory)
    a0 = v26.read_json(directory / "a0_result.json")
    capacity = protocol["rated_power_kw"]
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    sets = collect(directory, protocol, reference_root)
    for entries in sets.values():
        if entries:
            check_alignment(entries)
    check_anchor_column(sets["inner_selection"])
    historical, historical_status = historical_v25(protocol, sets["inner_selection"], reference_root)
    g1 = v26.read_json(directory / "inner_gate_seed2024.json")
    g2 = v26.read_json(directory / "inner_gate_all_seeds.json") if (directory / "inner_gate_all_seeds.json").exists() else None
    outer = bool(training["outer_evaluated"])
    if outer != bool(sets["outer_validation"]):
        raise ValueError("Outer predictions exist without a registered outer evaluation (or vice versa)")
    all_rows, all_leads, macro, macro_leads, stats, corrections = [], [], {}, {}, {}, []
    for split, entries in sets.items():
        if not entries:
            continue
        rows, leads = metric_rows(split, entries, capacity)
        if split == "inner_selection" and historical:
            hist_rows, hist_leads = metric_rows(split, historical, capacity)
            rows += [r for r in hist_rows if r["arm"] != "Persistence"]
            leads += [r for r in hist_leads if r["arm"] != "Persistence"]
        all_rows += rows
        all_leads += leads
        frame = pd.DataFrame(rows)
        seeds_by_arm = {arm: sorted({int(s) for s in group.seed.dropna()}) for arm, group in frame.groupby("arm")}
        macro[split] = macro_table(frame, seeds_by_arm)
        macro_leads[split] = macro_by_lead(pd.DataFrame(leads))
        stats[split] = statistics(entries, protocol, split)
        corrections += correction_disclosure(split, entries) if split == "inner_selection" else []
    if outer:
        outer_entries = dict(sets["outer_validation"])
        corrections += [r | {"note": "anchor = registered v20 outer forecast"}
                        for r in correction_disclosure("outer_validation", outer_entries)]
    pd.DataFrame(all_rows).to_csv(directory / "metrics_all_runs.csv", index=False)
    pd.DataFrame(all_leads).to_csv(directory / "metrics_by_lead.csv", index=False)
    v26.write_json(directory / "paired_statistics.json", stats)
    recompute = recomputation(directory, protocol, sets, reference_root)
    v26.write_json(directory / "independent_recomputation.json", recompute)
    epochs = epoch_disclosure(directory, a0)
    fit_side = fit_side_disclosure(directory)
    pd.DataFrame(epochs).to_csv(directory / "epoch_disclosure.csv", index=False)
    pd.DataFrame(corrections).to_csv(directory / "correction_disclosure.csv", index=False)
    pd.DataFrame(fit_side).to_csv(directory / "fit_side_disclosure.csv", index=False)
    resource_rows = resources(directory)
    pd.DataFrame(resource_rows).to_csv(directory / "resources.csv", index=False)
    split = "outer_validation" if outer else "inner_selection"
    lead_frame = pd.DataFrame([r for r in all_leads if r["split"] == split])
    figure_accuracy(macro[split], split, figures / "accuracy_overview.png")
    figure_leads(lead_frame, split, figures / "lead_metrics.png")
    figure_history(directory, protocol, a0, figures / "training_history.png")
    trace = figure_trace(sets[split], split, protocol, figures / "forecast_trace.png")
    figure_gain(stats[split], split, figures / "paired_gain.png")
    conclusion = conclusion_text(g1, g2, macro.get("outer_validation", {}))
    result = {"protocol_id": protocol["protocol_id"], "outer_evaluated": outer, "G1": g1["G1"], "I1": g1["I1"],
              "G2": g2["G2"] if g2 else None, "I2": g2["I2"] if g2 else None, "macro": macro, "macro_by_lead": macro_leads,
              "historical_v25": historical_status, "trace": trace, "conclusion": conclusion,
              "wiki": "not_applicable_wiki_disabled",
              "wiki_metrics": {k: "not_applicable" for k in ("coverage", "selected_gain", "harm", "abstention")},
              "evidence_scope": protocol["evidence_scope"], "sealed_test_evaluated": False,
              "independent_recomputation_max_abs_diff": recompute["max_abs_diff"]}
    v26.write_json(directory / "result.json", result)
    write_markdown(directory, protocol, result, macro, macro_leads, stats, resource_rows, epochs, corrections, fit_side)
    outputs = ["metrics_all_runs.csv", "metrics_by_lead.csv", "paired_statistics.json", "independent_recomputation.json",
               "epoch_disclosure.csv", "correction_disclosure.csv", "fit_side_disclosure.csv", "resources.csv",
               "result.json", "report.md"] + [f"figures/{p.name}" for p in sorted(figures.glob("*.png"))]
    v26.write_json(directory / "report_manifest.json", {
        "report_sources": {name: v26.file_sha256(ROOT / name) for name in REPORT_SOURCES},
        "inputs": {"training_completed.json": v26.file_sha256(directory / "training_completed.json")},
        "outputs": {name: v26.file_sha256(directory / name) for name in outputs},
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "note": "report fingerprint is separate from the training contract; editing figures never requires retraining"})
    return result


def write_markdown(directory, protocol, result, macro, macro_leads, stats, resource_rows, epochs, corrections, fit_side):
    split = "outer_validation" if result["outer_evaluated"] else "inner_selection"
    g1 = result["G1"]
    lines = [f"# v26 Kelmarsh anchored-residual report ({dt.date.today().isoformat()})", "",
             f"**Conclusion.** {result['conclusion']}", "",
             "Scope: Kelmarsh 2016, 24 h history -> 4 h direct forecast, six turbines, three rolling folds. Wiki disabled. "
             "Forecast = frozen v20 DLinear + zero-initialised TSMixerBasicSmall residual. All numbers are development "
             "evidence (outer and inner splits were inspected in earlier versions; the anchor's epoch was chosen on this "
             "inner split in v20); nothing here is an independent test. Seeds share one weather cohort.", "",
             "## Internal gates (selection split only)", "",
             f"- G1 passed: {g1['passed']} — checks: {g1.get('checks')}",
             f"- I1 (auxiliary package attribution, seed 2024): {result['I1']['supported']}",
             f"- G2 passed: {result['G2']['passed'] if result['G2'] else 'not run (G1 failed)'}",
             f"- I2: {result['I2']['supported'] if result['I2'] else 'not run'}",
             f"- historical v25 columns: {result['historical_v25']}", ""]
    for name, block in macro.items():
        rows = [{"arm": arm, **values} for arm, values in block.items()]
        lines += [f"## Macro metrics: {name}", "", table(rows, ["arm", "units", "mae", "mse", "rmse", "r2", "mae_skill_pct",
                                                                "rmse_skill_pct", "mbe", "within_5pct_capacity_pct",
                                                                "within_10pct_capacity_pct", "macro_error_ratio_skill_mae_pct"]), ""]
        lead_rows = [{"arm": arm, "lead": lead, **values} for arm, by in macro_leads[name].items() for lead, values in by.items()]
        lines += [f"### Per lead: {name}", "", table(lead_rows, ["arm", "lead", "mae", "rmse", "mbe", "r2"]), ""]
    lines += ["R² uses the truth variance of the same cohort; it is not a percentage accuracy. Skill = 1 - error/Persistence "
              "error, computed per fold/seed before averaging; macro error-ratio skill is reported separately.", "",
              "## Selected vs last vs epoch-0 (internal selection; epoch 0 = the anchor, not selectable)", "",
              table(epochs, ["run", "best_epoch", "epochs_completed", "dlinear_mae", "epoch0_mae", "best_mae", "last_mae",
                             "best_minus_dlinear_mae", "last_minus_dlinear_mae", "worst_trained_minus_dlinear_mae",
                             "epoch0_max_abs_diff_vs_anchor_kw"]), "",
              "## Correction magnitude (|forecast - anchor|) and whether it helped", "",
              table(corrections, ["split", "fold", "seed", "arm", "mean_abs_correction_kw", "max_abs_correction_kw",
                                  "mean_correction_kw", "share_changed_points_helped_pct", "mean_abs_error_gain_kw"]), "",
              "## Fit-side (training windows, eval mode) vs selection — same kW scale, informational only", "",
              table(fit_side, ["run", "fit_mae", "fit_rmse", "fit_mbe", "selection_mae", "selection_rmse",
                               "generalization_gap_mae"]), "",
              "The anchor was trained on the same fit windows, so fit-side residuals are training-internal; the two "
              "periods differ in difficulty, so a gap is descriptive, not a proof of over- or under-fitting.", "",
              f"## Paired statistics ({split})", ""]
    for fold, block in stats[split]["folds"].items():
        for key, _, _ in STAT_PAIRS:
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
    lines += ["", "On the internal split the candidate's epoch was chosen by MAE on the same windows (as the anchor's was in "
                  "v20); internal gains are therefore optimistic. The MAE-difference CI is not an RMSE-difference CI.", "",
              "## Resources", "",
              table(resource_rows, ["run", "trainable_parameters", "frozen_parameters", "best_epoch", "epochs_completed",
                                    "device", "train_seconds", "selection_eval_seconds", "outer_inference_seconds",
                                    "peak_memory_kind", "peak_memory_value"], 2),
              "", "## Figures", ""] + [f"![{p.stem}](figures/{p.name})" for p in sorted((directory / "figures").glob("*.png"))]
    lines += ["", "## Recovery", "", "- status: `bash scripts/train/Kelmarsh_launch_anchored_residual.sh --status`",
              "- resume (never retrains registered fits): `bash scripts/train/Kelmarsh_launch_anchored_residual.sh --resume --detach`",
              "- report only: `bash scripts/train/Kelmarsh_launch_anchored_residual.sh --report-only`", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(build_report(args.directory)["conclusion"])


if __name__ == "__main__":
    main()
