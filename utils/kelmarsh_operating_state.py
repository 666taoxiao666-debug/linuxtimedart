"""Torch-free helpers for v29 (Kelmarsh operating-state information group on the v28-chosen TSMixer_last10).

Protocol checks, D0 identity against the registered v28 preparation, the fixed Guard rule, the validation choice,
evaluation (both cohorts, C1-C6 with Holm, rules, 3-seed ensemble, ex-ante regime table), the post-test manifest and
the report. Generic metrics, paired statistics, hashing and JSON helpers are reused unchanged from the frozen v28 module
``utils.kelmarsh_multiyear_benchmark`` (its bytes enter the v29 fingerprint). In the reused
``verify_development_rows(new, reference)`` the "development" frame is the registered v28 preparation.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from utils.kelmarsh_multiyear_benchmark import (  # noqa: F401  (re-exported for the controller and tests)
    acquire_exclusive_lock, append_event, canonical_sha256, choose_pair_device, compare_all, contract_mismatches,
    expected_parameters, file_sha256, metrics, read_json, safe, seed_mean, text_sha256_lf, verify_development_rows,
    verify_power_curves, verify_registered_files, verify_training_completed, window_hash, write_json)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "kelmarsh_operating_state_v29"
PERSISTENCE_ARMS = ("PersistenceHour", "PersistenceLast10")
TRAINED_ARMS = ("TSMixer_last10_normal", "TSMixer_last10_all", "TSMixer_state_normal", "TSMixer_state_all")
GUARD_ARM = "Guard_last10_normal"
SEEDED_ARMS = TRAINED_ARMS + (GUARD_ARM,)
CHOICE_CANDIDATES = SEEDED_ARMS
ARMS = PERSISTENCE_ARMS + SEEDED_ARMS
COHORTS = ("all", "normal")          # all is primary in v29
COMPARISONS = ("C1", "C2", "C3", "C4", "C5", "C6")
CHANNELS = ["Wspd", "Etmp", "Ndir_sin", "Ndir_cos", "Pab_mean", "rotor_speed", "power_last10", "wind_last10",
            "available_capacity_frac", "stopped_frac", "power"]
CHANNEL_SETS = {"last10": [0, 1, 2, 3, 4, 5, 6, 7, 10], "state": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
STATE_CHANNELS = ["available_capacity_frac", "stopped_frac"]
ISSUE_ARRAYS = ("issue_stopped_frac", "issue_capacity_frac")
DERATED_BELOW = 0.98
HOUR = np.timedelta64(1, "h")
REPORT_SOURCES = ("utils/kelmarsh_operating_state.py", "scripts/report_kelmarsh_multiyear_benchmark.py",
                  "utils/kelmarsh_multiyear_benchmark.py", "utils/kelmarsh_backbone_information.py",
                  "utils/paired_forecast_statistics.py")


# --------------------------------------------------------------------------- protocol
def validate_protocol(p):
    d, arms = p["data"], p["arms"]
    if p["protocol_id"] != PROTOCOL_ID:
        raise ValueError("Unexpected protocol id")
    if tuple(arms) != ARMS:
        raise ValueError("Arm list departs from the frozen design")
    if (d["input_len"], d["pred_len"], d["stride"]) != (24, 4, 3):
        raise ValueError("Window design departs from 24 -> 4, stride 3")
    if d["channels"] != CHANNELS or {k: list(v) for k, v in d["channel_sets"].items()} != CHANNEL_SETS:
        raise ValueError("Channel design departs from the frozen design")
    if d["channels"][6:-1] != d["extras_in_common_series"] or d["state_group"]["channels"] != STATE_CHANNELS:
        raise ValueError("Common-series extras, state group and channel list disagree")
    splits, prep, ref = d["splits"], d["v29_preparation"], d["v28_registered_preparation"]
    if not (max(splits["train"]) < min(splits["validation"]) <= max(splits["validation"]) < min(splits["test"])
            <= max(splits["test"]) < min(d["sealed_years"])):
        raise ValueError("Years must be ordered train < validation < test < sealed")
    if not set(prep["curve_years"]) <= set(splits["train"]):
        raise ValueError("Power-curve years must belong to train")
    years = set(splits["train"]) | set(splits["validation"]) | set(splits["test"])
    if set(prep["years"]) != years or set(prep["years"]) & set(d["sealed_years"]):
        raise ValueError("Prepared years must be exactly train+validation+test and never sealed")
    if set(ref["years"]) != years - set(splits["test"]):
        raise ValueError("The registered v28 reference must cover exactly the non-test years")
    if d.get("yaw_error_channels_enabled") or any(c.startswith("yaw") for c in d["channels"]):
        raise ValueError("Yaw-error channels are disabled in v29")
    for arm in TRAINED_ARMS:
        spec = arms[arm]
        if spec["family"] != "TSMixerBasicSmall" or spec["channels"] not in CHANNEL_SETS:
            raise ValueError(f"{arm}: family or channel set departs from the design")
        if spec["train_cohort"] != ("all" if arm.endswith("_all") else "normal"):
            raise ValueError(f"{arm}: training cohort departs from the design")
        declared = p["families"][spec["family"]]["parameters"].get(spec["channels"])
        if declared != expected_parameters(spec["family"], len(CHANNEL_SETS[spec["channels"]])):
            raise ValueError(f"{arm}: declared parameters {declared} != formula")
    guard = arms[GUARD_ARM]
    if guard["family"] != "Guard" or guard["base_arm"] != "TSMixer_last10_normal" or guard.get("fit") is not False \
            or not 0 < float(guard["threshold_frac_rated"]) <= 0.05:
        raise ValueError("Guard arm departs from the frozen design")
    if not (p["cohorts"]["primary"] == p["cohorts"]["selection_cohort"] == p["training"]["selection_cohort"] == "all"):
        raise ValueError("v29 selects and decides on the ALL cohort")
    budget = p["budget"]
    if budget["fits"] != len(TRAINED_ARMS) * len(p["seeds"]) or budget["fits"] > budget["new_fits_max"]:
        raise ValueError("Budget does not match arms x seeds")
    if budget["fits"] * p["training"]["max_epochs"] > budget["new_epochs_max"]:
        raise ValueError("Epoch budget exceeded")
    if budget.get("additional_seeds") or budget.get("additional_arms") or budget.get("hyperparameter_search"):
        raise ValueError("No search, extra seeds or extra arms in v29")
    if tuple(sorted(p["comparisons"])) != COMPARISONS or p["comparisons"]["C5"]["candidate"] != "VALIDATION_CHOSEN":
        raise ValueError("Comparison family departs from the frozen design")
    for name, spec in p["comparisons"].items():
        if spec["reference"] not in ARMS or spec["candidate"] not in ARMS + ("VALIDATION_CHOSEN",):
            raise ValueError(f"{name}: unknown arm")
    if not float(p["rules"]["non_inferiority_margin_kw"]) > 0:
        raise ValueError("A positive non-inferiority margin is required")
    bins = p["diagnostics"]["anchor_bins_frac_rated"]
    if not bins or any(not 0 < a < c <= 1 for a, c in zip(bins, bins[1:])) or not 0 < bins[0]:
        raise ValueError("Anchor bins must increase within (0, 1]")
    return True


def arm_config(p, arm):
    spec = p["arms"][arm]
    channels = p["data"]["channel_sets"][spec["channels"]]
    return {"arm": arm, "family": spec["family"], "channel_set": spec["channels"], "channel_indices": list(channels),
            "channels": len(channels), "train_cohort": spec["train_cohort"],
            "selection_cohort": p["training"]["selection_cohort"],
            "expected_parameters": p["families"][spec["family"]]["parameters"][spec["channels"]],
            "args": dict(p["families"][spec["family"]].get("args", {}))}


def arm_parameters(p, arm):
    if arm in TRAINED_ARMS:
        return int(arm_config(p, arm)["expected_parameters"])
    if arm == GUARD_ARM:
        return arm_parameters(p, p["arms"][GUARD_ARM]["base_arm"])
    return 0


# --------------------------------------------------------------------------- data identity (D0)
def verify_preparation(meta, protocol, csv_sha256):
    prep = protocol["data"]["v29_preparation"]
    problems = []
    if meta.get("schema_version") != prep["schema_version"]:
        problems.append("schema_version")
    if sorted(meta.get("years", [])) != sorted(prep["years"]):
        problems.append(f"years {meta.get('years')}")
    if sorted(meta.get("curve_years", [])) != sorted(prep["curve_years"]):
        problems.append(f"curve_years {meta.get('curve_years')}")
    if meta.get("prepared", {}).get("sha256") != csv_sha256:
        problems.append("prepared csv sha256")
    if meta.get("labels_modified") or meta.get("clipping") or meta.get("interpolation"):
        problems.append("labels modified/clipped/interpolated")
    if any(int(y) in protocol["data"]["sealed_years"] for y in meta.get("years", [])):
        problems.append("sealed year prepared")
    if problems:
        raise ValueError(f"v29 preparation identity failed: {problems}")
    return True


# --------------------------------------------------------------------------- Guard rule
def guard_prediction(base, last10_anchor, rated, threshold_frac):
    """PersistenceLast10 for all leads where the issue-time 10-minute power < threshold_frac x rated, else the base."""
    base, anchor = np.asarray(base, dtype=np.float64), np.asarray(last10_anchor, dtype=np.float64)
    if base.shape != anchor.shape or base.ndim != 2 or not len(base):
        raise ValueError("Aligned [window, lead] base and anchor arrays required")
    if not (np.isfinite(anchor).all() and np.isfinite(base).all()):
        raise ValueError("Guard needs finite base forecasts and issue-time anchors")
    if not np.array_equal(anchor, np.repeat(anchor[:, :1], anchor.shape[1], axis=1)):
        raise ValueError("PersistenceLast10 must repeat the same issue-time observation across leads")
    fallback = anchor[:, 0] < float(threshold_frac) * float(rated)
    return np.where(fallback[:, None], anchor, base)


# --------------------------------------------------------------------------- validation choice
def validation_choice(seed_maes, p):
    """Lowest seed-mean validation ALL MAE; ties -> fewer parameters (Guard counts its base), then protocol order."""
    rows = []
    for order, arm in enumerate(CHOICE_CANDIDATES):
        maes = [float(v) for v in seed_maes.get(arm, [])]
        if len(maes) != len(p["seeds"]) or not np.isfinite(maes).all():
            raise ValueError(f"Incomplete validation records for {arm}")
        rows.append((float(np.mean(maes)), arm_parameters(p, arm), order, arm))
    rows.sort()
    return {"arm": rows[0][3], "selection_cohort": p["training"]["selection_cohort"],
            "ranking": [{"arm": a, "validation_mae_seed_mean": m, "parameters": n} for m, n, _, a in rows]}


# --------------------------------------------------------------------------- evaluation
def load_registered(directory, p):
    """Registered test cohort and per-seed predictions; the Guard arrays must equal the rule applied to the base."""
    folder = Path(directory) / "evaluation"
    with np.load(folder / "test_cohort.npz", allow_pickle=False) as saved:
        cohort = {k: saved[k] for k in saved.files}
    n, horizon = cohort["truth"].shape
    if horizon != p["data"]["pred_len"] or not n or cohort["times"].shape != (n, horizon):
        raise ValueError("Invalid common test queue dimensions")
    if cohort["normal_mask"].shape != (n,) or cohort["normal_mask"].dtype != bool or not cohort["normal_mask"].any():
        raise ValueError("A nonempty boolean normal cohort is required")
    if len(np.unique(cohort["window_starts"])) != n or cohort["turbines"].shape != (n,):
        raise ValueError("Duplicate or misaligned common windows")
    for key in ISSUE_ARRAYS:
        if cohort[key].shape != (n,) or not np.isfinite(cohort[key]).all():
            raise ValueError(f"Issue-time array {key} missing or non-finite")
    predictions = {arm: [cohort[arm]] for arm in PERSISTENCE_ARMS}
    for arm in SEEDED_ARMS:
        predictions[arm] = []
        for seed in p["seeds"]:
            with np.load(folder / f"{arm}_s{seed}.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["window_starts"], cohort["window_starts"]):
                    raise ValueError(f"{arm} s{seed} test windows differ from the common cohort")
                if saved["prediction"].shape != cohort["truth"].shape or not np.isfinite(saved["prediction"]).all():
                    raise ValueError(f"{arm} s{seed} prediction shape or finiteness differs")
                predictions[arm].append(saved["prediction"])
    guard = p["arms"][GUARD_ARM]
    for seed_index, saved in enumerate(predictions[GUARD_ARM]):
        expected = guard_prediction(predictions[guard["base_arm"]][seed_index], cohort["PersistenceLast10"],
                                    p["data"]["rated_power_kw"], guard["threshold_frac_rated"])
        if not np.array_equal(saved, expected):
            raise ValueError("Registered Guard forecasts differ from the fixed rule applied to the base arm")
    return cohort, predictions


def regime_masks(cohort, rated, bins):
    """Ex-ante regimes from issue-time observations only (anchor level, stopped/derated in the last input hour)."""
    anchor = cohort["PersistenceLast10"][:, 0] / float(rated)
    edges = [-np.inf] + [float(b) for b in bins] + [np.inf]
    masks = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        label = (f"anchor<{hi:g}" if lo == -np.inf else f"anchor>={lo:g}" if hi == np.inf else f"{lo:g}<=anchor<{hi:g}")
        masks[label + "_rated"] = (anchor >= lo) & (anchor < hi)
    masks["stopped_at_issue"] = cohort["issue_stopped_frac"] > 0
    masks["derated_at_issue"] = cohort["issue_capacity_frac"] < DERATED_BELOW
    masks["all_windows"] = np.ones(len(anchor), dtype=bool)
    return masks


def regime_table(cohort, predictions, rated, bins):
    total = len(cohort["truth"])
    rows = []
    for label, mask in regime_masks(cohort, rated, bins).items():
        row = {"regime": label, "windows": int(mask.sum()), "share_pct": float(100 * mask.sum() / total)}
        for arm, preds in predictions.items():
            row[f"{arm}_mae"] = (float(np.mean([np.abs(q[mask] - cohort["truth"][mask]).mean() for q in preds]))
                                 if mask.any() else None)
        rows.append(row)
    return rows


def _significant(comparison):
    return comparison["absolute_error"]["ci95_low"] > 0 and comparison["holm_p"] <= .05


def _positive(comparison):
    return _significant(comparison) and comparison["rmse_difference"]["rmse_gain_kw"] > 0


def _non_inferior(comparison, margin):
    return comparison["absolute_error"]["ci95_low"] > -float(margin)


def rules(cohort_comparisons, margin):
    """Predeclared v29 rules on {cohort: comparisons}; all is the primary (deployment) cohort."""
    everything, normal = cohort_comparisons["all"], cohort_comparisons["normal"]
    return {"state_information_supported": bool(_positive(everything["C1"]) and _non_inferior(normal["C1"], margin)),
            "all_window_training_supported": bool(_positive(everything["C3"]) and _non_inferior(normal["C3"], margin)),
            "learned_state_beats_rule": bool(_significant(everything["C6"])),
            "beats_naive": bool(_significant(everything["C5"]) and _significant(normal["C5"]))}


def evaluate_registered(directory, p, choice):
    """Metrics (both cohorts), C1-C6, rules, ensemble and regimes from the registered test arrays only."""
    if choice not in CHOICE_CANDIDATES:
        raise ValueError(f"Unknown validation choice {choice!r}")
    cohort, predictions = load_registered(directory, p)
    rated, settings = p["data"]["rated_power_kw"], p["statistics"]
    out = {"primary_cohort": "all", "windows": {"all": int(len(cohort["truth"])), "normal": int(cohort["normal_mask"].sum())},
           "normal_share_pct": float(100 * cohort["normal_mask"].mean()), "metrics": {}, "comparisons": {},
           "ensemble": {"arm": choice, "definition": "3-seed mean forecast (descriptive)"}, "regimes": {},
           "validation_choice": choice}
    issue = cohort["times"][:, 0] - HOUR
    for name in COHORTS:
        mask = cohort["normal_mask"].astype(bool) if name == "normal" else np.ones(len(cohort["truth"]), bool)
        truth = cohort["truth"][mask]
        subset = {arm: [q[mask] for q in preds] for arm, preds in predictions.items()}
        out["metrics"][name] = {arm: seed_mean([metrics(q, truth, rated) for q in preds]) for arm, preds in subset.items()}
        for arm, result in out["metrics"][name].items():
            for anchor in PERSISTENCE_ARMS:
                reference = out["metrics"][name][anchor]
                for loss in ("mae", "rmse"):
                    result[f"{loss}_skill_pct_vs_{anchor}"] = (100 * (1 - result[loss] / reference[loss])
                                                               if reference[loss] > 0 else None)
        out["comparisons"][name] = compare_all(subset, truth, issue[mask], p["comparisons"], choice, settings)
        out["ensemble"][name] = metrics(np.mean(subset[choice], axis=0), truth, rated)
        sub_cohort = {k: v[mask] for k, v in cohort.items() if k in ("truth", "PersistenceLast10") + ISSUE_ARRAYS}
        out["regimes"][name] = regime_table(sub_cohort, subset, rated, p["diagnostics"]["anchor_bins_frac_rated"])
    out["rules"] = rules(out["comparisons"], p["rules"]["non_inferiority_margin_kw"])
    return out


# --------------------------------------------------------------------------- post-test manifest (P1)
def post_test_manifest(directory, windows_by_split):
    """Completes the pre-test data_manifest.json (left unmodified) after E1: every split's all/normal counts."""
    directory = Path(directory)
    windows = {split: {"all": int(v["all"]), "normal": int(v["normal"])} for split, v in windows_by_split.items()}
    if set(windows) != {"train", "validation", "test"}:
        raise ValueError("Post-test manifest needs train, validation and test counts")
    with np.load(directory / "evaluation/test_cohort.npz", allow_pickle=False) as saved:
        registered = {"all": int(len(saved["truth"])), "normal": int(saved["normal_mask"].sum())}
    if windows["test"] != registered:
        raise ValueError("Post-freeze test window counts disagree with the registered test cohort")
    pre = read_json(directory / "data_manifest.json")
    if pre.get("test_windows_instantiated") or "test" in pre["series_audit"]["windows"]:
        raise ValueError("The pre-test data manifest must not contain test windows")
    for split in ("train", "validation"):
        if pre["series_audit"]["windows"][split] != windows[split]:
            raise ValueError(f"{split} window counts changed after opening the test year")
    predictions = {path.name: file_sha256(path) for path in sorted((directory / "evaluation").glob("*.npz"))
                   if path.name != "test_cohort.npz"}
    return {"windows": windows,
            "normal_share_pct": {s: 100 * v["normal"] / v["all"] if v["all"] else None for s, v in windows.items()},
            "pre_test_data_manifest_sha256": file_sha256(directory / "data_manifest.json"),
            "selection_freeze_sha256": file_sha256(directory / "selection_freeze.json"),
            "test_cohort_sha256": file_sha256(directory / "evaluation/test_cohort.npz"),
            "test_predictions_sha256": predictions,
            "policy": "data_manifest.json is written at D0 before any test window exists and is never modified; "
                      "this file completes the record after E1"}


# --------------------------------------------------------------------------- report (R1)
def build_report(directory):
    """Recompute everything from the registered arrays, refuse on any disagreement, write tables/figures/report."""
    from scripts import report_kelmarsh_multiyear_benchmark as r28  # frozen v28 helpers (Agg backend)
    directory = Path(directory)
    p = read_json(directory / "protocol.json")["protocol"]
    training = verify_training_completed(directory)
    freeze = read_json(directory / "selection_freeze.json")
    registered = read_json(directory / "decision.json")
    required = ["selection_freeze.json", "decision.json", "post_test_manifest.json", "evaluation/test_cohort.npz"]
    required += [f"evaluation/{arm}_s{seed}.npz" for arm in SEEDED_ARMS for seed in p["seeds"]]
    for name in required:
        if training["hashes"].get(name) != file_sha256(directory / name):
            raise ValueError(f"Registered artifact changed or missing: {name}")
    choice = freeze["validation_choice"]["arm"]
    fresh = evaluate_registered(directory, p, choice)
    r28.check_against_registered(fresh, registered)
    manifest = read_json(directory / "post_test_manifest.json")
    recomputed_manifest = post_test_manifest(directory, manifest["windows"])
    if canonical_sha256(manifest) != canonical_sha256(recomputed_manifest):
        raise ValueError("Post-test manifest hashes or development counts disagree with registered artifacts")
    if manifest["windows"]["test"] != fresh["windows"]:
        raise ValueError("Post-test manifest disagrees with the registered test cohort")
    rows = []
    for cohort in COHORTS:
        entries = [(arm, fresh["metrics"][cohort][arm]) for arm in ARMS]
        entries.append((f"ENSEMBLE({choice})", fresh["ensemble"][cohort] | {"seeds": len(p["seeds"])}))
        for arm, m in entries:
            rows.append({"cohort": cohort, "arm": arm, "seeds": m.get("seeds"), "mae": m["mae"], "mse": m["mse"],
                         "rmse": m["rmse"], "r2": m["r2"], "mbe": m["mbe"], "nmae_pct_rated": m["nmae_pct_rated"],
                         "nrmse_pct_rated": m["nrmse_pct_rated"],
                         **{k: v for k, v in m.items() if "skill_pct" in k or k.startswith("hit_")},
                         **{f"lead{h + 1}_{key}": v for key in ("mae", "mse", "rmse", "r2") for h, v in enumerate(m[f"lead_{key}"])}})
    pd.DataFrame(rows).to_csv(directory / "test_metrics.csv", index=False)
    comparison_rows = []
    for cohort in COHORTS:
        for name, c in sorted(fresh["comparisons"][cohort].items()):
            a, r = c["absolute_error"], c["rmse_difference"]
            comparison_rows.append({"cohort": cohort, "comparison": name, "candidate": c["candidate"],
                                    "reference": c["reference"], "mae_gain_kw": a["gain"], "mae_ci95_low": a["ci95_low"],
                                    "mae_ci95_high": a["ci95_high"], "nw_p": a["newey_west"]["p_value_two_sided"],
                                    "holm_p": c["holm_p"], "rmse_gain_kw": r["rmse_gain_kw"],
                                    "rmse_ci95_low": r["ci95_low_kw"], "rmse_ci95_high": r["ci95_high_kw"]})
    pd.DataFrame(comparison_rows).to_csv(directory / "test_comparisons.csv", index=False)
    regime_rows = [{"cohort": cohort, **row} for cohort in COHORTS for row in fresh["regimes"][cohort]]
    pd.DataFrame(regime_rows).to_csv(directory / "regime_metrics.csv", index=False)
    figures(fresh, directory / "figures")
    r28.forecast_trace(directory, p, choice)
    result = {"protocol_id": p["protocol_id"], "rules": fresh["rules"], "validation_choice": freeze["validation_choice"],
              "windows": fresh["windows"], "normal_share_pct": fresh["normal_share_pct"],
              "post_test_windows": manifest["windows"], "primary_cohort": "all",
              "scope": "single 2022 test evaluation; all = deployment (primary) cohort; normal = ex-post conditional "
                       "cohort; 2021 was the v28 test year and is the v29 validation year; 2023-2024 sealed",
              "independent_recomputation": "all metrics, comparisons, rules, ensemble and regimes recomputed from registered arrays"}
    write_json(directory / "result.json", result)
    write_markdown(directory, p, fresh, freeze, manifest, rows, comparison_rows, r28.table)
    outputs = ["test_metrics.csv", "test_comparisons.csv", "regime_metrics.csv", "result.json", "report.md",
               "figures/test_metrics.png", "figures/lead_mae.png", "figures/forecast_trace.png", "forecast_trace_selection.json"]
    write_json(directory / "report_manifest.json", {
        "report_sources": {n: file_sha256(ROOT / n) for n in REPORT_SOURCES},
        "inputs": {"training_completed.json": file_sha256(directory / "training_completed.json")},
        "outputs": {n: file_sha256(directory / n) for n in outputs},
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    return result


def figures(fresh, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    x = np.arange(len(ARMS))
    for ax, key, label in zip(axes, ("mae", "rmse"), ("MAE (kW)", "RMSE (kW)")):
        for offset, cohort in ((-.2, "all"), (.2, "normal")):
            ax.bar(x + offset, [fresh["metrics"][cohort][arm][key] for arm in ARMS], .4, label=f"test {cohort}")
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("_", "\n") for a in ARMS], fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("v29 test 2022 (single evaluation): all = deployment (primary) cohort, normal = conditional cohort", fontsize=9)
    fig.tight_layout()
    fig.savefig(folder / "test_metrics.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, cohort in zip(axes, COHORTS):
        for arm in ARMS:
            ax.plot([1, 2, 3, 4], fresh["metrics"][cohort][arm]["lead_mae"], marker="o", label=arm,
                    ls="--" if arm.startswith(("Persistence", "Guard")) else "-")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("lead (hours)")
        ax.set_title(f"test {cohort}: MAE by lead (kW)", fontsize=10)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(folder / "lead_mae.png", dpi=130)
    plt.close(fig)


def write_markdown(directory, p, fresh, freeze, manifest, rows, comparison_rows, table):
    validation = [{"arm": r["arm"], "seed": r["seed"], "best_epoch": r.get("best_epoch"), "train_cohort": r["train_cohort"],
                   **{f"val_{k}": v for k, v in r["validation"].items()}} for r in freeze["records"]]
    regimes = [row for row in fresh["regimes"]["all"]]
    regime_columns = ["regime", "windows", "share_pct"] + [f"{arm}_mae" for arm in ARMS]
    counts = ", ".join(f"{s} {v['all']}/{v['normal']}" for s, v in manifest["windows"].items())
    choice = freeze["validation_choice"]["arm"]
    lines = [f"# v29 Kelmarsh operating-state report ({dt.date.today().isoformat()})", "",
             f"**Rules:** {fresh['rules']}", "",
             f"Validation-chosen configuration (2021, ALL cohort, frozen before test): **{choice}**", "",
             f"Test 2022 windows: all {fresh['windows']['all']}, normal {fresh['windows']['normal']} "
             f"({fresh['normal_share_pct']:.2f}%). Every arm uses the same windows. In v29 the ALL cohort is primary "
             "(deployment accuracy); the normal cohort uses the target period's operating state and is conditional accuracy only.", "",
             f"Post-test manifest (all/normal windows): {counts}.", "",
             p["rules"].get("attribution_limits", ""), "",
             "Skill is reported against both Persistence anchors; capacity hit rates are tolerance statistics, not classification "
             "accuracy. Trained-arm rows average per-seed metrics; the ENSEMBLE row averages the three seed forecasts (descriptive).", "",
             "## Test metrics", "",
             table(rows, ["cohort", "arm", "mae", "rmse", "r2", "mbe", "nmae_pct_rated", "lead1_mae", "lead4_mae", "lead1_r2", "lead4_r2"]), "",
             "## Predeclared comparisons (positive gain favours the candidate; Holm over C1-C6 within each cohort)", "",
             table(comparison_rows, ["cohort", "comparison", "candidate", "reference", "mae_gain_kw", "mae_ci95_low", "mae_ci95_high",
                                     "nw_p", "holm_p", "rmse_gain_kw", "rmse_ci95_low", "rmse_ci95_high"]), "",
             "## Ex-ante regimes, ALL cohort (seed-mean MAE, kW; descriptive)", "",
             table(regimes, regime_columns, digits=1), "",
             "## Validation (2021, ALL cohort) selected states", "",
             table(validation, ["arm", "seed", "best_epoch", "train_cohort", "val_mae", "val_rmse"]), "",
             "![test metrics](figures/test_metrics.png)", "", "![lead MAE](figures/lead_mae.png)", "",
             "![first eligible forecast segment, not selected by error](figures/forecast_trace.png)", ""]
    (Path(directory) / "report.md").write_text("\n".join(lines), encoding="utf-8")
