"""Torch-free helpers for v32: confirmation of the frozen v31 design on the last sealed Kelmarsh year (2024).

Everything that does not depend on the protocol id or the years is reused unchanged from the frozen v31 module
(station parsing, upstream features, the composed series, validation choice, evaluation, farm metrics, rules). This
module only adds the v32 protocol/preparation checks and a report whose year labels come from the protocol splits.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from utils import kelmarsh_upstream_v31 as base
from utils.kelmarsh_upstream_v31 import (  # noqa: F401  (re-exported, unchanged)
    ARMS, CHANNEL_SETS, CHANNELS, COHORTS, COMPARISONS, ISSUE_ARRAYS, MISSING_CHANNEL, PERSISTENCE_ARMS, REFERENCE_ARM,
    SCADA_CHANNELS, TRAINED_ARMS, UPSTREAM_FEATURES, KelmarshUpstreamSeries, acquire_exclusive_lock, append_event,
    arm_config, arm_parameters, arm_protocol, canonical_sha256, compare_all, evaluate_registered, expected_parameters,
    farm_metrics, file_sha256, hourly_upstream, load_registered, load_station_observations, manifest_hashes, metrics,
    parse_station_file, post_test_manifest, read_json, regime_table, rules, safe, seed_mean, station_files,
    text_sha256_lf, validation_choice, verify_development_rows, verify_power_curves, verify_registered_files,
    verify_training_completed, window_hash, write_json)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "kelmarsh_upstream_v32"
PREPARATION_KEY = "v32_preparation"
REFERENCE_KEY = "v31_registered_preparation"
PREPARER_SOURCES = ("scripts/prepare_kelmarsh_multiyear_normalized.py", "scripts/prepare_kelmarsh_multiyear_v31.py")
REPORT_SOURCES = ("utils/kelmarsh_upstream_v32.py", "utils/kelmarsh_upstream_v31.py",
                  "scripts/report_kelmarsh_multiyear_benchmark.py", "utils/kelmarsh_multiyear_benchmark.py",
                  "utils/kelmarsh_backbone_information.py", "utils/paired_forecast_statistics.py",
                  "utils/kelmarsh_operating_state.py")


def years_of(p):
    s = p["data"]["splits"]
    return {"train": list(s["train"]), "validation": int(s["validation"][0]), "test": int(s["test"][0])}


# --------------------------------------------------------------------------- protocol
def validate_protocol(p):
    d, arms, up = p["data"], p["arms"], p["data"]["upstream"]
    if p["protocol_id"] != PROTOCOL_ID:
        raise ValueError("Unexpected protocol id")
    if tuple(arms) != ARMS:
        raise ValueError("Arm list departs from the frozen design")
    if (d["input_len"], d["pred_len"], d["stride"]) != (24, 4, 3):
        raise ValueError("Window design departs from 24 -> 4, stride 3")
    if d["channels"] != CHANNELS or {k: list(v) for k, v in d["channel_sets"].items()} != CHANNEL_SETS:
        raise ValueError("Channel design departs from the frozen design")
    if d["extras_in_common_series"] != SCADA_CHANNELS[6:] or up["features"] != UPSTREAM_FEATURES:
        raise ValueError("Extras or upstream features depart from the frozen design")
    splits, prep, ref = d["splits"], d[PREPARATION_KEY], d[REFERENCE_KEY]
    if splits != {"train": list(range(2016, 2023)), "validation": [2023], "test": [2024]} or d["sealed_years"]:
        raise ValueError("v32 confirms on 2024 after training 2016-2022 and validating on 2023; nothing else is sealed")
    years = set(splits["train"]) | set(splits["validation"]) | set(splits["test"])
    if set(prep["years"]) != years or not set(prep["curve_years"]) <= set(splits["train"]):
        raise ValueError("Prepared years must be exactly train+validation+test; curve years within train")
    if prep["curve_years"] != [2016, 2017, 2018, 2019] or prep["duplicate_normalization"] != "existing_dominating_row_v1":
        raise ValueError("Curve years and duplicate normalization are frozen from v31")
    if set(ref["years"]) != years - set(splits["test"]):
        raise ValueError("The registered v31 reference must cover exactly the non-test years")
    development = sorted(y for root in up["development_roots"] for y in root["years"])
    if development != sorted(up["development_years"]) or set(development) != years - set(splits["test"]):
        raise ValueError("Upstream development roots must cover exactly the non-test years once")
    if set(up["test_years"]) != set(splits["test"]) or int(up["lag_hours"]) != 1:
        raise ValueError("Upstream test years and the 1 h lag are frozen")
    ids = [int(s["src_id"]) for s in up["stations"]]
    if len(ids) != 36 or len(set(ids)) != len(ids):
        raise ValueError("The 36 screened stations are frozen")
    if d.get("yaw_error_channels_enabled") or any(c.startswith("yaw") for c in d["channels"]):
        raise ValueError("Yaw-error channels are disabled")
    for arm in TRAINED_ARMS:
        spec = arms[arm]
        if spec["family"] != "TSMixerBasicSmall" or spec["channels"] not in CHANNEL_SETS or spec["train_cohort"] != "all":
            raise ValueError(f"{arm}: family, channel set or training cohort departs from the design")
        declared = p["families"][spec["family"]]["parameters"].get(spec["channels"])
        if declared != expected_parameters(spec["family"], len(CHANNEL_SETS[spec["channels"]])):
            raise ValueError(f"{arm}: declared parameters {declared} != formula")
    if [a for a in TRAINED_ARMS if arms[a].get("training_overrides")] != ["TSMixer_upstream_all_balanced"] or \
            arms["TSMixer_upstream_all_balanced"]["training_overrides"] != {"loss_mae_weight": 0.5, "loss_mse_weight": 0.5}:
        raise ValueError("Only the balanced arm changes the loss, to 0.5/0.5")
    if not (p["cohorts"]["primary"] == p["cohorts"]["selection_cohort"] == p["training"]["selection_cohort"] == "all"):
        raise ValueError("v32 trains, selects and decides on the ALL cohort")
    budget = p["budget"]
    if budget["fits"] != len(TRAINED_ARMS) * len(p["seeds"]) or budget["fits"] > budget["new_fits_max"] or \
            budget["fits"] * p["training"]["max_epochs"] > budget["new_epochs_max"]:
        raise ValueError("Budget does not match arms x seeds x epochs")
    if budget.get("additional_seeds") or budget.get("additional_arms") or budget.get("hyperparameter_search"):
        raise ValueError("No search, extra seeds or extra arms in v32")
    if tuple(sorted(p["comparisons"])) != COMPARISONS or p["comparisons"]["C3"]["candidate"] != "VALIDATION_CHOSEN":
        raise ValueError("Comparison family departs from the frozen design")
    if float(p["rules"]["non_inferiority_margin_kw"]) != 2.0:
        raise ValueError("The 2 kW non-inferiority margin is frozen from v31")
    return True


def verify_preparation(meta, protocol, csv_sha256):
    prep = protocol["data"][PREPARATION_KEY]
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
    normalization = meta.get("duplicate_normalization", {})
    units = normalization.get("units", [])
    if normalization.get("version") != prep["duplicate_normalization"] or normalization.get("turbine_id_source") != "preparer unit loop":
        problems.append("duplicate normalization version or turbine-id source")
    if sorted(u.get("turbine_id") for u in units) != list(range(1, 7)):
        problems.append("duplicate normalization turbine identities")
    if any(u.get("cells_filled") != 0 or u.get("conflicting_nonmissing_values") != 0
           or not all(u.get(k) for k in ("timestamps_preserved", "nonmissing_values_preserved", "selected_existing_whole_rows"))
           for u in units):
        problems.append("duplicate normalization preservation evidence")
    for source in PREPARER_SOURCES:
        if meta.get("source_sha256", {}).get(source) != file_sha256(ROOT / source):
            problems.append(f"preparer source hash {source}")
    if problems:
        raise ValueError(f"v32 preparation identity failed: {problems}")
    return True


# --------------------------------------------------------------------------- report (R1), year labels from the protocol
def build_report(directory):
    from scripts import report_kelmarsh_multiyear_benchmark as r28  # frozen v28 helpers (Agg backend)
    directory = Path(directory)
    p = read_json(directory / "protocol.json")["protocol"]
    years = years_of(p)
    training = verify_training_completed(directory)
    freeze = read_json(directory / "selection_freeze.json")
    registered = read_json(directory / "decision.json")
    required = ["selection_freeze.json", "decision.json", "post_test_manifest.json", "evaluation/test_cohort.npz",
                "evaluation/upstream_test_files.json"]
    required += [f"evaluation/{arm}_s{seed}.npz" for arm in TRAINED_ARMS for seed in p["seeds"]]
    for name in required:
        if training["hashes"].get(name) != file_sha256(directory / name):
            raise ValueError(f"Registered artifact changed or missing: {name}")
    choice = freeze["validation_choice"]["arm"]
    fresh = evaluate_registered(directory, p, choice)
    r28.check_against_registered(fresh, registered)
    manifest = read_json(directory / "post_test_manifest.json")
    if manifest["windows"]["test"] != fresh["windows"]:
        raise ValueError("Post-test manifest disagrees with the registered test cohort")
    for key, name in (("selection_freeze_sha256", "selection_freeze.json"), ("test_cohort_sha256", "evaluation/test_cohort.npz"),
                      ("pre_test_data_manifest_sha256", "data_manifest.json")):
        if manifest[key] != file_sha256(directory / name):
            raise ValueError(f"Post-test manifest link {key} is stale")
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
            comparison_rows.append({"cohort": cohort, "comparison": name, "candidate": c["candidate"], "reference": c["reference"],
                                    "mae_gain_kw": a["gain"], "mae_ci95_low": a["ci95_low"], "mae_ci95_high": a["ci95_high"],
                                    "nw_p": a["newey_west"]["p_value_two_sided"], "holm_p": c["holm_p"],
                                    "rmse_gain_kw": r["rmse_gain_kw"], "rmse_ci95_low": r["ci95_low_kw"], "rmse_ci95_high": r["ci95_high_kw"]})
    pd.DataFrame(comparison_rows).to_csv(directory / "test_comparisons.csv", index=False)
    farm_rows = [{"arm": arm, "nrmse_pct": f["nrmse_pct"], "accuracy_pct": f["accuracy_pct"],
                  **{f"lead{h + 1}_nrmse_pct": v for h, v in enumerate(f["lead_nrmse_pct"])},
                  **{f"lead{h + 1}_accuracy_pct": v for h, v in enumerate(f["lead_accuracy_pct"])},
                  **{f"lead{h + 1}_mae_kw": v for h, v in enumerate(f["lead_mae_kw"])}} for arm, f in fresh["farm"]["arms"].items()]
    pd.DataFrame(farm_rows).to_csv(directory / "farm_metrics.csv", index=False)
    pd.DataFrame([{"cohort": c, **row} for c in COHORTS for row in fresh["regimes"][c]]).to_csv(directory / "regime_metrics.csv", index=False)
    figures(fresh, directory / "figures", p)
    r28.forecast_trace(directory, p, choice)
    result = {"protocol_id": p["protocol_id"], "rules": fresh["rules"], "validation_choice": freeze["validation_choice"],
              "windows": fresh["windows"], "normal_share_pct": fresh["normal_share_pct"], "post_test_windows": manifest["windows"],
              "primary_cohort": "all", "farm_accuracy_pct": {a: f["accuracy_pct"] for a, f in fresh["farm"]["arms"].items()},
              "scope": f"single {years['test']} test evaluation; all = deployment (primary) cohort; normal = ex-post conditional cohort; "
                       f"{years['validation']} was the v31 test year and is the v32 validation year; no sealed Kelmarsh year remains",
              "independent_recomputation": "metrics, comparisons, rules, ensemble, farm metrics and regimes recomputed from registered arrays"}
    write_json(directory / "result.json", result)
    write_markdown(directory, p, fresh, freeze, manifest, rows, comparison_rows, farm_rows, r28.table)
    outputs = ["test_metrics.csv", "test_comparisons.csv", "farm_metrics.csv", "regime_metrics.csv", "result.json", "report.md",
               "figures/test_metrics.png", "figures/lead_mae.png", "figures/forecast_trace.png", "forecast_trace_selection.json"]
    write_json(directory / "report_manifest.json", {
        "report_sources": {n: file_sha256(ROOT / n) for n in REPORT_SOURCES},
        "inputs": {"training_completed.json": file_sha256(directory / "training_completed.json")},
        "outputs": {n: file_sha256(directory / n) for n in outputs},
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    return result


def figures(fresh, folder, p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    years = years_of(p)
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(ARMS))
    for ax, key, label in zip(axes, ("mae", "rmse"), ("MAE (kW)", "RMSE (kW)")):
        for offset, cohort in ((-.2, "all"), (.2, "normal")):
            ax.bar(x + offset, [fresh["metrics"][cohort][arm][key] for arm in ARMS], .4, label=f"test {cohort}")
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("_", "\n") for a in ARMS], fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{p['protocol_id']} test {years['test']} (single evaluation): all = deployment (primary) cohort, "
                 "normal = conditional cohort", fontsize=9)
    fig.tight_layout()
    fig.savefig(folder / "test_metrics.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, cohort in zip(axes, COHORTS):
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


def write_markdown(directory, p, fresh, freeze, manifest, rows, comparison_rows, farm_rows, table):
    years = years_of(p)
    validation = [{"arm": r["arm"], "seed": r["seed"], "best_epoch": r.get("best_epoch"),
                   **{f"val_{k}": v for k, v in r["validation"].items()}} for r in freeze["records"]]
    counts = ", ".join(f"{s} {v['all']}/{v['normal']}" for s, v in manifest["windows"].items())
    choice = freeze["validation_choice"]["arm"]
    lines = [f"# {p['protocol_id']} Kelmarsh upstream-observation confirmation report ({dt.date.today().isoformat()})", "",
             f"**Rules:** {fresh['rules']}", "",
             f"Validation-chosen configuration ({years['validation']}, ALL cohort, joint MAE+RMSE score, frozen before test): **{choice}**", "",
             f"Test {years['test']} windows: all {fresh['windows']['all']}, normal {fresh['windows']['normal']} "
             f"({fresh['normal_share_pct']:.2f}%). Every arm uses the same windows. The ALL cohort is primary (deployment "
             "accuracy); the normal cohort uses the target period's operating state and is conditional accuracy only.", "",
             f"Post-test manifest (all/normal windows): {counts}.", "",
             p["rules"].get("attribution_limits", ""), "",
             "Upstream observations are archival MIDAS Open values used with an assumed 1 h availability lag. Trained-arm rows "
             "average per-seed metrics; the ENSEMBLE row averages the three seed forecasts (descriptive).", "",
             "## Test metrics (per turbine)", "",
             table(rows, ["cohort", "arm", "mae", "rmse", "r2", "mbe", "nmae_pct_rated", "lead1_mae", "lead4_mae", "lead1_r2", "lead4_r2"]), "",
             "## Predeclared comparisons (positive gain favours the candidate; Holm over C1-C3 within each cohort)", "",
             table(comparison_rows, ["cohort", "comparison", "candidate", "reference", "mae_gain_kw", "mae_ci95_low", "mae_ci95_high",
                                     "nw_p", "holm_p", "rmse_gain_kw", "rmse_ci95_low", "rmse_ci95_high"]), "",
             f"## Farm total (six turbines, {fresh['farm']['issue_times']} issue times; descriptive)", "",
             "Normalized by 6 x 2050 kW; accuracy = 100% - normalized RMSE. Reference only to the NEA 2011 real-time 15% guideline "
             "(15-minute resolution, opening capacity), not a compliance claim.", "",
             table(farm_rows, ["arm", "nrmse_pct", "accuracy_pct", "lead1_nrmse_pct", "lead2_nrmse_pct", "lead3_nrmse_pct",
                               "lead4_nrmse_pct", "lead4_accuracy_pct"], digits=2), "",
             f"## Validation ({years['validation']}, ALL cohort) selected states", "",
             table(validation, ["arm", "seed", "best_epoch", "val_mae", "val_rmse"]), "",
             "![test metrics](figures/test_metrics.png)", "", "![lead MAE](figures/lead_mae.png)", "",
             "![first eligible forecast segment, not selected by error](figures/forecast_trace.png)", ""]
    (Path(directory) / "report.md").write_text("\n".join(lines), encoding="utf-8")
