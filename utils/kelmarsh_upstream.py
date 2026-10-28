"""Torch-free helpers for v30 (Kelmarsh + upstream MIDAS Open station observations, test year 2023).

The upstream group is the eleven screened features of ``scripts/screen_kelmarsh_upstream_observations.upstream_features``
(screen passed 2026-10-28 on 2021/2022), evaluated for EVERY hourly label tau of the series from the observations at
tau - 1 h (and tau - 4 h for 3-hour changes) of the 36 stations frozen in the protocol. Station files are parsed with the
screen's rules (version_num 1, declared units, knots -> m/s, conflicting duplicates refused, row year = file year) and
must match the SHA256 of the registered download manifest; the 2023 files are opened only inside E1, after the S1
freeze, and 2024 files never. A missing upstream INPUT (never a label) is replaced by its training-year mean and flagged
in channel ``upstream_missing``, so the window queue is exactly the SCADA-only queue of the frozen v28 provider.

Generic metrics, paired statistics, hashing and the P1 post-test manifest are reused unchanged from the frozen v28/v29
modules; the screen script is imported read-only and never modified.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd

from data_provider.kelmarsh_multiyear import COHORTS as PROVIDER_COHORTS
from data_provider.kelmarsh_multiyear import KelmarshMultiYearSeries, MultiYearWindows
from scripts import screen_kelmarsh_upstream_observations as screen
from utils.kelmarsh_multiyear_benchmark import (  # noqa: F401  (re-exported for the controller and tests)
    acquire_exclusive_lock, append_event, canonical_sha256, choose_pair_device, compare_all, contract_mismatches,
    expected_parameters, file_sha256, metrics, read_json, safe, seed_mean, text_sha256_lf, verify_development_rows,
    verify_power_curves, verify_registered_files, verify_training_completed, window_hash, write_json)
from utils.kelmarsh_operating_state import post_test_manifest  # noqa: F401  (generic P1, frozen v29 module)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "kelmarsh_upstream_v30"
PERSISTENCE_ARMS = ("PersistenceHour", "PersistenceLast10")
REFERENCE_ARM = "TSMixer_last10_all"
TRAINED_ARMS = ("TSMixer_last10_all", "TSMixer_upstream_all", "TSMixer_upstream_all_balanced")
ARMS = PERSISTENCE_ARMS + TRAINED_ARMS
COHORTS = ("all", "normal")
COMPARISONS = ("C1", "C2", "C3")
SCADA_CHANNELS = ["Wspd", "Etmp", "Ndir_sin", "Ndir_cos", "Pab_mean", "rotor_speed", "power_last10", "wind_last10"]
UPSTREAM_FEATURES = ["upwind_ws", "upwind_dws3", "upwind_minus_local", "local_ws", "local_dws3", "regional_ws",
                     "regional_dws3", "local_dp3", "grad_east", "grad_north", "grad_magnitude_change3"]
MISSING_CHANNEL = "upstream_missing"
CHANNELS = SCADA_CHANNELS + UPSTREAM_FEATURES + [MISSING_CHANNEL, "power"]
CHANNEL_SETS = {"last10": [0, 1, 2, 3, 4, 5, 6, 7, 20], "upstream": list(range(21))}
ISSUE_ARRAYS = ("issue_upstream_missing",)
HOUR = np.timedelta64(1, "h")
WIND_FACTORS = {"knots": 1852 / 3600, "knot": 1852 / 3600, "kt": 1852 / 3600, "m/s": 1.0, "m s-1": 1.0, "ms-1": 1.0}
DIRECTION_UNITS = {"degrees", "degree", "degrees true", "degrees_true", "deg true"}
PRESSURE_UNITS = {"hpa", "mb", "millibars"}
REPORT_SOURCES = ("utils/kelmarsh_upstream.py", "scripts/report_kelmarsh_multiyear_benchmark.py",
                  "utils/kelmarsh_multiyear_benchmark.py", "utils/kelmarsh_backbone_information.py",
                  "utils/paired_forecast_statistics.py", "utils/kelmarsh_operating_state.py")


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
    splits, prep, ref = d["splits"], d["v30_preparation"], d["v29_registered_preparation"]
    if not (max(splits["train"]) < min(splits["validation"]) <= max(splits["validation"]) < min(splits["test"])
            <= max(splits["test"]) < min(d["sealed_years"])):
        raise ValueError("Years must be ordered train < validation < test < sealed")
    years = set(splits["train"]) | set(splits["validation"]) | set(splits["test"])
    if set(prep["years"]) != years or set(prep["years"]) & set(d["sealed_years"]):
        raise ValueError("Prepared years must be exactly train+validation+test and never sealed")
    if not set(prep["curve_years"]) <= set(splits["train"]):
        raise ValueError("Power-curve years must belong to train")
    if set(ref["years"]) != years - set(splits["test"]):
        raise ValueError("The registered v29 reference must cover exactly the non-test years")
    if set(up["development_years"]) != years - set(splits["test"]) or set(up["test_years"]) != set(splits["test"]):
        raise ValueError("Upstream development/test years must mirror the Kelmarsh splits")
    if int(up["lag_hours"]) < 1:
        raise ValueError("The upstream observations must lag the issue time by at least one hour")
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
        overrides = spec.get("training_overrides", {})
        if set(overrides) - {"loss_mae_weight", "loss_mse_weight"}:
            raise ValueError(f"{arm}: only the loss weights may be overridden")
        weights = {**{k: p["training"][k] for k in ("loss_mae_weight", "loss_mse_weight")}, **overrides}
        if abs(weights["loss_mae_weight"] + weights["loss_mse_weight"] - 1) > 1e-12:
            raise ValueError(f"{arm}: loss weights must sum to one")
    if [a for a in TRAINED_ARMS if arms[a].get("training_overrides")] != ["TSMixer_upstream_all_balanced"]:
        raise ValueError("Only the balanced arm changes the loss")
    if not (p["cohorts"]["primary"] == p["cohorts"]["selection_cohort"] == p["training"]["selection_cohort"] == "all"):
        raise ValueError("v30 trains, selects and decides on the ALL cohort")
    budget = p["budget"]
    if budget["fits"] != len(TRAINED_ARMS) * len(p["seeds"]) or budget["fits"] > budget["new_fits_max"]:
        raise ValueError("Budget does not match arms x seeds")
    if budget["fits"] * p["training"]["max_epochs"] > budget["new_epochs_max"]:
        raise ValueError("Epoch budget exceeded")
    if budget.get("additional_seeds") or budget.get("additional_arms") or budget.get("hyperparameter_search"):
        raise ValueError("No search, extra seeds or extra arms in v30")
    if tuple(sorted(p["comparisons"])) != COMPARISONS or p["comparisons"]["C3"]["candidate"] != "VALIDATION_CHOSEN":
        raise ValueError("Comparison family departs from the frozen design")
    for name, spec in p["comparisons"].items():
        if spec["reference"] not in ARMS or spec["candidate"] not in ARMS + ("VALIDATION_CHOSEN",):
            raise ValueError(f"{name}: unknown arm")
    if not float(p["rules"]["non_inferiority_margin_kw"]) > 0:
        raise ValueError("A positive non-inferiority margin is required")
    bins = p["diagnostics"]["anchor_bins_frac_rated"]
    if not bins or not 0 < bins[0] or any(not a < c <= 1 for a, c in zip(bins, bins[1:])):
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


def arm_protocol(p, arm):
    """The protocol as seen by the unchanged v28 fit_config for this arm (only the loss weights may differ)."""
    overrides = p["arms"][arm].get("training_overrides", {})
    return {**p, "training": {**p["training"], **overrides}} if overrides else p


def arm_parameters(p, arm):
    return int(arm_config(p, arm)["expected_parameters"]) if arm in TRAINED_ARMS else 0


def verify_preparation(meta, protocol, csv_sha256):
    prep = protocol["data"]["v30_preparation"]
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
        raise ValueError(f"v30 preparation identity failed: {problems}")
    return True


# --------------------------------------------------------------------------- MIDAS Open station observations
def manifest_hashes(manifest):
    """{file name: sha256} for every registered station file; names are unique (station and year in the name)."""
    out = {}
    for entry in manifest["files"]:
        name = Path(entry["path"]).name
        if name in out and out[name] != entry["sha256"]:
            raise ValueError(f"Conflicting manifest entries for {name}")
        out[name] = entry["sha256"]
    return out


def station_files(root, station_ids, years, hashes, required=True):
    """Exactly one '<...>_qcv-1_<year>.csv' per station and year under root/<src_id>/, SHA256 equal to the manifest."""
    found, missing = [], []
    for sid in sorted(station_ids):
        for year in sorted(years):
            paths = sorted((Path(root) / str(sid)).glob(f"*_qcv-1_{year}.csv"))
            if len(paths) > 1:
                raise ValueError(f"Several files for station {sid} year {year}")
            if not paths:
                missing.append({"src_id": sid, "year": year})
                continue
            path = paths[0]
            digest = file_sha256(path)
            if hashes.get(path.name) != digest:
                raise ValueError(f"{path.name}: SHA256 differs from the registered download manifest")
            found.append({"src_id": sid, "year": year, "path": path, "sha256": digest})
    if required and missing:
        raise ValueError(f"Missing development station files: {missing[:5]}")
    return found, missing


def parse_station_file(path, year, station_ids):
    """One MIDAS Open hourly file with the screen's rules (the screen script itself only opens 2016-2022)."""
    data = screen.read_badc_csv(path)
    units = dict(data.attrs.get("units", {}))
    version = screen.column(data, "version_num", required=False)
    if version is None:
        raise ValueError(f"{path}: no MIDAS version_num; refusing unidentified observation versions")
    release = re.search(r"dv-(\d{6})", Path(path).name)
    if release and release.group(1) != screen.DATASET["version"]:
        raise ValueError(f"{path}: observation release differs from the reviewed MIDAS version")
    data = data[pd.to_numeric(data[version], errors="coerce") == 1].copy()
    speed = screen.column(data, "wind_speed", "mean_wind_speed")
    factor = WIND_FACTORS.get(units.get(speed))
    if factor is None:
        raise ValueError(f"{path}: missing or unsupported declared wind-speed unit {units.get(speed)!r}")
    direction = screen.column(data, "wind_direction", "mean_wind_dir")
    if units.get(direction) not in DIRECTION_UNITS:
        raise ValueError(f"{path}: unsupported or missing wind-direction unit")
    pressure = screen.column(data, "msl_pressure", required=False)
    if pressure and units.get(pressure) not in PRESSURE_UNITS:
        raise ValueError(f"{path}: unsupported or missing pressure unit")
    sid = pd.to_numeric(data[screen.column(data, "src_id")], errors="coerce")
    keep = sid.isin(station_ids)
    frame = pd.DataFrame({"src_id": sid[keep].astype(int),
                          "time": pd.to_datetime(data.loc[keep, screen.column(data, "ob_time", "ob_end_time")], utc=True,
                                                 errors="coerce").dt.tz_localize(None),
                          "wind_speed": pd.to_numeric(data.loc[keep, speed], errors="coerce") * factor,
                          "wind_direction": pd.to_numeric(data.loc[keep, direction], errors="coerce"),
                          "msl_pressure": (pd.to_numeric(data.loc[keep, pressure], errors="coerce") if pressure else np.nan)})
    if (frame["time"].dropna().dt.year != year).any():
        raise ValueError(f"{path}: row year differs from the filename; no silent filtering")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame.loc[frame.wind_speed < 0, "wind_speed"] = np.nan
    frame.loc[(frame.wind_direction < 0) | (frame.wind_direction > 360) | (frame.wind_speed == 0), "wind_direction"] = np.nan
    frame.loc[frame.msl_pressure <= 0, "msl_pressure"] = np.nan
    return frame[frame["time"].notna() & (frame["time"].dt.minute == 0) & (frame["time"].dt.second == 0)]


def load_station_observations(files, station_ids):
    """{src_id: hourly frame indexed by naive-UTC ob_time}; conflicting duplicates are refused."""
    parts = {}
    for entry in files:
        frame = parse_station_file(entry["path"], int(entry["year"]), station_ids)
        for sid, group in frame.groupby("src_id"):
            parts.setdefault(int(sid), []).append(group.drop(columns="src_id"))
    out = {}
    values = ["wind_speed", "wind_direction", "msl_pressure"]
    for sid in station_ids:
        if sid not in parts:
            raise ValueError(f"Station {sid}: no observations")
        rows = pd.concat(parts[sid])
        if (rows.groupby("time")[values].nunique(dropna=False) > 1).any().any():
            raise ValueError(f"Station {sid}: conflicting duplicate observations")
        out[int(sid)] = rows.drop_duplicates("time").set_index("time").sort_index()
    return out


def hourly_upstream(observations, stations, labels, lag):
    """Feature row for every hour-end label tau from observations at tau - lag (screen definitions)."""
    labels = pd.DatetimeIndex(labels)
    features, names = screen.upstream_features(labels, observations, stations, int(lag))
    if names != UPSTREAM_FEATURES:
        raise ValueError("The screen feature definitions changed")
    return pd.DataFrame(features, index=labels, columns=names)


# --------------------------------------------------------------------------- series
class KelmarshUpstreamSeries:
    """Frozen v28 provider series (SCADA + last10, its window queue unchanged) with the upstream channels and the
    missing flag inserted before power. Exposes the attributes MultiYearWindows reads."""

    def __init__(self, path_or_frame, upstream, *, splits, input_len=24, horizon=4, stride=3,
                 extras=("power_last10", "wind_last10"), start=None, end=None, preparation=None):
        base = KelmarshMultiYearSeries(path_or_frame, splits=splits, input_len=input_len, horizon=horizon, stride=stride,
                                       extras=tuple(extras), start=start, end=end, preparation=preparation)
        if list(upstream.columns) != UPSTREAM_FEATURES:
            raise ValueError("Upstream frame columns depart from the frozen features")
        if upstream.index.has_duplicates:
            raise ValueError("Duplicate upstream labels")
        features = upstream.reindex(pd.DatetimeIndex(base.dates)).to_numpy(dtype=float)
        train_rows = np.isin(base.label_year, base.splits["train"])
        if not np.isfinite(features[train_rows]).any(axis=0).all():
            raise ValueError("An upstream feature has no finite training-year value")
        fill = np.array([np.nanmean(column[np.isfinite(column)]) for column in features[train_rows].T])
        missing = ~np.isfinite(features).all(axis=1)
        features = np.where(np.isfinite(features), features, fill[None, :])
        raw = np.column_stack([base.raw[:, :-1], features, missing.astype(float), base.raw[:, -1]])
        with np.errstate(invalid="ignore"):
            mean, std = np.nanmean(raw[train_rows], axis=0), np.nanstd(raw[train_rows], axis=0)
        scale = np.where(std == 0, 1.0, std)
        scada = list(range(base.raw.shape[1] - 1)) + [raw.shape[1] - 1]
        base_scada = list(range(base.raw.shape[1]))
        if not (np.allclose(mean[scada], base.scaler_mean[base_scada], rtol=0, atol=1e-9)
                and np.allclose(scale[scada], base.scaler_scale[base_scada], rtol=0, atol=1e-9)):
            raise ValueError("SCADA scaler differs from the frozen provider")
        self.base = base
        self.channels = base.channels[:-1] + UPSTREAM_FEATURES + [MISSING_CHANNEL, "power"]
        self.raw, self.scaler_mean, self.scaler_scale = raw, mean, scale
        self.values = ((raw - mean) / scale).astype(np.float32)
        for name in ("dates", "turbines", "normal_hour", "label_year", "power_last10", "input_len", "horizon", "splits",
                     "starts", "split_of", "normal_window"):
            setattr(self, name, getattr(base, name))
        shares = {name: float(missing[np.isin(self.label_year, years)].mean()) for name, years in self.splits.items()}
        self.audit = {**base.audit, "channels": self.channels, "scaler_mean": mean.tolist(), "scaler_scale": scale.tolist(),
                      "upstream": {"features": UPSTREAM_FEATURES, "fill_values_training_mean": fill.tolist(),
                                   "missing_row_share_by_split": shares,
                                   "missing_policy": "input-only training-mean fill plus upstream_missing flag; labels untouched",
                                   "window_queue": "identical to the frozen provider queue"}}

    def windows(self, split, cohort="all"):
        if split not in self.splits:
            raise ValueError(f"Unknown split {split!r}")
        if cohort not in PROVIDER_COHORTS:
            raise ValueError(f"Cohort must be one of {PROVIDER_COHORTS}")
        mask = self.split_of == split
        if cohort == "normal":
            mask &= self.normal_window
        if not mask.any():
            raise ValueError(f"No {cohort} windows in {split}")
        return MultiYearWindows(self, self.starts[mask], split, cohort)


# --------------------------------------------------------------------------- validation choice
def validation_choice(records, p):
    """Lowest joint score = mean MAE / reference MAE + mean RMSE / reference RMSE; ties: fewer parameters, order."""
    means = {}
    for arm in TRAINED_ARMS:
        rows = [r["validation"] for r in records if r["arm"] == arm]
        if len(rows) != len(p["seeds"]) or not all(np.isfinite([r["mae"], r["rmse"]]).all() for r in rows):
            raise ValueError(f"Incomplete validation records for {arm}")
        means[arm] = (float(np.mean([r["mae"] for r in rows])), float(np.mean([r["rmse"] for r in rows])))
    ref_mae, ref_rmse = means[REFERENCE_ARM]
    ranking = sorted((m / ref_mae + r / ref_rmse, arm_parameters(p, arm), order, arm, m, r)
                     for order, (arm, (m, r)) in enumerate(means.items()))
    return {"arm": ranking[0][3], "selection_cohort": p["training"]["selection_cohort"], "reference": REFERENCE_ARM,
            "ranking": [{"arm": a, "joint_score": s, "validation_mae_seed_mean": m, "validation_rmse_seed_mean": r,
                         "parameters": n} for s, n, _, a, m, r in ranking]}


# --------------------------------------------------------------------------- evaluation
def load_registered(directory, p):
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
    for arm in TRAINED_ARMS:
        predictions[arm] = []
        for seed in p["seeds"]:
            with np.load(folder / f"{arm}_s{seed}.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["window_starts"], cohort["window_starts"]):
                    raise ValueError(f"{arm} s{seed} test windows differ from the common cohort")
                if saved["prediction"].shape != cohort["truth"].shape or not np.isfinite(saved["prediction"]).all():
                    raise ValueError(f"{arm} s{seed} prediction shape or finiteness differs")
                predictions[arm].append(saved["prediction"])
    return cohort, predictions


def farm_metrics(cohort, predictions, rated, turbines):
    """Six-turbine sums on issue times with every turbine present; seed-mean of per-seed farm metrics."""
    issue = cohort["times"][:, 0].astype("datetime64[ns]").astype(np.int64)
    unique, inverse = np.unique(issue, return_inverse=True)
    column = cohort["turbines"].astype(int) - 1
    if column.min() < 0 or column.max() >= turbines:
        raise ValueError("Turbine ids outside the farm")

    def grid(values):
        out = np.full((len(unique), turbines, values.shape[1]), np.nan)
        out[inverse, column] = values
        return out
    truth = grid(cohort["truth"])
    complete = ~np.isnan(truth).any(axis=(1, 2))
    if not complete.any():
        raise ValueError("No issue time with every turbine")
    total = truth[complete].sum(axis=1)
    capacity = float(rated) * turbines
    out = {"issue_times": int(complete.sum()), "capacity_kw": capacity, "arms": {}}
    for arm, preds in predictions.items():
        per_seed = []
        for prediction in preds:
            error = grid(prediction)[complete].sum(axis=1) - total
            rmse = np.sqrt((error ** 2).mean(axis=0))
            per_seed.append({"lead_nrmse_pct": 100 * rmse / capacity, "lead_mae_kw": np.abs(error).mean(axis=0),
                             "nrmse_pct": 100 * float(np.sqrt((error ** 2).mean())) / capacity})
        lead = np.mean([s["lead_nrmse_pct"] for s in per_seed], axis=0)
        overall = float(np.mean([s["nrmse_pct"] for s in per_seed]))
        out["arms"][arm] = {"lead_nrmse_pct": lead.tolist(), "lead_accuracy_pct": (100 - lead).tolist(),
                            "nrmse_pct": overall, "accuracy_pct": 100 - overall,
                            "lead_mae_kw": np.mean([s["lead_mae_kw"] for s in per_seed], axis=0).tolist()}
    return out


def regime_table(cohort, predictions, rated, bins):
    anchor = cohort["PersistenceLast10"][:, 0] / float(rated)
    edges = [-np.inf] + [float(b) for b in bins] + [np.inf]
    masks = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        label = f"anchor<{hi:g}" if lo == -np.inf else f"anchor>={lo:g}" if hi == np.inf else f"{lo:g}<=anchor<{hi:g}"
        masks[label + "_rated"] = (anchor >= lo) & (anchor < hi)
    masks["upstream_missing_at_issue"] = cohort["issue_upstream_missing"] > 0
    masks["all_windows"] = np.ones(len(anchor), dtype=bool)
    rows = []
    for label, mask in masks.items():
        row = {"regime": label, "windows": int(mask.sum()), "share_pct": float(100 * mask.mean())}
        for arm, preds in predictions.items():
            row[f"{arm}_mae"] = (float(np.mean([np.abs(q[mask] - cohort["truth"][mask]).mean() for q in preds]))
                                 if mask.any() else None)
        rows.append(row)
    return rows


def _significant(c):
    return c["absolute_error"]["ci95_low"] > 0 and c["holm_p"] <= .05


def _rmse_positive(c):
    return c["rmse_difference"]["ci95_low_kw"] > 0


def _non_inferior(c, margin):
    return c["absolute_error"]["ci95_low"] > -float(margin)


def rules(cohort_comparisons, margin):
    """Predeclared v30 rules: MAE and RMSE are both primary; all is the primary cohort."""
    everything, normal = cohort_comparisons["all"], cohort_comparisons["normal"]
    return {"upstream_information_supported": bool(_significant(everything["C1"]) and _rmse_positive(everything["C1"])
                                                   and _non_inferior(normal["C1"], margin)),
            "balanced_loss_supported": bool(_rmse_positive(everything["C2"]) and _non_inferior(everything["C2"], margin)),
            "beats_naive": bool(_significant(everything["C3"]) and _significant(normal["C3"])
                                and _rmse_positive(everything["C3"]))}


def evaluate_registered(directory, p, choice):
    """Metrics (both cohorts), C1-C3, rules, ensemble, farm metrics and regimes from registered arrays only."""
    if choice not in TRAINED_ARMS:
        raise ValueError(f"Unknown validation choice {choice!r}")
    cohort, predictions = load_registered(directory, p)
    rated, settings = p["data"]["rated_power_kw"], p["statistics"]
    out = {"primary_cohort": "all", "windows": {"all": int(len(cohort["truth"])), "normal": int(cohort["normal_mask"].sum())},
           "normal_share_pct": float(100 * cohort["normal_mask"].mean()), "metrics": {}, "comparisons": {},
           "ensemble": {"arm": choice, "definition": "3-seed mean forecast (descriptive)"}, "regimes": {},
           "farm": farm_metrics(cohort, predictions, rated, int(p["data"]["farm_turbines"])),
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
        sub = {k: v[mask] for k, v in cohort.items() if k in ("truth", "PersistenceLast10") + ISSUE_ARRAYS}
        out["regimes"][name] = regime_table(sub, subset, rated, p["diagnostics"]["anchor_bins_frac_rated"])
    out["rules"] = rules(out["comparisons"], p["rules"]["non_inferiority_margin_kw"])
    return out


# --------------------------------------------------------------------------- report (R1)
def build_report(directory):
    """Recompute everything from the registered arrays, refuse on any disagreement, write tables/figures/report."""
    from scripts import report_kelmarsh_multiyear_benchmark as r28  # frozen v28 helpers (Agg backend)
    directory = Path(directory)
    p = read_json(directory / "protocol.json")["protocol"]
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
    figures(fresh, directory / "figures")
    r28.forecast_trace(directory, p, choice)
    result = {"protocol_id": p["protocol_id"], "rules": fresh["rules"], "validation_choice": freeze["validation_choice"],
              "windows": fresh["windows"], "normal_share_pct": fresh["normal_share_pct"], "post_test_windows": manifest["windows"],
              "primary_cohort": "all", "farm_accuracy_pct": {a: f["accuracy_pct"] for a, f in fresh["farm"]["arms"].items()},
              "scope": "single 2023 test evaluation; all = deployment (primary) cohort; normal = ex-post conditional cohort; "
                       "2021 and 2022 were opened by v28/v29 and are development years here; 2024 sealed",
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


def figures(fresh, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
    fig.suptitle("v30 test 2023 (single evaluation): all = deployment (primary) cohort, normal = conditional cohort", fontsize=9)
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
    validation = [{"arm": r["arm"], "seed": r["seed"], "best_epoch": r.get("best_epoch"),
                   **{f"val_{k}": v for k, v in r["validation"].items()}} for r in freeze["records"]]
    counts = ", ".join(f"{s} {v['all']}/{v['normal']}" for s, v in manifest["windows"].items())
    choice = freeze["validation_choice"]["arm"]
    lines = [f"# v30 Kelmarsh upstream-observation report ({dt.date.today().isoformat()})", "",
             f"**Rules:** {fresh['rules']}", "",
             f"Validation-chosen configuration (2022, ALL cohort, joint MAE+RMSE score, frozen before test): **{choice}**", "",
             f"Test 2023 windows: all {fresh['windows']['all']}, normal {fresh['windows']['normal']} "
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
             "## Validation (2022, ALL cohort) selected states", "",
             table(validation, ["arm", "seed", "best_epoch", "val_mae", "val_rmse"]), "",
             "![test metrics](figures/test_metrics.png)", "", "![lead MAE](figures/lead_mae.png)", "",
             "![first eligible forecast segment, not selected by error](figures/forecast_trace.png)", ""]
    (Path(directory) / "report.md").write_text("\n".join(lines), encoding="utf-8")
