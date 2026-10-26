"""Torch-free helpers for v28 (multi-year Kelmarsh base benchmark): protocol checks, data identity, arm configs,
metrics, paired comparisons, validation choice and the predeclared rules. Generic hashing / statistics come from the
v25 module ``utils.kelmarsh_backbone_information`` (its bytes enter the v28 fingerprint)."""
from __future__ import annotations

import numpy as np

from utils.kelmarsh_backbone_information import (  # noqa: F401  (re-exported)
    acquire_exclusive_lock, append_event, canonical_sha256, choose_pair_device, contract_mismatches, file_sha256,
    holm_adjust, paired_loss_statistics, read_json, rmse_difference_bootstrap, safe, text_sha256_lf,
    verify_registered_files, verify_training_completed, window_hash, write_json)

PROTOCOL_ID = "kelmarsh_multiyear_base_benchmark_v28"
PERSISTENCE_ARMS = ("PersistenceHour", "PersistenceLast10")
TRAINED_ARMS = ("DLinear_normal", "DLinear_alltrain", "LinearFlat_base", "LinearFlat_last10", "TSMixer_base", "TSMixer_last10")
CHOICE_CANDIDATES = ("LinearFlat_base", "LinearFlat_last10", "TSMixer_base", "TSMixer_last10")
COHORTS = ("normal", "all")
HOUR = np.timedelta64(1, "h")


# --------------------------------------------------------------------------- protocol
def expected_parameters(family, channels, seq_len=24, pred_len=4, ff_dim=32, blocks=2):
    if family == "DLinear":
        return 2 * (seq_len * pred_len + pred_len)
    if family == "LinearFlat":
        return seq_len * channels * pred_len + pred_len
    if family == "TSMixerBasicSmall":
        block = 2 * (2 * seq_len * channels) + (seq_len * seq_len + seq_len) + (channels * ff_dim + ff_dim) + (ff_dim * channels + channels)
        return blocks * block + seq_len * pred_len + pred_len
    raise ValueError(f"Unknown family {family!r}")


def validate_protocol(p):
    d = p["data"]
    if p["protocol_id"] != PROTOCOL_ID:
        raise ValueError("Unexpected protocol id")
    if tuple(p["arms"]) != PERSISTENCE_ARMS + TRAINED_ARMS:
        raise ValueError("Arm list departs from the frozen design")
    if (d["input_len"], d["pred_len"], d["stride"]) != (24, 4, 3):
        raise ValueError("Window design departs from 24 -> 4, stride 3")
    if d["channels"][-1] != "power" or any(s[-1] != len(d["channels"]) - 1 for s in d["channel_sets"].values()):
        raise ValueError("Power must be the last channel of every channel set")
    if d["channels"][6:-1] != d["extras_in_common_series"]:
        raise ValueError("Common-series extras and channel list disagree")
    splits = d["splits"]
    if not (max(splits["train"]) < min(splits["validation"]) <= max(splits["validation"]) < min(splits["test"])
            < min(d["sealed_years"])):
        raise ValueError("Years must be ordered train < validation < test < sealed")
    if not set(d["v28_preparation"]["curve_years"]) <= set(splits["train"]):
        raise ValueError("Power-curve years must belong to train")
    if set(d["v28_preparation"]["years"]) != set(splits["train"]) | set(splits["validation"]) | set(splits["test"]):
        raise ValueError("Prepared years must be exactly train+validation+test")
    if d.get("yaw_error_channels_enabled") or any(c.startswith("yaw") for c in d["channels"]):
        raise ValueError("Yaw-error channels are disabled in v28")
    for arm in TRAINED_ARMS:
        spec = p["arms"][arm]
        family = p["families"][spec["family"]]
        channels = len(d["channel_sets"][spec["channels"]])
        declared = family["parameters"].get(spec["channels"])
        if declared != expected_parameters(spec["family"], channels):
            raise ValueError(f"{arm}: declared parameters {declared} != formula")
        if spec["train_cohort"] not in COHORTS:
            raise ValueError(f"{arm}: unknown training cohort")
    if p["budget"]["fits"] != len(TRAINED_ARMS) * len(p["seeds"]) or p["budget"]["fits"] > p["budget"]["new_fits_max"]:
        raise ValueError("Budget does not match arms x seeds")
    if p["budget"]["fits"] * p["training"]["max_epochs"] > p["budget"]["new_epochs_max"]:
        raise ValueError("Epoch budget exceeded")
    if set(p["comparisons"]) != {"C1", "C2", "C3", "C4", "C5"}:
        raise ValueError("Comparison family departs from the frozen design")
    if p["training"].get("selection_cohort") != "normal":
        raise ValueError("Every arm must select on the same normal validation cohort")
    return True


def arm_config(p, arm):
    spec = p["arms"][arm]
    channels = p["data"]["channel_sets"][spec["channels"]]
    return {"arm": arm, "family": spec["family"], "channel_set": spec["channels"], "channel_indices": list(channels),
            "channels": len(channels), "train_cohort": spec["train_cohort"],
            "selection_cohort": p["training"]["selection_cohort"],
            "expected_parameters": p["families"][spec["family"]]["parameters"][spec["channels"]],
            "args": dict(p["families"][spec["family"]].get("args", {}))}


# --------------------------------------------------------------------------- data identity (D0)
def verify_preparation(meta, protocol, csv_sha256):
    prep = protocol["data"]["v28_preparation"]
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
        raise ValueError(f"v28 preparation identity failed: {problems}")
    return True


def verify_development_rows(v28_frame, dev_frame):
    """Every registered 2016-2019 development row must reappear unchanged in the v28 preparation."""
    key = ["turbine_id", "timestamp_hour_end_utc"]
    missing = [c for c in dev_frame.columns if c not in v28_frame.columns]
    if missing:
        raise ValueError(f"v28 preparation lacks development columns {missing}")
    dev = dev_frame.sort_values(key, kind="mergesort").reset_index(drop=True)
    stamps = set(zip(dev["turbine_id"].astype(str), dev["timestamp_hour_end_utc"].astype(str)))
    mask = [(str(t), str(s)) in stamps for t, s in zip(v28_frame["turbine_id"], v28_frame["timestamp_hour_end_utc"])]
    new = v28_frame.loc[mask, list(dev.columns)].sort_values(key, kind="mergesort").reset_index(drop=True)
    if len(new) != len(dev):
        raise ValueError(f"v28 preparation has {len(new)} of {len(dev)} development rows")
    for column in dev.columns:
        a, b = dev[column], new[column]
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            same = np.array_equal(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)
        else:
            same = a.astype(str).equals(b.astype(str))
        if not same:
            raise ValueError(f"Development column {column!r} differs in the v28 preparation")
    return {"rows_compared": int(len(dev)), "columns_compared": int(len(dev.columns))}


def verify_power_curves(v28_meta, dev_meta):
    if v28_meta.get("power_curves") != dev_meta.get("power_curves"):
        raise ValueError("Power curves differ from the registered development preparation")
    return True


# --------------------------------------------------------------------------- metrics
def metrics(prediction, truth, rated):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 2 or not p.size or rated <= 0 or not (np.isfinite(p).all() and np.isfinite(y).all()):
        raise ValueError("Aligned finite [window, lead] arrays required")
    e = p - y
    sst = float(((y - y.mean()) ** 2).sum())
    lead_sst = ((y - y.mean(0)) ** 2).sum(0)
    return {"windows": int(len(y)), "mae": float(np.abs(e).mean()), "mse": float((e ** 2).mean()),
            "rmse": float(np.sqrt((e ** 2).mean())), "r2": 1 - float((e ** 2).sum()) / sst if sst > 0 else None,
            "mbe": float(e.mean()), "nmae_pct_rated": float(100 * np.abs(e).mean() / rated),
            "nrmse_pct_rated": float(100 * np.sqrt((e ** 2).mean()) / rated),
            "hit_5pct_capacity_pct": float(100 * (np.abs(e) <= .05 * rated).mean()),
            "hit_10pct_capacity_pct": float(100 * (np.abs(e) <= .10 * rated).mean()),
            "lead_mse": (e ** 2).mean(0).tolist(),
            "lead_mae": np.abs(e).mean(0).tolist(), "lead_rmse": np.sqrt((e ** 2).mean(0)).tolist(),
            "lead_r2": (1 - (e ** 2).sum(0) / np.where(lead_sst > 0, lead_sst, np.nan)).tolist()}


def seed_mean(per_seed):
    keys = ("mae", "mse", "rmse", "r2", "mbe", "nmae_pct_rated", "nrmse_pct_rated",
            "hit_5pct_capacity_pct", "hit_10pct_capacity_pct")
    out = {k: None if any(m[k] is None for m in per_seed) else float(np.mean([m[k] for m in per_seed])) for k in keys}
    for k in ("lead_mae", "lead_mse", "lead_rmse", "lead_r2"):
        out[k] = safe(np.mean([m[k] for m in per_seed], axis=0).tolist())
    out["seeds"] = len(per_seed)
    return out


def validation_choice(validation_records, p):
    """Lowest seed-mean validation (normal) MAE among the choice candidates; ties -> fewer parameters."""
    rows = []
    for arm in CHOICE_CANDIDATES:
        maes = [r["validation"]["mae"] for r in validation_records if r["arm"] == arm]
        if len(maes) != len(p["seeds"]):
            raise ValueError(f"Incomplete validation records for {arm}")
        rows.append((float(np.mean(maes)), arm_config(p, arm)["expected_parameters"], arm))
    rows.sort()
    return {"arm": rows[0][2], "ranking": [{"arm": a, "validation_mae_seed_mean": m, "parameters": n} for m, n, a in rows]}


# --------------------------------------------------------------------------- comparisons and rules
def paired(candidate_preds, reference_preds, truth, issue_times, settings):
    """Seed-averaged losses on identical windows (not an ensemble)."""
    cand_abs = np.mean([np.abs(c - truth) for c in candidate_preds], axis=0)
    ref_abs = np.mean([np.abs(r - truth) for r in reference_preds], axis=0)
    cand_sq = np.mean([(c - truth) ** 2 for c in candidate_preds], axis=0)
    ref_sq = np.mean([(r - truth) ** 2 for r in reference_preds], axis=0)
    kwargs = {"block_length": settings["block_length"], "replicates": settings["replicates"], "seed": settings["seed"]}
    return {"absolute_error": paired_loss_statistics(cand_abs, ref_abs, issue_times, **kwargs),
            "rmse_difference": rmse_difference_bootstrap(cand_sq, ref_sq, issue_times, **kwargs),
            "per_lead_mae_gain_kw": (ref_abs - cand_abs).mean(0).tolist()}


def compare_all(predictions, truth, issue_times, comparisons, choice, settings):
    """predictions: {arm: [seed predictions]} on one cohort. Returns C1-C5 with Holm over MAE NW p-values."""
    out = {}
    for name, spec in comparisons.items():
        candidate = choice if spec["candidate"] == "VALIDATION_CHOSEN" else spec["candidate"]
        out[name] = {"candidate": candidate, "reference": spec["reference"], "question": spec["question"],
                     **paired(predictions[candidate], predictions[spec["reference"]], truth, issue_times, settings)}
    names = sorted(out)
    adjusted = holm_adjust([out[n]["absolute_error"]["newey_west"]["p_value_two_sided"] for n in names])
    for n, a in zip(names, adjusted):
        out[n]["holm_p"] = float(a)
    return out


def evaluate_registered(directory, p, choice):
    """Metrics (both cohorts), C1-C5 and the rules from the registered test arrays only (controller and report)."""
    from pathlib import Path
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
    rated, settings = p["data"]["rated_power_kw"], p["statistics"]
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
    out = {"windows": {"all": int(len(cohort["truth"])), "normal": int(cohort["normal_mask"].sum())},
           "normal_share_pct": float(100 * cohort["normal_mask"].mean()), "metrics": {}, "comparisons": {},
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
    out["rules"] = rules(out["comparisons"])
    return out


def _significant(comparison):
    return comparison["absolute_error"]["ci95_low"] > 0 and comparison["holm_p"] <= .05


def _positive(comparison):
    return _significant(comparison) and comparison["rmse_difference"]["rmse_gain_kw"] > 0


def rules(cohort_comparisons):
    """Predeclared protocol rules on {cohort: comparisons}."""
    normal, everything = cohort_comparisons["normal"], cohort_comparisons["all"]

    def holds_all(name):
        return everything[name]["absolute_error"]["gain"] >= 0
    flags = {"information_supported": any(_positive(normal[c]) and holds_all(c) for c in ("C1", "C2")),
             "information_supported_by": [c for c in ("C1", "C2") if _positive(normal[c]) and holds_all(c)],
             "capacity_supported": bool(_positive(normal["C3"]) and holds_all("C3")),
             "training_cleaning_supported": bool(_significant(normal["C4"]) and holds_all("C4")),
             "beats_naive": bool(_significant(normal["C5"]) and _significant(everything["C5"]))}
    return flags
