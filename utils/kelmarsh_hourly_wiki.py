"""Kelmarsh adaptation of the existing physical control-transition Wiki.

The source v15 implementation stays untouched. Here controls are observed blade
pitch and geographical nacelle heading, not an unverified yaw-misalignment field.
Raw negative power targets are retained; corrections alone have fixed bounds.
The expert/base identity and two chronological utility blocks are mandatory.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from utils import transport_event_wiki as history
from utils import local_transport_event_wiki as neighbors


EVENTS = ["hourly_wind_variability", "high_wind_low_power", "rated_saturation", "low_wind_idle"]
HISTORY_NAMES = [history.FEATURES[i] for i in (1, 2, 3, 4, 5, 6, 8, 9)]
CONTROLS = ["pitch_mean_div90", "pitch_last_div90", "pitch_change_div90",
            "nacelle_heading_sin_mean", "nacelle_heading_sin_last",
            "nacelle_heading_cos_mean", "nacelle_heading_cos_last"]
BASE_FIELDS = ("base_checkpoint_sha256", "base_manifest_sha256", "scaler_sha256")


def implementation_digest():
    return hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest()


def physical_events(context, rules):
    # Hourly variation is not claimed to measure ten-minute gust/turbulence.
    wind, power = context["wind_mean_mps"], context["power_ratio"]
    return np.column_stack((
        (context["wind_std_mps"] >= rules["wind_std_min_mps"]) |
        (context["wind_max_step_mps"] >= rules["wind_step_min_mps"]),
        (wind >= rules["active_wind_min_mps"]) & (power <= rules["low_power_max_ratio"]),
        power >= rules["rated_power_min_ratio"],
        (wind <= rules["low_wind_max_mps"]) & (power <= rules["low_power_max_ratio"]),
    ))


def keys(context, anchor, capacity, use_controls):
    matrix = history.physical_features(context, anchor, anchor, capacity)
    result = matrix[:, 0, (1, 2, 3, 4, 5, 6, 8, 9)]
    if use_controls:
        result = np.column_stack((result, *(context[name] for name in CONTROLS)))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite or missing observed knowledge coordinates")
    return result


def validate(model, contract, policy=None):
    if model["expert_id"] != history.digest({k: v for k, v in model.items() if k != "expert_id"}):
        raise ValueError("Knowledge memory/readout identity changed")
    bank_contract = {key: value for key, value in contract.items() if key not in BASE_FIELDS}
    if (model["bank_contract"] != bank_contract or model["implementation_sha256"] != implementation_digest()
            or model["history_sha256"] != history.implementation_digest()
            or model["neighbor_sha256"] != neighbors.implementation_digest()):
        raise ValueError("Expert, base, coordinate or implementation contract changed")
    if policy is not None:
        if policy["expert_id"] != model["expert_id"] or policy["contract"] != contract:
            raise ValueError("Utility evidence does not belong to this actual expert/base")
        if policy["policy_id"] != history.digest({k: v for k, v in policy.items() if k != "policy_id"}):
            raise ValueError("Frozen utility policy changed")


def fit_experts(block, contract, settings):
    ids = history._source_ids(block["turbines"], contract)
    x = keys(block["context"], block["anchor"], contract["rated_power_kw"], settings["use_controls"])
    labels = (block["truth"] - block["anchor"]) / contract["rated_power_kw"]
    if not np.isfinite(labels).all() or labels.shape[1] != contract["pred_len"]:
        raise ValueError("Memory outcome targets do not match the declared horizon")
    experts = []
    for k, event in enumerate(EVENTS):
        mask = block["events"][:, k]
        xx, unit_ids = x[mask], ids[mask]
        unique, counts = np.unique(unit_ids, return_counts=True)
        active = (len(xx) >= settings["min_fit_windows"] and
                  len(unique) >= settings["min_neighbor_turbines"] + 1 and
                  len(xx) - counts.max(initial=0) >= settings["neighbors"])
        expert = {"event_id": event, "active": bool(active), "fit_windows": int(mask.sum()),
                  "fit_turbines": int(len(unique))}
        if active:
            mean, scale = xx.mean(0), xx.std(0)
            scale = np.where(scale > 1e-6, scale, 1.)
            z = (xx - mean) / scale
            _, distance, diversity = neighbors._neighbors(z, z, unit_ids, settings["neighbors"], excluded_turbines=unit_ids)
            supported = np.isfinite(distance) & (diversity >= settings["min_neighbor_turbines"])
            expert.update(keys=xx.tolist(), values=labels[mask].tolist(), turbines=unit_ids.tolist(),
                          mean=mean.tolist(), scale=scale.tolist(),
                          support_radius=float(np.quantile(distance[supported], settings["support_quantile"])) if supported.any() else 0.)
        experts.append(expert)
    model = {"schema_version": 1, "readout": "power_transition_from_last_observation",
             "bank_contract": {key: value for key, value in contract.items() if key not in BASE_FIELDS},
             "fit_source_base": {key: contract.get(key) for key in BASE_FIELDS},
             "settings": settings, "events": EVENTS,
             "feature_names": HISTORY_NAMES + (CONTROLS if settings["use_controls"] else []),
             "implementation_sha256": implementation_digest(), "history_sha256": history.implementation_digest(),
             "neighbor_sha256": neighbors.implementation_digest(), "experts": experts,
             "targets_clipped": False, "base_refit_after_evidence": False}
    model["expert_id"] = history.digest(model)
    return model


def candidates(model, block, contract):
    validate(model, contract)
    settings = model["settings"]
    x = keys(block["context"], block["anchor"], contract["rated_power_kw"], settings["use_controls"])
    reference = np.asarray(block["reference"])
    n, horizon = reference.shape
    delta = np.zeros((n, horizon, len(EVENTS) + 1))
    physical = np.zeros_like(delta, bool)
    overlap = np.zeros_like(physical)
    for k, expert in enumerate(model["experts"]):
        mask = block["events"][:, k]
        if not expert["active"] or not mask.any():
            continue
        mean, scale = np.asarray(expert["mean"]), np.asarray(expert["scale"])
        bank = (np.asarray(expert["keys"]) - mean) / scale
        ids, values = np.asarray(expert["turbines"]), np.asarray(expert["values"])
        idx, distance, diversity = neighbors._neighbors((x[mask] - mean) / scale, bank, ids, settings["neighbors"])
        for h in range(horizon):
            transition = neighbors._balanced_median(values[idx, h], ids[idx]) * contract["rated_power_kw"]
            correction = block["anchor"][mask, h] + transition - reference[mask, h]
            delta[mask, h, k] = np.clip(correction, -settings["max_scale"] * contract["rated_power_kw"],
                                      settings["max_scale"] * contract["rated_power_kw"])
        physical[mask, :, k] = True
        overlap[mask, :, k] = ((distance <= expert["support_radius"]) &
                               (diversity >= settings["min_neighbor_turbines"]))[:, None]
    delta[..., -1] = np.clip(delta[..., :-1].sum(-1), -settings["composition_max_scale"] * contract["rated_power_kw"],
                            settings["composition_max_scale"] * contract["rated_power_kw"])
    physical[..., -1] = physical[..., :-1].sum(-1) >= 2
    overlap[..., -1] = (overlap[..., :-1] | ~physical[..., :-1]).all(-1)
    return reference[..., None] + delta, physical, physical & overlap


def fit_policy(model, blocks, contract, settings):
    validate(model, contract)
    if len(blocks) != 2 or blocks[0]["times"].max() >= blocks[1]["times"].min():
        raise ValueError("Two disjoint chronological utility blocks required")
    prepared, hsh = [], hashlib.sha256()
    for block in blocks:
        history._source_ids(block["turbines"], contract)
        candidate, _, support = candidates(model, block, contract)
        truth = block["truth"]
        if truth.shape != candidate.shape[:2] or not np.isfinite(truth).all():
            raise ValueError("Utility outcome alignment/validity error")
        gain = np.abs(block["reference"] - truth)[..., None] - np.abs(candidate - truth[..., None])
        mse = (block["reference"] - truth)[..., None] ** 2 - (candidate - truth[..., None]) ** 2
        for value in (truth, candidate, block["times"].astype(np.int64), block["turbines"]):
            hsh.update(np.ascontiguousarray(value).tobytes())
        prepared.append(dict(block, gain=gain, mse=mse, support=support))
    entries = {}
    for level, groups in (("event", 1), ("power", 5), ("joint", 20)):
        for group in range(groups):
            for k in range(len(EVENTS) + 1):
                for h in range(contract["pred_len"]):
                    evidence = []
                    for block in prepared:
                        strata = np.zeros(len(block["truth"]), int) if level == "event" else block["context"][level + "_bin"]
                        mask = (strata == group) & block["support"][:, h, k]
                        ids = np.unique(block["turbines"][mask])
                        gains, squared = [], []
                        for turbine in ids:
                            chosen = mask & (block["turbines"] == turbine)
                            gains.append(float(block["gain"][chosen, h, k].mean()))
                            squared.append(float(block["mse"][chosen, h, k].mean()))
                        evidence.append({"windows": int(mask.sum()), "turbines": len(ids),
                                         "gain_kw": float(np.mean(gains)) if gains else None,
                                         "mse_gain": float(np.mean(squared)) if squared else None,
                                         "positive_turbine_fraction": float(np.mean(np.asarray(gains) > 0)) if gains else None})
                    supported = all(row["windows"] >= settings["min_windows"] and
                                    row["turbines"] >= settings["min_turbines"] for row in evidence)
                    accepted = supported and all(row["gain_kw"] > 0 and row["mse_gain"] > 0 and
                                                 row["positive_turbine_fraction"] >= settings["positive_fraction"] for row in evidence)
                    entries[f"{level}:{group}:{h}:{k}"] = {"supported": supported, "accepted": accepted,
                        "score_kw": min(row["gain_kw"] for row in evidence) if accepted else 0., "blocks": evidence}
    policy = {"expert_id": model["expert_id"], "contract": contract, "settings": settings, "entries": entries,
              "utility_outputs_sha256": hsh.hexdigest(), "score": "worst two-block turbine-balanced MAE gain; not an LCB guarantee",
              "granularity": "finest supported; a harmful supported fine cell vetoes coarse fallback",
              "retired_cells": sum(row["supported"] and not row["accepted"] for row in entries.values())}
    policy["policy_id"] = history.digest(policy)
    return policy


def apply(model, policy, block, contract, event_only=False, allow_composition=True):
    validate(model, contract, policy)
    candidate, physical, overlap = candidates(model, block, contract)
    n, horizon, count = candidate.shape
    score = np.zeros_like(candidate)
    supported = np.zeros_like(physical)
    granularity = np.full(candidate.shape, -1, np.int8)
    levels = [("event", np.zeros(n, int), 0)] if event_only else [
        ("joint", block["context"]["joint_bin"], 2), ("power", block["context"]["power_bin"], 1),
        ("event", np.zeros(n, int), 0)]
    for h in range(horizon):
        for k in range(count):
            remaining = np.ones(n, bool)
            for level, bins, code in levels:
                for group in np.unique(bins):
                    row = policy["entries"][f"{level}:{group}:{h}:{k}"]
                    if not row["supported"]:
                        continue
                    selected = remaining & (bins == group)
                    score[selected, h, k] = row["score_kw"]
                    supported[selected, h, k] = True
                    granularity[selected, h, k] = code
                    remaining[selected] = False
    available = overlap & supported & (score > 0)
    if not allow_composition:
        available[..., -1] = False
    best = np.where(available, score, -np.inf).argmax(-1)
    active = available.any(-1)
    chosen = np.take_along_axis(candidate, best[..., None], -1).squeeze(-1)
    return {"prediction": np.where(active, chosen, block["reference"]), "action": np.where(active, best + 1, 0),
            "physical_available": physical, "history_overlap": overlap, "available": available,
            "granularity": granularity, "score_kw": score, "expert_id": model["expert_id"], "policy_id": policy["policy_id"]}
