"""Lead-aligned case keys; no new outcomes, cases, or numerical-model inputs.

The old 12-hour keys lose the last-hour state. Append fixed 1/3-hour observed
wind/power/rotor dynamics, then refit distances using the SAME donor cases.
No query outcome is read. Case/coordinate changes invalidate old certificates.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
from utils import kelmarsh_wiki_quality as quality

legacy, episode = quality.legacy, quality.episode
SHORT_NAMES = ("wind_last_z", "rotor_last_z", "power_diff1_z", "power_diff3_z",
               "wind_diff1_z", "wind_diff3_z", "rotor_diff1_z", "rotor_diff3_z")


def implementation_digest():
    return hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest()


def short_keys(history):
    """History is [window,7,past hours], scaled by the unchanged fit scaler."""
    x = np.asarray(history, float)
    if x.ndim != 3 or x.shape[1] != 7 or x.shape[2] < 4 or not np.isfinite(x).all():
        raise ValueError("Finite seven-channel causal history with at least four hours required")
    return np.column_stack((x[:, 0, -1], x[:, 5, -1],
        x[:, 6, -1] - x[:, 6, -2], x[:, 6, -1] - x[:, 6, -4],
        x[:, 0, -1] - x[:, 0, -2], x[:, 0, -1] - x[:, 0, -4],
        x[:, 5, -1] - x[:, 5, -2], x[:, 5, -1] - x[:, 5, -4]))


def keys(pack, info, capacity, use_controls):
    old = legacy.keys(info["context"], pack["anchor"], capacity, use_controls)
    return np.column_stack((old, short_keys(pack["history"])))


def fit_memory(source, pack, info, contract):
    # Reproducing the exact old bank verifies rows, ordering, quality states,
    # timestamps, raw labels and donor identities before replacing ONLY keys.
    reproduced = quality.fit_memory(pack, info, contract, source["settings"], source["quality_settings"])
    if reproduced != source:
        raise ValueError("Source case rows, quality context, outcomes or implementation changed")
    x = keys(pack, info, contract["rated_power_kw"], source["settings"]["use_controls"])
    cfg, groups = source["settings"], []
    for original in source["groups"]:
        group = dict(original)
        if group["active"]:
            mask = pack["events"][:, group["event"]] & (info["state"] == group["state"])
            xx, units = x[mask], pack["turbines"][mask]
            mean, scale = xx.mean(0), xx.std(0)
            scale = np.where(scale > 1e-6, scale, 1.)
            z = (xx - mean) / scale
            _, distance, diversity = legacy.neighbors._neighbors(z, z, units, cfg["neighbors"], excluded_turbines=units)
            supported = np.isfinite(distance) & (diversity >= cfg["min_neighbor_turbines"])
            group.update(keys=xx.tolist(), mean=mean.tolist(), scale=scale.tolist(),
                radius=float(np.quantile(distance[supported], cfg["support_quantile"])) if supported.any() else 0.)
        groups.append(group)
    bank = dict(source)
    bank.update(groups=groups, source_expert_id=source["expert_id"], short_names=list(SHORT_NAMES),
        short_implementation_sha256=implementation_digest(),
        feature_semantics="Same frozen scaler, causal last wind/rotor and lag1/lag3 differences; no future covariates")
    bank["expert_id"] = legacy.history.digest({k: v for k, v in bank.items() if k != "expert_id"})
    return bank


def validate(bank, contract):
    if (bank["expert_id"] != legacy.history.digest({k: v for k, v in bank.items() if k != "expert_id"})
            or bank["contract"] != contract or bank["short_names"] != list(SHORT_NAMES)
            or bank["short_implementation_sha256"] != implementation_digest()
            or bank["implementation_sha256"] != quality.implementation_digest()
            or bank["readout_sha256"] != episode.implementation_digest()
            or bank["coordinate_sha256"] != legacy.implementation_digest()
            or bank["neighbor_sha256"] != legacy.neighbors.implementation_digest()):
        raise ValueError("Short-dynamics case expert/coordinate/base identity changed")


def candidates(bank, query, info, contract):
    validate(bank, contract)
    reference = np.asarray(query["reference"], float)
    times = np.asarray(query["times"]).astype("datetime64[ns]")
    if times.shape != reference.shape or np.isnat(times).any():
        raise ValueError("Invalid query clock")
    origins = times[:, 0].astype(np.int64) - 3600 * 10**9
    if np.any(info["last_observed_ns"] > origins):
        raise ValueError("Future operating metadata at issue time")
    cfg, capacity = bank["settings"], contract["rated_power_kw"]
    x = keys(query, info, capacity, cfg["use_controls"])
    shape = (*reference.shape, len(legacy.EVENTS))
    prediction = np.repeat(reference[..., None], shape[-1], axis=-1)
    alpha, variance = np.zeros(shape), np.zeros(shape)
    physical, overlap = np.zeros(shape, bool), np.zeros(shape, bool)
    for group in bank["groups"]:
        k = group["event"]
        mask = query["events"][:, k] & (info["state"] == group["state"])
        if not group["active"] or not mask.any():
            continue
        if np.any(origins[mask] < group["available_ns"]):
            raise ValueError("Case outcomes not yet mature")
        mean, scale = np.asarray(group["mean"]), np.asarray(group["scale"])
        units, values = np.asarray(group["units"]), np.asarray(group["values"])
        idx, distance, diversity = legacy.neighbors._neighbors((x[mask] - mean) / scale,
            (np.asarray(group["keys"]) - mean) / scale, units, cfg["neighbors"])
        physical[mask, :, k] = True
        overlap[mask, :, k] = ((distance <= group["radius"]) & (diversity >= cfg["min_neighbor_turbines"]))[:, None]
        blocks = np.asarray(group["weather_blocks"])[idx]
        for h in range(reference.shape[1]):
            out = episode.readout(values[idx, h], units[idx], blocks,
                (reference[mask, h] - query["anchor"][mask, h]) / capacity, cfg["max_scale"], 2 / 3)
            prediction[mask, h, k] = query["anchor"][mask, h] + capacity * out["prediction"]
            alpha[mask, h, k], variance[mask, h, k] = out["alpha"], capacity**2 * out["variance"]
    return dict(prediction=prediction, alpha=alpha, variance_kw2=variance, physical=physical, overlap=overlap)


def transform(pack, info, bank, contract):
    """Replace only the Wiki token view; leave all other arrays unmodified."""
    query = {key: pack[key] for key in ("reference", "anchor", "context", "events", "times", "history")}
    out = candidates(bank, query, info, contract)
    result = dict(pack)
    result["memory"] = np.concatenate(((out["prediction"] - pack["reference"][..., None]) / contract["rated_power_kw"],
        out["alpha"], np.log1p(np.sqrt(out["variance_kw2"]) / contract["rated_power_kw"]),
        out["physical"], out["overlap"]), -1).astype(np.float32)
    result["overlap"] = out["overlap"]
    return result
