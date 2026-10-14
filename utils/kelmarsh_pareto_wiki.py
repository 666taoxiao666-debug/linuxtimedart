"""Forecast-relative, distributional event readout; frozen v1 memory is retained.

The empirical mean minimizes squared loss and the weighted median minimizes
absolute loss. Move towards the median only as far as the clipped mean permits.
This is an empirical local Pareto rule, NOT a guarantee under distribution shift.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from utils import kelmarsh_hourly_wiki as legacy


def implementation_digest():
    return hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest()


def local_readout(values, ids, base_change, max_scale, positive_fraction, mode="pareto"):
    """Inputs are normalized future transitions, not current/future query labels."""
    values, ids = np.asarray(values, float), np.asarray(ids)
    base_change = np.asarray(base_change, float)
    if values.ndim != 2 or values.shape != ids.shape or base_change.shape != values.shape[:1]:
        raise ValueError("Misaligned neighbor outcomes, units or forecast")
    if not np.isfinite(values).all() or not np.isfinite(base_change).all():
        raise ValueError("Nonfinite local distribution")
    if mode not in ("pareto", "median_local"):
        raise ValueError("Unknown predeclared readout")
    counts = (ids[:, :, None] == ids[:, None, :]).sum(-1)
    weights = 1. / counts
    weights /= weights.sum(1, keepdims=True)
    mean = (weights * values).sum(1)
    median = legacy.neighbors._balanced_median(values, ids)
    direction = np.clip(median - base_change, -max_scale, max_scale)
    alpha = np.zeros_like(direction)
    np.divide(mean - base_change, direction, out=alpha, where=np.abs(direction) > 1e-12)
    alpha = np.clip(alpha, 0., 1.) if mode == "pareto" else (np.abs(direction) > 1e-12).astype(float)
    prediction = base_change + alpha * direction
    absolute_gain = np.abs(base_change[:, None] - values) - np.abs(prediction[:, None] - values)
    squared_gain = (base_change[:, None] - values) ** 2 - (prediction[:, None] - values) ** 2
    mae_gain = (weights * absolute_gain).sum(1)
    mse_gain = (weights * squared_gain).sum(1)
    positive = np.zeros(len(values))
    for row in range(len(values)):
        unit_ids = np.unique(ids[row])
        positive[row] = np.mean([absolute_gain[row, ids[row] == unit].mean() > 0 and
                                 squared_gain[row, ids[row] == unit].mean() > 0 for unit in unit_ids])
    eligible = (mae_gain > 1e-12) & (mse_gain > 1e-12) & (positive >= positive_fraction)
    # No label-dependent clipping: this bounds only the forecast correction.
    return dict(prediction=prediction, alpha=alpha, mae_gain=mae_gain, mse_gain=mse_gain,
                positive_donor_fraction=positive, eligible=eligible, variance=(weights * (values - mean[:, None]) ** 2).sum(1))


def make_model(memory, contract, *, mode, protocol_sha256, positive_fraction):
    legacy.validate(memory, contract)
    if mode not in ("pareto", "median_local"):
        raise ValueError("Undeclared readout")
    model = dict(memory=memory, mode=mode, protocol_sha256=protocol_sha256,
                 positive_fraction=positive_fraction, implementation_sha256=implementation_digest(),
                 composition="disabled: conditional future levels are not additive causal residuals")
    model["expert_id"] = legacy.history.digest(model)
    return model


def validate(model, contract, policy=None):
    if model["expert_id"] != legacy.history.digest({k: v for k, v in model.items() if k != "expert_id"}):
        raise ValueError("Distributional readout identity changed")
    if model["implementation_sha256"] != implementation_digest():
        raise ValueError("Readout implementation changed")
    legacy.validate(model["memory"], contract)
    if policy is not None:
        if policy["expert_id"] != model["expert_id"] or policy["contract"] != contract:
            raise ValueError("Utility evidence does not belong to this readout and actual base")
        if policy["policy_id"] != legacy.history.digest({k: v for k, v in policy.items() if k != "policy_id"}):
            raise ValueError("Utility policy changed")


def candidates(model, block, contract):
    validate(model, contract)
    memory, capacity = model["memory"], contract["rated_power_kw"]
    settings = memory["settings"]
    x = legacy.keys(block["context"], block["anchor"], capacity, settings["use_controls"])
    reference = np.asarray(block["reference"], float)
    shape = (*reference.shape, len(legacy.EVENTS) + 1)
    prediction = np.repeat(reference[..., None], shape[-1], axis=-1)
    physical, overlap, eligible = [np.zeros(shape, bool) for _ in range(3)]
    alpha, gain, variance = [np.zeros(shape) for _ in range(3)]
    for k, expert in enumerate(memory["experts"]):
        mask = np.asarray(block["events"][:, k], bool)
        if not expert["active"] or not mask.any():
            continue
        mean, scale = np.asarray(expert["mean"]), np.asarray(expert["scale"])
        bank = (np.asarray(expert["keys"]) - mean) / scale
        ids, values = np.asarray(expert["turbines"]), np.asarray(expert["values"])
        idx, distance, diversity = legacy.neighbors._neighbors((x[mask] - mean) / scale, bank, ids, settings["neighbors"])
        physical[mask, :, k] = True
        overlap[mask, :, k] = ((distance <= expert["support_radius"]) &
                              (diversity >= settings["min_neighbor_turbines"]))[:, None]
        for h in range(reference.shape[1]):
            base_change = (reference[mask, h] - block["anchor"][mask, h]) / capacity
            output = local_readout(values[idx, h], ids[idx], base_change, settings["max_scale"],
                                   model["positive_fraction"], model["mode"])
            prediction[mask, h, k] = block["anchor"][mask, h] + capacity * output["prediction"]
            alpha[mask, h, k], gain[mask, h, k] = output["alpha"], capacity * output["mae_gain"]
            variance[mask, h, k] = capacity ** 2 * output["variance"]
            eligible[mask, h, k] = output["eligible"]
    return dict(prediction=prediction, physical=physical, overlap=overlap,
                support=physical & overlap & eligible, alpha=alpha, local_gain_kw=gain, variance_kw2=variance)


def fit_policy(model, blocks, contract, settings):
    validate(model, contract)
    if len(blocks) != 2 or blocks[0]["times"].max() >= blocks[1]["times"].min():
        raise ValueError("Two disjoint chronological train-utility blocks required")
    prepared, digest = [], hashlib.sha256()
    for block in blocks:
        legacy.history._source_ids(block["turbines"], contract)
        output = candidates(model, block, contract)
        truth = np.asarray(block["truth"])
        if truth.shape != output["prediction"].shape[:2] or not np.isfinite(truth).all():
            raise ValueError("Utility targets are invalid or misaligned")
        gain = np.abs(block["reference"] - truth)[..., None] - np.abs(output["prediction"] - truth[..., None])
        mse = (block["reference"] - truth)[..., None] ** 2 - (output["prediction"] - truth[..., None]) ** 2
        for value in (truth, output["prediction"], output["support"], block["times"].astype(np.int64), block["turbines"]):
            digest.update(np.ascontiguousarray(value).tobytes())
        prepared.append(dict(block, gain=gain, mse=mse, support=output["support"]))
    entries = {}
    for level, groups in (("event", 1), ("power", 5), ("joint", 20)):
        for group in range(groups):
            for k in range(len(legacy.EVENTS) + 1):
                for h in range(contract["pred_len"]):
                    evidence = []
                    for block in prepared:
                        strata = np.zeros(len(block["truth"]), int) if level == "event" else block["context"][level + "_bin"]
                        mask = (strata == group) & block["support"][:, h, k]
                        ids = np.unique(block["turbines"][mask])
                        gains = [block["gain"][mask & (block["turbines"] == unit), h, k].mean() for unit in ids]
                        mse = [block["mse"][mask & (block["turbines"] == unit), h, k].mean() for unit in ids]
                        evidence.append(dict(windows=int(mask.sum()), turbines=len(ids),
                            gain_kw=float(np.mean(gains)) if gains else None,
                            mse_gain=float(np.mean(mse)) if mse else None,
                            positive_turbine_fraction=float(np.mean(np.asarray(gains) > 0)) if gains else None))
                    supported = all(row["windows"] >= settings["min_windows"] and row["turbines"] >= settings["min_turbines"] for row in evidence)
                    accepted = supported and all(row["gain_kw"] > 0 and row["mse_gain"] > 0 and
                        row["positive_turbine_fraction"] >= settings["positive_fraction"] for row in evidence)
                    entries[f"{level}:{group}:{h}:{k}"] = dict(supported=supported, accepted=accepted,
                        score_kw=min(row["gain_kw"] for row in evidence) if accepted else 0., blocks=evidence)
    policy = dict(expert_id=model["expert_id"], contract=contract, settings=settings, entries=entries,
                  utility_outputs_sha256=digest.hexdigest(), score="min(two-block realized MAE gain, query empirical MAE gain); no shift guarantee")
    policy["policy_id"] = legacy.history.digest(policy)
    return policy


def apply(model, policy, block, contract):
    validate(model, contract, policy)
    output = candidates(model, block, contract)
    shape = output["prediction"].shape
    score, supported = np.zeros(shape), np.zeros(shape, bool)
    granularity = np.full(shape, -1, np.int8)
    for h in range(shape[1]):
        for k in range(shape[2]):
            remaining = np.ones(shape[0], bool)
            for level, bins, code in (("joint", block["context"]["joint_bin"], 2),
                                     ("power", block["context"]["power_bin"], 1),
                                     ("event", np.zeros(shape[0], int), 0)):
                for group in np.unique(bins):
                    row = policy["entries"][f"{level}:{group}:{h}:{k}"]
                    if row["supported"]:
                        selected = remaining & (bins == group)
                        score[selected, h, k] = np.minimum(row["score_kw"], output["local_gain_kw"][selected, h, k])
                        supported[selected, h, k] = True
                        granularity[selected, h, k] = code
                        remaining[selected] = False
    available = output["support"] & supported & (score > 0)
    best, active = np.where(available, score, -np.inf).argmax(-1), available.any(-1)
    chosen = np.take_along_axis(output["prediction"], best[..., None], -1).squeeze(-1)
    return dict(prediction=np.where(active, chosen, block["reference"]), action=np.where(active, best + 1, 0),
                physical_available=output["physical"], history_overlap=output["overlap"], available=available,
                granularity=granularity, alpha=output["alpha"], local_gain_kw=output["local_gain_kw"],
                variance_kw2=output["variance_kw2"], score_kw=score,
                expert_id=model["expert_id"], policy_id=policy["policy_id"])
