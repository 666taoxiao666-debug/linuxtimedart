"""Time-stamped, weather-block-balanced executable knowledge cases.

This is a new readout, not a retroactive change to the frozen v1/v2 experts.
Block balance changes probability mass, not the number of statistically
independent observations. All candidate outcomes must have matured at issue time.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from utils import kelmarsh_hourly_wiki as legacy


def implementation_digest():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def balanced_weights(turbines, time_blocks, mode="episode"):
    """Equal block mass, then equal turbine mass inside each represented block."""
    turbines, time_blocks = np.asarray(turbines), np.asarray(time_blocks)
    if turbines.ndim != 2 or turbines.shape != time_blocks.shape or not turbines.shape[1]:
        raise ValueError("Neighbor units and time blocks must align")
    if mode not in ("episode", "turbine_control"):
        raise ValueError("Unknown weighting rule")
    weights = np.zeros(turbines.shape, float)
    for row in range(len(turbines)):
        groups = np.unique(time_blocks[row]) if mode == "episode" else [None]
        for group in groups:
            mask = time_blocks[row] == group if group is not None else np.ones(turbines.shape[1], bool)
            units = np.unique(turbines[row, mask])
            for unit in units:
                selected = mask & (turbines[row] == unit)
                weights[row, selected] = 1. / (len(groups) * len(units) * selected.sum())
    return weights


def readout(values, turbines, time_blocks, base_change, max_scale, positive_fraction, mode="episode"):
    values, base = np.asarray(values, float), np.asarray(base_change, float)
    turbines, time_blocks = np.asarray(turbines), np.asarray(time_blocks)
    weights = balanced_weights(turbines, time_blocks, mode)
    if values.shape != weights.shape or base.shape != values.shape[:1]:
        raise ValueError("Forecast and neighbor outcomes must align")
    if not np.isfinite(values).all() or not np.isfinite(base).all() or max_scale <= 0:
        raise ValueError("Invalid outcomes or correction bound")
    order = np.argsort(values, axis=1, kind="stable")
    vv, ww = np.take_along_axis(values, order, 1), np.take_along_axis(weights, order, 1)
    position = (ww.cumsum(1) >= .5 - 1e-14).argmax(1)
    median = vv[np.arange(len(vv)), position]
    mean = (weights * values).sum(1)
    direction = np.clip(median - base, -max_scale, max_scale)
    alpha = np.zeros_like(base)
    np.divide(mean - base, direction, out=alpha, where=np.abs(direction) > 1e-12)
    alpha = np.clip(alpha, 0., 1.)
    prediction = base + alpha * direction
    gain = np.abs(base[:, None] - values) - np.abs(prediction[:, None] - values)
    squared = (base[:, None] - values) ** 2 - (prediction[:, None] - values) ** 2
    donor_positive = []
    for row in range(len(values)):
        unit_gains = []
        for unit in np.unique(turbines[row]):
            mask = turbines[row] == unit
            unit_weights = weights[row, mask] / weights[row, mask].sum()
            unit_gains.append((unit_weights @ gain[row, mask] > 0) and
                              (unit_weights @ squared[row, mask] > 0))
        donor_positive.append(np.mean(unit_gains))
    mae_gain, mse_gain = (weights * gain).sum(1), (weights * squared).sum(1)
    return dict(prediction=prediction, alpha=alpha, mae_gain=mae_gain, mse_gain=mse_gain,
                eligible=(mae_gain > 1e-12) & (mse_gain > 1e-12) &
                         (np.asarray(donor_positive) >= positive_fraction),
                weights=weights, variance=(weights * (values - mean[:, None]) ** 2).sum(1),
                time_blocks=np.array([len(np.unique(row)) for row in time_blocks]),
                kish_weight_count=1. / (weights ** 2).sum(1))


def make_model(memory, memory_block, contract, *, mode="episode", block_hours=24):
    """Bind immutable case timestamps to the exact pre-existing memory rows."""
    legacy.validate(memory, contract)
    if mode not in ("episode", "turbine_control") or not isinstance(block_hours, int) or block_hours <= 0:
        raise ValueError("Invalid predeclared case weighting")
    legacy.history._source_ids(memory_block["turbines"], contract)
    times = np.asarray(memory_block["times"]).astype("datetime64[ns]")
    if times.shape != np.asarray(memory_block["truth"]).shape or np.isnat(times).any():
        raise ValueError("Missing or misaligned target availability times")
    keys = legacy.keys(memory_block["context"], memory_block["anchor"], contract["rated_power_kw"], memory["settings"]["use_controls"])
    outcomes = (memory_block["truth"] - memory_block["anchor"]) / contract["rated_power_kw"]
    notes = []
    for k, expert in enumerate(memory["experts"]):
        mask = memory_block["events"][:, k]
        note = dict(event_id=expert["event_id"], active=expert["active"])
        if expert["active"]:
            for actual, expected in ((keys[mask], expert["keys"]), (outcomes[mask], expert["values"]),
                                     (memory_block["turbines"][mask], expert["turbines"])):
                if not np.array_equal(actual, expected):
                    raise ValueError("Timestamp ledger does not match exact memory row identity")
            note.update(target_times_ns=times[mask].astype(np.int64).tolist(),
                        weather_blocks=(times[mask, 0].astype("datetime64[h]").astype(np.int64) // block_hours).tolist(),
                        outcome="future power minus last observed power, divided by capacity")
        notes.append(note)
    model = dict(memory=memory, contract=contract, mode=mode, block_hours=block_hours,
                 notes=notes, implementation_sha256=implementation_digest(),
                 semantics="structured executable cases; no LLM-created numeric evidence",
                 uncertainty="time blocks and Kish count are descriptive, not independent sample sizes")
    model["expert_id"] = legacy.history.digest(model)
    return model


def validate(model, contract):
    if model["expert_id"] != legacy.history.digest({k: v for k, v in model.items() if k != "expert_id"}):
        raise ValueError("Executable case identity changed; recompute utility evidence")
    if contract != model["contract"] or model["implementation_sha256"] != implementation_digest():
        raise ValueError("Base, scaler, rule, horizon or execution implementation changed")
    legacy.validate(model["memory"], contract)


def candidates(model, block, contract):
    """History-only inference. Truth and turbine IDs are not required in a query."""
    validate(model, contract)
    capacity, memory = contract["rated_power_kw"], model["memory"]
    settings = memory["settings"]
    reference = np.asarray(block["reference"], float)
    times = np.asarray(block["times"]).astype("datetime64[ns]")
    if times.shape != reference.shape or np.isnat(times).any():
        raise ValueError("Issue-time schedule is required")
    # Hourly dataset target[0] is one hour after the forecast origin.
    origins = times[:, 0] - np.timedelta64(1, "h")
    x = legacy.keys(block["context"], block["anchor"], capacity, settings["use_controls"])
    shape = (*reference.shape, len(legacy.EVENTS) + 1)
    prediction = np.repeat(reference[..., None], shape[-1], axis=-1)
    support, physical = np.zeros(shape, bool), np.zeros(shape, bool)
    alpha, gain, variance = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    blocks_count, kish = np.zeros(shape, int), np.zeros(shape)
    for k, (expert, note) in enumerate(zip(memory["experts"], model["notes"])):
        mask = np.asarray(block["events"][:, k], bool)
        if not expert["active"] or not mask.any():
            continue
        # The complete fixed memory must be available; never silently rebuild it
        # per query and then reuse evidence for a different neighborhood rule.
        if np.any(origins[mask].astype(np.int64) < np.max(note["target_times_ns"])):
            raise ValueError("Knowledge outcome was not available at forecast origin")
        mean, scale = np.asarray(expert["mean"]), np.asarray(expert["scale"])
        ids, values = np.asarray(expert["turbines"]), np.asarray(expert["values"])
        idx, distance, diversity = legacy.neighbors._neighbors((x[mask] - mean) / scale,
            (np.asarray(expert["keys"]) - mean) / scale, ids, settings["neighbors"])
        overlap = (distance <= expert["support_radius"]) & (diversity >= settings["min_neighbor_turbines"])
        physical[mask, :, k] = True
        groups = np.asarray(note["weather_blocks"])[idx]
        for h in range(reference.shape[1]):
            out = readout(values[idx, h], ids[idx], groups,
                (reference[mask, h] - block["anchor"][mask, h]) / capacity,
                settings["max_scale"], 2/3, model["mode"])
            prediction[mask, h, k] = block["anchor"][mask, h] + capacity * out["prediction"]
            support[mask, h, k] = overlap & out["eligible"]
            alpha[mask, h, k], gain[mask, h, k] = out["alpha"], capacity * out["mae_gain"]
            variance[mask, h, k] = capacity ** 2 * out["variance"]
            blocks_count[mask, h, k], kish[mask, h, k] = out["time_blocks"], out["kish_weight_count"]
    return dict(prediction=prediction, physical=physical, support=support, alpha=alpha,
                local_gain_kw=gain, variance_kw2=variance, weather_blocks=blocks_count,
                kish_weight_count=kish, expert_id=model["expert_id"])


def fit_policy(model, blocks, contract, settings):
    """Reassess actual new outputs on donor-only chronological train blocks."""
    validate(model, contract)
    if len(blocks) != 2 or blocks[0]["times"].max() >= blocks[1]["times"].min():
        raise ValueError("Two disjoint chronological train-utility blocks required")
    prepared, identity = [], hashlib.sha256()
    for block in blocks:
        legacy.history._source_ids(block["turbines"], contract)
        output = candidates(model, block, contract)
        truth = np.asarray(block["truth"], float)
        if truth.shape != output["prediction"].shape[:2] or not np.isfinite(truth).all():
            raise ValueError("Misaligned utility outcomes")
        gain = np.abs(block["reference"] - truth)[..., None] - np.abs(output["prediction"] - truth[..., None])
        mse = (block["reference"] - truth)[..., None] ** 2 - (output["prediction"] - truth[..., None]) ** 2
        for value in (truth, output["prediction"], output["support"], block["times"].astype(np.int64), block["turbines"]):
            identity.update(np.ascontiguousarray(value).tobytes())
        prepared.append(dict(block, gain=gain, mse=mse, support=output["support"]))
    entries = {}
    for level, groups in (("event", 1), ("power", 5), ("joint", 20)):
        for group in range(groups):
            for h in range(contract["pred_len"]):
                for k in range(len(legacy.EVENTS) + 1):
                    evidence = []
                    for block in prepared:
                        strata = np.zeros(len(block["truth"]), int) if level == "event" else block["context"][level + "_bin"]
                        mask = (strata == group) & block["support"][:, h, k]
                        ids = np.unique(block["turbines"][mask])
                        gains = [block["gain"][mask & (block["turbines"] == unit), h, k].mean() for unit in ids]
                        squared = [block["mse"][mask & (block["turbines"] == unit), h, k].mean() for unit in ids]
                        time_groups = block["times"][mask, 0].astype("datetime64[h]").astype(np.int64) // model["block_hours"]
                        evidence.append(dict(windows=int(mask.sum()), turbines=len(ids), weather_blocks=len(np.unique(time_groups)),
                            gain_kw=float(np.mean(gains)) if gains else None, mse_gain=float(np.mean(squared)) if squared else None,
                            positive_turbine_fraction=float(np.mean(np.asarray(gains) > 0)) if gains else None))
                    supported = all(row["windows"] >= settings["min_windows"] and row["turbines"] >= settings["min_turbines"] for row in evidence)
                    accepted = supported and all(row["gain_kw"] > 0 and row["mse_gain"] > 0 and
                        row["positive_turbine_fraction"] >= settings["positive_fraction"] for row in evidence)
                    entries[f"{level}:{group}:{h}:{k}"] = dict(supported=supported, accepted=accepted,
                        score_kw=min(row["gain_kw"] for row in evidence) if accepted else 0., blocks=evidence)
    policy = dict(expert_id=model["expert_id"], contract=contract, settings=settings, entries=entries,
                  utility_outputs_sha256=identity.hexdigest(),
                  score="minimum realized two-block gain; not a transport confidence bound")
    policy["policy_id"] = legacy.history.digest(policy)
    return policy


def apply_with_policy(model, policy, output, block, contract):
    """Use only newly assessed evidence for this exact weighted readout.

    No v1/v2 policy reuse is allowed.
    """
    validate(model, contract)
    if policy["expert_id"] != model["expert_id"] or output["expert_id"] != model["expert_id"]:
        raise ValueError("Benefit evidence belongs to a different executing expert")
    if policy["contract"] != contract or policy["policy_id"] != legacy.history.digest({k: v for k, v in policy.items() if k != "policy_id"}):
        raise ValueError("Policy identity changed")
    score = np.zeros_like(output["prediction"])
    for h in range(score.shape[1]):
        for k in range(score.shape[2]):
            remaining = np.ones(score.shape[0], bool)
            for level, bins in (("joint", block["context"]["joint_bin"]), ("power", block["context"]["power_bin"]), ("event", np.zeros(score.shape[0], int))):
                for group in np.unique(bins):
                    row = policy["entries"][f"{level}:{group}:{h}:{k}"]
                    if row["supported"]:
                        mask = remaining & (bins == group)
                        score[mask, h, k] = np.minimum(row["score_kw"], output["local_gain_kw"][mask, h, k])
                        remaining[mask] = False
    available = output["support"] & (score > 0)
    best, active = np.where(available, score, -np.inf).argmax(-1), available.any(-1)
    chosen = np.take_along_axis(output["prediction"], best[..., None], -1).squeeze(-1)
    return dict(prediction=np.where(active, chosen, block["reference"]), action=np.where(active, best + 1, 0))
