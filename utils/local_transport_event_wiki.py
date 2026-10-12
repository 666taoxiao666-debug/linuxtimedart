"""Frozen physical case experts with utility evaluated on their actual outputs.

Each forecast uses a turbine-balanced median of nearby training residuals.
The stored cases, neighbor rule, support radius and base all belong to the
expert identity. Historical v12 code remains intact for reproducible audits.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from utils import transport_event_wiki as legacy

FEATURES = legacy.FEATURES
digest = legacy.digest
physical_features = legacy.physical_features
calibration_blocks = legacy.calibration_blocks


def implementation_digest():
    return hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest()


def validate_identity(model, contract, policy=None):
    actual = digest({k:v for k,v in model.items() if k != "expert_bundle_id"})
    if actual != model["expert_bundle_id"]:
        raise ValueError("Frozen expert cases or neighbor rule changed")
    if (model["implementation_sha256"] != implementation_digest()
            or model["physical_features_sha256"] != legacy.implementation_digest()):
        raise ValueError("Expert implementation changed; regenerate utility evidence")
    if model["contract"] != contract:
        raise ValueError("Base checkpoint, physical coordinates or event rules changed")
    if policy is not None:
        if policy["expert_bundle_id"] != actual or policy["contract"] != contract:
            raise ValueError("Utility evidence belongs to another actual expert/base")
        if policy["policy_id"] != digest({k:v for k,v in policy.items() if k != "policy_id"}):
            raise ValueError("Frozen utility policy changed")


def _neighbors(query, keys, turbines, k, excluded_turbines=None):
    # Chunked distance evaluation limits memory. Same-turbine cases are
    # excluded only when estimating fit-support distances, never at inference.
    indices, distances, distinct = [], [], []
    key_norm = np.sum(keys*keys, axis=1)
    for start in range(0, len(query), 128):
        q = query[start:start+128]
        d = np.maximum((q*q).sum(1)[:, None]+key_norm[None]-2*q@keys.T, 0.)/keys.shape[1]
        if excluded_turbines is not None:
            d[np.asarray(excluded_turbines)[start:start+len(q), None] == turbines[None]] = np.inf
        idx = np.argpartition(d, k-1, axis=1)[:, :k]
        selected = np.take_along_axis(d, idx, axis=1)
        indices.append(idx)
        distances.append(selected.max(1))
        distinct.extend(len(np.unique(turbines[row])) for row in idx)
    return np.concatenate(indices), np.concatenate(distances), np.asarray(distinct)


def _balanced_median(values, ids):
    # Give each represented source turbine the same total mass, independent
    # of how many correlated windows it contributes to the local neighborhood.
    counts = (ids[:, :, None] == ids[:, None, :]).sum(-1)
    weights = 1./counts
    order = np.argsort(values, axis=1, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=1)
    sorted_weights = np.take_along_axis(weights, order, axis=1)
    position = (sorted_weights.cumsum(1) >= sorted_weights.sum(1)[:, None]/2).argmax(1)
    return sorted_values[np.arange(len(values)), position]


def _keys(expert, h):
    history = np.asarray(expert["history_keys"])
    change = np.asarray(expert["forecast_change_keys"])[:, h:h+1]
    return np.concatenate((history[:, :7], change, history[:, 7:]), axis=1)


def fit_experts(context, reference, anchor, events, truth, turbines, contract, *,
                neighbors=32, min_neighbor_turbines=8, max_scale=.2,
                composition_max_scale=.05, min_fit_windows=128, support_quantile=.95):
    turbines = legacy._source_ids(turbines, contract)
    x = physical_features(context, reference, anchor, contract["rated_power_kw"])
    truth, events = np.asarray(truth, float), np.asarray(events, bool)
    n, horizon, dimension = x.shape
    if (truth.shape != (n, horizon) or horizon != contract["pred_len"]
            or events.shape != (n, len(contract["event_ids"])) or len(turbines) != n
            or not np.isfinite(truth).all()):
        raise ValueError("Expert labels, turbines and event vocabulary do not align")
    if not 1 <= min_neighbor_turbines <= neighbors < min_fit_windows or not 0 < support_quantile < 1:
        raise ValueError("Invalid fixed neighborhood/support specification")
    experts = []
    for k, event_id in enumerate(contract["event_ids"]):
        mask = events[:, k]
        xx, ids = x[mask], turbines[mask]
        unique, counts = np.unique(ids, return_counts=True)
        active = (len(ids) >= min_fit_windows and len(unique) >= min_neighbor_turbines+1
                  and len(ids)-counts.max(initial=0) >= neighbors)
        expert = dict(event_id=event_id, active=bool(active), fit_windows=int(mask.sum()),
            fit_turbines=int(len(unique)), history_keys=[], forecast_change_keys=[],
            residuals=[], turbines=[], mean=[], scale=[], support_radius=[])
        if active:
            mean, scale = xx.mean(0), xx.std(0)
            scale = np.where(scale > 1e-6, scale, 1.)
            mean[:, 0] = 0.; scale[:, 0] = 1.
            radius = []
            for h in range(horizon):
                z = (xx[:, h]-mean[h])/scale[h]
                _, distance, diversity = _neighbors(z, z, ids, neighbors, excluded_turbines=ids)
                eligible = np.isfinite(distance) & (diversity >= min_neighbor_turbines)
                radius.append(float(np.quantile(distance[eligible], support_quantile)) if eligible.any() else 0.)
            expert.update(history_keys=np.delete(xx[:, 0], 7, axis=1).tolist(),
                forecast_change_keys=xx[:, :, 7].tolist(),
                residuals=((truth[mask]-np.asarray(reference)[mask])/contract["rated_power_kw"]).tolist(),
                turbines=ids.tolist(), mean=mean.tolist(), scale=scale.tolist(), support_radius=radius)
        experts.append(expert)
    return legacy.seal_model(dict(schema_version=1, expert_type="physical_neighbor_balanced_median",
        contract=contract, feature_names=list(FEATURES), implementation_sha256=implementation_digest(),
        physical_features_sha256=legacy.implementation_digest(), neighbors=neighbors,
        min_neighbor_turbines=min_neighbor_turbines, max_scale=max_scale,
        composition_max_scale=composition_max_scale, min_fit_windows=min_fit_windows,
        support_quantile=support_quantile, experts=experts))


def candidates(model, context, reference, anchor, events, contract):
    validate_identity(model, contract)
    x = physical_features(context, reference, anchor, contract["rated_power_kw"])
    reference, events = np.asarray(reference, float), np.asarray(events, bool)
    n, horizon = reference.shape
    count = len(model["experts"])
    if horizon != contract["pred_len"] or events.shape != (n, count):
        raise ValueError("Event IDs/order changed")
    delta = np.zeros((n, horizon, count+1))
    physical = np.zeros_like(delta, dtype=bool); overlap = np.zeros_like(physical)
    for k, expert in enumerate(model["experts"]):
        if not expert["active"]:
            continue
        chosen = events[:, k]
        if not chosen.any():
            continue
        mean, scale = np.asarray(expert["mean"]), np.asarray(expert["scale"])
        residuals, ids = np.asarray(expert["residuals"]), np.asarray(expert["turbines"])
        for h in range(horizon):
            query = (x[chosen, h]-mean[h])/scale[h]
            keys = (_keys(expert, h)-mean[h])/scale[h]
            idx, distance, diversity = _neighbors(query, keys, ids, model["neighbors"])
            value = _balanced_median(residuals[idx, h], ids[idx])
            delta[chosen, h, k] = np.clip(value, -model["max_scale"], model["max_scale"])*contract["rated_power_kw"]
            physical[chosen, h, k] = True
            overlap[chosen, h, k] = ((distance <= expert["support_radius"][h])
                & (diversity >= model["min_neighbor_turbines"]))
    delta[..., count] = np.clip(delta[..., :count].sum(-1),
        -model["composition_max_scale"]*contract["rated_power_kw"],
        model["composition_max_scale"]*contract["rated_power_kw"])
    physical[..., count] = physical[..., :count].sum(-1) >= 2
    overlap[..., count] = (overlap[..., :count] | ~physical[..., :count]).all(-1)
    return np.clip(reference[..., None]+delta, 0., contract["rated_power_kw"]), physical, physical & overlap


def fit_transfer_policy(model, blocks, contract, *, min_windows=64, min_turbines=8,
                        min_positive_fraction=2/3):
    """Identical v12 utility decision, evaluated on the new actual candidates."""
    validate_identity(model, contract)
    if len(blocks) != 2:
        raise ValueError("Exactly two disjoint utility calibration blocks required")
    prepared = []
    for block in blocks:
        legacy._source_ids(block["turbines"], contract)
        candidate, _, supported = candidates(model, block["context"], block["reference"],
                                             block["anchor"], block["events"], contract)
        truth = np.asarray(block["truth"], float)
        times = np.asarray(block["times"], dtype="datetime64[ns]")
        if truth.shape != candidate.shape[:2] or times.shape != truth.shape or np.isnat(times).any() or not np.isfinite(truth).all():
            raise ValueError("Utility labels and timestamps do not align")
        prepared.append(dict(block, supported=supported,
            gain=np.abs(block["reference"]-truth)[..., None]-np.abs(candidate-truth[..., None]),
            mse_gain=(block["reference"]-truth)[..., None]**2-(candidate-truth[..., None])**2, times=times))
    if prepared[0]["times"].max() >= prepared[1]["times"].min():
        raise ValueError("Utility calibration target windows overlap in time")
    entries = {}
    for level, groups in (("event", 1), ("power", 5), ("joint", 20)):
        for group in range(groups):
            for k in range(len(model["experts"])+1):
                for h in range(contract["pred_len"]):
                    evidence = []
                    for block in prepared:
                        strata = np.zeros(len(block["truth"]), int) if level == "event" else np.asarray(block["context"][level+"_bin"])
                        mask = (strata == group) & block["supported"][:, h, k]
                        ids = np.unique(np.asarray(block["turbines"])[mask])
                        gains, squared = [], []
                        for turbine in ids:
                            chosen = mask & (np.asarray(block["turbines"]) == turbine)
                            gains.append(float(block["gain"][chosen, h, k].mean()))
                            squared.append(float(block["mse_gain"][chosen, h, k].mean()))
                        evidence.append(dict(windows=int(mask.sum()), turbines=len(ids),
                            turbine_balanced_gain_kw=float(np.mean(gains)) if gains else None,
                            turbine_balanced_mse_gain=float(np.mean(squared)) if squared else None,
                            positive_turbine_fraction=float(np.mean(np.asarray(gains)>0)) if gains else None))
                    supported = all(r["windows"] >= min_windows and r["turbines"] >= min_turbines for r in evidence)
                    accepted = supported and all(r["turbine_balanced_gain_kw"] > 0
                        and r["turbine_balanced_mse_gain"] > 0
                        and r["positive_turbine_fraction"] >= min_positive_fraction for r in evidence)
                    entries[f"{level}:{group}:{h}:{k}"] = dict(supported=supported, accepted=accepted,
                        score_kw=min(r["turbine_balanced_gain_kw"] for r in evidence) if accepted else 0., blocks=evidence)
    policy = dict(schema_version=1, expert_bundle_id=model["expert_bundle_id"], contract=contract,
        min_windows=min_windows, min_turbines=min_turbines, min_positive_fraction=min_positive_fraction,
        granularity_rule="finest supported physical cell; known harmful fine cell vetoes coarser backoff",
        score_definition="minimum of two time-block turbine-balanced gains; descriptive, not a confidence guarantee",
        entries=entries)
    policy["policy_id"] = digest(policy)
    return policy


def apply_transfer(model, policy, context, reference, anchor, events, contract, *, event_only=False):
    validate_identity(model, contract, policy)
    candidate, physical, overlap = candidates(model, context, reference, anchor, events, contract)
    n, horizon, count = candidate.shape
    score = np.zeros_like(candidate); supported = np.zeros_like(physical)
    granularity = np.full_like(candidate, -1, dtype=np.int8)
    levels = (("event", np.zeros(n, int), 0),) if event_only else (
        ("joint", np.asarray(context["joint_bin"]), 2),
        ("power", np.asarray(context["power_bin"]), 1), ("event", np.zeros(n, int), 0))
    for h in range(horizon):
        for k in range(count):
            remaining = np.ones(n, bool)
            for level, bins, code in levels:
                for group in np.unique(bins):
                    row = policy["entries"][f"{level}:{int(group)}:{h}:{k}"]
                    if not row["supported"]:
                        continue
                    chosen = remaining & (bins == group)
                    supported[chosen, h, k] = True; granularity[chosen, h, k] = code
                    score[chosen, h, k] = row["score_kw"]; remaining[chosen] = False
    available = overlap & supported & (score > 0)
    best = np.where(available, score, -np.inf).argmax(-1); active = available.any(-1)
    selected = np.take_along_axis(candidate, best[..., None], axis=-1).squeeze(-1)
    return dict(prediction=np.where(active, selected, np.asarray(reference, float)),
        action=np.where(active, best+1, 0), physical_available=physical, history_overlap=overlap,
        evidence_supported=supported, available=available, score_kw=score, granularity=granularity,
        expert_bundle_id=model["expert_bundle_id"], policy_id=policy["policy_id"])
