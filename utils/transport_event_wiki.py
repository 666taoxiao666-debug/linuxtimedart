"""Expert-bound, history-conditioned event residual knowledge.

Residual functions use physical coordinates, not a backbone's hidden states.
Utility evidence is valid only for the exact frozen expert/base/rule contract.
The temporal minimum is a conservative score, NOT a statistical guarantee.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FEATURES = ("intercept", "mean_power_ratio", "last_power_ratio", "wind_mean_div12",
            "wind_std_div3", "power_change_ratio", "wind_change_div5",
            "base_change_ratio", "wind_mean_times_change_div60", "last_minus_mean_power")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def implementation_digest():
    # Universal newlines preserve the same function identity on Windows/Linux.
    return hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest()


def seal_model(model):
    model["expert_bundle_id"] = digest({k: v for k, v in model.items() if k != "expert_bundle_id"})
    return model


def validate_identity(model, contract, policy=None):
    actual = digest({k: v for k, v in model.items() if k != "expert_bundle_id"})
    if actual != model["expert_bundle_id"]:
        raise ValueError("Frozen expert parameters or physical feature contract changed")
    if model["implementation_sha256"] != implementation_digest():
        raise ValueError("Expert implementation changed; regenerate utility evidence")
    if model["contract"] != contract:
        raise ValueError("Base checkpoint, normalization, event rules or forecast identity changed")
    if policy is not None:
        if policy["expert_bundle_id"] != actual or policy["contract"] != contract:
            raise ValueError("Utility evidence belongs to another actual expert/base")
        if policy["policy_id"] != digest({k: v for k, v in policy.items() if k != "policy_id"}):
            raise ValueError("Frozen utility policy changed")


def physical_features(context, reference, anchor, rated_power):
    reference, anchor = np.asarray(reference, float), np.asarray(anchor, float)
    if reference.ndim != 2 or reference.shape != anchor.shape or len(reference) == 0:
        raise ValueError("Aligned nonempty [window,horizon] forecasts required")
    if rated_power <= 0 or not np.isfinite(reference).all() or not np.isfinite(anchor).all():
        raise ValueError("Finite forecasts and a positive rated capacity required")
    n, horizon = reference.shape
    names = ("power_ratio", "wind_mean_mps", "wind_std_mps", "power_change_kw", "wind_change_mps")
    values = [np.asarray(context[k], float) for k in names]
    if any(v.shape != (n,) or not np.isfinite(v).all() for v in values):
        raise ValueError("Physical history coordinates do not align")
    p, wind, std, dp, dw = values
    last = anchor[:, 0] / rated_power
    fixed = np.column_stack((np.ones(n), p, last, wind / 12., std / 3., dp / rated_power,
                             dw / 5., wind * dw / 60., last - p))
    matrix = np.repeat(fixed[:, None], horizon, axis=1)
    return np.concatenate((matrix[:, :, :7], ((reference-anchor)/rated_power)[..., None],
                           matrix[:, :, 7:]), axis=-1)


def _source_ids(turbines, contract):
    turbines = np.asarray(turbines)
    source = set(contract["source_turbines"])
    if turbines.ndim != 1 or not set(turbines.tolist()).issubset(source):
        raise ValueError("Held-out turbine labels cannot enter expert fitting or evidence")
    return turbines


def fit_experts(context, reference, anchor, events, truth, turbines, contract, *,
                ridge_lambda=.01, max_scale=.2, composition_max_scale=.05,
                min_fit_windows=128, support_quantile=.95):
    turbines = _source_ids(turbines, contract)
    x = physical_features(context, reference, anchor, contract["rated_power_kw"])
    truth, events = np.asarray(truth, float), np.asarray(events, bool)
    n, horizon, dimension = x.shape
    if (horizon != contract["pred_len"] or truth.shape != (n, horizon)
            or events.shape != (n, len(contract["event_ids"]))):
        raise ValueError("Expert training labels and event vocabulary do not align")
    if not np.isfinite(truth).all() or ridge_lambda <= 0 or not 0 < max_scale <= 1:
        raise ValueError("Invalid expert training objective or bound")
    experts = []
    for index, event_id in enumerate(contract["event_ids"]):
        mask = events[:, index]
        count = int(mask.sum())
        mean = np.zeros((horizon, dimension)); scale = np.ones_like(mean)
        coef = np.zeros_like(mean); radius = np.zeros(horizon)
        if count >= min_fit_windows:
            xx = x[mask]
            mean = xx.mean(0); scale = xx.std(0)
            scale = np.where(scale > 1e-6, scale, 1.)
            mean[:, 0] = 0.; scale[:, 0] = 1.
            z = (xx-mean)/scale
            labels = (truth[mask]-np.asarray(reference)[mask]) / contract["rated_power_kw"]
            ids, counts = np.unique(turbines[mask], return_counts=True)
            weights = np.asarray([1./counts[np.searchsorted(ids, t)] for t in turbines[mask]])
            weights /= weights.sum()
            penalty = np.eye(dimension) * ridge_lambda; penalty[0, 0] = 0.
            for h in range(horizon):
                coef[h] = np.linalg.solve(z[:, h].T @ (weights[:, None]*z[:, h]) + penalty,
                                         z[:, h].T @ (weights*labels[:, h]))
                # Empirical history-support radius; target values never enter it.
                radius[h] = np.quantile(np.mean(z[:, h, 1:]**2, axis=1), support_quantile)
        experts.append(dict(event_id=event_id, active=count >= min_fit_windows,
            fit_windows=count, mean=mean.tolist(), scale=scale.tolist(),
            coefficients=coef.tolist(), support_radius=radius.tolist()))
    return seal_model(dict(schema_version=1, expert_type="capacity_normalized_physical_ridge",
        contract=contract, feature_names=list(FEATURES), implementation_sha256=implementation_digest(),
        ridge_lambda=ridge_lambda, max_scale=max_scale, composition_max_scale=composition_max_scale,
        support_quantile=support_quantile, min_fit_windows=min_fit_windows, experts=experts))


def candidates(model, context, reference, anchor, events, contract):
    validate_identity(model, contract)
    x = physical_features(context, reference, anchor, contract["rated_power_kw"])
    reference, events = np.asarray(reference, float), np.asarray(events, bool)
    n, horizon = reference.shape; count = len(model["experts"])
    if horizon != contract["pred_len"] or events.shape != (n, count):
        raise ValueError("Event IDs/order changed")
    delta = np.zeros((n, horizon, count+1)); physical = np.zeros_like(delta, dtype=bool)
    overlap = np.zeros_like(physical)
    for k, expert in enumerate(model["experts"]):
        if not expert["active"]:
            continue
        z = (x-np.asarray(expert["mean"])) / np.asarray(expert["scale"])
        value = np.einsum("nhd,hd->nh", z, np.asarray(expert["coefficients"]))
        delta[..., k] = np.clip(value, -model["max_scale"], model["max_scale"])*contract["rated_power_kw"]
        physical[..., k] = events[:, k, None]
        overlap[..., k] = (np.mean(z[..., 1:]**2, axis=-1) <= np.asarray(expert["support_radius"]))
    # Composition is an exact bounded sum of these frozen functions, not an
    # independently reinitialized adapter whose utility could be confused.
    delta[..., :count] *= physical[..., :count]
    delta[..., count] = np.clip(delta[..., :count].sum(-1),
        -model["composition_max_scale"]*contract["rated_power_kw"],
        model["composition_max_scale"]*contract["rated_power_kw"])
    physical[..., count] = physical[..., :count].sum(-1) >= 2
    overlap[..., count] = (overlap[..., :count] | ~physical[..., :count]).all(-1)
    prediction = np.clip(reference[..., None]+delta, 0., contract["rated_power_kw"])
    return prediction, physical, physical & overlap


def calibration_blocks(times, calibration_mask):
    times = np.asarray(times, dtype="datetime64[ns]")
    mask = np.asarray(calibration_mask, bool)
    cutoff = times[mask].min() + (times[mask].max()-times[mask].min()) // 2
    first = mask & (times[:, -1] < cutoff)
    second = mask & (times[:, 0] >= cutoff)
    if not first.any() or not second.any():
        raise ValueError("Two disjoint chronological utility blocks required")
    return first, second, cutoff


def fit_transfer_policy(model, blocks, contract, *, min_windows=64, min_turbines=8,
                        min_positive_fraction=2/3):
    validate_identity(model, contract)
    if len(blocks) != 2:
        raise ValueError("Exactly two disjoint utility calibration blocks required")
    prepared = []
    for block in blocks:
        _source_ids(block["turbines"], contract)
        candidate, physical, supported = candidates(model, block["context"], block["reference"],
                                                     block["anchor"], block["events"], contract)
        truth = np.asarray(block["truth"], float)
        if truth.shape != candidate.shape[:2] or not np.isfinite(truth).all():
            raise ValueError("Utility labels do not align")
        times = np.asarray(block["times"], dtype="datetime64[ns]")
        if times.shape != truth.shape or np.isnat(times).any():
            raise ValueError("Calibration target timestamps must align with forecasts")
        prepared.append(dict(block, supported=supported,
            gain=np.abs(block["reference"]-truth)[..., None]-np.abs(candidate-truth[..., None]),
            mse_gain=(block["reference"]-truth)[..., None]**2-(candidate-truth[..., None])**2,
            times=times))
    if prepared[0]["times"].max() >= prepared[1]["times"].min():
        raise ValueError("Utility calibration target windows overlap in time")
    levels = (("event", 1), ("power", 5), ("joint", 20))
    entries = {}
    for level, groups in levels:
        for group in range(groups):
            for k in range(len(model["experts"])+1):
                for h in range(contract["pred_len"]):
                    evidence = []
                    for block in prepared:
                        context = block["context"]
                        strata = np.zeros(len(block["truth"]), int) if level == "event" else np.asarray(context[level+"_bin"])
                        mask = (strata == group) & block["supported"][:, h, k]
                        ids = np.unique(np.asarray(block["turbines"])[mask])
                        wind_gains, wind_mse = [], []
                        for turbine in ids:
                            selected = mask & (np.asarray(block["turbines"]) == turbine)
                            wind_gains.append(float(block["gain"][selected, h, k].mean()))
                            wind_mse.append(float(block["mse_gain"][selected, h, k].mean()))
                        evidence.append(dict(windows=int(mask.sum()), turbines=len(ids),
                            turbine_balanced_gain_kw=float(np.mean(wind_gains)) if wind_gains else None,
                            turbine_balanced_mse_gain=float(np.mean(wind_mse)) if wind_mse else None,
                            positive_turbine_fraction=float(np.mean(np.asarray(wind_gains)>0)) if wind_gains else None))
                    supported = all(r["windows"] >= min_windows and r["turbines"] >= min_turbines for r in evidence)
                    accepted = supported and all(r["turbine_balanced_gain_kw"] > 0
                        and r["turbine_balanced_mse_gain"] > 0
                        and r["positive_turbine_fraction"] >= min_positive_fraction for r in evidence)
                    score = min(r["turbine_balanced_gain_kw"] for r in evidence) if accepted else 0.
                    entries[f"{level}:{group}:{h}:{k}"] = dict(supported=supported, accepted=accepted,
                        score_kw=score, blocks=evidence)
    policy = dict(schema_version=1, expert_bundle_id=model["expert_bundle_id"], contract=contract,
        min_windows=min_windows, min_turbines=min_turbines, min_positive_fraction=min_positive_fraction,
        granularity_rule="finest supported physical cell; known harmful fine cell vetoes coarser backoff",
        score_definition="minimum of two time-block turbine-balanced gains; descriptive, not a confidence guarantee",
        entries=entries)
    policy["policy_id"] = digest(policy)
    return policy


def apply_transfer(model, policy, context, reference, anchor, events, contract, *, event_only=False):
    """Prediction API deliberately has no target, turbine ID, or future argument."""
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
                    selected = remaining & (bins == group)
                    supported[selected, h, k] = True
                    granularity[selected, h, k] = code
                    score[selected, h, k] = row["score_kw"]
                    remaining[selected] = False
    available = overlap & supported & (score > 0)
    masked = np.where(available, score, -np.inf)
    best = masked.argmax(-1); active = available.any(-1)
    selected = np.take_along_axis(candidate, best[..., None], axis=-1).squeeze(-1)
    prediction = np.where(active, selected, np.asarray(reference, float))
    return dict(prediction=prediction, action=np.where(active, best+1, 0),
        physical_available=physical, history_overlap=overlap, evidence_supported=supported,
        available=available, score_kw=score, granularity=granularity,
        expert_bundle_id=model["expert_bundle_id"], policy_id=policy["policy_id"])
