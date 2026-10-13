"""Physical transition memory; utility belongs to an expert/base pair.

The transition readout stores Y - last_observed_power, not Y - a particular
backbone prediction. Memory can therefore survive a base replacement, but its
utility policy cannot. Residual and no-control arms are explicit ablations.
No transport guarantee follows from a matched history alone.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from utils import transport_event_wiki as legacy
from utils import local_transport_event_wiki as local

digest = legacy.digest
HISTORY_NAMES = tuple(legacy.FEATURES[i] for i in (1, 2, 3, 4, 5, 6, 8, 9))
CONTROL_NAMES = ("pitch_mean_div90", "pitch_last_div90", "pitch_change_div90",
                 "relative_yaw_cos_mean", "relative_yaw_cos_last")
BASE_FIELDS = ("base_checkpoint_sha256", "base_manifest_sha256")


def implementation_digest():
    return hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest()


def control_context(pitch, relative_yaw_cos):
    """Only last-12 observed, original-unit sensor histories are accepted."""
    pitch, cosine = np.asarray(pitch, float), np.asarray(relative_yaw_cos, float)
    if (pitch.ndim != 2 or pitch.shape[1] != 12 or pitch.shape != cosine.shape
            or not np.isfinite(pitch).all() or not np.isfinite(cosine).all()
            or np.any(np.abs(cosine) > 1 + 1e-5)):
        raise ValueError("Finite aligned last-12 pitch and relative-yaw histories required")
    return dict(pitch_mean_div90=pitch.mean(1)/90, pitch_last_div90=pitch[:, -1]/90,
                pitch_change_div90=(pitch[:, 6:].mean(1)-pitch[:, :6].mean(1))/90,
                relative_yaw_cos_mean=cosine.mean(1), relative_yaw_cos_last=cosine[:, -1])


def history_keys(context, anchor, capacity, use_controls):
    # Passing anchor twice removes all dependence on a base output, including
    # from key standardization/support estimation. The base-change slot is
    # excluded in BOTH readout ablations, so their key geometry is identical.
    values = legacy.physical_features(context, anchor, anchor, capacity)
    keys = values[:, 0, (1, 2, 3, 4, 5, 6, 8, 9)]
    if use_controls:
        controls = [np.asarray(context[name], float) for name in CONTROL_NAMES]
        if any(v.shape != (len(keys),) or not np.isfinite(v).all() for v in controls):
            raise ValueError("Observed control coordinates are missing or unaligned")
        keys = np.column_stack((keys, *controls))
    return keys


def _bank_contract(contract, readout):
    if readout not in ("transition", "residual"):
        raise ValueError("Unknown physical knowledge readout")
    if any(not isinstance(contract.get(key), str) or len(contract[key]) != 64 for key in BASE_FIELDS):
        raise ValueError("Explicit base checkpoint and manifest hashes required")
    # Residual targets DO depend on the source base. Transition targets do not.
    return {key: value for key, value in contract.items()
            if readout == "residual" or key not in BASE_FIELDS}


def validate_identity(model, contract, policy=None):
    bank_id = digest({k: v for k, v in model.items() if k != "expert_bundle_id"})
    if bank_id != model["expert_bundle_id"]:
        raise ValueError("Frozen transition memory or retrieval definition changed")
    if (model["implementation_sha256"] != implementation_digest()
            or model["history_implementation_sha256"] != legacy.implementation_digest()
            or model["neighbor_implementation_sha256"] != local.implementation_digest()):
        raise ValueError("Expert implementation changed; regenerate evidence")
    if model["bank_contract"] != _bank_contract(contract, model["readout"]):
        raise ValueError("Physical memory contract changed")
    if policy is not None:
        if (policy["expert_bundle_id"] != bank_id or policy["execution_contract"] != contract):
            raise ValueError("Utility evidence belongs to another expert/base execution pair")
        if policy["policy_id"] != digest({k: v for k, v in policy.items() if k != "policy_id"}):
            raise ValueError("Frozen execution policy changed")


def _forecasts(reference, anchor, contract):
    reference, anchor = np.asarray(reference, float), np.asarray(anchor, float)
    if (reference.ndim != 2 or reference.shape != anchor.shape or not len(reference)
            or reference.shape[1] != contract["pred_len"]
            or not np.isfinite(reference).all() or not np.isfinite(anchor).all()
            or not np.isfinite(contract["rated_power_kw"]) or contract["rated_power_kw"] <= 0
            or not np.allclose(anchor, anchor[:, :1], rtol=0, atol=1e-8)):
        raise ValueError("Aligned finite base and constant history-anchor forecasts required")
    if np.any(reference < 0) or np.any(reference > contract["rated_power_kw"]):
        raise ValueError("Base must use the declared [0,capacity] reporting convention")
    return reference, anchor


def fit_experts(context, reference, anchor, events, truth, turbines, contract, *,
                readout="transition", use_controls=True, neighbors=32,
                min_neighbor_turbines=8, min_fit_windows=128,
                support_quantile=.95, max_scale=.2, composition_max_scale=.05):
    bank_contract = _bank_contract(contract, readout)
    reference, anchor = _forecasts(reference, anchor, contract)
    turbines = legacy._source_ids(turbines, contract)
    truth, events = np.asarray(truth, float), np.asarray(events, bool)
    n, horizon = reference.shape
    if (truth.shape != reference.shape or not np.isfinite(truth).all()
            or len(turbines) != n or events.shape != (n, len(contract["event_ids"]))):
        raise ValueError("Source labels, turbines or events do not align")
    if (not 1 <= min_neighbor_turbines <= neighbors < min_fit_windows
            or not 0 < support_quantile < 1 or not 0 < max_scale <= 1
            or not 0 < composition_max_scale <= max_scale):
        raise ValueError("Invalid declared expert support/bounds")
    keys = history_keys(context, anchor, contract["rated_power_kw"], use_controls)
    target = (truth - (anchor if readout == "transition" else reference))/contract["rated_power_kw"]
    experts = []
    for k, event_id in enumerate(contract["event_ids"]):
        mask = events[:, k]; xx, ids = keys[mask], turbines[mask]
        unique, counts = np.unique(ids, return_counts=True)
        active = (len(ids) >= min_fit_windows and len(unique) >= min_neighbor_turbines+1
                  and len(ids)-counts.max(initial=0) >= neighbors)
        expert = dict(event_id=event_id, active=bool(active), fit_windows=int(mask.sum()),
                      fit_turbines=len(unique), keys=[], values=[], turbines=[],
                      mean=[], scale=[], support_radius=0.)
        if active:
            mean, scale = xx.mean(0), xx.std(0)
            scale = np.where(scale > 1e-6, scale, 1.)
            z = (xx-mean)/scale
            _, distance, diversity = local._neighbors(z, z, ids, neighbors, excluded_turbines=ids)
            eligible = np.isfinite(distance) & (diversity >= min_neighbor_turbines)
            radius = float(np.quantile(distance[eligible], support_quantile)) if eligible.any() else 0.
            expert.update(keys=xx.tolist(), values=target[mask].tolist(), turbines=ids.tolist(),
                          mean=mean.tolist(), scale=scale.tolist(), support_radius=radius)
        experts.append(expert)
    model = dict(schema_version=1, expert_type="physical_control_transition_memory",
                 bank_contract=bank_contract, readout=readout, use_controls=bool(use_controls),
                 feature_names=list(HISTORY_NAMES + (CONTROL_NAMES if use_controls else ())),
                 implementation_sha256=implementation_digest(),
                 history_implementation_sha256=legacy.implementation_digest(),
                 neighbor_implementation_sha256=local.implementation_digest(), neighbors=neighbors,
                 min_neighbor_turbines=min_neighbor_turbines, min_fit_windows=min_fit_windows,
                 support_quantile=support_quantile, max_scale=max_scale,
                 composition_max_scale=composition_max_scale, experts=experts)
    return legacy.seal_model(model)


def candidates(model, context, reference, anchor, events, contract):
    validate_identity(model, contract)
    reference, anchor = _forecasts(reference, anchor, contract)
    events = np.asarray(events, bool)
    n, horizon = reference.shape; count = len(model["experts"])
    if events.shape != (n, count):
        raise ValueError("Physical event IDs/order changed")
    keys = history_keys(context, anchor, contract["rated_power_kw"], model["use_controls"])
    delta = np.zeros((n, horizon, count+1))
    physical = np.zeros_like(delta, bool); overlap = np.zeros_like(physical)
    for k, expert in enumerate(model["experts"]):
        chosen = events[:, k]
        if not expert["active"] or not chosen.any():
            continue
        mean, scale = np.asarray(expert["mean"]), np.asarray(expert["scale"])
        bank = (np.asarray(expert["keys"])-mean)/scale
        ids, values = np.asarray(expert["turbines"]), np.asarray(expert["values"])
        idx, distance, diversity = local._neighbors((keys[chosen]-mean)/scale, bank, ids, model["neighbors"])
        for h in range(horizon):
            value = local._balanced_median(values[idx, h], ids[idx])*contract["rated_power_kw"]
            correction = (anchor[chosen, h]+value-reference[chosen, h]
                          if model["readout"] == "transition" else value)
            delta[chosen, h, k] = np.clip(correction, -model["max_scale"]*contract["rated_power_kw"],
                                        model["max_scale"]*contract["rated_power_kw"])
        physical[chosen, :, k] = True
        overlap[chosen, :, k] = ((distance <= expert["support_radius"])
                                & (diversity >= model["min_neighbor_turbines"]))[:, None]
    delta[..., count] = np.clip(delta[..., :count].sum(-1),
        -model["composition_max_scale"]*contract["rated_power_kw"],
        model["composition_max_scale"]*contract["rated_power_kw"])
    physical[..., count] = physical[..., :count].sum(-1) >= 2
    overlap[..., count] = (overlap[..., :count] | ~physical[..., :count]).all(-1)
    return np.clip(reference[..., None]+delta, 0, contract["rated_power_kw"]), physical, physical & overlap


def fit_transfer_policy(model, blocks, contract, *, min_windows=64, min_turbines=8,
                        min_positive_fraction=2/3):
    """Keep the v13 decision rule; assess this exact new execution object."""
    validate_identity(model, contract)
    if len(blocks) != 2 or min_windows < 1 or min_turbines < 1 or not 0 < min_positive_fraction <= 1:
        raise ValueError("Two utility blocks and valid fixed evidence requirements needed")
    prepared = []
    evidence_hash = hashlib.sha256()
    for block in blocks:
        ids = legacy._source_ids(block["turbines"], contract)
        candidate, _, support = candidates(model, block["context"], block["reference"],
                                           block["anchor"], block["events"], contract)
        truth, times = np.asarray(block["truth"], float), np.asarray(block["times"], dtype="datetime64[ns]")
        if (truth.shape != candidate.shape[:2] or times.shape != truth.shape or len(ids) != len(truth)
                or not np.isfinite(truth).all() or np.isnat(times).any()):
            raise ValueError("Utility labels and issue/target times do not align")
        for value in (truth, candidate, times.astype(np.int64), ids):
            evidence_hash.update(np.ascontiguousarray(value).tobytes())
        prepared.append(dict(block, supported=support, times=times,
            gain=np.abs(block["reference"]-truth)[..., None]-np.abs(candidate-truth[..., None]),
            mse_gain=(block["reference"]-truth)[..., None]**2-(candidate-truth[..., None])**2))
    if prepared[0]["times"].max() >= prepared[1]["times"].min():
        raise ValueError("Utility calibration target windows overlap in time")
    entries = {}
    for level, groups in (("event", 1), ("power", 5), ("joint", 20)):
        for group in range(groups):
            for k in range(len(model["experts"])+1):
                for h in range(contract["pred_len"]):
                    evidence = []
                    for block in prepared:
                        strata = (np.zeros(len(block["truth"]), int) if level == "event"
                                  else np.asarray(block["context"][level+"_bin"]))
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
    policy = dict(schema_version=1, expert_bundle_id=model["expert_bundle_id"], execution_contract=contract,
                  min_windows=min_windows, min_turbines=min_turbines,
                  min_positive_fraction=min_positive_fraction, entries=entries,
                  calibration_outputs_sha256=evidence_hash.hexdigest(),
                  score_definition="worst two-block turbine-balanced MAE mean, not a confidence bound",
                  granularity_rule="finest supported cell; known harmful fine cell vetoes coarse backoff")
    policy["policy_id"] = digest(policy)
    return policy


def apply_transfer(model, policy, context, reference, anchor, events, contract, *, event_only=False):
    validate_identity(model, contract, policy)
    candidate, physical, overlap = candidates(model, context, reference, anchor, events, contract)
    n, horizon, count = candidate.shape
    score = np.zeros_like(candidate); supported = np.zeros_like(physical)
    granularity = np.full(candidate.shape, -1, dtype=np.int8)
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
    selected = np.take_along_axis(candidate, best[..., None], -1).squeeze(-1)
    return dict(prediction=np.where(active, selected, np.asarray(reference, float)),
                action=np.where(active, best+1, 0), physical_available=physical,
                history_overlap=overlap, evidence_supported=supported, available=available,
                score_kw=score, granularity=granularity, expert_bundle_id=model["expert_bundle_id"],
                policy_id=policy["policy_id"])
