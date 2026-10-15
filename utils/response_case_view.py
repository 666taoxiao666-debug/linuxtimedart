"""Causal, cross-turbine local case pools for a response-weighted executor."""
from __future__ import annotations

import numpy as np
from utils import kelmarsh_short_dynamics_wiki as short


def build(pack, info, bank, contract, *, ready=True):
    short.validate(bank, contract)
    reference, anchor = np.asarray(pack["reference"]), np.asarray(pack["anchor"])
    numeric = np.asarray(pack["numeric"])
    clock = np.asarray(pack["times"]).astype("datetime64[ns]")
    units = np.asarray(pack["turbines"])
    n, horizon = reference.shape
    events, count = len(short.legacy.EVENTS), bank["settings"]["neighbors"]
    capacity = contract["rated_power_kw"]
    if (numeric.shape != reference.shape or anchor.shape != reference.shape or clock.shape != reference.shape
            or units.shape != (n,) or np.isnat(clock).any()):
        raise ValueError("Case query identities do not align")
    origins = clock[:, 0].astype(np.int64) - 3600 * 10**9
    if np.any(info["last_observed_ns"] > origins):
        raise ValueError("Future query metadata")
    x = short.keys(pack, info, capacity, bank["settings"]["use_controls"])
    dimension = x.shape[-1]
    view = dict(query=np.zeros((n, events, dimension), np.float32),
        neighbors=np.zeros((n, events, count, dimension), np.float32),
        values=np.zeros((n, horizon, events, count), np.float32),
        prior=np.zeros((n, events, count), np.float32),
        base_change=((numeric - anchor) / capacity).astype(np.float32),
        supported=np.zeros((n, horizon, events), bool),
        case_indices=np.full((n, events, count), -1, np.int64),
        case_units=np.zeros((n, events, count), np.int16))
    if not ready:
        return view
    for group in bank["groups"]:
        if not group["active"]:
            continue
        event = group["event"]
        mask = pack["events"][:, event] & (info["state"] == group["state"])
        if not mask.any():
            continue
        if np.any(origins[mask] < group["available_ns"]):
            raise ValueError("Case outcomes not yet available")
        mean, scale = np.asarray(group["mean"]), np.asarray(group["scale"])
        case_keys = (np.asarray(group["keys"]) - mean) / scale
        query_keys = (x[mask] - mean) / scale
        case_units = np.asarray(group["units"])
        # Same exclusion at fit and deployment. The target unit can never
        # retrieve its own source-unit outcomes, even when it is a donor.
        idx, distance, diversity = short.legacy.neighbors._neighbors(query_keys, case_keys, case_units, count,
                                                                    excluded_turbines=units[mask])
        selected_units = case_units[idx]
        if np.any(selected_units == units[mask, None]) or not np.isfinite(distance).all():
            raise ValueError("Insufficient cross-unit cases; never silently reuse own-unit outcomes")
        blocks = np.asarray(group["weather_blocks"])[idx]
        prior = short.episode.balanced_weights(selected_units, blocks)
        view["query"][mask, event] = query_keys
        view["neighbors"][mask, event] = case_keys[idx]
        view["values"][mask, :, event] = np.asarray(group["values"])[idx].transpose(0, 2, 1)
        view["prior"][mask, event] = prior
        view["supported"][mask, :, event] = ((distance <= group["radius"]) &
            (diversity >= bank["settings"]["min_neighbor_turbines"]))[:, None]
        view["case_indices"][mask, event] = idx
        view["case_units"][mask, event] = selected_units
    if not all(np.isfinite(value).all() for value in view.values()):
        raise ValueError("Nonfinite causal case view")
    return view


def certificate_pack(pack, view):
    result = dict(pack)
    result["overlap"] = view["supported"]
    return result
