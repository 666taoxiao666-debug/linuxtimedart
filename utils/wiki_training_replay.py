"""Training-only, retrospective diagnostics for actual frozen Wiki candidates."""
from __future__ import annotations

import hashlib
import numpy as np
from utils.wiki_evidence_support_audit import clock_support


def verify_replay(pack, numeric, candidate, certificate, digest_arrays):
    if digest_arrays(pack) != certificate["calibration_sha256"]:
        raise ValueError("Training calibration input hash mismatch")
    actual = hashlib.sha256(np.ascontiguousarray(np.stack((numeric, candidate), -1)).tobytes()).hexdigest()
    if actual != certificate["actual_outputs_sha256"]:
        raise ValueError("Frozen calibration outputs are not bit-identical; stop, do not replace the certificate")
    return actual


def replay_diagnostic(pack, numeric, candidate, certificate, *, train_end):
    clock_support(pack["times"], pack["turbines"], train_end=train_end)
    numeric, candidate, truth = (np.asarray(value, float) for value in (numeric, candidate, pack["truth"]))
    if numeric.shape != truth.shape or candidate.shape != truth.shape:
        raise ValueError("Training replay predictions/labels misaligned")
    supported = pack["overlap"].any(-1)
    gain = np.abs(numeric - truth) - np.abs(candidate - truth)
    squared = (numeric - truth)**2 - (candidate - truth)**2
    days = (pack["times"][:, 0] - np.timedelta64(1, "h")).astype("datetime64[D]")
    rows = {}
    for key, row in certificate["entries"].items():
        if not row["supported"]:
            continue
        level, group, h, k = key.split(":")
        group, h, k = int(group), int(h), int(k)
        strata = np.zeros(len(truth), int) if level == "event" else pack["context"][level + "_bin"]
        mask = (strata == group) & pack["overlap"][:, h, k]
        blocks = []
        for day in np.unique(days[mask]):
            selected = mask & (days == day)
            units = np.unique(pack["turbines"][selected])
            unit_gain = [gain[selected & (pack["turbines"] == unit), h].mean() for unit in units]
            unit_mse = [squared[selected & (pack["turbines"] == unit), h].mean() for unit in units]
            blocks.append(dict(utc_issue_day=str(day), windows=int(selected.sum()), turbines=len(units),
                               turbine_macro_gain_kw=float(np.mean(unit_gain)),
                               turbine_macro_mse_gain=float(np.mean(unit_mse))))
        values = np.asarray([block["turbine_macro_gain_kw"] for block in blocks])
        rows[key] = dict(accepted=row["accepted"], blocks=blocks,
                         day_macro_gain_kw=float(values.mean()),
                         day_macro_gain_sd_kw=float(values.std(ddof=1)) if len(values) > 1 else None,
                         positive_day_fraction=float((values > 0).mean()))
    return dict(scope="original outer-train fresh calibration, knowledge donors only",
                interpretation="In-sample frozen-candidate diagnostic, not a deployable oracle or independent transfer test",
                supported_point_pct=float(100 * supported.mean()),
                supported_candidate_gain_all_points_kw=float(np.where(supported, gain, 0.).mean()),
                oracle_gain_all_points_kw=float(np.where(supported, np.maximum(gain, 0.), 0.).mean()),
                cells=rows)
