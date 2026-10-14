"""Read-only support audit; counts are not a claim of statistical independence."""
from __future__ import annotations

import numpy as np


def clock_support(times, turbines, *, train_end):
    """Count issue clocks/UTC day blocks without looking at outcome magnitudes."""
    times = np.asarray(times).astype("datetime64[ns]")
    turbines = np.asarray(turbines)
    cutoff = np.datetime64(train_end, "ns")
    if (times.ndim != 2 or not times.shape[1] or len(times) != len(turbines)
            or np.isnat(times).any() or np.isnat(cutoff)):
        raise ValueError("Invalid training target clock")
    if np.any(times >= cutoff):
        raise ValueError("Audit cannot include outer validation/test target clocks")
    if times.shape[1] > 1 and np.any(np.diff(times, axis=1) != np.timedelta64(1, "h")):
        raise ValueError("Expected consecutive hourly targets")
    if not np.isin(turbines, [1, 2, 3, 4, 5]).all():
        raise ValueError("Only declared knowledge donor turbines 1-5 are permitted")
    issue = times[:, 0] - np.timedelta64(1, "h")
    days = issue.astype("datetime64[D]")
    dates, counts = np.unique(days, return_counts=True)
    return dict(windows=len(times), turbines=int(len(np.unique(turbines))),
                unique_issue_times=int(len(np.unique(issue))), utc_day_blocks=len(dates),
                first_issue=str(issue.min()) if len(issue) else None,
                last_issue=str(issue.max()) if len(issue) else None,
                largest_day_window_share=float(counts.max() / len(issue)) if len(issue) else None,
                day_blocks=[dict(utc_day=str(day), windows=int(count),
                                 unique_issue_times=int(len(np.unique(issue[days == day]))),
                                 turbines=int(len(np.unique(turbines[days == day]))))
                            for day, count in zip(dates, counts)])


def certificate_support_audit(pack, certificate, *, train_end):
    """Reproduce every saved cell's membership before describing its support."""
    times, turbines = np.asarray(pack["times"]), np.asarray(pack["turbines"])
    overlap = np.asarray(pack["overlap"], dtype=bool)
    if overlap.ndim != 3 or overlap.shape[:2] != times.shape:
        raise ValueError("Support and target shapes disagree")
    total = clock_support(times, turbines, train_end=train_end)
    rows = {}
    for key, entry in certificate["entries"].items():
        level, group, horizon, event = key.split(":")
        group, horizon, event = int(group), int(horizon), int(event)
        if level not in ("event", "power", "joint"):
            raise ValueError("Unknown certificate granularity")
        strata = np.zeros(len(times), int) if level == "event" else np.asarray(pack["context"][level + "_bin"])
        mask = (strata == group) & overlap[:, horizon, event]
        counts = clock_support(times[mask], turbines[mask], train_end=train_end)
        if counts["windows"] != entry["windows"] or counts["turbines"] != entry["turbines"]:
            raise ValueError(f"Saved certificate membership did not reproduce: {key}")
        if entry["supported"]:
            rows[key] = dict(counts, accepted=entry["accepted"],
                             saved_turbine_macro_gain_kw=entry["gain_kw"],
                             saved_turbine_macro_mse_gain=entry["mse_gain"])
    return dict(partition=total, supported_cells=rows,
                accepted_cells=[key for key, row in rows.items() if row["accepted"]],
                all_saved_cell_counts_reproduced=True,
                gain_by_time_block=None,
                gain_by_time_block_status="Not recomputed: frozen calibration predictions were not persisted; do not infer variance from aggregate gains.",
                interpretation="Issue counts/day blocks describe concentration, not an effective sample size or proof of independent weather episodes.")
