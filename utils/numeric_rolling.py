"""Expanding-origin numerical selection using only original training outcomes."""
from __future__ import annotations

import hashlib
import numpy as np
from data_provider.kelmarsh_hourly import KelmarshWindows
from utils.numeric_foundation import joint_score

ORIGINS = ("base_selection_end", "memory_fit_end", "calibration_early_end")
ENDS = ("memory_fit_end", "calibration_early_end", "train_end")


def rolling_partitions(series):
    starts = series.starts
    first = series.dates[starts + series.input_len]
    last = series.dates[starts + series.input_len + series.horizon - 1]
    result = {}
    for index, (begin, end) in enumerate(zip(ORIGINS, ENDS)):
        cut, stop = series.bounds[begin], series.bounds[end]
        fit = KelmarshWindows(series, starts[last < cut], "rolling_fit")
        query = KelmarshWindows(series, starts[(first >= cut) & (last < stop)], "rolling_selection")
        if not len(fit) or not len(query):
            raise ValueError("Empty expanding-origin interval")
        issue = query.target_times()[:, 0].min() - np.timedelta64(1, "h")
        if fit.target_times().max() > issue or series.bounds["base_fit_end"] > cut:
            raise ValueError("Training/scaler unavailable at internal issue time")
        fit.flag, query.flag = "train", "val"
        fit.train_cutoff = query.train_cutoff = cut
        fit.val_cutoff = query.val_cutoff = stop
        result[f"o{70 + index * 10}"] = dict(fit=fit, selection=query, cut=str(cut), end=str(stop))
    all_train = KelmarshWindows(series, starts[last < series.bounds["train_end"]], "rolling_refit")
    all_train.flag = "train"
    all_train.train_cutoff = series.bounds["train_end"]
    return result, all_train


def initializer_audit(series, first_query, contract):
    """Pretrain selection is part of initialization, so it must also be causal."""
    fit, selection = series.subset("fit"), series.subset("selection")
    for key, ds in (("fit_window_hash", fit), ("selection_window_hash", selection)):
        if contract.get(key) != hashlib.sha256(ds.window_starts.tobytes()).hexdigest():
            raise ValueError("Pretrain fitting/selection window identity changed")
    last_label = max(fit.target_times().max(), selection.target_times().max())
    issue = first_query.target_times()[:, 0].min() - np.timedelta64(1, "h")
    if last_label > issue:
        raise ValueError("Pretrain selected using outcomes unavailable at first rolling issue")
    return dict(latest_pretrain_outcome=str(last_label), first_internal_issue=str(issue),
                pretrain_fit_and_selection_both_causal=True)


def select_rolling(grid, baselines):
    """One architecture/config AND common epoch must pass every origin."""
    valid = []
    for row in grid:
        if set(row["per_origin"]) != set(baselines):
            raise ValueError("Candidate missing an internal origin")
        scores = {key: joint_score(row["per_origin"][key], baseline) for key, baseline in baselines.items()}
        score = float(np.mean(list(scores.values())))
        if row["best_epoch"] != 0 and np.isfinite(score) and score < 1 - 1e-8:
            valid.append(dict(row, inner_score=score, per_origin_scores=scores))
    if not valid:
        return dict(model="Persistence", best_epoch=0, inner_score=1.,
                    reason="No common configuration/epoch jointly passes all three training origins")
    return min(valid, key=lambda row: (row["inner_score"], row["parameters"], row["model"], row["best_epoch"] or 0))


def candidate_grid(records, epochs):
    """Build from all actual epoch metrics, NOT each origin's own best epoch."""
    origins = list(records)
    if len(origins) != 3:
        raise ValueError("Exactly three frozen training origins required")
    names = set(records[origins[0]])
    if any(set(records[key]) != names for key in origins):
        raise ValueError("Candidate pool differs across origins")
    rows = []
    for name in sorted(names):
        reference = records[origins[0]][name]
        options = [None] if name.startswith("DeltaRidge") else list(range(1, epochs + 1))
        for epoch in options:
            row = dict(model=name, best_epoch=epoch, parameters=reference["parameters"], per_origin={})
            if "alpha" in reference:
                row["alpha"] = reference["alpha"]
            for origin in origins:
                record = records[origin][name]
                if record["parameters"] != row["parameters"] or record.get("alpha") != reference.get("alpha"):
                    raise ValueError("Candidate configuration differs across origins")
                row["per_origin"][origin] = record["metrics"] if epoch is None else record["epochs"][str(epoch)]
            rows.append(row)
    return rows
