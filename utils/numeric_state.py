"""Shrunk, observed-state conditional end-point forecasts; no Wiki or future data."""
from __future__ import annotations

import numpy as np
from utils.numeric_foundation import observed_anchor
from utils.numeric_subhour import end_state_delta

NAME = "StateShrunkEndpoint"
MIN_ISSUES = 24
MIN_DAYS = 7


def design(dataset, tables):
    if dataset.pred_len != 2:
        raise ValueError("Two direct hourly forecast leads required")
    delta, _ = end_state_delta(dataset, tables)
    anchor = observed_anchor(dataset)
    # Both variables were already inputs of the v15 executor. This changes
    # the coefficient function, not the dataset or the prediction cohort.
    state = 2 * np.digitize(anchor[:, 0] / 2050., [.1, .9]) + (delta >= 0)
    issue = dataset.dates[dataset.window_starts + dataset.seq_len - 1]
    return delta, anchor, state, issue


def fit(dataset, tables, beta):
    delta, anchor, state, issue = design(dataset, tables)
    beta = np.asarray(beta, float)
    if beta.shape != (2,) or not np.isfinite(beta).all() or ((beta < 0) | (beta > 1)).any():
        raise ValueError("Frozen v15 coefficients must be finite and bounded")
    truth, _ = dataset.truth_and_anchor()
    innovation = truth - anchor
    coefficients = np.tile(beta, (6, 1))
    support = np.zeros((6, 3), dtype=np.int64)
    penalty = float(delta @ delta) / 6.
    for cell in range(6):
        mask = state == cell
        support[cell] = [int(mask.sum()), len(np.unique(issue[mask])),
                         len(np.unique(issue[mask].astype("datetime64[D]")))]
        if support[cell, 1] < MIN_ISSUES or support[cell, 2] < MIN_DAYS:
            continue
        x = delta[mask]
        denominator = float(x @ x) + penalty
        if denominator > 0:
            coefficients[cell] = np.clip((x @ innovation[mask] + penalty * beta) / denominator, 0., 1.)
    # The global prior is feasible. The penalized optimum therefore cannot
    # increase the fitting MSE, independently for each lead.
    prior_error = innovation - delta[:, None] * beta
    new_error = innovation - delta[:, None] * coefficients[state]
    if (np.mean(new_error**2, axis=0) > np.mean(prior_error**2, axis=0) + 1e-7).any():
        raise ValueError("Fitting MSE exceeds feasible frozen prior")
    return dict(coefficients=coefficients, support=support, prior=beta, penalty=np.array(penalty))


def predict(model, dataset, tables, enabled=(True, True)):
    delta, anchor, state, _ = design(dataset, tables)
    coefficients, beta = np.asarray(model["coefficients"]), np.asarray(model["prior"])
    enabled = np.asarray(enabled)
    if (coefficients.shape != (6, 2) or beta.shape != (2,) or enabled.shape != (2,)
            or enabled.dtype.kind != "b" or not np.isfinite(coefficients).all()
            or not np.isfinite(beta).all() or ((coefficients < 0) | (coefficients > 1)).any()
            or ((beta < 0) | (beta > 1)).any()):
        raise ValueError("Bounded six-state model and two Boolean lead flags required")
    weight = np.where(enabled[None, :], coefficients[state], beta[None, :])
    return anchor + delta[:, None] * weight


def choose(records, references):
    if set(records) != {"o70", "o80", "o90"} or set(records) != set(references):
        raise ValueError("Exactly three declared TRAIN query blocks required")
    enabled, failures = [], []
    for lead in ("horizon1", "horizon2"):
        good, scores = True, []
        for origin, row in records.items():
            if set(references[origin]) != {"v15", "persistence"}:
                raise ValueError("Both frozen v15 and Persistence required")
            for name, reference in references[origin].items():
                for metric in ("mae", "rmse"):
                    new, old = row[lead][metric], reference[lead][metric]
                    if not np.isfinite(new) or not np.isfinite(old) or old <= 0 or new > old + 1e-8:
                        good = False
                        failures.append(dict(origin=origin, lead=lead, reference=name, metric=metric))
            denominator = [references[origin]["v15"][lead][m] for m in ("mae", "rmse")]
            if all(np.isfinite(v) and v > 0 for v in denominator):
                scores.append(.5 * sum(row[lead][m] / old for m, old in zip(("mae", "rmse"), denominator)))
        good = good and len(scores) == 3 and np.isfinite(scores).all() and np.mean(scores) < 1 - 1e-8
        enabled.append(bool(good))
    return dict(model=NAME if any(enabled) else "SubhourEndState", enabled_leads=enabled,
                reused_v15=not any(enabled), failed_checks=failures,
                reason="Each lead admitted independently only by joint MAE/RMSE across three TRAIN blocks; disabled lead is exact v15")
