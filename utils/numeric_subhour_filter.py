"""One convex six-tick state estimator; no changes to the frozen v15 code."""
from __future__ import annotations

import numpy as np
import pandas as pd
from utils.numeric_foundation import observed_anchor


def centered_ticks(dataset, tables):
    rows = dataset.window_starts + dataset.seq_len - 1
    issues, units = dataset.dates[rows], dataset.turbines[rows]
    anchor = observed_anchor(dataset)[:, 0]
    values = np.empty((len(rows), 6))
    for unit in np.unique(units):
        mask = units == unit
        times = issues[mask, None] - np.arange(60, 0, -10)[None] * np.timedelta64(1, "m")
        block = tables[int(unit)].reindex(pd.DatetimeIndex(times.ravel())).to_numpy().reshape(-1, 6)
        if not np.isfinite(block).all():
            raise ValueError("Missing source tick; frozen cohort must not be filtered")
        if not np.allclose(block.mean(1), anchor[mask], atol=1e-8, rtol=0):
            raise ValueError("Six source ticks do not reconstruct the frozen hourly anchor")
        values[mask] = block
    return values - anchor[:, None]


def endpoint_weights(beta):
    beta = np.asarray(beta, float)
    if beta.ndim != 1 or not np.isfinite(beta).all() or np.any((beta < 0) | (beta > 1)):
        raise ValueError("Finite v15 coefficients in[0,1] required")
    weights = np.repeat(((1 - beta) / 6)[:, None], 6, axis=1)
    weights[:, -1] += beta
    return weights


def simplex_least_squares(x, y, prior):
    """Exact small convex solve by enumerating 63 nonempty simplex faces.

    Every face uses ONLY the same fit labels; these are optimizer active sets,
    not hyperparameter candidates. The prior is v15's feasible filter and wins
    numerical ties. Nonnegative weights with sum1 prevent extrapolative states.
    """
    x, y, prior = np.asarray(x, float), np.asarray(y, float), np.asarray(prior, float)
    if (x.ndim != 2 or x.shape[1] != 6 or y.shape != (len(x),) or not len(x)
            or prior.shape != (6,) or not all(np.isfinite(a).all() for a in (x, y, prior))
            or np.any(prior < 0) or not np.isclose(prior.sum(), 1, atol=1e-12, rtol=0)):
        raise ValueError("Finite paired fit arrays and a simplex prior required")
    scale = max(float(np.sqrt(np.mean(x**2))), float(np.sqrt(np.mean(y**2))), 1.)
    design, target = x / scale, y / scale
    gram, rhs = design.T @ design / len(x), design.T @ target / len(x)
    def loss(w):
        return float(np.mean((design @ w - target)**2))
    best, best_loss = prior.copy(), loss(prior)
    for mask in range(1, 64):
        support = np.flatnonzero([(mask >> j) & 1 for j in range(6)])
        n = len(support)
        system = np.zeros((n + 1, n + 1))
        system[:n, :n] = gram[np.ix_(support, support)]
        system[:n, -1], system[-1, :n] = 1., 1.
        vector = np.r_[rhs[support], 1.]
        solution = np.linalg.lstsq(system, vector, rcond=None)[0]
        if np.max(abs(system @ solution - vector)) > 1e-8 or np.any(solution[:n] < -1e-9):
            continue
        weights = np.zeros(6)
        weights[support] = np.maximum(solution[:n], 0.)
        weights /= weights.sum()
        objective = loss(weights)
        tolerance = 1e-12 * max(1., abs(best_loss))
        if (objective < best_loss - tolerance or
                (abs(objective - best_loss) <= tolerance and np.sum((weights - prior)**2) < np.sum((best - prior)**2))):
            best, best_loss = weights, objective
    if best_loss > loss(prior) + 1e-10:
        raise ValueError("Convex filter fit is worse than its feasible v15 prior")
    return best


def fit_weights(dataset, tables, beta):
    x = centered_ticks(dataset, tables)
    truth, anchor = dataset.truth_and_anchor()
    prior = endpoint_weights(beta)
    if prior.shape[0] != dataset.pred_len:
        raise ValueError("Baseline lead count mismatch")
    return np.array([simplex_least_squares(x, truth[:, h] - anchor[:, h], prior[h])
                     for h in range(dataset.pred_len)])


def predict(weights, dataset, tables):
    weights = np.asarray(weights, float)
    if (weights.shape != (dataset.pred_len, 6) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.allclose(weights.sum(1), 1., atol=1e-10, rtol=0)):
        raise ValueError("Finite nonnegative sum1 filter for each lead required")
    return observed_anchor(dataset) + centered_ticks(dataset, tables) @ weights.T


def select_filter(records, references):
    """Require a joint gain over BOTH v15 and Persistence in every train block."""
    if set(records) != {"o70", "o80", "o90"} or set(references) != set(records):
        raise ValueError("Exactly the three declared chronological training blocks required")
    scores = []
    for key, metrics in records.items():
        for reference in references[key].values():
            if any(not np.isfinite(metrics[k]) or metrics[k] > reference[k] + 1e-8 for k in ("mae", "rmse")):
                return dict(model="SubhourEndState", reused_v15=True,
                            reason="New filter fails joint training-block comparison; no new outer evaluation")
        scores.append(.5 * sum(metrics[k] / references[key]["v15"][k] for k in ("mae", "rmse")))
    if np.mean(scores) >= 1 - 1e-8:
        return dict(model="SubhourEndState", reused_v15=True, reason="No material aggregate training gain")
    return dict(model="SimplexSubhourFilter", parameters=10, serialized_weights=12,
                reused_v15=False, inner_score=float(np.mean(scores)), per_origin_scores=scores)
