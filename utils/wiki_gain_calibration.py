"""Train-only, time-block-stable event/horizon amplitude calibration.

This is an empirical stability penalty, not an IID confidence bound. Overlapping
windows are never presented as independent statistical replications.
"""
import numpy as np


def calibration_blocks(dataset, blocks=3):
    if getattr(dataset, 'flag', None) != 'train':
        raise ValueError('Policy calibration accepts only original training data')
    starts = np.asarray(dataset.window_starts)
    first = np.asarray(dataset.dates)[starts + dataset.seq_len]
    last = np.asarray(dataset.dates)[starts + dataset.seq_len + dataset.pred_len - 1]
    unique = np.unique(first)
    if blocks < 2 or len(unique) < blocks:
        raise ValueError('Need at least two nonempty temporal calibration blocks')
    cuts = unique[(np.arange(1, blocks) * len(unique) // blocks).astype(int)]
    first_block = np.searchsorted(cuts, first, side='right')
    last_block = np.searchsorted(cuts, last, side='right')
    ids = np.where(first_block == last_block, first_block, -1)
    return ids, {'cuts': [str(x) for x in cuts], 'purged_cross_block_windows': int((ids < 0).sum())}


def fit_gain_policy(base, candidates, target, available, trend, block_ids, *,
                    source_split, num_modes=3, min_windows=32, penalty=.25,
                    min_gain=.001, clip_bounds=None):
    if source_split != 'train':
        raise ValueError('Validation/test labels cannot fit the gain policy')
    base, candidates, target = (np.asarray(x, dtype=np.float64) for x in (base, candidates, target))
    available = np.asarray(available, dtype=bool)
    trend, block_ids = np.asarray(trend), np.asarray(block_ids)
    if base.ndim != 2 or candidates.shape[:2] != base.shape or target.shape != base.shape:
        raise ValueError('Expected base/target [N,H] and candidates [N,H,K]')
    n, horizon, count = candidates.shape
    if available.shape != (n, count) or trend.shape != (n,) or block_ids.shape != (n,):
        raise ValueError('Policy metadata shape mismatch')
    if not all(np.isfinite(x).all() for x in (base, candidates, target)):
        raise ValueError('Nonfinite calibration predictions/targets')
    if min_windows < 1 or not np.isfinite(penalty) or penalty < 0 or not np.isfinite(min_gain) or min_gain < 0:
        raise ValueError('Invalid policy calibration configuration')
    blocks = np.unique(block_ids[block_ids >= 0])
    if len(blocks) < 2 or ((trend < 0) | (trend >= num_modes)).any():
        raise ValueError('Invalid time blocks or history trend labels')
    clip = (lambda x: np.clip(x, *clip_bounds)) if clip_bounds is not None else (lambda x: x)
    base_error = np.abs(clip(base) - target)
    grid = (0., .25, .5, 1.)
    tables = np.zeros((num_modes + 1, horizon, count, 2), dtype=np.float64)
    audit = []
    # Row num_modes is a pooled fallback for rare trend/event combinations.
    for group in [num_modes, *range(num_modes)]:
        for k in range(count):
            mask = available[:, k] & (block_ids >= 0)
            if group != num_modes:
                mask &= trend == group
            masks = [mask & (block_ids == block) for block in blocks]
            enough = all(int(m.sum()) >= min_windows for m in masks)
            for h in range(horizon):
                alpha, score, mean_gain, positive_blocks = 0., 0., 0., 0
                fallback = not enough and group != num_modes
                if fallback:
                    alpha, score = tables[num_modes, h, k]
                elif enough:
                    for value in grid[1:]:
                        prediction = base[:, h] + value * (candidates[:, h, k] - base[:, h])
                        improvement = base_error[:, h] - np.abs(clip(prediction) - target[:, h])
                        gains = np.array([improvement[m].mean() for m in masks])
                        risk_adjusted = float(gains.mean() - penalty * gains.std())
                        positive = int((gains > 0).sum())
                        if risk_adjusted > max(score, min_gain) and positive > len(blocks) // 2:
                            alpha, score, mean_gain, positive_blocks = value, risk_adjusted, float(gains.mean()), positive
                tables[group, h, k] = alpha, score
                audit.append(dict(trend=group if group < num_modes else 'pooled', candidate=k,
                                  horizon=h + 1, windows=int(mask.sum()), block_counts=[int(m.sum()) for m in masks],
                                  alpha=alpha, score=score, mean_gain=None if fallback else mean_gain,
                                  positive_blocks=None if fallback else positive_blocks, pooled_fallback=fallback))
    return tables[:num_modes, ..., 0], tables[:num_modes, ..., 1], audit


def route_policy(base, candidates, target, available, trend, alpha, gain,
                 *, min_gain=.001, clip_bounds=None, return_actions=False):
    """Replay the hard, physically masked inference route on train windows."""
    base = np.asarray(base, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    trend = np.asarray(trend, dtype=np.int64)
    alpha = np.asarray(alpha, dtype=np.float64)
    gain = np.asarray(gain, dtype=np.float64)
    if (base.ndim != 2 or target.shape != base.shape or
            candidates.shape[:2] != base.shape or
            available.shape != (base.shape[0], candidates.shape[-1]) or
            trend.shape != (base.shape[0],) or alpha.shape != gain.shape or
            alpha.shape[1:] != candidates.shape[1:] or
            ((trend < 0) | (trend >= alpha.shape[0])).any()):
        raise ValueError('Joint route shape or trend mismatch')
    scores = np.where(available[:, None, :], gain[trend], -np.inf)
    selected = scores.argmax(-1)
    best = np.take_along_axis(scores, selected[..., None], -1)[..., 0]
    amplitude = np.take_along_axis(alpha[trend], selected[..., None], -1)[..., 0]
    correction = np.take_along_axis(candidates - base[..., None], selected[..., None], -1)[..., 0]
    prediction = base + np.where(best > min_gain, amplitude * correction, 0.)
    if clip_bounds is not None:
        prediction = np.clip(prediction, *clip_bounds)
    errors = np.abs(prediction - target)
    if return_actions:
        return errors, np.where(best > min_gain, selected + 1, 0)
    return errors


def joint_block_gains(base, candidates, target, available, trend, block_ids,
                      alpha, gain, *, blocks, min_gain=.001, clip_bounds=None):
    """Realized MAE improvement per chronological block, including abstention."""
    block_ids = np.asarray(block_ids)
    base_error = np.abs((np.clip(base, *clip_bounds) if clip_bounds is not None else base) - target)
    routed_error = route_policy(base, candidates, target, available, trend,
                                alpha, gain, min_gain=min_gain, clip_bounds=clip_bounds)
    return np.array([(base_error[block_ids == block] - routed_error[block_ids == block]).mean()
                     for block in blocks], dtype=np.float64)


def refine_joint_policy(base, candidates, target, available, trend, block_ids,
                        alpha, gain, *, fit_blocks, penalty=.25, min_gain=.001,
                        clip_bounds=None):
    """Remove harmful competing actions and tune one global amplitude on fit blocks.

    The chronological holdout block is excluded by fit_blocks. Selection uses
    the deployed hard router, not independent candidate averages.
    """
    alpha, gain = np.array(alpha, copy=True), np.array(gain, copy=True)
    chosen = np.isin(block_ids, fit_blocks)
    if not chosen.any() or any(not np.any(block_ids == block) for block in fit_blocks):
        raise ValueError('Empty joint-policy fitting time block')
    blocks = np.asarray(fit_blocks)
    def score(a, g):
        values = joint_block_gains(base, candidates, target, available, trend,
                                   block_ids, a, g, blocks=blocks,
                                   min_gain=min_gain, clip_bounds=clip_bounds)
        return float(values.mean() - penalty * values.std()), values
    before, before_blocks = score(alpha, gain)
    removed = []
    # One event/trend group can have positive standalone gain but negative
    # realized gain once a higher-scoring event wins on overlapping windows.
    while True:
        current, _ = score(alpha, gain)
        best_delta, best_group = 1e-8, None
        for mode in range(alpha.shape[0]):
            for event in range(alpha.shape[-1]):
                if not np.any(alpha[mode, :, event]):
                    continue
                trial_alpha, trial_gain = alpha.copy(), gain.copy()
                trial_alpha[mode, :, event] = 0.
                trial_gain[mode, :, event] = 0.
                improvement = score(trial_alpha, trial_gain)[0] - current
                if improvement > best_delta:
                    best_delta, best_group = improvement, (mode, event)
        if best_group is None:
            break
        mode, event = best_group
        alpha[mode, :, event] = 0.
        gain[mode, :, event] = 0.
        removed.append({'trend': mode, 'candidate': event, 'joint_score_gain': best_delta})
    # Correct for overall overshoot while retaining event-specific calibration.
    best_alpha, best_multiplier, best_score = alpha.copy(), 1., score(alpha, gain)[0]
    for multiplier in (.25, .5):
        trial = alpha * multiplier
        result = score(trial, gain)[0]
        if result > best_score + 1e-8:
            best_alpha, best_multiplier, best_score = trial, multiplier, result
    after, after_blocks = score(best_alpha, gain)
    return best_alpha, gain, {'fit_blocks': blocks.tolist(), 'score_before': before,
                              'score_after': after, 'block_gains_before': before_blocks.tolist(),
                              'block_gains_after': after_blocks.tolist(),
                              'removed_groups': removed, 'global_amplitude_multiplier': best_multiplier}


def prune_events_on_train_holdout(base, candidates, target, available, trend,
                                  block_ids, alpha, gain, *, holdout_block,
                                  min_gain=.001, clip_bounds=None):
    """Select a small set of event families using only the later TRAIN block.

    This block is a model-selection set, not an untouched estimate of policy
    performance. Evaluate the resulting frozen policy on the separate val set.
    """
    alpha, gain = np.array(alpha, copy=True), np.array(gain, copy=True)
    mask = np.asarray(block_ids) == holdout_block
    if not mask.any():
        raise ValueError('Empty training selection block')
    base, target = np.asarray(base), np.asarray(target)
    clipped_base = np.clip(base, *clip_bounds) if clip_bounds is not None else base
    def realized(a, g):
        errors, actions = route_policy(base, candidates, target, available, trend,
                                       a, g, min_gain=min_gain,
                                       clip_bounds=clip_bounds, return_actions=True)
        return float((np.abs(clipped_base - target)[mask] - errors[mask]).mean()), errors, actions
    before, _, _ = realized(alpha, gain)
    removed = []
    # Prune by the change in FINAL routed MAE, including any substitute event
    # that wins once an event is removed. No validation label is read here.
    while True:
        current, errors, actions = realized(alpha, gain)
        best_improvement, best_event, best_trial = 1e-8, None, None
        for event in range(alpha.shape[-1]):
            if not np.any(alpha[..., event]):
                continue
            trial_alpha, trial_gain = alpha.copy(), gain.copy()
            trial_alpha[..., event] = 0.
            trial_gain[..., event] = 0.
            trial_gain_value, _, _ = realized(trial_alpha, trial_gain)
            improvement = trial_gain_value - current
            if improvement > best_improvement:
                best_improvement = improvement
                best_event = event
                best_trial = trial_alpha, trial_gain
        if best_event is None:
            break
        selected = (actions[mask] == best_event + 1)
        base_error = np.abs(clipped_base - target)[mask]
        selected_gain = (float((base_error[selected] - errors[mask][selected]).mean())
                         if selected.any() else None)
        alpha, gain = best_trial
        removed.append({'candidate': best_event,
                        'selected_positions': int(selected.sum()),
                        'selected_gain_before_pruning': selected_gain,
                        'joint_gain_improvement': best_improvement})
    after, final_errors, final_actions = realized(alpha, gain)
    base_error = np.abs(clipped_base - target)[mask]
    event_outcomes = []
    for event in range(alpha.shape[-1]):
        selected = final_actions[mask] == event + 1
        event_outcomes.append({'candidate': event, 'selected_positions': int(selected.sum()),
                               'selected_gain': (float((base_error[selected] - final_errors[mask][selected]).mean())
                                                 if selected.any() else None)})
    return alpha, gain, {'selection_block': int(holdout_block),
                         'joint_gain_before': before, 'joint_gain_after': after,
                         'removed_events': removed, 'retained_event_outcomes': event_outcomes,
                         'uses_validation_labels': False,
                         'train_block_is_independent_evaluation': False}
