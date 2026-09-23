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
