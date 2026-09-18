import torch


def utility_supervision_loss(aux, target, eps=0.05):
    """Train horizon utilities from candidate outcomes on training batches only.

    Utility is the clipped relative absolute-error reduction against the
    trend-only branch.  Physically unavailable knowledge granularities are
    masked, so labels cannot teach the model to bypass the evidence filter.
    """

    required = {
        "utilities",
        "availability",
        "base_prediction",
        "event_prediction",
        "composition_prediction",
    }
    missing = sorted(required.difference(aux))
    if missing:
        raise KeyError("utility auxiliary output is missing: " + ", ".join(missing))
    base_error = (aux["base_prediction"].detach() - target).abs()
    candidate_errors = torch.stack(
        [
            (aux["event_prediction"].detach() - target).abs(),
            (aux["composition_prediction"].detach() - target).abs(),
        ],
        dim=-1,
    )
    relative_gain = (
        base_error.unsqueeze(-1) - candidate_errors
    ) / base_error.unsqueeze(-1).clamp_min(float(eps))
    targets = relative_gain.mean(dim=2).clamp(-1.0, 1.0)
    mask = aux["availability"].unsqueeze(1).expand_as(targets)
    if not bool(mask.any()):
        return aux["utilities"].sum() * 0.0, targets
    loss = torch.nn.functional.smooth_l1_loss(
        aux["utilities"][mask], targets[mask], reduction="mean"
    )
    return loss, targets


def utility_candidate_specialization_loss(aux, target, margin=0.01):
    """Teach physically available Wiki branches to become useful corrections.

    The utility head can only learn *which* branch is useful after candidate
    branches learn distinct corrections.  This train-only objective selects
    every available candidate at each sample/horizon and ranks it against a
    detached trend baseline.  This prevents the composition branch from being
    starved merely because the single-event branch is initially easier.
    Unavailable branches never receive supervision.
    """

    required = {
        "availability",
        "base_prediction",
        "event_prediction",
        "composition_prediction",
    }
    missing = sorted(required.difference(aux))
    if missing:
        raise KeyError("utility auxiliary output is missing: " + ", ".join(missing))
    if float(margin) < 0.0:
        raise ValueError("utility ranking margin cannot be negative")

    candidates = torch.stack(
        [aux["event_prediction"], aux["composition_prediction"]], dim=-1
    )
    # [batch, horizon, candidate]; channel-mean error supports c_out > 1.
    candidate_error = (candidates - target.unsqueeze(-1)).abs().mean(dim=2)
    availability = aux["availability"].unsqueeze(1).expand_as(candidate_error)
    masked_error = candidate_error.masked_fill(~availability, float("inf"))
    _, oracle_index = masked_error.min(dim=-1)
    available = availability.any(dim=-1)

    zero = candidates.sum() * 0.0
    if not bool(available.any()):
        return zero, zero, {
            "available_fraction": 0.0,
            "oracle_event_fraction": 0.0,
            "oracle_composition_fraction": 0.0,
        }

    channel_mask = availability.unsqueeze(2).expand_as(candidates)
    expanded_target = target.unsqueeze(-1).expand_as(candidates)
    specialization = torch.nn.functional.smooth_l1_loss(
        candidates[channel_mask], expanded_target[channel_mask], reduction="mean"
    )

    base_error = (aux["base_prediction"].detach() - target).abs().mean(dim=2)
    ranking_terms = torch.relu(
        candidate_error - base_error.unsqueeze(-1) + float(margin)
    )
    ranking = ranking_terms[availability].mean()
    available_count = available.sum().clamp_min(1)
    stats = {
        "available_fraction": float(available.float().mean().detach().cpu()),
        "oracle_event_fraction": float(
            ((oracle_index == 0) & available).sum().detach().cpu() / available_count
        ),
        "oracle_composition_fraction": float(
            ((oracle_index == 1) & available).sum().detach().cpu() / available_count
        ),
    }
    return specialization, ranking, stats
