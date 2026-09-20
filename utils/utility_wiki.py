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


def utility_decision_loss(
    aux,
    target,
    *,
    eps=0.05,
    min_gain=0.0,
    temperature=0.25,
):
    """Directly supervise abstain/event/composition decisions on train batches.

    The utility regressor learns the magnitude of candidate gains, while this
    balanced classification term teaches the exact decision made at inference:
    abstain unless the best physically available candidate beats ``min_gain``.
    Windows without any physical evidence are excluded because the inference
    gate already hard-masks them to abstention.
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
    if float(eps) <= 0.0:
        raise ValueError("utility decision eps must be positive")
    if float(temperature) <= 0.0:
        raise ValueError("utility decision temperature must be positive")

    base_error = (aux["base_prediction"].detach() - target).abs()
    candidate_errors = torch.stack(
        [
            (aux["event_prediction"].detach() - target).abs(),
            (aux["composition_prediction"].detach() - target).abs(),
        ],
        dim=-1,
    )
    realized_gain = (
        (base_error.unsqueeze(-1) - candidate_errors)
        / base_error.unsqueeze(-1).clamp_min(float(eps))
    ).mean(dim=2).clamp(-1.0, 1.0)
    availability = aux["availability"].unsqueeze(1).expand_as(realized_gain)
    has_evidence = availability.any(dim=-1)
    zero = aux["utilities"].sum() * 0.0
    if not bool(has_evidence.any()):
        return zero, {
            "oracle_abstain_fraction": 1.0,
            "oracle_event_fraction": 0.0,
            "oracle_composition_fraction": 0.0,
        }

    masked_gain = realized_gain.masked_fill(~availability, -1e4)
    best_gain, best_index = masked_gain.max(dim=-1)
    target_action = torch.where(
        best_gain > float(min_gain),
        best_index + 1,
        torch.zeros_like(best_index),
    )

    abstain_logit = torch.full_like(aux["utilities"][..., :1], float(min_gain))
    candidate_logits = aux["utilities"].masked_fill(~availability, -1e4)
    decision_logits = torch.cat([abstain_logit, candidate_logits], dim=-1)
    decision_logits = decision_logits / float(temperature)
    per_item = torch.nn.functional.cross_entropy(
        decision_logits[has_evidence],
        target_action[has_evidence],
        reduction="none",
    )
    evidence_actions = target_action[has_evidence]
    class_terms = [
        per_item[evidence_actions == class_index].mean()
        for class_index in range(3)
        if bool((evidence_actions == class_index).any())
    ]
    loss = torch.stack(class_terms).mean()
    denominator = has_evidence.sum().clamp_min(1)
    stats = {
        "oracle_abstain_fraction": float(
            ((target_action == 0) & has_evidence).sum().detach().cpu() / denominator
        ),
        "oracle_event_fraction": float(
            ((target_action == 1) & has_evidence).sum().detach().cpu() / denominator
        ),
        "oracle_composition_fraction": float(
            ((target_action == 2) & has_evidence).sum().detach().cpu() / denominator
        ),
    }
    return loss, stats


def selected_intervention_metrics(
    granularity,
    strength,
    base_prediction,
    candidate_prediction,
    target,
    *,
    selected_index,
):
    """Measure realized gain only where a particular Wiki branch was selected."""

    granularity = torch.as_tensor(granularity)
    strength = torch.as_tensor(strength, dtype=torch.float64)
    base_prediction = torch.as_tensor(base_prediction, dtype=torch.float64)
    candidate_prediction = torch.as_tensor(
        candidate_prediction, dtype=torch.float64
    )
    target = torch.as_tensor(target, dtype=torch.float64)
    if granularity.shape != strength.shape:
        raise ValueError("granularity and strength must have the same shape")
    if not (
        base_prediction.shape == candidate_prediction.shape == target.shape
        and base_prediction.shape[:2] == granularity.shape
    ):
        raise ValueError("utility predictions must have shape [sample,horizon,channel]")

    selected = granularity == int(selected_index)
    selected_count = int(selected.sum().item())
    if selected_count == 0:
        return {
            "count": 0,
            "gain_vs_base_pct": float("nan"),
            "harmful_fraction": float("nan"),
            "correction_abs_mean": float("nan"),
        }

    blended = base_prediction + strength.unsqueeze(-1) * (
        candidate_prediction - base_prediction
    )
    base_error = (base_prediction - target).abs().mean(dim=-1)
    selected_error = (blended - target).abs().mean(dim=-1)
    base_mae = base_error[selected].mean()
    selected_mae = selected_error[selected].mean()
    gain = 100.0 * (
        1.0 - selected_mae / base_mae.clamp_min(torch.finfo(torch.float64).eps)
    )
    correction = (blended - base_prediction).abs().mean(dim=-1)
    return {
        "count": selected_count,
        "gain_vs_base_pct": float(gain.item()),
        "harmful_fraction": float(
            (selected_error[selected] > base_error[selected]).double().mean().item()
        ),
        "correction_abs_mean": float(correction[selected].mean().item()),
    }


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

    expanded_target = target.unsqueeze(-1).expand_as(candidates)
    base_error = (aux["base_prediction"].detach() - target).abs().mean(dim=2)
    ranking_terms = torch.relu(
        candidate_error - base_error.unsqueeze(-1) + float(margin)
    )
    specialization_terms = []
    balanced_ranking_terms = []
    for candidate_index in range(candidates.size(-1)):
        branch_mask = availability[..., candidate_index]
        if not bool(branch_mask.any()):
            continue
        branch_channel_mask = branch_mask.unsqueeze(2).expand_as(
            candidates[..., candidate_index]
        )
        specialization_terms.append(
            torch.nn.functional.smooth_l1_loss(
                candidates[..., candidate_index][branch_channel_mask],
                expanded_target[..., candidate_index][branch_channel_mask],
                reduction="mean",
            )
        )
        balanced_ranking_terms.append(
            ranking_terms[..., candidate_index][branch_mask].mean()
        )
    # Equal branch weighting prevents the rarer composition adapter from being
    # overwhelmed by the much more frequent single-event windows.
    specialization = torch.stack(specialization_terms).mean()
    ranking = torch.stack(balanced_ranking_terms).mean()
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
