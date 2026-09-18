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
