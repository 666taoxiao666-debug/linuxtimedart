"""History-only event experts and horizon-wise utility selection.

Frozen text anchors are adapted to the frozen trend representation. Physical
support is mandatory; semantic similarity is a feature, not a hard prefilter.
"""
import torch
from torch import nn
from torch.nn import functional as F

from utils.utility_calibration import utility_action_probabilities


class FactorizedEvidenceAdapter(nn.Module):
    def __init__(self, d_model, pred_len, num_factors, semantic_dim,
                 physical_dim, num_modes=3, max_scale=.2):
        super().__init__()
        self.num_factors, self.pred_len = num_factors, pred_len
        self.num_modes, self.max_scale = num_modes, float(max_scale)
        self.semantic_projection = nn.Linear(semantic_dim, d_model, bias=False)
        self.query = nn.Sequential(nn.LayerNorm(2 * d_model + physical_dim),
                                   nn.Linear(2 * d_model + physical_dim, d_model), nn.GELU())
        self.experts = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(3 * d_model + physical_dim + 2),
                          nn.Linear(3 * d_model + physical_dim + 2, d_model), nn.GELU(),
                          nn.Linear(d_model, num_modes * pred_len))
            for _ in range(num_factors)
        ])
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def forward(self, hidden, trend_probs, rules, support, physical, semantic_keys):
        # hidden [B,C,P,D]; output [B,H,C,F]. No gradient enters the trend model.
        hidden = hidden.detach()
        pooled, last = hidden.mean(2), hidden[:, :, -1]
        physical = physical.detach()
        query = self.query(torch.cat([pooled.mean(1), last.mean(1), physical], -1))
        keys = self.semantic_projection(semantic_keys.detach())
        similarity = F.normalize(query, dim=-1) @ F.normalize(keys, dim=-1).T
        # A continuous semantic path can learn even when a frozen old retriever
        # would reject the event. Negative physical evidence still vetoes it.
        descriptors = keys[None] * torch.sigmoid(similarity)[..., None]
        corrections = []
        for index, expert in enumerate(self.experts):
            evidence = torch.stack([rules[:, index].clamp(0, 1), support[:, index]], -1)
            state = torch.cat([pooled, last,
                               descriptors[:, index, None].expand(-1, hidden.size(1), -1),
                               physical[:, None].expand(-1, hidden.size(1), -1),
                               evidence[:, None].expand(-1, hidden.size(1), -1)], -1)
            modes = expert(state).reshape(hidden.size(0), hidden.size(1), self.num_modes, self.pred_len)
            residual = (modes * trend_probs.detach()[:, None, :, None]).sum(2)
            residual = self.max_scale * residual.tanh()
            residual = residual * (support[:, index] > 0)[:, None, None]
            corrections.append(residual.permute(0, 2, 1))
        return torch.stack(corrections, -1), descriptors


class FactorUtilityGate(nn.Module):
    def __init__(self, d_model, pred_len, physical_dim, num_modes=3,
                 temperature=.05, min_gain=.001):
        super().__init__()
        self.temperature, self.min_gain = float(temperature), float(min_gain)
        self.pred_len = pred_len
        self.candidate_conditioned, self.physical_dim = True, physical_dim
        self.num_trend_modes, self.intervention_floor = num_modes, 1.
        self.estimator = nn.Sequential(
            nn.LayerNorm(3 * d_model + physical_dim + num_modes + pred_len + 2),
            nn.Linear(3 * d_model + physical_dim + num_modes + pred_len + 2, d_model),
            nn.GELU(), nn.Linear(d_model, pred_len))
        nn.init.zeros_(self.estimator[-1].weight)
        nn.init.zeros_(self.estimator[-1].bias)

    def forward(self, hidden, trend, physical, descriptors, corrections, support, rules):
        # Each event AND the composition receives a utility for every horizon.
        count = descriptors.size(1)
        common = torch.cat([hidden.mean(1), hidden[:, -1], trend, physical], -1).detach()
        state = torch.cat([common[:, None].expand(-1, count, -1), descriptors.detach(),
                           corrections.detach().mean(2).transpose(1, 2),
                           support[..., None], rules.clamp(0, 1)[..., None]], -1)
        return self.estimator(state).transpose(1, 2)


def factorized_route(base, corrections, scores, availability, min_gain, temperature, soft):
    masked = scores.masked_fill(~availability[:, None], float('-inf'))
    best, index = masked.max(-1)
    intervene = best > min_gain
    selected = corrections.gather(-1, index[:, :, None, None].expand(-1, -1, base.size(2), 1)).squeeze(-1)
    hard_prediction = base + selected * intervene[..., None]
    probability = utility_action_probabilities(scores, availability, min_gain, temperature)
    soft_prediction = base + (corrections * probability[:, :, None, 1:]).sum(-1)
    action = torch.where(intervene, index + 1, torch.zeros_like(index))
    return (soft_prediction if soft else hard_prediction), soft_prediction, hard_prediction, action


def factorized_loss(aux, target, kind, margin=0., min_gain=0., temperature=.05):
    predictions = aux['factor_predictions']
    available = aux['factor_availability'][:, None].expand(-1, target.size(1), -1)
    base_error = (aux['base_prediction'].detach() - target).abs().mean(2)
    errors = (predictions - target[..., None]).abs().mean(2)
    zero = predictions.sum() * 0. + aux['factor_utilities'].sum() * 0.
    if kind == 'candidate':
        if not available.any():
            return zero, zero, {}
        # Equal expert weighting, natural observations within each expert.
        fits, ranks = [], []
        for k in range(predictions.size(-1)):
            mask = available[..., k]
            if mask.any():
                correction = predictions[..., k] - aux['base_prediction'].detach()
                fit = errors[..., k] + .01 * correction.square().mean(2)
                fits.append(fit[mask].mean())
                ranks.append(F.relu(errors[..., k] - base_error + margin)[mask].mean())
        return torch.stack(fits).mean(), torch.stack(ranks).mean(), {}
    gain = (base_error[..., None] - errors.detach()).detach()
    if kind == 'supervision':
        loss = F.smooth_l1_loss(aux['factor_utilities'][available], gain[available]) if available.any() else zero
        return loss, gain
    # Cost-sensitive expected regret only; do not balance action-class counts.
    probability = utility_action_probabilities(aux['factor_utilities'], aux['factor_availability'], min_gain, temperature)
    costs = torch.cat([base_error[..., None], errors.detach()], -1)
    mask = torch.cat([torch.ones_like(available[..., :1]), available], -1)
    best = costs.masked_fill(~mask, float('inf')).min(-1).values
    regret = (probability * (costs - best[..., None])).sum(-1)
    has_evidence = available.any(-1)
    return (regret[has_evidence].mean() if has_evidence.any() else zero), {}
