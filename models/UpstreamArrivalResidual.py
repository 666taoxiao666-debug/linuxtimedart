"""v33 residual branch on a FROZEN TSMixerBasicSmall base (proposal sections 14.13, 14.17, 14.18).

prediction = base(x) + M_h * head([c_h, z, e_h]), with the head's last layer zero-initialised (epoch 0 = base).

* tokens  : "stations" (content/phys from utils.kelmarsh_station_history_v33.gather) or "base_rows" (B1: the base's own
            21-channel input rows, one token per step, no prior).
* prior   : "none" (B2/B1: q_tilde = 1) or "arrival" (A and fixed-kappa arms). The prior only RE-DISTRIBUTES attention
            (weights ~ exp(e) * q_tilde); the non-null mass M = S/(1+S), S = mean_valid exp(e), is identical to B2.
            q_tilde = q / mean_valid(q) where the (t, h) support holds, else 1 (B2 behaviour, reported).
* kappa   : learned in [lo, hi] via lo + (hi - lo) * sigmoid(theta) (init 1), or a fixed value.
* variant : full | age0 | hflat | geo180 (ablations change the prior only; content is unchanged).
The arrival prior must agree with utils.kelmarsh_station_history_v33.prior_numpy (tested on the server).
"""
from __future__ import annotations

import math

import torch
from torch import nn


def base_forward(base, x):
    """(prediction [B, H], hidden [B, L, C]) of the unchanged TSMixer base (no power_only masking in v33)."""
    h = x
    for block in base.blocks:
        h = block(h)
    return base.head(h[..., base.target_slot]), h


class ArrivalResidual(nn.Module):
    def __init__(self, base, *, tokens="stations", prior="none", kappa="learned", variant="full", content_dim=12,
                 width=16, query_width=16, lead_width=16, horizon=4, score_clamp=8.0, kappa_bounds=(0.5, 4.0),
                 kappa_init=1.0, min_speed=2.0, speed_floor=2.0, cross_scale=30.0, sigma_min=1.0, sigma_frac=0.5,
                 support_eps=1e-6):
        super().__init__()
        if tokens not in ("stations", "base_rows") or prior not in ("none", "arrival"):
            raise ValueError("Unknown token source or prior")
        if variant not in ("full", "age0", "hflat", "geo180"):
            raise ValueError("Unknown prior variant")
        if tokens == "base_rows" and prior != "none":
            raise ValueError("B1 tokens carry no geometry; no arrival prior")
        self.base = base
        for q in self.base.parameters():
            q.requires_grad_(False)
        self.tokens, self.prior, self.variant, self.horizon = tokens, prior, variant, int(horizon)
        self.clamp, self.eps = float(score_clamp), float(support_eps)
        self.min_speed, self.floor, self.cross = float(min_speed), float(speed_floor), float(cross_scale)
        self.sigma_min, self.sigma_frac = float(sigma_min), float(sigma_frac)
        lo, hi = kappa_bounds
        self.kappa_lo, self.kappa_hi = float(lo), float(hi)
        if prior == "arrival" and kappa == "learned":
            p = (kappa_init - lo) / (hi - lo)
            self.kappa_theta = nn.Parameter(torch.tensor(math.log(p / (1 - p)), dtype=torch.float32))
            self.fixed_kappa = None
        else:
            self.kappa_theta = None
            self.fixed_kappa = None if prior == "none" else float(kappa)
        seq, channels = base.seq_len, base.channels
        token_in = (channels + 1) if tokens == "base_rows" else content_dim
        self.token_mlp = nn.Sequential(nn.Linear(token_in, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.query = nn.Sequential(nn.Linear(seq * channels, query_width), nn.GELU())
        self.lead = nn.Embedding(self.horizon, lead_width)
        self.key = nn.Linear(width, width)
        self.qproj = nn.Linear(query_width + lead_width, width)
        self.score = nn.Linear(width, 1)
        self.value = nn.Linear(width, width)
        self.head = nn.Sequential(nn.Linear(width + query_width + lead_width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()                      # the base is frozen: no dropout, no statistics update
        return self

    def kappa(self):
        if self.kappa_theta is not None:
            return self.kappa_lo + (self.kappa_hi - self.kappa_lo) * torch.sigmoid(self.kappa_theta)
        return None if self.fixed_kappa is None else torch.tensor(self.fixed_kappa)

    def arrival_q(self, phys, valid):
        """q [B, K, H], support [B, H], q_tilde [B, K, H] (mirror of prior_numpy)."""
        d_par, d_perp, cos_own, vbar, age, base_ok = phys.unbind(-1)
        if self.variant == "geo180":
            d_par, d_perp, cos_own = -d_par, -d_perp, -cos_own
        eligible = (base_ok > 0.5) & (d_par > 0) & valid
        kappa = self.kappa().to(phys)
        t_adv = torch.where(eligible, d_par / (3.6 * kappa * torch.clamp(vbar, min=self.floor)), torch.zeros_like(d_par))
        a = torch.zeros_like(age) if self.variant == "age0" else age
        r = (t_adv - a).unsqueeze(-1)
        sigma = torch.clamp(self.sigma_frac * t_adv, min=self.sigma_min).unsqueeze(-1)
        geo = (torch.clamp(cos_own, min=0) * torch.exp(-0.5 * (d_perp / self.cross) ** 2)).unsqueeze(-1)
        h = torch.arange(1, self.horizon + 1, device=phys.device, dtype=phys.dtype)
        if self.variant == "hflat":
            lo, hi = torch.zeros_like(h), torch.full_like(h, float(self.horizon))
        else:
            lo, hi = h - 1, h
        mass = torch.special.ndtr((hi - r) / sigma) - torch.special.ndtr((lo - r) / sigma)
        if self.variant == "hflat":
            mass = mass / self.horizon
        e = eligible.unsqueeze(-1)
        q = torch.where(e, mass * geo, torch.zeros_like(mass))
        centre = e & (r >= lo) & (r <= hi)
        support = (q.sum(1) >= self.eps) & centre.any(1)
        n_valid = valid.sum(1).clamp(min=1).unsqueeze(-1).to(q)
        mean_q = q.sum(1) / n_valid
        q_tilde = torch.where(support.unsqueeze(1), q / torch.where(mean_q > 0, mean_q, torch.ones_like(mean_q)).unsqueeze(1),
                              torch.ones_like(q))
        return q, support, q_tilde

    def forward(self, x, content=None, phys=None, valid=None, return_diagnostics=False):
        with torch.no_grad():
            base_pred, hidden = base_forward(self.base, x)
        b = x.shape[0]
        z = self.query(hidden.reshape(b, -1))
        if self.tokens == "base_rows":
            steps = torch.arange(x.shape[1], device=x.device, dtype=x.dtype) / x.shape[1]
            tok_in = torch.cat([x, steps.view(1, -1, 1).expand(b, -1, 1)], dim=-1)
            valid = torch.ones(tok_in.shape[:2], dtype=torch.bool, device=x.device)
        else:
            tok_in = content
            valid = valid.bool()
        t = self.token_mlp(tok_in)                                                     # [B, K, W]
        lead = self.lead.weight                                                        # [H, Lw]
        qv = self.qproj(torch.cat([z.unsqueeze(1).expand(-1, self.horizon, -1), lead.unsqueeze(0).expand(b, -1, -1)], -1))
        e = self.score(torch.tanh(self.key(t).unsqueeze(2) + qv.unsqueeze(1))).squeeze(-1)   # [B, K, H]
        e = torch.clamp(e, -self.clamp, self.clamp)
        vmask = valid.unsqueeze(-1).to(e)
        expe = torch.exp(e) * vmask
        n_valid = valid.sum(1).unsqueeze(-1).to(e)
        s = expe.sum(1) / n_valid.clamp(min=1)
        mass = s / (1 + s)
        if self.prior == "arrival":
            _, support, q_tilde = self.arrival_q(phys, valid)
        else:
            support, q_tilde = None, torch.ones_like(expe)
        w = expe * q_tilde
        norm = w.sum(1, keepdim=True)
        w = torch.where(norm > 0, w / torch.where(norm > 0, norm, torch.ones_like(norm)), torch.zeros_like(w))
        c = torch.einsum("bkh,bkw->bhw", w, self.value(t))
        delta = mass * self.head(torch.cat([c, z.unsqueeze(1).expand(-1, self.horizon, -1),
                                            lead.unsqueeze(0).expand(b, -1, -1)], -1)).squeeze(-1)
        delta = torch.where(n_valid > 0, delta, torch.zeros_like(delta))
        out = (base_pred + delta).unsqueeze(-1)
        if return_diagnostics:
            return out, {"support": support, "mass": mass}
        return out


def trainable_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
