"""G1: upstream-innovation displacement correction of an NWP farm trajectory (arm M) and its matched controls.

Written by Claude (2026-11-11) per docs/STAGE_PLAN.md stage M7 part C (final after the R2-B1 ruling).
Codex writes the GFS point extraction, the per-sample station batch and the training driver on top of this module.

The numeric core is written once against a tiny array shim, so the same code runs on numpy (static tests in a
torch-free sandbox, reference values) and on torch (training with autograd on the server).

Per-sample inputs (a dict; every value finite; masked entries may hold any finite value, they are zeroed here):
  oS  [B,S,3]  station 10 m wind speed (m/s) observed at t0-3, t0-2, t0-1           oS_mask [B,S,3]
  gS  [B,S,8]  station GFS 10 m wind speed (m/s) from cycle C(t0), valid t0-5..t0+2  gS_mask [B,S,8]
  dS  [B,S,2]  station observed wind-FROM direction (sin, cos) at t0-1              dS_mask [B,S]
  gF  [B,13]   farm GFS wind speed (m/s) from cycle C(t0), valid t0-5..t0+7          gF_mask [B,13]
  age [B]      t0 - C(t0) in hours (6..11 under the 6 h publication rule)
  cycle_ok [B] 1 if C(t0) exists, else 0 (no fallback to another cycle)
  geo [S,3]    dist_100km, bearing_sin, bearing_cos from the farm to each station (fixed per farm)
All GFS values of one sample come from the single cycle C(t0); entries whose lead is outside [3, 18] h are masked
here again from `age` (defence in depth). oS and gS must be in the same unit (m/s).

Blocks (outputs are per-sample vectors, later broadcast as constant channels over the 24 input steps):
  pooled_block        (arm I_pool; also inside I and M)  28 ch: kernel-weighted station means of o(3), gS(8),
                      o - gS at the window (3), and the 14 matching coverages.
  phi_block           (arm I, generic control)           7 ch: per-station MLP phi(29 -> 16 -> 7, ReLU), then the
                      same kernel-weighted mean. phi has 599 parameters.
  displacement_block  (arm M, the method)                7 ch: corrected farm trajectory
                      g_F(t0+h+delta_F) + lambda * b_F (h = 1..4), delta_F (h), b_F (m/s), confidence.
  Shift delta > 0 means the weather observed upstream arrives earlier than the forecast: o(tau) ~ g(tau + delta).
  Arm M_self (control, added 2026-11-12 after the prior-art check, cf. Liu et al. 2023 Applied Energy 336:120815):
  the same displacement block, but the shift is estimated from the farm's OWN observed wind oF [B,3] (oF_mask)
  against the farm forecast gF at t0-5..t0+2; it also gets the 28 pooled station channels, so M vs M_self differs
  only in where the timing error is read (upstream stations vs the farm itself).
  Missing farm forecasts (R1-B2): the correction is applied only if every farm point that the shifted linear
  interpolation reads (all 4 horizons) is valid. Otherwise the sample gets the "no correction" state: the 4
  trajectory channels hold the uncorrected g_F(t0+h) (0 where masked), delta_F = b_F = 0 and confidence = 0.
  confidence = 0 always means "no correction applied" (also: no confident station, cycle missing).
  G1c (2026-11-12, after GPT's prior-art/identifiability review):
  - Confidence shrinkage (definite-defect fix): delta_F and b_F are normalised by the kernel weight of all usable
    stations, not by the confidence-weighted sum, i.e. delta_F = conf * (confidence-weighted mean shift). A weakly
    informative station can no longer impose its full soft-argmin shift (near-uniform costs over the admissible
    set {0..3} used to give ~ +1.5 h); uniform costs now give delta_F = 0.
  - Identifiability boundary (documented, not "fixed"): on a locally linear trajectory a shift and a level bias are
    indistinguishable (f(s+d) + b = f(s) + (v d + b)); M then reports an effective shift. Attribution to timing is
    tested empirically by M vs M_amp (and M_phase), never asserted from delta_F alone.
  - Extra controls: I_w (phi width 64), R_sp (direct linear transfer of pooled upstream residuals to the farm
    trajectory with wind-direction interactions, Tastu-style, no shift), M_amp (the M block with delta fixed at 0:
    level correction only), M_phase (the M block with lambda fixed at 0 and the b_F channel zeroed: shift only).
The upwind kernel is the expert program's sigmoid cone x anisotropic exponential distance (models/ProgramEncoder.py,
init [1.386, 0.3, 0.0, 0.0]); each arm owns its kernel parameters.
"""
from __future__ import annotations

import math

try:  # torch is optional at import time (the numpy path is torch-free)
    import torch as _torch
    from torch import nn as _nn
except ImportError:  # pragma: no cover - sandbox without torch
    _torch = None
    _nn = None

W_OFF = (-3, -2, -1)                 # observation window W relative to t0
GS_OFF = tuple(range(-5, 3))         # station forecast valid offsets t0-5..t0+2 (8)
GF_OFF = tuple(range(-5, 8))         # farm forecast valid offsets t0-5..t0+7 (13)
HORIZONS = (1, 2, 3, 4)
SHIFTS = (-2, -1, 0, 1, 2, 3)
LEAD_MIN, LEAD_MAX = 3.0, 18.0
KERNEL_INIT = (1.386, 0.3, 0.0, 0.0)
PHI_IN, PHI_HIDDEN, PHI_OUT = 29, 16, 7
PHI_HIDDEN_W = 64                    # arm I_w (wider generic control, G1c)
POOLED_OUT, DISP_OUT = 28, 7
RSP_FEATS = 3 * len(W_OFF)           # R_sp: pooled residual at each tau in W, x (1, sin, cos) of the regional wind
ARMS = ("B", "N", "I_pool", "I", "M", "M_self", "I_w", "R_sp", "M_amp", "M_phase")
DISP_MODES = {"M": "full", "M_self": "full", "M_amp": "amp", "M_phase": "phase"}
PHI_ARMS = {"I": PHI_HIDDEN, "I_w": PHI_HIDDEN_W}
TEMP_FLOOR = 0.05
T_RAW_INIT = math.log(math.e - 1.0)  # softplus(t_raw) = 1.0 -> temperature 1.05 (m/s)^2 / ws_scale^2
L_RAW_INIT = 0.0                     # lambda = sigmoid(0) = 0.5
EPS = 1e-6
_K0 = GS_OFF.index(W_OFF[0])         # index of t0-3 in the station forecast axis (= 2)
_F0 = GF_OFF.index(0)                # index of t0 in the farm forecast axis (= 5)
BATCH_KEYS = {"oS": 3, "oS_mask": 3, "gS": 8, "gS_mask": 8, "dS": 2, "dS_mask": None,
              "gF": None, "gF_mask": None, "age": None, "cycle_ok": None}
OPTIONAL_KEYS = ("oF", "oF_mask")    # farm's own observed wind at t0-3..t0-1 (required for arm M_self only)
STATION_ARMS = ("I_pool", "I", "M", "M_self", "I_w", "R_sp", "M_amp", "M_phase")


def block_width(arm):
    """Number of station-block channels each arm appends (B and N have none); every station arm except I_pool adds
    the same 7 channels on top of the 28 pooled ones."""
    if arm in ("B", "N"):
        return 0
    if arm == "I_pool":
        return POOLED_OUT
    if arm in STATION_ARMS:
        return POOLED_OUT + DISP_OUT
    raise KeyError(arm)


def phi_param_count(hidden=PHI_HIDDEN):
    return PHI_IN * hidden + hidden + hidden * PHI_OUT + PHI_OUT


def structural_param_counts():
    """Learnable parameters outside the TSMixer backbone, per arm."""
    k = len(KERNEL_INIT)
    return {"B": 0, "N": 0, "I_pool": k, "I": k + phi_param_count(), "M": k + 2, "M_self": k + 2,
            "I_w": k + phi_param_count(PHI_HIDDEN_W), "R_sp": k + RSP_FEATS * len(HORIZONS),
            "M_amp": k + 1, "M_phase": k + 1}


# --------------------------------------------------------------------------- array shim
class _NumpyOps:
    def __init__(self):
        import numpy as np
        self.np = np

    def const(self, values, like=None):
        return self.np.asarray(values, dtype=self.np.float64)

    def ge(self, a, b):
        return self.np.asarray(self.np.asarray(a) >= self.np.asarray(b), dtype=self.np.float64)

    def sum(self, x, axis, keepdims=False):
        return x.sum(axis=axis, keepdims=keepdims)

    def amax(self, x, axis, keepdims=False):
        return x.max(axis=axis, keepdims=keepdims)

    def cat(self, xs, axis):
        return self.np.concatenate(xs, axis=axis)

    def stack(self, xs, axis):
        return self.np.stack(xs, axis=axis)

    def where(self, cond, a, b):
        return self.np.where(cond, a, b)

    def exp(self, x):
        return self.np.exp(x)

    def sqrt(self, x):
        return self.np.sqrt(x)

    def abs(self, x):
        return self.np.abs(x)

    def clip_min(self, x, lo):
        return self.np.maximum(x, lo)

    def clip(self, x, lo, hi):
        return self.np.clip(x, lo, hi)

    def relu(self, x):
        return self.np.maximum(x, 0.0)

    def sigmoid(self, x):
        return 1.0 / (1.0 + self.np.exp(-x))

    def softplus(self, x):
        return self.np.log1p(self.np.exp(-self.np.abs(x))) + self.np.maximum(x, 0.0)


class _TorchOps:
    def __init__(self):
        if _torch is None:
            raise RuntimeError("torch is not installed")
        self.t = _torch

    def const(self, values, like):
        return self.t.as_tensor(values, dtype=like.dtype, device=like.device)

    def ge(self, a, b):
        ref = a if self.t.is_tensor(a) else b
        return (a >= b).to(ref.dtype)

    def sum(self, x, axis, keepdims=False):
        return x.sum(dim=axis, keepdim=keepdims)

    def amax(self, x, axis, keepdims=False):
        return x.amax(dim=axis, keepdim=keepdims)

    def cat(self, xs, axis):
        return self.t.cat(xs, dim=axis)

    def stack(self, xs, axis):
        return self.t.stack(xs, dim=axis)

    def where(self, cond, a, b):
        return self.t.where(cond, a, b)

    def exp(self, x):
        return self.t.exp(x)

    def sqrt(self, x):
        return self.t.sqrt(x)

    def abs(self, x):
        return self.t.abs(x)

    def clip_min(self, x, lo):
        return self.t.clamp(x, min=lo)

    def clip(self, x, lo, hi):
        return self.t.clamp(x, min=lo, max=hi)

    def relu(self, x):
        return self.t.relu(x)

    def sigmoid(self, x):
        return self.t.sigmoid(x)

    def softplus(self, x):
        return _nn.functional.softplus(x)


# --------------------------------------------------------------------------- numeric core (backend agnostic)
def lead_masks(o, age, cycle_ok):
    """[B,8] station and [B,13] farm masks: lead = offset + age within [LEAD_MIN, LEAD_MAX] and cycle present."""
    ok = cycle_ok[:, None]
    ls = age[:, None] + o.const(GS_OFF, age)[None, :]
    lf = age[:, None] + o.const(GF_OFF, age)[None, :]
    ms = o.ge(ls, LEAD_MIN) * o.ge(LEAD_MAX, ls) * ok
    mf = o.ge(lf, LEAD_MIN) * o.ge(LEAD_MAX, lf) * ok
    return ms, mf


def admissible_shifts(o, age, cycle_ok):
    """[B,6] whole-window rule (R1-S2): delta usable only if every station read tau+delta (tau in W) and every
    farm read h+delta (h = 1..4) lies inside the downloaded offsets and has lead within [LEAD_MIN, LEAD_MAX]."""
    cols = []
    for d in SHIFTS:
        s_lo, s_hi = min(W_OFF) + d, max(W_OFF) + d
        f_lo, f_hi = min(HORIZONS) + d, max(HORIZONS) + d
        inside = (s_lo >= GS_OFF[0] and s_hi <= GS_OFF[-1] and f_lo >= GF_OFF[0] and f_hi <= GF_OFF[-1])
        col = (o.ge(age + s_lo, LEAD_MIN) * o.ge(LEAD_MAX, age + s_hi)
               * o.ge(age + f_lo, LEAD_MIN) * o.ge(LEAD_MAX, age + f_hi) * cycle_ok)
        cols.append(col * (1.0 if inside else 0.0))
    return o.stack(cols, 1)


def regional_direction(o, dS, dS_mask):
    """Unit vector (sin, cos) [B] of the mean observed wind-from direction; (0, 0) if no station has a direction."""
    m = dS_mask
    n = o.sum(m, 1) + EPS
    rs = o.sum(dS[..., 0] * m, 1) / n
    rc = o.sum(dS[..., 1] * m, 1) / n
    rn = o.sqrt(rs * rs + rc * rc) + EPS
    return rs / rn, rc / rn


def kernel_weights(o, kp, geo, dS, dS_mask):
    """Expert upwind kernel: sigmoid cone on alignment with the regional wind-from direction x anisotropic
    exponential decay. Returns weights [B,S] and alignment [B,S]."""
    us, uc = regional_direction(o, dS, dS_mask)
    align = geo[None, :, 1] * us[:, None] + geo[None, :, 2] * uc[:, None]
    dist = geo[None, :, 0] + 0.0 * align
    cross = dist * o.sqrt(o.clip_min(1.0 - align * align, 0.0) + 1e-9)
    cone = o.sigmoid(o.exp(kp[0]) * (align - kp[1]))
    decay = o.exp(-dist / (o.softplus(kp[2]) + 0.05) - cross / (o.softplus(kp[3]) + 0.05))
    return cone * decay, align


def _station_parts(o, b, ms):
    om = b["oS_mask"]
    gm = b["gS_mask"] * ms[:, None, :]
    rm = om * gm[..., _K0:_K0 + len(W_OFF)]
    r = (b["oS"] - b["gS"][..., _K0:_K0 + len(W_OFF)]) * rm
    valid = o.clip(o.sum(om, -1) + o.sum(gm, -1), 0.0, 1.0)
    return om, gm, rm, r, valid


def pooled_block(o, b, w, ms):
    """[B,28]: kernel-weighted means of o(3), gS(8), residual(3), then their coverages (arm I_pool)."""
    om, gm, rm, r, _ = _station_parts(o, b, ms)
    vals = o.cat([b["oS"] * om, b["gS"] * gm, r], -1)
    masks = o.cat([om, gm, rm], -1)
    ww = w[..., None]
    den = o.sum(ww * masks, 1)
    mean = o.sum(ww * masks * vals, 1) / o.clip_min(den, EPS)
    cov = den / o.clip_min(o.sum(w, 1)[:, None], EPS)
    return o.cat([mean, cov], -1)


def phi_features(o, b, geo, align, ms):
    """[B,S,29] per-station raw inputs of the generic control: o, o-mask, gS, gS-mask, residual, geometry."""
    om, gm, rm, r, _ = _station_parts(o, b, ms)
    z = 0.0 * align[..., None]
    g = [geo[None, :, i, None] + z for i in range(3)]
    return o.cat([b["oS"] * om, om, b["gS"] * gm, gm, r] + g + [align[..., None]], -1)


def phi_block(o, b, w, geo, align, ms, phi_fn):
    """[B,7]: generic per-station encoder phi, then the same kernel-weighted mean (arm I, R2-B1 option 1)."""
    *_, valid = _station_parts(o, b, ms)
    z = phi_fn(phi_features(o, b, geo, align, ms))
    ww = (w * valid)[..., None]
    return o.sum(ww * z, 1) / o.clip_min(o.sum(ww, 1), EPS)


def residual_transfer_block(o, b, w, ms, mf, beta, us, uc):
    """[B,7] arm R_sp (G1c): direct linear transfer of upstream residuals to the farm trajectory, no shift.
    rbar_k = kernel-weighted mean of o - gS at tau_k in W (the same residuals I_pool pools); features
    x = [rbar_k, rbar_k * sin(theta), rbar_k * cos(theta)]_k (theta = regional wind-from direction, Tastu-style
    direction-conditioned coefficients); correction Delta_h = x . beta[:, h] (beta [9,4], init 0 = no correction).
    Applied per horizon only where g_F(t0+h) is valid and at least one residual exists; otherwise the plain g_F(t0+h).
    Channels: corrected g_F(t0+h) (4), mean applied Delta (1), residual coverage (1), applied flag (1)."""
    _, _, rm, r, _ = _station_parts(o, b, ms)
    ww = w[..., None]
    den = o.sum(ww * rm, 1)                                          # [B,3]
    rbar = o.sum(ww * r, 1) / o.clip_min(den, EPS)                   # r is already masked
    x = o.cat([rbar, rbar * us[:, None], rbar * uc[:, None]], -1)    # [B,9]
    delta = o.sum(x[:, :, None] * beta[None, :, :], 1)               # [B,4]
    has_r = o.ge(o.sum(o.ge(den, EPS), -1), 0.5) * b["cycle_ok"]
    fm = b["gF_mask"] * mf
    plain = (b["gF"] * fm)[:, _F0 + HORIZONS[0]:_F0 + HORIZONS[-1] + 1]
    app = fm[:, _F0 + HORIZONS[0]:_F0 + HORIZONS[-1] + 1] * has_r[:, None]
    g_hat = plain + app * delta
    mean_d = o.sum(app * delta, -1) / o.clip_min(o.sum(app, -1), 1.0)
    cov = o.sum(den, -1) / o.clip_min(len(W_OFF) * o.sum(w, 1), EPS)
    out = o.cat([g_hat, mean_d[:, None], cov[:, None], has_r[:, None]], -1)
    return out * b["cycle_ok"][:, None]


def displacement_block(o, b, w, ms, mf, adm, temp, lam, ws_scale=1.0, return_diagnostics=False, mode="full"):
    """[B,7]: per-station soft shift matching -> farm shift delta_F, bias b_F -> corrected farm trajectory.
    mode "full" (M, M_self); "amp" (M_amp: only delta = 0 is admissible, each station with a valid delta-0 match
    has confidence 1, so b_F is the kernel-weighted mean residual); "phase" (M_phase: lambda = 0 and b_F output 0)."""
    if mode not in ("full", "amp", "phase"):
        raise ValueError(f"unknown displacement mode {mode!r}")
    om = b["oS_mask"]
    gm = b["gS_mask"] * ms[:, None, :]
    n_obs = o.sum(om, -1)
    J, E, A = [], [], []
    nw = len(W_OFF)
    for i, d in enumerate(SHIFTS):
        lo = _K0 + d
        if lo < 0 or lo + nw > len(GS_OFF):
            raise ValueError("shift reads outside the station forecast axis")
        m_d = om * gm[..., lo:lo + nw]
        diff = (b["oS"] - b["gS"][..., lo:lo + nw]) * m_d
        n_d = o.sum(m_d, -1)
        full = o.ge(n_d, n_obs) * o.ge(n_obs, 1.0)  # every observed tau has a forecast at tau+delta
        A.append(full * adm[:, i, None])
        J.append(o.sum((diff / ws_scale) ** 2, -1) / o.clip_min(n_d, EPS))
        E.append(o.sum(diff, -1) / o.clip_min(n_d, EPS))
    J, E, A = o.stack(J, -1), o.stack(E, -1), o.stack(A, -1)          # [B,S,6]
    if mode == "amp":
        A = A * o.const([1.0 if d == 0 else 0.0 for d in SHIFTS], J)[None, None, :]
    on = A > 0.5
    z = o.where(on, -J / temp, J * 0.0 - 1e9)
    z = z - o.amax(z, -1, keepdims=True)
    p = o.exp(z) * A
    p = p / o.clip_min(o.sum(p, -1, keepdims=True), EPS)
    n_av = o.sum(A, -1)
    st_ok = o.ge(n_av, 1.0)
    shifts = o.const(SHIFTS, J)
    dhat = o.sum(p * shifts, -1)
    bres = o.sum(p * E, -1)
    conc = o.sum(p * p, -1)
    inv_n = 1.0 / o.clip_min(n_av, 1.0)
    if mode == "amp":
        c = st_ok                                                     # one candidate: no shift to be confident about
    else:
        c = o.clip((conc - inv_n) / o.clip_min(1.0 - inv_n, EPS), 0.0, 1.0) * o.ge(n_av, 2.0)
    omega = w * c * st_ok
    den = o.sum(omega, 1)
    usable = o.clip_min(o.sum(w * st_ok, 1), EPS)
    # G1c shrinkage: normalise by all usable stations' kernel weight, so delta_F = conf * (confidence-weighted mean);
    # low confidence -> towards no correction (the old den-normalisation let one weak station impose its full shift).
    d_f = o.sum(omega * dhat, 1) / usable
    b_f = o.sum(omega * bres, 1) / usable
    conf = den / usable
    if mode == "phase":
        lam = 0.0
        b_f = b_f * 0.0
    fm = b["gF_mask"] * mf
    grid = o.const(list(range(len(GF_OFF))), J)                     # farm axis positions 0..12
    hz = o.const(HORIZONS, J)
    pos = hz[None, :] + _F0 + d_f[:, None]                          # [B,4] fractional positions
    hat = o.relu(1.0 - o.abs(pos[..., None] - grid[None, None, :]))  # linear interpolation weights [B,4,13]
    num = o.sum(hat * (b["gF"] * fm)[:, None, :], -1)
    wsum = o.sum(hat * fm[:, None, :], -1)
    # R1-B2: the correction is applied only if every farm point read by the shifted interpolation (all 4 horizons)
    # is a valid forecast; no one-sided extrapolation. Otherwise the sample falls back to the "no correction" state:
    # g_hat = uncorrected g_F(t0+h) (0 where that forecast is masked; N's mask channels carry it), delta_F = b_F = 0,
    # confidence = 0. confidence = 0 therefore always means "no correction applied".
    v_h = o.ge(wsum, 1.0 - 1e-5)
    applied = o.ge(o.sum(v_h, -1), len(HORIZONS) - 0.5) * b["cycle_ok"]
    plain = (b["gF"] * fm)[:, _F0 + HORIZONS[0]:_F0 + HORIZONS[-1] + 1]
    g_corr = num / o.clip_min(wsum, EPS) + lam * b_f[:, None]
    g_hat = applied[:, None] * g_corr + (1.0 - applied[:, None]) * plain
    d_f, b_f, conf = d_f * applied, b_f * applied, conf * applied
    out = o.cat([g_hat, d_f[:, None], b_f[:, None], conf[:, None]], -1) * b["cycle_ok"][:, None]
    if return_diagnostics:
        return out, {"p": p, "available": A, "delta_station": dhat, "confidence_station": c, "cost": J,
                     "applied": applied}
    return out


def self_site_batch(o, b):
    """Arm M_self: the farm itself as the only 'station'. Its observed wind oF (t0-3..t0-1) is matched against the
    farm forecast gF at valid t0-5..t0+2 (the first 8 farm offsets equal GS_OFF), with weight 1."""
    if tuple(GF_OFF[:len(GS_OFF)]) != tuple(GS_OFF):
        raise ValueError("farm and station forecast axes are no longer aligned")
    n = len(GS_OFF)
    return {"oS": b["oF"][:, None, :], "oS_mask": b["oF_mask"][:, None, :],
            "gS": b["gF"][:, None, :n], "gS_mask": b["gF_mask"][:, None, :n],
            "gF": b["gF"], "gF_mask": b["gF_mask"], "cycle_ok": b["cycle_ok"]}


def station_blocks(o, arm, b, geo, params, phi_fn=None, ws_scale=1.0):
    """Station-block channels for one arm: I_pool -> 28, I -> 28 + 7 (phi), M -> 28 + 7 (upstream displacement),
    M_self -> 28 + 7 (the same displacement structure driven by the farm's own observed wind, TLA-like control)."""
    if arm not in STATION_ARMS:
        raise ValueError(f"station blocks exist only for {STATION_ARMS}")
    ms, mf = lead_masks(o, b["age"], b["cycle_ok"])
    w, align = kernel_weights(o, params["kp"], geo, b["dS"], b["dS_mask"])
    parts = [pooled_block(o, b, w, ms)]
    if arm in PHI_ARMS:
        parts.append(phi_block(o, b, w, geo, align, ms, phi_fn))
    elif arm == "R_sp":
        us, uc = regional_direction(o, b["dS"], b["dS_mask"])
        parts.append(residual_transfer_block(o, b, w, ms, mf, params["beta"], us, uc))
    elif arm in DISP_MODES:
        mode = DISP_MODES[arm]
        temp = o.softplus(params["t_raw"]) + TEMP_FLOOR if mode != "amp" else 1.0
        lam = o.sigmoid(params["l_raw"]) if mode != "phase" else 0.0
        adm = admissible_shifts(o, b["age"], b["cycle_ok"])
        if arm == "M_self":
            ones = b["cycle_ok"][:, None] * 0.0 + 1.0
            parts.append(displacement_block(o, self_site_batch(o, b), ones, ms, mf, adm, temp, lam, ws_scale))
        else:
            parts.append(displacement_block(o, b, w, ms, mf, adm, temp, lam, ws_scale, mode=mode))
    return o.cat(parts, -1)


# --------------------------------------------------------------------------- numpy reference
def validate_batch(batch, n_stations, arm=None):
    """Shape and finiteness checks for a numpy station batch (use in the data-preparation tests)."""
    import numpy as np
    B = np.asarray(batch["age"]).shape[0]
    expect = {"oS": (B, n_stations, 3), "oS_mask": (B, n_stations, 3), "gS": (B, n_stations, 8),
              "gS_mask": (B, n_stations, 8), "dS": (B, n_stations, 2), "dS_mask": (B, n_stations),
              "gF": (B, 13), "gF_mask": (B, 13), "age": (B,), "cycle_ok": (B,)}
    for k, shape in expect.items():
        a = np.asarray(batch[k], dtype=float)
        if a.shape != shape:
            raise ValueError(f"{k}: shape {a.shape} != {shape}")
        if not np.isfinite(a).all():
            raise ValueError(f"{k}: non-finite values (fill masked entries with any finite number)")
        if k.endswith("_mask") or k == "cycle_ok":
            if not np.isin(a, (0.0, 1.0)).all():
                raise ValueError(f"{k}: masks must be 0/1")
    present = [k for k in OPTIONAL_KEYS if k in batch]
    if present and len(present) != len(OPTIONAL_KEYS):
        raise ValueError("oF and oF_mask must be given together (missing data = mask 0, not a missing key)")
    if arm == "M_self" and not present:
        raise ValueError("arm M_self needs oF and oF_mask")
    for k in present:
        a = np.asarray(batch[k], dtype=float)
        if a.shape != (B, 3) or not np.isfinite(a).all():
            raise ValueError(f"{k}: expected finite [B, 3]")
        if k.endswith("_mask") and not np.isin(a, (0.0, 1.0)).all():
            raise ValueError(f"{k}: masks must be 0/1")


# --------------------------------------------------------------------------- observation calibration (G1b R1-B1)
CAL_N_MIN = 720                # valid fit-period pairs (30 days of hours)
CAL_SD_MIN = 0.5               # m/s, both series
CAL_R_MIN = 0.5                # Pearson correlation
CAL_B_RANGE = (1.0 / 3.0, 3.0)  # admissible slope


def fit_affine_calibration(o, g, n_min=CAL_N_MIN, sd_min=CAL_SD_MIN, r_min=CAL_R_MIN, b_range=CAL_B_RANGE):
    """Per-site calibration of observed wind o to the GFS scale, fitted on FIT-PERIOD pairs only.

    o, g: 1-D arrays of the same valid hours (o = observed wind, g = the causally available GFS value at that hour);
    pairs with any non-finite value are dropped. Rule (pre-registered, identical for every arm):
      1. n >= n_min, sd(o) >= sd_min, sd(g) >= sd_min, r >= r_min, and the reduced-major-axis slope
         b = sd(o) / sd(g) inside b_range  ->  method "rma": a = mean(o) - b * mean(g).
         (RMA is symmetric in o and g and never 0 when both spreads pass; OLS of o on g can give b = 0.)
      2. otherwise, if n >= n_min                                  ->  method "offset": b = 1, a = mean(o - g).
      3. otherwise (too few pairs)                                 ->  method "masked": the site's observations
         are masked (mask 0 at every time) in every arm.
    Returns a dict with a, b, method, reason and the diagnostics n, sd_o, sd_g, r.
    """
    import numpy as np
    o = np.asarray(o, dtype=float).ravel()
    g = np.asarray(g, dtype=float).ravel()
    if o.shape != g.shape:
        raise ValueError("o and g must be aligned 1-D arrays")
    keep = np.isfinite(o) & np.isfinite(g)
    o, g = o[keep], g[keep]
    n = int(o.size)
    out = {"n": n, "sd_o": None, "sd_g": None, "r": None, "a": None, "b": None}
    if n < n_min:
        return dict(out, method="masked", reason=f"n={n} < {n_min}")
    sd_o, sd_g = float(o.std()), float(g.std())
    r = float(np.corrcoef(o, g)[0, 1]) if sd_o > 0 and sd_g > 0 else float("nan")
    out.update(sd_o=sd_o, sd_g=sd_g, r=r if np.isfinite(r) else None)
    fails = []
    if sd_o < sd_min:
        fails.append(f"sd_o={sd_o:.3g} < {sd_min}")
    if sd_g < sd_min:
        fails.append(f"sd_g={sd_g:.3g} < {sd_min}")
    if not (np.isfinite(r) and r >= r_min):
        fails.append(f"r={r:.3g} < {r_min}")
    if not fails:
        b = sd_o / sd_g
        if not b_range[0] <= b <= b_range[1]:
            fails.append(f"b={b:.3g} outside {b_range}")
        else:
            return dict(out, a=float(o.mean() - b * g.mean()), b=float(b), method="rma", reason="ok")
    return dict(out, a=float((o - g).mean()), b=1.0, method="offset", reason="; ".join(fails))


def apply_calibration(o, mask, cal):
    """o' = (o - a) / b on valid entries; method "masked" zeroes the mask. Returns (o', mask')."""
    import numpy as np
    o = np.asarray(o, dtype=float)
    mask = np.asarray(mask, dtype=float)
    if cal["method"] == "masked":
        return np.zeros_like(o), np.zeros_like(mask)
    if not (np.isfinite(cal["b"]) and np.isfinite(cal["a"])
            and CAL_B_RANGE[0] <= cal["b"] <= CAL_B_RANGE[1]):
        raise ValueError("calibration with an inadmissible slope")
    out = np.where(mask > 0, (o - cal["a"]) / cal["b"], 0.0)
    return out, mask


def init_params_numpy(arm, seed=0):
    """Parameters at the registered init (phi: PyTorch-style uniform(+-1/sqrt(fan_in)), deterministic)."""
    import numpy as np
    p = {"kp": np.asarray(KERNEL_INIT, dtype=float), "t_raw": T_RAW_INIT, "l_raw": L_RAW_INIT}
    if arm in PHI_ARMS:
        hid = PHI_ARMS[arm]
        rng = np.random.default_rng(seed)
        a1, a2 = 1 / math.sqrt(PHI_IN), 1 / math.sqrt(hid)
        p.update(W1=rng.uniform(-a1, a1, (PHI_IN, hid)), b1=rng.uniform(-a1, a1, hid),
                 W2=rng.uniform(-a2, a2, (hid, PHI_OUT)), b2=rng.uniform(-a2, a2, PHI_OUT))
    if arm == "R_sp":
        p["beta"] = np.zeros((RSP_FEATS, len(HORIZONS)))
    return p


def station_blocks_numpy(arm, batch, geo, params=None, ws_scale=1.0, seed=0):
    """Numpy reference of station_blocks (float64)."""
    import numpy as np
    o = _NumpyOps()
    params = params or init_params_numpy(arm, seed)
    b = {k: np.asarray(v, dtype=float) for k, v in batch.items()}
    phi_fn = None
    if arm in PHI_ARMS:
        def phi_fn(x):
            return np.maximum(x @ params["W1"] + params["b1"], 0.0) @ params["W2"] + params["b2"]
    return station_blocks(o, arm, b, np.asarray(geo, dtype=float), params, phi_fn, ws_scale)


def displacement_diagnostics_numpy(batch, geo, params=None, ws_scale=1.0):
    """Per-station shift distributions etc. for the identifiability check (R1-S3), numpy."""
    import numpy as np
    o = _NumpyOps()
    params = params or init_params_numpy("M")
    b = {k: np.asarray(v, dtype=float) for k, v in batch.items()}
    ms, mf = lead_masks(o, b["age"], b["cycle_ok"])
    w, _ = kernel_weights(o, params["kp"], np.asarray(geo, dtype=float), b["dS"], b["dS_mask"])
    adm = admissible_shifts(o, b["age"], b["cycle_ok"])
    temp = o.softplus(np.asarray(params["t_raw"])) + TEMP_FLOOR
    lam = o.sigmoid(np.asarray(params["l_raw"]))
    return displacement_block(o, b, w, ms, mf, adm, temp, lam, ws_scale, return_diagnostics=True)


# --------------------------------------------------------------------------- torch modules
if _torch is not None:

    class StationBlocks(_nn.Module):
        """Learnable station block of one arm (I_pool, I or M); output [B, block_width(arm)]."""

        def __init__(self, arm, geo, ws_scale=1.0):
            super().__init__()
            if arm not in STATION_ARMS:
                raise ValueError(f"StationBlocks is only for {STATION_ARMS}")
            self.arm, self.ws_scale = arm, float(ws_scale)
            self.kp = _nn.Parameter(_torch.tensor(KERNEL_INIT, dtype=_torch.float32))
            self.register_buffer("geo", _torch.as_tensor(geo, dtype=_torch.float32).clone())
            if arm in PHI_ARMS:
                hid = PHI_ARMS[arm]
                self.phi = _nn.Sequential(_nn.Linear(PHI_IN, hid), _nn.ReLU(), _nn.Linear(hid, PHI_OUT))
            if arm == "R_sp":
                self.beta = _nn.Parameter(_torch.zeros(RSP_FEATS, len(HORIZONS)))
            if DISP_MODES.get(arm) in ("full", "phase"):
                self.t_raw = _nn.Parameter(_torch.tensor(T_RAW_INIT, dtype=_torch.float32))
            if DISP_MODES.get(arm) in ("full", "amp"):
                self.l_raw = _nn.Parameter(_torch.tensor(L_RAW_INIT, dtype=_torch.float32))

        def forward(self, batch):
            o = _TorchOps()
            b = {k: _torch.as_tensor(v, dtype=self.kp.dtype, device=self.kp.device) for k, v in batch.items()}
            params = {"kp": self.kp}
            for name in ("t_raw", "l_raw", "beta"):
                if hasattr(self, name):
                    params[name] = getattr(self, name)
            phi_fn = self.phi if self.arm in PHI_ARMS else None
            return station_blocks(o, self.arm, b, self.geo.to(self.kp.dtype), params, phi_fn, self.ws_scale)

    class G1TSMixerModel(_nn.Module):
        """x [B,24,C_base] (B's 21 channels, plus N's NWP channels for N/I_pool/I/M; power is the LAST channel)
        + station block channels broadcast over the 24 steps and inserted before power -> TSMixerBasicSmall."""

        def __init__(self, arm, base_channels, geo=None, *, blocks=2, ff_dim=32, dropout=0.1, norm_eps=1e-3,
                     seq_len=24, pred_len=4, ws_scale=1.0):
            super().__init__()
            from models.TSMixerBasicSmall import Model as TSMixer
            if arm not in ARMS:
                raise ValueError(f"arm must be one of {ARMS}")
            self.arm, self.k, self.base_channels = arm, block_width(arm), int(base_channels)
            self.station = StationBlocks(arm, geo, ws_scale) if self.k else None
            self.register_buffer("enc_mean", _torch.zeros(self.k))
            self.register_buffer("enc_std", _torch.ones(self.k))
            self.register_buffer("normalized", _torch.tensor(self.k == 0))
            ch = self.base_channels + self.k
            self.backbone = TSMixer(seq_len=seq_len, channels=ch, pred_len=pred_len, blocks=blocks, ff_dim=ff_dim,
                                    dropout=dropout, norm_eps=norm_eps, target_slot=ch - 1, input_mode="full")

        @_torch.no_grad()
        def fit_station_normalization(self, batches):
            """Fixed per-channel mean/std of the station block at its init parameters, fit-period batches only."""
            if not self.k:
                return
            total, s1, s2 = 0, _torch.zeros(self.k, dtype=_torch.float64), _torch.zeros(self.k, dtype=_torch.float64)
            for batch in batches:
                y = self.station(batch).double()
                if not bool(_torch.isfinite(y).all()):
                    raise ValueError("non-finite station block output on fit data")
                total, s1, s2 = total + y.shape[0], s1 + y.sum(0).cpu(), s2 + (y * y).sum(0).cpu()
            if total == 0:
                raise ValueError("station normalization requires non-empty fit-period batches")
            mean = s1 / total
            std = (s2 / total - mean * mean).clamp_min(0.0).sqrt().clamp_min(1e-3)
            self.enc_mean.copy_(mean.float())
            self.enc_std.copy_(std.float())
            self.normalized.fill_(True)

        def forward(self, x, batch=None):
            if not self.k:
                return self.backbone(x)
            if not bool(self.normalized):
                raise RuntimeError("call fit_station_normalization on fit-period batches first")
            enc = (self.station(batch) - self.enc_mean) / self.enc_std
            enc = enc[:, None, :].expand(x.shape[0], x.shape[1], self.k)
            return self.backbone(_torch.cat([x[..., :-1], enc, x[..., -1:]], dim=-1))
