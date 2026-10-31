"""Torch-free helpers for v33 (proposal sections 14.13, 14.17, 14.18): station token bank, geometry, masks,
a NumPy reference of the arrival prior, kappa* selection and the exhaustive A1 decision.

Conventions
* Station offsets r_s are (east, north) km from the farm, from the screen's distance/bearing.
* A wind-FROM direction theta (deg, meteorological) gives the unit vector u = (sin theta, cos theta) pointing towards
  where the air comes from; a station is upwind when r_s . u > 0.  Signed cross distance d_perp = r_x u_y - r_y u_x.
* Content geometry is rotated by the issue-time reference direction u_ref (circular mean of raw-valid, available
  station directions within ``radius_km`` of the farm); the arrival prior uses each station's OWN raw-valid direction
  (regional circular-mean fallback from raw-valid observations only).
* Three masks: availability (token ages >= lag, inside the bank) is structural - unavailable tokens never exist;
  per-variable raw validity (fills are only for content and carry flags); physical eligibility (raw speed >= 2 m/s and a
  raw-valid or legal regional-fallback direction) is the only gate of the prior, together with the upwind sign.
* Labels are never read or modified here.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

CONTENT_FEATURES = ["ws_z", "rel_dir_sin", "rel_dir_cos", "p_anom_z", "dp3_z", "valid_ws", "valid_dir", "valid_p",
                    "valid_dp3", "d_par_ref_100km", "d_perp_ref_100km", "age_24h"]
PHYS_FEATURES = ["d_par_own_km", "d_perp_own_km", "cos_own", "vbar_ms", "age_h", "base_ok"]
HOUR = pd.Timedelta(hours=1)


# --------------------------------------------------------------------------- geometry
def station_offsets(stations):
    """(S, 2) east/north offsets in km from distance_km and bearing_deg (farm -> station)."""
    d = np.array([float(s["distance_km"]) for s in stations])
    b = np.radians([float(s["bearing_deg"]) for s in stations])
    return np.column_stack([d * np.sin(b), d * np.cos(b)])


def from_unit(theta_deg):
    """Unit vector (east, north) pointing towards where the wind comes from."""
    t = np.radians(theta_deg)
    return np.sin(t), np.cos(t)


def project(offsets, theta_deg):
    """Along (d_par) and signed cross (d_perp) components of station offsets for wind-from directions theta.
    offsets: (S, 2); theta: (..., S) or broadcastable -> arrays of the broadcast shape."""
    ux, uy = from_unit(theta_deg)
    x, y = offsets[..., 0], offsets[..., 1]
    return x * ux + y * uy, x * uy - y * ux


def circular_mean_deg(theta_deg, mask, axis=-1):
    """Circular mean over ``axis`` of valid entries; NaN where none or the resultant vanishes."""
    t = np.radians(np.where(mask, theta_deg, 0.0))
    s, c = np.where(mask, np.sin(t), 0).sum(axis), np.where(mask, np.cos(t), 0).sum(axis)
    n = mask.sum(axis)
    out = np.degrees(np.arctan2(s, c)) % 360
    return np.where((n > 0) & (np.hypot(s, c) > 1e-9), out, np.nan)


# --------------------------------------------------------------------------- bank
@dataclass
class StationBank:
    hours: pd.DatetimeIndex
    offsets: np.ndarray            # (S, 2)
    ws: np.ndarray                 # (T, S) raw m/s, NaN = missing
    direction: np.ndarray          # (T, S) raw wind-from deg, NaN = missing/calm
    pressure: np.ndarray           # (T, S) raw hPa
    dp3: np.ndarray                # (T, S) raw 3 h pressure change
    content_ws: np.ndarray         # standardized, invalid -> 0 (fit mean)
    content_p: np.ndarray
    content_dp3: np.ndarray
    ref_dir: np.ndarray            # (T,) reference wind-from deg (never NaN)
    ref_flag: np.ndarray           # (T,) 0 = within radius, 1 = all-station fallback, 2 = last resort (north)
    prior_dir: np.ndarray          # (T, S) own raw-valid direction or regional raw-valid fallback; NaN = none
    vbar: np.ndarray               # (T,) upwind mean raw-valid speed (regional raw-valid fallback); NaN = none
    stats: dict

    def index_of(self, times):
        """Bank row of each timestamp (naive UTC on the hour); -1 when outside the bank."""
        times = pd.DatetimeIndex(times)
        idx = ((times - self.hours[0]) / HOUR).to_numpy()
        ok = np.isfinite(idx) & (idx >= 0) & (idx < len(self.hours)) & (np.round(idx) == idx)
        return np.where(ok, np.nan_to_num(idx, nan=-1).astype(np.int64), -1)


def build_bank(observations, stations, hours, fit_hours_mask, *, radius_km=50.0):
    """Token bank on an hourly grid. ``observations``: {src_id: frame(wind_speed, wind_direction, msl_pressure)};
    ``fit_hours_mask``: bool over ``hours`` selecting the fold's fitting period (scaler statistics only)."""
    hours = pd.DatetimeIndex(hours)
    if hours.has_duplicates or not hours.is_monotonic_increasing or (hours[1:] - hours[:-1] != HOUR).any():
        raise ValueError("The bank grid must be strictly hourly")
    fit_hours_mask = np.asarray(fit_hours_mask, bool)
    if fit_hours_mask.shape != (len(hours),) or not fit_hours_mask.any():
        raise ValueError("Fit-hour mask must select part of the grid")

    def matrix(name):
        return np.column_stack([observations[int(s["src_id"])][name].reindex(hours).to_numpy(dtype=float)
                                if int(s["src_id"]) in observations else np.full(len(hours), np.nan) for s in stations])
    ws, direction, pressure = matrix("wind_speed"), matrix("wind_direction"), matrix("msl_pressure")
    ws = np.where(np.isfinite(ws) & (ws >= 0), ws, np.nan)
    direction = np.where(np.isfinite(direction) & (direction >= 0) & (direction <= 360) & (ws > 0), direction % 360, np.nan)
    pressure = np.where(np.isfinite(pressure) & (pressure > 0), pressure, np.nan)
    dp3 = np.full_like(pressure, np.nan)
    dp3[3:] = pressure[3:] - pressure[:-3]
    offsets = station_offsets(stations)

    fit = fit_hours_mask[:, None]
    ws_fit = ws[fit[:, 0]]
    ws_mean, ws_std = float(np.nanmean(ws_fit)), float(np.nanstd(ws_fit)) or 1.0
    p_station_mean = np.array([np.nanmean(col) if np.isfinite(col).any() else np.nan for col in pressure[fit[:, 0]].T])
    p_anom = pressure - p_station_mean[None, :]
    p_std = float(np.nanstd(p_anom[fit[:, 0]])) or 1.0
    dp3_std = float(np.nanstd(dp3[fit[:, 0]])) or 1.0
    content_ws = np.where(np.isfinite(ws), (ws - ws_mean) / ws_std, 0.0)
    content_p = np.where(np.isfinite(p_anom), p_anom / p_std, 0.0)
    content_dp3 = np.where(np.isfinite(dp3), dp3 / dp3_std, 0.0)

    distance = np.hypot(offsets[:, 0], offsets[:, 1])
    near = distance <= radius_km
    valid_dir = np.isfinite(direction)
    ref = circular_mean_deg(direction[:, near], valid_dir[:, near]) if near.any() else np.full(len(hours), np.nan)
    flag = np.where(np.isfinite(ref), 0, 1)
    allref = circular_mean_deg(direction, valid_dir)
    ref = np.where(np.isfinite(ref), ref, allref)
    flag = np.where(np.isfinite(ref), flag, 2)
    ref = np.where(np.isfinite(ref), ref, 0.0)

    regional = allref                                                  # raw-valid only
    prior_dir = np.where(valid_dir, direction, regional[:, None])      # NaN where neither exists
    d_par, _ = project(offsets[None, :, :], np.nan_to_num(prior_dir, nan=0.0))
    upwind = np.isfinite(prior_dir) & (d_par > 0) & np.isfinite(ws)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN hours -> NaN (handled below)
        up_mean = np.nanmean(np.where(upwind, ws, np.nan), axis=1)
        reg_mean = np.nanmean(ws, axis=1)
    vbar = np.where(np.isfinite(up_mean), up_mean, reg_mean)
    stats = {"ws_mean": ws_mean, "ws_std": ws_std, "p_station_mean": p_station_mean.tolist(), "p_anom_std": p_std,
             "dp3_std": dp3_std, "fit_hours": int(fit_hours_mask.sum()), "radius_km": radius_km,
             "reference_flag_share": {str(k): float((flag == k).mean()) for k in (0, 1, 2)}}
    return StationBank(hours, offsets, ws, direction, pressure, dp3, content_ws, content_p, content_dp3, ref, flag,
                       prior_dir, vbar, stats)


def token_rows(bank, issue_times, lag, n_ages=24):
    """(N, A) bank rows of the tokens of each issue time (ages lag .. lag+n_ages-1) and (N,) row of the reference
    (issue - lag). Rows outside the bank are -1 (structurally unavailable)."""
    issue = pd.DatetimeIndex(issue_times)
    ages = np.arange(lag, lag + n_ages)
    base = bank.index_of(issue)
    rows = np.where(base[:, None] >= 0, base[:, None] - ages[None, :], -1)
    rows = np.where(rows >= 0, rows, -1)
    ref_rows = np.where(base - lag >= 0, base - lag, -1)
    return rows, ref_rows, ages


def gather(bank, rows, ref_rows, ages):
    """content (N, A*S, 12), phys (N, A*S, 6), valid (N, A*S) for token rows from ``token_rows``.
    Token order: age-major, then station (the model is permutation invariant over tokens)."""
    n, a = rows.shape
    s = bank.offsets.shape[0]
    ok_row = rows >= 0
    r = np.where(ok_row, rows, 0)
    ws, dirn, pres, dp3 = bank.ws[r], bank.direction[r], bank.pressure[r], bank.dp3[r]          # (N, A, S)
    v_ws, v_dir, v_p, v_dp3 = (np.isfinite(ws) & ok_row[..., None], np.isfinite(dirn) & ok_row[..., None],
                               np.isfinite(pres) & ok_row[..., None], np.isfinite(dp3) & ok_row[..., None])
    ref = bank.ref_dir[np.where(ref_rows >= 0, ref_rows, 0)]                                    # (N,)
    rel = np.radians(np.where(v_dir, dirn - ref[:, None, None], 0.0))
    d_par_ref, d_perp_ref = project(bank.offsets[None, None, :, :], np.broadcast_to(ref[:, None, None], (n, a, s)))
    content = np.stack([np.where(v_ws, bank.content_ws[r], 0.0), np.where(v_dir, np.sin(rel), 0.0),
                        np.where(v_dir, np.cos(rel), 0.0), np.where(v_p, bank.content_p[r], 0.0),
                        np.where(v_dp3, bank.content_dp3[r], 0.0), v_ws, v_dir, v_p, v_dp3,
                        d_par_ref / 100.0, d_perp_ref / 100.0,
                        np.broadcast_to(ages[None, :, None] / 24.0, (n, a, s))], axis=-1).astype(np.float32)
    pdir = bank.prior_dir[r]
    d_par_own, d_perp_own = project(bank.offsets[None, None, :, :], np.nan_to_num(pdir, nan=0.0))
    dist = np.hypot(bank.offsets[:, 0], bank.offsets[:, 1])[None, None, :]
    cos_own = np.where(dist > 0, d_par_own / np.maximum(dist, 1e-9), 0.0)
    vbar = bank.vbar[r]                                                                          # (N, A)
    base_ok = v_ws & (np.nan_to_num(ws, nan=0.0) >= 2.0) & np.isfinite(pdir) & np.isfinite(vbar)[..., None]
    phys = np.stack([d_par_own, d_perp_own, cos_own, np.broadcast_to(np.nan_to_num(vbar, nan=0.0)[..., None], (n, a, s)),
                     np.broadcast_to(ages[None, :, None].astype(float), (n, a, s)), base_ok], axis=-1).astype(np.float32)
    valid = (v_ws | v_dir | v_p | v_dp3)
    return (content.reshape(n, a * s, -1), phys.reshape(n, a * s, -1), valid.reshape(n, a * s))


# --------------------------------------------------------------------------- arrival prior (NumPy reference)
def _ndtr(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, float) / math.sqrt(2.0)))


def prior_numpy(phys, valid, kappa, *, variant="full", min_speed=2.0, speed_floor=2.0, cross_scale=30.0,
                sigma_min=1.0, sigma_frac=0.5, eps=1e-6, horizon=4):
    """q (N, K, H), support (N, H) and q_tilde (N, K, H) exactly as specified (sections 14.13/14.18).
    variant: full | age0 | hflat | geo180. The torch module must match this to 1e-5."""
    d_par, d_perp, cos_own, vbar, age, base_ok = [phys[..., i] for i in range(6)]
    if variant == "geo180":
        d_par, d_perp, cos_own = -d_par, -d_perp, -cos_own
    eligible = (base_ok > 0.5) & (d_par > 0) & valid
    t_adv = np.where(eligible, d_par / (3.6 * kappa * np.maximum(vbar, speed_floor)), 0.0)
    a = np.zeros_like(age) if variant == "age0" else age
    r = t_adv - a
    sigma = np.maximum(sigma_min, sigma_frac * t_adv)
    geo = np.maximum(cos_own, 0.0) * np.exp(-0.5 * (d_perp / cross_scale) ** 2)
    q = np.zeros(phys.shape[:-1] + (horizon,))
    centre = np.zeros_like(q, dtype=bool)
    for h in range(1, horizon + 1):
        lo, hi = (0.0, float(horizon)) if variant == "hflat" else (h - 1.0, float(h))
        mass = _ndtr((hi - r) / sigma) - _ndtr((lo - r) / sigma)
        if variant == "hflat":
            mass = mass / horizon
        q[..., h - 1] = np.where(eligible, mass * geo, 0.0)
        centre[..., h - 1] = eligible & (r >= lo) & (r <= hi)
    support = (q.sum(axis=1) >= eps) & centre.any(axis=1)
    n_valid = np.maximum(valid.sum(axis=1), 1)[:, None]
    mean_q = q.sum(axis=1) / n_valid
    q_tilde = np.where(support[:, None, :], q / np.where(mean_q > 0, mean_q, 1.0)[:, None, :], 1.0)
    return q, support, q_tilde


# --------------------------------------------------------------------------- selection and decision
def j_score(mae, rmse, mae_ref, rmse_ref):
    return float(mae / mae_ref + rmse / rmse_ref)


def kappa_star(scores, tolerance=1e-4):
    """scores: {kappa: J}. Minimum J; ties within tolerance -> closest to 1, then the smaller kappa."""
    if not scores or any(not np.isfinite(v) for v in scores.values()):
        raise ValueError("kappa grid incomplete or non-finite")
    best = min(scores.values())
    tied = [k for k, v in scores.items() if v - best < tolerance]
    return sorted(tied, key=lambda k: (abs(k - 1.0), k))[0]


def a1_decision(metrics, cfg):
    """Exhaustive, mutually exclusive A1 decision (section 14.13). metrics: {arm: {"mae":..., "rmse":...}} on the
    development-evaluation year, ALL cohort, kW. Returns {"branch": D0..D3, ...}."""
    need = ("B0", "B2", "A")
    if any(arm not in metrics for arm in need) or any(
            not np.isfinite(metrics[arm].get(key, np.nan)) for arm in need for key in ("mae", "rmse")):
        return {"branch": "D0", "reason": "missing arm or non-finite metric (technical failure)"}

    def dr(x, y):
        return 100.0 * (metrics[y]["rmse"] - metrics[x]["rmse"]) / metrics[y]["rmse"]

    def dm(x, y):
        return metrics[y]["mae"] - metrics[x]["mae"]
    d1, d2 = cfg["D1_mechanism"], cfg["D2_fusion"]
    numbers = {"dR_A_B2_pct": dr("A", "B2"), "dM_A_B2_kw": dm("A", "B2"), "dR_A_B0_pct": dr("A", "B0"),
               "dM_A_B0_kw": dm("A", "B0"), "dR_B2_B0_pct": dr("B2", "B0"), "dM_B2_B0_kw": dm("B2", "B0")}
    if (numbers["dR_A_B2_pct"] >= d1["rmse_gain_vs_B2_pct_min"] and numbers["dM_A_B2_kw"] >= d1["mae_gain_vs_B2_kw_min"]
            and numbers["dR_A_B0_pct"] >= d1["rmse_gain_vs_B0_pct_min"] and numbers["dM_A_B0_kw"] >= d1["mae_gain_vs_B0_kw_min"]):
        branch = "D1"
    elif numbers["dR_B2_B0_pct"] >= d2["rmse_gain_B2_vs_B0_pct_min"] and numbers["dM_B2_B0_kw"] >= d2["mae_gain_B2_vs_B0_kw_min"]:
        branch = "D2"
    else:
        branch = "D3"
    return {"branch": branch, **numbers}
