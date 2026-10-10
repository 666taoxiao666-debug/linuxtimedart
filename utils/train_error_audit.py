"""Read-only, history-conditioned error attribution for train-period OOF data."""
from __future__ import annotations

import numpy as np


EVENT_IDS = ("gust_or_turbulent", "high_wind_low_power", "rated_saturation", "low_wind_idle")


def observable_groups(wind, power, *, rated_power, thresholds):
    """Hard diagnostic membership, not the Wiki's soft activation or a new gate.

    Both inputs contain history only, in original physical units. Thresholds
    come from the existing frozen physical-event config, never from targets.
    """
    wind, power = np.asarray(wind, float), np.asarray(power, float)
    recent = int(thresholds["recent_steps"])
    if (wind.ndim != 2 or power.shape != wind.shape or wind.shape[1] < recent
            or recent < 2 or not np.isfinite(wind).all() or not np.isfinite(power).all()
            or not np.isfinite(rated_power) or rated_power <= 0):
        raise ValueError("Finite paired physical histories and positive capacity required")
    w, p = wind[:, -recent:], power[:, -recent:] / rated_power
    gust = ((w.std(axis=1) > thresholds["gust_std_min_mps"])
            | (np.abs(np.diff(w, axis=1)).max(axis=1) > thresholds["gust_step_min_mps"]))
    low_power = p.mean(axis=1) < thresholds["low_power_max_ratio"]
    masks = np.column_stack((gust,
        (w.mean(axis=1) > thresholds["active_wind_min_mps"]) & low_power,
        p.mean(axis=1) > thresholds["rated_power_min_ratio"],
        (w.mean(axis=1) < thresholds["low_wind_max_mps"]) & low_power))
    codes = (masks * (1 << np.arange(len(EVENT_IDS)))).sum(axis=1).astype(np.int16)
    # A fixed 5%-capacity history change is a diagnostic bin, not the model's
    # learned trend label. Compare consecutive halves of the same history.
    half = recent // 2
    delta = power[:, -half:].mean(axis=1) - power[:, -recent:-half].mean(axis=1)
    trend = np.where(delta > .05 * rated_power, 1,
                     np.where(delta < -.05 * rated_power, -1, 0)).astype(np.int8)
    return masks, codes, trend


def verify_audit_sources(manifests, plan):
    """Reject mismatched checkpoints or evidence that touches outer validation."""
    from utils.accuracy_tuning import verify_inner_manifest
    if (plan.get("sdwpf_fold"), plan.get("seed"), plan.get("pred_len")) != (1, 2024, 12):
        raise ValueError("Only the already frozen f1 seed2024 h12 diagnostic is supported")
    if len(manifests) != 2:
        raise ValueError("Exactly the frozen reference and v6 checkpoint are required")
    boundaries = []
    for manifest in manifests:
        verify_inner_manifest(manifest, plan, "finetune")
        args = manifest["args"]
        if (int(args["seq_len"]) != 336 or int(args["pred_len"]) != int(plan["pred_len"])
                or args.get("features") != "MS" or args.get("ramp_residual")
                or float(args.get("rated_power", 0)) != 1500.
                or int(manifest["extra"]["best_epoch"]) != 8):
            raise ValueError("Not the fixed-epoch plain-trend diagnostic source")
        boundaries.append(tuple(manifest["datasets"]["train"][key]
                                for key in ("train_cutoff", "val_cutoff")))
    if boundaries[0] != boundaries[1]:
        raise ValueError("Sources have different fit/selection boundaries")
    # Only the optional normalization contract may differ in the forecast data
    # path. Scalers, feature order, cleanup and physical labels must be equal.
    keys = ("feature_columns", "rated_power", "sdwpf_clip_power", "sdwpf_filter_abnormal",
            "sdwpf_causal_fill", "sdwpf_robust_pitch", "sdwpf_keep_curtailment",
            "sdwpf_physics_features", "sdwpf_drop_weak_features", "sdwpf_eval_stride")
    for key in keys:
        if manifests[0]["args"].get(key) != manifests[1]["args"].get(key):
            raise ValueError(f"Sources differ in data convention: {key}")
    for key in ("mean", "scale"):
        np.testing.assert_allclose(manifests[0]["datasets"]["train"]["scaler"][key],
            manifests[1]["datasets"]["train"]["scaler"][key], rtol=0, atol=0)
    if (manifests[0]["args"].get("consistent_physics_norm", False)
            or not manifests[1]["args"].get("consistent_physics_norm", False)):
        raise ValueError("Sources must be v5 reference then matched-normalization v6")
    return boundaries[0]


def error_report(predictions, truth, persistence, turbines, event_masks, codes, trend, *, rated_power):
    """Attribute errors without treating overlapping events as additive shares.

    Windows from one turbine/time are correlated; these are descriptive counts,
    not independent replications or statistical significance tests.
    """
    truth, persistence = np.asarray(truth, float), np.asarray(persistence, float)
    turbines, event_masks = np.asarray(turbines), np.asarray(event_masks, bool)
    codes, trend = np.asarray(codes), np.asarray(trend)
    if (not np.isfinite(rated_power) or rated_power <= 0 or truth.ndim != 2 or len(truth) == 0 or truth.shape[1] < 1 or persistence.shape != truth.shape
            or turbines.shape != (len(truth),) or codes.shape != turbines.shape
            or trend.shape != turbines.shape or event_masks.shape != (len(truth), len(EVENT_IDS))):
        raise ValueError("Forecasts and history metadata do not align")
    if not np.isin(trend, [-1, 0, 1]).all():
        raise ValueError("History trend must use the three frozen diagnostic bins")
    if not np.array_equal(codes, (event_masks * (1 << np.arange(len(EVENT_IDS)))).sum(axis=1)):
        raise ValueError("Disjoint event codes do not match overlapping masks")
    arrays = {name: np.asarray(value, float) for name, value in predictions.items()}
    if "reference" not in arrays or "physics_norm_v6" not in arrays or "persistence" in arrays:
        raise ValueError("Both frozen models are required; persistence is reserved")
    arrays["persistence"] = persistence
    if any(value.shape != truth.shape or not np.isfinite(value).all() for value in arrays.values()) or not np.isfinite(truth).all():
        raise ValueError("All paired predictions and labels must be finite and share a shape")
    errors = {name: value - truth for name, value in arrays.items()}
    totals = {name: (np.abs(error).sum(), np.square(error).sum()) for name, error in errors.items()}
    full_count = truth.size

    def row(window_mask, step=None):
        selected = np.flatnonzero(window_mask)
        y = truth[selected] if step is None else truth[selected, step]
        pe = errors["persistence"][selected] if step is None else errors["persistence"][selected, step]
        re = errors["reference"][selected] if step is None else errors["reference"][selected, step]
        persistence_mae, persistence_rmse = np.abs(pe).mean(), np.sqrt(np.square(pe).mean())
        sst = np.square(y - y.mean()).sum()
        result = {"window_count": int(len(selected)), "point_count": int(y.size),
                  "point_coverage_pct": float(100 * y.size / full_count), "models": {}}
        for name, error in errors.items():
            e = error[selected] if step is None else error[selected, step]
            absolute, squared = np.abs(e), np.square(e)
            mae, rmse = absolute.mean(), np.sqrt(squared.mean())
            sae, sse = totals[name]
            result["models"][name] = {
                "mae_kw": float(mae), "rmse_kw": float(rmse),
                "r2": float(1 - squared.sum() / sst) if sst > 0 else None,
                "bias_kw": float(e.mean()),
                "mae_skill_pct": float(100 * (1 - mae / persistence_mae)) if persistence_mae > 0 else None,
                "rmse_skill_pct": float(100 * (1 - rmse / persistence_rmse)) if persistence_rmse > 0 else None,
                "absolute_error_share_pct": float(100 * absolute.sum() / sae) if sae > 0 else 0.,
                "squared_error_share_pct": float(100 * squared.sum() / sse) if sse > 0 else 0.,
                "gain_vs_reference_kw": float(np.abs(re).mean() - mae),
                "harm_vs_reference_point_pct": float(100 * (absolute > np.abs(re)).mean()),
                "harm_vs_persistence_point_pct": float(100 * (absolute > np.abs(pe)).mean()),
                "within_5pct_capacity_pct": float(100 * (absolute <= .05 * rated_power).mean()),
                "within_10pct_capacity_pct": float(100 * (absolute <= .1 * rated_power).mean()),
                "above_20pct_capacity_error_pct": float(100 * (absolute > .2 * rated_power).mean()),
                "above_20pct_capacity_sse_share_pct": float(100 * squared[absolute > .2 * rated_power].sum() / sse) if sse > 0 else 0.,
            }
        return result

    all_windows = np.ones(len(truth), bool)
    report = {"schema_version": 1, "interpretation": {
        "event_masks": "hard history-only diagnostic membership, NOT Wiki activation/coverage",
        "overlap": "event and event_horizon groups overlap; only event_combinations partition total error",
        "sampling": "descriptive window-point counts, correlated over turbine/time; no significance claim",
        "scope": "forward OOF inside original outer train; no outer validation or sealed-test predictions",
        "historical_trend": "consecutive history half-means differ by more than 5% nameplate capacity; not learned prompt label"},
        "overall": row(all_windows), "horizon": [], "events": [], "event_horizon": [],
        "event_combinations": [], "turbines": [], "turbine_horizon": [], "historical_trend": []}
    for step in range(truth.shape[1]):
        report["horizon"].append(dict(step=step + 1, minutes=10 * (step + 1), **row(all_windows, step)))
    for index, name in enumerate(EVENT_IDS):
        mask = event_masks[:, index]
        if not mask.any():
            continue
        report["events"].append(dict(event=name, **row(mask)))
        for step in range(truth.shape[1]):
            report["event_horizon"].append(dict(event=name, step=step + 1, **row(mask, step)))
    for code in np.unique(codes):
        names = [name for index, name in enumerate(EVENT_IDS) if int(code) & (1 << index)]
        report["event_combinations"].append(dict(code=int(code), events=names,
                                               label="+".join(names) or "no_frozen_event", **row(codes == code)))
    for turbine in np.unique(turbines):
        mask = turbines == turbine
        report["turbines"].append(dict(turbine=str(turbine), **row(mask)))
        for step in range(truth.shape[1]):
            report["turbine_horizon"].append(dict(turbine=str(turbine), step=step + 1, **row(mask, step)))
    for code, name in ((-1, "down"), (0, "stable"), (1, "up")):
        if (trend == code).any():
            report["historical_trend"].append(dict(trend=name, **row(trend == code)))
    return report
