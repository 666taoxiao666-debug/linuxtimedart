"""History-only physical strata and descriptive train-period utility drift."""
from __future__ import annotations

import numpy as np

POWER_EDGES = (.08, .3, .7, .9)
WIND_EDGES = (3., 5., 10.)
POWER_LABELS = ("below_8pct", "8_to_30pct", "30_to_70pct", "70_to_90pct", "at_least_90pct")
WIND_LABELS = ("below_3mps", "3_to_5mps", "5_to_10mps", "at_least_10mps")


def history_context(wind, power, *, rated_power=1500.):
    """Only observed last-12 histories enter strata; no forecast or target input."""
    wind, power = np.asarray(wind, float), np.asarray(power, float)
    if (wind.ndim != 2 or wind.shape != power.shape or wind.shape[1] != 12
            or not np.isfinite(wind).all() or not np.isfinite(power).all()
            or not np.isfinite(rated_power) or rated_power <= 0):
        raise ValueError("Finite paired last-12 physical histories required")
    ratio, speed = power.mean(axis=1) / rated_power, wind.mean(axis=1)
    pbin = np.digitize(ratio, POWER_EDGES).astype(np.int8)
    wbin = np.digitize(speed, WIND_EDGES).astype(np.int8)
    return dict(power_ratio=ratio, wind_mean_mps=speed, wind_std_mps=wind.std(axis=1),
                power_change_kw=power[:, 6:].mean(axis=1) - power[:, :6].mean(axis=1),
                wind_change_mps=wind[:, 6:].mean(axis=1) - wind[:, :6].mean(axis=1),
                power_bin=pbin, wind_bin=wbin, joint_bin=pbin * len(WIND_LABELS) + wbin)


def decompose_gain(gain, strata, early, late, *, bins=20):
    """Exact symmetric mix/conditional decomposition, including unshared strata.

    Positive gain means reduced error. This is descriptive, not a causal effect
    or a significance test. Missing support is reported rather than imputed.
    """
    gain, strata = np.asarray(gain, float), np.asarray(strata)
    early, late = np.asarray(early, bool), np.asarray(late, bool)
    if (gain.ndim != 1 or any(v.shape != gain.shape for v in (strata, early, late))
            or not np.isfinite(gain).all() or not early.any() or not late.any()
            or (early & late).any() or not np.isin(strata, np.arange(bins)).all()):
        raise ValueError("Two disjoint supported blocks and aligned finite gains required")
    composition, conditional, unshared = 0., 0., 0.
    rows = []
    ne, nl = int(early.sum()), int(late.sum())
    for group in range(bins):
        e, l = early & (strata == group), late & (strata == group)
        ce, cl = int(e.sum()), int(l.sum())
        if not (ce or cl):
            continue
        we, wl = ce / ne, cl / nl
        ge, gl = float(gain[e].mean()) if ce else None, float(gain[l].mean()) if cl else None
        mix = within = absent = 0.
        if ce and cl:
            mix = (wl - we) * (gl + ge) / 2
            within = (gl - ge) * (wl + we) / 2
        else:
            absent = wl * gl if cl else -we * ge
        composition += mix
        conditional += within
        unshared += absent
        rows.append(dict(bin=group, early_windows=ce, late_windows=cl,
            early_weight=we, late_weight=wl, early_gain=ge, late_gain=gl,
            both_blocks_min64=ce >= 64 and cl >= 64,
            composition_contribution=mix, conditional_contribution=within,
            unshared_support_contribution=absent))
    delta = float(gain[late].mean() - gain[early].mean())
    np.testing.assert_allclose(composition + conditional + unshared, delta, rtol=1e-10, atol=1e-8)
    return dict(early_gain=float(gain[early].mean()), late_gain=float(gain[late].mean()),
        late_minus_early=delta, composition_shift=composition,
        conditional_utility_shift=conditional, unshared_support_shift=unshared, strata=rows)


def drift_report(model, arrays, context, early, late):
    """Inspect the existing five changed cells; never fit or search new scales."""
    from utils.residual_scale_calibration import apply_scale
    ref, anchor, target, state = (np.asarray(arrays[k]) for k in
                                  ("prediction", "persistence", "truth", "trend"))
    calibrated = apply_scale(model, ref, anchor, state)
    error, cal_error = ref - target, calibrated - target
    gain = np.abs(error) - np.abs(cal_error)
    squared_gain = error ** 2 - cal_error ** 2
    context = {k: np.asarray(v) for k, v in context.items()}
    if any(v.shape != state.shape for v in context.values()):
        raise ValueError("History context differs from forecast windows")
    report = dict(schema_version=1, fitting_performed=False, new_parameters_selected=False,
        outer_validation_used=False, sealed_test_evaluated=False,
        interpretation="descriptive exploratory train-OOF; not independent windows or causal attribution",
        units=dict(absolute_gain="kW", squared_gain="kW^2"), blocks={}, changed_cells=[])
    for name, mask in (("early", early), ("late", late)):
        report["blocks"][name] = dict(windows=int(mask.sum()),
            reference_mae_kw=float(np.abs(error[mask]).mean()),
            calibrated_mae_kw=float(np.abs(cal_error[mask]).mean()),
            gain_kw=float(gain[mask].mean()),
            reference_rmse_kw=float(np.sqrt(np.square(error[mask]).mean())),
            calibrated_rmse_kw=float(np.sqrt(np.square(cal_error[mask]).mean())))
    for row in model["states"]:
        for step, weight in enumerate(row["weights"]):
            if weight == 1:
                continue
            mask = state == row["state"]
            e, l = np.asarray(early, bool) & mask, np.asarray(late, bool) & mask
            if not e.any() or not l.any():
                report["changed_cells"].append(dict(state=row["state"], step=step + 1,
                    scale=weight, insufficient_state_support=True))
                continue
            absolute = decompose_gain(gain[:, step], context["joint_bin"], e, l)
            squared = decompose_gain(squared_gain[:, step], context["joint_bin"], e, l)
            for item in absolute["strata"]:
                index = item["bin"]
                item.update(power_label=POWER_LABELS[index // len(WIND_LABELS)],
                            wind_label=WIND_LABELS[index % len(WIND_LABELS)])
                for block, selected in (("early", e), ("late", l)):
                    selected = selected & (context["joint_bin"] == index)
                    if selected.any():
                        item[block + "_physics"] = dict(
                            wind_mean_mps=float(context["wind_mean_mps"][selected].mean()),
                            power_ratio=float(context["power_ratio"][selected].mean()),
                            reference_bias_kw=float(error[selected, step].mean()),
                            predicted_change_kw=float((ref - anchor)[selected, step].mean()),
                            realized_change_kw=float((target - anchor)[selected, step].mean()),
                            reference_gain_vs_persistence_kw=float((np.abs(anchor - target)
                                - np.abs(error))[selected, step].mean()),
                            calibration_harm_pct=float(100 * (gain[selected, step] < 0).mean()))
            report["changed_cells"].append(dict(state=row["state"], step=step + 1,
                scale=weight, absolute_error=absolute, squared_error=squared))
    return report
