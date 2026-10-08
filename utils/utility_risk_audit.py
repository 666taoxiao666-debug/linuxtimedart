"""Train-only calibration audit for the factorized Wiki harm-risk head.

The audit deliberately reuses only the chronological gate partition carved
from the original training split.  It never opens the outer validation or
sealed test labels and never changes a checkpoint or decision threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from utils.experiment_audit import checkpoint_info
from utils.utility_calibration import split_utility_training


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _safe_auc(target, probability, kind):
    target = np.asarray(target, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    if target.size == 0 or np.unique(target).size < 2:
        return None
    metric = roc_auc_score if kind == "roc" else average_precision_score
    return _finite_or_none(metric(target, probability))


def summarize_risk_head(probability, gain_kw, mask, threshold=0.5,
                        large_harm_quantile=0.9):
    """Summarize a fixed risk rule on train-only candidate outcomes.

    ``gain_kw`` is base absolute error minus candidate absolute error, so
    positive values favor intervention and negative values are harmful.
    """
    probability = np.asarray(probability, dtype=np.float64)
    gain_kw = np.asarray(gain_kw, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if probability.shape != gain_kw.shape or mask.shape != gain_kw.shape:
        raise ValueError("Risk probability, gain and mask must have identical shapes")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("Risk threshold must lie in (0,1)")
    if not 0.0 < float(large_harm_quantile) < 1.0:
        raise ValueError("Large-harm quantile must lie in (0,1)")

    p, gain = probability[mask], gain_kw[mask]
    if not np.isfinite(p).all() or not np.isfinite(gain).all():
        raise ValueError("Risk audit inputs must be finite")
    if p.size == 0:
        return {
            "samples": 0, "harm_prevalence_pct": None, "brier": None,
            "roc_auc": None, "pr_auc": None, "accepted_pct": None,
            "vetoed_pct": None, "harm_recall_pct": None,
            "large_harm_cutoff_kw": None, "large_harm_recall_pct": None,
            "accepted_gain_kw": None, "vetoed_gain_kw": None,
            "net_gain_before_veto_kw": None, "net_gain_after_veto_kw": None,
            "veto_value_kw_per_candidate": None,
        }

    harmful = gain < 0.0
    accepted = p < float(threshold)
    vetoed = ~accepted
    magnitude = np.maximum(-gain, 0.0)
    positive_harm = magnitude[harmful]
    large_cutoff = (
        float(np.quantile(positive_harm, large_harm_quantile))
        if positive_harm.size else None
    )
    large_harm = harmful & (magnitude >= large_cutoff) if large_cutoff is not None else harmful

    def masked_mean(values, selected):
        return float(np.mean(values[selected])) if selected.any() else None

    return {
        "samples": int(p.size),
        "harm_prevalence_pct": float(100.0 * harmful.mean()),
        "brier": float(np.mean((p - harmful.astype(np.float64)) ** 2)),
        "roc_auc": _safe_auc(harmful, p, "roc"),
        "pr_auc": _safe_auc(harmful, p, "pr"),
        "accepted_pct": float(100.0 * accepted.mean()),
        "vetoed_pct": float(100.0 * vetoed.mean()),
        "harm_recall_pct": (
            float(100.0 * (vetoed & harmful).sum() / harmful.sum())
            if harmful.any() else None
        ),
        "large_harm_cutoff_kw": large_cutoff,
        "large_harm_recall_pct": (
            float(100.0 * (vetoed & large_harm).sum() / large_harm.sum())
            if large_harm.any() else None
        ),
        "accepted_gain_kw": masked_mean(gain, accepted),
        "vetoed_gain_kw": masked_mean(gain, vetoed),
        "net_gain_before_veto_kw": float(gain.mean()),
        # Abstention contributes zero gain; denominator remains all eligible
        # candidates so before/after values are directly comparable.
        "net_gain_after_veto_kw": float(np.where(accepted, gain, 0.0).mean()),
        "veto_value_kw_per_candidate": float(np.where(vetoed, -gain, 0.0).mean()),
    }


def calibration_rows(probability, gain_kw, mask, bins=10, **identity):
    probability = np.asarray(probability, dtype=np.float64)
    gain_kw = np.asarray(gain_kw, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = mask & (probability >= lower)
        selected &= probability <= upper if index == len(edges) - 2 else probability < upper
        count = int(selected.sum())
        gain = gain_kw[selected]
        rows.append({
            **identity,
            "bin": index,
            "lower": float(lower),
            "upper": float(upper),
            "samples": count,
            "mean_probability": float(probability[selected].mean()) if count else None,
            "harm_rate_pct": float(100.0 * np.mean(gain < 0.0)) if count else None,
            "mean_gain_kw": float(gain.mean()) if count else None,
            "mean_harm_magnitude_kw": (
                float(np.maximum(-gain, 0.0).mean()) if count else None
            ),
        })
    return rows


def _target_scale(dataset):
    if not getattr(dataset, "scale", True):
        return 1.0
    scaler = getattr(dataset, "scaler", None)
    columns = list(getattr(dataset, "feature_columns", []))
    target = getattr(dataset, "target", None)
    index = columns.index(target) if target in columns else -1
    if scaler is not None and hasattr(scaler, "scale_"):
        return float(np.asarray(scaler.scale_).reshape(-1)[index])
    if scaler is not None and hasattr(scaler, "std"):
        return float(np.asarray(scaler.std).reshape(-1)[index])
    raise AttributeError("Training dataset has no usable target scale")


def run_utility_risk_audit(exp):
    args = exp.args
    output = Path(args.report_output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not getattr(args, "utility_harm_veto", False):
        raise ValueError("Risk audit requires a harm-veto checkpoint")
    if not getattr(args, "utility_factorized", False):
        raise ValueError("Risk audit requires the factorized Wiki")

    train_data, _ = exp._get_data(flag="train")
    _, gate_data, split_audit = split_utility_training(
        train_data, args.utility_calibration_fraction
    )
    loader = DataLoader(
        gate_data, batch_size=args.eval_batch_size, shuffle=False,
        drop_last=False, num_workers=0,
    )
    model = exp.model.module if isinstance(exp.model, torch.nn.DataParallel) else exp.model
    model.eval()
    parts = {name: [] for name in (
        "probability", "physical", "utility", "action", "gain"
    )}
    with torch.no_grad():
        for index, (x, y, _, _) in enumerate(loader):
            x = x.float().to(exp.device)
            y = y.float().to(exp.device)[:, -args.pred_len:, -1:]
            with exp._autocast():
                model(x)
            aux = getattr(model, "_last_utility_aux", None)
            required = {
                "base_prediction", "factor_predictions", "factor_utilities",
                "factor_physical_availability", "factor_action",
                "factor_harm_probability",
            }
            if not isinstance(aux, dict) or not required.issubset(aux):
                raise RuntimeError("Harm-veto checkpoint did not expose audited tensors")
            base_error = (aux["base_prediction"] - y).abs().mean(2)
            candidate_error = (aux["factor_predictions"] - y[..., None]).abs().mean(2)
            parts["gain"].append((base_error[..., None] - candidate_error).cpu().numpy())
            parts["probability"].append(aux["factor_harm_probability"].cpu().numpy())
            parts["physical"].append(aux["factor_physical_availability"].cpu().numpy())
            parts["utility"].append(aux["factor_utilities"].cpu().numpy())
            parts["action"].append(aux["factor_action"].cpu().numpy())
            if index % 100 == 0:
                print(f"[RISK-AUDIT] Train gate batch {index + 1}/{len(loader)}", flush=True)

    arrays = {name: np.concatenate(values) for name, values in parts.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Non-finite train-only risk audit tensors")
    arrays["physical"] = arrays["physical"].astype(bool)
    scale_kw = _target_scale(gate_data)
    gain_kw = arrays["gain"] * scale_kw
    gain_eligible = arrays["physical"] & (
        arrays["utility"] > float(args.utility_min_gain)
    )
    threshold = float(args.utility_harm_threshold)
    accepted = gain_eligible & (arrays["probability"] < threshold)

    factor_ids = [*model.scene_wiki_scene_ids, "composition"]
    if len(factor_ids) != gain_kw.shape[-1]:
        raise ValueError("Factor IDs do not match the risk audit candidate bank")
    rows = []
    bin_rows = []
    for factor_index, factor in enumerate(factor_ids):
        for horizon in range(args.pred_len):
            identity = {"factor": factor, "horizon": horizon + 1}
            mask = gain_eligible[:, horizon, factor_index]
            row = summarize_risk_head(
                arrays["probability"][:, horizon, factor_index],
                gain_kw[:, horizon, factor_index], mask, threshold,
            )
            rows.append({**identity, **row})
            bin_rows.extend(calibration_rows(
                arrays["probability"][:, horizon, factor_index],
                gain_kw[:, horizon, factor_index], mask, **identity,
            ))

    aggregate = summarize_risk_head(
        arrays["probability"], gain_kw, gain_eligible, threshold
    )
    physical_summary = summarize_risk_head(
        arrays["probability"], gain_kw, arrays["physical"], threshold
    )
    selected = np.zeros_like(accepted)
    for factor_index in range(len(factor_ids)):
        selected[..., factor_index] = arrays["action"] == factor_index + 1
    payload = {
        "audit_scope": "chronological_calibration_gate_from_original_train_only",
        "outer_validation_or_test_labels_used": False,
        "fold": int(args.sdwpf_fold),
        "seed": int(args.seed),
        "threshold": threshold,
        "gain_units": "kW (target scaler is fit on the outer training split)",
        "split": split_audit,
        "gate_windows": int(gain_kw.shape[0]),
        "physical_candidate_points": int(arrays["physical"].sum()),
        "gain_eligible_candidate_points": int(gain_eligible.sum()),
        "post_veto_candidate_points": int(accepted.sum()),
        "selected_points": int(selected.sum()),
        "gain_eligible_risk": aggregate,
        "all_physical_risk": physical_summary,
        "factor_ids": factor_ids,
        "checkpoint": checkpoint_info(args.loaded_finetune_checkpoint),
        "training_manifest": getattr(args, "loaded_finetune_manifest", None),
        "note": (
            "Read-only audit of the fixed v2 risk head. Metrics and bins are "
            "descriptive and must not be tuned against outer validation/test labels."
        ),
    }
    pd.DataFrame(rows).to_csv(output / "risk_by_factor_horizon.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(output / "risk_calibration_bins.csv", index=False)
    (output / "risk_audit.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, default=str), flush=True)
    print(f"[RISK-AUDIT] Results: {output}", flush=True)
    return payload
