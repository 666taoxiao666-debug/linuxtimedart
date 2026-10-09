"""Paired evaluation of the event residual, preserving the trained trend path."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import SequentialSampler

from utils.forecast_report import inverse_transform_target, _window_metadata
from utils.experiment_audit import checkpoint_info
from utils.metrics import forecast_metrics


def paired_forward(model, router, x):
    captured = []

    def observe(module, inputs, kwargs, output):
        rules = kwargs["rule_logits"]
        captured.append((rules.detach().cpu(), output[2].detach().cpu(),
                         output[3].detach().cpu()))

    handle = router.register_forward_hook(observe, with_kwargs=True)
    try:
        enabled = model(x)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Expected exactly one event router call per forecast")

    def suppress(module, inputs, output):
        # Replace ONLY the event prompt tensor. All model parameters, trend
        # prompts, classification outputs and physical rules stay unchanged.
        return (torch.zeros_like(output[0]), *output[1:])

    handle = router.register_forward_hook(suppress)
    try:
        disabled = model(x)
    finally:
        handle.remove()
    return enabled, disabled, captured[0]


def group_metrics(on, off, truth, mask, name):
    n = int(mask.sum())
    row = {"group": name, "windows": n}
    if not n:
        return {**row, "mae_on_kw": None, "mae_off_kw": None,
                "mae_gain_kw": None, "rmse_on_kw": None,
                "rmse_off_kw": None, "r2_on": None, "r2_off": None,
                "mae_skill_vs_off_pct": None,
                "rmse_skill_vs_off_pct": None, "win_fraction": None}
    a, b = on[mask] - truth[mask], off[mask] - truth[mask]
    aw, bw = np.abs(a).mean(axis=1), np.abs(b).mean(axis=1)
    on_metrics = forecast_metrics(on[mask], truth[mask])
    off_metrics = forecast_metrics(off[mask], truth[mask])
    eps = np.finfo(float).eps
    return {**row, "mae_on_kw": float(aw.mean()),
            "mae_off_kw": float(bw.mean()), "mae_gain_kw": float((bw-aw).mean()),
            "rmse_on_kw": float(np.sqrt(np.mean(a*a))),
            "rmse_off_kw": float(np.sqrt(np.mean(b*b))),
            "r2_on": float(on_metrics["r2"]),
            "r2_off": float(off_metrics["r2"]),
            "mae_skill_vs_off_pct": float(
                100.0 * (1.0 - on_metrics["mae"] / max(off_metrics["mae"], eps))
            ),
            "rmse_skill_vs_off_pct": float(
                100.0 * (1.0 - on_metrics["rmse"] / max(off_metrics["rmse"], eps))
            ),
            "win_fraction": float((aw < bw).mean())}


def summarize_intervention_policy(on, off, truth, supported, active):
    """Report policy coverage, abstention, selected benefit and selected harm."""
    on = np.asarray(on, dtype=np.float64)
    off = np.asarray(off, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    supported = np.asarray(supported, dtype=bool)
    active = np.asarray(active, dtype=bool)
    if on.ndim != 2 or on.shape != off.shape or on.shape != truth.shape:
        raise ValueError("Predictions and truth must share [window, horizon] shape")
    if supported.ndim != 2 or active.shape != supported.shape or len(active) != len(on):
        raise ValueError("Supported and active masks must share [window, factor] shape")
    candidate_window = supported.any(axis=1)
    intervention_window = active.any(axis=1)
    window_gain = np.abs(off - truth).mean(axis=1) - np.abs(on - truth).mean(axis=1)
    selected = window_gain[intervention_window]
    on_metrics = forecast_metrics(on, truth)
    off_metrics = forecast_metrics(off, truth)
    eps = np.finfo(float).eps
    return {
        "windows": int(len(on)),
        "candidate_coverage_pct": float(100.0 * candidate_window.mean()),
        "intervention_coverage_pct": float(100.0 * intervention_window.mean()),
        "abstention_pct": float(100.0 * (~intervention_window).mean()),
        "overall_mae_on_kw": float(np.abs(on - truth).mean()),
        "overall_mae_off_kw": float(np.abs(off - truth).mean()),
        "overall_rmse_on_kw": float(on_metrics["rmse"]),
        "overall_rmse_off_kw": float(off_metrics["rmse"]),
        "overall_r2_on": float(on_metrics["r2"]),
        "overall_r2_off": float(off_metrics["r2"]),
        "overall_mae_skill_vs_off_pct": float(
            100.0 * (1.0 - on_metrics["mae"] / max(off_metrics["mae"], eps))
        ),
        "overall_rmse_skill_vs_off_pct": float(
            100.0 * (1.0 - on_metrics["rmse"] / max(off_metrics["rmse"], eps))
        ),
        "overall_gain_kw": float(window_gain.mean()),
        "selected_windows": int(intervention_window.sum()),
        "selected_gain_kw": float(selected.mean()) if selected.size else None,
        "selected_harm_window_pct": (
            float(100.0 * np.mean(selected < 0.0)) if selected.size else None
        ),
        "false_intervention_on_no_evidence_pct": (
            float(100.0 * intervention_window[~candidate_window].mean())
            if (~candidate_window).any()
            else None
        ),
    }


def summarize_factorized_intervention_policy(on, off, truth, availability, action):
    """Summarize per-horizon expert decisions against the exact trend fallback.

    ``action == 0`` is abstention; ``action == k + 1`` selects candidate ``k``.
    Coverage is reported at both window and forecast-point granularity so a
    sparse correction at one horizon is not mistaken for intervention across
    the whole forecast window.
    """
    on = np.asarray(on, dtype=np.float64)
    off = np.asarray(off, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    availability = np.asarray(availability, dtype=bool)
    action = np.asarray(action, dtype=np.int64)
    if on.ndim != 2 or on.shape != off.shape or on.shape != truth.shape:
        raise ValueError("Predictions and truth must share [window, horizon] shape")
    if availability.ndim != 3 or availability.shape[:2] != on.shape:
        raise ValueError("Availability must have shape [window, horizon, candidate]")
    if action.shape != on.shape:
        raise ValueError("Actions must have shape [window, horizon]")
    if np.any(action < 0) or np.any(action > availability.shape[-1]):
        raise ValueError("Action index exceeds the candidate bank")
    chosen = action > 0
    candidate = availability.any(axis=-1)
    if np.any(chosen & ~candidate):
        raise ValueError("An intervention was selected where no candidate was available")
    selected_available = np.zeros_like(chosen)
    if chosen.any():
        rows, horizons = np.nonzero(chosen)
        selected_available[rows, horizons] = availability[
            rows, horizons, action[rows, horizons] - 1
        ]
    if not np.array_equal(selected_available, chosen):
        raise ValueError("An unavailable factor was selected")

    point_gain = np.abs(off - truth) - np.abs(on - truth)
    window_gain = point_gain.mean(axis=1)
    active_windows = chosen.any(axis=1)
    candidate_windows = candidate.any(axis=1)
    selected_point_gain = point_gain[chosen]
    selected_window_gain = window_gain[active_windows]
    on_metrics = forecast_metrics(on, truth)
    off_metrics = forecast_metrics(off, truth)
    eps = np.finfo(float).eps
    return {
        "windows": int(len(on)),
        "forecast_points": int(on.size),
        "candidate_coverage_pct": float(100.0 * candidate_windows.mean()),
        "candidate_point_coverage_pct": float(100.0 * candidate.mean()),
        "intervention_coverage_pct": float(100.0 * active_windows.mean()),
        "intervention_point_coverage_pct": float(100.0 * chosen.mean()),
        "abstention_pct": float(100.0 * (~active_windows).mean()),
        "abstention_point_pct": float(100.0 * (~chosen).mean()),
        "overall_mae_on_kw": float(np.abs(on - truth).mean()),
        "overall_mae_off_kw": float(np.abs(off - truth).mean()),
        "overall_rmse_on_kw": float(on_metrics["rmse"]),
        "overall_rmse_off_kw": float(off_metrics["rmse"]),
        "overall_r2_on": float(on_metrics["r2"]),
        "overall_r2_off": float(off_metrics["r2"]),
        "overall_mae_skill_vs_off_pct": float(
            100.0 * (1.0 - on_metrics["mae"] / max(off_metrics["mae"], eps))
        ),
        "overall_rmse_skill_vs_off_pct": float(
            100.0 * (1.0 - on_metrics["rmse"] / max(off_metrics["rmse"], eps))
        ),
        "overall_gain_kw": float(point_gain.mean()),
        "selected_windows": int(active_windows.sum()),
        "selected_points": int(chosen.sum()),
        "selected_gain_kw": (
            float(selected_point_gain.mean()) if selected_point_gain.size else None
        ),
        "selected_window_gain_kw": (
            float(selected_window_gain.mean()) if selected_window_gain.size else None
        ),
        "selected_harm_point_pct": (
            float(100.0 * np.mean(selected_point_gain < 0.0))
            if selected_point_gain.size else None
        ),
        "selected_harm_window_pct": (
            float(100.0 * np.mean(selected_window_gain < 0.0))
            if selected_window_gain.size else None
        ),
        "false_intervention_on_no_evidence_pct": (
            float(100.0 * chosen[~candidate].mean()) if (~candidate).any() else None
        ),
    }


def _run_factorized_wiki_diagnostic(exp, dataset, loader, output):
    """Evaluate the trained factor bank without legacy prompt deletion."""
    args, model = exp.args, exp.model
    parts = {key: [] for key in (
        "on", "off", "truth", "availability", "physical_availability",
        "action", "factor_predictions"
    )}
    if getattr(args, "utility_harm_veto", False):
        parts["harm_probability"] = []
    model.eval()
    with torch.no_grad():
        for i, (x, y, _, _) in enumerate(loader):
            x = x.float().to(exp.device)
            with exp._autocast():
                prediction = model(x)
            aux = getattr(model, "_last_utility_aux", None)
            required = {
                "base_prediction", "factor_predictions",
                "factor_availability", "factor_action",
            }
            if not isinstance(aux, dict) or not required.issubset(aux):
                raise RuntimeError("Factorized Wiki did not expose its audited candidate bank")
            parts["on"].append(
                prediction[:, -args.pred_len:, -1].detach().float().cpu().numpy()
            )
            parts["off"].append(
                aux["base_prediction"][:, -args.pred_len:, -1]
                .detach().float().cpu().numpy()
            )
            parts["truth"].append(
                y[:, -args.pred_len:, -1].detach().float().cpu().numpy()
            )
            parts["availability"].append(
                aux["factor_availability"].detach().cpu().numpy()
            )
            parts["physical_availability"].append(
                aux.get("factor_physical_availability", aux["factor_availability"])
                .detach().cpu().numpy()
            )
            if "harm_probability" in parts:
                if "factor_harm_probability" not in aux:
                    raise RuntimeError("Harm-veto checkpoint did not expose risk probabilities")
                parts["harm_probability"].append(
                    aux["factor_harm_probability"].detach().float().cpu().numpy()
                )
            parts["action"].append(aux["factor_action"].detach().cpu().numpy())
            parts["factor_predictions"].append(
                aux["factor_predictions"][:, -args.pred_len:, -1]
                .detach().float().cpu().numpy()
            )
            if i % 50 == 0:
                print(f"[WIKI-DIAG] Factorized validation batch {i+1}/{len(loader)}", flush=True)

    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    if len(arrays["on"]) != len(dataset):
        raise ValueError("Window count does not match dataset")
    for value in arrays.values():
        if not np.isfinite(value).all():
            raise ValueError("Non-finite values in factorized diagnostic outputs")
    for key in ("on", "off", "truth"):
        arrays[key] = inverse_transform_target(dataset, arrays[key])
    for candidate in range(arrays["factor_predictions"].shape[-1]):
        arrays["factor_predictions"][..., candidate] = inverse_transform_target(
            dataset, arrays["factor_predictions"][..., candidate]
        )
    if not all(np.isfinite(arrays[key]).all() for key in (
        "on", "off", "truth", "factor_predictions"
    )):
        raise ValueError("Non-finite inverse-transformed factorized predictions")
    np.savez_compressed(output / "paired_predictions.npz", **arrays)

    on, off, truth = (arrays[key] for key in ("on", "off", "truth"))
    availability, action = arrays["availability"].astype(bool), arrays["action"].astype(int)
    physical_availability = arrays["physical_availability"].astype(bool)
    summary = summarize_factorized_intervention_policy(
        on, off, truth, availability, action
    )
    physical_points = physical_availability.any(-1)
    routed_points = availability.any(-1)
    physical_windows = physical_points.any(-1)
    routed_windows = routed_points.any(-1)
    summary.update({
        "physical_candidate_coverage_pct": float(100.0 * physical_windows.mean()),
        "physical_candidate_point_coverage_pct": float(100.0 * physical_points.mean()),
        "harm_veto_window_pct_of_physical": (
            float(100.0 * (physical_windows & ~routed_windows).sum() / physical_windows.sum())
            if physical_windows.any() else None
        ),
        "harm_veto_point_pct_of_physical": (
            float(100.0 * (physical_points & ~routed_points).sum() / physical_points.sum())
            if physical_points.any() else None
        ),
    })
    pd.DataFrame([summary]).to_csv(output / "intervention_summary.csv", index=False)

    metadata = _window_metadata(dataset, len(on))
    if not {"TurbID", "forecast_start"}.issubset(metadata.columns):
        raise ValueError("Turbine/time metadata is required for paired comparison")
    metadata["mae_on_kw"] = np.abs(on - truth).mean(axis=1)
    metadata["mae_off_kw"] = np.abs(off - truth).mean(axis=1)
    metadata["mae_gain_kw"] = metadata.mae_off_kw - metadata.mae_on_kw
    metadata["candidate_available"] = availability.any(axis=(1, 2)).astype(int)
    metadata["physical_candidate_available"] = physical_availability.any(axis=(1, 2)).astype(int)
    metadata["intervention_active"] = (action > 0).any(axis=1).astype(int)

    factor_ids = [*model.scene_wiki_scene_ids, "composition"]
    if len(factor_ids) != availability.shape[-1]:
        raise ValueError("Factor IDs do not match the audited candidate bank")
    factor_rows = []
    base_point_error = np.abs(off - truth)
    for index, factor in enumerate(factor_ids):
        available = availability[..., index]
        physical_available = physical_availability[..., index]
        selected = action == index + 1
        candidate_error = np.abs(arrays["factor_predictions"][..., index] - truth)
        selected_gain = (base_point_error - candidate_error)[selected]
        candidate_gain = (base_point_error - candidate_error)[available]
        factor_rows.append({
            "factor": factor,
            "available_points": int(available.sum()),
            "physical_available_points": int(physical_available.sum()),
            "selected_points": int(selected.sum()),
            "availability_pct": float(100.0 * available.mean()),
            "physical_availability_pct": float(100.0 * physical_available.mean()),
            "harm_veto_pct_of_physical": (
                float(100.0 * (physical_available & ~available).sum() / physical_available.sum())
                if physical_available.any() else None
            ),
            "intervention_pct": float(100.0 * selected.mean()),
            "candidate_gain_kw": (
                float(candidate_gain.mean()) if candidate_gain.size else None
            ),
            "selected_gain_kw": (
                float(selected_gain.mean()) if selected_gain.size else None
            ),
            "selected_harm_point_pct": (
                float(100.0 * np.mean(selected_gain < 0.0))
                if selected_gain.size else None
            ),
        })
        metadata[f"{factor}_available_steps"] = available.sum(axis=1)
        metadata[f"{factor}_physical_available_steps"] = physical_available.sum(axis=1)
        metadata[f"{factor}_selected_steps"] = selected.sum(axis=1)
    metadata.to_csv(output / "window_metrics.csv", index=False)
    pd.DataFrame(factor_rows).to_csv(output / "factor_metrics.csv", index=False)

    masks = {
        "all": np.ones(len(on), dtype=bool),
        "candidate_available": availability.any(axis=(1, 2)),
        "intervention_active": (action > 0).any(axis=1),
        "abstained": ~(action > 0).any(axis=1),
    }
    pd.DataFrame([
        group_metrics(on, off, truth, mask, name) for name, mask in masks.items()
    ]).to_csv(output / "group_metrics.csv", index=False)
    pd.DataFrame([
        {
            **group_metrics(
                on[:, horizon:horizon + 1], off[:, horizon:horizon + 1],
                truth[:, horizon:horizon + 1],
                np.ones(len(on), dtype=bool), f"horizon_{horizon + 1}",
            ),
            "candidate_coverage_pct": float(100.0 * availability[:, horizon].any(-1).mean()),
            "intervention_coverage_pct": float(100.0 * (action[:, horizon] > 0).mean()),
        }
        for horizon in range(args.pred_len)
    ]).to_csv(output / "horizon_metrics.csv", index=False)

    payload = {
        "evaluation_split": "val",
        "fold": args.sdwpf_fold,
        "seed": args.seed,
        "windows": len(on),
        "gain_definition": "absolute error(base trend)-absolute error(selected Wiki); positive favors Wiki",
        "scale": "kW, inverse transformed, no reporting clip",
        "intervention_policy": summary,
        "factor_ids": factor_ids,
        "factors": factor_rows,
        "note": "Fixed-checkpoint factorized decision audit; action 0 is exact abstention",
        "checkpoint": checkpoint_info(args.loaded_finetune_checkpoint),
        "wiki_config": checkpoint_info(args.scene_wiki_config),
        "wiki_bundle": checkpoint_info(args.scene_wiki_embeddings),
        "parameters": vars(args),
    }
    (output / "diagnostic_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    brief = {key: value for key, value in payload.items()
             if key not in ("parameters", "checkpoint", "wiki_config", "wiki_bundle")}
    print(json.dumps(brief, indent=2), flush=True)
    print(f"[WIKI-DIAG] Results: {output.resolve()}", flush=True)


def without_event_forward(model, router, x, index):
    """Delete one additive contribution, keeping top-k and normalization fixed.

    This is a fixed-context intervention, not re-routing the remaining events.
    Re-normalizing after deletion would also change every retained contribution.
    """
    if model.training:
        raise ValueError("Event interventions require eval mode")
    if not 0 <= index < router.num_factors:
        raise ValueError("Invalid event index")
    def remove(module, inputs, output):
        activations = output[3]
        values = module.prompt_norm(
            module.semantic_projection(module.semantic_keys) + module.prompt_delta)
        denominator = (activations > 0).sum(-1, keepdim=True).to(activations.dtype).clamp_min(1).sqrt()
        contribution = (activations[:, index:index+1] * values[index:index+1]
                        / denominator * torch.sigmoid(module.prompt_gate_logit))
        return (output[0] - contribution, *output[1:])
    handle = router.register_forward_hook(remove)
    try:
        return model(x)
    finally:
        handle.remove()


def run_wiki_diagnostic(exp):
    args = exp.args
    if args.report_split != "val" or args.sdwpf_split != "rolling_holdout":
        raise ValueError("Diagnostic is restricted to rolling_holdout validation")
    if args.features != "MS" or not args.report_output_dir:
        raise ValueError("Diagnostics require features=MS and an explicit output directory")
    model = exp.model
    if isinstance(model, torch.nn.DataParallel):
        raise ValueError("Use a single GPU for ordered paired diagnostics")
    if getattr(model, "prompt_router", None) != "compositional_wiki":
        raise ValueError("A compositional Wiki checkpoint is required")
    dataset, loader = exp._get_data(flag="val")
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError("Diagnostics require all validation windows in sequential order")
    output = Path(args.report_output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    if getattr(args, "utility_factorized", False):
        _run_factorized_wiki_diagnostic(exp, dataset, loader, output)
        return
    parts = {k: [] for k in ("on", "off", "truth", "rules", "prob", "activation")}
    factors = list(model.scene_wiki_scene_ids)
    event_keys = ([f"without_{j}" for j in range(len(factors))]
                  if getattr(args, "wiki_diagnostic_single_events", False) else [])
    parts.update({k: [] for k in event_keys})
    with torch.no_grad():
        for i, (x, y, _, _) in enumerate(loader):
            x = x.float().to(exp.device)
            with exp._autocast():
                on, off, (rules, prob, activation) = paired_forward(
                    model, model.scene_wiki_router, x)
                for j, key in enumerate(event_keys):
                    prediction = without_event_forward(model, model.scene_wiki_router, x, j)
                    parts[key].append(prediction[:, -args.pred_len:, -1].detach().float().cpu().numpy())
            for key, value in (("on", on), ("off", off), ("truth", y)):
                parts[key].append(value[:, -args.pred_len:, -1].detach().float().cpu().numpy())
            for key, value in (("rules", rules), ("prob", prob), ("activation", activation)):
                parts[key].append(value.float().numpy())
            if i % 50 == 0:
                print(f"[WIKI-DIAG] Validation batch {i+1}/{len(loader)}", flush=True)
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    if len(arrays["on"]) != len(dataset):
        raise ValueError("Window count does not match dataset")
    for value in arrays.values():
        if not np.isfinite(value).all():
            raise ValueError("Non-finite values in diagnostic outputs")
    for key in ("on", "off", "truth", *event_keys):
        arrays[key] = inverse_transform_target(dataset, arrays[key])
        if not np.isfinite(arrays[key]).all():
            raise ValueError("Non-finite inverse-transformed predictions or labels")
    np.savez_compressed(output / "paired_predictions.npz", **arrays)
    supported = arrays["rules"] > 0
    active = arrays["activation"] > 0
    null = ~supported.any(axis=1)
    metadata = _window_metadata(dataset, len(null))
    if not {"TurbID", "forecast_start"}.issubset(metadata.columns):
        raise ValueError("Turbine/time metadata is required for paired comparison")
    on, off, truth = (arrays[k] for k in ("on", "off", "truth"))
    if event_keys:
        event_rows, event_horizons = [], []
        for j, key in enumerate(event_keys):
            for group, mask in (("all", np.ones(len(on), dtype=bool)),
                                ("event_active", active[:, j])):
                row = group_metrics(on, arrays[key], truth, mask, group)
                row["factor"] = factors[j]
                row["mean_abs_prediction_change_kw"] = (
                    float(np.abs(on[mask]-arrays[key][mask]).mean()) if mask.any() else None)
                event_rows.append(row)
                for h in range(args.pred_len):
                    row = group_metrics(on[:, h:h+1], arrays[key][:, h:h+1],
                                        truth[:, h:h+1], mask, group)
                    event_horizons.append({**row, "factor": factors[j], "horizon": h+1})
        pd.DataFrame(event_rows).to_csv(output / "single_event_metrics.csv", index=False)
        pd.DataFrame(event_horizons).to_csv(output / "single_event_horizon_metrics.csv", index=False)
    metadata["mae_on_kw"] = np.abs(on-truth).mean(axis=1)
    metadata["mae_off_kw"] = np.abs(off-truth).mean(axis=1)
    metadata["mae_gain_kw"] = metadata.mae_off_kw - metadata.mae_on_kw
    factors = list(model.scene_wiki_scene_ids)
    factor_rows = []
    for j, factor in enumerate(factors):
        predicted = arrays["prob"][:, j] >= 0.5
        target = supported[:, j]
        tp = int((predicted & target).sum())
        fp = int((predicted & ~target).sum())
        fn = int((~predicted & target).sum())
        factor_rows.append({"factor": factor, "support": int(target.sum()),
                            "precision": tp/max(1, tp+fp), "recall": tp/max(1, tp+fn),
                            "f1": 2*tp/max(1, 2*tp+fp+fn),
                            "actual_active_fraction": float(active[:, j].mean()),
                            "mean_strength": float(arrays["activation"][:, j].mean())})
        metadata[f"{factor}_supported"] = target.astype(int)
        metadata[f"{factor}_active"] = active[:, j].astype(int)
        metadata[f"{factor}_strength"] = arrays["activation"][:, j]
    metadata.to_csv(output / "window_metrics.csv", index=False)
    pd.DataFrame(factor_rows).to_csv(output / "factor_metrics.csv", index=False)
    masks = {"all": np.ones(len(null), dtype=bool), "rule_null": null,
             "rule_single": supported.sum(1) == 1, "rule_multiple": supported.sum(1) > 1,
             "actually_active": active.any(1), "actually_inactive": ~active.any(1)}
    masks.update({f"supported:{name}": supported[:, j] for j, name in enumerate(factors)})
    masks.update({f"active:{name}": active[:, j] for j, name in enumerate(factors)})
    rows = [group_metrics(on, off, truth, mask, name) for name, mask in masks.items()]
    pd.DataFrame(rows).to_csv(output / "group_metrics.csv", index=False)
    pd.DataFrame([group_metrics(on[:, h:h+1], off[:, h:h+1], truth[:, h:h+1],
                               masks["all"], f"horizon_{h+1}")
                  for h in range(args.pred_len)]).to_csv(output / "horizon_metrics.csv", index=False)
    null_count = int(null.sum())
    intervention_summary = summarize_intervention_policy(
        on, off, truth, supported, active
    )
    pd.DataFrame([intervention_summary]).to_csv(
        output / "intervention_summary.csv", index=False
    )
    summary = {"evaluation_split": "val", "fold": args.sdwpf_fold, "seed": args.seed,
               "windows": len(null), "gain_definition": "MAE(off)-MAE(on); positive favors Wiki",
               "scale": "kW, inverse transformed, no reporting clip",
               "null_windows": null_count,
               "actual_false_intervention_on_null": float(active[null].any(1).mean()) if null_count else None,
               "null_max_prediction_delta_kw": float(np.max(np.abs(on[null]-off[null]))) if null_count else None,
               "all": rows[0], "intervention_policy": intervention_summary,
               "factor_ids": factors,
               "note": "Fixed-checkpoint intervention, not a retrained ablation; groups overlap",
               "single_event_intervention": bool(event_keys),
               "single_event_definition": "Remove one additive prompt contribution; keep original top-k and composition denominator; positive gain favors retaining event",
               "checkpoint": checkpoint_info(args.loaded_finetune_checkpoint),
               "wiki_config": checkpoint_info(args.scene_wiki_config),
               "wiki_bundle": checkpoint_info(args.scene_wiki_embeddings),
               "parameters": vars(args)}
    (output / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    # Keep full checkpoint/config provenance in the JSON, not hundreds of
    # nested manifest lines in the terminal (which obscure completion).
    brief = {k: v for k, v in summary.items()
             if k not in ("parameters", "checkpoint", "wiki_config", "wiki_bundle")}
    print(json.dumps(brief, indent=2), flush=True)
    print(f"[WIKI-DIAG] Results: {output.resolve()}", flush=True)
