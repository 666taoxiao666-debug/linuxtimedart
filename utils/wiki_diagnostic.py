"""Paired evaluation of the event residual, preserving the trained trend path."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import SequentialSampler

from utils.forecast_report import inverse_transform_target, _window_metadata
from utils.experiment_audit import checkpoint_info


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
                "rmse_off_kw": None, "win_fraction": None}
    a, b = on[mask] - truth[mask], off[mask] - truth[mask]
    aw, bw = np.abs(a).mean(axis=1), np.abs(b).mean(axis=1)
    return {**row, "mae_on_kw": float(aw.mean()),
            "mae_off_kw": float(bw.mean()), "mae_gain_kw": float((bw-aw).mean()),
            "rmse_on_kw": float(np.sqrt(np.mean(a*a))),
            "rmse_off_kw": float(np.sqrt(np.mean(b*b))),
            "win_fraction": float((aw < bw).mean())}


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
    summary = {"evaluation_split": "val", "fold": args.sdwpf_fold, "seed": args.seed,
               "windows": len(null), "gain_definition": "MAE(off)-MAE(on); positive favors Wiki",
               "scale": "kW, inverse transformed, no reporting clip",
               "null_windows": null_count,
               "actual_false_intervention_on_null": float(active[null].any(1).mean()) if null_count else None,
               "null_max_prediction_delta_kw": float(np.max(np.abs(on[null]-off[null]))) if null_count else None,
               "all": rows[0], "factor_ids": factors,
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
