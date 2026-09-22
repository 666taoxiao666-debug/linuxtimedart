"""Reproducibility and provenance records for forecasting experiments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def sha256_file(path) -> str | None:
    """Hash a file once per process, keyed by resolved path, size, and mtime."""
    if not path:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return None
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    if key not in _HASH_CACHE:
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        _HASH_CACHE[key] = digest.hexdigest()
    return _HASH_CACHE[key]


def checkpoint_info(path) -> dict:
    if not path:
        return {"path": None, "exists": False, "sha256": None}
    resolved = Path(path).expanduser().resolve()
    exists = resolved.is_file()
    return {
        "path": str(resolved),
        "exists": exists,
        "size_bytes": resolved.stat().st_size if exists else None,
        "mtime": (
            datetime.fromtimestamp(resolved.stat().st_mtime)
            .astimezone()
            .isoformat()
            if exists
            else None
        ),
        "sha256": sha256_file(resolved) if exists else None,
    }


def _run_git(args, *, raw=False):
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    if raw:
        return completed.stdout
    return completed.stdout.decode("utf-8", errors="replace").strip()


def git_info() -> dict:
    status = _run_git(["status", "--porcelain"])
    diff = _run_git(["diff", "--binary", "--", "."], raw=True)
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "branch": _run_git(["branch", "--show-current"]),
        "dirty": bool(status) if status is not None else None,
        "status": status.splitlines() if status else [],
        "diff_sha256": (
            hashlib.sha256(diff).hexdigest()
            if diff is not None
            else None
        ),
    }


def environment_info() -> dict:
    cuda_device = None
    if torch.cuda.is_available():
        try:
            cuda_device = torch.cuda.get_device_name(torch.cuda.current_device())
        except (RuntimeError, AssertionError):
            cuda_device = None
    return {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "working_directory": str(Path.cwd().resolve()),
        "command_line": [str(argument) for argument in sys.argv],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_device": cuda_device,
    }


def args_info(args) -> dict:
    return {key: _jsonable(value) for key, value in sorted(vars(args).items())}


def data_file_info(args) -> dict:
    path = Path(args.root_path).expanduser() / args.data_path
    resolved = path.resolve()
    exists = resolved.is_file()
    should_hash = bool(getattr(args, "audit_hash_data", True))
    return {
        "path": str(resolved),
        "exists": exists,
        "size_bytes": resolved.stat().st_size if exists else None,
        "mtime": (
            datetime.fromtimestamp(resolved.stat().st_mtime)
            .astimezone()
            .isoformat()
            if exists
            else None
        ),
        "sha256": sha256_file(resolved) if exists and should_hash else None,
        "hash_enabled": should_hash,
    }


def dataset_summary(dataset, split_name: str) -> dict:
    result = {
        "split": split_name,
        "dataset_class": type(dataset).__name__,
        "windows": int(len(dataset)),
        "seq_len": int(getattr(dataset, "seq_len", 0)),
        "pred_len": int(getattr(dataset, "pred_len", 0)),
        "window_stride": int(getattr(dataset, "window_stride", 0)),
        "feature_columns": list(getattr(dataset, "feature_columns", []) or []),
    }
    try:
        sample = dataset[0]
    except (IndexError, KeyError, TypeError, ValueError):
        sample = None
    if isinstance(sample, (list, tuple)):
        result["sample_shapes"] = [
            list(getattr(value, "shape", ())) for value in sample
        ]
    dates = getattr(dataset, "dates", None)
    if dates is not None and len(dates):
        dates = np.asarray(dates)
        result["row_date_min"] = str(dates.min())
        result["row_date_max"] = str(dates.max())
    starts = getattr(dataset, "window_starts", None)
    if starts is not None and len(starts) and dates is not None:
        starts = np.asarray(starts, dtype=np.int64)
        seq_len = int(dataset.seq_len)
        total_len = seq_len + int(dataset.pred_len)
        target_start = dates[starts + seq_len]
        target_end = dates[starts + total_len - 1]
        result.update(
            {
                "target_date_min": str(target_start.min()),
                "target_date_max": str(target_end.max()),
                "first_window_history_start": str(dates[starts[0]]),
                "first_window_history_end": str(dates[starts[0] + seq_len - 1]),
                "first_window_target_start": str(target_start[0]),
                "first_window_target_end": str(target_end[0]),
            }
        )
    for name in (
        "train_cutoff",
        "val_cutoff",
        "test_start_cutoff",
        "test_cutoff",
    ):
        value = getattr(dataset, name, None)
        result[name] = str(value) if value is not None else None
    segments = getattr(dataset, "segments", None)
    if segments is not None:
        result["continuous_segments"] = int(len(segments))
    turbines = getattr(dataset, "turbines", None)
    if turbines is not None and len(turbines):
        turbines = np.asarray(turbines)
        ids, counts = np.unique(turbines, return_counts=True)
        result["turbines"] = int(len(ids))
        result["rows_by_turbine"] = {
            str(int(turbine)): int(count) for turbine, count in zip(ids, counts)
        }
        if starts is not None and len(starts):
            window_starts = np.asarray(starts, dtype=np.int64)
            target_turbines = turbines[window_starts + int(dataset.seq_len)]
            ids, counts = np.unique(target_turbines, return_counts=True)
            result["windows_by_turbine"] = {
                str(int(turbine)): int(count) for turbine, count in zip(ids, counts)
            }
            window_ends = (
                window_starts + int(dataset.seq_len) + int(dataset.pred_len) - 1
            )
            cross_turbine = turbines[window_starts] != turbines[window_ends]
            result["cross_turbine_windows"] = int(cross_turbine.sum())
            if cross_turbine.any():
                raise AssertionError(
                    f"{split_name} contains {int(cross_turbine.sum())} windows "
                    "that cross turbine identifiers"
                )
    available = getattr(dataset, "available_mask", None)
    if available is not None and len(available):
        result["available_fraction"] = float(np.asarray(available, dtype=bool).mean())
    audit_stats = getattr(dataset, "audit_stats", None)
    if audit_stats is not None:
        result["cleaning_audit"] = _jsonable(audit_stats)
    scaler = getattr(dataset, "scaler", None)
    if scaler is not None:
        result["scaler"] = {
            "mean": _jsonable(getattr(scaler, "mean_", None)),
            "scale": _jsonable(getattr(scaler, "scale_", None)),
            "var": _jsonable(getattr(scaler, "var_", None)),
        }
    return result


def split_boundary_checks(datasets) -> dict:
    bounds = {}
    for name, dataset in (datasets or {}).items():
        dates = getattr(dataset, "dates", None)
        starts = getattr(dataset, "window_starts", None)
        if dates is None or starts is None or not len(starts):
            continue
        dates = np.asarray(dates)
        starts = np.asarray(starts, dtype=np.int64)
        seq_len = int(dataset.seq_len)
        total_len = seq_len + int(dataset.pred_len)
        bounds[name] = (
            dates[starts + seq_len].min(),
            dates[starts + total_len - 1].max(),
        )
    checks = {}
    for left, right in (("train", "val"), ("val", "test"), ("train", "test")):
        if left not in bounds or right not in bounds:
            continue
        passed = bool(bounds[left][1] < bounds[right][0])
        checks[f"{left}_before_{right}"] = {
            "passed": passed,
            f"{left}_target_max": str(bounds[left][1]),
            f"{right}_target_min": str(bounds[right][0]),
        }
        if not passed:
            raise AssertionError(
                f"Target timestamp overlap: {left} max={bounds[left][1]} "
                f"is not before {right} min={bounds[right][0]}"
            )
    return checks


def model_runtime_summary(model, args=None) -> dict:
    core = model.module if isinstance(model, torch.nn.DataParallel) else model
    result = {
        "class": type(core).__name__,
        "parameter_count": int(sum(parameter.numel() for parameter in core.parameters())),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in core.parameters() if parameter.requires_grad)
        ),
    }
    for name in (
        "use_soft_prompt",
        "use_prompt_adaln",
        "mix_channels",
        "use_channel_prior",
        "use_op_context",
        "residual_forecast",
        "regime_label_method",
        "prompt_router",
        "utility_wiki",
        "utility_adapter_mode",
    ):
        result[name] = _jsonable(getattr(core, name, None))
    utility_gate = getattr(core, "utility_gate", None)
    if utility_gate is not None:
        result["utility_gate"] = {
            "candidate_conditioned": bool(getattr(utility_gate, "candidate_conditioned", False)),
            "physical_features": int(getattr(utility_gate, "physical_dim", 0)),
            "routing_train": "soft" if getattr(core, "utility_adapter_mode", "") == "calibrated_evidence" else "hard",
            "routing_eval": "hard",
            "gain_units": "train_target_std" if getattr(core, "utility_adapter_mode", "") == "calibrated_evidence" else "relative_ratio",
            "calibration_split": getattr(args, "utility_calibration_audit", None),
            "phase_lr_restart": getattr(core, "utility_adapter_mode", "") == "calibrated_evidence" and getattr(args, "lradj", "") == "step",
            "candidate_bounds": {
                "event": getattr(getattr(core, "utility_event_adapter", None), "max_scale", None),
                "composition_increment": getattr(getattr(core, "utility_composition_adapter", None), "max_scale", None),
                "units": "input_window_target_std",
            },
            "training_balance": {"hierarchical_evidence": "history_trend_event_sqrt_cap3", "calibrated_evidence": "natural"}.get(getattr(core, "utility_adapter_mode", "legacy"), "legacy"),
            "temperature": _jsonable(getattr(utility_gate, "temperature", None)),
            "min_gain": _jsonable(getattr(utility_gate, "min_gain", None)),
            "conditioning": "history_plus_event_evidence_plus_trend_probabilities",
            "num_trend_modes": _jsonable(
                getattr(utility_gate, "num_trend_modes", None)
            ),
            "adapter_warmup_epochs": int(
                getattr(args, "utility_adapter_warmup_epochs", 0)
            ),
            "intervention_floor": _jsonable(
                getattr(utility_gate, "intervention_floor", None)
            ),
        }
    if hasattr(core, "regime_down_thresh") and hasattr(core, "regime_up_thresh"):
        down = float(core.regime_down_thresh.detach().cpu().item())
        up = float(core.regime_up_thresh.detach().cpu().item())
        result["regime_thresholds"] = {
            "down": down if np.isfinite(down) else None,
            "up": up if np.isfinite(up) else None,
            "source": getattr(args, "regime_calibration_source", None),
        }
    calibration = getattr(core, "regime_calibration", None)
    if calibration is not None:
        result["regime_calibration"] = _jsonable(calibration)
    if getattr(core, "prompt_router", None) in (
        "scene_wiki",
        "hybrid_wiki",
        "compositional_wiki",
    ):
        router = core.scene_wiki_router
        channel_weights = None
        if getattr(router, "log_channel_weight", None) is not None:
            values = torch.softmax(router.log_channel_weight.detach(), dim=0).cpu().numpy()
            names = list(getattr(args, "feature_columns", []) or [])
            channel_weights = {
                (names[index] if index < len(names) else f"feature_{index}"): float(value)
                for index, value in enumerate(values)
            }
        result["scene_wiki"] = {
            "scene_ids": list(getattr(core, "scene_wiki_scene_ids", ())),
            "encoder": getattr(core, "scene_wiki_encoder_name", None),
            "config_sha256": getattr(core, "scene_wiki_config_sha256", None),
            "bundle_sha256": getattr(core, "scene_wiki_bundle_sha256", None),
            "prompt_gate": float(
                torch.sigmoid(router.prompt_gate_logit.detach())
                .cpu()
                .item()
            ),
            "retrieval_channel_weights": channel_weights,
        }
        if getattr(core, "prompt_router", None) == "compositional_wiki":
            result["scene_wiki"].update(
                {
                    "entry_type": "multi_label_event_factor",
                    "activation_threshold": float(router.activation_threshold),
                    "confidence_power": float(router.confidence_power),
                    "top_k": int(router.top_k),
                    "null_behavior": "exact_zero_without_positive_physical_support",
                    "factor_reliability": {
                        factor_id: float(value)
                        for factor_id, value in zip(
                            getattr(core, "scene_wiki_scene_ids", ()),
                            router.factor_reliability.detach().cpu().tolist(),
                        )
                    },
                }
            )
            utility_gate = getattr(core, "utility_gate", None)
            if utility_gate is not None:
                result["scene_wiki"]["utility_decision"] = {
                    "granularities": ["single_event", "composition"],
                    "prediction_scope": "per_horizon",
                    "trend_conditioning": ["down", "stable", "up"],
                    "temperature": float(utility_gate.temperature),
                    "min_gain": float(utility_gate.min_gain),
                    "abstention": "exact_trend_fallback_without_positive_utility",
                }
        intervention = getattr(core, "_last_wiki_intervention", None)
        if intervention is not None and intervention.numel():
            values = intervention.detach().float().cpu()
            result["scene_wiki"]["last_intervention_mean"] = float(values.mean())
            result["scene_wiki"]["last_intervention_max"] = float(values.max())
        factor_activations = getattr(core, "_last_wiki_factor_activations", None)
        if factor_activations is not None and factor_activations.numel():
            values = factor_activations.detach().float().cpu()
            active = values > 0.0
            result["scene_wiki"]["last_zero_intervention_fraction"] = float(
                (~active.any(dim=1)).float().mean().item()
            )
            result["scene_wiki"]["last_mean_active_factors"] = float(
                active.float().sum(dim=1).mean().item()
            )
            result["scene_wiki"]["last_factor_activation_fraction"] = {
                factor_id: float(active[:, index].float().mean().item())
                for index, factor_id in enumerate(
                    getattr(core, "scene_wiki_scene_ids", ())
                )
            }
    transfer_audit = getattr(core, "pretrain_transfer_audit", None)
    if transfer_audit is not None:
        result["pretrain_transfer"] = _jsonable(transfer_audit)
    overlay_audit = getattr(core, "overlay_transfer_audit", None)
    if overlay_audit is not None:
        result["overlay_transfer"] = _jsonable(overlay_audit)
    if getattr(core, "residual_gate_logit", None) is not None:
        result["residual_gate"] = float(
            torch.sigmoid(core.residual_gate_logit.detach()).cpu().item()
        )
    mixer = getattr(core, "channel_mixer", None)
    if mixer is not None:
        scales = mixer.channel_scale().detach().cpu().numpy()
        names = list(getattr(args, "feature_columns", []) or [])
        result["channel_scale"] = {
            (names[index] if index < len(names) else f"feature_{index}"): float(value)
            for index, value in enumerate(scales)
        }
    return result


def write_run_manifest(
    output_dir,
    args,
    stage: str,
    *,
    model=None,
    datasets=None,
    checkpoints=None,
    extra=None,
) -> str:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "stage": stage,
        "environment": environment_info(),
        "git": git_info(),
        "args": args_info(args),
        "data_file": data_file_info(args),
        "datasets": {
            name: dataset_summary(dataset, name)
            for name, dataset in (datasets or {}).items()
        },
        "split_boundary_checks": split_boundary_checks(datasets),
        "model": model_runtime_summary(model, args) if model is not None else None,
        "checkpoints": _jsonable(checkpoints or {}),
        "extra": _jsonable(extra or {}),
    }
    path = output / "run_manifest.json"
    temporary = output / "run_manifest.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)
    print(f"[AUDIT] Run manifest: {path.resolve()}")
    return str(path)
