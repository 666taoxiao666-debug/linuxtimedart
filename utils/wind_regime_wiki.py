"""Causal wind-regime Wiki loading, rule labels, and train-only audit helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


REQUIRED_SCENE_IDS = (
    "normal_stable",
    "normal_ramp_up",
    "normal_ramp_down",
    "gust_or_turbulent",
    "high_wind_low_power",
    "rated_saturation",
    "low_wind_idle",
)

# The full seven-scene Wiki is retained as a replacement-router ablation.  The
# proposed hybrid router deliberately removes the three ordinary trend scenes:
# those are already represented by the project's original quantile-calibrated
# soft prompts.  Its Wiki therefore models only exceptions plus an explicit
# "do not intervene" anchor.
EXCEPTION_SCENE_IDS = (
    "no_exception",
    "gust_or_turbulent",
    "high_wind_low_power",
    "rated_saturation",
    "low_wind_idle",
)

# Proposed compositional Wiki: these entries are independent physical event
# factors rather than mutually exclusive scenes.  The all-zero multi-label
# vector is the explicit "do not intervene" state, so it is intentionally not
# represented by a learned semantic anchor.
EVENT_FACTOR_IDS = (
    "gust_or_turbulent",
    "high_wind_low_power",
    "rated_saturation",
    "low_wind_idle",
)
VALID_SCENE_ORDERS = (REQUIRED_SCENE_IDS, EXCEPTION_SCENE_IDS)

EVENT_RULE_STATISTICS = {"mean", "std", "max_abs_step", "last", "trend_delta"}
EVENT_RULE_OPERATORS = {"above", "below"}
EVENT_RULE_COMBINERS = {"all", "any"}


BASE_EVENT_FACTOR_RULES = (
    {
        "combine": "any",
        "conditions": [
            {
                "feature": "Wspd",
                "statistic": "std",
                "operator": "above",
                "threshold_key": "gust_std_min_mps",
            },
            {
                "feature": "Wspd",
                "statistic": "max_abs_step",
                "operator": "above",
                "threshold_key": "gust_step_min_mps",
            },
        ],
    },
    {
        "combine": "all",
        "conditions": [
            {
                "feature": "Wspd",
                "statistic": "mean",
                "operator": "above",
                "threshold_key": "active_wind_min_mps",
            },
            {
                "feature": "power_ratio",
                "statistic": "mean",
                "operator": "below",
                "threshold_key": "low_power_max_ratio",
            },
        ],
    },
    {
        "combine": "all",
        "conditions": [
            {
                "feature": "power_ratio",
                "statistic": "mean",
                "operator": "above",
                "threshold_key": "rated_power_min_ratio",
                "scale": 0.1,
            }
        ],
    },
    {
        "combine": "all",
        "conditions": [
            {
                "feature": "Wspd",
                "statistic": "mean",
                "operator": "below",
                "threshold_key": "low_wind_max_mps",
            },
            {
                "feature": "power_ratio",
                "statistic": "mean",
                "operator": "below",
                "threshold_key": "low_power_max_ratio",
            },
        ],
    },
)


def validate_event_factor_rule(rule, rule_defaults, *, factor_id="event_factor"):
    """Validate the small observable-rule DSL; no executable code is allowed."""

    if not isinstance(rule, dict):
        raise ValueError(f"Event factor {factor_id!r} requires a rule object")
    combine = str(rule.get("combine", ""))
    if combine not in EVENT_RULE_COMBINERS:
        raise ValueError(f"Event factor {factor_id!r} has unsupported combine={combine!r}")
    conditions = rule.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError(f"Event factor {factor_id!r} requires rule conditions")
    for index, condition in enumerate(conditions):
        if not isinstance(condition, dict):
            raise ValueError(f"Event factor {factor_id!r} condition {index} is not an object")
        feature = str(condition.get("feature", "")).strip()
        statistic = str(condition.get("statistic", ""))
        operator = str(condition.get("operator", ""))
        if not feature:
            raise ValueError(f"Event factor {factor_id!r} condition {index} has no feature")
        if statistic not in EVENT_RULE_STATISTICS:
            raise ValueError(
                f"Event factor {factor_id!r} has unsupported statistic={statistic!r}"
            )
        if operator not in EVENT_RULE_OPERATORS:
            raise ValueError(
                f"Event factor {factor_id!r} has unsupported operator={operator!r}"
            )
        has_value = "threshold" in condition
        has_key = "threshold_key" in condition
        if has_value == has_key:
            raise ValueError(
                f"Event factor {factor_id!r} condition {index} must define exactly "
                "one of threshold or threshold_key"
            )
        threshold = (
            condition["threshold"]
            if has_value
            else rule_defaults.get(str(condition["threshold_key"]))
        )
        if threshold is None or not np.isfinite(float(threshold)):
            raise ValueError(
                f"Event factor {factor_id!r} condition {index} has no finite threshold"
            )
        if "scale" in condition and (
            not np.isfinite(float(condition["scale"])) or float(condition["scale"]) <= 0
        ):
            raise ValueError(f"Event factor {factor_id!r} condition scale must be positive")


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_wind_regime_wiki_spec(path) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Wind-regime Wiki config not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    scenes = spec.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Wind-regime Wiki must contain a non-empty scenes list")
    scene_ids = tuple(str(scene.get("id", "")).strip() for scene in scenes)
    entry_type = str(spec.get("entry_type", "scene"))
    if entry_type != "event_factor" and scene_ids not in VALID_SCENE_ORDERS:
        raise ValueError(
            "Wind-regime Wiki scene order is part of the checkpoint contract; "
            f"expected one of {VALID_SCENE_ORDERS}, got {scene_ids}"
        )
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("Wind-regime Wiki scene ids must be unique")
    for scene in scenes:
        if not str(scene.get("prompt", "")).strip():
            raise ValueError(f"Scene {scene.get('id')!r} has no semantic prompt")
    defaults = spec.get("rule_defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("rule_defaults must be a JSON object")
    factor_rules = []
    factor_reliability = []
    if entry_type == "event_factor":
        for scene in scenes:
            validate_event_factor_rule(
                scene.get("rule"), defaults, factor_id=scene.get("id")
            )
            factor_rules.append(scene["rule"])
            lifecycle = scene.get("lifecycle", {})
            if lifecycle is not None and not isinstance(lifecycle, dict):
                raise ValueError(
                    f"Event factor {scene.get('id')!r} lifecycle must be an object"
                )
            reliability = float((lifecycle or {}).get("deployment_weight", 1.0))
            if not np.isfinite(reliability) or not 0.0 <= reliability <= 1.0:
                raise ValueError(
                    f"Event factor {scene.get('id')!r} deployment_weight must be in [0, 1]"
                )
            factor_reliability.append(reliability)
    spec["scene_ids"] = list(scene_ids)
    spec["factor_rules"] = factor_rules
    spec["factor_reliability"] = factor_reliability
    spec["sha256"] = sha256_file(source)
    return spec


def load_wind_regime_wiki_bundle(path, expected_scene_ids=None) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Wind-regime Wiki embedding bundle not found: {source}. "
            "Run scripts/build_wind_regime_wiki.py once before training."
        )
    with np.load(source, allow_pickle=False) as bundle:
        required = {"embeddings", "scene_ids", "encoder_name", "config_sha256"}
        missing = sorted(required.difference(bundle.files))
        if missing:
            raise ValueError(f"Wiki embedding bundle is missing fields: {missing}")
        embeddings = np.asarray(bundle["embeddings"], dtype=np.float32)
        scene_ids = tuple(str(value) for value in bundle["scene_ids"].tolist())
        encoder_name = str(bundle["encoder_name"].item())
        config_sha256 = str(bundle["config_sha256"].item())
        factor_reliability = (
            np.asarray(bundle["factor_reliability"], dtype=np.float32)
            if "factor_reliability" in bundle.files
            else np.ones(len(scene_ids), dtype=np.float32)
        )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(scene_ids):
        raise ValueError(
            "Wiki embeddings must have shape [num_scenes, hidden_size], got "
            f"{embeddings.shape} for {len(scene_ids)} scenes"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Wiki embeddings contain non-finite values")
    if factor_reliability.shape != (len(scene_ids),):
        raise ValueError(
            "Wiki factor_reliability must have shape [num_scenes], got "
            f"{factor_reliability.shape}"
        )
    if not np.isfinite(factor_reliability).all() or (
        (factor_reliability < 0.0) | (factor_reliability > 1.0)
    ).any():
        raise ValueError("Wiki factor_reliability must be finite and in [0, 1]")
    if expected_scene_ids is not None and tuple(expected_scene_ids) != scene_ids:
        raise ValueError(
            "Wiki embedding scene order does not match the JSON config: "
            f"{scene_ids} != {tuple(expected_scene_ids)}"
        )
    return {
        "embeddings": embeddings,
        "scene_ids": scene_ids,
        "encoder_name": encoder_name,
        "config_sha256": config_sha256,
        "factor_reliability": factor_reliability,
        "sha256": sha256_file(source),
    }


def compute_scene_wiki_rule_logits(
    history_raw: torch.Tensor,
    feature_columns,
    *,
    rated_power: float,
    recent_steps: int = 12,
    low_wind_max_mps: float = 3.0,
    active_wind_min_mps: float = 5.0,
    low_power_max_ratio: float = 0.08,
    rated_power_min_ratio: float = 0.9,
    gust_std_min_mps: float = 1.0,
    gust_step_min_mps: float = 2.0,
    ramp_delta_min_ratio: float = 0.1,
    scene_ids=None,
):
    """Return deterministic scene priors and labels from observed history only.

    The rules provide leakage-safe supervision; the trainable Wiki retriever is
    still responsible for mapping encoder states to the frozen LLM anchors.
    """

    if history_raw.ndim != 3:
        raise ValueError(
            f"Scene Wiki history must be [batch,time,feature], got {history_raw.shape}"
        )
    names = {str(name): index for index, name in enumerate(feature_columns)}
    if "Wspd" not in names or "power" not in names:
        raise ValueError("Scene Wiki routing requires named Wspd and power channels")
    if not np.isfinite(float(rated_power)) or float(rated_power) <= 0:
        raise ValueError("rated_power must be finite and positive")
    steps = max(3, min(int(recent_steps), int(history_raw.size(1))))
    recent = history_raw[:, -steps:, :]
    wind = recent[:, :, names["Wspd"]]
    power_ratio = recent[:, :, names["power"]] / float(rated_power)
    wind_mean = wind.mean(dim=1)
    power_mean = power_ratio.mean(dim=1)
    wind_std = wind.std(dim=1, unbiased=False)
    max_wind_step = (
        wind[:, 1:].sub(wind[:, :-1]).abs().amax(dim=1)
        if steps > 1
        else torch.zeros_like(wind_mean)
    )
    edge = max(1, steps // 3)
    power_delta = power_ratio[:, -edge:].mean(dim=1) - power_ratio[:, :edge].mean(dim=1)

    labels = torch.zeros(history_raw.size(0), dtype=torch.long, device=history_raw.device)
    low_idle = (wind_mean <= float(low_wind_max_mps)) & (
        power_mean <= float(low_power_max_ratio)
    )
    high_wind_low_power = (wind_mean >= float(active_wind_min_mps)) & (
        power_mean <= float(low_power_max_ratio)
    )
    rated = power_mean >= float(rated_power_min_ratio)
    gust = (wind_std >= float(gust_std_min_mps)) | (
        max_wind_step >= float(gust_step_min_mps)
    )
    ramp_up = power_delta >= float(ramp_delta_min_ratio)
    ramp_down = power_delta <= -float(ramp_delta_min_ratio)

    # Priority is explicit and reproducible. Observable anomalous/physical
    # limits take precedence over generic trend labels.
    labels[ramp_up] = 1
    labels[ramp_down] = 2
    labels[gust] = 3
    labels[high_wind_low_power] = 4
    labels[rated] = 5
    labels[low_idle] = 6
    scene_ids = tuple(scene_ids or REQUIRED_SCENE_IDS)
    if scene_ids == EXCEPTION_SCENE_IDS:
        # normal stable/up/down -> no_exception; four exceptional states retain
        # their identity.  This avoids duplicating the original trend prompts.
        labels = torch.where(labels <= 2, torch.zeros_like(labels), labels - 2)
    elif scene_ids != REQUIRED_SCENE_IDS:
        raise ValueError(f"Unsupported Scene Wiki order: {scene_ids}")
    rule_logits = torch.nn.functional.one_hot(
        labels, num_classes=len(scene_ids)
    ).to(dtype=history_raw.dtype)
    return rule_logits, labels


def compute_event_factor_rule_logits(
    history_raw: torch.Tensor,
    feature_columns,
    *,
    rated_power: float,
    recent_steps: int = 12,
    low_wind_max_mps: float = 3.0,
    active_wind_min_mps: float = 5.0,
    low_power_max_ratio: float = 0.08,
    rated_power_min_ratio: float = 0.9,
    gust_std_min_mps: float = 1.0,
    gust_step_min_mps: float = 2.0,
    factor_ids=None,
    factor_rules=None,
    rule_defaults=None,
):
    """Return causal signed evidence and independent event-factor targets.

    Both outputs have shape ``[batch, num_factors]``.  Targets are multi-hot,
    so a gust can coexist with rated saturation or high-wind/low-power.  Rule
    logits are bounded signed margins in ``[-1, 1]``: positive values provide
    physical support, negative values oppose intervention, and values near zero
    deliberately express uncertainty around a rule boundary.
    """

    if history_raw.ndim != 3:
        raise ValueError(
            "Event Wiki history must be [batch,time,feature], got "
            f"{history_raw.shape}"
        )
    if not torch.isfinite(history_raw).all():
        raise ValueError("Event Wiki history contains non-finite values")
    names = {str(name): index for index, name in enumerate(feature_columns)}
    if "power" not in names:
        raise ValueError("Event Wiki routing requires a named power channel")
    if not np.isfinite(float(rated_power)) or float(rated_power) <= 0:
        raise ValueError("rated_power must be finite and positive")

    factor_ids = tuple(factor_ids or EVENT_FACTOR_IDS)
    factor_rules = tuple(factor_rules or BASE_EVENT_FACTOR_RULES)
    if len(factor_ids) != len(factor_rules) or not factor_ids:
        raise ValueError("Event factor ids and executable rules must have equal non-zero length")
    thresholds = {
        str(key): float(value) for key, value in (rule_defaults or {}).items()
    }
    thresholds.update({
        "low_wind_max_mps": float(low_wind_max_mps),
        "active_wind_min_mps": float(active_wind_min_mps),
        "low_power_max_ratio": float(low_power_max_ratio),
        "rated_power_min_ratio": float(rated_power_min_ratio),
        "gust_std_min_mps": float(gust_std_min_mps),
        "gust_step_min_mps": float(gust_step_min_mps),
    })
    for factor_id, rule in zip(factor_ids, factor_rules):
        validate_event_factor_rule(rule, thresholds, factor_id=factor_id)

    steps = max(3, min(int(recent_steps), int(history_raw.size(1))))
    recent = history_raw[:, -steps:, :]
    eps = torch.finfo(history_raw.dtype).eps

    def above(value, threshold, scale=None):
        denominator = max(
            abs(float(scale if scale is not None else threshold)), float(eps)
        )
        return ((value - float(threshold)) / denominator).clamp(-1.0, 1.0)

    def below(value, threshold, scale=None):
        denominator = max(
            abs(float(scale if scale is not None else threshold)), float(eps)
        )
        return ((float(threshold) - value) / denominator).clamp(-1.0, 1.0)

    def statistic(condition):
        feature = str(condition["feature"])
        if feature == "power_ratio":
            series = recent[:, :, names["power"]] / float(rated_power)
        else:
            if feature not in names:
                raise ValueError(
                    f"Event Wiki rule references unavailable feature {feature!r}"
                )
            series = recent[:, :, names[feature]]
        name = str(condition["statistic"])
        if name == "mean":
            return series.mean(dim=1)
        if name == "std":
            return series.std(dim=1, unbiased=False)
        if name == "max_abs_step":
            return series[:, 1:].sub(series[:, :-1]).abs().amax(dim=1)
        if name == "last":
            return series[:, -1]
        if name == "trend_delta":
            edge = max(1, steps // 3)
            return series[:, -edge:].mean(dim=1) - series[:, :edge].mean(dim=1)
        raise AssertionError(f"Unvalidated statistic: {name}")

    evidence_columns = []
    target_columns = []
    for rule in factor_rules:
        condition_evidence = []
        condition_targets = []
        for condition in rule["conditions"]:
            value = statistic(condition)
            threshold = float(
                condition["threshold"]
                if "threshold" in condition
                else thresholds[str(condition["threshold_key"])]
            )
            scale = condition.get("scale")
            margin = (
                above(value, threshold, scale=scale)
                if condition["operator"] == "above"
                else below(value, threshold, scale=scale)
            )
            condition_evidence.append(margin)
            condition_targets.append(margin > 0.0)
        stacked_evidence = torch.stack(condition_evidence, dim=-1)
        stacked_targets = torch.stack(condition_targets, dim=-1)
        if rule["combine"] == "all":
            evidence_columns.append(stacked_evidence.amin(dim=-1))
            target_columns.append(stacked_targets.all(dim=-1))
        else:
            evidence_columns.append(stacked_evidence.amax(dim=-1))
            target_columns.append(stacked_targets.any(dim=-1))
    rule_logits = torch.stack(evidence_columns, dim=-1)
    targets = torch.stack(target_columns, dim=-1).to(dtype=history_raw.dtype)
    return rule_logits, targets


def audit_scene_wiki_labels_from_dataset(
    dataset,
    feature_columns,
    *,
    rated_power: float,
    max_samples: int = 50000,
    rule_kwargs=None,
    scene_ids=None,
) -> dict:
    """Count Wiki pseudo-label support using deterministic training windows."""

    sample_count = min(int(max_samples), len(dataset))
    if sample_count < 1:
        raise ValueError("training split is empty; cannot audit Scene Wiki labels")
    selected = np.unique(
        np.linspace(0, len(dataset) - 1, num=sample_count, dtype=np.int64)
    )
    if not all(hasattr(dataset, name) for name in ("data_x", "window_starts", "seq_len")):
        raise TypeError("Scene Wiki audit requires indexed SDWPF window storage")
    values = np.asarray(dataset.data_x)
    starts = np.asarray(dataset.window_starts, dtype=np.int64)[selected]
    offsets = np.arange(int(dataset.seq_len), dtype=np.int64)
    mean = np.asarray(dataset.scaler.mean_, dtype=np.float32)
    scale = np.asarray(dataset.scaler.scale_, dtype=np.float32)
    scene_ids = tuple(scene_ids or REQUIRED_SCENE_IDS)
    if scene_ids not in VALID_SCENE_ORDERS:
        raise ValueError(f"Unsupported Scene Wiki order: {scene_ids}")
    counts = np.zeros(len(scene_ids), dtype=np.int64)
    kwargs = dict(rule_kwargs or {})
    for offset in range(0, len(starts), 2048):
        chunk_starts = starts[offset : offset + 2048]
        rows = chunk_starts[:, None] + offsets[None, :]
        scaled = values[rows]
        raw = scaled * scale.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
        _, labels = compute_scene_wiki_rule_logits(
            torch.as_tensor(raw, dtype=torch.float32),
            feature_columns,
            rated_power=rated_power,
            scene_ids=scene_ids,
            **kwargs,
        )
        counts += np.bincount(labels.cpu().numpy(), minlength=len(counts))
    return {
        "method": "scene_wiki",
        "source_split": "train",
        "sample_count": int(counts.sum()),
        "scene_ids": list(scene_ids),
        "class_counts": [int(value) for value in counts],
        "class_fractions": [float(value / max(1, counts.sum())) for value in counts],
    }


def audit_event_factor_labels_from_dataset(
    dataset,
    feature_columns,
    *,
    rated_power: float,
    max_samples: int = 50000,
    rule_kwargs=None,
    factor_ids=None,
) -> dict:
    """Audit multi-label event support using training histories only."""

    if getattr(dataset, "flag", "train") != "train":
        raise ValueError("event-factor calibration is restricted to the training split")
    sample_count = min(int(max_samples), len(dataset))
    if sample_count < 1:
        raise ValueError("training split is empty; cannot audit event-factor labels")
    selected = np.unique(
        np.linspace(0, len(dataset) - 1, num=sample_count, dtype=np.int64)
    )
    if not all(hasattr(dataset, name) for name in ("data_x", "window_starts", "seq_len")):
        raise TypeError("Event Wiki audit requires indexed SDWPF window storage")
    values = np.asarray(dataset.data_x)
    starts = np.asarray(dataset.window_starts, dtype=np.int64)[selected]
    offsets = np.arange(int(dataset.seq_len), dtype=np.int64)
    mean = np.asarray(dataset.scaler.mean_, dtype=np.float32)
    scale = np.asarray(dataset.scaler.scale_, dtype=np.float32)
    factor_ids = tuple(factor_ids or EVENT_FACTOR_IDS)
    if not factor_ids:
        raise ValueError("Event-factor audit requires at least one factor")

    positive_counts = np.zeros(len(factor_ids), dtype=np.int64)
    cooccurrence = np.zeros((len(factor_ids), len(factor_ids)), dtype=np.int64)
    no_intervention_count = 0
    combination_counts = np.zeros(1 << len(factor_ids), dtype=np.int64)
    kwargs = dict(rule_kwargs or {})
    # ramp_delta_min_ratio belongs to the legacy seven-scene rules only.
    kwargs.pop("ramp_delta_min_ratio", None)
    for offset in range(0, len(starts), 2048):
        chunk_starts = starts[offset : offset + 2048]
        rows = chunk_starts[:, None] + offsets[None, :]
        scaled = values[rows]
        raw = scaled * scale.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
        _, targets = compute_event_factor_rule_logits(
            torch.as_tensor(raw, dtype=torch.float32),
            feature_columns,
            rated_power=rated_power,
            factor_ids=factor_ids,
            **kwargs,
        )
        binary = targets.to(dtype=torch.int64).cpu().numpy()
        positive_counts += binary.sum(axis=0)
        cooccurrence += binary.T @ binary
        no_intervention_count += int((binary.sum(axis=1) == 0).sum())
        codes = (binary * (1 << np.arange(len(factor_ids)))).sum(axis=1)
        combination_counts += np.bincount(
            codes, minlength=len(combination_counts)
        )

    audited_samples = int(len(starts))
    return {
        "method": "causal_multi_label_event_factors",
        "source_split": "train",
        "sample_count": audited_samples,
        "factor_ids": list(factor_ids),
        "positive_counts": [int(value) for value in positive_counts],
        "negative_counts": [
            int(audited_samples - value) for value in positive_counts
        ],
        "positive_fractions": [
            float(value / max(1, audited_samples)) for value in positive_counts
        ],
        "no_intervention_count": int(no_intervention_count),
        "no_intervention_fraction": float(
            no_intervention_count / max(1, audited_samples)
        ),
        "mean_active_factors": float(positive_counts.sum() / max(1, audited_samples)),
        "cooccurrence": cooccurrence.tolist(),
        "combination_counts": {
            format(index, f"0{len(factor_ids)}b"): int(value)
            for index, value in enumerate(combination_counts)
            if value > 0
        },
    }


def update_event_factor_confusion(confusion, logits, targets, threshold=0.5):
    """Accumulate counts and return batch-level abstention statistics."""

    if confusion.ndim != 2 or confusion.shape[1] != 4:
        raise ValueError("event-factor confusion must have shape [num_factors, 4]")
    probabilities = torch.sigmoid(logits.detach())
    predicted = probabilities >= float(threshold)
    expected = targets.detach() >= 0.5
    if predicted.shape != expected.shape or predicted.shape[1] != confusion.shape[0]:
        raise ValueError(
            "event-factor prediction/target shape does not match confusion: "
            f"{predicted.shape}, {expected.shape}, {confusion.shape}"
        )
    predicted = predicted.cpu().numpy()
    expected = expected.cpu().numpy()
    confusion[:, 0] += np.logical_and(~predicted, ~expected).sum(axis=0)
    confusion[:, 1] += np.logical_and(predicted, ~expected).sum(axis=0)
    confusion[:, 2] += np.logical_and(~predicted, expected).sum(axis=0)
    confusion[:, 3] += np.logical_and(predicted, expected).sum(axis=0)
    true_null = ~expected.any(axis=1)
    predicted_null = ~predicted.any(axis=1)
    return np.asarray(
        [
            expected.shape[0],
            np.all(predicted == expected, axis=1).sum(),
            true_null.sum(),
            predicted_null.sum(),
            np.logical_and(true_null, predicted_null).sum(),
            np.logical_and(true_null, ~predicted_null).sum(),
        ],
        dtype=np.int64,
    )


def summarize_event_factor_confusion(confusion, sample_stats=None) -> dict:
    """Summarize deterministic multi-label routing diagnostics."""

    values = np.asarray(confusion, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("event-factor confusion must have shape [num_factors, 4]")
    tn, fp, fn, tp = values.T
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / np.maximum(1, tp + fn)
    f1 = 2.0 * precision * recall / np.maximum(np.finfo(float).eps, precision + recall)
    micro_precision = float(tp.sum() / max(1, int((tp + fp).sum())))
    micro_recall = float(tp.sum() / max(1, int((tp + fn).sum())))
    micro_f1 = float(
        2.0
        * micro_precision
        * micro_recall
        / max(np.finfo(float).eps, micro_precision + micro_recall)
    )
    result = {
        "event_hamming_accuracy": float((tp.sum() + tn.sum()) / max(1, values.sum())),
        "event_macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
        "event_micro_f1": micro_f1,
        "event_positive_counts": [int(value) for value in (tp + fn)],
        "event_pred_positive_counts": [int(value) for value in (tp + fp)],
        "event_precision": [float(value) for value in precision],
        "event_recall": [float(value) for value in recall],
        "event_f1": [float(value) for value in f1],
        "event_confusion_tn_fp_fn_tp": values.tolist(),
    }
    if sample_stats is not None:
        stats = np.asarray(sample_stats, dtype=np.int64).reshape(-1)
        if stats.shape != (6,):
            raise ValueError("event-factor sample stats must have six entries")
        samples, exact, true_null, predicted_null, correct_null, false_on_null = stats
        result.update(
            {
                "event_exact_match": float(exact / max(1, samples)),
                "event_true_null_count": int(true_null),
                "event_pred_null_count": int(predicted_null),
                "event_null_recall": float(correct_null / max(1, true_null)),
                "event_false_intervention_on_null": float(
                    false_on_null / max(1, true_null)
                ),
            }
        )
    return result
