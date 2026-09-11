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
VALID_SCENE_ORDERS = (REQUIRED_SCENE_IDS, EXCEPTION_SCENE_IDS)


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
    if scene_ids not in VALID_SCENE_ORDERS:
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
    spec["scene_ids"] = list(scene_ids)
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
    if embeddings.ndim != 2 or embeddings.shape[0] != len(scene_ids):
        raise ValueError(
            "Wiki embeddings must have shape [num_scenes, hidden_size], got "
            f"{embeddings.shape} for {len(scene_ids)} scenes"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Wiki embeddings contain non-finite values")
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
