"""Leakage-safe lifecycle management for the compositional wind-event Wiki.

The lifecycle is deliberately offline.  It consumes only out-of-fold evidence
from the training interval, merges candidates that share the same executable
physical rule, decays stale knowledge, and estimates whether a rule transfers
across source turbines.  Validation/test targets are never valid inputs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from utils.wind_regime_wiki import validate_event_factor_rule


TRAIN_ONLY_SOURCE = "train_oof"


def _canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite_float(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unique_ints(values, name: str) -> list[int]:
    result = [int(value) for value in values]
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicate turbine ids")
    return result


def _candidate_statistics(candidate, *, as_of_step: int, parameters: dict) -> dict:
    candidate_id = str(candidate.get("id", "")).strip()
    factor_id = str(candidate.get("factor_id", "")).strip()
    prompt = str(candidate.get("prompt", "")).strip()
    if not candidate_id or not factor_id or not prompt:
        raise ValueError("Every candidate requires non-empty id, factor_id, and prompt")

    created_step = int(candidate.get("created_step", 0))
    last_evidence_step = int(candidate.get("last_evidence_step", created_step))
    if created_step < 0 or last_evidence_step < created_step:
        raise ValueError(f"Candidate {candidate_id!r} has invalid lifecycle steps")
    if last_evidence_step > as_of_step:
        raise ValueError(f"Candidate {candidate_id!r} contains future evidence")

    evidence = candidate.get("turbine_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"Candidate {candidate_id!r} has no turbine_evidence")
    seen_turbines = set()
    turbine_rows = []
    for row in evidence:
        source_split = str(row.get("source_split", TRAIN_ONLY_SOURCE))
        if source_split != TRAIN_ONLY_SOURCE:
            raise ValueError(
                f"Candidate {candidate_id!r} uses forbidden source_split={source_split!r}; "
                "Wiki evolution requires train_oof evidence"
            )
        turbine_id = int(row["turbine_id"])
        if turbine_id in seen_turbines:
            raise ValueError(
                f"Candidate {candidate_id!r} repeats turbine_id={turbine_id}"
            )
        seen_turbines.add(turbine_id)
        n_windows = int(row["n_windows"])
        if n_windows < 2:
            raise ValueError("Each turbine needs at least two OOF windows")
        mean_utility = _finite_float(row["mean_utility"], "mean_utility")
        std_utility = _finite_float(row["std_utility"], "std_utility")
        if std_utility < 0:
            raise ValueError("std_utility cannot be negative")
        within_lcb = mean_utility - parameters["z_value"] * std_utility / math.sqrt(
            n_windows
        )
        turbine_rows.append(
            {
                "turbine_id": turbine_id,
                "n_windows": n_windows,
                "mean_utility": mean_utility,
                "std_utility": std_utility,
                "within_turbine_lcb": within_lcb,
            }
        )

    means = np.asarray([row["mean_utility"] for row in turbine_rows], dtype=np.float64)
    mean_utility = float(means.mean())
    between_std = float(means.std(ddof=1)) if len(means) > 1 else 0.0
    transfer_lcb = (
        mean_utility - parameters["z_value"] * between_std / math.sqrt(len(means))
    )
    positive_fraction = float(
        np.mean([row["within_turbine_lcb"] > 0.0 for row in turbine_rows])
    )
    total_windows = int(sum(row["n_windows"] for row in turbine_rows))
    support_score = min(1.0, total_windows / parameters["min_total_windows"])
    utility_score = float(
        np.clip(transfer_lcb / parameters["target_utility"], 0.0, 1.0)
    )
    transfer_score = positive_fraction * utility_score
    age_steps = int(as_of_step - last_evidence_step)
    forgetting_factor = float(
        2.0 ** (-age_steps / parameters["forget_half_life_steps"])
    )
    deployment_weight = float(transfer_score * support_score * forgetting_factor)

    reasons = []
    if len(turbine_rows) < parameters["min_source_turbines"]:
        reasons.append("insufficient_source_turbines")
    if positive_fraction < parameters["min_positive_turbine_fraction"]:
        reasons.append("unstable_across_turbines")
    if transfer_lcb <= parameters["min_transfer_lcb"]:
        reasons.append("non_positive_cross_turbine_lcb")
    if deployment_weight < parameters["retire_weight"]:
        reasons.append("forgotten_or_low_utility")

    return {
        "id": candidate_id,
        "factor_id": factor_id,
        "title": str(candidate.get("title", factor_id)).strip() or factor_id,
        "prompt": prompt,
        "observable_evidence": list(candidate.get("observable_evidence", [])),
        "rule": copy.deepcopy(candidate.get("rule")),
        "created_step": created_step,
        "last_evidence_step": last_evidence_step,
        "source_turbines": sorted(seen_turbines),
        "turbine_evidence": turbine_rows,
        "source_turbine_count": len(turbine_rows),
        "total_windows": total_windows,
        "mean_utility": mean_utility,
        "between_turbine_std": between_std,
        "cross_turbine_lcb": transfer_lcb,
        "positive_turbine_fraction": positive_fraction,
        "support_score": support_score,
        "transfer_score": transfer_score,
        "age_steps": age_steps,
        "forgetting_factor": forgetting_factor,
        "deployment_weight": deployment_weight,
        "accepted": not reasons,
        "decision_reasons": reasons or ["accepted_train_oof_cross_turbine_evidence"],
    }


def evolve_event_wiki(
    base_spec: dict,
    evidence_spec: dict,
    *,
    as_of_step: int | None = None,
    min_source_turbines: int = 3,
    min_total_windows: int = 300,
    min_positive_turbine_fraction: float = 2.0 / 3.0,
    min_transfer_lcb: float = 0.0,
    target_utility: float = 0.02,
    z_value: float = 1.645,
    forget_half_life_steps: int = 100_000,
    retire_weight: float = 0.10,
    max_merged_insights: int = 3,
) -> tuple[dict, dict]:
    """Return an evolved Wiki and a complete deterministic decision audit.

    ``mean_utility`` is a dimensionless OOF improvement ratio where positive is
    better (for example ``1 - candidate_MAE / reference_MAE``).  Turbines are
    treated as independent transfer units rather than pooling every window.
    """

    if str(evidence_spec.get("source_split", "")) != TRAIN_ONLY_SOURCE:
        raise ValueError("Knowledge evolution accepts source_split='train_oof' only")
    if evidence_spec.get("example_only"):
        raise ValueError("Example evidence cannot be used to evolve a deployable Wiki")
    if str(base_spec.get("entry_type", "")) != "event_factor":
        raise ValueError("Knowledge lifecycle requires an event_factor Wiki")
    base_scenes = base_spec.get("scenes")
    if not isinstance(base_scenes, list) or not base_scenes:
        raise ValueError("Base Wiki has no event-factor scenes")

    configured_step = evidence_spec.get("as_of_step")
    as_of_step = int(configured_step if as_of_step is None else as_of_step)
    if as_of_step < 0:
        raise ValueError("as_of_step must be non-negative")
    if min_source_turbines < 2:
        raise ValueError("min_source_turbines must be at least two")
    if min_total_windows < 1 or forget_half_life_steps < 1:
        raise ValueError("window support and forgetting half-life must be positive")
    if target_utility <= 0 or z_value < 0:
        raise ValueError("target_utility must be positive and z_value non-negative")
    for name, value in (
        ("min_positive_turbine_fraction", min_positive_turbine_fraction),
        ("retire_weight", retire_weight),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if max_merged_insights < 1:
        raise ValueError("max_merged_insights must be positive")

    source_turbines = _unique_ints(evidence_spec.get("source_turbines", []), "source_turbines")
    held_out_turbines = _unique_ints(
        evidence_spec.get("held_out_turbines", []), "held_out_turbines"
    )
    overlap = sorted(set(source_turbines).intersection(held_out_turbines))
    if overlap:
        raise ValueError(f"Held-out turbines leaked into source_turbines: {overlap}")
    if len(source_turbines) < min_source_turbines:
        raise ValueError("The evidence manifest declares too few source turbines")

    parameters = {
        "min_source_turbines": int(min_source_turbines),
        "min_total_windows": int(min_total_windows),
        "min_positive_turbine_fraction": float(min_positive_turbine_fraction),
        "min_transfer_lcb": float(min_transfer_lcb),
        "target_utility": float(target_utility),
        "z_value": float(z_value),
        "forget_half_life_steps": int(forget_half_life_steps),
        "retire_weight": float(retire_weight),
        "max_merged_insights": int(max_merged_insights),
    }

    base_by_id = {str(scene["id"]): copy.deepcopy(scene) for scene in base_scenes}
    base_order = [str(scene["id"]) for scene in base_scenes]
    base_rules = {
        factor_id: _canonical_json(scene.get("rule"))
        for factor_id, scene in base_by_id.items()
    }
    decisions = []
    candidate_ids = set()
    for raw_candidate in evidence_spec.get("candidates", []):
        candidate = _candidate_statistics(
            raw_candidate, as_of_step=as_of_step, parameters=parameters
        )
        if candidate["id"] in candidate_ids:
            raise ValueError(f"Duplicate candidate id: {candidate['id']}")
        candidate_ids.add(candidate["id"])
        leaked_turbines = sorted(
            set(candidate["source_turbines"]).intersection(held_out_turbines)
        )
        if leaked_turbines:
            raise ValueError(
                f"Candidate {candidate['id']!r} uses held-out turbines: {leaked_turbines}"
            )
        unknown_turbines = sorted(
            set(candidate["source_turbines"]).difference(source_turbines)
        )
        if unknown_turbines:
            raise ValueError(
                f"Candidate {candidate['id']!r} uses undeclared source turbines: "
                f"{unknown_turbines}"
            )

        if candidate["factor_id"] in base_by_id and candidate["rule"] is None:
            candidate["rule"] = copy.deepcopy(base_by_id[candidate["factor_id"]]["rule"])
        if candidate["rule"] is None:
            raise ValueError(
                f"New factor {candidate['factor_id']!r} requires an executable rule"
            )
        validate_event_factor_rule(
            candidate["rule"],
            base_spec.get("rule_defaults", {}),
            factor_id=candidate["factor_id"],
        )
        if (
            candidate["factor_id"] in base_rules
            and _canonical_json(candidate["rule"]) != base_rules[candidate["factor_id"]]
        ):
            candidate["accepted"] = False
            candidate["decision_reasons"] = ["conflicts_with_canonical_physical_rule"]
        decisions.append(candidate)

    accepted = [candidate for candidate in decisions if candidate["accepted"]]
    rules_by_factor: dict[str, set[str]] = {}
    for candidate in accepted:
        rules_by_factor.setdefault(candidate["factor_id"], set()).add(
            _canonical_json(candidate["rule"])
        )
    conflicts = sorted(
        factor_id for factor_id, rules in rules_by_factor.items() if len(rules) > 1
    )
    if conflicts:
        raise ValueError(
            "Accepted candidates assign conflicting physical rules to factor ids: "
            + ", ".join(conflicts)
        )
    # Exact executable-rule equality is the merge key.  This prevents a fluent
    # semantic description from collapsing physically different states.
    groups: dict[str, list[dict]] = {}
    for candidate in accepted:
        groups.setdefault(_canonical_json(candidate["rule"]), []).append(candidate)

    evolved_by_id = copy.deepcopy(base_by_id)
    group_audit = []
    active_ids = set()
    evaluated_rule_keys = {_canonical_json(candidate["rule"]) for candidate in decisions}
    for rule_key, group in sorted(groups.items(), key=lambda item: item[0]):
        group.sort(key=lambda item: (-item["deployment_weight"], item["id"]))
        existing_base_ids = [
            factor_id for factor_id in base_order if base_rules[factor_id] == rule_key
        ]
        canonical_id = (
            existing_base_ids[0]
            if existing_base_ids
            else group[0]["factor_id"]
        )
        if canonical_id in base_by_id:
            scene = copy.deepcopy(base_by_id[canonical_id])
        else:
            scene = {
                "id": canonical_id,
                "title": group[0]["title"],
                "prompt": group[0]["prompt"],
                "observable_evidence": group[0]["observable_evidence"],
                "rule": copy.deepcopy(group[0]["rule"]),
            }

        selected = group[:max_merged_insights]
        unique_insights = []
        for candidate in selected:
            insight = candidate["prompt"].strip()
            if insight and insight != scene["prompt"] and insight not in unique_insights:
                unique_insights.append(insight)
        if unique_insights:
            scene["prompt"] = (
                scene["prompt"].rstrip()
                + " Cross-turbine OOF validated refinements: "
                + " ".join(unique_insights)
            )
        deployment_weight = float(max(item["deployment_weight"] for item in group))
        source_ids = sorted(
            {turbine for item in group for turbine in item["source_turbines"]}
        )
        scene["lifecycle"] = {
            "status": "active",
            "deployment_weight": deployment_weight,
            "merged_candidate_ids": [item["id"] for item in group],
            "source_turbines": source_ids,
            "last_evidence_step": max(item["last_evidence_step"] for item in group),
            "cross_turbine_lcb": float(max(item["cross_turbine_lcb"] for item in group)),
            "forgetting_factor": float(max(item["forgetting_factor"] for item in group)),
        }
        evolved_by_id[canonical_id] = scene
        active_ids.add(canonical_id)
        group_audit.append(
            {
                "canonical_factor_id": canonical_id,
                "rule_sha256": hashlib.sha256(rule_key.encode("utf-8")).hexdigest(),
                "merged_candidate_ids": [item["id"] for item in group],
                "deployment_weight": deployment_weight,
                "action": "refine" if canonical_id in base_by_id else "add",
            }
        )

    # Existing factors keep a stable checkpoint slot. A static seed that has
    # never been assessed is carried forward; once lifecycle evidence exists,
    # an unsupported candidate is retired and an already-evolved entry keeps
    # decaying between lifecycle updates.
    previous_as_of_step = int(
        (base_spec.get("knowledge_lifecycle") or {}).get("as_of_step", as_of_step)
    )
    for factor_id in base_order:
        if factor_id in active_ids:
            continue
        scene = evolved_by_id[factor_id]
        prior = scene.get("lifecycle")
        rule_was_assessed = base_rules[factor_id] in evaluated_rule_keys
        if not rule_was_assessed and not prior:
            scene["lifecycle"] = {
                "status": "active_seed",
                "deployment_weight": 1.0,
                "merged_candidate_ids": [],
                "source_turbines": [],
                "last_evidence_step": None,
                "cross_turbine_lcb": None,
                "forgetting_factor": 1.0,
            }
            active_ids.add(factor_id)
            continue
        if not rule_was_assessed and prior:
            elapsed = max(0, as_of_step - previous_as_of_step)
            incremental_decay = 2.0 ** (
                -elapsed / parameters["forget_half_life_steps"]
            )
            carried_weight = float(prior.get("deployment_weight", 1.0)) * incremental_decay
            if carried_weight >= parameters["retire_weight"]:
                scene["lifecycle"] = {
                    **copy.deepcopy(prior),
                    "status": "active_carried",
                    "deployment_weight": carried_weight,
                    "forgetting_factor": float(
                        prior.get("forgetting_factor", 1.0)
                    )
                    * incremental_decay,
                }
                active_ids.add(factor_id)
                continue
        scene["lifecycle"] = {
            "status": "retired",
            "deployment_weight": 0.0,
            "merged_candidate_ids": [],
            "source_turbines": [],
            "last_evidence_step": None,
            "cross_turbine_lcb": None,
            "forgetting_factor": 0.0,
        }

    new_ids = sorted(set(evolved_by_id).difference(base_order))
    output_scenes = [evolved_by_id[factor_id] for factor_id in [*base_order, *new_ids]]
    evolved = copy.deepcopy(base_spec)
    evolved["schema_version"] = max(3, int(base_spec.get("schema_version", 1)))
    evolved["name"] = f"{base_spec.get('name', 'wind event Wiki')} (evolved)"
    evolved["scenes"] = output_scenes
    evolved["knowledge_lifecycle"] = {
        "method": "train_oof_rule_merge_time_decay_cross_turbine_transfer",
        "mutation_policy": "offline_train_only_frozen_for_validation_and_test",
        "source_split": TRAIN_ONLY_SOURCE,
        "as_of_step": as_of_step,
        "source_turbines": source_turbines,
        "held_out_turbines": held_out_turbines,
        "transfer_scope": (
            "source_to_held_out_turbines"
            if held_out_turbines
            else "stability_across_source_turbines"
        ),
        "evidence_sha256": _sha256_json(evidence_spec),
        "parameters": parameters,
        "seed_knowledge_policy": "retain_until_train_oof_assessed",
    }
    lifecycle_weights = [
        float(scene["lifecycle"]["deployment_weight"]) for scene in output_scenes
    ]
    audit = {
        "schema_version": 1,
        "method": evolved["knowledge_lifecycle"]["method"],
        "source_split": TRAIN_ONLY_SOURCE,
        "as_of_step": as_of_step,
        "base_config_sha256": _sha256_json(base_spec),
        "evidence_sha256": evolved["knowledge_lifecycle"]["evidence_sha256"],
        "source_turbines": source_turbines,
        "held_out_turbines": held_out_turbines,
        "transfer_scope": evolved["knowledge_lifecycle"]["transfer_scope"],
        "parameters": parameters,
        "candidate_decisions": decisions,
        "merge_groups": group_audit,
        "factor_ids": [scene["id"] for scene in output_scenes],
        "deployment_weights": lifecycle_weights,
        "active_factor_count": int(sum(weight > 0.0 for weight in lifecycle_weights)),
        "retired_factor_count": int(sum(weight == 0.0 for weight in lifecycle_weights)),
    }
    return evolved, audit


def load_json(path) -> dict:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {source}")
    return value
