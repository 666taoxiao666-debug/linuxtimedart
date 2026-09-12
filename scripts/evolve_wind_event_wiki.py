#!/usr/bin/env python3
"""Build a versioned event Wiki from train-only cross-turbine evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.wind_regime_wiki import load_wind_regime_wiki_spec
from utils.wind_wiki_lifecycle import evolve_event_wiki, load_json


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Merge, forget, and transfer event-Wiki knowledge using train_oof "
            "evidence only. The resulting config is frozen for validation/test."
        )
    )
    parser.add_argument(
        "--base_config", default="configs/wind_event_factor_wiki.json"
    )
    parser.add_argument("--evidence", required=True)
    parser.add_argument(
        "--output_config", default="outputs/wiki/evolved/wind_event_factor_wiki.json"
    )
    parser.add_argument(
        "--audit_output", default="outputs/wiki/evolved/lifecycle_audit.json"
    )
    parser.add_argument("--as_of_step", type=int, default=None)
    parser.add_argument("--min_source_turbines", type=int, default=3)
    parser.add_argument("--min_total_windows", type=int, default=300)
    parser.add_argument(
        "--min_positive_turbine_fraction", type=float, default=2.0 / 3.0
    )
    parser.add_argument("--min_transfer_lcb", type=float, default=0.0)
    parser.add_argument("--target_utility", type=float, default=0.02)
    parser.add_argument("--z_value", type=float, default=1.645)
    parser.add_argument("--forget_half_life_steps", type=int, default=100_000)
    parser.add_argument("--retire_weight", type=float, default=0.10)
    parser.add_argument("--max_merged_insights", type=int, default=3)
    return parser


def _atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main():
    args = build_parser().parse_args()
    base = load_wind_regime_wiki_spec(args.base_config)
    # Derived loader fields are not source content and must not be written back.
    for key in ("scene_ids", "factor_rules", "factor_reliability", "sha256"):
        base.pop(key, None)
    evidence = load_json(args.evidence)
    evolved, audit = evolve_event_wiki(
        base,
        evidence,
        as_of_step=args.as_of_step,
        min_source_turbines=args.min_source_turbines,
        min_total_windows=args.min_total_windows,
        min_positive_turbine_fraction=args.min_positive_turbine_fraction,
        min_transfer_lcb=args.min_transfer_lcb,
        target_utility=args.target_utility,
        z_value=args.z_value,
        forget_half_life_steps=args.forget_half_life_steps,
        retire_weight=args.retire_weight,
        max_merged_insights=args.max_merged_insights,
    )
    output = Path(args.output_config)
    audit_output = Path(args.audit_output)
    _atomic_json(output, evolved)
    # Re-load the written file so schema, physical rules, and deployment weights
    # are checked before the artifact can be used by training.
    validated = load_wind_regime_wiki_spec(output)
    audit["output_config"] = str(output.resolve())
    audit["output_config_sha256"] = validated["sha256"]
    audit["factor_ids"] = validated["scene_ids"]
    audit["deployment_weights"] = validated["factor_reliability"]
    _atomic_json(audit_output, audit)
    print(
        "[WIKI-LIFECYCLE] source=train_oof "
        f"candidates={len(audit['candidate_decisions'])} "
        f"active={audit['active_factor_count']} retired={audit['retired_factor_count']}"
    )
    print(f"[WIKI-LIFECYCLE] config={output.resolve()}")
    print(f"[WIKI-LIFECYCLE] audit={audit_output.resolve()}")


if __name__ == "__main__":
    main()
