#!/usr/bin/env python3
"""Return success only when a Wiki bundle exactly matches its JSON contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.wind_regime_wiki import (
    load_wind_regime_wiki_bundle,
    load_wind_regime_wiki_spec,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args()
    try:
        spec = load_wind_regime_wiki_spec(args.config)
        bundle = load_wind_regime_wiki_bundle(
            args.bundle, expected_scene_ids=spec["scene_ids"]
        )
    except (FileNotFoundError, ValueError):
        raise SystemExit(1)
    reliability_matches = np.allclose(
        bundle["factor_reliability"],
        np.asarray(
            spec.get("factor_reliability") or np.ones(len(spec["scene_ids"])),
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=1e-7,
    )
    raise SystemExit(0 if bundle["config_sha256"] == spec["sha256"] and reliability_matches else 1)


if __name__ == "__main__":
    main()
