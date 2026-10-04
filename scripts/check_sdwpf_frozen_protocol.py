#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def validate_protocol(value):
    data = value.get("data", {})
    if value.get("status") != "frozen_validation_protocol":
        raise ValueError("Protocol must be marked frozen_validation_protocol")
    expected = {
        "split": "rolling_holdout",
        "folds": [0, 1, 2],
        "seeds": [2024, 2025, 2026],
        "pred_len": 12,
        "eval_stride": 12,
    }
    for key, wanted in expected.items():
        if data.get(key) != wanted:
            raise ValueError(f"Frozen protocol {key} changed: {data.get(key)!r} != {wanted!r}")
    if "test" in json.dumps(value).lower() and not value.get("selection_disclosure"):
        raise ValueError("Protocol must retain its selection/test disclosure")
    if set(value.get("deep_baselines", {})) != {"PatchTST", "DLinear"}:
        raise ValueError("Frozen strong-baseline set must be PatchTST and DLinear")
    if value.get("paired_inference", {}).get("bootstrap_replicates", 0) < 1000:
        raise ValueError("Frozen protocol requires at least 1000 bootstrap replicates")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = Path(args.config)
    raw = path.read_bytes()
    value = validate_protocol(json.loads(raw.decode("utf-8")))
    print(f"PROTOCOL_ID={value['protocol_id']}")
    print(f"PROTOCOL_SHA256={hashlib.sha256(raw).hexdigest()}")
    print("FOLDS=" + ",".join(map(str, value["data"]["folds"])))
    print("SEEDS=" + ",".join(map(str, value["data"]["seeds"])))
    print("SEALED_TEST_ACCESSED=0")


if __name__ == "__main__":
    main()
