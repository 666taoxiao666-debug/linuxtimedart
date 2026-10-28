#!/usr/bin/env python3
"""Independent raw-power audit for v31; reads archives, never rewrites labels."""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_kelmarsh_multiyear_preparation as original


def audit(metadata, years, curve_years):
    reads = []
    raw_reader = original.read_power_ticks

    def unique_power(path, unit):
        ticks = raw_reader(path, unit)
        time, power = "Date and time", "Power (kW)"
        known = ticks.groupby(time)[power].nunique(dropna=True)
        if (known > 1).any():
            raise ValueError("Ambiguous original power: audit refuses multiple nonmissing values")
        # Independent implementation: sort known power first, select an original
        # row per timestamp; all-missing timestamps stay missing.
        selected = ticks.sort_values(power, na_position="last", kind="mergesort").drop_duplicates(time)
        selected = selected.sort_values(time, kind="mergesort").reset_index(drop=True)
        assert len(selected) == ticks[time].nunique()
        reads.append({"archive": str(path), "turbine": int(unit), "raw_rows": len(ticks),
                      "timestamps": len(selected), "multiple_nonmissing_power_values": 0})
        return selected

    with patch.object(original, "read_power_ticks", unique_power):
        result = original.audit(metadata, years, curve_years)
    result.pop("all_24_provider_headers_declare_utc", None)
    result["provider_utc_headers_checked"] = len(years) * 6
    result["independent_power_duplicate_checks"] = reads
    result["audit_v31_sha256"] = original.sha(Path(__file__))
    result["label_policy"] = "existing unique nonmissing source value per timestamp; missing timestamps remain missing; no interpolation or averaging duplicates"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--curve-years", nargs="+", type=int, required=True)
    args = parser.parse_args()
    if args.years != list(range(2016, 2024)) or args.curve_years != list(range(2016, 2020)):
        raise ValueError("v31 audit authorizes exactly 2016-2023 and curves 2016-2019")
    result = audit(args.metadata, args.years, args.curve_years)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("passed", "rows", "csv_sha256", "metadata_sha256", "flags_not_deletions")}))
