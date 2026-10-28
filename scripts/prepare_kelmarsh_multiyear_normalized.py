#!/usr/bin/env python3
"""Protocol-driven multi-year Kelmarsh preparation with the frozen v31 duplicate-row normalization (v32 onward).

Years, curve years, root and the normalization version come from the frozen protocol section given by
--preparation-key, so later protocols reuse this file instead of cloning a preparer per version.

  prepare  the frozen ``scripts/prepare_kelmarsh_multiyear.prepare`` with its tick merge replaced by the frozen
           ``scripts/prepare_kelmarsh_multiyear_v31.canonical_ticks`` (unchanged: keep one existing row that already
           holds every known value; refuse conflicts and complementary partial rows; never fill, average or clip).
           Each normalization record is paired with the turbine id of the preparer's own unit loop (fixes the v31
           note that ids were inferred from call order).
  audit    the frozen independent v31 power audit (``scripts/audit_kelmarsh_multiyear_v31.audit``) for the same years.

  python scripts/prepare_kelmarsh_multiyear_normalized.py prepare --protocol configs/kelmarsh_upstream_protocol_v32.json \
      --preparation-key v32_preparation
  python scripts/prepare_kelmarsh_multiyear_normalized.py audit --protocol ... --preparation-key v32_preparation --output <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_kelmarsh_multiyear_v31 as audit_v31  # noqa: E402
from scripts import prepare_kelmarsh_multiyear as original  # noqa: E402
from scripts import prepare_kelmarsh_multiyear_v31 as normalization  # noqa: E402

THIS = "scripts/prepare_kelmarsh_multiyear_normalized.py"
NORMALIZATION_SOURCE = "scripts/prepare_kelmarsh_multiyear_v31.py"


def section(protocol_path, key):
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    prep = protocol["data"][key]
    if prep.get("duplicate_normalization") != "existing_dominating_row_v1":
        raise ValueError("Unsupported duplicate normalization version")
    if not set(prep["curve_years"]) <= set(prep["years"]) or set(prep["years"]) & set(protocol["data"].get("sealed_years", [])):
        raise ValueError("Curve years must be prepared years and no sealed year may be prepared")
    return prep


def prepare(root, years, curve_years, version):
    """Run the frozen preparer with the frozen normalization; one record per turbine, id taken from the unit loop."""
    pending, records = [], []
    real_prepare_turbine = original.prepare_turbine

    def merge(frames):
        selected, record = normalization.canonical_ticks(frames)
        pending.append(record)
        return selected

    def prepare_turbine(unit, ticks, rated_kw, start, rules):
        if len(pending) != 1:
            raise ValueError("Normalization record is not paired with exactly one turbine")
        records.append(pending.pop() | {"turbine_id": int(unit)})
        return real_prepare_turbine(unit, ticks, rated_kw, start, rules)

    with patch.object(original, "merge_ticks", merge), patch.object(original, "prepare_turbine", prepare_turbine):
        metadata = original.prepare(root, years=list(years), curve_years=tuple(curve_years), download=False)
    if pending or len(records) != len({r["turbine_id"] for r in records}):
        raise ValueError("Unpaired or duplicated normalization records")
    for name in (THIS, NORMALIZATION_SOURCE):
        metadata["source_sha256"][name] = original.digest(ROOT / name)
    metadata["duplicate_normalization"] = {
        "version": version, "units": sorted(records, key=lambda r: r["turbine_id"]), "turbine_id_source": "preparer unit loop",
        "rule": "One existing whole row contains every known value; preserve timestamps and nonmissing values; no fill, "
                "averaging, clipping or conflict resolution"}
    original.save_json(Path(root) / original.SUBDIR / "metadata/preparation.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("prepare", "audit"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preparation-key", required=True)
    parser.add_argument("--output", type=Path, help="audit JSON (audit only)")
    args = parser.parse_args()
    prep = section(args.protocol, args.preparation_key)
    years, curve_years = [int(y) for y in prep["years"]], [int(y) for y in prep["curve_years"]]
    if args.command == "prepare":
        metadata = prepare(ROOT / prep["root"], years, curve_years, prep["duplicate_normalization"])
        print(json.dumps({"rows": metadata["rows"], "years": metadata["years"], "sha256": metadata["prepared"]["sha256"]}))
        return 0
    if args.output is None:
        raise ValueError("--output is required for audit")
    metadata_path = ROOT / prep["root"] / original.SUBDIR / "metadata/preparation.json"
    result = audit_v31.audit(metadata_path, years, curve_years)
    result["audit_wrapper_sha256"] = original.digest(ROOT / THIS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("passed", "rows", "csv_sha256", "metadata_sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
