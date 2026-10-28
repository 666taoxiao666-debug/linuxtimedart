#!/usr/bin/env python3
"""v31 source normalization: select an existing dominating duplicate row, never fill cells.

The frozen v30 attempt rejected 2023's repeated empty/partial export rows.
Keep every timestamp and every observed value, refuse conflicts or rows requiring
field-wise coalescing. The original archives and frozen preparer remain unchanged.
"""
import argparse
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts import prepare_kelmarsh_multiyear as original


def canonical_ticks(frames):
    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp", kind="mergesort")
    unique = combined.drop_duplicates().reset_index(drop=True)
    value_columns = list(unique.columns.drop("timestamp"))
    grouped = unique.groupby("timestamp", sort=True)
    known_counts = grouped[value_columns].nunique(dropna=True)
    if (known_counts > 1).any().any():
        raise ValueError("Conflicting nonmissing duplicate values; no label resolution allowed")
    completeness = unique[value_columns].notna().sum(axis=1)
    winner_indices = completeness.groupby(unique.timestamp).idxmax()
    selected = unique.loc[winner_indices.to_numpy()].sort_values("timestamp", kind="mergesort")
    # One original row must already contain every known value. No cell is filled.
    available = known_counts.gt(0).sum(axis=1).to_numpy()
    if not (selected[value_columns].notna().sum(axis=1).to_numpy() == available).all():
        raise ValueError("Complementary partial rows require forbidden cell coalescing")
    selected = selected.reset_index(drop=True)
    audit = {"input_rows": len(combined), "exact_unique_rows": len(unique),
             "output_rows": len(selected), "timestamps_preserved": True,
             "nonmissing_values_preserved": True, "selected_existing_whole_rows": True,
             "cells_filled": 0, "conflicting_nonmissing_values": 0}
    return selected, audit


def prepare(root, years, curve_years):
    records = []

    def merge(frames):
        selected, audit = canonical_ticks(frames)
        # Original prepare iterates units in numeric order; record that order.
        records.append(audit | {"turbine_id": len(records) + 1})
        return selected

    with patch.object(original, "merge_ticks", merge):
        metadata = original.prepare(root, years=years, curve_years=tuple(curve_years), download=False)
    metadata["source_sha256"][str(Path(__file__).resolve().relative_to(ROOT))] = original.digest(Path(__file__))
    metadata["duplicate_normalization"] = {"version": "existing_dominating_row_v1", "units": records,
        "rule": "One existing whole row contains every known value; preserve timestamps and nonmissing values; no fill, averaging, clipping or conflict resolution"}
    original.save_json(Path(root) / original.SUBDIR / "metadata/preparation.json", metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--curve-years", nargs="+", type=int, required=True)
    args = parser.parse_args()
    if args.years != list(range(2016, 2024)) or args.curve_years != list(range(2016, 2020)):
        raise ValueError("v31 authorizes exactly 2016-2023, curve years 2016-2019")
    prepare(args.root, args.years, args.curve_years)
