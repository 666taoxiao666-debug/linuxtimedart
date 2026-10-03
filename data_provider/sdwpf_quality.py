"""History-safe SDWPF sensor repairs; no target labels or split metrics used."""

from __future__ import annotations


def aggregate_blade_pitch(frame, columns, robust=False, disagreement_degrees=5.0):
    """Use the blade median only when same-timestamp pitch sensors disagree.

    Ordinary rows preserve their historical mean.  No power label or future
    sensor observation participates in the repair.
    """
    blades = frame[list(columns)]
    mean = blades.mean(axis=1)
    disagreement = blades.max(axis=1) - blades.min(axis=1)
    outlier = disagreement > float(disagreement_degrees)
    if robust:
        mean = mean.where(~outlier, blades.median(axis=1))
    return mean, outlier
