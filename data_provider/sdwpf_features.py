"""SDWPF feature layout, physics priors, and operating-point context.

Wind power is dominated by absolute wind speed (power curve) and, secondarily,
yaw error.  Reactive power and internal temperature are weak.  Instance
normalization would otherwise erase the wind-speed operating point before the
encoder, so the mixer keeps a named prior and a pre-norm context vector.
"""

from __future__ import annotations

from typing import Sequence


# Relative mixer scales.  These are initial values, not frozen weights.
# Wspd is largest because P ~ v^3 below rated; power is the persistence state;
# yaw is aerodynamic; pitch is a control/curtailment indicator; the rest are weak.
CHANNEL_PRIOR = {
    "Wspd": 2.5,
    "power": 1.5,
    "yaw_sin": 0.8,
    "yaw_cos": 0.8,
    "Wdir_sin": 0.4,
    "Wdir_cos": 0.4,
    "Pab_mean": 0.5,
    "Pab1": 0.2,
    "Pab2": 0.2,
    "Pab3": 0.2,
    "Etmp": 0.25,
    "Itmp": 0.1,
    "Prtv": 0.1,
    "Ndir_sin": 0.15,
    "Ndir_cos": 0.15,
    "Wdir": 0.2,
    "Ndir": 0.15,
}

WEAK_FEATURES = ("Prtv", "Itmp")
REVIN_KEEP_NAMES = ("Wspd",)
CONTEXT_LAST_NAMES = ("power", "yaw_sin", "yaw_cos", "Pab_mean")


def sdwpf_feature_columns(
    *,
    features: str = "MS",
    target: str = "power",
    circular_wind: bool = True,
    collapse_pitch: bool = True,
    physics_features: bool = True,
    drop_weak_features: bool = True,
) -> list[str]:
    """Canonical input column order.  Target is always last for MS."""
    if features == "S":
        return [target]

    columns: list[str] = ["Wspd", "Etmp"]
    drop_weak = bool(physics_features and drop_weak_features)
    if not drop_weak:
        columns.append("Itmp")

    if circular_wind:
        columns.extend(["Wdir_sin", "Wdir_cos"])
        if not physics_features:
            columns.extend(["Ndir_sin", "Ndir_cos"])
    else:
        columns.append("Wdir")
        if not physics_features:
            columns.append("Ndir")

    if physics_features:
        columns.extend(["yaw_sin", "yaw_cos"])

    if collapse_pitch:
        columns.append("Pab_mean")
    else:
        columns.extend(["Pab1", "Pab2", "Pab3"])

    if not drop_weak:
        columns.append("Prtv")

    columns.append(target)
    return columns


def channel_prior_vector(
    feature_columns: Sequence[str],
    physics_init: bool = True,
) -> list[float]:
    if not physics_init:
        return [1.0] * len(feature_columns)
    return [float(CHANNEL_PRIOR.get(name, 0.25)) for name in feature_columns]


def operating_context_dim(feature_columns: Sequence[str]) -> int:
    names = set(feature_columns)
    dim = 0
    if "Wspd" in names:
        dim += 2  # last observation + window mean
    for name in CONTEXT_LAST_NAMES:
        if name in names:
            dim += 1
    return dim


def revin_keep_indices(feature_columns: Sequence[str]) -> list[int]:
    index = {name: i for i, name in enumerate(feature_columns)}
    return [index[name] for name in REVIN_KEEP_NAMES if name in index]
