"""SDWPF feature layout, physics priors, and operating-point context.

Wind power is dominated by absolute wind speed (power curve) and, secondarily,
yaw error.  Reactive power and internal temperature are weak.  Instance
normalization would otherwise erase the wind-speed operating point before the
encoder, so the mixer keeps a named prior and a pre-norm context vector.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

ANGLE_MODES = ("legacy_difference", "relative_wdir")


def yaw_degrees(wdir, ndir, angle_mode="legacy_difference"):
    """SDWPF Wdir is already relative to the nacelle (official table 2).

    Legacy subtraction is retained solely to reproduce frozen checkpoints.
    Corrected inputs pair absolute nacelle heading with relative yaw, retaining
    both observed angles without assuming a convention for absolute wind.
    """
    if angle_mode not in ANGLE_MODES:
        raise ValueError("Unsupported SDWPF angle convention: " + str(angle_mode))
    return np.asarray(wdir) if angle_mode == "relative_wdir" else np.asarray(wdir)-np.asarray(ndir)


def check_angle_contract(source_args, target):
    source = source_args.get("sdwpf_angle_mode", "legacy_difference")
    current = getattr(target, "sdwpf_angle_mode", "legacy_difference")
    if source != current:
        raise RuntimeError("SDWPF angle checkpoint mismatch: retrain matched pretrain; "
                           f"checkpoint={source}, target={current}")
    if current == "relative_wdir":
        features = list(getattr(target, "feature_columns", []))
        if source_args.get("feature_columns") != features:
            raise RuntimeError("Corrected-angle checkpoint feature order mismatch")


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
    angle_mode: str = "legacy_difference",
) -> list[str]:
    """Canonical input column order.  Target is always last for MS."""
    if angle_mode not in ANGLE_MODES:
        raise ValueError("Unsupported SDWPF angle convention: " + str(angle_mode))
    if angle_mode == "relative_wdir" and (not physics_features or not circular_wind or features == "S"):
        raise ValueError("relative_wdir requires multivariate circular physics features")
    if features == "S":
        return [target]

    columns: list[str] = ["Wspd", "Etmp"]
    drop_weak = bool(physics_features and drop_weak_features)
    if not drop_weak:
        columns.append("Itmp")

    if circular_wind:
        prefix = "Ndir" if angle_mode == "relative_wdir" else "Wdir"
        columns.extend([prefix+"_sin", prefix+"_cos"])
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
    angle_mode: str = "legacy_difference",
) -> list[float]:
    if not physics_init:
        return [1.0] * len(feature_columns)
    # Preserve the old per-slot initial scales in the matched angle experiment.
    return [0.4 if angle_mode == "relative_wdir" and name in ("Ndir_sin", "Ndir_cos")
            else float(CHANNEL_PRIOR.get(name, 0.25)) for name in feature_columns]


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
