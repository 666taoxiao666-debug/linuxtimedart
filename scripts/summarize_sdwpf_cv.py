#!/usr/bin/env python3
"""Summarize best validation checkpoints from an SDWPF CV log.

The parser deliberately ignores pretraining epochs (``Epoch: 1/20``) and
selects fine-tuning checkpoints by the logged validation selection metric.
It never reads a test result.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


RUN_RE = re.compile(r"^===== CV fold=(?P<fold>\d+) seed=(?P<seed>\d+) =====$")
EPOCH_RE = re.compile(
    r"^Epoch:\s*(?P<epoch>\d+),\s*Steps:\s*\d+.*?"
    r"Val MAE\(kW\):\s*(?P<mae>[-+0-9.eE]+)\s+"
    r"Persist MAE\(kW\):\s*(?P<persistence>[-+0-9.eE]+)\s+"
    r"MAE Skill:\s*(?P<mae_skill>[-+0-9.eE]+)%\s+"
    r"RMSE Skill:\s*(?P<rmse_skill>[-+0-9.eE]+)%.*?"
    r"Select\([^)]*\):\s*(?P<selection>[-+0-9.eE]+)"
)


@dataclass(frozen=True)
class CVRun:
    fold: int
    seed: int
    best_epoch: int
    mae_kw: float
    persistence_mae_kw: float
    mae_skill_pct: float
    rmse_skill_pct: float
    selection_value: float

    @property
    def absolute_gain_kw(self) -> float:
        return self.persistence_mae_kw - self.mae_kw


def _values(text: str) -> list[int]:
    return [int(value) for value in re.split(r"[\s,]+", text.strip()) if value]


def parse_summary(path: Path) -> list[CVRun]:
    candidates: dict[tuple[int, int], list[CVRun]] = {}
    current: tuple[int, int] | None = None
    seen_markers: set[tuple[int, int]] = set()

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        marker = RUN_RE.match(line)
        if marker:
            current = (int(marker.group("fold")), int(marker.group("seed")))
            if current in seen_markers:
                raise ValueError(f"duplicate CV run marker: fold={current[0]} seed={current[1]}")
            seen_markers.add(current)
            candidates[current] = []
            continue

        epoch = EPOCH_RE.match(line)
        if epoch and current is not None:
            candidates[current].append(
                CVRun(
                    fold=current[0],
                    seed=current[1],
                    best_epoch=int(epoch.group("epoch")),
                    mae_kw=float(epoch.group("mae")),
                    persistence_mae_kw=float(epoch.group("persistence")),
                    mae_skill_pct=float(epoch.group("mae_skill")),
                    rmse_skill_pct=float(epoch.group("rmse_skill")),
                    selection_value=float(epoch.group("selection")),
                )
            )

    missing_metrics = [key for key, rows in candidates.items() if not rows]
    if missing_metrics:
        raise ValueError(f"CV runs contain no fine-tuning metrics: {missing_metrics}")
    if not candidates:
        raise ValueError(f"no CV runs found in {path}")

    return [
        min(rows, key=lambda row: row.selection_value)
        for _, rows in sorted(candidates.items())
    ]


def validate_matrix(
    runs: list[CVRun], expected_folds: list[int], expected_seeds: list[int]
) -> None:
    actual = {(run.fold, run.seed) for run in runs}
    expected = {(fold, seed) for fold in expected_folds for seed in expected_seeds}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"incomplete CV matrix; missing={missing}, extra={extra}")


def _mean(values) -> float:
    return statistics.fmean(values)


def _sample_sd(values) -> float:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_reports(runs: list[CVRun], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cv_metrics.csv"
    text_path = output_dir / "cv_metrics_summary.txt"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "fold",
                "seed",
                "best_epoch",
                "val_mae_kw",
                "persistence_mae_kw",
                "absolute_gain_kw",
                "mae_skill_pct",
                "rmse_skill_pct",
                "selection_value",
            ]
        )
        for run in runs:
            writer.writerow(
                [
                    run.fold,
                    run.seed,
                    run.best_epoch,
                    f"{run.mae_kw:.6f}",
                    f"{run.persistence_mae_kw:.6f}",
                    f"{run.absolute_gain_kw:.6f}",
                    f"{run.mae_skill_pct:.6f}",
                    f"{run.rmse_skill_pct:.6f}",
                    f"{run.selection_value:.7f}",
                ]
            )

    mean_mae = _mean(run.mae_kw for run in runs)
    mean_persistence = _mean(run.persistence_mae_kw for run in runs)
    gains = [run.absolute_gain_kw for run in runs]
    best_epoch_counts = {
        epoch: sum(run.best_epoch == epoch for run in runs)
        for epoch in sorted({run.best_epoch for run in runs})
    }
    lines = [
        f"RUNS={len(runs)}",
        "FOLDS=" + ",".join(str(value) for value in sorted({r.fold for r in runs})),
        "SEEDS=" + ",".join(str(value) for value in sorted({r.seed for r in runs})),
        f"ALL_BEST_CHECKPOINTS_BEAT_PERSISTENCE={int(all(gain > 0 for gain in gains))}",
        f"EPOCH_ZERO_SELECTED_RUNS={sum(run.best_epoch == 0 for run in runs)}",
        "BEST_EPOCH_COUNTS="
        + ",".join(f"{epoch}:{count}" for epoch, count in best_epoch_counts.items()),
        f"MACRO_VAL_MAE_KW={mean_mae:.6f}",
        f"MACRO_PERSISTENCE_MAE_KW={mean_persistence:.6f}",
        f"MACRO_ABSOLUTE_GAIN_KW={_mean(gains):.6f}",
        f"MACRO_ABSOLUTE_GAIN_SD_KW={_sample_sd(gains):.6f}",
        f"MACRO_MAE_SKILL_RATIO_PCT={100.0 * (1.0 - mean_mae / mean_persistence):.6f}",
        f"MEAN_REPORTED_MAE_SKILL_PCT={_mean(r.mae_skill_pct for r in runs):.6f}",
        f"MEAN_REPORTED_RMSE_SKILL_PCT={_mean(r.rmse_skill_pct for r in runs):.6f}",
    ]

    fold_sds = []
    for fold in sorted({run.fold for run in runs}):
        fold_runs = [run for run in runs if run.fold == fold]
        mae_sd = _sample_sd(run.mae_kw for run in fold_runs)
        fold_sds.append(mae_sd)
        prefix = f"FOLD_{fold}"
        lines.extend(
            [
                f"{prefix}_VAL_MAE_MEAN_KW={_mean(r.mae_kw for r in fold_runs):.6f}",
                f"{prefix}_VAL_MAE_SEED_SD_KW={mae_sd:.6f}",
                f"{prefix}_ABSOLUTE_GAIN_MEAN_KW={_mean(r.absolute_gain_kw for r in fold_runs):.6f}",
                f"{prefix}_MAE_SKILL_MEAN_PCT={_mean(r.mae_skill_pct for r in fold_runs):.6f}",
                f"{prefix}_RMSE_SKILL_MEAN_PCT={_mean(r.rmse_skill_pct for r in fold_runs):.6f}",
            ]
        )
    lines.append(f"MAX_WITHIN_FOLD_SEED_MAE_SD_KW={max(fold_sds):.6f}")
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, text_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-folds", default="")
    parser.add_argument("--expected-seeds", default="")
    args = parser.parse_args()

    runs = parse_summary(args.summary)
    if args.expected_folds or args.expected_seeds:
        if not args.expected_folds or not args.expected_seeds:
            parser.error("--expected-folds and --expected-seeds must be provided together")
        validate_matrix(runs, _values(args.expected_folds), _values(args.expected_seeds))
    csv_path, text_path = write_reports(runs, args.output_dir)
    print(f"[CV] Parsed metrics: {csv_path}")
    print(f"[CV] Aggregate metrics: {text_path}")


if __name__ == "__main__":
    main()
