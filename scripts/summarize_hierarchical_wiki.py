"""Pair Wiki and trend CV results by fold/seed without reading test data."""
import argparse
import csv
import math
import re
from pathlib import Path


def read_metrics(path):
    records = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["fold"]), int(row["seed"]))
            if key in records:
                raise ValueError(f"Duplicate fold/seed in {path}: {key}")
            for metric in ("val_mae_kw", "persistence_mae_kw"):
                if not math.isfinite(float(row[metric])):
                    raise ValueError(f"Non-finite {metric}: {key}")
            records[key] = row
    return records


def compare(wiki_path, trend_path):
    wiki, trend = read_metrics(wiki_path), read_metrics(trend_path)
    if not wiki:
        raise ValueError("No completed Wiki runs")
    rows = []
    for key, row in sorted(wiki.items()):
        if key not in trend:
            raise ValueError(f"No matching trend fold/seed: {key}")
        reference = trend[key]
        if abs(float(row["persistence_mae_kw"]) - float(reference["persistence_mae_kw"])) > 0.002:
            raise ValueError(f"Persistence mismatch for {key}; evaluation populations may differ")
        baseline = float(reference["val_mae_kw"])
        mae = float(row["val_mae_kw"])
        rows.append({"fold": key[0], "seed": key[1], "best_epoch": int(row["best_epoch"]),
                     "trend_mae_kw": baseline, "wiki_mae_kw": mae,
                     "gain_kw": baseline - mae,
                     "gain_pct": 100 * (baseline - mae) / max(baseline, 1e-12)})
    return rows


def attach_training_progress(rows, wiki_dir):
    """Expose degradation hidden by restoring epoch zero; never select oracle."""
    pattern = re.compile(r"^Epoch:\s*(\d+),.*?Val MAE\(kW\):\s*([-+0-9.eE]+).*?Select\([^)]*\):\s*([-+0-9.eE]+)")
    for row in rows:
        path = Path(wiki_dir) / "runs" / f"f{row['fold']}_s{row['seed']}" / "finetune.log"
        records = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                match = pattern.match(line.strip())
                if match and int(match[1]) > 0 and "Phase: adapter_warmup" not in line:
                    record = (int(match[1]), float(match[2]), float(match[3]))
                    if not all(math.isfinite(value) for value in record):
                        raise ValueError(f"Non-finite trained metric in {path}")
                    records.append(record)
        best = min(records, key=lambda item: item[2]) if records else None
        row.update({
            "trained_epochs_reported": len(records),
            "trained_best_epoch": best[0] if best else "",
            "trained_best_mae_kw": best[1] if best else "",
            "last_mae_kw": records[-1][1] if records else "",
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-dir", required=True, type=Path)
    parser.add_argument("--trend-dir", required=True, type=Path)
    args = parser.parse_args()
    rows = compare(args.wiki_dir / "cv_metrics.csv", args.trend_dir / "cv_metrics.csv")
    attach_training_progress(rows, args.wiki_dir)
    with (args.wiki_dir / "wiki_vs_trend.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    trend = sum(r["trend_mae_kw"] for r in rows) / len(rows)
    wiki = sum(r["wiki_mae_kw"] for r in rows) / len(rows)
    summary = (
        f"PAIRED_RUNS={len(rows)}\nTREND_MAE_KW={trend:.6f}\nWIKI_MAE_KW={wiki:.6f}\n"
        f"WIKI_GAIN_VS_TREND_KW={trend - wiki:.6f}\n"
        f"WIKI_GAIN_VS_TREND_PCT={100 * (trend - wiki) / max(trend, 1e-12):.6f}\n"
        f"RUNS_BEATING_TREND={sum(r['gain_kw'] > 0.002 for r in rows)}\n"
        f"EPOCH_ZERO_SELECTED={sum(r['best_epoch'] == 0 for r in rows)}\n"
        "EVALUATION_SPLIT=val\n"
    )
    reported = [r for r in rows if r["trained_epochs_reported"]]
    summary += f"TRAINED_PROGRESS_RUNS={len(reported)}\n"
    if len(reported) == len(rows):
        trained = sum(r["trained_best_mae_kw"] for r in rows) / len(rows)
        last = sum(r["last_mae_kw"] for r in rows) / len(rows)
        summary += (f"TRAINED_BEST_MAE_KW={trained:.6f}\nLAST_MAE_KW={last:.6f}\n"
                    f"TRAINED_BEST_GAIN_VS_TREND_KW={trend - trained:.6f}\n"
                    f"LAST_GAIN_VS_TREND_KW={trend - last:.6f}\n")
    (args.wiki_dir / "wiki_vs_trend.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
