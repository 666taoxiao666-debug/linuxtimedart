"""Read-only ramp / change-window diagnostics on stored test predictions (no training, no new data).

Compares a candidate arm (e.g. P-upstream) with a reference arm (e.g. P-last10) on identical windows.
The prediction of an arm is the mean over its seeds. The anchor is the last observed hourly power
(PersistenceHour), so "change" means truth or prediction minus the anchor at each lead.

Pre-registered judgement (docs/STAGE_PLAN.md, stage M4 continued; thresholds fixed before Penmanshiel is run):
  M1  change-cell MAE gain: cells (window, h) with h in 2..4 and |truth - anchor| >= 10% rated (ex-post strata).
      Pass: relative gain >= 10% and issue-time block-bootstrap 95% CI lower bound of the kW gain > 0.
  M2  h4 ramp detection: events truth - anchor >= +30% rated (up) or <= -30% rated (down); score = predicted
      change (negated for down). Statistic = mean of up and down AUC. Pass: AUC gain >= 0.05 and CI low > 0.
  M3  fallback: candidate with upstream-missing windows replaced by the reference. Adopt: MAE gain > 0 and CI low > 0.
  Decision over all folds: RAMP_REPLICATED if every fold passes M1 and M2, else RAMP_NOT_REPLICATED;
  ADOPT_FALLBACK if every fold adopts M3, else NO_FALLBACK.
Everything else in the output (strata table, Mincer-Zarnowitz slopes, POD/false warnings) is descriptive.
CIs: moving blocks of consecutive issue times (block 8, 5000 replicates, seed 2024), whole clusters resampled.

Usage:
  python scripts/ramp_diagnostics.py --out OUT.json --rated 2050 \
      --fold T22 COHORT.npz "DIR/P-upstream_s*.npz" "DIR/P-last10_s*.npz" --fold T23 ...
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.paired_forecast_statistics import moving_block_bootstrap_ratio  # noqa: E402

M1_REL_MIN, M2_AUC_MIN = 0.10, 0.05
CHANGE, RAMP, WARN = 0.10, 0.30, 0.10


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_arm(pattern, pred_key="prediction", window_key="window_starts"):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no prediction files match {pattern!r}")
    preds, windows, seeds = [], None, []
    for p in paths:
        with np.load(p, allow_pickle=False) as archive:
            z = {key: archive[key] for key in (window_key, pred_key)}
        w = z[window_key]
        if windows is not None and not np.array_equal(w, windows):
            raise ValueError(f"{p}: window_starts differ between seeds")
        windows = w
        preds.append(np.asarray(z[pred_key], dtype=np.float64))
        m = re.search(r"_s(\d+)\.npz$", p)
        seeds.append(int(m.group(1)) if m else p)
    return np.mean(preds, axis=0), windows, seeds, {p: sha256(p) for p in paths}


def cluster_order(issue):
    """Sort windows by issue time; return order, cluster start offsets and counts (contiguous after sorting)."""
    order = np.argsort(issue, kind="stable")
    _, starts, counts = np.unique(issue[order], return_index=True, return_counts=True)
    return order, starts, counts


def ratio_ci(cell_gain, cell_issue, block, reps, seed, all_issue=None):
    """Window-uniform mean gain over cells with whole-issue-time block bootstrap (utils.paired_forecast_statistics)."""
    if all_issue is not None:
        times = np.unique(all_issue)
        inv = np.searchsorted(times, cell_issue)
        sums = np.bincount(inv, weights=cell_gain, minlength=len(times))
        counts = np.bincount(inv, minlength=len(times)).astype(float)
        if counts.sum() == 0:
            raise ValueError("No change cells for M1")
        block = min(max(1, int(block)), len(times))
        needed = int(math.ceil(len(times) / block))
        rng = np.random.default_rng(int(seed))
        estimates = []
        for _ in range(int(reps)):
            chosen = rng.choice(np.arange(len(times) - block + 1), size=needed, replace=True)
            index = np.concatenate([np.arange(s, s + block) for s in chosen])[:len(times)]
            if counts[index].sum() > 0:
                estimates.append(sums[index].sum() / counts[index].sum())
        if not estimates:
            raise ValueError("No nonempty bootstrap replicates for M1")
        low, high = np.quantile(estimates, [.025, .975])
        return dict(mean_gain_kw=float(sums.sum() / counts.sum()), ci95_low_kw=float(low),
                    ci95_high_kw=float(high), finite_replicates=len(estimates),
                    total_time_clusters=len(times), zero_event_clusters=int((counts == 0).sum()))
    _, inv = np.unique(cell_issue, return_inverse=True)
    sums = np.bincount(inv, weights=cell_gain)
    counts = np.bincount(inv).astype(float)
    return moving_block_bootstrap_ratio(sums, counts, block_length=block, replicates=reps, seed=seed)


def auc(score, label):
    """Mann-Whitney AUC with average ranks for ties; nan if a class is empty."""
    label = np.asarray(label, bool)
    n1 = int(label.sum())
    n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = np.asarray(score)[order]
    first = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    cnt = np.diff(np.r_[first, len(s)])
    ranks = np.repeat(first + (cnt + 1) / 2.0, cnt)  # average 1-based rank within each tie group
    r = np.empty(len(s))
    r[order] = ranks
    return float((r[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ramp_auc(pred_change, true_change, rated):
    up = auc(pred_change, true_change >= RAMP * rated)
    down = auc(-pred_change, true_change <= -RAMP * rated)
    return up, down, 0.5 * (up + down)


def auc_gain_ci(dc, dr, d, issue, rated, block, reps, seed):
    order, starts, counts = cluster_order(issue)
    dc, dr, d = dc[order], dr[order], d[order]
    ends = starts + counts
    n = len(starts)
    block = min(max(1, int(block)), n)
    first = np.arange(n - block + 1)
    needed = int(math.ceil(n / block))
    rng = np.random.default_rng(int(seed))
    est = np.empty(int(reps))
    last_len = n - (needed - 1) * block  # the concatenated block list is truncated to n clusters
    for i in range(len(est)):
        chosen = rng.choice(first, size=needed, replace=True)
        # consecutive clusters are contiguous after sorting, so each block is one slice of windows
        idx = np.concatenate([np.arange(starts[c], ends[c + (block if j < needed - 1 else last_len) - 1])
                              for j, c in enumerate(chosen)])
        est[i] = ramp_auc(dc[idx], d[idx], rated)[2] - ramp_auc(dr[idx], d[idx], rated)[2]
    est = est[np.isfinite(est)]
    low, high = np.quantile(est, [.025, .975])
    return float(low), float(high), int(len(est))


def mae(pred, y):
    return float(np.abs(pred - y).mean())


def warn_scores(pc, d, rated):
    """Descriptive warning skill: warning = |predicted change| >= 10% rated, event = |true change| >= 30% rated."""
    warn = np.abs(pc) >= WARN * rated
    event = np.abs(d) >= RAMP * rated
    same = np.sign(pc) == np.sign(d)
    hit = event & warn & same
    useful = warn & (np.abs(d) >= CHANGE * rated) & same
    return {"pod": float(hit.sum() / max(event.sum(), 1)), "warning_rate": float(warn.mean()),
            "false_warning_ratio": float(1 - useful.sum() / max(warn.sum(), 1)),
            "csi": float(hit.sum() / max(event.sum() + (warn & ~hit).sum(), 1))}


def prepared_missing(prepared, target_times):
    """Recover the frozen raw missing flag at issue time, without refilling features."""
    import pandas as pd
    prepared = Path(prepared)
    manifest = prepared / "prepared_complete.json"
    path = prepared / "upstream.csv"
    expected = json.loads(manifest.read_text())["artifacts"]["upstream.csv"]
    actual = sha256(path)
    if actual != expected:
        raise ValueError("Prepared upstream SHA256 mismatch")
    frame = pd.read_csv(path, index_col="timestamp_hour_end_utc")
    columns = ["upwind_ws", "upwind_dws3", "upwind_minus_local", "local_ws", "local_dws3", "regional_ws",
               "regional_dws3", "local_dp3", "grad_east", "grad_north", "grad_magnitude_change3"]
    if list(frame.columns) != columns:
        raise ValueError("Frozen upstream column identity mismatch")
    frame.index = pd.to_datetime(frame.index, utc=True)
    times = np.asarray(target_times).astype("datetime64[ns]")
    if times.ndim != 2 or times.shape[1] != 4 or not np.all(np.diff(times, axis=1) == np.timedelta64(1, "h")):
        raise ValueError("Expected consecutive hourly targets")
    issue = pd.to_datetime(times[:, 0] - np.timedelta64(1, "h"), utc=True)
    if frame.index.has_duplicates or not issue.isin(frame.index).all():
        raise ValueError("Issue timestamps missing or ambiguous in prepared upstream")
    mask = ~np.isfinite(frame.reindex(issue).to_numpy(dtype=float)).all(axis=1)
    return mask, {str(path): actual, str(manifest): sha256(manifest)}


def analyse_fold(name, cohort_path, cand_glob, ref_glob, rated, keys, block=8, reps=5000, seed=2024, prepared=None):
    with np.load(cohort_path, allow_pickle=False) as archive:
        c = {keys[key]: archive[keys[key]] for key in ("truth", "anchor", "time", "window")}
        if keys["missing"] in archive.files:
            c[keys["missing"]] = archive[keys["missing"]]
    y = np.asarray(c[keys["truth"]], np.float64)
    anchor = np.asarray(c[keys["anchor"]], np.float64)
    missing_available = keys["missing"] in c
    missing = (np.asarray(c[keys["missing"]]).astype(float) > 0.5
               if missing_available else np.zeros(len(y), dtype=bool))
    recovered_hashes = {}
    if prepared is not None:
        recovered, recovered_hashes = prepared_missing(prepared, c[keys["time"]])
        if missing_available and not np.array_equal(missing, recovered):
            raise ValueError("Stored and recovered missing flags disagree")
        missing, missing_available = recovered, True
    issue = np.asarray(c[keys["time"]])[:, 0].astype("datetime64[ns]").astype(np.int64)
    cand, wc, sc, hc = load_arm(cand_glob, keys["pred"], keys["window"])
    ref, wr, sr, hr = load_arm(ref_glob, keys["pred"], keys["window"])
    if not (np.array_equal(wc, wr) and np.array_equal(wc, c[keys["window"]])):
        raise ValueError(f"{name}: window_starts of cohort, candidate and reference must be identical")
    if sorted(map(str, sc)) != sorted(map(str, sr)):
        raise ValueError(f"{name}: candidate and reference seeds differ ({sc} vs {sr})")
    if y.shape != cand.shape or y.shape != ref.shape or y.ndim != 2 or y.shape[1] != 4:
        raise ValueError(f"{name}: expected [window, 4] arrays")
    d = y - anchor
    dc, dr = cand - anchor, ref - anchor
    out = {"fold": name, "windows": int(len(y)), "seeds": sc, "rated_kw": rated,
           "inputs_sha256": {str(cohort_path): sha256(cohort_path), **hc, **hr, **recovered_hashes},
           "missing_source": "frozen_prepared_upstream_at_issue" if prepared else "cohort_or_unavailable",
           "overall_mae": {"candidate": mae(cand, y), "reference": mae(ref, y)}}

    # M1: change cells, h2..h4 (lead index 1..3)
    cells = np.abs(d[:, 1:]) >= CHANGE * rated
    gain_cells = (np.abs(ref - y) - np.abs(cand - y))[:, 1:][cells]
    issue_cells = np.broadcast_to(issue[:, None], cells.shape)[cells]
    ci = ratio_ci(gain_cells, issue_cells, block, reps, seed, all_issue=issue)
    ref_mae = float(np.abs(ref - y)[:, 1:][cells].mean())
    rel = ci["mean_gain_kw"] / ref_mae
    out["M1"] = {"cells": int(cells.sum()), "cell_share": float(cells.mean()), "reference_mae": ref_mae,
                 "candidate_mae": float(np.abs(cand - y)[:, 1:][cells].mean()), "gain_kw": ci["mean_gain_kw"],
                 "relative_gain": rel, "ci95_low_kw": ci["ci95_low_kw"], "ci95_high_kw": ci["ci95_high_kw"],
                 "pass": bool(rel >= M1_REL_MIN and ci["ci95_low_kw"] > 0),
                 "total_time_clusters": ci["total_time_clusters"],
                 "zero_event_clusters": ci["zero_event_clusters"],
                 "finite_replicates": ci["finite_replicates"]}

    # M2: h4 ramp AUC
    uc, dnc, mc = ramp_auc(dc[:, 3], d[:, 3], rated)
    ur, dnr, mr = ramp_auc(dr[:, 3], d[:, 3], rated)
    low, high, ok = auc_gain_ci(dc[:, 3], dr[:, 3], d[:, 3], issue, rated, block, reps, seed)
    out["M2"] = {"event_rate_up": float((d[:, 3] >= RAMP * rated).mean()),
                 "event_rate_down": float((d[:, 3] <= -RAMP * rated).mean()),
                 "candidate": {"auc_up": uc, "auc_down": dnc, "auc_mean": mc},
                 "reference": {"auc_up": ur, "auc_down": dnr, "auc_mean": mr},
                 "gain": mc - mr, "ci95_low": low, "ci95_high": high, "finite_replicates": ok,
                 "pass": bool(mc - mr >= M2_AUC_MIN and low > 0)}

    # M3: fallback to the reference when upstream is missing at issue time
    fb = np.where(missing[:, None], ref, cand)
    win_gain = (np.abs(cand - y) - np.abs(fb - y)).mean(1)
    ci3 = ratio_ci(win_gain, issue, block, reps, seed)
    out["M3"] = {"missing_windows": int(missing.sum()), "missing_share": float(missing.mean()),
                 "mae_in_missing": {"candidate": mae(cand[missing], y[missing]) if missing.any() else None,
                                    "reference": mae(ref[missing], y[missing]) if missing.any() else None},
                 "overall_gain_kw": ci3["mean_gain_kw"], "ci95_low_kw": ci3["ci95_low_kw"],
                 "ci95_high_kw": ci3["ci95_high_kw"],
                 "adopt": bool(missing.any() and ci3["mean_gain_kw"] > 0 and ci3["ci95_low_kw"] > 0)}
    if not missing_available:
        out["M3"] = {"status": "UNAVAILABLE", "reason": "upstream missing mask not stored",
                     "adopt": None}

    # descriptive
    strata, mz = [], []
    for h in range(4):
        q = np.abs(d[:, h]) / rated
        for lo, hi, lab in ((0, CHANGE, "calm"), (CHANGE, RAMP, "change_10_30"), (RAMP, np.inf, "ramp_30")):
            m = (q >= lo) & (q < hi)
            if m.any():
                strata.append({"h": h + 1, "stratum": lab, "share": float(m.mean()),
                               "mae_candidate": mae(cand[m, h], y[m, h]), "mae_reference": mae(ref[m, h], y[m, h])})
        row = {"h": h + 1}
        for lab, pc in (("candidate", dc[:, h]), ("reference", dr[:, h])):
            row[f"mz_slope_{lab}"] = float(np.cov(pc, d[:, h])[0, 1] / np.var(pc, ddof=1))
            row[f"amplitude_slope_{lab}"] = float(np.cov(pc, d[:, h])[0, 1] / np.var(d[:, h], ddof=1))
        mz.append(row)
    out["descriptive"] = {"strata": strata, "slopes": mz,
                          "h4_warning": {"candidate": warn_scores(dc[:, 3], d[:, 3], rated),
                                         "reference": warn_scores(dr[:, 3], d[:, 3], rated)}}
    return out


def _clean(v):
    """JSON without NaN: non-finite floats become null (e.g. an AUC with no events)."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def decide(folds):
    return {"ramp": "RAMP_REPLICATED" if all(f["M1"]["pass"] and f["M2"]["pass"] for f in folds) else "RAMP_NOT_REPLICATED",
            "fallback": ("UNDETERMINED" if any(f["M3"].get("adopt") is None for f in folds)
                         else "ADOPT_FALLBACK" if all(f["M3"]["adopt"] for f in folds) else "NO_FALLBACK")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fold", nargs=4, action="append", required=True, metavar=("NAME", "COHORT", "CAND_GLOB", "REF_GLOB"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--rated", type=float, default=2050.0)
    ap.add_argument("--truth-key", default="truth")
    ap.add_argument("--anchor-key", default="PersistenceHour")
    ap.add_argument("--missing-key", default="issue_upstream_missing")
    ap.add_argument("--time-key", default="times")
    ap.add_argument("--pred-key", default="prediction")
    ap.add_argument("--window-key", default="window_starts")
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--descriptive", action="store_true", help="No adoption or replication decision (T24)")
    ap.add_argument("--prepared", type=Path, help="Existing hash-locked P3 prepared directory, read-only")
    a = ap.parse_args(argv)
    keys = {"truth": a.truth_key, "anchor": a.anchor_key, "missing": a.missing_key, "time": a.time_key,
            "pred": a.pred_key, "window": a.window_key}
    folds = [analyse_fold(n, c, cg, rg, a.rated, keys, a.block, a.reps, a.seed, a.prepared) for n, c, cg, rg in a.fold]
    result = {"script_sha256": sha256(__file__), "thresholds": {"M1_rel_min": M1_REL_MIN, "M2_auc_min": M2_AUC_MIN,
              "change": CHANGE, "ramp": RAMP, "warn": WARN}, "bootstrap": {"block": a.block, "reps": a.reps, "seed": a.seed},
              "folds": folds, "decision": None if a.descriptive else decide(folds),
              "role": "descriptive" if a.descriptive else "development_replication"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(_clean(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for f in folds:
        print(f"{f['fold']}: M1 {f['M1']['relative_gain']*100:.1f}% [{f['M1']['ci95_low_kw']:.1f}, {f['M1']['ci95_high_kw']:.1f}] kW "
              f"pass={f['M1']['pass']} | M2 {f['M2']['gain']:+.3f} [{f['M2']['ci95_low']:+.3f}, {f['M2']['ci95_high']:+.3f}] "
              f"pass={f['M2']['pass']} | M3 gain={f['M3'].get('overall_gain_kw')} kW adopt={f['M3']['adopt']}")
    print(json.dumps(result["decision"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
