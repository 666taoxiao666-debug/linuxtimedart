"""CW3: local normalized/joint conformal; frozen exploratory J3/J4 rules."""
from __future__ import annotations
import argparse
import datetime as dt
import heapq
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import pilot_conformal_c0 as c0
from scripts import pilot_conformal_c0b as cb
from scripts import pilot_conformal_cw1 as cw1
from scripts import pilot_conformal_cw2 as cw2

save, sha, finite = cb.save, cb.sha, cb.finite
BASE = ("B0", "B1", "B2-A", "B3-A")
SCALE_ARMS = ("LN-P", "LN-R", "LJ-P", "LJ-R")
CANDIDATES = ("LN-R", "LJ-R")
MATCHED = {"LN-R": "LN-P", "LJ-R": "LJ-P"}
SELECTION_COHORTS = (
    ("selection_2022", 2022, False),
    ("selection_2023", 2023, False),
    ("J1", 2024, False),
    ("J2", 2024, True),
)
CHANGE_INDICES = (1, 4, 6, 7, 10)  # upwind_dws3, local_dws3, regional_dws3, local_dp3, grad_magnitude_change3
STATE_INDICES = (2, 5)  # upwind_minus_local, regional_ws
FEATURE_NAMES = {
    "N-P": ["pred_power"],
    "N-Rs": [
        "pred_power",
        "abs_upwind_dws3",
        "abs_local_dws3",
        "abs_regional_dws3",
        "abs_local_dp3",
        "abs_grad_magnitude_change3",
    ],
    "N-R": [
        "pred_power",
        "abs_upwind_dws3",
        "abs_local_dws3",
        "abs_regional_dws3",
        "abs_local_dp3",
        "abs_grad_magnitude_change3",
        "upwind_minus_local",
        "regional_ws",
        "upstream_missing",
        "dir_sector_0",
        "dir_sector_1",
        "dir_sector_2",
        "dir_sector_3",
        "dir_sector_4",
        "dir_sector_5",
        "dir_sector_6",
        "dir_sector_7",
        "dir_missing",
    ],
}
RAMP_THRESHOLD_KW = 205.0  # 10% of 2050 kW turbine capacity


def fit_scale_models(source, test, year):
    """Source-year-only HGB; preserve source sigma for local normalization/distance."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    cb.causal_check(source["times"], test["issue"], year)
    se = np.abs(source["truth"] - source["prediction"])
    finite(se, "source residuals")
    out = {}
    for arm in ("P", "R"):
        item = {k: [] for k in ("scores", "sigma_source", "sigma_test", "floors")}
        for h in range(4):
            med = float(np.median(se[:, h]))
            if not np.isfinite(med) or med <= 0:
                raise ValueError("Nonpositive source median residual")
            floor = .1 * med
            model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=200,
                learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=200,
                l2_regularization=1., random_state=2024, early_stopping=False)
            xs = cw2.scale_features(source, h, "N-" + arm)
            xt = cw2.scale_features(test, h, "N-" + arm)
            model.fit(xs, se[:, h])
            ss, st = np.maximum(model.predict(xs), floor), np.maximum(model.predict(xt), floor)
            finite(ss, "source sigma"); finite(st, "target sigma")
            item["scores"].append(se[:, h] / ss)
            item["sigma_source"].append(ss)
            item["sigma_test"].append(st)
            item["floors"].append(floor)
        out[arm] = item
    return out


def joint_neighbors(source_power, source_sigma, test_power, test_sigma, k=500):
    """Exact Euclidean neighbors, deterministic original-index tie breaking.

    A constant source coordinate receives divisor 1 (no source variation).
    Query batches avoid allocating N_test x N_source distances.
    """
    from scipy.spatial import cKDTree
    xs = np.c_[source_power, np.log(source_sigma)]
    xt = np.c_[test_power, np.log(test_sigma)]
    finite(xs, "joint source"); finite(xt, "joint target")
    scale = np.std(xs, axis=0)
    scale = np.where(scale > 0, scale, 1.)
    xs, xt = xs / scale, xt / scale
    k = min(k, len(xs))
    tree = cKDTree(xs)
    for x in xt:
        distance, _ = tree.query(x, k=k, workers=1)
        edge = float(np.max(distance))
        pool = np.asarray(tree.query_ball_point(x, np.nextafter(edge, np.inf)), dtype=int)
        d2 = np.sum((xs[pool] - x) ** 2, axis=1)
        yield pool[np.lexsort((pool, d2))[:k]]


def intervals(source, target_month, test, year, methods, scale_bundle=None):
    """Per-method, per-turbine, per-h ACI state; rule F on normalized or raw scores."""
    cb.causal_check(source["times"], test["issue"], year)
    cb.causal_check(target_month["times"], test["issue"], year)
    for pack in (source, target_month, test):
        finite(pack["truth"], "truth")
        finite(pack["prediction"], "prediction")
    se = np.abs(source["truth"] - source["prediction"])
    te = np.abs(target_month["truth"] - target_month["prediction"])
    error = np.abs(test["truth"] - test["prediction"])
    if scale_bundle is None and any(m in SCALE_ARMS for m in methods):
        scale_bundle = fit_scale_models(source, test, year)
    radius = {m: np.empty(error.shape) for m in methods}
    caps = {m: np.zeros(error.shape, bool) for m in methods}
    order = np.lexsort((test["turbines"], test["issue"]))
    for h in range(4):
        es = np.sort(se[:, h])
        cs = np.arange(1, len(es) + 1)
        q0 = cb.finite_sorted_quantile(es, cs, 1.0, 0.1)
        q1 = cb.finite_quantile(te[:, h], np.ones(len(te)), 1.0, 0.1)
        sorted_power = np.argsort(source["prediction"][:, h], kind="stable")
        power = source["prediction"][sorted_power, h]
        joint = {}
        for arm in ("P", "R"):
            if "LJ-" + arm in methods:
                bundle = scale_bundle[arm]
                joint[arm] = dict(zip(order, joint_neighbors(
                    source["prediction"][:, h], bundle["sigma_source"][h],
                    test["prediction"][order, h], bundle["sigma_test"][h][order])))
        states, queue = {}, {}
        for i in order:
            turbine, now = int(test["turbines"][i]), test["issue"][i]
            events = queue.setdefault(turbine, [])
            while events and events[0][0] <= now:
                _, _, feedback = heapq.heappop(events)
                for key, miss in feedback:
                    states[key] = float(np.clip(states[key] + 0.005 * (0.1 - miss), 0.005, 0.5))
            p = test["prediction"][i, h]
            j = np.searchsorted(power, p)
            pool = sorted_power[max(0, j - 500) : min(len(power), j + 500)]
            nearest = pool[np.argsort(np.abs(source["prediction"][pool, h] - p), kind="stable")[:500]]
            nearest = nearest[np.argsort(se[nearest, h], kind="stable")]
            feedback = []
            for m in methods:
                if m in ("B0", "B1"):
                    q, cap = q0 if m == "B0" else q1
                else:
                    key = (m, turbine)
                    a = states.setdefault(key, 0.1)
                    if m == "B2-A":
                        q, cap = cb.finite_sorted_quantile(es, cs, 1.0, a)
                    elif m == "B3-A":
                        q, cap = cb.finite_sorted_quantile(
                            se[nearest, h], np.arange(1, len(nearest) + 1), 1.0, a
                        )
                    elif m in SCALE_ARMS:
                        arm = m[-1]
                        bundle = scale_bundle[arm]
                        ids = nearest if m.startswith("LN-") else joint[arm][i]
                        scores = bundle["scores"][h][ids] if m.startswith("LN-") else se[ids, h]
                        scores = np.sort(scores, kind="stable")
                        q, cap = cb.finite_sorted_quantile(scores, np.arange(1, len(scores)+1), 1., a)
                        if m.startswith("LN-"):
                            q = float(q * bundle["sigma_test"][h][i])
                    else:
                        raise ValueError(f"Unknown method: {m}")
                    feedback.append((key, float(error[i, h] > q)))
                finite(q, "emitted radius")
                radius[m][i, h], caps[m][i, h] = q, cap
            arrival = test["times"][i, h]
            if arrival != now + np.timedelta64(h + 1, "h"):
                raise ValueError("Incorrect feedback delay")
            heapq.heappush(events, (arrival, int(i), feedback))
    return radius, caps, scale_bundle


def descriptive_strata(radii, caps, truth, prediction, anchor, features):
    """Descriptive only: ramp vs calm windows (|dP| >= 205 kW) and upstream missingness."""
    error = np.abs(truth - prediction)
    prev = np.c_[anchor[:, 0] if anchor.ndim == 2 else anchor, truth[:, :-1]]
    ramp_mask = np.abs(truth - prev) >= RAMP_THRESHOLD_KW
    miss_mask = np.broadcast_to((~np.isfinite(features[:, :11]).all(axis=1))[:, None], truth.shape)
    out = {}
    for m, r in radii.items():
        c = caps[m]
        out[m] = {
            "ramp": cb.checked_metrics(r[ramp_mask], error[ramp_mask], c[ramp_mask]) if ramp_mask.any() else {"n": 0},
            "calm": cb.checked_metrics(r[~ramp_mask], error[~ramp_mask], c[~ramp_mask]) if (~ramp_mask).any() else {"n": 0},
            "upstream_missing": cb.checked_metrics(r[miss_mask], error[miss_mask], c[miss_mask]) if miss_mask.any() else {"n": 0},
            "upstream_observed": cb.checked_metrics(r[~miss_mask], error[~miss_mask], c[~miss_mask]) if (~miss_mask).any() else {"n": 0},
        }
    return out


def select_candidate(cohort_reports):
    """All four cohorts must qualify; exact gain ties prefer LN-R."""
    gains = {}
    eligible = []
    for cand in CANDIDATES:
        rel = []
        ok = True
        for r in cohort_reports:
            m90 = r["mean_90"]
            ref = min(m90[MATCHED[cand]]["winkler_kw"], m90["B3-A"]["winkler_kw"])
            cov = m90[cand]["coverage"]
            wink = m90[cand]["winkler_kw"]
            if not (0.88 <= cov <= 0.92 and wink < ref):
                ok = False
            rel.append(100.0 * (ref - wink) / ref)
        gains[cand] = float(np.mean(rel))
        if ok:
            eligible.append(cand)
    if not eligible:
        return None, gains
    # Frozen tie-breaker prefers LN-R
    order = {"LN-R": 0, "LJ-R": 1}
    selected = min(eligible, key=lambda c: (-gains[c], order[c]))
    return selected, gains


def verdict(means, selected, cis):
    eligible = [m for m in ("B0", "B1", "B2-A") if means[m]["coverage"] >= 0.88] or ["B0", "B1", "B2-A"]
    comparators = [MATCHED[selected], "B3-A"]
    matched = min(comparators, key=lambda m: (means[m]["winkler_kw"], m))
    candidate = means[selected]
    flags = {
        "C1": 0.88 <= candidate["coverage"] <= 0.92,
        "C2": cb.finite_leq(candidate["winkler_kw"], 0.95 * min(means[m]["winkler_kw"] for m in eligible)),
        "C3": cb.finite_leq(candidate["winkler_kw"], 0.97 * means[matched]["winkler_kw"])
        and cis[matched]["ci95_low_kw"] > 0,
    }
    return {
        **flags,
        "pass_all": all(flags.values()),
        "matched_comparator": matched,
        "C2_comparators": eligible,
        "cis": cis,
    }


def summarize(out):
    rows = [json.loads((out / f"seed_{s}.json").read_text()) for s in (2024, 2025, 2026)]
    keys = ("coverage", "width_kw", "winkler_kw", "cap_fraction")
    methods = rows[0]["methods"]
    strata_keys = ("ramp", "calm", "upstream_missing", "upstream_observed")
    strata = {}
    for m in methods:
        strata[m] = {}
        for sk in strata_keys:
            if rows[0]["descriptive_strata"][m][sk].get("n", 1) == 0:
                strata[m][sk] = {"n": 0}
            else:
                strata[m][sk] = {
                    k: float(np.mean([r["descriptive_strata"][m][sk][k] for r in rows])) for k in keys
                }
    imp = {}
    for arm in rows[0]["importance"]:
        imp[arm] = []
        for h in range(4):
            fnames = rows[0]["importance"][arm][h].keys()
            imp[arm].append({f: float(np.mean([r["importance"][arm][h][f] for r in rows])) for f in fnames})
    result = {
        "mean_90": {
            m: {k: float(np.mean([r["methods"][m]["pooled"][k] for r in rows])) for k in keys}
            for m in methods
        },
        "horizons": {
            m: [
                {k: float(np.mean([r["methods"][m]["horizons"][h][k] for r in rows])) for k in keys}
                for h in range(4)
            ]
            for m in methods
        },
        "descriptive_strata": strata,
        "feature_importance": imp,
        "artifacts": {p.name: sha(p) for p in out.glob("*.npz")},
    }
    save(out / "result.json", result)
    return result


def phase(root, out, year, reverse, selected=None):
    import torch

    out.mkdir(parents=True, exist_ok=False)
    data = cw1.Data(root, year, reverse)
    p2 = data.p2
    methods = list(BASE) + (list(SCALE_ARMS) if selected is None else [MATCHED[selected], selected])
    windows = [
        data.windows(data.source),
        data.windows(data.target, month=True),
        data.windows(data.target, test=True),
    ]
    farms = [data.source, data.target, data.target]
    packs = [data.pack(w, f) for w, f in zip(windows, farms)]
    packs[-1]["anchor"] = windows[-1].truth_and_anchor()[1]
    for p in packs[:2]:
        cb.causal_check(p["times"], packs[-1]["issue"], year)
    audit = {
        "year": year,
        "direction": "P-to-K" if reverse else "K-to-P",
        "point_arm": data.arm,
        "point_training": False,
        "calibration_cutoff": f"{year - 1}-12-31T23:00",
        "windows": dict(zip(("source", "target_month", "evaluation"), map(len, windows))),
        "input_hashes": data.identities,
        "source_code": {**p2.source_code(), **{str(Path(x.__file__).resolve().relative_to(ROOT)): sha(x.__file__) for x in (cb, cw1, cw2)}, "scripts/pilot_conformal_cw3.py": sha(__file__)},
        "methods": methods,
        "scale_regressor": {
            "loss": "absolute_error",
            "max_iter": 200,
            "learning_rate": 0.05,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 200,
            "l2_regularization": 1.0,
            "random_state": 2024,
            "early_stopping": False,
            "floor_factor": 0.1,
        },
    }
    save(out / "calibration_freeze.json", audit)
    save(out / "evaluation_started.json", {"utc": p2.now(), "year": year})
    for seed in p2.SEEDS:
        model, _ = p2.trainer.build_model(p2.configuration(data.arm, data.recipe))
        state = torch.load(
            root / data.old["p2_run"] / f"fits/{data.arm}/s{seed}/best.pt",
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(state["model"], strict=True)
        for name, w, p in zip(("source", "target_month", "evaluation"), windows, packs):
            p["prediction"] = p2.trainer.v26run.predict(
                model, p2.view(w, data.arm), seed, torch.device("cpu"), 512
            )
            finite(p["prediction"], "prediction")
            np.savez_compressed(out / f"{name}_s{seed}.npz", **p)
        radii, caps, bundle = intervals(packs[0], packs[1], packs[2], year, methods)
        np.savez_compressed(out / f"intervals_s{seed}.npz", **radii)
        np.savez_compressed(out / f"caps_s{seed}.npz", **caps)
        error = np.abs(packs[2]["truth"] - packs[2]["prediction"])
        save(
            out / f"seed_{seed}.json",
            {
                "seed": seed,
                "methods": cw1.reports(radii, caps, error),
                "descriptive_strata": descriptive_strata(
                    radii, caps, packs[2]["truth"], packs[2]["prediction"], packs[2]["anchor"], packs[2]["features"]
                ),
                "floors": {arm: bundle[arm]["floors"] for arm in bundle},
                "importance": {},
            },
        )
        print(f"[CW3] {out.name} seed {seed} complete", flush=True)
    return summarize(out), audit


def judge(out, result, selected):
    comparisons = [MATCHED[selected], "B3-A"]
    differences = {m: [] for m in comparisons}
    issue = None
    for s in (2024, 2025, 2026):
        with np.load(out / f"evaluation_s{s}.npz") as p, np.load(out / f"intervals_s{s}.npz") as z:
            if issue is None:
                issue = p["issue"]
            else:
                np.testing.assert_array_equal(issue, p["issue"])
            error = np.abs(p["truth"] - p["prediction"])
            losses = {m: 2 * z[m] + 20 * np.maximum(error - z[m], 0) for m in [selected] + comparisons}
            for m in comparisons:
                differences[m].append(losses[m] - losses[selected])
    cis = {
        m: cw1.paired_ci(np.mean(differences[m], axis=0), issue, out / f"bootstrap_{m}.npz")
        for m in comparisons
    }
    result.update(verdict(result["mean_90"], selected, cis))
    result["artifacts"] = {p.name: sha(p) for p in out.glob("*.npz")}
    save(out / "result.json", result)
    return result


def combined_verdict(judgments):
    if len(judgments) != 2:
        raise ValueError("Both J3 and J4 required")
    if all(j["pass_all"] for j in judgments):
        return "EXPLORATORY_SUPPORT_PENDING_CONFIRMATION"
    if all(j["C1"] and j["C2"] and j["cis"][j["matched_comparator"]]["ci95_low_kw"] > 0
           for j in judgments):
        return "SMALL_SIGNIFICANT_EXPLORATORY"
    return "STOP_CW3"


def run(root, out):
    results, audits = [], []
    for name, year, reverse in SELECTION_COHORTS:
        r, a = phase(root, out / name, year, reverse)
        results.append(r)
        audits.append(a)
    selected, gains = select_candidate(results)
    freeze = {
        "selected": selected,
        "gains_pct": gains,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_results": {name: sha(out / f"{name}/result.json") for name, _, _ in SELECTION_COHORTS},
        "source_code": audits[-1]["source_code"],
        "input_hashes": {k: v for a in audits for k, v in a["input_hashes"].items()},
    }
    save(out / "selection_freeze.json", freeze)
    if selected is None:
        result = {
            "status": "COMPLETED",
            "selected": None,
            "decision": "STOP_AT_SELECTION",
            "gains_pct": gains,
            "selection_sha256": sha(out / "selection_freeze.json"),
        }
    else:
        for path, digest in freeze["input_hashes"].items():
            if sha(path) != digest:
                raise ValueError("Input changed after selection")
        for path, digest in freeze["source_code"].items():
            if sha(ROOT / path) != digest:
                raise ValueError("Code changed after selection")
        with (out / "judgment_started.json").open("x") as f:
            json.dump(
                {
                    "selection_sha256": sha(out / "selection_freeze.json"),
                    "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
                f,
            )
        judgments = {}
        for name, year in (("J3", 2023), ("J4", 2022)):
            cohort, _ = phase(root, out / name, year, True, selected)
            judgments[name] = judge(out / name, cohort, selected)
        result = {
            "status": "COMPLETED",
            "selected": selected,
            "gains_pct": gains,
            "decision": combined_verdict(list(judgments.values())),
            "judgments": judgments,
            "evidence_scope": "partially_exposed_exploratory_not_independent_confirmation",
            "selection_sha256": sha(out / "selection_freeze.json"),
        }
    save(out / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    save(
        args.out / "launch.json",
        {
            "argv": sys.argv,
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "script_sha256": sha(__file__),
            "plan_sha256": sha(ROOT / "docs/STAGE_PLAN.md"),
            "frozen_dependencies": {
                n: sha(ROOT / "scripts" / n)
                for n in ["pilot_conformal_c0.py", "pilot_conformal_c0b.py", "pilot_conformal_cw1.py", "pilot_conformal_cw2.py"]
            },
        },
    )
    try:
        if args.smoke:
            from tests.test_pilot_conformal_cw3 import smoke

            result = smoke(args.out)
        else:
            import torch

            torch.set_num_threads(2)
            result = run(args.data_root, args.out)
        save(
            args.out / "completed.json",
            {
                "status": result["status"],
                "seconds": time.monotonic() - start,
                "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
    except Exception as exc:
        save(
            args.out / "failed.json",
            {"type": type(exc).__name__, "error": str(exc), "seconds": time.monotonic() - start},
        )
        raise


if __name__ == "__main__":
    main()
