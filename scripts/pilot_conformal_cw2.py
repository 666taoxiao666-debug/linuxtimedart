"""CW2: source-only heteroscedastic error-scale conformal (N-P, N-R, N-Rs) + J3 judgment."""
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

save, sha, finite = cb.save, cb.sha, cb.finite
BASE = ("B0", "B1", "B2-A", "B3-A")
SCALE_ARMS = ("N-P", "N-R", "N-Rs")
CANDIDATES = ("N-R", "N-Rs")
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


def scale_features(pack, h, arm):
    p = pack["prediction"][:, [h]]
    finite(p, "prediction")
    if arm == "N-P":
        return p
    feat = pack["features"]
    changes = np.abs(feat[:, CHANGE_INDICES])
    if arm == "N-Rs":
        return np.c_[p, changes]
    if arm == "N-R":
        state = feat[:, STATE_INDICES]
        missing = (~np.isfinite(feat[:, :11]).all(axis=1)).astype(float)[:, None]
        sectors = feat[:, 11:20]
        return np.c_[p, changes, state, missing, sectors]
    raise ValueError(f"Unknown scale arm: {arm}")


def fit_scale_models(source, test, year):
    """Fit sigma_hat on source calibration year only; never uses target or test data."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance

    cb.causal_check(source["times"], test["issue"], year)
    finite(source["truth"], "source truth")
    finite(source["prediction"], "source prediction")
    se = np.abs(source["truth"] - source["prediction"])
    out = {arm: {"scores": [], "sigma_test": [], "floors": [], "importance": []} for arm in SCALE_ARMS}
    for h in range(4):
        err = se[:, h]
        med = float(np.median(err))
        if not np.isfinite(med) or med <= 0:
            raise ValueError(f"Nonpositive or nonfinite source median residual at h={h}")
        floor = 0.1 * med
        for arm in SCALE_ARMS:
            xs = scale_features(source, h, arm)
            xt = scale_features(test, h, arm)
            model = HistGradientBoostingRegressor(
                loss="absolute_error",
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=200,
                l2_regularization=1.0,
                random_state=2024,
                early_stopping=False,
            )
            model.fit(xs, err)
            sig_s = np.maximum(model.predict(xs), floor)
            sig_t = np.maximum(model.predict(xt), floor)
            finite(sig_s, f"{arm} source sigma")
            finite(sig_t, f"{arm} test sigma")
            scores = err / sig_s
            finite(scores, f"{arm} conformal scores")
            out[arm]["scores"].append(scores)
            out[arm]["sigma_test"].append(sig_t)
            out[arm]["floors"].append(floor)
            perm = permutation_importance(
                model, xs, err, scoring="neg_mean_absolute_error", n_repeats=5, random_state=2024
            )
            names = FEATURE_NAMES[arm]
            out[arm]["importance"].append(
                {names[i]: float(perm.importances_mean[i]) for i in range(len(names))}
            )
    return out


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
        sorted_scores = {}
        for arm in SCALE_ARMS:
            if arm in methods:
                sc = np.sort(scale_bundle[arm]["scores"][h], kind="stable")
                sorted_scores[arm] = (sc, np.arange(1, len(sc) + 1), scale_bundle[arm]["sigma_test"][h])
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
                        sc, sc_cum, sig_t = sorted_scores[m]
                        qs, cap = cb.finite_sorted_quantile(sc, sc_cum, 1.0, a)
                        q = float(sig_t[i] * qs)
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
    """4 cohorts must all have coverage in [0.88, 0.92] and Winkler < min(N-P, B3-A); tie -> N-Rs."""
    gains = {}
    eligible = []
    for cand in CANDIDATES:
        rel = []
        ok = True
        for r in cohort_reports:
            m90 = r["mean_90"]
            ref = min(m90["N-P"]["winkler_kw"], m90["B3-A"]["winkler_kw"])
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
    # Tie-breaker prefers N-Rs
    order = {"N-Rs": 0, "N-R": 1}
    selected = min(eligible, key=lambda c: (-gains[c], order[c]))
    return selected, gains


def verdict(means, selected, cis):
    eligible = [m for m in ("B0", "B1", "B2-A") if means[m]["coverage"] >= 0.88] or ["B0", "B1", "B2-A"]
    comparators = ["N-P", "B3-A"]
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
    methods = list(BASE) + (["N-P", "N-R", "N-Rs"] if selected is None else ["N-P", selected])
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
        "source_code": p2.source_code(),
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
                "floors": {arm: bundle[arm]["floors"] for arm in SCALE_ARMS if arm in methods},
                "importance": {arm: bundle[arm]["importance"] for arm in SCALE_ARMS if arm in methods},
            },
        )
        print(f"[CW2] {out.name} seed {seed} complete", flush=True)
    return summarize(out), audit


def judge(out, result, selected):
    comparisons = ["N-P", "B3-A"]
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
        r_j3, _ = phase(root, out / "J3", 2023, True, selected)
        judged = judge(out / "J3", r_j3, selected)
        result = {
            "status": "COMPLETED",
            "selected": selected,
            "gains_pct": gains,
            "decision": "CONTINUE" if judged["pass_all"] else "STOP_CW2",
            "J3": judged,
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
                for n in ["pilot_conformal_c0.py", "pilot_conformal_c0b.py", "pilot_conformal_cw1.py"]
            },
        },
    )
    try:
        if args.smoke:
            from tests.test_pilot_conformal_cw2 import smoke

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
