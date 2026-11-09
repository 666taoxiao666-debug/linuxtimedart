"""Read-only CW2 acceptance: hashes + metrics + selection & strata verification from saved arrays, no inference.

Usage: python verify_saved.py RUN_DIRECTORY CODE_WORKTREE DATA_ROOT
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric(radius, error, capped):
    assert np.isfinite(radius).all() and (radius >= 0).all()
    score = 2 * radius + 20 * np.maximum(error - radius, 0)
    assert np.isfinite(score).all()
    return dict(
        coverage=float((error <= radius).mean()),
        width_kw=float((2 * radius).mean()),
        winkler_kw=float(score.mean()),
        cap_fraction=float(capped.mean()),
    )


def main(run, code, data):
    assert read(run / "completed.json")["status"] == "COMPLETED"
    assert not (run / "failed.json").exists()
    launch = read(run / "launch.json")
    assert sha(code / "scripts/pilot_conformal_cw2.py") == launch["script_sha256"]
    assert sha(code / "docs/STAGE_PLAN.md") == launch["plan_sha256"]
    for name, digest in launch["frozen_dependencies"].items():
        assert sha(code / "scripts" / name) == digest, name

    freeze = read(run / "selection_freeze.json")
    cohorts = [
        (2022, "selection_2022"),
        (2023, "selection_2023"),
        (2024, "J1"),
        (2024, "J2"),
    ]
    for _, folder in cohorts:
        assert freeze["selection_results"][folder] == sha(run / f"{folder}/result.json")

    checked_inputs, checked_code, checked_arrays = set(), set(), 0
    phase_results = {}
    for year, folder in cohorts:
        phase = run / folder
        audit, result = read(phase / "calibration_freeze.json"), read(phase / "result.json")
        for name, digest in audit["input_hashes"].items():
            assert sha(name) == digest, name
            checked_inputs.add(name)
        for name, digest in audit["source_code"].items():
            assert sha(code / name) == digest, name
            checked_code.add(name)
        for name, digest in result["artifacts"].items():
            assert sha(phase / name) == digest, name
            checked_arrays += 1

        seeds = []
        for seed in (2024, 2025, 2026):
            report = read(phase / f"seed_{seed}.json")
            with np.load(phase / f"evaluation_s{seed}.npz") as p:
                truth, pred, anchor, feat = p["truth"], p["prediction"], p["anchor"], p["features"]
                error = np.abs(pred - truth)
                assert np.isfinite(error).all()
                assert np.array_equal(p["times"], p["issue"][:, None] + np.arange(1, 5) * np.timedelta64(1, "h"))
                first_issue = p["issue"].min()
            for name in ("source", "target_month"):
                with np.load(phase / f"{name}_s{seed}.npz") as p:
                    assert p["times"].max() <= np.datetime64(f"{year - 1}-12-31T23:00")
                    assert p["times"].max() <= first_issue
            prev = np.c_[anchor[:, 0] if anchor.ndim == 2 else anchor, truth[:, :-1]]
            ramp_mask = np.abs(truth - prev) >= 205.0
            miss_mask = np.broadcast_to((~np.isfinite(feat[:, :11]).all(axis=1))[:, None], truth.shape)
            seed_metrics = {}
            with np.load(phase / f"intervals_s{seed}.npz") as radii, np.load(phase / f"caps_s{seed}.npz") as caps:
                assert set(radii.files) == set(report["methods"]) == set(caps.files)
                for method in radii.files:
                    radius, capped = radii[method], caps[method]
                    assert radius.shape == error.shape == capped.shape and capped.dtype == bool
                    computed = metric(radius, error, capped)
                    for key, value in computed.items():
                        np.testing.assert_allclose(value, report["methods"][method]["pooled"][key], rtol=1e-12, atol=1e-9)
                    for h in range(4):
                        for key, value in metric(radius[:, h], error[:, h], capped[:, h]).items():
                            np.testing.assert_allclose(value, report["methods"][method]["horizons"][h][key], rtol=1e-12, atol=1e-9)
                    for sk, mask in (
                        ("ramp", ramp_mask),
                        ("calm", ~ramp_mask),
                        ("upstream_missing", miss_mask),
                        ("upstream_observed", ~miss_mask),
                    ):
                        if mask.any():
                            for key, value in metric(radius[mask], error[mask], capped[mask]).items():
                                np.testing.assert_allclose(
                                    value, report["descriptive_strata"][method][sk][key], rtol=1e-12, atol=1e-9
                                )
                    seed_metrics[method] = computed
            seeds.append(seed_metrics)
        means = {m: {k: float(np.mean([s[m][k] for s in seeds])) for k in seeds[0][m]} for m in seeds[0]}
        for m in means:
            for k in means[m]:
                np.testing.assert_allclose(means[m][k], result["mean_90"][m][k], rtol=1e-12, atol=1e-9)
        phase_results[folder] = result

    # Verify selection eligibility and gains
    candidates = ("N-R", "N-Rs")
    gains = {}
    eligible = []
    cohort_checks = {}
    for cand in candidates:
        rel = []
        ok = True
        cohort_checks[cand] = {}
        for _, folder in cohorts:
            m90 = phase_results[folder]["mean_90"]
            ref = min(m90["N-P"]["winkler_kw"], m90["B3-A"]["winkler_kw"])
            cov = m90[cand]["coverage"]
            wink = m90[cand]["winkler_kw"]
            pass_cov = 0.88 <= cov <= 0.92
            pass_wink = wink < ref
            if not (pass_cov and pass_wink):
                ok = False
            g = 100.0 * (ref - wink) / ref
            rel.append(g)
            cohort_checks[cand][folder] = dict(
                coverage=cov,
                winkler_kw=wink,
                ref_min_control_winkler_kw=ref,
                gain_pct=float(g),
                pass_coverage=bool(pass_cov),
                pass_winkler=bool(pass_wink),
            )
        gains[cand] = float(np.mean(rel))
        np.testing.assert_allclose(gains[cand], freeze["gains_pct"][cand], rtol=1e-12, atol=1e-9)
        if ok:
            eligible.append(cand)

    assert eligible == []
    assert freeze["selected"] is None
    assert not (run / "J3").exists(), "J3 must not be evaluated when STOP_AT_SELECTION"
    assert not (run / "judgment_started.json").exists()

    final = read(run / "result.json")
    assert final["decision"] == "STOP_AT_SELECTION"
    assert final["selected"] is None
    assert final["selection_sha256"] == sha(run / "selection_freeze.json")

    print(
        json.dumps(
            dict(
                status="PASSED",
                arrays_hashed=checked_arrays,
                input_files_hashed=len(checked_inputs),
                source_files_hashed=len(checked_code),
                selected=freeze["selected"],
                decision=final["decision"],
                gains_pct=gains,
                cohort_checks=cohort_checks,
                j3_evaluated=False,
                no_inference=True,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main(*(Path(p) for p in sys.argv[1:]))
