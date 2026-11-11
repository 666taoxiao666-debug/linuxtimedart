"""Read-only CW1 acceptance: hashes + metrics + bootstrap CI from saved arrays, no inference.

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
    return json.loads(Path(path).read_text())


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
    assert sha(code / "scripts/pilot_conformal_cw1.py") == launch["script_sha256"]
    assert sha(code / "docs/STAGE_PLAN.md") == launch["plan_sha256"]
    for name, digest in launch["frozen_dependencies"].items():
        assert sha(code / "scripts" / name) == digest, name

    freeze = read(run / "selection_freeze.json")
    started = read(run / "judgment_started.json")
    assert started["selection_sha256"] == sha(run / "selection_freeze.json")
    assert freeze["utc"] < started["utc"]
    for y in ("2022", "2023"):
        assert freeze["selection_results"][y] == sha(run / f"selection_{y}/result.json")

    checked_inputs, checked_code, checked_arrays = set(), set(), 0
    phases = [
        (2022, "selection_2022"),
        (2023, "selection_2023"),
        (2024, "J1"),
        (2024, "J2"),
    ]
    phase_results = {}
    for year, folder in phases:
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
                error = np.abs(p["prediction"] - p["truth"])
                assert np.isfinite(error).all()
                assert np.array_equal(p["times"], p["issue"][:, None] + np.arange(1, 5) * np.timedelta64(1, "h"))
                first_issue = p["issue"].min()
            for name in ("source", "target_month", "target_year"):
                with np.load(phase / f"{name}_s{seed}.npz") as p:
                    assert p["times"].max() <= np.datetime64(f"{year - 1}-12-31T23:00")
                    assert p["times"].max() <= first_issue
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
                    seed_metrics[method] = computed
            seeds.append(seed_metrics)
        means = {m: {k: float(np.mean([s[m][k] for s in seeds])) for k in seeds[0][m]} for m in seeds[0]}
        for m in means:
            for k in means[m]:
                np.testing.assert_allclose(means[m][k], result["mean_90"][m][k], rtol=1e-12, atol=1e-9)
        phase_results[folder] = result

    # Verify W3 choices and variant selection
    sel = [phase_results["selection_2022"], phase_results["selection_2023"]]
    choices = [
        "M-A"
        if np.mean([r["horizons"]["M-A"][h]["winkler_kw"] for r in sel])
        < np.mean([r["horizons"]["B3-A"][h]["winkler_kw"] for r in sel])
        else "B3-A"
        for h in range(4)
    ]
    assert choices == freeze["W3_choices"]
    variants = ("W1", "W2", "W3", "W4")
    gains = {
        m: float(
            np.mean(
                [
                    100 * (r["mean_90"]["B3-A"]["winkler_kw"] - r["mean_90"][m]["winkler_kw"]) / r["mean_90"]["B3-A"]["winkler_kw"]
                    for r in sel
                ]
            )
        )
        for m in variants
    }
    for m in variants:
        np.testing.assert_allclose(gains[m], freeze["gains_pct"][m], rtol=1e-12, atol=1e-9)
    eligible = [m for m in variants if all(0.88 <= r["mean_90"][m]["coverage"] <= 0.92 for r in sel)]
    chosen = min(eligible, key=lambda m: (-gains[m], variants.index(m)))
    assert chosen == freeze["selected"] == "W3"

    # Verify J1 and J2 bootstrap CI and verdicts
    for name in ("J1", "J2"):
        phase = run / name
        res = phase_results[name]
        with np.load(phase / "bootstrap_B3-A.npz") as z:
            starts = z["block_starts"]
            blen = int(z["block_length"])
            trunc = int(z["truncate_to"])
            idx = (starts[:, :, None] + np.arange(blen)).reshape(len(starts), -1)[:, :trunc]
            est = z["cluster_sums"][idx].sum(axis=1) / z["cluster_counts"][idx].sum(axis=1)
            np.testing.assert_allclose(est, z["estimates"], rtol=1e-12, atol=1e-9)
            low, high = np.quantile(est, [0.025, 0.975])
            np.testing.assert_allclose(low, res["cis"]["B3-A"]["ci95_low_kw"], rtol=1e-12, atol=1e-9)
            np.testing.assert_allclose(high, res["cis"]["B3-A"]["ci95_high_kw"], rtol=1e-12, atol=1e-9)

    final = read(run / "result.json")
    assert final["decision"] == "STOP_CW1"
    print(
        json.dumps(
            dict(
                status="PASSED",
                arrays_hashed=checked_arrays,
                input_files_hashed=len(checked_inputs),
                source_files_hashed=len(checked_code),
                selected=chosen,
                w3_choices=choices,
                gains_pct=gains,
                j1_verdict={k: final["judgments"]["J1"][k] for k in ("C1", "C2", "C3", "pass_all")},
                j2_verdict={k: final["judgments"]["J2"][k] for k in ("C1", "C2", "C3", "pass_all")},
                no_inference=True,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main(*(Path(p) for p in sys.argv[1:]))
