"""Verify copied P1 metadata without accessing any SCADA/observation data."""
from datetime import datetime, timezone
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main():
    transport = read("audit/export_sha256.json")
    for name, record in transport.items():
        path = OUT / name
        assert path.resolve().is_relative_to(OUT.resolve()), name
        assert path.stat().st_size == record["bytes"], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], name
    launch = read("audit/launch.json")
    completed = read("audit/prep_v35_c51baaf_r2.process.completed.json")
    assert completed["exit_code"] == 0 and completed["instance"] == 2
    assert hashlib.sha256((OUT / "audit/export_sha256.json").read_bytes()).hexdigest() == "616cbe9f4a461262b3036523a913ae218774fdc1ffee5ed053767d615fb578da"
    for name, sha in launch["source_code"].items():
        # Compare exact committed blobs: one historical __init__.py contains CRLF
        # in Git itself, while Windows may also convert other working files to CRLF.
        blob = subprocess.check_output(["git", "show", completed["source_commit"] + ":" + name], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == sha, name
    result, series, prep, routing = (read(n) for n in
                                   ("result.json", "series_audit.json", "preparation.json", "routing.json"))
    ids = [1, 2, *range(4, 16)]
    assert series["turbines"] == ids
    assert prep["curve_years"] == [2016, 2017, 2018, 2019]
    assert series["splits"] == {"train": [2016, 2017, 2018, 2019], "validation": [2020], "development_eval": [2021]}
    assert routing["start"] == result["start"] == prep["start"] == "2016-08-01T00:00:00Z"
    assert [s["src_id"] for s in routing["stations"]] == [150, 161, 177, 235, 982, 987, 1023, 1074, 16589, 24125, 30523]
    assert sorted(int(x) for x in prep["power_curves"]) == ids
    assert [x["turbine_id"] for x in prep["duplicate_normalization"]] == ids
    assert not prep["labels_modified"] and not prep["clipping"] and not prep["interpolation"]
    assert result["labels_modified"] == result["labels_clipped"] == result["labels_imputed"] == 0
    assert not result["protected_content_parsed"] and not result["training"] and result["source_unchanged"]
    assert series["prepared_hash_verified"]
    assert result["elapsed_seconds"] <= 3600
    expected_channels = ["Wspd", "Etmp", "Ndir_sin", "Ndir_cos", "Pab_mean", "rotor_speed", "power_last10", "wind_last10",
        "upwind_ws", "upwind_dws3", "upwind_minus_local", "local_ws", "local_dws3", "regional_ws", "regional_dws3",
        "local_dp3", "grad_east", "grad_north", "grad_magnitude_change3", "upstream_missing", "power"]
    gates = {"Q1": series["channels"] == expected_channels and series["turbines"] == ids,
             "Q2": all(v["all"] >= 5000 for v in series["windows"].values()),
             "Q3": all(v["missing_fraction"] is not None and v["missing_fraction"] <= .2
                       for v in series["upstream_missing_by_year"].values()),
             "Q4": all(v["cells_filled"] == 0 and v["nonmissing_values_preserved"] for v in prep["duplicate_normalization"])}
    assert gates == result["gates"]
    assert result["status"] == ("PASS" if all(gates.values()) else "GATE_FAILED")
    record = {"utc": datetime.now(timezone.utc).isoformat(), "verified_files": len(transport),
              "verified_bytes": sum(v["bytes"] for v in transport.values()),
              "source_files_verified_exact_git_blob": len(launch["source_code"]),
              "export_manifest_sha256_verified": True, "gates_recomputed": gates,
              "windows": series["windows"], "start": result["start"],
              "upstream_missing_by_year": series["upstream_missing_by_year"],
              "raw_data_copied_or_opened_locally": False}
    (OUT / "audit/local_verification.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
