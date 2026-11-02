"""Verify returned audit artifacts only; never read a dataset or rerun the audit."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[2]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify():
    exported = read(HERE / "export_sha256.json")
    checked = {}
    for name, item in exported["files"].items():
        p = OUT / name
        checked[name] = (p.stat().st_size == item["bytes"] and
                         hashlib.sha256(p.read_bytes()).hexdigest() == item["sha256"])
    assert all(checked.values()), "Transfer hash mismatch"
    source = read(HERE / "launch.json")["source_code"]
    assert all(hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == sha
               for p, sha in source.items()), "Frozen source mismatch"
    old = ROOT / "output/kelmarsh_hourly/d2_feasibility_v34/test_year_registration.json"
    assert (OUT / "test_year_registration.json").read_bytes() == old.read_bytes(), "Holdout registration changed"
    result = read(OUT / "result.json")
    a, b = result["A"], result["B"]
    assert a.get("status") != "UNASSESSABLE" and b.get("status") != "UNASSESSABLE", "Audit technical failure"
    assert a["registration"]["test_year"] == 2022
    protected = [r for r in a["scada"] if r["protected"]]
    assert all(not ({"rows", "timestamp_coverage", "missing_fraction"} & r.keys()) for r in protected)
    assert a["A1"] == bool(a["qualifying_development_years"])
    assert a["A2"] == (a["static"]["consistent"] and not any(r["missing_required"] for r in a["scada"]))
    m = a["midas"]
    stations = [s for s in m["stations"] if s["qualified"]]
    assert len(stations) == m["count"]
    assert sum(s["pressure_capable"] for s in stations) == m["pressure_count"]
    assert sorted({int(s["bearing_deg"] // 90) for s in stations}) == m["quadrants"]
    assert a["A3"] == (m["count"] >= 8 and m["pressure_count"] >= 5 and len(m["quadrants"]) >= 3)
    assert b["windows"] == 16949 and b["threshold_rmse"] == 180.98952
    bounds = b["O_curve"]["pooled"]
    assert all(math.isfinite(v) for v in bounds.values())
    assert 0 <= bounds["mae_lower"] <= bounds["mae_upper"]
    assert 0 <= bounds["rmse_lower"] <= bounds["rmse_upper"]
    decision = ("PASS" if bounds["rmse_upper"] <= 180.98952 else
                "FAIL" if bounds["rmse_lower"] > 180.98952 else "UNASSESSABLE")
    assert b["B1_status"] == decision and b["B1"] == (decision == "PASS")
    assert b["O_curve"]["missing_cells"] == 5 and b["O_curve"]["missing_by_horizon"] == [0, 2, 2, 1]
    assert result["continue_D2"] == (a["A1"] and a["A2"] and a["A3"] and b["B1"])
    return {"all_match": True, "files": len(checked), "bytes": sum(v["bytes"] for v in exported["files"].values()),
            "source_files_unchanged": len(source), "registration_identical": True,
            "protected_scada_members_header_only": len(protected), "A1": a["A1"], "A2": a["A2"], "A3": a["A3"],
            "B1_status": decision, "continue_D2": result["continue_D2"], "checks": checked}


if __name__ == "__main__":
    report = verify()
    with (HERE / "local_verification.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False))
