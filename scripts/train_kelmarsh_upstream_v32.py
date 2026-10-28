#!/usr/bin/env python3
"""v32: confirm the frozen v31 design once on the last sealed Kelmarsh year (train 2016-2022, select on 2023, test 2024).

Same arms, upstream features, 36 stations, 1 h lag, losses, seeds, joint selection score and rules as v31. The v31
controller is reused by subclassing: training (T1), selection freeze (S1), the single test evaluation (E1), the post-test
manifest (P1) and the decision are inherited unchanged; only the data identity (v32 preparation vs the registered v31
CSV), the upstream file roots (2023 station files are development data now), names/fingerprints and the year-aware
report are overridden. No 2024 station file or test window is opened before the S1 freeze.
"""
from __future__ import annotations

import argparse
import datetime as dt
import inspect
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_kelmarsh_upstream_v31 as v31  # noqa: E402  (frozen v31 controller; imports torch)

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from utils import kelmarsh_upstream_v32 as u  # noqa: E402

v28, v26run = v31.v28, v31.v26run
PROTOCOL = ROOT / "configs/kelmarsh_upstream_protocol_v32.json"
PROTOCOL_SHA256 = "b9fbdce22a83dd0e5128aa3d9a489cbcb725c4f298f2feba6fe15f028f7a3588"
TAG = "upstream_v32"
LOG_ROOT = v28.LOG_ROOT
POINTER = LOG_ROOT / f"{TAG}_latest.txt"
LOCK = LOG_ROOT / f"{TAG}.lock"
PREFLIGHT_DIR = LOG_ROOT / f"{TAG}_preflight"
LAUNCHER = v31.LAUNCHER
TRAINING_FILES = tuple(dict.fromkeys(v31.TRAINING_FILES + ("utils/kelmarsh_upstream_v32.py", "scripts/train_kelmarsh_upstream_v32.py")))
V32_TESTS = ("tests/test_kelmarsh_upstream_v32.py",)
REGRESSION_TESTS = tuple(dict.fromkeys(v31.V31_TESTS + v31.REGRESSION_TESTS))
V32_FILES = tuple(dict.fromkeys(TRAINING_FILES + V32_TESTS + REGRESSION_TESTS + v31.V31_FILES + (
    "scripts/prepare_kelmarsh_multiyear_normalized.py",)))
TRAINING_FUNCTIONS = v31.TRAINING_FUNCTIONS
now = v28.now


def training_code_sha256():
    return u.canonical_sha256({"files": {n: u.text_sha256_lf(ROOT / n) for n in TRAINING_FILES},
                               "functions": {f.__name__: inspect.getsource(f) for f in TRAINING_FUNCTIONS},
                               "torch_major_minor": ".".join(torch.__version__.split(".")[:2])})


def load_protocol(path=PROTOCOL, expected=PROTOCOL_SHA256):
    digest = u.text_sha256_lf(path)
    if expected is not None and digest != expected:
        raise ValueError("Frozen v32 protocol bytes changed; a new protocol needs a new id")
    protocol = u.read_json(path)
    u.validate_protocol(protocol)
    return protocol, digest


def full_source_digest(protocol_sha):
    return u.canonical_sha256({"protocol_sha256": protocol_sha,
                               "sources": {k: u.text_sha256_lf(ROOT / k) for k in V32_FILES}})


class V32Run(v31.V31Run):
    def __init__(self, directory, *, protocol=None, protocol_sha=None, series_factory=None, gpu_index=0,
                 require_preflight=True, write_pointer=True, lock_path=LOCK, verify_data=True):
        """``series_factory``/``verify_data=False`` are for synthetic tests only (never from the CLI)."""
        if protocol is None:
            protocol, protocol_sha = load_protocol()
        super().__init__(directory, protocol=protocol, protocol_sha=protocol_sha, series_factory=series_factory,
                         gpu_index=gpu_index, require_preflight=require_preflight, write_pointer=write_pointer,
                         lock_path=lock_path, verify_data=verify_data)

    # ------------------------------------------------------------------ v32 names and fingerprints
    def claim_pointer(self):
        if not self.write_pointer:
            return
        POINTER.parent.mkdir(parents=True, exist_ok=True)
        if POINTER.exists() and POINTER.read_text().strip() and Path(POINTER.read_text().strip()).resolve() != self.dir.resolve():
            raise ValueError(f"Pointer already registers another v32 run: {POINTER.read_text().strip()}")
        POINTER.write_text(str(self.dir.resolve()), encoding="utf-8")

    def record_sources(self):
        manifest = {"utc": now(), "training_code_sha256": training_code_sha256(),
                    "full_source_digest": full_source_digest(self.protocol_sha), "git_head": v28.git_head(),
                    "environment": {"host": socket.gethostname(), "python": sys.version, "platform": platform.platform(),
                                    "torch": torch.__version__, "cuda": torch.cuda.is_available(),
                                    "determinism": v26run.determinism_record()}}
        path = self.dir / "source_manifest.json"
        if not path.exists():
            u.write_json(path, manifest)
        elif u.read_json(path)["full_source_digest"] != manifest["full_source_digest"]:
            u.append_event(self.events, "source_changed_on_resume", current=manifest["full_source_digest"])
        self.training_code, self.source_digest = manifest["training_code_sha256"], manifest["full_source_digest"]

    def check_preflight(self):
        if not self.require_preflight:
            return
        record = PREFLIGHT_DIR / f"passed_{self.source_digest[:16]}.json"
        if not record.exists() or not u.read_json(record).get("passed") or u.read_json(record).get("full_source_digest") != self.source_digest:
            raise ValueError("No passed --preflight-only record for the current code/protocol; run it before any fit")

    # ------------------------------------------------------------------ data
    def upstream_frame(self, include_test):
        """Development station files from every declared development root; test-year files only after S1."""
        d, up = self.p["data"], self.p["data"]["upstream"]
        if include_test and not (self.dir / "selection_freeze.json").exists():
            raise PermissionError("Test-year station files require the S1 selection freeze")
        manifest_path = ROOT / up["download_manifest"]
        if u.text_sha256_lf(manifest_path) != up["download_manifest_sha256_lf"]:
            raise ValueError("Registered MIDAS download manifest changed")
        hashes = u.manifest_hashes(u.read_json(manifest_path))
        ids = [int(s["src_id"]) for s in up["stations"]]
        files = []
        for root in up["development_roots"]:
            found, _ = u.station_files(ROOT / root["root"], ids, root["years"], hashes, required=True)
            files += found
        self.upstream_identity = {"manifest_sha256_lf": up["download_manifest_sha256_lf"], "stations": ids,
                                  "development_files": {f"{e['src_id']}/{e['path'].name}": e["sha256"] for e in files}}
        if include_test:
            test, missing = u.station_files(ROOT / up["test_root"], ids, up["test_years"], hashes, required=False)
            files = files + test
            self.upstream_test_files = {"files": {f"{e['src_id']}/{e['path'].name}": e["sha256"] for e in test},
                                        "missing_station_years": missing}
        observations = u.load_station_observations(files, ids)
        last_year = max(up["test_years"]) if include_test else max(up["development_years"])
        start = pd.Timestamp(d["start_utc"]).tz_convert("UTC").tz_localize(None) - pd.Timedelta(hours=d["input_len"])
        labels = pd.date_range(start, pd.Timestamp(f"{last_year + 1}-01-01 01:00"), freq="h", inclusive="left")
        return u.hourly_upstream(observations, [dict(s) for s in up["stations"]], labels, up["lag_hours"])

    def load_series(self, include_test=False):
        d = self.p["data"]
        if include_test and not (self.dir / "selection_freeze.json").exists():
            raise PermissionError("Test series requires the S1 selection freeze")
        splits = d["splits"] if include_test else {k: v for k, v in d["splits"].items() if k != "test"}
        kwargs = {"splits": splits, "input_len": d["input_len"], "horizon": d["pred_len"], "stride": d["stride"],
                  "extras": tuple(d["extras_in_common_series"]), "start": d["start_utc"]}
        if not include_test:  # hour-ending midnight belongs to the preceding label year
            kwargs["end"] = f"{max(splits['validation']) + 1}-01-01T01:00:00Z"
        if self.series_factory is not None:
            return self.series_factory(kwargs)
        return u.KelmarshUpstreamSeries(ROOT / d[u.PREPARATION_KEY]["csv"], self.upstream_frame(include_test), **kwargs)

    def stage_d0(self):
        self.set_status(stage="D0", train="RUNNING")
        d = self.p["data"]
        identity = {"verification": "SKIPPED_SYNTHETIC_TEST_ONLY"}
        if self.verify_data:
            csv = ROOT / d[u.PREPARATION_KEY]["csv"]
            meta_path = csv.parent.parent / "metadata" / "preparation.json"
            meta = u.read_json(meta_path)
            u.verify_preparation(meta, self.p, u.file_sha256(csv))
            ref = d[u.REFERENCE_KEY]
            ref_csv = ROOT / ref["csv"]
            ref_meta_path = ref_csv.parent.parent / "metadata" / "preparation.json"
            if u.file_sha256(ref_csv) != ref["csv_sha256"] or u.file_sha256(ref_meta_path) != ref["preparation_json_sha256"]:
                raise ValueError("Registered v31 preparation changed")
            rows = u.verify_development_rows(pd.read_csv(csv), pd.read_csv(ref_csv))
            u.verify_power_curves(meta, u.read_json(ref_meta_path))
            identity = {"v32_csv_sha256": meta["prepared"]["sha256"], "v32_preparation_json_sha256": u.file_sha256(meta_path),
                        "v31_rows_identical": rows, "power_curves_identical": True,
                        "prepared_years": meta["years"], "curve_years": meta["curve_years"],
                        "duplicate_normalization": meta["duplicate_normalization"]}
        self.series = self.load_series()
        audit = self.series.audit
        if audit["channels"] != d["channels"]:
            raise ValueError("Series channels differ from the protocol")
        manifest = {"identity": identity, "upstream_identity": self.upstream_identity, "series_audit": audit,
                    "test_windows_instantiated": False}
        keys = ("identity", "upstream_identity", "series_audit")
        path = self.dir / "data_manifest.json"
        if path.exists():
            old = u.read_json(path)
            if u.canonical_sha256({k: old[k] for k in keys}) != u.canonical_sha256({k: manifest[k] for k in keys}):
                raise ValueError("D0 data identity differs from the registered D0 of this run")
        else:
            u.write_json(path, manifest)
            self.log("D0 data identity", identity=identity, windows=audit["windows"],
                     upstream_missing=audit["upstream"]["missing_row_share_by_split"])
        u.append_event(self.events, "D0_verified", windows=audit["windows"])

    # ------------------------------------------------------------------ report
    def report(self):
        self.set_status(stage="REPORT", report="RUNNING")
        try:
            summary = u.build_report(self.dir)
        except Exception as error:
            self.set_status(report="REPORT_FAILED", report_error=f"{type(error).__name__}: {error}")
            raise
        self.set_status(report="REPORT_COMPLETED", report_error=None)
        u.write_json(self.dir / "completed.json", {self.rel(q): u.file_sha256(q) for q in sorted(self.dir.rglob("*"))
                                                   if v28.registrable(q, self.dir)})
        self.set_status(stage="COMPLETED")
        return summary

    def run(self, report_only=False):
        """v31 run order unchanged; only the completed-run message names v32."""
        if (self.dir / "completed.json").exists() and not report_only:
            lock = u.acquire_exclusive_lock(self.lock_path)
            try:
                self.freeze_protocol_record()
                u.verify_training_completed(self.dir)
                u.verify_registered_files(self.dir, u.read_json(self.dir / "completed.json"))
                print("[REUSE] v32 already completed; nothing retrained", flush=True)
                return "COMPLETED"
            finally:
                lock.close()
        return super().run(report_only=report_only)


# =========================================================================== preflight and CLI
def protocol_namespace_check(p):
    own, configs, pointers = p["protocol_id"], [], {}
    for path in sorted((ROOT / "configs").glob("*protocol*.json")):
        if path.resolve() == PROTOCOL.resolve():
            continue
        try:
            if u.read_json(path).get("protocol_id") == own:
                configs.append(str(path))
        except (ValueError, AttributeError):
            continue
    for path in sorted(LOG_ROOT.glob("*latest.txt")):
        if path.resolve() == POINTER.resolve():
            continue
        target = Path(path.read_text(errors="replace").strip())
        marker = (target if target.is_absolute() else ROOT / target) / "protocol.json"
        if marker.exists() and u.read_json(marker).get("protocol_id") == own:
            pointers[str(path)] = str(target)
    return {"protocol_id": own, "configs": configs, "pointers": pointers, "passed": not configs and not pointers}


def preflight(gpu_index=0):
    p, protocol_sha = load_protocol()
    digest = full_source_digest(protocol_sha)
    folder = PREFLIGHT_DIR / f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}_{digest[:12]}"
    folder.mkdir(parents=True, exist_ok=False)
    v26run.configure_determinism()
    checks = {"namespace": protocol_namespace_check(p)}
    try:
        import pytest  # noqa: F401
        command = [sys.executable, "-m", "pytest", "-q", *V32_TESTS, *REGRESSION_TESTS]
    except ImportError:
        command = [sys.executable, "-m", "unittest", *[t.replace("/", ".")[:-3] for t in V32_TESTS + REGRESSION_TESTS]]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (folder / "tests.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    checks["tests"] = {"command": command, "exit_code": result.returncode, "passed": result.returncode == 0,
                       "tail": (result.stdout + result.stderr).splitlines()[-15:]}
    code = subprocess.run(["bash", "-n", LAUNCHER], cwd=ROOT).returncode
    checks["bash_syntax"] = {"exit_code": code, "passed": code == 0}
    try:
        d0 = V32Run(folder / "d0", protocol=p, protocol_sha=protocol_sha, require_preflight=False, write_pointer=False,
                    lock_path=PREFLIGHT_DIR / "preflight.lock")
        d0.dir.mkdir(parents=True)
        d0.freeze_protocol_record()
        d0.record_sources()
        d0.stage_d0()
        checks["D0"] = {"passed": True, **u.read_json(d0.dir / "data_manifest.json")}
    except Exception as error:
        checks["D0"] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    try:
        checks["synthetic_smoke"] = v31.smoke(p, folder / "smoke")
    except Exception as error:
        checks["synthetic_smoke"] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    checks["resources"] = {"gpu": v26run.gpu_snapshot(gpu_index), "cpu_count": os.cpu_count()}
    passed = all(checks[k]["passed"] for k in ("namespace", "tests", "bash_syntax", "D0", "synthetic_smoke"))
    record = {"passed": bool(passed), "full_source_digest": digest, "protocol_sha256": protocol_sha,
              "training_code_sha256": training_code_sha256(), "git_head": v28.git_head(), "utc": now(), "checks": checks}
    u.write_json(folder / "preflight.json", record)
    if passed:
        u.write_json(PREFLIGHT_DIR / f"passed_{digest[:16]}.json", record)
    print(f"[PREFLIGHT] passed={passed} tests={checks['tests']['passed']} D0={checks['D0']['passed']} "
          f"smoke={checks['synthetic_smoke']['passed']} namespace={checks['namespace']['passed']} record={folder}", flush=True)
    return record


def print_status():
    if not POINTER.exists():
        print("STATUS=NOT_STARTED")
        return
    directory = Path(POINTER.read_text().strip())
    print(f"RESULT_DIR={directory}")
    for name in ("status.env", "progress.json"):
        if (directory / name).exists():
            print((directory / name).read_text(encoding="utf-8").strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--print-tag", action="store_true", help="print the launcher tag and exit")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.print_tag:
        print(TAG)
        return 0
    if args.status:
        print_status()
        return 0
    if args.preflight_only:
        return 0 if preflight(args.gpu)["passed"] else 5
    if args.output is None:
        raise ValueError("An explicit independent --output directory is required")
    run = V32Run(args.output.resolve(), gpu_index=args.gpu)
    try:
        run.run(report_only=args.report_only)
        return 0
    except BlockingIOError:
        print("[BUSY] Another v32 controller holds the lock; nothing changed", flush=True)
        return 3
    except Exception as error:
        if args.output.exists():
            status = run.dir / "status.json"
            if not status.exists() or u.read_json(status).get("train") != "TRAIN_COMPLETED":
                run.set_status(train="FAILED", error=f"{type(error).__name__}: {error}")
            u.append_event(run.events, "controller_failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    sys.exit(main())
