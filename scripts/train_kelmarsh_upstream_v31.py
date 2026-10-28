#!/usr/bin/env python3
"""v31: upstream MIDAS Open station observations on the v29 all-window TSMixer_last10 recipe (24h -> 4h, Wiki off).

Train 2016-2021 on ALL windows, epoch selection on the 2022 ALL cohort, ONE test evaluation on 2023 reported on the all
(primary, deployment) and normal (secondary, conditional) cohorts; 2024 sealed. Arms: PersistenceHour, PersistenceLast10,
TSMixer_last10_all (reference), TSMixer_upstream_all (+ eleven screened upstream features and the missing flag) and
TSMixer_upstream_all_balanced (same inputs, 0.5 MAE + 0.5 MSE loss): 3 configurations x 3 seeds = 9 fits. The audited v28
trainer ``fit_config``, channel view, model factory and controller bookkeeping are reused unchanged by import. Stages:
  D0 data identity (v31 preparation vs the registered v29 CSV; development station files vs the download manifest)
  -> T1 9 fits -> S1 selection freeze (joint MAE+RMSE score) -> E1 open the 2023 station files and test windows, single
  evaluation -> P1 post-test manifest -> decision -> report. No test file or window is opened before the S1 freeze.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import inspect
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_kelmarsh_multiyear_benchmark as v28  # noqa: E402  (2-thread env, torch, audited trainer)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from utils import kelmarsh_upstream_v31 as u  # noqa: E402

v26run = v28.v26run
PROTOCOL = ROOT / "configs/kelmarsh_upstream_protocol_v31.json"
PROTOCOL_SHA256 = "d323c980e9a6a5293e2c95adc39a4210f56bf2e9953254ce0f66cb34af1ce850"
TAG = "upstream_v31"
LOG_ROOT = v28.LOG_ROOT
POINTER = LOG_ROOT / f"{TAG}_latest.txt"
LOCK = LOG_ROOT / f"{TAG}.lock"
PREFLIGHT_DIR = LOG_ROOT / f"{TAG}_preflight"
LAUNCHER = "scripts/train/Kelmarsh_launch_frozen_protocol.sh"
TRAINING_FILES = tuple(dict.fromkeys(v28.TRAINING_FILES + (
    "utils/kelmarsh_upstream_v31.py", "scripts/train_kelmarsh_upstream_v31.py", "scripts/screen_kelmarsh_upstream_observations.py",
    "utils/kelmarsh_operating_state.py")))
V31_TESTS = ("tests/test_kelmarsh_upstream_v31.py", "tests/test_kelmarsh_v31_duplicates.py")
REGRESSION_TESTS = tuple(dict.fromkeys(v28.V28_TESTS + v28.REGRESSION_TESTS + (
    "tests/test_kelmarsh_operating_state.py", "tests/test_screen_kelmarsh_upstream_observations.py")))
V31_FILES = tuple(dict.fromkeys(TRAINING_FILES + V31_TESTS + REGRESSION_TESTS + (
    LAUNCHER, "scripts/report_kelmarsh_multiyear_benchmark.py", "scripts/prepare_kelmarsh_multiyear.py",
    "scripts/acquire_hourly_wind_datasets.py", "scripts/prepare_kelmarsh_multiyear_v31.py",
    "scripts/audit_kelmarsh_multiyear_v31.py")))
TRAINING_FUNCTIONS = (v28.ChannelView, v28.build_model, v28.verify_trained, v28.fit_config, u.KelmarshUpstreamSeries,
                      u.hourly_upstream, u.parse_station_file, u.arm_protocol)
SYNTHETIC_PREPARATION = v28.SYNTHETIC_PREPARATION
COMPARE_CHUNK = 8192
now = v28.now


def training_code_sha256():
    return u.canonical_sha256({"files": {n: u.text_sha256_lf(ROOT / n) for n in TRAINING_FILES},
                               "functions": {f.__name__: inspect.getsource(f) for f in TRAINING_FUNCTIONS},
                               "torch_major_minor": ".".join(torch.__version__.split(".")[:2])})


def load_protocol(path=PROTOCOL, expected=PROTOCOL_SHA256):
    digest = u.text_sha256_lf(path)
    if expected is not None and digest != expected:
        raise ValueError("Frozen v31 protocol bytes changed; a new protocol needs a new id")
    protocol = u.read_json(path)
    u.validate_protocol(protocol)
    return protocol, digest


def full_source_digest(protocol_sha):
    return u.canonical_sha256({"protocol_sha256": protocol_sha,
                               "sources": {k: u.text_sha256_lf(ROOT / k) for k in V31_FILES}})


def synthetic_prepared_frame(years=(2016, 2017, 2018), days=14, seed=0):
    """v28 synthetic SCADA frame; NEVER real data."""
    return v28.synthetic_prepared_frame(years, days, seed)


def synthetic_upstream(frame, seed=0, missing=0.03):
    """Upstream features loosely tied to the synthetic wind, with missing cells; NEVER real data."""
    rng = np.random.default_rng(seed + 30)
    dates = pd.to_datetime(frame["timestamp_hour_end_utc"], utc=True).dt.tz_localize(None)
    wind = frame.assign(date=dates).groupby("date")["wind_speed_mps"].mean().sort_index()
    labels = pd.date_range(wind.index.min() - pd.Timedelta(hours=48), wind.index.max(), freq="h")
    base = wind.reindex(labels).interpolate(limit_direction="both").to_numpy()
    values = base[:, None] * rng.uniform(.5, 1.5, len(u.UPSTREAM_FEATURES))[None] + rng.normal(0, 1, (len(labels), len(u.UPSTREAM_FEATURES)))
    values[rng.random(values.shape) < missing] = np.nan
    return pd.DataFrame(values, index=labels, columns=u.UPSTREAM_FEATURES)


def synthetic_series(kwargs, seed=0):
    frame = synthetic_prepared_frame(seed=seed)
    return u.KelmarshUpstreamSeries(frame, synthetic_upstream(frame, seed), preparation=SYNTHETIC_PREPARATION, **kwargs)


# =========================================================================== controller
class V31Run(v28.V28Run):
    def __init__(self, directory, *, protocol=None, protocol_sha=None, series_factory=None, gpu_index=0,
                 require_preflight=True, write_pointer=True, lock_path=LOCK, verify_data=True):
        """``series_factory``/``verify_data=False`` are for synthetic tests only (never from the CLI)."""
        if protocol is None:
            protocol, protocol_sha = load_protocol()
        super().__init__(directory, protocol=protocol, protocol_sha=protocol_sha, series_factory=series_factory,
                         gpu_index=gpu_index, require_preflight=require_preflight, write_pointer=write_pointer,
                         lock_path=lock_path, verify_data=verify_data)
        self.upstream_identity = {"verification": "SKIPPED_SYNTHETIC_TEST_ONLY"}
        self.upstream_test_files = {"verification": "SKIPPED_SYNTHETIC_TEST_ONLY"}

    # ------------------------------------------------------------------ bookkeeping (v31 names and fingerprints)
    def claim_pointer(self):
        if not self.write_pointer:
            return
        POINTER.parent.mkdir(parents=True, exist_ok=True)
        if POINTER.exists() and POINTER.read_text().strip() and Path(POINTER.read_text().strip()).resolve() != self.dir.resolve():
            raise ValueError(f"Pointer already registers another v31 run: {POINTER.read_text().strip()}")
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
        """Hourly upstream features from the frozen stations; test-year files only when include_test (after S1)."""
        d, up = self.p["data"], self.p["data"]["upstream"]
        if include_test and not (self.dir / "selection_freeze.json").exists():
            raise PermissionError("Test-year station files require the S1 selection freeze")
        manifest_path = ROOT / up["download_manifest"]
        if u.text_sha256_lf(manifest_path) != up["download_manifest_sha256_lf"]:
            raise ValueError("Registered MIDAS download manifest changed")
        hashes = u.manifest_hashes(u.read_json(manifest_path))
        ids = [int(s["src_id"]) for s in up["stations"]]
        files, _ = u.station_files(ROOT / up["development_root"], ids, up["development_years"], hashes, required=True)
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
        # Hour-ending midnight belongs to the preceding label year. Exclude the next hour.
        if not include_test:
            kwargs["end"] = f"{max(splits['validation']) + 1}-01-01T01:00:00Z"
        if self.series_factory is not None:
            return self.series_factory(kwargs)
        return u.KelmarshUpstreamSeries(ROOT / d["v31_preparation"]["csv"], self.upstream_frame(include_test), **kwargs)

    def stage_d0(self):
        self.set_status(stage="D0", train="RUNNING")
        d = self.p["data"]
        identity = {"verification": "SKIPPED_SYNTHETIC_TEST_ONLY"}
        if self.verify_data:
            csv = ROOT / d["v31_preparation"]["csv"]
            meta_path = csv.parent.parent / "metadata" / "preparation.json"
            meta = u.read_json(meta_path)
            u.verify_preparation(meta, self.p, u.file_sha256(csv))
            ref = d["v29_registered_preparation"]
            ref_csv = ROOT / ref["csv"]
            ref_meta_path = ref_csv.parent.parent / "metadata" / "preparation.json"
            if u.file_sha256(ref_csv) != ref["csv_sha256"] or u.file_sha256(ref_meta_path) != ref["preparation_json_sha256"]:
                raise ValueError("Registered v29 preparation changed")
            rows = u.verify_development_rows(pd.read_csv(csv), pd.read_csv(ref_csv))
            u.verify_power_curves(meta, u.read_json(ref_meta_path))
            identity = {"v31_csv_sha256": meta["prepared"]["sha256"], "v31_preparation_json_sha256": u.file_sha256(meta_path),
                        "v29_rows_identical": rows, "power_curves_identical": True,
                        "prepared_years": meta["years"], "curve_years": meta["curve_years"]}
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

    # ------------------------------------------------------------------ T1
    def stage_t1(self):
        device = self.device()
        cache = {}
        for arm in u.TRAINED_ARMS:
            config = u.arm_config(self.p, arm)
            q = u.arm_protocol(self.p, arm)
            key = (config["train_cohort"], config["selection_cohort"])
            if key not in cache:
                cache[key] = (self.windows("train", key[0]), self.windows("validation", key[1]))
            fit_w, val_w = cache[key]
            fit, selection = v28.ChannelView(fit_w, config["channel_indices"]), v28.ChannelView(val_w, config["channel_indices"])
            for seed in self.p["seeds"]:
                out = self.dir / "runs" / arm / f"s{seed}"
                contract = {"protocol_sha256": self.protocol_sha, "training_code_sha256": self.training_code,
                            "arm": arm, "config": config, "seed": int(seed), "fit_window_hash": u.window_hash(fit.window_starts),
                            "selection_window_hash": u.window_hash(selection.window_starts),
                            "scaler_mean": fit.scaler.mean_.tolist(), "scaler_scale": fit.scaler.scale_.tolist(),
                            "training": q["training"], "device_type": device.type,
                            "determinism": v26run.determinism_record()}
                if not (out / "trained.json").exists():
                    self.check_preflight()
                    if not (out / "last.pt").exists() and len(self.registered_fits()) + 1 > self.p["budget"]["new_fits_max"]:
                        raise ValueError("The 9-fit budget would be exceeded; refuse")
                    self.set_status(stage="T1", train="RUNNING")
                    u.write_json(self.dir / "progress.json", {"arm": arm, "seed": seed, "utc": now()})
                    u.append_event(self.events, "fit_started", arm=arm, seed=seed, device=str(device))
                self.fit_calls.append((arm, seed))
                trained = v28.fit_config(config, q, fit=fit, selection=selection, contract=contract, seed=seed,
                                         device=device, out=out, events=self.events)
                if not trained.get("reused_from_registry"):
                    print(f"[FIT] {arm} s{seed} best_epoch={trained['best_epoch']} "
                          f"val_MAE={trained['validation_best']['mae']:.4f}", flush=True)
                gc.collect()
                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ S1
    def stage_s1(self):
        records, fits = [], {}
        for arm in u.TRAINED_ARMS:
            config = u.arm_config(self.p, arm)
            for seed in self.p["seeds"]:
                folder = self.dir / "runs" / arm / f"s{seed}"
                trained = u.read_json(folder / "trained.json")
                u.verify_registered_files(folder, trained["artifacts"])
                records.append({"arm": arm, "seed": seed, "best_epoch": trained["best_epoch"],
                                "validation": {k: trained["validation_best"][k] for k in ("mae", "rmse", "mse")},
                                "train_cohort": config["train_cohort"], "selection_cohort": config["selection_cohort"]})
                fits[self.rel(folder)] = {"trained_sha256": u.file_sha256(folder / "trained.json"),
                                          "best_sha256": u.file_sha256(folder / "best.pt")}
        choice = u.validation_choice(records, self.p)
        freeze = {"records": records, "fits": fits, "validation_choice": choice, "frozen_utc": now(),
                  "rule": "written before any test-year station file or test window is opened"}
        path = self.dir / "selection_freeze.json"
        if path.exists():
            old = u.read_json(path)
            if u.canonical_sha256({k: old[k] for k in ("records", "fits", "validation_choice")}) != \
                    u.canonical_sha256({k: freeze[k] for k in ("records", "fits", "validation_choice")}):
                raise ValueError("Selected states changed after the S1 freeze")
        else:
            u.write_json(path, freeze)
            self.log("S1 selection freeze", validation_choice=choice["arm"], ranking=choice["ranking"])
        u.append_event(self.events, "S1_frozen", choice=choice["arm"])
        return u.read_json(path)

    # ------------------------------------------------------------------ E1 and P1
    @staticmethod
    def _inputs_digest(windows):
        rows = windows.window_starts[:, None] + np.arange(windows.seq_len)
        digest = hashlib.sha256()
        for begin in range(0, len(rows), COMPARE_CHUNK):
            digest.update(np.ascontiguousarray(windows.data_x[rows[begin:begin + COMPARE_CHUNK]]).tobytes())
        return digest.hexdigest()

    def stage_e1(self, freeze):
        self.set_status(stage="E1", train="RUNNING")
        if u.canonical_sha256(freeze) != u.canonical_sha256(u.read_json(self.dir / "selection_freeze.json")):
            raise ValueError("Evaluation freeze identity changed")
        u.verify_registered_files(self.dir, {name + "/best.pt": entry["best_sha256"] for name, entry in freeze["fits"].items()})
        development = self.series
        self.series = self.load_series(include_test=True)
        folder = self.dir / "evaluation"
        folder.mkdir(exist_ok=True)
        files_path = folder / "upstream_test_files.json"
        if files_path.exists():
            if u.canonical_sha256(u.read_json(files_path)) != u.canonical_sha256(self.upstream_test_files):
                raise ValueError("Test-year station files changed on resume")
        else:
            u.write_json(files_path, self.upstream_test_files)
        for attr in ("scaler_mean", "scaler_scale"):
            if not np.array_equal(getattr(development, attr), getattr(self.series, attr)):
                raise ValueError("Adding the test year changed the frozen training scaler or upstream fill")
        for split in ("train", "validation"):
            for cohort_name in u.COHORTS:
                before, after = development.windows(split, cohort_name), self.series.windows(split, cohort_name)
                pairs = [(before.target_times(), after.target_times()),
                         (before.turbines[before.window_starts], after.turbines[after.window_starts])]
                pairs += list(zip(before.truth_and_anchor(), after.truth_and_anchor()))
                if any(not np.array_equal(a, c) for a, c in pairs) or self._inputs_digest(before) != self._inputs_digest(after):
                    raise ValueError(f"Adding the test year changed the {split}/{cohort_name} queue, inputs or labels")
        u.append_event(self.events, "test_series_opened_after_freeze", windows=self.series.audit["windows"]["test"])
        test_all = self.windows("test", "all")
        test_normal = self.windows("test", "normal")
        normal_mask = np.isin(test_all.window_starts, test_normal.window_starts)
        truth, persistence_hour = test_all.truth_and_anchor()
        issue_row = test_all.window_starts + test_all.seq_len - 1
        cohort = {"truth": truth, "times": test_all.target_times(), "turbines": test_all.turbines[test_all.window_starts],
                  "window_starts": test_all.window_starts, "normal_mask": normal_mask,
                  "PersistenceHour": persistence_hour, "PersistenceLast10": test_all.last10_anchor(),
                  "issue_upstream_missing": self.series.raw[issue_row, list(self.series.channels).index(u.MISSING_CHANNEL)]}
        self._register_npz(folder / "test_cohort.npz", cohort)
        cpu, batch = torch.device("cpu"), self.p["training"]["eval_batch_size"]
        for arm in u.TRAINED_ARMS:
            config = u.arm_config(self.p, arm)
            view = v28.ChannelView(test_all, config["channel_indices"])
            for seed in self.p["seeds"]:
                run = self.dir / "runs" / arm / f"s{seed}"
                saved_path = folder / f"{arm}_s{seed}.npz"
                registry = saved_path.with_suffix(".json")
                identity = {"best_sha256": u.file_sha256(run / "best.pt"), "cohort_sha256": u.file_sha256(folder / "test_cohort.npz")}
                if registry.exists():
                    registered = u.read_json(registry)
                    if registered["identity"] != identity or registered["prediction_sha256"] != u.file_sha256(saved_path):
                        raise ValueError("Registered test prediction identity changed")
                    continue
                trained = u.read_json(run / "trained.json")
                state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)["model"]
                if v26run.state_digest(state) != trained["best_state_sha256"]:
                    raise ValueError(f"{run}/best.pt differs from the frozen selected state")
                model, _ = v28.build_model(config)
                model.load_state_dict(state, strict=True)
                prediction = v26run.predict(model, view, seed, cpu, batch)
                self._register_npz(saved_path, {"prediction": prediction, "window_starts": test_all.window_starts})
                u.write_json(registry, {"identity": identity, "prediction_sha256": u.file_sha256(saved_path)})
        u.append_event(self.events, "E1_test_predicted", windows=int(len(truth)), normal=int(normal_mask.sum()))
        self.stage_p1()
        return self.decide(freeze)

    def stage_p1(self):
        manifest = u.safe(u.post_test_manifest(self.dir, self.series.audit["windows"]))
        path = self.dir / "post_test_manifest.json"
        if path.exists():
            if u.canonical_sha256(u.read_json(path)) != u.canonical_sha256(manifest):
                raise ValueError("Registered post-test manifest would change; refuse")
        else:
            u.write_json(path, manifest)
            self.log("P1 post-test manifest", windows=manifest["windows"])
        u.append_event(self.events, "P1_post_test_manifest", windows=manifest["windows"])

    def decide(self, freeze):
        payload = u.evaluate_registered(self.dir, self.p, freeze["validation_choice"]["arm"])
        path = self.dir / "decision.json"
        if path.exists():
            if u.read_json(path)["rules"] != payload["rules"]:
                raise ValueError("Registered decision would change; refuse")
        else:
            u.write_json(path, payload | {"utc": now()})
        self.log("E1 single test evaluation", rules=payload["rules"], choice=freeze["validation_choice"]["arm"])
        print(f"[RULES] {payload['rules']}", flush=True)
        return payload

    # ------------------------------------------------------------------ finish
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
        self.dir.mkdir(parents=True, exist_ok=True)
        lock = u.acquire_exclusive_lock(self.lock_path)
        try:
            if (self.dir / "completed.json").exists() and not report_only:
                self.freeze_protocol_record()
                u.verify_training_completed(self.dir)
                u.verify_registered_files(self.dir, u.read_json(self.dir / "completed.json"))
                print("[REUSE] v31 already completed; nothing retrained", flush=True)
                return "COMPLETED"
            v26run.configure_determinism()
            (self.dir / "controller.pid").write_text(str(os.getpid()), encoding="utf-8")
            self.freeze_protocol_record()
            self.claim_pointer()
            self.record_sources()
            u.append_event(self.events, "controller_started", argv=sys.argv, report_only=report_only)
            if (self.dir / "training_completed.json").exists():
                u.verify_training_completed(self.dir)
                self.report()
                return "COMPLETED"
            if report_only:
                raise ValueError("--report-only requires a registered TRAIN_COMPLETED run")
            self.stage_d0()
            self.stage_t1()
            freeze = self.stage_s1()
            self.stage_e1(freeze)
            self.finish_training()
            self.report()
            return "COMPLETED"
        finally:
            lock.close()


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


def smoke(p, folder):
    """Synthetic end-to-end smoke of both channel sets and both loss settings (CPU, 2 epochs)."""
    splits = {"train": (2016,), "validation": (2017,), "test": (2018,)}
    series = synthetic_series({"splits": splits, "extras": tuple(p["data"]["extras_in_common_series"])})
    out = {}
    for arm in ("TSMixer_last10_all", "TSMixer_upstream_all_balanced"):
        config = u.arm_config(p, arm)
        q = u.arm_protocol(p, arm)
        q = {**q, "training": {**q["training"], "max_epochs": 2, "scheduler_t_max": 2}}
        fit = v28.ChannelView(series.windows("train", config["train_cohort"]), config["channel_indices"])
        val = v28.ChannelView(series.windows("validation", config["selection_cohort"]), config["channel_indices"])
        trained = v28.fit_config(config, q, fit=fit, selection=val, contract={"smoke": arm}, seed=7,
                                 device=torch.device("cpu"), out=folder / arm)
        out[arm] = {"best_epoch": trained["best_epoch"], "validation_mae": trained["validation_best"]["mae"],
                    "parameters": trained["parameters"]}
    return {"passed": True, "arms": out, "data": "synthetic frame only"}


def preflight(gpu_index=0):
    p, protocol_sha = load_protocol()
    digest = full_source_digest(protocol_sha)
    folder = PREFLIGHT_DIR / f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}_{digest[:12]}"
    folder.mkdir(parents=True, exist_ok=False)
    v26run.configure_determinism()
    checks = {"namespace": protocol_namespace_check(p)}
    try:
        import pytest  # noqa: F401
        command = [sys.executable, "-m", "pytest", "-q", *V31_TESTS, *REGRESSION_TESTS]
    except ImportError:
        command = [sys.executable, "-m", "unittest", *[t.replace("/", ".")[:-3] for t in V31_TESTS + REGRESSION_TESTS]]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (folder / "tests.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    checks["tests"] = {"command": command, "exit_code": result.returncode, "passed": result.returncode == 0,
                       "tail": (result.stdout + result.stderr).splitlines()[-15:]}
    code = subprocess.run(["bash", "-n", LAUNCHER], cwd=ROOT).returncode
    checks["bash_syntax"] = {"exit_code": code, "passed": code == 0}
    try:
        d0 = V31Run(folder / "d0", protocol=p, protocol_sha=protocol_sha, require_preflight=False, write_pointer=False,
                    lock_path=PREFLIGHT_DIR / "preflight.lock")
        d0.dir.mkdir(parents=True)
        d0.freeze_protocol_record()
        d0.record_sources()
        d0.stage_d0()
        checks["D0"] = {"passed": True, **u.read_json(d0.dir / "data_manifest.json")}
    except Exception as error:
        checks["D0"] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    try:
        checks["synthetic_smoke"] = smoke(p, folder / "smoke")
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
    run = V31Run(args.output.resolve(), gpu_index=args.gpu)
    try:
        run.run(report_only=args.report_only)
        return 0
    except BlockingIOError:
        print("[BUSY] Another v31 controller holds the lock; nothing changed", flush=True)
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
