#!/usr/bin/env python3
"""v28: first multi-year Kelmarsh numeric base benchmark (24h -> 4h, six turbines, Wiki off).

Train 2016-2019 (normal-operation windows; DLinear_alltrain on all windows), epoch selection on 2020, ONE test
evaluation on 2021 reported on both the normal (primary, conditional) and all (deployment) cohorts; 2022-2024 sealed.
Arms: PersistenceHour, PersistenceLast10 (no fit) and 6 trained configurations x 3 seeds = 18 fits:
DLinear_normal, DLinear_alltrain, LinearFlat_base, LinearFlat_last10, TSMixer_base, TSMixer_last10.
Every arm uses the SAME windows (one series with the union of channels). Stages:
  D0 data identity (v28 preparation vs registered 2016-2019 development CSV) -> T1 18 fits -> S1 selection freeze
  -> E1 single test evaluation -> decision (predeclared rules) -> report. No test window exists before the S1 freeze.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import inspect
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from data_provider.kelmarsh_multiyear import KelmarshMultiYearSeries  # noqa: E402
from models.DLinear import Model as DLinearModel  # noqa: E402
from models.LinearFlat import Model as LinearFlatModel  # noqa: E402
from models.TSMixerBasicSmall import Model as TSMixerModel  # noqa: E402
from scripts import train_kelmarsh_anchored_residual as v26run  # noqa: E402  (audited numerics helpers)
from utils import kelmarsh_multiyear_benchmark as b  # noqa: E402
from utils.metrics import forecast_metrics  # noqa: E402

PROTOCOL = ROOT / "configs/kelmarsh_multiyear_base_benchmark_protocol_v28.json"
PROTOCOL_SHA256 = "86f7a8df11e42ceb041e08a381e739166b4c88c4038eacbfe05c48963292df3d"
LOG_ROOT = ROOT / "outputs/logs/Kelmarsh"
POINTER = LOG_ROOT / "multiyear_benchmark_latest.txt"
LOCK = LOG_ROOT / "multiyear_benchmark.lock"
PREFLIGHT_DIR = LOG_ROOT / "multiyear_benchmark_preflight"
TRAINING_FILES = ("models/DLinear.py", "models/LinearFlat.py", "models/TSMixerBasicSmall.py", "models/__init__.py",
                  "models/DLinearAnchoredResidual.py", "data_provider/kelmarsh_hourly.py", "data_provider/__init__.py",
                  "data_provider/kelmarsh_multiyear.py", "utils/kelmarsh_enrichment.py", "utils/metrics.py",
                  "scripts/train_kelmarsh_anchored_residual.py", "utils/kelmarsh_backbone_information.py",
                  "utils/kelmarsh_anchored_residual.py", "utils/kelmarsh_multiyear_benchmark.py",
                  "scripts/train_kelmarsh_multiyear_benchmark.py", "utils/paired_forecast_statistics.py")
V28_FILES = TRAINING_FILES + ("scripts/report_kelmarsh_multiyear_benchmark.py", "utils/paired_forecast_statistics.py",
                              "scripts/train/Kelmarsh_launch_multiyear_benchmark.sh", "tests/test_kelmarsh_multiyear_benchmark.py")
V28_TESTS = ("tests/test_kelmarsh_multiyear_benchmark.py",)
REGRESSION_TESTS = ("tests/test_kelmarsh_multiyear_preprocessing.py", "tests/test_tsmixer_basic_small.py",
                    "tests/test_hourly_wind_acquisition.py")
V28_FILES += REGRESSION_TESTS + ("scripts/prepare_kelmarsh_multiyear.py", "scripts/acquire_hourly_wind_datasets.py")
VOLATILE = {"status.json", "status.env", "progress.json", "launch.log", "launcher.pid", "controller.pid",
            "events.jsonl", "experiment_log.md", "fit_progress.json"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def registrable(path, directory):
    path = Path(path)
    return (path.is_file() and path.name not in VOLATILE and not path.name.endswith(".tmp")
            and path != Path(directory) / "completed.json")


class SimulatedInterrupt(RuntimeError):
    """Tests only."""


# =========================================================================== data views and models
class ChannelView(Dataset):
    """Same windows, a subset of channels (power stays last); scaler restricted accordingly."""

    def __init__(self, windows, indices):
        self.windows, self.indices = windows, np.asarray(indices, dtype=np.int64)
        if self.indices[-1] != len(windows.feature_columns) - 1:
            raise ValueError("Power must stay the last channel")
        self.window_starts = windows.window_starts
        self.scaler = SimpleNamespace(mean_=np.asarray(windows.scaler.mean_)[self.indices],
                                      scale_=np.asarray(windows.scaler.scale_)[self.indices])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        x, y, mx, my = self.windows[index]
        return x[:, self.indices], y[:, self.indices], mx, my

    def truth_and_anchor(self):
        return self.windows.truth_and_anchor()


def build_model(config):
    family, channels = config["family"], config["channels"]
    if family == "DLinear":
        model = DLinearModel(SimpleNamespace(**config["args"], seq_len=24, pred_len=4, enc_in=channels))
    elif family == "LinearFlat":
        model = LinearFlatModel(seq_len=24, channels=channels, pred_len=4)
    elif family == "TSMixerBasicSmall":
        a = config["args"]
        model = TSMixerModel(seq_len=24, channels=channels, pred_len=4, blocks=a["blocks"], ff_dim=a["ff_dim"],
                             dropout=a["dropout"], norm_eps=a["norm_eps"], target_slot=channels - 1, input_mode="full")
    else:
        raise ValueError(f"Unknown family {family!r}")
    count = int(sum(q.numel() for q in model.parameters() if q.requires_grad))
    if count != int(config["expected_parameters"]):
        raise ValueError(f"{config['arm']}: {count} parameters, protocol requires {config['expected_parameters']}")
    return model, count


# =========================================================================== trainer (v25 fit_arm semantics)
def verify_trained(out, contract):
    trained = b.read_json(out / "trained.json")
    mismatch = b.contract_mismatches(trained["contract"], contract)
    if mismatch:
        raise ValueError(f"Registered fit {out} has a different contract ({mismatch})")
    b.verify_registered_files(out, trained["artifacts"])
    return trained


def fit_config(config, p, *, fit, selection, contract, seed, device, out, events=None, stop_after_epoch=None):
    """seed_all -> construct -> epoch-0 assessment (selectable) -> epochs with loader(seed+epoch) shuffling,
    (1-w)|e|+w e^2 on standardized power, clip, AdamW, strict-MAE best, patience at the top of each epoch, cosine step
    after each epoch. last.pt carries the selected state (atomic source of truth on resume)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "trained.json").exists():
        return verify_trained(out, contract) | {"reused_from_registry": True}
    cfg = p["training"]
    weight = float(cfg["loss_mse_weight"])
    if abs((1 - weight) - float(cfg["loss_mae_weight"])) > 1e-12:
        raise ValueError("Loss weights must be (1-w, w)")
    device = torch.device(device)
    v26run.seed_all(seed)
    model, count = build_model(config)
    init_state = v26run.cpu_state(model)
    init_digest = v26run.state_digest(init_state)
    init_path, last_path, best_path = out / "init.pt", out / "last.pt", out / "best.pt"
    if init_path.exists():
        saved = torch.load(init_path, map_location="cpu", weights_only=False)
        if saved["state_sha256"] != init_digest or b.contract_mismatches(saved["contract"], contract):
            raise ValueError("Saved initial state differs from the deterministic construction")
    else:
        v26run.atomic_save({"model": init_state, "state_sha256": init_digest, "contract": contract}, init_path)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["scheduler_t_max"], eta_min=cfg["scheduler_eta_min"])
    truth, _ = selection.truth_and_anchor()
    rated = p["data"]["rated_power_kw"]

    def assess():
        begin = time.monotonic()
        prediction = v26run.predict(model, selection, seed, device, cfg["eval_batch_size"])
        if prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise ValueError("Nonfinite or misaligned selection predictions")
        metrics = forecast_metrics(prediction, truth, rated_power=rated)
        return metrics["mae"], metrics, time.monotonic() - begin

    def save_last(epoch):
        v26run.atomic_save({"model": v26run.cpu_state(model), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "contract": contract, "epoch": epoch, "best": best,
                            "best_epoch": best_epoch, "bad_epochs": bad, "best_model": best_state, "history": history,
                            "timing": timing, "rng": v26run.capture_rng(device)}, last_path)
        b.write_json(out / "fit_progress.json", {"last_completed_epoch": epoch, "epoch_in_progress": None})

    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if b.contract_mismatches(checkpoint["contract"], contract):
            raise ValueError("Interrupted fit identity changed")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history, start = checkpoint["history"], int(checkpoint["epoch"])
        best, best_epoch, bad, timing = checkpoint["best"], checkpoint["best_epoch"], checkpoint["bad_epochs"], checkpoint["timing"]
        best_state = checkpoint["best_model"]
        v26run.restore_rng(checkpoint["rng"], device)
        v26run.atomic_save({"model": best_state, "epoch": best_epoch, "contract": contract, "selection_mae_kw": best}, best_path)
        if events is not None:
            b.append_event(events, "fit_resumed", out=str(out), from_epoch=start)
    else:
        score, metrics, seconds = assess()
        history = [{"epoch": 0, "selection": metrics, "assess_seconds": seconds, "lr": optimizer.param_groups[0]["lr"]}]
        best, best_epoch, bad, start, best_state = score, 0, 0, 0, v26run.cpu_state(model)
        timing = {"train_seconds": 0.0, "assess_seconds": seconds}
        v26run.atomic_save({"model": best_state, "epoch": 0, "contract": contract, "selection_mae_kw": best}, best_path)
        save_last(0)
    for epoch in range(start + 1, int(cfg["max_epochs"]) + 1):
        if bad >= int(cfg["patience"]):
            break
        if stop_after_epoch is not None and epoch > stop_after_epoch:
            raise SimulatedInterrupt(f"after epoch {epoch - 1}")
        b.write_json(out / "fit_progress.json", {"last_completed_epoch": epoch - 1, "epoch_in_progress": epoch})
        model.train()
        total, batches, lr, begin = 0., 0, optimizer.param_groups[0]["lr"], time.monotonic()
        for x, y, _, _ in v26run.make_loader(fit, int(seed) + epoch, shuffle=True, batch=cfg["batch_size"], device=device):
            x, y = x.float().to(device), y.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            difference = model(x)[..., -1] - y[..., -1]
            loss = (1 - weight) * difference.abs().mean() + weight * difference.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            optimizer.step()
            total += loss.item()
            batches += 1
        train_seconds = time.monotonic() - begin
        score, metrics, assess_seconds = assess()
        timing["train_seconds"] += train_seconds
        timing["assess_seconds"] += assess_seconds
        history.append({"epoch": epoch, "train_loss": total / batches, "lr": lr, "selection": metrics,
                        "train_seconds": train_seconds, "assess_seconds": assess_seconds})
        if score < best:
            best, best_epoch, bad, best_state = score, epoch, 0, v26run.cpu_state(model)
            v26run.atomic_save({"model": best_state, "epoch": epoch, "contract": contract, "selection_mae_kw": best}, best_path)
        else:
            bad += 1
        scheduler.step()
        save_last(epoch)
        b.write_json(out / "training_history.json", history)
        if events is not None:
            b.append_event(events, "epoch", out=out.name, seed=seed, epoch=epoch, train_loss=total / batches,
                           validation_mae_kw=score, best_epoch=best_epoch)
        print(f"[EPOCH] {out.parent.name}/{out.name} epoch={epoch} train_loss={total / batches:.6f} "
              f"validation_mae={score:.4f} best_epoch={best_epoch}", flush=True)
    b.write_json(out / "training_history.json", history)
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    if int(best_checkpoint["epoch"]) != int(best_epoch):
        raise ValueError("best.pt does not register the selected epoch")
    model.load_state_dict(best_checkpoint["model"], strict=True)
    prediction = v26run.predict(model, selection, seed, device, cfg["eval_batch_size"])
    replay = forecast_metrics(prediction, truth, rated_power=rated)
    recorded = history[best_epoch]["selection"]
    if abs(replay["mae"] - recorded["mae"]) > 1e-6:
        raise ValueError("best.pt does not reproduce its registered validation MAE")
    np.savez_compressed(out / "validation_predictions.npz", prediction=prediction, truth=truth,
                        window_starts=selection.window_starts)
    trained = {"contract": contract, "arm": config["arm"], "seed": int(seed), "best_epoch": int(best_epoch),
               "epochs_completed": len(history) - 1, "validation_best": recorded,
               "validation_last": history[-1]["selection"], "parameters": count, "init_state_sha256": init_digest,
               "best_state_sha256": v26run.state_digest(best_checkpoint["model"]), "timing": timing,
               "device": str(device), "completed_utc": now(),
               "artifacts": {name: b.file_sha256(out / name) for name in
                             ("init.pt", "best.pt", "last.pt", "training_history.json", "validation_predictions.npz")}}
    b.write_json(out / "trained.json", trained)
    return trained


TRAINING_FUNCTIONS = (ChannelView, build_model, verify_trained, fit_config)


def training_code_sha256():
    return b.canonical_sha256({"files": {n: b.text_sha256_lf(ROOT / n) for n in TRAINING_FILES},
                               "functions": {f.__name__: inspect.getsource(f) for f in TRAINING_FUNCTIONS},
                               "torch_major_minor": ".".join(torch.__version__.split(".")[:2])})


def load_protocol(path=PROTOCOL, expected=PROTOCOL_SHA256):
    digest = b.text_sha256_lf(path)
    if expected is not None and digest != expected:
        raise ValueError("Frozen v28 protocol bytes changed; a new protocol needs a new id")
    protocol = b.read_json(path)
    b.validate_protocol(protocol)
    return protocol, digest


def full_source_digest(protocol_sha):
    return b.canonical_sha256({"protocol_sha256": protocol_sha,
                               "sources": {k: b.text_sha256_lf(ROOT / k) for k in V28_FILES}})


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as error:  # informational
        return f"unavailable: {type(error).__name__}"


# =========================================================================== synthetic data (smoke and tests only)
def synthetic_prepared_frame(years=(2016, 2017, 2018), days=14, seed=0):
    """Enriched-CSV-like frame with the columns the provider reads; NEVER real data."""
    rng = np.random.default_rng(seed)
    parts = []
    for year in years:
        dates = pd.date_range(f"{year}-12-{31 - days + 1:02d} 01:00", periods=days * 24, freq="h", tz="UTC")
        t = np.arange(len(dates), dtype=float)
        for unit in range(1, 7):
            wind = 7 + 3 * np.sin(t / 9 + unit) + rng.normal(0, .3, len(t))
            power = np.clip(2050 / (1 + np.exp(-(wind - 8) / 1.4)) + rng.normal(0, 30, len(t)), -10, 2060)
            parts.append(pd.DataFrame({
                "timestamp_hour_end_utc": dates.astype(str), "turbine_id": unit, "wind_speed_mps": wind,
                "ambient_temp_c": 5 + np.sin(t / 24), "nacelle_position_deg": (t * 7) % 360, "pitch_a_deg": 1.,
                "pitch_b_deg": 1.1, "pitch_c_deg": .9, "rotor_speed_rpm": 8 + wind / 2, "power_kw": power,
                "power_last10_kw": power + rng.normal(0, 40, len(t)), "wind_last10_mps": wind + rng.normal(0, .3, len(t)),
                "normal_operation": rng.random(len(t)) > .04}))
    return pd.concat(parts, ignore_index=True)


SYNTHETIC_PREPARATION = {"schema_version": 2, "curve_years": [2016], "prepared": {"sha256": "synthetic"}}


# =========================================================================== controller
class V28Run:
    def __init__(self, directory, *, protocol=None, protocol_sha=None, series_factory=None, gpu_index=0,
                 require_preflight=True, write_pointer=True, lock_path=LOCK, verify_data=True):
        """``series_factory``/``verify_data=False`` are for synthetic tests only (never from the CLI)."""
        self.dir = Path(directory)
        if protocol is None:
            protocol, protocol_sha = load_protocol()
        self.p, self.protocol_sha = protocol, protocol_sha
        self.series_factory, self.verify_data = series_factory, verify_data
        self.gpu_index, self.require_preflight, self.write_pointer, self.lock_path = gpu_index, require_preflight, write_pointer, lock_path
        self.events = self.dir / "events.jsonl"
        self.fit_calls = []

    # ------------------------------------------------------------------ bookkeeping
    def rel(self, path):
        return str(Path(path).relative_to(self.dir)).replace("\\", "/")

    def set_status(self, **fields):
        path = self.dir / "status.json"
        status = b.read_json(path) if path.exists() else {"train": "NOT_STARTED", "report": "PENDING", "stage": None}
        status.update(fields, updated_utc=now())
        b.write_json(path, status)
        (self.dir / "status.env").write_text(f"TRAIN={status['train']}\nREPORT={status['report']}\nSTAGE={status.get('stage')}\n",
                                             encoding="utf-8")
        return status

    def log(self, title, **items):
        with (self.dir / "experiment_log.md").open("a", encoding="utf-8") as handle:
            handle.writelines([f"\n## {now()} {title}\n"] + [f"- {k}: {v}\n" for k, v in items.items()])

    def freeze_protocol_record(self):
        record = {"protocol_id": self.p["protocol_id"], "protocol_sha256_lf": self.protocol_sha, "protocol": self.p}
        path = self.dir / "protocol.json"
        if path.exists():
            if b.canonical_sha256(b.read_json(path)) != b.canonical_sha256(record):
                raise ValueError("Frozen protocol changed on resume")
        else:
            b.write_json(path, record)

    def claim_pointer(self):
        if not self.write_pointer:
            return
        POINTER.parent.mkdir(parents=True, exist_ok=True)
        if POINTER.exists() and POINTER.read_text().strip() and Path(POINTER.read_text().strip()).resolve() != self.dir.resolve():
            raise ValueError(f"Pointer already registers another v28 run: {POINTER.read_text().strip()}")
        POINTER.write_text(str(self.dir.resolve()), encoding="utf-8")

    def record_sources(self):
        manifest = {"utc": now(), "training_code_sha256": training_code_sha256(),
                    "full_source_digest": full_source_digest(self.protocol_sha), "git_head": git_head(),
                    "environment": {"host": socket.gethostname(), "python": sys.version, "platform": platform.platform(),
                                    "torch": torch.__version__, "cuda": torch.cuda.is_available(),
                                    "determinism": v26run.determinism_record()}}
        path = self.dir / "source_manifest.json"
        if not path.exists():
            b.write_json(path, manifest)
        elif b.read_json(path)["full_source_digest"] != manifest["full_source_digest"]:
            b.append_event(self.events, "source_changed_on_resume", current=manifest["full_source_digest"])
        self.training_code, self.source_digest = manifest["training_code_sha256"], manifest["full_source_digest"]

    def check_preflight(self):
        if not self.require_preflight:
            return
        record = PREFLIGHT_DIR / f"passed_{self.source_digest[:16]}.json"
        if not record.exists() or not b.read_json(record).get("passed") or b.read_json(record).get("full_source_digest") != self.source_digest:
            raise ValueError("No passed --preflight-only record for the current code/protocol; run it before any fit")

    # ------------------------------------------------------------------ D0
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
        return KelmarshMultiYearSeries(ROOT / d["v28_preparation"]["csv"], **kwargs)

    def stage_d0(self):
        self.set_status(stage="D0", train="RUNNING")
        d = self.p["data"]
        identity = {"verification": "SKIPPED_SYNTHETIC_TEST_ONLY"}
        if self.verify_data:
            csv = ROOT / d["v28_preparation"]["csv"]
            meta_path = csv.parent.parent / "metadata" / "preparation.json"
            v28_meta = b.read_json(meta_path)
            b.verify_preparation(v28_meta, self.p, b.file_sha256(csv))
            dev = d["development_preparation"]
            dev_csv = ROOT / dev["csv"]
            dev_meta_path = dev_csv.parent.parent / "metadata" / "preparation.json"
            if b.file_sha256(dev_csv) != dev["csv_sha256"] or b.file_sha256(dev_meta_path) != dev["preparation_json_sha256"]:
                raise ValueError("Registered development preparation changed")
            rows = b.verify_development_rows(pd.read_csv(csv), pd.read_csv(dev_csv))
            b.verify_power_curves(v28_meta, b.read_json(dev_meta_path))
            identity = {"v28_csv_sha256": v28_meta["prepared"]["sha256"], "v28_preparation_json_sha256": b.file_sha256(meta_path),
                        "development_rows_identical": rows, "power_curves_identical": True,
                        "prepared_years": v28_meta["years"], "curve_years": v28_meta["curve_years"]}
        self.series = self.load_series()
        audit = self.series.audit
        if audit["channels"] != d["channels"]:
            raise ValueError("Series channels differ from the protocol")
        manifest = {"identity": identity, "series_audit": audit, "test_windows_instantiated": False}
        path = self.dir / "data_manifest.json"
        if path.exists():
            old = b.read_json(path)
            if b.canonical_sha256({k: old[k] for k in ("identity", "series_audit")}) != \
                    b.canonical_sha256({k: manifest[k] for k in ("identity", "series_audit")}):
                raise ValueError("D0 data identity differs from the registered D0 of this run")
        else:
            b.write_json(path, manifest)
            self.log("D0 data identity", identity=identity, windows=audit["windows"])
        b.append_event(self.events, "D0_verified", windows=audit["windows"])

    def windows(self, split, cohort):
        if split == "test" and not (self.dir / "selection_freeze.json").exists():
            raise PermissionError("Test windows may only be instantiated after the S1 selection freeze")
        b.append_event(self.events, "windows_instantiated", split=split, cohort=cohort)
        return self.series.windows(split, cohort)

    # ------------------------------------------------------------------ T1
    def device(self):
        path = self.dir / "device.json"
        if path.exists():
            record = b.read_json(path)
        else:
            record = b.choose_pair_device(v26run.gpu_snapshot(self.gpu_index), self.p["operations"]) | {"decided_utc": now()}
            b.write_json(path, record)
        if record["device"] == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Registered CUDA device unavailable; resume on the same host class")
            device = torch.device(f"cuda:{self.gpu_index}")
            torch.cuda.set_per_process_memory_fraction(record["memory_fraction"], device)
            return device
        return torch.device("cpu")

    def registered_fits(self):
        return sorted(self.rel(q.parent) for q in (self.dir / "runs").glob("*/s*/trained.json"))

    def stage_t1(self):
        device = self.device()
        cache = {}
        for arm in b.TRAINED_ARMS:
            config = b.arm_config(self.p, arm)
            key = (config["train_cohort"], config["selection_cohort"])
            if key not in cache:
                cache[key] = (self.windows("train", key[0]), self.windows("validation", key[1]))
            fit_w, val_w = cache[key]
            fit, selection = ChannelView(fit_w, config["channel_indices"]), ChannelView(val_w, config["channel_indices"])
            for seed in self.p["seeds"]:
                out = self.dir / "runs" / arm / f"s{seed}"
                contract = {"protocol_sha256": self.protocol_sha, "training_code_sha256": self.training_code,
                            "arm": arm, "config": config, "seed": int(seed), "fit_window_hash": b.window_hash(fit.window_starts),
                            "selection_window_hash": b.window_hash(selection.window_starts),
                            "scaler_mean": fit.scaler.mean_.tolist(), "scaler_scale": fit.scaler.scale_.tolist(),
                            "training": self.p["training"], "device_type": device.type,
                            "determinism": v26run.determinism_record()}
                if not (out / "trained.json").exists():
                    self.check_preflight()
                    if not (out / "last.pt").exists() and len(self.registered_fits()) + 1 > self.p["budget"]["new_fits_max"]:
                        raise ValueError("The 18-fit budget would be exceeded; refuse")
                    self.set_status(stage="T1", train="RUNNING")
                    b.write_json(self.dir / "progress.json", {"arm": arm, "seed": seed, "utc": now()})
                    b.append_event(self.events, "fit_started", arm=arm, seed=seed, device=str(device))
                self.fit_calls.append((arm, seed))
                trained = fit_config(config, self.p, fit=fit, selection=selection, contract=contract, seed=seed,
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
        for arm in b.TRAINED_ARMS:
            for seed in self.p["seeds"]:
                folder = self.dir / "runs" / arm / f"s{seed}"
                trained = b.read_json(folder / "trained.json")
                b.verify_registered_files(folder, trained["artifacts"])
                records.append({"arm": arm, "seed": seed, "best_epoch": trained["best_epoch"],
                                "validation": {k: trained["validation_best"][k] for k in ("mae", "rmse", "mse")},
                                "train_cohort": b.arm_config(self.p, arm)["train_cohort"]})
                records[-1]["selection_cohort"] = b.arm_config(self.p, arm)["selection_cohort"]
                fits[self.rel(folder)] = {"trained_sha256": b.file_sha256(folder / "trained.json"),
                                          "best_sha256": b.file_sha256(folder / "best.pt")}
        choice = b.validation_choice([r for r in records if r["train_cohort"] == "normal"], self.p)
        freeze = {"records": records, "fits": fits, "validation_choice": choice, "frozen_utc": now(),
                  "rule": "written before any test window is instantiated"}
        path = self.dir / "selection_freeze.json"
        if path.exists():
            old = b.read_json(path)
            if b.canonical_sha256({k: old[k] for k in ("records", "fits", "validation_choice")}) != \
                    b.canonical_sha256({k: freeze[k] for k in ("records", "fits", "validation_choice")}):
                raise ValueError("Selected states changed after the S1 freeze")
        else:
            b.write_json(path, freeze)
            self.log("S1 selection freeze", validation_choice=choice["arm"], ranking=choice["ranking"])
        b.append_event(self.events, "S1_frozen", choice=choice["arm"])
        return b.read_json(path)

    # ------------------------------------------------------------------ E1
    def stage_e1(self, freeze):
        self.set_status(stage="E1", train="RUNNING")
        if b.canonical_sha256(freeze) != b.canonical_sha256(b.read_json(self.dir / "selection_freeze.json")):
            raise ValueError("Evaluation freeze identity changed")
        b.verify_registered_files(self.dir, {name + "/best.pt": entry["best_sha256"]
                                            for name, entry in freeze["fits"].items()})
        development = self.series
        self.series = self.load_series(include_test=True)
        for attr in ("scaler_mean", "scaler_scale"):
            if not np.array_equal(getattr(development, attr), getattr(self.series, attr)):
                raise ValueError("Adding the test year changed the frozen training scaler")
        for split in ("train", "validation"):
            for cohort_name in b.COHORTS:
                before, after = development.windows(split, cohort_name), self.series.windows(split, cohort_name)
                pairs = [(before.target_times(), after.target_times()),
                         (before.turbines[before.window_starts], after.turbines[after.window_starts])]
                pairs += list(zip(before.truth_and_anchor(), after.truth_and_anchor()))
                if any(not np.array_equal(a, c) for a, c in pairs):
                    raise ValueError(f"Adding test year changed the {split}/{cohort_name} queue or labels")
        b.append_event(self.events, "test_series_opened_after_freeze", windows=self.series.audit["windows"]["test"])
        test_all = self.windows("test", "all")
        test_normal = self.windows("test", "normal")
        normal_mask = np.isin(test_all.window_starts, test_normal.window_starts)
        truth, persistence_hour = test_all.truth_and_anchor()
        folder = self.dir / "evaluation"
        folder.mkdir(exist_ok=True)
        cohort = {"truth": truth, "times": test_all.target_times(), "turbines": test_all.turbines[test_all.window_starts],
                  "window_starts": test_all.window_starts, "normal_mask": normal_mask,
                  "PersistenceHour": persistence_hour, "PersistenceLast10": test_all.last10_anchor()}
        self._register_npz(folder / "test_cohort.npz", cohort)
        cpu, batch = torch.device("cpu"), self.p["training"]["eval_batch_size"]
        for arm in b.TRAINED_ARMS:
            config = b.arm_config(self.p, arm)
            view = ChannelView(test_all, config["channel_indices"])
            for seed in self.p["seeds"]:
                run = self.dir / "runs" / arm / f"s{seed}"
                saved_path = folder / f"{arm}_s{seed}.npz"
                registry = saved_path.with_suffix(".json")
                identity = {"best_sha256": b.file_sha256(run / "best.pt"),
                            "cohort_sha256": b.file_sha256(folder / "test_cohort.npz")}
                if registry.exists():
                    registered = b.read_json(registry)
                    if registered["identity"] != identity or registered["prediction_sha256"] != b.file_sha256(saved_path):
                        raise ValueError("Registered test prediction identity changed")
                    continue
                trained = b.read_json(run / "trained.json")
                state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)["model"]
                if v26run.state_digest(state) != trained["best_state_sha256"]:
                    raise ValueError(f"{run}/best.pt differs from the frozen selected state")
                model, _ = build_model(config)
                model.load_state_dict(state, strict=True)
                prediction = v26run.predict(model, view, seed, cpu, batch)
                self._register_npz(saved_path, {"prediction": prediction, "window_starts": test_all.window_starts})
                b.write_json(registry, {"identity": identity, "prediction_sha256": b.file_sha256(saved_path)})
        b.append_event(self.events, "E1_test_predicted", windows=int(len(truth)), normal=int(normal_mask.sum()))
        return self.decide(freeze)

    @staticmethod
    def _register_npz(path, arrays):
        if path.exists():  # a resume must reproduce registered arrays exactly (CPU, deterministic)
            with np.load(path, allow_pickle=False) as saved:
                for key, value in arrays.items():
                    if not np.array_equal(np.asarray(value), saved[key]):
                        raise ValueError(f"{path.name}:{key} differs from the registered evaluation")
        else:
            temporary = path.with_name(path.stem + ".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, path)

    def decide(self, freeze):
        payload = b.evaluate_registered(self.dir, self.p, freeze["validation_choice"]["arm"])
        path = self.dir / "decision.json"
        if path.exists():
            if b.read_json(path)["rules"] != payload["rules"]:
                raise ValueError("Registered decision would change; refuse")
        else:
            b.write_json(path, payload | {"utc": now()})
        self.log("E1 single test evaluation", rules=payload["rules"], choice=freeze["validation_choice"]["arm"])
        print(f"[RULES] {payload['rules']}", flush=True)
        return payload

    # ------------------------------------------------------------------ finish
    def finish_training(self):
        hashes = {self.rel(q): b.file_sha256(q) for q in sorted(self.dir.rglob("*"))
                  if registrable(q, self.dir) and q.name != "training_completed.json"}
        b.write_json(self.dir / "training_completed.json", {"hashes": hashes, "fits": self.registered_fits(), "utc": now()})
        self.set_status(stage="TRAINING_DONE", train="TRAIN_COMPLETED")

    def report(self):
        from scripts import report_kelmarsh_multiyear_benchmark as reporting
        self.set_status(stage="REPORT", report="RUNNING")
        try:
            summary = reporting.build_report(self.dir)
        except Exception as error:
            self.set_status(report="REPORT_FAILED", report_error=f"{type(error).__name__}: {error}")
            raise
        self.set_status(report="REPORT_COMPLETED", report_error=None)
        b.write_json(self.dir / "completed.json", {self.rel(q): b.file_sha256(q) for q in sorted(self.dir.rglob("*"))
                                                   if registrable(q, self.dir)})
        self.set_status(stage="COMPLETED")
        return summary

    def run(self, report_only=False):
        self.dir.mkdir(parents=True, exist_ok=True)
        lock = b.acquire_exclusive_lock(self.lock_path)
        try:
            if (self.dir / "completed.json").exists() and not report_only:
                self.freeze_protocol_record()
                b.verify_training_completed(self.dir)
                b.verify_registered_files(self.dir, b.read_json(self.dir / "completed.json"))
                print("[REUSE] v28 already completed; nothing retrained", flush=True)
                return "COMPLETED"
            v26run.configure_determinism()
            (self.dir / "controller.pid").write_text(str(os.getpid()), encoding="utf-8")
            self.freeze_protocol_record()
            self.claim_pointer()
            self.record_sources()
            b.append_event(self.events, "controller_started", argv=sys.argv, report_only=report_only)
            if (self.dir / "training_completed.json").exists():
                b.verify_training_completed(self.dir)
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


evaluate_registered = b.evaluate_registered  # torch-free; shared by controller and report


# =========================================================================== preflight and CLI
def protocol_namespace_check(p):
    own, configs, pointers = p["protocol_id"], [], {}
    for path in sorted((ROOT / "configs").glob("*protocol*.json")):
        if path.resolve() == PROTOCOL.resolve():
            continue
        try:
            if b.read_json(path).get("protocol_id") == own:
                configs.append(str(path))
        except (ValueError, AttributeError):
            continue
    for path in sorted(LOG_ROOT.glob("*latest.txt")):
        if path.resolve() == POINTER.resolve():
            continue
        target = Path(path.read_text(errors="replace").strip())
        marker = (target if target.is_absolute() else ROOT / target) / "protocol.json"
        if marker.exists() and b.read_json(marker).get("protocol_id") == own:
            pointers[str(path)] = str(target)
    return {"protocol_id": own, "configs": configs, "pointers": pointers, "passed": not configs and not pointers}


def smoke(p, folder):
    """Synthetic end-to-end smoke of the three families on the provider's window views (CPU, 2 epochs)."""
    q = {**p, "training": {**p["training"], "max_epochs": 2, "scheduler_t_max": 2}}
    series = KelmarshMultiYearSeries(synthetic_prepared_frame(), splits={"train": (2016,), "validation": (2017,), "test": (2018,)},
                                     extras=tuple(p["data"]["extras_in_common_series"]), preparation=SYNTHETIC_PREPARATION)
    out = {}
    for arm in ("DLinear_normal", "LinearFlat_last10", "TSMixer_base"):
        config = b.arm_config(p, arm)
        fit = ChannelView(series.windows("train", "normal"), config["channel_indices"])
        val = ChannelView(series.windows("validation", "normal"), config["channel_indices"])
        trained = fit_config(config, q, fit=fit, selection=val, contract={"smoke": arm}, seed=7, device=torch.device("cpu"),
                             out=folder / arm)
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
        command = [sys.executable, "-m", "pytest", "-q", *V28_TESTS, *REGRESSION_TESTS]
    except ImportError:
        command = [sys.executable, "-m", "unittest", *[t.replace("/", ".")[:-3] for t in V28_TESTS + REGRESSION_TESTS]]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (folder / "tests.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    checks["tests"] = {"command": command, "exit_code": result.returncode, "passed": result.returncode == 0,
                       "tail": (result.stdout + result.stderr).splitlines()[-15:]}
    code = subprocess.run(["bash", "-n", "scripts/train/Kelmarsh_launch_multiyear_benchmark.sh"], cwd=ROOT).returncode
    checks["bash_syntax"] = {"exit_code": code, "passed": code == 0}
    try:
        d0 = V28Run(folder / "d0", protocol=p, protocol_sha=protocol_sha, require_preflight=False, write_pointer=False,
                    lock_path=PREFLIGHT_DIR / "preflight.lock")
        d0.dir.mkdir(parents=True)
        d0.freeze_protocol_record()
        d0.record_sources()
        d0.stage_d0()
        checks["D0"] = {"passed": True, **b.read_json(d0.dir / "data_manifest.json")}
    except Exception as error:
        checks["D0"] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    try:
        checks["synthetic_smoke"] = smoke(p, folder / "smoke")
    except Exception as error:
        checks["synthetic_smoke"] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    checks["resources"] = {"gpu": v26run.gpu_snapshot(gpu_index), "cpu_count": os.cpu_count()}
    passed = all(checks[k]["passed"] for k in ("namespace", "tests", "bash_syntax", "D0", "synthetic_smoke"))
    record = {"passed": bool(passed), "full_source_digest": digest, "protocol_sha256": protocol_sha,
              "training_code_sha256": training_code_sha256(), "git_head": git_head(), "utc": now(), "checks": checks}
    b.write_json(folder / "preflight.json", record)
    if passed:
        b.write_json(PREFLIGHT_DIR / f"passed_{digest[:16]}.json", record)
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
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.status:
        print_status()
        return 0
    if args.preflight_only:
        return 0 if preflight(args.gpu)["passed"] else 5
    if args.output is None:
        raise ValueError("An explicit independent --output directory is required")
    run = V28Run(args.output.resolve(), gpu_index=args.gpu)
    try:
        run.run(report_only=args.report_only)
        return 0
    except BlockingIOError:
        print("[BUSY] Another v28 controller holds the lock; nothing changed", flush=True)
        return 3
    except Exception as error:
        if args.output.exists():
            status = run.dir / "status.json"
            if not status.exists() or b.read_json(status).get("train") != "TRAIN_COMPLETED":
                run.set_status(train="FAILED", error=f"{type(error).__name__}: {error}")
            b.append_event(run.events, "controller_failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    sys.exit(main())
