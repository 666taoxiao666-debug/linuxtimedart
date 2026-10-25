#!/usr/bin/env python3
"""kelmarsh_forward_block_residual_diagnostic_v27 (Kelmarsh 24h->4h, Wiki off).

Only the registered v20-v26 fold-0 FIT period (2016-05-02 .. 2016-09-10, historically developed) is used. It is
cut into 6 equal chronological blocks. Step k = 3..6 refits everything on blocks 1..k-1 and scores block k:
  scaler      mean/std of rows before the step's fit cut (never later rows)
  anchor      DLinear, v20 recipe, trained by the UNCHANGED v25 trainer (fit 90% / selection 10% of the training span)
  residuals   AnchoredFull and AnchoredPowerOnly x 5 seeds, trained by the UNCHANGED v26 trainer on top of that anchor
  evaluation  anchor, each residual (selected and last epoch), 5-seed means, Persistence, on the next block only
Stages: B0 data/blocks/contracts (no fit) -> B1 four steps (<= 44 fits) -> B2 predeclared decision -> report -> stop.
No internal-selection or outer split of any fold, no sealed test, no change to v25/v26, no forecasting claim.
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

from data_provider.kelmarsh_hourly import SOURCE_COLUMNS, time_bounds, valid_window_starts  # noqa: E402
from scripts import train_kelmarsh_anchored_residual as v26run  # noqa: E402
from scripts import train_kelmarsh_backbone_information as v25run  # noqa: E402
from utils import kelmarsh_forward_block_residual as v27  # noqa: E402

PROTOCOL = ROOT / "configs/kelmarsh_forward_block_residual_protocol_v27.json"
PROTOCOL_SHA256 = "b9918206a61ca655957212fd03f4831dc9eda1d032cfe45ef7b328b154d6e4b4"
LOG_ROOT = ROOT / "outputs/logs/Kelmarsh"
POINTER = LOG_ROOT / "forward_block_residual_latest.txt"
LOCK = LOG_ROOT / "forward_block_residual.lock"
PREFLIGHT_DIR = LOG_ROOT / "forward_block_residual_preflight"
# Code that determines trained/evaluated numbers (the two reused trainers are part of it).
TRAINING_FILES = ("models/DLinearAnchoredResidual.py", "models/TSMixerBasicSmall.py", "models/DLinear.py", "models/__init__.py",
                  "data_provider/kelmarsh_hourly.py", "data_provider/__init__.py", "utils/metrics.py",
                  "scripts/train_kelmarsh_backbone_information.py", "scripts/train_kelmarsh_anchored_residual.py",
                  "utils/kelmarsh_backbone_information.py", "utils/kelmarsh_anchored_residual.py",
                  "utils/kelmarsh_forward_block_residual.py", "scripts/train_kelmarsh_forward_block_residual.py",
                  "layers/Autoformer_EncDec.py", "layers/SelfAttention_Family.py", "layers/Embed.py",
                  "layers/__init__.py", "utils/masking.py", "utils/paired_forecast_statistics.py",
                  "configs/tsmixer_upstream_pin_v25.json")
V27_FILES = TRAINING_FILES + ("scripts/train_kelmarsh_forward_block_residual.py", "scripts/report_kelmarsh_forward_block_residual.py",
                              "scripts/train/Kelmarsh_launch_forward_block_residual.sh", "utils/paired_forecast_statistics.py",
                              "tests/test_kelmarsh_forward_block_residual.py")
V27_TESTS = ("tests/test_kelmarsh_forward_block_residual.py",)
REGRESSION_TESTS = ("tests/test_dlinear_anchored_residual.py", "tests/test_kelmarsh_anchored_residual.py",
                    "tests/test_kelmarsh_backbone_information.py")
VOLATILE = {"status.json", "status.env", "progress.json", "launch.log", "launcher.pid", "controller.pid",
            "events.jsonl", "experiment_log.md", "fit_progress.json"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def registrable(path, directory):
    path = Path(path)
    return (path.is_file() and path.name not in VOLATILE and not path.name.endswith(".tmp")
            and path != Path(directory) / "completed.json")


# =========================================================================== data
class FitPrefixSeries:
    """Read numeric fields only inside the registered fold-0 fit prefix.

    The old unsealed grid remains as metadata/NaN padding to preserve the exact
    registered window indices. No default scaler is fitted, and no later numeric
    row is parsed, engineered, scaled or windowed. The first pass reads dates only.
    """

    def __init__(self, path, *, fold, input_len, horizon, stride, start, end):
        if fold != 0:
            raise ValueError("Only the registered fold-0 fit prefix is allowed")
        lo, hi = pd.Timestamp(start).tz_convert(None), pd.Timestamp(end).tz_convert(None)
        timeline = pd.date_range(lo, hi, freq="h", inclusive="left").to_numpy()
        self.bounds = time_bounds(timeline, fold)
        cutoff = self.bounds["calibration_early_end"]
        meta = pd.read_csv(path, usecols=["timestamp_hour_end_utc"])
        dates = pd.to_datetime(meta.timestamp_hour_end_utc, utc=True).dt.tz_localize(None)
        keep = set((np.flatnonzero(((dates >= lo) & (dates < cutoff)).to_numpy()) + 1).tolist())
        frame = pd.read_csv(path, usecols=SOURCE_COLUMNS, skiprows=lambda row: row != 0 and row not in keep)
        frame["date"] = pd.to_datetime(frame.timestamp_hour_end_utc, utc=True).dt.tz_localize(None)
        if frame.empty or not ((frame.date >= lo) & (frame.date < cutoff)).all():
            raise ValueError("Numeric loader escaped the registered fit prefix")
        if frame.duplicated(["turbine_id", "date"]).any():
            raise ValueError("Duplicate turbine/hour rows inside the fit prefix")
        if sorted(frame.turbine_id.unique()) != [1, 2, 3, 4, 5, 6]:
            raise ValueError("Expected the six registered turbines in the fit prefix")
        grid = timeline[timeline < self.bounds["sealed_test_start"]]
        prefix = grid[grid < cutoff]
        raw_parts = []
        for turbine in range(1, 7):
            unit = frame[frame.turbine_id == turbine].set_index("date").reindex(prefix)
            angles = np.deg2rad(unit.nacelle_position_deg.to_numpy(dtype=float))
            pitch = unit[["pitch_a_deg", "pitch_b_deg", "pitch_c_deg"]].mean(1, skipna=False)
            values = np.column_stack((unit.wind_speed_mps, unit.ambient_temp_c, np.sin(angles), np.cos(angles),
                                      pitch, unit.rotor_speed_rpm, unit.power_kw)).astype(float)
            padded = np.full((len(grid), 7), np.nan)
            padded[:len(prefix)] = values
            raw_parts.append(padded)
        self.raw = np.concatenate(raw_parts)
        self.raw[~np.isfinite(self.raw)] = np.nan
        self.dates = np.tile(grid, 6).astype("datetime64[ns]")
        self.turbines = np.repeat(np.arange(1, 7, dtype=np.int16), len(grid))
        self.input_len, self.horizon = int(input_len), int(horizon)
        self.starts, _ = valid_window_starts(self.raw, self.dates, self.turbines, input_len, horizon, stride)
        self.audit = {"numeric_rows_read": len(frame), "numeric_end_exclusive": str(cutoff),
                      "suffix_numeric_rows_read": 0, "default_scaler_fitted": False,
                      "suffix_storage": "metadata grid and NaN padding only"}

    def subset(self, name):
        raise ValueError("Named original splits are forbidden in the forward-block diagnostic")


class ForwardWindows(Dataset):
    """KelmarshWindows-compatible view of one step part, standardized with the STEP scaler (not the fold scaler)."""

    def __init__(self, series, starts, values, mean, scale, stage):
        self.window_starts = np.asarray(starts, dtype=np.int64)
        self.raw, self.values = series.raw, values
        self.dates, self.turbines = series.dates, series.turbines
        self.seq_len, self.pred_len = int(series.input_len), int(series.horizon)
        self.scaler = SimpleNamespace(mean_=np.asarray(mean, dtype=np.float64), scale_=np.asarray(scale, dtype=np.float64))
        self.series = SimpleNamespace(scaler=self.scaler)
        self.stage = stage

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, index):
        start = self.window_starts[index]
        end = start + self.seq_len
        return (self.values[start:end], self.values[end:end + self.pred_len],
                np.zeros((self.seq_len, 1), np.float32), np.zeros((self.pred_len, 1), np.float32))

    def truth_and_anchor(self):
        rows = self.window_starts[:, None] + self.seq_len + np.arange(self.pred_len)
        truth = self.raw[rows, -1]
        last = self.raw[self.window_starts + self.seq_len - 1, -1]
        return truth, np.repeat(last[:, None], self.pred_len, axis=1)

    def target_times(self):
        return self.dates[self.window_starts[:, None] + self.seq_len + np.arange(self.pred_len)]


def standardized(raw, mean, scale):
    return ((np.asarray(raw, dtype=np.float64) - mean) / scale).astype(np.float32)


def candidate_windows(series, protocol, verify_hash=True):
    """The registered fold-0 FIT windows (targets ending before fold-0 calibration_early_end) and the timeline."""
    first = series.dates[series.starts + series.input_len]
    last = series.dates[series.starts + series.input_len + series.horizon - 1]
    t_hi = series.bounds["calibration_early_end"]
    mask = last < t_hi
    starts = series.starts[mask]
    digest = v27.window_hash(starts)
    if verify_hash and digest != protocol["timeline"]["fold0_fit_window_hash"]:
        raise ValueError("fold-0 fit window set differs from the registered v20/v25/v26 queue; investigate, never bypass")
    t_lo = np.datetime64(pd.Timestamp(protocol["start_utc"]).tz_convert(None)) + np.timedelta64(int(protocol["input_len"]), "h")
    return {"starts": starts, "first": first[mask], "last": last[mask], "t_lo": t_lo, "t_hi": np.datetime64(t_hi),
            "window_hash": digest}


# =========================================================================== numerics identity
def training_code_sha256():
    return v27.canonical_sha256({"files": {name: v27.text_sha256_lf(ROOT / name) for name in TRAINING_FILES},
                                 "torch_major_minor": ".".join(torch.__version__.split(".")[:2])})


def load_protocol(path=PROTOCOL, expected=PROTOCOL_SHA256):
    digest = v27.text_sha256_lf(path)
    if expected is not None and digest != expected:
        raise ValueError("Frozen v27 protocol bytes changed; a new protocol needs a new id")
    protocol = v27.read_json(path)
    v27.validate_protocol(protocol)
    return protocol, digest


def full_source_digest(protocol_sha):
    return v27.canonical_sha256({"protocol_sha256": protocol_sha,
                                 "sources": {k: v27.text_sha256_lf(ROOT / k) for k in V27_FILES + REGRESSION_TESTS}})


def environment():
    import sklearn
    info = {"utc": now(), "hostname": socket.gethostname(), "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__, "determinism": v26run.determinism_record(),
            "argv": sys.argv}
    if torch.cuda.is_available():
        info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def git_info():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as error:  # informational only
            return f"unavailable: {type(error).__name__}"
    return {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD")}


# =========================================================================== controller
class V27Run:
    def __init__(self, directory, *, protocol=None, protocol_sha=None, series_factory=None, gpu_index=0,
                 require_preflight=True, write_pointer=True, lock_path=LOCK, verify_fold0_hash=True):
        """``verify_fold0_hash=False`` and an explicit ``series_factory`` are for synthetic tests only (never from the CLI)."""
        self.dir = Path(directory)
        if protocol is None:
            protocol, protocol_sha = load_protocol()
        self.p, self.protocol_sha = protocol, protocol_sha
        self.series_factory = series_factory or (lambda fold: FitPrefixSeries(
            ROOT / protocol["data_path"], fold=fold, input_len=protocol["input_len"], horizon=protocol["pred_len"],
            stride=protocol["train_stride"], start=protocol["start_utc"], end=protocol["end_utc_exclusive"]))
        self.gpu_index, self.require_preflight = gpu_index, require_preflight
        self.write_pointer, self.lock_path, self.verify_fold0_hash = write_pointer, lock_path, verify_fold0_hash
        self.events = self.dir / "events.jsonl"
        self.anchor_view, self.residual_view = v27.anchor_view(protocol), v27.residual_view(protocol)
        self.fit_calls, self.cache = [], {}

    # ------------------------------------------------------------------ bookkeeping
    def rel(self, path):
        return str(Path(path).relative_to(self.dir)).replace("\\", "/")

    def set_status(self, **fields):
        path = self.dir / "status.json"
        status = v27.read_json(path) if path.exists() else {"train": "NOT_STARTED", "report": "PENDING", "stage": None}
        status.update(fields, updated_utc=now())
        v27.write_json(path, status)
        overall = ("COMPLETED" if status.get("report") == "REPORT_COMPLETED" and status["train"] == "TRAIN_COMPLETED"
                   else status["report"] if status.get("report") == "REPORT_FAILED" else status["train"])
        (self.dir / "status.env").write_text(f"STATUS={overall}\nTRAIN={status['train']}\nREPORT={status['report']}\n"
                                             f"STAGE={status.get('stage')}\n", encoding="utf-8")
        return status

    def log(self, title, **items):
        with (self.dir / "experiment_log.md").open("a", encoding="utf-8") as handle:
            handle.writelines([f"\n## {now()} {title}\n"] + [f"- {k}: {v}\n" for k, v in items.items()])

    def freeze_protocol_record(self):
        record = {"protocol_id": self.p["protocol_id"], "protocol_sha256_lf": self.protocol_sha, "protocol": self.p}
        path = self.dir / "protocol.json"
        if path.exists():
            if v27.canonical_sha256(v27.read_json(path)) != v27.canonical_sha256(record):
                raise ValueError("Frozen protocol changed on resume; start a new protocol id instead")
        else:
            v27.write_json(path, record)

    def claim_pointer(self):
        if not self.write_pointer:
            return
        POINTER.parent.mkdir(parents=True, exist_ok=True)
        if POINTER.exists() and POINTER.read_text().strip() and Path(POINTER.read_text().strip()).resolve() != self.dir.resolve():
            raise ValueError(f"Pointer already registers another v27 run: {POINTER.read_text().strip()}; use --resume")
        POINTER.write_text(str(self.dir.resolve()), encoding="utf-8")

    def record_sources(self):
        manifest = {"utc": now(), "training_code_sha256": training_code_sha256(),
                    "full_source_digest": full_source_digest(self.protocol_sha),
                    "sources": {k: (v27.file_sha256(ROOT / k) if (ROOT / k).exists() else None) for k in V27_FILES},
                    "git": git_info()}
        path = self.dir / "source_manifest.json"
        if not path.exists():
            v27.write_json(path, manifest)
        elif v27.read_json(path)["full_source_digest"] != manifest["full_source_digest"]:
            v27.append_event(self.events, "source_changed_on_resume", current=manifest["full_source_digest"],
                             rule="finished fits are reused only if their per-fit contract still matches")
        if not (self.dir / "environment.json").exists():
            v27.write_json(self.dir / "environment.json", environment())
        self.training_code, self.source_digest = manifest["training_code_sha256"], manifest["full_source_digest"]
        return manifest

    def check_preflight(self):
        if not self.require_preflight:
            return
        record = PREFLIGHT_DIR / f"passed_{self.source_digest[:16]}.json"
        if not record.exists() or not v27.read_json(record).get("passed") \
                or v27.read_json(record).get("full_source_digest") != self.source_digest:
            raise ValueError("No passed --preflight-only record for the current code/protocol; run it before any fit")

    # ------------------------------------------------------------------ B0
    def stage_b0(self):
        self.set_status(stage="B0", train="RUNNING")
        if self.verify_fold0_hash and v27.file_sha256(ROOT / self.p["data_path"]) != self.p["data_sha256"]:
            raise ValueError("Prepared Kelmarsh source fingerprint changed")
        series = self.series_factory(self.p["timeline"]["source_fold"])
        windows = candidate_windows(series, self.p, verify_hash=self.verify_fold0_hash)
        edges = v27.block_edges(windows["t_lo"], windows["t_hi"], self.p["timeline"]["blocks"])
        steps = {}
        for k in self.p["timeline"]["eval_blocks"]:
            bounds = v27.step_bounds(edges, k)
            masks = v27.step_masks(windows["first"], windows["last"], bounds)
            mean, scale, rows, constant = v27.scaler_stats(series.raw, series.dates, bounds["fit_cut"])
            parts = {name: windows["starts"][mask] for name, mask in masks.items()}
            evaluation_issue = series.dates[parts["evaluation"] + series.input_len] - np.timedelta64(1, "h")
            steps[str(k)] = {"bounds_utc": {key: str(value) for key, value in bounds.items() if key != "step"},
                             "windows": {name: int(len(s)) for name, s in parts.items()},
                             "window_hashes": {name: v27.window_hash(s) for name, s in parts.items()},
                             "evaluation_issue_times": int(len(np.unique(evaluation_issue))),
                             "scaler_mean": mean.tolist(), "scaler_scale": scale.tolist(), "scaler_rows": rows,
                             "scaler_constant_channels": constant}
            self.cache[k] = {"series": series, "parts": parts, "mean": mean, "scale": scale}
        manifest = {"protocol_sha256": self.protocol_sha, "source_fold": self.p["timeline"]["source_fold"],
                    "fold0_fit_window_hash": windows["window_hash"], "fold0_fit_windows": int(len(windows["starts"])),
                    "timeline_utc": [str(windows["t_lo"]), str(windows["t_hi"])], "edges_utc": [str(e) for e in edges],
                    "steps": steps, "internal_selection_or_outer_instantiated": False,
                    "numeric_read_audit": getattr(series, "audit", {})}
        path = self.dir / "blocks_manifest.json"
        if path.exists():
            if v27.canonical_sha256(v27.read_json(path)) != v27.canonical_sha256(manifest):
                raise ValueError("Blocks/windows/scalers differ from the registered B0 of this run")
        else:
            v27.write_json(path, manifest)
            self.log("B0 blocks and contracts", edges=manifest["edges_utc"],
                     windows={k: s["windows"] for k, s in steps.items()}, new_fits=0)
        v27.append_event(self.events, "B0_verified")
        self.blocks = v27.read_json(path)
        return self.blocks

    def datasets(self, k):
        c = self.cache[k]
        values = standardized(c["series"].raw, c["mean"], c["scale"])
        return {name: ForwardWindows(c["series"], starts, values, c["mean"], c["scale"], name)
                for name, starts in c["parts"].items()}

    # ------------------------------------------------------------------ B1
    def step_device(self, k):
        path = self.dir / "steps" / f"k{k}" / "step_device.json"
        if path.exists():
            record = v27.read_json(path)
        else:
            record = v27.choose_pair_device(v26run.gpu_snapshot(self.gpu_index), self.p["operations"]) | {"step": k, "decided_utc": now()}
            v27.write_json(path, record)
            v27.append_event(self.events, "step_device_decided", **record)
        if record["device"] == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Registered CUDA step device unavailable; resume on the same host class (no silent switch)")
            device = torch.device(f"cuda:{self.gpu_index}")
            torch.cuda.set_per_process_memory_fraction(record["memory_fraction"], device)
            return device
        return torch.device("cpu")

    def registered_fits(self):
        return sorted(self.rel(p.parent) for p in (self.dir / "steps").glob("k*/**/trained.json"))

    def contract(self, k, arm, seed, data, device, anchor_sha=None):
        return {"protocol_sha256": self.protocol_sha, "training_code_sha256": self.training_code, "data_sha256": self.p["data_sha256"],
                "step": int(k), "arm": arm, "seed": int(seed), "bounds_utc": self.blocks["steps"][str(k)]["bounds_utc"],
                "window_hashes": {name: v27.window_hash(d.window_starts) for name, d in data.items()},
                "scaler_mean": data["fit"].scaler.mean_.tolist(), "scaler_scale": data["fit"].scaler.scale_.tolist(),
                "training": self.p["anchor"]["training"] if arm == v27.ANCHOR else self.p["candidate"]["training"],
                "anchor_state_sha256": anchor_sha, "device_type": torch.device(device).type,
                "determinism": v26run.determinism_record()}

    def before_fit(self, out, k, arm, seed, device):
        if (out / "trained.json").exists():
            return
        self.check_preflight()
        if not (out / "last.pt").exists() and len(self.registered_fits()) + 1 > self.p["budget"]["new_fits_max"]:
            raise ValueError("The 44-fit budget would be exceeded; refuse")
        self.set_status(stage=f"B1_k{k}", train="RUNNING")
        v27.write_json(self.dir / "progress.json", {"step": k, "arm": arm, "seed": seed, "device": str(device), "utc": now()})
        v27.append_event(self.events, "fit_started", step=k, arm=arm, seed=seed, device=str(device))

    def fit_anchor(self, k, data, device):
        out = self.dir / "steps" / f"k{k}" / "anchor" / v27.ANCHOR
        seed = self.p["anchor"]["seed"]
        self.before_fit(out, k, v27.ANCHOR, seed, device)
        self.fit_calls.append((k, v27.ANCHOR, seed))
        trained = v25run.fit_arm(v27.ANCHOR, self.anchor_view, fit=data["fit"], selection=data["selection"],
                                 contract=self.contract(k, v27.ANCHOR, seed, data, device), seed=seed, device=device,
                                 out=out, events=self.events)
        state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)["model"]
        state = {key: value.detach().cpu().clone() for key, value in state.items()}
        if v26run.state_digest(state) != trained["best_state_sha256"]:
            raise ValueError("Anchor best.pt differs from its registered selected state")
        return state

    def frozen_anchor(self, state):
        model, _ = v26run.build_model(v27.ANCHOR, self.residual_view, anchor_state=state)  # frozen, eval (v26 rule)
        return model

    def fit_residual(self, k, arm, seed, data, device, anchor_state, anchor_reference):
        out = self.dir / "steps" / f"k{k}" / f"s{seed}" / arm
        self.before_fit(out, k, arm, seed, device)
        self.fit_calls.append((k, arm, seed))
        init = self.dir / "steps" / f"k{k}" / f"s{seed}" / v27.FULL / "init.pt" if arm == v27.POWER else None
        return v26run.fit_arm(arm, self.residual_view, fit=data["fit"], selection=data["selection"],
                              contract=self.contract(k, arm, seed, data, device, v26run.state_digest(anchor_state)),
                              seed=seed, device=device, out=out, anchor_state=anchor_state,
                              anchor_reference_kw=anchor_reference, events=self.events, init_reference=init)

    def evaluate_step(self, k, data, anchor_state):
        """CPU forecasts on the evaluation block from registered states only; nothing here feeds back into training."""
        cpu, batch = torch.device("cpu"), self.p["candidate"]["training"]["eval_batch_size"]
        evaluation = data["evaluation"]
        truth, persistence = evaluation.truth_and_anchor()
        anchor = v26run.predict(self.frozen_anchor(anchor_state), evaluation, 0, cpu, batch)
        stacks = {f"{tag}_{which}": [] for tag in ("full", "power") for which in ("best", "last")}
        epochs = {"full": [], "power": []}
        anchor_sha = v26run.state_digest(anchor_state)
        for seed in self.p["residual_seeds"]:
            for arm, tag in ((v27.FULL, "full"), (v27.POWER, "power")):
                out = self.dir / "steps" / f"k{k}" / f"s{seed}" / arm
                trained = v27.read_json(out / "trained.json")
                v27.verify_registered_files(out, trained["artifacts"])
                if trained["anchor_state_sha256"] != anchor_sha:
                    raise ValueError(f"step {k} seed {seed} {arm}: residual was trained on a different anchor")
                epochs[tag].append(int(trained["best_epoch"]))
                for which in ("best", "last"):
                    state = torch.load(out / f"{which}.pt", map_location="cpu", weights_only=False)["model"]
                    if v26run.anchor_digest(state) != anchor_sha:
                        raise ValueError(f"{out / which}.pt carries a changed anchor")
                    if which == "best" and v26run.state_digest(state) != trained["best_state_sha256"]:
                        raise ValueError(f"{out}/best.pt differs from the registered selected state")
                    model, _ = v26run.build_model(arm, self.residual_view, anchor_state=anchor_state)
                    model.load_state_dict(state, strict=True)
                    stacks[f"{tag}_{which}"].append(v26run.predict(model, evaluation, seed, cpu, batch))
        arrays = {"truth": truth, "persistence": persistence, "anchor": anchor, "times": evaluation.target_times(),
                  "turbines": evaluation.turbines[evaluation.window_starts], "window_starts": evaluation.window_starts,
                  "seeds": np.asarray(self.p["residual_seeds"], dtype=np.int64),
                  "full_selected_epochs": np.asarray(epochs["full"], dtype=np.int64),
                  "power_selected_epochs": np.asarray(epochs["power"], dtype=np.int64),
                  **{key: np.stack(value) for key, value in stacks.items()}}
        path = self.dir / "steps" / f"k{k}" / "evaluation_predictions.npz"
        if path.exists():  # registered once; a resume must reproduce it exactly (CPU, deterministic)
            with np.load(path, allow_pickle=False) as saved:
                v27.assert_aligned(arrays, dict(saved), keys=tuple(arrays), label=f"step {k} evaluation")
        else:
            temporary = path.with_name(path.stem + ".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, path)
        record = v27.step_record(arrays)
        v27.write_json(self.dir / "steps" / f"k{k}" / "step_result.json",
                       {"step": k, "record": record, "evaluation_predictions_sha256": v27.file_sha256(path),
                        "selected_epochs": epochs, "scope": "evaluation block of a training-only period"})
        return arrays, record

    def run_step(self, k):
        data = self.datasets(k)
        device = self.step_device(k)
        anchor_state = self.fit_anchor(k, data, device)
        reference = v26run.predict(self.frozen_anchor(anchor_state), data["selection"], 0, torch.device("cpu"),
                                   self.p["candidate"]["training"]["eval_batch_size"])
        for seed in self.p["residual_seeds"]:
            for arm in v27.RESIDUAL_ARMS:
                self.fit_residual(k, arm, seed, data, device, anchor_state, reference)
        arrays, record = self.evaluate_step(k, data, anchor_state)
        v27.append_event(self.events, "step_completed", step=k, anchor_mae=record["anchor"]["mae"],
                         full_seed_mean_mae=record["full_seed_mean"]["mae"], power_seed_mean_mae=record["power_seed_mean"]["mae"])
        print(f"[STEP] k{k} anchor MAE={record['anchor']['mae']:.3f} full5 MAE={record['full_seed_mean']['mae']:.3f} "
              f"power5 MAE={record['power_seed_mean']['mae']:.3f}", flush=True)
        gc.collect()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        return arrays, record

    # ------------------------------------------------------------------ B2
    def stage_b2(self, arrays_list, records):
        pooled = v27.pooled_statistics(arrays_list, self.p["evaluation"]["pooled_statistics"])
        decision = v27.decide(records, pooled, self.p)
        payload = {"decision": decision, "pooled": pooled, "records": records, "utc": now(),
                   "decision_code_sha256": v27.text_sha256_lf(Path(v27.__file__))}
        path = self.dir / "decision.json"
        if path.exists():
            old = v27.read_json(path)
            if any(v27.canonical_sha256(old[key]) != v27.canonical_sha256(payload[key])
                   for key in ("decision", "pooled", "records", "decision_code_sha256")):
                raise ValueError("Registered decision would change; refuse and review")
            payload = old
        else:
            v27.write_json(path, payload)
        v27.append_event(self.events, "decision", outcome=payload["decision"]["outcome"], flags=payload["decision"]["flags"])
        self.log("B2 predeclared decision", outcome=payload["decision"]["outcome"], flags=payload["decision"]["flags"],
                 counts=payload["decision"]["counts"])
        print(f"[DECISION] {payload['decision']['outcome']} flags={payload['decision']['flags']}", flush=True)
        return payload

    def finish_training(self):
        hashes = {self.rel(p): v27.file_sha256(p) for p in sorted(self.dir.rglob("*"))
                  if registrable(p, self.dir) and p.name != "training_completed.json"}
        v27.write_json(self.dir / "training_completed.json", {"hashes": hashes, "registered_fits": self.registered_fits(),
                                                             "completed_utc": now()})
        self.set_status(stage="TRAINING_DONE", train="TRAIN_COMPLETED")
        v27.append_event(self.events, "TRAIN_COMPLETED", fits=len(self.registered_fits()))

    def report(self):
        from scripts import report_kelmarsh_forward_block_residual as reporting
        self.set_status(stage="REPORT", report="RUNNING")
        try:
            summary = reporting.build_report(self.dir)
        except Exception as error:
            self.set_status(report="REPORT_FAILED", report_error=f"{type(error).__name__}: {error}")
            v27.append_event(self.events, "REPORT_FAILED", error=f"{type(error).__name__}: {error}")
            raise
        self.set_status(report="REPORT_COMPLETED", report_error=None)
        v27.write_json(self.dir / "completed.json", {self.rel(p): v27.file_sha256(p) for p in sorted(self.dir.rglob("*"))
                                                     if registrable(p, self.dir)})
        self.set_status(stage="COMPLETED")
        v27.append_event(self.events, "COMPLETED")
        return summary

    def run(self, report_only=False):
        self.dir.mkdir(parents=True, exist_ok=True)
        lock = v27.acquire_exclusive_lock(self.lock_path)
        try:
            if (self.dir / "completed.json").exists() and not report_only:
                self.freeze_protocol_record()
                v27.verify_training_completed(self.dir)
                v27.verify_registered_files(self.dir, v27.read_json(self.dir / "completed.json"))
                print("[REUSE] v27 already completed; nothing retrained", flush=True)
                return "COMPLETED"
            v26run.configure_determinism()
            (self.dir / "controller.pid").write_text(str(os.getpid()), encoding="utf-8")
            self.freeze_protocol_record()
            self.claim_pointer()
            self.record_sources()
            v27.append_event(self.events, "controller_started", argv=sys.argv, pid=os.getpid(), report_only=report_only)
            if (self.dir / "training_completed.json").exists():
                v27.verify_training_completed(self.dir)
                self.report()
                return "COMPLETED"
            if report_only:
                raise ValueError("--report-only requires a registered TRAIN_COMPLETED run")
            self.stage_b0()
            arrays_list, records = [], []
            for k in self.p["timeline"]["eval_blocks"]:
                arrays, record = self.run_step(k)
                arrays_list.append(arrays)
                records.append(record)
            self.stage_b2(arrays_list, records)
            self.finish_training()
            self.report()
            return "COMPLETED"
        finally:
            lock.close()


# =========================================================================== preflight
def protocol_namespace_check(p):
    """Refuse a duplicate FULL protocol identity (configs or run pointers); unrelated shorthand versions are recorded only."""
    own = p["protocol_id"]
    configs, pointers = [], {}
    for path in sorted((ROOT / "configs").glob("*protocol*.json")):
        if path.resolve() == PROTOCOL.resolve():
            continue
        try:
            if v27.read_json(path).get("protocol_id") == own:
                configs.append(str(path))
        except (ValueError, AttributeError):
            continue
    for path in sorted(LOG_ROOT.glob("*latest.txt")):
        if path.resolve() == POINTER.resolve():
            continue
        target = Path(path.read_text(errors="replace").strip())
        marker = (target if target.is_absolute() else ROOT / target) / "protocol.json"
        if marker.exists() and v27.read_json(marker).get("protocol_id") == own:
            pointers[str(path)] = str(target)
    return {"protocol_id": own, "other_configs_with_same_id": configs, "other_pointers_with_same_id": pointers,
            "passed": not configs and not pointers}


def smoke(p, folder):
    """Synthetic smoke: the two reused trainers chained exactly as in a step (anchor -> frozen anchor -> paired residuals)."""
    q = json_copy(p)
    for block in (q["anchor"]["training"], q["candidate"]["training"]):
        block.update(max_epochs=2, scheduler_t_max=2)
    av, rv = v27.anchor_view(q), v27.residual_view(q)
    fit, selection, cpu = v26run.SyntheticWindows(96, 1), v26run.SyntheticWindows(40, 2), torch.device("cpu")
    trained = v25run.fit_arm(v27.ANCHOR, av, fit=fit, selection=selection, contract={"smoke": "anchor"}, seed=5, device=cpu,
                             out=folder / "anchor" / v27.ANCHOR)
    state = torch.load(folder / "anchor" / v27.ANCHOR / "best.pt", map_location="cpu", weights_only=False)["model"]
    frozen, _ = v26run.build_model(v27.ANCHOR, rv, anchor_state=state)
    reference = v26run.predict(frozen, selection, 0, cpu, 128)
    results = {}
    for arm in v27.RESIDUAL_ARMS:
        init = folder / "s7" / v27.FULL / "init.pt" if arm == v27.POWER else None
        results[arm] = v26run.fit_arm(arm, rv, fit=fit, selection=selection, contract={"smoke": arm}, seed=7, device=cpu,
                                      out=folder / "s7" / arm, anchor_state=state, anchor_reference_kw=reference,
                                      init_reference=init)
    paired = results[v27.FULL]["init_state_sha256"] == results[v27.POWER]["init_state_sha256"]
    anchored = all(r["anchor_state_sha256"] == v26run.state_digest(state) for r in results.values())
    identity = all(r["epoch0_max_abs_diff_vs_anchor_kw"] == 0.0 for r in results.values())
    return {"passed": bool(paired and anchored and identity), "anchor_best_epoch": trained["best_epoch"],
            "paired_init": bool(paired), "anchor_bound": bool(anchored), "epoch0_identity_exact_cpu": bool(identity),
            "data": "synthetic tensors only"}


def json_copy(value):
    import copy
    return copy.deepcopy(value)


def preflight(gpu_index=0):
    p, protocol_sha = load_protocol()
    digest = full_source_digest(protocol_sha)
    folder = PREFLIGHT_DIR / f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}_{digest[:12]}"
    folder.mkdir(parents=True, exist_ok=False)
    v26run.configure_determinism()
    checks = {"namespace": protocol_namespace_check(p)}
    try:
        import pytest  # noqa: F401
        command = [sys.executable, "-m", "pytest", "-q", *V27_TESTS, *REGRESSION_TESTS]
    except ImportError:
        command = [sys.executable, "-m", "unittest", *[t.replace("/", ".")[:-3] for t in V27_TESTS + REGRESSION_TESTS]]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (folder / "tests.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    checks["tests"] = {"command": command, "exit_code": result.returncode, "passed": result.returncode == 0,
                       "tail": (result.stdout + result.stderr).splitlines()[-15:]}
    code = subprocess.run(["bash", "-n", "scripts/train/Kelmarsh_launch_forward_block_residual.sh"], cwd=ROOT).returncode
    checks["bash_syntax"] = {"exit_code": code, "passed": code == 0}
    try:
        b0 = V27Run(folder / "b0", protocol=p, protocol_sha=protocol_sha, require_preflight=False, write_pointer=False,
                    lock_path=PREFLIGHT_DIR / "preflight.lock")
        b0.dir.mkdir(parents=True)
        b0.freeze_protocol_record()
        b0.record_sources()
        blocks = b0.stage_b0()
        checks["B0"] = {"passed": True, "edges_utc": blocks["edges_utc"],
                        "windows": {k: s["windows"] for k, s in blocks["steps"].items()},
                        "evaluation_issue_times": {k: s["evaluation_issue_times"] for k, s in blocks["steps"].items()}}
    except Exception as error:
        checks["B0"] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    try:
        checks["synthetic_smoke"] = smoke(p, folder / "smoke")
    except Exception as error:
        checks["synthetic_smoke"] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    checks["resources"] = {"gpu": v26run.gpu_snapshot(gpu_index), "cpu_count": os.cpu_count(),
                           "decision_if_step_started_now": v27.choose_pair_device(v26run.gpu_snapshot(gpu_index), p["operations"])}
    passed = all(checks[k]["passed"] for k in ("namespace", "tests", "bash_syntax", "B0", "synthetic_smoke"))
    record = {"passed": bool(passed), "full_source_digest": digest, "protocol_sha256": protocol_sha,
              "training_code_sha256": training_code_sha256(), "git": git_info(), "utc": now(), "folder": str(folder),
              "checks": checks}
    v27.write_json(folder / "preflight.json", record)
    if passed:
        v27.write_json(PREFLIGHT_DIR / f"passed_{digest[:16]}.json", record)
    print(f"[PREFLIGHT] passed={passed} tests={checks['tests']['passed']} B0={checks['B0']['passed']} "
          f"smoke={checks['synthetic_smoke']['passed']} namespace={checks['namespace']['passed']} record={folder}", flush=True)
    return record


# =========================================================================== CLI
def print_status():
    if not POINTER.exists():
        print("STATUS=NOT_STARTED")
        return
    directory = Path(POINTER.read_text().strip())
    print(f"RESULT_DIR={directory}")
    for name in ("status.env", "progress.json"):
        if (directory / name).exists():
            print((directory / name).read_text(encoding="utf-8").strip())
    if (directory / "events.jsonl").exists():
        print("LAST_EVENTS:")
        for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()[-6:]:
            print("  " + line[:400])


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
    run = V27Run(args.output.resolve(), gpu_index=args.gpu)
    try:
        run.run(report_only=args.report_only)
        return 0
    except BlockingIOError:
        print("[BUSY] Another v27 controller holds the lock; nothing changed", flush=True)
        return 3
    except Exception as error:
        if args.output.exists():
            status = run.dir / "status.json"
            current = v27.read_json(status) if status.exists() else {}
            if current.get("train") != "TRAIN_COMPLETED":
                run.set_status(train="FAILED", error=f"{type(error).__name__}: {error}")
            v27.append_event(run.events, "controller_failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    sys.exit(main())
