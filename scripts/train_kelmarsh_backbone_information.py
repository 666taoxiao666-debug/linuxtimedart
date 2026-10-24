#!/usr/bin/env python3
"""v25: bounded Kelmarsh 24h->4h numeric-backbone comparison with an information ablation (Wiki off).

TSMixerBasicSmall Full vs PowerOnly (same 3,618 parameters, identical initial state) against the frozen
v20 DLinear and Persistence on the unchanged v20 queue and scaler. Stages:
  A0 contracts + read-only replay of the seed-2024 DLinear checkpoints (no fit)
  A1 seed 2024: f0 Full, f0 PowerOnly, f1 ..., f2 ...            (6 fits)      -> G1 / I1 on internal selection
  A2 only if G1: seeds 2025, 2026 x folds x {Full, PowerOnly, DLinear} (<=18) -> G2 / I2 on internal selection
  A3 only if G2: one outer-validation evaluation of every fold/seed/arm, nothing re-selected
  A4 independent recomputation, report and stop (scripts/report_kelmarsh_backbone_information.py)
Gates are budget rules on development evidence, not significance tests. Never touches the sealed test.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import glob
import hashlib
import inspect
import os
import platform
import random
import re
import resource
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

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data_provider.kelmarsh_hourly import FEATURES, KelmarshSeries
from models.DLinear import Model as DLinearModel
from models.TSMixerBasicSmall import Model as TSMixerModel
from utils import kelmarsh_backbone_information as v25
from utils.metrics import forecast_metrics

PROTOCOL = ROOT / "configs/kelmarsh_backbone_information_protocol_v25.json"
PROTOCOL_SHA256 = "e111a7947004c5a58b486d57c4544a97e14192f966a33740523b16857b947653"
LOG_ROOT = ROOT / "outputs/logs/Kelmarsh"
POINTER = LOG_ROOT / "backbone_information_latest.txt"
LOCK = LOG_ROOT / "backbone_information.lock"
PREFLIGHT_DIR = LOG_ROOT / "backbone_information_preflight"
# Files whose code determines the trained numbers (data, models, metric used for epoch selection).
TRAINING_FILES = ("models/TSMixerBasicSmall.py", "models/DLinear.py", "models/__init__.py",
                  "data_provider/kelmarsh_hourly.py", "data_provider/__init__.py", "utils/metrics.py")
# All v25 code; any change requires a new preflight before further fits (but never invalidates finished fits).
V25_FILES = TRAINING_FILES + ("scripts/train_kelmarsh_backbone_information.py", "utils/kelmarsh_backbone_information.py",
                              "scripts/report_kelmarsh_backbone_information.py", "utils/paired_forecast_statistics.py",
                              "scripts/train/Kelmarsh_launch_backbone_information.sh",
                              "tests/test_tsmixer_basic_small.py", "tests/test_kelmarsh_backbone_information.py",
                              "tests/test_kelmarsh_backbone_resume.py")
V25_TESTS = ("tests/test_tsmixer_basic_small.py", "tests/test_kelmarsh_backbone_information.py",
             "tests/test_kelmarsh_backbone_resume.py")
REGRESSION_TESTS = ("tests/test_kelmarsh_recent_numeric.py", "tests/test_kelmarsh_24h4h.py",
                    "tests/test_paired_gain_estimand.py", "tests/test_strong_baselines_and_statistics.py")
VOLATILE = {"status.json", "status.env", "progress.json", "launch.log", "launcher.pid", "controller.pid",
            "events.jsonl", "experiment_log.md", "fit_progress.json"}


def registrable(path, directory):
    """Files that enter hash manifests: not volatile logs/status, not temporaries, not the suite marker itself."""
    path = Path(path)
    return (path.is_file() and path.name not in VOLATILE and not path.name.endswith(".tmp")
            and path != Path(directory) / "completed.json")


class SimulatedInterrupt(RuntimeError):
    """Raised only by tests to emulate a crash at an epoch boundary or inside an epoch."""


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# =========================================================================== training numerics
# Every function in this block enters the per-fit contract via training_code_sha256().
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism():
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # may only be set once per process
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Prefer deterministic kernels (CUBLAS_WORKSPACE_CONFIG is set before torch import); warn instead of abort.
    torch.use_deterministic_algorithms(True, warn_only=True)
    return determinism_record()


def determinism_record():
    return {"dtype": "float32", "amp": False, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_threads": torch.get_num_threads(), "num_workers": 0}


def make_loader(dataset, seed, *, shuffle=False, batch=64, device=None):
    generator = torch.Generator().manual_seed(int(seed))
    pin = device is not None and torch.device(device).type == "cuda"
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=0, drop_last=False,
                      generator=generator, pin_memory=pin)


@torch.no_grad()
def predict(model, dataset, seed, device, batch=128):
    """Original-scale power forecast; only the power channel of the frozen scaler is inverted."""
    model.eval()
    chunks = []
    for x, _, _, _ in make_loader(dataset, seed, batch=batch, device=device):
        chunks.append(model(x.float().to(device))[..., -1].cpu().numpy())
    scaled = np.concatenate(chunks).astype(float)
    return scaled * dataset.scaler.scale_[-1] + dataset.scaler.mean_[-1]


def build_model(arm, p):
    spec = p["arms"][arm]
    if spec["family"] == "TSMixerBasicSmall":
        c = p["candidate"]
        kwargs = {"seq_len": p["input_len"], "channels": len(p["feature_columns"]), "pred_len": p["pred_len"],
                  "blocks": c["blocks"], "ff_dim": c["feature_hidden_dim"], "dropout": c["dropout"],
                  "norm_eps": c["norm_eps"], "target_slot": p["target_slot"], "input_mode": spec["input_mode"]}
        model, factory, expected = TSMixerModel(**kwargs), "models.TSMixerBasicSmall.Model", c["expected_trainable_parameters"]
    elif spec["family"] == "DLinear":
        kwargs = dict(spec["model_args"])
        model, factory, expected = DLinearModel(SimpleNamespace(**kwargs)), "models.DLinear.Model", spec["expected_parameters"]
    else:
        raise ValueError("Undeclared model family")
    count = int(sum(q.numel() for q in model.parameters() if q.requires_grad))
    if count != int(expected):
        raise ValueError(f"{arm} has {count} trainable parameters, protocol requires {expected}; never enlarge to 'fix'")
    return model, {"factory": factory, "kwargs": kwargs, "trainable_parameters": count}


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_digest(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def atomic_save(obj, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(obj, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def capture_rng(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.device(device).type == "cuda" and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def verify_trained(out, contract):
    trained = v25.read_json(out / "trained.json")
    mismatch = v25.contract_mismatches(trained["contract"], contract)
    if mismatch:
        raise ValueError(f"Registered fit {out} has a different contract ({mismatch}); never mix or overwrite")
    for name, digest in trained["artifacts"].items():
        if v25.file_sha256(out / name) != digest:
            raise ValueError(f"Registered artifact changed: {out / name}")
    return trained


def fit_arm(arm, p, *, fit, selection, contract, seed, device, out, events=None, init_reference=None,
            stop_after_epoch=None, stop_inside_epoch=None):
    """Train one arm with the frozen recipe; epoch-boundary resumable; returns the trained.json record.

    Order exactly as v20 shared.train_stage: seed_all -> construct -> epoch-0 selection -> epochs with
    loader(seed+epoch) shuffling, (1-w)|e|+w e^2 on the standardized power, clip, AdamW, strict-MAE best,
    patience checked at the top of each epoch, cosine step after each epoch.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "trained.json").exists():
        return verify_trained(out, contract) | {"reused_from_registry": True}
    cfg = p["training"]
    weight = float(cfg["loss_mse_weight"])
    if abs((1 - weight) - float(cfg["loss_mae_weight"])) > 1e-12:
        raise ValueError("Loss weights must be (1-w, w)")
    device = torch.device(device)
    seed_all(seed)
    model, factory = build_model(arm, p)
    init_state = cpu_state(model)
    init_digest = state_digest(init_state)
    if init_reference is not None:
        reference = torch.load(init_reference, map_location="cpu", weights_only=False)
        if reference["state_sha256"] != init_digest or state_digest(reference["model"]) != init_digest:
            raise ValueError("Paired arms do not start from the identical initial state")
    init_path = out / "init.pt"
    if init_path.exists():
        saved = torch.load(init_path, map_location="cpu", weights_only=False)
        if saved["state_sha256"] != init_digest or v25.contract_mismatches(saved["contract"], contract):
            raise ValueError("Saved initial state differs from the deterministic construction")
    else:
        atomic_save({"model": init_state, "state_sha256": init_digest, "arm": arm, "seed": seed,
                     "contract": contract}, init_path)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["scheduler_t_max"],
                                                           eta_min=cfg["scheduler_eta_min"])
    truth, persistence = selection.truth_and_anchor()
    rated, power_scale = p["rated_power_kw"], float(fit.scaler.scale_[-1])

    def assess():
        begin = time.monotonic()
        metrics = forecast_metrics(predict(model, selection, seed, device, cfg["eval_batch_size"]), truth,
                                   rated_power=rated)
        return metrics["mae"], metrics, time.monotonic() - begin

    def save_last(epoch):
        atomic_save({"model": cpu_state(model), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "contract": contract, "epoch": epoch, "best": best, "best_epoch": best_epoch, "bad_epochs": bad,
                     "history": history, "timing": timing, "rng": capture_rng(device),
                     "next_shuffle_seed": int(seed) + epoch + 1}, last_path)
        v25.write_json(out / "fit_progress.json", {"last_completed_epoch": epoch, "epoch_in_progress": None})

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    last_path, best_path = out / "last.pt", out / "best.pt"
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        mismatch = v25.contract_mismatches(checkpoint["contract"], contract)
        if mismatch:
            raise ValueError(f"Interrupted fit identity changed: {mismatch}")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history, start = checkpoint["history"], int(checkpoint["epoch"])
        best, best_epoch, bad, timing = checkpoint["best"], checkpoint["best_epoch"], checkpoint["bad_epochs"], checkpoint["timing"]
        restore_rng(checkpoint["rng"], device)
        if not best_path.exists():
            raise ValueError("best.pt missing for an interrupted fit")
        marker = v25.read_json(out / "fit_progress.json") if (out / "fit_progress.json").exists() else {}
        if events is not None:
            v25.append_event(events, "fit_resumed", arm=arm, seed=seed, out=str(out), from_epoch=start,
                             discarded_partial_epoch=marker.get("epoch_in_progress"))
    else:
        score, metrics, seconds = assess()
        history = [{"epoch": 0, "selection": metrics, "assess_seconds": seconds, "lr": optimizer.param_groups[0]["lr"]}]
        best, best_epoch, bad, start = score, 0, 0, 0
        timing = {"train_seconds": 0.0, "assess_seconds": seconds}
        atomic_save({"model": cpu_state(model), "epoch": 0, "contract": contract, "selection_mae_kw": best}, best_path)
        save_last(0)
    for epoch in range(start + 1, int(cfg["max_epochs"]) + 1):
        if bad >= int(cfg["patience"]):
            break
        if stop_after_epoch is not None and epoch > stop_after_epoch:
            raise SimulatedInterrupt(f"after epoch {epoch - 1}")
        v25.write_json(out / "fit_progress.json", {"last_completed_epoch": epoch - 1, "epoch_in_progress": epoch})
        model.train()
        total, batches, abs_kw, points, lr = 0., 0, 0., 0, optimizer.param_groups[0]["lr"]
        begin = time.monotonic()
        for x, y, _, _ in make_loader(fit, int(seed) + epoch, shuffle=True, batch=cfg["batch_size"], device=device):
            if stop_inside_epoch is not None and epoch == stop_inside_epoch[0] and batches == stop_inside_epoch[1]:
                raise SimulatedInterrupt(f"inside epoch {epoch}")
            x, y = x.float().to(device), y.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            difference = model(x)[..., -1] - y[..., -1]
            loss = (1 - weight) * difference.abs().mean() + weight * difference.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; never continue silently")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            optimizer.step()
            total += loss.item()
            batches += 1
            abs_kw += difference.detach().abs().sum().item() * power_scale
            points += difference.numel()
        train_seconds = time.monotonic() - begin
        score, metrics, assess_seconds = assess()
        timing["train_seconds"] += train_seconds
        timing["assess_seconds"] += assess_seconds
        history.append({"epoch": epoch, "train_loss": total / batches, "train_loss_space": "standardized (1-w)|e|+w e^2",
                        "train_mae_kw_running_with_dropout": abs_kw / points, "lr": lr, "selection": metrics,
                        "train_seconds": train_seconds, "assess_seconds": assess_seconds})
        if score < best:
            best, best_epoch, bad = score, epoch, 0
            atomic_save({"model": cpu_state(model), "epoch": epoch, "contract": contract, "selection_mae_kw": best}, best_path)
        else:
            bad += 1
        scheduler.step()
        save_last(epoch)
        v25.write_json(out / "training_history.json", history)
        if events is not None:
            v25.append_event(events, "epoch", arm=arm, seed=seed, out=str(out.relative_to(out.parents[2])),
                             epoch=epoch, train_loss=total / batches, selection_mae_kw=score, best_epoch=best_epoch)
        print(f"[EPOCH] {out.parent.name}/{arm} epoch={epoch} train_loss={total / batches:.6f} "
              f"selection_mae={score:.6f} best_epoch={best_epoch}", flush=True)
    v25.write_json(out / "training_history.json", history)
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model"], strict=True)
    prediction = predict(model, selection, seed, device, cfg["eval_batch_size"])
    replay = forecast_metrics(prediction, truth, rated_power=rated)
    recorded = history[best_epoch]["selection"]
    if history[best_epoch]["epoch"] != best_epoch or abs(replay["mae"] - recorded["mae"]) > 1e-6:
        raise ValueError("best.pt does not reproduce its registered selection MAE")
    np.savez_compressed(out / "selection_predictions.npz", prediction=prediction, truth=truth, persistence=persistence,
                        times=selection.target_times(), turbines=selection.turbines[selection.window_starts],
                        window_starts=selection.window_starts)
    if device.type == "cuda":
        memory = {"kind": "cuda_max_memory_allocated_bytes", "value": int(torch.cuda.max_memory_allocated(device))}
    else:
        memory = {"kind": "process_max_rss_kib_cumulative", "value": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    trained = {"contract": contract, "arm": arm, "seed": int(seed), "best_epoch": int(best_epoch),
               "epochs_completed": len(history) - 1, "stopped_early": len(history) - 1 < int(cfg["max_epochs"]),
               "selection_best": recorded, "selection_replay_abs_mae_diff_kw": abs(replay["mae"] - recorded["mae"]),
               "trainable_parameters": factory["trainable_parameters"], "init_state_sha256": init_digest,
               "best_state_sha256": state_digest(best_checkpoint["model"]), "timing": timing, "peak_memory": memory,
               "device": str(device), "reused": False, "completed_utc": now(),
               "artifacts": {name: v25.file_sha256(out / name) for name in
                             ("init.pt", "best.pt", "last.pt", "training_history.json", "selection_predictions.npz")}}
    v25.write_json(out / "trained.json", trained)
    return trained


TRAINING_FUNCTIONS = (seed_all, configure_determinism, determinism_record, make_loader, predict, build_model,
                      cpu_state, state_digest, capture_rng, restore_rng, verify_trained, fit_arm)


def training_code_sha256():
    """Identity of code that determines trained numbers; orchestration/report edits do not change it."""
    return v25.canonical_sha256({"files": {name: v25.text_sha256_lf(ROOT / name) for name in TRAINING_FILES},
                                 "functions": {f.__name__: inspect.getsource(f) for f in TRAINING_FUNCTIONS},
                                 "torch_major_minor": ".".join(torch.__version__.split(".")[:2])})


def gate_code_sha256():
    """Identity of the gate/statistics module; a change never silently flips a registered decision."""
    return v25.text_sha256_lf(Path(v25.__file__))


# =========================================================================== identity helpers
def load_protocol(path=PROTOCOL, expected=PROTOCOL_SHA256):
    digest = v25.text_sha256_lf(path)
    if expected is not None and digest != expected:
        raise ValueError("Frozen v25 protocol bytes changed; a new protocol needs a new id")
    protocol = v25.read_json(path)
    v25.validate_protocol(protocol)
    return protocol, digest


def load_pin(p, require_resolved=True):
    pin = v25.read_json(ROOT / p["upstream_pin_file"])
    resolved = (bool(re.fullmatch(r"[0-9a-f]{40}", pin.get("commit") or ""))
                and bool(re.fullmatch(r"[0-9a-f]{64}", pin.get("file_sha256") or ""))
                and pin.get("license") == "Apache-2.0")
    if require_resolved and not resolved:
        raise ValueError("Upstream TSMixer commit/file SHA-256 not pinned; run scripts/deploy/v25_server_ops.sh resolve-pin first")
    return pin, resolved


def git_info():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as error:  # informational only
            return f"unavailable: {type(error).__name__}"
    return {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "tracked_changes": git("status", "--porcelain", "--untracked-files=no").splitlines()}


def v25_sources():
    return {name: (v25.file_sha256(ROOT / name) if (ROOT / name).exists() else None) for name in V25_FILES}


def full_source_digest(protocol_sha, pin):
    return v25.canonical_sha256({"protocol_sha256": protocol_sha, "pin": pin,
                                 "sources": {k: v25.text_sha256_lf(ROOT / k) for k in V25_FILES if (ROOT / k).exists()}})


def imported_closure():
    closure = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        try:
            relative = Path(filename).resolve().relative_to(ROOT)
        except ValueError:
            continue
        closure[str(relative).replace("\\", "/")] = v25.file_sha256(ROOT / relative)
    return dict(sorted(closure.items()))


def environment():
    import sklearn
    info = {"utc": now(), "hostname": socket.gethostname(), "python": sys.version, "platform": platform.platform(),
            "executable": sys.executable, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cuda_available": torch.cuda.is_available(), "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "determinism": determinism_record(),
            "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                   "NUMEXPR_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES")},
            "argv": sys.argv}
    if torch.cuda.is_available():
        info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def gpu_snapshot(index=0):
    """Read-only resource probe. Never changes another process or global device settings."""
    if not torch.cuda.is_available():
        return None
    try:
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True, timeout=60).stdout.strip().splitlines()
        # cuda:{index} is a LOGICAL device; map it through CUDA_VISIBLE_DEVICES to nvidia-smi's physical index/UUID.
        visible = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
        token = visible[index] if visible else str(index)
        row = next(line.split(", ") for line in gpus
                   if line.split(", ")[0].strip() == token or line.split(", ")[1].strip().startswith(token))
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True, timeout=60).stdout.strip().splitlines()
        others = [int(parts[1]) for parts in (a.split(", ") for a in apps if a.strip())
                  if parts[0].strip() == row[1].strip() and parts[1].strip().isdigit() and int(parts[1]) != os.getpid()]
        return {"index": index, "total_mib": float(row[2]), "free_mib": float(row[3]), "other_compute_pids": others,
                "source": "nvidia-smi"}
    except Exception as error:
        free, total = torch.cuda.mem_get_info(index)
        return {"index": index, "total_mib": total / 2 ** 20, "free_mib": free / 2 ** 20, "other_compute_pids": ["unknown"],
                "source": f"torch.mem_get_info ({type(error).__name__}); others assumed present"}


def verify_completed_suite(directory):
    """Every registered completed.json (suite and nested) must still hash-match."""
    directory = Path(directory)
    markers = [directory / "completed.json"] + sorted((directory / "runs").glob("f*_s*/*/completed.json"))
    for marker in markers:
        for name, digest in v25.read_json(marker).items():
            if v25.file_sha256(marker.parent / name) != digest:
                raise ValueError(f"Registered reference artifact changed: {marker.parent / name}")
    return len(markers)


# =========================================================================== controller
class V25Run:
    def __init__(self, directory, *, protocol=None, protocol_sha=None, pin=None, series_factory=None, gpu_index=0,
                 require_preflight=True, test_reference=None, write_pointer=True, lock_path=LOCK, reference_root=ROOT):
        self.dir = Path(directory)
        if protocol is None:
            protocol, protocol_sha = load_protocol()
        self.p, self.protocol_sha = protocol, protocol_sha
        self.pin = pin if pin is not None else load_pin(protocol, require_resolved=False)[0]
        self.folds = list(protocol["folds"])
        self.series_factory = series_factory or (lambda fold: KelmarshSeries(
            ROOT / protocol["data_path"], fold=fold, input_len=protocol["input_len"], horizon=protocol["pred_len"],
            stride=protocol["train_stride"], start=protocol["start_utc"], end=protocol["end_utc_exclusive"]))
        self.gpu_index, self.require_preflight = gpu_index, require_preflight
        self.test_reference, self.write_pointer, self.lock_path = test_reference, write_pointer, lock_path
        self.reference_root = Path(reference_root)
        self.events = self.dir / "events.jsonl"
        self.access = v25.DataAccessLog(self.events)
        self.prepared, self.fit_calls = {}, []
        self.tol = protocol["gates"]["numeric_tie_tolerance_kw"]

    # ------------------------------------------------------------------ bookkeeping
    def rel(self, path):
        return str(Path(path).relative_to(self.dir)).replace("\\", "/")

    def set_status(self, **fields):
        path = self.dir / "status.json"
        status = v25.read_json(path) if path.exists() else {"train": "NOT_STARTED", "report": "PENDING", "stage": None}
        status.update(fields, updated_utc=now())
        v25.write_json(path, status)
        overall = ("COMPLETED" if status.get("report") == "REPORT_COMPLETED" and status["train"] == "TRAIN_COMPLETED"
                   else status["report"] if status.get("report") == "REPORT_FAILED" else status["train"])
        (self.dir / "status.env").write_text(f"STATUS={overall}\nTRAIN={status['train']}\nREPORT={status['report']}\n"
                                             f"STAGE={status.get('stage')}\n", encoding="utf-8")
        return status

    def log(self, title, **items):
        lines = [f"\n## {now()} {title}\n"]
        lines += [f"- {key}: {value}\n" for key, value in items.items()]
        with (self.dir / "experiment_log.md").open("a", encoding="utf-8") as handle:
            handle.writelines(lines)

    def freeze_protocol_record(self):
        record = {"protocol_id": self.p["protocol_id"], "protocol_sha256_lf": self.protocol_sha, "protocol": self.p,
                  "upstream_pin": self.pin, "upstream_pin_sha256": v25.canonical_sha256(self.pin)}
        path = self.dir / "protocol.json"
        if path.exists():
            saved = v25.read_json(path)
            if v25.canonical_sha256(saved) != v25.canonical_sha256(record):
                raise ValueError("Frozen protocol/pin changed on resume; start a new protocol id instead")
        else:
            v25.write_json(path, record)
        self.pin_sha = record["upstream_pin_sha256"]

    def claim_pointer(self):
        if not self.write_pointer:
            return
        POINTER.parent.mkdir(parents=True, exist_ok=True)
        if POINTER.exists() and POINTER.read_text().strip() and Path(POINTER.read_text().strip()).resolve() != self.dir.resolve():
            raise ValueError(f"Pointer already registers another v25 run: {POINTER.read_text().strip()}; use --resume")
        POINTER.write_text(str(self.dir.resolve()), encoding="utf-8")

    def record_sources(self):
        manifest = {"utc": now(), "training_code_sha256": training_code_sha256(), "training_files": list(TRAINING_FILES),
                    "training_functions": [f.__name__ for f in TRAINING_FUNCTIONS], "gate_code_sha256": gate_code_sha256(),
                    "v25_sources": v25_sources(), "full_source_digest": full_source_digest(self.protocol_sha, self.pin),
                    "imported_closure_under_root": imported_closure(), "git": git_info()}
        path = self.dir / "source_manifest.json"
        if not path.exists():
            v25.write_json(path, manifest)
        elif v25.read_json(path)["full_source_digest"] != manifest["full_source_digest"]:
            v25.append_event(self.events, "source_changed_on_resume", previous=v25.read_json(path)["full_source_digest"],
                             current=manifest["full_source_digest"],
                             training_code_changed=v25.read_json(path)["training_code_sha256"] != manifest["training_code_sha256"],
                             rule="finished fits are reused only if their per-fit contract still matches")
            v25.write_json(self.dir / f"source_manifest_resume_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}.json", manifest)
        if not (self.dir / "environment.json").exists():
            v25.write_json(self.dir / "environment.json", environment())
        else:
            v25.append_event(self.events, "environment_on_resume", environment=environment())
        if not (self.dir / "parameter_changes.json").exists():
            v25.write_json(self.dir / "parameter_changes.json", parameter_changes(self.p))
        self.training_code = manifest["training_code_sha256"]
        self.source_digest = manifest["full_source_digest"]
        return manifest

    def check_preflight(self):
        if not self.require_preflight:
            return
        record = PREFLIGHT_DIR / f"passed_{self.source_digest[:16]}.json"
        if not record.exists() or v25.read_json(record).get("full_source_digest") != self.source_digest \
                or not v25.read_json(record).get("passed"):
            raise ValueError("No passed --preflight-only record for the current code/protocol; run it before any fit")

    # ------------------------------------------------------------------ data
    def prepare_fold(self, fold, stage):
        self.access.request(stage, "fit", fold)
        self.access.request(stage, "selection", fold)
        if fold not in self.prepared:
            series = self.series_factory(fold)
            self.prepared[fold] = (series, v25.inner_datasets(series))
        return self.prepared[fold]

    def fold_manifest(self, fold, series, data):
        selection = data["selection"]
        truth, persistence = selection.truth_and_anchor()
        return {"fold": fold, "bounds_utc": {k: str(v) for k, v in series.bounds.items()},
                "fit_windows": len(data["fit"]), "selection_windows": len(selection),
                "fit_window_hash": v25.window_hash(data["fit"].window_starts),
                "selection_window_hash": v25.window_hash(selection.window_starts),
                "fit_target_range_utc": [str(data["fit"].target_times().min()), str(data["fit"].target_times().max())],
                "selection_target_range_utc": [str(selection.target_times().min()), str(selection.target_times().max())],
                "scaler_mean": series.scaler.mean_.tolist(), "scaler_scale": series.scaler.scale_.tolist(),
                "scaler_sha256": v25.canonical_sha256([series.scaler.mean_.tolist(), series.scaler.scale_.tolist()]),
                "selection_truth_sha256": v25.array_sha256(truth), "selection_persistence_sha256": v25.array_sha256(persistence),
                "selection_times_sha256": v25.array_sha256(selection.target_times()),
                "selection_turbines_sha256": v25.array_sha256(selection.turbines[selection.window_starts]),
                "selection_issue_coverage": v25.issue_time_coverage(selection.target_times()[:, 0] - np.timedelta64(1, "h"),
                                                                    self.p["eval_stride"]),
                "series_audit": series.audit, "outer_validation_instantiated": False}

    # ------------------------------------------------------------------ A0
    def verify_reference_identity(self):
        ref = self.p["reference"]
        if v25.text_sha256_lf(ROOT / ref["protocol_path"]) != ref["protocol_sha256"]:
            raise ValueError("v20 protocol bytes changed")
        refdir = self.reference_root / ref["dir"]
        if v25.file_sha256(refdir / "completed.json") != ref["completed_sha256"]:
            raise ValueError("v20 completed.json identity changed")
        markers = verify_completed_suite(refdir)
        if v25.file_sha256(ROOT / self.p["data_path"]) != self.p["data_sha256"]:
            raise ValueError("Prepared Kelmarsh source fingerprint changed")
        return {"v20_dir": ref["dir"], "v20_completed_sha256": ref["completed_sha256"], "verified_completed_markers": markers,
                "data_sha256": self.p["data_sha256"]}

    def verify_reference_fold(self, fold, series, data):
        ref = self.p["reference"]
        spec = next(item for item in ref["dlinear_checkpoints"] if item["fold"] == fold)
        run = self.reference_root / ref["dir"] / "runs" / f"f{fold}_s{ref['seed']}" / "DLinear"
        trained, contract = v25.read_json(run / "trained.json"), v25.read_json(run / "trained.json")["contract"]
        errors = []
        checkpoint_sha = v25.file_sha256(run / "checkpoint.pt")
        if checkpoint_sha != spec["checkpoint_sha256"] or trained["artifacts"]["checkpoint.pt"] != spec["checkpoint_sha256"]:
            errors.append("checkpoint sha")
        if trained["best_epoch"] != spec["best_epoch"]:
            errors.append("best epoch")
        for key, stage in (("fit_window_hash", "fit"), ("selection_window_hash", "selection")):
            computed = v25.window_hash(data[stage].window_starts)
            if not (contract[key] == spec[key] == computed):
                errors.append(f"{key}: registered {contract[key][:12]} design {spec[key][:12]} computed {computed[:12]}")
        if not (np.array_equal(series.scaler.mean_, contract["scaler_mean"]) and np.array_equal(series.scaler.scale_, contract["scaler_scale"])):
            errors.append("scaler arrays")
        expected = {"data_sha256": self.p["data_sha256"], "protocol_sha256": ref["protocol_sha256"], "name": "DLinear",
                    "seed": ref["seed"], "pretrain": False, "initial_checkpoint_sha256": None}
        errors += [f"contract {k}" for k, v in expected.items() if contract.get(k) != v]
        history = v25.read_json(run / "training_history.json")
        entry = next(r["selection"] for r in history if r["epoch"] == spec["best_epoch"])
        if entry["mae"] != spec["internal_mae_kw"] or entry["rmse"] != spec["internal_rmse_kw"]:
            errors.append("registered internal MAE/RMSE")
        if v25.read_json(run / "manifest.json")["parameters"] != ref["expected_parameters"]:
            errors.append("parameter count")
        if errors:
            raise ValueError(f"v20 DLinear f{fold} reference identity failed: {errors}; investigate, never bypass hashes")
        model, _ = build_model("DLinear", self.p)
        state = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(state, strict=True)
        cpu = torch.device("cpu")
        selection = data["selection"]
        truth, persistence = selection.truth_and_anchor()
        prediction = predict(model, selection, ref["seed"], cpu, self.p["training"]["eval_batch_size"])
        replay = forecast_metrics(prediction, truth, rated_power=self.p["rated_power_kw"])
        fit_truth, _ = data["fit"].truth_and_anchor()
        fit_replay = forecast_metrics(predict(model, data["fit"], ref["seed"], cpu, self.p["training"]["eval_batch_size"]),
                                      fit_truth, rated_power=self.p["rated_power_kw"])
        diff = {k: abs(replay[k] - spec[f"internal_{k}_kw"]) for k in ("mae", "rmse")}
        if max(diff.values()) > ref["replay_tolerance_kw"]:
            raise ValueError(f"Replayed v20 DLinear f{fold} does not reproduce internal metrics: {diff}")
        folder = self.dir / "reference_replay"
        folder.mkdir(parents=True, exist_ok=True)
        arrays = {"prediction": prediction, "truth": truth, "persistence": persistence, "times": selection.target_times(),
                  "turbines": selection.turbines[selection.window_starts], "window_starts": selection.window_starts}
        target = folder / f"f{fold}_s{ref['seed']}_DLinear_selection_predictions.npz"
        if target.exists():  # registered once; a resume must reproduce it exactly, never rewrite it
            with np.load(target, allow_pickle=False) as saved:
                v25.assert_aligned(arrays, dict(saved), keys=tuple(arrays), label=f"reference replay f{fold}")
        else:
            np.savez_compressed(target, **arrays)
        return {"fold": fold, "checkpoint": self.rel_root(run / "checkpoint.pt"), "checkpoint_sha256": checkpoint_sha,
                "best_epoch": spec["best_epoch"], "registered_selection": {"mae": entry["mae"], "rmse": entry["rmse"]},
                "replay_device": "cpu", "replay_selection": {"mae": replay["mae"], "rmse": replay["rmse"]},
                "replay_abs_diff_kw": diff, "replay_fit_informational": {"mae": fit_replay["mae"], "rmse": fit_replay["rmse"]},
                "parameters": ref["expected_parameters"], "retrained": False}

    @staticmethod
    def rel_root(path):
        """Repository-relative path when possible (portable manifests), else absolute."""
        try:
            return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            return str(Path(path).resolve())

    def stage_a0(self):
        self.set_status(stage="A0", train="RUNNING")
        identity = (self.verify_reference_identity() if self.test_reference is None
                    else {"verification": "SKIPPED_TEST_OVERRIDE_NEVER_FROM_CLI"})
        folds, reference, persistence = {}, {}, {}
        for fold in self.folds:
            series, data = self.prepare_fold(fold, "A0")
            folds[str(fold)] = self.fold_manifest(fold, series, data)
            truth, anchor = data["selection"].truth_and_anchor()
            pm = forecast_metrics(anchor, truth, rated_power=self.p["rated_power_kw"])
            persistence[str(fold)] = {"mae": pm["mae"], "rmse": pm["rmse"], "mse": pm["mse"], "r2": pm["r2"]}
            reference[str(fold)] = (self.test_reference[fold] if self.test_reference is not None
                                    else self.verify_reference_fold(fold, series, data))
        a0 = {"stage": "A0", "identity": identity, "data": folds, "persistence_inner": persistence,
              "dlinear_seed2024_inner": {f: {k: (r["registered_selection"][k] if "registered_selection" in r else r[k])
                                             for k in ("mae", "rmse")} for f, r in reference.items()},
              "new_fits": 0, "outer_access": False}
        stable = {k: a0[k] for k in ("data", "persistence_inner", "dlinear_seed2024_inner")}
        path = self.dir / "a0_result.json"
        if path.exists() and v25.canonical_sha256({k: v25.read_json(path)[k] for k in stable}) != v25.canonical_sha256(stable):
            raise ValueError("A0 contracts differ from the registered A0 of this run; data or references drifted")
        if not path.exists():
            v25.write_json(self.dir / "data_manifest.json", {"protocol_sha256": self.protocol_sha, "folds": folds,
                           "data_path": self.p["data_path"], "data_sha256": self.p["data_sha256"],
                           "feature_columns": FEATURES, "sealed_test_requested": False})
            v25.write_json(self.dir / "reference_manifest.json", {"identity": identity, "dlinear_seed2024": reference,
                           "v21_historical": self.p["v21_historical"], "reuse": "seed-2024 DLinear never retrained"})
            v25.write_json(path, a0 | {"completed_utc": now()})
            self.log("A0 contracts and baseline replay", purpose="verify v20 queue/scaler/checkpoints before any fit",
                     persistence_inner=persistence, dlinear_inner=a0["dlinear_seed2024_inner"], new_fits=0)
        v25.append_event(self.events, "A0_verified")
        self.a0 = v25.read_json(path)
        return self.a0

    # ------------------------------------------------------------------ devices and fits
    def pair_device(self, fold, seed, arms):
        path = self.dir / "runs" / f"f{fold}_s{seed}" / "pair_device.json"
        ops = self.p["operations"]
        if path.exists():
            record = v25.read_json(path)
            folder = path.parent
            if all((folder / arm / "trained.json").exists() or (folder / arm / "reused_cache.json").exists() for arm in arms):
                # Finished pair: only registry verification follows; never wait for, or touch, the GPU.
                return torch.device(f"cuda:{self.gpu_index}" if record["device"] == "cuda" else "cpu")
            if record["device"] == "cuda":
                waited, began = False, time.monotonic()
                while True:
                    snapshot = gpu_snapshot(self.gpu_index)
                    if snapshot and snapshot["free_mib"] / 1024. >= ops["resume_gpu_free_gib_min"]:
                        break
                    if not waited:
                        v25.append_event(self.events, "waiting_for_registered_pair_device", fold=fold, seed=seed, snapshot=snapshot)
                        waited = True
                    if time.monotonic() - began > ops["resume_device_wait_seconds_max"]:
                        raise TimeoutError("Registered pair device unavailable; resume later (no silent device switch)")
                    time.sleep(60)
        else:
            record = v25.choose_pair_device(gpu_snapshot(self.gpu_index), ops) | {"fold": fold, "seed": seed, "decided_utc": now()}
            v25.write_json(path, record)
            v25.append_event(self.events, "pair_device_decided", **record)
        if record["device"] == "cuda":
            device = torch.device(f"cuda:{self.gpu_index}")
            torch.cuda.set_per_process_memory_fraction(record["memory_fraction"], device)
            return device
        return torch.device("cpu")

    def contract(self, arm, fold, seed, data, device):
        _, factory = build_model(arm, self.p)
        series = data["fit"].series
        return {"protocol_sha256": self.protocol_sha, "upstream_pin_sha256": self.pin_sha, "training_code_sha256": self.training_code,
                "data_sha256": self.p["data_sha256"], "arm": arm, "fold": int(fold), "seed": int(seed),
                "model_factory": factory["factory"], "model_kwargs": factory["kwargs"],
                "trainable_parameters": factory["trainable_parameters"], "scaler_mean": series.scaler.mean_.tolist(),
                "scaler_scale": series.scaler.scale_.tolist(), "fit_window_hash": v25.window_hash(data["fit"].window_starts),
                "selection_window_hash": v25.window_hash(data["selection"].window_starts), "training": self.p["training"],
                "device_type": torch.device(device).type, "determinism": determinism_record()}

    def registered_new_fits(self):
        return sorted(self.rel(p.parent) for p in (self.dir / "runs").glob("f*_s*/*/trained.json")
                      if not v25.read_json(p).get("reused", False))

    def fit(self, fold, seed, arm, device, stage, init_reference=None):
        _, data = self.prepare_fold(fold, stage)
        out = self.dir / "runs" / f"f{fold}_s{seed}" / arm
        contract = self.contract(arm, fold, seed, data, device)
        if not (out / "trained.json").exists():
            self.check_preflight()
            started = (out / "last.pt").exists()
            if not started and len(self.registered_new_fits()) + 1 > self.p["budget"]["new_fits_max"]:
                raise ValueError("The 24-fit budget would be exceeded; refuse")
            self.set_status(stage=stage, train="RUNNING")
            v25.write_json(self.dir / "progress.json", {"stage": stage, "fold": fold, "seed": seed, "arm": arm,
                                                         "device": str(device), "utc": now()})
            v25.append_event(self.events, "fit_started" if not started else "fit_resuming", stage=stage, fold=fold,
                             seed=seed, arm=arm, device=str(device))
        self.fit_calls.append((fold, seed, arm))
        trained = fit_arm(arm, self.p, fit=data["fit"], selection=data["selection"], contract=contract, seed=seed,
                          device=device, out=out, events=self.events, init_reference=init_reference)
        if not trained.get("reused_from_registry"):
            v25.append_event(self.events, "fit_completed", stage=stage, fold=fold, seed=seed, arm=arm,
                             best_epoch=trained["best_epoch"], selection_mae_kw=trained["selection_best"]["mae"],
                             selection_rmse_kw=trained["selection_best"]["rmse"])
            print(f"[FIT] {stage} f{fold} s{seed} {arm} best_epoch={trained['best_epoch']} "
                  f"MAE={trained['selection_best']['mae']:.6f} RMSE={trained['selection_best']['rmse']:.6f}", flush=True)
        gc.collect()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        return trained

    def tsmixer_pair(self, fold, seed, stage, device):
        full = self.fit(fold, seed, v25.PRIMARY, device, stage)
        self.fit(fold, seed, v25.INFO_CONTROL, device, stage,
                 init_reference=self.dir / "runs" / f"f{fold}_s{seed}" / v25.PRIMARY / "init.pt")
        return full

    # ------------------------------------------------------------------ DLinear cache (exact same contract only)
    def find_dlinear_cache(self, fold, seed, data):
        rule = self.p["dlinear_cache"]
        for path in sorted(glob.glob(str(ROOT / rule["search_glob"].format(fold=fold, seed=seed)))):
            run = Path(path).parent
            if self.dir.resolve() in run.resolve().parents:
                continue
            try:
                trained = v25.read_json(path)
                c = trained["contract"]
                ok = (c.get("name") == "DLinear" and c.get("seed") == seed and c.get("pretrain") is False
                      and c.get("initial_checkpoint_sha256") is None and c.get("data_sha256") == self.p["data_sha256"]
                      and c.get("protocol_sha256") == rule["required_contract_protocol_sha256"]
                      and c.get("fit_window_hash") == v25.window_hash(data["fit"].window_starts)
                      and c.get("selection_window_hash") == v25.window_hash(data["selection"].window_starts)
                      and np.array_equal(c.get("scaler_mean"), data["fit"].scaler.mean_)
                      and np.array_equal(c.get("scaler_scale"), data["fit"].scaler.scale_))
                if not ok:
                    continue
                suite = run.parents[2]
                verify_completed_suite(suite)
                if v25.file_sha256(run / "checkpoint.pt") != trained["artifacts"]["checkpoint.pt"]:
                    continue
                history = v25.read_json(run / "training_history.json")
                entry = next(r["selection"] for r in history if r["epoch"] == trained["best_epoch"])
                return {"source_run": self.rel_root(run), "checkpoint_sha256": trained["artifacts"]["checkpoint.pt"],
                        "best_epoch": trained["best_epoch"], "selection": {"mae": entry["mae"], "rmse": entry["rmse"]},
                        "raw_predictions": self.rel_root(run / "raw_predictions.npz") if (run / "raw_predictions.npz").exists() else None}
            except (KeyError, ValueError, StopIteration, OSError):
                continue
        return None

    def dlinear(self, fold, seed, device, stage):
        out = self.dir / "runs" / f"f{fold}_s{seed}" / v25.BASELINE
        registration = out / "reused_cache.json"
        if registration.exists():
            return v25.read_json(registration)
        if not (out / "trained.json").exists() and not (out / "last.pt").exists():
            _, data = self.prepare_fold(fold, stage)
            cache = self.find_dlinear_cache(fold, seed, data)
            if cache is not None:
                v25.write_json(registration, cache | {"reused": True, "registered_utc": now()})
                v25.append_event(self.events, "dlinear_cache_reused", fold=fold, seed=seed, **cache)
                return v25.read_json(registration)
        return self.fit(fold, seed, v25.BASELINE, device, stage)

    # ------------------------------------------------------------------ records and gates
    def inner_record(self, fold, seed, arm):
        if arm == v25.BASELINE and seed == self.p["reference"]["seed"]:
            values, source = self.a0["dlinear_seed2024_inner"][str(fold)], "v20 registered history (not retrained)"
        else:
            base = self.dir / "runs" / f"f{fold}_s{seed}" / arm
            if (base / "reused_cache.json").exists():
                values, source = v25.read_json(base / "reused_cache.json")["selection"], "exact-contract cache"
            elif (base / "trained.json").exists():
                values, source = v25.read_json(base / "trained.json")["selection_best"], "v25 fit"
            else:
                return None
        return {"fold": int(fold), "seed": int(seed), "arm": arm, "mae": values["mae"], "rmse": values["rmse"], "source": source}

    def records(self, seeds, arms):
        rows = [self.inner_record(f, s, a) for f in self.folds for s in seeds for a in arms]
        return [r for r in rows if r is not None]

    def persistence_inner(self):
        return {int(f): v for f, v in self.a0["persistence_inner"].items()}

    def register_gate(self, path, payload, key):
        if path.exists():
            old = v25.read_json(path)
            if old[key]["passed"] != payload[key]["passed"] or \
                    v25.canonical_sha256(old["records"]) != v25.canonical_sha256(payload["records"]):
                raise ValueError(f"Registered {key} decision/records would change; refuse and review")
            if old["gate_code_sha256"] != payload["gate_code_sha256"]:
                v25.append_event(self.events, "gate_code_changed_same_decision", gate=key)
            return old
        v25.write_json(path, payload)
        return payload

    def gate1(self):
        g = self.p["gates"]
        records = self.records([g["G1"]["seed"]], (v25.PRIMARY, v25.INFO_CONTROL, v25.BASELINE))
        payload = {"G1": v25.gate_g1(records, self.persistence_inner(), g["G1"], self.folds, self.tol),
                   "I1": v25.info_i1(records, g["I1"], self.folds, self.tol), "records": records,
                   "gate_code_sha256": gate_code_sha256(), "evaluated_utc": now(), "scope": "internal selection only"}
        payload = self.register_gate(self.dir / "inner_gate_seed2024.json", payload, "G1")
        v25.append_event(self.events, "G1", passed=payload["G1"]["passed"], I1=payload["I1"]["supported"])
        self.log("A1 complete: G1/I1", hypothesis="TSMixerFull beats DLinear and Persistence on internal selection (seed 2024)",
                 G1_passed=payload["G1"]["passed"], G1_checks=payload["G1"].get("checks"), I1_supported=payload["I1"]["supported"],
                 next_step="A2 fixed seeds 2025/2026" if payload["G1"]["passed"] else "stop: no outer predictions")
        print(f"[G1] passed={payload['G1']['passed']} I1={payload['I1']['supported']}", flush=True)
        return payload["G1"]

    def gate2(self):
        g = self.p["gates"]
        records = self.records(g["G2"]["seeds"], v25.ALL_ARMS)
        payload = {"G2": v25.gate_g2(records, self.persistence_inner(), g["G2"], self.folds, self.tol),
                   "I2": v25.info_i2(records, g["I2"], self.folds, self.tol), "records": records,
                   "gate_code_sha256": gate_code_sha256(), "evaluated_utc": now(), "scope": "internal selection only"}
        payload = self.register_gate(self.dir / "inner_gate_all_seeds.json", payload, "G2")
        v25.append_event(self.events, "G2", passed=payload["G2"]["passed"], I2=payload["I2"]["supported"])
        self.log("A2 complete: G2/I2", G2_passed=payload["G2"]["passed"], G2_checks=payload["G2"].get("checks"),
                 I2_supported=payload["I2"]["supported"],
                 next_step="A3 single outer evaluation" if payload["G2"]["passed"] else "stop: no outer predictions")
        print(f"[G2] passed={payload['G2']['passed']} I2={payload['I2']['supported']}", flush=True)
        return payload["G2"]

    # ------------------------------------------------------------------ stages
    def stage_a1(self):
        seed = self.p["seeds"]["A1"][0]
        for fold in self.folds:
            self.tsmixer_pair(fold, seed, "A1", self.pair_device(fold, seed, v25.TSMIXER_ARMS))
        return self.gate1()

    def stage_a2(self):
        for seed in self.p["seeds"]["A2"]:
            for fold in self.folds:
                device = self.pair_device(fold, seed, v25.ALL_ARMS)
                self.tsmixer_pair(fold, seed, "A2", device)
                self.dlinear(fold, seed, device, "A2")
        return self.gate2()

    def outer_freeze(self):
        fits = {self.rel(p.parent): {"trained_sha256": v25.file_sha256(p),
                                     "best_sha256": v25.file_sha256(p.parent / "best.pt")}
                for p in sorted((self.dir / "runs").glob("f*_s*/*/trained.json"))}
        freeze = {"protocol.json": v25.file_sha256(self.dir / "protocol.json"),
                  "inner_gate_seed2024.json": v25.file_sha256(self.dir / "inner_gate_seed2024.json"),
                  "inner_gate_all_seeds.json": v25.file_sha256(self.dir / "inner_gate_all_seeds.json"),
                  "fits": fits, "cache_registrations": {self.rel(p): v25.file_sha256(p)
                                                        for p in sorted((self.dir / "runs").glob("f*_s*/*/reused_cache.json"))}}
        path = self.dir / "outer_freeze.json"
        if path.exists():
            if v25.canonical_sha256(v25.read_json(path)["freeze"]) != v25.canonical_sha256(freeze):
                raise ValueError("Selected checkpoints/gates changed after the outer freeze")
        else:
            v25.write_json(path, {"freeze": freeze, "frozen_utc": now(), "rule": "written before any outer request"})
        return freeze

    def outer_device(self, fold, seed):
        record = v25.read_json(self.dir / "runs" / f"f{fold}_s{seed}" / "pair_device.json")
        if record["device"] == "cuda" and torch.cuda.is_available():
            snapshot = gpu_snapshot(self.gpu_index)
            if snapshot and snapshot["free_mib"] / 1024. >= self.p["operations"]["resume_gpu_free_gib_min"]:
                device = torch.device(f"cuda:{self.gpu_index}")
                torch.cuda.set_per_process_memory_fraction(record["memory_fraction"], device)
                return device
        if record["device"] == "cuda":
            v25.append_event(self.events, "outer_inference_on_cpu", fold=fold, seed=seed,
                             note="inference device only; weights and selection unchanged")
        return torch.device("cpu")

    def stage_a3(self):
        if not v25.read_json(self.dir / "inner_gate_all_seeds.json")["G2"]["passed"]:
            raise PermissionError("A3 requires a registered passed G2")
        self.set_status(stage="A3", train="RUNNING")
        self.outer_freeze()
        seeds = self.p["gates"]["G2"]["seeds"]
        index = {}
        refdir = self.reference_root / self.p["reference"]["dir"]
        for fold in self.folds:
            series, data = self.prepare_fold(fold, "A3")
            self.access.request("A3", "validation", fold, g2_passed=True)
            validation = series.subset("validation")
            truth, persistence = validation.truth_and_anchor()
            cohort = {"times": validation.target_times(), "turbines": validation.turbines[validation.window_starts],
                      "window_starts": validation.window_starts, "truth": truth, "persistence": persistence}
            with np.load(refdir / "runs" / f"f{fold}_s2024/Persistence/raw_predictions.npz", allow_pickle=False) as saved:
                v25.assert_aligned(cohort, dict(saved), label=f"v20 Persistence f{fold}")
            cohort_path = self.dir / "runs" / f"f{fold}_outer_cohort.npz"
            if not cohort_path.exists():
                np.savez_compressed(cohort_path, **cohort)
            index[f"f{fold}/Persistence"] = {"fold": fold, "seed": None, "arm": "Persistence", "path": self.rel_root(cohort_path),
                                              "sha256": v25.file_sha256(cohort_path), "array": "persistence"}
            for seed in seeds:
                for arm in v25.ALL_ARMS:
                    index[f"f{fold}/s{seed}/{arm}"] = self.outer_prediction(fold, seed, arm, validation, cohort)
        v25.write_json(self.dir / "outer_index.json", {"entries": index, "scope": "outer validation, development evidence",
                                                     "selection_on_outer": False, "registered_utc": now()})
        self.log("A3 single outer evaluation", folds=self.folds, seeds=seeds, arms=list(v25.ALL_ARMS),
                 note="every fold/seed/arm registered, harmed ones included; no outer selection or refit")
        return index

    def outer_prediction(self, fold, seed, arm, validation, cohort):
        base = self.dir / "runs" / f"f{fold}_s{seed}" / arm
        if arm == v25.BASELINE and seed == self.p["reference"]["seed"]:
            path = self.reference_root / self.p["reference"]["dir"] / "runs" / f"f{fold}_s{seed}" / "DLinear" / "raw_predictions.npz"
            source = "v20 registered outer prediction, hash-verified, not recomputed"
        elif (base / "reused_cache.json").exists():
            raw = v25.read_json(base / "reused_cache.json")["raw_predictions"]
            if raw is None:
                raise ValueError("Cached DLinear has no registered outer prediction")
            path, source = ROOT / raw, "exact-contract cached outer prediction"
        else:
            path, source = base / "outer_predictions.npz", "v25 best.pt"
            if not path.exists():
                trained = v25.read_json(base / "trained.json")
                device = self.outer_device(fold, seed)
                model, _ = build_model(arm, self.p)
                state = torch.load(base / "best.pt", map_location="cpu", weights_only=False)["model"]
                if state_digest(state) != trained["best_state_sha256"]:
                    raise ValueError("best.pt differs from the registered selected state")
                model.load_state_dict(state, strict=True)
                model.to(device)
                begin = time.monotonic()
                prediction = predict(model, validation, seed, device, self.p["training"]["eval_batch_size"])
                seconds = time.monotonic() - begin
                np.savez_compressed(path, prediction=prediction, **cohort)
                metrics = forecast_metrics(prediction, cohort["truth"], rated_power=self.p["rated_power_kw"])
                v25.write_json(base / "result.json", {"arm": arm, "fold": fold, "seed": seed, "split": "outer_validation",
                               "best_epoch": trained["best_epoch"], "metrics": metrics, "inference_seconds": seconds,
                               "inference_device": str(device), "outer_predictions_sha256": v25.file_sha256(path),
                               "evidence_scope": "development; outer previously inspected in earlier versions"})
                print(f"[OUTER] f{fold} s{seed} {arm} MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f}", flush=True)
        with np.load(path, allow_pickle=False) as saved:
            v25.assert_aligned(cohort, dict(saved), label=f"outer f{fold} s{seed} {arm}")
        return {"fold": fold, "seed": seed, "arm": arm, "path": self.rel_root(path), "sha256": v25.file_sha256(path),
                "array": "prediction", "source": source}

    def finish_training(self, outer_evaluated):
        for marker in sorted((self.dir / "runs").glob("f*_s*/*/trained.json")) + sorted(
                (self.dir / "runs").glob("f*_s*/*/reused_cache.json")):
            folder = marker.parent
            v25.write_json(folder / "completed.json", {p.name: v25.file_sha256(p) for p in sorted(folder.iterdir())
                                                        if registrable(p, self.dir) and p.name != "completed.json"})
        hashes = {self.rel(p): v25.file_sha256(p) for p in sorted(self.dir.rglob("*"))
                  if registrable(p, self.dir) and p.name != "training_completed.json"}
        v25.write_json(self.dir / "training_completed.json", {"outer_evaluated": bool(outer_evaluated), "hashes": hashes,
                       "registered_new_fits": self.registered_new_fits(), "completed_utc": now()})
        self.set_status(stage="TRAINING_DONE", train="TRAIN_COMPLETED", outer_evaluated=bool(outer_evaluated))
        v25.append_event(self.events, "TRAIN_COMPLETED", outer_evaluated=bool(outer_evaluated),
                         new_fits=len(self.registered_new_fits()))
        self.log("Training complete", outer_evaluated=outer_evaluated, new_fits=len(self.registered_new_fits()),
                 budget=f"<= {self.p['budget']['new_fits_max']} fits")

    def verify_training_completed(self):
        record = v25.read_json(self.dir / "training_completed.json")
        for name, digest in record["hashes"].items():
            if v25.file_sha256(self.dir / name) != digest:
                raise ValueError(f"Registered training artifact changed: {name}")
        return record

    def report(self):
        from scripts import report_kelmarsh_backbone_information as reporting
        self.set_status(stage="A4", report="RUNNING")
        try:
            summary = reporting.build_report(self.dir, reference_root=self.reference_root)
        except Exception as error:
            self.set_status(report="REPORT_FAILED", report_error=f"{type(error).__name__}: {error}")
            v25.append_event(self.events, "REPORT_FAILED", error=f"{type(error).__name__}: {error}")
            self.log("A4 report failed", error=f"{type(error).__name__}: {error}",
                     recovery="fix the report code only, then launcher --report-only (no retraining)")
            raise
        self.set_status(stage="A4", report="REPORT_COMPLETED", report_error=None)
        v25.write_json(self.dir / "completed.json", {self.rel(p): v25.file_sha256(p) for p in sorted(self.dir.rglob("*"))
                                                     if registrable(p, self.dir)})
        self.set_status(stage="COMPLETED")
        v25.append_event(self.events, "COMPLETED")
        self.log("A4 complete; v25 stops here", conclusion=summary.get("conclusion"),
                 next_step="cross-year exposure audit and a new information-budget protocol (not automatic)")
        return summary

    def run(self, report_only=False):
        self.dir.mkdir(parents=True, exist_ok=True)
        lock = v25.acquire_exclusive_lock(self.lock_path)
        try:
            if (self.dir / "completed.json").exists() and not report_only:
                self.freeze_protocol_record()  # verifies identity only; writes nothing into a completed run
                print("[REUSE] v25 already completed; nothing retrained", flush=True)
                return "COMPLETED"
            configure_determinism()
            (self.dir / "controller.pid").write_text(str(os.getpid()), encoding="utf-8")
            if not (self.dir / "training_completed.json").exists():
                load_pin(self.p, require_resolved=True)  # the pin is frozen into protocol.json before any fit
            self.freeze_protocol_record()
            self.claim_pointer()
            self.record_sources()
            v25.append_event(self.events, "controller_started", argv=sys.argv, pid=os.getpid(), report_only=report_only)
            if (self.dir / "training_completed.json").exists():
                self.verify_training_completed()
                print("[REUSE] training registered as complete; report stage only", flush=True)
                self.report()
                return "COMPLETED"
            if report_only:
                raise ValueError("--report-only requires a registered TRAIN_COMPLETED run")
            self.stage_a0()
            outer = False
            if self.stage_a1()["passed"]:
                if self.stage_a2()["passed"]:
                    self.stage_a3()
                    outer = True
            self.finish_training(outer)
            self.report()
            return "COMPLETED"
        finally:
            lock.close()


def parameter_changes(p):
    """Old (v20 DLinear reference) -> new values; unchanged recipe keys are recorded as inherited."""
    inherited = {k: "inherited from v20 DLinear recipe" for k in ("learning_rate", "weight_decay", "batch_size",
                 "max_epochs", "patience", "grad_clip_norm", "scheduler", "loss", "selection_metric", "data", "queue",
                 "scaler", "features", "horizon", "stride")}
    return {"reference": "v20 DLinear seed 2024 (configs/kelmarsh_recent_train_protocol_v20.json)",
            "changes": [
                {"parameter": "model_family", "old": "DLinear (models/DLinear.py)", "new": "TSMixerBasicSmall (models/TSMixerBasicSmall.py)"},
                {"parameter": "trainable_parameters", "old": 200, "new": 3618},
                {"parameter": "effective_input_use", "old": "power history only (channel-independent DLinear, loss on power slot)",
                 "new": "TSMixerFull: all 7 standardized channels mixed; TSMixerPowerOnly: channels 0-5 set to 0"},
                {"parameter": "initialization", "old": "PyTorch Linear default", "new": "Xavier uniform / zero bias; LayerNorm 1/0"},
                {"parameter": "seeds", "old": [2024], "new": "A1 [2024]; A2 [2025, 2026] only if G1 passes"},
                {"parameter": "decision_rule", "old": "none", "new": "G1/G2 internal budget gates; I1/I2 information diagnostics"},
                {"parameter": "device_policy", "old": "exclusive GPU wait", "new": "per-pair GPU if >=6 GiB free (4 GiB allocator cap when shared) else CPU 2 threads"}],
            "inherited": inherited, "wiki": "disabled (unchanged)"}


# =========================================================================== preflight
class SyntheticWindows(Dataset):
    """Tiny KelmarshWindows-like dataset for the pre-launch smoke test (never real data)."""

    def __init__(self, n=96, seed=0, offset=0):
        rng = np.random.default_rng(seed)
        self.values = rng.normal(size=(n + 28, 7)).astype(np.float32)
        self.values[:, -1] = np.sin(np.arange(n + 28) / 5.).astype(np.float32) + .1 * self.values[:, 0]
        self.window_starts = np.arange(n, dtype=np.int64)
        self.scaler = SimpleNamespace(mean_=np.zeros(7), scale_=np.ones(7) * 100.)
        self.turbines = np.ones(n + 28, dtype=np.int16)
        self.dates = pd.date_range("2030-01-01", periods=n + 28, freq="h").to_numpy() + np.timedelta64(offset, "h")
        self.series = SimpleNamespace(scaler=self.scaler)

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, i):
        s = self.window_starts[i]
        return (self.values[s:s + 24], self.values[s + 24:s + 28], np.zeros((24, 1), np.float32), np.zeros((4, 1), np.float32))

    def truth_and_anchor(self):
        rows = self.window_starts[:, None] + 24 + np.arange(4)
        truth = self.values[rows, -1].astype(float) * 100.
        last = self.values[self.window_starts + 23, -1].astype(float) * 100.
        return truth, np.repeat(last[:, None], 4, axis=1)

    def target_times(self):
        return self.dates[self.window_starts[:, None] + 24 + np.arange(4)]


def smoke(p, folder):
    """Synthetic smoke: both TSMixer arms, identical init, interrupt+resume equals uninterrupted (CPU)."""
    q = {**p, "training": {**p["training"], "max_epochs": 3, "scheduler_t_max": 3}}
    fit, selection, cpu = SyntheticWindows(96, 1), SyntheticWindows(40, 2), torch.device("cpu")
    results = {}
    for arm in v25.TSMIXER_ARMS:
        contract = {"smoke": True, "arm": arm}
        init = folder / "full" / v25.PRIMARY / "init.pt" if arm == v25.INFO_CONTROL else None
        results[arm] = fit_arm(arm, q, fit=fit, selection=selection, contract=contract, seed=7, device=cpu,
                               out=folder / "full" / arm, init_reference=init)
    contract = {"smoke": True, "arm": v25.PRIMARY}
    try:
        fit_arm(v25.PRIMARY, q, fit=fit, selection=selection, contract=contract, seed=7, device=cpu,
                out=folder / "interrupted" / v25.PRIMARY, stop_inside_epoch=(2, 1))
    except SimulatedInterrupt:
        pass
    resumed = fit_arm(v25.PRIMARY, q, fit=fit, selection=selection, contract=contract, seed=7, device=cpu,
                      out=folder / "interrupted" / v25.PRIMARY)
    a = torch.load(folder / "full" / v25.PRIMARY / "last.pt", map_location="cpu", weights_only=False)
    b = torch.load(folder / "interrupted" / v25.PRIMARY / "last.pt", map_location="cpu", weights_only=False)
    same = state_digest(a["model"]) == state_digest(b["model"]) and resumed["best_epoch"] == results[v25.PRIMARY]["best_epoch"]
    return {"passed": bool(same and results[v25.PRIMARY]["init_state_sha256"] == results[v25.INFO_CONTROL]["init_state_sha256"]),
            "arms": {k: {"best_epoch": v["best_epoch"], "selection_mae": v["selection_best"]["mae"]} for k, v in results.items()},
            "resume_matches_uninterrupted": bool(same), "data": "synthetic tensors only; not a real-data candidate"}


def preflight(gpu_index=0, skip_tests=False):
    p, protocol_sha = load_protocol()
    pin, resolved = load_pin(p, require_resolved=False)
    digest = full_source_digest(protocol_sha, pin)
    folder = PREFLIGHT_DIR / f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}_{digest[:12]}"
    folder.mkdir(parents=True, exist_ok=False)
    configure_determinism()
    checks = {"pin_resolved": resolved, "upstream_pin": pin}
    others = sorted({str(path.relative_to(ROOT)) for path in (ROOT / "configs").glob("*v25*")} - {
        "configs/kelmarsh_backbone_information_protocol_v25.json", "configs/tsmixer_upstream_pin_v25.json"})
    pointers = {str(path.relative_to(ROOT)): path.read_text(errors="replace").strip()
                for path in LOG_ROOT.glob("*latest.txt") if "v25" in path.read_text(errors="replace")
                and path.resolve() != POINTER.resolve()}
    checks["v25_id_free"] = {"other_v25_configs": others, "other_pointers_mentioning_v25": pointers,
                             "own_pointer": POINTER.read_text().strip() if POINTER.exists() else None,
                             "passed": not others and not pointers}
    if skip_tests:
        checks["tests"] = {"passed": False, "skipped": True}
    else:
        try:
            import pytest  # noqa: F401
            command = [sys.executable, "-m", "pytest", "-q", *V25_TESTS, *REGRESSION_TESTS]
        except ImportError:
            command = [sys.executable, "-m", "unittest", *[t.replace("/", ".")[:-3] for t in V25_TESTS + REGRESSION_TESTS]]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (folder / "tests.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        checks["tests"] = {"command": command, "exit_code": result.returncode, "passed": result.returncode == 0,
                           "tail": (result.stdout + result.stderr).splitlines()[-15:]}
    checks["bash_syntax"] = {"exit_code": subprocess.run(["bash", "-n", "scripts/train/Kelmarsh_launch_backbone_information.sh"],
                                                          cwd=ROOT).returncode}
    checks["bash_syntax"]["passed"] = checks["bash_syntax"]["exit_code"] == 0
    try:
        a0 = V25Run(folder / "a0", protocol=p, protocol_sha=protocol_sha, pin=pin, require_preflight=False,
                    write_pointer=False, lock_path=PREFLIGHT_DIR / "preflight.lock")
        a0.dir.mkdir(parents=True)
        a0.freeze_protocol_record()
        a0.record_sources()
        result = a0.stage_a0()
        checks["A0"] = {"passed": True, "persistence_inner": result["persistence_inner"],
                        "dlinear_seed2024_inner": result["dlinear_seed2024_inner"],
                        "windows": {f: [d["fit_windows"], d["selection_windows"]] for f, d in result["data"].items()},
                        "subsets_requested": sorted({r[1] for r in a0.access.requests})}
    except Exception as error:
        checks["A0"] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    try:
        checks["synthetic_smoke"] = smoke(p, folder / "smoke")
    except Exception as error:
        checks["synthetic_smoke"] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    checks["resources"] = {"gpu": gpu_snapshot(gpu_index), "loadavg": os.getloadavg(), "cpu_count": os.cpu_count(),
                           "decision_if_pair_started_now": v25.choose_pair_device(gpu_snapshot(gpu_index), p["operations"])}
    passed = all(checks[k]["passed"] for k in ("v25_id_free", "tests", "bash_syntax", "A0", "synthetic_smoke")) and resolved
    record = {"passed": bool(passed), "full_source_digest": digest, "protocol_sha256": protocol_sha,
              "training_code_sha256": training_code_sha256(), "git": git_info(), "utc": now(), "folder": str(folder),
              "checks": checks}
    v25.write_json(folder / "preflight.json", record)
    if passed:
        v25.write_json(PREFLIGHT_DIR / f"passed_{digest[:16]}.json", record)
    print(f"[PREFLIGHT] passed={passed} pin_resolved={resolved} tests={checks['tests'].get('passed')} "
          f"A0={checks['A0']['passed']} smoke={checks['synthetic_smoke']['passed']} record={folder}", flush=True)
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
    pid_file = directory / "controller.pid"
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
        print(f"CONTROLLER_PID={pid} ALIVE={'yes' if alive else 'no'}")
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
    run = V25Run(args.output.resolve(), gpu_index=args.gpu)
    try:
        run.run(report_only=args.report_only)
        return 0
    except BlockingIOError:
        print("[BUSY] Another v25 controller holds the lock; nothing changed", flush=True)
        return 3
    except Exception as error:
        if args.output.exists():
            status = run.dir / "status.json"
            current = v25.read_json(status) if status.exists() else {}
            if current.get("train") != "TRAIN_COMPLETED":
                run.set_status(train="FAILED", error=f"{type(error).__name__}: {error}")
            v25.append_event(run.events, "controller_failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    sys.exit(main())
