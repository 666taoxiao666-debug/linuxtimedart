#!/usr/bin/env python3
"""v26: frozen v20 DLinear forecast + zero-initialised TSMixerBasicSmall residual, Kelmarsh 24h->4h (Wiki off).

Single mechanism versus v25: the residual branch (the unchanged 3,618-parameter TSMixerBasicSmall with its head
zeroed) learns only an increment on top of the frozen same-fold v20 DLinear forecast. AnchoredFull vs
AnchoredPowerOnly (identical initial state incl. the anchor) attributes any increment to the six auxiliary channels.
Stages:
  A0 contracts + read-only replay of the three frozen anchors + CPU epoch-0 identity of both arms (no fit)
  A1 seed 2024: f0 Full, f0 PowerOnly, f1 ..., f2 ...                (6 fits)  -> G1 / I1 on internal selection
  A2 only if G1: residual seeds 2025, 2026 x folds x {Full, PowerOnly} (<= 12) -> G2 / I2 on internal selection
  A3 only if G2: one outer-validation evaluation of every fold/seed/arm, nothing re-selected
  A4 independent recomputation, report and stop (scripts/report_kelmarsh_anchored_residual.py)
Epoch 0 is recorded (it must equal DLinear) but is not selectable. The anchor is never trained. Gates are budget
rules on development evidence, not significance tests. Never touches the sealed test. Never resumes or edits v25.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
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
from models.DLinearAnchoredResidual import Model as AnchoredModel
from utils import kelmarsh_anchored_residual as v26
from utils.metrics import forecast_metrics

PROTOCOL = ROOT / "configs/kelmarsh_anchored_residual_protocol_v26.json"
PROTOCOL_SHA256 = "e11f80a64ae50f3782c99ec16b7eb591099714866f7842d6b226677aeb698eb9"
LOG_ROOT = ROOT / "outputs/logs/Kelmarsh"
POINTER = LOG_ROOT / "anchored_residual_latest.txt"
LOCK = LOG_ROOT / "anchored_residual.lock"
PREFLIGHT_DIR = LOG_ROOT / "anchored_residual_preflight"
# Files whose code determines the trained numbers (data, models, metric used for epoch selection).
TRAINING_FILES = ("models/DLinearAnchoredResidual.py", "models/TSMixerBasicSmall.py", "models/DLinear.py",
                  "models/__init__.py", "data_provider/kelmarsh_hourly.py", "data_provider/__init__.py",
                  "utils/metrics.py")
# All v26 code; any change requires a new preflight before further fits (but never invalidates finished fits).
V26_FILES = TRAINING_FILES + ("scripts/train_kelmarsh_anchored_residual.py", "utils/kelmarsh_anchored_residual.py",
                              "utils/kelmarsh_backbone_information.py", "utils/paired_forecast_statistics.py",
                              "scripts/report_kelmarsh_anchored_residual.py",
                              "scripts/train/Kelmarsh_launch_anchored_residual.sh",
                              "tests/test_dlinear_anchored_residual.py", "tests/test_kelmarsh_anchored_residual.py")
V26_TESTS = ("tests/test_dlinear_anchored_residual.py", "tests/test_kelmarsh_anchored_residual.py")
REGRESSION_TESTS = ("tests/test_tsmixer_basic_small.py", "tests/test_kelmarsh_backbone_information.py",
                    "tests/test_paired_gain_estimand.py", "tests/test_strong_baselines_and_statistics.py")
VOLATILE = {"status.json", "status.env", "progress.json", "launch.log", "launcher.pid", "controller.pid",
            "events.jsonl", "experiment_log.md", "fit_progress.json"}
ANCHOR_REPLAY_DIR = "anchor_replay"


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


@torch.no_grad()
def predict_standardized(model, dataset, device, batch=128):
    """Standardized power forecast (for the per-sample epoch-0 identity check in both spaces)."""
    model.eval()
    chunks = []
    for x, _, _, _ in make_loader(dataset, 0, batch=batch, device=device):
        chunks.append(model(x.float().to(device))[..., -1].cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def build_model(arm, p, anchor_state=None):
    """Construct an arm. Candidate arms load ``anchor_state`` (the frozen DLinear state_dict) when given."""
    spec = p["arms"][arm]
    if spec["family"] == "DLinearAnchoredResidual":
        c, r = p["candidate"], p["candidate"]["residual"]
        kwargs = {"seq_len": p["input_len"], "channels": len(p["feature_columns"]), "pred_len": p["pred_len"],
                  "blocks": r["blocks"], "ff_dim": r["feature_hidden_dim"], "dropout": r["dropout"],
                  "norm_eps": r["norm_eps"], "target_slot": p["target_slot"], "input_mode": spec["input_mode"],
                  "anchor_args": dict(c["anchor"]["model_args"])}
        model, factory = AnchoredModel(**kwargs), "models.DLinearAnchoredResidual.Model"
        if anchor_state is not None:
            model.load_anchor_state(anchor_state)
        expected, frozen_expected = c["expected_trainable_parameters"], c["expected_frozen_parameters"]
    elif spec["family"] == "DLinear":
        kwargs = dict(spec["model_args"])
        model, factory = DLinearModel(SimpleNamespace(**kwargs)), "models.DLinear.Model"
        if anchor_state is not None:
            model.load_state_dict(anchor_state, strict=True)
        # Replay the actual frozen executor. requires_grad can affect Linear's
        # backend dispatch even under no_grad, so eval() alone is insufficient
        # for the exact epoch-0 comparison with the embedded frozen anchor.
        model.requires_grad_(False)
        model.eval()
        expected, frozen_expected = 0, spec["expected_parameters"]
    else:
        raise ValueError("Undeclared model family")
    count = int(sum(q.numel() for q in model.parameters() if q.requires_grad))
    frozen = int(sum(q.numel() for q in model.parameters() if not q.requires_grad))
    if count != int(expected) or frozen != int(frozen_expected):
        raise ValueError(f"{arm} has {count} trainable / {frozen} frozen parameters, protocol requires "
                         f"{expected} / {frozen_expected}; never enlarge to 'fix'")
    return model, {"factory": factory, "kwargs": kwargs, "trainable_parameters": count, "frozen_parameters": frozen}


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


def anchor_digest(state):
    """Digest of the anchor sub-state only (``anchor.*`` keys, prefix stripped) = digest of the DLinear state."""
    sub = {key[len("anchor."):]: value for key, value in state.items() if key.startswith("anchor.")}
    if not sub:
        raise ValueError("State has no anchor tensors")
    return state_digest(sub)


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
    trained = v26.read_json(out / "trained.json")
    mismatch = v26.contract_mismatches(trained["contract"], contract)
    if mismatch:
        raise ValueError(f"Registered fit {out} has a different contract ({mismatch}); never mix or overwrite")
    for name, digest in trained["artifacts"].items():
        if v26.file_sha256(out / name) != digest:
            raise ValueError(f"Registered artifact changed: {out / name}")
    return trained


def fit_arm(arm, p, *, fit, selection, contract, seed, device, out, anchor_state, anchor_reference_kw,
            events=None, init_reference=None, stop_after_epoch=None, stop_inside_epoch=None):
    """Train one residual arm with the frozen recipe on top of the frozen anchor; epoch-boundary resumable.

    Order as v25 fit_arm: seed_all -> construct (anchor loaded, frozen) -> epoch-0 assessment (recorded, must equal the
    anchor per sample, NOT selectable) -> epochs with loader(seed+epoch) shuffling, (1-w)|e|+w e^2 on the standardized
    final forecast, clip and AdamW over trainable parameters only, strict-MAE best over epochs >= 1, patience checked at
    the top of each epoch (counted from epoch 1), cosine step after each epoch.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "trained.json").exists():
        return verify_trained(out, contract) | {"reused_from_registry": True}
    cfg = p["training"]
    weight = float(cfg["loss_mse_weight"])
    if abs((1 - weight) - float(cfg["loss_mae_weight"])) > 1e-12:
        raise ValueError("Loss weights must be (1-w, w)")
    if cfg["epoch0_selectable"] or int(cfg["first_selectable_epoch"]) != 1:
        raise ValueError("v26 selects only trained epochs (>= 1)")
    device = torch.device(device)
    seed_all(seed)
    model, factory = build_model(arm, p, anchor_state=anchor_state)
    init_state = cpu_state(model)
    init_digest = state_digest(init_state)
    anchor_sha = anchor_digest(init_state)
    if anchor_sha != state_digest({k: v.detach().cpu() for k, v in anchor_state.items()}):
        raise ValueError("Loaded anchor differs from the supplied frozen state")
    if init_reference is not None:
        reference = torch.load(init_reference, map_location="cpu", weights_only=False)
        if reference["state_sha256"] != init_digest or state_digest(reference["model"]) != init_digest:
            raise ValueError("Paired arms do not start from the identical initial state")
    init_path = out / "init.pt"
    if init_path.exists():
        saved = torch.load(init_path, map_location="cpu", weights_only=False)
        if saved["state_sha256"] != init_digest or v26.contract_mismatches(saved["contract"], contract):
            raise ValueError("Saved initial state differs from the deterministic construction")
    # init.pt is written only after the epoch-0 identity check passed (fresh start below), so a refused start
    # never leaves a stale init.pt that would block a corrected retry.
    model.to(device)
    trainable = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["scheduler_t_max"],
                                                           eta_min=cfg["scheduler_eta_min"])
    truth, persistence = selection.truth_and_anchor()
    rated, power_scale = p["rated_power_kw"], float(fit.scaler.scale_[-1])
    tolerance = float(p["candidate"]["epoch0_identity_tolerance_kw"])

    def assess():
        begin = time.monotonic()
        prediction = predict(model, selection, seed, device, cfg["eval_batch_size"])
        metrics = forecast_metrics(prediction, truth, rated_power=rated)
        return metrics["mae"], metrics, time.monotonic() - begin, prediction

    def save_last(epoch):
        # last.pt also carries the selected state, so it is the single atomic source of truth on resume: a crash
        # between writing best.pt and last.pt can never leave best.pt pointing at an unregistered epoch.
        atomic_save({"model": cpu_state(model), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "contract": contract, "epoch": epoch, "best": best, "best_epoch": best_epoch, "bad_epochs": bad,
                     "best_model": best_state, "history": history, "timing": timing, "rng": capture_rng(device),
                     "next_shuffle_seed": int(seed) + epoch + 1}, last_path)
        v26.write_json(out / "fit_progress.json", {"last_completed_epoch": epoch, "epoch_in_progress": None})

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    last_path, best_path = out / "last.pt", out / "best.pt"
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        mismatch = v26.contract_mismatches(checkpoint["contract"], contract)
        if mismatch:
            raise ValueError(f"Interrupted fit identity changed: {mismatch}")
        if anchor_digest(checkpoint["model"]) != anchor_sha:
            raise ValueError("Interrupted fit carries a changed anchor; refuse")
        if not init_path.exists():
            raise ValueError("init.pt missing for an interrupted fit")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history, start = checkpoint["history"], int(checkpoint["epoch"])
        best, best_epoch, bad, timing = checkpoint["best"], checkpoint["best_epoch"], checkpoint["bad_epochs"], checkpoint["timing"]
        best_state = checkpoint["best_model"]
        restore_rng(checkpoint["rng"], device)
        if best_epoch is not None:
            if best_state is None or anchor_digest(best_state) != anchor_sha:
                raise ValueError("last.pt carries no valid selected state for an interrupted fit")
            atomic_save({"model": best_state, "epoch": best_epoch, "contract": contract, "selection_mae_kw": best}, best_path)
        marker = v26.read_json(out / "fit_progress.json") if (out / "fit_progress.json").exists() else {}
        if events is not None:
            v26.append_event(events, "fit_resumed", arm=arm, seed=seed, out=str(out), from_epoch=start,
                             discarded_partial_epoch=marker.get("epoch_in_progress"))
    else:
        score, metrics, seconds, prediction = assess()
        identity = float(np.max(np.abs(prediction - np.asarray(anchor_reference_kw, dtype=np.float64))))
        if not identity <= tolerance:
            raise ValueError(f"Epoch-0 forecast differs from the frozen anchor by {identity} kW > {tolerance}")
        if not init_path.exists():
            atomic_save({"model": init_state, "state_sha256": init_digest, "anchor_state_sha256": anchor_sha, "arm": arm,
                         "seed": seed, "contract": contract}, init_path)
        history = [{"epoch": 0, "selection": metrics, "assess_seconds": seconds, "lr": optimizer.param_groups[0]["lr"],
                    "selectable": False, "epoch0_max_abs_diff_vs_anchor_kw": identity}]
        best, best_epoch, bad, start, best_state = float("inf"), None, 0, 0, None
        timing = {"train_seconds": 0.0, "assess_seconds": seconds}
        save_last(0)
    for epoch in range(start + 1, int(cfg["max_epochs"]) + 1):
        if bad >= int(cfg["patience"]):
            break
        if stop_after_epoch is not None and epoch > stop_after_epoch:
            raise SimulatedInterrupt(f"after epoch {epoch - 1}")
        v26.write_json(out / "fit_progress.json", {"last_completed_epoch": epoch - 1, "epoch_in_progress": epoch})
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
            torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip_norm"])
            optimizer.step()
            total += loss.item()
            batches += 1
            abs_kw += difference.detach().abs().sum().item() * power_scale
            points += difference.numel()
        train_seconds = time.monotonic() - begin
        score, metrics, assess_seconds, _ = assess()
        timing["train_seconds"] += train_seconds
        timing["assess_seconds"] += assess_seconds
        history.append({"epoch": epoch, "train_loss": total / batches, "train_loss_space": "standardized (1-w)|e|+w e^2",
                        "train_mae_kw_running_with_dropout": abs_kw / points, "lr": lr, "selection": metrics,
                        "selectable": True, "train_seconds": train_seconds, "assess_seconds": assess_seconds})
        if score < best:
            best, best_epoch, bad, best_state = score, epoch, 0, cpu_state(model)
            atomic_save({"model": best_state, "epoch": epoch, "contract": contract, "selection_mae_kw": best}, best_path)
        else:
            bad += 1
        scheduler.step()
        save_last(epoch)
        v26.write_json(out / "training_history.json", history)
        if events is not None:
            v26.append_event(events, "epoch", arm=arm, seed=seed, out=str(out.relative_to(out.parents[2])),
                             epoch=epoch, train_loss=total / batches, selection_mae_kw=score, best_epoch=best_epoch)
        print(f"[EPOCH] {out.parent.name}/{arm} epoch={epoch} train_loss={total / batches:.6f} "
              f"selection_mae={score:.6f} best_epoch={best_epoch}", flush=True)
    v26.write_json(out / "training_history.json", history)
    if best_epoch is None:
        raise ValueError("No trained epoch was completed; refuse to register a fit")
    rule_epoch, _ = v26.select_epoch(history, cfg["first_selectable_epoch"])
    if rule_epoch != best_epoch:
        raise ValueError("Online best epoch disagrees with the registered selection rule")
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    last_checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    if int(best_checkpoint["epoch"]) != int(best_epoch) or int(last_checkpoint["best_epoch"]) != int(best_epoch):
        raise ValueError("best.pt / last.pt do not register the selected epoch")
    for name, state in (("best", best_checkpoint["model"]), ("last", last_checkpoint["model"])):
        if anchor_digest(state) != anchor_sha:
            raise ValueError(f"The frozen anchor changed during training ({name}.pt); refuse")
    model.load_state_dict(best_checkpoint["model"], strict=True)
    prediction = predict(model, selection, seed, device, cfg["eval_batch_size"])
    replay = forecast_metrics(prediction, truth, rated_power=rated)
    recorded = history[best_epoch]["selection"]
    if history[best_epoch]["epoch"] != best_epoch or abs(replay["mae"] - recorded["mae"]) > 1e-6:
        raise ValueError("best.pt does not reproduce its registered selection MAE")
    leads = v26.lead_summary(prediction, truth, persistence, rated)
    fit_truth, fit_persistence = fit.truth_and_anchor()
    fit_prediction = predict(model, fit, seed, device, cfg["eval_batch_size"])
    fit_overall = v26.point_metrics(fit_prediction, fit_truth, fit_persistence, rated)
    fit_leads = v26.lead_summary(fit_prediction, fit_truth, fit_persistence, rated)
    np.savez_compressed(out / "selection_predictions.npz", prediction=prediction, truth=truth, persistence=persistence,
                        anchor=np.asarray(anchor_reference_kw, dtype=np.float64),
                        times=selection.target_times(), turbines=selection.turbines[selection.window_starts],
                        window_starts=selection.window_starts)
    if device.type == "cuda":
        memory = {"kind": "cuda_max_memory_allocated_bytes", "value": int(torch.cuda.max_memory_allocated(device))}
    else:
        memory = {"kind": "process_max_rss_kib_cumulative", "value": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    trained = {"contract": contract, "arm": arm, "seed": int(seed), "best_epoch": int(best_epoch),
               "epochs_completed": len(history) - 1, "stopped_early": len(history) - 1 < int(cfg["max_epochs"]),
               "selection_best": recorded, "selection_best_by_lead": leads,
               "selection_epoch0": history[0]["selection"], "selection_last": history[-1]["selection"],
               "epoch0_max_abs_diff_vs_anchor_kw": history[0]["epoch0_max_abs_diff_vs_anchor_kw"],
               "selection_replay_abs_mae_diff_kw": abs(replay["mae"] - recorded["mae"]),
               "fit_eval_best": {"overall": fit_overall, "by_lead": fit_leads,
                                 "note": "eval mode, fit windows, informational only; never used for any choice"},
               "trainable_parameters": factory["trainable_parameters"], "frozen_parameters": factory["frozen_parameters"],
               "init_state_sha256": init_digest, "anchor_state_sha256": anchor_sha,
               "best_state_sha256": state_digest(best_checkpoint["model"]), "timing": timing, "peak_memory": memory,
               "device": str(device), "reused": False, "completed_utc": now(),
               "artifacts": {name: v26.file_sha256(out / name) for name in
                             ("init.pt", "best.pt", "last.pt", "training_history.json", "selection_predictions.npz")}}
    v26.write_json(out / "trained.json", trained)
    return trained


TRAINING_FUNCTIONS = (seed_all, configure_determinism, determinism_record, make_loader, predict, predict_standardized,
                      build_model, cpu_state, state_digest, anchor_digest, capture_rng, restore_rng, verify_trained, fit_arm)


def training_code_sha256():
    """Identity of code that determines trained numbers; orchestration/report edits do not change it."""
    return v26.canonical_sha256({"files": {name: v26.text_sha256_lf(ROOT / name) for name in TRAINING_FILES},
                                 "functions": {f.__name__: inspect.getsource(f) for f in TRAINING_FUNCTIONS},
                                 "selection_rule": inspect.getsource(v26.select_epoch),
                                 "torch_major_minor": ".".join(torch.__version__.split(".")[:2])})


def gate_code_sha256():
    """Identity of the gate/statistics module; a change never silently flips a registered decision."""
    return v26.text_sha256_lf(Path(v26.__file__))


# =========================================================================== identity helpers
def load_protocol(path=PROTOCOL, expected=PROTOCOL_SHA256):
    digest = v26.text_sha256_lf(path)
    if expected is not None and digest != expected:
        raise ValueError("Frozen v26 protocol bytes changed; a new protocol needs a new id")
    protocol = v26.read_json(path)
    v26.validate_protocol(protocol)
    return protocol, digest


def load_pin(p, require_resolved=True):
    pin = v26.read_json(ROOT / p["upstream_pin_file"])
    resolved = (bool(re.fullmatch(r"[0-9a-f]{40}", pin.get("commit") or ""))
                and bool(re.fullmatch(r"[0-9a-f]{64}", pin.get("file_sha256") or ""))
                and pin.get("license") == "Apache-2.0")
    if require_resolved and not resolved:
        raise ValueError("Upstream TSMixer commit/file SHA-256 not pinned (configs/tsmixer_upstream_pin_v25.json)")
    return pin, resolved


def git_info():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as error:  # informational only
            return f"unavailable: {type(error).__name__}"
    return {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "tracked_changes": git("status", "--porcelain", "--untracked-files=no").splitlines()}


def v26_sources():
    return {name: (v26.file_sha256(ROOT / name) if (ROOT / name).exists() else None) for name in V26_FILES}


def full_source_digest(protocol_sha, pin):
    return v26.canonical_sha256({"protocol_sha256": protocol_sha, "pin": pin,
                                 "sources": {k: v26.text_sha256_lf(ROOT / k) for k in V26_FILES if (ROOT / k).exists()}})


def imported_closure():
    closure = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename)
        # torch.ops/torch.classes expose synthetic relative __file__ names; skip them (v25 fix 54a8d3a).
        if not path.is_absolute() or not path.is_file():
            continue
        try:
            relative = path.resolve().relative_to(ROOT)
        except ValueError:
            continue
        closure[str(relative).replace("\\", "/")] = v26.file_sha256(ROOT / relative)
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


def preflight_device(p, gpu_index):
    """Apply the same own-process resource policy before any preflight CUDA probe."""
    decision = v26.choose_pair_device(gpu_snapshot(gpu_index), p["operations"])
    if decision["device"] == "cuda":
        device = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_per_process_memory_fraction(decision["memory_fraction"], device)
        return device, decision
    return torch.device("cpu"), decision


def verify_completed_suite(directory):
    """Every registered completed.json (suite and nested) must still hash-match."""
    directory = Path(directory)
    markers = [directory / "completed.json"] + sorted((directory / "runs").glob("f*_s*/*/completed.json"))
    for marker in markers:
        for name, digest in v26.read_json(marker).items():
            if v26.file_sha256(marker.parent / name) != digest:
                raise ValueError(f"Registered reference artifact changed: {marker.parent / name}")
    return len(markers)


# =========================================================================== controller
class V26Run:
    def __init__(self, directory, *, protocol=None, protocol_sha=None, pin=None, series_factory=None, gpu_index=0,
                 require_preflight=True, test_anchors=None, write_pointer=True, lock_path=LOCK, reference_root=ROOT):
        """``test_anchors`` (tests only, never from the CLI): {fold: {"state": DLinear state_dict, "mae": .., "rmse": ..}}
        replaces the hash-verified v20 checkpoints with synthetic anchors."""
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
        self.test_anchors, self.write_pointer, self.lock_path = test_anchors, write_pointer, lock_path
        self.reference_root = Path(reference_root)
        self.events = self.dir / "events.jsonl"
        self.access = v26.DataAccessLog(self.events)
        self.prepared, self.fit_calls, self.anchors = {}, [], {}
        self.tol = protocol["gates"]["numeric_tie_tolerance_kw"]

    # ------------------------------------------------------------------ bookkeeping
    def rel(self, path):
        return str(Path(path).relative_to(self.dir)).replace("\\", "/")

    @staticmethod
    def rel_root(path):
        """Repository-relative path when possible (portable manifests), else absolute."""
        try:
            return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            return str(Path(path).resolve())

    def rel_reference(self, path):
        """Path relative to reference_root (the report resolves 'base': 'reference' entries against it), else absolute."""
        try:
            return str(Path(path).resolve().relative_to(self.reference_root.resolve())).replace("\\", "/")
        except ValueError:
            return str(Path(path).resolve())

    def set_status(self, **fields):
        path = self.dir / "status.json"
        status = v26.read_json(path) if path.exists() else {"train": "NOT_STARTED", "report": "PENDING", "stage": None}
        status.update(fields, updated_utc=now())
        v26.write_json(path, status)
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
                  "upstream_pin": self.pin, "upstream_pin_sha256": v26.canonical_sha256(self.pin)}
        path = self.dir / "protocol.json"
        if path.exists():
            saved = v26.read_json(path)
            if v26.canonical_sha256(saved) != v26.canonical_sha256(record):
                raise ValueError("Frozen protocol/pin changed on resume; start a new protocol id instead")
        else:
            v26.write_json(path, record)
        self.pin_sha = record["upstream_pin_sha256"]

    def claim_pointer(self):
        if not self.write_pointer:
            return
        POINTER.parent.mkdir(parents=True, exist_ok=True)
        if POINTER.exists() and POINTER.read_text().strip() and Path(POINTER.read_text().strip()).resolve() != self.dir.resolve():
            raise ValueError(f"Pointer already registers another v26 run: {POINTER.read_text().strip()}; use --resume")
        POINTER.write_text(str(self.dir.resolve()), encoding="utf-8")

    def record_sources(self):
        manifest = {"utc": now(), "training_code_sha256": training_code_sha256(), "training_files": list(TRAINING_FILES),
                    "training_functions": [f.__name__ for f in TRAINING_FUNCTIONS], "gate_code_sha256": gate_code_sha256(),
                    "v26_sources": v26_sources(), "full_source_digest": full_source_digest(self.protocol_sha, self.pin),
                    "imported_closure_under_root": imported_closure(), "git": git_info()}
        path = self.dir / "source_manifest.json"
        if not path.exists():
            v26.write_json(path, manifest)
        elif v26.read_json(path)["full_source_digest"] != manifest["full_source_digest"]:
            v26.append_event(self.events, "source_changed_on_resume", previous=v26.read_json(path)["full_source_digest"],
                             current=manifest["full_source_digest"],
                             training_code_changed=v26.read_json(path)["training_code_sha256"] != manifest["training_code_sha256"],
                             rule="finished fits are reused only if their per-fit contract still matches")
            v26.write_json(self.dir / f"source_manifest_resume_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}.json", manifest)
        if not (self.dir / "environment.json").exists():
            v26.write_json(self.dir / "environment.json", environment())
        else:
            v26.append_event(self.events, "environment_on_resume", environment=environment())
        if not (self.dir / "parameter_changes.json").exists():
            v26.write_json(self.dir / "parameter_changes.json", parameter_changes(self.p))
        self.training_code = manifest["training_code_sha256"]
        self.source_digest = manifest["full_source_digest"]
        return manifest

    def check_preflight(self):
        if not self.require_preflight:
            return
        record = PREFLIGHT_DIR / f"passed_{self.source_digest[:16]}.json"
        if not record.exists() or v26.read_json(record).get("full_source_digest") != self.source_digest \
                or not v26.read_json(record).get("passed"):
            raise ValueError("No passed --preflight-only record for the current code/protocol; run it before any fit")

    # ------------------------------------------------------------------ data
    def prepare_fold(self, fold, stage):
        self.access.request(stage, "fit", fold)
        self.access.request(stage, "selection", fold)
        if fold not in self.prepared:
            series = self.series_factory(fold)
            self.prepared[fold] = (series, v26.inner_datasets(series))
        return self.prepared[fold]

    def fold_manifest(self, fold, series, data):
        selection = data["selection"]
        truth, persistence = selection.truth_and_anchor()
        return {"fold": fold, "bounds_utc": {k: str(v) for k, v in series.bounds.items()},
                "fit_windows": len(data["fit"]), "selection_windows": len(selection),
                "fit_window_hash": v26.window_hash(data["fit"].window_starts),
                "selection_window_hash": v26.window_hash(selection.window_starts),
                "fit_target_range_utc": [str(data["fit"].target_times().min()), str(data["fit"].target_times().max())],
                "selection_target_range_utc": [str(selection.target_times().min()), str(selection.target_times().max())],
                "scaler_mean": series.scaler.mean_.tolist(), "scaler_scale": series.scaler.scale_.tolist(),
                "scaler_sha256": v26.canonical_sha256([series.scaler.mean_.tolist(), series.scaler.scale_.tolist()]),
                "selection_truth_sha256": v26.array_sha256(truth), "selection_persistence_sha256": v26.array_sha256(persistence),
                "selection_times_sha256": v26.array_sha256(selection.target_times()),
                "selection_turbines_sha256": v26.array_sha256(selection.turbines[selection.window_starts]),
                "selection_issue_coverage": v26.issue_time_coverage(selection.target_times()[:, 0] - np.timedelta64(1, "h"),
                                                                    self.p["eval_stride"]),
                "series_audit": series.audit, "outer_validation_instantiated": False}

    # ------------------------------------------------------------------ A0: frozen anchors
    def verify_reference_identity(self):
        ref = self.p["reference"]
        if v26.text_sha256_lf(ROOT / ref["protocol_path"]) != ref["protocol_sha256"]:
            raise ValueError("v20 protocol bytes changed")
        refdir = self.reference_root / ref["dir"]
        if v26.file_sha256(refdir / "completed.json") != ref["completed_sha256"]:
            raise ValueError("v20 completed.json identity changed")
        markers = verify_completed_suite(refdir)
        if v26.file_sha256(ROOT / self.p["data_path"]) != self.p["data_sha256"]:
            raise ValueError("Prepared Kelmarsh source fingerprint changed")
        return {"v20_dir": ref["dir"], "v20_completed_sha256": ref["completed_sha256"], "verified_completed_markers": markers,
                "data_sha256": self.p["data_sha256"]}

    def anchor_run_dir(self, fold):
        ref = self.p["reference"]
        return self.reference_root / ref["dir"] / "runs" / f"f{fold}_s{ref['seed']}" / "DLinear"

    def load_anchor(self, fold, series=None, data=None):
        """Hash-verified frozen anchor state for ``fold`` (cached). Contract checks need the fold's data."""
        if fold in self.anchors:
            return self.anchors[fold]
        if self.test_anchors is not None:
            spec = self.test_anchors[fold]
            entry = {"state": spec["state"], "checkpoint_sha256": None, "state_sha256": state_digest(spec["state"]),
                     "registered": {"mae": spec["mae"], "rmse": spec["rmse"]}, "source": "SYNTHETIC_TEST_ANCHOR"}
            self.anchors[fold] = entry
            return entry
        ref = self.p["reference"]
        spec = next(item for item in ref["dlinear_checkpoints"] if item["fold"] == fold)
        run = self.anchor_run_dir(fold)
        trained = v26.read_json(run / "trained.json")
        contract = trained["contract"]
        errors = []
        checkpoint_sha = v26.file_sha256(run / "checkpoint.pt")
        if checkpoint_sha != spec["checkpoint_sha256"] or trained["artifacts"]["checkpoint.pt"] != spec["checkpoint_sha256"]:
            errors.append("checkpoint sha")
        if trained["best_epoch"] != spec["best_epoch"]:
            errors.append("best epoch")
        if data is not None:
            for key, stage in (("fit_window_hash", "fit"), ("selection_window_hash", "selection")):
                computed = v26.window_hash(data[stage].window_starts)
                if not (contract[key] == spec[key] == computed):
                    errors.append(f"{key}: registered {contract[key][:12]} design {spec[key][:12]} computed {computed[:12]}")
        if series is not None and not (np.array_equal(series.scaler.mean_, contract["scaler_mean"])
                                       and np.array_equal(series.scaler.scale_, contract["scaler_scale"])):
            errors.append("scaler arrays")
        expected = {"data_sha256": self.p["data_sha256"], "protocol_sha256": ref["protocol_sha256"], "name": "DLinear",
                    "seed": ref["seed"], "pretrain": False, "initial_checkpoint_sha256": None}
        errors += [f"contract {k}" for k, v in expected.items() if contract.get(k) != v]
        history = v26.read_json(run / "training_history.json")
        entry = next(r["selection"] for r in history if r["epoch"] == spec["best_epoch"])
        if entry["mae"] != spec["internal_mae_kw"] or entry["rmse"] != spec["internal_rmse_kw"]:
            errors.append("registered internal MAE/RMSE")
        if v26.read_json(run / "manifest.json")["parameters"] != ref["expected_parameters"]:
            errors.append("parameter count")
        if errors:
            raise ValueError(f"v20 DLinear f{fold} anchor identity failed: {errors}; investigate, never bypass hashes")
        state = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)["model"]
        state = {k: v.detach().cpu().clone() for k, v in state.items()}
        result = {"state": state, "checkpoint_sha256": checkpoint_sha, "state_sha256": state_digest(state),
                  "registered": {"mae": entry["mae"], "rmse": entry["rmse"]}, "best_epoch": spec["best_epoch"],
                  "checkpoint": self.rel_root(run / "checkpoint.pt"), "source": "v20 registered checkpoint (hash-verified)"}
        self.anchors[fold] = result
        return result

    def anchor_replay(self, fold, series, data):
        """CPU replay of the frozen anchor on fit and selection; registered once, reproduced exactly on resume."""
        anchor = self.load_anchor(fold, series, data)
        model, _ = build_model(v26.BASELINE, self.p, anchor_state=anchor["state"])
        cpu, batch = torch.device("cpu"), self.p["training"]["eval_batch_size"]
        seed = self.p["seeds"]["anchor_seed"]
        rated = self.p["rated_power_kw"]
        selection = data["selection"]
        truth, persistence = selection.truth_and_anchor()
        prediction = predict(model, selection, seed, cpu, batch)
        replay = forecast_metrics(prediction, truth, rated_power=rated)
        diff = {k: abs(replay[k] - anchor["registered"][k]) for k in ("mae", "rmse")}
        if self.test_anchors is None and max(diff.values()) > self.p["reference"]["replay_tolerance_kw"]:
            raise ValueError(f"Replayed v20 DLinear f{fold} does not reproduce internal metrics: {diff}")
        fit_truth, fit_persistence = data["fit"].truth_and_anchor()
        fit_prediction = predict(model, data["fit"], seed, cpu, batch)
        folder = self.dir / ANCHOR_REPLAY_DIR
        folder.mkdir(parents=True, exist_ok=True)
        arrays = {"prediction": prediction, "truth": truth, "persistence": persistence, "times": selection.target_times(),
                  "turbines": selection.turbines[selection.window_starts], "window_starts": selection.window_starts}
        target = folder / f"f{fold}_s{seed}_DLinear_selection_predictions.npz"
        if target.exists():
            with np.load(target, allow_pickle=False) as saved:
                v26.assert_aligned(arrays, dict(saved), keys=tuple(arrays), label=f"anchor replay f{fold}")
        else:
            np.savez_compressed(target, **arrays)
        leads = v26.lead_summary(prediction, truth, persistence, rated)
        return {"fold": fold, "checkpoint": anchor.get("checkpoint"), "checkpoint_sha256": anchor["checkpoint_sha256"],
                "anchor_state_sha256": anchor["state_sha256"], "best_epoch": anchor.get("best_epoch"),
                "registered_selection": anchor["registered"], "replay_device": "cpu",
                "replay_selection": {"mae": replay["mae"], "rmse": replay["rmse"]}, "replay_abs_diff_kw": diff,
                "replay_selection_by_lead": leads,
                "replay_fit": {"overall": v26.point_metrics(fit_prediction, fit_truth, fit_persistence, rated),
                               "by_lead": v26.lead_summary(fit_prediction, fit_truth, fit_persistence, rated)},
                "selection_predictions": self.rel(target), "selection_predictions_sha256": v26.file_sha256(target),
                "parameters": self.p["reference"]["expected_parameters"], "retrained": False, "source": anchor["source"]}

    def epoch0_identity(self, fold, data, seed):
        """Both candidate arms at construction equal the anchor per sample on CPU (standardized and kW space)."""
        anchor = self.load_anchor(fold)
        reference, _ = build_model(v26.BASELINE, self.p, anchor_state=anchor["state"])
        cpu, batch = torch.device("cpu"), self.p["training"]["eval_batch_size"]
        selection = data["selection"]
        ref_std = predict_standardized(reference, selection, cpu, batch)
        ref_kw = predict(reference, selection, seed, cpu, batch)
        out, digests = {}, {}
        for arm in v26.CANDIDATE_ARMS:
            seed_all(seed)
            model, _ = build_model(arm, self.p, anchor_state=anchor["state"])
            digests[arm] = state_digest(cpu_state(model))
            std = predict_standardized(model, selection, cpu, batch)
            kw = predict(model, selection, seed, cpu, batch)
            out[arm] = {"max_abs_diff_standardized": float(np.max(np.abs(std - ref_std))),
                        "max_abs_diff_kw": float(np.max(np.abs(kw - ref_kw)))}
            if out[arm]["max_abs_diff_standardized"] != 0.0 or out[arm]["max_abs_diff_kw"] != 0.0:
                raise ValueError(f"f{fold} {arm} epoch-0 forecast is not the frozen anchor on CPU: {out[arm]}")
        if digests[v26.PRIMARY] != digests[v26.INFO_CONTROL]:
            raise ValueError("Paired candidate arms are not constructed identically")
        return {"fold": fold, "seed": seed, "device": "cpu", "arms": out, "paired_init_state_sha256": digests[v26.PRIMARY]}

    def device_identity_probe(self, fold, device):
        """Preflight only: the anchored model at construction on ``device`` vs the registered CPU anchor replay.
        Measures the per-sample CPU/GPU float32 gap before the first real fit relies on epoch0_identity_tolerance_kw."""
        _, data = self.prepare_fold(fold, "A0")
        seed = self.p["seeds"]["A1"][0]
        seed_all(seed)
        model, _ = build_model(v26.PRIMARY, self.p, anchor_state=self.load_anchor(fold)["state"])
        model.to(device)
        prediction = predict(model, data["selection"], seed, device, self.p["training"]["eval_batch_size"])
        difference = float(np.max(np.abs(prediction - self.anchor_reference_kw(fold))))
        tolerance = float(self.p["candidate"]["epoch0_identity_tolerance_kw"])
        return {"fold": fold, "device": str(device), "max_abs_diff_kw": difference, "tolerance_kw": tolerance,
                "passed": bool(difference <= tolerance)}

    def stage_a0(self):
        self.set_status(stage="A0", train="RUNNING")
        identity = (self.verify_reference_identity() if self.test_anchors is None
                    else {"verification": "SKIPPED_TEST_OVERRIDE_NEVER_FROM_CLI"})
        folds, reference, persistence, epoch0 = {}, {}, {}, {}
        for fold in self.folds:
            series, data = self.prepare_fold(fold, "A0")
            folds[str(fold)] = self.fold_manifest(fold, series, data)
            truth, anchor = data["selection"].truth_and_anchor()
            pm = forecast_metrics(anchor, truth, rated_power=self.p["rated_power_kw"])
            persistence[str(fold)] = {"mae": pm["mae"], "rmse": pm["rmse"], "mse": pm["mse"], "r2": pm["r2"]}
            reference[str(fold)] = self.anchor_replay(fold, series, data)
            epoch0[str(fold)] = self.epoch0_identity(fold, data, self.p["seeds"]["A1"][0])
        dlinear_inner = {}
        for f, r in reference.items():
            lead1 = next(row for row in r["replay_selection_by_lead"] if row["lead_hours"] == 1)
            dlinear_inner[f] = {"mae": r["registered_selection"]["mae"], "rmse": r["registered_selection"]["rmse"],
                                "lead1_mae": lead1["mae"], "lead1_rmse": lead1["rmse"],
                                "source": "registered v20 history (overall) + CPU replay (lead 1)"}
        a0 = {"stage": "A0", "identity": identity, "data": folds, "persistence_inner": persistence,
              "dlinear_anchor_inner": dlinear_inner, "epoch0_identity": epoch0, "new_fits": 0, "outer_access": False}
        stable = {k: a0[k] for k in ("data", "persistence_inner", "dlinear_anchor_inner")}
        path = self.dir / "a0_result.json"
        if path.exists() and v26.canonical_sha256({k: v26.read_json(path)[k] for k in stable}) != v26.canonical_sha256(stable):
            raise ValueError("A0 contracts differ from the registered A0 of this run; data or anchors drifted")
        if not path.exists():
            v26.write_json(self.dir / "data_manifest.json", {"protocol_sha256": self.protocol_sha, "folds": folds,
                           "data_path": self.p["data_path"], "data_sha256": self.p["data_sha256"],
                           "feature_columns": FEATURES, "sealed_test_requested": False})
            v26.write_json(self.dir / "reference_manifest.json", {"identity": identity, "dlinear_anchor": reference,
                           "v25_historical": self.p["v25_historical"], "reuse": "seed-2024 DLinear never retrained"})
            v26.write_json(path, a0 | {"completed_utc": now()})
            self.log("A0 contracts, anchor replay and epoch-0 identity",
                     purpose="verify v20 queue/scaler/anchors and that both arms start exactly at the anchor before any fit",
                     persistence_inner=persistence, dlinear_inner=dlinear_inner,
                     epoch0_identity={f: e["arms"] for f, e in epoch0.items()}, new_fits=0)
        v26.append_event(self.events, "A0_verified")
        self.a0 = v26.read_json(path)
        return self.a0

    # ------------------------------------------------------------------ devices and fits
    def pair_device(self, fold, seed, arms):
        path = self.dir / "runs" / f"f{fold}_s{seed}" / "pair_device.json"
        ops = self.p["operations"]
        if path.exists():
            record = v26.read_json(path)
            folder = path.parent
            if all((folder / arm / "trained.json").exists() for arm in arms):
                return torch.device(f"cuda:{self.gpu_index}" if record["device"] == "cuda" else "cpu")
            if record["device"] == "cuda":
                waited, began = False, time.monotonic()
                while True:
                    snapshot = gpu_snapshot(self.gpu_index)
                    if snapshot and snapshot["free_mib"] / 1024. >= ops["resume_gpu_free_gib_min"]:
                        break
                    if not waited:
                        v26.append_event(self.events, "waiting_for_registered_pair_device", fold=fold, seed=seed, snapshot=snapshot)
                        waited = True
                    if time.monotonic() - began > ops["resume_device_wait_seconds_max"]:
                        raise TimeoutError("Registered pair device unavailable; resume later (no silent device switch)")
                    time.sleep(60)
        else:
            record = v26.choose_pair_device(gpu_snapshot(self.gpu_index), ops) | {"fold": fold, "seed": seed, "decided_utc": now()}
            v26.write_json(path, record)
            v26.append_event(self.events, "pair_device_decided", **record)
        if record["device"] == "cuda":
            device = torch.device(f"cuda:{self.gpu_index}")
            torch.cuda.set_per_process_memory_fraction(record["memory_fraction"], device)
            return device
        return torch.device("cpu")

    def contract(self, arm, fold, seed, data, device):
        anchor = self.load_anchor(fold)
        _, factory = build_model(arm, self.p)
        series = data["fit"].series
        return {"protocol_sha256": self.protocol_sha, "upstream_pin_sha256": self.pin_sha, "training_code_sha256": self.training_code,
                "data_sha256": self.p["data_sha256"], "arm": arm, "fold": int(fold), "seed": int(seed),
                "model_factory": factory["factory"], "model_kwargs": factory["kwargs"],
                "trainable_parameters": factory["trainable_parameters"], "frozen_parameters": factory["frozen_parameters"],
                "anchor_checkpoint_sha256": anchor["checkpoint_sha256"], "anchor_state_sha256": anchor["state_sha256"],
                "scaler_mean": series.scaler.mean_.tolist(), "scaler_scale": series.scaler.scale_.tolist(),
                "fit_window_hash": v26.window_hash(data["fit"].window_starts),
                "selection_window_hash": v26.window_hash(data["selection"].window_starts), "training": self.p["training"],
                "device_type": torch.device(device).type, "determinism": determinism_record()}

    def registered_new_fits(self):
        return sorted(self.rel(p.parent) for p in (self.dir / "runs").glob("f*_s*/*/trained.json")
                      if not v26.read_json(p).get("reused", False))

    def anchor_reference_kw(self, fold):
        """The registered CPU replay of the anchor on the selection windows (epoch-0 identity target)."""
        seed = self.p["seeds"]["anchor_seed"]
        path = self.dir / ANCHOR_REPLAY_DIR / f"f{fold}_s{seed}_DLinear_selection_predictions.npz"
        with np.load(path, allow_pickle=False) as saved:
            return saved["prediction"].astype(np.float64)

    def fit(self, fold, seed, arm, device, stage, init_reference=None):
        _, data = self.prepare_fold(fold, stage)
        out = self.dir / "runs" / f"f{fold}_s{seed}" / arm
        contract = self.contract(arm, fold, seed, data, device)
        if not (out / "trained.json").exists():
            self.check_preflight()
            started = (out / "last.pt").exists()
            if not started and len(self.registered_new_fits()) + 1 > self.p["budget"]["new_fits_max"]:
                raise ValueError("The 18-fit budget would be exceeded; refuse")
            self.set_status(stage=stage, train="RUNNING")
            v26.write_json(self.dir / "progress.json", {"stage": stage, "fold": fold, "seed": seed, "arm": arm,
                                                         "device": str(device), "utc": now()})
            v26.append_event(self.events, "fit_started" if not started else "fit_resuming", stage=stage, fold=fold,
                             seed=seed, arm=arm, device=str(device))
        self.fit_calls.append((fold, seed, arm))
        trained = fit_arm(arm, self.p, fit=data["fit"], selection=data["selection"], contract=contract, seed=seed,
                          device=device, out=out, anchor_state=self.load_anchor(fold)["state"],
                          anchor_reference_kw=self.anchor_reference_kw(fold), events=self.events,
                          init_reference=init_reference)
        if not trained.get("reused_from_registry"):
            v26.append_event(self.events, "fit_completed", stage=stage, fold=fold, seed=seed, arm=arm,
                             best_epoch=trained["best_epoch"], selection_mae_kw=trained["selection_best"]["mae"],
                             selection_rmse_kw=trained["selection_best"]["rmse"])
            print(f"[FIT] {stage} f{fold} s{seed} {arm} best_epoch={trained['best_epoch']} "
                  f"MAE={trained['selection_best']['mae']:.6f} RMSE={trained['selection_best']['rmse']:.6f}", flush=True)
        gc.collect()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        return trained

    def candidate_pair(self, fold, seed, stage, device):
        full = self.fit(fold, seed, v26.PRIMARY, device, stage)
        self.fit(fold, seed, v26.INFO_CONTROL, device, stage,
                 init_reference=self.dir / "runs" / f"f{fold}_s{seed}" / v26.PRIMARY / "init.pt")
        return full

    # ------------------------------------------------------------------ records and gates
    def inner_record(self, fold, seed, arm):
        if arm == v26.BASELINE:
            if seed != self.p["seeds"]["anchor_seed"]:
                return None
            values = self.a0["dlinear_anchor_inner"][str(fold)]
            return {"fold": int(fold), "seed": int(seed), "arm": arm, "mae": values["mae"], "rmse": values["rmse"],
                    "lead1_mae": values["lead1_mae"], "lead1_rmse": values["lead1_rmse"], "best_epoch": None,
                    "source": values["source"]}
        base = self.dir / "runs" / f"f{fold}_s{seed}" / arm
        if not (base / "trained.json").exists():
            return None
        trained = v26.read_json(base / "trained.json")
        return v26.record_from_metrics(fold, seed, arm, trained["selection_best"], trained["selection_best_by_lead"],
                                       best_epoch=trained["best_epoch"], source="v26 fit (best.pt)")

    def records(self, seeds, arms):
        rows = [self.inner_record(f, s, a) for f in self.folds for s in seeds for a in arms]
        return [r for r in rows if r is not None]

    def persistence_inner(self):
        return {int(f): v for f, v in self.a0["persistence_inner"].items()}

    def register_gate(self, path, payload, key):
        if path.exists():
            old = v26.read_json(path)
            if old[key]["passed"] != payload[key]["passed"] or \
                    v26.canonical_sha256(old["records"]) != v26.canonical_sha256(payload["records"]):
                raise ValueError(f"Registered {key} decision/records would change; refuse and review")
            if old["gate_code_sha256"] != payload["gate_code_sha256"]:
                v26.append_event(self.events, "gate_code_changed_same_decision", gate=key)
            return old
        v26.write_json(path, payload)
        return payload

    def gate1(self):
        g = self.p["gates"]
        records = self.records([g["G1"]["seed"]], v26.ALL_ARMS)
        payload = {"G1": v26.gate_g1(records, self.persistence_inner(), g["G1"], self.folds, self.tol),
                   "I1": v26.info_i1(records, g["I1"], self.folds, self.tol), "records": records,
                   "gate_code_sha256": gate_code_sha256(), "evaluated_utc": now(), "scope": "internal selection only"}
        payload = self.register_gate(self.dir / "inner_gate_seed2024.json", payload, "G1")
        v26.append_event(self.events, "G1", passed=payload["G1"]["passed"], I1=payload["I1"]["supported"])
        self.log("A1 complete: G1/I1",
                 hypothesis="AnchoredFull beats the frozen DLinear and Persistence on internal selection without harming lead 1 (seed 2024)",
                 G1_passed=payload["G1"]["passed"], G1_checks=payload["G1"].get("checks"), I1_supported=payload["I1"]["supported"],
                 next_step="A2 fixed residual seeds 2025/2026" if payload["G1"]["passed"] else "stop: no outer predictions")
        print(f"[G1] passed={payload['G1']['passed']} I1={payload['I1']['supported']}", flush=True)
        return payload["G1"]

    def gate2(self):
        g = self.p["gates"]
        anchor_seed = self.p["seeds"]["anchor_seed"]
        records = self.records(g["G2"]["seeds"], v26.CANDIDATE_ARMS) + self.records([anchor_seed], (v26.BASELINE,))
        payload = {"G2": v26.gate_g2(records, self.persistence_inner(), g["G2"], self.folds, anchor_seed, self.tol),
                   "I2": v26.info_i2(records, g["I2"], self.folds, self.tol), "records": records,
                   "gate_code_sha256": gate_code_sha256(), "evaluated_utc": now(), "scope": "internal selection only"}
        payload = self.register_gate(self.dir / "inner_gate_all_seeds.json", payload, "G2")
        v26.append_event(self.events, "G2", passed=payload["G2"]["passed"], I2=payload["I2"]["supported"])
        self.log("A2 complete: G2/I2", G2_passed=payload["G2"]["passed"], G2_checks=payload["G2"].get("checks"),
                 I2_supported=payload["I2"]["supported"],
                 next_step="A3 single outer evaluation" if payload["G2"]["passed"] else "stop: no outer predictions")
        print(f"[G2] passed={payload['G2']['passed']} I2={payload['I2']['supported']}", flush=True)
        return payload["G2"]

    # ------------------------------------------------------------------ stages
    def stage_a1(self):
        seed = self.p["seeds"]["A1"][0]
        for fold in self.folds:
            self.candidate_pair(fold, seed, "A1", self.pair_device(fold, seed, v26.CANDIDATE_ARMS))
        return self.gate1()

    def stage_a2(self):
        for seed in self.p["seeds"]["A2"]:
            for fold in self.folds:
                self.candidate_pair(fold, seed, "A2", self.pair_device(fold, seed, v26.CANDIDATE_ARMS))
        return self.gate2()

    def outer_freeze(self):
        fits = {self.rel(p.parent): {"trained_sha256": v26.file_sha256(p),
                                     "best_sha256": v26.file_sha256(p.parent / "best.pt")}
                for p in sorted((self.dir / "runs").glob("f*_s*/*/trained.json"))}
        freeze = {"protocol.json": v26.file_sha256(self.dir / "protocol.json"),
                  "a0_result.json": v26.file_sha256(self.dir / "a0_result.json"),
                  "inner_gate_seed2024.json": v26.file_sha256(self.dir / "inner_gate_seed2024.json"),
                  "inner_gate_all_seeds.json": v26.file_sha256(self.dir / "inner_gate_all_seeds.json"), "fits": fits}
        path = self.dir / "outer_freeze.json"
        if path.exists():
            if v26.canonical_sha256(v26.read_json(path)["freeze"]) != v26.canonical_sha256(freeze):
                raise ValueError("Selected checkpoints/gates changed after the outer freeze")
        else:
            v26.write_json(path, {"freeze": freeze, "frozen_utc": now(), "rule": "written before any outer request"})
        return freeze

    def outer_device(self, fold, seed):
        record = v26.read_json(self.dir / "runs" / f"f{fold}_s{seed}" / "pair_device.json")
        if record["device"] == "cuda" and torch.cuda.is_available():
            snapshot = gpu_snapshot(self.gpu_index)
            if snapshot and snapshot["free_mib"] / 1024. >= self.p["operations"]["resume_gpu_free_gib_min"]:
                device = torch.device(f"cuda:{self.gpu_index}")
                torch.cuda.set_per_process_memory_fraction(record["memory_fraction"], device)
                return device
        if record["device"] == "cuda":
            v26.append_event(self.events, "outer_inference_on_cpu", fold=fold, seed=seed,
                             note="inference device only; weights and selection unchanged")
        return torch.device("cpu")

    def outer_anchor_check(self, fold, validation, cohort, registered_path):
        """The frozen anchor replayed on outer must reproduce the registered v20 DLinear outer forecast."""
        anchor = self.load_anchor(fold)
        model, _ = build_model(v26.BASELINE, self.p, anchor_state=anchor["state"])
        prediction = predict(model, validation, self.p["seeds"]["anchor_seed"], torch.device("cpu"),
                             self.p["training"]["eval_batch_size"])
        with np.load(registered_path, allow_pickle=False) as saved:
            v26.assert_aligned(cohort, dict(saved), label=f"v20 DLinear outer f{fold}")
            difference = float(np.max(np.abs(prediction - saved["prediction"].astype(np.float64))))
        if difference > self.p["reference"]["outer_replay_max_abs_tolerance_kw"]:
            raise ValueError(f"Anchor executor on outer differs from the registered v20 DLinear by {difference} kW")
        return {"fold": fold, "max_abs_diff_kw": difference, "device": "cpu"}

    def stage_a3(self):
        if not v26.read_json(self.dir / "inner_gate_all_seeds.json")["G2"]["passed"]:
            raise PermissionError("A3 requires a registered passed G2")
        self.set_status(stage="A3", train="RUNNING")
        self.outer_freeze()
        seeds = self.p["gates"]["G2"]["seeds"]
        index, anchor_checks = {}, {}
        refdir = self.reference_root / self.p["reference"]["dir"]
        anchor_seed = self.p["seeds"]["anchor_seed"]
        for fold in self.folds:
            series, data = self.prepare_fold(fold, "A3")
            self.access.request("A3", "validation", fold, g2_passed=True)
            validation = series.subset("validation")
            truth, persistence = validation.truth_and_anchor()
            cohort = {"times": validation.target_times(), "turbines": validation.turbines[validation.window_starts],
                      "window_starts": validation.window_starts, "truth": truth, "persistence": persistence}
            with np.load(refdir / "runs" / f"f{fold}_s{anchor_seed}/Persistence/raw_predictions.npz", allow_pickle=False) as saved:
                v26.assert_aligned(cohort, dict(saved), label=f"v20 Persistence f{fold}")
            dlinear_path = refdir / "runs" / f"f{fold}_s{anchor_seed}" / "DLinear" / "raw_predictions.npz"
            anchor_checks[str(fold)] = self.outer_anchor_check(fold, validation, cohort, dlinear_path)
            cohort_path = self.dir / "runs" / f"f{fold}_outer_cohort.npz"
            if not cohort_path.exists():
                np.savez_compressed(cohort_path, **cohort)
            index[f"f{fold}/Persistence"] = {"fold": fold, "seed": None, "arm": "Persistence", "path": self.rel(cohort_path),
                                              "base": "run", "sha256": v26.file_sha256(cohort_path), "array": "persistence"}
            index[f"f{fold}/s{anchor_seed}/DLinear"] = {"fold": fold, "seed": anchor_seed, "arm": v26.BASELINE,
                                                         "path": self.rel_reference(dlinear_path), "base": "reference",
                                                         "sha256": v26.file_sha256(dlinear_path), "array": "prediction",
                                                         "source": "v20 registered outer prediction (= the frozen anchor), hash-verified"}
            for seed in seeds:
                for arm in v26.CANDIDATE_ARMS:
                    index[f"f{fold}/s{seed}/{arm}"] = self.outer_prediction(fold, seed, arm, validation, cohort)
        v26.write_json(self.dir / "outer_index.json", {"entries": index, "anchor_executor_checks": anchor_checks,
                                                     "scope": "outer validation, development evidence",
                                                     "selection_on_outer": False, "registered_utc": now()})
        self.log("A3 single outer evaluation", folds=self.folds, seeds=seeds, arms=list(v26.ALL_ARMS),
                 anchor_executor_checks=anchor_checks,
                 note="every fold/seed/arm registered, harmed ones included; no outer selection or refit")
        return index

    def outer_prediction(self, fold, seed, arm, validation, cohort):
        base = self.dir / "runs" / f"f{fold}_s{seed}" / arm
        path = base / "outer_predictions.npz"
        trained = v26.read_json(base / "trained.json")
        v26.verify_registered_files(base, trained["artifacts"])
        result_path = base / "result.json"
        if result_path.exists():
            result = v26.read_json(result_path)
            if result["best_state_sha256"] != trained["best_state_sha256"]:
                raise ValueError("Outer prediction executor differs from the registered selected state")
            v26.verify_registered_files(base, {path.name: result["outer_predictions_sha256"]})
        else:
            device = self.outer_device(fold, seed)
            model, _ = build_model(arm, self.p, anchor_state=self.load_anchor(fold)["state"])
            state = torch.load(base / "best.pt", map_location="cpu", weights_only=False)["model"]
            if state_digest(state) != trained["best_state_sha256"] or anchor_digest(state) != trained["anchor_state_sha256"]:
                raise ValueError("best.pt differs from the registered selected state or anchor")
            model.load_state_dict(state, strict=True)
            model.to(device)
            begin = time.monotonic()
            prediction = predict(model, validation, seed, device, self.p["training"]["eval_batch_size"])
            seconds = time.monotonic() - begin
            arrays = {"prediction": prediction, **cohort}
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    v26.assert_aligned(arrays, dict(saved), keys=tuple(arrays), label="interrupted outer inference")
            else:
                temporary = path.with_name(path.stem + ".tmp.npz")
                np.savez_compressed(temporary, **arrays)
                os.replace(temporary, path)
            metrics = forecast_metrics(prediction, cohort["truth"], rated_power=self.p["rated_power_kw"])
            v26.write_json(base / "result.json", {"arm": arm, "fold": fold, "seed": seed, "split": "outer_validation",
                           "best_epoch": trained["best_epoch"], "metrics": metrics, "inference_seconds": seconds,
                           "best_state_sha256": trained["best_state_sha256"], "anchor_state_sha256": trained["anchor_state_sha256"],
                           "inference_device": str(device), "outer_predictions_sha256": v26.file_sha256(path),
                           "evidence_scope": "development; outer previously inspected in earlier versions"})
            print(f"[OUTER] f{fold} s{seed} {arm} MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f}", flush=True)
        with np.load(path, allow_pickle=False) as saved:
            v26.assert_aligned(cohort, dict(saved), label=f"outer f{fold} s{seed} {arm}")
        return {"fold": fold, "seed": seed, "arm": arm, "path": self.rel(path), "base": "run", "sha256": v26.file_sha256(path),
                "array": "prediction", "source": "v26 best.pt (anchor + selected residual)"}

    def finish_training(self, outer_evaluated):
        for marker in sorted((self.dir / "runs").glob("f*_s*/*/trained.json")):
            folder = marker.parent
            v26.write_json(folder / "completed.json", {p.name: v26.file_sha256(p) for p in sorted(folder.iterdir())
                                                        if registrable(p, self.dir) and p.name != "completed.json"})
        hashes = {self.rel(p): v26.file_sha256(p) for p in sorted(self.dir.rglob("*"))
                  if registrable(p, self.dir) and p.name != "training_completed.json"}
        v26.write_json(self.dir / "training_completed.json", {"outer_evaluated": bool(outer_evaluated), "hashes": hashes,
                       "registered_new_fits": self.registered_new_fits(), "completed_utc": now()})
        self.set_status(stage="TRAINING_DONE", train="TRAIN_COMPLETED", outer_evaluated=bool(outer_evaluated))
        v26.append_event(self.events, "TRAIN_COMPLETED", outer_evaluated=bool(outer_evaluated),
                         new_fits=len(self.registered_new_fits()))
        self.log("Training complete", outer_evaluated=outer_evaluated, new_fits=len(self.registered_new_fits()),
                 budget=f"<= {self.p['budget']['new_fits_max']} fits")

    def verify_training_completed(self):
        return v26.verify_training_completed(self.dir)

    def report(self):
        from scripts import report_kelmarsh_anchored_residual as reporting
        self.set_status(stage="A4", report="RUNNING")
        try:
            summary = reporting.build_report(self.dir, reference_root=self.reference_root)
        except Exception as error:
            self.set_status(report="REPORT_FAILED", report_error=f"{type(error).__name__}: {error}")
            v26.append_event(self.events, "REPORT_FAILED", error=f"{type(error).__name__}: {error}")
            self.log("A4 report failed", error=f"{type(error).__name__}: {error}",
                     recovery="fix the report code only, then launcher --report-only (no retraining)")
            raise
        self.set_status(stage="A4", report="REPORT_COMPLETED", report_error=None)
        v26.write_json(self.dir / "completed.json", {self.rel(p): v26.file_sha256(p) for p in sorted(self.dir.rglob("*"))
                                                     if registrable(p, self.dir)})
        self.set_status(stage="COMPLETED")
        v26.append_event(self.events, "COMPLETED")
        self.log("A4 complete; v26 stops here", conclusion=summary.get("conclusion"),
                 next_step="decide from the registered result; no automatic follow-up")
        return summary

    def run(self, report_only=False):
        self.dir.mkdir(parents=True, exist_ok=True)
        lock = v26.acquire_exclusive_lock(self.lock_path)
        try:
            if (self.dir / "completed.json").exists() and not report_only:
                self.freeze_protocol_record()  # verifies identity only; writes nothing into a completed run
                self.verify_training_completed()
                verify_completed_suite(self.dir)
                print("[REUSE] v26 already completed; nothing retrained", flush=True)
                return "COMPLETED"
            configure_determinism()
            (self.dir / "controller.pid").write_text(str(os.getpid()), encoding="utf-8")
            if not (self.dir / "training_completed.json").exists():
                load_pin(self.p, require_resolved=True)
            self.freeze_protocol_record()
            self.claim_pointer()
            self.record_sources()
            v26.append_event(self.events, "controller_started", argv=sys.argv, pid=os.getpid(), report_only=report_only)
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
    """Old (v25 TSMixerFull) -> new values; unchanged recipe keys are recorded as inherited."""
    inherited = {k: "inherited from v25 / v20 recipe" for k in ("learning_rate", "weight_decay", "batch_size",
                 "max_epochs", "patience", "grad_clip_norm", "scheduler", "loss", "selection_metric", "data", "queue",
                 "scaler", "features", "horizon", "stride", "residual_architecture", "dropout", "seeds_A1")}
    return {"reference": "v25 TSMixerFull seed 2024 (configs/kelmarsh_backbone_information_protocol_v25.json)",
            "changes": [
                {"parameter": "forecast", "old": "y = TSMixer(x)", "new": "y = DLinear_frozen(x) + TSMixer_zero_head(x)"},
                {"parameter": "head_initialization", "old": "xavier_uniform / zero bias", "new": "weight 0, bias 0 (draws unchanged)"},
                {"parameter": "frozen_parameters", "old": 0, "new": 200},
                {"parameter": "trainable_parameters", "old": 3618, "new": 3618},
                {"parameter": "epoch0_selectable", "old": "included (random init, never selected in practice)",
                 "new": "recorded, not selectable (epoch 0 = the reference itself)"},
                {"parameter": "patience_origin", "old": "epoch 0", "new": "epoch 1"},
                {"parameter": "decision_rule", "old": "G1/G2 macro + 2 folds", "new": "same + lead-1 macro MAE/RMSE not worse than DLinear"},
                {"parameter": "A2_arms", "old": "TSMixerFull, TSMixerPowerOnly, DLinear (18 fits)",
                 "new": "AnchoredFull, AnchoredPowerOnly (12 fits; anchor never retrained)"},
                {"parameter": "budget", "old": "24 fits / 480 epochs", "new": "18 fits / 360 epochs"},
                {"parameter": "information_ablation_mask", "old": "before the TSMixer", "new": "before anchor AND residual (anchor unaffected)"}],
            "inherited": inherited, "wiki": "disabled (unchanged)"}


# =========================================================================== preflight
class SyntheticWindows(Dataset):
    """Tiny KelmarshWindows-like dataset for the pre-launch smoke test (never real data)."""

    def __init__(self, n=96, seed=0, offset=0):
        rng = np.random.default_rng(seed)
        self.values = rng.normal(size=(n + 28, 7)).astype(np.float32)
        self.values[:, -1] = np.sin(np.arange(n + 28) / 5.).astype(np.float32) + .3 * self.values[:, 0]
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


def synthetic_anchor_state(p, seed=5):
    """A deterministic DLinear state for synthetic tests (never a real checkpoint)."""
    seed_all(seed)
    model = DLinearModel(SimpleNamespace(**p["arms"][v26.BASELINE]["model_args"]))
    return cpu_state(model)


def anchor_reference_for(p, anchor_state, dataset):
    model, _ = build_model(v26.BASELINE, p, anchor_state=anchor_state)
    return predict(model, dataset, 0, torch.device("cpu"), p["training"]["eval_batch_size"])


def smoke(p, folder):
    """Synthetic smoke: both arms, identical init, frozen anchor unchanged, interrupt+resume equals uninterrupted."""
    q = {**p, "training": {**p["training"], "max_epochs": 3, "scheduler_t_max": 3}}
    fit, selection, cpu = SyntheticWindows(96, 1), SyntheticWindows(40, 2), torch.device("cpu")
    anchor = synthetic_anchor_state(q)
    reference = anchor_reference_for(q, anchor, selection)
    results = {}
    for arm in v26.CANDIDATE_ARMS:
        contract = {"smoke": True, "arm": arm}
        init = folder / "full" / v26.PRIMARY / "init.pt" if arm == v26.INFO_CONTROL else None
        results[arm] = fit_arm(arm, q, fit=fit, selection=selection, contract=contract, seed=7, device=cpu,
                               out=folder / "full" / arm, anchor_state=anchor, anchor_reference_kw=reference,
                               init_reference=init)
    contract = {"smoke": True, "arm": v26.PRIMARY}
    try:
        fit_arm(v26.PRIMARY, q, fit=fit, selection=selection, contract=contract, seed=7, device=cpu,
                out=folder / "interrupted" / v26.PRIMARY, anchor_state=anchor, anchor_reference_kw=reference,
                stop_inside_epoch=(2, 1))
    except SimulatedInterrupt:
        pass
    resumed = fit_arm(v26.PRIMARY, q, fit=fit, selection=selection, contract=contract, seed=7, device=cpu,
                      out=folder / "interrupted" / v26.PRIMARY, anchor_state=anchor, anchor_reference_kw=reference)
    a = torch.load(folder / "full" / v26.PRIMARY / "last.pt", map_location="cpu", weights_only=False)
    b = torch.load(folder / "interrupted" / v26.PRIMARY / "last.pt", map_location="cpu", weights_only=False)
    same = state_digest(a["model"]) == state_digest(b["model"]) and resumed["best_epoch"] == results[v26.PRIMARY]["best_epoch"]
    anchors_unchanged = all(r["anchor_state_sha256"] == state_digest(anchor) for r in results.values())
    paired = results[v26.PRIMARY]["init_state_sha256"] == results[v26.INFO_CONTROL]["init_state_sha256"]
    return {"passed": bool(same and paired and anchors_unchanged),
            "arms": {k: {"best_epoch": v["best_epoch"], "selection_mae": v["selection_best"]["mae"],
                         "epoch0_max_abs_diff_vs_anchor_kw": v["epoch0_max_abs_diff_vs_anchor_kw"]} for k, v in results.items()},
            "resume_matches_uninterrupted": bool(same), "paired_init": bool(paired), "anchor_unchanged": bool(anchors_unchanged),
            "data": "synthetic tensors and a synthetic anchor only; not a real-data candidate"}


def preflight(gpu_index=0, skip_tests=False):
    p, protocol_sha = load_protocol()
    pin, resolved = load_pin(p, require_resolved=False)
    digest = full_source_digest(protocol_sha, pin)
    folder = PREFLIGHT_DIR / f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}_{digest[:12]}"
    folder.mkdir(parents=True, exist_ok=False)
    configure_determinism()
    checks = {"pin_resolved": resolved, "upstream_pin": pin}
    others = sorted({str(path.relative_to(ROOT)) for path in (ROOT / "configs").glob("*v26*")} - {
        "configs/kelmarsh_anchored_residual_protocol_v26.json"})
    pointers = {str(path.relative_to(ROOT)): path.read_text(errors="replace").strip()
                for path in LOG_ROOT.glob("*latest.txt") if "v26" in path.read_text(errors="replace")
                and path.resolve() != POINTER.resolve()}
    checks["v26_id_free"] = {"other_v26_configs": others, "other_pointers_mentioning_v26": pointers,
                             "own_pointer": POINTER.read_text().strip() if POINTER.exists() else None,
                             "passed": not others and not pointers}
    if skip_tests:
        checks["tests"] = {"passed": False, "skipped": True}
    else:
        try:
            import pytest  # noqa: F401
            command = [sys.executable, "-m", "pytest", "-q", *V26_TESTS, *REGRESSION_TESTS]
        except ImportError:
            command = [sys.executable, "-m", "unittest", *[t.replace("/", ".")[:-3] for t in V26_TESTS + REGRESSION_TESTS]]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (folder / "tests.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        checks["tests"] = {"command": command, "exit_code": result.returncode, "passed": result.returncode == 0,
                           "tail": (result.stdout + result.stderr).splitlines()[-15:]}
    checks["bash_syntax"] = {"exit_code": subprocess.run(["bash", "-n", "scripts/train/Kelmarsh_launch_anchored_residual.sh"],
                                                          cwd=ROOT).returncode}
    checks["bash_syntax"]["passed"] = checks["bash_syntax"]["exit_code"] == 0
    try:
        a0 = V26Run(folder / "a0", protocol=p, protocol_sha=protocol_sha, pin=pin, require_preflight=False,
                    write_pointer=False, lock_path=PREFLIGHT_DIR / "preflight.lock")
        a0.dir.mkdir(parents=True)
        a0.freeze_protocol_record()
        a0.record_sources()
        result = a0.stage_a0()
        checks["A0"] = {"passed": True, "persistence_inner": result["persistence_inner"],
                        "dlinear_anchor_inner": result["dlinear_anchor_inner"],
                        "epoch0_identity": {f: e["arms"] for f, e in result["epoch0_identity"].items()},
                        "windows": {f: [d["fit_windows"], d["selection_windows"]] for f, d in result["data"].items()},
                        "subsets_requested": sorted({r[1] for r in a0.access.requests})}
    except Exception as error:
        checks["A0"] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    if not checks["A0"]["passed"]:
        checks["gpu_epoch0_identity"] = {"passed": False, "reason": "A0 failed"}
    else:
        try:
            device, decision = preflight_device(p, gpu_index)
            probes = [a0.device_identity_probe(fold, device) for fold in p["folds"]]
            checks["gpu_epoch0_identity"] = {"passed": all(r["passed"] for r in probes), "probes": probes,
                                             "resource_decision": decision,
                                             "cuda_probed": device.type == "cuda"}
        except Exception as error:
            checks["gpu_epoch0_identity"] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    try:
        checks["synthetic_smoke"] = smoke(p, folder / "smoke")
    except Exception as error:
        checks["synthetic_smoke"] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    checks["resources"] = {"gpu": gpu_snapshot(gpu_index), "loadavg": os.getloadavg(), "cpu_count": os.cpu_count(),
                           "decision_if_pair_started_now": v26.choose_pair_device(gpu_snapshot(gpu_index), p["operations"])}
    passed = all(checks[k]["passed"] for k in ("v26_id_free", "tests", "bash_syntax", "A0", "gpu_epoch0_identity",
                                               "synthetic_smoke")) and resolved
    record = {"passed": bool(passed), "full_source_digest": digest, "protocol_sha256": protocol_sha,
              "training_code_sha256": training_code_sha256(), "git": git_info(), "utc": now(), "folder": str(folder),
              "checks": checks}
    v26.write_json(folder / "preflight.json", record)
    if passed:
        v26.write_json(PREFLIGHT_DIR / f"passed_{digest[:16]}.json", record)
    print(f"[PREFLIGHT] passed={passed} pin_resolved={resolved} tests={checks['tests'].get('passed')} "
          f"A0={checks['A0']['passed']} gpu_epoch0={checks['gpu_epoch0_identity']['passed']} "
          f"smoke={checks['synthetic_smoke']['passed']} record={folder}", flush=True)
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
    run = V26Run(args.output.resolve(), gpu_index=args.gpu)
    try:
        run.run(report_only=args.report_only)
        return 0
    except BlockingIOError:
        print("[BUSY] Another v26 controller holds the lock; nothing changed", flush=True)
        return 3
    except Exception as error:
        if args.output.exists():
            status = run.dir / "status.json"
            current = v26.read_json(status) if status.exists() else {}
            if current.get("train") != "TRAIN_COMPLETED":
                run.set_status(train="FAILED", error=f"{type(error).__name__}: {error}")
            v26.append_event(run.events, "controller_failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    sys.exit(main())
