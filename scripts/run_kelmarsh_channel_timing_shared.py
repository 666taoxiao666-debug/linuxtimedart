#!/usr/bin/env python3
"""Explicitly authorized, process-local sharing for the unchanged v23 trial.

This entry point changes execution controls only. It does not modify the frozen
model/data/selection recipe, relax its source locks, or signal any other job.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_kelmarsh_channel_timing as trial
from utils.gpu_runtime_policy import GPURuntimePolicy


@contextmanager
def sharing_adapter(directory, resources):
    """Wrap the original loader without changing batches, seeds or optimizer."""
    original_loader = trial.shared.loader
    original_guard = trial.shared.gpu_available_to_this_process
    original_wait = trial.prior.wait_for_gpu

    def check():
        if resources.mode is None:
            resources.configure()
        else:
            resources.check(force=True)

    def enter_stage(output, fold, seed, stage):
        check()
        trial.shared.write(output / "progress.json", dict(
            stage=stage, fold=fold, seed=seed,
            execution_policy="authorized_adaptive_sharing",
            resource_mode=resources.mode))

    def loader(*args, **kwargs):
        for batch in original_loader(*args, **kwargs):
            resources.check()
            yield batch
            # Runs after each optimizer/evaluation step; synchronize and yield
            # only our process. No allocation/compute quota is claimed here.
            resources.after_step()

    trial.shared.loader = loader
    trial.shared.gpu_available_to_this_process = check
    trial.prior.wait_for_gpu = enter_stage
    try:
        yield
    finally:
        trial.shared.loader = original_loader
        trial.shared.gpu_available_to_this_process = original_guard
        trial.prior.wait_for_gpu = original_wait


def bind_execution(directory):
    """Append a separate immutable runtime-policy record; retain old provenance."""
    contract = dict(
        execution_policy="authorized_adaptive_sharing",
        authorization="User explicitly requested sharing rather than GPU-idle queueing on 2026-10-16",
        training_protocol_sha256=trial.PROTOCOL_SHA256,
        gpu=0, shared_allocator_gib=2., shared_headroom_gib=4.,
        shared_cpu_threads=2, shared_step_delay_ms=20.,
        solo_policy="allocator unrestricted; CPU affinity threads; no artificial delay",
        policy_sources={name: trial.shared.sha(ROOT / name) for name in (
            "scripts/run_kelmarsh_channel_timing_shared.py", "utils/gpu_runtime_policy.py")},
        model_data_loss_lr_seeds_unchanged=True,
        original_runtime_snapshot_preserved=True,
        controls_scope="Our allocator/threads/yield only, not a total-VRAM or GPU-utilization quota",
        limitation="CUDA/driver allocations are outside the allocator cap; no zero-impact or crash-free guarantee")
    path = directory / "shared_execution_contract.json"
    if path.exists():
        if trial.shared.read(path) != contract:
            raise ValueError("Shared execution identity changed; do not overwrite prior contract")
    else:
        trial.shared.write(path, contract)
    record = dict(
        recorded_at_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        actual_python_argv=sys.argv, executable=sys.executable,
        execution_contract_sha256=trial.shared.sha(path))
    # The original source_commit/runtime_snapshot document the first queued
    # launch. This append-only record identifies the later sharing entry point.
    import subprocess
    record["git_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    attempt = directory / f"shared_launch_{os.getpid()}.json"
    if attempt.exists():
        raise ValueError("Shared launch record already exists")
    trial.shared.write(attempt, record)


def run_shared(directory):
    import fcntl
    directory = directory.resolve()
    if not directory.is_dir() or not (directory / "protocol.json").exists():
        raise ValueError("Only the existing, identified v23 directory may be resumed")
    # Probe the same suite lock before recording anything, then release it
    # before trial.run acquires its own descriptor. A wrapper-specific lock
    # excludes duplicate sharing entry points throughout the whole execution.
    with (directory / "shared_execution.lock").open("a") as wrapper_lock:
        fcntl.flock(wrapper_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (trial.POINTER.parent / "channel_timing.lock").open("a") as suite_lock:
            fcntl.flock(suite_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (directory / "completed.json").exists():
                trial.prior.verify_completed_suite(directory)
                print("[REUSE] Completed v23; no new shared execution", flush=True)
                return
            bind_execution(directory)
        resources = GPURuntimePolicy(shared=True, allocator_gib=2., headroom_gib=4.,
            step_delay_ms=20., on_change=lambda state: trial.shared.write(
                directory / "resource_policy.json", state))
        print("[SHARED] Explicit user authorization: no idle-only queue; allocator2GiB/CPU2/yield20ms when sharing", flush=True)
        with sharing_adapter(directory, resources):
            trial.run(directory, gpu=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_shared(args.output)
    except BlockingIOError:
        print("[BUSY] Existing v23 process left unchanged", flush=True)
        raise SystemExit(2)
    except Exception as error:
        # Only our run is stopped on memory pressure; never signal another PID
        # or silently change batch size, learning rate or model configuration.
        if args.output.is_dir():
            (args.output / "status.env").write_text(
                f"STATUS=FAILED\nERROR={type(error).__name__}: {error}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
