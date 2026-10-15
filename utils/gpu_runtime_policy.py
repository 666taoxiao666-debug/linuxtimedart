"""Process-local resource controls. Never signal or reconfigure another job."""
from __future__ import annotations

import os
import subprocess
import time
import torch


class GPURuntimePolicy:
    def __init__(self, *, shared=False, allocator_gib=2., headroom_gib=4., step_delay_ms=20., on_change=None):
        if allocator_gib <= 0 or headroom_gib < 0 or step_delay_ms < 0:
            raise ValueError("Invalid process-local GPU resource limits")
        self.shared = shared
        self.allocator_gib, self.headroom_gib = allocator_gib, headroom_gib
        self.step_delay_ms, self.last_check = step_delay_ms, float("-inf")
        self.mode, self.total_gib, self.events = None, None, []
        self.on_change = on_change

    @staticmethod
    def competitors():
        result = subprocess.run(["nvidia-smi", "--id=0", "--query-compute-apps=pid",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if any(not line.isdigit() for line in lines):
            raise RuntimeError("Cannot establish GPU process ownership; do not assume idle")
        return sorted({int(line) for line in lines if int(line) != os.getpid()})

    @staticmethod
    def available_cpu_threads():
        return max(1, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1))

    @staticmethod
    def memory():
        result = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=memory.free,memory.total",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        fields = result.stdout.strip().split(",")
        if len(fields) != 2:
            raise RuntimeError("Cannot establish GPU free-memory headroom")
        return tuple(float(value.strip()) / 1024 for value in fields)

    def configure(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no silent CPU fallback")
        self.check(force=True)
        return self.snapshot()

    def snapshot(self):
        restricted = self.mode == "shared_limited"
        return dict(mode=self.mode, policy="adaptive_sharing" if self.shared else "idle_only",
                    pid=os.getpid(), cpu_threads=2 if restricted else self.available_cpu_threads(),
                    pytorch_allocator_limit_gib=self.allocator_gib if restricted else None,
                    pytorch_allocator_fraction=self.allocator_gib / self.total_gib if restricted else 1.,
                    minimum_free_headroom_gib=self.headroom_gib if restricted else None,
                    step_delay_ms=self.step_delay_ms if restricted else 0.,
                    cpu_nice_unchanged=True, transitions=list(self.events),
                    caveat="Allocator cap excludes CUDA/driver allocations; yielding is not a GPU quota. Detection is periodic, not hardware isolation; no zero-impact guarantee.")

    def _cpu_threads(self, count):
        torch.set_num_threads(count)
        # The runner's startup BLAS defaults are conservative. Relax them too
        # when alone, without changing model hyperparameters or DataLoader data.
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            return False
        threadpool_limits(limits=count)
        return True

    def _transition(self, mode, others, free):
        restricted = mode == "shared_limited"
        if restricted:
            if self.mode is not None:
                torch.cuda.synchronize(0)
                torch.cuda.empty_cache()  # Our unoccupied cache only.
                free, _ = self.memory()
            if (free < self.allocator_gib + self.headroom_gib
                    or torch.cuda.memory_reserved(0) > self.allocator_gib * 1024**3
                    or self.allocator_gib >= self.total_gib):
                raise RuntimeError("Insufficient GPU headroom for bounded shared execution; only this experiment stops")
        torch.cuda.set_per_process_memory_fraction(self.allocator_gib / self.total_gib if restricted else 1., 0)
        cpu_threads = 2 if restricted else self.available_cpu_threads()
        blas_updated = self._cpu_threads(cpu_threads)
        self.mode = mode
        event = dict(mode=mode, other_gpu_pids=others, free_gib=free, cpu_threads=cpu_threads,
                     blas_threads_updated=blas_updated, unix_time=time.time())
        self.events.append(event)
        print(f"[GPU_MODE] {event}", flush=True)
        if self.on_change is not None:
            self.on_change(self.snapshot())

    def check(self, *, force=False):
        if not force and time.monotonic() - self.last_check < 1.:
            return
        others = self.competitors()
        if others and not self.shared:
            raise RuntimeError("Another GPU task is present; idle-only execution cannot share")
        free, total = self.memory()
        self.total_gib = total
        mode = "shared_limited" if others else "solo_unrestricted"
        if mode != self.mode:
            self._transition(mode, others, free)
        elif mode == "shared_limited" and free < self.headroom_gib:
            raise RuntimeError("GPU headroom fell below shared-run guard; only this experiment stops")
        self.last_check = time.monotonic()

    def after_step(self):
        self.check()
        if self.mode == "shared_limited":
            torch.cuda.synchronize(0)
            time.sleep(self.step_delay_ms / 1000)
