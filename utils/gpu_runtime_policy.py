"""Process-local resource controls. Never signal or reconfigure another job."""
from __future__ import annotations

import os
import subprocess
import time
import torch


class GPURuntimePolicy:
    def __init__(self, *, shared=False, allocator_gib=2., headroom_gib=4., step_delay_ms=20.):
        if allocator_gib <= 0 or headroom_gib < 0 or step_delay_ms < 0:
            raise ValueError("Invalid process-local GPU resource limits")
        self.shared = shared
        self.allocator_gib, self.headroom_gib = allocator_gib, headroom_gib
        self.step_delay_ms, self.last_check = step_delay_ms, float("-inf")

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
        if not self.shared:
            return dict(mode="idle_only", cpu_threads=2)
        free, total = self.memory()
        if free < self.allocator_gib + self.headroom_gib or self.allocator_gib >= total:
            raise RuntimeError("Insufficient GPU headroom for bounded shared execution")
        torch.cuda.set_per_process_memory_fraction(self.allocator_gib / total, 0)
        # This affects this process only; it does not reserve CPU cores.
        if hasattr(os, "nice"):
            os.nice(10)
        self.last_check = float("-inf")
        return dict(mode="shared_explicit_user_authorization", pid=os.getpid(), cpu_threads=2,
                    pytorch_allocator_limit_gib=self.allocator_gib, minimum_free_headroom_gib=self.headroom_gib,
                    step_delay_ms=self.step_delay_ms, initial_free_gib=free, total_gib=total,
                    caveat="Allocator cap excludes CUDA/driver allocations; delay is not a GPU percentage quota. Other jobs may slow down or OOM; no zero-impact guarantee.")

    def check(self):
        if not self.shared:
            from scripts.train_kelmarsh_hourly import gpu_available_to_this_process
            gpu_available_to_this_process()
        elif time.monotonic() - self.last_check >= 1.:
            free, _ = self.memory()
            self.last_check = time.monotonic()
            if free < self.headroom_gib:
                raise RuntimeError("GPU headroom fell below shared-run guard; only this experiment stops")

    def after_step(self):
        if self.shared:
            torch.cuda.synchronize(0)
            self.check()
            time.sleep(self.step_delay_ms / 1000)
