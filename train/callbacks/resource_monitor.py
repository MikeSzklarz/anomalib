"""Lightweight CPU/GPU/memory resource sampler, run on a background thread so it
works identically for stat-based "no-epoch" models (padim, patchcore, anomalyvfm)
and multi-epoch gradient-trained models: it samples on a wall-clock timer between
Lightning's fit()/test() start/end hooks, not on epoch boundaries.

CPU/RAM are reported as absolute, process-scoped numbers (cores used, GB used by
this job's own PID) rather than host-wide percentages - on a shared SLURM node,
psutil.cpu_percent()/virtual_memory() reflect every job on the node, not just
this one, so a host-wide "80%" is not attributable to this run. GPU memory is
process-scoped too (torch.cuda.memory_allocated()). GPU utilization/power stay
as NVML's device-wide readings, since there's no lightweight per-process
alternative - this is normally fine because SLURM typically grants exclusive
GPU access per job (no MPS sharing), so a device-wide reading isn't muddied by
other jobs the way host CPU/RAM would be.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import psutil
import torch
from lightning.pytorch.callbacks import Callback

try:
    import pynvml

    pynvml.nvmlInit()
except Exception:
    pynvml = None


class ResourceMonitorCallback(Callback):
    """Samples CPU/RAM/GPU usage every `interval_sec` while fit()/test() runs.

    `self.stage` is mutated by pipeline.py before each fit()/test() call (same
    pattern as RearrangeVisualizationsCallback.subfolder) so samples can be
    grouped by pipeline stage (fit, test, val, contamination) afterwards.
    """

    def __init__(self, output_path: Path, logger=None, interval_sec: float = 5.0):
        self.output_path = output_path
        self.logger = logger or logging.getLogger("train_script")
        self.interval_sec = interval_sec
        self.stage = "fit"

        self.samples: list[dict] = []
        self._lock = threading.Lock()
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._run_started_at: float = time.time()
        self._warned_tick_error = False
        self._process = psutil.Process(os.getpid())
        self._gpu_mem_total_mb = None
        if torch.cuda.is_available():
            try:
                self._gpu_mem_total_mb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1e6
            except Exception as e:
                self.logger.warning(f"[resource_monitor] Could not read total GPU memory: {e}")
        self._nvml_handle = None
        if pynvml is not None:
            try:
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            except Exception as e:
                self.logger.warning(f"[resource_monitor] pynvml available but device handle failed, GPU util/power will be skipped: {e}")

    def _sample(self, stage: str) -> dict:
        row = {
            "elapsed_sec": time.time() - self._run_started_at,
            "stage": stage,
            # Cores used by this job's own process (e.g. 2.3 = ~2.3 CPU cores worth
            # of work), not a host-wide percentage - meaningful even if other jobs
            # are running on the same shared node.
            "cpu_cores_used": self._process.cpu_percent() / 100.0,
            "ram_used_gb": self._process.memory_info().rss / 1e9,
        }

        if torch.cuda.is_available():
            row["gpu_mem_used_mb"] = torch.cuda.memory_allocated() / 1e6
            row["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved() / 1e6
            row["gpu_mem_total_mb"] = self._gpu_mem_total_mb

        if self._nvml_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                row["gpu_util_percent"] = util.gpu
                row["gpu_power_watts"] = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
            except Exception:
                pass

        return row

    def _loop(self, stage: str) -> None:
        self._process.cpu_percent()  # prime — first real call needs a baseline
        while not self._stop_event.wait(self.interval_sec):
            try:
                row = self._sample(stage)
                with self._lock:
                    self.samples.append(row)
            except Exception as e:
                if not self._warned_tick_error:
                    self.logger.warning(f"[resource_monitor] Sample failed (further errors suppressed): {e}")
                    self._warned_tick_error = True

    def _start(self) -> None:
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, args=(self.stage,), daemon=True)
        self._thread.start()

    def _stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval_sec)

    def on_fit_start(self, trainer, pl_module):
        self._start()

    def on_fit_end(self, trainer, pl_module):
        self._stop()

    def on_test_start(self, trainer, pl_module):
        self._start()

    def on_test_end(self, trainer, pl_module):
        self._stop()

    def on_exception(self, trainer, pl_module, exception):
        self._stop()
