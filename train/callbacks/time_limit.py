"""Wall-clock cap on training (--max_time), so a run that never early-stops still
reaches test/val/contamination/export before SLURM kills the job and nothing gets
written at all.

Lightning's own Trainer(max_time=...) is not enough here: its stop request is
ignored until min_epochs is met, so it can't stop the fixed-length configs
(min_epochs == max_epochs) or a run where even min_epochs takes longer than the
limit. This callback lifts the minimums when the limit is hit, then stops.
"""

from __future__ import annotations

import time
from datetime import timedelta

from lightning.pytorch.callbacks import Callback


def parse_max_time(value: str) -> timedelta:
    """'HH:MM:SS' or 'D:HH:MM:SS' (SLURM --time style) -> timedelta."""
    parts = value.split(":")
    if len(parts) not in (3, 4) or not all(p.isdigit() for p in parts):
        msg = f"--max_time must look like HH:MM:SS or D:HH:MM:SS, got {value!r}"
        raise ValueError(msg)
    nums = [int(p) for p in parts]
    days = nums[0] if len(nums) == 4 else 0
    hours, minutes, seconds = nums[-3:]
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


class TrainingTimeLimitCallback(Callback):
    """Stops fit() once `limit` of training wall-clock time has passed, checked after
    every training batch. Lightning then runs its usual end-of-training validation,
    so the best-epoch checkpoint still gets its chance to update before fit returns."""

    def __init__(self, limit: timedelta, logger) -> None:
        self.limit = limit
        self.logger = logger
        self.fired = False
        self._started_at: float | None = None

    def on_fit_start(self, trainer, pl_module) -> None:
        self._started_at = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.fired or self._started_at is None:
            return
        if time.monotonic() - self._started_at < self.limit.total_seconds():
            return
        self.fired = True
        # min_epochs/min_steps would otherwise make Lightning ignore should_stop.
        trainer.fit_loop.min_epochs = 0
        trainer.fit_loop.epoch_loop.min_steps = 0
        trainer.should_stop = True
        self.logger.warning(
            f"[max_time] Training time limit {self.limit} reached during epoch {trainer.current_epoch} - "
            f"stopping training now; the best epoch so far will be restored and evaluated."
        )
