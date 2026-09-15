from lightning.pytorch.callbacks import Callback


class FileLoggingCallback(Callback):
    """Logs metrics to the python logger at the end of every epoch."""

    def __init__(self, logger):
        self.logger = logger
        self.epoch_rows: list[dict] = []

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        log_parts = [f"Epoch {epoch}"]
        row = {"epoch": epoch}
        for name, value in metrics.items():
            if isinstance(value, float) or hasattr(value, "item"):
                float_value = float(value)
                log_parts.append(f"{name}: {float_value:.4f}")
                row[name] = float_value
        self.logger.info(" | ".join(log_parts))
        self.epoch_rows.append(row)
