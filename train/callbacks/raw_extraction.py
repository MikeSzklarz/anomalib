import io
import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback
from PIL import Image


def _get_batch_field(obj, key):
    if isinstance(obj, dict):
        return obj.get(key, None)
    return getattr(obj, key, None)


def _normalize_like_postprocessor(raw_map: torch.Tensor, pl_module, logger=None) -> tuple[torch.Tensor, bool]:
    """Reproduce OneClassPostProcessor._normalize exactly, given a RAW anomaly map.

    RawDataExtractionCallback runs as a trainer callback, which Lightning invokes
    BEFORE model callbacks (PostProcessor is a model callback) - so the anomaly_map
    seen here is always pre-normalization, on the model-native scale. Rather than
    fighting callback ordering, this replicates the PostProcessor's own formula
    (pixel_min/pixel_max/pixel_threshold, already calibrated on the val set) so the
    result is numerically identical to what the default anomalib visualizer draws.

    Falls back to a per-image min-max normalization (the old behavior) only if the
    PostProcessor was never calibrated (e.g. normalization disabled) - callers
    should treat that case as a heatmap that is NOT comparable across images.

    Returns (normalized_tensor_in_[0,1], used_global_normalization).
    """
    post_processor = getattr(pl_module, "post_processor", None)
    if post_processor is not None:
        try:
            pixel_min = post_processor.pixel_min
            pixel_max = post_processor.pixel_max
            pixel_threshold = post_processor.pixel_threshold
            if not (pixel_min.isnan() or pixel_max.isnan()):
                threshold = pixel_threshold if not pixel_threshold.isnan() else (pixel_max + pixel_min) / 2
                normalized = ((raw_map - threshold) / (pixel_max - pixel_min)) + 0.5
                return normalized.clamp(min=0, max=1), True
        except Exception as e:  # noqa: BLE001
            if logger is not None:
                logger.warning(f"[raw_outputs] Could not apply global normalization, falling back per-image: {e}")

    # Fallback: per-image min-max (not comparable across images - only used when the
    # PostProcessor has no calibrated min/max, e.g. normalization was disabled).
    flat_min = raw_map.amin(dim=tuple(range(1, raw_map.dim())), keepdim=True)
    flat_max = raw_map.amax(dim=tuple(range(1, raw_map.dim())), keepdim=True)
    span = (flat_max - flat_min).clamp(min=1e-8)
    return ((raw_map - flat_min) / span).clamp(min=0, max=1), False


def save_npz_lzma(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Writes a standard .npz (ZIP archive of .npy entries), but using LZMA instead
    of the DEFLATE numpy defaults to - roughly 40% smaller for float anomaly maps
    with zero precision loss (see train/analysis/offline_viewer.py's benchmark notes).

    Still a completely standard .npz: plain `np.load(path)` reads it back with no
    special handling - zipfile (which np.load uses internally) already supports
    LZMA as one of the ZIP format's standard compression methods.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_LZMA) as zf:
        for key, value in arrays.items():
            buf = io.BytesIO()
            np.save(buf, value)
            zf.writestr(f"{key}.npy", buf.getvalue())


class RawDataExtractionCallback(Callback):
    """Collects per-image anomaly maps and masks during a test pass, then writes
    ONE consolidated, LZMA-compressed arrays.npz per split - instead of one PNG +
    one .npz per image - so a split's worth of raw data is a couple of files to
    download instead of thousands.

    compact=True quantizes anomaly_maps to uint8 (storing the min/max needed to
    reverse it) for ~10x smaller files at a small (~0.2% of range) precision cost -
    an explicit opt-in, not the default, since it's lossy.
    """

    def __init__(self, output_path: Path, subfolder: str = "test", logger=None, compact: bool = False):
        self.output_path = output_path
        self.subfolder = subfolder
        self.logger = logger or logging.getLogger("train_script")
        self.compact = compact

        self._image_names: list[str] = []
        self._anomaly_maps: list[np.ndarray] = []  # raw, model-native scale
        self._pred_masks: list[np.ndarray] = []
        self._gt_masks: list[np.ndarray] = []
        self._norm_params: dict[str, float] | None = None
        self._pixel_rows: list[dict] = []
        self.pixel_stats_by_split: dict[str, pd.DataFrame] = {}
        self._warned_global_norm = False

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        anomaly_map = _get_batch_field(outputs, "anomaly_map")
        if anomaly_map is None: anomaly_map = _get_batch_field(batch, "anomaly_map")

        pred_mask = _get_batch_field(outputs, "pred_mask")
        if pred_mask is None: pred_mask = _get_batch_field(batch, "pred_mask")

        if pred_mask is None and anomaly_map is not None:
            try:
                # anomaly_map is RAW (model-native scale) here — PostProcessor has not yet run.
                # Use the raw F1-optimal pixel threshold, which is on the same scale.
                thresh = None
                if hasattr(pl_module, "post_processor"):
                    t = pl_module.post_processor.pixel_threshold
                    if not t.isnan():
                        thresh = t.item()
                if thresh is not None:
                    pred_mask = (anomaly_map >= thresh).to(torch.uint8)
            except Exception:
                pass

        gt_mask = _get_batch_field(outputs, "gt_mask")
        if gt_mask is None: gt_mask = _get_batch_field(batch, "gt_mask")

        paths = _get_batch_field(outputs, "image_path")
        if paths is None: paths = _get_batch_field(batch, "image_path")

        # Reproduce the PostProcessor's own global normalization (calibrated pixel_min/
        # pixel_max/pixel_threshold from the val set) so the colorized heatmap - rendered
        # later, offline, from the stored raw values - uses the SAME scale as the default
        # anomalib visualization. Captured once per split (these are constant across the
        # whole test pass) and stored alongside the arrays so reconstruction never needs
        # the live model.
        normalized_map = None
        if anomaly_map is not None:
            normalized_map, used_global_norm = _normalize_like_postprocessor(anomaly_map, pl_module, self.logger)
            if used_global_norm and self._norm_params is None:
                try:
                    pp = pl_module.post_processor
                    self._norm_params = {
                        "pixel_min": float(pp.pixel_min.item()),
                        "pixel_max": float(pp.pixel_max.item()),
                        "pixel_threshold": float(
                            pp.pixel_threshold.item() if not pp.pixel_threshold.isnan()
                            else (pp.pixel_max.item() + pp.pixel_min.item()) / 2
                        ),
                    }
                except Exception:
                    pass
            if not used_global_norm and not self._warned_global_norm:
                self.logger.warning(
                    "[raw_outputs] PostProcessor has no calibrated pixel_min/pixel_max — "
                    "anomaly_map heatmaps are falling back to per-image min-max normalization "
                    "and are NOT comparable in severity across images."
                )
                self._warned_global_norm = True

        if paths is not None:
            for i, path in enumerate(paths):
                path_obj = Path(str(path))
                image_name = f"{path_obj.parent.name}/{path_obj.name}"
                row: dict[str, Any] = {"Image_Name": image_name}

                pred_bool = None
                gt_bool = None

                if anomaly_map is not None:
                    raw_np = anomaly_map[i].detach().cpu().numpy().squeeze().astype(np.float32)  # model-native scale
                    norm_np = normalized_map[i].detach().cpu().numpy().squeeze()  # [0, 1]
                    self._anomaly_maps.append(raw_np)
                    row["Max_Score_In_Map"] = float(norm_np.max())
                    row["Mean_Score_In_Map"] = float(norm_np.mean())
                    row["P95_Score_In_Map"] = float(np.percentile(norm_np, 95))

                if pred_mask is not None:
                    pred_bool = pred_mask[i].detach().cpu().numpy().squeeze().astype(bool)
                    self._pred_masks.append(pred_bool.astype(np.uint8))
                    row["Anomalous_Pixel_Count"] = int(pred_bool.sum())
                    row["Anomalous_Pixel_Pct"] = 100.0 * float(pred_bool.mean())

                if gt_mask is not None:
                    gt_bool = gt_mask[i].detach().cpu().numpy().squeeze() > 0.5
                    self._gt_masks.append(gt_bool.astype(np.uint8))

                if pred_bool is not None and gt_bool is not None:
                    intersection = int(np.logical_and(pred_bool, gt_bool).sum())
                    union = int(np.logical_or(pred_bool, gt_bool).sum())
                    denom = int(pred_bool.sum() + gt_bool.sum())
                    row["GT_Mask_Available"] = True
                    row["GT_Anomalous_Pixel_Pct"] = 100.0 * float(gt_bool.mean())
                    row["IoU"] = float(intersection / union) if union > 0 else float("nan")
                    row["Dice"] = float(2 * intersection / denom) if denom > 0 else float("nan")
                else:
                    row["GT_Mask_Available"] = False

                self._image_names.append(image_name)
                self._pixel_rows.append(row)

    def _write_arrays(self) -> None:
        if not self._image_names:
            return

        raw_out_dir = self.output_path / "raw_outputs" / self.subfolder
        arrays: dict[str, np.ndarray] = {"image_names": np.array(self._image_names)}

        if self._anomaly_maps:
            stack = np.stack(self._anomaly_maps).astype(np.float32)
            if self.compact:
                lo, hi = float(stack.min()), float(stack.max())
                span = (hi - lo) or 1.0
                arrays["anomaly_maps"] = np.round((stack - lo) / span * 255).astype(np.uint8)
                arrays["anomaly_maps_scale_min"] = np.float32(lo)
                arrays["anomaly_maps_scale_max"] = np.float32(hi)
            else:
                arrays["anomaly_maps"] = stack.astype(np.float16)

        if self._pred_masks:
            arrays["pred_masks"] = np.stack(self._pred_masks)

        if len(self._gt_masks) == len(self._image_names) and self._gt_masks:
            arrays["gt_masks"] = np.stack(self._gt_masks)

        if self._norm_params:
            for key, value in self._norm_params.items():
                arrays[key] = np.float32(value)

        out_path = raw_out_dir / "arrays.npz"
        save_npz_lzma(out_path, arrays)
        size_mb = out_path.stat().st_size / 1e6
        self.logger.info(
            f"[raw_outputs] Wrote {len(self._image_names)} images' arrays to {out_path} "
            f"({size_mb:.2f} MB{', compact uint8' if self.compact else ', lossless float16'})"
        )

    def on_test_end(self, trainer, pl_module):
        del trainer, pl_module
        if self._pixel_rows:
            self.pixel_stats_by_split[self.subfolder] = pd.DataFrame(self._pixel_rows)
        self._write_arrays()

        self._image_names = []
        self._anomaly_maps = []
        self._pred_masks = []
        self._gt_masks = []
        self._norm_params = None
        self._pixel_rows = []
