"""Post-training predictions on a dataset's holdout images.

If <root_dir>/holdout/ exists, every folder directly inside it (except ground_truth*)
is a holdout *group*, e.g. holdout/conflict/<class>/*.jpg or a flat holdout/undecided/*.jpg.
After each checkpoint's normal evaluation, every group is run through the trained model
and written to <checkpoint>/holdout/. Holdout images carry no labels, so nothing here is
scored against a ground-truth label and nothing is added to predictions.csv, metrics.json
or the analysis charts - those all assume real labels.

Scores are captured RAW (model-native scale), before the PostProcessor normalizes them, and
classified against the same raw validation threshold as the test split - so Anomaly_Score
here is directly comparable with predictions.csv. (Lightning re-attaches the model's own
callbacks, PostProcessor included, at the end of the callback list on every run, so a
callback added to the trainer always sees the raw batch first.)

With --task segmentation, masks are picked up from holdout/ground_truth/<group>/ (same
relative folder, <stem>_mask.<ext> or <stem>.<ext>) wherever they exist; groups can be only
partly masked. With any other task, masks are ignored.

Output (all checkpoints evaluated; arrays only for the primary, like raw_outputs/):
    <checkpoint>/holdout/predictions.csv            one row per image, every group
    <checkpoint>/holdout/summary.csv                per group x class counts and score stats
    <checkpoint>/holdout/per_image_stats.csv        map stats (+ IoU/Dice where a mask exists)
    <checkpoint>/holdout/raw_outputs/<group>/       index.csv + arrays.npz (primary only),
                                                    same format as raw_outputs/<split>/ so
                                                    analysis/offline_viewer.py reads it as-is
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, Dataset
from torchvision.tv_tensors import Mask

from anomalib.data import ImageBatch, ImageItem
from anomalib.data.utils.image import is_image_file, read_image, read_mask

from .callbacks.raw_extraction import _normalize_like_postprocessor, save_npz_lzma

HOLDOUT_DIRNAME = "holdout"
_MASK_DIR_PREFIX = "ground_truth"
_MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class HoldoutImage:
    path: Path
    group: str
    cls: str               # immediate parent folder; the group name for a flat group
    image_name: str        # "<parent>/<file>", the same convention as predictions.csv
    relative_path: str     # relative to holdout/, e.g. "conflict/scratch/T0127_003.jpg"
    mask_path: Path | None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def _mask_stem(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_mask")] if stem.endswith("_mask") else stem


def _index_masks(mask_root: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    if not mask_root.is_dir():
        return index
    for mask in mask_root.rglob("*"):
        if mask.is_file() and mask.suffix.lower() in _MASK_EXTENSIONS:
            index.setdefault(_mask_stem(mask), []).append(mask)
    return index


def _find_mask(index: Dict[str, List[Path]], mask_root: Path, relative: Path) -> Path | None:
    candidates = index.get(relative.stem, [])
    for candidate in candidates:  # same sub-folder as the image first
        if candidate.parent.relative_to(mask_root) == relative.parent:
            return candidate
    return candidates[0] if len(candidates) == 1 else None


def discover_holdout(root_dir: str | Path, use_masks: bool, logger) -> Dict[str, List[HoldoutImage]]:
    """{group: [HoldoutImage, ...]} for <root_dir>/holdout/, or {} if there is none."""
    holdout_root = Path(root_dir) / HOLDOUT_DIRNAME
    if not holdout_root.is_dir():
        return {}

    loose = [f for f in holdout_root.iterdir() if f.is_file() and is_image_file(f)]
    if loose:
        logger.warning(f"[holdout] {len(loose)} image(s) directly in {holdout_root} ignored - put them in a group folder.")

    groups: Dict[str, List[HoldoutImage]] = {}
    group_dirs = sorted(
        p for p in holdout_root.iterdir()
        if p.is_dir() and not p.name.startswith((".", _MASK_DIR_PREFIX))
    )
    for group_dir in group_dirs:
        images = sorted(
            f for f in group_dir.rglob("*")
            if f.is_file() and is_image_file(f)
            and not any(part.startswith(".") for part in f.relative_to(group_dir).parts)
        )
        if not images:
            continue
        mask_root = holdout_root / _MASK_DIR_PREFIX / group_dir.name
        mask_index = _index_masks(mask_root) if use_masks else {}
        items = []
        for image in images:
            relative = image.relative_to(group_dir)
            items.append(HoldoutImage(
                path=image,
                group=group_dir.name,
                cls=image.parent.name,
                image_name=f"{image.parent.name}/{image.name}",
                relative_path=image.relative_to(holdout_root).as_posix(),
                mask_path=_find_mask(mask_index, mask_root, relative) if mask_index else None,
            ))
        groups[group_dir.name] = items
    return groups


def describe_groups(groups: Dict[str, List[HoldoutImage]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for group, items in groups.items():
        classes: Dict[str, int] = {}
        for item in items:
            classes[item.cls] = classes.get(item.cls, 0) + 1
        out[group] = {
            "images": len(items),
            "gt_masks": sum(item.mask_path is not None for item in items),
            "classes": dict(sorted(classes.items())),
        }
    return out


def check_overlap(groups: Dict[str, List[HoldoutImage]], datamodule, logger) -> Dict[str, int]:
    """Holdout file NAMES also present in train/val/test. Full paths always differ (the
    holdout lives in its own folder), so a copied image only shows up by name."""
    names = {item.path.name for items in groups.values() for item in items}
    result: Dict[str, int] = {}
    for split in ("train", "val", "test"):
        dataset = getattr(datamodule, f"{split}_data", None)
        samples = getattr(dataset, "samples", None)
        split_names = set() if samples is None else {Path(p).name for p in samples.image_path.astype(str)}
        overlap = names & split_names
        result[split] = len(overlap)
        if overlap:
            logger.error(f"[holdout] CRITICAL: {len(overlap)} holdout file name(s) also in {split.upper()}, "
                         f"e.g. {sorted(overlap)[0]}")
        else:
            logger.info(f"[holdout] No holdout file names overlap {split.upper()}")
    return result


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class _HoldoutDataset(Dataset):
    """Mirrors AnomalibDataset.__getitem__ for unlabelled images. With masks on, an image
    without a mask gets an all-zero placeholder (collation needs one per item); it is never
    read back as ground truth - HoldoutImage.mask_path says whether a real mask exists."""

    def __init__(self, items: List[HoldoutImage], with_masks: bool, augmentations=None):
        self.items = items
        self.with_masks = with_masks
        self.augmentations = augmentations

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> ImageItem:
        item = self.items[index]
        image = read_image(item.path, as_tensor=True)
        gt_mask = None
        if self.with_masks:
            gt_mask = (read_mask(item.mask_path, as_tensor=True) if item.mask_path
                       else Mask(torch.zeros(image.shape[-2:])).to(torch.uint8))
        if self.augmentations:
            if gt_mask is None:
                image = self.augmentations(image)
            else:
                image, gt_mask = self.augmentations(image, gt_mask)
        return ImageItem(image=image, gt_mask=gt_mask, image_path=str(item.path),
                         mask_path=str(item.mask_path) if item.mask_path else "")


class _RawCapture(Callback):
    """Collects raw scores/maps in on_predict_batch_end, before the PostProcessor (a model
    callback, so later in the chain) normalizes the batch in place."""

    def __init__(self, items: List[HoldoutImage], keep_arrays: bool, logger):
        self.by_path = {str(item.path): item for item in items}
        self.keep_arrays = keep_arrays
        self.logger = logger
        self.records: List[Dict[str, Any]] = []
        self.maps: List[np.ndarray] = []
        self.pred_masks: List[np.ndarray] = []
        self.gt_masks: List[np.ndarray] = []
        self.norm_params: Dict[str, float] | None = None

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        scores = outputs.pred_score.detach().float().cpu().reshape(-1)
        anomaly_map = outputs.anomaly_map
        gt_mask = outputs.gt_mask

        pixel_threshold = None
        normalized = None
        if anomaly_map is not None:
            anomaly_map = anomaly_map.detach().float()
            pp = getattr(pl_module, "post_processor", None)
            if pp is not None and not pp.pixel_threshold.isnan():
                pixel_threshold = float(pp.pixel_threshold.item())
            normalized, used_global = _normalize_like_postprocessor(anomaly_map, pl_module, self.logger)
            if used_global and self.norm_params is None:
                self.norm_params = {
                    "pixel_min": float(pp.pixel_min.item()),
                    "pixel_max": float(pp.pixel_max.item()),
                    "pixel_threshold": float(pp.pixel_threshold.item() if not pp.pixel_threshold.isnan()
                                             else (pp.pixel_max.item() + pp.pixel_min.item()) / 2),
                }

        for i, path in enumerate(outputs.image_path):
            item = self.by_path[str(path)]
            record: Dict[str, Any] = {"item": item, "raw_score": float(scores[i])}
            pred_bool = gt_bool = None
            if anomaly_map is not None:
                raw_np = anomaly_map[i].cpu().numpy().squeeze().astype(np.float32)
                norm_np = normalized[i].cpu().numpy().squeeze()
                record["Max_Score_In_Map"] = float(norm_np.max())
                record["Mean_Score_In_Map"] = float(norm_np.mean())
                record["P95_Score_In_Map"] = float(np.percentile(norm_np, 95))
                if pixel_threshold is not None:
                    pred_bool = raw_np >= pixel_threshold
                    record["Anomalous_Pixel_Count"] = int(pred_bool.sum())
                    record["Anomalous_Pixel_Pct"] = 100.0 * float(pred_bool.mean())
                if self.keep_arrays:
                    self.maps.append(raw_np)
                    if pred_bool is not None:
                        self.pred_masks.append(pred_bool.astype(np.uint8))
            if gt_mask is not None:
                gt_bool = gt_mask[i].detach().cpu().numpy().squeeze() > 0.5
                if self.keep_arrays:
                    self.gt_masks.append(gt_bool.astype(np.uint8))
            record["GT_Mask_Available"] = item.mask_path is not None
            if item.mask_path is not None and gt_bool is not None and pred_bool is not None:
                intersection = int(np.logical_and(pred_bool, gt_bool).sum())
                union = int(np.logical_or(pred_bool, gt_bool).sum())
                denom = int(pred_bool.sum() + gt_bool.sum())
                record["GT_Anomalous_Pixel_Pct"] = 100.0 * float(gt_bool.mean())
                record["IoU"] = float(intersection / union) if union > 0 else float("nan")
                record["Dice"] = float(2 * intersection / denom) if denom > 0 else float("nan")
            self.records.append(record)


@contextmanager
def _visualizer_predict_disabled(model):
    """No images for holdout - only raw maps. Same shadowing trick as pipeline._images_disabled,
    for the predict hook."""
    visualizer = getattr(model, "visualizer", None)
    if visualizer is None:
        yield
        return
    visualizer.on_predict_batch_end = lambda *args, **kwargs: None
    try:
        yield
    finally:
        del visualizer.on_predict_batch_end


@contextmanager
def _attached(trainer, callback):
    trainer.callbacks.append(callback)
    try:
        yield
    finally:
        if callback in trainer.callbacks:
            trainer.callbacks.remove(callback)


# ---------------------------------------------------------------------------
# Run + write
# ---------------------------------------------------------------------------
def _image_threshold(model, logger) -> float:
    """Raw image threshold, exactly as RearrangeVisualizationsCallback picks it for test."""
    try:
        threshold = model.post_processor.image_threshold
        if not threshold.isnan():
            return float(threshold.item())
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[holdout] Error extracting raw threshold: {e}.")
    logger.warning("[holdout] Raw threshold is NaN - falling back to 0.5; classifications may be incorrect.")
    return 0.5


def _normalized_scores(model, raw: torch.Tensor) -> List[float]:
    pp = getattr(model, "post_processor", None)
    if pp is None:
        return [float("nan")] * len(raw)
    return pp._normalize(raw, pp.image_min.cpu(), pp.image_max.cpu(), pp.image_threshold.cpu()).tolist()


def _write_arrays(out_dir: Path, capture: _RawCapture, image_names: List[str], compact: bool, logger) -> None:
    arrays: Dict[str, np.ndarray] = {"image_names": np.array(image_names)}
    if capture.maps:
        stack = np.stack(capture.maps).astype(np.float32)
        if compact:
            lo, hi = float(stack.min()), float(stack.max())
            span = (hi - lo) or 1.0
            arrays["anomaly_maps"] = np.round((stack - lo) / span * 255).astype(np.uint8)
            arrays["anomaly_maps_scale_min"] = np.float32(lo)
            arrays["anomaly_maps_scale_max"] = np.float32(hi)
        else:
            arrays["anomaly_maps"] = stack.astype(np.float16)
    if len(capture.pred_masks) == len(image_names) and capture.pred_masks:
        arrays["pred_masks"] = np.stack(capture.pred_masks)
    if len(capture.gt_masks) == len(image_names) and capture.gt_masks:
        # Partly-masked groups: images without a mask hold zeros, flagged False here.
        arrays["gt_masks"] = np.stack(capture.gt_masks)
        arrays["gt_mask_available"] = np.array([r["item"].mask_path is not None for r in capture.records])
    for key, value in (capture.norm_params or {}).items():
        arrays[key] = np.float32(value)
    path = out_dir / "arrays.npz"
    save_npz_lzma(path, arrays)
    logger.info(f"[holdout] Wrote {len(image_names)} images' arrays to {path} "
                f"({path.stat().st_size / 1e6:.2f} MB{', compact uint8' if compact else ', lossless float16'})")


def _summary(predictions: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(
        stats[[c for c in ("Relative_Path", "IoU", "Dice") if c in stats.columns]], on="Relative_Path", how="left",
    )
    rows = []
    for group, group_df in merged.groupby("Group", sort=True):
        parts = [("ALL", group_df)] + list(group_df.groupby("Class", sort=True))
        for cls, df in parts:
            flagged = int((df["Classification"] == "anomalous").sum())
            row = {
                "Group": group, "Class": cls, "N": len(df),
                "N_Anomalous": flagged, "N_Normal": len(df) - flagged, "Flag_Rate": flagged / len(df),
                "Score_Min": df["Anomaly_Score"].min(), "Score_Median": df["Anomaly_Score"].median(),
                "Score_Mean": df["Anomaly_Score"].mean(), "Score_Max": df["Anomaly_Score"].max(),
                "Threshold": df["Threshold"].iloc[0],
            }
            if "GT_Mask_Available" in df.columns:
                row["N_GT_Masks"] = int(df["GT_Mask_Available"].sum())
                for metric in ("IoU", "Dice"):
                    if metric in df.columns:
                        row[f"Mean_{metric}"] = df[metric].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def run_holdout(
    trainer, model, groups: Dict[str, List[HoldoutImage]], out_dir: Path, *,
    save_arrays: bool, compact: bool, with_masks: bool, batch_size: int, num_workers: int,
    augmentations, logger,
) -> Dict[str, Any]:
    """Predict every holdout group with the weights currently loaded in `model` and write
    <out_dir>/{predictions,summary,per_image_stats}.csv (+ raw_outputs/<group>/ if save_arrays).
    Returns the per-group counts for manifest.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = _image_threshold(model, logger)
    prediction_frames, stats_frames = [], []
    summary: Dict[str, Any] = {"threshold": threshold, "groups": {}, "durations_sec": {}}

    with _visualizer_predict_disabled(model):
        for group, items in groups.items():
            group_masks = with_masks and any(item.mask_path is not None for item in items)
            dataset = _HoldoutDataset(items, with_masks=group_masks, augmentations=augmentations)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=ImageBatch.collate)
            capture = _RawCapture(items, keep_arrays=save_arrays, logger=logger)
            logger.info(f"[holdout] Predicting '{group}': {len(items)} images"
                        f"{f', {sum(i.mask_path is not None for i in items)} with masks' if group_masks else ''}")
            start = time.time()
            with _attached(trainer, capture):
                trainer.predict(model=model, dataloaders=loader, return_predictions=False)
            summary["durations_sec"][group] = time.time() - start

            if len(capture.records) != len(items):
                logger.warning(f"[holdout] '{group}': captured {len(capture.records)} of {len(items)} images.")
            raw = torch.tensor([r["raw_score"] for r in capture.records])
            normalized = _normalized_scores(model, raw)
            predictions = pd.DataFrame({
                "Group": group,
                "Class": [r["item"].cls for r in capture.records],
                "Image_Name": [r["item"].image_name for r in capture.records],
                "Relative_Path": [r["item"].relative_path for r in capture.records],
                "Anomaly_Score": raw.tolist(),
                "Normalized_Score": normalized,
                "Threshold": threshold,
                "Classification": ["anomalous" if s >= threshold else "normal" for s in raw.tolist()],
            })
            if with_masks:
                predictions["GT_Mask_Available"] = [r["GT_Mask_Available"] for r in capture.records]
            # Union over all images, in first-seen order: IoU/Dice exist only for masked images,
            # which need not include the first one.
            stat_keys = list(dict.fromkeys(k for r in capture.records for k in r if k not in {"item", "raw_score"}))
            stats = pd.DataFrame([
                {"Group": group, "Image_Name": r["item"].image_name, "Relative_Path": r["item"].relative_path,
                 **{k: r.get(k) for k in stat_keys}}
                for r in capture.records
            ])
            prediction_frames.append(predictions)
            stats_frames.append(stats)

            n_flagged = int((predictions["Classification"] == "anomalous").sum())
            summary["groups"][group] = {"images": len(predictions), "anomalous": n_flagged,
                                        "normal": len(predictions) - n_flagged}
            logger.info(f"[holdout] '{group}': {n_flagged}/{len(predictions)} classified anomalous "
                        f"(raw threshold {threshold:.6f})")

            if save_arrays and capture.records:
                group_dir = out_dir / "raw_outputs" / group
                group_dir.mkdir(parents=True, exist_ok=True)
                predictions.to_csv(group_dir / "index.csv", index=False)
                _write_arrays(group_dir, capture, predictions["Image_Name"].tolist(), compact, logger)

    if prediction_frames:
        all_predictions = pd.concat(prediction_frames, ignore_index=True)
        all_stats = pd.concat(stats_frames, ignore_index=True)
        all_predictions.to_csv(out_dir / "predictions.csv", index=False)
        all_stats.to_csv(out_dir / "per_image_stats.csv", index=False)
        _summary(all_predictions, all_stats).to_csv(out_dir / "summary.csv", index=False)
        logger.info(f"[holdout] Wrote predictions, summary and per-image stats to {out_dir}")
    return summary
