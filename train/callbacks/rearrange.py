import json
import logging
import shutil
from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback


class RearrangeVisualizationsCallback(Callback):
    def __init__(self, output_path: Path, subfolder: str = "test", logger=None, model_name: str = "model"):
        self.output_path = output_path
        self.subfolder = subfolder
        self.logger = logger or logging.getLogger("train_script")
        self.model_name = model_name
        self.preds_stats = []
        self.predictions_by_split: dict[str, pd.DataFrame] = {}
        self.thresholds_by_split: dict[str, float] = {}

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        def get_item(obj, key):
            if isinstance(obj, dict): return obj.get(key, None)
            return getattr(obj, key, None)

        gt = get_item(outputs, "gt_label")
        if gt is None: gt = get_item(batch, "gt_label")

        pred_score = get_item(outputs, "pred_score")
        if pred_score is None: pred_score = get_item(batch, "pred_score")

        paths = get_item(outputs, "image_path")
        if paths is None: paths = get_item(batch, "image_path")

        if gt is not None and pred_score is not None and paths is not None:
            gt = gt.cpu().squeeze()
            score = pred_score.cpu().squeeze()

            if gt.ndim == 0: gt = gt.unsqueeze(0)
            if score.ndim == 0: score = score.unsqueeze(0)

            path_list = [str(p) for p in paths]
            self.preds_stats.append((gt, score, path_list))

    def on_test_end(self, trainer, pl_module):
        if not self.preds_stats:
            self.logger.warning(f"No predictions collected. Skipping rearrangement for {self.subfolder}.")
            return

        all_gt = torch.cat([x[0] for x in self.preds_stats])
        all_scores = torch.cat([x[1] for x in self.preds_stats])
        all_paths = []
        for x in self.preds_stats: all_paths.extend(x[2])

        # IMPORTANT: pred_score captured in on_test_batch_end is RAW (model-native scale).
        # Reason: PostProcessor is a model callback and is appended AFTER trainer callbacks
        # in the Lightning callback chain, so it normalizes AFTER we collect scores here.
        # We must compare raw scores against the raw F1-optimal threshold stored in the model.
        selected_thresh = float("nan")
        try:
            if hasattr(pl_module, "post_processor"):
                pp = pl_module.post_processor
                thresh_tensor = pp.image_threshold  # raw F1-optimal threshold
                if not thresh_tensor.isnan():
                    selected_thresh = float(thresh_tensor.item())
        except Exception as e:
            self.logger.warning(f"[{self.subfolder.upper()}] Error extracting raw threshold: {e}.")

        if selected_thresh != selected_thresh:  # isnan check without import
            selected_thresh = 0.5
            self.logger.warning(
                f"[{self.subfolder.upper()}] Raw threshold is NaN (no anomalous val samples?). "
                f"Falling back to 0.5 — classifications may be incorrect."
            )

        self.logger.info(
            f" [{self.subfolder.upper()}] Using raw F1-adaptive threshold: {selected_thresh:.6f} "
            f"(model-native score scale)"
        )

        pred_labels = (all_scores >= selected_thresh).long()

        is_anom_gt = (all_gt == 1)
        is_norm_gt = (all_gt == 0)

        tp = torch.logical_and(is_anom_gt, (pred_labels == 1)).sum().item()
        fn = torch.logical_and(is_anom_gt, (pred_labels == 0)).sum().item()
        fp = torch.logical_and(is_norm_gt, (pred_labels == 1)).sum().item()
        tn = torch.logical_and(is_norm_gt, (pred_labels == 0)).sum().item()

        f1_denom = 2 * tp + fp + fn
        target_f1 = (2 * tp / f1_denom) if f1_denom > 0 else 0.0

        stats_msg = (
            f"\n FINAL CLASSIFICATION STATS ({self.subfolder.upper()}) \n"
            f" Threshold : {selected_thresh:.4f}\n"
            f" TP: {tp:<5} | FN: {fn}\n"
            f" TN: {tn:<5} | FP: {fp}\n"
            f" F1: {target_f1:.4f}\n"
        )
        self.logger.info(stats_msg)

        stats_data = {
            "custom_threshold": float(selected_thresh),
            "custom_F1_score": float(target_f1),
            "TP": int(tp), "FN": int(fn), "TN": int(tn), "FP": int(fp),
            "Total_Anomalous": int(tp + fn), "Total_Normal": int(tn + fp)
        }

        temp_stats_path = self.output_path / f".tmp_{self.model_name}_custom_stats_{self.subfolder}.json"
        try:
            with open(temp_stats_path, "w") as f:
                json.dump(stats_data, f)
        except Exception as e:
            self.logger.error(f"Failed to stage custom stats: {e}")

        csv_image_names = []
        csv_scores = []
        csv_thresholds = []
        csv_class = []
        csv_gt_label = []
        csv_defect_type = []

        for i, original_path in enumerate(all_paths):
            gt_val = all_gt[i].item()
            pred_val = pred_labels[i].item()
            score_val = all_scores[i].item()

            if gt_val == 1 and pred_val == 1: sub_cat = "TP"
            elif gt_val == 1 and pred_val == 0: sub_cat = "FN"
            elif gt_val == 0 and pred_val == 1: sub_cat = "FP"
            else: sub_cat = "TN"

            path_obj = Path(original_path)
            short_name = f"{path_obj.parent.name}/{path_obj.name}"

            csv_image_names.append(short_name)
            csv_scores.append(score_val)
            csv_thresholds.append(selected_thresh)
            csv_class.append(sub_cat)
            csv_gt_label.append(int(gt_val))
            csv_defect_type.append(path_obj.parent.name)

        try:
            df = pd.DataFrame({
                "Image_Name": csv_image_names,
                "Anomaly_Score": csv_scores,
                "Threshold": csv_thresholds,
                "Classification": csv_class,
                "GT_Label": csv_gt_label,
                "Defect_Type": csv_defect_type,
            })
            # Kept in memory (not written to disk per-split) so pipeline.py can build the
            # single consolidated predictions.csv across whichever splits actually ran,
            # instead of leaving separate test/val/contamination CSV files that all
            # duplicate the same columns.
            self.predictions_by_split[self.subfolder] = df
            self.thresholds_by_split[self.subfolder] = float(selected_thresh)
            self.logger.info(f"[{self.subfolder.upper()}] Collected {len(df)} predictions for consolidated output.")
        except Exception as e:
            self.logger.error(f"[{self.subfolder.upper()}] Failed to build predictions dataframe: {e}")

        # Image Rearrangement
        base_search_dir = Path(trainer.default_root_dir)
        sample_file_name = Path(all_paths[0]).name
        found_files = list(base_search_dir.rglob(sample_file_name))
        vis_candidates = [f for f in found_files if "images" in str(f.parent) and "results" in str(f)]

        if not vis_candidates:
            vis_candidates = [f for f in found_files if "datasets" not in str(f)]

        if not vis_candidates:
            self.logger.warning(f"Could not locate visualization for {sample_file_name} in {base_search_dir}")
            return

        sample_path = vis_candidates[0]
        images_root = None
        for parent in sample_path.parents:
            if parent.name == "images":
                images_root = parent
                break
        if images_root is None:
            if sample_path.parent.name == "images":
                images_root = sample_path.parent
            else:
                images_root = sample_path.parent.parent

        moved_count = 0
        ops = []

        for i, original_path in enumerate(all_paths):
            gt_val = all_gt[i].item()
            pred_val = pred_labels[i].item()

            main_cat = "anomalous" if gt_val == 1 else "normal"
            if gt_val == 1 and pred_val == 1: sub_cat = "TP"
            elif gt_val == 1 and pred_val == 0: sub_cat = "FN"
            elif gt_val == 0 and pred_val == 1: sub_cat = "FP"
            else: sub_cat = "TN"

            dest_folder = images_root / self.subfolder / main_cat / sub_cat
            fname = Path(original_path).name

            potential_paths = [
                images_root / fname,
                images_root / Path(original_path).parent.name / fname,
                images_root / Path(original_path).parent.parent.name / Path(original_path).parent.name / fname,
            ]

            source_file = None
            for p in potential_paths:
                if p.exists():
                    source_file = p
                    break

            if not source_file:
                found = list(images_root.rglob(fname))
                candidates = [f for f in found if f.parent.name not in {"TP", "FN", "FP", "TN"}]
                if candidates:
                    source_file = candidates[0]

            if source_file and source_file.exists():
                folder_prefix = Path(original_path).parent.name
                new_name = f"{folder_prefix}_{source_file.stem}_{sub_cat}{source_file.suffix}"
                ops.append((source_file, dest_folder / new_name))

        for src, dst in ops:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                moved_count += 1
            except Exception as e:
                pass

        def _remove_empty_dirs(path: Path) -> None:
            for child in path.iterdir():
                if child.is_dir():
                    _remove_empty_dirs(child)
            try:
                path.rmdir()
            except OSError:
                pass

        split_dirs = {"test", "val", "contamination"}
        label_dirs = {"normal", "anomalous"}

        for item in images_root.iterdir():
            if not item.is_dir():
                continue
            if item.name in split_dirs:
                for sub in item.iterdir():
                    if sub.is_dir() and sub.name not in label_dirs:
                        _remove_empty_dirs(sub)
            elif item.name not in label_dirs:
                _remove_empty_dirs(item)

        self.logger.info(f"Reorganization complete. Updated {moved_count} images in '{self.subfolder}'.")
        self.preds_stats = []
