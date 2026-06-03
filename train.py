import argparse
import logging
import sys
import warnings
import yaml
import json
import os
import types
import pandas as pd
import numpy as np
import cv2

from pathlib import Path
from typing import Dict, Any, Type, Set
from PIL import Image

import torch
from torchvision.transforms import v2
import shutil

from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch import seed_everything

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")

# Anomalib Imports
from anomalib.engine import Engine
from anomalib.deploy import ExportType
from anomalib.loggers import AnomalibTensorBoardLogger
from anomalib.metrics import Evaluator, AUROC, F1Score, F1Max, AUPR
from anomalib.data.utils.split import ValSplitMode, TestSplitMode

from anomalib.visualization.image.item_visualizer import DEFAULT_TEXT_CONFIG
DEFAULT_TEXT_CONFIG["enable"] = False

from anomalib.data import (
    MVTecAD, MVTecLOCO, MVTecAD2, MVTec3D, 
    BTech, Visa, Folder, Kolektor, 
    Avenue, ShanghaiTech, UCSDped
)

from anomalib.models import (
    AnomalyDINO, AnomalyVFM, CFM, Cfa, Cflow, Csflow, Dfkde, Dfm,
    Dinomaly, Draem, Dsr, EfficientAd, Fastflow,
    Fre, Ganomaly, GeneralAD, Glass, InpFormer, L2BT, Padim, Patchcore, Patchflow,
    ReverseDistillation, Stfpm, Supersimplenet, Uflow, UniNet
)

# -----------------------------------------------------------------------------
# 1. Mappings
# -----------------------------------------------------------------------------

MODEL_MAP: Dict[str, Type] = {
    "anomalydino": AnomalyDINO,
    "anomalyvfm": AnomalyVFM,
    "cfm": CFM,
    "cfa": Cfa,
    "cflow": Cflow,
    "csflow": Csflow,
    "dfkde": Dfkde,
    "dfm": Dfm,
    "dinomaly": Dinomaly,
    "draem": Draem,
    "dsr": Dsr,
    "efficientad": EfficientAd,
    "fastflow": Fastflow,
    "fre": Fre,
    "ganomaly": Ganomaly,
    "generalad": GeneralAD,
    "glass": Glass,
    "inpformer": InpFormer,
    "l2bt": L2BT,
    "padim": Padim,
    "patchcore": Patchcore,
    "patchflow": Patchflow,
    "reversedistillation": ReverseDistillation,
    "stfpm": Stfpm,
    "supersimplenet": Supersimplenet,
    "uflow": Uflow,
    "uninet": UniNet
}

DATASET_MAP: Dict[str, Type] = {
    "mvtecad": MVTecAD,
    "mvtecloco": MVTecLOCO,
    "mvtecad2": MVTecAD2,
    "mvtec3d": MVTec3D,
    "btech": BTech,
    "visa": Visa,
    "kolektor": Kolektor,
    "folder": Folder,
    "avenue": Avenue,
    "shanghaitech": ShanghaiTech,
    "ucsdped": UCSDped
}

# -----------------------------------------------------------------------------
# 2. Logger Setup & Utils
# -----------------------------------------------------------------------------

def setup_logger(output_dir: Path, model_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"{model_name}_training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True
    )
    return logging.getLogger("train_script")

def log_dataset_details(datamodule, logger, export_paths=False, output_path=None):
    """Logs detailed stats and checks for TRUE data leakage using full paths."""
    logger.info("Setting up datamodule to inspect splits...")
    
    # This triggers datamodule._setup() (which we intercept), then it performs splits
    datamodule.setup()

    def get_info(dataset):
        if not dataset or not hasattr(dataset, 'samples'):
            return 0, 0, 0, set()
        
        samples = dataset.samples
        n_total = len(samples)
        
        if 'label_index' in samples:
            n_normal = (samples.label_index == 0).sum()
            n_anom = (samples.label_index == 1).sum()
        else:
            n_normal = n_total
            n_anom = 0
            
        filepaths = set(samples.image_path.astype(str).tolist())
        return n_total, n_normal, n_anom, filepaths

    n_train, norm_train, anom_train, files_train = get_info(getattr(datamodule, "train_data", None))
    n_val, norm_val, anom_val, files_val = get_info(getattr(datamodule, "val_data", None))
    n_test, norm_test, anom_test, files_test = get_info(getattr(datamodule, "test_data", None))

    logger.info("=== Dataset Split Statistics ===")
    logger.info(f"  [TRAIN] Total: {n_train} | Normal: {norm_train} | Anomalous: {anom_train}")
    logger.info(f"[VAL  ] Total: {n_val} | Normal: {norm_val} | Anomalous: {anom_val}")
    logger.info(f"  [TEST ] Total: {n_test} | Normal: {norm_test} | Anomalous: {anom_test}")

    if export_paths:
        def print_samples(name, files):
            logger.info(f"    Sample Files ({name}):")
            short_paths = sorted([f"{Path(f).parent.name}/{Path(f).name}" for f in files])
            for p in short_paths[:5]:
                logger.info(f"      - .../{p}")
            if len(short_paths) > 5: logger.info(f"      ... ({len(short_paths)-5} more)")
        
        print_samples("VAL", files_val)
        print_samples("TEST", files_test)

        if output_path is not None:
            csv_data = []
            splits =[("Train", files_train), ("Validation", files_val), ("Test", files_test)]
            
            for split_name, files in splits:
                for f in files:
                    path_obj = Path(f)
                    short_name = f"{path_obj.parent.name}/{path_obj.name}"
                    csv_data.append({"Image_Name": short_name, "Split": split_name})
            
            try:
                df = pd.DataFrame(csv_data)
                csv_out_path = Path(output_path) / "dataset_splits.csv"
                df.to_csv(csv_out_path, index=False)
                logger.info(f"=== Exported detailed dataset splits CSV to: {csv_out_path} ===")
            except Exception as e:
                logger.error(f"Failed to export dataset splits CSV: {e}")
    
    def check_overlap(set_a, set_b, name_a, name_b):
        overlap = set_a.intersection(set_b)
        if overlap:
            logger.error(f"   CRITICAL: {len(overlap)} images overlap between {name_a} and {name_b}!")
            logger.error(f"    Example overlap: {list(overlap)[0]}")
        else:
            logger.info(f"      No overlap between {name_a} and {name_b}")

    check_overlap(files_train, files_val, "TRAIN", "VAL")
    check_overlap(files_train, files_test, "TRAIN", "TEST")
    check_overlap(files_val, files_test, "VAL", "TEST")
    logger.info("================================")
    
def _get_batch_field(obj, key):
    if isinstance(obj, dict):
        return obj.get(key, None)
    return getattr(obj, key, None)

class FileLoggingCallback(Callback):
    """Logs metrics to the python logger at the end of every epoch."""
    def __init__(self, logger):
        self.logger = logger

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        log_parts =[f"Epoch {epoch}"]
        for name, value in metrics.items():
            if isinstance(value, float) or hasattr(value, 'item'):
                log_parts.append(f"{name}: {float(value):.4f}")
        self.logger.info(" | ".join(log_parts))    
        
class RearrangeVisualizationsCallback(Callback):
    def __init__(self, output_path: Path, subfolder: str = "test", logger=None, model_name: str = "model"):
        self.output_path = output_path
        self.subfolder = subfolder
        self.logger = logger or logging.getLogger("train_script")
        self.model_name = model_name
        self.preds_stats =[]

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
        all_paths =[]
        for x in self.preds_stats: all_paths.extend(x[2])

        selected_thresh = 0.5
        try:
            if hasattr(pl_module, "post_processor") and hasattr(pl_module.post_processor, "image_threshold"):
                thresh = pl_module.post_processor.image_threshold
                if not torch.isnan(thresh):
                    selected_thresh = float(thresh.item())
        except Exception as e:
            self.logger.warning(f"[{self.subfolder.upper()}] Error extracting threshold: {e}. Using default 0.5")

        self.logger.info(f" [{self.subfolder.upper()}] Using model's internal adaptive threshold: {selected_thresh:.4f}")

        pred_labels = (all_scores >= selected_thresh).long()

        target_f1 = 0.0
        if "image_F1Max" in trainer.callback_metrics:
            target_f1 = trainer.callback_metrics["image_F1Max"].item()
        elif "F1Score" in trainer.callback_metrics:
            target_f1 = trainer.callback_metrics["F1Score"].item()

        is_anom_gt = (all_gt == 1)
        is_norm_gt = (all_gt == 0)

        tp = torch.logical_and(is_anom_gt, (pred_labels == 1)).sum().item()
        fn = torch.logical_and(is_anom_gt, (pred_labels == 0)).sum().item()
        fp = torch.logical_and(is_norm_gt, (pred_labels == 1)).sum().item()
        tn = torch.logical_and(is_norm_gt, (pred_labels == 0)).sum().item()

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

        csv_image_names =[]
        csv_scores =[]
        csv_thresholds =[]
        csv_class =[]

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

        try:
            df = pd.DataFrame({
                "Image_Name": csv_image_names,
                "Anomaly_Score": csv_scores,
                "Threshold": csv_thresholds,
                "Classification": csv_class
            })

            csv_out_path = self.output_path / f"{self.model_name}_{self.subfolder}_predictions.csv"
            df.to_csv(csv_out_path, index=False)
            self.logger.info(f"[{self.subfolder.upper()}] Exported detailed predictions CSV to: {csv_out_path}")
        except Exception as e:
            self.logger.error(f"[{self.subfolder.upper()}] Failed to export predictions CSV: {e}")

        # Image Rearrangement
        base_search_dir = Path(trainer.default_root_dir)
        sample_file_name = Path(all_paths[0]).name
        found_files = list(base_search_dir.rglob(sample_file_name))
        vis_candidates =[f for f in found_files if "images" in str(f.parent) and "results" in str(f)]

        if not vis_candidates:
            vis_candidates =[f for f in found_files if "datasets" not in str(f)]

        if not vis_candidates:
            self.logger.warning(f"Could not locate visualization for {sample_file_name} in {base_search_dir}")
            return

        sample_path = vis_candidates[0]
        if sample_path.parent.name == "images":
            images_root = sample_path.parent
        else:
            images_root = sample_path.parent.parent

        moved_count = 0
        ops =[]

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

            potential_paths =[
                images_root / fname,
                images_root / Path(original_path).parent.name / fname
            ]

            source_file = None
            for p in potential_paths:
                if p.exists():
                    source_file = p
                    break

            if not source_file:
                found = list(images_root.rglob(fname))
                candidates =[f for f in found if sub_cat not in str(f.parent) and self.subfolder not in str(f.parent)]
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

        for item in images_root.iterdir():
            if item.is_dir() and item.name not in["normal", "anomalous", "test", "val", "contamination"]:
                try: item.rmdir()
                except OSError: pass

        self.logger.info(f"Reorganization complete. Updated {moved_count} images in '{self.subfolder}'.")
        self.preds_stats =[]
        
class RawDataExtractionCallback(Callback):
    def __init__(self, output_path: Path, subfolder: str = "test", logger=None):
        self.output_path = output_path
        self.subfolder = subfolder
        self.logger = logger or logging.getLogger("train_script")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        anomaly_map = _get_batch_field(outputs, "anomaly_map")
        if anomaly_map is None: anomaly_map = _get_batch_field(batch, "anomaly_map")

        pred_mask = _get_batch_field(outputs, "pred_mask")
        if pred_mask is None: pred_mask = _get_batch_field(batch, "pred_mask")

        if pred_mask is None and anomaly_map is not None:
            try:
                if hasattr(pl_module, "post_processor") and hasattr(pl_module.post_processor, "pixel_threshold"):
                    thresh = pl_module.post_processor.pixel_threshold
                    if not torch.isnan(thresh):
                        pred_mask = (anomaly_map >= thresh.item()).to(torch.uint8)
            except Exception:
                pass

        gt_mask = _get_batch_field(outputs, "gt_mask")
        if gt_mask is None: gt_mask = _get_batch_field(batch, "gt_mask")

        paths = _get_batch_field(outputs, "image_path")
        if paths is None: paths = _get_batch_field(batch, "image_path")

        raw_out_dir = self.output_path / "raw_outputs" / self.subfolder
        amap_dir = raw_out_dir / "anomaly_maps"
        pmask_dir = raw_out_dir / "pred_masks"
        gmask_dir = raw_out_dir / "gt_masks"

        if paths is not None:
            for i, path in enumerate(paths):
                path_obj = Path(str(path))
                save_name = f"{path_obj.parent.name}_{path_obj.stem}.png"

                if anomaly_map is not None:
                    amap_dir.mkdir(parents=True, exist_ok=True)
                    amap_np = anomaly_map[i].cpu().numpy()

                    if not (amap_np.min() >= 0.0 and amap_np.max() <= 1.0):
                        a_min, a_max = amap_np.min(), amap_np.max()
                        amap_np = (amap_np - a_min) / (a_max - a_min) if a_max > a_min else np.zeros_like(amap_np)

                    amap_uint8 = (amap_np * 255).astype(np.uint8)
                    heatmap = cv2.applyColorMap(amap_uint8, cv2.COLORMAP_JET)
                    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    Image.fromarray(heatmap_rgb).save(amap_dir / save_name)
                    
                if pred_mask is not None:
                    pmask_dir.mkdir(parents=True, exist_ok=True)
                    mask_arr = (pred_mask[i].cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(mask_arr, mode='L').save(pmask_dir / save_name)
                    
                if gt_mask is not None:
                    gmask_dir.mkdir(parents=True, exist_ok=True)
                    mask_arr = (gt_mask[i].cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(mask_arr, mode='L').save(gmask_dir / save_name)

# -----------------------------------------------------------------------------
# 3. Helpers
# -----------------------------------------------------------------------------

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_init_args(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key in config and "init_args" in config[key]:
        return config[key]["init_args"]
    return {}

# -----------------------------------------------------------------------------
# 4. Main Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # Basics
    parser.add_argument("--model", type=str, required=True, choices=MODEL_MAP.keys())
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_MAP.keys())
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--config_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results")
    
    # Data params
    parser.add_argument("--root_dir", type=str, default="./datasets")
    parser.add_argument("--category", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--exclude_root", type=str, default=None, help="Directory containing CSV/TXT files of images to exclude per model.")
    
    # Training params
    parser.add_argument("--max_epochs", type=int, default=999)
    parser.add_argument("--min_epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--task", type=str, default="segmentation", choices=["classification", "segmentation", "detection"])
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    
    # Export & Saving
    parser.add_argument("--export_types", nargs="+", default=[], choices=["torch", "openvino", "onnx"])
    parser.add_argument("--no_checkpoint", action="store_true")
    parser.add_argument("--image_size", type=int, nargs="+", default=None)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--export_paths", action="store_true")
    parser.add_argument("--check_contamination", action="store_true")
    parser.add_argument("--eval_val_test", action="store_true", 
                        help="Run a secondary test pass over the Validation set and merge the CSV outputs.")

    args = parser.parse_args()
    
    if args.seed is not None:
        seed_everything(args.seed, workers=True)
    
    output_path = Path(args.output_dir) / args.model / args.dataset / args.category
    logger = setup_logger(output_path, args.model)
    logger.info(f"Experiment Args: {vars(args)}")

    config_path = args.config
    if config_path is None:
        target_folder = Path(args.config_root) if args.config_root else Path(f"./{args.dataset}_configs")
        auto_path = target_folder / f"{args.model}.yaml"
        if auto_path.exists():
            config_path = str(auto_path)

    yaml_config = load_yaml_config(config_path) if config_path else {}

    # Extract dynamic exclusion lists matching the active model
    exclude_files = set()
    if args.exclude_root:
        exclude_dir = Path(args.exclude_root)
        csv_path = exclude_dir / f"{args.model}.csv"
        txt_path = exclude_dir / f"{args.model}.txt"
        
        target_path = None
        if csv_path.exists(): target_path = csv_path
        elif txt_path.exists(): target_path = txt_path
        
        if target_path:
            logger.info(f"Loading exclusion list from: {target_path}")
            if target_path.suffix == ".csv":
                try:
                    df_ex = pd.read_csv(target_path)
                    if 'Filename' in df_ex.columns:
                        exclude_files = set(df_ex['Filename'].astype(str).str.strip().tolist())
                    else:
                        df_ex = pd.read_csv(target_path, header=None)
                        exclude_files = set(df_ex.iloc[:, 0].astype(str).str.strip().tolist())
                except pd.errors.EmptyDataError:
                    logger.warning(f"Exclusion file {target_path} is empty.")
                except Exception as e:
                    logger.error(f"Failed to read exclusion CSV: {e}")
            else:
                try:
                    with open(target_path, "r") as f:
                        exclude_files = set([line.strip() for line in f.readlines() if line.strip()])
                except Exception as e:
                    logger.error(f"Failed to read exclusion TXT: {e}")
        else:
            logger.warning(f"Exclude root '{args.exclude_root}' provided but no exclusion file found for model '{args.model}'")

    # -------------------------------------------------------------------------
    # Dataset Initialization
    # -------------------------------------------------------------------------
    logger.info(f"Initializing DataModule: {args.dataset}")
    DataClass = DATASET_MAP[args.dataset]
    dataset_kwargs = get_init_args(yaml_config, "data")
    
    dataset_kwargs.update({
        "root": args.root_dir,
        "train_batch_size": args.batch_size,
        "eval_batch_size": args.batch_size,
        "seed": args.seed,
    })

    if args.model == "efficientad" and dataset_kwargs["train_batch_size"] != 1:
        dataset_kwargs["train_batch_size"] = 1
    
    import inspect
    valid_args = inspect.signature(DataClass.__init__).parameters
    
    if "category" in valid_args:
        dataset_kwargs["category"] = args.category
    elif "category" in dataset_kwargs:
        dataset_kwargs.pop("category")
            
    if args.dataset == "kolektor":
        dataset_kwargs["val_split_mode"] = "from_test"
        dataset_kwargs["val_split_ratio"] = 0.5 

    if args.dataset == "Folder":
        dataset_kwargs["name"] = args.category
        
    if args.dataset == "folder":
        dataset_kwargs["name"] = args.category if args.category else "custom_folder"
        root_p = Path(args.root_dir)
        
        if "normal_dir" not in dataset_kwargs:
            if (root_p / "train" / "good").exists(): dataset_kwargs["normal_dir"] = "train/good"
            elif (root_p / "good").exists(): dataset_kwargs["normal_dir"] = "good"
            else: dataset_kwargs["normal_dir"] = "train/good"
        
        if "abnormal_dir" not in dataset_kwargs:
            if (root_p / "test").exists(): dataset_kwargs["abnormal_dir"] = "test"
            elif (root_p / "defect").exists(): dataset_kwargs["abnormal_dir"] = "defect"

    filtered_kwargs = {k: v for k, v in dataset_kwargs.items() if k in valid_args}

    try:
        datamodule = DataClass(**filtered_kwargs)

        # ==============================================================================
        # INJECT PRE-SPLIT FILTERING 
        # Here we Monkey-Patch the _setup method before datamodule.setup() is called
        # so that files are removed from the DataFrames BEFORE they are mathematically 
        # split into val and test pools.
        # ==============================================================================
        if exclude_files:
            original_setup = datamodule._setup
            
            def custom_setup(self_dm, stage=None):
                # 1. First, call the original _setup to load the physical files into memory
                original_setup(stage)
                
                logger.info("Applying exclusion filter BEFORE dataset splits...")
                
                # 2. Safely filter the Pandas DataFrames in memory
                def filter_samples(dataset, subset_name):
                    if not dataset or not hasattr(dataset, "samples"):
                        return 0
                        
                    orig_len = len(dataset.samples)
                    if orig_len == 0:
                        return 0
                        
                    keep_mask =[]
                    for img_path in dataset.samples["image_path"]:
                        path_obj = Path(img_path)
                        expected_fp_name = f"{path_obj.parent.name}_{path_obj.stem}_FP{path_obj.suffix}"
                        exact_name = path_obj.name
                        
                        if expected_fp_name in exclude_files or exact_name in exclude_files or str(path_obj) in exclude_files:
                            keep_mask.append(False)
                        else:
                            keep_mask.append(True)
                    
                    if not all(keep_mask):
                        dataset.samples = dataset.samples[keep_mask].reset_index(drop=True)
                        return orig_len - len(dataset.samples)
                    return 0

                r_train = filter_samples(getattr(self_dm, "train_data", None), "train_data")
                r_test = filter_samples(getattr(self_dm, "test_data", None), "test_data")
                
                total_removed = r_train + r_test
                if total_removed > 0:
                    logger.info(f"Successfully removed {total_removed} contaminated files before splitting.")
                else:
                    logger.info("No files matched the exclusion list during pre-split filtering.")
            
            # Rebind the method to the instantiated datamodule
            datamodule._setup = types.MethodType(custom_setup, datamodule)
        # ==============================================================================

        # This triggers datamodule.setup() which now calls our injected code!
        log_dataset_details(datamodule, logger, export_paths=args.export_paths, output_path=output_path)

    except Exception as e:
        logger.error(f"DataModule Error: {e}")
        sys.exit(1)
        
    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    logger.info(f"Initializing Model: {args.model}")
    ModelClass = MODEL_MAP[args.model]
    model_kwargs = get_init_args(yaml_config, "model")
    
    if args.image_size or args.grayscale:
        try:
            if args.image_size:
                img_size = (args.image_size[0], args.image_size[0]) if len(args.image_size) == 1 else tuple(args.image_size[:2])
                pre_processor = ModelClass.configure_pre_processor(image_size=img_size)
            else:
                pre_processor = ModelClass.configure_pre_processor()

            if args.grayscale:
                pre_processor.transform = v2.Compose([v2.Grayscale(num_output_channels=3), pre_processor.transform])
            model_kwargs["pre_processor"] = pre_processor
        except Exception as e:
            logger.warning("Falling back to default model initialization.")

    if args.task == "classification":
        val_metrics =[AUROC(fields=["pred_score", "gt_label"], prefix="image_"), F1Max(fields=["pred_score", "gt_label"], prefix="image_"), AUPR(fields=["pred_score", "gt_label"], prefix="image_")]
        test_metrics = [AUROC(fields=["pred_score", "gt_label"]), F1Score(fields=["pred_label", "gt_label"])]
        evaluator = Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)
        monitor_metric = "image_AUROC"
    elif args.task == "segmentation":
        pixel_metrics =[AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"), F1Max(fields=["anomaly_map", "gt_mask"], prefix="pixel_")]
        image_metrics =[AUROC(fields=["pred_score", "gt_label"], prefix="image_"), F1Max(fields=["pred_score", "gt_label"], prefix="image_"), AUPR(fields=["pred_score", "gt_label"], prefix="image_")]
        evaluator = Evaluator(val_metrics=pixel_metrics + image_metrics, test_metrics=pixel_metrics + image_metrics)
        monitor_metric = "pixel_AUROC"
    else:
        evaluator = None 
        monitor_metric = "train_loss"

    if evaluator: model_kwargs["evaluator"] = evaluator

    model = ModelClass(**model_kwargs)

    # -------------------------------------------------------------------------
    # Callbacks & Engine
    # -------------------------------------------------------------------------
    tb_logger = AnomalibTensorBoardLogger(save_dir=str(output_path), name="tensorboard_logs", version="")
    
    rearrange_cb = RearrangeVisualizationsCallback(output_path=output_path, subfolder="test", logger=logger, model_name=args.model)
    raw_data_cb = RawDataExtractionCallback(output_path=output_path, subfolder="test", logger=logger)
    
    callbacks =[
        EarlyStopping(monitor=monitor_metric, mode="max" if "loss" not in monitor_metric else "min", patience=args.patience, verbose=True),
        FileLoggingCallback(logger=logger),
        rearrange_cb,
        raw_data_cb,
    ]
    
    trainer_config = yaml_config.get("trainer", {})
    engine = Engine(
        callbacks=callbacks, logger=tb_logger,
        min_epochs=trainer_config.get("min_epochs", args.min_epochs),
        max_epochs=trainer_config.get("max_epochs", args.max_epochs),
        accelerator=args.accelerator, devices=args.devices,
        default_root_dir=str(output_path),
    )

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------
    try:
        logger.info("Starting Fit...")
        engine.fit(model=model, datamodule=datamodule)
        
        # --- 1. NORMAL TEST (Test Set Only) ---
        logger.info("Starting Standard Test...")
        test_results = engine.test(model=model, datamodule=datamodule)

        temp_stats_path = output_path / f".tmp_{args.model}_custom_stats_test.json"
        if temp_stats_path.exists():
            with open(temp_stats_path, "r") as f: custom_stats = json.load(f)
            os.remove(temp_stats_path)
            if isinstance(test_results, list): test_results[0].update(custom_stats)
            elif isinstance(test_results, dict): test_results.update(custom_stats)

        with open(output_path / f"{args.model}_metrics.json", "w") as f:
            json.dump(test_results, f, indent=4)

        # --- 2. VALIDATION SET EVALUATION PASS ---
        if args.eval_val_test:
            logger.info("=======================================================")
            logger.info("   STARTING EVALUATION PASS ON VALIDATION SET  ")
            logger.info("=======================================================")
            
            rearrange_cb.subfolder = "val"
            raw_data_cb.subfolder = "val"
            
            logger.info(f"   Validation images to check: {len(datamodule.val_data)}")

            val_results = engine.test(model=model, dataloaders=datamodule.val_dataloader())
            logger.info(f"Validation Pass Metrics: {val_results}")
            
            temp_stats_path_val = output_path / f".tmp_{args.model}_custom_stats_val.json"
            if temp_stats_path_val.exists():
                with open(temp_stats_path_val, "r") as f: val_stats = json.load(f)
                os.remove(temp_stats_path_val)
                if isinstance(val_results, list): val_results[0].update(val_stats)
                elif isinstance(val_results, dict): val_results.update(val_stats)
                
            with open(output_path / f"{args.model}_metrics_val.json", "w") as f:
                json.dump(val_results, f, indent=4)
                
            try:
                df_test = pd.read_csv(output_path / f"{args.model}_test_predictions.csv")
                df_test.insert(0, "Split", "Test")
                
                df_val = pd.read_csv(output_path / f"{args.model}_val_predictions.csv")
                df_val.insert(0, "Split", "Validation")
                
                df_combined = pd.concat([df_val, df_test], ignore_index=True)
                combined_csv_path = output_path / f"{args.model}_val_test_combined_predictions.csv"
                df_combined.to_csv(combined_csv_path, index=False)
                logger.info(f"Combined Validation & Test CSV exported to: {combined_csv_path}")
            except Exception as e:
                logger.warning(f"Failed to merge validation and test CSVs: {e}")

        # --- 3. CONTAMINATION CHECK ---
        if args.check_contamination:
            logger.info("=======================================================")
            logger.info("   STARTING CONTAMINATION CHECK (Scanning ALL Normal Data)  ")
            logger.info("=======================================================")
            
            rearrange_cb.subfolder = "contamination"
            raw_data_cb.subfolder = "contamination"

            contamination_kwargs = filtered_kwargs.copy()
            contamination_kwargs["val_split_mode"] = ValSplitMode.NONE
            contamination_kwargs["test_split_mode"] = TestSplitMode.NONE
            
            contam_datamodule = DataClass(**contamination_kwargs)

            # Re-apply the same monkey patch so it also filters out contaminants from this secondary check!
            if exclude_files:
                original_setup_contam = contam_datamodule._setup
                def custom_setup_contam(self_dm, stage=None):
                    original_setup_contam(stage)
                    def filter_samples(dataset):
                        if not dataset or not hasattr(dataset, "samples"): return
                        keep_mask = []
                        for img_path in dataset.samples["image_path"]:
                            path_obj = Path(img_path)
                            expected_fp_name = f"{path_obj.parent.name}_{path_obj.stem}_FP{path_obj.suffix}"
                            if expected_fp_name in exclude_files or path_obj.name in exclude_files or str(path_obj) in exclude_files:
                                keep_mask.append(False)
                            else: keep_mask.append(True)
                        if not all(keep_mask):
                            dataset.samples = dataset.samples[keep_mask].reset_index(drop=True)
                    
                    filter_samples(getattr(self_dm, "train_data", None))
                    filter_samples(getattr(self_dm, "test_data", None))
                contam_datamodule._setup = types.MethodType(custom_setup_contam, contam_datamodule)

            contam_datamodule.setup()

            if hasattr(datamodule.test_data, "transform") and hasattr(contam_datamodule.train_data, "transform"):
                contam_datamodule.train_data.transform = datamodule.test_data.transform

            contamination_loader = contam_datamodule.train_dataloader()
            engine.test(model=model, dataloaders=contamination_loader)

        # ---------------------------------------------------------------------
        # Export
        # ---------------------------------------------------------------------
        if args.export_types:
            for ext in args.export_types:
                try:
                    engine.export(model=model, export_type=ExportType[ext.upper()], export_root=str(output_path / "weights"))
                except Exception as e:
                    logger.error(f"Failed to export {ext}. Reason: {e}")

        logger.info(f"Experiment finished. Results in {output_path}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise e

if __name__ == "__main__":
    main()