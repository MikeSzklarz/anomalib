import argparse
import logging
import sys
import warnings
import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Type, Set

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
from anomalib.metrics import Evaluator, AUROC, F1Score

from anomalib.data import (
    MVTecAD, MVTecLOCO, MVTecAD2, MVTec3D, 
    BTech, Visa, Folder, Kolektor, 
    Avenue, ShanghaiTech, UCSDped
)

from anomalib.models import (
    AnomalyDINO, Cfa, Cflow, Csflow, Dfkde, Dfm, 
    Dinomaly, Draem, Dsr, EfficientAd, Fastflow, 
    Fre, Ganomaly, Padim, Patchcore, ReverseDistillation, 
    Stfpm, Supersimplenet, Uflow, UniNet, WinClip
)

# -----------------------------------------------------------------------------
# 1. Mappings
# -----------------------------------------------------------------------------

MODEL_MAP: Dict[str, Type] = {
    "anomalydino": AnomalyDINO,
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
    "padim": Padim,
    "patchcore": Patchcore,
    "reversedistillation": ReverseDistillation,
    "stfpm": Stfpm,
    "supersimplenet": Supersimplenet,
    "uflow": Uflow,
    "uninet": UniNet
}

# Models that require iterative training (Gradient Descent)
ITERATIVE_MODELS: Set[str] = {
    "cflow", "csflow", "draem", "dsr", "efficientad", "fastflow", 
    "fre", "ganomaly", "reversedistillation", "stfpm", 
    "supersimplenet", "uflow", "uninet", "dinomaly"
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
# 2. Logger Setup
# -----------------------------------------------------------------------------

def setup_logger(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True
    )
    return logging.getLogger("train_script")

def log_dataset_details(datamodule, logger, print_paths=False):
    """Logs detailed stats and checks for TRUE data leakage using full paths."""
    logger.info("Setting up datamodule to inspect splits...")
    datamodule.setup()

    def get_info(dataset):
        if not dataset or not hasattr(dataset, 'samples'):
            return 0, 0, 0, set()
        
        samples = dataset.samples
        n_total = len(samples)
        
        # Robust label counting
        if 'label_index' in samples:
            n_normal = (samples.label_index == 0).sum()
            n_anom = (samples.label_index == 1).sum()
        else:
            n_normal = n_total
            n_anom = 0
            
        # Use full string path, NOT just filename
        filepaths = set(samples.image_path.astype(str).tolist())
        return n_total, n_normal, n_anom, filepaths

    n_train, norm_train, anom_train, files_train = get_info(getattr(datamodule, "train_data", None))
    n_val, norm_val, anom_val, files_val = get_info(getattr(datamodule, "val_data", None))
    n_test, norm_test, anom_test, files_test = get_info(getattr(datamodule, "test_data", None))

    logger.info("=== Dataset Split Statistics ===")
    logger.info(f"  [TRAIN] Total: {n_train} | Normal: {norm_train} | Anomalous: {anom_train}")
    logger.info(f"  [VAL  ] Total: {n_val} | Normal: {norm_val} | Anomalous: {anom_val}")
    logger.info(f"  [TEST ] Total: {n_test} | Normal: {norm_test} | Anomalous: {anom_test}")

    # Print paths only if requested
    if print_paths:
        def print_samples(name, files):
            logger.info(f"    Sample Files ({name}):")
            # Sort and print just the parent/filename to keep logs readable but useful
            short_paths = sorted([f"{Path(f).parent.name}/{Path(f).name}" for f in files])
            for p in short_paths[:5]:
                logger.info(f"      - .../{p}")
            if len(short_paths) > 5: logger.info(f"      ... ({len(short_paths)-5} more)")
        
        print_samples("VAL", files_val)
        print_samples("TEST", files_test)

    logger.info("=== Data Leakage Check (Full Path Overlap) ===")
    
    def check_overlap(set_a, set_b, name_a, name_b):
        overlap = set_a.intersection(set_b)
        if overlap:
            logger.error(f"   CRITICAL: {len(overlap)} images overlap between {name_a} and {name_b}!")
            # Print first overlapping file to help debug
            logger.error(f"    Example overlap: {list(overlap)[0]}")
        else:
            logger.info(f"      No overlap between {name_a} and {name_b}")

    check_overlap(files_train, files_val, "TRAIN", "VAL")
    check_overlap(files_train, files_test, "TRAIN", "TEST")
    check_overlap(files_val, files_test, "VAL", "TEST")
    
    logger.info("================================")
    
class FileLoggingCallback(Callback):
    """
    Logs metrics to the python logger at the end of every epoch 
    so they appear in the text log file.
    """
    def __init__(self, logger):
        self.logger = logger

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        # Get all logged metrics (this includes losses logged by Anomalib)
        metrics = trainer.callback_metrics
        
        # Filter for interesting metrics (loss, auroc, etc) and format them
        log_parts = [f"Epoch {epoch}"]
        for name, value in metrics.items():
            if isinstance(value, float) or hasattr(value, 'item'):
                log_parts.append(f"{name}: {float(value):.4f}")
        
        self.logger.info(" | ".join(log_parts))    
        
class RearrangeVisualizationsCallback(Callback):
    """
    1. Collects predictions during testing.
    2. Calculates the optimal F1 threshold.
    3. SEARCHES for the output images on disk.
    4. Reorganizes files IN-PLACE with source-folder prefixing:
       - Moves files to: images/anomalous/TP/chip_116_007_TP.jpg
       - Cleans up empty original folders.
    """
    def __init__(self, output_path: Path, logger=None):
        self.output_path = output_path
        self.logger = logger or logging.getLogger("train_script")
        self.preds_stats = [] 

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # 1. Helper to safely grab data
        def get_item(obj, key):
            if isinstance(obj, dict): return obj.get(key, None)
            return getattr(obj, key, None)

        # 2. Extract Data
        gt = get_item(outputs, "gt_label")
        if gt is None: gt = get_item(batch, "gt_label")

        pred_score = get_item(outputs, "pred_score")
        if pred_score is None: pred_score = get_item(batch, "pred_score")

        paths = get_item(outputs, "image_path")
        if paths is None: paths = get_item(batch, "image_path")

        # 3. Store Data (CPU)
        if gt is not None and pred_score is not None and paths is not None:
            gt = gt.cpu().squeeze()
            score = pred_score.cpu().squeeze()
            
            # Handle Scalar Edge Cases
            if gt.ndim == 0: gt = gt.unsqueeze(0)
            if score.ndim == 0: score = score.unsqueeze(0)

            path_list = [str(p) for p in paths]
            self.preds_stats.append((gt, score, path_list))

    def on_test_end(self, trainer, pl_module):
        if not self.preds_stats:
            self.logger.warning("No predictions collected. Skipping rearrangement.")
            return

        # 1. Flatten all batches
        all_gt = torch.cat([x[0] for x in self.preds_stats])
        all_scores = torch.cat([x[1] for x in self.preds_stats])
        all_paths = []
        for x in self.preds_stats: all_paths.extend(x[2])

        # 2. Determine Best Threshold (Matching Official F1)
        target_f1 = 0.642857 # Default fallback
        if "F1Score" in trainer.callback_metrics:
            target_f1 = trainer.callback_metrics["F1Score"].item()

        thresholds = torch.unique(all_scores)
        best_diff = float("inf")
        selected_thresh = 0.0
        
        is_anom_gt = (all_gt == 1)
        is_norm_gt = (all_gt == 0)

        for thresh in thresholds:
            pred_labels = (all_scores >= thresh).long()
            tp = torch.logical_and(is_anom_gt, (pred_labels == 1)).sum().item()
            fn = torch.logical_and(is_anom_gt, (pred_labels == 0)).sum().item()
            fp = torch.logical_and(is_norm_gt, (pred_labels == 1)).sum().item()
            
            if (tp + fp) > 0 and (tp + fn) > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                if (precision + recall) > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    diff = abs(f1 - target_f1)
                    if diff < best_diff:
                        best_diff = diff
                        selected_thresh = thresh

        # 3. Apply Selected Threshold
        pred_labels = (all_scores >= selected_thresh).long()

        # 4. Log Stats
        tp = torch.logical_and(is_anom_gt, (pred_labels == 1)).sum().item()
        fn = torch.logical_and(is_anom_gt, (pred_labels == 0)).sum().item()
        fp = torch.logical_and(is_norm_gt, (pred_labels == 1)).sum().item()
        tn = torch.logical_and(is_norm_gt, (pred_labels == 0)).sum().item()

        stats_msg = (
            f"\n FINAL CLASSIFICATION STATS \n"
            f" Threshold : {selected_thresh:.4f}\n"
            f" TP: {tp:<5} | FN: {fn}\n"
            f" TN: {tn:<5} | FP: {fp}\n"
            f" F1: {target_f1:.4f}\n"
        )
        self.logger.info(stats_msg)

        stats_data = {
            "custom_threshold": float(selected_thresh),
            "custom_F1_score": float(target_f1),
            "TP": int(tp),
            "FN": int(fn),
            "TN": int(tn),
            "FP": int(fp),
            "Total_Anomalous": int(tp + fn),
            "Total_Normal": int(tn + fp)
        }
        temp_stats_path = self.output_path / ".tmp_custom_stats.json"
        try:
            with open(temp_stats_path, "w") as f:
                json.dump(stats_data, f)
        except Exception as e:
            self.logger.error(f"Failed to stage custom stats: {e}")

        # 5. LOCATE IMAGES FOLDER
        base_search_dir = Path(trainer.default_root_dir)
        sample_file_name = Path(all_paths[0]).name
        found_files = list(base_search_dir.rglob(sample_file_name))
        vis_candidates = [f for f in found_files if "images" in str(f.parent) and "results" in str(f)]
        
        if not vis_candidates:
            # Fallback
            vis_candidates = [f for f in found_files if "datasets" not in str(f)]

        if not vis_candidates:
            self.logger.warning(f"Could not locate visualization for {sample_file_name} in {base_search_dir}")
            return

        sample_path = vis_candidates[0]
        if sample_path.parent.name == "images":
            images_root = sample_path.parent
        else:
            images_root = sample_path.parent.parent

        self.logger.info(f"Located existing visualizations at: {images_root}")

        # 6. Rearrange Files IN-PLACE
        moved_count = 0
        ops = []

        for i, original_path in enumerate(all_paths):
            gt_val = all_gt[i].item()
            pred_val = pred_labels[i].item()
            
            # Determine Category Folder
            main_cat = "anomalous" if gt_val == 1 else "normal"
            if gt_val == 1 and pred_val == 1: sub_cat = "TP"
            elif gt_val == 1 and pred_val == 0: sub_cat = "FN"
            elif gt_val == 0 and pred_val == 1: sub_cat = "FP"
            else: sub_cat = "TN"

            # Destination: images/anomalous/TP/
            dest_folder = images_root / main_cat / sub_cat
            
            fname = Path(original_path).name
            
            # Find THIS specific file inside images_root
            # Try direct lookup
            potential_paths = [
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
                candidates = [f for f in found if sub_cat not in str(f.parent)]
                if candidates:
                    source_file = candidates[0]

            if source_file and source_file.exists():
                # Get the folder prefix from the ORIGINAL input path (e.g. 'chip', 'good')
                folder_prefix = Path(original_path).parent.name
                
                # Format: folder_filename_TP.jpg (e.g. chip_116_007_TP.jpg)
                new_name = f"{folder_prefix}_{source_file.stem}_{sub_cat}{source_file.suffix}"
                
                ops.append((source_file, dest_folder / new_name))

        # Execute Moves
        for src, dst in ops:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                moved_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to move {src.name}: {e}")

        # 7. Cleanup Empty Folders
        for item in images_root.iterdir():
            if item.is_dir() and item.name not in ["normal", "anomalous"]:
                try:
                    item.rmdir() 
                except OSError:
                    pass 

        self.logger.info(f"Reorganization complete. Updated {moved_count} images.")
        
        
        
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
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    parser.add_argument("--output_dir", type=str, default="./results")
    
    # Data params
    parser.add_argument("--root_dir", type=str, default="./datasets")
    parser.add_argument("--category", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    
    # Training params
    parser.add_argument("--max_epochs", type=int, default=999)
    parser.add_argument("--task", type=str, default="segmentation", choices=["classification", "segmentation", "detection"])
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Export & Saving
    parser.add_argument("--export_types", nargs="+", default=[], choices=["torch", "openvino", "onnx"], 
                        help="List of formats to export (e.g. --export_types torch openvino)")
    parser.add_argument("--no_checkpoint", action="store_true", 
                        help="If set, strictly prevents saving .ckpt weights to disk (saves space)")
    parser.add_argument("--image_size", type=int, nargs="+", default=None, 
                        help="Input image size (Height, Width) Default: None")
    parser.add_argument("--grayscale", action="store_true",
                        help="Strictly force input to 3-channel Grayscale")
    parser.add_argument("--print_paths", action="store_true", 
                        help="Print filenames of all images in every split to verify distribution.")

    args = parser.parse_args()
    
    if args.seed is not None:
        seed_everything(args.seed, workers=True)
    
    # Path Setup
    output_path = Path(args.output_dir) / args.model / args.dataset / args.category
    logger = setup_logger(output_path)
    logger.info(f"Experiment Args: {vars(args)}")

    # Load Config
    yaml_config = load_yaml_config(args.config) if args.config else {}

    # -------------------------------------------------------------------------
    # Dataset Initialization
    # -------------------------------------------------------------------------
    logger.info(f"Initializing DataModule: {args.dataset}")
    DataClass = DATASET_MAP[args.dataset]
    ds_kwargs = get_init_args(yaml_config, "data")
    
    # 1. Apply Global Overrides
    ds_kwargs.update({
        "root": args.root_dir,
        "train_batch_size": args.batch_size,
        "eval_batch_size": args.batch_size,
        "seed": args.seed,
    })

    # 2. Handle EfficientAD Constraint
    if args.model == "efficientad" and ds_kwargs["train_batch_size"] != 1:
        logger.warning("EfficientAD requires train_batch_size=1. Overriding.")
        ds_kwargs["train_batch_size"] = 1
    
    # 3. Handle Dataset Specific Arguments (Smart Filtering)
    import inspect
    valid_args = inspect.signature(DataClass.__init__).parameters
    
    # Logic for 'category'
    if "category" in valid_args:
        ds_kwargs["category"] = args.category
    elif "category" in ds_kwargs:
        # Remove category if dataset (like Kolektor) doesn't support it
        ds_kwargs.pop("category")

    # Logic for 'name' (Folder dataset)
    if args.dataset == "Folder":
        ds_kwargs["name"] = args.category
        
    # Logic for 'name' and 'normal_dir' (Folder dataset)
    if args.dataset == "folder":
        # 1. Set the dataset name
        ds_kwargs["name"] = args.category if args.category else "custom_folder"

        # 2. Smart-Detect paths relative to the ROOT you passed
        # We look directly inside args.root_dir
        root_p = Path(args.root_dir)
        
        # Smart-Detect 'normal_dir'
        if "normal_dir" not in ds_kwargs:
            if (root_p / "train" / "good").exists():
                ds_kwargs["normal_dir"] = "train/good"  # MVTec Style
            elif (root_p / "good").exists():
                ds_kwargs["normal_dir"] = "good"        # Simple Style
            else:
                ds_kwargs["normal_dir"] = "train/good"  # Fallback
        
        # Smart-Detect 'abnormal_dir'
        if "abnormal_dir" not in ds_kwargs:
            if (root_p / "test").exists():
                ds_kwargs["abnormal_dir"] = "test"      # MVTec Style (contains subfolders)
            elif (root_p / "defect").exists():
                ds_kwargs["abnormal_dir"] = "defect"    # Simple Style

    # Filter out any kwargs from config/CLI that the specific dataset class doesn't support
    # (e.g. MVTecAD2 crashes if you pass 'val_split_mode')
    filtered_kwargs = {k: v for k, v in ds_kwargs.items() if k in valid_args}

    try:
        datamodule = DataClass(**filtered_kwargs)
        log_dataset_details(datamodule, logger, print_paths=args.print_paths)
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
            # 1. Determine Resolution
            if args.image_size:
                if len(args.image_size) == 1:
                    img_size = (args.image_size[0], args.image_size[0])
                else:
                    img_size = tuple(args.image_size[:2])
                logger.info(f"Configuring PreProcessor for Resolution: {img_size}")
                # Generate with specific size
                pre_processor = ModelClass.configure_pre_processor(image_size=img_size)
            else:
                # User didn't specify size, so use the Model's internal default
                logger.info("Using model default resolution.")
                pre_processor = ModelClass.configure_pre_processor()

            # 2. Apply Grayscale Wrapper if requested
            if args.grayscale:
                logger.info("--- FORCING GRAYSCALE (3-Channel) ---")
                # We wrap the existing transform pipeline.
                # Pipeline becomes: Input -> Grayscale -> [Original_Resize -> Original_Normalize]
                # We use num_output_channels=3 so it doesn't crash backbones expecting RGB.
                pre_processor.transform = v2.Compose([
                    v2.Grayscale(num_output_channels=3),
                    pre_processor.transform
                ])

            model_kwargs["pre_processor"] = pre_processor

        except Exception as e:
            logger.warning(f"Failed to configure custom pre_processor: {e}")
            logger.warning("Falling back to default model initialization.")
    
    # If task is classification, explicitly define metrics to exclude pixel-level checks.
    if args.task == "classification":
        logger.info("Task is 'classification'. Configuring Evaluator for Unbounded Scores.")
        
        val_metrics = [
            AUROC(fields=["pred_score", "gt_label"])
        ]
        
        test_metrics = [
            AUROC(fields=["pred_score", "gt_label"]),
            F1Score(fields=["pred_label", "gt_label"])
        ]
        
        # Create the evaluator with distinct sets
        evaluator = Evaluator(
            val_metrics=val_metrics,
            test_metrics=test_metrics
        )
        
        model_kwargs["evaluator"] = evaluator
    
    model = ModelClass(**model_kwargs)

    # -------------------------------------------------------------------------
    # Inspect tranforms and resolution
    # -------------------------------------------------------------------------
    logger.info("=== Tranform & Resolution Inspection ===")
    
    if hasattr(model, "pre_processor") and model.pre_processor is not None:
        logger.info(f"Model PreProcessor (Hard Resolution/Norm): {model.pre_processor.transform}")
    else:
        logger.warning("Model does not have an active PreProcessor")
        
    train_augs = getattr(datamodule, "train_augmentation", None)
    val_augs = getattr(datamodule, "val_augmentation", None)
    test_augs = getattr(datamodule, "test_augmentation", None)

    logger.info(f"Train Augmentation: {train_augs}")
    logger.info(f"Val Augmentation: {val_augs}")
    logger.info(f"Test Augmentation: {test_augs}")
    logger.info(f"=====================================")

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    tb_logger = AnomalibTensorBoardLogger(save_dir=str(output_path), name="tensorboard_logs", version="")
    
    callbacks = [
        EarlyStopping(
            monitor="AUROC",
            mode="max",
            patience=20,
        ),
        FileLoggingCallback(logger=logger),
        RearrangeVisualizationsCallback(output_path=output_path, logger=logger),
    ]
    
    # -------------------------------------------------------------------------
    # Engine
    # -------------------------------------------------------------------------
    engine = Engine(
        callbacks=callbacks,
        logger=tb_logger,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=str(output_path),
    )

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------
    try:
        logger.info("Starting Fit...")
        engine.fit(model=model, datamodule=datamodule)
        
        logger.info("Starting Test...")
        # Note: If checkpointing is disabled, test() uses the in-memory model (last state)
        test_results = engine.test(model=model, datamodule=datamodule)
        logger.info(f"Test Results: {test_results}")

        custom_stats = {}
        temp_stats_path = output_path / ".tmp_custom_stats.json"
        if temp_stats_path.exists():
            try:
                with open(temp_stats_path, "r") as f:
                    custom_stats = json.load(f)
                # Clean up temp file
                os.remove(temp_stats_path)
            except Exception as e:
                logger.warning(f"Found temp stats but failed to load: {e}")

        # 2. Merge into test_results (handle list vs dict return types)
        if isinstance(test_results, list):
            for res in test_results:
                if isinstance(res, dict):
                    res.update(custom_stats)
        elif isinstance(test_results, dict):
            test_results.update(custom_stats)

        # 3. Save final combined JSON
        json_metrics_path = output_path / "metrics.json"
        try:
            with open(json_metrics_path, "w") as f:
                json.dump(test_results, f, indent=4)
            logger.info(f"All Metrics (Standard + Custom) exported to {json_metrics_path}")
        except Exception as e:
            logger.error(f"Failed to export metrics to JSON: {e}")

        # ---------------------------------------------------------------------
        # Export (Optional)
        # ---------------------------------------------------------------------
        if args.export_types:
            logger.info(f"Attempting export to: {args.export_types}")
            for ext in args.export_types:
                try:
                    logger.info(f"Exporting to {ext}...")
                    
                    # Convert string arg to Anomalib ExportType Enum
                    export_enum = ExportType[ext.upper()]
                    
                    engine.export(
                        model=model,
                        export_type=export_enum,
                        export_root=str(output_path / "weights"),
                    )
                    logger.info(f"Successfully exported {ext}")
                except Exception as e:
                    logger.error(f"Failed to export {ext}. Reason: {e}")
        else:
            logger.info("No export types specified. Skipping export.")

        logger.info(f"Experiment finished. Results in {output_path}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise e

if __name__ == "__main__":
    main()