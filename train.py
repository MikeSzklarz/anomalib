import argparse
import logging
import sys
import warnings
import yaml
from pathlib import Path
from typing import Dict, Any, Type, Set

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")

# Anomalib Imports
from anomalib.engine import Engine
from anomalib.deploy import ExportType
from anomalib.loggers import AnomalibTensorBoardLogger

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
    "uninet": UniNet,
    "winclip": WinClip
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
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--task", type=str, default="segmentation", choices=["classification", "segmentation", "detection"])
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    
    # Export & Saving
    parser.add_argument("--export_types", nargs="+", default=[], choices=["torch", "openvino", "onnx"], 
                        help="List of formats to export (e.g. --export_types torch openvino)")
    parser.add_argument("--no_checkpoint", action="store_true", 
                        help="If set, strictly prevents saving .ckpt weights to disk (saves space)")

    parser.add_argument("--print_paths", action="store_true", 
                        help="Print filenames of all images in every split to verify distribution.")

    args = parser.parse_args()
    
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
        "eval_batch_size": args.batch_size
    })

    # 2. Handle EfficientAD Constraint
    if args.model == "EfficientAD" and ds_kwargs["train_batch_size"] != 1:
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
    model = ModelClass(**model_kwargs)

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    tb_logger = AnomalibTensorBoardLogger(save_dir=str(output_path), name="tensorboard_logs", version="")
    
    # -------------------------------------------------------------------------
    # Engine
    # -------------------------------------------------------------------------
    engine = Engine(
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
        
        with open(output_path / "metrics.yaml", "w") as f:
            yaml.dump(test_results, f)

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