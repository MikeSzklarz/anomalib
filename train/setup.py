"""Dataset and model construction.

build_datamodule() turns CLI args + yaml config into a ready-to-use datamodule
(handling the exclusion-list monkeypatch and per-dataset special-casing),
build_contamination_datamodule() builds the secondary "scan everything labeled
normal" datamodule used by --check_contamination, and build_model() turns them
into a configured AnomalibModule + the metric name to monitor for early stopping.
"""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path
from typing import Any, Dict, Set, Tuple, Type

import pandas as pd
from torchvision.transforms import v2

from anomalib.data.utils.split import TestSplitMode, ValSplitMode
from anomalib.metrics import AUPR, AUROC, Evaluator, F1Max, F1Score

from .registry import DATASET_MAP, MODEL_MAP
from .utils import get_init_args


def load_exclusion_files(args, logger) -> Set[str]:
    """Loads the per-model exclusion list (CSV/TXT of filenames to drop) if --exclude_root is set."""
    exclude_files: Set[str] = set()
    if not args.exclude_root:
        return exclude_files

    exclude_dir = Path(args.exclude_root)
    csv_path = exclude_dir / f"{args.model}.csv"
    txt_path = exclude_dir / f"{args.model}.txt"

    target_path = None
    if csv_path.exists(): target_path = csv_path
    elif txt_path.exists(): target_path = txt_path

    if not target_path:
        logger.warning(f"Exclude root '{args.exclude_root}' provided but no exclusion file found for model '{args.model}'")
        return exclude_files

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

    return exclude_files


def _log_dataset_details(datamodule, logger, export_paths=False, output_path=None):
    """Logs detailed stats and checks for TRUE data leakage using full paths.

    Returns a JSON-serializable dict of the same split counts and overlap
    results that get logged, for use in manifest.json and the split
    composition chart.
    """
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
            splits = [("Train", files_train), ("Validation", files_val), ("Test", files_test)]

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
        return {"pair": f"{name_a}-{name_b}", "overlap_count": len(overlap)}

    overlap_results = [
        check_overlap(files_train, files_val, "TRAIN", "VAL"),
        check_overlap(files_train, files_test, "TRAIN", "TEST"),
        check_overlap(files_val, files_test, "VAL", "TEST"),
    ]
    logger.info("================================")

    return {
        "train": {"total": n_train, "normal": int(norm_train), "anomalous": int(anom_train)},
        "val": {"total": n_val, "normal": int(norm_val), "anomalous": int(anom_val)},
        "test": {"total": n_test, "normal": int(norm_test), "anomalous": int(anom_test)},
        "overlap_checks": overlap_results,
    }


def _apply_exclusion_monkeypatch(datamodule, exclude_files: Set[str], logger, contamination: bool = False) -> None:
    """Monkey-patches datamodule._setup so excluded files are dropped from the
    DataFrames BEFORE they are mathematically split into val/test pools."""
    original_setup = datamodule._setup

    def filter_samples(dataset, subset_name=""):
        if not dataset or not hasattr(dataset, "samples"):
            return 0
        orig_len = len(dataset.samples)
        if orig_len == 0:
            return 0
        keep_mask = []
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

    def custom_setup(self_dm, stage=None):
        original_setup(stage)
        if contamination:
            filter_samples(getattr(self_dm, "train_data", None))
            filter_samples(getattr(self_dm, "test_data", None))
            return
        logger.info("Applying exclusion filter BEFORE dataset splits...")
        r_train = filter_samples(getattr(self_dm, "train_data", None), "train_data")
        r_test = filter_samples(getattr(self_dm, "test_data", None), "test_data")
        total_removed = r_train + r_test
        if total_removed > 0:
            logger.info(f"Successfully removed {total_removed} contaminated files before splitting.")
        else:
            logger.info("No files matched the exclusion list during pre-split filtering.")

    datamodule._setup = types.MethodType(custom_setup, datamodule)


def build_datamodule(
    args, yaml_config: Dict[str, Any], exclude_files: Set[str], logger, output_path: Path,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], Type]:
    """Returns (datamodule, dataset_stats, filtered_kwargs, DataClass)."""
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

    valid_args = inspect.signature(DataClass.__init__).parameters

    if "category" in valid_args:
        dataset_kwargs["category"] = args.category
    elif "category" in dataset_kwargs:
        dataset_kwargs.pop("category")

    if args.dataset == "kolektor":
        dataset_kwargs["val_split_mode"] = "from_test"
        dataset_kwargs["val_split_ratio"] = 0.5

    # Explicit CLI overrides win over the dataset-specific default above and anything in the yaml config.
    split_overrides = {
        "val_split_mode": args.val_split_mode,
        "val_split_ratio": args.val_split_ratio,
        "test_split_mode": args.test_split_mode,
        "test_split_ratio": args.test_split_ratio,
    }
    for key, value in split_overrides.items():
        if value is None:
            continue
        if key not in valid_args:
            logger.warning(
                f"[split] --{key.replace('_', '-')} was set to '{value}' but '{args.dataset}' does not accept "
                f"this parameter - it will have no effect."
            )
            continue
        dataset_kwargs[key] = value
        logger.info(f"[split] Overriding {key} = {value}")

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

        if args.task == "segmentation" and "mask_dir" not in dataset_kwargs:
            if (root_p / "ground_truth").exists():
                dataset_kwargs["mask_dir"] = "ground_truth"

    filtered_kwargs = {k: v for k, v in dataset_kwargs.items() if k in valid_args}

    for key in ("test_split_mode", "test_split_ratio", "val_split_mode", "val_split_ratio"):
        if key in filtered_kwargs:
            source = "CLI override" if split_overrides.get(key) is not None else "train dataset-specific default / yaml config"
            logger.info(f"[split] {key} = {filtered_kwargs[key]} ({source})")
        elif key in valid_args:
            logger.info(f"[split] {key} = {valid_args[key].default} ('{args.dataset}' class default)")
        else:
            logger.info(f"[split] {key} = N/A (not supported by '{args.dataset}')")

    try:
        datamodule = DataClass(**filtered_kwargs)

        # ==============================================================================
        # INJECT PRE-SPLIT FILTERING
        # Here we Monkey-Patch the _setup method before datamodule.setup() is called
        # so that files are removed from the DataFrames BEFORE they are mathematically
        # split into val and test pools.
        # ==============================================================================
        if exclude_files:
            _apply_exclusion_monkeypatch(datamodule, exclude_files, logger)
        # ==============================================================================

        # This triggers datamodule.setup() which now calls our injected code!
        dataset_stats = _log_dataset_details(datamodule, logger, export_paths=args.export_paths, output_path=output_path)

    except Exception as e:
        logger.error(f"DataModule Error: {e}")
        sys.exit(1)

    return datamodule, dataset_stats, filtered_kwargs, DataClass


def build_contamination_datamodule(args, DataClass: Type, filtered_kwargs: Dict[str, Any], exclude_files: Set[str], datamodule, logger):
    """Builds the secondary datamodule used by --check_contamination: every
    file the dataset considers 'normal' (no val/test split), scanned for
    scores that suggest mislabeled/contaminated training data.

    Returns (contamination_loader, contamination_pool_size).
    """
    contamination_kwargs = filtered_kwargs.copy()
    contamination_kwargs["val_split_mode"] = ValSplitMode.NONE
    contamination_kwargs["test_split_mode"] = TestSplitMode.NONE

    contam_datamodule = DataClass(**contamination_kwargs)

    # Re-apply the same monkey patch so it also filters out contaminants from this secondary check!
    if exclude_files:
        _apply_exclusion_monkeypatch(contam_datamodule, exclude_files, logger, contamination=True)

    contam_datamodule.setup()

    if hasattr(datamodule.test_data, "transform") and hasattr(contam_datamodule.train_data, "transform"):
        contam_datamodule.train_data.transform = datamodule.test_data.transform

    contamination_loader = contam_datamodule.train_dataloader()
    contamination_pool_size = len(contam_datamodule.train_data)
    return contamination_loader, contamination_pool_size


def build_model(args, yaml_config: Dict[str, Any], logger, resolved_image_size: int | None = None) -> Tuple[Any, str]:
    """Returns (model, monitor_metric). resolved_image_size comes from the
    --image_size_ratio resolution in cli.py, if that flag was used."""
    logger.info(f"Initializing Model: {args.model}")
    ModelClass = MODEL_MAP[args.model]
    model_kwargs = get_init_args(yaml_config, "model")

    _use_color_prep = args.color_preprocessing and args.color_preprocessing != "none"
    if resolved_image_size or args.image_size or _use_color_prep:
        try:
            if resolved_image_size:
                img_size = (resolved_image_size, resolved_image_size)
                pre_processor = ModelClass.configure_pre_processor(image_size=img_size)
            elif args.image_size:
                img_size = (args.image_size[0], args.image_size[0]) if len(args.image_size) == 1 else tuple(args.image_size[:2])
                pre_processor = ModelClass.configure_pre_processor(image_size=img_size)
            else:
                pre_processor = ModelClass.configure_pre_processor()

            if _use_color_prep:
                from .preprocessing import get_color_transform
                color_transform = get_color_transform(args.color_preprocessing)
                if color_transform is not None:
                    pre_processor.transform = v2.Compose([color_transform, pre_processor.transform])
                    logger.info(f"Color preprocessing: {color_transform}")
            model_kwargs["pre_processor"] = pre_processor
        except Exception as e:
            logger.warning(f"Falling back to default model initialization: {e}")

    if args.task == "classification":
        val_metrics = [AUROC(fields=["pred_score", "gt_label"], prefix="image_"), F1Max(fields=["pred_score", "gt_label"], prefix="image_"), AUPR(fields=["pred_score", "gt_label"], prefix="image_")]
        test_metrics = [AUROC(fields=["pred_score", "gt_label"]), F1Score(fields=["pred_label", "gt_label"])]
        evaluator = Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)
        monitor_metric = "image_AUROC"
    elif args.task == "segmentation":
        pixel_metrics = [AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"), F1Max(fields=["anomaly_map", "gt_mask"], prefix="pixel_")]
        image_metrics = [AUROC(fields=["pred_score", "gt_label"], prefix="image_"), F1Max(fields=["pred_score", "gt_label"], prefix="image_"), AUPR(fields=["pred_score", "gt_label"], prefix="image_")]
        evaluator = Evaluator(val_metrics=pixel_metrics + image_metrics, test_metrics=pixel_metrics + image_metrics)
        monitor_metric = "pixel_AUROC"
    else:
        evaluator = None
        monitor_metric = "train_loss"

    if evaluator: model_kwargs["evaluator"] = evaluator

    model = ModelClass(**model_kwargs)
    return model, monitor_metric
