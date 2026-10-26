"""Command-line entry point: argument parsing, --experiment file merging, image
size resolution, and wiring dataset/model setup into the run pipeline."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import yaml
from lightning.pytorch import seed_everything

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")

from anomalib.visualization.image.item_visualizer import DEFAULT_TEXT_CONFIG
DEFAULT_TEXT_CONFIG["enable"] = False

from . import pipeline
from . import setup as dataset_model_setup
from .callbacks.time_limit import parse_max_time
from .image_size import get_resize_target, log_image_size_analysis
from .image_size_rules import MODEL_IMAGE_SIZE_RULES, locked_image_size, resolve_from_ratio, validate_image_size
from .registry import DATASET_MAP, MODEL_MAP
from .utils import load_yaml_config, setup_logger


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # Basics
    parser.add_argument("--model", type=str, default=None, choices=MODEL_MAP.keys())
    parser.add_argument("--dataset", type=str, default=None, choices=DATASET_MAP.keys())
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--config_root", type=str, default=None)
    parser.add_argument("--experiment", type=str, default=None,
                        help="Path to a YAML file of script-level arguments (root_dir, output_dir, "
                             "task, split ratios, flags, etc.) for this run - keys match the CLI flag "
                             "names. Any flag also given explicitly on the command line overrides the "
                             "file's value; unknown keys in the file are ignored with a warning.")
    parser.add_argument("--output_dir", type=str, default="./results")

    # Data params
    parser.add_argument("--root_dir", type=str, default="./datasets")
    parser.add_argument("--category", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--exclude_root", type=str, default=None, help="Directory containing CSV/TXT files of images to exclude per model.")
    parser.add_argument("--val_split_mode", type=str, default=None,
                        choices=["none", "same_as_test", "from_train", "from_test", "synthetic", "from_dir"],
                        help="Override how the validation set is derived (default: whatever the dataset class "
                             "itself defaults to - often 'same_as_test', meaning validation reuses the test set "
                             "verbatim). Use 'from_test' for a genuinely held-out val/test split.")
    parser.add_argument("--val_split_ratio", type=float, default=None,
                        help="Fraction of data used for validation when --val_split_mode is from_train/from_test/"
                             "synthetic. Ignored by other modes.")
    parser.add_argument("--test_split_mode", type=str, default=None,
                        choices=["none", "from_dir", "synthetic"],
                        help="Override how the test set is derived (default: whatever the dataset class itself "
                             "defaults to).")
    parser.add_argument("--test_split_ratio", type=float, default=0.3,
                        help="Fraction of normal training data pulled into the test set when the test directory "
                             "has no normal images of its own (then halved with validation under the default "
                             "val_split_mode). Default 0.3 instead of anomalib's 0.2, for larger val/test normal "
                             "sets. Ignored when the test directory has its own normal images.")

    # Training params
    parser.add_argument("--max_epochs", type=int, default=999)
    parser.add_argument("--min_epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=20,
                        help="Epochs without improvement before early stopping. The best epoch is restored "
                             "afterwards, so a longer patience only costs time, not accuracy.")
    parser.add_argument("--max_time", type=str, default="05:00:00",
                        help="Wall-clock cap on TRAINING, as HH:MM:SS or D:HH:MM:SS (default 05:00:00, sized for "
                             "the 6h SLURM jobs). When hit, training stops (even before min_epochs) and the run "
                             "continues to restore-best/test/val/contamination/export. 'none' disables it.")
    parser.add_argument("--task", type=str, default="segmentation", choices=["classification", "segmentation", "detection"])
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # Export & Saving
    parser.add_argument("--export_types", nargs="+", default=[], choices=["torch", "openvino", "onnx"])
    parser.add_argument("--no_checkpoint", action="store_true")
    parser.add_argument("--image_size", type=int, nargs="+", default=None)
    parser.add_argument("--image_size_ratio", type=float, default=None,
                        help="Scale this model's own baseline default image_size by this ratio "
                             "(e.g. 1.2). Resolved per-model against known architecture constraints; "
                             "if the resolved size would equal the baseline (architecturally frozen "
                             "model, or ratio too small to move the snapped size), the run is skipped. "
                             "Mutually exclusive with --image_size.")
    parser.add_argument("--color_preprocessing", type=str, default="none",
                        choices=["none", "reinhard_neutral", "reinhard_cool", "clahe",
                                 "reinhard_clahe", "desaturate", "grayscale_3ch",
                                 "viridis", "composite"],
                        help="Inline color preprocessing applied before ImageNet normalization.")
    parser.add_argument("--export_paths", action="store_true")
    parser.add_argument("--compact_raw_outputs", action="store_true",
                        help="Quantize raw_outputs/<split>/arrays.npz anomaly maps to uint8 instead of "
                             "float16 (~10x smaller, ~0.2%% of the value range in max quantization error). "
                             "Off by default since it's lossy; the default float16 + LZMA storage is already "
                             "~40%% smaller than the old per-image PNG/npz files with zero precision cost.")
    parser.add_argument("--check_contamination", action="store_true")
    parser.add_argument("--eval_val_test", action="store_true",
                        help="Run a secondary test pass over the Validation set and merge the CSV outputs.")

    # Resource monitoring
    parser.add_argument("--disable_resource_monitor", action="store_true",
                        help="Disable CPU/GPU/memory resource monitoring (on by default).")
    return parser


def _apply_experiment_file(args, parser: argparse.ArgumentParser) -> dict | None:
    if not args.experiment:
        return None

    explicit_cli_flags = {
        tok[2:].split("=")[0] for tok in sys.argv[1:] if tok.startswith("--")
    }
    exp_path = Path(args.experiment)
    if not exp_path.exists():
        parser.error(f"--experiment file not found: {exp_path}")
    with open(exp_path, "r") as f:
        experiment_config = yaml.safe_load(f) or {}

    valid_dests = vars(args).keys()
    applied, skipped = {}, []
    for key, value in experiment_config.items():
        if key not in valid_dests:
            skipped.append(key)
        elif key in explicit_cli_flags:
            continue  # CLI always wins over the experiment file
        else:
            setattr(args, key, value)
            applied[key] = value
    return {"file": str(exp_path), "applied": applied, "skipped": skipped}


def _validate_args(args, parser: argparse.ArgumentParser) -> None:
    if args.model is None or args.model not in MODEL_MAP:
        parser.error(f"--model is required and must be one of {sorted(MODEL_MAP)} (via --model or --experiment)")
    if args.dataset is None or args.dataset not in DATASET_MAP:
        parser.error(f"--dataset is required and must be one of {sorted(DATASET_MAP)} (via --dataset or --experiment)")
    if args.image_size and args.image_size_ratio is not None:
        parser.error("--image_size and --image_size_ratio are mutually exclusive.")
    if args.max_time is not None and args.max_time.lower() == "none":
        args.max_time = None
    if args.max_time is not None:
        try:
            parse_max_time(args.max_time)
        except ValueError as e:
            parser.error(str(e))


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    experiment_summary = _apply_experiment_file(args, parser)
    _validate_args(args, parser)

    if args.seed is not None:
        seed_everything(args.seed, workers=True)

    resolved_image_size = None
    baseline_size = skip_run = size_explanation = None
    if args.image_size_ratio is not None:
        resolved_image_size, baseline_size, skip_run, size_explanation = resolve_from_ratio(
            args.model, args.image_size_ratio,
        )

    output_path = Path(args.output_dir) / args.model / args.dataset / args.category
    if resolved_image_size is not None:
        # ratio 1.0 is the model's own default size: name it so it stands out among the ratio runs.
        prefix = "baseline" if args.image_size_ratio == 1.0 else "imgsize"
        output_path = output_path / f"{prefix}_{resolved_image_size}"
    if args.seed is not None:
        # Deepest level, so every seed of one model/size config sits side by side.
        output_path = output_path / f"seed_{args.seed}"
    logger = setup_logger(output_path, args.model)
    logger.info(f"Experiment Args: {vars(args)}")
    if experiment_summary:
        logger.info(f"[experiment] Loaded {experiment_summary['file']}: applied {experiment_summary['applied']}")
        if experiment_summary["skipped"]:
            logger.warning(f"[experiment] Unknown keys in {experiment_summary['file']}, ignored: {experiment_summary['skipped']}")

    target_h = target_w = None
    if args.image_size_ratio is not None:
        logger.info(f"[image_size_ratio] {size_explanation}")
        if skip_run:
            logger.info(
                f"[image_size_ratio] Skipping run: '{args.model}' has no valid image_size at "
                f"ratio={args.image_size_ratio} distinct from its baseline default ({baseline_size})."
            )
            sys.exit(0)
        target_h = target_w = resolved_image_size
    elif args.image_size:
        baseline_size = MODEL_IMAGE_SIZE_RULES.get(args.model, {}).get("default", "unknown")
        logger.info(f"[image_size] Explicit override: {args.image_size} (model baseline default: {baseline_size})")
        target_h, target_w = (args.image_size[0], args.image_size[0]) if len(args.image_size) == 1 else tuple(args.image_size[:2])
        locked_size = locked_image_size(args.model)
        if locked_size is not None and (target_h, target_w) != (locked_size, locked_size):
            logger.warning(
                f"[image_size] Ignoring --image_size {args.image_size} for '{args.model}': its pre-processor "
                f"center-crops to a hardcoded size, so another value would only crop away more of the image or "
                f"zero-pad it. Using the locked default ({locked_size})."
            )
            args.image_size = None
            target_h = target_w = locked_size
        else:
            size_warning = validate_image_size(args.model, target_h, target_w)
            if size_warning:
                logger.warning(f"[image_size] {size_warning}")
    else:
        baseline_size = MODEL_IMAGE_SIZE_RULES.get(args.model, {}).get("default", "unknown")
        logger.info(f"[image_size] Using model baseline default: {baseline_size}")
        try:
            resize_target = get_resize_target(MODEL_MAP[args.model].configure_pre_processor())
            if resize_target:
                target_h, target_w = resize_target
        except Exception as e:
            logger.warning(f"[image_size] Could not determine baseline resize target for native-resolution check: {e}")

    config_path = args.config
    if config_path is None:
        target_folder = Path(args.config_root) if args.config_root else Path(f"./{args.dataset}_configs")
        auto_path = target_folder / f"{args.model}.yaml"
        if auto_path.exists():
            config_path = str(auto_path)

    yaml_config = load_yaml_config(config_path) if config_path else {}

    exclude_files = dataset_model_setup.load_exclusion_files(args, logger)

    datamodule, dataset_stats, filtered_kwargs, DataClass = dataset_model_setup.build_datamodule(
        args, yaml_config, exclude_files, logger, output_path,
    )

    if target_h and target_w:
        log_image_size_analysis(datamodule, logger, args.model, target_h, target_w)

    model, monitor_metric = dataset_model_setup.build_model(args, yaml_config, logger, resolved_image_size)

    pipeline.run(
        args=args,
        model=model,
        datamodule=datamodule,
        DataClass=DataClass,
        filtered_kwargs=filtered_kwargs,
        exclude_files=exclude_files,
        dataset_stats=dataset_stats,
        monitor_metric=monitor_metric,
        logger=logger,
        output_path=output_path,
        config_path=config_path,
        experiment_summary=experiment_summary,
        target_h=target_h,
        target_w=target_w,
        trainer_config=yaml_config.get("trainer", {}),
    )
