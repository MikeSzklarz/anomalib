"""Runs one full experiment: callbacks/engine construction, fit, evaluation of BOTH the
last and the best epoch (test -> val -> contamination), export, output consolidation,
manifest.json, and the post-test analysis charts. Called from cli.main() once the
datamodule and model are built.

Run folder layout:
    metrics/metrics.json        {"primary", "last": {...}, "best": {...}, "comparison"}
    manifest.json               run provenance; checkpoint_selection describes both checkpoints
    epoch_metrics.csv, resource_usage/, tensorboard_logs/   run-level, shared
    last/, best/                predictions.csv, analysis/ - the primary checkpoint
                                (--primary_checkpoint, default last) also gets raw_outputs/<split>/,
                                images/, analysis/figures/ and the contamination check
    weights/lightning/          last.ckpt, best.ckpt (exports go to weights/, from the primary)
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping

from anomalib.callbacks.checkpoint import ModelCheckpoint
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.loggers import AnomalibTensorBoardLogger

from . import setup as dataset_model_setup
from .analysis.orchestrate import run_post_test_analysis
from .callbacks import (
    FileLoggingCallback,
    RawDataExtractionCallback,
    RearrangeVisualizationsCallback,
    ResourceMonitorCallback,
    TrainingTimeLimitCallback,
)
from .callbacks.time_limit import parse_max_time
from .utils import dir_size_bytes, get_git_commit

CHECKPOINT_NAMES = ("last", "best")
# Confusion counts and the threshold itself aren't scores - left out of the best-vs-last comparison.
_NON_SCORE_KEYS = {"TP", "FN", "TN", "FP", "Total_Anomalous", "Total_Normal", "custom_threshold"}
_HEADLINE_METRICS = ("image_AUROC", "pixel_AUROC", "image_F1Max", "pixel_F1Max", "image_AUPR", "custom_F1_score")


def _build_engine(args, monitor_metric, logger, output_path, trainer_config: Dict[str, Any]):
    """Returns (engine, rearrange_cb, raw_data_cb, file_logging_cb, resource_cb, ckpt_cb)."""
    tb_logger = AnomalibTensorBoardLogger(save_dir=str(output_path), name="tensorboard_logs", version="")

    rearrange_cb = RearrangeVisualizationsCallback(output_path=output_path, subfolder="test", logger=logger, model_name=args.model)
    raw_data_cb = RawDataExtractionCallback(
        output_path=output_path, subfolder="test", logger=logger, compact=getattr(args, "compact_raw_outputs", False),
    )
    file_logging_cb = FileLoggingCallback(logger=logger)
    resource_cb = None if args.disable_resource_monitor else ResourceMonitorCallback(output_path=output_path, logger=logger)

    monitor_mode = "max" if "loss" not in monitor_metric else "min"
    # Like min/max_epochs, a model's yaml `trainer.patience` wins over the --patience flag.
    patience = trainer_config.get("patience", args.patience)
    if patience != args.patience:
        logger.info(f"[early_stopping] Using config patience {patience} (overrides --patience {args.patience}).")
    callbacks = [
        EarlyStopping(monitor=monitor_metric, mode=monitor_mode, patience=patience, verbose=True),
        file_logging_cb,
        rearrange_cb,
        raw_data_cb,
    ]
    if resource_cb is not None:
        callbacks.append(resource_cb)
    if args.max_time:
        callbacks.append(TrainingTimeLimitCallback(parse_max_time(args.max_time), logger))
        logger.info(f"[max_time] Training will stop after {parse_max_time(args.max_time)} of wall-clock time.")

    # best.ckpt = best-scoring epoch on the metric early stopping watches; last.ckpt = most
    # recent validated epoch. Both are evaluated after fit (see _resolve_checkpoints).
    # Having our own ModelCheckpoint also stops Engine adding its default last-epoch one.
    # With --no_checkpoint both still have to hit disk to be loaded, so they go to a temp
    # dir that run() deletes after evaluation; nothing is left in the results.
    if args.no_checkpoint:
        ckpt_dir = Path(tempfile.mkdtemp(prefix="ckpt_"))
        logger.info("[checkpoint] --no_checkpoint set: best/last epochs are kept in a temp dir only, deleted after evaluation.")
    else:
        ckpt_dir = output_path / "weights" / "lightning"
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename="best", monitor=monitor_metric, mode=monitor_mode,
        save_top_k=1, save_last=True, auto_insert_metric_name=False,
    )
    callbacks.append(ckpt_cb)

    engine = Engine(
        callbacks=callbacks, logger=tb_logger,
        min_epochs=trainer_config.get("min_epochs", args.min_epochs),
        max_epochs=trainer_config.get("max_epochs", args.max_epochs),
        accelerator=args.accelerator, devices=args.devices,
        default_root_dir=str(output_path),
    )
    return engine, rearrange_cb, raw_data_cb, file_logging_cb, resource_cb, ckpt_cb


def _read_checkpoint(path: str | Path) -> Dict[str, Any]:
    # mmap: reading just the epoch out of a multi-GB checkpoint shouldn't load all of it.
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _resolve_checkpoints(engine, ckpt_cb, monitor_metric: str, args, logger) -> Dict[str, Any]:
    """Works out which epochs last.ckpt and best.ckpt hold, whether they are the same epoch,
    and which one is primary. Returns the checkpoint_selection block for manifest.json."""
    epochs_trained = int(engine.trainer.current_epoch)
    final_epoch = max(epochs_trained - 1, 0)
    callbacks = engine.trainer.callbacks
    time_cb = next((c for c in callbacks if isinstance(c, TrainingTimeLimitCallback)), None)
    early_cb = next((c for c in callbacks if isinstance(c, EarlyStopping)), None)
    if time_cb is not None and time_cb.fired:
        stop_reason = "max_time"
    elif early_cb is not None and early_cb.stopped_epoch > 0:
        stop_reason = "early_stopping"
    else:
        stop_reason = "max_epochs"  # the trainer's or the model's own (e.g. single-pass models)

    checkpoints: Dict[str, Dict[str, Any]] = {}
    for name, path in (("last", ckpt_cb.last_model_path), ("best", ckpt_cb.best_model_path)):
        if path and Path(path).is_file():
            checkpoints[name] = {"available": True, "source": "file", "path": str(path),
                                 "epoch": int(_read_checkpoint(path)["epoch"]), "identical_to": None}
        else:
            checkpoints[name] = {"available": False, "source": None, "path": None, "epoch": None, "identical_to": None}

    last, best = checkpoints["last"], checkpoints["best"]
    if not last["available"]:
        # Nothing was saved at all - a zero-shot model (validate only) or a single-pass model whose
        # fit takes no optimizer step - so the weights in memory ARE the last (and only) epoch.
        last.update(available=True, source="in_memory", epoch=final_epoch)
        if not best["available"]:
            best.update(available=True, source="in_memory", epoch=final_epoch)
    if best["available"] and best["epoch"] == last["epoch"]:
        best["identical_to"] = "last"
    elif not best["available"]:
        best["reason"] = f"no best checkpoint was saved ('{monitor_metric}' never logged?)"
        logger.warning(f"[checkpoint] {best['reason']} - only the last epoch will be evaluated.")

    primary = args.primary_checkpoint
    if primary == "best" and not best["available"]:
        logger.warning("[checkpoint] --primary_checkpoint best, but there is no best checkpoint - using last as primary.")
        primary = "last"
    if best["identical_to"] and primary == "best":
        # Same weights either way; keep a single evaluated folder and point the other at it.
        best["identical_to"] = None
        last["identical_to"] = "best"

    last_is_final_epoch = last["epoch"] == final_epoch
    if not last_is_final_epoch:
        logger.warning(
            f"[checkpoint] last.ckpt is epoch {last['epoch']}, but training ran into epoch {final_epoch} "
            f"(stopped by {stop_reason}) without validating it - 'last' means the last VALIDATED epoch."
        )
    best_score = ckpt_cb.best_model_score
    score_txt = f" ({monitor_metric}={float(best_score):.4f} on val)" if best_score is not None else ""
    logger.info(
        f"[checkpoint] last = epoch {last['epoch']}, best = epoch {best['epoch']}{score_txt}; "
        f"{epochs_trained} epochs trained (stopped by {stop_reason}); primary checkpoint: {primary}."
    )
    return {
        "monitor": monitor_metric,
        "epochs_trained": epochs_trained,
        "stop_reason": stop_reason,
        "requested_primary": args.primary_checkpoint,
        "primary": primary,
        "last_epoch": last["epoch"],
        "best_epoch": best["epoch"],
        "best_score": float(best_score) if best_score is not None else None,
        "last_is_final_epoch": last_is_final_epoch,
        "checkpoints": checkpoints,
    }


def _apply_checkpoint(model, info: Dict[str, Any], logger, name: str) -> None:
    """Load a checkpoint's weights into `model` in place. The state_dict also carries the
    post-processor threshold/normalization computed in that same epoch's validation, so
    weights and threshold stay consistent."""
    if info["source"] != "file":
        return  # in-memory weights are already loaded
    model.load_state_dict(_read_checkpoint(info["path"])["state_dict"])
    logger.info(f"[checkpoint] Loaded {name} checkpoint (epoch {info['epoch']}) from {info['path']}")


@contextmanager
def _images_disabled(model):
    """Stop anomalib's visualizer writing images for the duration of a test pass. The model's
    visualizer instance is the same object Lightning attaches as a callback, so shadowing its
    hook on the instance is enough; deleting the shadow restores the class method."""
    visualizer = getattr(model, "visualizer", None)
    if visualizer is None:
        yield
        return
    visualizer.on_test_batch_end = lambda *args, **kwargs: None
    try:
        yield
    finally:
        del visualizer.on_test_batch_end


def _write_checkpoint_outputs(out_dir: Path, result: Dict[str, Any], logger) -> Dict[str, pd.DataFrame]:
    """predictions.csv, raw_outputs/<split>/index.csv and analysis/per_image_stats.csv for one
    checkpoint's folder. Returns per_image_stats_by_split for the analysis step."""
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_by_split: Dict[str, pd.DataFrame] = result["predictions"]

    # ---------------------------------------------------------------------
    # Consolidated predictions.csv - all splits, one Split column, built
    # directly from what RearrangeVisualizationsCallback already collected
    # in memory (no separate {model}_{split}_predictions.csv files, no
    # re-reading CSVs back off disk just to merge them).
    # ---------------------------------------------------------------------
    if predictions_by_split:
        combined_frames = []
        for split_name, split_df in predictions_by_split.items():
            tagged = split_df.copy()
            tagged.insert(0, "Split", split_name.capitalize())
            combined_frames.append(tagged)
        pd.concat(combined_frames, ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)
        logger.info(f"Exported consolidated predictions CSV to: {out_dir / 'predictions.csv'}")

    # ---------------------------------------------------------------------
    # raw_outputs/<split>/index.csv - the same per-split predictions slice,
    # copied alongside that split's arrays.npz so the folder is self-sufficient:
    # download raw_outputs/test/ alone and you have the arrays AND the labels/
    # scores/threshold tying each array row to an image, with nothing to fetch
    # from the parent directory.
    # ---------------------------------------------------------------------
    for split_name, split_df in predictions_by_split.items():
        raw_split_dir = out_dir / "raw_outputs" / split_name
        if raw_split_dir.exists():
            split_df.to_csv(raw_split_dir / "index.csv", index=False)

    # ---------------------------------------------------------------------
    # Consolidated analysis/per_image_stats.csv - segmentation pixel stats
    # (IoU/Dice, in-map score peaks, anomalous-area %) joined with the
    # matching split's predictions on Image_Name. Only produced for splits
    # where RawDataExtractionCallback actually saw an anomaly_map.
    # ---------------------------------------------------------------------
    per_image_stats_by_split: Dict[str, pd.DataFrame] = {}
    for split_name, pixel_df in result["pixel_stats"].items():
        preds_df = predictions_by_split.get(split_name)
        if preds_df is None:
            continue
        merged = preds_df.merge(pixel_df, on="Image_Name", how="left")
        per_image_stats_by_split[split_name] = merged.rename(columns={"Anomaly_Score": "Pred_Score"})

    if per_image_stats_by_split:
        analysis_dir = out_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        combined_stats_frames = []
        for split_name, stats_df in per_image_stats_by_split.items():
            tagged = stats_df.copy()
            tagged.insert(0, "Split", split_name.capitalize())
            combined_stats_frames.append(tagged)
        pd.concat(combined_stats_frames, ignore_index=True).to_csv(analysis_dir / "per_image_stats.csv", index=False)
    return per_image_stats_by_split


def _compare_checkpoints(metrics_json: Dict[str, Any]) -> Dict[str, Any]:
    """{split: {metric: {"last", "best", "best_minus_last"}}} for every score both checkpoints have."""
    comparison: Dict[str, Any] = {}
    for split in ("test", "val"):
        last_m, best_m = metrics_json["last"].get(split), metrics_json["best"].get(split)
        if not last_m or not best_m:
            continue
        rows = {}
        for key in sorted(set(last_m) & set(best_m) - _NON_SCORE_KEYS):
            try:
                last_v, best_v = float(last_m[key]), float(best_m[key])
            except (TypeError, ValueError):
                continue
            rows[key] = {"last": last_v, "best": best_v, "best_minus_last": best_v - last_v}
        comparison[split] = rows
    return comparison


def _log_checkpoint_comparison(logger, checkpoint_selection: Dict[str, Any], metrics_json: Dict[str, Any]) -> None:
    ckpts = checkpoint_selection["checkpoints"]
    logger.info("=======================================================")
    logger.info("   BEST vs LAST CHECKPOINT  ")
    logger.info("=======================================================")
    for name in CHECKPOINT_NAMES:
        info = ckpts[name]
        role = "primary" if name == checkpoint_selection["primary"] else "secondary"
        detail = f"same weights as {info['identical_to']}" if info.get("identical_to") else (
            "not available" if not info["available"] else f"evaluated, outputs in {name}/")
        logger.info(f"[checkpoint] {name:<4}: epoch {info['epoch']} ({role}; {detail})")
    comparison = metrics_json.get("comparison") or {}
    if not comparison:
        logger.info("[checkpoint] No best-vs-last comparison (only one distinct checkpoint was evaluated).")
        return
    for split, rows in comparison.items():
        for key in _HEADLINE_METRICS:
            if key in rows:
                r = rows[key]
                logger.info(
                    f"[checkpoint] {split:<4} {key:<16} last={r['last']:.4f}  best={r['best']:.4f}  "
                    f"best-last={r['best_minus_last']:+.4f}"
                )


def run(
    *,
    args,
    model,
    datamodule,
    DataClass,
    filtered_kwargs: Dict[str, Any],
    exclude_files: set,
    dataset_stats: Dict[str, Any],
    monitor_metric: str,
    logger,
    output_path: Path,
    config_path: str | None,
    experiment_summary: Dict[str, Any] | None,
    target_h: int | None,
    target_w: int | None,
    trainer_config: Dict[str, Any] | None = None,
) -> None:
    engine, rearrange_cb, raw_data_cb, file_logging_cb, resource_cb, ckpt_cb = _build_engine(
        args, monitor_metric, logger, output_path, trainer_config or {},
    )

    run_started_at = datetime.now(timezone.utc)
    durations_sec: Dict[str, Any] = {}
    throughput_images_per_sec: Dict[str, Any] = {}

    def evaluate(name: str, primary: bool) -> Dict[str, Any]:
        """Test (+ val with --eval_val_test, + contamination for the primary) passes on whatever
        weights are loaded now, with this checkpoint's outputs going to output_path/<name>/.
        Only the primary pass writes images and raw_outputs arrays."""
        out_dir = output_path / name
        rearrange_cb.predictions_by_split = {}
        rearrange_cb.thresholds_by_split = {}
        rearrange_cb.move_images = primary
        rearrange_cb.images_dir = out_dir / "images"
        raw_data_cb.pixel_stats_by_split = {}
        raw_data_cb.output_path = out_dir
        raw_data_cb.save_arrays = primary
        result: Dict[str, Any] = {
            "metrics": {"test": None, "val": None, "contamination": None},
            "durations": {}, "throughput": {}, "contamination_pool_size": None,
        }

        def run_pass(split: str, n_images: int | None, **test_kwargs) -> None:
            rearrange_cb.subfolder = split
            raw_data_cb.subfolder = split
            if resource_cb is not None:
                resource_cb.stage = f"{split}_{name}"
            start = time.time()
            results = engine.test(model=model, **test_kwargs)
            elapsed = time.time() - start
            result["durations"][split] = elapsed
            if elapsed > 0 and n_images:
                result["throughput"][split] = n_images / elapsed
            metrics = dict(results[0]) if results else {}
            stats_path = output_path / f".tmp_{args.model}_custom_stats_{split}.json"
            if stats_path.exists():
                with open(stats_path, "r") as f: metrics.update(json.load(f))
                os.remove(stats_path)
            result["metrics"][split] = metrics
            logger.info(f"[{name}] {split.capitalize()} Pass Metrics: {metrics}")

        with nullcontext() if primary else _images_disabled(model):
            # --- 1. NORMAL TEST (Test Set Only) ---
            logger.info(f"Starting Standard Test ({name} checkpoint)...")
            n_test = len(datamodule.test_data) if getattr(datamodule, "test_data", None) is not None else 0
            run_pass("test", n_test, datamodule=datamodule)

            # --- 2. VALIDATION SET EVALUATION PASS ---
            if args.eval_val_test:
                logger.info("=======================================================")
                logger.info(f"   STARTING EVALUATION PASS ON VALIDATION SET ({name} checkpoint)  ")
                logger.info("=======================================================")
                logger.info(f"   Validation images to check: {len(datamodule.val_data)}")
                run_pass("val", len(datamodule.val_data), dataloaders=datamodule.val_dataloader())

            # --- 3. CONTAMINATION CHECK (primary checkpoint only) ---
            if args.check_contamination and primary:
                logger.info("=======================================================")
                logger.info(f"   STARTING CONTAMINATION CHECK (Scanning ALL Normal Data, {name} checkpoint)  ")
                logger.info("=======================================================")
                contamination_loader, pool_size = dataset_model_setup.build_contamination_datamodule(
                    args, DataClass, filtered_kwargs, exclude_files, datamodule, logger,
                )
                result["contamination_pool_size"] = pool_size
                run_pass("contamination", pool_size, dataloaders=contamination_loader)

        result["predictions"] = dict(rearrange_cb.predictions_by_split)
        result["thresholds"] = dict(rearrange_cb.thresholds_by_split)
        result["pixel_stats"] = dict(raw_data_cb.pixel_stats_by_split)
        return result

    try:
        logger.info("Starting Fit...")
        if resource_cb is not None:
            resource_cb.stage = "fit"
        fit_start = time.time()
        engine.fit(model=model, datamodule=datamodule)
        durations_sec["fit"] = time.time() - fit_start
        checkpoint_selection = _resolve_checkpoints(engine, ckpt_cb, monitor_metric, args, logger)
        # The epoch limits that actually applied: config values win over the CLI flags, and the
        # model's own trainer_arguments (e.g. max_epochs=1 for single-pass models) win over both.
        checkpoint_selection["effective_epochs"] = {
            "min_epochs": (trainer_config or {}).get("min_epochs", args.min_epochs),
            "max_epochs": engine.trainer.max_epochs,
            "patience": (trainer_config or {}).get("patience", args.patience),
            "max_time": args.max_time,
        }
        ckpts = checkpoint_selection["checkpoints"]
        primary = checkpoint_selection["primary"]
        secondary = "best" if primary == "last" else "last"

        results: Dict[str, Dict[str, Any]] = {}
        try:
            # ---------------------------------------------------------------------
            # Primary checkpoint: every output, images and raw arrays included, plus contamination.
            # Secondary checkpoint: test (+ val) predictions, metrics and analysis data only.
            # Identical epochs (single-pass models, or best == last) are evaluated once.
            # ---------------------------------------------------------------------
            _apply_checkpoint(model, ckpts[primary], logger, primary)
            visualizer = getattr(model, "visualizer", None)
            if visualizer is not None and hasattr(visualizer, "output_dir"):
                visualizer.output_dir = output_path / primary / "images"
            results[primary] = evaluate(primary, primary=True)

            sec_info = ckpts[secondary]
            if sec_info.get("identical_to"):
                logger.info(f"[checkpoint] {secondary} is the same epoch as {primary} ({sec_info['epoch']}) - not evaluated twice.")
            elif sec_info["available"]:
                # In-memory primary weights have no file to reload from, so keep a copy first.
                primary_state = None if ckpts[primary]["source"] == "file" else {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                _apply_checkpoint(model, sec_info, logger, secondary)
                try:
                    results[secondary] = evaluate(secondary, primary=False)
                except Exception as e:  # noqa: BLE001
                    # Never lose the primary results to a failure in the comparison pass.
                    sec_info["error"] = str(e)
                    logger.error(f"[checkpoint] Evaluating the {secondary} checkpoint failed (non-fatal): {e}")
                if primary_state is not None:
                    model.load_state_dict(primary_state)
                else:
                    _apply_checkpoint(model, ckpts[primary], logger, primary)  # export uses the primary weights

            # ---------------------------------------------------------------------
            # Export (primary checkpoint weights)
            # ---------------------------------------------------------------------
            if args.export_types:
                for ext in args.export_types:
                    try:
                        engine.export(model=model, export_type=ExportType[ext.upper()], export_root=str(output_path / "weights"))
                    except Exception as e:
                        logger.error(f"Failed to export {ext}. Reason: {e}")
        finally:
            if args.no_checkpoint:
                shutil.rmtree(ckpt_cb.dirpath, ignore_errors=True)

        for name, res in results.items():
            durations_sec[name] = res["durations"]
            throughput_images_per_sec[name] = res["throughput"]
        for name, info in ckpts.items():
            info["outputs_dir"] = name if name in results else info.get("identical_to")

        # ---------------------------------------------------------------------
        # metrics.json - one section per checkpoint, explicitly named, each with
        # test/val/contamination present (as null) or a full metrics dict. A
        # checkpoint with the same weights as the other carries a copy of its
        # metrics, so both sections can always be read the same way.
        # ---------------------------------------------------------------------
        metrics_json: Dict[str, Any] = {"primary": primary, "monitor": monitor_metric}
        for name in CHECKPOINT_NAMES:
            info = ckpts[name]
            source = name if name in results else info.get("identical_to")
            section: Dict[str, Any] = {
                "epoch": info["epoch"],
                "is_primary": name == primary,
                "evaluated": name in results,
                "identical_to": info.get("identical_to"),
                "outputs_dir": info.get("outputs_dir"),
            }
            section.update(results[source]["metrics"] if source in results else {"test": None, "val": None, "contamination": None})
            if name == "best":
                section["selection_note"] = (
                    f"Epoch chosen by the highest val {monitor_metric}, so this section's val metrics "
                    f"are optimistically biased; its test metrics are not."
                )
            metrics_json[name] = section
        metrics_json["comparison"] = _compare_checkpoints(metrics_json) if len(results) == 2 else {}

        metrics_dir = output_path / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_dir / "metrics.json", "w") as f:
            json.dump(metrics_json, f, indent=4, default=str)
        _log_checkpoint_comparison(logger, checkpoint_selection, metrics_json)

        per_image_stats: Dict[str, Dict[str, pd.DataFrame]] = {
            name: _write_checkpoint_outputs(output_path / name, res, logger) for name, res in results.items()
        }

        # ---------------------------------------------------------------------
        # epoch_metrics.csv - structured per-epoch training curve, written
        # directly instead of needing to be regexed back out of the log later.
        # ---------------------------------------------------------------------
        epoch_df = pd.DataFrame(file_logging_cb.epoch_rows) if file_logging_cb.epoch_rows else pd.DataFrame()
        if not epoch_df.empty:
            epoch_df.to_csv(output_path / "epoch_metrics.csv", index=False)

        # ---------------------------------------------------------------------
        # resource_usage/samples.csv - raw CPU/RAM/GPU samples from ResourceMonitorCallback,
        # plus a per-stage mean/max summary folded into manifest.json below.
        # ---------------------------------------------------------------------
        resource_df = pd.DataFrame()
        resource_summary: Dict[str, Any] = {}
        if resource_cb is not None and resource_cb.samples:
            try:
                resource_df = pd.DataFrame(resource_cb.samples)
                resource_dir = output_path / "resource_usage"
                resource_dir.mkdir(parents=True, exist_ok=True)
                resource_df.to_csv(resource_dir / "samples.csv", index=False)
                numeric_cols = resource_df.select_dtypes("number").columns.drop("elapsed_sec", errors="ignore")
                resource_summary = {
                    stage: {col: {"mean": float(g[col].mean()), "max": float(g[col].max())} for col in numeric_cols}
                    for stage, g in resource_df.groupby("stage")
                }
            except Exception as e:
                logger.error(f"[resource_monitor] Failed to write resource_usage/samples.csv (non-fatal): {e}")

        # ---------------------------------------------------------------------
        # Split composition, for the one-per-run chart + manifest.
        # ---------------------------------------------------------------------
        split_counts: Dict[str, Dict[str, int]] = {}
        if dataset_stats:
            if dataset_stats.get("train", {}).get("total"):
                split_counts["Train"] = dataset_stats["train"]
            if dataset_stats.get("val", {}).get("total"):
                split_counts["Validation"] = dataset_stats["val"]
            if dataset_stats.get("test", {}).get("total"):
                split_counts["Test"] = dataset_stats["test"]
        contamination_pool_size = results[primary]["contamination_pool_size"]
        if contamination_pool_size:
            split_counts["Contamination"] = {
                "total": contamination_pool_size, "normal": contamination_pool_size, "anomalous": 0,
            }

        # ---------------------------------------------------------------------
        # manifest.json - run-level provenance that previously only existed as
        # unstructured lines in the .log file.
        # ---------------------------------------------------------------------
        try:
            run_ended_at = datetime.now(timezone.utc)
            lightning_ckpt_bytes = sum(f.stat().st_size for f in output_path.rglob("*.ckpt") if f.is_file())
            exported_weights_bytes = dir_size_bytes(output_path / "weights")
            manifest = {
                "model": args.model,
                "dataset": args.dataset,
                "category": args.category,
                "task": args.task,
                "seed": args.seed,
                "image_size": {"height": target_h, "width": target_w},
                "config_path": config_path,
                "args": vars(args),
                "experiment_file": experiment_summary,
                "dataset_stats": dataset_stats,
                "timestamps": {"start": run_started_at.isoformat(), "end": run_ended_at.isoformat()},
                "checkpoint_selection": checkpoint_selection,
                "durations_sec": durations_sec,
                "throughput_images_per_sec": throughput_images_per_sec,
                "resource_usage": resource_summary,
                "checkpoint_sizes_bytes": {
                    "lightning_checkpoint_total": lightning_ckpt_bytes,
                    "exported_weights_total": exported_weights_bytes,
                },
                "git_commit": get_git_commit(),
                "package_versions": {"torch": torch.__version__},
            }
            with open(output_path / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"[manifest] Failed to write manifest.json (non-fatal): {e}")

        # ---------------------------------------------------------------------
        # Static analysis charts (matplotlib/seaborn) + the JSON/CSV backing
        # each one, per evaluated checkpoint - charts for the primary only.
        # Never allowed to fail the run - a plotting bug must not cost hours
        # of already-completed training/testing.
        # ---------------------------------------------------------------------
        for name, res in results.items():
            try:
                run_post_test_analysis(
                    output_path=output_path / name,
                    predictions_by_split=res["predictions"],
                    thresholds_by_split=res["thresholds"],
                    per_image_stats_by_split=per_image_stats[name],
                    split_counts=split_counts,
                    epoch_df=epoch_df,
                    resource_df=resource_df,
                    make_charts=name == primary,
                )
                logger.info(f"Analysis {'charts + ' if name == primary else ''}summary written to: {output_path / name / 'analysis'}")
            except Exception as e:
                logger.error(f"[analysis] Post-test analysis for {name} failed (non-fatal): {e}")

        logger.info(f"Experiment finished. Results in {output_path}")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise e
