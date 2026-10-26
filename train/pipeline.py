"""Runs one full experiment: callbacks/engine construction, fit -> test -> val
-> contamination -> export, output consolidation (metrics.json, predictions.csv,
per_image_stats.csv, epoch_metrics.csv), manifest.json, and the post-test
analysis charts. Called from cli.main() once the datamodule and model are built.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
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


def _build_engine(args, monitor_metric, logger, output_path, trainer_config: Dict[str, Any]):
    """Returns (engine, rearrange_cb, raw_data_cb, file_logging_cb, resource_cb, best_cb)."""
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

    # Keep only the best-scoring epoch on the same metric early stopping watches, so it can
    # be restored after fit (see _restore_best_checkpoint). Without this, everything after
    # fit - test, val, contamination, export - used the LAST epoch: best + `patience`.
    # Having our own ModelCheckpoint also stops Engine adding its default last-epoch one.
    # With --no_checkpoint the best epoch still has to hit disk to be restored, so it goes
    # to a temp dir that _restore_best_checkpoint deletes; nothing is left in the results.
    if args.no_checkpoint:
        ckpt_dir = Path(tempfile.mkdtemp(prefix="best_ckpt_"))
        logger.info("[checkpoint] --no_checkpoint set: best epoch is kept in a temp file only, deleted after restoring.")
    else:
        ckpt_dir = output_path / "weights" / "lightning"
    best_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename="model", monitor=monitor_metric, mode=monitor_mode,
        save_top_k=1, save_last=False, auto_insert_metric_name=False,
    )
    callbacks.append(best_cb)

    engine = Engine(
        callbacks=callbacks, logger=tb_logger,
        min_epochs=trainer_config.get("min_epochs", args.min_epochs),
        max_epochs=trainer_config.get("max_epochs", args.max_epochs),
        accelerator=args.accelerator, devices=args.devices,
        default_root_dir=str(output_path),
    )
    return engine, rearrange_cb, raw_data_cb, file_logging_cb, resource_cb, best_cb


def _restore_best_checkpoint(model, engine, best_cb, monitor_metric: str, args, logger) -> Dict[str, Any]:
    """Load the best epoch's weights into `model` in place, so every later step (test,
    val pass, contamination, export) uses them. The checkpoint's state_dict also carries
    the post-processor threshold/normalization computed in that same epoch's validation,
    so weights and threshold stay consistent. Returns a summary for manifest.json."""
    epochs_trained = int(engine.trainer.current_epoch)
    callbacks = engine.trainer.callbacks
    time_cb = next((c for c in callbacks if isinstance(c, TrainingTimeLimitCallback)), None)
    early_cb = next((c for c in callbacks if isinstance(c, EarlyStopping)), None)
    if time_cb is not None and time_cb.fired:
        stop_reason = "max_time"
    elif early_cb is not None and early_cb.stopped_epoch > 0:
        stop_reason = "early_stopping"
    else:
        stop_reason = "max_epochs"  # the trainer's or the model's own (e.g. single-pass models)
    summary: Dict[str, Any] = {"monitor": monitor_metric, "epochs_trained": epochs_trained, "stop_reason": stop_reason,
                               "restored": False, "best_epoch": None, "best_score": None}
    best_path = best_cb.best_model_path
    try:
        if not best_path or not Path(best_path).is_file():
            logger.warning(
                f"[checkpoint] No best checkpoint was saved ('{monitor_metric}' never logged?) - "
                f"evaluating the final weights (epoch {epochs_trained - 1})."
            )
            return summary
        checkpoint = torch.load(best_path, map_location=model.device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        best_score = best_cb.best_model_score
        summary.update(restored=True, best_epoch=int(checkpoint["epoch"]),
                       best_score=float(best_score) if best_score is not None else None)
        logger.info(
            f"[checkpoint] Restored best epoch {summary['best_epoch']} ({monitor_metric}={summary['best_score']:.4f}) "
            f"out of {epochs_trained} trained (stopped by {stop_reason}); test/val/contamination/export use these weights."
        )
    finally:
        if args.no_checkpoint:
            shutil.rmtree(best_cb.dirpath, ignore_errors=True)
    return summary


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
    engine, rearrange_cb, raw_data_cb, file_logging_cb, resource_cb, best_cb = _build_engine(
        args, monitor_metric, logger, output_path, trainer_config or {},
    )

    run_started_at = datetime.now(timezone.utc)
    durations_sec: Dict[str, float] = {}
    throughput_images_per_sec: Dict[str, float] = {}
    all_metrics: Dict[str, Any] = {"test": None, "val": None, "contamination": None}
    contamination_pool_size = None

    try:
        logger.info("Starting Fit...")
        if resource_cb is not None:
            resource_cb.stage = "fit"
        fit_start = time.time()
        engine.fit(model=model, datamodule=datamodule)
        durations_sec["fit"] = time.time() - fit_start
        checkpoint_selection = _restore_best_checkpoint(model, engine, best_cb, monitor_metric, args, logger)
        # The epoch limits that actually applied: config values win over the CLI flags, and the
        # model's own trainer_arguments (e.g. max_epochs=1 for single-pass models) win over both.
        checkpoint_selection["effective_epochs"] = {
            "min_epochs": (trainer_config or {}).get("min_epochs", args.min_epochs),
            "max_epochs": engine.trainer.max_epochs,
            "patience": (trainer_config or {}).get("patience", args.patience),
            "max_time": args.max_time,
        }

        # --- 1. NORMAL TEST (Test Set Only) ---
        logger.info("Starting Standard Test...")
        if resource_cb is not None:
            resource_cb.stage = "test"
        test_start = time.time()
        test_results = engine.test(model=model, datamodule=datamodule)
        durations_sec["test"] = time.time() - test_start
        n_test_images = len(datamodule.test_data) if getattr(datamodule, "test_data", None) is not None else 0
        if durations_sec["test"] > 0 and n_test_images:
            throughput_images_per_sec["test"] = n_test_images / durations_sec["test"]

        temp_stats_path = output_path / f".tmp_{args.model}_custom_stats_test.json"
        if temp_stats_path.exists():
            with open(temp_stats_path, "r") as f: custom_stats = json.load(f)
            os.remove(temp_stats_path)
            if isinstance(test_results, list): test_results[0].update(custom_stats)
            elif isinstance(test_results, dict): test_results.update(custom_stats)
        all_metrics["test"] = test_results[0] if isinstance(test_results, list) else test_results

        # --- 2. VALIDATION SET EVALUATION PASS ---
        if args.eval_val_test:
            logger.info("=======================================================")
            logger.info("   STARTING EVALUATION PASS ON VALIDATION SET  ")
            logger.info("=======================================================")

            rearrange_cb.subfolder = "val"
            raw_data_cb.subfolder = "val"
            if resource_cb is not None:
                resource_cb.stage = "val"

            logger.info(f"   Validation images to check: {len(datamodule.val_data)}")

            val_start = time.time()
            val_results = engine.test(model=model, dataloaders=datamodule.val_dataloader())
            durations_sec["val"] = time.time() - val_start
            n_val_images = len(datamodule.val_data)
            if durations_sec["val"] > 0 and n_val_images:
                throughput_images_per_sec["val"] = n_val_images / durations_sec["val"]
            logger.info(f"Validation Pass Metrics: {val_results}")

            temp_stats_path_val = output_path / f".tmp_{args.model}_custom_stats_val.json"
            if temp_stats_path_val.exists():
                with open(temp_stats_path_val, "r") as f: val_stats = json.load(f)
                os.remove(temp_stats_path_val)
                if isinstance(val_results, list): val_results[0].update(val_stats)
                elif isinstance(val_results, dict): val_results.update(val_stats)
            all_metrics["val"] = val_results[0] if isinstance(val_results, list) else val_results
            # predictions.csv and metrics.json are written once, consolidated across every
            # split that ran, after the contamination check below - see that section.

        # --- 3. CONTAMINATION CHECK ---
        if args.check_contamination:
            logger.info("=======================================================")
            logger.info("   STARTING CONTAMINATION CHECK (Scanning ALL Normal Data)  ")
            logger.info("=======================================================")

            rearrange_cb.subfolder = "contamination"
            raw_data_cb.subfolder = "contamination"
            if resource_cb is not None:
                resource_cb.stage = "contamination"

            contamination_loader, contamination_pool_size = dataset_model_setup.build_contamination_datamodule(
                args, DataClass, filtered_kwargs, exclude_files, datamodule, logger,
            )

            contam_start = time.time()
            contamination_results = engine.test(model=model, dataloaders=contamination_loader)
            durations_sec["contamination"] = time.time() - contam_start
            if durations_sec["contamination"] > 0 and contamination_pool_size:
                throughput_images_per_sec["contamination"] = contamination_pool_size / durations_sec["contamination"]

            # Persist contamination results - previously computed (engine.test's return value
            # was discarded) and never written anywhere; the RearrangeVisualizationsCallback's
            # staged .tmp_*_custom_stats_contamination.json was also never read or cleaned up.
            temp_stats_path_contam = output_path / f".tmp_{args.model}_custom_stats_contamination.json"
            if temp_stats_path_contam.exists():
                with open(temp_stats_path_contam, "r") as f: contam_stats = json.load(f)
                os.remove(temp_stats_path_contam)
                if isinstance(contamination_results, list): contamination_results[0].update(contam_stats)
                elif isinstance(contamination_results, dict): contamination_results.update(contam_stats)
            all_metrics["contamination"] = (
                contamination_results[0] if isinstance(contamination_results, list) else contamination_results
            )
            logger.info(f"Contamination Pass Metrics: {all_metrics['contamination']}")

        # ---------------------------------------------------------------------
        # Consolidated metrics.json - test/val/contamination in one file, each
        # key present (as null) or a full metrics dict, instead of separate
        # {model}_metrics.json / {model}_metrics_val.json files and a silently
        # discarded contamination result.
        # ---------------------------------------------------------------------
        metrics_dir = output_path / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_dir / "metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=4, default=str)

        # ---------------------------------------------------------------------
        # Consolidated predictions.csv - all splits, one Split column, built
        # directly from what RearrangeVisualizationsCallback already collected
        # in memory (no separate {model}_{split}_predictions.csv files, no
        # re-reading CSVs back off disk just to merge them).
        # ---------------------------------------------------------------------
        predictions_by_split: Dict[str, pd.DataFrame] = dict(rearrange_cb.predictions_by_split)
        thresholds_by_split: Dict[str, float] = dict(rearrange_cb.thresholds_by_split)
        if predictions_by_split:
            combined_frames = []
            for split_name, split_df in predictions_by_split.items():
                tagged = split_df.copy()
                tagged.insert(0, "Split", split_name.capitalize())
                combined_frames.append(tagged)
            combined_predictions = pd.concat(combined_frames, ignore_index=True)
            combined_predictions.to_csv(output_path / "predictions.csv", index=False)
            logger.info(f"Exported consolidated predictions CSV to: {output_path / 'predictions.csv'}")

        # ---------------------------------------------------------------------
        # raw_outputs/<split>/index.csv - the same per-split predictions slice,
        # copied alongside that split's arrays.npz so the folder is self-sufficient:
        # download raw_outputs/test/ alone and you have the arrays AND the labels/
        # scores/threshold tying each array row to an image, with nothing to fetch
        # from the parent directory.
        # ---------------------------------------------------------------------
        for split_name, split_df in predictions_by_split.items():
            raw_split_dir = output_path / "raw_outputs" / split_name
            if raw_split_dir.exists():
                raw_split_dir.mkdir(parents=True, exist_ok=True)
                split_df.to_csv(raw_split_dir / "index.csv", index=False)

        # ---------------------------------------------------------------------
        # Consolidated analysis/per_image_stats.csv - segmentation pixel stats
        # (IoU/Dice, in-map score peaks, anomalous-area %) joined with the
        # matching split's predictions on Image_Name. Only produced for splits
        # where RawDataExtractionCallback actually saw an anomaly_map.
        # ---------------------------------------------------------------------
        pixel_stats_by_split: Dict[str, pd.DataFrame] = dict(raw_data_cb.pixel_stats_by_split)
        per_image_stats_by_split: Dict[str, pd.DataFrame] = {}
        for split_name, pixel_df in pixel_stats_by_split.items():
            preds_df = predictions_by_split.get(split_name)
            if preds_df is None:
                continue
            merged = preds_df.merge(pixel_df, on="Image_Name", how="left")
            merged = merged.rename(columns={"Anomaly_Score": "Pred_Score"})
            per_image_stats_by_split[split_name] = merged

        analysis_dir = output_path / "analysis"
        if per_image_stats_by_split:
            analysis_dir.mkdir(parents=True, exist_ok=True)
            combined_stats_frames = []
            for split_name, stats_df in per_image_stats_by_split.items():
                tagged = stats_df.copy()
                tagged.insert(0, "Split", split_name.capitalize())
                combined_stats_frames.append(tagged)
            pd.concat(combined_stats_frames, ignore_index=True).to_csv(
                analysis_dir / "per_image_stats.csv", index=False
            )

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
        if contamination_pool_size:
            split_counts["Contamination"] = {
                "total": contamination_pool_size, "normal": contamination_pool_size, "anomalous": 0,
            }

        # ---------------------------------------------------------------------
        # Export
        # ---------------------------------------------------------------------
        if args.export_types:
            for ext in args.export_types:
                try:
                    engine.export(model=model, export_type=ExportType[ext.upper()], export_root=str(output_path / "weights"))
                except Exception as e:
                    logger.error(f"Failed to export {ext}. Reason: {e}")

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
        # each one. Never allowed to fail the run - a plotting bug must not
        # cost hours of already-completed training/testing.
        # ---------------------------------------------------------------------
        try:
            run_post_test_analysis(
                output_path=output_path,
                predictions_by_split=predictions_by_split,
                thresholds_by_split=thresholds_by_split,
                per_image_stats_by_split=per_image_stats_by_split,
                split_counts=split_counts,
                epoch_df=epoch_df,
                resource_df=resource_df,
            )
            logger.info(f"Analysis charts + summary written to: {analysis_dir}")
        except Exception as e:
            logger.error(f"[analysis] Post-test analysis failed (non-fatal): {e}")

        logger.info(f"Experiment finished. Results in {output_path}")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise e
