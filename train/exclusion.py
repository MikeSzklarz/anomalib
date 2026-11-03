"""Train-only exclusion list (--exclude_list).

The datamodule splits ALL images first - same pool, same seed, so every image lands in
the same train/val/test split as in a run without a list - and only then are the listed
images dropped from train. Val and test keep them: they are still scored, and tagged
(In_Exclusion_List) so they can be filtered or relabeled offline without retraining.
The contamination pass is not filtered either, so every listed image gets a score.

Contrast with --exclude_root (setup.py), which drops files BEFORE the split and so
reshuffles every split: torch.randperm over a different-length pool is a different
permutation.

Outputs, in their own folder (output_path/exclusion/):
    exclusion_list.txt       verbatim copy of the list used (its sha256 is in the summary)
    excluded_samples.csv     one row per list entry: matched image, the split it was
                             assigned to, and what happened to it
    exclusion_summary.json   source, sha256, split counts before/after removal, per-split
                             breakdown, and whether every setup() call produced the same split
    <ckpt>_predictions.csv   predictions.csv rows for the listed images (val, test,
                             contamination) for each evaluated checkpoint
"""

from __future__ import annotations

import hashlib
import json
import shutil
import types
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

EXCLUSION_DIRNAME = "exclusion"
FLAG_COLUMN = "In_Exclusion_List"
# Split names match predictions.csv ("Val", not "Validation") so the files join directly.
_SPLIT_ATTRS = (("Train", "train_data"), ("Val", "val_data"), ("Test", "test_data"))
_ACTIONS = {"Train": "removed_from_train", "Val": "kept_in_val", "Test": "kept_in_test"}


def short_name(image_path) -> str:
    """'<parent folder>/<file name>' - the Image_Name form used in predictions.csv."""
    p = Path(str(image_path))
    return f"{p.parent.name}/{p.name}"


def _normalize_entry(entry: str) -> str:
    entry = entry.strip().replace("\\", "/")
    while entry.startswith("./"):
        entry = entry[2:]
    return entry


def _split_counts(dataset) -> Dict[str, int]:
    if dataset is None or not hasattr(dataset, "samples"):
        return {"total": 0, "normal": 0, "anomalous": 0}
    samples = dataset.samples
    if "label_index" in samples:
        normal = int((samples.label_index == 0).sum())
        anomalous = int((samples.label_index == 1).sum())
    else:
        normal, anomalous = len(samples), 0
    return {"total": len(samples), "normal": normal, "anomalous": anomalous}


def _normal_train_fraction(counts: Dict[str, Dict[str, int]]) -> float | None:
    normals = sum(counts[s]["normal"] for s in ("train", "val", "test"))
    return round(counts["train"]["normal"] / normals, 4) if normals else None


class TrainOnlyExclusion:
    """Loads the list, wraps datamodule.setup() so listed images are removed from train after
    every split, and writes the exclusion/ outputs. One instance per run."""

    def __init__(self, path: str | Path, logger):
        self.logger = logger
        self.source = Path(path)
        if not self.source.is_file():
            raise FileNotFoundError(f"--exclude_list file not found: {self.source}")
        raw = self.source.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()

        # One entry per line: a bare file name (T0127_030.jpg) or a path relative to the
        # dataset root (good/T0127_030.jpg). Blank lines and '#' comments are ignored.
        lines = [_normalize_entry(line) for line in raw.decode("utf-8").splitlines()]
        lines = [line for line in lines if line and not line.startswith("#")]
        self.entries: List[str] = list(dict.fromkeys(lines))
        self.duplicate_entries = sorted({line for line in lines if lines.count(line) > 1})

        self.record: pd.DataFrame | None = None
        self.counts_before: Dict[str, Dict[str, int]] | None = None
        self.counts_after: Dict[str, Dict[str, int]] | None = None
        self.setup_calls = 0
        self.split_stable = True
        self._filtered_train = None

        logger.info(f"[exclusion] Loaded {len(self.entries)} entries from {self.source} (sha256 {self.sha256[:12]}...)")
        logger.info("[exclusion] Mode: train-only. The split is computed on ALL images first, then listed "
                    "images are removed from train; val and test keep them.")
        if self.duplicate_entries:
            logger.warning(f"[exclusion] {len(self.duplicate_entries)} duplicate entries ignored: {self.duplicate_entries[:10]}")

    # ------------------------------------------------------------------
    # Datamodule hook
    # ------------------------------------------------------------------
    def attach(self, datamodule) -> None:
        """Wraps setup() rather than _setup(): anomalib's setup() runs _setup() (load all
        images) and then the seeded test/val splits, so filtering after it leaves the split
        untouched. setup() runs more than once per job (once to log split stats, again from
        trainer.fit, which re-splits from scratch), so the filter runs after every call."""
        original_setup = datamodule.setup
        exclusion = self

        def setup_then_exclude(_dm, stage=None):
            original_setup(stage)
            exclusion._exclude_from_train(_dm)

        datamodule.setup = types.MethodType(setup_then_exclude, datamodule)

    def _exclude_from_train(self, datamodule) -> None:
        train = getattr(datamodule, "train_data", None)
        if train is None or not hasattr(train, "samples"):
            return
        if train is self._filtered_train:
            return  # setup() was a no-op (already set up) - train is already filtered
        self.setup_calls += 1

        splits = {name: getattr(datamodule, attr, None) for name, attr in _SPLIT_ATTRS}
        counts_before = {name.lower(): _split_counts(ds) for name, ds in splits.items()}
        record = self._match(splits)

        removed_paths = set(record.loc[record.Action == "removed_from_train", "Image_Path"])
        if removed_paths:
            keep = ~train.samples.image_path.astype(str).isin(removed_paths)
            train.samples = train.samples[keep].reset_index(drop=True)
        self._filtered_train = train

        if self.record is not None:
            previous = set(zip(self.record.Image_Path, self.record.Assigned_Split))
            current = set(zip(record.Image_Path, record.Assigned_Split))
            if previous != current or counts_before != self.counts_before:
                self.split_stable = False
                self.logger.error(
                    f"[exclusion] setup() call #{self.setup_calls} produced a DIFFERENT split than the previous "
                    f"call - the logged split stats and dataset_splits.csv may not match what was trained on."
                )

        self.record = record
        self.counts_before = counts_before
        self.counts_after = {name.lower(): _split_counts(ds) for name, ds in splits.items()}
        self._log(record)

    def _match(self, splits: Dict[str, Any]) -> pd.DataFrame:
        by_name: Dict[str, List[Dict[str, Any]]] = {}
        for split_name, dataset in splits.items():
            if dataset is None or not hasattr(dataset, "samples"):
                continue
            samples = dataset.samples
            labels = samples.label_index if "label_index" in samples else pd.Series(0, index=samples.index)
            for image_path, label in zip(samples.image_path.astype(str), labels):
                p = Path(image_path)
                by_name.setdefault(p.name, []).append({
                    "split": split_name, "path": image_path, "posix": p.as_posix(), "label": int(label),
                })

        rows, ambiguous = [], {}
        for entry in self.entries:
            candidates = by_name.get(entry.rsplit("/", 1)[-1], [])
            if "/" in entry:
                candidates = [c for c in candidates if c["posix"] == entry or c["posix"].endswith("/" + entry)]
            if len(candidates) > 1:
                ambiguous[entry] = [short_name(c["path"]) for c in candidates]
                continue
            if not candidates:
                rows.append({"Entry": entry, "Image_Name": None, "Image_Path": None, "Assigned_Split": None,
                             "GT_Label": None, "Defect_Type": None, "Action": "not_found"})
                continue
            c = candidates[0]
            rows.append({
                "Entry": entry, "Image_Name": short_name(c["path"]), "Image_Path": c["path"],
                "Assigned_Split": c["split"], "GT_Label": c["label"], "Defect_Type": Path(c["path"]).parent.name,
                "Action": _ACTIONS[c["split"]],
            })

        if ambiguous:
            details = "; ".join(f"{e} -> {m}" for e, m in list(ambiguous.items())[:5])
            raise ValueError(
                f"[exclusion] {len(ambiguous)} entries match more than one image, so it is unclear which to "
                f"exclude. Write them as '<folder>/<file>' instead. {details}"
            )
        record = pd.DataFrame(rows, columns=["Entry", "Image_Name", "Image_Path", "Assigned_Split",
                                             "GT_Label", "Defect_Type", "Action"])
        record["GT_Label"] = record.GT_Label.astype("Int64")  # stays an int despite not_found rows
        if not (record.Action != "not_found").any():
            raise ValueError(
                f"[exclusion] None of the {len(self.entries)} entries in {self.source} matched an image in the "
                f"dataset - wrong list, or wrong --root_dir?"
            )
        return record

    def _log(self, record: pd.DataFrame) -> None:
        by_split = record.Assigned_Split.value_counts().to_dict()
        not_found = record.loc[record.Action == "not_found", "Entry"].tolist()
        before, after = self.counts_before, self.counts_after
        n_removed = before["train"]["total"] - after["train"]["total"]

        self.logger.info(f"[exclusion] setup() call #{self.setup_calls}: listed images by assigned split: "
                         f"Train {by_split.get('Train', 0)} | Val {by_split.get('Val', 0)} | "
                         f"Test {by_split.get('Test', 0)} | not found {len(not_found)}")
        self.logger.info(f"[exclusion] Train: {before['train']['total']} -> {after['train']['total']} "
                         f"({n_removed} removed). Val ({after['val']['total']}) and Test ({after['test']['total']}) "
                         f"unchanged; their {by_split.get('Val', 0) + by_split.get('Test', 0)} listed images "
                         f"are kept and tagged {FLAG_COLUMN}=True in predictions.csv.")
        self.logger.info(f"[exclusion] Train share of normal images: {_normal_train_fraction(before):.1%} "
                         f"(as split) -> {_normal_train_fraction(after):.1%} (after removal)")
        anomalous_removed = record[(record.Action == "removed_from_train") & (record.GT_Label == 1)]
        if len(anomalous_removed):
            self.logger.warning(f"[exclusion] {len(anomalous_removed)} listed images removed from train are "
                                f"labeled anomalous: {anomalous_removed.Image_Name.tolist()[:10]}")
        if not_found:
            self.logger.warning(f"[exclusion] {len(not_found)} entries matched no image: {not_found[:10]}"
                                f"{' ...' if len(not_found) > 10 else ''}")

    # ------------------------------------------------------------------
    # Queries used by the rest of the pipeline
    # ------------------------------------------------------------------
    @property
    def listed_image_names(self) -> Set[str]:
        if self.record is None:
            return set()
        return set(self.record.Image_Name.dropna())

    @property
    def removed_image_names(self) -> Set[str]:
        if self.record is None:
            return set()
        return set(self.record.loc[self.record.Action == "removed_from_train", "Image_Name"])

    def tag(self, predictions: pd.DataFrame) -> pd.DataFrame:
        tagged = predictions.copy()
        tagged[FLAG_COLUMN] = tagged.Image_Name.isin(self.listed_image_names)
        return tagged

    def summary(self) -> Dict[str, Any]:
        record = self.record if self.record is not None else pd.DataFrame(columns=["Action", "Assigned_Split"])
        return {
            "mode": "train_only",
            "description": "Split computed on all images with the run's seed; listed images then removed "
                           "from train only. Val/test/contamination keep them, tagged In_Exclusion_List.",
            "source_file": str(self.source.resolve()),
            "sha256": self.sha256,
            "entries": len(self.entries),
            "duplicate_entries": self.duplicate_entries,
            "matched": int((record.Action != "not_found").sum()),
            "not_found": record.loc[record.Action == "not_found", "Entry"].tolist() if len(record) else [],
            "listed_by_assigned_split": {s: int((record.Assigned_Split == s).sum()) for s, _ in _SPLIT_ATTRS},
            "removed_from_train": int((record.Action == "removed_from_train").sum()),
            "split_counts_before_removal": self.counts_before,
            "split_counts_after_removal": self.counts_after,
            "normal_train_fraction": {
                "as_split": _normal_train_fraction(self.counts_before) if self.counts_before else None,
                "after_removal": _normal_train_fraction(self.counts_after) if self.counts_after else None,
            },
            "setup_calls": self.setup_calls,
            "split_stable_across_setup_calls": self.split_stable,
        }

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    def write_outputs(self, output_path: Path) -> None:
        """exclusion_list.txt, excluded_samples.csv and exclusion_summary.json. Safe to call
        again later - after fit, the summary also covers the re-split trainer.fit did."""
        out_dir = Path(output_path) / EXCLUSION_DIRNAME
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.source, out_dir / "exclusion_list.txt")
        if self.record is not None:
            self.record.to_csv(out_dir / "excluded_samples.csv", index=False)
        with open(out_dir / "exclusion_summary.json", "w") as f:
            json.dump(self.summary(), f, indent=2)

    def write_checkpoint_predictions(self, output_path: Path, checkpoint: str, predictions: pd.DataFrame) -> None:
        """<checkpoint>_predictions.csv: the listed images' rows from that checkpoint's predictions.csv,
        with the split each was assigned to (Assigned_Split) and whether the model trained on it."""
        if FLAG_COLUMN not in predictions or self.record is None:
            return
        listed = predictions[predictions[FLAG_COLUMN]].drop(columns=[FLAG_COLUMN])
        listed = listed.merge(self.record[["Image_Name", "Assigned_Split", "Action"]], on="Image_Name", how="left")
        out_dir = Path(output_path) / EXCLUSION_DIRNAME
        out_dir.mkdir(parents=True, exist_ok=True)
        listed.to_csv(out_dir / f"{checkpoint}_predictions.csv", index=False)
        self.logger.info(f"[exclusion] Wrote {len(listed)} prediction rows for listed images to "
                         f"{out_dir / f'{checkpoint}_predictions.csv'}")
