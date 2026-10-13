"""Download every external file the models in train/registry.py need, into this repo.

Run once from the repo root on a machine WITH internet, in the anomalib environment:

    python download_assets.py                     # everything (~12.5 GB)
    python download_assets.py --models padim rad  # just these

It builds each model the same way train.py does, which makes the model's own code
fetch (and hash-check) its files, so this stays correct as models are added:

    model_cache/huggingface/   timm backbones + AnomalyVFM weights
    model_cache/torch/         torchvision backbones (cfa, uninet, csflow)
    model_cache/anomalib/      EfficientAd teacher, DSR and Point-MAE weights
    datasets/dtd               texture images DRAEM trains with
    datasets/imagenette        extra images EfficientAd trains with

Then copy the repo folder (including model_cache/ and datasets/) to the cluster.
train/__init__.py points training at model_cache/ whenever that folder exists;
run_slurm.sh sets HF_HUB_OFFLINE=1 so nothing tries to reach the internet.
"""

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
os.chdir(REPO_DIR)  # datasets/dtd and datasets/imagenette are relative to where training runs
sys.path.insert(0, str(REPO_DIR))

import train  # noqa: E402

train.MODEL_CACHE_DIR.mkdir(exist_ok=True)
train.use_model_cache()
os.environ["HF_HUB_OFFLINE"] = "0"  # downloading is the whole point here

from train.registry import MODEL_MAP  # noqa: E402
from train.utils import get_init_args, load_yaml_config  # noqa: E402


def fetch(name: str, config_root: str | None) -> None:
    kwargs = {}
    if config_root:
        kwargs = get_init_args(load_yaml_config(str(Path(config_root) / f"{name}.yaml")), "model")
    model = MODEL_MAP[name](**kwargs)  # backbones (and DRAEM's DTD) download here
    if hasattr(model, "prepare_pretrained_model"):  # efficientad, dsr
        model.prepare_pretrained_model()
    if name == "efficientad" and not Path(model.imagenet_dir).is_dir():
        from anomalib.data.utils import download_and_extract
        from anomalib.models.image.efficient_ad.lightning_model import IMAGENETTE_DOWNLOAD_INFO

        download_and_extract(Path(model.imagenet_dir), IMAGENETTE_DOWNLOAD_INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=sorted(MODEL_MAP), choices=sorted(MODEL_MAP), metavar="MODEL")
    parser.add_argument("--config_root", default=None,
                        help="Same folder you pass to train.py; only matters if a config changes a backbone.")
    args = parser.parse_args()

    failed = {}
    for i, name in enumerate(args.models, 1):
        print(f"[{i}/{len(args.models)}] {name}", flush=True)
        try:
            fetch(name, args.config_root)
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            failed[name] = f"{type(e).__name__}: {e}"
            print(f"    FAILED: {failed[name]}", flush=True)
        gc.collect()

    print("\nOn disk:")
    for path in (train.MODEL_CACHE_DIR, Path("datasets/dtd"), Path("datasets/imagenette")):
        if path.exists():
            print("  " + subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True).stdout.strip())
    print(f"\n{len(args.models) - len(failed)}/{len(args.models)} models ready.")
    if failed:
        print("Failed: " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
