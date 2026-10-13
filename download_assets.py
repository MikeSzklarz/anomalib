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

Safe to re-run: anything already present is skipped, and an interrupted dataset
download resumes where it stopped.

Then copy the repo folder (including model_cache/ and datasets/) to the cluster.
train/__init__.py points training at model_cache/ whenever that folder exists;
run_slurm.sh sets HF_HUB_OFFLINE=1 so nothing tries to reach the internet.
"""

import argparse
import gc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

REPO_DIR = Path(__file__).resolve().parent
os.chdir(REPO_DIR)  # datasets/dtd and datasets/imagenette are relative to where training runs
sys.path.insert(0, str(REPO_DIR))

import train  # noqa: E402

train.MODEL_CACHE_DIR.mkdir(exist_ok=True)
train.use_model_cache()
os.environ["HF_HUB_OFFLINE"] = "0"  # downloading is the whole point here

from train.registry import MODEL_MAP  # noqa: E402
from train.utils import get_init_args, load_yaml_config  # noqa: E402


def download(url: str, dest: Path, retries: int = 8) -> None:
    """Download url to dest, resuming a partial file and retrying dropped connections
    (the DTD server in particular is slow and resets connections)."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        try:
            request = Request(url, headers={"Range": f"bytes={have}-"} if have else {})  # noqa: S310
            with urlopen(request, timeout=60) as response:  # noqa: S310
                resumed = have and response.status == 206
                total = int(response.headers.get("Content-Length", 0)) + (have if resumed else 0)
                print(f"    {dest.name}: {'resuming at' if resumed else 'starting,'} "
                      f"{have / 1e6 if resumed else total / 1e6:.0f} MB{f' of {total / 1e6:.0f} MB' if resumed else ''}", flush=True)
                with open(part, "ab" if resumed else "wb") as f:
                    shutil.copyfileobj(response, f, 1 << 20)
            if total and part.stat().st_size < total:
                raise ConnectionError(f"connection closed at {part.stat().st_size / 1e6:.0f} of {total / 1e6:.0f} MB")
            part.rename(dest)
            return
        except HTTPError as e:
            if e.code == 416 and have:  # asked to resume past the end: the .part file is already complete
                part.rename(dest)
                return
            raise
        except OSError as e:  # URLError, timeouts, resets
            if attempt == retries:
                raise
            print(f"    {dest.name}: {e} - retry {attempt}/{retries - 1} in {5 * attempt}s", flush=True)
            time.sleep(5 * attempt)


def ensure_dataset(root: Path, info) -> None:
    """Make sure the archive behind `info` is downloaded, hash-checked and extracted in `root`.

    Skips when `root` already holds an extracted folder. A bare `root`, or one holding only
    an archive, is NOT treated as done: anomalib creates `root` before it starts downloading,
    so a failed attempt leaves exactly that behind and the model would then skip the download
    and silently train without the data.
    """
    from anomalib.data.utils.download import check_hash, extract

    if root.is_dir() and any(child.is_dir() for child in root.iterdir()):
        return
    archive = root / info.url.split("/")[-1]
    if archive.exists():
        try:
            check_hash(archive, info.hashsum)
        except ValueError:
            archive.unlink()  # truncated leftover from a failed attempt
    if not archive.exists():
        download(info.url, archive)
        check_hash(archive, info.hashsum)
    print(f"    extracting {archive.name}", flush=True)
    extract(archive, root)


def fetch(name: str, config_root: str | None) -> None:
    kwargs = {}
    if config_root:
        kwargs = get_init_args(load_yaml_config(str(Path(config_root) / f"{name}.yaml")), "model")

    if name == "draem":
        from anomalib.models.image.draem.lightning_model import DTD_DOWNLOAD_INFO

        ensure_dataset(Path(kwargs.get("dtd_dir", "./datasets/dtd")), DTD_DOWNLOAD_INFO)
    if name == "cfm":  # a truncated Point-MAE file would be accepted as-is; drop it so it re-downloads
        from anomalib.data.utils.download import generate_hash
        from anomalib.models.image.cfm.components import POINTMAE_DOWNLOAD_INFO
        from anomalib.utils.path import get_pretrained_weights_dir

        weights = get_pretrained_weights_dir() / "pointmae" / "pointmae_pretrain.pth"
        if weights.is_file() and generate_hash(weights, "sha256") != POINTMAE_DOWNLOAD_INFO.hashsum:
            weights.unlink()

    model = MODEL_MAP[name](**kwargs)  # backbones download here
    if hasattr(model, "prepare_pretrained_model"):  # efficientad, dsr
        model.prepare_pretrained_model()
    if name == "efficientad":
        from anomalib.models.image.efficient_ad.lightning_model import IMAGENETTE_DOWNLOAD_INFO

        ensure_dataset(Path(model.imagenet_dir), IMAGENETTE_DOWNLOAD_INFO)


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
