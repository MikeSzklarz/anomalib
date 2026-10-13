"""Training CLI and pipeline for the anomalib model zoo used by submit_sweep.sh.

Entry point: train.cli.main() (invoked via the repo-root train.py shim, or
`python -m train`). See cli.py, setup.py, pipeline.py, callbacks/, analysis/.
"""

import os
from pathlib import Path

# Repo-local home for every external file the models need (pretrained backbones
# from Hugging Face / torch hub, anomalib's own pretrained weights). Filled by
# download_assets.py on a machine with internet; once it exists, training reads
# from it instead of ~/.cache, so the repo folder can be copied as-is to a
# cluster whose nodes cannot download. Must run before timm/huggingface_hub are
# imported, which is why it lives here.
MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent / "model_cache"


def use_model_cache() -> None:
    os.environ["XDG_CACHE_HOME"] = str(MODEL_CACHE_DIR)  # anomalib's pre_trained/ cache
    os.environ["HF_HOME"] = str(MODEL_CACHE_DIR / "huggingface")
    os.environ["TORCH_HOME"] = str(MODEL_CACHE_DIR / "torch")


if MODEL_CACHE_DIR.is_dir():
    use_model_cache()
