"""Small standalone helpers shared across the train package."""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def setup_logger(output_dir: Path, model_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"{model_name}_training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True
    )
    return logging.getLogger("train_script")


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


def get_git_commit() -> str | None:
    """Best-effort git commit hash of this checkout, for manifest.json provenance."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
