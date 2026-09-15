"""Standalone helper for offline analysis of a downloaded raw_outputs/<split>/
folder - needs only numpy + Pillow, NOT anomalib/torch/lightning. Meant to be
copied alongside arrays.npz + index.csv onto a laptop with no training
environment installed.

What's in arrays.npz:
    image_names               (N,)          str          same order as every other array
    anomaly_maps               (N, H, W)     float16 or uint8 (see anomaly_maps_scale_min/max)
    anomaly_maps_scale_min/max scalar        only present if anomaly_maps is uint8 (--compact_raw_outputs)
    pred_masks                  (N, H, W)     uint8 0/1     the model's binary prediction
    gt_masks                     (N, H, W)     uint8 0/1     only if the dataset provides masks
    pixel_min/pixel_max/         scalar        the PostProcessor's calibrated normalization,
    pixel_threshold                             needed to reproduce the exact default-visualization colors

anomaly_maps is always on the model-native ("raw") scale - the same scale as
Anomaly_Score in index.csv/predictions.csv - not yet colorized or thresholded.
render_heatmap() below reproduces the exact colors anomalib's own visualizer
would draw, entirely offline, using only what's stored in this file.

Usage:
    python offline_viewer.py arrays.npz                         # render every image
    python offline_viewer.py arrays.npz --image "debris/T0127_013.jpg"
    python offline_viewer.py arrays.npz --upsample-to 1210 1210  # match the original photo size

Or import directly:
    from offline_viewer import load_split, render_heatmap
    data = load_split("arrays.npz")
    img = render_heatmap(data["anomaly_maps"][3], data.get("pixel_min"), data.get("pixel_max"), data.get("pixel_threshold"))
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# Same 9-stop palette as anomalib.visualization.image.functional.apply_colormap,
# copied here (not imported) so this script has zero anomalib dependency.
_COLORMAP_STOPS = np.array([
    (0, 0, 143), (0, 0, 255), (0, 127, 255), (0, 255, 255), (127, 255, 127),
    (255, 255, 0), (255, 127, 0), (255, 0, 0), (127, 0, 0),
], dtype=np.float64)


def _bilinear_resize_1d(values: np.ndarray, out_size: int) -> np.ndarray:
    """Replicates torch.nn.functional.interpolate(mode='bilinear', align_corners=False)
    for a resize along axis 0 - verified byte-for-byte identical to the torch call
    anomalib's apply_colormap makes to build its 9-stop-to-256-color palette."""
    in_size = values.shape[0]
    scale = in_size / out_size
    out = np.zeros((out_size, values.shape[1]), dtype=np.float64)
    for i in range(out_size):
        src = min(max((i + 0.5) * scale - 0.5, 0.0), in_size - 1)
        lo = int(np.floor(src))
        hi = min(lo + 1, in_size - 1)
        frac = src - lo
        out[i] = values[lo] * (1 - frac) + values[hi] * frac
    return out


_PALETTE = np.floor(_bilinear_resize_1d(_COLORMAP_STOPS, 256)).astype(np.uint8)  # torch's .byte() truncates


def apply_colormap(gray_uint8: np.ndarray) -> Image.Image:
    """Pixel-identical to anomalib.visualization.image.functional.apply_colormap,
    reimplemented with zero anomalib dependency."""
    img = Image.fromarray(gray_uint8, mode="L").convert("P")
    img.putpalette(_PALETTE.flatten().tolist())
    return img.convert("RGB")


def load_split(npz_path: str | Path) -> dict:
    """Loads arrays.npz, de-quantizing anomaly_maps back to float32 if it was
    stored compact (uint8 + scale_min/max). Returns a plain dict; anomaly_maps
    is always float32 on the model-native scale regardless of storage mode."""
    data = np.load(npz_path, allow_pickle=False)
    result: dict = {"image_names": data["image_names"]}

    if "anomaly_maps" in data.files:
        maps = data["anomaly_maps"]
        if maps.dtype == np.uint8 and "anomaly_maps_scale_min" in data.files:
            lo = float(data["anomaly_maps_scale_min"])
            hi = float(data["anomaly_maps_scale_max"])
            maps = maps.astype(np.float32) / 255.0 * (hi - lo) + lo
        result["anomaly_maps"] = maps.astype(np.float32)

    for key in ("pred_masks", "gt_masks"):
        if key in data.files:
            result[key] = data[key]

    for key in ("pixel_min", "pixel_max", "pixel_threshold"):
        if key in data.files:
            result[key] = float(data[key])

    return result


def render_heatmap(
    raw_map: np.ndarray,
    pixel_min: float | None,
    pixel_max: float | None,
    pixel_threshold: float | None,
) -> Image.Image:
    """Reconstructs the exact colorized heatmap the default anomalib visualization
    shows, from one raw (model-native scale) anomaly_map plus the three
    normalization scalars stored alongside it in arrays.npz."""
    if pixel_min is not None and pixel_max is not None and pixel_max > pixel_min:
        threshold = pixel_threshold if pixel_threshold is not None else (pixel_max + pixel_min) / 2
        normalized = ((raw_map - threshold) / (pixel_max - pixel_min)) + 0.5
        normalized = np.clip(normalized, 0.0, 1.0)
    else:
        # No calibrated normalization was stored (PostProcessor normalization was
        # disabled at train time) - falls back to per-image min-max, which is NOT
        # comparable in severity across images.
        lo, hi = float(raw_map.min()), float(raw_map.max())
        normalized = (raw_map - lo) / (hi - lo) if hi > lo else np.zeros_like(raw_map)

    gray = np.round(normalized * 255).astype(np.uint8)
    return apply_colormap(gray)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz_path", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("./rendered_heatmaps"))
    parser.add_argument("--image", type=str, default=None,
                        help="Render only this Image_Name (e.g. 'debris/T0127_013.jpg'). Default: render all.")
    parser.add_argument("--upsample-to", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None,
                        help="Resize the rendered heatmap to this size (e.g. the original photo's resolution).")
    args = parser.parse_args()

    data = load_split(args.npz_path)
    if "anomaly_maps" not in data:
        raise SystemExit(f"{args.npz_path} has no anomaly_maps array.")

    names = data["image_names"]
    indices = range(len(names)) if args.image is None else [i for i, n in enumerate(names) if n == args.image]
    if not list(indices):
        raise SystemExit(f"Image '{args.image}' not found in {args.npz_path}.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i in indices:
        img = render_heatmap(data["anomaly_maps"][i], data.get("pixel_min"), data.get("pixel_max"), data.get("pixel_threshold"))
        if args.upsample_to:
            img = img.resize(tuple(args.upsample_to), Image.BILINEAR)
        out_name = str(names[i]).replace("/", "_")
        img.save(args.out_dir / f"{out_name}.png")
        count += 1
    print(f"Rendered {count} heatmap(s) to {args.out_dir}")


if __name__ == "__main__":
    _main()
