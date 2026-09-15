"""Compares the resolved image_size for a run against the dataset's real native
resolution. The actual per-model size rules (baseline defaults, ratio scaling,
validation) live in image_size_rules.py - this module is just the logging/
comparison layer, plus the helper that reads a PreProcessor's resize target."""

from PIL import Image


def get_resize_target(pre_processor) -> tuple[int, int] | None:
    """Extract the (height, width) a PreProcessor's Resize transform will produce, if any."""
    transform = getattr(pre_processor, "transform", None)
    if transform is None:
        return None
    for t in getattr(transform, "transforms", [transform]):
        size = getattr(t, "size", None)
        if size is not None and len(size) == 2:
            return int(size[0]), int(size[1])
    return None


def log_image_size_analysis(datamodule, logger, model_name: str, target_h: int, target_w: int, sample_count: int = 5):
    """Compares the resolved image_size against the dataset's real native resolution.

    Flags axes that will be upsampled (no extra real detail gained beyond native resolution)
    and, for square targets, the aspect-ratio skew a non-square source will suffer - which is
    constant across every image_size choice for a given dataset, not something a bigger or
    smaller target fixes.
    """
    train_data = getattr(datamodule, "train_data", None)
    if not train_data or not hasattr(train_data, "samples") or len(train_data.samples) == 0:
        logger.warning("[image_size] Skipping native-resolution check: no train samples found.")
        return

    native_sizes = []
    for image_path in train_data.samples["image_path"].tolist()[:sample_count]:
        try:
            with Image.open(image_path) as im:
                native_sizes.append(im.size)  # PIL gives (W, H)
        except Exception:
            continue

    if not native_sizes:
        logger.warning("[image_size] Skipping native-resolution check: could not read sample images.")
        return

    widths = [s[0] for s in native_sizes]
    heights = [s[1] for s in native_sizes]
    native_w, native_h = widths[0], heights[0]
    variability_note = ""
    if len(set(widths)) > 1 or len(set(heights)) > 1:
        variability_note = (
            f" [sampled {len(native_sizes)} images, sizes vary: width {min(widths)}-{max(widths)}, "
            f"height {min(heights)}-{max(heights)}; using first sample as reference]"
        )

    scale_w = target_w / native_w
    scale_h = target_h / native_h

    def axis_desc(name: str, native: int, target: int, scale: float) -> str:
        direction = "UPSAMPLED, no extra real detail beyond native" if scale > 1.0 else "downsampled"
        return f"{name} {native}->{target} ({scale:.2f}x, {direction})"

    logger.info(
        f"[image_size] Native-resolution check for '{model_name}': native={native_w}x{native_h}{variability_note}, "
        f"target={target_w}x{target_h}. {axis_desc('width', native_w, target_w, scale_w)}; "
        f"{axis_desc('height', native_h, target_h, scale_h)}."
    )

    if scale_w > 1.0 and scale_h > 1.0:
        logger.warning(
            "[image_size] Both axes exceed native resolution - this target adds no additional real detail "
            "over a smaller one, only interpolation and extra compute."
        )

    if target_w == target_h and min(scale_w, scale_h) > 0:
        skew = max(scale_w, scale_h) / min(scale_w, scale_h)
        if skew > 1.15:
            logger.warning(
                f"[image_size] Non-square source ({native_w}x{native_h}) forced into a square resize distorts "
                f"object geometry by ~{skew:.2f}x (width and height scaled by different factors). This skew is "
                f"constant across every image_size choice for this dataset - it's a property of the native "
                f"aspect ratio vs. the forced square resize, not something image_size affects."
            )
