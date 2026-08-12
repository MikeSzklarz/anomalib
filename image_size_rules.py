"""Rule-based image_size resolver for train.py's MODEL_MAP.

Built from empirical forward-pass testing of every model's underlying torch_model at
image_size in [224..768]. See conversation history / test scripts in scratchpad for
the raw per-model results this table is derived from.

Two independent use cases:
  - resolve_image_size(model, native_w, native_h, max_size): pick the best absolute
    size given your real image resolution and a compute/VRAM budget cap.
  - resolve_from_ratio(model, ratio, max_size=None): scale the model's own baseline
    default image_size by `ratio` for small, controlled resolution-sensitivity
    experiments (e.g. ratio=1.2). Snaps to the nearest size the architecture accepts,
    and flags `skipped=True` when that snaps back to exactly the baseline (models
    that are architecturally frozen, or where the ratio is too small to move the
    snapped value) so callers can skip the experiment instead of wasting a run that
    is identical to the baseline.

Usage:
    python image_size_rules.py native --model dinomaly --width 2880 --height 2160
    python image_size_rules.py ratio --model dinomaly --ratio 1.2
"""

import argparse

# kind values:
#   "free"        - no constraint found; any size in the tested range works
#   "divisor"     - must be a multiple of `divisor`; optional `floor` (hard minimum)
#   "fixed"       - architecture hardcodes a single resolution; image_size can't change
#   "patch_floor" - must be >= `floor` (its patch_size); no other constraint found
#   "safe_set"    - no clean formula; snap to nearest confirmed-working value
MODEL_IMAGE_SIZE_RULES = {
    # no constraint found in tested range
    "anomalydino": {"kind": "free", "default": 252},
    "anomalyvfm": {"kind": "free", "default": 768},
    "cflow": {"kind": "free", "default": 256},
    "dfkde": {"kind": "free", "default": 256},
    "dfm": {"kind": "free", "default": 256},
    "ganomaly": {"kind": "free", "default": 256},
    "generalad": {"kind": "free", "default": 518},
    "padim": {"kind": "free", "default": 256},
    "patchcore": {"kind": "free", "default": 256},
    "patchflow": {"kind": "free", "default": 768},
    "reversedistillation": {"kind": "free", "default": 256},
    "stfpm": {"kind": "free", "default": 256},
    "cfa": {"kind": "free", "default": 256},
    "glass": {"kind": "free", "default": 288},
    "supersimplenet": {"kind": "free", "default": 256},
    # divisible by 8: LayerNorm shapes computed via floor(size/8) mismatch otherwise
    "fastflow": {"kind": "divisor", "divisor": 8, "default": 256},
    # divisible by 32: U-Net-style skip connections mismatch otherwise
    "draem": {"kind": "divisor", "divisor": 32, "default": 256},
    "dsr": {"kind": "divisor", "divisor": 32, "default": 256},
    "uninet": {"kind": "divisor", "divisor": 32, "default": 256},
    # divisible by 14 (ViT patch14 backbone), no minimum floor
    "l2bt": {"kind": "divisor", "divisor": 14, "default": 224},
    # divisible by 14 AND >= 392 (hardcoded CenterCrop(392) ahead of the ViT)
    "dinomaly": {"kind": "divisor", "divisor": 14, "floor": 392, "default": 392},
    "inpformer": {"kind": "divisor", "divisor": 14, "floor": 392, "default": 392},
    # architecture is locked to one resolution; image_size cannot meaningfully change
    "fre": {"kind": "fixed", "value": 256, "default": 256},
    "uflow": {"kind": "fixed", "value": 448, "default": 448},
    # must be >= patch_size (default 448); no other constraint found up to 768
    "superadd": {"kind": "patch_floor", "floor": 448, "default": 448},
    # needs a minimum floor (~256) for its conv kernels to fit; no divisor rule found
    "efficientad": {"kind": "divisor", "divisor": 1, "floor": 256, "default": 256},
    # irregular multi-scale coupling layers; no formula found, only a confirmed-safe set
    "csflow": {"kind": "safe_set", "values": [256, 384, 512, 640, 768], "default": 256},
    # requires a depth/xyz channel alongside RGB (use with mvtec3d/adam3d); untested here
    "cfm": {"kind": "free", "default": 224, "note": "requires depth data; unverified"},
}


def _known_model(model: str) -> dict:
    if model not in MODEL_IMAGE_SIZE_RULES:
        msg = f"Unknown model '{model}'. Known models: {sorted(MODEL_IMAGE_SIZE_RULES)}"
        raise ValueError(msg)
    return MODEL_IMAGE_SIZE_RULES[model]


def _snap_to_multiple(target: float, divisor: int, floor: int, bias: str) -> int:
    if bias == "floor":
        snapped = (int(target) // divisor) * divisor
    else:  # "nearest"
        snapped = round(target / divisor) * divisor
    return int(snapped) if snapped >= floor else floor


def _resolve(spec: dict, model: str, target: float, bias: str, max_size: int | None) -> tuple[int, str]:
    """Snap `target` to a valid image_size for `model`, per its constraint kind."""
    if max_size is not None:
        target = min(target, max_size)

    if spec["kind"] == "fixed":
        value = spec["value"]
        return value, f"{model} hardcodes its resolution; image_size is locked to {value}."

    if spec["kind"] == "free":
        value = int(round(target))
        note = f" Note: {spec['note']}." if "note" in spec else ""
        return value, f"No architecture constraint found; using {value}.{note}"

    if spec["kind"] == "divisor":
        divisor = spec["divisor"]
        floor = spec.get("floor", divisor)
        value = _snap_to_multiple(max(target, floor), divisor, floor, bias)
        return value, f"Snapped {target:.1f} to nearest multiple of {divisor} (floor {floor}, bias={bias}) -> {value}."

    if spec["kind"] == "patch_floor":
        floor = spec["floor"]
        value = max(int(round(target)), floor)
        return value, f"{model} requires image_size >= patch_size ({floor}); using {value}."

    if spec["kind"] == "safe_set":
        values = spec["values"]
        value = min(values, key=lambda v: abs(v - target))
        return value, f"No clean divisor rule; snapping to nearest confirmed-safe value from {values} -> {value}."

    msg = f"Unhandled rule kind for {model}: {spec['kind']}"
    raise AssertionError(msg)


def resolve_image_size(model: str, native_width: int, native_height: int, max_size: int = 768) -> tuple[int, str]:
    """Pick the best single (square) image_size for `model` given a native resolution.

    Targets the longer native side (the axis that suffers the most compression under
    a square resize) capped at `max_size`, then snaps (flooring, so the result never
    exceeds `max_size`) to whatever value the model's architecture actually accepts.
    """
    spec = _known_model(model)
    long_side = max(native_width, native_height)
    ideal = min(long_side, max_size)
    value, explanation = _resolve(spec, model, ideal, bias="floor", max_size=max_size)
    return value, (
        f"native={native_width}x{native_height}, long_side={long_side} capped at {max_size} -> "
        f"target={ideal}. {explanation}"
    )


def resolve_from_ratio(model: str, ratio: float, max_size: int | None = None) -> tuple[int, int, bool, str]:
    """Scale `model`'s own baseline default image_size by `ratio` and snap it.

    Returns (resolved_size, baseline_size, skipped, explanation). `skipped` is True
    when the resolved size is identical to the baseline - either because the model's
    resolution is architecturally frozen, or because `ratio` was too small to move
    the snapped value - meaning an experiment at this ratio would be indistinguishable
    from the baseline run and should be skipped rather than wasting a training run.
    """
    spec = _known_model(model)
    baseline = spec["default"]
    target = baseline * ratio
    value, explanation = _resolve(spec, model, target, bias="nearest", max_size=max_size)
    skipped = value == baseline
    skip_note = " SKIPPED: resolves to the same size as baseline, nothing to compare." if skipped else ""
    full_explanation = f"baseline={baseline}, ratio={ratio} -> target={target:.1f}. {explanation}{skip_note}"
    return value, baseline, skipped, full_explanation


def validate_image_size(model: str, height: int, width: int) -> str | None:
    """Sanity-check an explicit (height, width) against `model`'s known constraints.

    Returns a human-readable warning if the value looks likely to fail deep in training
    (e.g. a non-multiple for a divisor-constrained model, or any non-default value for a
    fixed-resolution model), or None if it looks fine. This is a heads-up based on the
    empirically-derived rule table, not a guarantee - untested sizes outside what was
    sampled during testing could still behave differently.
    """
    spec = _known_model(model)

    if height != width:
        return (
            f"'{model}' image_size {height}x{width} is non-square; every model tested so far was only "
            f"verified with square sizes - a non-square resize is unverified and may behave unexpectedly."
        )

    valid_value, _ = _resolve(spec, model, height, bias="nearest", max_size=None)
    if valid_value != height:
        return (
            f"image_size={height} may not be valid for '{model}' (kind={spec['kind']}): the nearest size "
            f"matching its known constraints is {valid_value}. This may fail once training actually starts."
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    native_parser = subparsers.add_parser("native", help="Resolve from a real image resolution")
    native_parser.add_argument("--model", required=True, choices=sorted(MODEL_IMAGE_SIZE_RULES))
    native_parser.add_argument("--width", type=int, required=True)
    native_parser.add_argument("--height", type=int, required=True)
    native_parser.add_argument("--max-size", type=int, default=768, help="Compute/VRAM budget cap (default 768)")

    ratio_parser = subparsers.add_parser("ratio", help="Resolve from a ratio of the model's baseline default")
    ratio_parser.add_argument("--model", required=True, choices=sorted(MODEL_IMAGE_SIZE_RULES))
    ratio_parser.add_argument("--ratio", type=float, required=True)
    ratio_parser.add_argument("--max-size", type=int, default=None, help="Optional compute/VRAM budget cap")

    args = parser.parse_args()

    if args.mode == "native":
        size, explanation = resolve_image_size(args.model, args.width, args.height, args.max_size)
        print(explanation)
        print(f"--image_size {size}")
    else:
        size, baseline, skipped, explanation = resolve_from_ratio(args.model, args.ratio, args.max_size)
        print(explanation)
        if not skipped:
            print(f"--image_size {size}")


if __name__ == "__main__":
    main()
