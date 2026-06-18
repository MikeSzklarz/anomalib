"""
Color preprocessing transforms for train.py.

Each class is a torch.nn.Module that accepts PIL Images (as delivered by the
anomalib datamodule) and returns PIL Images, making them safe to prepend to
any pre_processor.transform chain.

Techniques mirror the analysis in research/notebooks/02_color/run.py:
  reinhard_neutral  — T1: Reinhard transfer targeting ImageNet LAB centroid (L=62, a=2, b=10)
  reinhard_cool     — T2: Reinhard transfer to achromatic neutral (L=62, a=0, b=0)
  clahe             — T3: CLAHE on L channel only
  reinhard_clahe    — T4: T1 then T3
  desaturate        — T5: 75% original + 25% grayscale
  grayscale_3ch     — T6: grayscale replicated to 3 channels
  viridis           — T7: grayscale mapped through viridis colormap
  composite         — T8: channels = [L, CLAHE-L, Sobel edges]
"""

import numpy as np
import cv2
import torch
from PIL import Image


# ── Core numpy helpers (float32 [0,1] RGB in and out) ────────────────────────

def _reinhard(src: np.ndarray, tL: float, ta: float, tb: float) -> np.ndarray:
    u8 = (src * 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    for i, t in enumerate([tL, ta, tb]):
        m, s = lab[:, :, i].mean(), lab[:, :, i].std() + 1e-8
        lab[:, :, i] = (lab[:, :, i] - m) * (20.0 / s) + t
    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 100) * (255.0 / 100.0)
    lab[:, :, 1] = np.clip(lab[:, :, 1] + 128.0, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + 128.0, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def _clahe(src: np.ndarray) -> np.ndarray:
    u8 = (src * 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def _grayscale(src: np.ndarray) -> np.ndarray:
    g = np.dot(src, [0.2989, 0.5870, 0.1140])
    return np.stack([g] * 3, axis=-1).astype(np.float32)


def _composite(src: np.ndarray) -> np.ndarray:
    u8 = (src * 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    Lo = lab[:, :, 0].astype(np.float32) / 255.0
    Lc = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0]).astype(np.float32) / 255.0
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(sx ** 2 + sy ** 2)
    edges = edges / edges.max() if edges.max() > 0 else edges
    return np.stack([Lo, Lc, edges], axis=-1).astype(np.float32)


# ── Base class ────────────────────────────────────────────────────────────────

class _ColorTransformBase(torch.nn.Module):
    def _apply_np(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _apply_single_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """Process one (C, H, W) tensor, any dtype, any device."""
        device, dtype = t.device, t.dtype
        arr = t.cpu().float().permute(1, 2, 0).numpy()
        if dtype == torch.uint8 or arr.max() > 1.5:
            arr /= 255.0
        out = np.clip(self._apply_np(arr), 0, 1)
        result = torch.from_numpy(out).permute(2, 0, 1).to(device)
        if dtype == torch.uint8:
            return (result * 255).to(torch.uint8)
        return result.to(dtype)

    def forward(self, img, *passthrough):
        """Process image; pass mask/extra args through unchanged.

        v2.Compose calls each transform as transform(image, gt_mask), so
        *passthrough captures gt_mask and returns it untouched.
        """
        if isinstance(img, Image.Image):
            arr = np.array(img).astype(np.float32) / 255.0
            out = np.clip(self._apply_np(arr), 0, 1)
            result = Image.fromarray((out * 255).astype(np.uint8))
        elif isinstance(img, torch.Tensor):
            if img.ndim == 4:  # (B, C, H, W) batch
                result = torch.stack(
                    [self._apply_single_tensor(img[i]) for i in range(img.shape[0])], dim=0
                )
            else:  # (C, H, W) single
                result = self._apply_single_tensor(img)
        else:
            result = img

        if passthrough:
            return (result,) + passthrough
        return result


# ── Concrete transforms ───────────────────────────────────────────────────────

class ReinhardNeutralTransform(_ColorTransformBase):
    """T1: Reinhard color transfer targeting ImageNet LAB centroid (L=62, a=2, b=10)."""
    def _apply_np(self, img): return _reinhard(img, 62.0, 2.0, 10.0)
    def __repr__(self): return "ReinhardNeutralTransform(target=ImageNet_centroid)"


class ReinhardCoolTransform(_ColorTransformBase):
    """T2: Reinhard color transfer to achromatic neutral (L=62, a=0, b=0)."""
    def _apply_np(self, img): return _reinhard(img, 62.0, 0.0, 0.0)
    def __repr__(self): return "ReinhardCoolTransform(target=achromatic)"


class CLAHETransform(_ColorTransformBase):
    """T3: Contrast-limited adaptive histogram equalization on L channel only."""
    def _apply_np(self, img): return _clahe(img)
    def __repr__(self): return "CLAHETransform(clip=2.0, grid=8x8)"


class ReinhardCLAHETransform(_ColorTransformBase):
    """T4: Reinhard neutral followed by CLAHE."""
    def _apply_np(self, img): return _clahe(_reinhard(img, 62.0, 2.0, 10.0))
    def __repr__(self): return "ReinhardCLAHETransform()"


class Desaturate25Transform(_ColorTransformBase):
    """T5: Blend 75% original + 25% grayscale (mild desaturation)."""
    def _apply_np(self, img):
        g = np.dot(img, [0.2989, 0.5870, 0.1140])
        return (0.75 * img + 0.25 * np.stack([g] * 3, axis=-1)).astype(np.float32)
    def __repr__(self): return "Desaturate25Transform(alpha=0.25)"


class Grayscale3chTransform(_ColorTransformBase):
    """T6: Convert to grayscale and replicate to 3 channels."""
    def _apply_np(self, img): return _grayscale(img)
    def __repr__(self): return "Grayscale3chTransform()"


class ViridasTransform(_ColorTransformBase):
    """T7: Convert to grayscale then map through viridis colormap."""
    def _apply_np(self, img):
        import matplotlib.cm as cm
        g = np.dot(img, [0.2989, 0.5870, 0.1140])
        return cm.viridis(g)[:, :, :3].astype(np.float32)
    def __repr__(self): return "ViridasTransform(colormap=viridis)"


class Composite3chTransform(_ColorTransformBase):
    """T8: Composite 3-channel image — [L, CLAHE-L, Sobel edges].

    NOTE: The output channels are NOT RGB. ImageNet normalization downstream
    will still run but is semantically mismatched. This is an advanced option
    that changes the input representation entirely.
    """
    def _apply_np(self, img): return _composite(img)
    def __repr__(self): return "Composite3chTransform(channels=[L, CLAHE-L, Sobel])"


# ── Factory ───────────────────────────────────────────────────────────────────

_TRANSFORM_MAP = {
    "reinhard_neutral": ReinhardNeutralTransform,
    "reinhard_cool":    ReinhardCoolTransform,
    "clahe":            CLAHETransform,
    "reinhard_clahe":   ReinhardCLAHETransform,
    "desaturate":       Desaturate25Transform,
    "grayscale_3ch":    Grayscale3chTransform,
    "viridis":          ViridasTransform,
    "composite":        Composite3chTransform,
}

COLOR_PREPROCESSING_CHOICES = ["none"] + list(_TRANSFORM_MAP.keys())


def get_color_transform(name: str) -> _ColorTransformBase | None:
    cls = _TRANSFORM_MAP.get(name)
    return cls() if cls else None
