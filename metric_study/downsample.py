"""Downsampling to 512x512, matching the training loader's geometry.

The training loader (``train/data_loader.py:56-58``) does a direct anisotropic
resize, ``transforms.Resize([512, 512])`` (PIL bilinear + antialias) applied straight
to the native 1679x1475 frame, with no crop or pad. The study reproduces that same
geometry with OpenCV's ``INTER_AREA``, mask-normalized so gaps/beamstop don't bleed
into neighboring output pixels and so the valid-pixel sum is preserved (plan §1.3).

Only ``INTER_AREA`` is used in the main degradation/metric grid; a one-time check
against the training loader's own resize is provided by
:func:`compare_to_torchvision_resize` (plan §1.3, F11), not as an alternative
downsampler.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover - dependency is required, not optional
    raise ImportError("metric_study.downsample requires opencv-python-headless") from e

TARGET_SIZE = (512, 512)  # (rows, cols)


def downsample_to_512(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: tuple[int, int] = TARGET_SIZE,
    tau: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask-normalized area-average resize to ``target_size`` (plan §1.3).

    Args:
        image: Native-resolution image (rows, cols), any float dtype. Should already
            have degradations/masking applied where relevant, i.e. this is the last
            step before metrics.
        mask: Boolean valid-pixel mask, same shape as ``image``.
        target_size: (rows, cols) of the output. cv2 expects (width, height), so
            this function does the axis-order flip internally.
        tau: An output pixel is valid only if at least this fraction of its native-
            resolution footprint was a valid input pixel (excludes gaps/beamstop
            that partially overlap a downsampled pixel).

    Returns:
        (image_512, mask_512): the mask-normalized area-mean image and its boolean
        valid mask, both of ``target_size``.
    """
    if image.shape != mask.shape:
        raise ValueError(f"image/mask shape mismatch: {image.shape} vs {mask.shape}")
    cv_size = (target_size[1], target_size[0])  # cv2 wants (width, height)
    m = mask.astype(np.float32)
    num = cv2.resize((image.astype(np.float32) * m), cv_size, interpolation=cv2.INTER_AREA)
    cov = cv2.resize(m, cv_size, interpolation=cv2.INTER_AREA)
    image_512 = np.divide(num, cov, out=np.zeros_like(num), where=cov > 0)
    mask_512 = cov >= tau
    return image_512.astype(np.float32), mask_512


def valid_pixel_sum_ratio(image: np.ndarray, mask: np.ndarray, image_512: np.ndarray, mask_512: np.ndarray) -> float:
    """Ratio of the downsampled to native valid-pixel sum; should be ~1.0 (plan §5.3 test).

    Because ``downsample_to_512`` stores an area *mean* (not sum), the native and
    512 sums are compared directly with no scale factor: mean-preserving area
    averaging keeps ``sum(image_512[mask_512])`` proportional to
    ``sum(image[mask]) / n_native_valid`` times ``n_512_valid`` when intensity is
    uniform, and keeps the two *means* equal in general, which is what this checks.
    """
    native_mean = float(image[mask].mean()) if mask.any() else 0.0
    ds_mean = float(image_512[mask_512].mean()) if mask_512.any() else 0.0
    if native_mean == 0:
        return float("nan")
    return ds_mean / native_mean


def compare_to_torchvision_resize(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """One-time consistency check against the training loader's own resize (plan §1.3).

    Compares this module's ``downsample_to_512`` (cv2 INTER_AREA, mask-normalized)
    against ``torchvision.transforms.Resize([512, 512])`` (PIL bilinear + antialias,
    no masking), applied to the same input, to establish whether conclusions from
    this study transfer to what the training loader actually does.

    Args:
        image: Native-resolution image in model space ([0, 1] or [0, 255]).
        mask: Boolean valid-pixel mask, same shape as ``image``.

    Returns:
        Dict with ``mae`` (mean absolute difference over valid 512 pixels) and
        ``max_abs_diff``.
    """
    import torch
    from PIL import Image
    from torchvision import transforms

    image_512_cv, mask_512 = downsample_to_512(image, mask)

    pil_img = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    resize = transforms.Compose([transforms.Resize([512, 512]), transforms.ToTensor()])
    tv_tensor: torch.Tensor = resize(pil_img)
    image_512_tv = tv_tensor.squeeze(0).numpy() * 255.0

    diff = np.abs(image_512_cv - image_512_tv)
    diff_valid = diff[mask_512] if mask_512.any() else diff
    return {
        "mae": float(diff_valid.mean()),
        "max_abs_diff": float(diff_valid.max()) if diff_valid.size else float("nan"),
    }
