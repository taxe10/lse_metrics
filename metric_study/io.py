"""Image readers.

Production inputs are expected to be 8-bit (or 16-bit) grayscale PNG/JPEG exports of
Pilatus 2M frames (see ``plans/metric_study.md`` Q2/Q3 for open questions on the exact
export). ``read_image`` never raises: read/decode failures are reported through the
returned ``status`` field so the pipeline can log-and-skip instead of crashing
(plan §6.3).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm", ".pgm", ".webp")


@dataclass
class ImageReadResult:
    """Result of attempting to read one image file.

    Attributes:
        path: The file that was read.
        image: The decoded image as float32, or None if reading failed.
        meta: Shape, dtype and other header-derived fields.
        status: "ok", or an error description if reading failed.
    """

    path: Path
    image: np.ndarray | None
    meta: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def read_image(path: str | Path) -> ImageReadResult:
    """Read a grayscale image file, never raising.

    Uses ``cv2.imread`` with ``IMREAD_UNCHANGED`` so both 8-bit and 16-bit PNGs are
    read at their native bit depth (plan §6.1). Multi-channel images are averaged
    down to grayscale, since the sample data and training loader both treat scattering
    frames as single-channel.

    Args:
        path: Path to the image file.

    Returns:
        An ``ImageReadResult`` with ``status == "ok"`` and a populated ``image`` on
        success, or ``status`` describing the failure and ``image is None`` otherwise.
    """
    path = Path(path)
    if not path.exists():
        return ImageReadResult(path, None, status=f"missing_file: {path}")
    try:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            return ImageReadResult(path, None, status="decode_failed")
        if raw.ndim == 3:
            # BGR(A) -> grayscale by simple channel mean; scattering exports are
            # expected to be single-channel already, so this is a defensive fallback.
            raw = raw[..., :3].astype(np.float32).mean(axis=2)
        bit_depth = 16 if raw.dtype == np.uint16 else 8
        image = raw.astype(np.float32)
        meta = {
            "shape": tuple(image.shape),
            "dtype": str(raw.dtype),
            "bit_depth": bit_depth,
            "file_size": path.stat().st_size,
            "mtime": path.stat().st_mtime,
        }
        return ImageReadResult(path, image, meta=meta, status="ok")
    except Exception as e:  # noqa: BLE001 - any decode error must be caught, not raised
        logger.warning("Failed to read %s: %s", path, e)
        return ImageReadResult(path, None, status=f"exception: {type(e).__name__}: {e}")


def iter_image_paths(root: str | Path, extensions: tuple[str, ...] = VALID_EXTENSIONS):
    """Recursively yield image file paths under ``root``, matching the loader's glob (data_loader.py:69-70)."""
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in extensions:
            yield p
