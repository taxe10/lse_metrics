"""Detector masks: static module gaps plus data-driven beamstop/dead/hot pixels.

No calibration geometry is used anywhere in this module (plan scope note in the
plan header) — the static mask below is derived purely from the Pilatus 2M module
layout (a detector property), and the data-driven mask is estimated straight from a
stack of frames, with no beam-center or PONI dependency.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)

# Pilatus 2M module layout, confirmed against the sample data (plan §0.2):
# 8 module rows x 3 module columns, each module 487 (w) x 195 (h) px, with
# 7 px gaps between columns and 17 px gaps between rows.
# 8*195 + 7*17 = 1679 (rows); 3*487 + 2*7 = 1475 (cols).
PILATUS2M_SHAPE = (1679, 1475)
_MODULE_H, _GAP_ROW = 195, 17
_MODULE_W, _GAP_COL = 487, 7
_N_ROWS, _N_COLS = 8, 3


def static_gap_mask(shape: tuple[int, int] = PILATUS2M_SHAPE, dilation_px: int = 2) -> np.ndarray:
    """Boolean valid-pixel mask for the Pilatus 2M module gaps (True = valid detector pixel).

    Args:
        shape: (rows, cols) of the native detector image. Only the confirmed Pilatus 2M
            shape is supported; any other shape raises, so callers can catch it and treat
            the frame as "unexpected shape" (plan §6.6) rather than silently masking wrong.
        dilation_px: How many pixels to grow the gap region by, to absorb JPEG ringing
            and dark module edges observed in the sample data (plan §0.2). 0 disables.

    Returns:
        A boolean array of ``shape``, True where the pixel is real detector area.

    Raises:
        ValueError: If ``shape`` is not the confirmed Pilatus 2M shape.
    """
    if tuple(shape) != PILATUS2M_SHAPE:
        raise ValueError(
            f"static_gap_mask only supports the confirmed Pilatus 2M shape {PILATUS2M_SHAPE}, got {shape}. "
            "Frames of another shape should be recorded and skipped (plan §6.6), not masked with this layout."
        )
    valid = np.zeros(shape, dtype=bool)
    for i in range(_N_ROWS):
        r0 = i * (_MODULE_H + _GAP_ROW)
        for j in range(_N_COLS):
            c0 = j * (_MODULE_W + _GAP_COL)
            valid[r0 : r0 + _MODULE_H, c0 : c0 + _MODULE_W] = True
    if dilation_px > 0:
        # Dilate the *gap* (invalid) region, i.e. erode the valid region.
        invalid = ~valid
        invalid = ndi.binary_dilation(invalid, iterations=dilation_px)
        valid = ~invalid
    return valid


@dataclass
class DataDrivenMask:
    """A beamstop/dead/hot pixel mask estimated from a stack of frames in one mask group.

    Attributes:
        valid: Boolean array, True where the pixel is usable (not beamstop, dead, or hot).
        dark_frac: Per-pixel fraction of frames where the pixel was at or below the dark
            threshold (diagnostic; high values indicate beamstop/dead pixels).
            frames.
        n_frames: Number of frames the mask was estimated from.
    """

    valid: np.ndarray
    dark_frac: np.ndarray
    n_frames: int


def estimate_data_driven_mask(
    stack: np.ndarray,
    dark_value: float = 3.0,
    dark_frac_thresh: float = 0.99,
    hot_frac_thresh: float = 0.5,
    saturation_value: float | None = 255.0,
) -> DataDrivenMask:
    """Estimate a beamstop/dead/hot pixel mask from a stack of frames (plan §6.6).

    A pixel is masked invalid if it is at or below ``dark_value`` in at least
    ``dark_frac_thresh`` of frames (beamstop shadow or dead pixel), or at/above
    ``saturation_value`` in at least ``hot_frac_thresh`` of frames (hot/saturated pixel).

    Args:
        stack: Array of shape (N, H, W), N frames from the same mask group.
        dark_value: Pixel values at or below this are considered "dark" for one frame.
        dark_frac_thresh: Fraction of frames a pixel must be dark in, to be masked.
        hot_frac_thresh: Fraction of frames a pixel must be saturated in, to be masked.
        saturation_value: Value considered "saturated"; None disables hot-pixel masking
            (e.g. for float/count data with no fixed ceiling).

    Returns:
        A ``DataDrivenMask``.
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected a stack of shape (N, H, W), got {stack.shape}")
    n = stack.shape[0]
    dark_frac = (stack <= dark_value).mean(axis=0)
    valid = dark_frac < dark_frac_thresh
    if saturation_value is not None:
        hot_frac = (stack >= saturation_value).mean(axis=0)
        valid &= hot_frac < hot_frac_thresh
    n_masked = int((~valid).sum())
    logger.info(
        "Data-driven mask from %d frames: %d pixels masked (%.3f%%)",
        n,
        n_masked,
        100.0 * n_masked / valid.size,
    )
    return DataDrivenMask(valid=valid, dark_frac=dark_frac, n_frames=n)


def combine_masks(*masks: np.ndarray) -> np.ndarray:
    """Logical AND of any number of boolean valid-pixel masks (must all be the same shape)."""
    if not masks:
        raise ValueError("combine_masks requires at least one mask")
    out = masks[0].copy()
    for m in masks[1:]:
        if m.shape != out.shape:
            raise ValueError(f"Mask shape mismatch: {m.shape} vs {out.shape}")
        out &= m
    return out
