"""Shared fixtures: synthetic scattering-like images and access to the sample data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from metric_study.masks import PILATUS2M_SHAPE, static_gap_mask

SAMPLE_DATA_DIR = Path("/Users/aianalyticscomputing/Documents/sample_data")


@pytest.fixture(scope="session")
def native_shape():
    return PILATUS2M_SHAPE


@pytest.fixture(scope="session")
def native_mask(native_shape):
    return static_gap_mask(native_shape, dilation_px=2)


@pytest.fixture(scope="session")
def synthetic_rings_image(native_shape):
    """A synthetic image with one broad ring, one fine ring, and a few spots, on flat background."""
    h, w = native_shape
    cy, cx = h * 0.65, w * 0.5  # off-center "beam", like the real sample frames
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    rng = np.random.default_rng(42)
    background = 20.0 + 0.0 * r
    # Widths chosen within the ridge detector's tested scale range (sigma up to 8 px,
    # metric_study.features.RIDGE_SCALES), and with the broad ring's much larger
    # amplitude giving it the majority of total ridge excess intensity, so both
    # `is_fine` and `is_broad` are exercised (plan §2 fine/broad ridge thresholds).
    broad_ring = 200.0 * np.exp(-((r - 300) ** 2) / (2 * 6.0**2))  # sigma=6 -> FWHM ~14 px (broad)
    fine_ring = 40.0 * np.exp(-((r - 500) ** 2) / (2 * 1.5**2))  # sigma=1.5 -> FWHM ~3.5 px (fine)
    image = background + broad_ring + fine_ring
    # A handful of compact Bragg-like spots.
    for (sy, sx) in [(400, 600), (420, 640), (380, 580), (450, 700), (300, 500), (350, 550), (500, 400), (550, 450)]:
        yy_s, xx_s = np.mgrid[0:h, 0:w]
        d = np.sqrt((yy_s - sy) ** 2 + (xx_s - sx) ** 2)
        image += 80.0 * np.exp(-(d**2) / (2 * 1.5**2))
    # Kept low: gamma-normalized Hessian ridge response is roughly scale-invariant for
    # white noise, so noisier fixtures compete with real structure across all tested
    # scales (features.RIDGE_SCALES) and fragment ridge detection into noise blobs.
    # Real-data noise robustness is a separate tuning question for the actual study.
    image += rng.normal(0, 0.05, size=image.shape)
    return image.astype(np.float32)


@pytest.fixture
def small_synthetic_pair():
    """A small (64, 64) image pair for fast metric unit tests (no ring/spot structure needed)."""
    rng = np.random.default_rng(0)
    target = rng.uniform(0, 1, size=(64, 64)).astype(np.float32)
    mask = np.ones((64, 64), dtype=bool)
    mask[:4, :] = False  # a small invalid strip, like a module gap
    return target, mask


def has_sample_data() -> bool:
    return SAMPLE_DATA_DIR.exists() and any(SAMPLE_DATA_DIR.glob("*.jpg"))


requires_sample_data = pytest.mark.skipif(not has_sample_data(), reason="sample_data directory not found")
