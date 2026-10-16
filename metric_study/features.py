"""Geometry-free ring and peak detection, used by degradations, the peak-weighted
metric, and the automatic pattern-type classifier (plan §2, §6.5).

No beam center, detector distance or wavelength is used anywhere here — everything
is derived directly from the 2D image, per the plan's scope decision to defer
calibration-dependent analysis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import blob_log, hessian_matrix, hessian_matrix_eigvals
from skimage.measure import label, regionprops

logger = logging.getLogger(__name__)

RIDGE_SCALES = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
FWHM_FACTOR = 2.3548  # 2*sqrt(2*ln2), converts a Gaussian sigma to FWHM


@dataclass
class RidgeComponent:
    """One connected ring-like (ridge) component.

    Attributes:
        mask: Boolean pixel mask of this component, native image shape.
        width_px: Estimated FWHM in pixels (from the best-response scale).
        excess_intensity: Sum of ``image - background`` over the component.
        is_fine: True if ``width_px <= fine_threshold``.
        is_broad: True if ``width_px >= broad_threshold`` and it carries a large
            share of the total ridge excess intensity.
    """

    mask: np.ndarray
    width_px: float
    excess_intensity: float
    is_fine: bool = False
    is_broad: bool = False


@dataclass
class SpotComponent:
    """One connected Bragg-peak-like (compact, non-ridge) component.

    Attributes:
        mask: Boolean pixel mask of this component.
        area_px: Number of pixels in the component.
        excess_intensity: Sum of ``image - background`` over the component.
        centroid: (row, col) centroid.
    """

    mask: np.ndarray
    area_px: float
    excess_intensity: float
    centroid: tuple[float, float]


@dataclass
class FeatureSet:
    """All geometry-free features extracted from one target image.

    Attributes:
        background: Smooth background estimate, same shape as the input image.
        ridges: Detected ring-like components.
        spots: Detected Bragg-peak-like components.
        snr: A simple global signal-to-noise proxy (see :func:`extract_features`).
        saturated_fraction: Fraction of valid pixels at/above the saturation value.
    """

    background: np.ndarray
    ridges: list[RidgeComponent] = field(default_factory=list)
    spots: list[SpotComponent] = field(default_factory=list)
    snr: float = 0.0
    saturated_fraction: float = 0.0


def estimate_background(image: np.ndarray, mask: np.ndarray, opening_size: int = 25) -> np.ndarray:
    """Smooth background via mask-aware inpainting + large-kernel morphological opening.

    Args:
        image: Native or downsampled intensity image.
        mask: Boolean valid-pixel mask.
        opening_size: Size (px) of the morphological opening structuring element;
            should be larger than the widest ring of interest so rings are removed,
            leaving only the slowly-varying background.

    Returns:
        A background image the same shape as ``image``, with invalid (gap/beamstop)
        pixels filled in by normalized-convolution inpainting so they don't bleed
        dark values into the opening.
    """
    m = mask.astype(np.float32)
    sigma = max(opening_size / 3.0, 1.0)
    num = ndi.gaussian_filter(image * m, sigma=sigma)
    den = ndi.gaussian_filter(m, sigma=sigma)
    inpainted = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)
    background = ndi.grey_opening(inpainted, size=opening_size)
    return background.astype(np.float32)


def _best_scale_ridge_response(
    signal: np.ndarray, mask: np.ndarray, scales: tuple[float, ...] = RIDGE_SCALES
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-scale Hessian ridge filter (plan §2). Returns (response, best_scale) maps."""
    best_response = np.zeros_like(signal, dtype=np.float32)
    best_scale = np.zeros_like(signal, dtype=np.float32)
    for sigma in scales:
        hxx, hxy, hyy = hessian_matrix(signal, sigma=sigma, mode="reflect", use_gaussian_derivatives=True)
        eig1, eig2 = hessian_matrix_eigvals((hxx, hxy, hyy))
        # Ridges are bright, elongated structures: large-magnitude negative eigenvalue
        # in the direction across the ridge. Normalize by scale (gamma-normalized
        # derivatives) so responses are comparable across sigma.
        response = np.maximum(0.0, -eig2) * (sigma**2)
        better = response > best_response
        best_response = np.where(better, response, best_response)
        best_scale = np.where(better, sigma, best_scale)
    best_response = np.where(mask, best_response, 0.0)
    return best_response, best_scale


def detect_ridges(
    image: np.ndarray,
    mask: np.ndarray,
    background: np.ndarray,
    fine_threshold_px: float = 8.0,
    broad_threshold_px: float = 12.0,
    response_percentile: float = 99.0,
    min_component_px: int = 30,
    scales: tuple[float, ...] = RIDGE_SCALES,
) -> list[RidgeComponent]:
    """Detect ring-like (ridge) components in ``image - background`` (plan §2).

    Binarization and connected-component labeling are done **per matched scale**
    (i.e. restricted to pixels whose best-response scale is ``s``, using a threshold
    computed within that scale's own response distribution), not on one combined map
    across all scales. Gamma-normalized Hessian response is comparable in magnitude
    across a wide range of feature amplitudes and widths, so a single global
    threshold lets a strong, broad feature's response dominate the percentile cutoff
    and swamp a real but weaker fine ring; per-scale thresholding is scale-space
    non-maximum suppression, which avoids that and also keeps components from
    different physical rings from merging into one connected blob.

    Args:
        image: Target image (native or downsampled).
        mask: Boolean valid-pixel mask.
        background: Smooth background, from :func:`estimate_background`.
        fine_threshold_px: Rings with FWHM at or below this are "fine".
        broad_threshold_px: Rings with FWHM at or above this are "broad" candidates.
        response_percentile: Percentile of the ridge response, computed *within*
            each scale's own assigned pixels, used as that scale's binarization
            threshold.
        min_component_px: Discard connected components smaller than this.
        scales: The scale set to bucket pixels into (must match what produced
            ``best_scale`` via :func:`_best_scale_ridge_response`).

    Returns:
        A list of :class:`RidgeComponent`, with the single largest-excess-intensity
        component flagged ``is_broad`` if it clears ``broad_threshold_px`` and
        carries more than half of the total ridge excess intensity.
    """
    signal = image - background
    response, best_scale = _best_scale_ridge_response(signal, mask, scales)
    if not np.any(response > 0):
        return []

    components: list[RidgeComponent] = []
    scale_of: list[float] = []
    total_excess = 0.0
    for s in scales:
        scale_mask = (best_scale == s) & mask
        if not scale_mask.any():
            continue
        thresh = np.percentile(response[scale_mask], response_percentile)
        binary = scale_mask & (response >= thresh)
        if not binary.any():
            continue
        # A light closing bridges single-pixel noise dips along an otherwise
        # continuous ring; it does not need to (and at this size cannot) merge an
        # entire fragmented annulus into one component -- that's handled below by
        # aggregating excess intensity per matched scale, not per connected piece.
        binary = ndi.binary_closing(binary, structure=np.ones((3, 3)), iterations=1) & mask
        labeled, n = label(binary, return_num=True)
        for region in regionprops(labeled):
            if region.area < min_component_px:
                continue
            comp_mask = labeled == region.label
            width_px = float(s * FWHM_FACTOR)
            excess = float(signal[comp_mask].sum())
            total_excess += max(excess, 0.0)
            components.append(RidgeComponent(mask=comp_mask, width_px=width_px, excess_intensity=excess))
            scale_of.append(s)

    for comp in components:
        comp.is_fine = comp.width_px <= fine_threshold_px

    # "Broad" is a property of a physical ring, which may be fragmented into many
    # small connected pieces (e.g. by noise dips along its annulus, plan §2's
    # ridge components). Aggregate excess intensity by matched scale -- not by
    # connected fragment -- so a ring that dominates the image's total ridge excess
    # is correctly flagged even when no single fragment alone clears 50% (plan §2's
    # broad-ring definition: "carries high integrated excess intensity").
    if components and total_excess > 0:
        scale_totals: dict[float, float] = {}
        for comp, s in zip(components, scale_of):
            scale_totals[s] = scale_totals.get(s, 0.0) + max(comp.excess_intensity, 0.0)
        dominant_scale = max(scale_totals, key=scale_totals.get)
        if dominant_scale >= broad_threshold_px / FWHM_FACTOR and scale_totals[dominant_scale] / total_excess > 0.5:
            for comp, s in zip(components, scale_of):
                if s == dominant_scale:
                    comp.is_broad = True
    return components


def detect_spots(
    image: np.ndarray,
    mask: np.ndarray,
    background: np.ndarray,
    min_sigma: float = 1.0,
    max_sigma: float = 4.0,
    threshold_sigma: float = 4.0,
    min_area_px: int = 1,
    max_area_px: int = 400,
    max_elongation: float = 3.0,
) -> list[SpotComponent]:
    """Detect compact, non-ridge Bragg-peak-like components (plan §2).

    Args:
        image: Target image.
        mask: Boolean valid-pixel mask.
        background: Smooth background, from :func:`estimate_background`.
        min_sigma, max_sigma: LoG blob-detection scale range, in pixels.
        threshold_sigma: Detection threshold as a multiple of the noise std
            (estimated from the masked, background-subtracted signal).
        min_area_px, max_area_px: Component-size filter (plan: 1-400 px).
        max_elongation: Discard components more elongated than this (major/minor
            axis ratio) — those are ridge fragments, not spots.

    Returns:
        A list of :class:`SpotComponent`.
    """
    signal = np.where(mask, image - background, 0.0)
    positive_signal = np.clip(signal, 0, None)
    noise_std = float(np.std(signal[mask])) if mask.any() else 0.0
    if noise_std <= 0:
        return []
    # `threshold` must be on the SAME scale as the image passed to blob_log (here,
    # unscaled `positive_signal`) -- an earlier version divided the threshold by 255
    # while sometimes passing an unscaled image, so the effective threshold could be
    # ~250x too low. On a near-flat/low-signal image that made blob_log detect a
    # blob at nearly every pixel across the full ~2.5M-pixel frame, which is what
    # actually hung (not looped, just extremely slow) on the low-signal test case.
    threshold = threshold_sigma * noise_std
    blobs = blob_log(
        positive_signal,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=5,
        threshold=max(threshold, 1e-6),
    )
    if len(blobs) > 2000:
        # A safety net, not a normal code path: this many candidates means the
        # threshold has effectively collapsed (e.g. a future scale-unit bug, or a
        # pathological input), and continuing would be a de facto hang at 1.5M-image
        # scale. Keep only the strongest candidates instead of silently stalling.
        logger.warning("detect_spots: %d candidate blobs, truncating to the 2000 strongest", len(blobs))
        strengths = [positive_signal[int(r), int(c)] for r, c, _ in blobs]
        keep = np.argsort(strengths)[::-1][:2000]
        blobs = blobs[keep]
    spots: list[SpotComponent] = []
    for r, c, sigma in blobs:
        rad = max(sigma * np.sqrt(2), 1.0)
        r0, r1 = int(max(r - rad, 0)), int(min(r + rad + 1, image.shape[0]))
        c0, c1 = int(max(c - rad, 0)), int(min(c + rad + 1, image.shape[1]))
        comp_mask = np.zeros_like(mask)
        comp_mask[r0:r1, c0:c1] = True
        comp_mask &= mask
        area = int(comp_mask.sum())
        if not (min_area_px <= area <= max_area_px):
            continue
        props = regionprops(comp_mask.astype(int))
        if props:
            axis_major = props[0].axis_major_length or 1.0
            axis_minor = props[0].axis_minor_length or 1.0
            elongation = axis_major / max(axis_minor, 1e-6)
            if elongation > max_elongation:
                continue
        excess = float(signal[comp_mask].sum())
        spots.append(SpotComponent(mask=comp_mask, area_px=area, excess_intensity=excess, centroid=(r, c)))
    return spots


def extract_features(image: np.ndarray, mask: np.ndarray, saturation_value: float = 255.0) -> FeatureSet:
    """Run the full geometry-free feature decomposition on one target image (plan §2)."""
    background = estimate_background(image, mask)
    ridges = detect_ridges(image, mask, background)
    spots = detect_spots(image, mask, background)
    valid = image[mask]
    # SNR is computed on the background-SUBTRACTED signal (excess above the smooth
    # background), not on raw intensity: a flat, bright detector background has a
    # high raw percentile/std ratio despite having no rings or peaks at all, which
    # would wrongly avoid the "low_signal" classification below. Excess intensity is
    # near zero for a genuinely flat frame, giving a low ratio as intended.
    signal = (image - background)[mask]
    signal_level = float(np.percentile(signal, 99)) if signal.size else 0.0
    noise_level = float(np.std(signal)) if signal.size else 0.0
    snr = signal_level / noise_level if noise_level > 0 else 0.0
    saturated_fraction = float((valid >= saturation_value).mean()) if valid.size else 0.0
    return FeatureSet(background=background, ridges=ridges, spots=spots, snr=snr, saturated_fraction=saturated_fraction)


def classify_pattern(features: FeatureSet, spot_count_threshold: int = 8, snr_threshold: float = 3.0) -> str:
    """Rule-based pattern classifier (plan §6.5): assigns one primary class.

    Rules, in priority order:
        (x) low-signal: SNR below ``snr_threshold``.
        (a) broad: a ridge is flagged ``is_broad`` and carries most excess intensity.
        (c) peaks/texture: spot count at or above ``spot_count_threshold``.
        (b) fine/weak rings: any ridge flagged ``is_fine``.
        (a) broad (fallback): any ridge present at all.
        (x) low-signal (default): no ridges or spots detected.

    Args:
        features: Output of :func:`extract_features`.
        spot_count_threshold: Minimum spot count for class (c).
        snr_threshold: SNR below this is treated as low-signal, regardless of
            detected ridges/spots (matches the "low-signal/blank" stratum, plan §0.2).

    Returns:
        One of ``"broad"``, ``"fine"``, ``"peaks"``, ``"low_signal"``.
    """
    if features.snr < snr_threshold:
        return "low_signal"
    if any(r.is_broad for r in features.ridges):
        return "broad"
    if len(features.spots) >= spot_count_threshold:
        return "peaks"
    if any(r.is_fine for r in features.ridges):
        return "fine"
    if features.ridges:
        return "broad"
    return "low_signal"
