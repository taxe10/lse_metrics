"""Controlled, parameterized degradations applied to ground-truth patterns (plan §2).

Every function here is pure: ``degrade(image, mask, features, severity, rng) -> image'``.
They operate on pseudo-linear (or true linear, where available) intensity, at native
resolution, inside the valid mask, before the image is re-encoded to model space and
downsampled (plan §1.1, §1.3). Determinism is the caller's responsibility: pass a
``numpy.random.Generator`` seeded per plan §5.4
(``np.random.default_rng([seed, hash(image_id), deg_id, sev_idx, rep])``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import ndimage as ndi

from .features import FeatureSet, RidgeComponent, SpotComponent


def attenuate_component(
    image: np.ndarray, component_mask: np.ndarray, background: np.ndarray, a: float
) -> np.ndarray:
    """Attenuate one component toward its local background: ``I' = I - a*(I - B)``.

    Args:
        image: Native-resolution image.
        component_mask: Boolean mask of the component (ring or spot) to attenuate.
        background: Local background estimate, same shape as ``image``.
        a: Attenuation fraction in [0, 1]. 0 = no change, 1 = fully removed to background.

    Returns:
        A new image with the component attenuated; input is not modified in place.
    """
    out = image.copy()
    out[component_mask] = image[component_mask] - a * (image[component_mask] - background[component_mask])
    return out


def fine_ring_attenuation(
    image: np.ndarray, mask: np.ndarray, features: FeatureSet, a: float, ring_index: int = 0
) -> np.ndarray:
    """Attenuate one fine ring by fraction ``a`` (plan §2, "fine-ring attenuation").

    A physical ring may be fragmented into several connected components sharing the
    same matched scale (e.g. by noise dips along its annulus; see
    ``features.detect_ridges``). All fragments at the selected scale are attenuated
    together, so this degrades the whole ring, not one arc of it.

    Args:
        image: Native-resolution image.
        mask: Boolean valid-pixel mask (unused directly; kept for signature symmetry
            with other degradations and future mask-aware variants).
        features: Output of :func:`~metric_study.features.extract_features`.
        a: Attenuation fraction in [0, 1].
        ring_index: Which fine ring (by distinct matched scale) to target,
            ordered narrowest-first; clamped to the available count. The "weakest"
            and "median" variants (plan §2) select by sorting the same distinct-scale
            groups on total ``excess_intensity`` before calling this.

    Returns:
        The degraded image, unchanged if no fine ridge was detected.
    """
    fine_ridges = [r for r in features.ridges if r.is_fine]
    if not fine_ridges:
        return image.copy()
    widths = sorted({r.width_px for r in fine_ridges})
    target_width = widths[min(ring_index, len(widths) - 1)]
    combined_mask = np.zeros_like(mask)
    for r in fine_ridges:
        if r.width_px == target_width:
            combined_mask |= r.mask
    return attenuate_component(image, combined_mask, features.background, a)


def broad_ring_damage(image: np.ndarray, mask: np.ndarray, features: FeatureSet, a: float) -> np.ndarray:
    """Attenuate the dominant broad ridge by fraction ``a`` (plan §2, MSE-visible control).

    A physical ring can be fragmented into several connected components (e.g. by
    noise dips along its annulus; see ``features.detect_ridges``), all flagged
    ``is_broad`` together. Every such fragment is attenuated, not just one, so this
    degrades the whole ring rather than one arc of it.
    """
    broad = [r for r in features.ridges if r.is_broad]
    if not broad:
        return image.copy()
    combined_mask = np.zeros_like(mask)
    for r in broad:
        combined_mask |= r.mask
    return attenuate_component(image, combined_mask, features.background, a)


def peak_removal(
    image: np.ndarray, mask: np.ndarray, features: FeatureSet, f: float, a: float, rng: np.random.Generator
) -> np.ndarray:
    """Attenuate a random fraction ``f`` of detected spots by amplitude ``a`` (plan §2).

    Args:
        image: Native-resolution image.
        mask: Boolean valid-pixel mask.
        features: Output of :func:`~metric_study.features.extract_features`.
        f: Fraction of detected spots to attenuate, in [0, 1].
        a: Attenuation fraction applied to each selected spot, in [0, 1].
        rng: Seeded random generator controlling which spots are selected.

    Returns:
        The degraded image, unchanged if no spots were detected.
    """
    if not features.spots:
        return image.copy()
    n_select = max(1, round(f * len(features.spots))) if f > 0 else 0
    if n_select == 0:
        return image.copy()
    idx = rng.choice(len(features.spots), size=n_select, replace=False)
    out = image.copy()
    for i in idx:
        spot = features.spots[i]
        out = attenuate_component(out, spot.mask, features.background, a)
    return out


def gaussian_blur(image: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Mask-normalized Gaussian blur, ``(G*(I*M))/(G*M)`` (plan §2), so gaps don't bleed in.

    Args:
        image: Native-resolution image.
        mask: Boolean valid-pixel mask.
        sigma: Blur std, in pixels. sigma=0 returns the input unchanged.

    Returns:
        The blurred image; invalid pixels are left at 0.
    """
    if sigma <= 0:
        return image.copy()
    m = mask.astype(np.float32)
    num = ndi.gaussian_filter(image * m, sigma=sigma)
    den = ndi.gaussian_filter(m, sigma=sigma)
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)
    return np.where(mask, out, 0.0).astype(image.dtype)


def _local_normal(image: np.ndarray, mask: np.ndarray, centroid: tuple[float, float]) -> np.ndarray:
    """Unit gradient direction at ``centroid`` — a geometry-free stand-in for the ring
    normal (no beam center is used; this is purely the local intensity gradient)."""
    gy, gx = np.gradient(ndi.gaussian_filter(np.where(mask, image, 0.0), sigma=2.0))
    r, c = int(round(centroid[0])), int(round(centroid[1]))
    r = np.clip(r, 0, image.shape[0] - 1)
    c = np.clip(c, 0, image.shape[1] - 1)
    vec = np.array([gy[r, c], gx[r, c]], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.array([1.0, 0.0])
    return vec / norm


def feature_displacement(
    image: np.ndarray,
    mask: np.ndarray,
    features: FeatureSet,
    component: RidgeComponent | SpotComponent,
    delta_px: float,
) -> np.ndarray:
    """Shift one component by ``delta_px`` along its local gradient direction (plan §2).

    A geometry-free stand-in for a radial shift: the "normal" direction is the local
    image intensity gradient at the component's centroid, not a beam-center-relative
    radius (no calibration is used anywhere in this study). The vacated region is
    filled with the local background.

    Args:
        image: Native-resolution image.
        mask: Boolean valid-pixel mask.
        features: Output of :func:`~metric_study.features.extract_features` (provides
            the background used to fill the vacated region).
        component: The ridge or spot component to move.
        delta_px: Signed displacement magnitude, in pixels.

    Returns:
        The degraded image.
    """
    comp_mask = component.mask
    centroid = (
        component.centroid if isinstance(component, SpotComponent) else tuple(np.argwhere(comp_mask).mean(axis=0))
    )
    direction = _local_normal(image, mask, centroid)
    out = image.copy()
    out[comp_mask] = features.background[comp_mask]  # vacate
    shift = (direction * delta_px).tolist()
    moved = ndi.shift(np.where(comp_mask, image, 0.0), shift=shift, order=1, mode="constant", cval=0.0)
    moved_mask = ndi.shift(comp_mask.astype(np.float32), shift=shift, order=1, mode="constant", cval=0.0) > 0.5
    moved_mask &= mask
    out[moved_mask] = moved[moved_mask]
    return out


def global_scaling(image: np.ndarray, mask: np.ndarray, s: float) -> np.ndarray:
    """``I' = s * I`` (plan §2)."""
    return (image * s).astype(image.dtype)


def additive_offset(image: np.ndarray, mask: np.ndarray, c: float) -> np.ndarray:
    """``I' = I + c * median(I_valid)`` (plan §2)."""
    med = float(np.median(image[mask])) if mask.any() else 0.0
    return (image + c * med).astype(image.dtype)


def poisson_resample(image: np.ndarray, mask: np.ndarray, mean_counts: float, rng: np.random.Generator) -> np.ndarray:
    """Resample at a target mean-counts-per-pixel level: ``Poisson(lambda*I)/lambda`` (plan §2).

    This is the noise-floor degradation: a metric should NOT strongly penalize this,
    since it's what a perfect reconstruction of noisy data looks like. Flagged
    pseudo-count unless ``image`` is true linear counts (plan §1.1, §3 metric #3).

    Args:
        image: Native-resolution image, pseudo-linear (or true linear) space.
        mask: Boolean valid-pixel mask.
        mean_counts: Target mean counts per valid pixel.
        rng: Seeded random generator.

    Returns:
        The resampled image.
    """
    valid_mean = float(image[mask].mean()) if mask.any() and image[mask].mean() > 0 else 1.0
    lam = mean_counts / valid_mean
    scaled = np.clip(image * lam, 0, None)
    sampled = rng.poisson(scaled).astype(np.float64)
    out = sampled / lam
    return np.where(mask, out, image).astype(image.dtype)


@dataclass
class DegradationSpec:
    """One entry in the degradation registry.

    Attributes:
        name: Registry key, matches the ``degradation`` column in metrics.parquet.
        needs_features: Whether the function requires a ``FeatureSet``.
        needs_rng: Whether the function requires an ``np.random.Generator``.
        fn: The pure degradation function.
    """

    name: str
    needs_features: bool
    needs_rng: bool
    fn: Callable


REGISTRY: dict[str, DegradationSpec] = {
    "fine_ring_attenuation": DegradationSpec("fine_ring_attenuation", True, False, fine_ring_attenuation),
    "broad_ring_damage": DegradationSpec("broad_ring_damage", True, False, broad_ring_damage),
    "peak_removal": DegradationSpec("peak_removal", True, True, peak_removal),
    "gaussian_blur": DegradationSpec("gaussian_blur", False, False, gaussian_blur),
    "global_scaling": DegradationSpec("global_scaling", False, False, global_scaling),
    "additive_offset": DegradationSpec("additive_offset", False, False, additive_offset),
    "poisson_resampling": DegradationSpec("poisson_resampling", False, True, poisson_resample),
}


def severity_rng(seed: int, image_id: str, deg_id: str, severity_idx: int, rep: int = 0) -> np.random.Generator:
    """Deterministic per-(image, degradation, severity, rep) RNG (plan §5.4).

    Uses a stable hash (``hashlib.sha256``), not Python's built-in ``hash()``, which
    is randomized per-process (``PYTHONHASHSEED``) and would silently break the
    plan's requirement that results be identical regardless of chunking or worker
    order (plan §5.3 "Pipeline" tests, §5.4).
    """
    import hashlib

    key = f"{image_id}:{deg_id}:{severity_idx}:{rep}"
    key_hash = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng([seed, key_hash])
