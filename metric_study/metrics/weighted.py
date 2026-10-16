"""Peak-weighted MSE — metric #9 (plan §3).

Up-weights pixels the geometry-free feature detector (``metric_study.features``)
flagged as ring or spot signal on the *target*, so a reconstruction that reproduces
the smooth background well but drops fine rings/peaks is penalized more than plain
MSE would penalize it. The weight map is computed on the fly per image (never
stored), since it's cheap and depends only on the target.
"""
from __future__ import annotations

import numpy as np
import torch

from .base import check_shapes, masked_fill_like


def build_weight_map(shape: tuple[int, int], ridges, spots, beta: float) -> np.ndarray:
    """Build ``W = 1 + beta * S(target)`` where ``S`` is 1 on any detected ridge/spot pixel.

    Args:
        shape: (H, W) of the target image.
        ridges: List of ``RidgeComponent`` (from ``metric_study.features``).
        spots: List of ``SpotComponent``.
        beta: Weight multiplier for feature pixels (plan §3 tests beta in {10, 100}).

    Returns:
        A float32 weight map, ``shape``, with value 1.0 on background and
        ``1 + beta`` on detected ring/spot pixels.
    """
    feature_mask = np.zeros(shape, dtype=bool)
    for r in ridges:
        feature_mask |= r.mask
    for s in spots:
        feature_mask |= s.mask
    return np.where(feature_mask, 1.0 + beta, 1.0).astype(np.float32)


def peak_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    geometry=None,
    weight_map: torch.Tensor | None = None,
    **cfg,
) -> torch.Tensor:
    """Peak-weighted MSE (plan §3, metric #9).

    Args:
        pred, target: Model-space images, shape (B, H, W).
        mask: Boolean valid-pixel mask.
        weight_map: Precomputed per-image weight map from :func:`build_weight_map`
            (as a (B, H, W) tensor), stacked by the caller from each image's own
            target-derived features. Required — there is no default, since the
            weight map must come from the target's own feature decomposition, not a
            generic prior.

    Returns:
        Per-image mean weighted squared error, shape (B,).
    """
    check_shapes(pred, target, mask)
    if weight_map is None:
        raise ValueError("peak_weighted_mse requires weight_map (from features.build_weight_map per image)")
    if weight_map.shape != pred.shape:
        raise ValueError(f"weight_map shape {weight_map.shape} does not match pred shape {pred.shape}")
    pred = masked_fill_like(pred, target, mask)
    mask_f = mask.to(dtype=pred.dtype)
    w = weight_map.to(dtype=pred.dtype) * mask_f
    denom = w.flatten(1).sum(dim=1).clamp_min(1e-12)
    return (w * (pred - target) ** 2).flatten(1).sum(dim=1) / denom
