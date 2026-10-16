"""Shared masking helper used by every metric (plan §3).

Every metric receives an explicit boolean mask and must be exactly insensitive to
what happens on invalid (gap/beamstop) pixels. The standard way to guarantee that
for local and FFT-based metrics (not just simple means) is to overwrite ``pred``
with ``target`` on invalid pixels before computing anything, so those pixels
contribute zero difference and cannot leak into neighboring valid pixels through a
convolution or FFT.
"""
from __future__ import annotations

import torch


def masked_fill_like(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return a copy of ``pred`` with invalid pixels replaced by ``target``.

    Args:
        pred: Prediction, shape (B, H, W) or (B, 1, H, W).
        target: Target, same shape as ``pred``.
        mask: Boolean valid-pixel mask, broadcastable to ``pred``'s shape (True = valid).

    Returns:
        A new tensor, same shape as ``pred``: equal to ``pred`` where ``mask`` is
        True, and equal to ``target`` where ``mask`` is False.
    """
    mask = mask.to(dtype=torch.bool)
    return torch.where(mask, pred, target)


def check_shapes(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
    """Raise ValueError if pred/target/mask shapes don't match (fail fast, not silently)."""
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")
    if mask.shape != pred.shape:
        raise ValueError(f"mask shape {mask.shape} does not match pred shape {pred.shape}")
