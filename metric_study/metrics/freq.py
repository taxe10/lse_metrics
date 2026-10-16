"""Focal frequency loss — metric #6 (plan §3).

This is a **vendored reimplementation** of the formula from Jiang et al., "Focal
Frequency Loss for Image Reconstruction and Synthesis" (ICCV 2021), written directly
against the paper rather than installed from the authors' GitHub repo (the plan
originally named ``EndlessSora/focal-frequency-loss`` as the source; per review, we
do not pull third-party code from a GitHub URL into the study env, and instead
implement the published formula ourselves). It is **not** byte-identical to the
official implementation — differences in exact weight normalization should be
expected, and this is flagged wherever FFL results are reported.

Formula: for each frequency bin, the "focal" weight emphasizes bins where pred and
target disagree most, ``w = |F_pred - F_target|^(2*alpha)`` (stop-gradient,
normalized by its own per-patch max), and the loss is
``mean(w * |F_pred - F_target|^2)``.

Masking (plan §3's ``masked_fill_like`` convention) works correctly here because the
FFT is linear: with ``pred := target`` on invalid pixels, ``F_pred - F_target ==
FFT(pred - target)`` where ``(pred - target)`` is exactly zero on invalid pixels, so
those pixels contribute exactly zero to every frequency bin's difference — no special
per-frequency masking is needed on top of the spatial-domain fill.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import check_shapes, masked_fill_like


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """Zero-pad (bottom/right) so H and W are both multiples of ``multiple``."""
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)


def _patchify(x: torch.Tensor, patch_factor: int) -> torch.Tensor:
    """(B, H, W) -> (B*patch_factor^2, H/patch_factor, W/patch_factor)."""
    b, h, w = x.shape
    ph, pw = h // patch_factor, w // patch_factor
    x = x.view(b, patch_factor, ph, patch_factor, pw)
    x = x.permute(0, 1, 3, 2, 4).contiguous()  # (B, pf, pf, ph, pw)
    return x.view(b * patch_factor * patch_factor, ph, pw)


def focal_frequency_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    geometry=None,
    alpha: float = 1.0,
    patch_factor: int = 1,
    **cfg,
) -> torch.Tensor:
    """Focal frequency loss (plan §3, metric #6, vendored reimplementation — see module docstring).

    Args:
        pred, target: Model-space images, shape (B, H, W).
        mask: Boolean valid-pixel mask.
        alpha: Focal exponent controlling how strongly high-error frequency bins are
            up-weighted (paper default 1.0).
        patch_factor: Split the image into a ``patch_factor x patch_factor`` grid and
            take the FFT of each patch separately (plan tests both 1 and 4).

    Returns:
        Per-image mean focal frequency loss, shape (B,).
    """
    check_shapes(pred, target, mask)
    b = pred.shape[0]
    pred = masked_fill_like(pred, target, mask)
    pred = _pad_to_multiple(pred, patch_factor)
    target = _pad_to_multiple(target, patch_factor)

    pred_p = _patchify(pred, patch_factor)
    target_p = _patchify(target, patch_factor)

    freq_pred = torch.fft.fft2(pred_p, norm="ortho")
    freq_target = torch.fft.fft2(target_p, norm="ortho")
    freq_diff = freq_pred - freq_target
    freq_distance = freq_diff.real**2 + freq_diff.imag**2  # (B*pf^2, ph, pw)

    with torch.no_grad():
        weight = freq_distance**alpha
        max_per_patch = weight.flatten(1).max(dim=1).values.clamp_min(1e-12)
        weight = weight / max_per_patch.view(-1, 1, 1)

    loss_per_patch = (weight * freq_distance).flatten(1).mean(dim=1)  # (B*pf^2,)
    loss_per_image = loss_per_patch.view(b, -1).mean(dim=1)
    return loss_per_image
