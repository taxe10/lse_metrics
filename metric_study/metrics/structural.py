"""SSIM and MS-SSIM on log/model-space images — metric #4 (plan §3).

Implemented directly (not via ``pytorch_msssim``) so the SSIM/contrast-structure
maps can be masked correctly: each output pixel's SSIM value depends on a
``win_size`` neighborhood, so a masked pixel must invalidate every SSIM-map pixel
whose window touches it, not just itself. That's done with a min-pool erosion of
the mask by ``win_size // 2`` before averaging. ``pytorch_msssim`` doesn't expose the
per-pixel map, so it can't do this; it remains an installed cross-check dependency
for unmasked reference values in tests only.

Constants (plan §3, metric #4): Gaussian window, sigma=1.5, win_size=11 (7 at 512
where noted), K=(0.01, 0.03), data_range=1.0 (model space is already in [0, 1]).
MS-SSIM needs ``min(H, W) > (win_size - 1) * 2**(n_scales - 1)``; at 5 scales and
win_size=11, that's 160 px, which both native and 512 satisfy (plan §3).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import check_shapes, masked_fill_like

_MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _gaussian_window(win_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(win_size, dtype=dtype, device=device) - (win_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = g.outer(g)
    return window_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, win, win)


def _erode_mask(mask: torch.Tensor, win_size: int) -> torch.Tensor:
    """Min-pool ``mask`` so a pixel is True only if its whole win_size window is valid."""
    pad = win_size // 2
    m = mask.to(dtype=torch.float32).unsqueeze(1)  # (B, 1, H, W)
    eroded = -F.max_pool2d(-m, kernel_size=win_size, stride=1, padding=pad)
    return (eroded.squeeze(1) > 0.5)


def _ssim_maps(
    pred: torch.Tensor, target: torch.Tensor, win_size: int, win_sigma: float, data_range: float, k1: float, k2: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (ssim_map, cs_map), each (B, H', W') with H'=H-win_size+1 (valid conv)."""
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    window = _gaussian_window(win_size, win_sigma, pred.device, pred.dtype)
    x = pred.unsqueeze(1)
    y = target.unsqueeze(1)

    mu_x = F.conv2d(x, window)
    mu_y = F.conv2d(y, window)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window) - mu_x2
    sigma_y2 = F.conv2d(y * y, window) - mu_y2
    sigma_xy = F.conv2d(x * y, window) - mu_xy

    cs_map = (2 * sigma_xy + c2) / (sigma_x2 + sigma_y2 + c2)
    l_map = (2 * mu_xy + c1) / (mu_x2 + mu_y2 + c1)
    ssim_map = l_map * cs_map
    return ssim_map.squeeze(1), cs_map.squeeze(1)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    geometry=None,
    win_size: int = 11,
    win_sigma: float = 1.5,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
    **cfg,
) -> torch.Tensor:
    """Masked single-scale SSIM (plan §3, metric #4). Returns per-image mean SSIM, shape (B,).

    Lower-is-worse convention for this study is ``1 - ssim``; this function returns
    raw SSIM (1 = identical) — callers building a "loss" should use ``1 - ssim(...)``.
    """
    check_shapes(pred, target, mask)
    pred = masked_fill_like(pred, target, mask)
    ssim_map, _ = _ssim_maps(pred, target, win_size, win_sigma, data_range, k1, k2)
    eroded = _erode_mask(mask, win_size)[:, : ssim_map.shape[-2], : ssim_map.shape[-1]]
    eroded_f = eroded.to(dtype=ssim_map.dtype)
    denom = eroded_f.flatten(1).sum(dim=1).clamp_min(1.0)
    return (ssim_map * eroded_f).flatten(1).sum(dim=1) / denom


def ms_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    geometry=None,
    win_size: int = 11,
    win_sigma: float = 1.5,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
    weights: tuple[float, ...] = _MS_SSIM_WEIGHTS,
    **cfg,
) -> torch.Tensor:
    """Masked multi-scale SSIM (plan §3, metric #4). Returns per-image mean, shape (B,).

    Requires ``min(H, W) > (win_size - 1) * 2**(len(weights) - 1)``; raises
    ValueError otherwise (plan's noted MS-SSIM size constraint), so a caller can
    catch it and fall back to a smaller ``win_size`` or fewer scales rather than
    silently getting garbage from a too-small image.
    """
    check_shapes(pred, target, mask)
    n_scales = len(weights)
    min_side = min(pred.shape[-2], pred.shape[-1])
    required = (win_size - 1) * (2 ** (n_scales - 1))
    if min_side <= required:
        raise ValueError(
            f"ms_ssim: image size {tuple(pred.shape[-2:])} too small for {n_scales} scales at "
            f"win_size={win_size} (needs > {required}px on the shorter side)"
        )
    pred = masked_fill_like(pred, target, mask)
    p, t, m = pred.unsqueeze(1), target.unsqueeze(1), mask.to(dtype=pred.dtype).unsqueeze(1)

    mcs = []
    for level in range(n_scales):
        ssim_map, cs_map = _ssim_maps(p.squeeze(1), t.squeeze(1), win_size, win_sigma, data_range, k1, k2)
        eroded = _erode_mask(m.squeeze(1) > 0.5, win_size)[:, : cs_map.shape[-2], : cs_map.shape[-1]]
        eroded_f = eroded.to(dtype=cs_map.dtype)
        denom = eroded_f.flatten(1).sum(dim=1).clamp_min(1.0)
        if level < n_scales - 1:
            level_val = (cs_map * eroded_f).flatten(1).sum(dim=1) / denom
            mcs.append(torch.clamp(level_val, min=1e-6))
            p = F.avg_pool2d(p, kernel_size=2)
            t = F.avg_pool2d(t, kernel_size=2)
            m = F.avg_pool2d(m, kernel_size=2)
        else:
            level_val = (ssim_map * eroded_f).flatten(1).sum(dim=1) / denom
            mcs.append(torch.clamp(level_val, min=1e-6))

    out = torch.ones_like(mcs[0])
    for level_val, w in zip(mcs, weights):
        out = out * (level_val**w)
    return out
