"""Per-pixel metrics: #1 (MSE linear/model-space) and #2 (log/asinh MSE+L1), plus
negative Pearson correlation (#5) (plan §3).

All functions share the signature ``metric(pred, target, mask, geometry=None, **cfg)
-> Tensor`` of shape (B,). ``geometry`` is accepted and ignored everywhere in this
module (kept for the deferred I(q)-based metrics, plan §3 note on #7/#8).
"""
from __future__ import annotations

import torch

from .base import check_shapes, masked_fill_like

_EPS_DEFAULT = 1.0


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-image mean of ``x`` over valid pixels, batched over the leading dim."""
    mask_f = mask.to(dtype=x.dtype)
    denom = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
    return (x * mask_f).flatten(1).sum(dim=1) / denom


def mse_linear(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, geometry=None, **cfg) -> torch.Tensor:
    """Masked MSE in (pseudo-)linear intensity space — baseline metric #1."""
    check_shapes(pred, target, mask)
    pred = masked_fill_like(pred, target, mask)
    return _masked_mean((pred - target) ** 2, mask)


def mse_model_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, geometry=None, **cfg) -> torch.Tensor:
    """Masked MSE in model ([0,1]) space — same as ``mse_linear`` but named for clarity
    when both domains are logged side by side (plan §3, metric #1)."""
    return mse_linear(pred, target, mask, geometry, **cfg)


def mse_model_unmasked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, geometry=None, **cfg) -> torch.Tensor:
    """The literal current training loss: unmasked ``mean((recon - x)**2)`` over ALL
    pixels including gaps/beamstop (``training_2.py:203``, ``F.mse_loss(recon, x,
    reduction="mean")``). ``mask`` is accepted for signature symmetry but ignored —
    that is the point of this metric: it shows how much of the training loss is
    spent on pixels that carry no scientific signal (plan §4.6, §8)."""
    check_shapes(pred, target, mask)
    return ((pred - target) ** 2).flatten(1).mean(dim=1)


def _log_transform(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=0.0) + eps)


def _asinh_transform(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.asinh(x / scale)


def log_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, geometry=None, eps: float = _EPS_DEFAULT, **cfg) -> torch.Tensor:
    """MSE on ``log(I + eps)`` (plan §3, metric #2)."""
    check_shapes(pred, target, mask)
    pred = masked_fill_like(pred, target, mask)
    diff = _log_transform(pred, eps) - _log_transform(target, eps)
    return _masked_mean(diff**2, mask)


def log_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, geometry=None, eps: float = _EPS_DEFAULT, **cfg) -> torch.Tensor:
    """L1 on ``log(I + eps)`` (plan §3, metric #2)."""
    check_shapes(pred, target, mask)
    pred = masked_fill_like(pred, target, mask)
    diff = _log_transform(pred, eps) - _log_transform(target, eps)
    return _masked_mean(diff.abs(), mask)


def asinh_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, geometry=None, scale: float = 1.0, **cfg) -> torch.Tensor:
    """MSE on ``asinh(I / scale)`` (plan §3, metric #2)."""
    check_shapes(pred, target, mask)
    pred = masked_fill_like(pred, target, mask)
    diff = _asinh_transform(pred, scale) - _asinh_transform(target, scale)
    return _masked_mean(diff**2, mask)


def asinh_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, geometry=None, scale: float = 1.0, **cfg) -> torch.Tensor:
    """L1 on ``asinh(I / scale)`` (plan §3, metric #2)."""
    check_shapes(pred, target, mask)
    pred = masked_fill_like(pred, target, mask)
    diff = _asinh_transform(pred, scale) - _asinh_transform(target, scale)
    return _masked_mean(diff.abs(), mask)


def neg_pearson(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, geometry=None, **cfg) -> torch.Tensor:
    """Negative Pearson correlation over valid pixels (plan §3, metric #5).

    Invariant to per-image affine transforms of ``pred`` (scale and offset), by
    construction — tested in ``tests/test_metrics.py``.
    """
    check_shapes(pred, target, mask)
    pred = masked_fill_like(pred, target, mask)
    mask_f = mask.to(dtype=pred.dtype)
    n = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)

    p = pred.flatten(1)
    t = target.flatten(1)
    m = mask_f.flatten(1)

    p_mean = (p * m).sum(dim=1) / n
    t_mean = (t * m).sum(dim=1) / n
    p_c = (p - p_mean.unsqueeze(1)) * m
    t_c = (t - t_mean.unsqueeze(1)) * m

    cov = (p_c * t_c).sum(dim=1)
    p_std = torch.sqrt((p_c**2).sum(dim=1).clamp_min(1e-12))
    t_std = torch.sqrt((t_c**2).sum(dim=1).clamp_min(1e-12))
    r = cov / (p_std * t_std)
    return -r
