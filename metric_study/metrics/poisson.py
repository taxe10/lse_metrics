"""Poisson NLL and chi-squared — metric #3 (plan §3).

Meaningful only when ``pred``/``target`` are true photon counts. On the sample data
(8-bit tone-mapped JPEGs, no known export transform — plan §9 Q2/Q3) these are only
approximate "pseudo-count" values; callers must set ``pseudo_count=True`` explicitly
to acknowledge that, and the returned values are recorded in a separate column so
they are never silently mixed with true-count results (plan §3 feasibility flag).
"""
from __future__ import annotations

import torch

from .base import check_shapes, masked_fill_like

_EPS = 1e-6


def poisson_nll(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, geometry=None, pseudo_count: bool = True, **cfg
) -> torch.Tensor:
    """Poisson negative log-likelihood, Stirling-offset so the optimum is 0.

    ``NLL = pred - target*log(pred) + (target*log(target) - target)``, using the
    Stirling approximation ``log(target!) ~= target*log(target) - target`` (with the
    ``0*log(0) = 0`` convention) as a constant offset, so that ``pred == target``
    gives exactly 0 per pixel: at ``p == t``, ``t - t*log(t) + t*log(t) - t == 0``.

    Args:
        pred, target: Non-negative count-like tensors, shape (B, H, W).
        mask: Boolean valid-pixel mask.
        pseudo_count: Must be passed explicitly; True means ``pred``/``target`` are
            *not* verified true counts (plan §3 feasibility flag). This does not
            change the computation, only documents/enforces caller intent.

    Returns:
        Per-image mean Poisson NLL over valid pixels, shape (B,).
    """
    check_shapes(pred, target, mask)
    pred = masked_fill_like(pred, target, mask)
    p = torch.clamp(pred, min=_EPS)
    t = torch.clamp(target, min=0.0)
    t_log_t = torch.where(t > 0, t * torch.log(torch.clamp(t, min=_EPS)), torch.zeros_like(t))
    nll = p - t * torch.log(p) + (t_log_t - t)
    mask_f = mask.to(dtype=pred.dtype)
    denom = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
    return (nll * mask_f).flatten(1).sum(dim=1) / denom


def poisson_chi2(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, geometry=None, pseudo_count: bool = True, **cfg
) -> torch.Tensor:
    """Poisson chi-squared: ``sum((pred - target)^2 / max(pred, 1))`` (plan §3, metric #3).

    See :func:`poisson_nll` for the ``pseudo_count`` caveat.
    """
    check_shapes(pred, target, mask)
    pred = masked_fill_like(pred, target, mask)
    var = torch.clamp(pred, min=1.0)
    chi2 = (pred - target) ** 2 / var
    mask_f = mask.to(dtype=pred.dtype)
    denom = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
    return (chi2 * mask_f).flatten(1).sum(dim=1) / denom
