"""Metric registry: every candidate metric behind the common signature (plan §3)

    metric(pred, target, mask, geometry=None, **cfg) -> Tensor  # shape (B,)

``METRICS`` maps the name used in ``metrics.parquet`` columns to the callable.
"""
from __future__ import annotations

from .freq import focal_frequency_loss
from .pixel import asinh_l1, asinh_mse, log_l1, log_mse, mse_linear, mse_model_unmasked, neg_pearson
from .poisson import poisson_chi2, poisson_nll
from .structural import ms_ssim, ssim
from .weighted import peak_weighted_mse

METRICS = {
    "mse_linear": mse_linear,
    "mse_model_unmasked": mse_model_unmasked,
    "log_mse": log_mse,
    "log_l1": log_l1,
    "asinh_mse": asinh_mse,
    "asinh_l1": asinh_l1,
    "poisson_nll": poisson_nll,
    "poisson_chi2": poisson_chi2,
    "ssim": ssim,
    "ms_ssim": ms_ssim,
    "neg_pearson": neg_pearson,
    "focal_frequency_loss": focal_frequency_loss,
    "peak_weighted_mse": peak_weighted_mse,
}

__all__ = ["METRICS"] + list(METRICS)
