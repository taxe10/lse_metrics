"""The current export/training transform: log + percentile clip.

The exact parameters used to produce the training PNG/JPEG exports are an open
question (plan §9 Q3: log base/eps, percentile values, per-image vs. dataset-wide,
clip-before-or-after-log). Everything here is parameterized so the config can be
updated once that's confirmed, and ``export_transform``/``inverse_transform`` round-trip
to within one grey level by construction (tested in ``tests/test_preprocess.py``).

Two spaces are used throughout the study (plan §1.1):
  - "model space": ``x in [0, 1]``, exactly what the VAE sees and reconstructs.
  - "pseudo-linear space": an approximate inverse of the export, used only to apply
    degradations the way a physical corruption would (plan §1.1, §2). It is a true
    inverse only for a raw-frame subsample where the real export parameters are known;
    otherwise it is flagged as pseudo-linear/pseudo-count in metrics that require it
    (Poisson NLL/chi2, "linear" MSE).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExportParams:
    """Parameters of one image's log + percentile-clip export.

    Attributes:
        log_eps: Additive offset before taking the log, so log(0) is finite.
        clip_low: Lower clip bound, in log space, before rescaling to [0, 1].
        clip_high: Upper clip bound, in log space, before rescaling to [0, 1].
    """

    log_eps: float
    clip_low: float
    clip_high: float

    @property
    def log_range(self) -> float:
        return self.clip_high - self.clip_low


def masked_percentile(values: np.ndarray, mask: np.ndarray, pct: float) -> float:
    """Percentile of ``values`` restricted to ``mask``-valid pixels (plan §1.1)."""
    valid = values[mask]
    if valid.size == 0:
        raise ValueError("masked_percentile: mask has no valid pixels")
    return float(np.percentile(valid, pct))


def export_transform(
    raw: np.ndarray,
    mask: np.ndarray,
    log_eps: float = 1.0,
    clip_low_pct: float = 0.5,
    clip_high_pct: float = 99.5,
    apply_upper_clip: bool = True,
) -> tuple[np.ndarray, ExportParams]:
    """Apply log transform + percentile clip + rescale to [0, 1] (model space).

    Percentiles are computed on valid (masked-in) pixels only, matching plan §1.1's
    rule that clip parameters come from the target, never from the prediction, so
    global-scale and offset degradations aren't silently normalized away.

    Args:
        raw: Native or downsampled intensity, in pseudo-linear (or true linear) space.
        mask: Boolean valid-pixel mask, same shape as ``raw``.
        log_eps: Offset added before the log.
        clip_low_pct: Lower percentile (of valid pixels) for the clip.
        clip_high_pct: Upper percentile (of valid pixels) for the clip.
        apply_upper_clip: If False, skip the upper-percentile clip (plan §1.1's
            clip ablation, to test whether the export itself hides peak signal).

    Returns:
        (x, params) where ``x`` is in [0, 1] (invalid pixels are set to 0), and
        ``params`` records the per-image clip bounds needed to invert or to transform
        a prediction with the *target's* parameters.
    """
    log_img = np.log(np.clip(raw, 0, None) + log_eps)
    clip_low = masked_percentile(log_img, mask, clip_low_pct)
    clip_high = masked_percentile(log_img, mask, clip_high_pct) if apply_upper_clip else float(log_img[mask].max())
    if clip_high <= clip_low:
        clip_high = clip_low + 1e-6
    params = ExportParams(log_eps=log_eps, clip_low=clip_low, clip_high=clip_high)
    x = np.clip((log_img - clip_low) / params.log_range, 0.0, 1.0)
    x = np.where(mask, x, 0.0).astype(np.float32)
    return x, params


def apply_export_params(raw: np.ndarray, mask: np.ndarray, params: ExportParams) -> np.ndarray:
    """Transform ``raw`` to model space using someone else's (the target's) ``params``.

    Used to transform a prediction with the target's clip bounds (plan §1.1), so a
    prediction that is globally brighter or dimmer than the target is not silently
    renormalized away before the metric ever sees it.
    """
    log_img = np.log(np.clip(raw, 0, None) + params.log_eps)
    x = np.clip((log_img - params.clip_low) / params.log_range, 0.0, 1.0)
    return np.where(mask, x, 0.0).astype(np.float32)


def inverse_transform(x: np.ndarray, params: ExportParams) -> np.ndarray:
    """Approximate inverse of :func:`export_transform`: model space -> pseudo-linear space.

    This is exact only where ``x`` was not clipped (i.e. the true value was within
    [clip_low, clip_high]); values that were clipped cannot be recovered, which is the
    physical basis for flagging Poisson-family metrics as "pseudo-count" unless a raw
    frame is available (plan §1.1, §3 metric #3).
    """
    log_img = x * params.log_range + params.clip_low
    return np.exp(log_img) - params.log_eps
