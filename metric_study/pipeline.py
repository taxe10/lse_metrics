"""Chunked pipeline: read -> mask -> features -> degrade -> downsample -> metrics -> Parquet.

Implements the per-chunk contract from plan §6.3: atomic writes (``.tmp`` then
rename), a ``.done`` marker so a resumed run skips finished chunks, and a
``skipped/`` record for unreadable/corrupt files instead of crashing the job.

This first version processes one image at a time (not yet GPU-batched across
severities per plan §6.2) — it is deliberately written for correctness and to make
the dry run (plan §6.7) reproducible end to end; batching is a follow-up
optimization once ``cli.py benchmark`` shows where CPU/GPU time actually goes
(plan §7).
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .config import StudyConfig
from .degradations import REGISTRY, severity_rng
from .downsample import downsample_to_512
from .features import FeatureSet, classify_pattern, extract_features
from .io import read_image
from .masks import PILATUS2M_SHAPE, static_gap_mask
from .metrics import METRICS
from .metrics.weighted import build_weight_map
from .preprocess import export_transform

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """Summary of one chunk's processing, returned for logging/status (plan §6.3)."""

    chunk_id: int
    n_rows: int
    n_skipped: int
    metrics_path: Path | None
    skipped_path: Path | None


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(dtype=torch.float32)


def _severity_grid(cfg: StudyConfig, degradation: str) -> list[float]:
    grid = cfg["degradations"].get(degradation)
    if grid is None:
        raise KeyError(f"No severity grid configured for degradation '{degradation}'")
    return list(grid)


def _compute_metrics_for_pair(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    weight_map: np.ndarray,
    metric_names: list[str],
    metric_cfg: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Run every requested metric on one (pred, target, mask) pair, returning scalars."""
    p_t = _to_tensor(pred)
    t_t = _to_tensor(target)
    m_t = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
    w_t = _to_tensor(weight_map)

    out: dict[str, float] = {}
    for name in metric_names:
        fn = METRICS[name]
        kwargs = dict(metric_cfg.get(name, {}))
        try:
            if name == "peak_weighted_mse":
                value = fn(p_t, t_t, m_t, weight_map=w_t, **kwargs)
            else:
                value = fn(p_t, t_t, m_t, **kwargs)
            out[name] = float(value.item())
        except Exception as e:  # noqa: BLE001 - one bad metric must not kill the whole row
            logger.warning("Metric %s failed: %s", name, e)
            out[name] = float("nan")
    return out


def process_one_image(
    image_id: str,
    raw: np.ndarray,
    native_mask: np.ndarray,
    cfg: StudyConfig,
    features: FeatureSet | None = None,
) -> list[dict[str, Any]]:
    """Run the clean baseline plus every configured degradation on one image, both resolutions.

    Args:
        image_id: Manifest ``image_id``.
        raw: Native-resolution model-space image (float32).
        native_mask: Boolean valid-pixel mask, native resolution (static gap mask,
            optionally combined with a data-driven group mask).
        cfg: Study config (degradation grids, metric list, seed, tau).
        features: Precomputed :class:`FeatureSet`; computed from ``raw``/``native_mask``
            if not provided.

    Returns:
        A list of row-dicts, one per (resolution, degradation, severity, rep),
        ready to append into the chunk's output table.
    """
    if features is None:
        features = extract_features(raw, native_mask)
    pattern_class = classify_pattern(features)
    tau = cfg["downsample"].get("tau", 0.9)
    metric_names = list(cfg["metrics"].get("names", list(METRICS)))
    metric_cfg = cfg["metrics"].get("params", {})
    beta = cfg["metrics"].get("peak_weight_beta", 10.0)

    # Computed once per image, not per (degradation, severity, rep): the clean target
    # never changes across that loop, only `pred_native` does, so re-running feature
    # extraction (the expensive multi-scale ridge/spot detector) inside the emit
    # closure below would redo the same ~seconds-long computation ~100+ times per
    # image for no reason.
    target_512, mask_512 = downsample_to_512(raw, native_mask, tau=tau)
    features_512 = extract_features(target_512, mask_512)
    weight_map_native = build_weight_map(raw.shape, features.ridges, features.spots, beta)
    # The peak-weighted-MSE weight map is built from the *target's own* detected
    # features at each resolution (plan §3, metric #9), so the 512 map comes from
    # `features_512` (what actually survived downsampling), not from resizing the
    # native map, which would mismatch pixel-for-pixel against the 512 target.
    weight_map_512 = build_weight_map(target_512.shape, features_512.ridges, features_512.spots, beta)

    rows: list[dict[str, Any]] = []

    def _emit(pred_native: np.ndarray, degradation: str, severity: float, rep: int):
        pred_512, mask_512_pred = downsample_to_512(pred_native, native_mask, tau=tau)
        # Use the intersection with the clean target's own valid-512 mask: a pixel is
        # only scored at 512 if both the (possibly-shifted) prediction and the target
        # agree it's real detector area.
        mask_512_pair = mask_512 & mask_512_pred

        for resolution, pred, target, mask, weight_map in (
            ("native", pred_native, raw, native_mask, weight_map_native),
            ("512", pred_512, target_512, mask_512_pair, weight_map_512),
        ):
            metric_values = _compute_metrics_for_pair(pred, target, mask, weight_map, metric_names, metric_cfg)
            rows.append(
                {
                    "image_id": image_id,
                    "resolution": resolution,
                    "degradation": degradation,
                    "severity": severity,
                    "rep": rep,
                    "pattern_class": pattern_class,
                    **metric_values,
                }
            )

    _emit(raw, "clean", 0.0, 0)

    for degradation, spec in REGISTRY.items():
        severities = _severity_grid(cfg, degradation)
        for sev_idx, severity in enumerate(severities):
            n_reps = 3 if degradation == "poisson_resampling" else 1
            for rep in range(n_reps):
                rng = severity_rng(cfg.seed, image_id, degradation, sev_idx, rep)
                kwargs: dict[str, Any] = {}
                if spec.needs_features:
                    kwargs["features"] = features
                if spec.needs_rng:
                    kwargs["rng"] = rng
                if degradation == "peak_removal":
                    degraded = spec.fn(raw, native_mask, f=severity, a=1.0, **kwargs)
                elif degradation in ("fine_ring_attenuation", "broad_ring_damage"):
                    degraded = spec.fn(raw, native_mask, a=severity, **kwargs)
                elif degradation == "gaussian_blur":
                    degraded = spec.fn(raw, native_mask, sigma=severity)
                elif degradation == "global_scaling":
                    degraded = spec.fn(raw, native_mask, s=severity)
                elif degradation == "additive_offset":
                    degraded = spec.fn(raw, native_mask, c=severity)
                elif degradation == "poisson_resampling":
                    degraded = spec.fn(raw, native_mask, mean_counts=severity, **kwargs)
                else:
                    raise NotImplementedError(f"Unknown degradation dispatch for '{degradation}'")
                _emit(degraded, degradation, severity, rep)

    return rows


def process_chunk(
    manifest_chunk: pd.DataFrame,
    cfg: StudyConfig,
    out_dir: str | Path,
    chunk_id: int,
) -> ChunkResult:
    """Process one manifest chunk end to end, with idempotent, resumable output (plan §6.3).

    Args:
        manifest_chunk: Rows from ``manifest.parquet`` to process (already filtered
            to ``is_duplicate == False`` and ``read_status == "ok"`` by the caller).
        cfg: Study config.
        out_dir: Base output directory; writes to ``{out_dir}/metrics/chunk=<id>.parquet``
            and ``{out_dir}/skipped/chunk=<id>.parquet``.
        chunk_id: Integer chunk index, used in output filenames and the ``.done`` marker.

    Returns:
        A :class:`ChunkResult` summary.
    """
    out_dir = Path(out_dir)
    metrics_dir = out_dir / "metrics"
    skipped_dir = out_dir / "skipped"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    skipped_dir.mkdir(parents=True, exist_ok=True)

    done_marker = metrics_dir / f"chunk={chunk_id:06d}.done"
    final_path = metrics_dir / f"chunk={chunk_id:06d}.parquet"
    if done_marker.exists():
        logger.info("Chunk %d already done, skipping (resume).", chunk_id)
        return ChunkResult(chunk_id, len(manifest_chunk), 0, final_path, None)

    all_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for _, record in manifest_chunk.iterrows():
        image_id, path = record["image_id"], record["path"]
        try:
            result = read_image(path)
            if not result.ok:
                skipped_rows.append({"image_id": image_id, "path": path, "stage": "read", "error": result.status})
                continue
            shape = tuple(result.image.shape)
            if shape != PILATUS2M_SHAPE:
                skipped_rows.append(
                    {"image_id": image_id, "path": path, "stage": "mask", "error": f"unexpected_shape:{shape}"}
                )
                continue
            gap_mask = static_gap_mask(shape, dilation_px=cfg["masks"].get("gap_dilation_px", 2))
            # Model space is already [0, 255] (or [0, 65535] for 16-bit); normalize to [0, 1]
            # to match the training loader's ToTensor() output range (data_loader.py:61).
            max_val = 255.0 if result.meta.get("bit_depth", 8) == 8 else 65535.0
            raw = result.image / max_val
            rows = process_one_image(image_id, raw, gap_mask, cfg)
            all_rows.extend(rows)
        except Exception as e:  # noqa: BLE001 - one bad image must not kill the chunk
            logger.warning("Skipping %s: %s", image_id, e)
            skipped_rows.append({"image_id": image_id, "path": path, "stage": "process", "error": str(e)})

    df = pd.DataFrame(all_rows)
    df["config_hash"] = cfg.config_hash
    tmp_path = metrics_dir / f"chunk={chunk_id:06d}.parquet.tmp"
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp_path, compression="zstd")
    tmp_path.replace(final_path)  # atomic on POSIX (plan §6.3)

    skipped_path = None
    if skipped_rows:
        skipped_path = skipped_dir / f"chunk={chunk_id:06d}.parquet"
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(skipped_rows), preserve_index=False), skipped_path)

    file_hash = hashlib.sha256(final_path.read_bytes()).hexdigest()
    done_marker.write_text(
        json.dumps({"config_hash": cfg.config_hash, "n_rows": len(df), "sha256": file_hash}, indent=2)
    )
    logger.info(
        "Chunk %d done: %d rows written, %d images skipped.", chunk_id, len(df), len(skipped_rows)
    )
    return ChunkResult(chunk_id, len(df), len(skipped_rows), final_path, skipped_path)


def make_chunks(manifest: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    """Split a (deduplicated, readable) manifest into fixed-size chunks (plan §6.2)."""
    usable = manifest[(~manifest["is_duplicate"]) & (manifest["read_status"] == "ok")].reset_index(drop=True)
    return [usable.iloc[i : i + chunk_size] for i in range(0, len(usable), chunk_size)]
