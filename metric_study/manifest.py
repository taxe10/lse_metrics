"""Build a manifest of every input image, with content-hash deduplication.

The sample data showed 56 of 100 files are byte-identical duplicates (plan §0.2,
Q4). Deduplication happens here, once, so every later stage (features, degradations,
metrics) only ever sees ``is_duplicate == False`` rows unless it explicitly asks
for duplicates.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .io import iter_image_paths, read_image

logger = logging.getLogger(__name__)


def _content_hash(path: Path) -> str:
    """MD5 of the raw file bytes. Cheap and sufficient for exact-duplicate detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(data_dir: str | Path) -> pd.DataFrame:
    """Scan ``data_dir`` and build a manifest with one row per image file.

    Args:
        data_dir: Root directory to scan recursively for image files.

    Returns:
        A DataFrame with columns: ``image_id``, ``path``, ``content_hash``,
        ``is_duplicate`` (True for every occurrence after the first of a given
        ``content_hash``), ``first_of_hash_id`` (the ``image_id`` of the kept
        original for duplicates), ``shape``, ``dtype``, ``bit_depth``, ``file_size``,
        ``mtime``, ``read_status``.
    """
    rows = []
    seen_hashes: dict[str, str] = {}
    for path in iter_image_paths(data_dir):
        image_id = path.stem
        content_hash = _content_hash(path)
        is_duplicate = content_hash in seen_hashes
        first_of_hash_id = seen_hashes.get(content_hash, image_id)
        if not is_duplicate:
            seen_hashes[content_hash] = image_id

        result = read_image(path)
        rows.append(
            {
                "image_id": image_id,
                "path": str(path),
                "content_hash": content_hash,
                "is_duplicate": is_duplicate,
                "first_of_hash_id": first_of_hash_id,
                "shape": str(result.meta.get("shape")),
                "dtype": result.meta.get("dtype"),
                "bit_depth": result.meta.get("bit_depth"),
                "file_size": result.meta.get("file_size"),
                "mtime": result.meta.get("mtime"),
                "read_status": result.status,
            }
        )
    if not rows:
        logger.warning("No image files found under %s", data_dir)
    df = pd.DataFrame(rows)
    n_dupe = int(df["is_duplicate"].sum()) if len(df) else 0
    n_bad = int((df["read_status"] != "ok").sum()) if len(df) else 0
    logger.info(
        "Manifest: %d files, %d unique by content hash, %d duplicates, %d unreadable",
        len(df),
        len(df) - n_dupe,
        n_dupe,
        n_bad,
    )
    return df


def write_manifest(df: pd.DataFrame, out_path: str | Path) -> None:
    """Write the manifest to Parquet, zstd-compressed (plan §6.4)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="zstd")
    logger.info("Wrote manifest with %d rows to %s", len(df), out_path)


def read_manifest(path: str | Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()
