"""Command-line entry point: ``python -m metric_study.cli <subcommand> ...`` (plan §5.2).

Subcommands implemented so far: ``manifest``, ``run``, ``status``, ``dry-run``.
``masks``, ``features``, ``reconstruct``, ``benchmark``, ``analyze`` and ``atlas``
are listed in the plan's layout (§5.2) as separate subcommands; ``features`` and
``masks`` are folded into ``run`` for now (they're cheap per-image steps computed
inline in ``pipeline.process_one_image``/``process_chunk``), and
``reconstruct``/``benchmark``/``analyze``/``atlas`` are not yet wired up pending a
checkpoint and a production run to benchmark and analyze (plan §9 Q1).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import StudyConfig
from .manifest import build_manifest, read_manifest, write_manifest
from .pipeline import make_chunks, process_chunk

logger = logging.getLogger(__name__)


def _setup_logging(out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / f"{name}.log")]
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s|%(levelname)s] %(message)s", handlers=handlers)


def cmd_manifest(args: argparse.Namespace) -> None:
    _setup_logging(Path(args.output_dir), "manifest")
    df = build_manifest(args.data_dir)
    write_manifest(df, Path(args.output_dir) / "manifest.parquet")


def cmd_run(args: argparse.Namespace) -> None:
    _setup_logging(Path(args.output_dir), "run")
    cfg = StudyConfig.load(args.config, args.section)
    cfg.log()
    manifest_path = Path(args.output_dir) / "manifest.parquet"
    if not manifest_path.exists():
        logger.info("No manifest found at %s; building one from %s", manifest_path, cfg.data_dir)
        df = build_manifest(cfg.data_dir)
        write_manifest(df, manifest_path)
    else:
        df = read_manifest(manifest_path)

    if args.limit:
        eligible = df[(~df["is_duplicate"]) & (df["read_status"] == "ok")]
        keep_ids = set(eligible["image_id"].iloc[: args.limit])
        df = df[df["image_id"].isin(keep_ids) | df["is_duplicate"]]

    chunks = make_chunks(df, cfg.chunk_size)
    logger.info("Processing %d chunks (chunk_size=%d)", len(chunks), cfg.chunk_size)
    for chunk_id, chunk in enumerate(chunks):
        process_chunk(chunk, cfg, args.output_dir, chunk_id)


def cmd_status(args: argparse.Namespace) -> None:
    metrics_dir = Path(args.output_dir) / "metrics"
    skipped_dir = Path(args.output_dir) / "skipped"
    done = sorted(metrics_dir.glob("chunk=*.done")) if metrics_dir.exists() else []
    parquet = sorted(metrics_dir.glob("chunk=*.parquet")) if metrics_dir.exists() else []
    skipped = sorted(skipped_dir.glob("chunk=*.parquet")) if skipped_dir.exists() else []
    print(f"Done chunks:    {len(done)}")
    print(f"Parquet chunks: {len(parquet)}")
    print(f"Chunks with skips: {len(skipped)}")


def cmd_dry_run(args: argparse.Namespace) -> None:
    """Run the exact production path on a small dataset with tiny chunks (plan §6.7)."""
    args.section = args.section or "dry_run"
    _setup_logging(Path(args.output_dir), "dry_run")
    logger.info("=== DRY RUN: %s -> %s ===", args.data_dir, args.output_dir)
    cfg = StudyConfig.load(args.config, args.section)
    manifest_path = Path(args.output_dir) / "manifest.parquet"
    df = build_manifest(cfg.data_dir if not args.data_dir else args.data_dir)
    write_manifest(df, manifest_path)

    n_dupe = int(df["is_duplicate"].sum())
    n_unique = len(df) - n_dupe
    n_bad = int((df["read_status"] != "ok").sum())
    logger.info("Dry run manifest: %d files, %d unique, %d duplicates, %d unreadable", len(df), n_unique, n_dupe, n_bad)

    if args.limit:
        eligible = df[(~df["is_duplicate"]) & (df["read_status"] == "ok")]
        keep_ids = set(eligible["image_id"].iloc[: args.limit])
        df = df[df["image_id"].isin(keep_ids) | df["is_duplicate"]]
        logger.info("--limit %d: restricting dry run to %d unique image(s)", args.limit, len(keep_ids))

    chunks = make_chunks(df, cfg.chunk_size)
    logger.info("Dry run: %d chunks of up to %d images each", len(chunks), cfg.chunk_size)
    results = [process_chunk(chunk, cfg, args.output_dir, i) for i, chunk in enumerate(chunks)]

    # Resume check: re-run chunk 0 and confirm it is skipped via the .done marker.
    if results:
        before = results[0]
        again = process_chunk(chunks[0], cfg, args.output_dir, 0)
        resumed_ok = again.metrics_path == before.metrics_path
        logger.info("Resume check (re-processing chunk 0): %s", "OK (skipped, .done honored)" if resumed_ok else "FAILED")

    total_rows = sum(r.n_rows for r in results)
    total_skipped = sum(r.n_skipped for r in results)
    logger.info("=== DRY RUN COMPLETE: %d metric rows, %d images skipped ===", total_rows, total_skipped)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="metric_study")
    sub = parser.add_subparsers(dest="command", required=True)

    p_manifest = sub.add_parser("manifest", help="Scan a directory and write manifest.parquet")
    p_manifest.add_argument("--data-dir", required=True)
    p_manifest.add_argument("--output-dir", required=True)
    p_manifest.set_defaults(func=cmd_manifest)

    p_run = sub.add_parser("run", help="Run the full pipeline over a manifest")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--section", default="default")
    p_run.add_argument("--output-dir", required=True)
    p_run.add_argument("--limit", type=int, default=None, help="Only process the first N unique images")
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="Report done/pending/skipped chunk counts")
    p_status.add_argument("--output-dir", required=True)
    p_status.set_defaults(func=cmd_status)

    p_dry = sub.add_parser("dry-run", help="Run the production path on a small dataset (plan §6.7)")
    p_dry.add_argument("--config", required=True)
    p_dry.add_argument("--section", default="dry_run")
    p_dry.add_argument("--data-dir", default=None, help="Overrides config's data_dir if given")
    p_dry.add_argument("--output-dir", required=True)
    p_dry.add_argument("--limit", type=int, default=None, help="Only process the first N unique images (for a quick smoke test)")
    p_dry.set_defaults(func=cmd_dry_run)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
