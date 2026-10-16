import numpy as np
from PIL import Image

from metric_study.config import StudyConfig
from metric_study.manifest import build_manifest
from metric_study.pipeline import make_chunks, process_chunk

_TEST_YAML = """
test_pipeline: &test_pipeline
  seed: 0
  resolutions: [native, "512"]
  data_dir: "{data_dir}"
  output_dir: "{output_dir}"
  chunk_size: 2
  downsample: {{interpolation: INTER_AREA, tau: 0.9}}
  preprocess: {{log_eps: 1.0, clip_low_pct: 0.5, clip_high_pct: 99.5, apply_upper_clip: true}}
  masks: {{gap_dilation_px: 2}}
  degradations:
    fine_ring_attenuation: [0.0, 1.0]
    broad_ring_damage: [0.0, 1.0]
    peak_removal: [1.0]
    gaussian_blur: [2.0]
    global_scaling: [1.5]
    additive_offset: [0.05]
    poisson_resampling: [100.0]
  metrics:
    names: [mse_linear, mse_model_unmasked, log_mse, neg_pearson, poisson_nll, peak_weighted_mse]
    peak_weight_beta: 10.0
    params: {{poisson_nll: {{pseudo_count: true}}}}
"""


def _write_synthetic_frame(path, native_shape):
    h, w = native_shape
    cy, cx = h * 0.65, w * 0.5
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    image = 20.0 + 150.0 * np.exp(-((r - 300) ** 2) / (2 * 6.0**2)) + 30.0 * np.exp(-((r - 500) ** 2) / (2 * 1.5**2))
    image += np.random.default_rng(0).normal(0, 0.5, size=image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def test_process_chunk_end_to_end(tmp_path, native_shape):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_synthetic_frame(data_dir / "img1.jpg", native_shape)

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(_TEST_YAML.format(data_dir=data_dir, output_dir=output_dir))
    cfg = StudyConfig.load(yaml_path, "test_pipeline")

    manifest = build_manifest(data_dir)
    chunks = make_chunks(manifest, cfg.chunk_size)
    assert len(chunks) == 1

    result = process_chunk(chunks[0], cfg, output_dir, chunk_id=0)
    assert result.n_rows > 0
    assert result.n_skipped == 0
    assert result.metrics_path.exists()

    import pyarrow.parquet as pq

    df = pq.read_table(result.metrics_path).to_pandas()
    assert set(df["resolution"]) == {"native", "512"}
    assert "clean" in set(df["degradation"])
    assert (df["config_hash"] == cfg.config_hash).all()


def test_process_chunk_is_idempotent_and_resumable(tmp_path, native_shape):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_synthetic_frame(data_dir / "img1.jpg", native_shape)

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(_TEST_YAML.format(data_dir=data_dir, output_dir=output_dir))
    cfg = StudyConfig.load(yaml_path, "test_pipeline")

    manifest = build_manifest(data_dir)
    chunks = make_chunks(manifest, cfg.chunk_size)

    first = process_chunk(chunks[0], cfg, output_dir, chunk_id=0)
    mtime_1 = first.metrics_path.stat().st_mtime

    second = process_chunk(chunks[0], cfg, output_dir, chunk_id=0)
    mtime_2 = second.metrics_path.stat().st_mtime
    assert mtime_1 == mtime_2  # resumed: file was not rewritten


def test_process_chunk_skips_corrupt_files(tmp_path, native_shape):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    data_dir.mkdir()
    _write_synthetic_frame(data_dir / "good.jpg", native_shape)
    (data_dir / "bad.jpg").write_bytes(b"not an image")

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(_TEST_YAML.format(data_dir=data_dir, output_dir=output_dir))
    cfg = StudyConfig.load(yaml_path, "test_pipeline")

    manifest = build_manifest(data_dir)
    # `make_chunks` already filters out rows the manifest itself flagged unreadable
    # (plan §6.5's "cheap pass" reuses that), so process_chunk's own per-row
    # skip-and-log path (plan §6.3, defense in depth for a manifest passed through
    # directly rather than via make_chunks) is exercised here by chunking the raw,
    # unfiltered manifest instead.
    result = process_chunk(manifest, cfg, output_dir, chunk_id=0)
    assert result.n_skipped == 1
    assert result.skipped_path.exists()
    assert result.n_rows > 0  # the good image still produced rows
