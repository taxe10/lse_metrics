import numpy as np

from metric_study.preprocess import apply_export_params, export_transform, inverse_transform, masked_percentile


def test_masked_percentile_ignores_invalid_pixels():
    values = np.array([[0.0, 100.0], [1.0, 2.0]])
    mask = np.array([[False, True], [True, True]])
    # Excluding the 0 outlier, the median of [100, 1, 2] is 2.0.
    assert masked_percentile(values, mask, 50) == 2.0


def test_export_transform_round_trip_within_one_grey_level():
    rng = np.random.default_rng(0)
    raw = rng.uniform(1, 5000, size=(64, 64)).astype(np.float32)
    mask = np.ones((64, 64), dtype=bool)
    x, params = export_transform(raw, mask, clip_low_pct=0.0, clip_high_pct=100.0)
    recovered = inverse_transform(x, params)
    # With the full 0-100 percentile range, nothing is clipped, so the round trip
    # should recover raw values to within float32 precision.
    rel_err = np.abs(recovered - raw) / np.clip(raw, 1e-3, None)
    assert np.median(rel_err) < 1e-4


def test_export_transform_clips_to_unit_range():
    rng = np.random.default_rng(1)
    raw = rng.exponential(scale=50, size=(64, 64)).astype(np.float32)
    mask = np.ones((64, 64), dtype=bool)
    x, _ = export_transform(raw, mask)
    assert x.min() >= 0.0
    assert x.max() <= 1.0


def test_apply_export_params_uses_targets_clip_not_its_own():
    """A prediction that is globally brighter than the target should NOT be
    renormalized back to [0,1] using its own percentiles (plan §1.1 design rule)."""
    rng = np.random.default_rng(2)
    target_raw = rng.uniform(1, 100, size=(32, 32)).astype(np.float32)
    mask = np.ones((32, 32), dtype=bool)
    _, target_params = export_transform(target_raw, mask)

    pred_raw = target_raw * 3.0  # globally brighter prediction
    pred_x = apply_export_params(pred_raw, mask, target_params)
    target_x = apply_export_params(target_raw, mask, target_params)
    # Using the target's own params, a uniformly brighter pred must map to
    # systematically higher values than the target (not silently normalized away).
    assert pred_x[mask].mean() > target_x[mask].mean()


def test_upper_clip_ablation_changes_saturation():
    rng = np.random.default_rng(3)
    raw = rng.exponential(scale=20, size=(64, 64)).astype(np.float32)
    mask = np.ones((64, 64), dtype=bool)
    x_clipped, _ = export_transform(raw, mask, apply_upper_clip=True, clip_high_pct=50.0)
    x_unclipped, _ = export_transform(raw, mask, apply_upper_clip=False)
    # With a much stricter clip, more pixels should saturate at 1.0.
    assert (x_clipped >= 0.999).sum() > (x_unclipped >= 0.999).sum()
