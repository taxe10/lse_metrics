import numpy as np

from metric_study.downsample import downsample_to_512, valid_pixel_sum_ratio


def test_constant_image_stays_constant(native_mask, native_shape):
    image = np.full(native_shape, 7.0, dtype=np.float32)
    image_512, mask_512 = downsample_to_512(image, native_mask)
    assert mask_512.any()
    np.testing.assert_allclose(image_512[mask_512], 7.0, atol=1e-3)


def test_coverage_and_mask_are_bounded(native_mask, native_shape):
    image = np.random.default_rng(0).uniform(0, 1, size=native_shape).astype(np.float32)
    image_512, mask_512 = downsample_to_512(image, native_mask)
    assert image_512.shape == (512, 512)
    assert mask_512.shape == (512, 512)
    assert mask_512.dtype == bool
    # Most of the 512 grid should be valid, since gaps are a small fraction of area.
    assert 0.7 < mask_512.mean() < 1.0


def test_fully_masked_region_stays_masked():
    shape = (64, 64)
    mask = np.ones(shape, dtype=bool)
    mask[:32, :] = False
    image = np.ones(shape, dtype=np.float32)
    image_512, mask_512 = downsample_to_512(image, mask, target_size=(16, 16), tau=0.9)
    # Native rows 0-31 are invalid; each output row i covers native rows [4i, 4i+4),
    # so output rows 0-7 (native rows 0-31) should be fully invalid.
    assert not mask_512[:8, :].any()
    assert mask_512[8:, :].all()


def test_valid_pixel_mean_is_preserved(native_mask, native_shape):
    image = np.random.default_rng(1).uniform(0, 100, size=native_shape).astype(np.float32)
    image_512, mask_512 = downsample_to_512(image, native_mask)
    ratio = valid_pixel_sum_ratio(image, native_mask, image_512, mask_512)
    assert 0.95 < ratio < 1.05
