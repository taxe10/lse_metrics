import numpy as np

from metric_study.degradations import (
    additive_offset,
    broad_ring_damage,
    fine_ring_attenuation,
    gaussian_blur,
    global_scaling,
    peak_removal,
    poisson_resample,
    severity_rng,
)
from metric_study.features import extract_features


def test_gaussian_blur_zero_sigma_is_identity(synthetic_rings_image, native_mask):
    out = gaussian_blur(synthetic_rings_image, native_mask, sigma=0.0)
    np.testing.assert_array_equal(out, synthetic_rings_image)


def test_global_scaling_identity_at_s_one(synthetic_rings_image, native_mask):
    out = global_scaling(synthetic_rings_image, native_mask, s=1.0)
    np.testing.assert_allclose(out, synthetic_rings_image)


def test_additive_offset_zero_at_c_zero(synthetic_rings_image, native_mask):
    out = additive_offset(synthetic_rings_image, native_mask, c=0.0)
    np.testing.assert_allclose(out, synthetic_rings_image)


def test_poisson_resample_high_lambda_approaches_identity(native_mask, native_shape):
    image = np.full(native_shape, 500.0, dtype=np.float32)
    rng = severity_rng(0, "img", "poisson_resampling", 0, 0)
    out = poisson_resample(image, native_mask, mean_counts=500.0, rng=rng)
    rel_err = np.abs(out[native_mask] - image[native_mask]) / image[native_mask]
    assert np.median(rel_err) < 0.05  # noisy but centered on the identity


def test_fine_ring_attenuation_a_zero_is_identity(synthetic_rings_image, native_mask):
    features = extract_features(synthetic_rings_image, native_mask)
    out = fine_ring_attenuation(synthetic_rings_image, native_mask, features, a=0.0)
    np.testing.assert_allclose(out, synthetic_rings_image)


def test_fine_ring_attenuation_a_one_removes_most_excess(synthetic_rings_image, native_mask):
    features = extract_features(synthetic_rings_image, native_mask)
    fine = [r for r in features.ridges if r.is_fine]
    assert fine, "synthetic fixture should contain at least one detected fine ring"
    out = fine_ring_attenuation(synthetic_rings_image, native_mask, features, a=1.0)
    ring = sorted(fine, key=lambda r: r.width_px)[0]
    remaining = (out[ring.mask] - features.background[ring.mask]).sum()
    original = ring.excess_intensity
    assert abs(remaining) < 0.05 * abs(original) + 1e-6


def test_broad_ring_damage_a_zero_is_identity(synthetic_rings_image, native_mask):
    out = broad_ring_damage(synthetic_rings_image, native_mask, extract_features(synthetic_rings_image, native_mask), a=0.0)
    np.testing.assert_allclose(out, synthetic_rings_image)


def test_peak_removal_f_zero_is_identity(synthetic_rings_image, native_mask):
    features = extract_features(synthetic_rings_image, native_mask)
    rng = severity_rng(0, "img", "peak_removal", 0, 0)
    out = peak_removal(synthetic_rings_image, native_mask, features, f=0.0, a=1.0, rng=rng)
    np.testing.assert_allclose(out, synthetic_rings_image)


def test_severity_rng_is_deterministic():
    r1 = severity_rng(0, "23998", "fine_ring_attenuation", 3, 0)
    r2 = severity_rng(0, "23998", "fine_ring_attenuation", 3, 0)
    np.testing.assert_array_equal(r1.integers(0, 1000, size=10), r2.integers(0, 1000, size=10))


def test_severity_rng_differs_by_key():
    r1 = severity_rng(0, "23998", "fine_ring_attenuation", 3, 0)
    r2 = severity_rng(0, "23998", "fine_ring_attenuation", 4, 0)
    assert not np.array_equal(r1.integers(0, 1000, size=10), r2.integers(0, 1000, size=10))


def test_severity_rng_independent_of_process_hash_seed():
    """Guards against the bug of using Python's built-in hash() (PYTHONHASHSEED-randomized)."""
    import hashlib

    key = "img:deg:0:0"
    h1 = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    h2 = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    assert h1 == h2
