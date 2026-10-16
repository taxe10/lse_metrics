import numpy as np

from metric_study.features import classify_pattern, detect_ridges, detect_spots, estimate_background, extract_features


def test_estimate_background_removes_narrow_structure(synthetic_rings_image, native_mask):
    background = estimate_background(synthetic_rings_image, native_mask, opening_size=25)
    # The background should be much flatter (lower variance) than the original,
    # since the fine ring and spots are removed by the opening.
    assert background[native_mask].std() < synthetic_rings_image[native_mask].std()


def test_detect_ridges_finds_fine_and_broad(synthetic_rings_image, native_mask):
    background = estimate_background(synthetic_rings_image, native_mask)
    ridges = detect_ridges(synthetic_rings_image, native_mask, background)
    assert any(r.is_fine for r in ridges), "should detect the synthetic fine ring"
    assert any(r.is_broad for r in ridges), "should detect the synthetic broad ring"


def test_detect_spots_finds_bragg_like_peaks(synthetic_rings_image, native_mask):
    background = estimate_background(synthetic_rings_image, native_mask)
    spots = detect_spots(synthetic_rings_image, native_mask, background)
    assert len(spots) >= 1


def test_classify_pattern_low_signal_for_flat_image(native_mask, native_shape):
    flat = np.full(native_shape, 10.0, dtype=np.float32) + np.random.default_rng(0).normal(
        0, 0.01, size=native_shape
    ).astype(np.float32)
    features = extract_features(flat, native_mask)
    assert classify_pattern(features) == "low_signal"


def test_classify_pattern_on_synthetic_rings(synthetic_rings_image, native_mask):
    features = extract_features(synthetic_rings_image, native_mask)
    label = classify_pattern(features)
    assert label in {"broad", "fine", "peaks"}  # has real structure, not low_signal
