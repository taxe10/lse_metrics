import numpy as np
import pytest

from metric_study.masks import (
    PILATUS2M_SHAPE,
    combine_masks,
    estimate_data_driven_mask,
    static_gap_mask,
)


def test_static_gap_mask_shape():
    mask = static_gap_mask(PILATUS2M_SHAPE, dilation_px=0)
    assert mask.shape == PILATUS2M_SHAPE
    assert mask.dtype == bool


def test_static_gap_mask_matches_observed_gap_fraction():
    # Sample-data analysis found ~9.4% dark area (plan §0.2); the undilated module
    # geometry alone accounts for 1 - (8*195*3*487)/(1679*1475).
    mask = static_gap_mask(PILATUS2M_SHAPE, dilation_px=0)
    invalid_frac = 1.0 - mask.mean()
    assert 0.07 < invalid_frac < 0.10


def test_dilation_grows_invalid_region():
    m0 = static_gap_mask(PILATUS2M_SHAPE, dilation_px=0)
    m2 = static_gap_mask(PILATUS2M_SHAPE, dilation_px=2)
    assert m2.sum() < m0.sum()
    # Every pixel valid under dilation=2 must also be valid under dilation=0.
    assert np.all(m0[m2])


def test_unexpected_shape_raises():
    with pytest.raises(ValueError):
        static_gap_mask((100, 100))


def test_data_driven_mask_flags_always_dark_and_hot_pixels():
    stack = np.full((20, 32, 32), 50.0, dtype=np.float32)
    stack[:, 5, 5] = 0.0  # always dark -> beamstop/dead pixel
    stack[:, 10, 10] = 255.0  # always saturated -> hot pixel
    result = estimate_data_driven_mask(stack, dark_value=1.0, saturation_value=255.0)
    assert result.valid[5, 5] == False  # noqa: E712
    assert result.valid[10, 10] == False  # noqa: E712
    assert result.valid[0, 0] == True  # noqa: E712
    assert result.n_frames == 20


def test_combine_masks_is_logical_and():
    a = np.array([[True, True], [False, True]])
    b = np.array([[True, False], [False, True]])
    out = combine_masks(a, b)
    assert np.array_equal(out, np.array([[True, False], [False, True]]))


def test_combine_masks_requires_matching_shape():
    a = np.ones((2, 2), dtype=bool)
    b = np.ones((3, 3), dtype=bool)
    with pytest.raises(ValueError):
        combine_masks(a, b)
