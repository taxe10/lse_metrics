import numpy as np
import pytest
import torch

from metric_study.metrics import METRICS
from metric_study.metrics.pixel import neg_pearson
from metric_study.metrics.weighted import build_weight_map, peak_weighted_mse

torch.manual_seed(0)


def _batch(target_np, mask_np, pred_np=None):
    target = torch.from_numpy(target_np).unsqueeze(0).float()
    mask = torch.from_numpy(mask_np).unsqueeze(0)
    pred = target.clone() if pred_np is None else torch.from_numpy(pred_np).unsqueeze(0).float()
    return pred, target, mask


OPTIMAL_VALUE = {
    "mse_linear": 0.0,
    "mse_model_unmasked": 0.0,
    "log_mse": 0.0,
    "log_l1": 0.0,
    "asinh_mse": 0.0,
    "asinh_l1": 0.0,
    "poisson_nll": 0.0,
    "poisson_chi2": 0.0,
    "ssim": 1.0,
    "ms_ssim": 1.0,
    "neg_pearson": -1.0,
    "focal_frequency_loss": 0.0,
}


# ms_ssim needs min(H, W) > (win_size-1)*2**(n_scales-1); at win_size=7 and 5 scales
# that's 96px, so it cannot run on the 64x64 small_synthetic_pair fixture at all
# (this is the documented plan §3 size constraint, not a bug) and gets its own
# fixture/tests below (test_ms_ssim_*, using large_synthetic_pair).
SMALL_IMAGE_METRICS = [n for n in OPTIMAL_VALUE if n != "ms_ssim"]

# mse_model_unmasked is ALSO excluded from the masked-pixel-invariance test below,
# but for a different reason (plan §3): it is the literal current (unmasked)
# training loss (`F.mse_loss(recon, x, reduction="mean")`, training_2.py:203), so it
# is *supposed* to be sensitive to what happens on masked pixels -- that sensitivity
# is exactly what plan §4.6/§8 uses it to quantify. It has its own dedicated test below.
MASK_INVARIANT_METRICS = [n for n in SMALL_IMAGE_METRICS if n != "mse_model_unmasked"]


@pytest.mark.parametrize("name", SMALL_IMAGE_METRICS)
def test_identical_inputs_give_optimum(name, small_synthetic_pair):
    target_np, mask_np = small_synthetic_pair
    pred, target, mask = _batch(target_np, mask_np)
    fn = METRICS[name]
    kwargs = {"win_size": 7} if name == "ssim" else {}
    value = fn(pred, target, mask, **kwargs)
    assert value.shape == (1,)
    np.testing.assert_allclose(value.item(), OPTIMAL_VALUE[name], atol=1e-4)


@pytest.mark.parametrize("name", MASK_INVARIANT_METRICS)
def test_masked_pixel_changes_have_no_effect(name, small_synthetic_pair):
    target_np, mask_np = small_synthetic_pair
    rng = np.random.default_rng(1)
    pred_np = target_np.copy()
    pred_np[~mask_np] = rng.uniform(0, 1, size=(~mask_np).sum()).astype(np.float32)  # perturb ONLY invalid pixels

    pred, target, mask = _batch(target_np, mask_np, pred_np)
    pred_unperturbed, _, _ = _batch(target_np, mask_np)

    fn = METRICS[name]
    kwargs = {"win_size": 7} if name == "ssim" else {}
    v_perturbed = fn(pred, target, mask, **kwargs)
    v_clean = fn(pred_unperturbed, target, mask, **kwargs)
    np.testing.assert_allclose(v_perturbed.item(), v_clean.item(), atol=1e-5)


def test_mse_model_unmasked_is_sensitive_to_masked_pixels(small_synthetic_pair):
    """The literal training loss is NOT mask-invariant by design (see comment above)."""
    target_np, mask_np = small_synthetic_pair
    rng = np.random.default_rng(1)
    pred_np = target_np.copy()
    pred_np[~mask_np] = rng.uniform(0, 1, size=(~mask_np).sum()).astype(np.float32)

    pred, target, mask = _batch(target_np, mask_np, pred_np)
    pred_unperturbed, _, _ = _batch(target_np, mask_np)
    fn = METRICS["mse_model_unmasked"]
    v_perturbed = fn(pred, target, mask)
    v_clean = fn(pred_unperturbed, target, mask)
    assert v_perturbed.item() != pytest.approx(v_clean.item())


@pytest.mark.parametrize("name", SMALL_IMAGE_METRICS)
def test_batched_equals_per_image(name, small_synthetic_pair):
    target_np, mask_np = small_synthetic_pair
    rng = np.random.default_rng(2)
    pred_np = target_np + rng.normal(0, 0.05, size=target_np.shape).astype(np.float32)

    pred = torch.from_numpy(np.stack([pred_np, pred_np])).float()
    target = torch.from_numpy(np.stack([target_np, target_np])).float()
    mask = torch.from_numpy(np.stack([mask_np, mask_np]))

    fn = METRICS[name]
    kwargs = {"win_size": 7} if name == "ssim" else {}
    batched = fn(pred, target, mask, **kwargs)
    single = fn(pred[:1], target[:1], mask[:1], **kwargs)
    assert batched.shape == (2,)
    np.testing.assert_allclose(batched[0].item(), single[0].item(), atol=1e-5)
    np.testing.assert_allclose(batched[1].item(), single[0].item(), atol=1e-5)


@pytest.fixture(scope="session")
def large_synthetic_pair():
    """A (192, 192) pair, large enough for ms_ssim's 5-scale win_size=7 requirement (>96px)."""
    rng = np.random.default_rng(3)
    target = rng.uniform(0, 1, size=(192, 192)).astype(np.float32)
    mask = np.ones((192, 192), dtype=bool)
    mask[:8, :] = False
    return target, mask


def test_ms_ssim_identical_inputs_give_optimum(large_synthetic_pair):
    from metric_study.metrics.structural import ms_ssim

    target_np, mask_np = large_synthetic_pair
    pred, target, mask = _batch(target_np, mask_np)
    value = ms_ssim(pred, target, mask, win_size=7)
    np.testing.assert_allclose(value.item(), 1.0, atol=1e-4)


def test_ms_ssim_masked_pixel_changes_have_no_effect(large_synthetic_pair):
    from metric_study.metrics.structural import ms_ssim

    target_np, mask_np = large_synthetic_pair
    rng = np.random.default_rng(4)
    pred_np = target_np.copy()
    pred_np[~mask_np] = rng.uniform(0, 1, size=(~mask_np).sum()).astype(np.float32)
    pred, target, mask = _batch(target_np, mask_np, pred_np)
    pred_unperturbed, _, _ = _batch(target_np, mask_np)
    v_perturbed = ms_ssim(pred, target, mask, win_size=7)
    v_clean = ms_ssim(pred_unperturbed, target, mask, win_size=7)
    np.testing.assert_allclose(v_perturbed.item(), v_clean.item(), atol=1e-5)


def test_ms_ssim_batched_equals_per_image(large_synthetic_pair):
    from metric_study.metrics.structural import ms_ssim

    target_np, mask_np = large_synthetic_pair
    rng = np.random.default_rng(5)
    pred_np = target_np + rng.normal(0, 0.05, size=target_np.shape).astype(np.float32)
    pred = torch.from_numpy(np.stack([pred_np, pred_np])).float()
    target = torch.from_numpy(np.stack([target_np, target_np])).float()
    mask = torch.from_numpy(np.stack([mask_np, mask_np]))
    batched = ms_ssim(pred, target, mask, win_size=7)
    single = ms_ssim(pred[:1], target[:1], mask[:1], win_size=7)
    np.testing.assert_allclose(batched[0].item(), single[0].item(), atol=1e-5)


def test_neg_pearson_invariant_to_scale_and_offset(small_synthetic_pair):
    target_np, mask_np = small_synthetic_pair
    pred_np = target_np * 3.0 + 0.5
    pred, target, mask = _batch(target_np, mask_np, pred_np)
    value = neg_pearson(pred, target, mask)
    np.testing.assert_allclose(value.item(), -1.0, atol=1e-4)


def test_peak_weighted_mse_requires_weight_map(small_synthetic_pair):
    target_np, mask_np = small_synthetic_pair
    pred, target, mask = _batch(target_np, mask_np)
    with pytest.raises(ValueError):
        peak_weighted_mse(pred, target, mask)


def test_peak_weighted_mse_upweights_feature_pixels(small_synthetic_pair):
    target_np, mask_np = small_synthetic_pair
    pred_np = target_np.copy()
    pred_np[10, 10] += 0.5  # a single-pixel error
    pred, target, mask = _batch(target_np, mask_np, pred_np)

    weight_flat = np.ones_like(target_np)
    weight_flat_t = torch.from_numpy(weight_flat).unsqueeze(0).float()
    weight_focused = np.ones_like(target_np)
    weight_focused[10, 10] = 100.0
    weight_focused_t = torch.from_numpy(weight_focused).unsqueeze(0).float()

    v_flat = peak_weighted_mse(pred, target, mask, weight_map=weight_flat_t)
    v_focused = peak_weighted_mse(pred, target, mask, weight_map=weight_focused_t)
    assert v_focused.item() > v_flat.item()


def test_ms_ssim_raises_on_too_small_image():
    from metric_study.metrics.structural import ms_ssim

    small = torch.rand(1, 32, 32)
    mask = torch.ones(1, 32, 32, dtype=torch.bool)
    with pytest.raises(ValueError):
        ms_ssim(small, small, mask)


def test_build_weight_map_matches_component_pixels():
    from metric_study.features import RidgeComponent

    shape = (10, 10)
    ring_mask = np.zeros(shape, dtype=bool)
    ring_mask[2:4, 2:4] = True
    ridge = RidgeComponent(mask=ring_mask, width_px=3.0, excess_intensity=10.0, is_fine=True)
    weights = build_weight_map(shape, [ridge], [], beta=9.0)
    assert weights[3, 3] == 10.0
    assert weights[0, 0] == 1.0
