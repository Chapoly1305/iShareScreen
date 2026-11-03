import torch
import torch.nn.functional as F
from benchmarks.super_resolution.mac_readability.stroke_loss import (
    structure_loss,
    objective,
)


def test_identity_and_finite_gradients():
    torch.manual_seed(7)
    truth = torch.rand(2, 3, 32, 32)
    pred = truth.clone().requires_grad_()
    loss = structure_loss(pred, truth)
    assert abs(float(loss)) < 1e-6
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_blurred_strokes_and_inverted_contrast_are_penalized():
    truth = torch.ones(1, 3, 40, 40)
    truth[:, :, 8:32, 18:20] = 0
    truth[:, :, 19:21, 8:32] = 0
    blurred = F.avg_pool2d(truth, 3, stride=1, padding=1).requires_grad_()
    loss, rgb, struct = objective(blurred, truth)
    assert loss > 0 and rgb > 0 and struct > 0
    assert structure_loss(1 - truth, truth) > structure_loss(blurred, truth)
    loss.backward()
    assert torch.isfinite(blurred.grad).all() and blurred.grad.abs().max() > 0


def test_matches_independent_gaussian_ssim_reference():
    import numpy as np
    from skimage.metrics import structural_similarity

    torch.manual_seed(8)
    x = torch.rand(1, 3, 40, 40)
    y = (x + 0.03 * torch.randn_like(x)).clamp(0, 1)
    weights = np.array([0.2126, 0.7152, 0.0722], np.float32)
    a = x[0].permute(1, 2, 0).numpy() @ weights
    b = y[0].permute(1, 2, 0).numpy() @ weights
    expected = 1 - structural_similarity(
        a,
        b,
        data_range=1,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    assert abs(float(structure_loss(x, y)) - expected) < 1e-5
