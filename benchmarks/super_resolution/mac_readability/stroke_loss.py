"""Differentiable luminance structure loss for an isolated training experiment.

Local SSIM: Gaussian11x11 sigma1.5, C1=.01^2,C2=.03^2, valid windows.
RGB L1 anchors color. All calculations occur in FP32, outside AMP inference.
"""

import torch
import torch.nn.functional as F


def structure_loss(pred, reference):
    assert pred.shape == reference.shape and pred.ndim == 4 and pred.shape[1] == 3
    pred = pred.float()
    reference = reference.float()
    weights = pred.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    x = (pred * weights).sum(1, keepdim=True)
    y = (reference * weights).sum(1, keepdim=True)
    coordinates = torch.arange(-5, 6, device=pred.device, dtype=torch.float32)
    g = torch.exp(-coordinates.square() / (2 * 1.5**2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).view(1, 1, 11, 11)
    mu_x = F.conv2d(x, kernel)
    mu_y = F.conv2d(y, kernel)
    var_x = (F.conv2d(x * x, kernel) - mu_x.square()).clamp_min(0)
    var_y = (F.conv2d(y * y, kernel) - mu_y.square()).clamp_min(0)
    cov = F.conv2d(x * y, kernel) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + 0.01**2) * (2 * cov + 0.03**2)) / (
        (mu_x.square() + mu_y.square() + 0.01**2) * (var_x + var_y + 0.03**2)
    )
    return (1 - ssim).mean()


def objective(pred, reference, structural=True):
    rgb = F.l1_loss(pred.float(), reference.float())
    structure = structure_loss(pred, reference)
    return (0.2 * rgb + 0.8 * structure if structural else rgb), rgb, structure
