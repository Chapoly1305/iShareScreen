"""RT4KSR-S x2 with strict official training weights and deployment fusion."""
import importlib
from types import SimpleNamespace

from ...models import _packages


def build(source):
    with _packages({"model": source / "model", "model.utils": source / "model/utils"}):
        cls = importlib.import_module("model.arch").rt4ksr_rep
        return cls(SimpleNamespace(feature_channels=24, num_blocks=4, scale=2, act_type="gelu", is_train=True))


def deploy(model):
    """Fold the official residual 1x1/3x3/1x1 blocks on CPU, including biases.

    Same algebra as the official test.py reparameterize(), without its CUDA-only
    tensor allocation or in-place mutation of the source checkpoint kernels.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F

    class InferenceNorm(nn.Module):
        """Same channel normalization, without a training-only PythonOp."""
        def __init__(self, original):
            super().__init__()
            self.weight, self.bias, self.eps = original.weight, original.bias, original.eps

        def forward(self, x):
            mean = x.mean(1, keepdim=True)
            variance = (x-mean).pow(2).mean(1, keepdim=True)
            normalized = (x-mean)/(variance+self.eps).sqrt()
            return self.weight.view(1, -1, 1, 1)*normalized + self.bias.view(1, -1, 1, 1)

    def fuse(parent):
        for name, child in list(parent.named_children()):
            if type(child).__name__ == "LayerNorm2d":
                setattr(parent, name, InferenceNorm(child))
                continue
            if type(child).__name__ != "ResBlock":
                fuse(child)
                continue
            first, middle, last = child.expand_conv, child.fea_conv, child.reduce_conv
            kernel = middle.weight.detach().clone()
            index = torch.arange(kernel.shape[0])
            kernel[index, index, 1, 1] += 1
            combined = F.conv2d(kernel, first.weight.permute(1, 0, 2, 3))
            bias = F.conv2d(first.bias.view(1, -1, 1, 1).expand(1, -1, 3, 3), kernel, middle.bias)
            combined = F.conv2d(combined.permute(1, 0, 2, 3), last.weight).permute(1, 0, 2, 3)
            bias = F.conv2d(bias, last.weight, last.bias).view(-1)
            index = torch.arange(combined.shape[0])
            combined[index, index, 1, 1] += 1
            replacement = nn.Conv2d(combined.shape[1], combined.shape[0], 3, padding=1)
            with torch.no_grad():
                replacement.weight.copy_(combined)
                replacement.bias.copy_(bias)
            setattr(parent, name, replacement)

    fuse(model)
    if model.forget:
        raise ValueError("The pinned RT4KSR-S configuration requires forget=False")

    class Deployment(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.down, self.head = original.down, original.head
            self.body, self.tail, self.upsample = original.body, original.tail, original.upsample

        def forward(self, x):
            # forget=False leaves the Gaussian/high-frequency branch unused.
            # Drop only that dead branch; all contributing weights are retained.
            return self.upsample(self.tail(self.body(self.head(self.down(x)))))

    return Deployment(model)
