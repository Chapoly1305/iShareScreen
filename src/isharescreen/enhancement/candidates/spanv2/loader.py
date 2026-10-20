"""Official SPANV2 x4 reference attention, without the custom CUDA extension."""
import importlib

from ...models import _packages


def build(source, scale=4):
    with _packages({"span_source": source}):
        cls = importlib.import_module("span_source.architecture").SPANV2_ESR
        return cls(3, 3, feature_channels=32, upscale=scale, bias=False, use_span_attn=False)
