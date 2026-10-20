"""Official EfRLFN x2, 52 features, six residual blocks."""
import importlib

from ...models import _packages


def build(source):
    with _packages({"code": source / "code"}):
        return importlib.import_module("code.model").EfRLFN(upscale=2)
