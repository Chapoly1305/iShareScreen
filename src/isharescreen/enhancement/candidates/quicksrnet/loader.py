"""Official QuickSRNet Small x2 architecture."""
import importlib

from ...models import _packages


def build(source):
    with _packages({"quicksr_source": source}):
        return importlib.import_module("quicksr_source.models").QuickSRNetSmall(scaling_factor=2)
