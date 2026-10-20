"""Official Ultra deployment graph: 16 features, partial widths 16/8, ×2."""
import importlib
import sys
from types import ModuleType, SimpleNamespace

from ...models import _packages


def build(source):
    # The source uses BasicSR only as a registration decorator. Isolate an
    # identity registrar without changing the official architecture file.
    with _packages({"plainusr_source": source, "basicsr": source,
                    "basicsr.utils": source}):
        registry = ModuleType("basicsr.utils.registry")
        registry.ARCH_REGISTRY = SimpleNamespace(register=lambda: lambda cls: cls)
        sys.modules[registry.__name__] = registry
        cls = importlib.import_module("plainusr_source.architecture").PlainUSR_ultra_inference
        return cls(n_feat=16, im_feat=[16, 8], attn_feat=4, scale=2)
