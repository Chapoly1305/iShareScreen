"""Official already-deployed CASR x4; never relabel these weights as x2."""
import importlib

from ...models import _packages


def build(source):
    # The official file imports but never uses the removed stdlib imp module.
    # An isolated empty namespace keeps inference compatible with Python 3.12+.
    with _packages({"source": source / "source", "source.models": source / "source/models", "imp": source}):
        cls = importlib.import_module("source.models.RepNetworkPlain").RepNetwork_V010_BestStruct_deploy
        return cls(module_nums=1, channel_nums=32, act_type="gelu", scale=4, colors=3)
