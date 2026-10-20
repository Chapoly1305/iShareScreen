"""Profiling keeps cold calls separate and removes diagnostic hooks on errors."""
import numpy as np
import pytest

from benchmarks.super_resolution.profile_live import leaf_hotspots, measure


def test_warmed_stage_samples_and_percentiles():
    class Adapter:
        calls = 0

        def __call__(self, image):
            self.calls += 1
            return image, {"total_ms": float(self.calls), "inference_ms": float(self.calls) - 0.5}

    adapter = Adapter()
    result = measure(adapter, np.zeros((8, 8, 3), np.uint8), 2, 3)
    assert adapter.calls == 5
    assert [r["total_ms"] for r in result["samples"]] == [3, 4, 5]
    assert result["p50_ms"] == {"total_ms": 4.0, "inference_ms": 3.5}
    assert result["p95_total_ms"] == pytest.approx(4.9)
    with pytest.raises(ValueError):
        measure(adapter, np.zeros((8, 8, 3), np.uint8), 0, 3)


@pytest.mark.parametrize("fail", [False, True])
def test_leaf_hooks_are_removed_even_if_model_fails(fail):
    torch = pytest.importorskip("torch")

    class Adapter:
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.ReLU())

        def sync(self):
            pass

        def __call__(self, image):
            self.model(torch.zeros(1, 3, 8, 8))
            if fail:
                raise RuntimeError("Inference failed")
            return image, {"total_ms": 1}

    adapter = Adapter()
    if fail:
        with pytest.raises(RuntimeError, match="Inference failed"):
            leaf_hotspots(adapter, np.zeros((8, 8, 3), np.uint8))
    else:
        costs = leaf_hotspots(adapter, np.zeros((8, 8, 3), np.uint8))
        assert set(costs) == {"Conv2d", "ReLU"} and all(v >= 0 for v in costs.values())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in adapter.model.modules())
