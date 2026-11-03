import numpy as np
import torch
from benchmarks.super_resolution.mac_readability.train_strokes import Cache


def test_arms_share_crops_and_mixture_only_changes_degradation():
    cache = Cache.__new__(Cache)
    row = {
        "transform": {
            "content_box": [0, 0, 1920, 1080],
            "line_boxes": [[200, 200, 800, 240]],
        }
    }
    cache.rows = [row] * 512
    codec = np.full((720, 1280, 3), 10, np.uint8)
    steady = np.full_like(codec, 20)
    high = np.broadcast_to(
        (np.arange(1080) % 256).astype(np.uint8)[:, None, None], (1080, 1920, 3)
    )
    cache.pairs = [(codec, steady, high)] * 512
    a, truth_a = cache.batch({"steady_probability": 1.0}, 3, 20261104)
    b, truth_b = cache.batch({"steady_probability": 0.5}, 3, 20261104)
    c, truth_c = cache.batch({"steady_probability": 0.0}, 3, 20261104)
    assert torch.equal(truth_a, truth_b) and torch.equal(truth_a, truth_c)
    assert torch.all(a == 20 / 255) and torch.all(c == 10 / 255)
    assert set(b[:, 0, 0, 0].tolist()) == set(
        [float(a[0, 0, 0, 0]), float(c[0, 0, 0, 0])]
    )
    again, truth_again = cache.batch({"steady_probability": 0.5}, 3, 20261104)
    assert torch.equal(b, again) and torch.equal(truth_b, truth_again)
