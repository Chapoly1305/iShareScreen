import numpy as np
import torch

from benchmarks.super_resolution.mac_readability.train_distill import Cache


def test_teacher_and_truth_share_source_and_exact_three_over_two_crop_phase():
    cache = Cache.__new__(Cache)
    cache.rows = []
    cache.pairs = []
    ly, lx = np.indices((720, 1280))
    hy, hx = np.indices((1080, 1920))
    for identity in [17, 91]:
        cache.rows.append(
            {
                "transform": {
                    "content_box": [0, 0, 1920, 1080],
                    "line_boxes": [[300, 200, 1200, 260], [40, 1000, 1800, 1040]],
                }
            }
        )
        low = np.stack(
            [lx % 251, ly % 251, np.full_like(lx, identity)], axis=-1
        ).astype(np.uint8)
        high = np.stack(
            [(hx * 2 // 3) % 251, (hy * 2 // 3) % 251, np.full_like(hx, identity)],
            axis=-1,
        ).astype(np.uint8)
        cache.pairs.append((low, high, 255 - high))
    a = cache.batch(5, 20261105)
    b = cache.batch(5, 20261105)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    low, truth, teacher = a
    assert low.shape == (16, 3, 128, 128) and truth.shape == teacher.shape == (
        16,
        3,
        192,
        192,
    )
    assert torch.allclose(truth + teacher, torch.ones_like(truth), atol=1e-7)
    assert torch.equal(low[:, :, 0, 0], truth[:, :, 0, 0])
    assert torch.equal(low[:, :, 2, 2], truth[:, :, 3, 3])
    assert torch.equal(low[:, :, 126, 126], truth[:, :, 189, 189])
    assert set(low[:, 2, 0, 0].tolist()) == {
        float(torch.tensor(17 / 255)),
        float(torch.tensor(91 / 255)),
    }
    assert not torch.equal(low, cache.batch(6, 20261105)[0])
