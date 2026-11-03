import numpy as np
from benchmarks.super_resolution.mac_readability.consistency import Projection, correct


def test_adjoint_and_constant_preservation():
    p = Projection(48, 72)
    rng = np.random.default_rng(9)
    x = rng.normal(size=(48, 72)).astype(np.float32)
    y = rng.normal(size=(32, 48)).astype(np.float32)
    assert np.allclose(
        np.sum(p.down(x) * y), np.sum(x * p.adjoint(y)), rtol=1e-5, atol=1e-5
    )
    assert np.allclose(p.down(np.ones_like(x)), 1, atol=1e-6)


def test_consistent_flat_image_unchanged_and_residual_reduced():
    p = Projection(48, 72)
    prior = np.full((48, 72, 3), 128, np.uint8)
    low = np.full((32, 48, 3), 128, np.uint8)
    assert np.array_equal(correct(prior, low, p, [3])[3], prior)
    low[8:24, 12:36] = 140
    out = correct(prior, low, p, [3])[3]
    assert np.abs(out.astype("i2") - prior.astype("i2")).max() <= 16
    assert np.mean(
        (p.down(out[:, :, 0].astype(np.float32)) - low[:, :, 0]) ** 2
    ) < np.mean((p.down(prior[:, :, 0].astype(np.float32)) - low[:, :, 0]) ** 2)
