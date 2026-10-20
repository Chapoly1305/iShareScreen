"""Renderer RGB packing is exact, optional, and keeps caller pixels separate."""
import sys

import numpy as np
import pytest

from isharescreen.frontend.desktop.gpu import _pack_rgba


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_rgba_pack_is_exact_for_fast_and_dependency_free_paths(monkeypatch, fallback, strided):
    if fallback:
        monkeypatch.setitem(sys.modules, "cv2", None)
    else:
        pytest.importorskip("cv2")
    source = np.random.default_rng(19).integers(0, 256, size=(31, 47, 3), dtype=np.uint8)
    if strided:
        source = source[::-1, ::-1]
    before = source.copy()
    staging = np.zeros((31, 47, 4), np.uint8)
    _pack_rgba(source, staging)
    np.testing.assert_array_equal(staging[..., :3], before)
    assert (staging[..., 3] == 255).all()
    np.testing.assert_array_equal(source, before)
    assert not np.shares_memory(source, staging)
    source[:] = 0
    np.testing.assert_array_equal(staging[..., :3], before)


def test_opencv_packer_writes_the_existing_staging_buffer(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    original = cv2.cvtColor
    calls = []

    def convert(source, code, *, dst):
        calls.append(dst)
        return original(source, code, dst=dst)

    monkeypatch.setattr(cv2, "cvtColor", convert)
    staging = np.empty((5, 7, 4), np.uint8)
    for value in (23, 197):
        _pack_rgba(np.full((5, 7, 3), value, np.uint8), staging)
        assert (staging[..., :3] == value).all() and (staging[..., 3] == 255).all()
    assert len(calls) == 2 and all(destination is staging for destination in calls)
