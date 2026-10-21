"""The faster tile packer preserves every uint8 BT.709 conversion exactly."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from isharescreen.enhancement.live import tile_rgb
from isharescreen.proxy.media.tiles import TileFrame


def reference(y, u, v):
    yf, cb, cr = y.astype(np.float32)/255, u.astype(np.float32)/255-.5, v.astype(np.float32)/255-.5
    rgb = np.stack((yf+1.5748*cr, yf-.187324*cb-.468124*cr, yf+1.8556*cb), axis=-1)
    return np.clip(np.rint(rgb*255), 0, 255).astype(np.uint8)


def test_every_uint8_yuv_triplet_preserves_float32_rounding():
    u, v = np.meshgrid(np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8))
    for value in range(256):
        y = np.full((256,256), value, np.uint8)
        tile = TileFrame(y.tobytes(), u.tobytes(), v.tobytes(), 256,256,256,256,256,256)
        np.testing.assert_array_equal(tile_rgb(tile), reference(y,u,v))


@pytest.mark.parametrize("subsampled", [False, True])
def test_nonuniform_padded_planar_and_biplanar_match_reference(subsampled):
    rng = np.random.default_rng(91)
    w, h, stride = 30, 18, 40
    cw, ch = (w//2, h//2) if subsampled else (w,h)
    y = rng.integers(0,256,(h,stride), dtype=np.uint8)
    u = rng.integers(0,256,(ch,stride), dtype=np.uint8)
    v = rng.integers(0,256,(ch,stride), dtype=np.uint8)
    uv = np.full((ch,stride*2), 71, np.uint8)
    uv[:,:2*cw:2], uv[:,1:2*cw:2] = u[:,:cw], v[:,:cw]
    up, vp = u[:,:cw], v[:,:cw]
    if subsampled:
        up = cv2.resize(up,(w,h),interpolation=cv2.INTER_LINEAR)
        vp = cv2.resize(vp,(w,h),interpolation=cv2.INTER_LINEAR)
    expected = reference(y[:,:w],up,vp)
    for uu, vv, uv_stride in [(u.tobytes(),v.tobytes(),stride), (uv.tobytes(),None,stride*2)]:
        tile = TileFrame(y.tobytes(),uu,vv,w,h,stride,uv_stride,cw,ch)
        actual = tile_rgb(tile)
        np.testing.assert_array_equal(actual,expected)
        assert actual.flags.c_contiguous and actual.dtype == np.uint8
