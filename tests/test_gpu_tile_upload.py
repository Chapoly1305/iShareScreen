"""Verify multi-tile chroma packing without a GPU."""
from types import SimpleNamespace

import numpy as np
import pytest

from isharescreen.frontend.desktop.gpu import Renderer
from isharescreen.proxy.media.tiles import TileFrame


@pytest.mark.parametrize("subsample", [1, 2])
@pytest.mark.parametrize("biplanar", [False, True])
def test_four_tiles_pack_chroma_without_gaps_and_crop_padding(subsample, biplanar):
    renderer = Renderer.__new__(Renderer)
    renderer._h = 30  # Four 8-row slots; final two luma rows are padding.
    renderer._mode = "biplanar" if biplanar else "planar"
    renderer._content_w = renderer._content_h = 0
    renderer._dbg_tiles = {}
    renderer._y_tex, renderer._u_tex, renderer._v_tex, renderer._uv_tex = "y", "u", "v", "uv"
    planes = {name: np.zeros((30, 16), dtype=np.uint8) for name in ("y", "u", "v", "uv")}
    def write(target, data, layout, size):
        width, height, _ = size
        if target["texture"] == "uv": width *= 2
        origin = target["origin"][1]
        planes[target["texture"]][origin:origin+height, :width] = np.asarray(data)[:height, :width]
    renderer._device = SimpleNamespace(queue=SimpleNamespace(write_texture=write))
    cw = ch = 8 // subsample
    for tile in range(4):
        value = tile + 1
        frame = TileFrame(bytes([value])*64, bytes([value])*(cw*ch*(2 if biplanar else 1)),
                          None if biplanar else bytes([value])*(cw*ch),
                          8, 8, 8, cw*(2 if biplanar else 1), cw, ch)
        renderer.upload_tile(tile, frame, 8)
    expected = np.repeat([1, 2, 3, 4], 8)[:30]
    np.testing.assert_array_equal(planes["y"][:30, 0], expected)
    chroma_expected = np.repeat([1, 2, 3, 4], ch)[:30//subsample]
    for name in (("uv",) if biplanar else ("u", "v")):
        np.testing.assert_array_equal(planes[name][:len(chroma_expected), 0], chroma_expected)
        assert not planes[name][len(chroma_expected):].any()
