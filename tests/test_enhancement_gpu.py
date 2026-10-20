"""Opt-in real GPU readback: enhancement, bypass, crop and cursor composition."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("ISS_TEST_GPU") != "1",reason="Set ISS_TEST_GPU=1 for real GPU readback")


def test_enhanced_rgb_bypass_crop_and_cursor():
    import wgpu
    from isharescreen.frontend.desktop.gpu import Renderer
    from isharescreen.proxy.media.tiles import TileFrame
    adapter=wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device=adapter.request_device_sync()
    renderer=Renderer(device,"rgba8unorm",8,8)
    y=np.full((8,8),90,np.uint8).tobytes()
    uv=np.full((8,8),128,np.uint8).tobytes()
    renderer.upload_tile(0,TileFrame(y,uv,uv,8,8,8,8,8,8),8)
    target=device.create_texture(size=(64,32,1),format="rgba8unorm",
                                 usage=wgpu.TextureUsage.RENDER_ATTACHMENT|wgpu.TextureUsage.COPY_SRC)
    def read():
        renderer.draw(target.create_view(),64,32)
        return np.frombuffer(device.queue.read_texture({"texture":target},{"bytes_per_row":256},(64,32,1)),np.uint8).reshape(32,64,4)
    baseline=read().copy()
    rgb=np.empty((16,16,3),np.uint8);rgb[:]=[17,83,149];rgb[:,8:]=[40,180,20]
    renderer.upload_enhanced(rgb)
    enhanced=read()
    np.testing.assert_allclose(enhanced[16,8,:3],[17,83,149],atol=1)
    np.testing.assert_allclose(enhanced[16,56,:3],[40,180,20],atol=1)
    renderer.set_source_rect((4,0,4,8))
    np.testing.assert_allclose(read()[16,32,:3],[40,180,20],atol=1)
    renderer.set_source_rect(None)
    cursor=np.zeros((2,2,4),np.uint8);cursor[:]=[250,0,0,255]
    renderer.set_cursor_image(SimpleNamespace(width=2,height=2,rgba=cursor.tobytes(),hotspot_x=0,hotspot_y=0))
    renderer.set_cursor_pos((2,2))
    with_cursor=read()
    np.testing.assert_allclose(with_cursor[10,18,:3],[250,0,0],atol=1)
    renderer.set_cursor_pos(None)
    renderer.clear_enhancement()
    np.testing.assert_array_equal(read(),baseline)


def test_real_hevc444_decoder_canvas_matches_original_shader(tmp_path):
    import shutil
    import subprocess
    Image = pytest.importorskip("PIL.Image")
    import wgpu
    import av
    from isharescreen.proxy.media.decode_common import _av_frame_to_tile
    from isharescreen.enhancement.live import TileCanvas, compose
    from isharescreen.frontend.desktop.gpu import Renderer
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg with libx265 is required for the real decode fixture")
    x, y = np.meshgrid(np.arange(64), np.arange(64))
    rgb = np.stack((x*4, y*4, (x*2+y*2)), axis=-1).astype(np.uint8)
    source, encoded = tmp_path/"source.png", tmp_path/"fixture.hevc"
    Image.fromarray(rgb).save(source)
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-vf", "scale=in_range=full:out_range=full:out_color_matrix=bt709",
        "-color_range", "pc", "-colorspace", "bt709", "-frames:v", "1",
        "-c:v", "libx265", "-pix_fmt", "yuv444p", "-crf", "18",
        "-x265-params", "pools=1:frame-threads=1:log-level=error", str(encoded),
    ], check=True, capture_output=True)
    with av.open(str(encoded)) as container:
        frame = next(container.decode(video=0))
        assert frame.colorspace == 1 and frame.color_range == 2
        tile, had_error = _av_frame_to_tile(frame, [None], set())
    assert tile is not None and not had_error
    assert tile.is_yuv444 and (tile.width, tile.height) == (64, 64)
    canvas = TileCanvas(64, 64, 1)
    canvas.update(0, tile, 64)
    expected = compose(canvas.snapshot())
    device = wgpu.gpu.request_adapter_sync(power_preference="high-performance").request_device_sync()
    renderer = Renderer(device, "rgba8unorm", 64, 64)
    renderer.upload_tile(0, tile, 64)
    target = device.create_texture(size=(64,64,1), format="rgba8unorm",
                                   usage=wgpu.TextureUsage.RENDER_ATTACHMENT|wgpu.TextureUsage.COPY_SRC)
    renderer.draw(target.create_view(), 64, 64)
    actual = np.frombuffer(device.queue.read_texture(
        {"texture":target}, {"bytes_per_row":256}, (64,64,1)), np.uint8).reshape(64,64,4)[...,:3]
    np.testing.assert_allclose(actual, expected, atol=1)
