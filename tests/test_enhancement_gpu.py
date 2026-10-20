"""Opt-in real GPU readback: enhancement, bypass, crop and cursor composition."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("ISS_TEST_GPU") != "1",reason="Set ISS_TEST_GPU=1 for real GPU readback")


@pytest.mark.parametrize("fallback", [False, True])
def test_enhanced_rgb_bypass_crop_and_cursor(monkeypatch, fallback):
    if fallback:
        import sys
        monkeypatch.setitem(sys.modules, "cv2", None)
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
    staging = renderer._rgb_staging
    replacement = rgb.copy()
    replacement[:] = [90, 40, 170]
    renderer.upload_enhanced(replacement)
    assert renderer._rgb_staging is staging
    replacement[:] = 0  # Upload must not retain caller-owned pixels.
    np.testing.assert_allclose(read()[16,32,:3], [90,40,170], atol=1)
    assert (staging[...,3] == 255).all()
    renderer.upload_enhanced(rgb)
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


@pytest.mark.skipif(os.environ.get("ISS_TEST_COREML") != "1", reason="Set ISS_TEST_COREML=1 for installed Core ML inference")
@pytest.mark.parametrize("key,gpu_only", [(key, False) for key in ("quicksrnet", "plainusr-coreml", "rt4ksr-coreml", "efrlfn-coreml")]
                         + [(key, True) for key in ("quicksrnet-torch", "plainusr", "rt4ksr", "efrlfn")])
def test_real_model_worker_to_metal_renderer(key, gpu_only, monkeypatch):
    """Exercise real canvas assembly, inference, upload/readback, and bypass."""
    import time
    import wgpu
    from isharescreen.enhancement import coreml, variants
    from isharescreen.enhancement.live import EnhancementDisplay, EnhancementWorker
    from isharescreen.frontend.desktop.gpu import Renderer
    from isharescreen.proxy.media.tiles import TileFrame

    if gpu_only:
        monkeypatch.setenv("ISS_SR_GPU_ONLY", "1")
        monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    option = next(row for row in variants.availability() if row["id"] == key)
    if not option["available"]:
        pytest.skip(option["reason"])
    adapter = variants.load(key)
    captured = []

    class Capture:
        device, provenance = adapter.device, adapter.provenance

        def __call__(self, rgb):
            output, timing = adapter(rgb)
            captured.append(output.copy())
            return output, timing

    device = wgpu.gpu.request_adapter_sync(power_preference="high-performance").request_device_sync()
    renderer = Renderer(device, "rgba8unorm", 960, 540)
    y = np.full((540, 960), 90, np.uint8).tobytes()
    uv = np.full((540, 960), 128, np.uint8).tobytes()
    tile = TileFrame(y, uv, uv, 960, 540, 960, 960, 960, 540)
    renderer.upload_tile(0, tile, 540)
    target = device.create_texture(size=(1920, 1080, 1), format="rgba8unorm",
                                   usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)

    def read():
        renderer.draw(target.create_view(), 1920, 1080)
        return np.frombuffer(device.queue.read_texture(
            {"texture": target}, {"bytes_per_row": 1920*4}, (1920, 1080, 1)), np.uint8).reshape(1080, 1920, 4).copy()

    baseline = read()
    worker = EnhancementWorker(factory=lambda *args, **kwargs: Capture())
    display = EnhancementDisplay(worker, renderer, 960, 540, 1)
    try:
        worker.select(key)
        display.update_tile(0, tile, 540)
        deadline = time.monotonic() + 30
        while worker.snapshot()["displayed"] != key and time.monotonic() < deadline:
            display.step()
            assert worker.snapshot()["state"] != "error", worker.snapshot()["error"]
            time.sleep(0.002)
        assert worker.snapshot()["displayed"] == key
        np.testing.assert_allclose(read()[..., :3], captured[-1], atol=1)
        assert worker.snapshot()["timing_ms"]["assembly_ms"] >= 0
        assert worker.snapshot()["display_upload_ms"] >= 0
        worker.select("off")
        display.step()
        np.testing.assert_array_equal(read(), baseline)
    finally:
        worker.close()
