"""Regression gates for tile geometry, queue bounds and stale model output."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2")

from isharescreen.enhancement.live import EnhancementDisplay, EnhancementWorker, TileCanvas, compose, tile_rgb
from isharescreen.enhancement.models import Upscaler, availability
from isharescreen.proxy.media.tiles import TileFrame


def tile(y_value=100, biplanar=False, subsampled=False):
    width,height,stride=8,8,12
    cw,ch=(4,4) if subsampled else (8,8)
    y=np.full((height,stride),y_value,np.uint8)
    u=np.full((ch,stride if not biplanar else stride*2),128,np.uint8)
    v=np.full((ch,stride),128,np.uint8)
    return TileFrame(y.tobytes(),u.tobytes(),None if biplanar else v.tobytes(),
                     width,height,stride,u.shape[1],cw,ch)


def wait_until(predicate):
    deadline=time.monotonic()+4
    while time.monotonic()<deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("worker did not reach expected state")


@pytest.mark.parametrize("subsampled",[False,True])
def test_padded_planar_and_biplanar_match(subsampled):
    planar=tile_rgb(tile(subsampled=subsampled))
    biplanar=tile_rgb(tile(biplanar=True,subsampled=subsampled))
    np.testing.assert_array_equal(planar,biplanar)
    assert planar.shape==(8,8,3)
    np.testing.assert_allclose(planar,np.broadcast_to([101,100,101],(8,8,3)),atol=1)


def test_canvas_waits_for_all_tiles_and_clips_ctu_padding():
    canvas=TileCanvas(8,12,2)
    canvas.update(0,tile(60),8)
    assert canvas.snapshot() is None
    canvas.update(1,tile(180),8)
    out=compose(canvas.snapshot())
    assert out.shape==(12,8,3)
    assert out[:8].mean()<70 and out[8:].mean()>170


def test_uninstalled_checkpoint_has_no_untrained_fallback(tmp_path):
    assert not next(x for x in availability(tmp_path) if x["id"]=="edsr")["available"]
    with pytest.raises(RuntimeError):
        Upscaler("edsr",home=tmp_path)


def test_ecbsr_preserves_official_digital_y_input_range():
    torch=pytest.importorskip("torch")
    seen=[]
    class NativeRange(torch.nn.Module):
        def forward(self,tensor):
            seen.append(float(tensor.mean()))
            return torch.nn.functional.interpolate(tensor,scale_factor=2,mode="nearest")
    adapter=Upscaler.__new__(Upscaler)
    adapter.key="ecbsr";adapter.device="cpu";adapter.dtype=torch.float32;adapter.model=NativeRange()
    out,_=adapter(np.full((8,8,3),255,np.uint8))
    assert seen[0]==pytest.approx(235)
    assert out.shape==(16,16,3)
    np.testing.assert_allclose(out,255,atol=1)


def test_nonfinite_model_output_is_rejected_before_uint8_quantization():
    torch=pytest.importorskip("torch")
    class Broken(torch.nn.Module):
        def forward(self,tensor):return torch.full((1,3,16,16),float("nan"))
    adapter=Upscaler.__new__(Upscaler)
    adapter.key="imdn";adapter.device="cpu";adapter.dtype=torch.float32;adapter.model=Broken()
    with pytest.raises(FloatingPointError,match="non-finite"):
        adapter(np.zeros((8,8,3),np.uint8))


def test_latest_frame_only_and_immediate_bypass():
    entered,release=threading.Event(),threading.Event()
    calls=[]
    class Fake:
        def __init__(self,key,device): self.key=key
        def __call__(self,im):
            calls.append(int(im[0,0,0])); entered.set(); release.wait(4)
            return im,{"total_ms":1}
    worker=EnhancementWorker(factory=Fake)
    try:
        worker.select("bicubic")
        worker.submit(np.full((8,8,3),1,np.uint8))
        assert entered.wait(2)
        for value in range(2,10):
            worker.submit(np.full((8,8,3),value,np.uint8))
        assert worker.snapshot()["pending"]==1
        assert worker.snapshot()["dropped"]==7
        release.set()
        wait_until(lambda:len(calls)==2)
        assert calls==[1,9]
        worker.select("off")
        assert worker.poll() is None
        assert worker.snapshot()["displayed"]=="off"
    finally:
        release.set(); worker.close()


def test_switch_and_resize_reject_inflight_outputs():
    entered,release=threading.Event(),threading.Event()
    class Fake:
        def __init__(self,key,device): self.key=key
        def __call__(self,im):
            entered.set(); release.wait(4)
            return im,{"total_ms":1}
    worker=EnhancementWorker(factory=Fake)
    try:
        worker.select("bicubic"); worker.submit(np.zeros((8,8,3),np.uint8))
        assert entered.wait(2)
        worker.invalidate(); release.set()
        wait_until(lambda:worker.snapshot()["state"]=="ready")
        assert worker.poll() is None
        worker.select("bilinear")
        worker.submit(np.zeros((16,16,3),np.uint8))
        wait_until(lambda:worker.snapshot()["processed"]==1)
        result=worker.poll()
        assert result[1]=="bilinear" and result[2].shape==(16,16,3)
        worker.select("off")
        worker.displayed(result[0],result[1])
        assert worker.snapshot()["displayed"]=="off"
    finally:
        release.set(); worker.close()


def test_error_stops_retry_until_new_selection():
    calls=[]
    def bad_factory(key,device):
        calls.append(key)
        raise RuntimeError("incompatible checkpoint")
    worker=EnhancementWorker(factory=bad_factory)
    try:
        worker.select("bicubic")
        wait_until(lambda:worker.snapshot()["state"]=="error")
        worker.submit(np.zeros((8,8,3),np.uint8))
        assert calls==["bicubic"] and worker.snapshot()["pending"]==0
        worker.select("off")
        assert worker.snapshot()["state"]=="bypass"
    finally:
        worker.close()


def test_cold_model_load_starts_with_latest_waiting_frame():
    entered,release=threading.Event(),threading.Event()
    calls=[]
    class Fake:
        def __init__(self,key,device):
            entered.set(); release.wait(4)
        def __call__(self,im):
            calls.append(int(im[0,0,0]));return im,{"total_ms":1}
    worker=EnhancementWorker(factory=Fake)
    try:
        worker.select("bicubic");worker.submit(np.full((8,8,3),1,np.uint8))
        assert entered.wait(2)
        worker.submit(np.full((8,8,3),9,np.uint8));release.set()
        wait_until(lambda:worker.snapshot()["processed"]==1)
        assert calls==[9]
        assert worker.snapshot()["p50_ms"] is not None
        worker.select("off")
        assert worker.snapshot()["p50_ms"] is None
    finally:
        release.set();worker.close()


@pytest.mark.parametrize("replacement", ["bilinear", "bicubic"])
def test_cold_load_does_not_start_invalidated_selection_or_geometry(replacement):
    entered, release = threading.Event(), threading.Event()
    calls = []
    class Fake:
        def __init__(self, key, device):
            self.key = key
            if key == "bicubic":
                entered.set()
                release.wait(4)
        def __call__(self, im):
            calls.append((self.key, im.shape[:2]))
            return im, {"total_ms": 1}
    worker = EnhancementWorker(factory=Fake)
    try:
        worker.select("bicubic")
        worker.submit(np.zeros((8, 8, 3), np.uint8))
        assert entered.wait(2)
        if replacement == "bicubic":
            worker.invalidate()  # Same model, but a new canvas geometry.
        else:
            worker.select(replacement)
        worker.submit(np.zeros((16, 16, 3), np.uint8))
        release.set()
        wait_until(lambda: worker.snapshot()["processed"] == 1)
        assert calls == [(replacement, (16, 16))], "Obsolete cold-load input must not consume an inference"
        result = worker.poll()
        assert result[1] == replacement and result[2].shape == (16, 16, 3)
    finally:
        release.set()
        worker.close()


def test_canvas_invalidated_during_assembly_does_not_enter_model(monkeypatch):
    import isharescreen.enhancement.live as live
    entered, release = threading.Event(), threading.Event()
    original_compose = live.compose
    calls = []
    def slow_compose(snapshot):
        entered.set()
        release.wait(4)
        return original_compose(snapshot)
    class Fake:
        def __init__(self, key, device): pass
        def __call__(self, im):
            calls.append(im.shape[:2])
            return im, {"total_ms": 1}
    monkeypatch.setattr(live, "compose", slow_compose)
    worker = EnhancementWorker(factory=Fake)
    canvas = TileCanvas(8, 8, 1)
    canvas.update(0, tile(), 8)
    try:
        worker.select("bicubic")
        worker.submit(canvas.snapshot())
        assert entered.wait(2)
        worker.invalidate()
        worker.submit(np.zeros((16, 16, 3), np.uint8))
        release.set()
        wait_until(lambda: worker.snapshot()["processed"] == 1)
        assert calls == [(16, 16)]
    finally:
        release.set()
        worker.close()


def test_dropdown_only_in_post_login_dashboard():
    from isharescreen.gui import connect
    dashboard=connect._dash_page().decode()
    assert 'id="enhancement-model"' in dashboard
    assert 'onchange="setEnhancement(this.value)"' in dashboard
    assert "enhancementState(d.enhancement)" in dashboard
    assert 'id="enhancement-model"' not in connect._FORM


class DisplayRenderer:
    def __init__(self):
        self.frames = []
        self.rgb = None

    def clear_enhancement(self):
        self.rgb = None

    def upload_enhanced(self, rgb):
        self.rgb = rgb.copy()
        self.frames.append(self.rgb)


def test_stage_timings_and_upload_cost_follow_the_current_generation():
    class Fake:
        def __init__(self, key, device):
            pass

        def __call__(self, image):
            return image.repeat(2, 0).repeat(2, 1), {"total_ms": 2, "inference_ms": 1.5}

    worker = EnhancementWorker(factory=Fake)
    renderer = DisplayRenderer()
    display = EnhancementDisplay(worker, renderer, 8, 8, 1)
    try:
        worker.select("bicubic")
        display.update_tile(0, tile(), 8)
        display.step()
        wait_until(lambda: worker.snapshot()["processed"] == 1)
        assert display.step()
        status = worker.snapshot()
        assert status["timing_ms"]["inference_ms"] == 1.5
        assert status["timing_ms"]["assembly_ms"] >= 0
        assert status["display_upload_ms"] >= 0
        status["timing_ms"].clear()
        assert worker.snapshot()["timing_ms"]["inference_ms"] == 1.5
        generation = worker.snapshot()["generation"]
        worker.select("off")
        assert not worker.displayed(generation, "bicubic", 1000)
        assert worker.snapshot()["timing_ms"] == {} and worker.snapshot()["display_upload_ms"] is None
    finally:
        worker.close()


@pytest.mark.parametrize("change", ["off", "bilinear", "resize", "close"])
@pytest.mark.parametrize("boundary", ["after_poll", "during_upload"])
def test_late_selection_change_never_presents_obsolete_output(monkeypatch, change, boundary):
    class Fake:
        def __init__(self, key, device): pass
        def __call__(self, im):
            return im.repeat(2, axis=0).repeat(2, axis=1), {"total_ms": 1}
    worker = EnhancementWorker(factory=Fake)
    renderer = DisplayRenderer()
    display = EnhancementDisplay(worker, renderer, 8, 8, 1)
    def change_selection():
        if change == "resize": worker.invalidate()
        elif change == "close": worker.close()
        else: worker.select(change)
    try:
        worker.select("bicubic")
        display.update_tile(0, tile(), 8)
        display.step()
        wait_until(lambda: worker.snapshot()["processed"] == 1)
        if boundary == "after_poll":
            original_poll = worker.poll
            def late_poll():
                result = original_poll()
                if result is not None: change_selection()
                return result
            monkeypatch.setattr(worker, "poll", late_poll)
        else:
            original_upload = renderer.upload_enhanced
            def late_upload(rgb):
                original_upload(rgb)
                change_selection()
            monkeypatch.setattr(renderer, "upload_enhanced", late_upload)
        assert display.step()
        assert renderer.rgb is None, "An obsolete output must be bypassed before drawing"
        assert worker.snapshot()["displayed"] == "off"
        if boundary == "after_poll": assert renderer.frames == []
    finally:
        worker.close()


def test_fast_same_model_reselection_repaints_static_canvas():
    class Fake:
        def __init__(self, key, device): pass
        def __call__(self, im):
            return im.repeat(2, axis=0).repeat(2, axis=1), {"total_ms": 1}
    worker = EnhancementWorker(factory=Fake)
    renderer = DisplayRenderer()
    display = EnhancementDisplay(worker, renderer, 8, 8, 1)
    try:
        display.update_tile(0, tile(), 8)
        worker.select("bicubic")
        display.step()
        wait_until(lambda: worker.snapshot()["processed"] == 1)
        assert display.step()
        assert len(renderer.frames) == 1
        # Both control commands can arrive between two render-loop iterations.
        worker.select("off")
        worker.select("bicubic")
        assert display.step(), "A→Original→A must be observed even though the final model name matches"
        assert renderer.rgb is None
        wait_until(lambda: worker.snapshot()["processed"] == 2)
        assert display.step()
        assert len(renderer.frames) == 2
        assert worker.snapshot()["displayed"] == "bicubic"
    finally:
        worker.close()


def test_close_during_model_loading_never_starts_obsolete_inference():
    entered, release = threading.Event(), threading.Event()
    calls = []
    class Fake:
        def __init__(self, key, device):
            entered.set()
            release.wait(10)
        def __call__(self, im):
            calls.append(True)
            return im, {"total_ms": 1}
    worker = EnhancementWorker(factory=Fake)
    try:
        worker.select("bicubic")
        worker.submit(np.zeros((8, 8, 3), np.uint8))
        assert entered.wait(2)
        worker.close()
        release.set()
        worker._thread.join(timeout=2)
        assert not worker._thread.is_alive()
        assert calls == []
        assert worker.poll() is None
        assert worker.snapshot()["state"] == "closed"
        with pytest.raises(ValueError, match="closed"):
            worker.select("bilinear")
    finally:
        release.set()
        worker.close()


def test_display_upload_failure_restores_original_without_retry():
    class Fake:
        def __init__(self, key, device): pass
        def __call__(self, im): return im, {"total_ms": 1}
    class FailingRenderer(DisplayRenderer):
        def upload_enhanced(self, rgb): raise RuntimeError("texture allocation failed")
    worker = EnhancementWorker(factory=Fake)
    renderer = FailingRenderer()
    renderer.rgb = np.ones((16, 16, 3), np.uint8)
    display = EnhancementDisplay(worker, renderer, 8, 8, 1)
    try:
        worker.select("bicubic")
        display.update_tile(0, tile(), 8)
        display.step()
        wait_until(lambda: worker.snapshot()["processed"] == 1)
        assert display.step()
        assert renderer.rgb is None
        assert worker.snapshot()["displayed"] == "off"
        assert "texture allocation" in worker.snapshot()["error"]
        submitted = worker.snapshot()["submitted"]
        display.step()
        assert worker.snapshot()["submitted"] == submitted
    finally:
        worker.close()


def test_reconfigured_display_rejects_inflight_old_canvas():
    entered, release = threading.Event(), threading.Event()
    class Fake:
        def __init__(self, key, device): pass
        def __call__(self, im):
            entered.set();release.wait(4)
            return im.repeat(2, axis=0).repeat(2, axis=1), {"total_ms": 1}
    worker = EnhancementWorker(factory=Fake)
    old, new = DisplayRenderer(), DisplayRenderer()
    display = EnhancementDisplay(worker, old, 8, 8, 1)
    try:
        worker.select("bicubic");display.update_tile(0, tile(), 8);display.step()
        assert entered.wait(2)
        display.reconfigure(new, 16, 8, 1)
        y = np.full((8, 16), 90, np.uint8).tobytes()
        uv = np.full((8, 16), 128, np.uint8).tobytes()
        display.update_tile(0, TileFrame(y, uv, uv, 16, 8, 16, 16, 16, 8), 8)
        display.step();release.set()
        wait_until(lambda: worker.snapshot()["processed"] == 1)
        assert display.step()
        assert not old.frames
        assert new.rgb.shape == (16, 32, 3)
        assert worker.snapshot()["active"] == "bicubic"
    finally:
        release.set();worker.close()
