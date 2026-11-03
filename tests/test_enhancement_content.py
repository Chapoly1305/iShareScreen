"""A late refinement must never hide newly decoded text."""

import threading
import time

import numpy as np
import pytest

from isharescreen.enhancement import live
from isharescreen.proxy.media.tiles import TileFrame


def tile(value):
    return TileFrame(
        bytes([value]) * 64, bytes([128]) * 16, bytes([128]) * 16, 8, 8, 8, 4, 4, 4
    )


def wait_until(predicate):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("Worker did not reach expected state")


class Renderer:
    def __init__(self):
        self.rgb = None
        self.uploads = []

    def clear_enhancement(self):
        self.rgb = None

    def upload_enhanced(self, rgb):
        self.rgb = rgb.copy()
        self.uploads.append(self.rgb)


@pytest.fixture
def available(monkeypatch):
    monkeypatch.setattr(
        live,
        "availability",
        lambda: [dict(id="off", available=True), dict(id="bicubic", available=True)],
    )
    # Keep the concurrency regression independent of OpenCV and model assets.
    monkeypatch.setattr(
        live,
        "compose",
        lambda frame: np.full(
            (frame.height, frame.width, 3), frame.tiles[0][0].y[0], np.uint8
        ),
    )


def test_new_text_bypasses_immediately_and_rejects_older_inference(available):
    entered = {v: threading.Event() for v in (100, 150, 200)}
    release = {v: threading.Event() for v in entered}
    release[100].set()

    class Model:
        def __call__(self, rgb):
            value = int(rgb[0, 0, 0])
            entered[value].set()
            assert release[value].wait(4)
            return rgb, {"total_ms": 1}

    worker = live.EnhancementWorker(factory=lambda *a, **k: Model())
    renderer = Renderer()
    display = live.EnhancementDisplay(worker, renderer, 8, 8, 1)
    try:
        worker.select("bicubic")
        display.update_tile(0, tile(100), 8)
        display.step()
        wait_until(lambda: worker.snapshot()["processed"] == 1)
        display.step()
        assert renderer.rgb[0, 0, 0] == 100

        display.update_tile(0, tile(150), 8)
        assert renderer.rgb is None, "New decoded text must be exposed before inference"
        assert worker.snapshot()["displayed"] == "off"
        display.step()
        assert entered[150].wait(2)
        display.update_tile(0, tile(200), 8)
        display.step()
        release[150].set()
        assert entered[200].wait(2)
        assert worker.snapshot()["discarded_stale"] == 1
        display.step()
        assert renderer.rgb is None
        assert len(renderer.uploads) == 1

        release[200].set()
        wait_until(lambda: worker.snapshot()["processed"] == 2)
        display.step()
        assert renderer.rgb[0, 0, 0] == 200
        assert len(renderer.uploads) == 2
    finally:
        for event in release.values():
            event.set()
        worker.close()


def test_identical_republished_tiles_keep_static_refinement(available):
    class Model:
        def __call__(self, rgb):
            return rgb, {"total_ms": 1}

    worker = live.EnhancementWorker(factory=lambda *a, **k: Model())
    renderer = Renderer()
    display = live.EnhancementDisplay(worker, renderer, 8, 8, 1)
    try:
        worker.select("bicubic")
        display.update_tile(0, tile(100), 8)
        display.step()
        wait_until(lambda: worker.snapshot()["processed"] == 1)
        display.step()
        before = worker.snapshot()
        for _ in range(10):
            display.update_tile(0, tile(100), 8)
            assert not display.step()
            assert renderer.rgb is not None
        after = worker.snapshot()
        assert after["submitted"] == before["submitted"]
        assert after["content_revision"] == before["content_revision"]
    finally:
        worker.close()


@pytest.mark.parametrize("boundary", ["after_poll", "during_upload"])
def test_content_change_at_presentation_boundary_rejects_old_pixels(
    available, monkeypatch, boundary
):
    class Model:
        def __call__(self, rgb):
            return rgb, {"total_ms": 1}

    worker = live.EnhancementWorker(factory=lambda *a, **k: Model())
    renderer = Renderer()
    display = live.EnhancementDisplay(worker, renderer, 8, 8, 1)
    try:
        worker.select("bicubic")
        display.update_tile(0, tile(100), 8)
        display.step()
        wait_until(lambda: worker.snapshot()["processed"] == 1)
        if boundary == "after_poll":
            original = worker.poll

            def poll():
                result = original()
                if result is not None:
                    display.update_tile(0, tile(200), 8)
                return result

            monkeypatch.setattr(worker, "poll", poll)
        else:
            original = renderer.upload_enhanced

            def upload(rgb):
                original(rgb)
                display.update_tile(0, tile(200), 8)

            monkeypatch.setattr(renderer, "upload_enhanced", upload)
        display.step()
        assert renderer.rgb is None
        assert worker.snapshot()["displayed"] == "off"
    finally:
        worker.close()


def test_error_from_obsolete_content_does_not_poison_new_text(available):
    entered, release = threading.Event(), threading.Event()

    class Model:
        def __call__(self, rgb):
            if rgb[0, 0, 0] == 100:
                entered.set()
                assert release.wait(4)
                raise RuntimeError("An obsolete input failed")
            return rgb, {"total_ms": 1}

    worker = live.EnhancementWorker(factory=lambda *a, **k: Model())
    renderer = Renderer()
    display = live.EnhancementDisplay(worker, renderer, 8, 8, 1)
    try:
        worker.select("bicubic")
        display.update_tile(0, tile(100), 8)
        display.step()
        assert entered.wait(2)
        display.update_tile(0, tile(200), 8)
        display.step()
        release.set()
        wait_until(lambda: worker.snapshot()["processed"] == 1)
        display.step()
        assert worker.snapshot()["state"] == "ready"
        assert renderer.rgb[0, 0, 0] == 200
    finally:
        release.set()
        worker.close()


def test_obsolete_upload_error_cannot_disable_latest_content(available):
    class Model:
        def __call__(self, rgb):
            return rgb, {"total_ms": 1}

    class FailingRenderer(Renderer):
        def upload_enhanced(self, rgb):
            display.update_tile(0, tile(200), 8)
            raise RuntimeError("Obsolete texture upload failed")

    worker = live.EnhancementWorker(factory=lambda *a, **k: Model())
    display = live.EnhancementDisplay(worker, FailingRenderer(), 8, 8, 1)
    try:
        worker.select("bicubic")
        display.update_tile(0, tile(100), 8)
        display.step()
        wait_until(lambda: worker.snapshot()["processed"] == 1)
        display.step()
        assert worker.snapshot()["state"] == "ready"
        assert worker.snapshot()["error"] == ""
    finally:
        worker.close()
