"""Control/render races without model assets, OpenCV, GPU or a remote host."""
import time

import numpy as np
import pytest

from isharescreen.enhancement import live


def wait_until(predicate):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Enhancement worker did not reach the expected state")


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr(live, "availability", lambda: [
        {"id": key, "available": True, "label": key, "reason": ""}
        for key in ("off", "bicubic", "bilinear")
    ])

    class Model:
        def __init__(self, key, device):
            self.device = device

        def __call__(self, image):
            return image.repeat(2, axis=0).repeat(2, axis=1), {"total_ms": 1}

    instance = live.EnhancementWorker(factory=Model)
    try:
        yield instance
    finally:
        instance.close()


@pytest.mark.parametrize("change", ["off", "bilinear", "resize", "close"])
def test_obsolete_upload_failure_cannot_poison_new_control_state(worker, change):
    class Renderer:
        def clear_enhancement(self):
            pass

        def upload_enhanced(self, rgb):
            if change == "resize":
                worker.invalidate()
            elif change == "close":
                worker.close()
            else:
                worker.select(change)
            raise RuntimeError("Previous texture upload failed")

    display = live.EnhancementDisplay(worker, Renderer(), 8, 8, 1)
    worker.select("bicubic")
    worker.submit(np.zeros((8, 8, 3), dtype=np.uint8))
    wait_until(lambda: worker.snapshot()["processed"] == 1)
    assert display.step()
    status = worker.snapshot()
    assert status["state"] != "error"
    assert status["error"] == ""
    if change == "close":
        assert status["state"] == "closed"
    elif change == "off":
        assert status["selected"] == "off" and status["state"] == "bypass"
    else:
        worker.submit(np.zeros((8, 8, 3), dtype=np.uint8))
        wait_until(lambda: worker.snapshot()["processed"] == 2)


def test_closed_worker_cannot_be_reopened_by_a_late_error(worker):
    worker.close()
    worker.fail(RuntimeError("Late upload error"))
    assert worker.snapshot()["state"] == "closed"


def test_diagnostic_options_are_not_mutable_worker_state(worker):
    status = worker.snapshot()
    status["options"][0]["available"] = False
    status["options"].clear()
    assert len(worker.snapshot()["options"]) == 3
    worker.select("off")
