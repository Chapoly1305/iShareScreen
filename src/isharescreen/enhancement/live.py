"""Bounded latest-canvas inference; all Torch work lives off the render thread."""
from __future__ import annotations

from collections import deque
import logging
import os
import threading
import time
from typing import NamedTuple

import numpy as np

from .variants import availability, load
from ..proxy.media.tiles import TileFrame

log = logging.getLogger(__name__)


class CanvasSnapshot(NamedTuple):
    """Immutable tile references captured on the render thread."""

    width: int
    height: int
    tiles: tuple[tuple[TileFrame, int], ...]


class FrameRequest(NamedTuple):
    """One replaceable request, owned by a control/geometry generation."""

    generation: int
    submitted_at: float
    frame: CanvasSnapshot | np.ndarray


class EnhancedFrame(NamedTuple):
    """CPU RGB result; only the render thread may upload it to a texture."""

    generation: int
    model: str
    rgb: np.ndarray
    timing: dict[str, float]


def tile_rgb(tile):
    """Match the viewer's full-range BT.709 shader, including row padding."""
    import cv2
    y = np.frombuffer(tile.y, np.uint8).reshape(tile.height, tile.y_stride)[:, :tile.width]
    ch, cw = tile.chroma_height, tile.chroma_width
    u = np.frombuffer(tile.u, np.uint8).reshape(ch, tile.uv_stride)
    if tile.v is None:
        u, v = u[:, :2*cw:2], u[:, 1:2*cw:2]
    else:
        u = u[:, :cw]
        v = np.frombuffer(tile.v, np.uint8).reshape(ch, tile.uv_stride)[:, :cw]
    if (cw,ch) != (tile.width,tile.height):
        u = cv2.resize(u, (tile.width,tile.height), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v, (tile.width,tile.height), interpolation=cv2.INTER_LINEAR)
    yf, cb, cr = y.astype(np.float32)/255, u.astype(np.float32)/255-0.5, v.astype(np.float32)/255-0.5
    rgb = np.stack((yf+1.5748*cr, yf-0.187324*cb-0.468124*cr, yf+1.8556*cb), axis=-1)
    return np.clip(np.rint(rgb*255), 0,255).astype(np.uint8)


class TileCanvas:
    """Snapshot latest state of every tile, clipping encoder CTU padding.

    Tiles have no common frame timestamp in TileFrame. This composes a fully
    populated latest-state canvas, not an atomic cross-tile video frame.
    """
    def __init__(self, width, height, num_tiles):
        self.width, self.height, self.num_tiles = width, height, num_tiles
        self.tiles = {}

    def update(self, index, tile, slot_height):
        self.tiles[index] = (tile, slot_height)

    def snapshot(self):
        if len(self.tiles) != self.num_tiles:
            return None
        return CanvasSnapshot(self.width, self.height, tuple(self.tiles[i] for i in range(self.num_tiles)))


def compose(snapshot):
    width, height, tiles = snapshot
    out = np.zeros((height,width,3), dtype=np.uint8)
    for i, (tile, slot) in enumerate(tiles):
        y = i*slot
        rows, cols = min(tile.height, slot, height-y), min(tile.width,width)
        if rows > 0 and cols > 0:
            out[y:y+rows,:cols] = tile_rgb(tile)[:rows,:cols]
    return out


class EnhancementWorker:
    """Single inference thread with one pending request and one consumable result.

    Model selection, canvas replacement and shutdown invalidate generations.
    Neither late results nor late errors may affect a newer generation. Model
    construction and inference never hold the control condition lock.
    """

    def __init__(self, device="auto", factory=load):
        self.device, self.factory = device, factory
        self._cv = threading.Condition()
        self._selected, self._generation, self._pending, self._result = "off", 0, None, None
        self._closed = False
        self._options = availability()
        self._stats = {"selected":"off", "active":"off", "displayed":"off", "state":"bypass",
                       "error":"", "submitted":0, "processed":0, "dropped":0, "device":device,
                       "provenance":{}, "timing_ms":{}, "display_upload_ms":None}
        self._latencies = deque(maxlen=120)
        self._thread = threading.Thread(target=self._run, name="iss-super-resolution", daemon=True)
        self._thread.start()

    def select(self, key):
        if not any(x["id"] == key and x["available"] for x in self._options):
            raise ValueError(f"Model unavailable: {key}")
        with self._cv:
            if self._closed:
                raise ValueError("Enhancement worker is closed")
            if key == self._selected:
                return
            self._selected = key
            self._generation += 1
            self._pending = self._result = None
            self._latencies.clear()
            self._stats.update(selected=key, active="off", displayed="off",
                               state="bypass" if key == "off" else "loading", error="",
                               last_ms=None,age_ms=None,provenance={},timing_ms={},display_upload_ms=None)
            self._cv.notify_all()

    def invalidate(self):
        """Discard in-flight old-geometry output after a canvas rebuild."""
        with self._cv:
            self._generation += 1
            self._pending = self._result = None
            self._latencies.clear()
            self._stats.update(displayed="off",last_ms=None,age_ms=None,timing_ms={},display_upload_ms=None)
            self._cv.notify_all()

    def submit(self, frame):
        with self._cv:
            if self._selected == "off" or self._closed or self._stats["state"] == "error":
                return
            if self._pending is not None:
                self._stats["dropped"] += 1
            # TileFrames own immutable byte buffers; retaining a snapshot is cheap.
            self._pending = FrameRequest(self._generation, time.monotonic(), frame)
            self._stats["submitted"] += 1
            self._cv.notify_all()

    def poll(self):
        with self._cv:
            result, self._result = self._result, None
            if result is None or result.generation != self._generation:
                return None
            return result

    def displayed(self, generation, key, upload_ms=None):
        """Confirm presentation only if the control state still owns this output."""
        with self._cv:
            if (not self._closed and generation == self._generation and key == self._selected
                    and self._stats["state"] != "error"):
                self._stats["displayed"] = key
                self._stats["display_upload_ms"] = upload_ms
                return True
            return False

    def fail(self, error, generation=None):
        """Fail only the generation that owns a render-thread upload error."""
        with self._cv:
            if self._closed or (generation is not None and generation != self._generation):
                return False
            self._generation += 1
            self._pending = self._result = None
            self._stats.update(state="error",active="off",displayed="off",error=str(error))
            return True

    def snapshot(self):
        with self._cv:
            return dict(self._stats, provenance=dict(self._stats["provenance"]),
                        timing_ms=dict(self._stats["timing_ms"]),
                        options=[dict(option) for option in self._options], pending=int(self._pending is not None),
                        generation=self._generation,
                        p50_ms=float(np.median(self._latencies)) if self._latencies else None)

    def close(self):
        with self._cv:
            self._closed = True
            self._generation += 1
            self._pending = self._result = None
            self._stats.update(state="closed", active="off", displayed="off")
            self._cv.notify_all()
        self._thread.join(timeout=2)

    def _run(self):
        model, loaded = None, "off"
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._closed or self._pending is not None or self._selected != loaded)
                if self._closed:
                    return
                generation, key = self._generation, self._selected
                item, self._pending = self._pending, None
            try:
                if key != loaded:
                    model = None  # release previous model before loading the next
                    if key != "off":
                        options = {"device":self.device}
                        if os.environ.get("ISS_SR_PRECISION", "fp32") != "fp32":
                            options["precision"] = os.environ["ISS_SR_PRECISION"]
                        model = self.factory(key, **options)
                    loaded = key
                    with self._cv:
                        if self._closed:
                            return
                        if generation == self._generation:
                            self._stats.update(active=key, state="bypass" if key == "off" else "ready",
                                               provenance=dict(getattr(model, "provenance", {})))
                            # Loading may take seconds. Start with the latest
                            # canvas that arrived during loading, rather than
                            # briefly displaying a cold, obsolete first frame.
                            if self._pending is not None:
                                if item is not None:
                                    self._stats["dropped"] += 1
                                item, self._pending = self._pending, None
                if item is None or key == "off" or item.generation != generation:
                    continue
                with self._cv:
                    if self._closed:
                        return
                    if generation != self._generation:
                        continue  # Selection/geometry changed during cold loading.
                started = time.monotonic()
                frame = item.frame
                rgb = compose(frame) if isinstance(frame, tuple) else frame
                assembly_ms = (time.monotonic()-started)*1000
                with self._cv:
                    if self._closed:
                        return
                    if generation != self._generation:
                        continue  # Canvas assembly also runs outside the control lock.
                out, timing = model(rgb)
                timing["assembly_ms"] = assembly_ms
                timing["assembly_and_processing_ms"] = (time.monotonic()-started)*1000
                timing["age_ms"] = (time.monotonic()-item.submitted_at)*1000
                with self._cv:
                    if not self._closed and generation == self._generation and self._stats["state"] != "error":
                        self._result = EnhancedFrame(generation, key, out, timing)
                        self._stats.update(active=key, state="ready", last_ms=timing["assembly_and_processing_ms"],
                                           age_ms=timing["age_ms"], device=getattr(model,"device",self.device),
                                           timing_ms=dict(timing))
                        self._stats["processed"] += 1
                        self._latencies.append(timing["assembly_and_processing_ms"])
            except Exception as exc:
                log.exception("enhancement %s failed", key)
                with self._cv:
                    if not self._closed and generation == self._generation:
                        # Do not spin/retry a broken model on every render iteration.
                        self._stats.update(state="error", active="off", displayed="off", error=str(exc))
                        self._pending = self._result = None
                        loaded = key


class EnhancementDisplay:
    """Render-thread bridge between decoded tiles, inference and presentation."""

    def __init__(self, worker, renderer, width, height, num_tiles):
        self.worker, self.renderer = worker, renderer
        self.canvas = TileCanvas(width, height, num_tiles)
        self._selected, self._state, self._dirty = "off", "bypass", False
        self._generation = worker.snapshot()["generation"]

    def reconfigure(self, renderer, width, height, num_tiles):
        self.worker.invalidate()
        self.renderer = renderer
        self.canvas = TileCanvas(width, height, num_tiles)
        self._dirty = False

    def update_tile(self, index, tile, slot_height):
        self.canvas.update(index, tile, slot_height)
        self._dirty = True

    def step(self):
        """Return whether a model switch/result/error requires repainting."""
        status = self.worker.snapshot()
        # A→Original→A can fit between render iterations. Model names alone
        # cannot distinguish those requests or a same-model canvas rebuild.
        selection_changed = status["generation"] != self._generation
        self._generation = status["generation"]
        error_changed = status["state"] == "error" and self._state != "error"
        self._state = status["state"]
        if selection_changed or status["state"] == "error":
            self.renderer.clear_enhancement()
            self._selected = status["selected"]
        if (self._dirty or selection_changed) and self._selected != "off":
            frame = self.canvas.snapshot()
            if frame is not None:
                self.worker.submit(frame)
        self._dirty = False
        result = self.worker.poll()
        if result is None:
            return selection_changed or error_changed
        generation, key, rgb, timing = result
        if self.worker.snapshot()["generation"] != generation:
            self.renderer.clear_enhancement()
            return True
        started = time.perf_counter()
        try:
            self.renderer.upload_enhanced(rgb)
            # Controls stay responsive during potentially slow GPU allocation.
            # If selection/geometry changed during upload, bypass before draw.
            if not self.worker.displayed(generation, key, (time.perf_counter()-started)*1000):
                self.renderer.clear_enhancement()
        except Exception as exc:
            log.exception("enhancement output upload failed")
            self.renderer.clear_enhancement()
            self.worker.fail(exc, generation=generation)
        log.debug("enhancement %s processing %.2fms, display upload enqueue %.2fms",
                  key, timing["assembly_and_processing_ms"],
                  (time.perf_counter() - started) * 1000)
        return True
