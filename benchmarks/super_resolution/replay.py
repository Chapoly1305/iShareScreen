"""Paced recorded-video source with one replaceable decoded tile, not a backlog."""
from pathlib import Path
import threading
import time

import av

from isharescreen.proxy.media.decode_common import _av_frame_to_tile


def preview_tile(frame,reformatter_holder=None,seen_fmts=None):
    if frame.colorspace != 1 or frame.color_range != 2:
        raise ValueError("Video preview requires full-range BT.709; prepare with full709")
    tile, had_error = _av_frame_to_tile(frame,
        [None] if reformatter_holder is None else reformatter_holder,
        set() if seen_fmts is None else seen_fmts)
    if tile is None or had_error:
        raise ValueError("Video did not produce a clean supported TileFrame")
    return tile


def video_tiles(path):
    with av.open(str(path)) as container:
        first=None;reformatter_holder=[None];seen_fmts=set()
        for frame in container.decode(video=0):
            if frame.pts is None:raise ValueError("Recorded frame has no timestamp")
            stamp=float(frame.pts*frame.time_base)
            if first is None:first=stamp
            yield stamp-first,preview_tile(frame,reformatter_holder,seen_fmts)


class VideoReplay:
    def __init__(self,path,loop=False,frame_source=video_tiles,fps=None):
        self.path=Path(path);self.loop=loop;self.frame_source=frame_source
        if fps is None:
            with av.open(str(path)) as container:
                fps=float(container.streams.video[0].average_rate)
        if fps<=0:raise ValueError("Video replay requires a positive frame rate")
        self.fps=fps
        self._lock=threading.Lock();self._stop=threading.Event();self._pending=None
        self._stats=dict(state="starting",emitted=0,delivered=0,superseded=0,loops=0,
                         pending=0,time_s=0.,fps=fps,error="")
        self._thread=threading.Thread(target=self._run,name="iss-video-replay",daemon=True)
        self._thread.start()

    def poll(self):
        with self._lock:
            item,self._pending=self._pending,None
            if item is not None:self._stats["delivered"]+=1
            return item

    def snapshot(self):
        with self._lock:return dict(self._stats,pending=int(self._pending is not None))

    def close(self):
        self._stop.set();self._thread.join(timeout=2)
        with self._lock:
            self._pending=None
            self._stats["state"]="closed"

    def _run(self):
        try:
            while not self._stop.is_set():
                started=time.monotonic();last_stamp=-1;count=0
                for stamp,tile in self.frame_source(self.path):
                    if stamp<last_stamp:raise ValueError("Recorded timestamps moved backwards")
                    if self._stop.wait(max(0,started+stamp-time.monotonic())):return
                    with self._lock:
                        if self._pending is not None:self._stats["superseded"]+=1
                        self._pending=tile
                        self._stats.update(state="playing",time_s=stamp)
                        self._stats["emitted"]+=1
                    last_stamp=stamp;count+=1
                if not count:raise ValueError("Video has no decoded frames")
                if not self.loop:
                    with self._lock:self._stats["state"]="ended"
                    return
                if self._stop.wait(max(0,started+last_stamp+1/self.fps-time.monotonic())):return
                with self._lock:self._stats["loops"]+=1
        except Exception as exc:
            with self._lock:self._stats.update(state="error",error=str(exc))
        finally:
            if self._stop.is_set():
                with self._lock:self._stats["state"]="closed"
