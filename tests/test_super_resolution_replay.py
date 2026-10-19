"""Recorded-video timing/queue checks plus the production HEVC tile extractor."""
from fractions import Fraction
import time

import numpy as np
import pytest

av=pytest.importorskip("av")
from benchmarks.super_resolution.replay import VideoReplay, video_tiles


def test_replay_keeps_only_latest_tile_and_closes():
    def frames(path):
        for i in range(5):yield 0.,i
    source=VideoReplay("fixture.mp4",frame_source=frames,fps=60)
    try:
        source._thread.join(timeout=1)
        assert not source._thread.is_alive()
        assert source.snapshot()["state"]=="ended"
        assert source.snapshot()["emitted"]==5 and source.snapshot()["superseded"]==4
        assert source.snapshot()["pending"]==1
        assert source.poll()==4 and source.poll() is None
        assert source.snapshot()["delivered"]==1
    finally:source.close()
    assert source.snapshot()["state"]=="closed"


def test_replay_close_interrupts_future_frame_deadline():
    def frames(path):
        yield 0.,"first"
        yield 100.,"future"
    source=VideoReplay("fixture.mp4",frame_source=frames,fps=60)
    deadline=time.monotonic()+1
    while not source.snapshot()["emitted"] and time.monotonic()<deadline:time.sleep(.005)
    assert source.snapshot()["emitted"]==1
    source.close()
    assert not source._thread.is_alive() and source.poll() is None


def test_replay_rejects_backward_timestamps():
    def frames(path):
        yield 0.,"first"
        yield -.1,"bad"
    source=VideoReplay("fixture.mp4",frame_source=frames,fps=60)
    try:
        source._thread.join(timeout=1)
        assert source.snapshot()["state"]=="error"
        assert "backwards" in source.snapshot()["error"]
    finally:source.close()


def test_recorded_hevc_is_extracted_as_full709_native_tiles(tmp_path):
    path=tmp_path/"recording.mp4"
    with av.open(str(path),"w") as container:
        stream=container.add_stream("libx265",rate=60)
        stream.width=64;stream.height=64;stream.pix_fmt="yuv444p"
        stream.codec_context.color_range=2;stream.codec_context.colorspace=1
        stream.options={"x265-params":"pools=1:frame-threads=1:log-level=error:range=full:colormatrix=bt709:colorprim=bt709:transfer=bt709"}
        for i,value in enumerate([20,90,180]):
            frame=av.VideoFrame.from_ndarray(np.full((64,64,3),value,np.uint8),format="rgb24")
            frame.pts=i;frame.time_base=Fraction(1,60)
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    frames=list(video_tiles(path))
    assert len(frames)==3
    for i,(stamp,tile) in enumerate(frames):
        assert stamp==pytest.approx(i/60,abs=.001)
        assert (tile.width,tile.height)==(64,64) and tile.is_yuv444
    assert frames[0][1].y!=frames[-1][1].y
