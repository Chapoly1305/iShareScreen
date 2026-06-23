"""Opus encoder for the WebTransport audio path.

PyAV libopus wrapped to take 48 kHz stereo float32 PCM frames (the
shape `Session.set_audio_callback` delivers) and emit Annex-format
Opus packets. Tuned for low-latency screen-share:

  application=lowdelay   — Opus's "lowdelay" mode reorders nothing
                           and removes the ~2.5 ms lookahead vs the
                           default voip mode.
  frame_duration=5 ms    — minimum the libopus reference accepts in
                           lowdelay; 5 ms is a 240-sample frame at
                           48 kHz, well-suited for typing-feedback.
  bitrate=64000          — transparent for stereo system audio per
                           Xiph listening tests; ~50% of voip-typical.

Typical encode time per packet: ~0.3 ms on Apple Silicon. Negligible
vs the 5 ms framing cost itself.
"""
from __future__ import annotations

import logging
from fractions import Fraction
from typing import Optional

import av
import numpy as np


log = logging.getLogger(__name__)


_OPUS_SAMPLE_RATE = 48000
_OPUS_FRAME_SAMPLES = 240   # 5 ms @ 48 kHz


class OpusEncoder:
    """Stateful libopus encoder. Construct once, call `encode(pcm)`
    with arbitrary-length stereo float32 PCM. Returns a list of opus
    packet bytes (one per Opus frame; usually one per `encode` call
    because the source produces ≥5 ms blocks)."""

    def __init__(self, *, bitrate_bps: int = 64000) -> None:
        # PyAV's raw `CodecContext.create('libopus', 'w')` path returns
        # EINVAL from `avcodec_open2` even with all expected fields
        # set — same configuration via the container API works. So we
        # construct a stream attached to a discardable container; the
        # encoder packets are still accessible via `stream.encode()`.
        import io
        self._container = av.open(io.BytesIO(), "w", format="ogg")
        stream = self._container.add_stream("libopus", rate=_OPUS_SAMPLE_RATE)
        codec = stream.codec_context
        codec.layout = "stereo"
        codec.bit_rate = bitrate_bps
        codec.time_base = Fraction(1, _OPUS_SAMPLE_RATE)
        codec.options = {
            "application": "lowdelay",
            "frame_duration": "5",
        }
        # Container API opens the encoder lazily on first encode().
        self._stream = stream
        self._codec = codec
        self._pts = 0
        # libopus only accepts exact frame-size inputs (240 / 480 / 960 /
        # 1920 / 2880 samples). Apple's AAC-ELD-SBR decode delivers
        # blocks that don't always divide evenly, so buffer leftover.
        self._buf = np.empty((0, 2), dtype=np.float32)
        log.info(
            "libopus low-latency: 48kHz stereo %d kbps lowdelay 5ms framing",
            bitrate_bps // 1000,
        )

    def encode(self, pcm: np.ndarray) -> list[tuple[bytes, int]]:
        """Encode a stereo float32 PCM block of arbitrary length.
        Returns a list of `(opus_bytes, pts_us)` for each 5 ms frame."""
        if pcm.size == 0:
            return []
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)
        if pcm.ndim != 2 or pcm.shape[1] != 2:
            raise ValueError(f"expected (N, 2) stereo, got {pcm.shape}")
        # Append to leftover buffer.
        self._buf = np.concatenate([self._buf, pcm])
        out: list[tuple[bytes, int]] = []
        while self._buf.shape[0] >= _OPUS_FRAME_SAMPLES:
            chunk = self._buf[:_OPUS_FRAME_SAMPLES]
            self._buf = self._buf[_OPUS_FRAME_SAMPLES:]
            # PyAV expects planar fltp = (channels, samples).
            planar = np.ascontiguousarray(chunk.T)
            frame = av.AudioFrame.from_ndarray(
                planar, format="fltp", layout="stereo",
            )
            frame.sample_rate = _OPUS_SAMPLE_RATE
            frame.pts = self._pts
            self._pts += _OPUS_FRAME_SAMPLES
            for pkt in self._stream.encode(frame):
                ts = pkt.pts if pkt.pts is not None else self._pts
                ts_us = ts * 1_000_000 // _OPUS_SAMPLE_RATE
                # libopus's first packet carries a negative pts from
                # the encoder's lookahead. Clamp to 0; the relative
                # spacing across packets is what the browser uses for
                # scheduling, not the absolute value.
                if ts_us < 0:
                    ts_us = 0
                out.append((bytes(pkt), ts_us))
        return out

    def close(self) -> None:
        try:
            list(self._stream.encode(None))
        except Exception:
            pass
        try:
            self._container.close()
        except Exception:
            pass


__all__ = ["OpusEncoder"]
