"""WebTransport + WebCodecs bridge.

Lowest-latency browser frontend. Replaces aiortc/WebRTC with raw QUIC
over HTTP/3, lifting the WebRTC playout-buffer floor (~30 ms in tuned
Chrome) down to ~5 ms. Browsers receive H.264 NALUs as unidirectional
streams and decode via WebCodecs `VideoDecoder` straight to a canvas.

Architecture:
    aioquic HTTP/3 server
        GET /                    → static/index.html
        WebTransport CONNECT /wt → per-viewer session
            unidirectional streams (server → client) — H.264 NALUs
            bidirectional stream  (client ↔ server) — JSON input events

One iss `Session` is shared across viewers. The encoder runs once per
frame (not per viewer); encoded bytes fan out to every active viewer.
"""
from .server import WebTransportBridge, run

__all__ = ["WebTransportBridge", "run"]
