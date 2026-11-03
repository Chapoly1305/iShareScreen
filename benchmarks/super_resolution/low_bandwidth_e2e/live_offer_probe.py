"""Measure a live Screen Sharing offer without storing the login password.

This is a smoke probe. It records socket payload and frame event counts, not
network-layer overhead or human text accuracy. Run from the Mac against its
own Screen Sharing service when the remote client cannot reach its VPN address.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from isharescreen import __version__ as isharescreen_version
from isharescreen.proxy.protocol.media_config import MediaOfferConfig
from isharescreen.proxy.protocol.negotiation import AdvertiseDims
from isharescreen.proxy.session import Session, SessionConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--user", required=True)
    parser.add_argument(
        "--cap-kbps",
        type=int,
        default=None,
        help="experimental offered desktop video cap",
    )
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--warmup-seconds", type=int, default=5)
    parser.add_argument(
        "--curtain",
        action="store_true",
        help="request a private 1080p virtual display; blanks the Mac screen",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _counts(session: Session) -> dict[str, int]:
    received, lost = session.packet_stats
    return {
        "video_socket_bytes": session._rx_bytes_video,
        "control_socket_bytes": session._rx_bytes_ctrl,
        "audio_socket_bytes": session._rx_bytes_audio,
        "rtcp_socket_bytes": session._rx_bytes_rtcp,
        "video_packets": session._rx_pkts_video,
        "received_packets": received,
        "lost_packets": lost,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.cap_kbps is not None and not 1 <= args.cap_kbps <= 4_294_967:
        raise SystemExit("--cap-kbps must be between 1 and 4294967")
    if args.seconds < 1 or args.warmup_seconds < 0:
        raise SystemExit("duration must be positive and warmup nonnegative")
    password = getpass.getpass("Mac Screen Sharing password: ")
    media = MediaOfferConfig.from_environment()
    media.video.codec = "hevc"
    if args.cap_kbps is not None:
        media.set_video_bitrate(args.cap_kbps * 1000)
    media.validate()
    config = SessionConfig(
        host=args.host,
        username=args.user,
        password=password,
        audio=False,
        curtain=args.curtain,
        dynamic_resolution=False,
        advertise=AdvertiseDims(width=1920, height=1080, hidpi_scale=1.0),
        media_offer=media,
    )
    del password
    with Session(config) as session:
        print(
            "connected",
            "canvas",
            session.canvas_dims,
            "server",
            session.server_dims,
            "decoder",
            session.hw_accel,
            flush=True,
        )
        warmup_end = time.monotonic() + args.warmup_seconds
        while time.monotonic() < warmup_end and session.is_connected:
            session.wait_for_fresh_tile(timeout=0.25)
        start_counts = _counts(session)
        start = time.monotonic()
        event_count = 0
        while time.monotonic() - start < args.seconds and session.is_connected:
            if session.wait_for_fresh_tile(timeout=0.25):
                event_count += 1
        elapsed = time.monotonic() - start
        end_counts = _counts(session)
        delta = {name: end_counts[name] - start_counts[name] for name in end_counts}
        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "host": args.host,
            "client_system": platform.platform(),
            "python_version": platform.python_version(),
            "isharescreen_version": isharescreen_version,
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "user": args.user,
            "offered_cap_kbps": args.cap_kbps,
            "curtain": args.curtain,
            "requested_canvas": [1920, 1080, 1.0],
            "actual_canvas": list(session.canvas_dims),
            "server_dims": list(session.server_dims),
            "decoder": session.hw_accel,
            "connected_at_end": session.is_connected,
            "elapsed_seconds": elapsed,
            "fresh_tile_events": event_count,
            "socket_deltas": delta,
            "video_socket_payload_kbps": delta["video_socket_bytes"]
            * 8
            / elapsed
            / 1000,
            "all_socket_payload_kbps": sum(
                delta[k]
                for k in (
                    "video_socket_bytes",
                    "control_socket_bytes",
                    "audio_socket_bytes",
                    "rtcp_socket_bytes",
                )
            )
            * 8
            / elapsed
            / 1000,
            "limits": "Socket payload excludes IP/UDP/TCP overhead; local loopback is not a weak-network test. Fresh tile events are not exact displayed frames.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["connected_at_end"] and event_count else 2


if __name__ == "__main__":
    raise SystemExit(main())
