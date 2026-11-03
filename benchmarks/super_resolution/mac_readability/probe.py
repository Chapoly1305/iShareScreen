"""Bounded Mac-local Screen Sharing capture using native Keychain login.

Launch in the logged-in Mac GUI session. Socket PCAP framing is synthetic;
this probe does not claim a physical network cap or atomic cross-tile frames.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import signal
import socket
import struct
import subprocess
import time

import numpy as np
import Quartz

from isharescreen.credentials import get_password
from isharescreen.proxy.config import SessionConfig
from isharescreen.proxy.protocol.media_config import (
    MediaOfferConfig,
    PixelFormat,
    ScreenFormat,
)
from isharescreen.proxy.protocol.negotiation import AdvertiseDims
from isharescreen.proxy.session import Session


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def counters(session):
    from recovery_snapshot import snapshot

    received, lost = session.packet_stats
    decoder = session._decoder
    return dict(
        video_udp_bytes=session._rx_bytes_video,
        ctrl_udp_bytes=session._rx_bytes_ctrl,
        audio_subset_of_ctrl_bytes=session._rx_bytes_audio,
        rtcp_subset_of_ctrl_bytes=session._rx_bytes_rtcp,
        received_packets=received,
        lost_packets=lost,
        video_queue_depth=session._video_q.qsize(),
        control_queue_depth=session._ctrl_q.qsize(),
        video_queue_dropped=session._video_q_dropped,
        decode_latency_ms=getattr(decoder, "decode_latency_ms", None),
        decoder_good_counts=list(getattr(decoder, "good_counts", [])),
        pending_groups=len(session._pending_groups),
        recovery=snapshot(session),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=12)
    p.add_argument("--kbps", type=int, default=600)
    p.add_argument("--height", type=int, choices=[720, 1080], default=720)
    p.add_argument("--codec", choices=["hevc", "avc"], default="hevc")
    p.add_argument("--native-canvas", action="store_true")
    p.add_argument("--restrict-format", action="store_true")
    p.add_argument("--curtain", action="store_true")
    p.add_argument("--default-bitrate", action="store_true")
    p.add_argument(
        "--local-bind",
        help="Separate the local client address from the local Screen Sharing server",
    )
    p.add_argument("--stimulus", action="store_true")
    p.add_argument(
        "--no-60fps",
        action="store_true",
        help="Experimental offer: clear both 60-fps stream capability bits; verify actual output",
    )
    p.add_argument(
        "--native-420",
        action="store_true",
        help="Experimental Mac decode: preserve full-range NV12 instead of expanding source 420 to 444",
    )
    p.add_argument(
        "--snapshot-interval",
        type=float,
        default=0.5,
        help="Seconds between raw tile saves; zero disables saves for lightweight timing",
    )
    p.add_argument(
        "--link-kbps",
        type=int,
        default=0,
        help="Optional per-direction socket-boundary link cap, including modeled IP overhead",
    )
    p.add_argument("--link-queue-ms", type=float, default=250)
    p.add_argument("--link-delay-ms", type=float, default=10)
    args = p.parse_args()
    if (
        not 1 <= args.seconds <= 120
        or not 1 <= args.kbps <= 6000
        or not 0 <= args.snapshot_interval <= 5
    ):
        raise SystemExit("Probe duration/bitrate out of bounds")
    if (
        not 0 <= args.link_kbps <= 6000
        or not 1 <= args.link_queue_ms <= 2000
        or not 0 <= args.link_delay_ms <= 200
    ):
        raise SystemExit("Link parameters out of bounds")
    os.umask(0o077)
    os.environ["ISS_CLIP"] = "0"
    if args.native_420:
        if args.codec != "hevc":
            raise ValueError("Native 420 diagnostic is HEVC-only")
        from isharescreen.proxy.media import vtdecode

        if hasattr(vtdecode, "_NV12_FMT"):
            raise ValueError(
                "Native 420 is now automatic; the prototype flag requires the preserved baseline runtime"
            )
        from isharescreen.proxy.media.bitstream import (
            BitReader,
            remove_emulation_prevention,
        )
        from isharescreen.proxy.media.hevc_rps import _skip_profile_tier_level

        original_params = vtdecode.VTHevcDecoder.set_params

        def require_420(self, vps, sps, pps):
            br = BitReader(remove_emulation_prevention(sps[2:]))
            br.read(4)
            layers = br.read(3)
            br.read1()
            _skip_profile_tier_level(br, layers + 1)
            br.read_ue()
            if br.read_ue() != 1:
                raise ValueError(
                    "Native 420 diagnostic requires actual HEVC 420 source"
                )
            return original_params(self, vps, sps, pps)

        original_convert = vtdecode.VTHevcDecoder._cvbuf_to_tile

        def convert_420(self, buf):
            tile = original_convert(self, buf)
            return (
                dataclasses.replace(
                    tile, chroma_width=tile.width // 2, chroma_height=tile.height // 2
                )
                if tile
                else None
            )

        vtdecode._NV24_FMT = 0x34323066  # 420f; existing native guard now enforces this requested format.
        vtdecode.VTHevcDecoder.set_params = require_420
        vtdecode.VTHevcDecoder._cvbuf_to_tile = convert_420
    if args.no_60fps:
        from isharescreen.proxy.protocol import negotiation

        original_0x1c = negotiation.build_0x1c

        def build_without_60fps(*positional, **options):
            message = bytearray(original_0x1c(*positional, **options))
            flags = struct.unpack_from(">I", message, 6)[0]
            struct.pack_into(">I", message, 6, flags & ~3)
            return bytes(message)

        negotiation.build_0x1c = build_without_60fps
    if args.local_bind:
        local_ip = subprocess.check_output(
            ["/usr/sbin/ipconfig", "getifaddr", "en0"], text=True
        ).strip()
        if args.local_bind != local_ip:
            raise SystemExit(
                "Local capture bind address must match this Mac's en0 address"
            )
        original_create_connection = socket.create_connection

        def bound_connection(address, *positional, **options):
            if address == ("127.0.0.1", 5900):
                options["source_address"] = (args.local_bind, 0)
            return original_create_connection(address, *positional, **options)

        socket.create_connection = bound_connection
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "runner.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "recovery-snapshot.py").write_bytes(
        Path(__file__).with_name("recovery_snapshot.py").read_bytes()
    )
    if args.link_kbps:
        (args.output / "link-runner.py").write_bytes(
            Path(__file__).with_name("link.py").read_bytes()
        )
    if args.stimulus:
        (args.output / "stimulus-runner.py").write_bytes(
            Path(__file__).with_name("scene.py").read_bytes()
        )
    logging.basicConfig(level=logging.WARNING)
    result = dict(
        schema="mac_screen_sharing_capture_v2",
        started_unix=time.time(),
        state="initializing",
        platform=platform.platform(),
        pid=os.getpid(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        offered=dict(
            kbps=None if args.default_bitrate else args.kbps,
            height=args.height,
            width=args.height * 16 // 9,
            chroma="420",
            range="full_or_video",
            audio=False,
            curtain=args.curtain,
            codec=args.codec,
            custom_dimensions_requested=not args.native_canvas,
        ),
        samples=[],
        snapshots=[],
        tile_events=[],
        marker_events=[],
        local_bind=args.local_bind,
        snapshot_interval=args.snapshot_interval,
        experimental_60fps_capabilities_cleared=args.no_60fps,
        experimental_native_420_decode=args.native_420,
        scope="Mac-local encoder/decoder. Latest per-tile states are not atomic frames. PCAP uses synthetic IP/TCP framing; no physical WAN claim.",
    )
    status = args.output / "STATUS.json"
    write(status, result)
    session = stimulus = awake = None
    link = None
    restore_connection = restore_udp = None
    password = None

    def interrupted(*unused):
        raise TimeoutError("Capture stopped by its bounded-run watchdog")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if (Quartz.CGSessionCopyCurrentDictionary() or {}).get(
            "CGSSessionScreenIsLocked"
        ):
            raise RuntimeError(
                "Mac is locked; unlock locally before a controlled screen capture"
            )
        awake = subprocess.Popen(["/usr/bin/caffeinate", "-di", "-w", str(os.getpid())])
        password = get_password("127.0.0.1", 5900, "alex")
        if not password:
            raise RuntimeError("No locally saved Screen Sharing credential")
        if args.link_kbps:
            from link import SocketLink

            link = SocketLink(
                args.link_kbps * 1000, args.link_queue_ms, args.link_delay_ms
            )
            restore_connection = socket.create_connection
            restore_udp = Session._bind_udp

            def shaped_connection(address, *positional, **options):
                connected = restore_connection(address, *positional, **options)
                return (
                    link.wrap_tcp(connected)
                    if address == ("127.0.0.1", 5900)
                    else connected
                )

            def shaped_udp(host, port):
                return link.wrap_udp(restore_udp(host, port))

            socket.create_connection = shaped_connection
            Session._bind_udp = staticmethod(shaped_udp)
            result["link_emulation"] = dict(
                link.config,
                runner_sha256=hashlib.sha256(
                    Path(__file__).with_name("link.py").read_bytes()
                ).hexdigest(),
            )
        media = MediaOfferConfig.from_environment()
        media.video.codec = args.codec
        media.video.ltrp_enabled = args.codec == "hevc"
        media.video.tiles_per_frame = 4 if args.codec == "hevc" else 1
        media.video.pixel_formats = (
            PixelFormat.YUV420_8_FULL | PixelFormat.YUV420_8_VIDEO
        )
        if args.restrict_format:
            chosen = (
                ScreenFormat.HD720_30 if args.height == 720 else ScreenFormat.HD1080_30
            )
            for payload in media.video.payloads:
                for rule in payload.rules:
                    rule.formats = rule.preferred_format = chosen
                    rule.formats_ext1 = rule.preferred_format_ext1 = None
        if not args.native_canvas:
            media.video.custom_width, media.video.custom_height = (
                args.height * 16 // 9,
                args.height,
            )
        if not args.default_bitrate:
            media.set_video_bitrate(args.kbps * 1000)
        media.validate()
        result["offer"] = dataclasses.asdict(media)
        config = SessionConfig(
            host="127.0.0.1",
            username="alex",
            password=password,
            curtain=args.curtain,
            audio=False,
            dynamic_resolution=False,
            media_offer=media,
            udp_bind_host=args.local_bind or "",
            advertise=AdvertiseDims(
                width=args.height * 16 // 9, height=args.height, hidpi_scale=1.0
            )
            if args.curtain
            else None,
            record_pcap=str(args.output / "session.pcap"),
        )
        session = Session(config)
        result["state"] = "connecting"
        write(status, result)
        session.connect()
        source_root = args.output / "runtime-sources"
        source_root.mkdir()
        import inspect
        from isharescreen.proxy.protocol import negotiation, media_config
        from isharescreen.proxy.media import hevc_rps

        runtime = {}
        for item in [
            Session,
            type(session._decoder),
            negotiation,
            media_config,
            hevc_rps,
        ]:
            file = Path(inspect.getfile(item))
            data = file.read_bytes()
            (source_root / file.name).write_bytes(data)
            runtime[file.name] = hashlib.sha256(data).hexdigest()
        result["runtime_source_sha256"] = runtime
        result.update(
            state="capturing",
            canvas=list(session.canvas_dims),
            num_tiles=session.num_tiles,
            decoder=session.hw_accel,
            connected_unix=time.time(),
            connected_monotonic_ns=time.monotonic_ns(),
        )
        decoder = session._decoder
        parameters = getattr(decoder, "_params", None)
        if parameters:
            vps, sps, pps = parameters
            nalus = (vps, sps) + tuple(pps)
        else:
            nalus = tuple(
                x
                for x in (
                    getattr(decoder, "_vps", None),
                    getattr(decoder, "_sps", None),
                    getattr(decoder, "_pps", None),
                )
                if x
            )
        if nalus:
            parameter_path = args.output / (
                "codec-parameters.hevc"
                if args.codec == "hevc"
                else "codec-parameters.h264"
            )
            parameter_path.write_bytes(b"".join(b"\x00\x00\x00\x01" + x for x in nalus))
            result["codec_parameters"] = dict(
                path=parameter_path.name,
                sha256=hashlib.sha256(parameter_path.read_bytes()).hexdigest(),
                description="Public video format parameter sets only; no session keys",
            )
            inspected = subprocess.run(
                [
                    "/opt/homebrew/bin/ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-of",
                    "json",
                    str(parameter_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            result["encoded_format_probe"] = (
                json.loads(inspected.stdout)
                if inspected.returncode == 0
                else {"error": inspected.stderr[-500:]}
            )
        write(status, result)
        if args.stimulus:
            stimulus = subprocess.Popen(
                [
                    os.sys.executable,
                    str(Path(__file__).with_name("scene.py")),
                    "--output",
                    str(args.output / "stimulus"),
                    "--seconds-per-state",
                    "3",
                ]
            )
        start = time.monotonic()
        latest, received_at = {}, {}
        sample_at = snapshot_at = status_at = start
        marker_geometry = None
        last_marker = None
        while time.monotonic() - start < args.seconds and session.is_connected:
            session.wait_for_fresh_tile(timeout=0.02)
            for i in range(session.num_tiles):
                tile = session.get_frame(i)
                if tile is not None:
                    latest[i], received_at[i] = tile, time.monotonic_ns()
                    result["tile_events"].append(dict(tile=i, time_ns=received_at[i]))
                    result.setdefault("actual_tile_dimensions", {})[str(i)] = dict(
                        width=tile.width,
                        height=tile.height,
                        y_stride=tile.y_stride,
                        chroma_width=tile.chroma_width,
                        chroma_height=tile.chroma_height,
                    )
            now = time.monotonic()
            if args.stimulus and marker_geometry is None:
                stimulus_manifest = args.output / "stimulus/manifest.json"
                if stimulus_manifest.exists():
                    source = json.loads(stimulus_manifest.read_text())
                    own = next(
                        (
                            w
                            for w in source.get("own_windows", [])
                            if w["number"] == source["window_id"]
                        ),
                        None,
                    )
                    if (
                        own is not None
                        and own["onscreen"]
                        and own["bounds"]["Width"] == 960
                        and own["bounds"]["Height"] == 540
                        and source.get("states")
                        and source["states"][0].get("reference")
                    ):
                        scale = source["backing_scale"]
                        bounds = own["bounds"]
                        marker_geometry = dict(
                            x=round(bounds["X"] * scale),
                            y=round(bounds["Y"] * scale),
                            scale=scale,
                        )
                        result["marker_geometry"] = marker_geometry
            if marker_geometry and len(latest) == session.num_tiles:
                values = []
                times = []
                g = marker_geometry
                slot = latest[0].height
                for offset_y in (12, 524):
                    y = g["y"] + round(offset_y * g["scale"])
                    index = y // slot
                    tile = latest.get(index)
                    if tile is None:
                        break
                    bits = []
                    for bit in range(8):
                        x = g["x"] + round((440 + bit * 10 + 4) * g["scale"])
                        bits.append(
                            tile.y[(y - index * slot) * tile.y_stride + x] > 128
                        )
                    values.append(sum(value << bit for bit, value in enumerate(bits)))
                    times.append(received_at[index])
                if len(values) == 2 and values != last_marker:
                    result["marker_events"].append(
                        dict(
                            markers=values,
                            observed_ns=time.monotonic_ns(),
                            tile_received_ns=times,
                        )
                    )
                    last_marker = values
            if now >= sample_at:
                if (Quartz.CGSessionCopyCurrentDictionary() or {}).get(
                    "CGSSessionScreenIsLocked"
                ):
                    raise RuntimeError(
                        "Mac locked during capture; this run cannot qualify for scene timing"
                    )
                result["samples"].append(
                    dict(time_ns=time.monotonic_ns(), **counters(session))
                )
                sample_at = now + 0.1
            if (
                args.snapshot_interval
                and now >= snapshot_at
                and len(latest) == session.num_tiles
            ):
                arrays, tiles = {}, []
                for i, tile in sorted(latest.items()):
                    meta = {
                        f.name: getattr(tile, f.name)
                        for f in dataclasses.fields(tile)
                        if f.name not in ("y", "u", "v")
                    }
                    for plane in ("y", "u", "v"):
                        data = getattr(tile, plane)
                        if data is not None:
                            arrays[f"tile_{i}_{plane}"] = np.frombuffer(data, np.uint8)
                    tiles.append(
                        dict(
                            index=i,
                            received_ns=received_at[i],
                            biplanar=tile.v is None,
                            **meta,
                        )
                    )
                name = f"tiles-{len(result['snapshots']):04d}.npz"
                np.savez_compressed(args.output / name, **arrays)
                result["snapshots"].append(
                    dict(
                        path=name,
                        sampled_ns=time.monotonic_ns(),
                        tiles=tiles,
                        observed_markers=last_marker,
                        marker_geometry=marker_geometry,
                    )
                )
                snapshot_at = now + args.snapshot_interval
            if now >= status_at:
                write(status, result)
                status_at = now + 1
        result.update(
            state="completed" if latest else "failed_no_decoded_tiles",
            elapsed_seconds=time.monotonic() - start,
            connected_at_end=session.is_connected,
            final_counters=counters(session),
        )
    except Exception as error:
        result.update(
            state="failed",
            error_type=type(error).__name__,
            error=str(error).replace(password, "<redacted>")
            if password
            else str(error),
        )
    finally:
        if stimulus is not None and stimulus.poll() is None:
            stimulus.terminate()
            try:
                stimulus.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stimulus.kill()
                stimulus.wait(timeout=5)
        if session is not None:
            try:
                session.close()
            except Exception as error:
                result["close_error_type"] = type(error).__name__
        if link is not None:
            link.close()
            write(args.output / "LINK.json", link.snapshot())
        if restore_connection is not None:
            socket.create_connection = restore_connection
        if restore_udp is not None:
            Session._bind_udp = staticmethod(restore_udp)
        if awake is not None:
            awake.terminate()
            awake.wait(timeout=5)
        result["finished_unix"] = time.time()
        write(status, result)
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ("state", "canvas", "decoder", "error_type", "error")
            }
        )
    )
    return 0 if result["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
