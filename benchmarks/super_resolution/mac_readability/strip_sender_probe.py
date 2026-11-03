"""Actual Mac lossless420 strip costs; fixed parameters, no OCR-guided routing."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

try:
    from .strip_transport import (
        StripReceiver,
        frame_strips,
        planes_nv12,
        encode_plane,
        wire_bytes,
    )
    from .detail_sender_probe import sha, save
except ImportError:
    from strip_transport import (
        StripReceiver,
        frame_strips,
        planes_nv12,
        encode_plane,
        wire_bytes,
    )
    from detail_sender_probe import sha, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controls", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    controls = json.loads((a.controls / "MANIFEST.json").read_text())
    assert controls["complete"]
    rows = [
        x
        for x in controls["images"]
        if x["height"] == 1080 and x["stage"] == "420-no-codec"
    ]
    assert len(rows) == 32
    a.output.mkdir(exist_ok=False)
    result = dict(
        complete=False,
        scope=__doc__,
        started=time.time(),
        platform=platform.platform(),
        machine=platform.machine(),
        source_sha256=controls["source_sha256"],
        controls_sha256=sha(a.controls / "MANIFEST.json"),
        scripts={
            name: sha(Path(__file__).with_name(name))
            for name in [
                "strip_sender_probe.py",
                "strip_transport.py",
                "detail_transport.py",
                "detail_sender_probe.py",
            ]
        },
        parameters=dict(
            source="actualMacnative1080limitedBT709nv12",
            target_strip_height=64,
            min_height=32,
            max_height=96,
            boundary="minimum source horizontal contrast across4rows, nearest64 tie break; even420 coordinates",
            codec="separate-plane zlib6, shortest none/left/up bytefilter",
            priority="luma edgeenergy/wirebyte",
            quality_budgets_wire_bytes=[15000, 60000],
            stability_wait_ms=100,
            scope="same two budgets as tiles; fixed-strips andmonolithic controls arebytecostonly; noOCR/categoryinput",
        ),
        images=[],
        test_used=False,
    )
    save(a.output / "RESULTS.json", result)
    for row in rows:
        path = a.controls / row["path"]
        assert sha(path) == row["sha256"]
        raw = path.read_bytes()
        start = time.perf_counter()
        tiles = frame_strips(raw, row["width"], row["height"], row["state_id"])
        encode_ms = (time.perf_counter() - start) * 1000
        receiver = StripReceiver(row["width"], row["height"])
        identity = hashlib.sha256(raw).digest()[:16]
        receiver.begin(row["state_id"], identity)
        folder = a.output / row["id"]
        folder.mkdir()
        metadata = []
        for i, tile in enumerate(tiles):
            packet = tile.pop("packet")
            dest = folder / f"{i:03d}.strip"
            dest.write_bytes(packet)
            tile.update(
                path=str(dest.relative_to(a.output)),
                sha256=sha(dest),
                payload_bytes=len(packet),
                wire_bytes=wire_bytes(len(packet)),
            )
            metadata.append(tile)
            assert receiver.accept(packet)
        assert receiver.mask.all() and all(
            np.array_equal(x, y)
            for x, y in zip(
                receiver.planes, planes_nv12(raw, row["width"], row["height"])
            )
        )
        start = time.perf_counter()
        fixed = frame_strips(
            raw, row["width"], row["height"], row["state_id"], aligned=False
        )
        fixed_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        mono = b"".join(
            encode_plane(p) for p in planes_nv12(raw, row["width"], row["height"])
        )
        mono_ms = (time.perf_counter() - start) * 1000
        total = wire_bytes(32) + sum(t["wire_bytes"] for t in metadata)
        result["images"].append(
            {
                k: row[k]
                for k in ["id", "group", "category", "state_id", "width", "height"]
            }
            | dict(
                raw_sha256=row["sha256"],
                identity=identity.hex(),
                tiles=metadata,
                encode_ms=encode_ms,
                wire_bytes=total,
                exact_420_roundtrip=True,
                controls=dict(
                    fixed_strips_wire_bytes=wire_bytes(32)
                    + sum(wire_bytes(len(t["packet"])) for t in fixed),
                    fixed_strips_encode_ms=fixed_ms,
                    monolithic_wire_bytes=wire_bytes(32) + wire_bytes(36 + len(mono)),
                    monolithic_encode_ms=mono_ms,
                ),
            )
        )
        save(a.output / "RESULTS.json", result)
        print(row["state_id"], total, round(encode_ms, 1), flush=True)
    result.update(complete=True, finished=time.time())
    save(a.output / "RESULTS.json", result)


if __name__ == "__main__":
    main()
