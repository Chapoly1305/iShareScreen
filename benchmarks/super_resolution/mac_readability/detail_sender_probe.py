"""Measure lossless 420 tile generation on the actual Mac, using frozen sources.

Offline development replay; no screen access, credentials, final-test data or
production transport. Both network wire costs and serialization times are
MODELED lower bounds, not end-to-end measurements.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

try:
    from .detail_transport import Receiver, frame_tiles, planes_nv12, wire_bytes
except ImportError:
    from detail_transport import Receiver, frame_tiles, planes_nv12, wire_bytes

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controls", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    controls = json.loads((a.controls / "MANIFEST.json").read_text())
    assert controls["complete"]
    rows = [
        r
        for r in controls["images"]
        if r["height"] == 1080 and r["stage"] == "420-no-codec"
    ]
    assert len(rows) == 32
    a.output.mkdir(exist_ok=False)
    result = dict(
        schema="source_detail_tiles_v1",
        scope=__doc__,
        complete=False,
        started=time.time(),
        platform=platform.platform(),
        machine=platform.machine(),
        python=platform.python_version(),
        numpy=np.__version__,
        sender_script_sha256=sha(Path(__file__)),
        codec_script_sha256=sha(Path(__file__).with_name("detail_transport.py")),
        controls_sha256=sha(a.controls / "MANIFEST.json"),
        source_sha256=controls["source_sha256"],
        parameters=dict(
            tile_size=256,
            zlib_level=6,
            filtering="shortest of none/left/up, separately per 420 plane",
            priority="source luma edge energy / modeled wire bytes",
            stability_wait_ms=100,
            quality_budgets_wire_bytes=[15000, 60000],
            limits="No OCR/category-based routing. Base-video bytes must be added; quality byte budgets are not time deadlines.",
        ),
        wire_model="1200-byte payload fragments + 16-byte app framing + 28-byte IPv4/UDP; one 32-byte revision announcement. Excludes VPN, retransmission, ACK and encryption.",
        test_used=False,
        images=[],
    )
    save(a.output / "RESULTS.json", result)
    previous = {}
    for row in rows:
        path = a.controls / row["path"]
        assert sha(path) == row["sha256"]
        raw = path.read_bytes()
        start = time.perf_counter()
        tiles = frame_tiles(raw, row["width"], row["height"], row["state_id"])
        encode_ms = 1000 * (time.perf_counter() - start)
        identity = hashlib.sha256(raw).digest()[:16]
        receiver = Receiver(row["width"], row["height"])
        receiver.begin(row["state_id"], identity)
        metadata = []
        output = a.output / row["id"]
        output.mkdir()
        for index, tile in enumerate(tiles):
            packet = tile.pop("packet")
            dest = output / f"{index:03d}.tile"
            dest.write_bytes(packet)
            tile.update(
                path=str(dest.relative_to(a.output)),
                sha256=sha(dest),
                payload_bytes=len(packet),
                wire_bytes=wire_bytes(len(packet)),
            )
            metadata.append(tile)
            assert receiver.accept(packet)
        assert receiver.mask.all()
        assert all(
            np.array_equal(a, b)
            for a, b in zip(
                receiver.planes, planes_nv12(raw, row["width"], row["height"])
            )
        )
        assert (
            frame_tiles(
                raw,
                row["width"],
                row["height"],
                row["state_id"] + 1000,
                previous_raw=raw,
            )
            == []
        )
        changed = None
        if row["group"] in previous:
            changed_tiles = frame_tiles(
                raw,
                row["width"],
                row["height"],
                row["state_id"],
                previous_raw=previous[row["group"]],
            )
            changed = dict(
                tiles=len(changed_tiles),
                wire_bytes=wire_bytes(32)
                + sum(wire_bytes(len(t["packet"])) for t in changed_tiles),
                scope="Encoder-only cache eligibility; receiver cache restoration not implemented",
            )
        previous[row["group"]] = raw
        total = wire_bytes(32) + sum(t["wire_bytes"] for t in metadata)
        result["images"].append(
            {
                k: row[k]
                for k in ("id", "group", "category", "state_id", "width", "height")
            }
            | dict(
                raw_sha256=row["sha256"],
                identity=identity.hex(),
                tiles=metadata,
                encode_ms=encode_ms,
                wire_bytes=total,
                serialization_floor_full600_ms=total * 8 / 600,
                serialization_if_detail300_ms=total * 8 / 300,
                changed_from_previous_group=changed,
                exact_420_roundtrip=True,
                repeated_identical_source_changed_tiles=0,
            )
        )
        save(a.output / "RESULTS.json", result)
        print(row["id"], total, "bytes", round(encode_ms, 2), "ms", flush=True)
    result.update(complete=True, finished=time.time())
    save(a.output / "RESULTS.json", result)


if __name__ == "__main__":
    main()
