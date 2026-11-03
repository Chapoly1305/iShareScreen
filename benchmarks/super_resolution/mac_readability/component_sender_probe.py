"""Actual M1 whole-frame exact-pattern encoding/decoding; frozen420 sources only."""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import time

import cv2
import numpy as np

try:
    from .component_codec import encode, decode
    from .detail_sender_probe import sha, save
    from .detail_transport import wire_bytes
except ImportError:
    from component_codec import encode, decode
    from detail_sender_probe import sha, save
    from detail_transport import wire_bytes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controls", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cv2.setNumThreads(1)
    d = json.loads((a.controls / "MANIFEST.json").read_text())
    assert d["complete"]
    frames = [
        x for x in d["images"] if x["height"] == 1080 and x["stage"] == "420-no-codec"
    ]
    assert len(frames) == 32
    a.output.mkdir(exist_ok=False)
    result = dict(
        complete=False,
        scope=__doc__,
        platform=platform.platform(),
        machine=platform.machine(),
        hardware=subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        numpy=np.__version__,
        opencv=cv2.__version__,
        started=time.time(),
        source_sha256=d["source_sha256"],
        controls_sha256=sha(a.controls / "MANIFEST.json"),
        scripts={
            name: sha(Path(__file__).with_name(name))
            for name in [
                "component_codec.py",
                "component_sender_probe.py",
                "detail_transport.py",
                "detail_sender_probe.py",
            ]
        },
        protocol="Mostfrequentluma background;8connected components; dictionary reuse only for byte-identical dimensioned luma blocks; positions+dictionary zlib6; UV lossless420; fullNV12 SHA256 checked after decode",
        limits=[
            "Offlinecodec timing, no sourcecapture/network/GPUdisplay",
            "32 single-shotframes; includesfirstcoldencode; fileI/Oexcluded",
            "Wholeframecompletion needed beforedisplay; no progressivecomponentdisplay orcrossframecache",
        ],
        images=[],
        test_used=False,
    )
    save(a.output / "RESULTS.json", result)
    for v in frames:
        path = a.controls / v["path"]
        assert sha(path) == v["sha256"]
        raw = path.read_bytes()
        start = time.perf_counter()
        packet, counts = encode(raw, 1920, 1080)
        encode_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        actual = decode(packet)
        decode_ms = (time.perf_counter() - start) * 1000
        assert actual == raw
        dest = a.output / (v["id"] + ".isc")
        dest.write_bytes(packet)
        result["images"].append(
            {
                k: v[k]
                for k in ["id", "group", "category", "state_id", "width", "height"]
            }
            | dict(
                raw_sha256=v["sha256"],
                path=dest.name,
                sha256=sha(dest),
                payload_bytes=len(packet),
                wire_bytes=wire_bytes(32) + wire_bytes(len(packet)),
                encode_ms=encode_ms,
                decode_ms=decode_ms,
                exact_raw=True,
                **counts,
            )
        )
        save(a.output / "RESULTS.json", result)
        print(
            v["state_id"],
            result["images"][-1]["wire_bytes"],
            round(encode_ms, 1),
            round(decode_ms, 1),
            flush=True,
        )
    result.update(complete=True, finished=time.time())
    save(a.output / "RESULTS.json", result)


if __name__ == "__main__":
    main()
