"""Compare native 420 and expanded 444 decoding of identical Mac HEVC bytes.

This checks luma preservation and output sampling, not a new chroma encoding
sweep. Both decoder paths consume the same existing 4:2:0 elementary stream.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from isharescreen.proxy.media.vtdecode import VTHevcDecoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = subprocess.check_output(
        [
            "/opt/homebrew/bin/ffmpeg",
            "-v",
            "error",
            "-i",
            str(args.video),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-bsf:v",
            "hevc_mp4toannexb",
            "-f",
            "hevc",
            "pipe:1",
        ]
    )
    nalus = [n for n in re.split(b"\x00\x00\x00?\x01", raw) if n]
    vps = next(n for n in nalus if (n[0] >> 1) & 63 == 32)
    sps = next(n for n in nalus if (n[0] >> 1) & 63 == 33)
    pps = next(n for n in nalus if (n[0] >> 1) & 63 == 34)
    wanted = {0, 1, 2, 6, 30, 60, 89, 90, 91, 96, 120, 179}
    reference = {}
    records = []
    result = dict(
        scope=__doc__,
        source_sha256=hashlib.sha256(args.video.read_bytes()).hexdigest(),
        stream_sha256=hashlib.sha256(raw).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runs=[],
    )
    for mode in ["expanded444", "native420"]:
        decoder = VTHevcDecoder(1)
        decoder.set_params(vps, sps, {0: pps})
        if mode == "expanded444":
            with decoder._lock:
                decoder._output_format = 0x34343466
                decoder._build_session_locked()
        count = 0
        try:
            for nalu in nalus:
                if (nalu[0] >> 1) & 63 >= 32:
                    continue
                decoder.feed_nalu(nalu, 0)
                frame = decoder.get_frame(0)
                if frame is None:
                    raise RuntimeError(f"Missing decoded frame: {mode} {count}")
                if count in wanted:
                    y = (
                        np.frombuffer(frame.y, np.uint8)
                        .reshape(frame.height, frame.y_stride)[:, : frame.width]
                        .copy()
                    )
                    if mode == "expanded444":
                        reference[count] = y
                    else:
                        delta = np.abs(
                            y.astype(np.int16) - reference[count].astype(np.int16)
                        )
                        records.append(
                            dict(
                                frame=count,
                                luma_max_abs_difference=int(delta.max()),
                                luma_mean_abs_difference=float(delta.mean()),
                                width=frame.width,
                                height=frame.height,
                                chroma_width=frame.chroma_width,
                                chroma_height=frame.chroma_height,
                            )
                        )
                count += 1
            result["runs"].append(dict(mode=mode, decoded_frames=count))
        finally:
            decoder.close()
    assert len(records) == len(wanted) and all(
        r["luma_max_abs_difference"] == 0 for r in records
    )
    result.update(luma_checks=records, complete=True)
    (args.output / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Both paths decoded",
        count,
        "identical-source frames;",
        len(records),
        "luma comparisons are byte-identical",
    )


if __name__ == "__main__":
    main()
