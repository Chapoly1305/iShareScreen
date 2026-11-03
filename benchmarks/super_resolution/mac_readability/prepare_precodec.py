"""Convert Mac raw controls through the same planar420 RGB path as codec scoring."""

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

from benchmarks.super_resolution.mac_readability.quality import markers, save, sha


def planar_nv12(raw, width, height):
    flat = np.frombuffer(raw, np.uint8)
    n = width * height
    assert len(flat) == n * 3 // 2
    return np.concatenate((flat[:n], flat[n::2], flat[n + 1 :: 2])).reshape(
        height * 3 // 2, width
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controls", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--codec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    controls = json.loads((a.controls / "MANIFEST.json").read_text())
    source = json.loads((a.source / "manifest.json").read_text())
    assert controls["complete"] and controls["source_sha256"] == sha(
        a.source / "manifest.json"
    )
    # Verify byte-layout conversion against an existing decoded frame. Using the
    # planar representation avoids a different NV12-to-RGB swscale kernel.
    codec = json.loads((a.codec / "MANIFEST.json").read_text())
    run = codec["runs"][0]
    with av.open(a.codec / run["path"]) as container:
        f = next(container.decode(video=0))
        flat = f.to_ndarray().ravel()
        n = f.width * f.height
        packed = np.concatenate(
            (
                flat[:n],
                np.stack((flat[n : n + n // 4], flat[n + n // 4 :]), axis=1).ravel(),
            )
        )
        planar = planar_nv12(packed.tobytes(), f.width, f.height)
        assert np.array_equal(planar, f.to_ndarray())
        copy = av.VideoFrame.from_ndarray(planar, format="yuv420p")
        copy.colorspace = f.colorspace
        copy.color_range = f.color_range
        assert np.array_equal(
            copy.to_ndarray(format="rgb24"), f.to_ndarray(format="rgb24")
        )
    images = []
    inputs = []
    for row in controls["images"]:
        path = a.controls / row["path"]
        assert sha(path) == row["sha256"]
        if row["stage"] == "rgb-resize":
            rgb = np.asarray(Image.open(path).convert("RGB"))
        else:
            frame = av.VideoFrame.from_ndarray(
                planar_nv12(path.read_bytes(), row["width"], row["height"]),
                format="yuv420p",
            )
            frame.colorspace = 1
            frame.color_range = 1
            rgb = frame.to_ndarray(format="rgb24")
        assert markers(rgb) == [row["state_id"]] * 2
        low = a.output / (row["id"] + "-input.png")
        Image.fromarray(rgb).save(low)
        out = a.output / (row["id"] + "-display.png")
        Image.fromarray(
            cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        ).save(out)
        base = {k: row[k] for k in ["id", "group", "category", "state_id", "height"]}
        base.update(
            config=f"{row['height']}p-{row['stage']}",
            method="bicubic" if row["height"] == 720 else "native",
            source_age_s=None,
            path=str(out.resolve()),
            sha256=sha(out),
            input_path=low.name,
            decoded_sha256=sha(low),
            codec_frame_count_valid=None,
        )
        images.append(base)
        if row["height"] == 720:
            inputs.append(base)
    common = dict(
        source_sha256=controls["source_sha256"],
        script_sha256=sha(Path(__file__)),
        controls_sha256=sha(a.controls / "MANIFEST.json"),
        av_version=av.__version__,
        scope=__doc__,
        raw_layout_roundtrip_byte_identical=True,
        complete=True,
    )
    save(a.output / "IMAGES.json", dict(common, images=images))
    save(a.output / "INPUTS.json", dict(common, images=inputs))


if __name__ == "__main__":
    main()
