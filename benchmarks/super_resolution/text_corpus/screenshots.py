"""Faithful same-page HEVC screenshots at 128/600/1000 Kbps, with ML x2.

Fixed held-out PDF view and fixed seed 682 are selected without examining
quality. The PNG panels paste native pixels; no synthetic image generation.
"""

import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess
import time

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from benchmarks.super_resolution.text_corpus.scratch import build
from benchmarks.super_resolution.training_state import atomic_json, fingerprint
from isharescreen.enhancement.models import project_home, sha256


def read(path):
    return json.loads(path.read_text())


def panel(out, reference, decoded, enhanced, roi, kbps, second, actual):
    x, y, w, h = roi
    box = (x, y, x + w, y + h)
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    canvas = Image.new("RGB", (2 * w + 36, h + 110), "#f1f4f6")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 10),
        f"{kbps} Kbps | page held {second} s | same 100% pixel crop",
        font=font,
        fill="#15232a",
    )
    draw.text((12, 40), "Decoded 540p -> FHD (bilinear)", font=small, fill="#35434a")
    draw.text(
        (w + 24, 40), "ML x2: text + edge, fixed seed 682", font=small, fill="#11614f"
    )
    canvas.paste(decoded.crop(box), (12, 65))
    canvas.paste(enhanced.crop(box), (w + 24, 65))
    draw.text(
        (12, h + 77),
        f"HEVC 420 / 30 fps | requested {kbps} Kbps | measured 3 s video avg {actual / 1000:.1f} Kbps",
        font=small,
        fill="#35434a",
    )
    path = out / f"{kbps}k-{second}s-comparison.png"
    canvas.save(path)
    return path


def run(args):
    root = args.root.resolve()
    out = root / "screenshots-same-source-v1"
    out.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    protocol = read(root / "evaluation-protocol/protocol.json")
    case = sorted(
        [c for c in protocol["cases"] if c["split"] == "test" and c["kind"] == "paper"],
        key=lambda c: hashlib.sha256(c["id"].encode()).hexdigest(),
    )[0]
    source = root / case["hr"]
    assert sha256(source) == case["hr_sha256"]
    reference = Image.open(source).convert("RGB")
    assert reference.size == (1920, 1080)
    reference.save(out / "reference-fhd.png")
    x = max(0, case["xywh"][0] - 6)
    y = max(0, case["xywh"][1] - 6)
    roi = [x, y, min(500, 1920 - x), min(230, 1080 - y)]
    reference.crop((x, y, x + roi[2], y + roi[3])).save(out / "reference-crop.png")
    training = root / "hopper-results/runs/text-edge-seed-682"
    policy = read(training / "protocol.json")
    done = read(training / "COMPLETED.json")
    assert (
        fingerprint(policy) == done["protocol_binding"]
        and sha256(training / "best.pt") == done["best_sha256"]
    )
    model, _ = build(project_home(), 682)
    checkpoint = torch.load(training / "best.pt", map_location="cpu", weights_only=True)
    assert checkpoint["protocol_binding"] == done["protocol_binding"]
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    low = cv2.resize(np.array(reference), (960, 540), interpolation=cv2.INTER_LANCZOS4)
    raw = low.tobytes()
    results = []
    for kbps in [128, 600, 1000]:
        folder = out / f"{kbps}k"
        folder.mkdir()
        video = folder / "static-page.mkv"
        log = folder / "encode.log"
        params = f"pools=1:frame-threads=1:keyint=60:min-keyint=60:scenecut=0:bframes=0:rc-lookahead=0:bitrate={kbps}:vbv-maxrate={kbps}:vbv-bufsize={kbps}:vbv-init=0.5:strict-cbr=1:hrd=1:log-level=info"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "960x540",
            "-framerate",
            "30",
            "-i",
            "pipe:0",
            "-an",
            "-vf",
            "scale=in_range=full:out_range=full:out_color_matrix=bt709",
            "-c:v",
            "libx265",
            "-preset",
            "fast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "pc",
            "-colorspace",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_primaries",
            "bt709",
            "-x265-params",
            params,
            str(video),
        ]
        with log.open("w") as stream:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stream
            )
            try:
                for _ in range(90):
                    process.stdin.write(raw)
            finally:
                process.stdin.close()
            if process.wait(timeout=120):
                raise RuntimeError("HEVC encode failed; inspect " + str(log))
        decoded = {}
        count = 0
        packet_bytes = 0
        with av.open(video) as container:
            for packet in container.demux(video=0):
                packet_bytes += packet.size
                for frame in packet.decode():
                    if (
                        frame.is_corrupt
                        or frame.width != 960
                        or frame.height != 540
                        or frame.color_range != 2
                        or frame.colorspace != 1
                    ):
                        raise ValueError("Unexpected decoded pixels")
                    if count in [29, 89]:
                        decoded[count] = dict(
                            rgb=frame.to_ndarray(format="rgb24"),
                            keyframe=bool(frame.key_frame),
                            pts_s=float(frame.pts * frame.time_base),
                        )
                    count += 1
        assert count == 90
        actual = packet_bytes * 8 / 3
        frames = []
        for index, second in [(29, 1), (89, 3)]:
            rgb = decoded[index]["rgb"]
            Image.fromarray(rgb).save(folder / f"{second}s-lr-native.png")
            baseline = Image.fromarray(
                cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_LINEAR)
            )
            baseline.save(folder / f"{second}s-decoded-fhd.png")
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].float() / 255
            with torch.inference_mode():
                prediction = model(tensor)
            assert (
                prediction.shape == (1, 3, 1080, 1920)
                and torch.isfinite(prediction).all()
            )
            enhanced = Image.fromarray(
                prediction[0]
                .mul(255)
                .round()
                .clamp(0, 255)
                .byte()
                .permute(1, 2, 0)
                .numpy()
            )
            enhanced.save(folder / f"{second}s-ml-fhd.png")
            comparison = panel(
                out, reference, baseline, enhanced, roi, kbps, second, actual
            )
            frames.append(
                dict(
                    frame=index,
                    nominal_hold_s=second,
                    pts_s=decoded[index]["pts_s"],
                    keyframe=decoded[index]["keyframe"],
                    comparison=comparison.name,
                    decoded_fhd=f"{kbps}k/{second}s-decoded-fhd.png",
                    ml_fhd=f"{kbps}k/{second}s-ml-fhd.png",
                )
            )
        results.append(
            dict(
                requested_kbps=kbps,
                actual_average_video_bps=actual,
                vbv_underflows=log.read_text().count("VBV underflow"),
                video_sha256=sha256(video),
                command=command,
                frames=frames,
            )
        )
        print(
            "SCREENSHOTS COMPLETE",
            kbps,
            "Kbps, actual",
            round(actual / 1000, 1),
            "Kbps",
            flush=True,
        )
    receipt = dict(
        schema="isharescreen_same_source_screenshots_v1",
        source_case=case,
        source_hr_sha256=sha256(source),
        selection="First SHA256(case id) among frozen held-out paper cases; fixed text-edge seed 682, not best-test seed.",
        geometry=dict(hr=[1920, 1080], lr=[960, 540], roi=roi),
        fps=30,
        seconds=3,
        static=True,
        device="CPU FP32",
        checkpoint_sha256=sha256(training / "best.pt"),
        checkpoint_epoch=done["best_epoch"],
        results=results,
        runner_sha256=sha256(Path(__file__)),
        finished=time.time(),
        scope="Offline screenshots from actual HEVC decoding and actual trained model inference; not a live remote-connection screenshot or a physical-bandwidth test. Native crop pixels are pasted without resizing or sharpening. 1s/3s mean frames 29/89; periodic GOP is 60 frames.",
    )
    atomic_json(out / "manifest.json", receipt)
    page = [
        '<!doctype html><html><meta charset="utf-8"><title>Same PDF, three bitrates</title><style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:0 18px}img{max-width:100%}h2{margin-top:40px}</style><h1>Same held-out PDF: 128 / 600 / 1000 Kbps</h1><p>Left: decoded 540p upscaled with bilinear. Right: trained text + edge model, seed 682. Same source, same crop, same display geometry. Exact full-resolution images are linked below each comparison. Offline codec/model screenshots, not a live network measurement.</p><p><a href="reference-fhd.png">Clean FHD reference</a> · <a href="reference-crop.png">Clean native crop</a> · <a href="manifest.json">Source, model, timing and actual bitrate provenance</a></p>'
    ]
    for row in results:
        page.append(
            f"<h2>{row['requested_kbps']} Kbps requested · {row['actual_average_video_bps'] / 1000:.1f} Kbps measured video average</h2>"
        )
        for frame in row["frames"]:
            page.append(
                f'<h3>After {frame["nominal_hold_s"]} second(s)</h3><img src="{frame["comparison"]}"><p><a href="{frame["decoded_fhd"]}">Decoded FHD PNG</a> · <a href="{frame["ml_fhd"]}">ML FHD PNG</a></p>'
            )
    page.append("</html>")
    (out / "index.html").write_text("\n".join(page))
    atomic_json(
        out / "COMPLETED.json",
        dict(manifest_sha256=sha256(out / "manifest.json"), finished=time.time()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    run(p.parse_args())
