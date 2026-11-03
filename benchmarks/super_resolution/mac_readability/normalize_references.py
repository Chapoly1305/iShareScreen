"""Normalize captured Mac window ICC colors and verify source state markers."""

import argparse
import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--dense",
        action="store_true",
        help="Normalize the separately frozen 32-state development documents",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.source / "manifest.json"
    source = json.loads(path.read_text())
    count = 32 if args.dense else 8
    assert source["complete"] and len(source["states"]) == count
    result = copy.deepcopy(source)
    result["schema"] = "mac_captured_srgb_text_reference_v2"
    result["source_manifest_sha256"] = sha(path)
    result["source_manifest_path"] = str(path.resolve())
    result["color_normalization"] = (
        "Embedded display ICC -> sRGB, relative colorimetric intent. Originals preserved."
    )
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    hashes = []
    for row in result["states"]:
        if row.get("labels_state_id") != row["state_id"]:
            raise ValueError("Reference lacks labels tied to its actual draw state")
        body = row["labels"] if args.dense else row["labels"][:-4]
        assert [label["text"] for label in body] == [
            line for line in row["lines"] if line
        ]
        original = args.source / row["reference"]["path"]
        assert sha(original) == row["reference"]["sha256"]
        image = Image.open(original)
        icc = image.info.get("icc_profile")
        if not icc:
            raise ValueError("Captured reference has no ICC profile")
        profile = ImageCms.ImageCmsProfile(BytesIO(icc))
        if "A" in image.getbands():
            assert np.all(np.asarray(image.getchannel("A")) == 255)
        rgb = ImageCms.profileToProfile(
            image, profile, srgb, renderingIntent=1, outputMode="RGB"
        )
        pixels = np.asarray(rgb)
        assert pixels.shape == (1080, 1920, 3)
        markers = []
        for y in (24, 1048):
            markers.append(
                sum(
                    (int(pixels[y, (440 + bit * 10 + 4) * 2].mean()) > 128) << bit
                    for bit in range(8)
                )
            )
        assert markers == [row["state_id"]] * 2, (
            "Wrong/mixed source state",
            row["id"],
            markers,
        )
        output = args.output / original.name
        rgb.save(output, icc_profile=srgb.tobytes())
        row["original_reference"] = row["reference"]
        row["reference"] = dict(
            path=output.name, sha256=sha(output), dimensions=[1920, 1080]
        )
        row["input_icc_sha256"] = hashlib.sha256(icc).hexdigest()
        row["input_icc_name"] = ImageCms.getProfileDescription(profile).strip()
        row["verified_markers"] = markers
        row["decoded_pixels_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()
        hashes.append(row["decoded_pixels_sha256"])
        for label in row["labels"]:
            x, y, w, h = label["rect_points"]
            label["xywh"] = [round(x * 2), round(y * 2), round(w * 2), round(h * 2)]
    assert len(set(hashes)) == count
    result["complete"] = True
    (args.output / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(count, "ICC-normalized unique reference frames; both state markers verified")


if __name__ == "__main__":
    main()
