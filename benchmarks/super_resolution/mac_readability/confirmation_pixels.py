"""Native-reference full-frame and text-region fidelity for fixed confirmation."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from benchmarks.super_resolution.mac_readability.quality import save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    output = root / "PIXEL-QUALITY.json"
    assert not output.exists()
    source = json.loads((a.source / "manifest.json").read_text())
    assert source["complete"]
    names = {
        "official": "OFFICIAL-IMAGES.json",
        "trained": "TRAINED-IMAGES.json",
        "native": "NATIVE-IMAGES.json",
        "candidate": "stream-runs/confirm-refresh-01/FINAL-IMAGES.json",
    }
    indexes = {}
    binding = {}
    for name, path in names.items():
        index = json.loads((root / path).read_text())
        assert index["source_sha256"] == sha(a.source / "manifest.json")
        indexes[name] = {x["state_id"]: x for x in index["images"]}
        assert len(indexes[name]) == 32
        binding[name] = sha(root / path)
    result = dict(
        complete=False,
        script_sha256=sha(Path(__file__)),
        source_manifest_sha256=sha(a.source / "manifest.json"),
        index_sha256=binding,
        rows=[],
        started=time.time(),
        ssim="skimage default7x7 window,RGBchannel_axis2,data_range255;ROI mean over identical source-label mask on full SSIMmap",
        test_used=False,
    )
    for state in source["states"]:
        ref = np.asarray(
            Image.open(a.source / state["reference"]["path"]).convert("RGB")
        )
        mask = np.zeros(ref.shape[:2], bool)
        for label in state["labels"]:
            x, y, w, h = label["xywh"]
            mask[y : y + h, x : x + w] = True
        for name, index in indexes.items():
            item = index[state["state_id"]]
            assert sha(item["path"]) == item["sha256"]
            pixels = np.asarray(Image.open(item["path"]).convert("RGB"))
            assert pixels.shape == ref.shape
            error = np.square(ref.astype(np.float32) - pixels.astype(np.float32))
            mse = float(error.mean())
            roi_mse = float(error[mask].mean())
            ssim, ssim_map = structural_similarity(
                ref, pixels, channel_axis=2, data_range=255, full=True
            )
            result["rows"].append(
                dict(
                    method=name,
                    state_id=state["state_id"],
                    group=state["group"],
                    category=state["category"],
                    full_mse=mse,
                    roi_mse=roi_mse,
                    full_psnr=float(10 * np.log10(255**2 / mse)),
                    roi_psnr=float(10 * np.log10(255**2 / roi_mse)),
                    full_ssim=float(ssim),
                    roi_ssim=float(ssim_map[mask].mean()),
                )
            )
        save(root / "PIXEL-PROGRESS.json", dict(states=state["state_id"], total=32))
        print("Pixel metrics", state["state_id"], "/32", flush=True)
    result.update(complete=True, finished=time.time())
    save(output, result)


if __name__ == "__main__":
    main()
