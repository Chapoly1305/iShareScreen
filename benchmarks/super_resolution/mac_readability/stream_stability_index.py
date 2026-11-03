"""Hash-bound timed screenshots from actual receiver-visible canvases."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmarks.super_resolution.mac_readability.quality import save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--run-name", required=True)
    a = p.parse_args()
    folder = a.root / "stream-runs" / a.run_name
    receiver = folder / "receiver"
    data = json.loads((receiver / "RESULTS.json").read_text())
    sender = json.loads((folder / "sender/RESULTS.json").read_text())
    assert data["complete"] and sender["complete"] and len(data["source_events"]) == 32
    states = {x["state_id"]: x for x in sender["states"]}
    images = []
    assert len(data["timed_images"]) == 128
    for row in data["timed_images"]:
        path = receiver / row["path"]
        pixels = np.asarray(Image.open(path).convert("RGB"))
        assert list(pixels.shape) == row["shape"]
        assert hashlib.sha256(pixels.tobytes()).hexdigest() == row["pixel_sha256"]
        state = states[row["state_id"]]
        age = row["age_target_s"]
        images.append(
            dict(
                id=f"{a.run_name}-s{row['state_id']:02d}-age{round(age * 1000):04d}",
                state_id=row["state_id"],
                group=state["group"],
                config=f"receiver-announcement-age-{age:.1f}s",
                method="stream-" + data["policy"],
                source_age_s=age,
                actual_receiver_age_s=row["age_actual_s"],
                phase=row["phase"],
                path=str(path.resolve()),
                sha256=sha(path),
            )
        )
    save(
        folder / "TIMED-IMAGES.json",
        dict(
            source_sha256=sender["source_sha256"],
            images=images,
            complete=True,
            receiver_sha256=sha(receiver / "RESULTS.json"),
            sender_sha256=sha(folder / "sender/RESULTS.json"),
            scope="Actual visible pixels at local receiver-source-announcement ages; retain blank/late/low-quality outputs. Separate from frozen offline2.8s codec-age qualification.",
        ),
    )
    print(a.run_name, "128hash-verified timed images", flush=True)


if __name__ == "__main__":
    main()
