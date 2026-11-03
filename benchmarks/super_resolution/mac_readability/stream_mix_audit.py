"""Audit actual displayed source/SR composites before scoring any OCR."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from benchmarks.super_resolution.mac_readability.quality import markers, save, sha


def read(path):
    return json.loads(path.read_text())


def rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    folder = args.root / "stream-runs" / args.run_name
    receiver = folder / "receiver"
    sender = read(folder / "sender/RESULTS.json")
    data = read(receiver / "RESULTS.json")
    assert sender["complete"] and data["complete"] and sender["ssh_returncode"] == 0
    assert (
        not data["errors"] and len(data["supplements"]) == len(sender["states"]) == 32
    )
    native = {
        x["state_id"]: x
        for x in read(args.root / "precodec-images-v1/IMAGES.json")["images"]
        if x["config"] == "1080p-420-no-codec"
    }
    inputs = {
        x["state_id"]: x
        for x in read(args.root / "dense-inference-inputs/INPUTS.json")["images"]
    }
    states = {x["state_id"]: x for x in sender["states"]}
    checks, images = [], []
    for item in data["supplements"]:
        sid = item["state_id"]
        state = states[sid]
        refreshes = [
            x
            for x in data.get("mix_refresh_events", [])
            if x["generation"] == state["generation"]
        ]
        refreshed = bool(refreshes)
        if refreshed:
            item = refreshes[-1]
        assert item["generation"] == state["generation"]
        assert item["raw_sha256"] == state["raw_sha256"]
        path = receiver / (item["path"] if refreshed else f"s{sid:02d}-detail.png")
        output, reference = rgb(path), rgb(native[sid]["path"])
        weight = item["source_weight_actual"]
        if weight < 1:
            source = rgb(
                receiver
                / (
                    item["source_path"]
                    if refreshed
                    else f"s{sid:02d}-refinement-source.png"
                )
            )
            base = rgb(
                receiver
                / (
                    item["sr_base_path"]
                    if refreshed
                    else f"s{sid:02d}-refinement-sr-base.png"
                )
            )
            assert (
                hashlib.sha256(source.tobytes()).hexdigest()
                == item["source_rgb_sha256"]
            )
            assert (
                hashlib.sha256(base.tobytes()).hexdigest() == item["sr_base_rgb_sha256"]
            )
            sr_record = item if refreshed else item["sr_base_record"]
            assert sr_record["generation"] == state["generation"]
            assert sr_record["input_revision"] == sr_record["latest_codec_revision"]
            assert np.abs(source.astype("int16") - reference.astype("int16")).max() <= 1
            expected = cv2.addWeighted(source, weight, base, 1 - weight, 0)
        else:
            expected = reference
        delta = int(np.abs(output.astype("int16") - expected.astype("int16")).max())
        assert delta <= 1
        first = markers(rgb(receiver / f"s{sid:02d}-first.png"))
        assert first == [sid, sid]
        decoded = rgb(receiver / f"s{sid:02d}-steady-decoded.png")
        frozen = rgb(inputs[sid]["decoded_path"])
        decoded_delta = int(
            np.abs(decoded.astype("int16") - frozen.astype("int16")).max()
        )
        assert decoded_delta <= 1
        checks.append(
            dict(
                state_id=sid,
                source_weight=weight,
                refreshed=refreshed,
                output_max_delta=delta,
                first_markers=first,
                frozen_decode_max_delta=decoded_delta,
            )
        )
        images.append(
            dict(
                id=f"{args.run_name}-s{sid:02d}-final",
                state_id=sid,
                group=state["group"],
                config="actual-stream-final",
                method="refresh-source-mix-25"
                if data.get("refresh_source_mix")
                else "hold-source-mix-25",
                source_age_s=None,
                source_weight_actual=weight,
                path=str(path.resolve()),
                sha256=sha(path),
            )
        )
    save(
        folder / "FINAL-IMAGES.json",
        dict(
            source_sha256=sender["source_sha256"],
            images=images,
            complete=True,
            scope="Actual final GPU readbacks; includes fallback cases; arrival ages vary. Not frozen 2.8s codec-age images.",
        ),
    )
    save(
        folder / "MIX-AUDIT.json",
        dict(
            complete=True,
            checks=checks,
            mixed_states=sum(x["source_weight"] < 1 for x in checks),
            fallback_states=sum(x["source_weight"] == 1 for x in checks),
            evidence={
                part: sha(folder / part / "RESULTS.json")
                for part in ["sender", "receiver"]
            },
        ),
    )
    print(
        "Verified actual composites:",
        len(checks),
        "fallbacks:",
        sum(x["source_weight"] == 1 for x in checks),
    )


if __name__ == "__main__":
    main()
