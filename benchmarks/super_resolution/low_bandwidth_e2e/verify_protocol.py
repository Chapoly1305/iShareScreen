"""Verify frozen case bindings and original-document split provenance."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
import zipfile

from benchmarks.super_resolution.text_corpus.ocr_protocol import regions
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def run(root: Path, public: Path) -> None:
    protocol_path = root / "EVAL_PROTOCOL.json"
    protocol = read(protocol_path)
    core_path = public / "doclaynet/DocLayNet_core.zip"
    split_docs = {}
    selected_page = {}
    categories = {}
    with zipfile.ZipFile(core_path) as archive:
        for split in ("train", "val", "test"):
            data = json.loads(archive.read(f"COCO/{split}.json"))
            split_docs[split] = {
                (row["collection"], row["doc_name"]) for row in data["images"]
            }
            if split in ("val", "test"):
                selected_page.update(
                    {row["file_name"][:-4]: (split, row) for row in data["images"]}
                )
                categories[split] = {
                    row["id"]: row["name"] for row in data["categories"]
                }
    if any(
        split_docs[a] & split_docs[b]
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
    ):
        raise ValueError("DocLayNet original documents cross official splits")
    records = []
    for case in protocol["cases"]:
        image, labels = root / case["hr"], root / case["labels"]
        if (
            sha256(image) != case["hr_sha256"]
            or sha256(labels) != case["labels_sha256"]
        ):
            raise ValueError(f"Frozen pixels or labels differ: {case['id']}")
        row = read(labels)
        candidates = regions(row)
        if not candidates or candidates[0] != case["line"]:
            raise ValueError(
                f"Frozen line is not derived from visible source geometry: {case['id']}"
            )
        if case["truth_origin"] == "pdf":
            key = row["source"]
            official, entry = selected_page[key]
            if official != ("val" if case["split"] == "validation" else "test"):
                raise ValueError(f"Wrong official split: {case['id']}")
            expected_group = f"doclaynet:{entry['collection']}:{entry['doc_name']}"
            if case["source_group"] != expected_group:
                raise ValueError(f"Wrong original-document group: {case['id']}")
            folder = root / "independent-pdf/sources" / key
            source = root / "independent-pdf/independent-pdf-protocol.json"
            bound = next(x for x in read(source)["cases"] if x["id"] == case["id"])
            if (
                sha256(folder / "source.pdf") != bound["pdf_sha256"]
                or sha256(folder / "source.json") != bound["metadata_sha256"]
            ):
                raise ValueError(f"Original PDF/metadata changed: {case['id']}")
        elif case["truth_origin"] == "html":
            source = root / case["source_original"]
            html = root / "independent-html-protocol.json"
            bound = next(x for x in read(html)["cases"] if x["id"] == case["id"])
            if sha256(source) != bound["source_sha256"]:
                raise ValueError(f"Original HTML source changed: {case['id']}")
        records.append(
            dict(
                id=case["id"],
                split=case["split"],
                kind=case["source_kind"],
                source_group=case["source_group"],
                line_text=case["line"]["text"],
                image_sha256=case["hr_sha256"],
                labels_sha256=case["labels_sha256"],
            )
        )
    groups = {
        split: {row["source_group"] for row in records if row["split"] == split}
        for split in ("validation", "test")
    }
    if groups["validation"] & groups["test"]:
        raise ValueError("New source group crosses validation/test")
    atomic_json(
        root / "protocol-verification.json",
        dict(
            schema="isharescreen_low_bandwidth_protocol_verification_v1",
            protocol_sha256=sha256(protocol_path),
            core_receipt_sha256=sha256(
                core_path.with_name(core_path.name + ".receipt.json")
            ),
            official_document_counts={s: len(v) for s, v in split_docs.items()},
            official_document_intersections_zero=True,
            checked_cases=len(records),
            source_groups={s: len(v) for s, v in groups.items()},
            case_counts={
                f"{s}/{k}": n
                for (s, k), n in Counter(
                    (r["split"], r["kind"]) for r in records
                ).items()
            },
            all_case_sources_and_renders_match=True,
            visible_line_geometry_match=True,
            overlap_audit_hashes=[
                sha256(root / "independent-pdf/overlap-audit.json"),
                sha256(root / "html-overlap-audit.json"),
            ],
            records=records,
            scope="Exact source/render bindings, official original-document partition, visible line derivation and prior visible-text overlap screens; no proof against every hidden image or paraphrase overlap.",
            runner_sha256=sha256(Path(__file__)),
            finished=time.time(),
        ),
    )
    print("VERIFIED FROZEN PROTOCOL", len(records), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.public.resolve())


if __name__ == "__main__":
    main()
