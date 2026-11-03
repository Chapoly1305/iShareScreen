"""Freeze unseen, source-derived PDF evaluation pages from official DocLayNet splits.

This selects original documents before examining OCR or model predictions. It
does not use DocLayNet layout labels as transcription truth. The PDF text layer
and the same render transform provide line strings and geometric boxes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile

import fitz

from benchmarks.super_resolution.text_corpus.catalog import text_identity
from benchmarks.super_resolution.text_corpus.ocr_protocol import regions
from benchmarks.super_resolution.text_corpus.render import pdf_view
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def order(value: object) -> str:
    return hashlib.sha256(
        ("isharescreen-low-bandwidth-e2e-v1:" + str(value)).encode()
    ).hexdigest()


def old_identities(corpus: Path) -> tuple[set[str], set[str]]:
    catalog = json.loads((corpus / "sources.json").read_text())
    page_hashes = {
        p["text_sha256"] for s in catalog["sources"] for p in s.get("pages", [])
    }
    titles = {text_identity(s.get("title", "")) for s in catalog["sources"]}
    return page_hashes, titles


def document_pages(core: zipfile.ZipFile, split: str) -> list[list[dict]]:
    data = json.loads(core.read(f"COCO/{split}.json"))
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for image in data["images"]:
        groups[(image["collection"], image["doc_name"])].append(image)
    return [
        sorted(group, key=lambda row: order(row["file_name"]))
        for _, group in sorted(groups.items(), key=lambda item: order(item[0]))
    ]


def freeze(
    root: Path, corpus: Path, public: Path, validation_docs: int, test_docs: int
) -> None:
    if root.exists():
        raise FileExistsError(f"Frozen protocol already exists: {root}")
    root.mkdir(parents=True)
    core_path = public / "doclaynet/DocLayNet_core.zip"
    extra_path = public / "doclaynet/DocLayNet_extra.zip"
    core_receipt = json.loads(
        (core_path.with_name(core_path.name + ".receipt.json")).read_text()
    )
    extra_receipt = json.loads(
        (extra_path.with_name(extra_path.name + ".receipt.json")).read_text()
    )
    if not all(
        r.get("archive_check", {}).get("kind") == "ZIP CRC of every member"
        for r in (core_receipt, extra_receipt)
    ):
        raise ValueError("Full archive integrity receipt missing")
    old_pages, old_titles = old_identities(corpus)
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen_text: set[str] = set()
    seen_docs: set[tuple[str, str]] = set()
    with zipfile.ZipFile(core_path) as core, zipfile.ZipFile(extra_path) as extra:
        extra_names = set(extra.namelist())
        for split, quota in [("validation", validation_docs), ("test", test_docs)]:
            official = "val" if split == "validation" else "test"
            for pages in document_pages(core, official):
                if sum(row["split"] == split for row in accepted) == quota:
                    break
                for image in pages:
                    key = image["file_name"].removesuffix(".png")
                    doc_key = (image["collection"], image["doc_name"])
                    if doc_key in seen_docs:
                        raise ValueError(
                            "Original document appears twice across selected splits"
                        )
                    pdf_member, json_member = f"PDF/{key}.pdf", f"JSON/{key}.json"
                    if pdf_member not in extra_names or json_member not in extra_names:
                        rejected.append(
                            dict(
                                split=split,
                                page=key,
                                reason="original PDF or metadata absent",
                            )
                        )
                        continue
                    pdf_bytes = extra.read(
                        pdf_member
                    )  # ZIP CRC checked again for selected member.
                    metadata_bytes = extra.read(json_member)
                    metadata = json.loads(metadata_bytes)["metadata"]
                    if (
                        metadata["original_filename"] != image["doc_name"]
                        or metadata["collection"] != image["collection"]
                    ):
                        raise ValueError(f"COCO/source document mismatch: {key}")
                    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
                        if len(document) != 1:
                            rejected.append(
                                dict(
                                    split=split,
                                    page=key,
                                    reason="source PDF is not one page",
                                )
                            )
                            continue
                        page = document[0]
                        normalized = text_identity(page.get_text("text", sort=True))
                        text_sha = hashlib.sha256(normalized.encode()).hexdigest()
                        if (
                            not 300 <= len(normalized)
                            or text_sha in old_pages
                            or text_sha in seen_text
                        ):
                            rejected.append(
                                dict(
                                    split=split,
                                    page=key,
                                    reason="short or exact text overlap",
                                )
                            )
                            continue
                        if text_identity(image["doc_name"]) in old_titles:
                            rejected.append(
                                dict(
                                    split=split,
                                    page=key,
                                    reason="existing source title overlap",
                                )
                            )
                            continue
                    folder = root / "sources" / key
                    folder.mkdir(parents=True)
                    pdf_path = folder / "source.pdf"
                    metadata_path = folder / "source.json"
                    pdf_path.write_bytes(pdf_bytes)
                    metadata_path.write_bytes(metadata_bytes)
                    width = 1120 if int(order(key)[:2], 16) % 2 else 1480
                    job = dict(
                        id=f"doclaynet-{key[:16]}",
                        kind="paper",
                        split=split,
                        source_group=f"doclaynet:{image['collection']}:{image['doc_name']}",
                        source=key,
                        page=1,
                        page_width=width,
                        top_points=0.0,
                    )
                    source = dict(
                        path=str(pdf_path.relative_to(root)), sha256=sha256(pdf_path)
                    )
                    try:
                        view = pdf_view((root, job, source))
                        labels = json.loads((root / view["labels"]).read_text())
                        candidates = regions(labels)
                        if not candidates:
                            raise ValueError("no visible source-known English line")
                    except ValueError as error:
                        rejected.append(dict(split=split, page=key, reason=str(error)))
                        continue
                    line = candidates[0]
                    accepted.append(
                        dict(
                            **view,
                            official_split=official,
                            original_filename=image["doc_name"],
                            collection=image["collection"],
                            doc_category=image["doc_category"],
                            page_no=image["page_no"],
                            pdf=str(pdf_path.relative_to(root)),
                            pdf_sha256=source["sha256"],
                            metadata=str(metadata_path.relative_to(root)),
                            metadata_sha256=sha256(metadata_path),
                            text_sha256=text_sha,
                            rendering_width=width,
                            line=line,
                            source_truth="PDF embedded text and visible word geometry; no OCR prediction",
                        )
                    )
                    seen_text.add(text_sha)
                    seen_docs.add(doc_key)
                    print("FROZEN SOURCE", split, len(accepted), key, flush=True)
                    break
            if sum(row["split"] == split for row in accepted) != quota:
                raise ValueError(
                    f"Insufficient eligible {split} source documents; see selected records"
                )
    if len({row["source_group"] for row in accepted}) != len(accepted):
        raise ValueError("Source document leakage across cases")
    protocol = dict(
        schema="isharescreen_low_bandwidth_independent_pdf_v1",
        core_receipt_sha256=sha256(
            core_path.with_name(core_path.name + ".receipt.json")
        ),
        extra_receipt_sha256=sha256(
            extra_path.with_name(extra_path.name + ".receipt.json")
        ),
        old_sources_sha256=sha256(corpus / "sources.json"),
        old_views_sha256=sha256(corpus / "dataset-views-v3.json"),
        runner_sha256=sha256(Path(__file__)),
        cases=accepted,
        rejected=rejected,
        counts=dict(Counter(r["split"] for r in accepted)),
        selection="Official validation/test original-document splits; SHA256 document/page order, first source-eligible page and first source-eligible visible line. No OCR-success filter.",
        image_truth="Fresh PyMuPDF FHD render of original page PDF; never interpolated HR.",
        exact_overlap="Reject exact normalized full-page text hashes and matching normalized source titles against prior corpus; near-duplicate audit pending.",
        license_notice=str((public / "notices/doclaynet.upstream.txt").resolve()),
        status="frozen_sources_pending_near_duplicate_audit",
        finished=time.time(),
    )
    atomic_json(root / "independent-pdf-protocol.json", protocol)
    (root / "freeze-runner.py").write_bytes(Path(__file__).read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--validation-docs", type=int, default=24)
    parser.add_argument("--test-docs", type=int, default=48)
    args = parser.parse_args()
    freeze(
        args.root.resolve(),
        args.corpus.resolve(),
        args.public.resolve(),
        args.validation_docs,
        args.test_docs,
    )


if __name__ == "__main__":
    main()
