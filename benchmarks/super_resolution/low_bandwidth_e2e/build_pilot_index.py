"""Build an isolated 2,000-view source index with new public PDFs and old train-only desktop content."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import time
import zipfile

import fitz

from benchmarks.super_resolution.text_corpus.catalog import text_identity
from benchmarks.super_resolution.text_corpus.render import pdf_view
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def order(value: object) -> str:
    return hashlib.sha256(
        ("isharescreen-new-data-pilot-v1:" + str(value)).encode()
    ).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def old_views(corpus: Path, quotas: dict[str, int]) -> list[dict]:
    data = read(corpus / "dataset-views-v3.json")
    chosen = []
    for kind, quota in quotas.items():
        groups = defaultdict(list)
        for view in data["views"]:
            if view["split"] == "train" and view["kind"] == kind:
                groups[view["source_group"]].append(view)
        ordered_groups = sorted(groups, key=order)
        for group in ordered_groups:
            groups[group].sort(key=lambda row: order(row["id"]))
        candidates = []
        for depth in range(max(map(len, groups.values()))):
            candidates.extend(
                groups[group][depth]
                for group in ordered_groups
                if depth < len(groups[group])
            )
        if len(candidates) < quota:
            raise ValueError(f"Not enough old train {kind} views")
        chosen.extend(candidates[:quota])
    return chosen


def source_plan(root: Path, corpus: Path, public: Path) -> dict:
    core_path = public / "doclaynet/DocLayNet_core.zip"
    with zipfile.ZipFile(core_path) as core:
        data = json.loads(core.read("COCO/train.json"))
    groups = defaultdict(list)
    for image in data["images"]:
        groups[(image["collection"], image["doc_name"])].append(image)
    candidates = []
    for key in sorted(groups, key=order):
        pages = sorted(groups[key], key=lambda row: order(row["file_name"]))[:3]
        candidates.extend(
            dict(
                page_hash=row["file_name"].removesuffix(".png"),
                original_filename=row["doc_name"],
                collection=row["collection"],
                doc_category=row["doc_category"],
                page_no=row["page_no"],
            )
            for row in pages
        )
    quotas = dict(code=400, web=200, table=100, terminal=100)
    old = old_views(corpus, quotas)
    return dict(
        schema="isharescreen_new_data_pilot_source_plan_v1",
        doclaynet_train_quota=1200,
        old_quotas=quotas,
        doclaynet_candidates=candidates,
        old_views=old,
        old_views_sha256=sha256(corpus / "dataset-views-v3.json"),
        doclaynet_core_receipt_sha256=sha256(
            core_path.with_name(core_path.name + ".receipt.json")
        ),
        doclaynet_extra_receipt_sha256=sha256(
            public / "doclaynet/DocLayNet_extra.zip.receipt.json"
        ),
        selection="Official DocLayNet train original documents; SHA256 group/page order, at most three candidate pages per document, first 1200 eligible rendered views. Existing content selected train-only with source-group round robin.",
        filter="Require normalized source PDF text >=300 chars, no exact old-page text hash/title match, renderer minimum text/pixel coverage. Record every rejection. No OCR prediction filter.",
        created=time.time(),
    )


def run(root: Path, corpus: Path, public: Path) -> None:
    work = root / "pilot"
    work.mkdir(parents=True, exist_ok=True)
    with (work / "TASK.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = work / "source-index.json"
        if output.exists():
            raise FileExistsError("Pilot source index already complete")
        plan_path = work / "source-plan.json"
        if not plan_path.exists():
            atomic_json(plan_path, source_plan(root, corpus, public))
        plan = read(plan_path)
        if plan["old_views_sha256"] != sha256(corpus / "dataset-views-v3.json"):
            raise ValueError("Old corpus changed after source plan")
        old_catalog = read(corpus / "sources.json")
        old_pages = {
            p["text_sha256"] for s in old_catalog["sources"] for p in s.get("pages", [])
        }
        old_titles = {text_identity(s.get("title", "")) for s in old_catalog["sources"]}
        extra_path = public / "doclaynet/DocLayNet_extra.zip"
        accepted = []
        rejected = []
        per_doc = Counter()
        with zipfile.ZipFile(extra_path) as extra:
            members = set(extra.namelist())
            for candidate in plan["doclaynet_candidates"]:
                if len(accepted) >= plan["doclaynet_train_quota"]:
                    break
                key = candidate["page_hash"]
                group = f"doclaynet:{candidate['collection']}:{candidate['original_filename']}"
                if per_doc[group] >= 2:
                    continue
                pdf_member, meta_member = f"PDF/{key}.pdf", f"JSON/{key}.json"
                if pdf_member not in members or meta_member not in members:
                    rejected.append(dict(key=key, reason="missing source PDF/metadata"))
                    continue
                folder = work / "sources" / key
                folder.mkdir(parents=True, exist_ok=True)
                source = folder / "source.pdf"
                metadata = folder / "source.json"
                if not source.exists() or not metadata.exists():
                    source.write_bytes(extra.read(pdf_member))
                    metadata.write_bytes(extra.read(meta_member))
                meta = read(metadata)["metadata"]
                if (
                    meta["original_filename"] != candidate["original_filename"]
                    or meta["collection"] != candidate["collection"]
                ):
                    raise ValueError(f"DocLayNet metadata mismatch {key}")
                with fitz.open(source) as doc:
                    if len(doc) != 1:
                        rejected.append(dict(key=key, reason="not one source page"))
                        continue
                    page = doc[0]
                    normalized = text_identity(page.get_text("text", sort=True))
                    text_sha = hashlib.sha256(normalized.encode()).hexdigest()
                    if (
                        len(normalized) < 300
                        or text_sha in old_pages
                        or text_identity(candidate["original_filename"]) in old_titles
                    ):
                        rejected.append(
                            dict(key=key, reason="short/exact old-source overlap")
                        )
                        continue
                    width = (1120, 1480, 1840)[int(order(key)[:2], 16) % 3]
                    span = 1000 / (width / page.rect.width)
                    top = min(
                        max(0, page.rect.height - span),
                        max(0, page.rect.height - span)
                        * (0.5 if int(order(key)[2:4], 16) % 2 else 0),
                    )
                job = dict(
                    id="newpdf-" + key[:20],
                    kind="paper",
                    split="train",
                    source_group=group,
                    source=key,
                    page=1,
                    page_width=width,
                    top_points=round(top, 4),
                )
                try:
                    view = pdf_view(
                        (
                            work,
                            job,
                            dict(
                                path=str(source.relative_to(work)),
                                sha256=sha256(source),
                            ),
                        )
                    )
                except ValueError as error:
                    rejected.append(dict(key=key, reason=str(error)))
                    continue
                accepted.append(
                    dict(
                        **view,
                        source_pdf=str(source.relative_to(work)),
                        source_pdf_sha256=sha256(source),
                        source_metadata=str(metadata.relative_to(work)),
                        source_metadata_sha256=sha256(metadata),
                        official_split="train",
                        page_no=candidate["page_no"],
                        doc_category=candidate["doc_category"],
                        text_sha256=text_sha,
                    )
                )
                per_doc[group] += 1
                if len(accepted) % 50 == 0:
                    atomic_json(
                        work / "build-status.json",
                        dict(
                            phase="rendering",
                            accepted_new_pdf=len(accepted),
                            rejected=len(rejected),
                            latest_page=key,
                            updated=time.time(),
                        ),
                    )
                    print("PILOT NEW PDF", len(accepted), flush=True)
        if len(accepted) != plan["doclaynet_train_quota"]:
            atomic_json(
                work / "build-status.json",
                dict(
                    phase="failed",
                    accepted_new_pdf=len(accepted),
                    rejected=len(rejected),
                    reason="Insufficient eligible original train documents",
                ),
            )
            raise ValueError("Insufficient eligible DocLayNet train pages")
        old_rows = []
        for view in plan["old_views"]:
            image = corpus / view["hr"]
            labels = corpus / view["labels"]
            if (
                sha256(image) != view["hr_sha256"]
                or sha256(labels) != view["labels_sha256"]
            ):
                raise ValueError(f"Old training source changed: {view['id']}")
            old_rows.append(
                {
                    **view,
                    "hr": str(image.resolve()),
                    "labels": str(labels.resolve()),
                    "origin": "existing_train_only",
                }
            )
        result = dict(
            schema="isharescreen_new_data_pilot_index_v1",
            plan_sha256=sha256(plan_path),
            new_pdf=accepted,
            existing_train=old_rows,
            rejected=rejected,
            count=len(accepted) + len(old_rows),
            counts=dict(Counter(row["kind"] for row in accepted + old_rows)),
            complete=len(accepted) + len(old_rows) == 2000,
            source_groups=len({row["source_group"] for row in accepted + old_rows}),
            runner_sha256=sha256(Path(__file__)),
            finished=time.time(),
        )
        atomic_json(output, result)
        atomic_json(
            work / "SOURCE-INDEX-COMPLETED.json",
            dict(
                index_sha256=sha256(output), count=result["count"], finished=time.time()
            ),
        )
        print(
            "PILOT SOURCE INDEX COMPLETE", result["count"], result["counts"], flush=True
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.corpus.resolve(), args.public.resolve())


if __name__ == "__main__":
    main()
