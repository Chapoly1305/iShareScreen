"""Freeze new grouped code/path and DocLayNet sources for a bitrate frontier."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time
import zipfile

import fitz

from benchmarks.super_resolution.low_bandwidth_e2e.freeze_doclaynet import (
    document_pages,
    old_identities,
    order,
)
from benchmarks.super_resolution.text_corpus.catalog import text_identity
from benchmarks.super_resolution.text_corpus.ocr_protocol import regions
from benchmarks.super_resolution.text_corpus.render import html_views, pdf_view
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


TEMPLATES = {
    "validation": (
        "quota_file",
        "deploy_log",
        "csv_export",
        "build_flag",
        "route_map",
        "config_json",
        "audit_path",
        "metric_line",
    ),
    "test": (
        "package_lock",
        "stack_frame",
        "shell_env",
        "request_id",
        "service_unit",
        "parser_rule",
        "storage_key",
        "release_tag",
    ),
}


def generated_code(template: str) -> str:
    seed = int(
        hashlib.sha256(("bitrate-frontier-v2:" + template).encode()).hexdigest()[:16],
        16,
    )
    rng = random.Random(seed)
    lines = ["from pathlib import Path", "import json", "import re", ""]
    for index in range(46):
        number = rng.randrange(100000, 999999)
        tenant = rng.randrange(1000, 9999)
        percent = f"{rng.randrange(10, 99)}.{rng.randrange(10)}%"
        path = f"/srv/{template[:5]}/{tenant}/r{number}.json"
        parts = {
            "quota_file": f'QUOTA_{index:02d} = ("{path}", "{percent}")',
            "deploy_log": f'DEPLOY_{index:02d} = "node={tenant} {percent} {path}"',
            "csv_export": f'EXPORT_{index:02d} = "{number},{percent},{path}"',
            "build_flag": f'BUILD_{index:02d} = ["--in={path}", "--rate={percent}"]',
            "route_map": f'ROUTES_{index:02d} = {{"/job/{number}": ("{path}", "{percent}")}}',
            "config_json": f'CONFIG_{index:02d} = {{"rate": "{percent}", "path": "{path}"}}',
            "audit_path": f'AUDIT_{index:02d} = "user_{tenant}: {percent}; {path}"',
            "metric_line": f'METRIC_{index:02d} = "job_{number} {percent} {path}"',
            "package_lock": f'PACKAGE_{index:02d} = {{"id": {number}, "file": "{path}", "pct": "{percent}"}}',
            "stack_frame": f'FRAME_{index:02d} = "error_{number}; {percent}; {path}"',
            "shell_env": f'ENV_{index:02d} = "DATA={path} LOAD={percent}"',
            "request_id": f'REQUEST_{index:02d} = ("PUT /jobs/{number}", "{percent}", "{path}")',
            "service_unit": f'SERVICE_{index:02d} = "tenant-{tenant}: {percent}; {path}"',
            "parser_rule": f'RULE_{index:02d} = re.compile(r"{number}/{percent}/{path}")',
            "storage_key": f'STORE_{index:02d} = Path("{path}").with_name("{number}_{percent}.json")',
            "release_tag": f'RELEASE_{index:02d} = "v{tenant}.{index % 20}: {percent} {path}"',
        }
        lines.append(parts[template])
    return "\n".join(lines) + "\n"


def run(root: Path, parent: Path, corpus: Path, public: Path) -> None:
    root = root.resolve()
    protocol_path = root / "EVAL_PROTOCOL.json"
    if protocol_path.exists():
        raise FileExistsError("The new evaluation protocol is already frozen")
    old_eval = json.loads((parent / "EVAL_PROTOCOL.json").read_text())
    percent_eval = json.loads(
        (parent / "symbol-focus-v1/holdout/EVAL_PROTOCOL.json").read_text()
    )
    old_groups = {
        case["source_group"] for p in (old_eval, percent_eval) for case in p["cases"]
    }
    old_lines = {
        text_identity(case["line"]["text"])
        for p in (old_eval, percent_eval)
        for case in p["cases"]
    }
    prior_pdf = json.loads(
        (parent / "independent-pdf/independent-pdf-protocol.json").read_text()
    )
    old_groups.update(case["source_group"] for case in prior_pdf["cases"])
    prior_pages, prior_titles = old_identities(corpus)
    pdf_cases, rejected, seen_doc, seen_page_text = [], [], set(), set()
    core_path = public / "doclaynet/DocLayNet_core.zip"
    extra_path = public / "doclaynet/DocLayNet_extra.zip"
    for path in (core_path, extra_path):
        receipt = json.loads((path.with_name(path.name + ".receipt.json")).read_text())
        if receipt.get("archive_check", {}).get("kind") != "ZIP CRC of every member":
            raise ValueError(f"Missing full ZIP CRC audit: {path}")
    with zipfile.ZipFile(core_path) as core, zipfile.ZipFile(extra_path) as extra:
        names = set(extra.namelist())
        for split, quota, official in [("validation", 8, "val"), ("test", 16, "test")]:
            for pages in document_pages(core, official):
                if sum(case["split"] == split for case in pdf_cases) >= quota:
                    break
                for image in pages:
                    key = image["file_name"].removesuffix(".png")
                    group = f"doclaynet:{image['collection']}:{image['doc_name']}"
                    doc_key = image["collection"], image["doc_name"]
                    if (
                        group in old_groups
                        or doc_key in seen_doc
                        or text_identity(image["doc_name"]) in prior_titles
                    ):
                        continue
                    member = f"PDF/{key}.pdf"
                    if member not in names:
                        rejected.append(
                            dict(split=split, page=key, reason="original PDF absent")
                        )
                        continue
                    pdf_bytes = extra.read(member)
                    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
                        if len(document) != 1:
                            rejected.append(
                                dict(split=split, page=key, reason="not a one-page PDF")
                            )
                            continue
                        text = text_identity(document[0].get_text("text", sort=True))
                    text_hash = hashlib.sha256(text.encode()).hexdigest()
                    if (
                        len(text) < 300
                        or text_hash in prior_pages
                        or text_hash in seen_page_text
                    ):
                        rejected.append(
                            dict(
                                split=split,
                                page=key,
                                reason="short or exact page overlap",
                            )
                        )
                        continue
                    source = root / "sources/pdf" / key / "source.pdf"
                    source.parent.mkdir(parents=True, exist_ok=True)
                    source.write_bytes(pdf_bytes)
                    width = 1120 if int(order(key)[:2], 16) % 2 else 1480
                    job = dict(
                        id="frontier-pdf-" + key[:16],
                        kind="paper",
                        split=split,
                        source_group=group,
                        source=key,
                        page=1,
                        page_width=width,
                        top_points=0.0,
                    )
                    source_spec = dict(
                        path=str(source.relative_to(root)), sha256=sha256(source)
                    )
                    try:
                        view = pdf_view((root, job, source_spec))
                        labels = json.loads((root / view["labels"]).read_text())
                        lines = [
                            line
                            for line in regions(labels)
                            if text_identity(line["text"]) not in old_lines
                        ]
                        if not lines:
                            raise ValueError("No new source-known visible English line")
                    except ValueError as error:
                        rejected.append(dict(split=split, page=key, reason=str(error)))
                        continue
                    line = lines[0]
                    pdf_cases.append(
                        dict(
                            id=view["id"],
                            source_kind="paper",
                            split=split,
                            source_group=group,
                            content_category=image["doc_category"],
                            truth_origin="original PDF text layer and visible render geometry",
                            hr=view["hr"],
                            hr_sha256=view["hr_sha256"],
                            labels=view["labels"],
                            labels_sha256=view["labels_sha256"],
                            line=line,
                            font_size=None,
                            rendering_width=width,
                            source_original=source_spec["path"],
                            source_sha256=source_spec["sha256"],
                            original_filename=image["doc_name"],
                            text_sha256=text_hash,
                            official_split=official,
                        )
                    )
                    old_lines.add(text_identity(line["text"]))
                    seen_page_text.add(text_hash)
                    seen_doc.add(doc_key)
                    print("FRONTIER PDF", split, len(pdf_cases), group, flush=True)
                    break
            if sum(case["split"] == split for case in pdf_cases) != quota:
                raise ValueError(
                    f"Only {sum(case['split'] == split for case in pdf_cases)}/{quota} eligible {split} PDFs"
                )
    code_cases, jobs, sources = [], [], {}
    for split, templates in TEMPLATES.items():
        for index, template in enumerate(templates):
            identifier = f"frontier-{split}-{template}"
            path = root / "sources/code" / (identifier + ".py")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(generated_code(template))
            sources[identifier] = dict(
                path=str(path.relative_to(root)), title=path.name, sha256=sha256(path)
            )
            jobs.append(
                dict(
                    id=identifier,
                    kind="code",
                    split=split,
                    source_group=f"frontier-template:{template}",
                    source=identifier,
                    line_start=0,
                    font_size=16,
                    theme="light",
                )
            )
    for view in html_views(root, jobs, sources):
        labels = json.loads((root / view["labels"]).read_text())
        lines = [
            line
            for line in regions(labels)
            if line["digits"] >= 3
            and line["punctuation"] >= 3
            and line["xywh"][2] <= 900
            and text_identity(line["text"]) not in old_lines
        ]
        if not lines:
            raise ValueError(f"No eligible source-derived code line: {view['id']}")
        line = lines[0]
        code_cases.append(
            dict(
                id=view["id"],
                source_kind="code",
                split=view["split"],
                source_group=view["source_group"],
                content_category="code_path",
                truth_origin="literal source and browser DOM geometry",
                hr=view["hr"],
                hr_sha256=view["hr_sha256"],
                labels=view["labels"],
                labels_sha256=view["labels_sha256"],
                line=line,
                font_size=labels["font_size"],
                source_original=sources[labels["source"]]["path"],
                source_sha256=sources[labels["source"]]["sha256"],
            )
        )
        old_lines.add(text_identity(line["text"]))
    cases = pdf_cases + code_cases
    if Counter((case["split"], case["source_kind"]) for case in cases) != {
        ("validation", "paper"): 8,
        ("validation", "code"): 8,
        ("test", "paper"): 16,
        ("test", "code"): 8,
    }:
        raise ValueError("New source quotas incomplete")
    val = {c["source_group"] for c in cases if c["split"] == "validation"}
    test = {c["source_group"] for c in cases if c["split"] == "test"}
    if val & test or (val | test) & old_groups:
        raise ValueError("Source group crossed split or prior evaluation")
    protocol = dict(
        schema="isharescreen_bitrate_frontier_eval_v1",
        frozen=time.time(),
        runner_sha256=sha256(Path(__file__)),
        parent_protocol_sha256=sha256(parent / "EVAL_PROTOCOL.json"),
        plan_sha256=sha256(root / "PLAN.json"),
        cases=cases,
        counts={
            f"{split}/{kind}": count
            for (split, kind), count in Counter(
                (c["split"], c["source_kind"]) for c in cases
            ).items()
        },
        fixed_display_ids={
            split: [c["id"] for c in cases if c["split"] == split][:4]
            for split in ("validation", "test")
        },
        main_grid=dict(
            old_eval["main_grid"],
            base_geometry=[[1280, 720], [1920, 1080]],
            requested_video_bps=[1000000, 1250000, 1500000, 2000000],
        ),
        codec=old_eval["codec"],
        ocr=old_eval["ocr"],
        selection="One deterministic source-derived visible line per new source; never selected by OCR or model prediction.",
        output_geometry=[1920, 1080],
        limits="Static synthetic code or official DocLayNet page, sampled frames 60/90; not live desktop interaction or Mac client.",
    )
    atomic_json(protocol_path, protocol)
    atomic_json(
        root / "OVERLAP_AUDIT.json",
        dict(
            schema="isharescreen_bitrate_overlap_audit_v1",
            protocol_sha256=sha256(protocol_path),
            prior_source_groups_excluded=len(old_groups),
            prior_normalized_lines_excluded=len(old_lines),
            new_source_groups=len(val | test),
            validation_test_group_overlap=0,
            prior_group_overlap=0,
            exact_selected_line_overlap=0,
            pdf_rejections=rejected,
            result="passed exact source/line audit; visual near duplicate remains limited",
            finished=time.time(),
        ),
    )
    print(
        "FRONTIER SOURCES FROZEN", protocol["counts"], sha256(protocol_path), flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.parent.resolve(), args.corpus.resolve(), args.public.resolve())
