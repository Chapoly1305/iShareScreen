"""Freeze source-known percent/code views before symbol-focused model training."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import time

from benchmarks.super_resolution.text_corpus.ocr_protocol import regions
from benchmarks.super_resolution.text_corpus.render import html_views
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


TEMPLATES = {
    "validation": ("quota", "billing", "status", "request"),
    "test": (
        "threshold",
        "assertion",
        "sql",
        "warning",
        "command",
        "style",
        "format",
        "metric",
    ),
}


def make_line(template: str, ordinal: int, rng: random.Random) -> str:
    pct = f"{rng.randrange(1, 99)}.{rng.randrange(10)}%"
    tenant = rng.randrange(1000, 9999)
    item = rng.randrange(100000, 999999)
    path = f"/srv/tenant_{tenant}/runs/2026-10/item_{item}.json"
    lines = {
        "quota": f'QUOTA_{ordinal:02d} = {{"used": "{pct}", "path": "{path}", "max": {item}}}',
        "billing": f'INVOICE_{item} = "discount {pct} on tenant {tenant}; export={path}"',
        "status": f'PROGRESS_{ordinal:02d} = "job {item}: {pct} done; log={path}"',
        "request": f'HEADER_{ordinal:02d} = ("POST /v3/jobs/{item}", "X-Percent: {pct}", "{path}")',
        "threshold": f'THRESHOLD_{item} = {{"ratio": "{pct}", "artifact": "{path}"}}',
        "assertion": f'assert usage_{tenant} <= "{pct}", "failed item={item}; file={path}"',
        "sql": f'SQL_{ordinal:02d} = "SELECT id FROM jobs WHERE load_pct < {pct} AND id={item};"',
        "warning": f'WARN_{ordinal:02d} = "2026-10-21 node={tenant} at {pct}; trace={path}"',
        "command": f'CMD_{ordinal:02d} = "tool --rate={pct} --item={item} --output={path}"',
        "style": f'STYLE_{ordinal:02d} = "width:{pct}; margin-left:{ordinal}%; asset:{path}"',
        "format": f'FORMAT_{ordinal:02d} = "error={pct}; key=k_{item}; source={path}"',
        "metric": f'METRIC_{ordinal:02d} = "tenant_{tenant}{{job={item}}} {pct} path={path}"',
    }
    return lines[template]


def generate_source(template: str, index: int) -> str:
    seed = int(
        hashlib.sha256(
            f"isharescreen-percent-holdout-v1:{template}:{index}".encode()
        ).hexdigest()[:16],
        16,
    )
    rng = random.Random(seed)
    return "\n".join(make_line(template, ordinal, rng) for ordinal in range(45)) + "\n"


def run(campaign: Path, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if (root / "EVAL_PROTOCOL.json").exists():
        raise FileExistsError("Percent holdout was already frozen")
    old_path = campaign / "EVAL_PROTOCOL.json"
    old = json.loads(old_path.read_text())
    old_lines = {"".join(c["line"]["text"].split()) for c in old["cases"]}
    jobs, sources = [], {}
    for split, templates in TEMPLATES.items():
        for template in templates:
            for index in range(3):
                identifier = f"percent-{split}-{template}-{index:02d}"
                source_file = root / "sources" / f"{identifier}.py"
                source_file.parent.mkdir(parents=True, exist_ok=True)
                content = generate_source(template, index)
                if source_file.exists() and source_file.read_text() != content:
                    raise ValueError(f"Generated source changed: {source_file}")
                source_file.write_text(content)
                sources[identifier] = dict(
                    path=str(source_file.relative_to(root)),
                    title=source_file.name,
                    sha256=sha256(source_file),
                )
                jobs.append(
                    dict(
                        id=identifier,
                        kind="code",
                        split=split,
                        source_group=f"percent-template:{template}",
                        source=identifier,
                        line_start=0,
                        font_size=(14, 16, 18)[index],
                        theme="dark" if index == 1 else "light",
                    )
                )
    cases = []
    for view in html_views(root, jobs, sources):
        labels_path = root / view["labels"]
        labels = json.loads(labels_path.read_text())
        choices = [
            x for x in regions(labels) if re.search(r"\d+(?:\.\d+)?\s*%", x["text"])
        ]
        if not choices:
            raise ValueError(
                f"No visible source-derived numeric percent line: {view['id']}"
            )
        line = choices[0]
        if "".join(line["text"].split()) in old_lines:
            raise ValueError(
                f"Exact normalized overlap with old evaluation: {view['id']}"
            )
        job = next(x for x in jobs if x["id"] == view["id"])
        src = sources[view["id"]]
        cases.append(
            dict(
                id=view["id"],
                source_kind="code",
                truth_origin="literal source + DOM geometry",
                split=view["split"],
                source_group=view["source_group"],
                hr=view["hr"],
                hr_sha256=view["hr_sha256"],
                labels=view["labels"],
                labels_sha256=view["labels_sha256"],
                line=line,
                content_category="percent_code",
                source_original=src["path"],
                source_sha256=src["sha256"],
                font_size=job["font_size"],
                rendering_width=1920,
            )
        )
    if len(cases) != 36 or len({c["id"] for c in cases}) != 36:
        raise ValueError("Expected 12 validation plus 24 independent test views")
    val_groups = {c["source_group"] for c in cases if c["split"] == "validation"}
    test_groups = {c["source_group"] for c in cases if c["split"] == "test"}
    if val_groups & test_groups:
        raise ValueError("Template family crossed split")
    if {c["source_group"] for c in old["cases"]} & (val_groups | test_groups):
        raise ValueError("Template source overlaps original evaluation")
    protocol = dict(
        schema="isharescreen_percent_code_holdout_v1",
        frozen=time.time(),
        runner_sha256=sha256(Path(__file__)),
        old_protocol_sha256=sha256(old_path),
        cases=cases,
        counts=dict(Counter(c["split"] for c in cases)),
        fixed_display_ids={
            split: [c["id"] for c in cases if c["split"] == split][:4]
            for split in ("validation", "test")
        },
        main_grid=old["main_grid"],
        codec=old["codec"],
        ocr=old["ocr"],
        metrics=[
            "numeric_percent_token_exact",
            "percent_symbol_recall",
            "equal_source_CER",
            "digit_edits",
            "punctuation_edits",
            "exact_line_rate",
            "PSNR",
            "SSIM",
        ],
        selection="One visible source-derived numeric-percent line per rendered view, chosen before OCR or model inference.",
        split_policy="Four validation and eight test template families, three values/font views each; no family crosses split or original frozen evaluation.",
        comparison="Same 540p/720p/1080p HEVC 600/1000 Kbps input and FHD output, baseline vs historical EDSR vs symbol-tuned EDSR; H100 pinned OCR.",
        limits="Synthetic code-like views only; no natural PDF, full-screen user study, live connection or Mac performance claim.",
    )
    atomic_json(root / "EVAL_PROTOCOL.json", protocol)
    atomic_json(
        root / "OVERLAP_AUDIT.json",
        dict(
            schema="isharescreen_percent_holdout_overlap_v1",
            protocol_sha256=sha256(root / "EVAL_PROTOCOL.json"),
            old_protocol_sha256=sha256(old_path),
            exact_normalized_line_overlap=0,
            old_source_group_overlap=0,
            validation_test_source_group_overlap=0,
            source_file_sha256={c["id"]: c["source_sha256"] for c in cases},
            result="passed",
            finished=time.time(),
        ),
    )
    print(
        "PERCENT HOLDOUT FROZEN",
        protocol["counts"],
        sha256(root / "EVAL_PROTOCOL.json"),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.campaign.resolve(), args.root.resolve())
