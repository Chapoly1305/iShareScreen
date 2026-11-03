"""Freeze held-out code/path and source-document HTML reader cases."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

from benchmarks.super_resolution.text_corpus.ocr_protocol import regions
from benchmarks.super_resolution.text_corpus.render import html_views
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


TEMPLATES = {
    "validation": ("settings", "migration", "query", "routes"),
    "test": ("parser", "filesystem", "process", "api"),
}


def order(value: object) -> str:
    return hashlib.sha256(
        ("isharescreen-html-independent-v1:" + str(value)).encode()
    ).hexdigest()


def code_lines(template: str, index: int) -> str:
    rng = random.Random(int(order((template, index))[:16], 16))
    rows = [
        "from pathlib import Path",
        "import json",
        "import re",
        "",
        f"# Frozen source template: {template}; document: {index:03d}",
        "",
    ]
    for number in range(50):
        tenant = rng.randrange(1000, 9999)
        part = rng.randrange(100000, 999999)
        path = f"/srv/tenant_{tenant}/archive/2026-10/item_{part:06d}.json"
        key = f"k_{rng.randrange(16**8):08x}"
        if template == "settings":
            line = f'CONFIG_{number:02d} = {{"path": "{path}", "retry": {number % 7}, "key": "{key}"}}'
        elif template == "migration":
            line = f"STEP_{number:02d} = \"UPDATE records SET path='{path}' WHERE shard={number % 17} AND id={part};\""
        elif template == "query":
            line = f"QUERY_{number:02d} = f\"SELECT key, value FROM cache_{tenant} WHERE id={part} AND tag='{key}'\""
        elif template == "routes":
            line = f'ROUTE_{number:02d} = ("/api/v2/tenant/{tenant}/items/{part}", "GET", "{key}")'
        elif template == "parser":
            line = f'PATTERN_{number:02d} = re.compile(r"^(?P<node>{tenant})/(?P<item>{part})/(?P<key>{key})$")'
        elif template == "filesystem":
            line = (
                f'TARGET_{number:02d} = Path("{path}").with_suffix(".{number % 9}.bak")'
            )
        elif template == "process":
            line = f'COMMAND_{number:02d} = ["/usr/bin/tool", "--source", "{path}", "--attempt", "{number % 8}"]'
        elif template == "api":
            line = f'RESPONSE_{number:02d} = {{"uri": "https://node-{tenant}.example.test/v2/{part}", "etag": "{key}"}}'
        else:
            raise ValueError(template)
        rows.append(line)
    return "\n".join(rows) + "\n"


def freeze(root: Path, pdf_root: Path) -> None:
    if (root / "independent-html-protocol.json").exists():
        raise FileExistsError("Independent HTML protocol is already frozen")
    pdf = json.loads((pdf_root / "independent-pdf-protocol.json").read_text())
    sources = {}
    jobs = []
    for split, templates in TEMPLATES.items():
        for index in range(8 if split == "validation" else 16):
            template = templates[index % len(templates)]
            identifier = f"code-{split}-{index:02d}"
            path = root / "sources" / (identifier + ".py")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(code_lines(template, index))
            sources[identifier] = dict(
                path=str(path.relative_to(root)), title=path.name, sha256=sha256(path)
            )
            jobs.append(
                dict(
                    id=identifier,
                    kind="code",
                    split=split,
                    source_group=f"new-code-template:{template}",
                    source=identifier,
                    template=template,
                    line_start=0,
                    font_size=(14, 16, 18, 20)[index % 4],
                    theme="dark" if index % 3 == 0 else "light",
                )
            )
        pages = sorted(
            (row for row in pdf["cases"] if row["split"] == split),
            key=lambda row: order(row["source_group"]),
        )
        for index, row in enumerate(pages[: 4 if split == "validation" else 8]):
            identifier = f"web-{split}-{index:02d}"
            # root is the campaign root; the selected original PDF is its child.
            relative = (pdf_root / row["pdf"]).relative_to(root)
            sources[identifier] = dict(
                path=str(relative),
                title=row["original_filename"],
                sha256=row["pdf_sha256"],
            )
            jobs.append(
                dict(
                    id=identifier,
                    kind="web",
                    split=split,
                    source_group=row["source_group"],
                    source=identifier,
                    page=1,
                    variant=30000 + index,
                    font_size=(14, 16, 18, 20)[index % 4],
                    theme="light",
                )
            )
    cases = []
    for view in html_views(root, jobs, sources):
        labels = json.loads((root / view["labels"]).read_text())
        eligible = regions(labels)
        if not eligible:
            raise ValueError(f"No source-derived visible line: {view['id']}")
        chosen = eligible[0]
        cases.append(
            dict(
                **view,
                line=chosen,
                source_file=sources[labels["source"]]["path"],
                source_sha256=sources[labels["source"]]["sha256"],
                source_truth="Literal source strings and browser-measured DOM geometry; no OCR prediction",
                font_size=labels["font_size"],
                template=labels.get("template"),
            )
        )
    if len(cases) != len(jobs):
        raise ValueError("Missing HTML cases")
    if set(TEMPLATES["validation"]) & set(TEMPLATES["test"]):
        raise ValueError("Code templates cross validation/test")
    atomic_json(
        root / "independent-html-protocol.json",
        dict(
            schema="isharescreen_low_bandwidth_independent_html_v1",
            pdf_protocol_sha256=sha256(pdf_root / "independent-pdf-protocol.json"),
            runner_sha256=sha256(Path(__file__)),
            cases=cases,
            counts={
                f"{split}/{kind}": count
                for (split, kind), count in Counter(
                    (row["split"], row["kind"]) for row in cases
                ).items()
            },
            template_policy="Code template families are disjoint by split; each family and all source variants stay within one split. Web reader views inherit the original PDF source group/split.",
            web_scope="Synthetic browser reader layout of unseen source PDFs, not a captured natural website.",
            selection="Fixed source/template hash and first eligible DOM-derived visible line before OCR/model scoring.",
            finished=time.time(),
        ),
    )
    (root / "freeze-html-runner.py").write_bytes(Path(__file__).read_bytes())
    print(
        "FROZEN HTML",
        len(cases),
        dict(Counter((r["split"], r["kind"]) for r in cases)),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pdf-root", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.root.resolve(), args.pdf_root.resolve())


if __name__ == "__main__":
    main()
