"""Freeze read-only Zotero PDF attachments and selected local source code.

No PDF text is treated as an instruction. Original library files are never
modified. Only copied, hashed sources are subsequently rendered.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import unicodedata
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256

API = "http://127.0.0.1:23119/api/users/0"
SCHEMA = "isharescreen_text_sources_v1"


def get(route):
    with urlopen(
        Request(API + route, headers={"Zotero-API-Version": "3"}), timeout=30
    ) as response:
        body = response.read().decode("utf-8")
        return (
            json.loads(body)
            if "json" in response.headers.get("Content-Type", "")
            else body
        )


def text_identity(text):
    return " ".join(
        unicodedata.normalize("NFKC", html.unescape(text)).casefold().split()
    )


def document_family(title, doi, fallback):
    title = text_identity(re.sub("<[^>]+>", "", title))
    title = re.sub(r"[_\W]+", " ", title)
    # Versions and sections of a protocol share substantial material.
    if re.search(r"\bmatter\b", title) and re.search(
        r"specification|namespace|device.library|core.spec|application.cluster", title
    ):
        return "spec:matter"
    if re.search(r"zig.?bee|green power", title) and re.search(
        r"specification|functionality|feature|device", title
    ):
        return "spec:zigbee"
    if re.search(r"bluetooth", title) and re.search(
        r"specification|core|profile", title
    ):
        return "spec:bluetooth"
    if "homekit" in title and "specification" in title:
        return "spec:homekit"
    if "aliro" in title and "specification" in title:
        return "spec:aliro"
    if (
        "cluster library specification" in title
        or "base device behavior specification" in title
    ):
        return "spec:zigbee"
    if doi:
        return "doi:" + doi.strip().lower().removeprefix("https://doi.org/")
    normalized = re.sub(r"[^\w]+", " ", title).strip()
    return "title:" + normalized if len(normalized) > 20 else "item:" + fallback


def split_for(group, seed=682):
    value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else "validation" if value < 90 else "test"


class Union:
    def __init__(self):
        self.parents = {}

    def find(self, x):
        self.parents.setdefault(x, x)
        if self.parents[x] != x:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parents[max(a, b)] = min(a, b)


def freeze_file(source, root, suffix):
    digest = sha256(source)
    relative = Path("sources") / suffix / (digest + source.suffix.lower())
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(source, target)
    if sha256(target) != digest:
        raise ValueError("Frozen source copy differs")
    return str(relative), digest


def build(args):
    import pymupdf

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "sources.json").exists():
        raise ValueError(
            "Source catalogue already frozen; use a fresh corpus directory"
        )
    items = []
    for start in range(0, 100000, 100):
        batch = get(f"/items?limit=100&start={start}&format=json")
        if not isinstance(batch, list):
            raise ValueError("Unexpected Zotero response")
        # Notes and embedded instructions are not corpus sources.
        items.extend(v for v in batch if v.get("data", {}).get("itemType") != "note")
        if len(batch) < 100:
            break
    else:
        raise ValueError("Zotero pagination did not terminate")
    by_key = {v["key"]: v for v in items}
    rows = []
    excluded = []
    union = Union()
    page_owners = {}
    seen_files = {}
    attachments = [
        v
        for v in items
        if v.get("data", {}).get("itemType") == "attachment"
        and v["data"].get("contentType") == "application/pdf"
    ]
    for n, item in enumerate(attachments, 1):
        data = item["data"]
        key = item["key"]
        parent = by_key.get(data.get("parentItem"), item)["data"]
        title = parent.get("title") or data.get("title") or key
        try:
            url = item.get("links", {}).get("enclosure", {}).get("href") or get(
                f"/items/{key}/file/view/url"
            )
            if not isinstance(url, str) or urlparse(url).scheme != "file":
                raise ValueError("No local PDF attachment")
            path = Path(unquote(urlparse(url).path))
            relative, digest = freeze_file(path, root, "pdf")
            family = document_family(
                title, parent.get("DOI", ""), data.get("parentItem", key)
            )
            union.find(family)
            if digest in seen_files:
                union.join(family, seen_files[digest])
                excluded.append(
                    dict(kind="pdf", key=key, reason="duplicate_file", sha256=digest)
                )
                continue
            pages = []
            with pymupdf.open(root / relative) as doc:
                if doc.needs_pass:
                    raise ValueError("Encrypted PDF")
                for index, page in enumerate(doc):
                    text = page.get_text("text", sort=True)
                    normalized = text_identity(text)
                    chars = sum(not c.isspace() for c in text)
                    if chars < 300 or text.count("\ufffd") > max(2, chars * 0.002):
                        continue
                    identity = hashlib.sha256(normalized.encode()).hexdigest()
                    if chars >= 500 and identity in page_owners:
                        union.join(family, page_owners[identity])
                    elif chars >= 500:
                        page_owners[identity] = family
                    pages.append(
                        dict(
                            page=index + 1,
                            characters=chars,
                            text_sha256=identity,
                            width=float(page.rect.width),
                            height=float(page.rect.height),
                        )
                    )
                count = len(doc)
            if not pages:
                raise ValueError("No usable text-layer pages")
            kind = (
                "specification"
                if family.startswith("spec:") or "specification" in title.lower()
                else "paper"
            )
            rows.append(
                dict(
                    id="pdf-" + digest[:20],
                    kind=kind,
                    path=relative,
                    sha256=digest,
                    source_group=family,
                    title=title,
                    zotero_key=key,
                    parent_key=data.get("parentItem"),
                    doi=parent.get("DOI"),
                    original_path=str(path),
                    page_count=count,
                    pages=pages,
                    truth="PDF text layer with geometric word boxes at rendering; not OCR labels",
                )
            )
            seen_files[digest] = family
        except (OSError, ValueError, RuntimeError) as error:
            excluded.append(dict(kind="pdf", key=key, title=title, reason=str(error)))
        if n % 20 == 0:
            print(
                "PDF SOURCES",
                n,
                "/",
                len(attachments),
                "accepted",
                len(rows),
                flush=True,
            )
    code_seen = {}
    for source_root in args.code_roots:
        source_root = source_root.resolve()
        repo = source_root.parent.name
        paths = subprocess.check_output(
            ["rg", "--files", str(source_root)], text=True
        ).splitlines()
        for filename in sorted(paths):
            path = Path(filename)
            rel = path.relative_to(source_root)
            if path.suffix.lower() not in (
                ".py",
                ".cpp",
                ".h",
                ".c",
                ".js",
                ".ts",
                ".rs",
                ".go",
                ".xml",
                ".md",
            ):
                continue
            if any(
                p in ("third_party", "out", "vendor", "generated", "__pycache__")
                for p in rel.parts
            ):
                continue
            if path.stat().st_size > 160000 or path.is_symlink():
                continue
            try:
                text = path.read_text("utf-8")
            except (OSError, UnicodeError):
                continue
            if len(text) < 700 or len(text.splitlines()) < 18:
                continue
            # Only source directories are read, never credentials/configuration stores.
            relative, digest = freeze_file(path, root, "code")
            family = (
                "code:"
                + repo
                + ":"
                + ("/".join(rel.parts[:2]) if len(rel.parts) > 2 else rel.parts[0])
            )
            union.find(family)
            if digest in code_seen:
                union.join(family, code_seen[digest])
                continue
            code_seen[digest] = family
            rows.append(
                dict(
                    id="code-" + digest[:20],
                    kind="code",
                    path=relative,
                    sha256=digest,
                    source_group=family,
                    title=f"{repo}/{rel}",
                    original_path=str(path),
                    lines=len(text.splitlines()),
                    language=path.suffix[1:],
                    truth="Literal source text and browser DOM geometry",
                )
            )
    for row in rows:
        row["source_group"] = union.find(row["source_group"])
        row["split"] = split_for(row["source_group"], args.seed)
    result = dict(
        schema=SCHEMA,
        created=time.time(),
        seed=args.seed,
        sources=rows,
        excluded=excluded,
        zotero_api=API,
        zotero_pdf_attachments=len(attachments),
        grouping="DOI/title/protocol family, exact source bytes and shared full-page text; code repository/subdirectory with exact-file unions.",
        split_policy="Deterministic 80/10/10 source-group assignment before viewport generation; versions/pages/crops stay together.",
        source_code_sha256=sha256(Path(__file__)),
        pymupdf_version=pymupdf.__version__,
        counts=dict(Counter(r["kind"] for r in rows)),
        split_counts=dict(Counter(r["split"] for r in rows)),
        eligible_pdf_pages=sum(len(r.get("pages", [])) for r in rows),
    )
    atomic_json(root / "sources.json", result)
    print(
        json.dumps(
            {k: result[k] for k in ["counts", "split_counts", "eligible_pdf_pages"]},
            indent=2,
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=682)
    p.add_argument(
        "--code-roots",
        nargs="+",
        type=Path,
        default=[Path("src"), Path("/Volumes/Tools/connectedhomeip/src")],
    )
    build(p.parse_args())


if __name__ == "__main__":
    main()
