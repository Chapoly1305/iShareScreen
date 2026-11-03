"""Small source-bound DocLayNet intake, isolated from the current experiment.

Read explicitly selected PDF/JSON members using HTTP ranges, rather than
downloading the full archive. Retain original assets, license and content hashes.
Intake remains quarantined until original-document split mapping and overlap
checks are complete. It never updates training inputs or model checkpoints.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen
import zipfile

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256

ARCHIVE = "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_extra.zip"


class RangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.position = 0
        self.received = 0
        with urlopen(Request(url, method="HEAD"), timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers["ETag"]
            if response.headers.get("Accept-Ranges") != "bytes":
                raise ValueError("HTTP range reads unavailable")

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = (
            offset
            if whence == 0
            else self.position + offset
            if whence == 1
            else self.size + offset
        )
        if not 0 <= position <= self.size:
            raise ValueError("Invalid archive seek")
        self.position = position
        return position

    def read(self, size=-1):
        size = min(self.size - self.position, size if size >= 0 else self.size)
        if not size:
            return b""
        if size > 32 * 2**20:
            raise ValueError("Unexpectedly large archive range")
        start = self.position
        end = start + size - 1
        with urlopen(
            Request(
                self.url,
                headers={"Range": f"bytes={start}-{end}", "If-Match": self.etag},
            ),
            timeout=60,
        ) as response:
            if (
                response.status != 206
                or response.headers["Content-Range"]
                != f"bytes {start}-{end}/{self.size}"
            ):
                raise ValueError("Server did not honor exact byte range")
            value = response.read(size + 1)
        if len(value) != size:
            raise ValueError("Incomplete archive range")
        self.position += size
        self.received += size
        return value


def run(args):
    import fitz

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    for name, url in [
        (
            "README.upstream.md",
            "https://raw.githubusercontent.com/DS4SD/DocLayNet/main/README.md",
        ),
        (
            "LICENSE.upstream.txt",
            "https://raw.githubusercontent.com/DS4SD/DocLayNet/main/LICENSE",
        ),
    ]:
        with urlopen(url, timeout=30) as response:
            content = response.read(256 * 1024)
        (out / name).write_bytes(content)
    stream = RangeReader(ARCHIVE)
    rows = []
    seen = set()
    with zipfile.ZipFile(stream) as archive:
        names = set(archive.namelist())
        eligible = []
        for name in names:
            match = re.fullmatch(r"PDF/([0-9a-f]{64})\.pdf", name)
            if match and "JSON/" + match[1] + ".json" in names:
                eligible.append(match[1])
        chosen = sorted(
            eligible,
            key=lambda key: hashlib.sha256(
                ("ishare-public-pilot-682:" + key).encode()
            ).hexdigest(),
        )[: args.count]
        if len(chosen) != args.count:
            raise ValueError("Insufficient PDF/JSON pairs")
        for index, key in enumerate(chosen):
            folder = out / "sources" / key
            folder.mkdir(parents=True)
            for prefix, suffix in [("PDF", "pdf"), ("JSON", "json")]:
                member = f"{prefix}/{key}.{suffix}"
                info = archive.getinfo(member)
                if info.file_size > 20 * 2**20:
                    raise ValueError("Pilot member exceeds intake limit")
                (folder / ("source." + suffix)).write_bytes(archive.read(member))
            metadata = json.loads((folder / "source.json").read_text())
            with fitz.open(folder / "source.pdf") as document:
                if len(document) != 1:
                    raise ValueError("Expected source single-page PDF")
                page = document[0]
                words = page.get_text("words")
                text = page.get_text()
                normalized = " ".join(text.split())
                identity = hashlib.sha256(normalized.encode()).hexdigest()
                duplicate = identity in seen
                seen.add(identity)
                atomic_json(
                    folder / "pdf-words.json",
                    dict(
                        page_points=list(page.rect),
                        rotation=page.rotation,
                        words=[list(w) for w in words],
                        text=text,
                    ),
                )
                preview = None
                if index < 4:
                    width = 1920
                    clip = fitz.Rect(
                        0,
                        0,
                        page.rect.width,
                        min(page.rect.height, page.rect.width * 1080 / 1920),
                    )
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(
                            width / page.rect.width, width / page.rect.width
                        ),
                        clip=clip,
                        alpha=False,
                    )
                    preview = folder / "preview-screen.png"
                    pix.save(preview)
                rows.append(
                    dict(
                        id="doclaynet:" + key,
                        archive_pdf_member=f"PDF/{key}.pdf",
                        archive_json_member=f"JSON/{key}.json",
                        pdf=str((folder / "source.pdf").relative_to(out)),
                        pdf_sha256=sha256(folder / "source.pdf"),
                        text_cells_sha256=sha256(folder / "source.json"),
                        extracted_words=len(words),
                        characters=len(normalized),
                        text_sha256=identity,
                        duplicate_in_pilot=duplicate,
                        page_points=list(page.rect),
                        rotation=page.rotation,
                        upstream_json_fields=list(metadata)
                        if isinstance(metadata, dict)
                        else [],
                        preview=str(preview.relative_to(out)) if preview else None,
                        training_ready=False,
                        reason="Original-document split mapping and overlap audit pending; source intake only.",
                    )
                )
            atomic_json(
                out / "status.json",
                dict(
                    phase="collecting",
                    completed=len(rows),
                    total=args.count,
                    http_bytes=stream.received,
                    updated=time.time(),
                ),
            )
            print(
                "DOC SOURCE",
                index + 1,
                "/",
                args.count,
                "words",
                len(words),
                flush=True,
            )
    atomic_json(
        out / "manifest.json",
        dict(
            schema="isharescreen_public_text_intake_v1",
            upstream="https://github.com/DS4SD/DocLayNet",
            archive=ARCHIVE,
            archive_etag=stream.etag,
            archive_bytes=stream.size,
            http_bytes_received=stream.received,
            license_sha256=sha256(out / "LICENSE.upstream.txt"),
            readme_sha256=sha256(out / "README.upstream.md"),
            rows=rows,
            selected_before_inspecting_text=True,
            selection="First SHA256(ishare-public-pilot-682 + page hash); no OCR-success filtering.",
            scope="Public source intake/embedded-text audit only. Not training data until source grouping, split and cross-corpus overlap checks pass.",
            runner_sha256=sha256(Path(__file__)),
            elapsed_s=time.time() - started,
            finished=time.time(),
        ),
    )
    atomic_json(
        out / "COMPLETED.json",
        dict(
            pages=len(rows),
            with_embedded_text=sum(r["extracted_words"] >= 30 for r in rows),
            unique_texts=len(seen),
            manifest_sha256=sha256(out / "manifest.json"),
            finished=time.time(),
        ),
    )
    print(
        "PUBLIC TEXT INTAKE COMPLETE",
        len(rows),
        "pages; bytes downloaded",
        stream.received,
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=24)
    run(p.parse_args())
