"""Durable public-source queue for the additional text/screen datasets."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
from urllib.request import Request, urlopen
import zipfile

from benchmarks.super_resolution.text_corpus.public_downloads import download
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def small(out, name, url, size=None, expected_sha=None, git_oid=None):
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_name(path.name + ".receipt.json")
    if receipt.exists():
        row = json.loads(receipt.read_text())
        assert sha256(path) == row["sha256"]
        return row
    with urlopen(url, timeout=60) as response:
        value = response.read(20 * 2**20 + 1)
    if len(value) > 20 * 2**20 or (size is not None and len(value) != size):
        raise ValueError("Small metadata download size differs")
    identity = hashlib.sha256(value).hexdigest()
    if expected_sha and identity != expected_sha:
        raise ValueError("Metadata SHA256 differs")
    if (
        git_oid
        and hashlib.sha1(
            b"blob " + str(len(value)).encode() + b"\0" + value
        ).hexdigest()
        != git_oid
    ):
        raise ValueError("Git metadata identity differs")
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(value)
    temporary.replace(path)
    row = dict(
        name=name,
        url=url,
        bytes=len(value),
        sha256=identity,
        upstream_sha256=expected_sha,
        git_blob_oid=git_oid,
        finished=time.time(),
    )
    atomic_json(receipt, row)
    return row


def plan(root):
    result = {}
    for name, repo in [
        ("websight", "HuggingFaceM4/WebSight"),
        ("docbank", "liminghao1630/DocBank"),
    ]:
        rows = json.loads((root / (name + "-files.upstream.json")).read_text())
        items = []
        for row in rows:
            if row["type"] != "file":
                continue
            relative = PurePosixPath(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe upstream path")
            items.append(
                dict(
                    name=row["path"],
                    url="https://huggingface.co/datasets/"
                    + repo
                    + "/resolve/main/"
                    + row["path"],
                    size=row["size"],
                    sha256=row.get("lfs", {}).get("oid"),
                    git_oid=None if "lfs" in row else row["oid"],
                )
            )
        result[name] = items
    release = json.loads((root / "xfund-files.upstream.json").read_text())
    result["xfund"] = [
        dict(
            name=a["name"],
            url=a["browser_download_url"],
            size=a["size"],
            sha256=(a.get("digest") or "").removeprefix("sha256:") or None,
        )
        for a in release["assets"]
    ]
    base = "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/"
    result["rico"] = [
        dict(name=name, url=base + name, size=None, sha256=None)
        for name in ["unique_uis.tar.gz", "ui_details.csv", "semantic_annotations.zip"]
    ]
    return result


def dataset(root, name, items):
    out = root / name
    out.mkdir(exist_ok=True)
    finished = []
    failed = []
    started = time.time()

    def one(item):
        path = PurePosixPath(item["name"])
        folder = out / path.parent
        if item["size"] is not None and item["size"] < 8 * 2**20:
            return small(
                folder,
                path.name,
                item["url"],
                item["size"],
                item["sha256"],
                item.get("git_oid"),
            )
        return download(
            folder, path.name, item["url"], item["size"], item["sha256"], workers=2
        )

    atomic_json(
        out / "queue-status.json",
        dict(
            phase="downloading",
            total_files=len(items),
            completed=[],
            failed=[],
            pid=os.getpid(),
            started=started,
        ),
    )
    with ThreadPoolExecutor(max_workers=3 if name == "websight" else 2) as pool:
        futures = {pool.submit(one, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                row = future.result()
                finished.append(
                    dict(file=item["name"], bytes=row["bytes"], sha256=row["sha256"])
                )
                print("DATASET FILE COMPLETE", name, item["name"], flush=True)
            except Exception as error:
                failed.append(dict(file=item["name"], error=repr(error)))
                print("DATASET FILE FAILED", name, failed[-1], flush=True)
            atomic_json(
                out / "queue-status.json",
                dict(
                    phase="downloading",
                    total_files=len(items),
                    completed=finished,
                    failed=failed,
                    pid=os.getpid(),
                    started=started,
                    updated=time.time(),
                ),
            )
    phase = "completed" if not failed else "partial_failure"
    receipt = dict(
        phase=phase,
        total_files=len(items),
        completed=finished,
        failed=failed,
        finished=time.time(),
    )
    atomic_json(out / "queue-status.json", receipt)
    if not failed:
        atomic_json(root / (name + "-DOWNLOADS-COMPLETED.json"), receipt)
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    jobs = plan(root)
    total = sum(i["size"] or 0 for group in jobs.values() for i in group)
    if shutil.disk_usage(root).free < total * 1.3:
        raise ValueError("Insufficient storage headroom")
    atomic_json(
        root / "extended-download-plan.json",
        dict(
            datasets=jobs,
            known_bytes=total,
            pid=os.getpid(),
            created=time.time(),
            scope="All public WebSight repository data versions; complete DocBank repository; all XFUND release assets; RICO unique UI screenshots/hierarchies, UI metadata and semantic labels. Interaction videos/animations are outside the text-screen intake.",
        ),
    )
    # Preserve each upstream notice separately from data; absence is recorded,
    # never interpreted as permission for a downstream commercial deployment.
    notices = [
        ("doclaynet", "https://raw.githubusercontent.com/DS4SD/DocLayNet/main/LICENSE"),
        (
            "pubtabnet",
            "https://raw.githubusercontent.com/ibm-aur-nlp/PubTabNet/master/LICENSE.md",
        ),
        (
            "docbank",
            "https://raw.githubusercontent.com/doc-analysis/DocBank/master/LICENSE",
        ),
        ("rico", "https://www.interactionmining.org/archive/rico/copyright.txt"),
    ]
    for name, url in notices:
        try:
            small(root / "notices", name + ".upstream.txt", url)
        except Exception as error:
            atomic_json(
                root / "notices" / (name + "-notice-error.json"),
                dict(url=url, error=repr(error)),
            )
    results = {}
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {
            pool.submit(dataset, root, name, items): name
            for name, items in jobs.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()
            atomic_json(
                root / "extended-queue-status.json",
                dict(results=results, pid=os.getpid(), updated=time.time()),
            )
    atomic_json(
        root / "extended-queue-finished.json",
        dict(results=results, finished=time.time()),
    )


if __name__ == "__main__":
    main()
