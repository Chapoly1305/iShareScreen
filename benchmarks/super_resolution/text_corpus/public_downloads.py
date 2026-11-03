"""Resumable full public archives with range, size, hash and archive checks."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen
import zipfile

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def download(out, name, url, expected=None, upstream_sha256=None, workers=4):
    out.mkdir(parents=True, exist_ok=True)
    state = out / (name + ".status.json")
    receipt = out / (name + ".receipt.json")
    target = out / name
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["url"] != url or sha256(target) != saved["sha256"]:
            raise ValueError("Previously completed download differs")
        return saved
    with urlopen(Request(url, headers={"Range": "bytes=0-0"}), timeout=60) as response:
        if response.status != 206:
            raise ValueError("Range download is required for this large archive")
        size = int(response.headers["Content-Range"].split("/")[-1])
        etag = response.headers.get("ETag")
        response.read(1)
    if expected is not None and size != expected:
        raise ValueError("Upstream file size differs")
    parts = out / (name + ".parts")
    parts.mkdir(exist_ok=True)
    binding = dict(url=url, size=size, etag=etag, chunk_bytes=64 * 2**20)
    manifest = parts / "source.json"
    if manifest.exists() and json.loads(manifest.read_text()) != binding:
        raise ValueError("Partial download belongs to another object")
    atomic_json(manifest, binding)
    lock = threading.Lock()
    started = time.time()
    completed = 0

    def status(phase, **extra):
        atomic_json(
            state,
            dict(
                phase=phase,
                received_bytes=completed,
                total_bytes=size,
                elapsed_s=time.time() - started,
                updated=time.time(),
                **extra,
            ),
        )

    chunks = [
        (i, start, min(size - start, binding["chunk_bytes"]))
        for i, start in enumerate(range(0, size, binding["chunk_bytes"]))
    ]

    def fetch(item):
        index, start, length = item
        path = parts / f"{index:06d}.bin"
        if path.exists():
            if path.stat().st_size != length:
                raise ValueError("Completed part size differs")
            return length
        temporary = path.with_suffix(".partial")
        for attempt in range(5):
            try:
                headers = {"Range": f"bytes={start}-{start + length - 1}"}
                if etag:
                    headers["If-Match"] = etag
                with urlopen(Request(url, headers=headers), timeout=90) as response:
                    if (
                        response.status != 206
                        or response.headers["Content-Range"]
                        != f"bytes {start}-{start + length - 1}/{size}"
                    ):
                        raise ValueError("Invalid partial response")
                    count = 0
                    with temporary.open("wb") as stream:
                        while block := response.read(2**20):
                            stream.write(block)
                            count += len(block)
                    if count != length:
                        raise IOError("Truncated byte range")
                temporary.replace(path)
                return length
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)

    status("downloading")
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(fetch, item) for item in chunks]):
                completed += future.result()
                with lock:
                    status("downloading")
                print(
                    name,
                    round(completed / size * 100, 1),
                    "%",
                    round(completed / 2**30, 2),
                    "GiB",
                    flush=True,
                )
        status("assembling")
        digest = hashlib.sha256()
        temporary = out / (name + ".assembling")
        with temporary.open("wb") as stream:
            for index, start, length in chunks:
                with (parts / f"{index:06d}.bin").open("rb") as chunk:
                    while block := chunk.read(4 * 2**20):
                        stream.write(block)
                        digest.update(block)
        identity = digest.hexdigest()
        if temporary.stat().st_size != size or (
            upstream_sha256 and identity != upstream_sha256
        ):
            raise ValueError("Archive size or upstream SHA256 differs")
        temporary.replace(target)
        status("checking_archive", sha256=identity)
        archive_check = None
        if name.endswith(".zip"):
            with zipfile.ZipFile(target) as archive:
                bad = archive.testzip()
                if bad is not None:
                    raise ValueError("ZIP CRC failed: " + bad)
                archive_check = dict(
                    kind="ZIP CRC of every member", members=len(archive.infolist())
                )
        saved = dict(
            name=name,
            url=url,
            bytes=size,
            etag=etag,
            sha256=identity,
            upstream_sha256=upstream_sha256,
            upstream_sha256_matched=bool(upstream_sha256),
            archive_check=archive_check,
            finished=time.time(),
        )
        atomic_json(receipt, saved)
        status("completed", sha256=identity)
        # Completed byte ranges are redundant after the final hash/CRC receipt.
        for index, _, _ in chunks:
            (parts / f"{index:06d}.bin").unlink()
        return saved
    except BaseException as error:
        status("failed", error=repr(error))
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", choices=["doclaynet", "pubtabnet"], required=True)
    a = p.parse_args()
    if a.dataset == "doclaynet":
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [
                pool.submit(
                    download,
                    a.output / "doclaynet",
                    f"DocLayNet_{kind}.zip",
                    f"https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_{kind}.zip",
                )
                for kind in ["core", "extra"]
            ]
            results = [j.result() for j in jobs]
    else:
        tree = json.loads((a.output / "pubtabnet-hf-tree.json").read_text())
        entry = next(e for e in tree if e.get("path") == "pubtabnet.tar.gz")
        results = [
            download(
                a.output / "pubtabnet",
                "pubtabnet.tar.gz",
                "https://huggingface.co/datasets/ajimeno/PubTabNet/resolve/main/pubtabnet.tar.gz",
                entry["size"],
                entry["lfs"]["oid"],
                workers=4,
            )
        ]
    atomic_json(a.output / (a.dataset + "-DOWNLOADS-COMPLETED.json"), results)


if __name__ == "__main__":
    main()
