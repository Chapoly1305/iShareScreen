"""Download the complete official TextZoom folder, retaining source hashes."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

import gdown
from benchmarks.super_resolution.text_corpus.public_downloads import download

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    files = json.loads((root / "textzoom-files.json").read_text())
    out = root / "textzoom"
    out.mkdir(exist_ok=True)
    started = time.time()
    completed = []
    errors = []

    def task(spec):
        target = Path(spec["local_path"]).resolve()
        target.relative_to(out.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        receipt = target.with_name(target.name + ".receipt.json")
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            assert sha256(target) == prior["sha256"]
            return prior
        transport = None
        try:
            result = gdown.download(
                id=spec["id"],
                output=str(target),
                quiet=True,
                use_cookies=False,
                resume=True,
            )
        except gdown.exceptions.FileURLRetrievalError:
            # Same publicly shared Drive file through Google's public download
            # endpoint; no cookies, account tokens or permission changes.
            url = (
                "https://drive.usercontent.google.com/download?id="
                + spec["id"]
                + "&export=download&confirm=t"
            )
            transport = download(target.parent, target.name, url, workers=2)
            result = str(target)
        if result is None:
            raise RuntimeError("Official Drive download unavailable: " + spec["path"])
        row = dict(
            id=spec["id"],
            path=spec["path"],
            file=str(target.relative_to(root.resolve())),
            bytes=target.stat().st_size,
            sha256=sha256(target),
            transport=transport,
            finished=time.time(),
        )
        if target.name == "data.mdb":
            import lmdb

            env = lmdb.open(
                str(target.parent),
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=2,
            )
            with env.begin() as tx:
                count = int(tx.get(b"num-samples"))
                row["samples"] = count
                for i in range(1, count + 1):
                    if any(
                        tx.get((prefix + "-%09d" % i).encode()) is None
                        for prefix in ["label", "image_hr", "image_lr"]
                    ):
                        raise ValueError("Incomplete LMDB record")
            env.close()
            row["all_sample_keys_verified"] = True
        atomic_json(receipt, row)
        return row

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(task, s): s for s in files}
        for future in as_completed(futures):
            try:
                row = future.result()
                completed.append(row)
                print("TEXTZOOM COMPLETE", row["path"], row["bytes"], flush=True)
            except Exception as error:
                errors.append(dict(file=futures[future]["path"], error=repr(error)))
                print("TEXTZOOM FAILED", errors[-1], flush=True)
            atomic_json(
                out / "status.json",
                dict(
                    phase="downloading" if not errors else "partial_failure",
                    completed=completed,
                    errors=errors,
                    total=len(files),
                    elapsed_s=time.time() - started,
                    updated=time.time(),
                ),
            )
    if errors:
        raise RuntimeError(
            "Some official TextZoom files could not be downloaded; successful files retained"
        )
    atomic_json(
        root / "textzoom-DOWNLOADS-COMPLETED.json",
        dict(
            files=completed,
            finished=time.time(),
            scope="Full official folder, all sample keys checked; upstream does not publish checksums. Local SHA256 values retained.",
        ),
    )


if __name__ == "__main__":
    main()
