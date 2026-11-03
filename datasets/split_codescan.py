"""Create repository- and hash-isolated CodeSCAN source splits, without training.

Raw files and upstream splits remain immutable. Exact duplicate color images
are represented once in the admitted source layout; all originals remain in
sources/codescan/raw. Other duplicate assets connect repository groups so no
file content, decoded PNG pixels or normalized complete source file crosses
the new train/validation/test boundaries.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image


ROLES = (
    "img.png",
    "bw.png",
    "source.txt",
    "annotations.json",
    "info.json",
    "page.html",
    "page_bw.html",
)
SPLITS = ("train", "validation", "test")
FRACTIONS = np.array([0.8, 0.1, 0.1])
DATASETS = Path(__file__).resolve().parent
SOURCE = DATASETS / "sources/codescan"
RAW = SOURCE / "raw/codescan"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def inspect_sample(image):
    folder = image.parent
    relative = folder.relative_to(RAW)
    language, repository, *_ = relative.parts
    files = {}
    stats = {}
    for name in ROLES:
        path = folder / name
        assert path.is_file() and not path.is_symlink(), path
        files[name] = digest(path)
        s = path.stat()
        stats[name] = [s.st_size, s.st_mtime_ns]
    pixels = {}
    sizes = {}
    alpha = {}
    for name in ("img.png", "bw.png"):
        with Image.open(folder / name) as image:
            rgba = image.convert("RGBA")
            alpha[name] = list(rgba.getchannel("A").getextrema())
            rgb = image.convert("RGB")
            sizes[name] = list(rgb.size)
            h = hashlib.sha256()
            h.update(f"RGB:{rgb.width}x{rgb.height}:".encode())
            h.update(rgb.tobytes())
            pixels[name] = h.hexdigest()
    text = (
        (folder / "source.txt")
        .read_bytes()
        .replace(b"\r\n", b"\n")
        .replace(b"\r", b"\n")
    )
    text = b"\n".join(line.rstrip(b" \t") for line in text.split(b"\n")).strip(b"\n")
    directory = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return dict(
        id=hashlib.sha256(str(relative).encode()).hexdigest()[:24],
        sample=str(relative),
        language=language,
        repository=repository.casefold().replace("~", "/"),
        files=files,
        stats=stats,
        pixels=pixels,
        sizes=sizes,
        alpha=alpha,
        normalized_source_sha256=hashlib.sha256(text).hexdigest(),
        source_empty=not text.strip(),
        directory_sha256=directory,
    )


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def inventory(args):
    release = SOURCE / args.release
    release.mkdir(parents=True, exist_ok=True)
    path = release / "inventory.jsonl"
    if path.exists():
        raise FileExistsError(
            "Inventory already exists; use split or verify without recomputing it."
        )
    images = sorted(RAW.glob("*/*/*/img.png"))
    assert len(images) == 12000
    started = time.time()
    with path.open("w") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(inspect_sample, images), 1):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 1000 == 0:
                out.flush()
                print("HASHED", index, "/", len(images), flush=True)
    write(
        release / "INVENTORY-COMPLETED.json",
        dict(
            samples=len(images),
            files=len(images) * len(ROLES),
            inventory_sha256=digest(path),
            elapsed_s=time.time() - started,
            complete=True,
        ),
    )


def load(release):
    receipt = json.loads((release / "INVENTORY-COMPLETED.json").read_text())
    assert receipt["inventory_sha256"] == digest(release / "inventory.jsonl")
    return [
        json.loads(s) for s in (release / "inventory.jsonl").read_text().splitlines()
    ]


def clusters(rows):
    uf = UnionFind(len(rows))
    first = {}
    seen_pixels = {}
    retained = []
    duplicates = []
    hash_counts = Counter()
    for i, row in enumerate(rows):
        keys = [
            ("repository", row["repository"]),
            ("directory", row["directory_sha256"]),
            ("source_normalized", row["normalized_source_sha256"]),
        ]
        keys += [("file", v) for v in row["files"].values()]
        keys += [("pixels", v) for v in row["pixels"].values()]
        for key in keys:
            if key in first:
                uf.join(i, first[key])
                hash_counts[key[0]] += 1
            else:
                first[key] = i
        pixel = row["pixels"]["img.png"]
        if pixel in seen_pixels:
            duplicates.append(
                dict(
                    sample=row["sample"],
                    canonical_sample=rows[seen_pixels[pixel]]["sample"],
                    reason="identical_decoded_color_pixels",
                    pixel_sha256=pixel,
                )
            )
        else:
            seen_pixels[pixel] = i
            retained.append(i)
    grouped = defaultdict(list)
    for i in retained:
        grouped[uf.find(i)].append(i)
    return uf, grouped, duplicates, dict(hash_counts)


def assign(rows, groups, seed):
    languages = sorted({r["language"] for r in rows})
    lang_index = {s: i for i, s in enumerate(languages)}
    vectors = {
        k: np.array(
            [sum(rows[i]["language"] == lang for i in v) for lang in languages],
            dtype=np.float64,
        )
        for k, v in groups.items()
    }
    totals = sum(vectors.values())
    targets = FRACTIONS[:, None] * totals[None, :]
    total_targets = FRACTIONS * totals.sum()
    current = np.zeros_like(targets)
    counts = np.zeros(3)
    result = {}
    order = sorted(
        groups,
        key=lambda k: (
            -len(groups[k]),
            hashlib.sha256(f"{seed}:{rows[k]['sample']}".encode()).hexdigest(),
        ),
    )
    for k in order:
        v = vectors[k]
        n = len(groups[k])
        cost = []
        for s in range(3):
            language_cost = np.sum(
                ((current[s] + v - targets[s]) ** 2 - (current[s] - targets[s]) ** 2)
                / np.maximum(targets[s], 1)
            )
            global_cost = (
                (counts[s] + n - total_targets[s]) ** 2
                - (counts[s] - total_targets[s]) ** 2
            ) / max(total_targets[s], 1)
            cost.append(language_cost + global_cost)
        chosen = min(range(3), key=lambda s: (cost[s], s))
        result[k] = SPLITS[chosen]
        current[chosen] += v
        counts[chosen] += n
    return result


def split(args):
    release = SOURCE / args.release
    if (release / "PLAN.json").exists():
        raise FileExistsError("Frozen split already exists; run verify instead.")
    rows = load(release)
    assert len(rows) == 12000 and len({r["sample"] for r in rows}) == 12000
    uf, groups, duplicates, collisions = clusters(rows)
    assignment = assign(rows, groups, args.seed)
    assigned = {i: assignment[k] for k, values in groups.items() for i in values}
    kept = set(assigned)
    all_members = defaultdict(list)
    for i, row in enumerate(rows):
        all_members[uf.find(i)].append(row["sample"])
    group_id = {
        k: hashlib.sha256("\n".join(sorted(all_members[k])).encode()).hexdigest()[:24]
        for k in groups
    }
    manifests = {s: [] for s in SPLITS}
    for i in sorted(kept):
        row = rows[i]
        s = assigned[i]
        folder = DATASETS / s / "codescan"
        relative = Path("samples") / row["sample"]
        link = folder / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            raise FileExistsError(link)
        link.symlink_to(
            os.path.relpath(RAW / row["sample"], link.parent), target_is_directory=True
        )
        manifests[s].append(
            dict(
                row,
                split=s,
                group_id=group_id[uf.find(i)],
                directory=str(relative),
                source_image=str(relative / "img.png"),
                black_white_image=str(relative / "bw.png"),
                source_text=str(relative / "source.txt"),
                annotations=str(relative / "annotations.json"),
            )
        )
    # The temporary intake entry must not expose the full archive under train.
    intake = SOURCE / "intake-records"
    intake.mkdir(exist_ok=True)
    entry = DATASETS / "train/codescan"
    for name in (
        "incoming.json",
        "ARCHIVE-VERIFIED.json",
        "UPSTREAM-SPLIT-INSPECTION.json",
        "README.md",
    ):
        p = entry / name
        if p.exists():
            target = intake / name
            if target.exists():
                raise FileExistsError(target)
            p.rename(target)
    archive_link = entry / "source.zip"
    if archive_link.is_symlink():
        assert archive_link.resolve() == (SOURCE / "codescan.zip").resolve()
        archive_link.unlink()
    counts = {s: len(v) for s, v in manifests.items()}
    plan = dict(
        schema="isharescreen_codescan_grouped_split_v1",
        seed=args.seed,
        target_fractions=dict(zip(SPLITS, FRACTIONS.tolist())),
        raw_samples=len(rows),
        retained_samples=len(kept),
        excluded_identical_color_images=len(duplicates),
        original_repositories=len({r["repository"] for r in rows}),
        connected_groups=len(groups),
        largest_group=max(map(len, groups.values())),
        counts=counts,
        grouping="Canonical repository across all languages, plus any identical file SHA256, decoded RGB image, normalized full source text, or full sample-directory digest.",
        source_normalization="CRLF/CR to LF; remove trailing spaces/tabs per line and outer blank lines; preserve indentation and source tokens.",
        duplicate_policy="Retain lexicographically first sample for identical decoded color image; preserve every original in immutable raw corpus.",
        inventory_sha256=digest(release / "inventory.jsonl"),
        archive_sha256=json.loads((SOURCE / "SOURCE.json").read_text())[
            "archive_sha256"
        ],
        training_ready=False,
        training_hold=True,
    )
    for s, values in manifests.items():
        folder = DATASETS / s / "codescan"
        write(
            folder / "manifest.json",
            dict(
                schema="isharescreen_codescan_source_split_v1",
                split=s,
                count=len(values),
                language_counts=dict(Counter(r["language"] for r in values)),
                source_release=str(release.relative_to(DATASETS)),
                training_ready=False,
                codec_pairs_generated=False,
                rows=values,
            ),
        )
        (folder / "README.md").write_text(
            f"# CodeSCAN {s}\n\n{len(values):,} unique color screenshot samples. Target fraction: {dict(zip(SPLITS, FRACTIONS.tolist()))[s]:.0%}.\n\nUse manifest.json; samples/ contains only this split. All files from a repository or exact-duplicate component stay in one split. Original archive, upstream splits and excluded duplicates remain under datasets/sources/codescan/.\n\nThese are source screenshots and annotations, not admitted 720p/600kbps/420 LR/HR pairs. Training remains on hold.\n"
        )
    write(release / "duplicates-excluded.json", duplicates)
    write(release / "PLAN.json", plan)
    write(
        DATASETS / "codescan-splits.json",
        dict(
            schema="isharescreen_codescan_split_catalog_v1",
            release=str(release.relative_to(DATASETS)),
            splits={
                s: dict(manifest=f"{s}/codescan/manifest.json", count=counts[s])
                for s in SPLITS
            },
            training_ready=False,
            training_hold=True,
        ),
    )
    print(json.dumps(plan, indent=2), flush=True)


def verify(args):
    release = SOURCE / args.release
    plan = json.loads((release / "PLAN.json").read_text())
    inventories = load(release)
    expected = {r["sample"]: r for r in inventories}
    sets = {}
    counts = {}
    within = {}
    checked = 0
    for s in SPLITS:
        folder = DATASETS / s / "codescan"
        manifest = json.loads((folder / "manifest.json").read_text())
        rows = manifest["rows"]
        assert manifest["count"] == len(rows)
        assert (
            len(rows)
            == len({r["sample"] for r in rows})
            == len({r["pixels"]["img.png"] for r in rows})
        )
        files = set()
        pixels = set()
        paths = set()
        directories = set()
        repos = set()
        groups = set()
        normalized = set()
        dir_hashes = set()
        for row in rows:
            original = expected[row["sample"]]
            assert (
                row["files"] == original["files"]
                and row["pixels"] == original["pixels"]
            )
            link = folder / row["directory"]
            assert (
                link.is_symlink() and link.resolve() == (RAW / row["sample"]).resolve()
            )
            assert link.resolve() not in paths
            paths.add(link.resolve())
            directories.add(row["sample"])
            repos.add(row["repository"])
            groups.add(row["group_id"])
            normalized.add(row["normalized_source_sha256"])
            dir_hashes.add(row["directory_sha256"])
            for name in ROLES:
                path = link / name
                assert digest(path) == row["files"][name], path
                files.add(row["files"][name])
                checked += 1
            pixels.update(row["pixels"].values())
        within[s] = dict(
            samples=len(rows),
            unique_color_images=len({r["pixels"]["img.png"] for r in rows}),
            unique_file_hashes=len(files),
        )
        counts[s] = len(rows)
        sets[s] = dict(
            file_sha256=files,
            decoded_rgb=pixels,
            sample_path=directories,
            resolved_source_directory=paths,
            repository=repos,
            connected_group=groups,
            normalized_source=normalized,
            directory_sha256=dir_hashes,
        )
    overlaps = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = {key: len(sets[a][key] & sets[b][key]) for key in sets[a]}
        assert not any(overlap.values()), (a, b, overlap)
        overlaps[a + "__" + b] = overlap
    assert sum(counts.values()) == plan["retained_samples"]
    audit = dict(
        passed=True,
        counts=counts,
        retained_samples=sum(counts.values()),
        raw_samples=12000,
        excluded_identical_color_images=plan["excluded_identical_color_images"],
        actual_files_rehashed=checked,
        cross_split_overlaps=overlaps,
        within_split=within,
        manifests_sha256={
            s: digest(DATASETS / s / "codescan/manifest.json") for s in SPLITS
        },
        inventory_sha256=digest(release / "inventory.jsonl"),
        training_hold=True,
        limits="Exact bytes, decoded RGB and normalized complete-source isolation; not proof of no near-duplicate snippets, fork similarity, shared short passages or unseen pretraining overlap.",
    )
    write(release / "AUDIT.json", audit)
    (release / "REPORT.md").write_text(
        "# CodeSCAN source split audit\n\n"
        + json.dumps(audit, ensure_ascii=False, indent=2)
        + "\n"
    )
    p = SOURCE / "SOURCE.json"
    source = json.loads(p.read_text())
    source.update(
        status="source_splits_verified_codec_pairs_pending",
        split_release=args.release,
        new_split_counts=counts,
        split_audit=str((release / "AUDIT.json").relative_to(SOURCE)),
        training_ready=False,
        training_hold=True,
    )
    write(p, source)
    print(json.dumps(audit, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("inventory", "split", "verify"))
    parser.add_argument("--release", default="split-v1")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=682)
    args = parser.parse_args()
    globals()[args.action](args)


if __name__ == "__main__":
    main()
