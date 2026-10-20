"""Prepare denser, aligned candidates without changing the frozen training dataset."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

from benchmarks.super_resolution.folds import validate
from benchmarks.super_resolution.training_data import validate_pairs
from isharescreen.enhancement.models import sha256


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Media path escapes dataset root")
    return path


def candidate_indices(count, every, anchors=()):
    if type(count) is not int or count < 1 or type(every) is not int or every < 1:
        raise ValueError("Frame count and candidate spacing must be positive integers")
    if any(type(i) is not int or not 0 <= i < count for i in anchors):
        raise ValueError("Invalid baseline frame")
    return sorted(set(range(0, count, every)) | set(anchors))


def prepare(args):
    from PIL import Image
    # Reuse the production offline alignment/count/timestamp checks.
    from benchmarks.super_resolution.video_benchmark import aligned_frames

    started = time.monotonic()
    plan = validate(json.loads(args.folds.read_text()))
    baseline = json.loads(args.baseline.read_text())
    manifest = json.loads(args.manifest.read_text())
    manifest_hash, folds_hash = sha256(args.manifest), sha256(args.folds)
    if (manifest_hash != plan["manifest_sha256"]
            or baseline["manifest_sha256"] != manifest_hash
            or baseline["folds_sha256"] != folds_hash):
        raise ValueError("Frozen source/baseline/folds provenance mismatch")
    validate_pairs(baseline, plan)
    membership = {c["id"]: c for c in plan["clips"]}
    if {c["id"] for c in manifest["clips"]} != set(membership):
        raise ValueError("Source clips differ from frozen folds")
    anchors = {(r["capture"], r["frame"]): r for r in baseline["pairs"]}
    # Validate arguments before creating any output.
    for clip in manifest["clips"]:
        candidate_indices(clip["frames"], args.every,
                          [i for capture, i in anchors if capture == clip["id"]])
    args.output.mkdir(parents=True, exist_ok=False)

    def extract(item):
        index, clip = item
        paths = {}
        for role in ("reference", "low"):
            path = inside(args.manifest.parent, clip[role])
            if sha256(path) != clip[f"{role}_sha256"]:
                raise ValueError(f"Changed source media: {clip['id']}/{role}")
            paths[role] = path
        wanted = set(candidate_indices(clip["frames"], args.every,
                     [i for capture, i in anchors if capture == clip["id"]]))
        folder = args.output / f"clip-{index:03d}"
        folder.mkdir()
        rows = []
        for i, hr, lr in aligned_frames(paths["reference"], paths["low"],
                                        clip["frames"], Fraction(clip["fps"])):
            if i not in wanted:
                continue
            anchor = anchors.get((clip["id"], i))
            row = dict(capture=clip["id"], frame=i,
                       group=membership[clip["id"]]["group"],
                       fold=membership[clip["id"]]["fold"],
                       content=clip.get("content") or "unknown",
                       time_s=float(Fraction(i, 1) / Fraction(clip["fps"])),
                       baseline=anchor is not None)
            for role, rgb in (("hr", hr), ("lr", lr)):
                digest = hashlib.sha256(rgb.tobytes()).hexdigest()
                if anchor and anchor[f"{role}_rgb_sha256"] != digest:
                    raise ValueError("Re-decoded baseline pixels changed")
                path = folder / f"{i:06d}-{role}.png"
                Image.fromarray(rgb).save(path, compress_level=1)
                row[role] = str(path.relative_to(args.output))
                row[f"{role}_sha256"] = sha256(path)
                row[f"{role}_rgb_sha256"] = digest
            rows.append(row)
        print(f"CANDIDATES {index + 1}/{len(manifest['clips'])} {clip['id']}: {len(rows)}", flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        groups = list(pool.map(extract, enumerate(manifest["clips"])))
    rows = [row for group in groups for row in group]
    data = dict(schema="sr_curation_candidates_v1", candidates=rows,
                every=args.every, source_manifest=str(args.manifest.resolve()),
                manifest_sha256=manifest_hash, folds_sha256=folds_hash,
                baseline_sha256=sha256(args.baseline),
                baseline_manifest=str(args.baseline.resolve()),
                generator_sha256=sha256(Path(__file__)),
                alignment_generator_sha256=sha256(Path(__file__).with_name("video_benchmark.py")),
                elapsed_s=time.monotonic() - started,
                reference="Lossless PNG of the existing decoded RGB8 proxy; not lossless acquisition",
                frame_indexing="zero-based source decoded frame index; time_s = frame / fps",
                training_ready=False)
    (args.output / "candidates.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"PREPARED {len(rows)} candidates; preserved {sum(r['baseline'] for r in rows)} baseline frames", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--folds", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.workers < 1 or args.every < 1:
        p.error("workers and every must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
