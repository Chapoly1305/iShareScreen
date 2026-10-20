"""Sample aligned decoded video pairs before any random training crop is taken."""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from benchmarks.super_resolution.folds import validate
from isharescreen.enhancement.models import sha256


def keep_unique(rows):
    """Remove shared exact HR frames from ALL groups, then deduplicate within groups.

    This is fold-blind filtering, not automatic semantic content classification.
    Shared opening/closing screens must not create artificially easy validation.
    """
    owners = defaultdict(set)
    for row in rows:
        owners[row["hr_rgb_sha256"]].add(row["group"])
    seen, kept, dropped = set(), [], []
    for row in rows:
        digest = row["hr_rgb_sha256"]
        reason = "shared_exact_reference" if len(owners[digest]) > 1 else "repeated_reference"
        if len(owners[digest]) > 1 or digest in seen:
            dropped.append({"capture": row["capture"], "frame": row["frame"], "reason": reason})
        else:
            kept.append(row)
            seen.add(digest)
    return kept, dropped


def validate_pairs(data, plan):
    validate(plan)
    clips = {c["id"]: c for c in plan["clips"]}
    seen, hashes, captures = set(), {}, set()
    for pair in data["pairs"]:
        capture = pair["capture"]
        if capture not in clips or pair["group"] != clips[capture]["group"]:
            raise ValueError("Pair has unknown capture or incorrect content group")
        if pair["fold"] != clips[capture]["fold"]:
            raise ValueError("Pair fold differs from frozen assignment")
        identity = (capture, pair["frame"])
        if identity in seen or not 0 <= pair["frame"] < clips[capture]["frames"]:
            raise ValueError("Duplicate or invalid source frame")
        seen.add(identity)
        captures.add(capture)
        digest = pair["hr_rgb_sha256"]
        if digest in hashes and hashes[digest] != pair["group"]:
            raise ValueError("Exact reference duplicated across content groups")
        hashes[digest] = pair["group"]
    # Every capture must retain data; no silent disappearance of difficult clips.
    if captures != set(clips):
        raise ValueError(f"Captures without retained samples: {sorted(set(clips) - captures)}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--folds", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--every", type=int, default=60, help="Sample spacing in decoded source frames")
    args = p.parse_args()
    if args.every < 1:
        p.error("every must be positive")
    plan = validate(json.loads(args.folds.read_text()))
    if sha256(args.manifest) != plan["manifest_sha256"]:
        raise ValueError("Video manifest changed since fold creation")
    manifest = json.loads(args.manifest.read_text())
    # Import heavy dependencies only for extraction, not split planning.
    from PIL import Image
    from benchmarks.super_resolution.video_benchmark import aligned_frames
    args.output.mkdir(parents=True, exist_ok=False)
    membership = {c["id"]: c for c in plan["clips"]}
    rows = []
    for index, clip in enumerate(manifest["clips"]):
        for role in ("reference", "low"):
            if sha256(args.manifest.parent / clip[role]) != clip[f"{role}_sha256"]:
                raise ValueError(f"Changed video input: {clip['id']}/{role}")
        folder = args.output / f"clip-{index:03d}"
        folder.mkdir()
        for i, hr, lr in aligned_frames(args.manifest.parent / clip["reference"],
                                        args.manifest.parent / clip["low"],
                                        clip["frames"], Fraction(clip["fps"])):
            # No appended final adjacent frame; the schedule is fixed for all models.
            if i % args.every:
                continue
            row = dict(capture=clip["id"], group=membership[clip["id"]]["group"],
                       fold=membership[clip["id"]]["fold"], frame=i,
                       time_s=float(Fraction(i, 1) / Fraction(clip["fps"])))
            for role, rgb in (("hr", hr), ("lr", lr)):
                path = folder / f"{i:06d}-{role}.png"
                Image.fromarray(rgb).save(path)
                row[role] = str(path.relative_to(args.output))
                row[f"{role}_sha256"] = sha256(path)
                row[f"{role}_rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
            rows.append(row)
        print(f"SAMPLED {index+1}/{len(manifest['clips'])} {clip['id']}", flush=True)
    kept, dropped = keep_unique(rows)
    data = dict(schema="sr_training_pairs_v1", pairs=kept, excluded=dropped,
                every=args.every, folds_sha256=sha256(args.folds),
                manifest_sha256=sha256(args.manifest), generator_sha256=sha256(Path(__file__)),
                filtering="fold-blind exact-HR dedup; all cross-group exact references excluded")
    (args.output / "sampling-audit.json").write_text(json.dumps(data, indent=2))
    validate_pairs(data, plan)
    # Excluded PNGs stay in the audit directory but are never consumed for training.
    (args.output / "manifest.json").write_text(json.dumps(data, indent=2))
    print(f"VERIFIED {len(kept)} unique paired frames; excluded {len(dropped)}", flush=True)


if __name__ == "__main__":
    main()
