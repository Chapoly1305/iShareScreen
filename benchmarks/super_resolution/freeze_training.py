"""Freeze a signed-off curated pool into a portable, independently copied dataset."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

from benchmarks.super_resolution.folds import validate
from benchmarks.super_resolution.training_data import validate_pairs
from isharescreen.enhancement.models import sha256


def read(path):
    return json.loads(path.read_text())


def freeze(candidates, proposal, pool, qc, folds, baseline, output, workers=4):
    """Fail closed on stale provenance; never change the source inventory or folds."""
    source, review, index, signoff = map(read, (candidates, proposal, pool, qc))
    plan, anchors = read(folds), read(baseline)
    validate(plan)
    bindings = [(index["candidates_sha256"], candidates),
                (index["proposal_sha256"], proposal),
                (review["candidates_sha256"], candidates),
                (source["folds_sha256"], folds), (review["folds_sha256"], folds),
                (source["baseline_sha256"], baseline),
                (signoff["evidence_sha256"]["proposal"], proposal),
                (signoff["evidence_sha256"]["training_pool_index"], pool),
                (signoff["evidence_sha256"]["original_baseline"], baseline),
                (signoff["evidence_sha256"]["original_folds"], folds)]
    if any(digest != sha256(path) for digest, path in bindings):
        raise ValueError("Curation signoff, candidate, baseline or fold identity changed")
    if (signoff["status"] != "complete_with_known_legacy_baseline_warnings"
            or not signoff["assistant_visual_review"]["complete"]):
        raise ValueError("Completed quality-control signoff required")
    decisions = review["decisions"]
    if [d["index"] for d in decisions] != list(range(len(source["candidates"]))):
        raise ValueError("Curation decision inventory is incomplete or reordered")
    eligible = [d["index"] for d in decisions if d["selected"] and not d.get("quality_flags")]
    if eligible != index["eligible_candidate_indices"] or len(eligible) != index["eligible_pair_count"]:
        raise ValueError("Eligible pool differs from reviewed decisions")
    anchor_map = {(r["capture"], r["frame"]): r for r in anchors["pairs"]}
    rows = []
    for i in eligible:
        row = dict(source["candidates"][i])
        identity = (row["capture"], row["frame"])
        if bool(row["baseline"]) != (identity in anchor_map):
            raise ValueError("Baseline membership differs from frozen inventory")
        if identity in anchor_map and any(row[f"{role}_rgb_sha256"] != anchor_map[identity][f"{role}_rgb_sha256"]
                                         for role in ("lr", "hr")):
            raise ValueError("Baseline pixels changed")
        row.update(candidate_index=i, validation_eligible=bool(row["baseline"]))
        rows.append(row)
    data = dict(schema="sr_frozen_curated_pairs_v1", training_ready=True, pairs=rows,
                folds_sha256=sha256(folds), manifest_sha256=source["manifest_sha256"],
                provenance={name: sha256(path) for name, path in
                            (("candidates", candidates), ("proposal", proposal), ("pool", pool),
                             ("quality_control", qc), ("baseline", baseline))},
                generator_sha256=sha256(Path(__file__)),
                validation_policy="Clean legacy baseline only; all other held-out frames excluded from training",
                sampling_policy="Each training frame once per shuffled epoch; new aligned crop on each visit")
    validate_pairs(data, plan)
    expected = {c["id"] for c in plan["clips"]}
    if {r["capture"] for r in rows if r["validation_eligible"]} != expected:
        raise ValueError("Clean validation silently drops a capture")
    output.mkdir(parents=True, exist_ok=False)
    def copy_pair(row):
        import numpy as np
        from PIL import Image
        shapes = {}
        for role in ("lr", "hr"):
            relative = Path(row[role])
            origin = (candidates.parent / relative).resolve()
            if relative.is_absolute() or not origin.is_relative_to(candidates.parent.resolve()):
                raise ValueError("Candidate path escapes dataset root")
            if sha256(origin) != row[f"{role}_sha256"]:
                raise ValueError("Candidate PNG changed")
            with Image.open(origin) as image:
                rgb = np.array(image.convert("RGB"))
            if hashlib.sha256(rgb.tobytes()).hexdigest() != row[f"{role}_rgb_sha256"]:
                raise ValueError("Candidate decoded pixels changed")
            shapes[role] = rgb.shape
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)  # Not a hardlink to mutable extraction files.
            if sha256(target) != row[f"{role}_sha256"]:
                raise ValueError("Frozen copy differs")
        if shapes["hr"] != (2*shapes["lr"][0], 2*shapes["lr"][1], 3):
            raise ValueError("Not an aligned x2 pair")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in executor.map(copy_pair, rows):
            pass
    shutil.copy2(folds, output / "folds.json")
    for name, path in (("pool", pool), ("quality-control", qc), ("proposal", proposal)):
        shutil.copy2(path, output / f"source-{name}.json")
    # Manifest is the commit point: partial freezes have no consumable manifest.
    (output / "manifest.json").write_text(json.dumps(data, indent=2, allow_nan=False))
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("candidates", "proposal", "pool", "qc", "folds", "baseline", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.workers < 1:
        p.error("Positive worker count required")
    result = freeze(**vars(args))
    print(f"FROZEN {len(result['pairs'])} pairs; "
          f"{sum(r['validation_eligible'] for r in result['pairs'])} validation anchors", flush=True)


if __name__ == "__main__":
    main()
