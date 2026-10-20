"""Deterministic content-grouped development folds, shared by every SR model."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from isharescreen.enhancement.models import sha256


def validate(plan):
    clips = plan["clips"]
    n = plan["n_folds"]
    if n < 2 or not clips:
        raise ValueError("Need at least two nonempty folds")
    ids, groups = set(), {}
    for clip in clips:
        if not isinstance(clip["id"], str) or not clip["id"] or clip["id"] in ids:
            raise ValueError("Empty or duplicate capture ID")
        ids.add(clip["id"])
        group, fold = clip["group"], clip["fold"]
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Empty content group")
        if type(fold) is not int or not 0 <= fold < n:
            raise ValueError("Invalid fold index")
        if group in groups and groups[group] != fold:
            raise ValueError("Content group crosses folds")
        groups[group] = fold
    if set(groups.values()) != set(range(n)):
        raise ValueError("Empty fold")
    return plan


def partition(plan, fold):
    validate(plan)
    if type(fold) is not int or not 0 <= fold < plan["n_folds"]:
        raise ValueError("Invalid held-out fold")
    train = [c for c in plan["clips"] if c["fold"] != fold]
    val = [c for c in plan["clips"] if c["fold"] == fold]
    return train, val


def make_plan(manifest, n_folds=5, seed=0, group_map=None):
    if type(n_folds) is not int or n_folds < 2:
        raise ValueError("Need at least two folds")
    clips = manifest["clips"]
    group_map = group_map or {}
    ids = [c["id"] for c in clips]
    if len(set(ids)) != len(ids) or any(not i for i in ids):
        raise ValueError("Empty or duplicate capture ID")
    if set(group_map) - set(ids):
        raise ValueError("Group map contains unknown captures")
    grouped = defaultdict(list)
    for clip in sorted(clips, key=lambda c: c["id"]):
        group = group_map.get(clip["id"], clip["id"])
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Empty content group")
        seconds = float(Fraction(clip["frames"], 1) / Fraction(clip["fps"]))
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid clip duration")
        grouped[group].append({**clip, "group": group, "duration_s": seconds})
    if len(grouped) < n_folds:
        raise ValueError("Fewer content groups than folds")
    # Longest groups first; deterministic seeded ties, independent of input order.
    def tie(value):
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    order = sorted(grouped, key=lambda g: (-sum(c["duration_s"] for c in grouped[g]), tie(g)))
    counts, durations = [0] * n_folds, [0.0] * n_folds
    contents = [Counter() for _ in range(n_folds)]
    capacities = [len(order) // n_folds + (i < len(order) % n_folds) for i in range(n_folds)]
    result = []
    for group in order:
        members = grouped[group]
        labels = Counter(c.get("content") or "unknown" for c in members)
        eligible = [i for i in range(n_folds) if counts[i] < capacities[i]]
        fold = min(eligible, key=lambda i: (
            durations[i], sum(contents[i][label] for label in labels), tie(f"fold-{i}")))
        counts[fold] += 1
        durations[fold] += sum(c["duration_s"] for c in members)
        contents[fold].update(labels)
        result.extend({**c, "fold": fold} for c in members)
    plan = {
        "schema": "sr_grouped_development_folds_v1", "n_folds": n_folds, "seed": seed,
        "scope": "cross-content development CV; not an independent-session final test",
        "grouping": "capture ID unless explicitly overridden; content independence asserted by dataset owner",
        "algorithm": "longest-duration-first; balanced group capacities; content counts break duration ties",
        "clips": sorted(result, key=lambda c: c["id"]),
    }
    validate(plan)
    plan["summary"] = [dict(
        fold=i, groups=counts[i], clips=sum(c["fold"] == i for c in result),
        duration_s=durations[i], content=dict(sorted(contents[i].items()))) for i in range(n_folds)]
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--groups", type=Path, help="Optional JSON capture-ID to content-group map")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    groups = json.loads(args.groups.read_text()) if args.groups else None
    plan = make_plan(json.loads(args.manifest.read_text()), args.folds, args.seed, groups)
    plan.update(manifest_sha256=sha256(args.manifest), generator_sha256=sha256(Path(__file__)))
    if args.groups:
        plan["groups_sha256"] = sha256(args.groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(plan, f, indent=2)
    print(json.dumps(plan["summary"], indent=2))


if __name__ == "__main__":
    main()
