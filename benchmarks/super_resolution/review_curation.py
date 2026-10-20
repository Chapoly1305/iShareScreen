"""Apply explicit, hash-bound visual quality gates to an audited proposal."""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path

from benchmarks.super_resolution.curate import write_report
from isharescreen.enhancement.models import sha256


def quality_flags(row, review):
    return sorted({r["reason"] for r in review["ranges"] if r["capture"] == row["capture"]
                   and r["start"] <= row["frame"] <= r["end"]})


def apply_review(rows, decisions, review):
    result = copy.deepcopy(decisions)
    for row, decision in zip(rows, result):
        flags = quality_flags(row, review)
        if flags:
            decision["quality_flags"] = flags
            if row["baseline"]:
                decision["future_training_eligible"] = False
            else:
                decision["parent_reason"] = decision["reason"]
                decision.update(selected=False, reason="manual_quality_quarantine")
        elif (review.get("require_exact_hr_for_near_dedup")
              and decision["reason"] == "near_duplicate_confirmed"
              and decision["differences"]["hr"]["mae"] > 0):
            decision.update(selected=True, reason="hr_change_preserved_after_visual_review")
    # A clean candidate must not disappear only because its old representative
    # was quarantined. Restore it unless a surviving exact paired copy exists.
    retained = {(r["capture"], r["hr_rgb_sha256"], r["lr_rgb_sha256"]): i
                for i, (r, d) in enumerate(zip(rows, result)) if d["selected"]}
    for i, (row, decision) in enumerate(zip(rows, result)):
        if (decision["reason"] not in ("near_duplicate_confirmed", "exact_pair_duplicate")
                or result[decision["reference_index"]]["selected"]):
            continue
        key = (row["capture"], row["hr_rgb_sha256"], row["lr_rgb_sha256"])
        if key in retained:
            decision.update(reason="exact_pair_duplicate", reference_index=retained[key])
        else:
            decision.update(selected=True, reason="quality_reference_unavailable")
            retained[key] = i
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True, help="Audited parent review directory")
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--quality-review", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    parent_path = args.review / "proposal.json"
    parent = json.loads(parent_path.read_text())
    audit = json.loads(args.audit.read_text())
    policy = json.loads(args.quality_review.read_text())
    candidate_path = Path(parent["candidates_manifest"])
    data = json.loads(candidate_path.read_text())
    if (sha256(parent_path) != audit["proposal_sha256"]
            or sha256(args.review / "embeddings.npy") != audit["embeddings_sha256"]
            or sha256(candidate_path) != policy["candidates_sha256"]
            or policy["candidates_sha256"] != parent["candidates_sha256"]):
        raise ValueError("Parent/audit/quality policy identity mismatch")
    rows = data["candidates"]
    captures = {r["capture"] for r in rows}
    for interval in policy["ranges"]:
        if (interval["capture"] not in captures or type(interval["start"]) is not int
                or type(interval["end"]) is not int or not 0 <= interval["start"] <= interval["end"]
                or not interval["reason"]):
            raise ValueError("Invalid visual quality interval")
    decisions = apply_review(rows, parent["decisions"], policy)
    selected = [r for r, d in zip(rows, decisions) if d["selected"]]
    warnings = [dict(capture=r["capture"], frame=r["frame"], fold=r["fold"],
                     reasons=d["quality_flags"]) for r, d in zip(rows, decisions)
                if r["baseline"] and d.get("quality_flags")]
    args.output.mkdir(parents=True, exist_ok=False)
    overview = write_report(rows, decisions, args.output, candidate_path.parent)
    proposal = dict(parent, selected=selected, decisions=decisions, overview=overview,
                    reasons=dict(Counter(d["reason"] for d in decisions)),
                    status="visually_gated_proposal_with_legacy_baseline_warnings",
                    dataset_name=None, parent_dataset_name=parent["dataset_name"],
                    parent_proposal=str(parent_path.resolve()), parent_proposal_sha256=sha256(parent_path),
                    parent_audit=str(args.audit.resolve()), parent_audit_sha256=sha256(args.audit),
                    quality_review=str(args.quality_review.resolve()), quality_review_sha256=sha256(args.quality_review),
                    quality_generator_sha256=sha256(Path(__file__)), baseline_quality_warnings=warnings)
    (args.output / "proposal.json").write_text(json.dumps(proposal, indent=2) + "\n")
    os.link(args.review / "embeddings.npy", args.output / "embeddings.npy")
    folds = json.loads((args.review / "fold-additions.json").read_text())
    baseline = json.loads(Path(data["baseline_manifest"]).read_text())["pairs"]
    flagged = {(r["capture"], r["frame"]) for r in warnings}
    identity = lambda r: dict(capture=r["capture"], frame=r["frame"])
    for fold in folds:
        held_out = fold["held_out_fold"]
        fold["training_additions"] = [identity(r) for r in selected if not r["baseline"] and r["fold"] != held_out]
        fold["training_exclusions"] = [identity(r) for r in baseline if r["fold"] != held_out
                                        and (r["capture"], r["frame"]) in flagged]
        fold["validation_quality_warnings"] = [r for r in warnings if r["fold"] == held_out]
    (args.output / "fold-additions.json").write_text(json.dumps(folds, indent=2) + "\n")
    page = args.output / "index.html"
    page.write_text(page.read_text().replace("<h1>",
        f"<p><strong>Quality review: {len(warnings)} legacy baseline frames are flagged for capture UI or motion video. "
        "Their inventory and validation membership remain unchanged; proposed future training exclusions are explicit. "
        "New flagged candidates are quarantined. This does not modify active training.</strong></p><h1>", 1))
    print(json.dumps(dict(selected=len(selected), baseline_quality_warnings=len(warnings),
                          reasons=proposal["reasons"]), indent=2))


if __name__ == "__main__":
    main()
