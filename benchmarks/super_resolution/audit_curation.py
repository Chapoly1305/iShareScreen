"""Audit a frame-curation proposal and make capture-wide visual QC sheets."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from benchmarks.super_resolution.curate import load_rgb, pixel_difference, safe_duplicate, shared_references
from benchmarks.super_resolution.folds import validate
from benchmarks.super_resolution.prepare_curation import candidate_indices, inside
from benchmarks.super_resolution.training_data import validate_pairs
from isharescreen.enhancement.models import sha256


def audit(proposal, candidates, plan, baseline, quality_review=None):
    rows, decisions = candidates["candidates"], proposal["decisions"]
    if [d["index"] for d in decisions] != list(range(len(rows))):
        raise ValueError("Decision coverage/order differs from candidate manifest")
    selected = [r for r, d in zip(rows, decisions) if d["selected"]]
    if selected != proposal["selected"] or proposal["training_ready"] is not False:
        raise ValueError("Proposal selected records or readiness flag changed")
    paired = {(r["capture"], r["hr_rgb_sha256"], r["lr_rgb_sha256"]) for r in selected}
    if len(paired) != len(selected):
        raise ValueError("Selected dataset retains an exact paired duplicate")
    validate_pairs({"pairs": selected}, plan)
    original = {(r["capture"], r["frame"]): r for r in baseline["pairs"]}
    retained = {(r["capture"], r["frame"]): r for r in selected}
    for identity, row in original.items():
        if identity not in retained:
            raise ValueError("Frozen baseline frame disappeared")
        for role in ("hr", "lr"):
            if retained[identity][f"{role}_rgb_sha256"] != row[f"{role}_rgb_sha256"]:
                raise ValueError("Frozen baseline pixels changed")
    shared = shared_references(rows)
    for clip in plan["clips"]:
        actual = sorted(r["frame"] for r in rows if r["capture"] == clip["id"])
        anchors = [frame for capture, frame in original if capture == clip["id"]]
        if actual != candidate_indices(clip["frames"], candidates["every"], anchors):
            raise ValueError("Candidate schedule has missing or unexpected frames")
    for i, (row, decision) in enumerate(zip(rows, decisions)):
        if quality_review is not None:
            from benchmarks.super_resolution.review_curation import quality_flags
            expected = quality_flags(row, quality_review)
            if decision.get("quality_flags", []) != expected:
                raise ValueError("Visual quality flags differ from reviewed intervals")
            if expected and ((not row["baseline"] and decision["selected"])
                             or (row["baseline"] and decision.get("future_training_eligible") is not False)):
                raise ValueError("Quality review did not quarantine new content or flag legacy training exclusions")
            if (quality_review.get("require_exact_hr_for_near_dedup")
                    and decision["reason"] == "near_duplicate_confirmed"
                    and decision["differences"]["hr"]["mae"] != 0):
                raise ValueError("Near-duplicate exclusion changes HR text/detail pixels")
        if row["baseline"] and (not decision["selected"] or decision["reason"] != "frozen_baseline_anchor"):
            raise ValueError("Baseline anchor decision changed")
        if not row["baseline"] and row["hr_rgb_sha256"] in shared:
            if decision["selected"] or decision["reason"] != "quarantined_reference_overlap":
                raise ValueError("New cross-group exact reference was not quarantined")
        if decision["reason"] in ("near_duplicate_confirmed", "exact_pair_duplicate"):
            reference = decision["reference_index"]
            if not 0 <= reference < len(rows) or not decisions[reference]["selected"]:
                raise ValueError("Duplicate refers to a missing/skipped representative")
            if rows[reference]["capture"] != row["capture"]:
                raise ValueError("Duplicate representative crosses captures")
            if decision["reason"] == "exact_pair_duplicate":
                if any(row[f"{role}_rgb_sha256"] != rows[reference][f"{role}_rgb_sha256"] for role in ("hr", "lr")):
                    raise ValueError("Exact-pair exclusion changed HR or LR pixels")
            elif not safe_duplicate(decision["differences"], proposal["parameters"]["mean_limit"],
                                   proposal["parameters"]["tile_limit"]):
                raise ValueError("Skipped frame fails native-pixel guards")
    original_hr = {r["hr_rgb_sha256"] for r in original.values()}
    selected_hr = {r["hr_rgb_sha256"] for r in selected}
    return dict(source_frames=sum(c["frames"] for c in plan["clips"]),
                candidates=len(rows), selected=len(selected), baseline_preserved=len(original),
                additions=len(selected) - len(original), captures=len(plan["clips"]),
                unique_hr_pixel_targets=len(selected_hr), new_hr_pixel_targets=len(selected_hr - original_hr),
                legacy_baseline_quality_warnings=sum(r["baseline"] and bool(d.get("quality_flags"))
                                                     for r, d in zip(rows, decisions)),
                note="Unique HR pixel hashes are not a count of independent semantic scenes",
                reasons=dict(Counter(d["reason"] for d in decisions)))


def overview_sheets(rows, selected, root, output):
    from PIL import Image, ImageDraw
    groups = defaultdict(list)
    for row in rows:
        if (row["capture"], row["frame"]) in selected:
            groups[row["capture"]].append(row)
    sheets = []
    for page_start in range(0, len(groups), 6):
        captures = sorted(groups)[page_start:page_start + 6]
        sheet = Image.new("RGB", (1280, len(captures) * 214), "#1c2025")
        draw = ImageDraw.Draw(sheet)
        for line, capture in enumerate(captures):
            group = groups[capture]
            choices = [group[round(i * (len(group) - 1) / 3)] for i in range(4)]
            y = line * 214
            draw.text((4, y + 2), f"{capture} / selected {len(group)}", fill="white")
            for column, row in enumerate(choices):
                with Image.open(inside(root, row["hr"])) as image:
                    image = image.convert("RGB"); image.thumbnail((320, 180))
                    sheet.paste(image, (column * 320, y + 17))
                draw.text((column * 320 + 4, y + 198),
                          f"frame {row['frame']} / {row['time_s']:.3f}s / " +
                          ("baseline" if row["baseline"] else "addition"), fill="white")
        name = f"coverage-{page_start // 6:02d}.jpg"
        sheet.save(output / name, quality=92)
        sheets.append(name)
    return sheets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--folds", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.workers < 1:
        p.error("workers must be positive")
    proposal_path = args.review / "proposal.json"
    proposal = json.loads(proposal_path.read_text())
    candidate_path = Path(proposal["candidates_manifest"])
    candidates = json.loads(candidate_path.read_text())
    baseline_path = Path(candidates["baseline_manifest"])
    baseline = json.loads(baseline_path.read_text())
    plan = validate(json.loads(args.folds.read_text()))
    if (sha256(candidate_path) != proposal["candidates_sha256"]
            or sha256(args.folds) != proposal["folds_sha256"]
            or sha256(baseline_path) != proposal["baseline_sha256"]
            or sha256(Path(candidates["source_manifest"])) != proposal["manifest_sha256"]
            or sha256(Path(__file__).with_name("curate.py")) != proposal["generator_sha256"]
            or sha256(Path(__file__).with_name("prepare_curation.py")) != candidates["generator_sha256"]):
        raise ValueError("Audit inputs changed")
    quality_review = None
    if "quality_review" in proposal:
        quality_path = Path(proposal["quality_review"])
        quality_review = json.loads(quality_path.read_text())
        if (sha256(quality_path) != proposal["quality_review_sha256"]
                or quality_review["candidates_sha256"] != proposal["candidates_sha256"]
                or sha256(Path(__file__).with_name("review_curation.py")) != proposal["quality_generator_sha256"]
                or sha256(Path(proposal["parent_proposal"])) != proposal["parent_proposal_sha256"]
                or sha256(Path(proposal["parent_audit"])) != proposal["parent_audit_sha256"]):
            raise ValueError("Visual review/parent provenance changed")
        limits = {clip["id"]: clip["frames"] for clip in plan["clips"]}
        for interval in quality_review["ranges"]:
            if interval["capture"] not in limits or not 0 <= interval["start"] <= interval["end"] < limits[interval["capture"]]:
                raise ValueError("Visual quality range escapes source frames")
    summary = audit(proposal, candidates, plan, baseline, quality_review)
    fold_additions = json.loads((args.review / "fold-additions.json").read_text())
    if [entry["held_out_fold"] for entry in fold_additions] != list(range(plan["n_folds"])):
        raise ValueError("Fold-addition coverage changed")
    for entry in fold_additions:
        fold = entry["held_out_fold"]
        identity = lambda r: dict(capture=r["capture"], frame=r["frame"])
        if (entry["validation_unchanged"] is not True
                or entry["validation_frames"] != [identity(r) for r in baseline["pairs"] if r["fold"] == fold]
                or entry["training_additions"] != [identity(r) for r in proposal["selected"]
                                                  if not r["baseline"] and r["fold"] != fold]):
            raise ValueError("Fold plan changes validation or introduces held-out training frames")
        if quality_review is not None:
            from benchmarks.super_resolution.review_curation import quality_flags
            exclusions = [identity(r) for r in baseline["pairs"]
                          if r["fold"] != fold and quality_flags(r, quality_review)]
            if entry.get("training_exclusions") != exclusions:
                raise ValueError("Future training exclusions differ from reviewed legacy frames")
    # Independently re-read selected PNGs and repeat the native-pixel guard for
    # EVERY confirmed duplicate, not only contact-sheet examples.
    import numpy as np
    root, rows = candidate_path.parent, candidates["candidates"]
    def check_pixels(item):
        i, decision = item
        row = rows[i]
        if decision["selected"]:
            hr, lr = load_rgb(root, row, "hr"), load_rgb(root, row, "lr")
            if hr.shape != (2 * lr.shape[0], 2 * lr.shape[1], 3):
                raise ValueError("Selected pair geometry changed")
        elif decision["reason"] == "near_duplicate_confirmed":
            reference = rows[decision["reference_index"]]
            for role in ("hr", "lr"):
                actual = pixel_difference(load_rgb(root, row, role), load_rgb(root, reference, role))
                if actual != decision["differences"][role]:
                    raise ValueError("Independent native-pixel guard differs from producer")
        elif decision["reason"] == "exact_pair_duplicate":
            for role in ("hr", "lr"):
                load_rgb(root, row, role)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(check_pixels, enumerate(proposal["decisions"]))):
            if (i + 1) % 500 == 0:
                print(f"AUDIT {i + 1}/{len(rows)}", flush=True)
    array = np.load(args.review / "embeddings.npy", allow_pickle=False)
    if array.shape[0] != len(rows) or not np.isfinite(array).all():
        raise ValueError("Embedding coverage or values invalid")
    args.output.mkdir(parents=True, exist_ok=False)
    selected_ids = {(r["capture"], r["frame"]) for r in proposal["selected"]}
    sheets = overview_sheets(rows, selected_ids, root, args.output)
    result = dict(status="automated_checks_passed_visual_review_pending", summary=summary,
                  proposal_sha256=sha256(proposal_path), baseline_sha256=sha256(baseline_path),
                  folds_sha256=sha256(args.folds), embeddings_sha256=sha256(args.review / "embeddings.npy"),
                  generator_sha256=sha256(Path(__file__)), coverage_sheets=sheets,
                  selected_pixels_rechecked=True, all_confirmed_duplicate_guards_recomputed=True,
                  exact_paired_duplicates_remaining=0,
                  validation_membership_unchanged=True, training_started=False,
                  limitation="Visual spot checks and no downstream training comparison; not exhaustive semantic QC")
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
