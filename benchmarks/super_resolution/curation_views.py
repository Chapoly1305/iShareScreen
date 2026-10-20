"""Make native-detail review images from curation decisions; previews only, never GT."""
import argparse
import json
from pathlib import Path

from benchmarks.super_resolution.prepare_curation import inside
from isharescreen.enhancement.models import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeline-captures", nargs="+", help="Review every dense candidate in these captures")
    args = p.parse_args()
    import numpy as np
    from PIL import Image, ImageDraw
    proposal_path = args.review / "proposal.json"
    proposal = json.loads(proposal_path.read_text())
    candidate_path = Path(proposal["candidates_manifest"])
    if sha256(candidate_path) != proposal["candidates_sha256"]:
        raise ValueError("Candidate manifest changed")
    rows = json.loads(candidate_path.read_text())["candidates"]
    root = candidate_path.parent
    if args.timeline_captures:
        if set(args.timeline_captures) - {r["capture"] for r in rows}:
            raise ValueError("Unknown timeline capture")
        args.output.mkdir(parents=True, exist_ok=False)
        for capture in args.timeline_captures:
            group = [r for r in rows if r["capture"] == capture]
            for start in range(0, len(group), 60):
                page = group[start:start + 60]
                sheet = Image.new("RGB", (1600, ((len(page) + 9) // 10) * 114), "#20242a")
                draw = ImageDraw.Draw(sheet)
                for i, row in enumerate(page):
                    x, y = i % 10 * 160, i // 10 * 114
                    with Image.open(inside(root, row["hr"])) as image:
                        image = image.convert("RGB"); image.thumbnail((160, 90))
                        sheet.paste(image, (x, y))
                    draw.text((x + 2, y + 92), f"{row['frame']} / {row['time_s']:.2f}s", fill="white")
                sheet.save(args.output / f"{capture}-{start // 60:02d}.jpg", quality=92)
        return
    decisions = proposal["decisions"]
    categories = {
        "exact_pair": [d for d in decisions if d["reason"] == "exact_pair_duplicate"],
        "near_duplicate": sorted([d for d in decisions if d["reason"] == "near_duplicate_confirmed"],
                                 key=lambda d: -max(x["max_tile_mae"] for x in d["differences"].values())),
        "lr_difference_preserved": sorted([d for d in decisions if d["reason"] == "native_detail_or_degradation_preserved"
                                            and d["differences"]["hr"]["mae"] == 0],
                                           key=lambda d: -d["differences"]["lr"]["max_tile_mae"]),
        "hr_detail_preserved": sorted([d for d in decisions if d["reason"] == "native_detail_or_degradation_preserved"
                                        and d["differences"]["hr"]["max_tile_mae"] > 1],
                                       key=lambda d: d["differences"]["hr"]["max_tile_mae"]),
        "hr_change_restored": [d for d in decisions if d["reason"] == "hr_change_preserved_after_visual_review"],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for category, choices in categories.items():
        seen = set()
        for decision in choices:
            row, reference = rows[decision["index"]], rows[decision["reference_index"]]
            if row["capture"] in seen:
                continue
            seen.add(row["capture"])
            images = {}
            for name, record in (("reference", reference), ("candidate", row)):
                for role in ("hr", "lr"):
                    with Image.open(inside(root, record[role])) as image:
                        images[name, role] = image.convert("RGB")
            a = np.asarray(images["reference", "hr"]).astype(np.int16)
            b = np.asarray(images["candidate", "hr"]).astype(np.int16)
            difference = np.abs(a - b).mean(axis=2)
            lr_difference = np.abs(np.asarray(images["reference", "lr"]).astype(np.int16) -
                                   np.asarray(images["candidate", "lr"]).astype(np.int16)).mean(axis=2)
            difference += lr_difference.repeat(2, 0).repeat(2, 1)
            y, x = np.unravel_index(np.argmax(difference), difference.shape)
            width, height = images["reference", "hr"].size
            x0 = max(0, min(int(x) - 240, width - 480)) // 2 * 2
            y0 = max(0, min(int(y) - 120, height - 240)) // 2 * 2
            box = (x0, y0, x0 + 480, y0 + 240)
            sheet = Image.new("RGB", (960, 840), "#20242a")
            draw = ImageDraw.Draw(sheet)
            draw.text((6, 4), f"{category}: {row['capture']} / ref {reference['frame']} vs candidate {row['frame']}", fill="white")
            for column, name in enumerate(("reference", "candidate")):
                context = images[name, "hr"].resize((480, 270))
                sheet.paste(context, (column * 480, 28))
                draw.text((column * 480 + 4, 300), f"{name}: whole-frame context (viewing resize)", fill="white")
                sheet.paste(images[name, "hr"].crop(box), (column * 480, 326))
                draw.text((column * 480 + 4, 568), f"HR native pixels / crop {box}", fill="white")
                low_box = tuple(value // 2 for value in box)
                low = images[name, "lr"].crop(low_box).resize((480, 240), Image.Resampling.NEAREST)
                sheet.paste(low, (column * 480, 594))
                draw.text((column * 480 + 4, 818), "LR same region / nearest x2 for viewing", fill="white")
            filename = f"{category}-{len(seen):02d}.png"
            sheet.save(args.output / filename)
            cases.append(dict(category=category, image=filename, capture=row["capture"],
                              candidate_frame=row["frame"], reference_frame=reference["frame"],
                              hr_crop=box, candidate_index=decision["index"],
                              reference_index=decision["reference_index"],
                              image_sha256=sha256(args.output / filename)))
            if len(seen) == 2:
                break
    result = dict(proposal_sha256=sha256(proposal_path), generator_sha256=sha256(Path(__file__)),
                  cases=cases, purpose="Visual spot-check artifacts only; never training targets")
    (args.output / "cases.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
