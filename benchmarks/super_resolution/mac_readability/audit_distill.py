"""Audit the complete GT-continuation/teacher-distillation queue and paired regressions."""

import argparse
import json
from pathlib import Path
import numpy as np
from benchmarks.super_resolution.mac_readability.quality import save, sha, summarize


def read(p):
    return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    curve = read(root / "training-ocr-v4/CURVES.json")
    assert curve["complete"] and len(curve["checkpoints"]) == 13
    assert read(root / "training-v4/STATUS.json")["state"] == "training_completed"
    freeze = read(root / "COMPARATOR-FROZEN.json")
    hard = freeze["hard_groups"]
    base = [
        x
        for x in read(root / "dense-model-quality-v1/ROWS.json")
        if x["method"] == "edsr-official" and x["unit"] == "line"
    ]
    bsummary = summarize(base)[0]
    bg = {g["group"]: g["cer"] for g in bsummary["groups"]}
    lookup = {(x["state_id"], x["label_id"]): x for x in base}
    candidates = [x for x in curve["checkpoints"] if x["step"] > 0]
    best = min(candidates, key=lambda r: (r["hard_macro_cer"], r["arm"], r["step"]))
    audited = []
    for arm in sorted({r["arm"] for r in candidates}):
        pick = min(
            [x for x in candidates if x["arm"] == arm],
            key=lambda r: (r["hard_macro_cer"], r["step"]),
        )
        path = root / "training-ocr-v4" / arm / f"step-{pick['step']:06d}" / "ROWS.json"
        rows = read(path)
        assert len(rows) == 1046
        summary = summarize(rows)[0]
        groups = {g["group"]: g["cer"] for g in summary["groups"]}
        delta = np.array([bg[g] - groups[g] for g in hard])
        draw = np.random.default_rng(20261102).integers(0, 15, (10000, 15))
        new = [
            x
            for x in rows
            if lookup[x["state_id"], x["label_id"]]["exact_nonspace"]
            and not x["exact_nonspace"]
        ]
        repaired = [
            x
            for x in rows
            if not lookup[x["state_id"], x["label_id"]]["exact_nonspace"]
            and x["exact_nonspace"]
        ]
        audited.append(
            dict(
                arm=arm,
                step=pick["step"],
                hard_cer=pick["hard_macro_cer"],
                relative_reduction=1
                - pick["hard_macro_cer"] / freeze["hard_macro_cer"],
                absolute_gain_ci95=np.quantile(
                    delta[draw].mean(1), [0.025, 0.975]
                ).tolist(),
                summary=summary,
                categories={
                    c: summarize([x for x in rows if x["category"] == c])[0]
                    for c in sorted({x["category"] for x in rows})
                },
                new_errors_on_correct_lines=new,
                repaired_lines=repaired,
                zero_error_group_edits=sum(
                    x["edits"]
                    for x in rows
                    if x["group"] in freeze["zero_error_groups"]
                ),
                rows_sha256=sha(path),
            )
        )
    folder = root / "training-v4/models" / best["arm"]
    receipt = read(folder / "COMPLETED.json")
    checkpoint = next(x for x in receipt["checkpoints"] if x["step"] == best["step"])
    images_root = folder / f"validation-{best['step']:06d}"
    index = read(images_root / "RESULTS.json")
    for row in index["images"]:
        row["path"] = str((images_root / row["path"]).resolve())
        assert sha(Path(row["path"])) == row["sha256"]
    selection = dict(
        schema="teacher_distillation_selection_v1",
        complete=True,
        comparator_sha256=sha(root / "COMPARATOR-FROZEN.json"),
        chosen_arm=best["arm"],
        chosen_step=best["step"],
        checkpoint_sha256=checkpoint["checkpoint_sha256"],
        hard_cer=best["hard_macro_cer"],
        target_cer=freeze["target_hard_cer"],
        quality_target_met=best["hard_macro_cer"] <= freeze["target_hard_cer"],
        method="Lowest frozen hard15 CER after all12 trained snapshots andone common initialization control; development selection only",
        production_selected=False,
        pending="Cell/visual/latency guards and independent confirmation; quality goal remains active if target not met",
        bootstrap_scope="Descriptive paired development-group interval after checkpoint selection; not an unbiased held-out generalization guarantee",
        arms=audited,
        test_used=False,
    )
    save(root / "TRAINING-V4-SELECTION.json", selection)
    save(root / "TRAINING-V4-SELECTED-IMAGES.json", index)
    for x in audited:
        print(
            x["arm"],
            x["step"],
            round(x["hard_cer"] * 100, 4),
            round(x["relative_reduction"] * 100, 2),
            x["absolute_gain_ci95"],
        )


if __name__ == "__main__":
    main()
