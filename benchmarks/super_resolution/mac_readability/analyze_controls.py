"""Audit fixed development controls against the frozen comparator, retaining regressions."""

import argparse
import json
from pathlib import Path
import numpy as np
from benchmarks.super_resolution.mac_readability.quality import save, sha, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    r = a.root
    freeze = json.loads((r / "COMPARATOR-FROZEN.json").read_text())
    hard = freeze["hard_groups"]
    base = [
        x
        for x in json.loads((r / "dense-model-quality-v1/ROWS.json").read_text())
        if x["method"] == "edsr-official" and x["unit"] == "line"
    ]
    lookup = {(x["state_id"], x["label_id"]): x for x in base}
    bs = summarize(base)[0]
    bg = {g["group"]: g["cer"] for g in bs["groups"]}
    outputs = []
    files = list((r / "detail-controls-v1").glob("*/ocr/ROWS.json"))
    if (r / "ensemble-controls-v1/ocr/ROWS.json").exists():
        files.append(r / "ensemble-controls-v1/ocr/ROWS.json")
    for path in files:
        rows = json.loads(path.read_text())
        for s in summarize(rows):
            selected = [x for x in rows if x["method"] == s["method"]]
            assert len(selected) == 1046
            groups = {x["group"]: x["cer"] for x in s["groups"]}
            hc = float(np.mean([groups[g] for g in hard]))
            delta = np.array([bg[g] - groups[g] for g in hard])
            draw = np.random.default_rng(20261102).integers(
                0, len(hard), (10000, len(hard))
            )
            outputs.append(
                dict(
                    method=s["method"],
                    hard_cer=hc,
                    relative_reduction=1 - hc / freeze["hard_macro_cer"],
                    target_cer_met=hc <= freeze["target_hard_cer"],
                    absolute_gain_ci95=np.quantile(
                        delta[draw].mean(1), [0.025, 0.975]
                    ).tolist(),
                    full_summary=s,
                    category_summary={
                        c: summarize([x for x in selected if x["category"] == c])[0]
                        for c in sorted({x["category"] for x in selected})
                    },
                    zero_error_group_edits=sum(
                        x["edits"]
                        for x in selected
                        if x["group"] in freeze["zero_error_groups"]
                    ),
                    formerly_correct_lines_now_wrong=sum(
                        not x["exact_nonspace"]
                        and lookup[x["state_id"], x["label_id"]]["exact_nonspace"]
                        for x in selected
                    ),
                    formerly_wrong_lines_now_correct=sum(
                        x["exact_nonspace"]
                        and not lookup[x["state_id"], x["label_id"]]["exact_nonspace"]
                        for x in selected
                    ),
                    row_file=str(path),
                    row_sha256=sha(path),
                )
            )
    outputs.sort(key=lambda x: x["hard_cer"])
    save(
        r / "LOW-COST-CONTROLS.json",
        dict(
            complete=len(outputs) == 9,
            scope=__doc__,
            script_sha256=sha(Path(__file__)),
            comparator_sha256=sha(r / "COMPARATOR-FROZEN.json"),
            expected_candidates=9,
            candidates=outputs,
            selected_for_production=None,
            pending="Cell correctness, visual regressions, actual4090 pipeline latency, and independent confirmation required for any qualifying candidate",
            test_used=False,
        ),
    )
    for x in outputs:
        print(
            x["method"],
            round(x["hard_cer"] * 100, 3),
            round(x["relative_reduction"] * 100, 2),
            x["target_cer_met"],
        )


if __name__ == "__main__":
    main()
