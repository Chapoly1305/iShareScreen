"""Choose the stronger existing control before confirmation-candidate scoring."""

import argparse
import json
from pathlib import Path

from benchmarks.super_resolution.mac_readability.quality import save, sha, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    output = root / "COMPARATOR-FROZEN.json"
    assert not output.exists()
    assert not (root / "candidate-quality").exists(), (
        "Comparator must precede candidate OCR"
    )
    definitions = json.loads((args.source / "manifest.json").read_text())
    assert definitions["complete"]
    expected = {
        (x["state_id"], label["label_id"])
        for x in definitions["states"]
        for label in x["labels"]
    }
    ranking = []
    evidence = {}
    for name in ["official", "trained"]:
        folder = root / (name + "-quality")
        protocol = json.loads((folder / "PROTOCOL.json").read_text())
        assert protocol["complete"] and protocol["source_sha256"] == sha(
            args.source / "manifest.json"
        )
        rows = json.loads((folder / "ROWS.json").read_text())
        lines = [x for x in rows if x["unit"] == "line"]
        assert {(x["state_id"], x["label_id"]) for x in lines} == expected and len(
            lines
        ) == len(expected)
        summary = summarize(lines)
        assert len(summary) == 1
        ranking.append(summary[0])
        evidence[name] = dict(
            rows_sha256=sha(folder / "ROWS.json"),
            protocol_sha256=sha(folder / "PROTOCOL.json"),
        )
    ranking.sort(key=lambda x: (x["macro_cer"], x["method"]))
    chosen = ranking[0]
    hard = [x for x in chosen["groups"] if x["cer"] > 0]
    assert hard
    hard_cer = sum(x["cer"] for x in hard) / len(hard)
    save(
        output,
        dict(
            complete=True,
            script_sha256=sha(Path(__file__)),
            source_manifest_sha256=sha(args.source / "manifest.json"),
            chosen_method=chosen["method"],
            selection="Lowest full-set equal-document lineCER among fixedofficialandbesttrainedcontrols, beforecandidateOCR",
            ranking=ranking,
            hard_groups=[x["group"] for x in hard],
            zero_error_groups=[x["group"] for x in chosen["groups"] if x["cer"] == 0],
            hard_macro_cer=hard_cer,
            target_hard_cer=hard_cer * 0.75,
            target_relative_reduction=0.25,
            evidence=evidence,
            candidate_scored=False,
            test_used=False,
            retain_full_set_and_zero_error_regressions=True,
        ),
    )
    print(
        "Frozen comparator",
        chosen["method"],
        "hardCER",
        hard_cer,
        "groups",
        len(hard),
        "target",
        hard_cer * 0.75,
    )


if __name__ == "__main__":
    main()
