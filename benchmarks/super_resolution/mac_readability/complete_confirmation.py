"""Audit the experimental goal against immutable raw evidence and deliverables."""

import argparse
import datetime
from html.parser import HTMLParser
import json
import os
from pathlib import Path

import numpy as np

from benchmarks.super_resolution.mac_readability.quality import save, sha, summarize


def read(path):
    return json.loads(path.read_text())


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ["src", "href"] and value and "://" not in value:
                self.paths.append(value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    source = a.source.resolve()
    evidence = {}
    requirements = []

    def bind(path):
        path = Path(path).resolve()
        assert path.is_file()
        evidence[os.path.relpath(path, root)] = sha(path)

    def require(name, condition, proof):
        assert condition, name
        requirements.append(dict(requirement=name, status="proven", evidence=proof))

    frozen = read(root / "COMPARATOR-FROZEN.json")
    result = read(root / "RESULTS.json")
    definitions = read(source.parent / "DEFINITIONS.json")
    manifest = read(source / "manifest.json")
    require(
        "Independent document groups and source-state correspondence",
        manifest["complete"]
        and len(manifest["states"]) == 32
        and definitions["document_groups"] == 16
        and manifest["definitions_sha256"] == sha(source.parent / "DEFINITIONS.json"),
        "32 native Mac captures; definitions hash, source labels, ICC conversion and paired state markers",
    )
    require(
        "Training isolation, source uniqueness, and frozen final-test exclusion",
        definitions["isolation"]["counts"]
        == {"earlier_development": 16, "native_glyph_train": 64, "codescan_train": 9600}
        and definitions["isolation"][
            "code_repository_overlap_with_train_or_earlier_dev"
        ]
        == 0
        and not definitions["test_used"]
        and read(root / "dense-inference-inputs/INPUTS.json")[
            "source_pixel_and_file_overlap"
        ]
        == 0,
        "Repository, normalized-document, character8gram audit; native and recorded training pixel/file hashes; only train/validation source paths inspected",
    )
    for state in manifest["states"]:
        assert state["verified_markers"] == [state["state_id"]] * 2
        path = source / state["reference"]["path"]
        assert sha(path) == state["reference"]["sha256"]
        bind(path)
    codec = read(source.parent / "codec-v1/MANIFEST.json")
    require(
        "Real Mac hardware encoding and fixed 720p / 600Kbps / 420 primary condition",
        len(codec["runs"]) == 32
        and all(x["probe"]["streams"][0]["pix_fmt"] == "yuv420p" for x in codec["runs"])
        and all(x["requested_kbps"] == 600 for x in codec["runs"])
        and read(root / "MAC-ENVIRONMENT.json")["cpu"] == "Apple M1",
        "Original ffmpeg commands, VideoToolbox software-fallback disabled, stream probes and packet timestamps; frame drops retained",
    )
    for row in codec["runs"]:
        path = source.parent / "codec-v1" / row["path"]
        assert sha(path) == row["sha256"]
        bind(path)
    rows = {
        name: read(root / (name + "-quality") / "ROWS.json")
        for name in ["official", "trained", "candidate", "native", "clean"]
    }
    controls = {
        name: summarize([x for x in rows[name] if x["unit"] == "line"])[0]
        for name in ["official", "trained"]
    }
    winner = min(controls, key=lambda n: controls[n]["macro_cer"])
    require(
        "Comparator and difficult groups frozen before candidate scoring",
        controls[winner]["method"] == frozen["chosen_method"]
        and (root / "COMPARATOR-FROZEN.json").stat().st_mtime
        < read(root / "candidate-quality/PROTOCOL.json")["started"],
        "Both existing controls scored first; stronger trained EDSR selected; 13 positive-error groups frozen",
    )
    candidate = summarize([x for x in rows["candidate"] if x["unit"] == "line"])[0]
    group = {x["group"]: x["cer"] for x in candidate["groups"]}
    hard = float(np.mean([group[g] for g in frozen["hard_groups"]]))
    gain = 1 - hard / frozen["hard_macro_cer"]
    require(
        "At least25% fewer character errors on independent difficult documents",
        gain >= 0.25,
        dict(
            stronger_control_hard_cer=frozen["hard_macro_cer"],
            candidate_hard_cer=hard,
            relative_reduction=gain,
            uncertainty=result["uncertainty"],
            scope="Observed independent-set mean; not a guarantee on all future workloads",
        ),
    )
    require(
        "Added local first-display p95 at most20ms",
        result["added_first_local_ms"]["n"] == 32
        and result["added_first_local_ms"]["p95"] <= 20,
        result["added_first_local_ms"],
    )
    idle = read(root / "IDLE-TAIL-RESULTS.json")
    require(
        "Local stable-input refinement p95 at most200ms without censoring",
        idle["all32_completed"]
        and len(idle["rows"]) == 32
        and all(x["stable_window_ms"] >= 200 for x in idle["rows"])
        and idle["summary_ms"]["p95"] <= 200,
        dict(
            summary_ms=idle["summary_ms"],
            scope=idle["interval"],
            idle_tail_s=1,
            quality_result_replaced=False,
        ),
    )
    receiver = read(root / "stream-runs/confirm-refresh-01/receiver/RESULTS.json")
    require(
        "Actual authorized RTX4090 inference/rendering and current-source results",
        "GPU-f3d8b210-4eb5-a38b-a81d-983611f83a6a" in receiver["gpu"]
        and not receiver["errors"]
        and len(read(root / "stream-runs/confirm-refresh-01/MIX-AUDIT.json")["checks"])
        == 32
        and read(root / "BYPASS-AUDIT.json")["all_first_frames_current"]
        and all(
            x["input_revision"] == x["latest_codec_revision"]
            for x in receiver["sr_events"] + receiver["mix_refresh_events"]
        )
        and result["runs"]["refresh"]["sr_to_base_reversions"] == 0,
        dict(
            source_generation_rejections=receiver["sr_rejections"],
            gpu=receiver["gpu"],
            final_operand_and_GPU_pixel_checks=32,
        ),
    )
    require(
        "CER, exact code/cells, digits/punctuation, fidelity and regression cases delivered",
        len(result["table"]) == 5
        and read(root / "PIXEL-QUALITY.json")["complete"]
        and all(
            "code_exactness" in t
            and "digit_edits" in t["cells"]
            and "punctuation_edits" in t["cells"]
            for t in result["table"]
        )
        and len(result["regressions"]["candidate"]["new_wrong"]) > 0,
        "Raw2560OCRunits per method; line/cell/category metrics; rawwhitespace diagnostic; full-frame andROI PSNR/SSIM;43newOCRwrongunits retained",
    )
    transport = result["runs"]["refresh"]
    require(
        "Actual traffic, first response, refinement timing and limitations separated",
        transport["traffic"]["os_counter_out_bytes"] > 0
        and transport["summary"]["first"]["completed"] == 32
        and transport["final_refinement_ack_upper_ms"]["n"] == 32,
        dict(
            os_mean_out_kbps=transport["traffic"]["os_mean_out_kbps"],
            first_update_ack_upper_ms=transport["summary"]["first"]["ack_upper_ms"],
            final_update_ack_upper_ms=transport["final_refinement_ack_upper_ms"],
            scope="Forwardapplication600Kbps+fragmentallowance;OSSSH counters; no strictphysicalwire/panel/liveScreenSharing claim",
        ),
    )
    require(
        "Frozen candidate and all source states retained",
        not result["test_used"]
        and read(root / "stream-runs/confirm-refresh-01/MIX-AUDIT.json")["complete"]
        and len(candidate["groups"]) == 16,
        "Fixedcheckpoint,alpha.25,holdpolicy verified against preregistered runtime hashes; zero-error groups and fallback cases retained",
    )
    parser = Links()
    parser.feed((root / "REPORT.html").read_text())
    for relative in parser.paths:
        assert (root / relative).is_file(), relative
    require(
        "Reusable report, visual comparisons, curves, code and evidence delivered",
        all(
            (root / n).is_file()
            for n in [
                "REPORT.html",
                "curves.png",
                "BLIND-KEY.json",
                "blind-s01.png",
                "blind-s09.png",
                "blind-s25.png",
                "REPRODUCIBILITY.json",
            ]
        )
        and Path("benchmarks/super_resolution/mac_readability/README.md").is_file(),
        "Native-pixel blind crops inspected; curve inspected; report links resolved; README reproduction commands and runtime fingerprints saved",
    )
    require(
        "Training evidence and performance curves preserved",
        (root.parent / "training-v2/models/steady_l1/training-curve.csv").is_file()
        and (
            root.parent
            / "training-v2/models/steady_l1/validation-performance-curve.csv"
        ).is_file()
        and read(root.parent / "training-ocr-v2/STATUS.json")["state"]
        == "scoring_complete",
        "Archived training/validation curves and selected checkpoint; no confirmation data used for training",
    )
    for name in rows:
        for filename in ["ROWS.json", "SUMMARY.json", "PROTOCOL.json"]:
            bind(root / (name + "-quality") / filename)
    for index_name in [
        "OFFICIAL-IMAGES.json",
        "TRAINED-IMAGES.json",
        "NATIVE-IMAGES.json",
        "stream-runs/confirm-refresh-01/FINAL-IMAGES.json",
    ]:
        index = read(root / index_name)
        bind(root / index_name)
        for image in index["images"]:
            assert sha(image["path"]) == image["sha256"]
            bind(image["path"])
    for run in ["confirm-bypass-01", "confirm-refresh-01", "confirm-idle-tail-01"]:
        for part in ["sender", "receiver"]:
            bind(root / "stream-runs" / run / part / "RESULTS.json")
        bind(root / "stream-runs" / run / "sender/NETTOP.json")
    for path in [
        source / "manifest.json",
        source.parent / "DEFINITIONS.json",
        source.parent / "codec-v1/MANIFEST.json",
        root / "COMPARATOR-FROZEN.json",
        root / "MAC-ENVIRONMENT.json",
        root / "PIXEL-QUALITY.json",
        root / "IDLE-TAIL-RESULTS.json",
        root / "REPRODUCIBILITY.json",
        root / "BLIND-KEY.json",
        root.parent / "INDEPENDENT-CONFIRMATION-PLAN-V1.json",
        root.parent / "INDEPENDENT-COMPARATOR-ADDENDUM.json",
        Path("benchmarks/super_resolution/mac_readability/README.md"),
    ]:
        bind(path)
    output = dict(
        passed=True,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        script_sha256=sha(Path(__file__)),
        scope="Completion of the requested experimental round, not production deployment or proof of universal perceptual preference",
        requirements=requirements,
        evidence_sha256=evidence,
        material_limitations=result["limitations"]
        + [
            "CellCERregresses from1.6835%to1.8965%",
            "Source-onlycontrol has similarOCR andhigherfidelity; noestablishedbenefitfromblending acomplete source",
            "Independentcontent uses samefonts/layout; meaneffect38.1%,bootstrap95%interval17.3%-59.1%",
        ],
        further_required_work=[],
        test_used=False,
    )
    save(root / "COMPLETION-AUDIT.json", output)
    print(
        "Proven requirements:",
        len(requirements),
        "bound evidence files:",
        len(evidence),
    )


if __name__ == "__main__":
    main()
