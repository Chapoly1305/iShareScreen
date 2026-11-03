"""Independent fixed-candidate quality and real-transport evidence."""

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmarks.super_resolution.mac_readability.quality import save, sha, summarize
from benchmarks.super_resolution.mac_readability.stream_stability_report import audit
from benchmarks.super_resolution.mac_readability.stream_report import stats


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    source = args.source
    frozen = read(root / "COMPARATOR-FROZEN.json")
    assert frozen["complete"] and not frozen["candidate_scored"]
    specs = {
        "clean": "clean-quality",
        "official": "official-quality",
        "trained": "trained-quality",
        "native": "native-quality",
        "candidate": "candidate-quality",
    }
    protocols = {
        name: read(root / folder / "PROTOCOL.json") for name, folder in specs.items()
    }
    original = read(root.parent / "dense-clean-v1/PROTOCOL.json")
    for protocol in protocols.values():
        assert protocol["complete"] and protocol["source_sha256"] == sha(
            source / "manifest.json"
        )
        for key in ["tesseract", "tessdata", "preprocessing"]:
            assert protocol[key] == original[key], key
    assert (root / "COMPARATOR-FROZEN.json").stat().st_mtime < (
        root / "candidate-quality/PROTOCOL.json"
    ).stat().st_mtime
    rows = {name: read(root / folder / "ROWS.json") for name, folder in specs.items()}
    comparator = (
        "trained" if frozen["chosen_method"] == "steady_l1-step001000" else "official"
    )
    baseline = {(x["state_id"], x["label_id"]): x for x in rows[comparator]}
    tables = []
    regressions = {}
    for name, data in rows.items():
        assert {(x["state_id"], x["label_id"]) for x in data} == set(baseline)
        lines = summarize([x for x in data if x["unit"] == "line"])[0]
        cells = summarize([x for x in data if x["unit"] == "cell"])[0]
        groups = {x["group"]: x["cer"] for x in lines["groups"]}
        hard = float(np.mean([groups[g] for g in frozen["hard_groups"]]))
        tables.append(
            dict(
                name=name,
                hard_cer=hard,
                relative_gain=1 - hard / frozen["hard_macro_cer"],
                lines=lines,
                cells=cells,
                categories={
                    c: summarize(
                        [x for x in data if x["unit"] == "line" and x["category"] == c]
                    )[0]
                    for c in ["code", "table", "english", "chinese"]
                },
                zero_group_edits=sum(
                    x["edits"]
                    for x in data
                    if x["unit"] == "line" and x["group"] in frozen["zero_error_groups"]
                ),
            )
        )
        code = [x for x in data if x["unit"] == "line" and x["category"] == "code"]
        tables[-1]["code_exactness"] = dict(
            lines=len(code),
            nonspace_exact=sum(x["exact_nonspace"] for x in code),
            raw_whitespace_exact=sum(
                x["text_predicted"].rstrip("\n\f") == x["text_gt"] for x in code
            ),
            limitation="TesseractlineOCR doesnotreliablyrecoverindentation;rawwhitespaceexactness isdiagnostic,notproofofrenderedcodeindentation",
        )
        regressions[name] = dict(
            new_wrong=[
                x
                for x in data
                if baseline[x["state_id"], x["label_id"]]["exact_nonspace"]
                and not x["exact_nonspace"]
            ],
            repaired=[
                x
                for x in data
                if not baseline[x["state_id"], x["label_id"]]["exact_nonspace"]
                and x["exact_nonspace"]
            ],
        )
    by_name = {x["name"]: x for x in tables}
    candidate = by_name["candidate"]
    a = np.array(
        [
            next(
                x["cer"]
                for x in by_name[comparator]["lines"]["groups"]
                if x["group"] == g
            )
            for g in frozen["hard_groups"]
        ]
    )
    b = np.array(
        [
            next(x["cer"] for x in candidate["lines"]["groups"] if x["group"] == g)
            for g in frozen["hard_groups"]
        ]
    )
    draws = np.random.default_rng(20261103).integers(0, len(a), (10000, len(a)))
    uncertainty = dict(
        paired_absolute_gain_ci95=np.quantile(
            (a[draws] - b[draws]).mean(1), [0.025, 0.975]
        ).tolist(),
        paired_relative_gain_ci95=np.quantile(
            1 - b[draws].mean(1) / a[draws].mean(1), [0.025, 0.975]
        ).tolist(),
        unit="document group",
        resamples=10000,
        seed=20261103,
    )
    runs = {
        name: audit(root, "confirm-" + name + "-01") for name in ["bypass", "refresh"]
    }
    control = {x["state_id"]: x for x in runs["bypass"]["points"]}
    added = stats(
        [
            x["first"]["local_ms"] - control[x["state_id"]]["first"]["local_ms"]
            for x in runs["refresh"]["points"]
        ]
    )
    pixels = read(root / "PIXEL-QUALITY.json")
    assert pixels["complete"]
    for table in tables:
        if table["name"] == "clean":
            continue
        values = [x for x in pixels["rows"] if x["method"] == table["name"]]
        table["pixels"] = {
            k: float(
                np.mean(
                    [
                        np.mean([x[k] for x in values if x["group"] == g])
                        for g in {x["group"] for x in values}
                    ]
                )
            )
            for k in ["full_psnr", "roi_psnr", "full_ssim", "roi_ssim"]
        }
    raw = read(root / "stream-runs/confirm-refresh-01/receiver/RESULTS.json")
    plan = read(root.parent / "INDEPENDENT-CONFIRMATION-PLAN-V1.json")
    sender = read(root / "stream-runs/confirm-refresh-01/sender/RESULTS.json")
    assert raw["script_sha256"] == plan["candidate"]["scripts"]["stream_receiver.py"]
    assert sender["script_sha256"] == plan["candidate"]["scripts"]["stream_sender.py"]
    assert (
        raw["display_state_sha256"]
        == plan["candidate"]["scripts"]["stream_display_state.py"]
    )
    assert raw["checkpoint_sha256"] == plan["candidate"]["checkpoint_sha256"]
    result = dict(
        complete=True,
        scope=__doc__,
        script_sha256=sha(Path(__file__)),
        source_manifest_sha256=sha(source / "manifest.json"),
        comparator_sha256=sha(root / "COMPARATOR-FROZEN.json"),
        comparator=comparator,
        table=tables,
        regressions=regressions,
        uncertainty=uncertainty,
        runs=runs,
        added_first_local_ms=added,
        quality_mean_target_met=candidate["hard_cer"] <= frozen["target_hard_cer"],
        first_local_target_met=added["p95"] <= 20,
        accepted_refresh_local_target_met=runs["refresh"][
            "refresh_job_input_to_readback_ms"
        ]["p95"]
        <= 200,
        full_goal_met=False,
        production_selected=False,
        test_used=False,
        limitations=[
            "New independent document content; same fonts and layout templates",
            "Mac hardware-coded frozen replay, not liveScreenSharing capture; softwareHEVC decoding and GPUreadback, noNVDEC/panelclaim",
            "One sequential run per mode; processSSHcounters and pacedapplicationbudget, not strictphysicalwirecap",
            "Accepted-job input latency excludes waiting for admissible inputs; not a200ms source-change orend-to-end claim",
            "OCR is not human visual preference; sourcefidelity and newcell/digiterrors remain explicit",
        ],
        evidence={
            name: sha(root / folder / "ROWS.json") for name, folder in specs.items()
        },
    )
    result["rejected_obsolete_results"] = dict(
        count=raw["stale_sr_results"], reasons=raw["sr_rejections"]
    )
    result["primary_final_revision_latency_audit"] = read(
        root / "FINAL-REVISION-LATENCY-AUDIT.json"
    )
    idle_path = root / "IDLE-TAIL-RESULTS.json"
    if idle_path.exists():
        result["uncensored_local_stability"] = read(idle_path)
    completion_path = root / "COMPLETION-AUDIT.json"
    if completion_path.exists():
        completion = read(completion_path)
        for relative, expected_hash in completion["evidence_sha256"].items():
            assert sha(root / relative) == expected_hash, relative
        result["full_goal_met"] = completion["passed"]
        result["completion_audit_sha256"] = sha(completion_path)
    save(root / "RESULTS.json", result)
    # Fixed illustrations, plus randomized blind keys saved separately.
    manifests = {
        "official": "OFFICIAL-IMAGES.json",
        "trained": "TRAINED-IMAGES.json",
        "native": "NATIVE-IMAGES.json",
        "candidate": "stream-runs/confirm-refresh-01/FINAL-IMAGES.json",
    }
    indexes = {
        name: {x["state_id"]: x for x in read(root / path)["images"]}
        for name, path in manifests.items()
    }
    states = {x["state_id"]: x for x in read(source / "manifest.json")["states"]}
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    rng = np.random.default_rng(20261103)
    blind = []
    examples = []
    for sid in [1, 9, 25]:
        order = list(rng.permutation(list(indexes)))
        canvas = Image.new("RGB", (1240, 500), "#eeeeee")
        draw = ImageDraw.Draw(canvas)
        draw.text((320, 7), "Ground truth", font=font, fill="#111111")
        box = (36, 78, 636, 208)
        canvas.paste(
            Image.open(source / states[sid]["reference"]["path"])
            .convert("RGB")
            .crop(box),
            (320, 30),
        )
        mapping = {}
        for i, name in enumerate(order):
            x = 10 + (i % 2) * 620
            y = 175 + (i // 2) * 160
            letter = chr(65 + i)
            draw.text((x, y), letter, font=font, fill="#111111")
            canvas.paste(
                Image.open(indexes[name][sid]["path"]).convert("RGB").crop(box),
                (x, y + 24),
            )
            mapping[letter] = name
        filename = f"blind-s{sid:02d}.png"
        canvas.save(root / filename)
        examples.append(filename)
        blind.append(dict(state_id=sid, mapping=mapping, crop_xyxy=box))
    save(
        root / "BLIND-KEY.json",
        dict(
            seed=20261103,
            examples=blind,
            selection="Fixedsource states1,9,25 andfixedcrop, not selectedbyquality outcomes",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    names = ["official", "trained", "native", "candidate"]
    positions = np.arange(4)
    axes[0].bar(names, [by_name[n]["hard_cer"] * 100 for n in names])
    axes[0].axhline(
        frozen["target_hard_cer"] * 100,
        color="red",
        ls="--",
        label="25% reduction target",
    )
    axes[0].set(ylabel="Hard-group macro CER (%)", title="Independent documents")
    axes[0].legend(fontsize=8)
    for n in ["bypass", "refresh"]:
        timed = sorted(runs[n]["timed_ocr"], key=lambda x: x["source_age_s"])
        axes[1].plot(
            [x["source_age_s"] for x in timed],
            [x["macro_cer"] * 100 for x in timed],
            ".-",
            label=n,
        )
    axes[1].set(
        xlabel="Receiver source-announcement age(s)",
        ylabel="All16 document CER(%)",
        title="Actual visible pixels, including blanks",
    )
    axes[1].legend()
    fig.savefig(root / "curves.png", dpi=150)
    plt.close(fig)
    table_html = ""
    for t in tables:
        table_html += f"<tr><td>{t['name']}</td><td>{100 * t['hard_cer']:.3f}%</td><td>{100 * t['relative_gain']:.1f}%</td><td>{100 * t['cells']['macro_cer']:.3f}%</td><td>{t['cells']['exact_units']}/{t['cells']['units']}</td><td>{t['cells']['digit_edits']}</td><td>{t.get('pixels', {}).get('roi_psnr', float('inf')):.2f}</td></tr>"
    v = runs["refresh"]
    traffic = v["traffic"]
    passed = result["quality_mean_target_met"]
    idle_note = ""
    if "uncensored_local_stability" in result:
        idle = result["uncensored_local_stability"]
        idle_note = f'<p><b>Separate local-stability validation:</b> each source was held for one additional second after its original packet sequence. All32 latest-input refinements completed: {idle["all32_completed"]}; p95 {idle["summary_ms"]["p95"]:.1f}ms. This prevents source switches from censoring the local completion check. Receiver/model/source/packets are unchanged. The quality table continues to use the original3second primary run. <a href="IDLE-TAIL-RESULTS.json">Every input, completion and stable window</a>.</p>'
    document = f"""<!doctype html><meta charset="utf-8"><title>Independent source-backed text confirmation</title>
<style>body{{font:17px system-ui;max-width:1250px;margin:40px auto;line-height:1.5}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:9px}}img{{max-width:100%}}</style>
<h1>Independent confirmation: fixed source-backed refinement</h1>
<p>This confirmation uses 16 new documents and 32 native Mac screenshots, with unchanged 14/18 px fonts. Code repositories are disjoint from training and earlier development. Source text, pixel hashes, and character-shingle similarity audits are preserved. The original final test remains untouched. One overlong mixed-language line was shortened to fit its native font width before any model evaluation; the original text and failed capture evidence remain.</p>
<p>The stronger existing control was frozen before candidate scoring: <b>{comparator}</b>. Its {len(frozen["hard_groups"])} positive-error document groups define the hard set; all 16 groups and zero-error regressions are retained. The fixed candidate blends 25% exact native 4:2:0 source with 75% current-source SR, and updates the blend when later valid SR results arrive. No parameters were tuned on this confirmation set.</p>
<table><tr><th>Method</th><th>Hard CER</th><th>Reduction vs stronger control</th><th>Cell CER</th><th>Exact cells</th><th>Cell digit edits</th><th>Text ROI PSNR dB</th></tr>{table_html}</table>
<p>Mean 25% quality target met: <b>{passed}</b>. The paired document-bootstrap 95% interval for relative error reduction is {uncertainty["paired_relative_gain_ci95"][0] * 100:.1f}% to {uncertainty["paired_relative_gain_ci95"][1] * 100:.1f}%. This does not guarantee a 25% improvement on every new workload. Newly incorrect OCR units versus the stronger control: {len(regressions["candidate"]["new_wrong"])}; repaired units: {len(regressions["candidate"]["repaired"])}. These count OCR outputs, not proven visual character substitutions.</p>
<p><b>Source detail is the main practical result.</b> Source-only pixels and the blend have nearly identical line CER on this set. Source-only pixels are much closer to the original in PSNR/SSIM. The experiment does not establish a benefit from mixing SR into a complete source image. The blend also regresses in cell CER versus trained EDSR, despite improving overall line CER. No human preference result is claimed.</p>
<p>Added local first-display p95: {added["p95"]:.2f} ms. Accepted source-backed refresh input to GPU readback p95: {v["refresh_job_input_to_readback_ms"]["p95"]:.1f} ms. First packet to first accepted SR p95: {v["first_packet_receipt_to_first_sr_ms"]["p95"]:.1f} ms. Scheduled source change to the final refinement acknowledgment p95: {v["final_refinement_ack_upper_ms"]["p95"]:.1f} ms. This is not a 200 ms end-to-end result. SR-to-base reversions within unchanged sources: {v["sr_to_base_reversions"]}.</p>
{idle_note}
<p>OS-reported SSH outgoing mean: {traffic["os_mean_out_kbps"]:.1f} Kbps; largest approximate one-second sample: {traffic["os_approximate_1s_out_kbps"]["maximum"]:.1f} Kbps. The shared 600 Kbps application/framing allowance is not a strict physical-wire cap. Mac VideoToolbox output includes recorded frame drops; exact evaluation timestamps and markers exist for all 32 primary frames. Actual transmission, software HEVC decoding, CUDA AMP inference, and WGPU readback are measured. Live Screen Sharing and physical panel latency are outside this replay measurement.</p>
<img src="curves.png"><h2>Fixed blind native-pixel crops</h2><p>The same source crop is shown for each method, with randomized candidate identities. Open at 100% before judging. No human preference result has been collected.</p>
{"".join('<p><img src="' + x + '"></p>' for x in examples)}
<p><a href="BLIND-KEY.json">Blind key</a> · <a href="RESULTS.json">Raw results, timing and regression cases</a> · <a href="PIXEL-QUALITY.json">Full-frame and text-region PSNR/SSIM</a> · <a href="COMPARATOR-FROZEN.json">Frozen stronger comparator</a></p>"""
    (root / "REPORT.html").write_text(document)
    print(
        "Independent quality target",
        passed,
        "hardCER",
        candidate["hard_cer"],
        "gain",
        candidate["relative_gain"],
        "addedfirst",
        added,
    )


if __name__ == "__main__":
    main()
