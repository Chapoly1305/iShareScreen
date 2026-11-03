"""Report offline and actual-stream source blending, including regressions."""

import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmarks.super_resolution.mac_readability.quality import save, sha, summarize


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--refresh-run")
    args = parser.parse_args()
    root = args.root
    source = args.source
    frozen = read(root / "COMPARATOR-FROZEN.json")
    specs = [
        (
            "Official EDSR",
            "dense-model-quality-v1",
            "dense-models-v1/IMAGES-LOCAL.json",
            "edsr-official",
        ),
        (
            "Best trained EDSR",
            "training-v2-selected-full-quality",
            "TRAINING-V2-SELECTED-IMAGES.json",
            None,
        ),
        (
            "Exact native420 source",
            "component-quality-v1",
            "precodec-images-v1/IMAGES.json",
            "native",
        ),
        (
            "Offline 25% source +75% SR",
            "source-mix-selected-quality",
            "SOURCE-MIX-SELECTED-IMAGES.json",
            None,
        ),
        (
            "Stream initial blend",
            "stream-runs/stream-stability-holdmix-01/final-quality",
            "stream-runs/stream-stability-holdmix-01/FINAL-IMAGES.json",
            None,
        ),
    ]
    if args.refresh_run:
        specs.append(
            (
                "Stream refreshed blend",
                f"stream-runs/{args.refresh_run}/final-quality",
                f"stream-runs/{args.refresh_run}/FINAL-IMAGES.json",
                None,
            )
        )
    tables = []
    indexes = {}
    evidence = {}
    line_sets = {}
    for name, quality, index, method in specs:
        rows = read(root / quality / "ROWS.json")
        if method == "edsr-official":
            rows = [x for x in rows if x["method"] == method]
        assert len(rows) == 2852 and read(root / quality / "PROTOCOL.json")["complete"]
        lines = [x for x in rows if x["unit"] == "line"]
        cells = [x for x in rows if x["unit"] == "cell"]
        line_sets[name] = lines
        line_summary = summarize(lines)[0]
        cell_summary = summarize(cells)[0]
        group_cer = {x["group"]: x["cer"] for x in line_summary["groups"]}
        hard = float(np.mean([group_cer[g] for g in frozen["hard_groups"]]))
        tables.append(
            dict(
                name=name,
                hard_cer=hard,
                relative_gain_vs_frozen=1 - hard / frozen["hard_macro_cer"],
                mean_target_met=hard <= frozen["target_hard_cer"],
                lines=line_summary,
                cells=cell_summary,
                categories={
                    c: summarize([x for x in lines if x["category"] == c])[0]
                    for c in ["code", "table", "chinese", "english"]
                },
                zero_error_group_edits=sum(
                    x["edits"]
                    for x in lines
                    if x["group"] in frozen["zero_error_groups"]
                ),
            )
        )
        images = read(root / index)["images"]
        if method == "edsr-official":
            images = [x for x in images if x["method"] == method]
        if method == "native":
            images = [x for x in images if x["config"] == "1080p-420-no-codec"]
        indexes[name] = {x["state_id"]: x for x in images}
        assert len(indexes[name]) == 32
        evidence[name] = dict(
            rows_sha256=sha(root / quality / "ROWS.json"),
            index_sha256=sha(root / index),
        )
    base = {(x["state_id"], x["label_id"]): x for x in line_sets["Best trained EDSR"]}
    regressions = {}
    for name, lines in line_sets.items():
        regressions[name] = dict(
            new_wrong_vs_best=[
                x
                for x in lines
                if base[x["state_id"], x["label_id"]]["exact_nonspace"]
                and not x["exact_nonspace"]
            ],
            repaired_vs_best=[
                x
                for x in lines
                if not base[x["state_id"], x["label_id"]]["exact_nonspace"]
                and x["exact_nonspace"]
            ],
        )
    # Independent full/cell evaluation must reproduce the earlier selected line scores.
    selected = [
        x
        for x in read(root / "source-mix-quality-v1/ROWS.json")
        if x["method"] == "source-mix-25"
    ]
    earlier = {(x["state_id"], x["label_id"]): x["text_predicted"] for x in selected}
    assert all(
        earlier[x["state_id"], x["label_id"]] == x["text_predicted"]
        for x in line_sets["Offline 25% source +75% SR"]
    )
    states = {x["state_id"]: x for x in read(source / "manifest.json")["states"]}
    roi = []
    for sid, state in states.items():
        reference = np.asarray(
            Image.open(source / state["reference"]["path"]).convert("RGB")
        ).astype(np.float32)
        mask = np.zeros(reference.shape[:2], bool)
        for label in state["labels"]:
            x, y, w, h = label["xywh"]
            mask[y : y + h, x : x + w] = True
        for name, index in indexes.items():
            item = index[sid]
            assert sha(item["path"]) == item["sha256"]
            output = np.asarray(Image.open(item["path"]).convert("RGB")).astype(
                np.float32
            )
            mse = float(np.square(reference[mask] - output[mask]).mean())
            roi.append(
                dict(
                    method=name,
                    state_id=sid,
                    group=state["group"],
                    mse=mse,
                    psnr=float(10 * np.log10(255**2 / mse)),
                )
            )
    for table in tables:
        values = [x for x in roi if x["method"] == table["name"]]
        table["text_roi_psnr"] = float(
            np.mean(
                [
                    np.mean([x["psnr"] for x in values if x["group"] == g])
                    for g in states_group(states)
                ]
            )
        )
    save(
        root / "SOURCE-MIX-RESULTS.json",
        dict(
            complete=True,
            script_sha256=sha(Path(__file__)),
            table=tables,
            evidence=evidence,
            regressions=regressions,
            text_roi=roi,
            all_ratios=read(root / "SOURCE-MIX-SELECTION.json")["candidates"],
            target_hard_cer=frozen["target_hard_cer"],
            test_used=False,
            full_goal_met=False,
            production_selected=False,
            limitations=[
                "Repeated development selection; independent confirmation pending",
                "OCR does not establish human readability preference; native source itself has OCR errors",
                "Offline blend uses frozen2.8s SR; stream blend uses actually available SR and variable arrival times",
                "Source supplements share the paced transport; 25% pixel weight does not mean25% payload",
                "Cell/digit regressions versus trained SR retained, even when line average improves",
            ],
        ),
    )
    # Fixed source states and crop coordinates, chosen independently of OCR scores.
    examples = []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    chosen_names = [
        "Official EDSR",
        "Best trained EDSR",
        "Exact native420 source",
        "Offline 25% source +75% SR",
        specs[-1][0],
    ]
    for sid in [1, 9, 25]:
        box = (36, 78, 636, 208)
        labels = ["Ground truth"] + chosen_names
        canvas = Image.new("RGB", (640, len(labels) * 166 + 20), "#eeeeee")
        draw = ImageDraw.Draw(canvas)
        for row, name in enumerate(labels):
            path = (
                source / states[sid]["reference"]["path"]
                if name == "Ground truth"
                else Path(indexes[name][sid]["path"])
            )
            draw.text((20, row * 166 + 6), name, fill="#111111", font=font)
            canvas.paste(
                Image.open(path).convert("RGB").crop(box), (20, row * 166 + 29)
            )
        filename = f"source-mix-comparison-s{sid:02d}.png"
        canvas.save(root / filename)
        examples.append(filename)
    rows = ""
    for t in tables:
        rows += f"<tr><td>{html.escape(t['name'])}</td><td>{t['hard_cer'] * 100:.3f}%</td><td>{t['relative_gain_vs_frozen'] * 100:.1f}%</td><td>{t['cells']['macro_cer'] * 100:.3f}%</td><td>{t['cells']['exact_units']}/1806</td><td>{t['cells']['digit_edits']}</td><td>{t['text_roi_psnr']:.2f}</td></tr>"
    document = f"""<!doctype html><meta charset="utf-8"><title>Source-backed blending: quality and regressions</title>
<style>body{{font:17px system-ui;max-width:1180px;margin:40px auto;line-height:1.5}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:9px}}img{{max-width:100%}}</style>
<h1>Source-backed blending: offline gains and actual-stream checks</h1>
<p>All results use the frozen32 native Mac source states,16 development documents and unchanged OCR. The primary hard15 metric excludes the previously frozen zero-error English statistics group. No final test was opened.</p>
<table><tr><th>Method</th><th>Hard15 line CER</th><th>Reduction vs official</th><th>Cell CER</th><th>Exact cells</th><th>Digit edits in cells</th><th>Text ROI PSNR dB</th></tr>{rows}</table>
<p>Three global source weights25%,50%,75% were scored;25% was selected on development. Its offline mean clears the25% relative target, but cell CER and digit edits regress versus the best trained SR. Exact native source has the best cell accuracy among these candidates. A good OCR score does not mean the image is more faithful than its source. Native420 source reaches50.80dB in the text regions versus30.09dB for the selected offline blend. This large fidelity gap and visible source sharpness show why OCR alone must not choose the final display.</p>
<p>The first actual-stream blend froze SR at supplement arrival.30 of31 SR inputs differed from the frozen steady frame; one state fell back to native source. Its hard CER was2.187%, so it did not reproduce the offline gain. This failure is retained rather than replaced by the offline number.</p>
<p>Pixel blending weights are not bandwidth fractions: the full exact source supplement is still transmitted. Network and local latency are reported separately in the <a href="STREAM-STABILITY-REPORT.html">stream report</a>. Independent development confirmation and human visual preference remain open; these are not production acceptance results.</p>
<h2>Fixed native-pixel crops</h2><p>Source states1,9,25; same crop coordinates for every method. Open images at100% to inspect characters. These are fixed illustrations, not selected wins.</p>
{"".join('<p><img src="' + x + '"></p>' for x in examples)}
<p><a href="SOURCE-MIX-RESULTS.json">Per-category, per-group, new errors, repaired lines, hashes and ROI metrics</a></p>"""
    (root / "SOURCE-MIX-REPORT.html").write_text(document)
    print(
        json.dumps(
            [
                {
                    k: t[k]
                    for k in [
                        "name",
                        "hard_cer",
                        "relative_gain_vs_frozen",
                        "text_roi_psnr",
                    ]
                }
                for t in tables
            ],
            indent=2,
        )
    )


def states_group(states):
    return sorted({x["group"] for x in states.values()})


if __name__ == "__main__":
    main()
