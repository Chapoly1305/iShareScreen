"""Render and OCR frozen source-truth regions after bound model training.

Validation-proxy-selected weights are fixed before the test predictions. Both
clean-reference floors and every OCR prediction are retained; no text correction.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import html
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from benchmarks.super_resolution.training_state import atomic_json, fingerprint
from isharescreen.enhancement.models import sha256, project_home
from benchmarks.super_resolution.text_corpus.runtime_paths import tessdata


def read(path):
    return json.loads(path.read_text())


def render(args):
    import cv2
    import torch
    from benchmarks.super_resolution.text_corpus.scratch import build
    from benchmarks.super_resolution.text_corpus.ocr_protocol import prepare_crop
    from isharescreen.enhancement import candidates

    root = args.root.resolve()
    training = args.training
    out = (args.output or root / "evaluation-v1").resolve()
    out.relative_to(root)
    out.mkdir(parents=True, exist_ok=False)
    protocol = read(root / "evaluation-protocol/protocol.json")
    policy = read(training / "protocol.json")
    done = read(training / "COMPLETED.json")
    if done["protocol_binding"] != fingerprint(policy) or done["best_sha256"] != sha256(
        training / "best.pt"
    ):
        raise ValueError("Training checkpoint receipt differs")
    if protocol["finished"] >= read(training / "host.json")["started"]:
        raise ValueError("Evaluation regions were not frozen before training")
    initialization = policy["initialization"]["kind"]
    if initialization not in ("from_scratch", "official_pretrained"):
        raise ValueError("Unrecognized initialization")
    paired = read(root / "hevc-v2/pairs.json")
    compat = read(root / "evaluation-protocol/source-compatibility.json")
    if policy["ocr_protocol_sha256"] != sha256(
        root / "evaluation-protocol/protocol.json"
    ) or policy["ocr_source_compatibility_sha256"] != sha256(
        root / "evaluation-protocol/source-compatibility.json"
    ):
        raise ValueError("Evaluation protocol differs from its pre-training binding")
    if compat["final_views_sha256"] != paired["views_sha256"]:
        raise ValueError("Evaluation source compatibility differs")
    pairs = {r["id"]: r for r in paired["pairs"]}
    device = args.device
    if any(
        pairs[c["view"]]["requested_bps"] in policy.get("excluded_bps", [])
        for c in protocol["cases"]
    ):
        raise ValueError("Evaluation contains excluded bitrate")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    identities = {}
    for method in ["bilinear", "bicubic", "pretrained", "initial", "trained"]:
        model = None
        if method == "pretrained":
            from benchmarks.super_resolution.text_corpus.official import load

            model, _ = load(args.home)
            identities[method] = dict(
                kind="official_pretrained",
                checkpoint_sha256=candidates.specification("quicksrnet")["files"][
                    "weights.pth"
                ]["sha256"],
            )
        elif method in ("initial", "trained"):
            model, _ = build(args.home, policy["seed"])
            path = training / ("initial.pt" if method == "initial" else "best.pt")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["protocol_binding"] != done["protocol_binding"]:
                raise ValueError("Wrong checkpoint binding")
            model.load_state_dict(payload["state_dict"], strict=True)
            identities[method] = dict(
                kind=initialization
                + ("_initial" if method == "initial" else "_trained"),
                epoch=payload["epoch"],
                checkpoint_sha256=sha256(path),
            )
        if model is not None:
            model.to(device).eval().requires_grad_(False)
        folder = out / method
        folder.mkdir()
        for n, case in enumerate(protocol["cases"], 1):
            pair = pairs[case["view"]]
            if (
                sha256(root / pair["lr"]) != pair["lr_sha256"]
                or pair["hr_sha256"] != case["hr_sha256"]
                or pair["labels_sha256"] != case["labels_sha256"]
            ):
                raise ValueError("Evaluation pair differs from protocol")
            rgb = np.array(Image.open(root / pair["lr"]).convert("RGB"))
            if model is None:
                pixels = cv2.resize(
                    rgb,
                    (1920, 1080),
                    interpolation=cv2.INTER_LINEAR
                    if method == "bilinear"
                    else cv2.INTER_CUBIC,
                )
            else:
                tensor = (
                    torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None]
                    .to(device)
                    .float()
                    / 255
                )
                with torch.inference_mode():
                    prediction = model(tensor)
                if (
                    prediction.shape != (1, 3, 1080, 1920)
                    or not torch.isfinite(prediction).all()
                ):
                    raise ValueError("Invalid reconstructed pixels")
                pixels = (
                    prediction[0]
                    .float()
                    .mul(255)
                    .round()
                    .clamp(0, 255)
                    .byte()
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
            path = folder / (case["id"] + ".png")
            prepare_crop(Image.fromarray(pixels), case["xywh"], path)
            records.append(
                dict(
                    id=case["id"],
                    method=method,
                    view=case["view"],
                    crop=str(path.relative_to(root)),
                    sha256=sha256(path),
                    requested_bps=pair["requested_bps"],
                    actual_episode_bps=pair["actual_episode_video_bps"],
                    keyframe=pair["keyframe"],
                )
            )
            if n % 40 == 0:
                print("EVAL RENDER", method, n, "/", len(protocol["cases"]), flush=True)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    atomic_json(
        out / "rendered.json",
        dict(
            schema="isharescreen_text_predictions_v2",
            records=records,
            identities=identities,
            ocr_protocol_sha256=sha256(root / "evaluation-protocol/protocol.json"),
            training_policy_sha256=sha256(training / "protocol.json"),
            training_receipt_sha256=sha256(training / "COMPLETED.json"),
            device=device,
            torch=str(torch.__version__),
            gpu=torch.cuda.get_device_name() if device.startswith("cuda") else device,
            script_sha256=sha256(Path(__file__)),
            finished=time.time(),
            scope="Same decoded 540p RGB, common FP32 x2 reconstruction and final FHD line crops. CPU bilinear/bicubic are offline RGB baselines, not the native YUV viewer shader or live latency.",
        ),
    )
    (out / "render-runner.py").write_bytes(Path(__file__).read_bytes())


def summarize(rows):
    chars = sum(r["characters"] for r in rows)
    edits = sum(r["edit_distance"] for r in rows)
    groups = defaultdict(list)
    for r in rows:
        groups[r["source_group"]].append(r["cer"])
    return dict(
        cases=len(rows),
        source_groups=len(groups),
        mean_line_cer=float(np.mean([r["cer"] for r in rows])),
        equal_source_group_cer=float(np.mean([np.mean(v) for v in groups.values()])),
        pooled_cer=edits / chars,
        characters=chars,
        edits=edits,
        exact_lines=sum(r["exact"] for r in rows),
        worst_line_cer=max(r["cer"] for r in rows),
        digits=sum(r["digits"] for r in rows),
        digit_edits=sum(r["digit_edits"] for r in rows),
        punctuation=sum(r["punctuation"] for r in rows),
        punctuation_edits=sum(r["punctuation_edits"] for r in rows),
    )


def score(args):
    from benchmarks.super_resolution.text_corpus.ocr_protocol import score_crop

    root = args.root.resolve()
    out = (args.output or root / "evaluation-v1").resolve()
    out.relative_to(root)
    protocol = read(root / "evaluation-protocol/protocol.json")
    predictions = read(out / "rendered.json")
    if predictions["ocr_protocol_sha256"] != sha256(
        root / "evaluation-protocol/protocol.json"
    ):
        raise ValueError("OCR protocol changed")
    if protocol["tessdata_sha256"] != sha256(tessdata() / "eng.traineddata"):
        raise ValueError("Different OCR model")
    if protocol["tesseract"] != subprocess.check_output(
        ["tesseract", "--version"], stderr=subprocess.STDOUT, text=True
    ):
        raise ValueError("Different OCR implementation")
    cases = {c["id"]: c for c in protocol["cases"]}

    def task(row):
        if sha256(root / row["crop"]) != row["sha256"]:
            raise ValueError("Prediction crop changed")
        case = cases[row["id"]]
        return dict(
            row,
            split=case["split"],
            kind=case["kind"],
            source_group=case["source_group"],
            text_gt=case["text"],
            **score_crop(root / row["crop"], case["text"]),
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(task, predictions["records"]))
    for case in protocol["cases"]:
        rows.append(
            dict(
                id=case["id"],
                method="reference",
                split=case["split"],
                kind=case["kind"],
                source_group=case["source_group"],
                text_gt=case["text"],
                crop=case["clean_crop"],
                sha256=case["clean_crop_sha256"],
                **case["clean_ocr"],
            )
        )
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["method"])].append(row)
    summary = [
        dict(split=split, method=method, **summarize(values))
        for (split, method), values in grouped.items()
    ]
    atomic_json(out / "ocr-rows.json", rows)
    atomic_json(out / "summary.json", summary)
    lines = [
        "# New text corpus: QuickSRNet result",
        "",
        "Model selection uses the fixed validation pixel/text/edge proxy, before inspecting test OCR. Frozen source-derived English line cases retain clean-reference OCR failures. No OCR or language-model correction is applied.",
        "",
        "| Split | Method | Lines | Mean line CER % | Equal-source-group CER % | Exact lines | Digit edits / GT | Punctuation edits / GT |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['split']} | {r['method']} | {r['cases']} | {r['mean_line_cer'] * 100:.3f} | {r['equal_source_group_cer'] * 100:.3f} | {r['exact_lines']} | {r['digit_edits']}/{r['digits']} | {r['punctuation_edits']}/{r['punctuation']} |"
        )
    lines += [
        "",
        f"These are {sum(c['split'] == 'validation' for c in protocol['cases'])} validation and {sum(c['split'] == 'test' for c in protocol['cases'])} test single-line regions frozen before training. PSM7 line scores cannot be compared directly with the earlier PSM6 multi-line benchmark. Whitespace/indentation, full-screen transcription, temporal stability, Chinese, native viewer latency and constrained-network usability are not established.",
        "",
        "[Every prediction](ocr-rows.json) · [Model and input provenance](rendered.json) · [Frozen source truth]("
        + os.path.relpath(root / "evaluation-protocol/protocol.json", out)
        + ")",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    page = [
        '<!doctype html><html><meta charset="utf-8"><title>Text model evaluation</title><style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 20px}img{max-width:100%}pre{white-space:pre-wrap}details{margin:18px 0}</style><h1>Source-held-out text evaluation</h1><p><a href="REPORT.md">Results and scope</a> · <a href="summary.json">Metrics</a> · <a href="ocr-rows.json">Exact OCR predictions</a></p>'
    ]
    for case in [c for c in protocol["cases"] if c["split"] == "test"]:
        page.append(
            f"<details><summary>{html.escape(case['kind'] + ' / ' + case['id'])}</summary><pre>GT: {html.escape(case['text'])}</pre>"
        )
        for method in [
            "reference",
            "bilinear",
            "bicubic",
            "pretrained",
            "initial",
            "trained",
        ]:
            row = next(
                r for r in rows if r["id"] == case["id"] and r["method"] == method
            )
            page.append(
                f'<p>{method}: OCR CER {row["cer"] * 100:.2f}%</p><img loading="lazy" src="{html.escape(os.path.relpath(root / row["crop"], out), quote=True)}"><pre>{html.escape(row["predicted"])}</pre>'
            )
        page.append("</details>")
    page.append("</html>")
    (out / "index.html").write_text("\n".join(page))
    atomic_json(
        out / "COMPLETED.json",
        dict(
            rows_sha256=sha256(out / "ocr-rows.json"),
            summary_sha256=sha256(out / "summary.json"),
            predictions_sha256=sha256(out / "rendered.json"),
            script_sha256=sha256(Path(__file__)),
            finished=time.time(),
        ),
    )
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["render", "ocr"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--training", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--home", type=Path, default=project_home())
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    dict(render=render, ocr=score)[args.action](args)


if __name__ == "__main__":
    main()
