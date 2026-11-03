"""Compare frozen legacy checkpoints on matched 720p/600k/420 validation inputs."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np
from PIL import Image
import torch

from benchmarks.super_resolution.training_state import atomic_json
from benchmarks.super_resolution.validation_gpu_quality import quality
from benchmarks.super_resolution.video_perceptual import normalized_tensor
from benchmarks.super_resolution.train import perceptual_network
from isharescreen.enhancement.models import Upscaler, sha256
from isharescreen.enhancement import candidates

CONFIG = "720p-600k-420"


def read(path):
    return json.loads(path.read_text())


def adapter(model, device):
    return (
        Upscaler(model, device=device, precision="fp32")
        if model in ("edsr", "imdn", "safmn")
        else candidates.TorchUpscaler(model, device=device, precision="fp32")
    )


def display(model, rgb):
    native, _ = model(rgb)
    assert native.shape == (1440, 2560, 3)
    return cv2.resize(native, (1920, 1080), interpolation=cv2.INTER_AREA)


def evaluate(args):
    root = args.root.resolve()
    base = args.baseline.resolve()
    plan = read(root / "PLAN.json")
    assert sha256(base / "PROTOCOL.json") == plan["baseline_protocol_sha256"]
    assert sha256(base / "INPUTS.json") == plan["baseline_inputs_sha256"]
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    net, lpips = perceptual_network(args.device)
    cases = []
    for run in read(base / "INPUTS.json")["runs"]:
        if run["config"] != CONFIG:
            continue
        for sample in run["samples"]:
            for key in ("lr", "hr"):
                assert sha256(base / sample[key]) == sample[key + "_sha256"]
            cases.append(
                dict(
                    clip=run["clip"],
                    content=run["content"],
                    frame=sample["frame"],
                    lr=np.asarray(Image.open(base / sample["lr"]).convert("RGB")),
                    hr=np.asarray(Image.open(base / sample["hr"]).convert("RGB")),
                    middle=sample == run["samples"][1],
                )
            )
    assert len(cases) == 72

    def score(model, case):
        output = display(model, case["lr"])
        scores = quality(case["hr"], output)
        with torch.inference_mode():
            scores["lpips"] = float(
                net(
                    normalized_tensor(case["hr"], args.device),
                    normalized_tensor(output, args.device),
                ).item()
            )
        return output, scores

    # A rerendered official result must reproduce the original evaluation.
    check = adapter("quicksrnet", args.device)
    case = cases[0]
    _, scores = score(check, case)
    baseline = next(
        r
        for r in (
            json.loads(s)
            for s in (base / "evaluations/quicksrnet/records.jsonl")
            .read_text()
            .splitlines()
        )
        if r["config"] == CONFIG
        and r["clip"] == case["clip"]
        and r["frame"] == case["frame"]
    )
    for metric in ("psnr", "ssim", "lpips"):
        assert abs(scores[metric] - baseline[metric]) < 1e-5, (
            metric,
            scores[metric],
            baseline[metric],
        )
    atomic_json(
        root / "BASELINE-REPRODUCTION.json",
        dict(
            passed=True,
            actual=scores,
            expected={k: baseline[k] for k in ("psnr", "ssim", "lpips")},
        ),
    )
    del check
    torch.cuda.empty_cache()
    atomic_json(
        root / "HOST.json",
        dict(
            host=platform.node(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            lpips=lpips,
            runner_sha256=sha256(Path(__file__)),
            started=time.time(),
        ),
    )
    started = time.time()
    for index, job in enumerate(plan["jobs"]):
        out = root / "results" / job["id"]
        out.mkdir(parents=True, exist_ok=True)
        if (out / "COMPLETED.json").exists():
            assert read(out / "COMPLETED.json")["results_sha256"] == sha256(
                out / "results.json"
            )
            continue
        model = adapter(job["model"], args.device)
        current = None
        rows = []
        for case in cases:
            fold = (
                str(plan["fold_by_clip"][case["clip"]])
                if job["kind"] == "development_cv"
                else "all"
            )
            entry = job["checkpoints"][fold]
            if fold != current:
                weight = root / entry["weights"]
                assert sha256(weight) == entry["weights_sha256"]
                state = torch.load(weight, map_location="cpu", weights_only=True)
                assert all(torch.isfinite(v).all().item() for v in state.values())
                torch.nn.Module.load_state_dict(model.model, state, strict=True)
                model.model.eval().requires_grad_(False)
                current = fold
                model(case["lr"])
            output, scores = score(model, case)
            image = None
            if case["middle"]:
                path = out / "images" / (case["clip"] + ".png")
                path.parent.mkdir(exist_ok=True)
                Image.fromarray(output).save(path)
                image = str(path.relative_to(root))
            rows.append(
                dict(
                    clip=case["clip"],
                    content=case["content"],
                    frame=case["frame"],
                    model=job["model"],
                    variant=job["id"],
                    checkpoint_fold=fold,
                    checkpoint_sha256=entry["weights_sha256"],
                    image=image,
                    **scores,
                )
            )
        assert len(rows) == 72
        atomic_json(
            out / "results.json",
            dict(
                job=job,
                rows=rows,
                baseline_protocol_sha256=plan["baseline_protocol_sha256"],
            ),
        )
        atomic_json(
            out / "COMPLETED.json",
            dict(results_sha256=sha256(out / "results.json"), rows=len(rows)),
        )
        atomic_json(
            root / "STATUS.json",
            dict(
                completed_jobs=index + 1,
                total_jobs=len(plan["jobs"]),
                latest=job["id"],
                elapsed_s=time.time() - started,
            ),
        )
        print("COMPLETED", index + 1, "/", len(plan["jobs"]), job["id"], flush=True)
        del model
        torch.cuda.empty_cache()
    atomic_json(
        root / "EVALUATION-COMPLETED.json",
        dict(jobs=len(plan["jobs"]), finished=time.time()),
    )


def report(args):
    root = args.root.resolve()
    base = args.baseline.resolve()
    plan = read(root / "PLAN.json")
    metrics = ("psnr", "ssim", "lpips")
    baselines = {}
    for method in (
        "baseline",
        "edsr",
        "imdn",
        "safmn",
        "quicksrnet",
        "plainusr",
        "rt4ksr",
        "efrlfn",
    ):
        path = base / "evaluations" / method / "records.jsonl"
        assert (
            sha256(path) == read(base / "COVERAGE-AUDIT.json")["results_sha256"][method]
        )
        rows = [
            r
            for r in map(json.loads, path.read_text().splitlines())
            if r["config"] == CONFIG
        ]
        baselines[method] = {(r["clip"], r["frame"]): r for r in rows}
    summaries = []
    for job in plan["jobs"]:
        folder = root / "results" / job["id"]
        assert read(folder / "COMPLETED.json")["results_sha256"] == sha256(
            folder / "results.json"
        )
        rows = read(folder / "results.json")["rows"]
        official = baselines[job["model"]]
        plain = baselines["baseline"]
        assert {(r["clip"], r["frame"]) for r in rows} == set(official) == set(plain)
        score = dict(
            id=job["id"],
            model=job["model"],
            kind=job["kind"],
            frames=len(rows),
            clips=24,
        )
        for metric in metrics:
            score[metric] = float(np.mean([r[metric] for r in rows]))
            score["delta_official_" + metric] = float(
                np.mean(
                    [
                        r[metric] - official[(r["clip"], r["frame"])][metric]
                        for r in rows
                    ]
                )
            )
            score["delta_no_sr_" + metric] = float(
                np.mean(
                    [r[metric] - plain[(r["clip"], r["frame"])][metric] for r in rows]
                )
            )
        summaries.append(score)
    atomic_json(
        root / "SUMMARY.json",
        dict(
            config=CONFIG,
            baselines={
                m: {k: float(np.mean([r[k] for r in values.values()])) for k in metrics}
                for m, values in baselines.items()
            },
            variants=summaries,
        ),
    )
    lines = [
        "# Legacy checkpoints at 720p / 600 Kbps / 4:2:0",
        "",
        "24 matched Mac recordings, three frames each; FP32; identical decoded inputs and 1080p display reconstruction to the official baseline campaign.",
        "",
        "The 1,500-epoch CV models use the checkpoint from the fold holding out each recording. Checkpoints were selected by the historical validation rule before this experiment. These recordings share a capture session and were previously inspected; this remains development regression, not an independent final test.",
        "",
        "| Variant | PSNR ↑ | Δ vs official | SSIM ↑ | Δ vs official | LPIPS ↓ | Δ vs official |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        lines.append(
            f"| {s['id']} | {s['psnr']:.3f} | {s['delta_official_psnr']:+.3f} | {s['ssim']:.5f} | {s['delta_official_ssim']:+.5f} | {s['lpips']:.5f} | {s['delta_official_lpips']:+.5f} |"
        )
    lines += [
        "",
        "No-SR reference: "
        + str(
            {
                k: round(
                    float(np.mean([r[k] for r in baselines["baseline"].values()])), 5
                )
                for k in metrics
            }
        ),
        "",
        "Higher PSNR/SSIM and lower LPIPS are preferred. These are image-quality metrics, not verified text correctness. Text-corpus models did not train on the Mac recording pool according to their frozen policies, but possible document-content overlap is not fully excluded.",
        "",
        "No new weights were trained in this comparison. The planned new 720p/600k/420 training remains a separate fresh experiment.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(
        root / "COMPLETED.json",
        dict(
            variants=len(summaries),
            rows=len(summaries) * 72,
            summary_sha256=sha256(root / "SUMMARY.json"),
            finished=time.time(),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("evaluate", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    globals()[args.action](args)


if __name__ == "__main__":
    main()
