"""FP32 pretrained single-frame evaluation with synchronized stage timings."""
import argparse
import csv
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch

from isharescreen.enhancement.models import Upscaler, availability, project_home, REPOS, sha256


def create_output(path):
    """Never mix a rerun with an old marker, CSV, or saved model output."""
    if path.exists():
        raise FileExistsError(f"Benchmark output already exists: {path}. Choose a fresh output directory.")
    path.mkdir(parents=True, exist_ok=False)


def metrics(reference, output, border=0):
    if reference.shape != output.shape:
        raise ValueError(f"Geometry mismatch: {reference.shape} vs {output.shape}")
    if border:
        reference, output = reference[border:-border, border:-border], output[border:-border, border:-border]
    return {"psnr": float(peak_signal_noise_ratio(reference, output, data_range=255)),
            "ssim": float(structural_similarity(reference, output, channel_axis=2, data_range=255))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=Path("benchmarks/super_resolution/artifacts/pairs/manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["bilinear", "bicubic", "sharpen", "edsr", "imdn", "ecbsr", "safmn"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["fp32","fp16","amp-fp16"], default="fp32")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timing-repeats", type=int, default=30)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--save-images", type=int, default=12, help="Save this many full outputs per model")
    args = parser.parse_args()
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    create_output(args.output)
    pairs = json.loads(args.pairs.read_text())[:args.limit]
    if not pairs:
        raise ValueError("No aligned pairs")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    import isharescreen.enhancement.models as adapters
    env = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "device": device, "command": sys.argv,
           "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else platform.machine(),
           "manifest_sha256": sha256(args.pairs), "tf32": False, "precision":args.precision,
           "pid": __import__("os").getpid(), "started": time.time(),
           "runner_sha256":sha256(Path(__file__)), "adapter_sha256":sha256(Path(adapters.__file__))}
    (args.output / "environment.json").write_text(json.dumps(env, indent=2))
    (args.output / "pip-freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    perceptual = None
    if args.lpips:
        import lpips
        perceptual = lpips.LPIPS(net="alex").eval().to(device)
    rows, statuses, timing, visuals = [], {}, {}, {}
    fields = ["model", "sample", "capture", "degradation", "psnr", "ssim", "psnr_border2", "ssim_border2",
              "lpips", "prepare_ms", "upload_ms", "inference_ms", "download_ms", "total_ms"]
    with (args.output / "per_frame.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); f.flush()
        for key in args.models:
            available = next((x for x in availability() if x["id"] == key), None)
            if not available or not available["available"]:
                statuses[key] = {"status": "unavailable", "reason": available["reason"] if available else "unknown model"}
                print(f"UNAVAILABLE {key}: {statuses[key]['reason']}", flush=True)
                continue
            print(f"loading {key}", flush=True)
            model = Upscaler(key, device=device, precision=args.precision)
            provenance = dict(model.provenance)
            if key in REPOS:
                provenance["source_sha"] = subprocess.check_output(
                    ["git", "-C", str(project_home()/"vendor"/REPOS[key]), "rev-parse", "HEAD"], text=True).strip()
            sample = np.array(Image.open(args.pairs.parent / pairs[0]["lr"]).convert("RGB"))
            t = time.monotonic(); model(sample)
            first_ms = (time.monotonic()-t)*1000
            for _ in range(args.warmup):
                model(sample)
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            repeated = [model(sample)[1] for _ in range(args.timing_repeats)]
            timing[key] = {stage: {"p50": float(np.percentile([r[stage] for r in repeated], 50)),
                                   "p95": float(np.percentile([r[stage] for r in repeated], 95))}
                           for stage in repeated[0]}
            timing[key].update(first_call_ms=first_ms, warmup=args.warmup, repeats=args.timing_repeats,
                               input_wh=[sample.shape[1], sample.shape[0]],
                               peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20 if device.startswith("cuda") else None)
            for i, pair in enumerate(pairs):
                for role in ["lr","hr"]:
                    if f"{role}_sha256" in pair and sha256(args.pairs.parent/pair[role]) != pair[f"{role}_sha256"]:
                        raise RuntimeError(f"Input changed: {pair[role]}")
                lr = np.array(Image.open(args.pairs.parent / pair["lr"]).convert("RGB"))
                hr = np.array(Image.open(args.pairs.parent / pair["hr"]).convert("RGB"))
                out, stages = model(lr)
                quality = metrics(hr, out); cropped = metrics(hr, out, border=2)
                lp = ""
                if perceptual is not None:
                    def tensor(im):
                        return torch.from_numpy(im.copy()).permute(2,0,1).unsqueeze(0).float().to(device)/127.5-1
                    with torch.inference_mode():
                        lp = float(perceptual(tensor(hr), tensor(out)).item())
                row = dict(model=key, sample=pair["id"], capture=pair["capture"], degradation=pair["degradation"],
                           **quality, psnr_border2=cropped["psnr"], ssim_border2=cropped["ssim"], lpips=lp, **stages)
                rows.append(row); writer.writerow(row); f.flush()
                if i in set(np.linspace(0,len(pairs)-1,min(args.save_images,len(pairs)),dtype=int)):
                    folder = args.output/"images"/pair["id"]; folder.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(out).save(folder/f"{key}.png")
                    if key == args.models[0]:
                        Image.fromarray(hr).save(folder/"reference.png")
                        Image.fromarray(lr).save(folder/"input.png")
                    # Fixed same-coordinate crop across all methods.
                    crop = Image.fromarray(out).crop((40,40,680,280))
                    visuals.setdefault(pair["id"], {"reference": Image.fromarray(hr).crop((40,40,680,280))})[key] = crop
                print(f"{key} {i+1}/{len(pairs)} {pair['id']} PSNR={quality['psnr']:.3f} SSIM={quality['ssim']:.5f} total={stages['total_ms']:.2f}ms", flush=True)
            statuses[key] = {"status": "completed", **provenance}
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            (args.output/"status.json").write_text(json.dumps(statuses, indent=2))
    summary = {}
    for key in args.models:
        summary[key] = {}
        for deg in sorted({r["degradation"] for r in rows}):
            selected = [r for r in rows if r["model"] == key and r["degradation"] == deg]
            if not selected:
                continue
            captures = sorted({r["capture"] for r in selected})
            summary[key][deg] = {"n_frames": len(selected), "n_captures": len(captures)}
            for metric in ["psnr", "ssim", "psnr_border2", "ssim_border2", "lpips"]:
                if selected[0][metric] == "":
                    continue
                means = [np.mean([r[metric] for r in selected if r["capture"] == c]) for c in captures]
                summary[key][deg][metric] = float(np.mean(means))
            base = {r["sample"]:r for r in rows if r["model"] == "bicubic" and r["degradation"] == deg}
            if all(r["sample"] in base for r in selected):
                for metric in ["psnr", "ssim"]:
                    summary[key][deg][f"delta_{metric}_vs_bicubic"] = float(np.mean([
                        np.mean([r[metric]-base[r["sample"]][metric] for r in selected if r["capture"] == c]) for c in captures]))
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2))
    (args.output/"timing.json").write_text(json.dumps(timing, indent=2))
    (args.output/"status.json").write_text(json.dumps(statuses, indent=2))
    for sample, crops in visuals.items():
        sheet = Image.new("RGB", (640, 268*len(crops)), "#242424"); draw=ImageDraw.Draw(sheet)
        for i, (key, crop) in enumerate(crops.items()):
            draw.text((8,i*268+6), key, fill="white"); sheet.paste(crop, (0,i*268+28))
        sheet.save(args.output/"images"/sample/"comparison.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,4), layout="constrained")
    for key, values in summary.items():
        if values:
            axes[0].plot(list(values), [v["psnr"] for v in values.values()], "o-", label=key)
    axes[0].set(xlabel="Degradation", ylabel="Capture-mean RGB PSNR (dB)"); axes[0].legend()
    keys=list(timing); axes[1].bar(keys,[timing[k]["total_ms"]["p50"] for k in keys])
    axes[1].set(ylabel="Warm processing p50 (ms), 540p → 1080p")
    axes[1].axhline(33.3,color="r",linestyle="--",label="33.3 ms"); axes[1].legend()
    fig.savefig(args.output/"quality_latency.png", dpi=160)
    (args.output/"COMPLETED.json").write_text(json.dumps({"finished":time.time(), "rows":len(rows), "models":statuses}, indent=2))
    print(f"COMPLETED: {len(rows)} measurements in {args.output}", flush=True)


if __name__ == "__main__":
    main()
