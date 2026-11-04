"""Isolated, hash-bound inference experiments; never changes published results.

Run each model/variant in a fresh process. Baseline reproduces the published
resident-forward boundary. Optional CPU RGB pipeline timings are separate.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import cv2
import numpy as np
from PIL import Image
import torch

from benchmarks.super_resolution.benchmark_best_latency import guard, save
from benchmarks.super_resolution.codescan_campaign.train import forward
from benchmarks.super_resolution.legacy_validation_compare import adapter


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return dict(p50_ms=float(np.percentile(values, 50)),
                p95_ms=float(np.percentile(values, 95)),
                p99_ms=float(np.percentile(values, 99)),
                mean_ms=float(np.mean(values)), samples_ms=values)


def amp():
    return torch.autocast("cuda", dtype=torch.float16)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--variant", choices=["baseline", "channels_last", "compile", "compile_cl", "graph_cl"], required=True)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--quality", action="store_true")
    p.add_argument("--pipeline", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dest = args.output / f"{args.model}-{args.variant}.json"
    if dest.exists():
        raise FileExistsError(dest)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1234)
    plan = json.loads((args.weights / "PLAN.json").read_text())
    item = next(row for row in plan["models"] if row["model"] == args.model)
    ckpt = args.weights / "checkpoints" / f"{args.model}.pt"
    assert digest(ckpt) == item["sha256"]
    result = dict(complete=False, model=args.model, variant=args.variant,
                  checkpoint=item, host=platform.node(), gpu_start=guard(),
                  torch=str(torch.__version__), cuda=torch.version.cuda,
                  cudnn=torch.backends.cudnn.version(), script_sha256=digest(Path(__file__)),
                  started=time.time(), pid=os.getpid(), samples=args.samples,
                  warmup=args.warmup, input_shape=[1, 3, 720, 1280],
                  precision="AMP FP16; FP32 weights/input; TF32 disabled",
                  scope="Resident forward 720p to native1440p, same normalization as published benchmark. Fixed synthetic input. CUDA event timing without profiler.",
                  command=os.sys.argv)
    save(dest, result)
    try:
        wrapper = adapter(args.model, "cuda:0")
        model = wrapper.model.eval().requires_grad_(False)
        payload = torch.load(ckpt, map_location="cpu", weights_only=True)
        torch.nn.Module.load_state_dict(model, payload["state_dict"], strict=True)
        del payload
        x = torch.linspace(0, 1, 1280 * 720 * 3, device="cuda").reshape(1, 3, 720, 1280)
        with torch.inference_mode():
            with amp():
                reference = forward(model, args.model, x).float().clone()
            use_cl = args.variant in ("channels_last", "compile_cl", "graph_cl")
            fmt = torch.channels_last if use_cl else torch.contiguous_format
            model.to(memory_format=fmt)
            x = x.contiguous(memory_format=fmt)

            def eager(t):
                return forward(model, args.model, t)

            fn = eager
            if args.variant.startswith("compile"):
                fn = torch.compile(eager, fullgraph=True, dynamic=False)
            compile_started = time.perf_counter()
            with amp():
                for _ in range(args.warmup):
                    y = fn(x)
            torch.cuda.synchronize()
            result["setup_and_warmup_s"] = time.perf_counter() - compile_started
            if args.variant == "graph_cl":
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream), amp():
                    for _ in range(5):
                        graph_output = eager(x)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph), amp():
                    graph_output = eager(x)

                def replay(t):
                    if t.data_ptr() != x.data_ptr():
                        x.copy_(t)
                    graph.replay()
                    return graph_output

                fn = replay
                with amp():
                    y = fn(x)
            assert y.shape == reference.shape and torch.isfinite(y).all().item()
            diff = y.float() - reference
            result["synthetic_vs_baseline_amp"] = dict(max_abs=float(diff.abs().max()),
                                                       mean_abs=float(diff.abs().mean()))
            del reference, diff
            # Keep autocast cache active for the timed block, as in original.
            timings, wall = [], []
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.reset_peak_memory_stats()
            with amp():
                for _ in range(10):
                    y = fn(x)
                torch.cuda.synchronize()
                for i in range(args.samples):
                    t0 = time.perf_counter()
                    start.record()
                    y = fn(x)
                    end.record()
                    end.synchronize()
                    timings.append(start.elapsed_time(end))
                    wall.append((time.perf_counter() - t0) * 1000)
            result.update(forward=stats(timings), host_wall=stats(wall),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(), gpu_after=guard())
            save(dest, result)
            print(args.model, args.variant, "forward p95", result["forward"]["p95_ms"], flush=True)
            if args.profile:
                with amp(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                              torch.profiler.ProfilerActivity.CUDA],
                                                    record_shapes=True) as prof:
                    for _ in range(5):
                        y = fn(x)
                    torch.cuda.synchronize()
                events = prof.key_averages(group_by_input_shape=True)
                rows = [dict(op=e.key, count=e.count, shapes=e.input_shapes,
                             self_cuda_us=e.self_device_time_total,
                             total_cuda_us=e.device_time_total,
                             self_cpu_us=e.self_cpu_time_total) for e in events]
                rows.sort(key=lambda e: e["self_cuda_us"], reverse=True)
                result["profile"] = rows
                result["profile_calls"] = 5
                prof.export_chrome_trace(str(args.output / f"{args.model}-{args.variant}.trace.json"))
                save(dest, result)

            manifest = json.loads((args.validation / "INPUTS.json").read_text())
            runs = [r for r in manifest["runs"] if r["config"] == "720p-600k-420"]
            if args.pipeline:
                sample = runs[0]["samples"][1]
                path = args.validation / sample["lr"]
                assert digest(path) == sample["lr_sha256"]
                rgb = np.asarray(Image.open(path).convert("RGB"))
                stages = {k: [] for k in ["prepare", "upload", "model", "pack_download", "resize", "total"]}
                # Diagnostic stage synchronizations deliberately expose costs.
                # This is a CPU-RGB-to-CPU-RGB path, not live input-to-display.
                with amp():
                    for i in range(120):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        t = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).float().div_(255)
                        t1 = time.perf_counter()
                        t = t.to("cuda").contiguous(memory_format=fmt)
                        torch.cuda.synchronize()
                        t2 = time.perf_counter()
                        out = fn(t)
                        torch.cuda.synchronize()
                        t3 = time.perf_counter()
                        # Include the same finite check and RGB quantization.
                        if not torch.isfinite(out).all().item():
                            raise ValueError("Nonfinite pipeline output")
                        pixels = out[0].float().mul(255).round().clamp(0, 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                        t4 = time.perf_counter()
                        display = cv2.resize(pixels, (1920, 1080), interpolation=cv2.INTER_AREA)
                        t5 = time.perf_counter()
                        if i >= 20:
                            for name, delta in zip(stages, [t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t5-t0]):
                                stages[name].append(delta * 1000)
                result["pipeline"] = {k: stats(v) for k, v in stages.items()}
                result["pipeline_scope"] = "CPU RGB normalization, H2D, forward, finite check, native1440p uint8 packing/D2H, OpenCV area1080p. Synchronized stages, no decode/network/queue/display."
                save(dest, result)

            if args.quality:
                # Separate FP32 reference retains original NCHW and exact weights.
                ref_wrapper = adapter(args.model, "cuda:0")
                ref_model = ref_wrapper.model.eval().requires_grad_(False)
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
                torch.nn.Module.load_state_dict(ref_model, state["state_dict"], strict=True)
                del state
                records = []
                for run in runs:
                    for sample in run["samples"]:
                        for kind in ("lr", "hr"):
                            assert digest(args.validation / sample[kind]) == sample[kind + "_sha256"]
                        low = np.asarray(Image.open(args.validation / sample["lr"]).convert("RGB"))
                        target = np.asarray(Image.open(args.validation / sample["hr"]).convert("RGB"))
                        t = torch.from_numpy(np.ascontiguousarray(low.transpose(2, 0, 1))).unsqueeze(0).to("cuda").float() / 255
                        ref = forward(ref_model, args.model, t)
                        with amp():
                            baseline = forward(ref_model, args.model, t)
                            out = fn(t.contiguous(memory_format=fmt)).clone()
                        assert torch.isfinite(out).all().item()
                        outputs = {}
                        for name, tensor in [("fp32", ref), ("baseline_amp", baseline), ("optimized", out)]:
                            pixels = tensor[0].float().mul(255).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                            outputs[name] = cv2.resize(pixels, (1920, 1080), interpolation=cv2.INTER_AREA)
                        row = dict(clip=run["clip"], frame=sample["frame"], lr_sha256=sample["lr_sha256"], psnr={})
                        for name, pixels in outputs.items():
                            mse = np.mean((pixels.astype(np.float64)-target.astype(np.float64))**2)
                            row["psnr"][name] = float(10 * np.log10(255**2 / mse))
                        for name in ("fp32", "baseline_amp"):
                            delta = np.abs(outputs["optimized"].astype(np.int16)-outputs[name].astype(np.int16))
                            row["vs_" + name] = dict(max_rgb=int(delta.max()), mean_abs_rgb=float(delta.mean()),
                                                    fraction_gt1=float((delta > 1).mean()))
                        records.append(row)
                result["quality"] = dict(samples=len(records), scope="72 Mac validation frames; FP32/NCHW vs AMP/NCHW vs optimized. No test data or checkpoint tuning. RGB quantized final1080p PSNR; not SSIM/LPIPS equivalence.",
                                         mean_psnr={name: float(np.mean([r["psnr"][name] for r in records])) for name in outputs}, records=records)
            result.update(complete=True, finished=time.time(), gpu_end=guard())
            save(dest, result)
    except BaseException as exc:
        result.update(error=repr(exc), failed=time.time())
        save(dest, result)
        raise


if __name__ == "__main__":
    main()
