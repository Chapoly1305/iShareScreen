"""Prototype same-weight CUDA pipeline with fixed 4:3 area downsampling.

This is an experiment only, not a viewer backend. The GPU postprocessor first
rounds/clamps each native SR pixel, then applies the same area footprint as
OpenCV 1440p -> 1080p. Compare outputs against OpenCV before reporting speed.
"""
import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
import triton
import triton.language as tl

from benchmarks.super_resolution.profile_inference_optimization import adapter, amp, digest, forward, guard, save, stats


@triton.jit
def pack_area_kernel(X, Y, SC: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
                     N: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = idx % 3
    col = (idx // 3) % 1920
    row = idx // (3 * 1920)
    sx = (col * 4) // 3
    sy = (row * 4) // 3
    wx = (3 - col % 3).to(tl.float32) * .25
    wy = (3 - row % 3).to(tl.float32) * .25
    base = c * SC + sy * SH + sx * SW
    a = tl.load(X + base, idx < N, 0).to(tl.float32)
    b = tl.load(X + base + SW, idx < N, 0).to(tl.float32)
    d = tl.load(X + base + SH, idx < N, 0).to(tl.float32)
    e = tl.load(X + base + SH + SW, idx < N, 0).to(tl.float32)
    a = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(a * 255), 0), 255)
    b = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(b * 255), 0), 255)
    d = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(d * 255), 0), 255)
    e = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(e * 255), 0), 255)
    val = (a * wx + b * (1-wx)) * wy + (d * wx + e * (1-wx)) * (1-wy)
    val = tl.extra.cuda.libdevice.nearbyint(val)
    tl.store(Y + idx, val.to(tl.uint8), idx < N)


def pack_area(x):
    assert x.shape == (1, 3, 1440, 2560)
    out = torch.empty((1080, 1920, 3), dtype=torch.uint8, device=x.device)
    pack_area_kernel[(triton.cdiv(out.numel(), 256),)](x, out, *x.stride()[1:], out.numel(), 256, enable_fp_fusion=False)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--variant", choices=["channels_last", "compile_cl"], required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    dest = a.output / f"{a.model}-{a.variant}-fused.json"
    if dest.exists(): raise FileExistsError(dest)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    plan = json.loads((a.weights / "PLAN.json").read_text())
    item = next(i for i in plan["models"] if i["model"] == a.model)
    ckpt = a.weights / "checkpoints" / f"{a.model}.pt"
    assert digest(ckpt) == item["sha256"]
    result = dict(complete=False, checkpoint=item, variant=a.variant,
                  gpu_start=guard(), script_sha256=digest(Path(__file__)), pid=os.getpid(),
                  started=time.time(), torch=str(torch.__version__),
                  scope="CPU uint8 720p RGB -> GPU normalization/forward/round-clamp/area1080p -> CPU uint8 RGB. No decode/network/display. AMP FP16; same weights and native model resolution.")
    save(dest,result)
    try:
        model = adapter(a.model,"cuda:0").model.eval().requires_grad_(False)
        torch.nn.Module.load_state_dict(model,torch.load(ckpt,map_location="cpu",weights_only=True)["state_dict"],strict=True)
        model.to(memory_format=torch.channels_last)
        def predict(t): return forward(model,a.model,t)
        fn = torch.compile(predict,fullgraph=True,dynamic=False) if a.variant == "compile_cl" else predict
        runs = [r for r in json.loads((a.validation/"INPUTS.json").read_text())["runs"] if r["config"]=="720p-600k-420"]
        with torch.inference_mode(),amp():
            # Validate the fused operation on all 72 Mac validation frames.
            errors=[]
            for run in runs:
                for s in run["samples"]:
                    path=a.validation/s["lr"]
                    assert digest(path)==s["lr_sha256"]
                    rgb=np.asarray(Image.open(path).convert("RGB"))
                    gpu=torch.from_numpy(rgb.copy()).to("cuda").permute(2,0,1).unsqueeze(0).float()/255
                    out=fn(gpu)
                    assert torch.isfinite(out).all().item()
                    native=out[0].float().mul(255).round().clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy()
                    expected=cv2.resize(native,(1920,1080),interpolation=cv2.INTER_AREA)
                    actual=pack_area(out).cpu().numpy()
                    delta=np.abs(expected.astype(np.int16)-actual.astype(np.int16))
                    errors.append(dict(clip=run["clip"],frame=s["frame"],max_rgb=int(delta.max()),mean_abs_rgb=float(delta.mean()),fraction_different=float((delta!=0).mean())))
            assert max(r["max_rgb"] for r in errors)<=1
            result["area_validation"]=errors
            # Fixed frame for stage timing, identical to original pipeline probe.
            s=runs[0]["samples"][1]
            rgb=np.asarray(Image.open(a.validation/s["lr"]).convert("RGB"))
            stages={k:[] for k in ("prepare","upload_normalize","model","pack_resize_download","total")}
            for i in range(550):
                torch.cuda.synchronize(); t0=time.perf_counter()
                t=torch.from_numpy(rgb); t1=time.perf_counter()
                t=t.to("cuda").permute(2,0,1).unsqueeze(0).float()/255
                torch.cuda.synchronize(); t2=time.perf_counter()
                out=fn(t)
                torch.cuda.synchronize(); t3=time.perf_counter()
                if not torch.isfinite(out).all().item(): raise ValueError("Nonfinite")
                display=pack_area(out).cpu().numpy(); t4=time.perf_counter()
                if i>=50:
                    for k,v in zip(stages,(t1-t0,t2-t1,t3-t2,t4-t3,t4-t0)): stages[k].append(v*1000)
            result.update(complete=True,finished=time.time(),gpu_end=guard(),pipeline={k:stats(v) for k,v in stages.items()})
            save(dest,result)
            print(a.model,a.variant,"fused pipeline p95",result["pipeline"]["total"]["p95_ms"],"max area error",max(r["max_rgb"] for r in errors),flush=True)
    except BaseException as e:
        result.update(error=repr(e));save(dest,result);raise


if __name__=="__main__": main()
