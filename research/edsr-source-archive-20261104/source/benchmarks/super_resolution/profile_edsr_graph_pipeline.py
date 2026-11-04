"""Fixed-buffer EDSR runtime: one CUDA Graph per frame, GPU postprocessing.

The GPU path has no host copies or per-frame finite reduction. The host path
keeps both transfers in its graph. A separately measured checked host path
preserves the older per-frame finite check with only one final synchronization.
All paths must match the unfused engine's OpenCV output on 72 validation frames.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np
from PIL import Image
import torch

from benchmarks.super_resolution.profile_edsr_tiles import Engine
from benchmarks.super_resolution.profile_fused_pipeline import pack_area_kernel
from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save, stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--samples",type=int,default=1000)
    p.add_argument("--capture",action="store_true")
    p.add_argument("--gpu-only",action="store_true",help="Measure only resident model and resident GPU pipeline")
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);cv2.setNumThreads(1)
    result=dict(complete=False,started=time.time(),pid=os.getpid(),host=platform.node(),gpu=guard(),
        engine_sha256=digest(a.engine),script_sha256=digest(Path(__file__)),
        area_kernel_source_sha256=digest(Path(__file__).with_name("profile_fused_pipeline.py")),
        command=os.sys.argv,profiler=a.capture,samples=a.samples,warmup=50,
        scope="GPU: resident RGB uint8 720p -> normalized EDSR native1440p -> rounded/clamped/area RGB uint8 1080p resident. Host: CPU RGB720p -> same GPU pipeline -> CPU RGB1080p. No decode/network/display; per-frame latency, not pipelined throughput.")
    save(a.output,result)
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
    engine=Engine(a.engine)
    gpu_rgb=torch.empty((720,1280,3),dtype=torch.uint8,device="cuda")
    host_in=torch.empty_like(gpu_rgb,device="cpu",pin_memory=True)
    gpu_out=torch.empty((1080,1920,3),dtype=torch.uint8,device="cuda")
    host_out=torch.empty_like(gpu_out,device="cpu",pin_memory=True)
    finite_gpu=torch.empty((),dtype=torch.bool,device="cuda")
    finite_host=torch.empty((),dtype=torch.bool,pin_memory=True)
    source=torch.zeros_like(host_in)

    def compute():
        engine.input.copy_(gpu_rgb.permute(2,0,1).unsqueeze(0).float()/255)
        y=engine.run()
        pack_area_kernel[((gpu_out.numel()+255)//256,)](y,gpu_out,*y.stride()[1:],gpu_out.numel(),256,enable_fp_fusion=False)

    def gpu():compute()

    def host():
        gpu_rgb.copy_(host_in,non_blocking=True);compute()
        host_out.copy_(gpu_out,non_blocking=True)

    def host_checked():
        host()
        finite_gpu.copy_(torch.isfinite(engine.output).all())
        finite_host.copy_(finite_gpu,non_blocking=True)

    functions=dict(model=engine.run,gpu=gpu)
    if not a.gpu_only:functions.update(host=host,host_checked=host_checked)
    graphs={}
    with torch.inference_mode():
        gpu_rgb.zero_();host_in.zero_()
        for name,fn in functions.items():
            for _ in range(5):fn()
            stream.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):fn()
            graphs[name]=graph

        records=[]
        runs=[r for r in json.loads((a.validation/"INPUTS.json").read_text())["runs"] if r["config"]=="720p-600k-420"]
        for run in runs:
            for s in run["samples"]:
                path=a.validation/s["lr"];assert digest(path)==s["lr_sha256"]
                rgb=np.asarray(Image.open(path).convert("RGB")).copy();source=torch.from_numpy(rgb)
                engine.input.copy_(source.permute(2,0,1).unsqueeze(0).float()/255)
                y=engine.run();assert torch.isfinite(y).all().item()
                native=y[0].mul(255).round().clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy()
                expected=cv2.resize(native,(1920,1080),interpolation=cv2.INTER_AREA)
                host_in.copy_(source);gpu_rgb.copy_(host_in,non_blocking=True)
                row=dict(clip=run["clip"],frame=s["frame"],lr_sha256=s["lr_sha256"],paths={})
                for name in (n for n in functions if n!="model"):
                    graphs[name].replay();stream.synchronize()
                    actual=gpu_out.cpu().numpy() if name=="gpu" else host_out.numpy()
                    delta=np.abs(actual.astype(np.int16)-expected.astype(np.int16))
                    row["paths"][name]=dict(max_rgb=int(delta.max()),different_values=int(np.count_nonzero(delta)))
                    assert int(delta.max())==0,row
                    if name=="host_checked":assert bool(finite_host)
                records.append(row)
        assert len(records)==72
        result["area_validation"]=records;save(a.output,result)
        sample=runs[0]["samples"][1]
        rgb=np.asarray(Image.open(a.validation/sample["lr"]).convert("RGB")).copy();source=torch.from_numpy(rgb)
        host_in.copy_(source);gpu_rgb.copy_(host_in,non_blocking=True)
        engine.input.copy_(gpu_rgb.permute(2,0,1).unsqueeze(0).float()/255)
        for name,graph in graphs.items():
            for _ in range(50):graph.replay()
        stream.synchronize()
        result["timings"]={}
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        if a.capture:torch.cuda.cudart().cudaProfilerStart()
        for name,graph in graphs.items():
            wall=[];device=[]
            torch.cuda.nvtx.range_push("phase_"+name)
            for i in range(a.samples):
                torch.cuda.nvtx.range_push("frame_"+name)
                t0=time.perf_counter();begin.record()
                if name.startswith("host"):host_in.copy_(source)
                graph.replay();end.record();end.synchronize()
                if name=="host_checked":assert bool(finite_host)
                wall.append((time.perf_counter()-t0)*1000);device.append(begin.elapsed_time(end))
                torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()
            result["timings"][name]=dict(wall=stats(wall),cuda_event=stats(device))
        if a.capture:torch.cuda.cudart().cudaProfilerStop()
        result.update(complete=True,finished=time.time(),gpu_end=guard(),
            context_memory_bytes=engine.engine.device_memory_size_v2,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save(a.output,result)
        print(json.dumps({k:{"wall_p95":v["wall"]["p95_ms"],"gpu_p95":v["cuda_event"]["p95_ms"]} for k,v in result["timings"].items()}),flush=True)


if __name__=="__main__":main()
