"""Minimal Nsight target with NVTX stages and explicit memory provenance.

Nsight Compute: --mode resident --iterations 1 --capture profiler
Nsight Systems: --mode all --iterations 50 --capture profiler
Run without an attached profiler for latency claims.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import cv2
import numpy as np
import torch
import tensorrt as trt


@contextmanager
def nvtx(name):
    torch.cuda.nvtx.range_push(name)
    try: yield
    finally: torch.cuda.nvtx.range_pop()


def tensor_info(t):
    return dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),device=str(t.device),
                pointer=t.data_ptr(),alignment_mod256=t.data_ptr()%256,storage_offset=t.storage_offset(),
                contiguous=t.is_contiguous(),channels_last=t.ndim==4 and t.is_contiguous(memory_format=torch.channels_last),
                pinned=t.is_pinned(),bytes=t.numel()*t.element_size())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--uuid",required=True);p.add_argument("--mode",choices=["resident","cpu","pinned","fused","all"],default="all")
    p.add_argument("--validation",type=Path,help="Check all 72 Mac validation frames against CPU packing/resize")
    p.add_argument("--iterations",type=int,default=50);p.add_argument("--capture",choices=["none","profiler"],default="none")
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    use_fused=a.mode in ("all","fused") or a.validation is not None
    if use_fused:
        from benchmarks.super_resolution.profile_fused_pipeline import pack_area_kernel
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu","--format=csv,noheader"],text=True).strip()
    assert a.uuid in gpu
    processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True)
    assert all(int(pid)==os.getpid() for pid in processes.splitlines() if pid.strip().isdigit()),processes
    torch.set_num_threads(1);cv2.setNumThreads(1)
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
    logger=trt.Logger(trt.Logger.WARNING);runtime=trt.Runtime(logger)
    engine=runtime.deserialize_cuda_engine(a.engine.read_bytes());ctx=engine.create_execution_context()
    x=torch.linspace(0,1,1280*720*3,device="cuda").reshape(1,3,720,1280)
    y=torch.empty((1,3,1440,2560),dtype=torch.float32,device="cuda")
    ctx.set_tensor_address("input",x.data_ptr());ctx.set_tensor_address("output",y.data_ptr())
    rgb=(np.arange(720*1280*3,dtype=np.uint32)%256).astype(np.uint8).reshape(720,1280,3)
    pinned_in=torch.empty((720,1280,3),dtype=torch.uint8,pin_memory=True)
    pinned_out=torch.empty((1440,2560,3),dtype=torch.uint8,pin_memory=True)
    gpu_rgb=torch.empty_like(pinned_in,device="cuda")
    source=torch.from_numpy(rgb)
    if use_fused:
        fused_out=torch.empty((1080,1920,3),dtype=torch.uint8,device="cuda")
        fused_host=torch.empty_like(fused_out,device="cpu",pin_memory=True)

    def infer():
        if not ctx.execute_async_v3(stream.cuda_stream):raise RuntimeError("TensorRT failed")

    def resident():
        with nvtx("model_forward"):infer()
        stream.synchronize()

    def cpu():
        with nvtx("cpu_prepare_fp32_nchw"):
            host=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).unsqueeze(0).float()/255
        with nvtx("h2d_pageable_fp32"):
            x.copy_(host);stream.synchronize()
        with nvtx("model_forward"):infer();stream.synchronize()
        with nvtx("finite_check_pack_d2h"):
            if not torch.isfinite(y).all().item():raise ValueError("nonfinite")
            image=y[0].mul(255).round().clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy()
        with nvtx("cpu_area_resize"):
            return cv2.resize(image,(1920,1080),interpolation=cv2.INTER_AREA)

    def pinned():
        with nvtx("cpu_copy_to_reused_pinned_input"):pinned_in.copy_(source)
        with nvtx("h2d_pinned_uint8_gpu_normalize"):
            gpu_rgb.copy_(pinned_in,non_blocking=True)
            x.copy_(gpu_rgb.permute(2,0,1).unsqueeze(0).float()/255)
        with nvtx("model_forward"):infer()
        with nvtx("finite_check_pack_pinned_d2h"):
            if not torch.isfinite(y).all().item():raise ValueError("nonfinite")
            image=y[0].mul(255).round().clamp(0,255).byte().permute(1,2,0).contiguous()
            pinned_out.copy_(image,non_blocking=True);stream.synchronize()
        with nvtx("cpu_area_resize"):
            return cv2.resize(pinned_out.numpy(),(1920,1080),interpolation=cv2.INTER_AREA)

    def fused():
        with nvtx("cpu_copy_to_reused_pinned_input"):pinned_in.copy_(source)
        with nvtx("h2d_pinned_uint8_gpu_normalize"):
            gpu_rgb.copy_(pinned_in,non_blocking=True)
            x.copy_(gpu_rgb.permute(2,0,1).unsqueeze(0).float()/255)
        with nvtx("model_forward"):infer()
        with nvtx("finite_check"):
            if not torch.isfinite(y).all().item():raise ValueError("nonfinite")
        with nvtx("gpu_pack_area1080p"):
            pack_area_kernel[((fused_out.numel()+255)//256,)](
                y,fused_out,*y.stride()[1:],fused_out.numel(),256,enable_fp_fusion=False)
        with nvtx("d2h_pinned_1080p"):
            fused_host.copy_(fused_out,non_blocking=True);stream.synchronize()
        return fused_host.numpy()

    modes=["resident","cpu","pinned","fused"] if a.mode=="all" else [a.mode]
    funcs=dict(resident=resident,cpu=cpu,pinned=pinned,fused=fused)
    with torch.inference_mode():
        validation=[]
        if a.validation is not None:
            from PIL import Image
            runs=[r for r in json.loads((a.validation/"INPUTS.json").read_text())["runs"] if r["config"]=="720p-600k-420"]
            for run in runs:
                for sample in run["samples"]:
                    path=a.validation/sample["lr"]
                    assert hashlib.sha256(path.read_bytes()).hexdigest()==sample["lr_sha256"]
                    rgb=np.asarray(Image.open(path).convert("RGB")).copy();source=torch.from_numpy(rgb)
                    expected=cpu();actual=fused()
                    delta=np.abs(expected.astype(np.int16)-actual.astype(np.int16))
                    validation.append(dict(clip=run["clip"],frame=sample["frame"],lr_sha256=sample["lr_sha256"],
                        max_rgb=int(delta.max()),mean_abs_rgb=float(delta.mean()),different_values=int(np.count_nonzero(delta))))
            assert len(validation)==72
            # The pipeline change must preserve all output pixels exactly.
            assert max(r["max_rgb"] for r in validation)==0,validation
            sample=runs[0]["samples"][1]
            rgb=np.asarray(Image.open(a.validation/sample["lr"]).convert("RGB")).copy();source=torch.from_numpy(rgb)
            x.copy_(source.permute(2,0,1).unsqueeze(0).float()/255)
        for name in modes:
            for _ in range(50):funcs[name]()
        # This comparison establishes exact data preparation/postprocessing parity.
        if a.mode=="all":
            first=cpu();second=pinned();assert np.array_equal(first,second)
            assert np.array_equal(first,fused())
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        result=dict(host=platform.node(),gpu=gpu,pid=os.getpid(),uid=os.getuid(),torch=str(torch.__version__),trt=trt.__version__,
                    started=time.time(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    engine_sha256=hashlib.sha256(a.engine.read_bytes()).hexdigest(),command=os.sys.argv,
                    tensors={n:tensor_info(t) for n,t in dict(input=x,output=y,pinned_input=pinned_in,pinned_output=pinned_out,gpu_uint8=gpu_rgb).items()},
                    context_device_memory_bytes=engine.device_memory_size_v2,
                    gpu_memory_before=dict(zip(["free","total"],torch.cuda.mem_get_info())),
                    scope="resident excludes copies; cpu/pinned use CPU area resize; fused uses GPU round/clamp/area1080p and pinned output. All pipeline timings include CPU RGB input, transfers, model and finite check through CPU RGB1080p output. No network/decode/display.",timings={},
                    area_validation=validation,validation_input_manifest_sha256=hashlib.sha256((a.validation/"INPUTS.json").read_bytes()).hexdigest() if a.validation else None)
        if a.capture=="profiler":torch.cuda.cudart().cudaProfilerStart()
        for name in modes:
            times=[];begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            with nvtx("phase_"+name):
                for i in range(a.iterations):
                    with nvtx("frame_"+name):
                        t0=time.perf_counter();funcs[name]();times.append((time.perf_counter()-t0)*1000)
            result["timings"][name]=dict(p50_ms=float(np.percentile(times,50)),p95_ms=float(np.percentile(times,95)),samples_ms=times)
        torch.cuda.synchronize()
        if a.capture=="profiler":torch.cuda.cudart().cudaProfilerStop()
        result.update(finished=time.time(),torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_memory_after=dict(zip(["free","total"],torch.cuda.mem_get_info())),
                      profiler_capture_requested=a.capture=="profiler",cpu_pinned_exact_match=a.mode=="all")
        a.output.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps({k:{x:v[x] for x in ["p50_ms","p95_ms"]} for k,v in result["timings"].items()}),flush=True)


if __name__=="__main__":main()
