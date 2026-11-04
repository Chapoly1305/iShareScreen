"""Screen cuDNN convolution algorithms directly, outside framework heuristics.

This is an isolated 64->64, 3x3, 720p operator experiment, not model latency.
No model acceptance can follow from this microbenchmark alone.
"""
import argparse
import ctypes as c
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save, stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--samples",type=int,default=100)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(1);torch.manual_seed(1729)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    result=dict(complete=False,started=time.time(),gpu=guard(),pid=os.getpid(),command=os.sys.argv,
        script_sha256=digest(Path(__file__)),cudnn=torch.backends.cudnn.version(),
        scope="Isolated FP16 1x64x720x1280 convolution, 64x64x3x3 filter, padding1 stride1. Random data and weights. 16 repeated calls per captured graph. No bias/activation/residual, no EDSR quality or full-model speed claim.",rows=[])
    save(a.output,result)
    libpath=Path(torch.__file__).resolve().parent.parent/"nvidia/cudnn/lib/libcudnn.so.9"
    lib=c.CDLL(str(libpath));vp=c.c_void_p;ci=c.c_int;sz=c.c_size_t
    signatures={
        "cudnnCreate":[c.POINTER(vp)],"cudnnSetStream":[vp,vp],
        "cudnnCreateTensorDescriptor":[c.POINTER(vp)],"cudnnSetTensor4dDescriptor":[vp,ci,ci,ci,ci,ci,ci],
        "cudnnCreateFilterDescriptor":[c.POINTER(vp)],"cudnnSetFilter4dDescriptor":[vp,ci,ci,ci,ci,ci,ci],
        "cudnnCreateConvolutionDescriptor":[c.POINTER(vp)],"cudnnSetConvolution2dDescriptor":[vp,ci,ci,ci,ci,ci,ci,ci,ci],
        "cudnnSetConvolutionMathType":[vp,ci],
        "cudnnGetConvolutionForwardWorkspaceSize":[vp,vp,vp,vp,vp,ci,c.POINTER(sz)],
        "cudnnConvolutionForward":[vp,vp,vp,vp,vp,vp,vp,ci,vp,sz,vp,vp,vp],
        "cudnnDestroyTensorDescriptor":[vp],"cudnnDestroyFilterDescriptor":[vp],"cudnnDestroyConvolutionDescriptor":[vp],"cudnnDestroy":[vp]}
    for name,args in signatures.items():
        getattr(lib,name).argtypes=args;getattr(lib,name).restype=ci
    lib.cudnnGetErrorString.argtypes=[ci];lib.cudnnGetErrorString.restype=c.c_char_p

    def check(status):
        if status:raise RuntimeError(lib.cudnnGetErrorString(status).decode())

    stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
    handle=vp();check(lib.cudnnCreate(c.byref(handle)));check(lib.cudnnSetStream(handle,stream.cuda_stream))
    alpha,beta=c.c_float(1),c.c_float(0)
    names=["implicit_gemm","implicit_precomp_gemm","gemm","direct","fft","fft_tiling","winograd","winograd_nonfused"]
    with torch.inference_mode():
        seed_x=torch.randn((1,64,720,1280),device="cuda",dtype=torch.float16)
        seed_w=torch.randn((64,64,3,3),device="cuda",dtype=torch.float16)*.02
        expected=F.conv2d(seed_x.float(),seed_w.float(),padding=1)
        for layout,fmt in [("nchw",0),("nhwc",1)]:
            memory_format=torch.channels_last if fmt else torch.contiguous_format
            x=seed_x.contiguous(memory_format=memory_format);w=seed_w.contiguous(memory_format=memory_format)
            y=torch.empty_like(x);xd,wd,yd,cd=vp(),vp(),vp(),vp()
            check(lib.cudnnCreateTensorDescriptor(c.byref(xd)));check(lib.cudnnCreateTensorDescriptor(c.byref(yd)))
            check(lib.cudnnCreateFilterDescriptor(c.byref(wd)));check(lib.cudnnCreateConvolutionDescriptor(c.byref(cd)))
            check(lib.cudnnSetTensor4dDescriptor(xd,fmt,2,1,64,720,1280));check(lib.cudnnSetTensor4dDescriptor(yd,fmt,2,1,64,720,1280))
            check(lib.cudnnSetFilter4dDescriptor(wd,2,fmt,64,64,3,3))
            check(lib.cudnnSetConvolution2dDescriptor(cd,1,1,1,1,1,1,1,0))
            check(lib.cudnnSetConvolutionMathType(cd,1))
            for algorithm,name in enumerate(names):
                row=dict(layout=layout,algorithm=name,index=algorithm)
                workspace_size=sz()
                status=lib.cudnnGetConvolutionForwardWorkspaceSize(handle,xd,wd,cd,yd,algorithm,c.byref(workspace_size))
                if status:
                    row["unsupported"]=lib.cudnnGetErrorString(status).decode();result["rows"].append(row);continue
                row["workspace_bytes"]=workspace_size.value
                if workspace_size.value>4<<30:
                    row["skipped"]="workspace exceeds 4 GiB";result["rows"].append(row);continue
                workspace=torch.empty(max(workspace_size.value,1),device="cuda",dtype=torch.uint8)

                def launch():
                    return lib.cudnnConvolutionForward(handle,c.byref(alpha),xd,x.data_ptr(),wd,w.data_ptr(),cd,algorithm,
                        workspace.data_ptr(),workspace_size.value,c.byref(beta),yd,y.data_ptr())

                status=launch()
                if status:
                    row["unsupported"]=lib.cudnnGetErrorString(status).decode();result["rows"].append(row);continue
                stream.synchronize()
                delta=(y.float()-expected).abs();row.update(max_abs_error=float(delta.max()),mean_abs_error=float(delta.mean()))
                if not torch.isfinite(y).all().item():raise ValueError(row)
                for _ in range(20):check(launch())
                stream.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    for _ in range(16):check(launch())
                for _ in range(5):graph.replay()
                stream.synchronize();begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                samples=[]
                for _ in range(a.samples):
                    begin.record();graph.replay();end.record();end.synchronize();samples.append(begin.elapsed_time(end)/16)
                row["latency"]=stats(samples);result["rows"].append(row);save(a.output,result)
                print(layout,name,"p95",row["latency"]["p95_ms"],"max_error",row["max_abs_error"],flush=True)
                del graph,workspace,delta
            check(lib.cudnnDestroyTensorDescriptor(xd));check(lib.cudnnDestroyTensorDescriptor(yd))
            check(lib.cudnnDestroyFilterDescriptor(wd));check(lib.cudnnDestroyConvolutionDescriptor(cd))
        check(lib.cudnnDestroy(handle));result.update(complete=True,finished=time.time(),gpu_end=guard());save(a.output,result)


if __name__=="__main__":main()
