"""Screen the custom fused Winograd kernel before any whole-model experiment."""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from benchmarks.super_resolution.edsr_winograd import transform_weights,winograd
from benchmarks.super_resolution.profile_inference_optimization import digest,guard,save,stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(1);torch.manual_seed(1729)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    result=dict(complete=False,started=time.time(),gpu=guard(),script_sha256=digest(Path(__file__)),
        kernel_sha256=digest(Path(__file__).with_name("edsr_winograd.py")),rows=[],correctness=[],
        scope="Isolated fused 64->64 convolution. Random weights/data; no model quality or full inference-speed claim.")
    save(a.output,result)
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
    w=torch.randn((64,64,3,3),dtype=torch.float32)*.02
    u=transform_weights(w).to(device="cuda",dtype=torch.float16)
    bias=torch.randn(64,device="cuda")*.01
    with torch.inference_mode():
        for h,width in [(4,6),(10,14),(32,64)]:
            for kind in ["random","corner_impulse"]:
                x=torch.randn((1,64,h,width),device="cuda",dtype=torch.float16).contiguous(memory_format=torch.channels_last)
                if kind=="corner_impulse":x.zero_();x[:,:,0,0]=1;x[:,:,-1,-1]=-1
                skip=torch.randn_like(x);out=torch.empty_like(x)
                reference=F.conv2d(x.float(),w.cuda(),bias,padding=1)
                for relu,residual in [(True,False),(False,True)]:
                    expected=reference.relu() if relu else reference+skip.float()
                    winograd(x,u,bias,out,skip if residual else None,relu=relu)
                    delta=(out.float()-expected).abs()
                    row=dict(shape=[h,width],kind=kind,relu=relu,residual=residual,max_abs=float(delta.max()),mean_abs=float(delta.mean()))
                    result["correctness"].append(row)
                    assert torch.isfinite(out).all().item() and row["max_abs"]<.01,row
        save(a.output,result)
        x=torch.randn((1,64,720,1280),device="cuda",dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        out=torch.empty_like(x)
        reference=F.conv2d(x.float(),w.cuda(),bias,padding=1).relu()
        for block_m,warps,unroll in [(16,4,1),(32,4,1),(32,8,1),(64,4,1),(64,8,1),(32,4,4),(64,8,4)]:
            def fn():return winograd(x,u,bias,out,relu=True,block_m=block_m,warps=warps,unroll=unroll)
            t0=time.time();kernel=fn();stream.synchronize()
            delta=(out.float()-reference).abs()
            row=dict(block_m=block_m,warps=warps,unroll=unroll,compile_s=time.time()-t0,n_regs=kernel.n_regs,n_spills=kernel.n_spills,
                shared_bytes=kernel.metadata.shared,max_abs_error=float(delta.max()),mean_abs_error=float(delta.mean()))
            assert torch.isfinite(out).all().item() and row["max_abs_error"]<.01,row
            for _ in range(20):fn()
            stream.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                for _ in range(16):fn()
            for _ in range(5):graph.replay()
            stream.synchronize();begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            values=[]
            for _ in range(100):
                begin.record();graph.replay();end.record();end.synchronize();values.append(begin.elapsed_time(end)/16)
            row["latency"]=stats(values);result["rows"].append(row);save(a.output,result)
            print(json.dumps({k:v for k,v in row.items() if k!="latency"}),"p95",row["latency"]["p95_ms"],flush=True)
            del graph,delta
        result.update(complete=True,finished=time.time(),gpu_end=guard());save(a.output,result)


if __name__=="__main__":main()
