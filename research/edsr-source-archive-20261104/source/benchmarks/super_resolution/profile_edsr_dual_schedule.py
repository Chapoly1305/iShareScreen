"""Expand scheduling tests for the accepted bit-checked INT8 residual kernel."""
import argparse
from pathlib import Path
import time
import torch
import triton

from benchmarks.super_resolution.profile_edsr_int8_epilogue import conv_dual,check
from benchmarks.super_resolution.profile_inference_optimization import digest,guard,save,stats


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=dict(complete=False,started=time.time(),gpu=guard(),scope=__doc__,source_sha256=digest(Path(__file__)),
        kernel_source_sha256=digest(Path(__file__).with_name('profile_edsr_int8_epilogue.py')),triton=triton.__version__,records=[])
    save(a.output,r);torch.set_num_threads(1);torch.manual_seed(1729)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream);h,w=720,1280
    x=torch.randint(-127,128,(1,2,h,w,32),device='cuda',dtype=torch.int8)
    weight=torch.randint(-127,128,(9,64,64),device='cuda',dtype=torch.int8)
    skip=torch.randn((1,2,h,w,32),device='cuda',dtype=torch.float16)
    out=torch.empty_like(skip);q=torch.empty_like(x)
    scale=torch.rand(64,device='cuda')*.001;bias=torch.randn(64,device='cuda');factor=torch.rand(64,device='cuda')*.1
    configs=((64,4,2),(16,2,2),(16,4,2),(32,2,2),(32,4,2),(32,4,3),(64,2,2),(64,2,3),(64,4,4),(64,4,2))
    for m,warps,stages in configs:
        config=dict(M=m,num_warps=warps,num_stages=stages);row=dict(config=config)
        try:
            row['correctness']=check(config)
            def run():return conv_dual[(triton.cdiv(h*w,m),)](x,weight,skip,scale,bias,factor,out,q,h,w,**config,enable_fp_fusion=False)
            kernel=run();stream.synchronize();row.update(registers=kernel.n_regs,spills=kernel.n_spills,shared_bytes=kernel.metadata.shared)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):run()
            for _ in range(50):graph.replay()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);values=[]
            for _ in range(300):
                begin.record();graph.replay();end.record();end.synchronize();values.append(begin.elapsed_time(end))
            row['latency']=stats(values);print(config,'p95',row['latency']['p95_ms'],'regs',kernel.n_regs,'spills',kernel.n_spills,flush=True)
        except Exception as exc:row['error']=repr(exc);print(config,'failed',repr(exc),flush=True)
        r['records'].append(row);save(a.output,r)
    r.update(complete=True,finished=time.time(),gpu_end=guard());save(a.output,r)


if __name__=='__main__':main()
