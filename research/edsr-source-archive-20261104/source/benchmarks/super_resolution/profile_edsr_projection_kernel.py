"""Microbenchmark a fused 7x7 projection/head branch, not full-model latency."""
import argparse
from pathlib import Path
import time

import torch
from torch.nn import functional as F
import triton
import triton.language as tl

from benchmarks.super_resolution.profile_inference_optimization import digest,guard,save,stats


@triton.jit
def projection_head(X,RGB,W,HW,B,HB,Y,H:tl.constexpr,WIDTH:tl.constexpr,M:tl.constexpr):
    pixel=tl.program_id(0)*M+tl.arange(0,M)
    y=pixel//WIDTH;x=pixel%WIDTH
    channel=tl.arange(0,64);rgb_channel=tl.arange(0,16);output_channel=tl.arange(0,16)
    acc=tl.full((M,16),0,tl.float32);rgb_acc=tl.full((M,16),0,tl.float32)
    for rs in range(49):
        iy=y+rs//7-3;ix=x+rs%7-3
        valid=(pixel<H*WIDTH)&(iy>=0)&(iy<H)&(ix>=0)&(ix<WIDTH)
        a=tl.load(X+(iy[:,None]*WIDTH+ix[:,None])*64+channel[None,:],valid[:,None],0)
        b=tl.load(W+(rs*64+channel[:,None])*16+output_channel[None,:])
        acc=tl.dot(a,b,acc)
        rgb=tl.load(RGB+rgb_channel[None,:]*(H*WIDTH)+iy[:,None]*WIDTH+ix[:,None],
                    valid[:,None]&(rgb_channel[None,:]<3),0)
        hw=tl.load(HW+(rs*16+rgb_channel[:,None])*16+output_channel[None,:])
        rgb_acc=tl.dot(rgb,hw,rgb_acc)
    bias=tl.load(B+output_channel);head_bias=tl.load(HB+output_channel)
    # Preserve a separate half rounding for each original branch before add.
    a=(acc+bias[None,:]).to(tl.float16).to(tl.float32)
    b=(rgb_acc+head_bias[None,:]).to(tl.float16).to(tl.float32)
    value=(a+b).to(tl.float16)
    tl.store(Y+pixel[:,None]*16+output_channel[None,:],value,pixel[:,None]<H*WIDTH)


def weights(w):
    result=torch.zeros(49,64 if w.shape[1]==64 else 16,16,device=w.device,dtype=w.dtype)
    result[:,:w.shape[1],:12]=w.permute(2,3,1,0).reshape(49,w.shape[1],12)
    return result


@torch.inference_mode()
def check(config):
    rows=[]
    for h,w in ((2,3),(17,35)):
        for kind in ('random','constant','corner'):
            a=torch.randn(1,64,h,w,device='cuda',dtype=torch.float16)*.5
            rgb=torch.randn(1,3,h,w,device='cuda',dtype=torch.float16)*.5
            if kind=='constant':a.fill_(1);rgb.fill_(1)
            if kind=='corner':a.zero_();rgb.zero_();a[:,:,0,0]=1;rgb[:,:,-1,-1]=-1
            wa=torch.randn(12,64,7,7,device='cuda',dtype=torch.float16)*.02
            wh=torch.randn(12,3,7,7,device='cuda',dtype=torch.float16)*.02
            bias=torch.randn(16,device='cuda',dtype=torch.float32)*.1;hb=torch.randn_like(bias)*.1
            bias[12:]=0;hb[12:]=0
            ref=(F.conv2d(a.float(),wa.float(),bias[:12],padding=3).half()+
                 F.conv2d(rgb.float(),wh.float(),hb[:12],padding=3).half()).half()
            out=torch.empty(h,w,16,device='cuda',dtype=torch.float16)
            projection_head[(triton.cdiv(h*w,config['M']),)](a.permute(0,2,3,1).contiguous(),rgb,weights(wa),weights(wh),bias,hb,out,h,w,
                                                           **config,enable_fp_fusion=False)
            actual=out[:,:,:12].permute(2,0,1).unsqueeze(0)
            error=float((actual.float()-ref.float()).abs().max())
            torch.testing.assert_close(actual,ref,atol=.002,rtol=.002)
            assert torch.count_nonzero(out[:,:,12:]).item()==0
            rows.append(dict(shape=[h,w],kind=kind,max_abs_error=error,padded_channels_zero=True))
    return rows


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=dict(complete=False,scope=__doc__,started=time.time(),gpu=guard(),source_sha256=digest(Path(__file__)),
                triton=triton.__version__,records=[])
    save(a.output,result);torch.set_num_threads(1);torch.manual_seed(1729)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
    h,w=720,1280
    x=torch.randn(h,w,64,device='cuda',dtype=torch.float16);rgb=torch.randn(3,h,w,device='cuda',dtype=torch.float16)
    wa=torch.randn(49,64,16,device='cuda',dtype=torch.float16)*.02
    wh=torch.randn(49,16,16,device='cuda',dtype=torch.float16)*.02;wh[:,3:]=0
    bias=torch.randn(16,device='cuda',dtype=torch.float32);hb=torch.randn_like(bias)
    out=torch.empty(h,w,16,device='cuda',dtype=torch.float16)
    for m,warps,stages in ((32,4,2),(64,4,2),(64,4,3),(128,4,2),(128,4,3),(128,8,2),(256,8,2)):
        config=dict(M=m,num_warps=warps,num_stages=stages);row=dict(config=config)
        try:
            row['correctness']=check(config)
            def run():return projection_head[(triton.cdiv(h*w,m),)](x,rgb,wa,wh,bias,hb,out,h,w,**config,enable_fp_fusion=False)
            kernel=run();stream.synchronize()
            row.update(registers=kernel.n_regs,spills=kernel.n_spills,shared_bytes=kernel.metadata.shared)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):run()
            for _ in range(30):graph.replay()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);values=[]
            for _ in range(200):
                begin.record();graph.replay();end.record();end.synchronize();values.append(begin.elapsed_time(end))
            row['latency']=stats(values);print(config,'p95',row['latency']['p95_ms'],'registers',kernel.n_regs,'spills',kernel.n_spills,flush=True)
        except Exception as exc:row['error']=repr(exc);print(config,repr(exc),flush=True)
        result['records'].append(row);save(a.output,result)
    result.update(complete=True,finished=time.time(),gpu_end=guard());save(a.output,result)


if __name__=='__main__':main()
