"""Reuse each input row across three INT8 convolution taps.

This is a component experiment, not a full-model latency claim. Integer
convolution and both residual/quantized outputs must match independent PyTorch
operations bit for bit before each full-resolution timing configuration.
"""
import argparse
from pathlib import Path
import time

import torch
from torch.nn import functional as F
import triton
import triton.language as tl

from benchmarks.super_resolution.profile_edsr_int8_epilogue import conv_dual,packed
from benchmarks.super_resolution.profile_inference_optimization import digest,guard,save,stats


@triton.jit
def conv_window(X,W,Skip,Scale,Bias,Factor,Out,Quant,
                H:tl.constexpr,WIDTH:tl.constexpr,M:tl.constexpr,REUSE:tl.constexpr,
                WRITE_Q:tl.constexpr=True):
    x0=tl.program_id(0)*M;y=tl.program_id(1)
    x=x0+tl.arange(0,M);pixel=y*WIDTH+x
    channel=tl.arange(0,64);output_channel=tl.arange(0,64)
    acc=tl.full((M,64),0,tl.int32)
    for ky in range(3):
        iy=y+ky-1
        if REUSE:
            R:tl.constexpr=triton.next_power_of_2(M+2)
            px=x0-1+tl.arange(0,R)
            address=channel[None,:]//32*(H*WIDTH*32)+(iy*WIDTH+px[:,None])*32+channel[None,:]%32
            patch=tl.load(X+address,(tl.arange(0,R)[:,None]<M+2)&(px[:,None]>=0)&(px[:,None]<WIDTH)&(iy>=0)&(iy<H),0)
        for kx in tl.static_range(3):
            if REUSE:
                indices=tl.broadcast_to(tl.arange(0,M)[:,None]+kx,(M,64))
                value=tl.gather(patch,indices,axis=0)
            else:
                ix=x+kx-1
                address=channel[None,:]//32*(H*WIDTH*32)+(iy*WIDTH+ix[:,None])*32+channel[None,:]%32
                value=tl.load(X+address,(x[:,None]<WIDTH)&(ix[:,None]>=0)&(ix[:,None]<WIDTH)&(iy>=0)&(iy<H),0)
            weight=tl.load(W+((ky*3+kx)*64+channel[:,None])*64+output_channel[None,:])
            acc=tl.dot(value,weight,acc,out_dtype=tl.int32)
    offset=output_channel[None,:]//32*(H*WIDTH*32)+pixel[:,None]*32+output_channel[None,:]%32
    alpha=tl.load(Scale+output_channel);bias=tl.load(Bias+output_channel)
    skip=tl.load(Skip+offset,x[:,None]<WIDTH,0).to(tl.float32)
    result=(acc.to(tl.float32)*alpha[None,:]+bias[None,:]+skip).to(tl.float16)
    tl.store(Out+offset,result,x[:,None]<WIDTH)
    if WRITE_Q:
        factor=tl.load(Factor+output_channel)
        q=tl.extra.cuda.libdevice.nearbyint(result.to(tl.float32)*factor[None,:])
        q=tl.minimum(tl.maximum(q,-128),127).to(tl.int8)
        tl.store(Quant+offset,q,x[:,None]<WIDTH)


def launch(kind,config,args,h,w):
    if kind=='baseline':
        return conv_dual[(triton.cdiv(h*w,config['M']),)](*args,h,w,**config,enable_fp_fusion=False)
    return conv_window[(triton.cdiv(w,config['M']),h)](*args,h,w,REUSE=kind=='reuse',**config,enable_fp_fusion=False)


@torch.inference_mode()
def check(kind,config):
    rows=[];torch.manual_seed(1729)
    for h,w in ((2,3),(17,35),(5,1280)):
        for pattern in ('random','constant','corner'):
            x=torch.randint(-128,128,(1,64,h,w),device='cuda',dtype=torch.int8)
            if pattern=='constant':x.fill_(1)
            if pattern=='corner':x.zero_();x[:,:,0,0]=127;x[:,:,-1,-1]=-128
            weight=torch.randint(-127,128,(64,64,3,3),device='cuda',dtype=torch.int8)
            skip=torch.randn((1,64,h,w),device='cuda',dtype=torch.float16)*20
            alpha=torch.rand(64,device='cuda')*.001;bias=torch.randn(64,device='cuda');factor=torch.rand(64,device='cuda')*.1
            ref=(F.conv2d(x.float(),weight.float(),padding=1)*alpha[None,:,None,None]+bias[None,:,None,None]+skip.float()).half()
            expected_q=(ref.float()*factor[None,:,None,None]).round().clamp(-128,127).to(torch.int8)
            xp=packed(x);sp=packed(skip);wp=weight.permute(2,3,1,0).contiguous();out=torch.empty_like(sp);q=torch.empty_like(xp)
            launch(kind,config,(xp,wp,sp,alpha,bias,factor,out,q),h,w)
            torch.cuda.synchronize()
            assert torch.equal(out,packed(ref)),(kind,config,h,w,pattern,'residual')
            assert torch.equal(q,packed(expected_q)),(kind,config,h,w,pattern,'quantizer')
            rows.append(dict(shape=[h,w],pattern=pattern,bit_exact=True))
    return rows


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);dest=a.output/'RESULTS.json'
    if dest.exists():raise FileExistsError(dest)
    result=dict(complete=False,started=time.time(),gpu=guard(),scope=__doc__,source_sha256=digest(Path(__file__)),
        baseline_sha256=digest(Path(__file__).with_name('profile_edsr_int8_epilogue.py')),triton=triton.__version__,records=[])
    save(dest,result);torch.set_num_threads(1);torch.manual_seed(1729)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream);h,w=720,1280
    x=torch.randint(-128,128,(1,2,h,w,32),device='cuda',dtype=torch.int8)
    weight=torch.randint(-127,128,(9,64,64),device='cuda',dtype=torch.int8)
    skip=torch.randn((1,2,h,w,32),device='cuda',dtype=torch.float16);out=torch.empty_like(skip);q=torch.empty_like(x)
    alpha=torch.rand(64,device='cuda')*.001;bias=torch.randn(64,device='cuda');factor=torch.rand(64,device='cuda')*.1
    args=(x,weight,skip,alpha,bias,factor,out,q)
    configs=[('baseline',64,4,2),('row',64,4,2),('reuse',32,2,2),('reuse',64,2,2),('reuse',64,4,2),
             ('reuse',64,4,3),('reuse',128,4,2),('reuse',128,8,2),('baseline',64,4,2)]
    for index,(kind,m,warps,stages)in enumerate(configs):
        config=dict(M=m,num_warps=warps,num_stages=stages);row=dict(kind=kind,config=config)
        try:
            row['correctness']=check(kind,config)
            def run():return launch(kind,config,args,h,w)
            kernel=run();stream.synchronize();row.update(registers=kernel.n_regs,spills=kernel.n_spills,shared_bytes=kernel.metadata.shared)
            ptx=a.output/f'{index}-{kind}.ptx';ptx.write_text(kernel.asm['ptx'])
            row['kernel']=dict(name=kernel.metadata.name,ptx=ptx.name,ptx_sha256=digest(ptx),
                grid=[triton.cdiv(h*w,m),1,1]if kind=='baseline'else[triton.cdiv(w,m),h,1],
                block=[warps*32,1,1],shared_bytes=kernel.metadata.shared)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):run()
            for _ in range(50):graph.replay()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);values=[]
            for _ in range(300):
                begin.record();graph.replay();end.record();end.synchronize();values.append(begin.elapsed_time(end))
            row['latency']=stats(values)
            print(kind,config,'p95',row['latency']['p95_ms'],'regs',kernel.n_regs,'spills',kernel.n_spills,flush=True)
        except Exception as exc:row['error']=repr(exc);print(kind,config,repr(exc),flush=True)
        result['records'].append(row);save(dest,result)
    result.update(complete=True,finished=time.time(),gpu_end=guard());save(dest,result)


if __name__=='__main__':main()
