"""INT8 3x3 convolution with FP16 residual and INT8 next-input dual outputs.

Kernel microbenchmark only, not an EDSR end-to-end or quality result. Eliminates
a separate residual-feature read for the following quantization. Integer
convolution, border handling, FP16 store and INT8 store are checked against
independent PyTorch operations before each full-resolution timing candidate.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
import triton
import triton.language as tl

from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save, stats


@triton.jit
def conv_dual(X, W, Skip, Scale, Bias, Factor, Out, Quant,
              H:tl.constexpr, WIDTH:tl.constexpr, M:tl.constexpr,
              DUAL:tl.constexpr=True, OUT_HWC:tl.constexpr=False, WRITE_Q:tl.constexpr=True):
    pixel = tl.program_id(0) * M + tl.arange(0, M)
    channel = tl.arange(0,64)
    output_channel = tl.arange(0,64)
    y = pixel // WIDTH; x = pixel % WIDTH
    acc = tl.full((M,64), 0, tl.int32)
    for rs in range(9):
        iy = y + rs // 3 - 1; ix = x + rs % 3 - 1
        address = channel[None,:] // 32 * (H*WIDTH*32) + (iy[:,None]*WIDTH+ix[:,None])*32 + channel[None,:]%32
        value = tl.load(X + address,
            (pixel[:,None]<H*WIDTH)&(iy[:,None]>=0)&(iy[:,None]<H)&(ix[:,None]>=0)&(ix[:,None]<WIDTH), 0)
        weight = tl.load(W + (rs*64+channel[:,None])*64 + output_channel[None,:])
        acc = tl.dot(value, weight, acc, out_dtype=tl.int32)
    offset = output_channel[None,:] // 32 * (H*WIDTH*32) + pixel[:,None]*32 + output_channel[None,:]%32
    alpha = tl.load(Scale + output_channel)
    bias = tl.load(Bias + output_channel)
    value = acc.to(tl.float32)*alpha[None,:] + bias[None,:]
    if DUAL:
        skip = tl.load(Skip + offset, pixel[:,None]<H*WIDTH, 0).to(tl.float32)
        result = (value + skip).to(tl.float16)
        out_offset = pixel[:,None]*64 + output_channel[None,:] if OUT_HWC else offset
        tl.store(Out + out_offset, result, pixel[:,None]<H*WIDTH)
        value = result.to(tl.float32)
    else:
        value = tl.maximum(value, 0)
    if WRITE_Q:
        factor = tl.load(Factor + output_channel)
        q = tl.extra.cuda.libdevice.nearbyint(value*factor[None,:])
        q = tl.minimum(tl.maximum(q,-128),127).to(tl.int8)
        tl.store(Quant + offset, q, pixel[:,None]<H*WIDTH)


def packed(x):
    n,c,h,w = x.shape
    assert c == 64
    return x.reshape(n,2,32,h,w).permute(0,1,3,4,2).contiguous()


def unpacked(x):
    n,_,h,w,_ = x.shape
    return x.permute(0,1,4,2,3).reshape(n,64,h,w)


@torch.inference_mode()
def check(config, seed=1729):
    torch.manual_seed(seed)
    records=[]
    for h,w in ((2,3),(17,35)):
        for kind in ("random","constant","corner"):
            x=torch.randint(-127,128,(1,64,h,w),device="cuda",dtype=torch.int8)
            if kind=="constant": x.fill_(1)
            if kind=="corner": x.zero_();x[:,:,0,0]=127;x[:,:,-1,-1]=-127
            weight=torch.randint(-127,128,(64,64,3,3),device="cuda",dtype=torch.int8)
            skip=torch.randn((1,64,h,w),device="cuda",dtype=torch.float16)*20
            alpha=torch.rand(64,device="cuda")*.001
            bias=torch.randn(64,device="cuda");factor=torch.rand(64,device="cuda")*.1
            ref=F.conv2d(x.float(),weight.float(),padding=1)*alpha[None,:,None,None]+bias[None,:,None,None]
            ref=(ref+skip.float()).half() if config.get("DUAL",True) else ref.relu()
            expected_q=(ref.float()*factor[None,:,None,None]).round().clamp(-128,127).to(torch.int8)
            xp=packed(x);sp=packed(skip);wp=weight.permute(2,3,1,0).contiguous()
            out=torch.empty_like(sp);q=torch.empty_like(xp)
            kernel=conv_dual[(triton.cdiv(h*w,config["M"]),)](xp,wp,sp,alpha,bias,factor,out,q,h,w,
                **config,enable_fp_fusion=False)
            torch.cuda.synchronize()
            if config.get("DUAL",True):
                assert torch.equal(out,packed(ref)),(h,w,kind,config,float((unpacked(out)-ref).abs().max()))
            assert torch.equal(q,packed(expected_q))
            records.append(dict(shape=[h,w],kind=kind,bit_exact=True))
    return records


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=["dual","relu"],default="dual")
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=dict(complete=False,started=time.time(),scope=__doc__,gpu=guard(),
        source_sha256=digest(Path(__file__)),triton=triton.__version__,mode=a.mode,records=[])
    save(a.output,result)
    torch.set_num_threads(1);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream);torch.manual_seed(1729)
    h,w=720,1280
    x=torch.randint(-127,128,(1,2,h,w,32),device="cuda",dtype=torch.int8)
    weight=torch.randint(-127,128,(9,64,64),device="cuda",dtype=torch.int8)
    skip=torch.randn((1,2,h,w,32),device="cuda",dtype=torch.float16)
    out=torch.empty_like(skip);q=torch.empty_like(x)
    alpha=torch.rand(64,device="cuda")*.001;bias=torch.randn(64,device="cuda");factor=torch.rand(64,device="cuda")*.1
    for m,warps,stages in ((64,4,2),(64,4,3),(128,4,2),(128,4,3),(128,8,2),(128,8,3),(256,8,2),(256,8,3)):
        config=dict(M=m,num_warps=warps,num_stages=stages,DUAL=a.mode=="dual")
        row=dict(config=config)
        try:
            row["correctness"]=check(config)
            def run():return conv_dual[(triton.cdiv(h*w,m),)](x,weight,skip,alpha,bias,factor,out,q,h,w,
                **config,enable_fp_fusion=False)
            kernel=run();stream.synchronize()
            row.update(registers=kernel.n_regs,spills=kernel.n_spills,shared_bytes=kernel.metadata.shared)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):run()
            for _ in range(50):graph.replay()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            values=[]
            for _ in range(200):
                begin.record();graph.replay();end.record();end.synchronize();values.append(begin.elapsed_time(end))
            row["latency"]=stats(values)
            print(config,"p95",row["latency"]["p95_ms"],"regs",kernel.n_regs,"spills",kernel.n_spills,flush=True)
        except Exception as exc:
            row["error"]=repr(exc);print(config,"failed",repr(exc),flush=True)
        result["records"].append(row);save(a.output,result)
    result.update(complete=True,finished=time.time(),gpu_end=guard());save(a.output,result)


if __name__=="__main__":main()
