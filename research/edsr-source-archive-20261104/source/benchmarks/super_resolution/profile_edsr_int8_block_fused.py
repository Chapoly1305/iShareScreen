"""Spatially fuse both INT8 convolutions of one EDSR residual block.

The quantized intermediate remains inside one GPU block. Halo activations
outside the image are explicitly zeroed before conv2, including its bias.
This is a kernel microbenchmark; whole-model accuracy/performance are separate.
"""
import argparse
from pathlib import Path
import time

import torch
from torch.nn import functional as F
import triton
import triton.language as tl

from benchmarks.super_resolution.profile_edsr_int8_epilogue import packed
from benchmarks.super_resolution.profile_inference_optimization import digest,guard,save,stats


@triton.jit
def block_kernel(X,W1,W2,Skip,A1,B1,MidFactor,A2,B2,NextFactor,Out,Quant,
                 H:tl.constexpr,WIDTH:tl.constexpr,TY:tl.constexpr,TX:tl.constexpr,
                 MH:tl.constexpr,MO:tl.constexpr):
    tile=tl.program_id(0)
    base_y=tile//tl.cdiv(WIDTH,TX)*TY
    base_x=tile%tl.cdiv(WIDTH,TX)*TX
    m=tl.arange(0,MH);c=tl.arange(0,64);o=tl.arange(0,64)
    hy=base_y-1+m//(TX+2);hx=base_x-1+m%(TX+2)
    mid_valid=(m<(TY+2)*(TX+2))&(hy>=0)&(hy<H)&(hx>=0)&(hx<WIDTH)
    acc1=tl.full((MH,64),0,tl.int32)
    for rs in range(9):
        iy=hy+rs//3-1;ix=hx+rs%3-1
        address=c[None,:]//32*(H*WIDTH*32)+(iy[:,None]*WIDTH+ix[:,None])*32+c[None,:]%32
        v=tl.load(X+address,mid_valid[:,None]&(iy[:,None]>=0)&(iy[:,None]<H)&(ix[:,None]>=0)&(ix[:,None]<WIDTH),0)
        weight=tl.load(W1+(rs*64+c[:,None])*64+o[None,:])
        acc1=tl.dot(v,weight,acc1,out_dtype=tl.int32)
    a1=tl.load(A1+o);b1=tl.load(B1+o);factor=tl.load(MidFactor+o)
    mid=tl.maximum(acc1.to(tl.float32)*a1[None,:]+b1[None,:],0)*factor[None,:]
    mid=tl.minimum(tl.extra.cuda.libdevice.nearbyint(mid),127).to(tl.int8)
    mid=tl.where(mid_valid[:,None],mid,0).to(tl.int8)

    output_pixel=tl.arange(0,MO)
    oy=base_y+output_pixel//TX;ox=base_x+output_pixel%TX
    valid=(output_pixel<TY*TX)&(oy<H)&(ox<WIDTH)
    acc2=tl.full((MO,64),0,tl.int32)
    for rs in range(9):
        row=(output_pixel//TX+rs//3)*(TX+2)+output_pixel%TX+rs%3
        index=tl.broadcast_to(row[:,None],(MO,64))
        v=tl.gather(mid,index,axis=0)
        weight=tl.load(W2+(rs*64+c[:,None])*64+o[None,:])
        acc2=tl.dot(v,weight,acc2,out_dtype=tl.int32)
    offset=o[None,:]//32*(H*WIDTH*32)+(oy[:,None]*WIDTH+ox[:,None])*32+o[None,:]%32
    skip=tl.load(Skip+offset,valid[:,None],0).to(tl.float32)
    a2=tl.load(A2+o);b2=tl.load(B2+o)
    out=(acc2.to(tl.float32)*a2[None,:]+b2[None,:]+skip).to(tl.float16)
    factor=tl.load(NextFactor+o)
    q=tl.extra.cuda.libdevice.nearbyint(out.to(tl.float32)*factor[None,:])
    q=tl.minimum(tl.maximum(q,-128),127).to(tl.int8)
    tl.store(Out+offset,out,valid[:,None]);tl.store(Quant+offset,q,valid[:,None])


def launch(args,h,w,config):
    return block_kernel[(triton.cdiv(h,config["TY"])*triton.cdiv(w,config["TX"]),)](
        *args,h,w,**config,enable_fp_fusion=False)


@torch.inference_mode()
def inputs(h,w,kind="random"):
    x=torch.randint(-127,128,(1,64,h,w),device="cuda",dtype=torch.int8)
    if kind=="constant":x.fill_(1)
    if kind=="corner":x.zero_();x[:,:,0,0]=127;x[:,:,-1,-1]=-127
    weights=[torch.randint(-127,128,(64,64,3,3),device="cuda",dtype=torch.int8) for _ in range(2)]
    skip=torch.randn((1,64,h,w),device="cuda",dtype=torch.float16)*20
    a1,a2=[torch.rand(64,device="cuda")*.001 for _ in range(2)]
    b1,b2=[torch.randn(64,device="cuda") for _ in range(2)]
    midfactor,nextfactor=[torch.rand(64,device="cuda")*.1 for _ in range(2)]
    out=torch.empty_like(packed(skip));q=torch.empty_like(packed(x))
    args=[packed(x),*[w.permute(2,3,1,0).contiguous() for w in weights],packed(skip),a1,b1,midfactor,a2,b2,nextfactor,out,q]
    return args,(x,weights,skip)


@torch.inference_mode()
def check(config):
    torch.manual_seed(1729);records=[]
    for h,w in ((2,3),(17,35)):
        for kind in ("random","constant","corner"):
            args,(x,weights,skip)=inputs(h,w,kind)
            a1,b1,mf,a2,b2,nf=args[4:10]
            mid=(F.conv2d(x.float(),weights[0].float(),padding=1)*a1[None,:,None,None]+b1[None,:,None,None]).relu()
            mid=(mid*mf[None,:,None,None]).round().clamp(0,127)
            out=(F.conv2d(mid,weights[1].float(),padding=1)*a2[None,:,None,None]+b2[None,:,None,None]+skip.float()).half()
            q=(out.float()*nf[None,:,None,None]).round().clamp(-128,127).to(torch.int8)
            launch(args,h,w,config);torch.cuda.synchronize()
            assert torch.equal(args[-2],packed(out)),(kind,h,w,float((args[-2]-packed(out)).abs().max()))
            assert torch.equal(args[-1],packed(q))
            records.append(dict(shape=[h,w],kind=kind,bit_exact=True))
    return records


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=dict(complete=False,started=time.time(),gpu=guard(),scope=__doc__,source_sha256=digest(Path(__file__)),
                triton=triton.__version__,torch=str(torch.__version__),records=[])
    save(a.output,result)
    torch.set_num_threads(1);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream);torch.manual_seed(1729)
    args,_=inputs(720,1280)
    for ty,tx in ((4,8),(8,8),(4,16),(8,16)):
        for warps in (4,8):
            config=dict(TY=ty,TX=tx,MH=triton.next_power_of_2((ty+2)*(tx+2)),MO=ty*tx,num_warps=warps,num_stages=2)
            row=dict(config=config)
            try:
                row["correctness"]=check(config)
                def run():return launch(args,720,1280,config)
                kernel=run();stream.synchronize()
                row.update(registers=kernel.n_regs,spills=kernel.n_spills,shared_bytes=kernel.metadata.shared)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):run()
                for _ in range(50):graph.replay()
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);times=[]
                for _ in range(200):
                    begin.record();graph.replay();end.record();end.synchronize();times.append(begin.elapsed_time(end))
                row["latency"]=stats(times);print(config,"p95",row["latency"]["p95_ms"],"regs",kernel.n_regs,"spills",kernel.n_spills,flush=True)
            except Exception as exc:
                row["error"]=repr(exc);print(config,"failed",repr(exc),flush=True)
            result["records"].append(row);save(a.output,result)
    result.update(complete=True,finished=time.time(),gpu_end=guard());save(a.output,result)


if __name__=="__main__":main()
