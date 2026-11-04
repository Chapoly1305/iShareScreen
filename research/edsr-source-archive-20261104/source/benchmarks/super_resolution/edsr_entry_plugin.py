"""AOT fused head quantization and FP16 residual layout conversion."""
from functools import lru_cache
from typing import Tuple

import tensorrt as trt
import tensorrt.plugin as trtp
import triton
import triton.language as tl
from triton.backends.compiler import AttrsDescriptor


@triton.jit
def entry(X, Factor, Skip, Quant, N:tl.constexpr, PIXELS:tl.constexpr, BLOCK:tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0,BLOCK)
    channel = pos // (PIXELS*32) * 32 + pos % 32
    pixel = pos // 32 % PIXELS
    value = tl.load(X + pixel*64+channel,pos<N,0)
    factor = tl.load(Factor+channel,pos<N,0)
    q = tl.extra.cuda.libdevice.nearbyint(value.to(tl.float32)*factor)
    q = tl.minimum(tl.maximum(q,-128),127).to(tl.int8)
    tl.store(Skip+pos,value,pos<N)
    tl.store(Quant+pos,q,pos<N)


@trtp.register("iss::edsr_entry")
def describe(x:trtp.TensorDesc,factor:trtp.TensorDesc,height:int,width:int) -> Tuple[trtp.TensorDesc,trtp.TensorDesc]:
    q=x.like();q.dtype=trt.int8
    return x.like(),q


@trtp.autotune("iss::edsr_entry")
def autotune(x,factor,height,width,outputs):
    return [trtp.AutoTuneCombination("FP16,FP32,FP16,INT8","HWC8,LINEAR,CHW32,CHW32")]


@lru_cache(maxsize=8)
def compile_kernel(height=720,width=1280):
    source=triton.compiler.ASTSource(fn=entry,
        signature={"X":"*fp16","Factor":"*fp32","Skip":"*fp16","Quant":"*i8"},
        constants={"N":height*width*64,"PIXELS":height*width,"BLOCK":1024},
        attrs=AttrsDescriptor.from_hints({i:16 for i in range(4)}))
    return triton.compile(source,options=dict(num_warps=4,num_stages=2,enable_fp_fusion=False))


@trtp.aot_impl("iss::edsr_entry")
def ahead_of_time(x,factor,height,width,outputs,tactic) -> Tuple[str,str,trtp.KernelLaunchParams,trtp.SymIntExprs]:
    kernel=compile_kernel(height,width);launch=trtp.KernelLaunchParams()
    launch.grid_x=trtp.cdiv(x.shape_expr.numel(),1024)
    launch.block_x=kernel.metadata.num_warps*32;launch.shared_mem=kernel.metadata.shared
    return kernel.metadata.name,kernel.asm["ptx"],launch,trtp.SymIntExprs(0)
