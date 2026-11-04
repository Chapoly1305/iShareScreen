"""AOT final INT8 residual convolution with one FP16 output and one rounding."""
from functools import lru_cache
from typing import Tuple

import tensorrt.plugin as trtp
import triton
from triton.backends.compiler import AttrsDescriptor
import triton.language as tl

from benchmarks.super_resolution.profile_edsr_int8_epilogue import conv_dual


@triton.jit
def final_kernel(X,W,Skip,Alpha,Bias,Out,H:tl.constexpr,WIDTH:tl.constexpr,M:tl.constexpr):
    conv_dual(X,W,Skip,Alpha,Bias,Alpha,Out,Out,H,WIDTH,M,DUAL=True,OUT_HWC=False,WRITE_Q=False)


@trtp.register("iss::edsr_final")
def describe(x:trtp.TensorDesc,weight:trtp.TensorDesc,skip:trtp.TensorDesc,
             alpha:trtp.TensorDesc,bias:trtp.TensorDesc,height:int,width:int) -> trtp.TensorDesc:
    return skip.like()


@trtp.autotune("iss::edsr_final")
def autotune(x,weight,skip,alpha,bias,height,width,outputs):
    return [trtp.AutoTuneCombination("INT8,INT8,FP16,FP32,FP32,FP16","CHW32,LINEAR,CHW32,LINEAR,LINEAR,CHW32")]


@lru_cache(maxsize=8)
def compile_kernel(height=720,width=1280):
    source=triton.compiler.ASTSource(fn=final_kernel,
        signature={"X":"*i8","W":"*i8","Skip":"*fp16","Alpha":"*fp32","Bias":"*fp32","Out":"*fp16"},
        constants={"H":height,"WIDTH":width,"M":64},attrs=AttrsDescriptor.from_hints({i:16 for i in range(6)}))
    return triton.compile(source,options=dict(num_warps=4,num_stages=2,enable_fp_fusion=False))


@trtp.aot_impl("iss::edsr_final")
def ahead_of_time(x,weight,skip,alpha,bias,height,width,outputs,tactic) -> Tuple[str,str,trtp.KernelLaunchParams,trtp.SymIntExprs]:
    kernel=compile_kernel(height,width);launch=trtp.KernelLaunchParams()
    launch.grid_x=trtp.cdiv(x.shape_expr[2]*x.shape_expr[3],64)
    launch.block_x=kernel.metadata.num_warps*32;launch.shared_mem=kernel.metadata.shared
    return kernel.metadata.name,kernel.asm["ptx"],launch,trtp.SymIntExprs(0)
