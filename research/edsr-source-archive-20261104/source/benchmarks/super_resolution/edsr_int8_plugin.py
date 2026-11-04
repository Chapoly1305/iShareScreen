"""Ahead-of-time TensorRT operator for an INT8 residual convolution.

The registered operator embeds one compiled Triton kernel in the engine. Its
six tensor inputs are quantized activations, RSIO weights, FP16 residuals,
per-output-channel convolution scales, biases, and next-input quantizer factors.
Both outputs use CHW32, matching the adjacent native INT8 convolutions.
"""
from functools import lru_cache
from typing import Tuple

import tensorrt as trt
import tensorrt.plugin as trtp
import triton
from triton.backends.compiler import AttrsDescriptor

from benchmarks.super_resolution.profile_edsr_int8_epilogue import conv_dual


@trtp.register("iss::edsr_dual")
def describe(x: trtp.TensorDesc, weight: trtp.TensorDesc, skip: trtp.TensorDesc,
             alpha: trtp.TensorDesc, bias: trtp.TensorDesc, factor: trtp.TensorDesc,
             height: int, width: int
             ) -> Tuple[trtp.TensorDesc, trtp.TensorDesc]:
    return skip.like(), x.like()


@trtp.autotune("iss::edsr_dual")
def autotune(x, weight, skip, alpha, bias, factor, height, width, outputs):
    return [trtp.AutoTuneCombination(
        "INT8,INT8,FP16,FP32,FP32,FP32,FP16,INT8",
        "CHW32,LINEAR,CHW32,LINEAR,LINEAR,LINEAR,CHW32,CHW32")]


@lru_cache(maxsize=8)
def compile_kernel(height=720, width=1280):
    source = triton.compiler.ASTSource(fn=conv_dual,
        signature={"X":"*i8", "W":"*i8", "Skip":"*fp16", "Scale":"*fp32",
                   "Bias":"*fp32", "Factor":"*fp32", "Out":"*fp16", "Quant":"*i8"},
        constants={"H":height, "WIDTH":width, "M":64, "DUAL":True, "OUT_HWC":False, "WRITE_Q":True},
        # All fixed-shape tensors/weights are TensorRT-managed aligned buffers.
        # Match the pointer divisibility specialization used by the measured
        # Triton JIT kernel; otherwise AOT emits scalar byte loads and spills.
        attrs=AttrsDescriptor.from_hints({i:16 for i in range(8)}))
    return triton.compile(source, options=dict(num_warps=4, num_stages=2, enable_fp_fusion=False))


@trtp.aot_impl("iss::edsr_dual")
def ahead_of_time(x, weight, skip, alpha, bias, factor, height, width, outputs, tactic
                  ) -> Tuple[str, str, trtp.KernelLaunchParams, trtp.SymIntExprs]:
    kernel = compile_kernel(height, width)
    launch = trtp.KernelLaunchParams()
    launch.grid_x = trtp.cdiv(x.shape_expr[2] * x.shape_expr[3], 64)
    launch.block_x = kernel.metadata.num_warps * 32
    launch.shared_mem = kernel.metadata.shared
    return kernel.metadata.name, kernel.asm["ptx"], launch, trtp.SymIntExprs(0)
