"""Raw-address launcher for an explicitly owned NVDEC mapping.

NVDEC mappings reject a pointer attribute queried by the generic Triton
launcher. The decoder validates the device address, pitch, context and lifetime.
"""

import triton
import triton.language as tl
from .cuda_kernels import normalize_nv12_709_full


@triton.jit
def normalize_address(
    address, Y, SH: tl.constexpr, SW: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr
):
    X = tl.cast(address, tl.pointer_type(tl.uint8))
    normalize_nv12_709_full(X, Y, SH, SW, N, BLOCK)
