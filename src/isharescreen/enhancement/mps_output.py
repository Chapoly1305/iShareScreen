"""Exact FP16 CHW to uint8 RGB packing on the existing PyTorch MPS stream."""
from __future__ import annotations


_SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void pack_rgb(
    device const half *input [[buffer(0)]],
    device uchar *rgb [[buffer(1)]],
    device atomic_uint *invalid [[buffer(2)]],
    constant uint &pixels [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= pixels * 3) return;
    float value = float(input[(i % 3) * pixels + i / 3]);
    if (!isfinite(value)) {
        atomic_store_explicit(invalid, 1u, memory_order_relaxed);
        value = 0.0f;
    }
    // Preserve FP32 scale, round-to-even and saturation before byte conversion.
    rgb[i] = uchar(clamp(rint(value * 255.0f), 0.0f, 255.0f));
}
"""


class MPSRGBPacker:
    """Fuse quantization, channel interleaving and full nonfinite detection.

    Compilation is lazy, on the model's worker thread. The public compiler is
    preferred; PyTorch 2.6 exposes the same facility as ``_compile_shader``.
    The returned MPS tensor owns its storage independently of later calls.
    """

    def __init__(self):
        import torch
        compiler = (getattr(torch.mps, "compile_shader", None)
                    or getattr(torch.mps, "_compile_shader", None))
        if compiler is None:
            raise RuntimeError("This PyTorch build has no MPS shader compiler")
        self._library = compiler(_SOURCE)
        self._kernel = self._library.pack_rgb

    def __call__(self, output):
        import torch
        if (output.device.type != "mps" or output.dtype != torch.float16
                or output.ndim != 4 or output.shape[:2] != (1,3)
                or not output.is_contiguous() or not output.numel()):
            raise ValueError("MPS RGB packing requires contiguous 1×3×H×W FP16 output")
        height, width = output.shape[2:]
        rgb = torch.empty((height,width,3), dtype=torch.uint8, device=output.device)
        invalid = torch.zeros(1, dtype=torch.int32, device=output.device)
        self._kernel(output, rgb, invalid, height*width, threads=rgb.numel())
        if invalid.item():
            raise ValueError("Candidate model returned invalid pixels")
        return rgb
