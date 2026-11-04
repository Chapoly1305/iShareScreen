"""CUTLASS INT8 convolution + FP16 skip + INT8 quantization microbenchmark.

All timed operands already reside on GPU. This is not an end-to-end EDSR
measurement. The C ABI performs no allocation or tensor copying on replay.
"""
import argparse
import ctypes as c
from pathlib import Path
import time

import torch
from torch.nn import functional as F
import triton

from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save, stats
from benchmarks.super_resolution.profile_edsr_int8_epilogue import conv_dual, packed


class Library:
    def __init__(self, path):
        self.lib = c.CDLL(str(path.resolve()))
        self.lib.iss_cutlass_create.argtypes = [c.POINTER(c.c_void_p)] + [c.c_int] * 3 + [c.c_void_p] * 8
        self.lib.iss_cutlass_create.restype = c.c_int
        self.lib.iss_cutlass_run.argtypes = [c.c_void_p, c.c_void_p]
        self.lib.iss_cutlass_run.restype = c.c_int
        self.lib.iss_cutlass_destroy.argtypes = [c.c_void_p]
        self.lib.iss_cutlass_destroy.restype = None
        self.lib.iss_cutlass_attributes.argtypes = [c.c_void_p, c.POINTER(c.c_int), c.POINTER(c.c_int)]
        self.lib.iss_cutlass_attributes.restype = c.c_int

    def create(self, config, h, w, tensors):
        handle = c.c_void_p()
        code = self.lib.iss_cutlass_create(c.byref(handle), config, h, w,
                                          *(t.data_ptr() for t in tensors))
        if code:
            raise RuntimeError(f"CUTLASS initialization status {code}")
        return handle

    def run(self, handle, stream):
        code = self.lib.iss_cutlass_run(handle, stream.cuda_stream)
        if code:
            raise RuntimeError(f"CUTLASS launch status {code}")

    def attributes(self, handle):
        regs, shared = c.c_int(), c.c_int()
        code = self.lib.iss_cutlass_attributes(handle, c.byref(regs), c.byref(shared))
        if code:
            raise RuntimeError(f"CUDA function attribute status {code}")
        return dict(registers=regs.value, shared_bytes=shared.value)


def filter_packed(weight):
    # TensorCxRSKx<32>: [input channel group, R, S, output channel, input lane].
    # CUTLASS interleaved integer MMA also permutes output channels offline
    # (cutlass/util/host_reorder.h::reorder_column<32>) so accumulator fragments
    # can be stored contiguously without an epilogue shared-memory transpose.
    destination = torch.tensor([n // 32 * 32 + (n % 8 // 2) * 8 +
                                (n % 32 // 8) * 2 + n % 2 for n in range(64)],
                               device=weight.device)
    reordered = torch.empty_like(weight)
    reordered[destination] = weight
    return reordered.reshape(64, 2, 32, 3, 3).permute(1, 3, 4, 0, 2).contiguous()


@torch.inference_mode()
def check(library, config, stream):
    torch.manual_seed(1729)
    records = []
    for h, w in ((1, 1), (2, 3), (17, 35), (5, 1280)):
        for kind in ("random", "constant", "corner"):
            x = torch.randint(-128, 128, (1, 64, h, w), device="cuda", dtype=torch.int8)
            if kind == "constant":
                x.fill_(1)
            elif kind == "corner":
                x.zero_()
                x[:, :, 0, 0] = 127
                x[:, :, -1, -1] = -128
            weight = torch.randint(-128, 128, (64, 64, 3, 3), device="cuda", dtype=torch.int8)
            skip = torch.randn((1, 64, h, w), device="cuda", dtype=torch.float16) * 20
            alpha = torch.rand(64, device="cuda") * .001
            bias = torch.randn(64, device="cuda")
            factor = torch.rand(64, device="cuda") * .1
            ref = F.conv2d(x.float(), weight.float(), padding=1)
            ref = (ref * alpha[None, :, None, None] + bias[None, :, None, None] + skip.float()).half()
            rq = (ref.float() * factor[None, :, None, None]).round().clamp(-128, 127).to(torch.int8)
            xp, wp, sp = packed(x), filter_packed(weight), packed(skip)
            out, quant = torch.full_like(sp, float("nan")), torch.empty_like(xp)
            tensors = (xp, wp, sp, alpha, bias, factor, out, quant)
            handle = library.create(config, h, w, tensors)
            try:
                library.run(handle, stream)
                stream.synchronize()
                assert torch.equal(out, packed(ref)), dict(shape=[h, w], kind=kind,
                    half_max_error=float((out - packed(ref)).abs().max()),
                    differing=int((out != packed(ref)).sum()))
                assert torch.equal(quant, packed(rq)), dict(shape=[h, w], kind=kind, int8_mismatch=True)
                records.append(dict(shape=[h, w], kind=kind, half_bit_exact=True, int8_bit_exact=True))
            finally:
                stream.synchronize()
                library.lib.iss_cutlass_destroy(handle)
    # Isolate nearest-even ties and saturation beyond the int32 conversion
    # range; ordinary random convolution checks rarely exercise those values.
    h, w = 2, 3
    xp = torch.zeros((1, 2, h, w, 32), device="cuda", dtype=torch.int8)
    wp = filter_packed(torch.zeros((64, 64, 3, 3), device="cuda", dtype=torch.int8))
    values = torch.tensor([-65504, -129.5, -128.5, -127.5, -126.5, -2.5, -1.5, -.5,
                           .5, 1.5, 2.5, 126.5, 127.5, 128.5, 129.5, 65504],
                          device="cuda", dtype=torch.float16).repeat(4)
    sp = packed(values.view(1, 64, 1, 1).expand(1, 64, h, w))
    alpha = torch.zeros(64, device="cuda")
    bias = torch.zeros_like(alpha)
    factor = torch.ones_like(alpha)
    factor[32:] = 1e6
    refq = packed((values.float() * factor).round().clamp(-128, 127).to(torch.int8)
                  .view(1, 64, 1, 1).expand(1, 64, h, w))
    out, quant = torch.empty_like(sp), torch.empty_like(xp)
    handle = library.create(config, h, w, (xp, wp, sp, alpha, bias, factor, out, quant))
    try:
        library.run(handle, stream)
        stream.synchronize()
        assert torch.equal(out, sp) and torch.equal(quant, refq), "Quantization ties/saturation differ"
        records.append(dict(shape=[h, w], kind="ties_and_saturation", half_bit_exact=True, int8_bit_exact=True))
    finally:
        stream.synchronize()
        library.lib.iss_cutlass_destroy(handle)
    return records


def timing(run, stream, samples):
    run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    for _ in range(50):
        graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(samples):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    graph.reset()
    return stats(values)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--configs", default="0,1,2,3,4,5")
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--capture-only", action="store_true",
                   help="Bracket one validated launch per configuration with cudaProfilerStart/Stop")
    a = p.parse_args()
    if a.check_only and a.capture_only:
        p.error("--check-only and --capture-only are mutually exclusive")
    if a.output.exists():
        raise FileExistsError(a.output)
    result = dict(complete=False, started=time.time(), scope=__doc__, gpu=guard(),
                  source_sha256=digest(Path(__file__)), library_sha256=digest(a.library),
                  torch=torch.__version__, triton=triton.__version__, records=[])
    save(a.output, result)
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    torch.manual_seed(1729)
    lib = Library(a.library)
    h, w = 720, 1280
    x = torch.randint(-128, 128, (1, 2, h, w, 32), device="cuda", dtype=torch.int8)
    weight = torch.randint(-128, 128, (64, 64, 3, 3), device="cuda", dtype=torch.int8)
    wp, wt = filter_packed(weight), weight.permute(2, 3, 1, 0).contiguous()
    skip = torch.randn((1, 2, h, w, 32), device="cuda", dtype=torch.float16)
    out, quant = torch.empty_like(skip), torch.empty_like(x)
    ref, rq = torch.empty_like(skip), torch.empty_like(x)
    alpha = torch.rand(64, device="cuda") * .001
    bias = torch.randn(64, device="cuda")
    factor = torch.rand(64, device="cuda") * .1

    def baseline():
        return conv_dual[(triton.cdiv(h * w, 64),)](x, wt, skip, alpha, bias, factor,
            ref, rq, h, w, 64, num_warps=4, num_stages=2, enable_fp_fusion=False)

    if not a.check_only and not a.capture_only:
        result["baseline_before"] = timing(baseline, stream, a.samples)
        print("baseline_before p95_ms", result["baseline_before"]["p95_ms"], flush=True)
    for config in map(int, a.configs.split(",")):
        row = dict(config=config)
        try:
            row["correctness"] = check(lib, config, stream)
            if not a.check_only:
                tensors = (x, wp, skip, alpha, bias, factor, out, quant)
                handle = lib.create(config, h, w, tensors)
                try:
                    row.update(lib.attributes(handle))
                    lib.run(handle, stream)
                    baseline()
                    stream.synchronize()
                    assert torch.equal(out, ref), "Full-resolution FP16 result differs"
                    assert torch.equal(quant, rq), "Full-resolution INT8 result differs"
                    row["full_resolution_bit_exact"] = True
                    if a.capture_only:
                        # Inputs and outputs are fully resident; capture follows
                        # correctness checks and excludes allocation/reference work.
                        torch.cuda.cudart().cudaProfilerStart()
                        lib.run(handle, stream)
                        stream.synchronize()
                        torch.cuda.cudart().cudaProfilerStop()
                        row["profiled_launch"] = True
                    else:
                        row["latency"] = timing(lambda: lib.run(handle, stream), stream, a.samples)
                finally:
                    stream.synchronize()
                    lib.lib.iss_cutlass_destroy(handle)
            print(config, "p95_ms", row.get("latency", {}).get("p95_ms", "correctness passed"), flush=True)
        except Exception as exc:
            row["error"] = repr(exc)
            print(config, repr(exc), flush=True)
        result["records"].append(row)
        save(a.output, result)
    if not a.check_only and not a.capture_only:
        result["baseline_after"] = timing(baseline, stream, a.samples)
    result.update(complete=True, finished=time.time(), gpu_end=guard())
    save(a.output, result)


if __name__ == "__main__":
    main()
