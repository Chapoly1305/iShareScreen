"""Test whether load-time quantization saves EDSR inter-layer latency.

Microbenchmark: previous residual convolution -> next conv1/ReLU/quantizer.
Both compared conv1 implementations are CUTLASS, not the accepted TensorRT
conv1 tactic. No full-model speed claim follows from a local improvement.
"""
import argparse
import ctypes as c
from pathlib import Path
import time

import torch
from torch.nn import functional as F
import triton

from benchmarks.super_resolution.profile_edsr_cutlass import Library, filter_packed, timing
from benchmarks.super_resolution.profile_edsr_int8_epilogue import conv_dual, packed, unpacked
from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save


class QuantizedLoadLibrary(Library):
    def __init__(self, path):
        super().__init__(path)
        self.lib.iss_quantized_load_create.argtypes = [c.POINTER(c.c_void_p)] + [c.c_int] * 4 + [c.c_void_p] * 7
        self.lib.iss_quantized_load_create.restype = c.c_int

    def create_load(self, config, fused, h, w, tensors):
        handle = c.c_void_p()
        code = self.lib.iss_quantized_load_create(c.byref(handle), config, int(fused), h, w,
                                                *(t.data_ptr() for t in tensors))
        if code:
            raise RuntimeError(f"quantized-load initialization status {code}")
        return handle


@torch.inference_mode()
def check(library, config, stream):
    torch.manual_seed(1729)
    records = []
    for h, w in ((1, 1), (2, 3), (17, 35), (5, 1280)):
        for kind in ("random", "constant", "corner"):
            x = torch.randn((1, 64, h, w), device="cuda", dtype=torch.float16) * 20
            if kind == "constant":
                x.fill_(1)
            elif kind == "corner":
                x.zero_()
                x[:, :, 0, 0] = 65504
                x[:, :, -1, -1] = -65504
            weight = torch.randint(-128, 128, (64, 64, 3, 3), device="cuda", dtype=torch.int8)
            input_factor = torch.rand(64, device="cuda") * 2
            # Powers of two keep independent FP32 integer-product evaluation exact.
            scale = torch.tensor([2. ** (-11 - i % 4) for i in range(64)], device="cuda")
            bias = torch.randn(64, device="cuda")
            factor = torch.full((64,), .125, device="cuda")
            quant = (x.float() * input_factor[None, :, None, None]).round().clamp(-128, 127).to(torch.int8)
            expected = F.conv2d(quant.float(), weight.float(), padding=1)
            expected = (expected * scale[None, :, None, None] + bias[None, :, None, None]).relu()
            expected = (expected * factor[None, :, None, None]).round().clamp(0, 127).to(torch.int8)
            xp, qp, wp = packed(x), packed(quant), filter_packed(weight)
            out = torch.empty_like(qp)
            for fused in (False, True):
                tensors = (xp if fused else qp, wp, input_factor, scale, bias, factor, out)
                handle = library.create_load(config, fused, h, w, tensors)
                try:
                    library.run(handle, stream)
                    stream.synchronize()
                    assert torch.equal(out, packed(expected)), dict(config=config, fused=fused,
                        shape=[h, w], kind=kind, differing=int((out != packed(expected)).sum()),
                        max_error=int((out.int() - packed(expected).int()).abs().max()))
                    records.append(dict(fused=fused, shape=[h, w], kind=kind, bit_exact=True))
                finally:
                    stream.synchronize()
                    library.lib.iss_cutlass_destroy(handle)
    return records


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--configs", default="0,1,2,3")
    p.add_argument("--samples", type=int, default=300)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    r = dict(complete=False, started=time.time(), gpu=guard(), scope=__doc__,
             source_sha256=digest(Path(__file__)), library_sha256=digest(a.library),
             torch=torch.__version__, triton=triton.__version__, records=[])
    save(a.output, r)
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    torch.manual_seed(1729)
    lib = QuantizedLoadLibrary(a.library)
    h, w = 720, 1280
    shape = (1, 2, h, w, 32)
    upstream = torch.randint(-128, 128, shape, device="cuda", dtype=torch.int8)
    upstream_weight = torch.randint(-128, 128, (9, 64, 64), device="cuda", dtype=torch.int8)
    skip = torch.randn(shape, device="cuda", dtype=torch.float16) * 20
    upstream_scale = torch.rand(64, device="cuda") * .001
    upstream_bias = torch.randn(64, device="cuda")
    input_factor = torch.rand(64, device="cuda") * 2
    x = torch.empty_like(skip)
    q = torch.empty_like(upstream)
    weight = torch.randint(-128, 128, (64, 64, 3, 3), device="cuda", dtype=torch.int8)
    wp = filter_packed(weight)
    scale = torch.tensor([2. ** (-11 - i % 4) for i in range(64)], device="cuda")
    bias = torch.randn(64, device="cuda")
    factor = torch.full((64,), .125, device="cuda")

    def previous(write_q):
        return conv_dual[(triton.cdiv(h * w, 64),)](upstream, upstream_weight, skip,
            upstream_scale, upstream_bias, input_factor, x, q, h, w, 64,
            WRITE_Q=write_q, num_warps=4, num_stages=2, enable_fp_fusion=False)

    previous(True)
    stream.synchronize()
    r["previous_conv_timings"] = {
        "dual_output": timing(lambda: previous(True), stream, a.samples),
        "half_output_only": timing(lambda: previous(False), stream, a.samples)}
    r["removed_intermediate_bytes"] = q.numel() * q.element_size()
    save(a.output, r)
    for config in map(int, a.configs.split(",")):
        row = dict(config=config)
        handles = []
        try:
            row["correctness"] = check(lib, config, stream)
            tensors = {}
            outputs = {}
            contexts = {}
            for fused in (False, True):
                output = torch.empty_like(q)
                tensors[fused] = (x if fused else q, wp, input_factor, scale, bias, factor, output)
                contexts[fused] = lib.create_load(config, fused, h, w, tensors[fused])
                handles.append(contexts[fused])
                outputs[fused] = output
            row["metadata"] = {str(f): lib.attributes(handle) for f, handle in contexts.items()}
            for handle in contexts.values():
                lib.run(handle, stream)
            stream.synchronize()
            assert torch.equal(outputs[False], outputs[True]), "Full-resolution output differs"
            row["full_resolution_bit_exact"] = True
            row["conv1_timings"] = {}
            row["pair_timings"] = {}
            for fused in (False, True):
                row["conv1_timings"][str(fused)] = timing(lambda: lib.run(contexts[fused], stream), stream, a.samples)

                def pair():
                    previous(not fused)
                    lib.run(contexts[fused], stream)

                row["pair_timings"][str(fused)] = timing(pair, stream, a.samples)
            stream.synchronize()
            assert torch.equal(outputs[False], outputs[True]), "Paired output differs"
            row["paired_bit_exact"] = True
            print(config, "conv1", {k: v["p95_ms"] for k, v in row["conv1_timings"].items()},
                  "pair", {k: v["p95_ms"] for k, v in row["pair_timings"].items()}, flush=True)
        except Exception as exc:
            row["error"] = repr(exc)
            print(config, repr(exc), flush=True)
        finally:
            stream.synchronize()
            for handle in handles:
                lib.lib.iss_cutlass_destroy(handle)
        r["records"].append(row)
        save(a.output, r)
    r.update(complete=True, finished=time.time(), gpu_end=guard())
    save(a.output, r)


if __name__ == "__main__":
    main()
